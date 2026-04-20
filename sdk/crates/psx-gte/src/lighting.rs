//! Up-to-3-directional-light rig, rendered via the GTE's
//! hardware lighting pipeline (`NCCS` / `NCDS` / `NCS` ops).
//!
//! # Model
//!
//! Each [`Light`] is a *directional* light — a direction vector
//! + an RGB intensity. The PSX GTE evaluates lighting as:
//!
//! ```text
//!   IR   = LLM * V0                         (N dot each light direction)
//!   IR   = [BK] * 0x1000 + LCM * IR         (mix light colours + ambient)
//!   MAC  = IR * RGBC                        (modulate by material/vertex colour)
//!   RGB2 = MAC                              (push to RGB FIFO, readable as u32)
//! ```
//!
//! - `LLM` (light direction matrix, 3×3): each **row** is one
//!   light's direction vector.
//! - `LCM` (light colour matrix, 3×3): each **column** is one
//!   light's RGB contribution (so `LCM * IR` mixes light colours
//!   weighted by the corresponding dot-product magnitudes).
//! - `BK` (background colour, 3×i32): ambient term added
//!   before the material multiply.
//! - `RGBC` (data reg 6): per-vertex material / palette tint —
//!   the GTE multiplies the lit colour by this before emitting.
//!
//! # Object-local vs world-space lights
//!
//! The GTE operates on normals in their native mesh space
//! (typically object-local). Lights packed into LLM therefore
//! also need to be in object-local space for the dot products to
//! come out right. Use [`LightRig::for_object`] once per object
//! per frame to pre-rotate world-space lights into its local
//! frame.
//!
//! # Cost
//!
//! - Rig upload (`load`): ~20 CTC2 writes per frame (cheap).
//! - Per-object rotate: 9 integer multiplies × 3 lights = 27 mul
//!   + 9 adds on the CPU — still cheap next to projection cost.
//! - Per vertex: 2 MTC2 + 1 RTPS + 2 MTC2 + 1 MTC2 + 1 NCCS + 3
//!   MFC2 ≈ 10 GTE ops, ~30-40 cycles total.

use crate::math::{Mat3I16, Vec3I16, Vec3I32};
use crate::scene;
use crate::{mfc2, mtc2, ops};

/// A single directional light in some reference frame (caller's
/// choice of world / object / eye space).
#[derive(Copy, Clone, Debug)]
pub struct Light {
    /// Direction FROM the surface TOWARD the light source. Q3.12
    /// unit vector; if you computed a "light-to-surface" direction
    /// instead, negate it before storing here.
    pub direction: Vec3I16,
    /// Per-channel intensity in Q3.12 (`0x1000` = full intensity,
    /// `0x0800` = half, `0x2000` = double / over-bright). The GTE
    /// clamps out-of-range values at the MAC stage.
    pub colour: (i16, i16, i16),
}

impl Light {
    /// Convenience constructor for a "warm-white at 75%" feel —
    /// useful while prototyping light rigs.
    pub const fn warm_white_at(direction: Vec3I16) -> Self {
        Self {
            direction,
            colour: (0x0C00, 0x0A00, 0x0800),
        }
    }

    /// "Cool fill" preset — bluish, dimmer, useful as a
    /// secondary / rim light.
    pub const fn cool_fill_at(direction: Vec3I16) -> Self {
        Self {
            direction,
            colour: (0x0400, 0x0600, 0x0A00),
        }
    }

    /// Null light — contributes nothing. Use as a filler for
    /// unused slots in a rig.
    pub const OFF: Self = Self {
        direction: Vec3I16::ZERO,
        colour: (0, 0, 0),
    };
}

/// A rig of up to **3** directional lights — the maximum the
/// PSX GTE supports natively.
#[derive(Copy, Clone, Debug)]
pub struct LightRig {
    /// Three light slots; unused slots should be [`Light::OFF`].
    pub lights: [Light; 3],
    /// Ambient term added before material multiply. Q3.12 per
    /// channel in an `i32` (GTE's BK register is 20-bit signed).
    pub ambient: (i32, i32, i32),
}

impl LightRig {
    /// Empty rig: no lights, black ambient.
    pub const OFF: Self = Self {
        lights: [Light::OFF; 3],
        ambient: (0, 0, 0),
    };

    /// Build a rig from 3 lights + an ambient term.
    pub const fn new(lights: [Light; 3], ambient: (i32, i32, i32)) -> Self {
        Self { lights, ambient }
    }

    /// Pack this rig into the GTE's control registers:
    /// - LLM ← rows are light directions
    /// - LCM ← columns are light RGB contributions
    /// - BK ← ambient
    ///
    /// Call once per object (or once per frame if lights are
    /// in a universal frame). Any subsequent [`NCCS`][ops::nccs]
    /// / [`NCS`][ops::ncs] / [`NCDS`][ops::ncds] reads these.
    pub fn load(&self) {
        let [l0, l1, l2] = self.lights;

        // LLM: rows = light directions.
        let llm = Mat3I16 {
            m: [
                [l0.direction.x, l0.direction.y, l0.direction.z],
                [l1.direction.x, l1.direction.y, l1.direction.z],
                [l2.direction.x, l2.direction.y, l2.direction.z],
            ],
        };
        scene::load_light_matrix(&llm);

        // LCM: `LCM * IR` mixes light colours weighted by the
        // dot-product intensities. IR[i] = light-i's dot product,
        // so LCM's COLUMN i must be light-i's RGB triple. In our
        // row-major layout that means m[channel][light].
        let lcm = Mat3I16 {
            m: [
                [l0.colour.0, l1.colour.0, l2.colour.0], // R row
                [l0.colour.1, l1.colour.1, l2.colour.1], // G row
                [l0.colour.2, l1.colour.2, l2.colour.2], // B row
            ],
        };
        scene::load_light_colour_matrix(&lcm);

        // BK: ambient term.
        scene::load_background_colour(Vec3I32::new(
            self.ambient.0,
            self.ambient.1,
            self.ambient.2,
        ));
    }

    /// Return a new rig with every light's direction rotated into
    /// the frame in which `rotation` is the object's local → world
    /// transform. I.e., if you pass the same `Mat3I16` you pass to
    /// [`scene::load_rotation`] for RTPS, the returned rig is in
    /// object-local space — correct for feeding LLM when lighting
    /// mesh vertices that are themselves in local space.
    ///
    /// Math: world_vert = R × local_vert, so local = R⁻¹ × world,
    /// and since R is a rotation, R⁻¹ = Rᵀ. We compute
    /// `local_dir = Rᵀ × world_dir` component-wise.
    pub fn for_object(&self, rotation: &Mat3I16) -> LightRig {
        let rotate = |dir: Vec3I16| {
            // R^T * dir: the dot of `dir` with each COLUMN of R.
            let mut out = [0i32; 3];
            for i in 0..3 {
                out[i] = ((rotation.m[0][i] as i32) * (dir.x as i32)
                    + (rotation.m[1][i] as i32) * (dir.y as i32)
                    + (rotation.m[2][i] as i32) * (dir.z as i32))
                    >> 12;
            }
            Vec3I16::new(
                out[0].clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                out[1].clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                out[2].clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            )
        };
        let rot_light = |l: &Light| Light {
            direction: rotate(l.direction),
            colour: l.colour,
        };
        LightRig {
            lights: [
                rot_light(&self.lights[0]),
                rot_light(&self.lights[1]),
                rot_light(&self.lights[2]),
            ],
            ambient: self.ambient,
        }
    }
}

/// Result of [`project_lit`]: screen-space vertex + its computed
/// lit colour.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct ProjectedLit {
    /// Screen-space X, post-projection.
    pub sx: i16,
    /// Screen-space Y.
    pub sy: i16,
    /// Depth (MAC3 after divide), 0..=0xFFFF.
    pub sz: u16,
    /// GTE-computed RGB after lighting + material modulation.
    pub r: u8,
    /// Lit green.
    pub g: u8,
    /// Lit blue.
    pub b: u8,
}

/// Project a vertex AND compute its lit colour in one call.
///
/// Does RTPS on `vert` + NCCS on `normal`, with a per-vertex
/// `material` RGB loaded into RGBC (data reg 6) to modulate the
/// GTE's lit output. Pass `(128, 128, 128)` as material for
/// "unmodulated" behaviour — the lit colour comes through as the
/// light rig + ambient dictate.
///
/// Prerequisites (caller's responsibility, as with
/// [`scene::project_vertex`]):
///
/// - Rotation matrix loaded ([`scene::load_rotation`])
/// - Translation loaded ([`scene::load_translation`])
/// - Projection plane / screen offset set
///   ([`scene::set_projection_plane`], [`scene::set_screen_offset`])
/// - Light rig loaded ([`LightRig::load`], with `for_object`
///   applied so lights are in the same frame as the normal)
pub fn project_lit(
    vert: Vec3I16,
    normal: Vec3I16,
    material: (u8, u8, u8),
) -> ProjectedLit {
    // --- Position: RTPS ---
    mtc2!(0, vert.xy_packed());
    mtc2!(1, vert.z_packed());
    // SAFETY: V0 loaded; scene setup is caller's responsibility.
    unsafe { ops::rtps() };
    let sxy = mfc2!(14); // SXY2 (packed xy)
    let sz = mfc2!(19) as u16; // SZ3

    // --- Lighting: NCCS ---
    mtc2!(0, normal.xy_packed());
    mtc2!(1, normal.z_packed());
    // RGBC layout (data reg 6): 0x00CC_BBGG_RR — low 8 bits R,
    // next 8 G, next 8 B, top 8 "CODE" (GPU command byte, used by
    // some prim ops; 0 for our purposes).
    let rgbc = (material.0 as u32) | ((material.1 as u32) << 8) | ((material.2 as u32) << 16);
    mtc2!(6, rgbc);
    // SAFETY: V0 holds the normal, RGBC holds the material;
    // LLM/LCM/BK were loaded via `LightRig::load`.
    unsafe { ops::nccs() };
    // Read lit colour from RGB2 (data reg 22). Same 0x00BB_GGRR
    // layout as RGBC.
    let lit = mfc2!(22);

    ProjectedLit {
        sx: sxy as i16,
        sy: (sxy >> 16) as i16,
        sz,
        r: (lit & 0xFF) as u8,
        g: ((lit >> 8) & 0xFF) as u8,
        b: ((lit >> 16) & 0xFF) as u8,
    }
}

