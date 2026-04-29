//! World-space coord types + per-actor transforms.
//!
//! # Why a separate layer
//!
//! The SDK's [`psx_gte::math`] types are **register-shaped**: they
//! pack into the exact 16-bit and 32-bit slots the GTE's MTC2 and
//! CTC2 opcodes expect. That's the right shape for the SDK — a thin
//! hardware wrapper — but it leaves the engine with a "one type,
//! many meanings" problem:
//!
//! - [`psx_gte::math::Vec3I16`] is used for mesh-local vertices **and**
//!   light directions **and** un-scaled world normals.
//! - [`psx_gte::math::Vec3I32`] is used for world-space translations
//!   **and** colour biases **and** the GTE's far-colour register.
//!
//! Call sites lose the "which coordinate space is this?" information
//! the moment the value leaves its source function. Worse: every
//! engine example so far has re-implemented its own `scale_mat`
//! helper to fold a per-actor uniform scale into a rotation matrix,
//! because the SDK's `Mat3I16::mul` is the only composition op and
//! taking "scale × rotation" through that would saturate.
//!
//! This module introduces two engine-level types that shift those
//! concerns back into the type system:
//!
//! - [`Vec3World`] — a position in **world space**, Q19.12. Cannot be
//!   mixed up with a mesh-local `Vec3I16` or a GTE colour register.
//! - [`ActorTransform`] — a pose for one "actor" (game object) — its
//!   world-space position, rotation, and uniform scale. A single
//!   [`ActorTransform::load_gte`] call composes them into the
//!   rotation + translation control-register writes the GTE needs
//!   for subsequent [`project_vertex`][psx_gte::scene::project_vertex]
//!   calls.
//!
//! # OoT-style scaling
//!
//! The shape mirrors a convention Nintendo 64 games like Ocarina of
//! Time used for their actor system: each actor carried a
//! `(position, rotation, scale)` triple, and the render path
//! multiplied the scale into the rotation matrix before loading the
//! RSP's vertex transform. Here the GTE replaces the RSP, but the
//! API shape is the same: one per-frame update per actor, one
//! `.load_gte()` call, then draw its mesh with mesh-local vertex
//! coords and the hardware does the rest.
//!
//! # Scale semantics
//!
//! `ActorTransform::scale` is a **uniform** Q3.12 factor. `0x1000`
//! is 1.0× (identity); `0x0800` halves mesh size, `0x2000` doubles
//! it, and so on. Non-uniform (per-axis) scale is deliberately out
//! of scope — every engine example that's wanted scale so far has
//! wanted it uniform, and folding per-axis scale into the rotation
//! matrix interacts badly with normal transforms (distorts
//! lighting). If a non-uniform case comes up later, we add it; in
//! the meantime this keeps the type narrow.

use psx_gte::math::{Mat3I16, Vec3I32};
use psx_gte::scene;

/// A position in world space. Stored as raw Q19.12 per-axis `i32`
/// — the same representation the GTE's translation register (TR,
/// control-register slots 5..=7) expects, so `load_gte` forwards
/// them unchanged.
///
/// # Units
///
/// The engine adopts the GTE's native "Q3.12 unit" as its world
/// unit: `1.0` in world space is `0x1000` in the stored `i32`.
/// Mesh vertices in `Vec3I16` coords use the same scale — a cube
/// mesh whose corners are at `±0x0800` is a 1.0-unit cube, which
/// means an `ActorTransform` placing it at
/// `Vec3World::from_units(3, 0, 10)` puts its centre at world
/// `(3.0, 0.0, 10.0)`.
///
/// # Range
///
/// `i32` gives ±524,288 world units comfortably — well past any
/// scene a PSX game would actually render, and a big headroom
/// over the GTE's internal 16-bit vertex inputs (which is why
/// meshes still use `Vec3I16` — they live in local space where
/// ±8.0 units is plenty).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash)]
#[repr(C)]
pub struct Vec3World {
    /// X axis, Q19.12.
    pub x: i32,
    /// Y axis, Q19.12.
    pub y: i32,
    /// Z axis, Q19.12.
    pub z: i32,
}

impl Vec3World {
    /// Origin.
    pub const ZERO: Vec3World = Vec3World { x: 0, y: 0, z: 0 };

    /// Construct from integer world units. `from_units(3, 0, -2)` puts
    /// you at world `(3.0, 0.0, -2.0)`. Internally shifts each axis
    /// left by 12 so the result lands in the GTE's native Q19.12 scale.
    #[inline]
    pub const fn from_units(x: i32, y: i32, z: i32) -> Vec3World {
        Vec3World {
            x: x << 12,
            y: y << 12,
            z: z << 12,
        }
    }

    /// Construct from raw Q19.12 components. Use when you already
    /// have hand-packed values (e.g. copied from an older `Vec3I32`
    /// site). Prefer [`Vec3World::from_units`] when the value
    /// conceptually is an integer world-unit count.
    #[inline]
    pub const fn from_raw(x: i32, y: i32, z: i32) -> Vec3World {
        Vec3World { x, y, z }
    }

    /// Drop into the GTE's native `Vec3I32` register-shape — what
    /// [`psx_gte::scene::load_translation`] wants. Zero-cost.
    #[inline]
    pub const fn as_gte_translation(self) -> Vec3I32 {
        Vec3I32::new(self.x, self.y, self.z)
    }
}

/// Per-actor pose: where, how oriented, and how big.
///
/// # Typical use
///
/// ```ignore
/// let actor = ActorTransform::at(Vec3World::from_units(0, 0, 10))
///     .with_rotation(Mat3I16::rotate_y(angle))
///     .with_scale_q12(0x0800); // half-size
///
/// actor.load_gte();
/// for v in mesh.vertices() {
///     let p = psx_gte::scene::project_vertex(v);
///     // ... draw triangle using p
/// }
/// ```
///
/// One `load_gte` call per actor, then drive the mesh through the
/// GTE as normal. The scale folds into the rotation matrix before
/// upload, so the projection pipeline runs at normal RTPS cost
/// — no per-vertex multiply overhead.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ActorTransform {
    /// World-space position.
    pub position: Vec3World,
    /// Rotation about the actor's local origin. Q3.12 rotation
    /// matrix — composable through [`Mat3I16::mul`] when building
    /// multi-axis tumbles on the way in.
    pub rotation: Mat3I16,
    /// Uniform scale, Q3.12. `0x1000` = 1.0× (identity). The
    /// [`load_gte`][Self::load_gte] path multiplies this into each
    /// cell of `rotation` before uploading, so the vertex shader
    /// path runs at "just rotation" cost.
    pub scale: i16,
}

impl ActorTransform {
    /// Identity pose — origin, identity rotation, 1.0× scale.
    /// Equivalent to "draw the mesh exactly as authored, in world
    /// space at the origin".
    pub const IDENTITY: ActorTransform = ActorTransform {
        position: Vec3World::ZERO,
        rotation: Mat3I16::IDENTITY,
        scale: 0x1000,
    };

    /// Start from [`Self::IDENTITY`] at the given world position.
    /// Chain [`Self::with_rotation`] / [`Self::with_scale_q12`] for
    /// the remaining fields; each returns `Self` so the builder
    /// reads top-to-bottom at the call site.
    #[inline]
    pub const fn at(position: Vec3World) -> ActorTransform {
        ActorTransform {
            position,
            rotation: Mat3I16::IDENTITY,
            scale: 0x1000,
        }
    }

    /// Replace the rotation. Typically called with
    /// [`Mat3I16::rotate_y`] / `rotate_x` / `rotate_z`, or a
    /// composed tumble like `rotate_y(a).mul(&rotate_x(b))`.
    #[inline]
    pub const fn with_rotation(mut self, rotation: Mat3I16) -> ActorTransform {
        self.rotation = rotation;
        self
    }

    /// Replace the uniform scale. `0x1000` = 1.0×, `0x0800` = 0.5×,
    /// `0x2000` = 2.0×. Naming is `_q12` so the caller can't miss
    /// the fixed-point contract.
    #[inline]
    pub const fn with_scale_q12(mut self, scale_q12: i16) -> ActorTransform {
        self.scale = scale_q12;
        self
    }

    /// The rotation matrix after folding in the uniform scale — the
    /// exact matrix [`load_gte`][Self::load_gte] will upload to the
    /// GTE's RT control registers.
    ///
    /// Exposed for callers that also need to do CPU-side transform
    /// math alongside the GTE pipeline — e.g. a CPU point-light
    /// shader computing world-space positions via the same scaled
    /// rotation the GTE uses. Reusing this accessor guarantees the
    /// CPU and GTE paths see bit-identical matrices.
    #[inline]
    pub fn scaled_rotation(&self) -> Mat3I16 {
        scale_mat_uniform(&self.rotation, self.scale as i32)
    }

    /// Compose `rotation × uniform_scale` and upload along with the
    /// world translation to the GTE. After this, each
    /// [`psx_gte::scene::project_vertex`] call transforms a mesh-
    /// local vertex through the full actor transform into screen
    /// space.
    ///
    /// Cost: 9 multiplies + 9 shifts on the CPU (scaling the 3×3
    /// rotation matrix in place), plus the 8 `ctc2` register writes
    /// that the SDK's [`load_rotation`][scene::load_rotation] and
    /// [`load_translation`][scene::load_translation] helpers do.
    pub fn load_gte(&self) {
        scene::load_rotation(&self.scaled_rotation());
        scene::load_translation(self.position.as_gte_translation());
    }
}

/// Fold a uniform Q3.12 scale into every cell of a Q3.12 matrix.
///
/// Rotation matrices have `|cell|` in `[0, 0x1000]`; scale factors
/// up to `0x8000` (8.0×) keep the product within `i16`. Anything
/// larger saturates — callers who need huge scale should split
/// into a mesh-local pre-scale and a smaller runtime factor.
///
/// Exposed at module-level (not via `impl Mat3I16`) deliberately:
/// it's engine-policy scaling math, not a hardware-register
/// abstraction, so it stays in the engine crate alongside
/// [`ActorTransform`].
fn scale_mat_uniform(m: &Mat3I16, scale_q12: i32) -> Mat3I16 {
    let mut out = [[0i16; 3]; 3];
    // Unrolled 3×3 keeps the code-path branchless + inline-friendly.
    // Values pre-shift fit in `i32` (16-bit × 16-bit); the Q3.12
    // fractional drop happens once per cell.
    let mut i = 0;
    while i < 3 {
        let mut j = 0;
        while j < 3 {
            let v = ((m.m[i][j] as i32) * scale_q12) >> 12;
            // Clamp to `i16` so pathologically large scale factors
            // saturate cleanly instead of wrapping.
            out[i][j] = if v > i16::MAX as i32 {
                i16::MAX
            } else if v < i16::MIN as i32 {
                i16::MIN
            } else {
                v as i16
            };
            j += 1;
        }
        i += 1;
    }
    Mat3I16 { m: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Vec3World
    // ------------------------------------------------------------------

    #[test]
    fn vec3world_from_units_scales_by_4096() {
        let v = Vec3World::from_units(1, 2, -3);
        assert_eq!(v.x, 0x1000);
        assert_eq!(v.y, 0x2000);
        assert_eq!(v.z, -0x3000);
    }

    #[test]
    fn vec3world_from_raw_passes_through() {
        let v = Vec3World::from_raw(0x1234, -1, 0x7FFF_FFFF);
        assert_eq!(v.x, 0x1234);
        assert_eq!(v.y, -1);
        assert_eq!(v.z, 0x7FFF_FFFF);
    }

    #[test]
    fn vec3world_zero_is_all_zero() {
        assert_eq!(Vec3World::ZERO.x, 0);
        assert_eq!(Vec3World::ZERO.y, 0);
        assert_eq!(Vec3World::ZERO.z, 0);
    }

    #[test]
    fn vec3world_as_gte_translation_roundtrips_values() {
        let v = Vec3World::from_raw(100, -200, 300);
        let t = v.as_gte_translation();
        assert_eq!(t.x, 100);
        assert_eq!(t.y, -200);
        assert_eq!(t.z, 300);
    }

    // ------------------------------------------------------------------
    // ActorTransform — builder
    // ------------------------------------------------------------------

    #[test]
    fn actor_transform_identity_is_neutral() {
        let t = ActorTransform::IDENTITY;
        assert_eq!(t.position, Vec3World::ZERO);
        assert_eq!(t.rotation, Mat3I16::IDENTITY);
        assert_eq!(t.scale, 0x1000);
    }

    #[test]
    fn actor_transform_at_sets_position_only() {
        let p = Vec3World::from_units(5, 0, -10);
        let t = ActorTransform::at(p);
        assert_eq!(t.position, p);
        assert_eq!(t.rotation, Mat3I16::IDENTITY);
        assert_eq!(t.scale, 0x1000);
    }

    #[test]
    fn actor_transform_scaled_rotation_matches_manual_compose() {
        let rot = Mat3I16::rotate_y(32);
        let t = ActorTransform::at(Vec3World::ZERO)
            .with_rotation(rot)
            .with_scale_q12(0x0800);
        // `scaled_rotation` must return exactly what `load_gte`
        // would upload — i.e. the same byte output the old hand-
        // rolled `scale_mat` helpers produced.
        assert_eq!(t.scaled_rotation(), scale_mat_uniform(&rot, 0x0800));
    }

    #[test]
    fn actor_transform_scaled_rotation_identity_scale_noop() {
        // A scale of 1.0× must leave the rotation matrix bytewise
        // unchanged — critical for the showcase-3d migration where
        // meshes don't use scale but still go through `load_gte`.
        let rot = Mat3I16::rotate_z(96);
        let t = ActorTransform::at(Vec3World::ZERO).with_rotation(rot);
        assert_eq!(t.scaled_rotation(), rot);
    }

    #[test]
    fn actor_transform_builder_chains_cleanly() {
        let rot = Mat3I16::rotate_y(64); // quarter turn
        let t = ActorTransform::at(Vec3World::from_units(1, 2, 3))
            .with_rotation(rot)
            .with_scale_q12(0x0800);
        assert_eq!(t.position.x, 0x1000);
        assert_eq!(t.position.y, 0x2000);
        assert_eq!(t.position.z, 0x3000);
        assert_eq!(t.rotation, rot);
        assert_eq!(t.scale, 0x0800);
    }

    // ------------------------------------------------------------------
    // scale_mat_uniform
    // ------------------------------------------------------------------

    #[test]
    fn scale_mat_uniform_identity_scale_preserves_matrix() {
        let rot = Mat3I16::rotate_y(32);
        let out = scale_mat_uniform(&rot, 0x1000);
        assert_eq!(out, rot, "scaling by 1.0× must not change any cell");
    }

    #[test]
    fn scale_mat_uniform_half_shrinks_every_cell() {
        let out = scale_mat_uniform(&Mat3I16::IDENTITY, 0x0800);
        // 0x1000 × 0x0800 >> 12 = 0x0800.
        assert_eq!(out.m[0][0], 0x0800);
        assert_eq!(out.m[1][1], 0x0800);
        assert_eq!(out.m[2][2], 0x0800);
        // Off-diagonal cells stay zero.
        assert_eq!(out.m[0][1], 0);
        assert_eq!(out.m[1][2], 0);
    }

    #[test]
    fn scale_mat_uniform_double_grows_diagonal() {
        let out = scale_mat_uniform(&Mat3I16::IDENTITY, 0x2000);
        // 0x1000 × 0x2000 >> 12 = 0x2000 — still in i16 range.
        assert_eq!(out.m[0][0], 0x2000);
        assert_eq!(out.m[1][1], 0x2000);
        assert_eq!(out.m[2][2], 0x2000);
    }

    #[test]
    fn scale_mat_uniform_saturates_without_wrapping() {
        // 0x1000 × 0x1_0000 >> 12 = 0x1_0000 — one above i16::MAX.
        // The clamp must prevent a wrap-around sign flip that would
        // silently mirror the actor.
        let out = scale_mat_uniform(&Mat3I16::IDENTITY, 0x1_0000);
        assert_eq!(out.m[0][0], i16::MAX);
    }

    #[test]
    fn scale_mat_uniform_preserves_sign_on_negative_cells() {
        // Build a matrix with a negative off-diagonal and confirm
        // scaling preserves sign.
        let m = Mat3I16 {
            m: [[0x1000, -0x0800, 0], [0, 0x1000, 0], [0, 0, 0x1000]],
        };
        let out = scale_mat_uniform(&m, 0x0800);
        // 0x1000 × 0x0800 >> 12 = 0x0800 on the diagonal.
        assert_eq!(out.m[0][0], 0x0800);
        // -0x0800 × 0x0800 >> 12 = -0x0400 off-diagonal.
        assert_eq!(out.m[0][1], -0x0400);
    }

    // ------------------------------------------------------------------
    // Unused-field guard
    // ------------------------------------------------------------------

    #[test]
    fn vec3world_is_copy_and_sized_as_three_i32() {
        // Regression: keep `Vec3World` byte-compatible with the
        // natural `[i32; 3]` layout so future MIPS FFI / DMA code
        // can reinterpret-cast without surprises.
        assert_eq!(
            core::mem::size_of::<Vec3World>(),
            3 * core::mem::size_of::<i32>()
        );
    }
}
