//! `showcase-lights` -- four coloured moving point lights
//! illuminating a small set of scaled cubes.
//!
//! Complementary to `showcase-3d`:
//!
//! | demo             | lighting path                   | light count | light type   |
//! |------------------|---------------------------------|-------------|--------------|
//! | showcase-3d      | GTE NCCS (hardware)             | 3           | directional  |
//! | showcase-lights  | CPU per-vertex N·L (this file)  | 4           | point (pos + radius) |
//!
//! The PSX GTE only natively supports **directional** lights, and
//! only up to **3** at a time. Point lights need per-vertex
//! direction + distance math which the GTE can't do, so this demo
//! stays on the CPU. Even with 4 lights × 48 verts (6 cubes × 8),
//! it's about 1k integer-math ops per frame -- trivially under
//! budget on a 33 MHz MIPS R3000.
//!
//! What's on screen:
//!
//! 1. **Dark-room backdrop** -- a Gouraud gradient behind the scene.
//! 2. **6 scaled cubes** -- different heights + XZ
//!    positions, one canonical cube mesh `include_bytes!`'d from
//!    `assets/cube.psxm` (cooked from a face-split OBJ so per-face
//!    normals come out of `psxed --compute-normals` flat).
//! 3. **4 point lights** in R / G / B / Y, each on its own orbit
//!    with a different period so they visibly "dance" around the
//!    scene.
//! 4. **HUD** -- frame / live-tri / light positions.
//!
//! Ported to `psx-engine` in Phase 3e. The `Scene` struct (renamed
//! to `Lighting` to avoid collision with the engine trait) now
//! holds per-frame state inline; GTE one-time setup moved to
//! `Scene::init`; geometry animation drives off `ctx.frame` instead
//! of an incrementing `static mut SCENE.frame`.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::Mesh;
use psx_engine::{
    ActorTransform, App, Config, Ctx, DepthBand, DepthRange, GouraudMeshOptions, GouraudRenderPass,
    GouraudTriCommand, OtDepth, OtFrame, PrimitiveArena, Scene, Vec3World,
};
use psx_font::{fonts::BASIC_8X16, FontAtlas};
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat, TriGouraud};
use psx_gte::lighting::ProjectedLit;
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_math::sincos;
use psx_vram::{Clut, TexDepth, Tpage};

// ----------------------------------------------------------------------
// Screen / scene geometry
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;
const OT_DEPTH: usize = 16;
const BG_SLOT: usize = OT_DEPTH - 1;
const WORLD_BAND: DepthBand = OtDepth::<OT_DEPTH>::band(3, OT_DEPTH - 3);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(0x1800, 0x8000);
const LIGHT_MARKER_SLOT: usize = 2;

const PROJ_H: u16 = 280;
/// Camera setback -- tuned so the cubes sit comfortably in the
/// middle distance rather than crowding the screen. Kept just
/// under `i16::MAX / 2` so light orbits ±0x2C00 around it still
/// fit in Vec3I16 (the GTE's native vertex type).
const WORLD_Z: i32 = 0x5000;

const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

// Cube mesh: 24 verts, 12 triangles, face-split so per-vertex
// normals come out equal to their face normal after psxed's
// area-weighted average (only adjacent to same-face tris).
static CUBE_BLOB: &[u8] = include_bytes!("../assets/cube.psxm");

// ----------------------------------------------------------------------
// Light rig (4 point lights, CPU-shaded)
// ----------------------------------------------------------------------

const NUM_LIGHTS: usize = 4;

/// Radius large enough to reach every cube from any orbit
/// position, small enough that distant cubes fall off visibly.
/// Cubes span ±0x2200 on X and sit near z ≈ WORLD_Z; lights
/// orbit at radii up to 0x2C00 around scene centre.
const LIGHT_RADIUS_SQ: i32 = 0x1800_0000;

/// Tuned to match `sqrt(LIGHT_RADIUS_SQ) ≈ 0x4E00`. Used as the
/// "reference distance" the dot product divides by to turn its
/// N · D magnitude (which carries distance) into a rough
/// direction-only scalar.
const RADIUS_LINEAR: i32 = 0x4E00;
const LIGHT_ATT_DENOM_Q12: i32 = LIGHT_RADIUS_SQ >> 12;
const LIGHT_DELTA_LIMIT: i32 = 0x4F00;

/// Per-channel ambient term, added before clamping. Dim blueish
/// so shadowed faces read as "in the dark room" rather than
/// pure-black holes.
const AMBIENT: (i32, i32, i32) = (20, 24, 36);

#[derive(Copy, Clone)]
struct PointLight {
    /// World-space position (Q3.12-ish integers -- we don't treat
    /// them as fractional, just as "world-space units" where 1.0 =
    /// 0x1000 matching the cube scale).
    pos: Vec3I16,
    /// Linear RGB intensity, 0..=255 per channel.
    colour: (u8, u8, u8),
}

impl PointLight {
    const fn new(pos: Vec3I16, colour: (u8, u8, u8)) -> Self {
        Self { pos, colour }
    }
}

// ----------------------------------------------------------------------
// Cube instances
// ----------------------------------------------------------------------

const NUM_CUBES: usize = 4;

/// One cube placed in the room. `scale` is a Q3.12 factor
/// applied to each vertex before rotation; the cube mesh has
/// unit-side verts at ±0x0800, so `scale = 0x1000` renders a
/// 1.0×1.0×1.0 cube and `scale = 0x0800` renders 0.5×0.5×0.5.
/// Position uses the engine's `Vec3World` so the type system
/// catches "wrong coord space" bugs that the old raw `Vec3I32`
/// layout couldn't.
#[derive(Copy, Clone)]
struct CubeInstance {
    position: Vec3World,
    scale: i16,
    /// Angular velocity per frame (Q0.12 units), for Y-axis spin.
    y_spin_per_frame: u16,
}

/// Gallery-style layout: 4 cubes in a gentle arc left-to-right
/// with staggered depth + height so each one is clearly
/// independent in the frame. Sizes vary so the scene reads as a
/// "collection of objects" rather than a uniform grid.
///
/// Positions are relative to WORLD_Z -- the `z` component is the
/// additional depth offset from the camera plane. Raw Q19.12
/// coordinates (not the `from_units` constructor) because the
/// original values were hand-tuned in that scale.
const CUBE_LAYOUT: [CubeInstance; NUM_CUBES] = [
    // Far-left, low, medium-large, slow spin.
    CubeInstance {
        position: Vec3World::from_raw(-0x2000, -0x0400, WORLD_Z + 0x0600),
        scale: 0x0B00,
        y_spin_per_frame: 3,
    },
    // Mid-left, higher, smaller, slightly faster.
    CubeInstance {
        position: Vec3World::from_raw(-0x0A00, 0x0400, WORLD_Z - 0x0400),
        scale: 0x0800,
        y_spin_per_frame: 5,
    },
    // Mid-right, low, medium, different rate.
    CubeInstance {
        position: Vec3World::from_raw(0x0A00, -0x0600, WORLD_Z - 0x0200),
        scale: 0x0900,
        y_spin_per_frame: 4,
    },
    // Far-right, mid-height, largest, slowest.
    CubeInstance {
        position: Vec3World::from_raw(0x2200, 0, WORLD_Z + 0x0400),
        scale: 0x0D00,
        y_spin_per_frame: 2,
    },
];

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

/// Renamed from the pre-engine `Scene` struct to avoid name
/// collision with [`psx_engine::Scene`] -- this is *our* scene
/// type; the trait is what the engine calls back into.
struct Lighting {
    tri_count: u16,
    lights: [PointLight; NUM_LIGHTS],
    font: Option<FontAtlas>,
}

// ----------------------------------------------------------------------
// OT + primitive buffers
// ----------------------------------------------------------------------

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();

/// Gouraud cube tris: 4 cubes × 12 tris = 48. 96 slots leaves
/// headroom for a denser layout later.
static mut GOURAUD_TRIS: [TriGouraud; 96] = [const {
    TriGouraud {
        tag: 0,
        color0_cmd: 0,
        v0: 0,
        color1: 0,
        v1: 0,
        color2: 0,
        v2: 0,
    }
}; 96];

/// Rects for light position markers (one per light -- small bright
/// square at each light's projected screen pos so the viewer can
/// see where the lights are).
static mut LIGHT_MARKERS: [RectFlat; 16] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 16];
static mut GOURAUD_COMMANDS: [GouraudTriCommand; 96] = [GouraudTriCommand::EMPTY; 96];

static mut BG_QUAD: QuadGouraud = QuadGouraud {
    tag: 0,
    color0_cmd: 0,
    v0: 0,
    color1: 0,
    v1: 0,
    color2: 0,
    v2: 0,
    color3: 0,
    v3: 0,
};

/// Per-cube projected vertex cache.
const EMPTY_VERT: ProjectedLit = ProjectedLit {
    sx: 0,
    sy: 0,
    sz: 0,
    r: 0,
    g: 0,
    b: 0,
};
static mut CUBE_PROJ: [ProjectedLit; 32] = [EMPTY_VERT; 32];

// ----------------------------------------------------------------------
// Per-vertex CPU lighting
// ----------------------------------------------------------------------

/// Compute a lit colour for one vertex given its world-space
/// position + normal.
///
/// For each light: compute `direction = light_pos - vert_pos`,
/// then the un-normalised dot product with the normal. This mixes
/// "distance × cos(angle)" into one scalar -- we skip the explicit
/// normalise (which would need a sqrt) and instead rely on a
/// separate quadratic distance-falloff term to take the distance
/// contribution out of the dot.
///
/// Fixed-point: normal is Q3.12 (components in ±0x1000), position
/// deltas are "world units" where 1 unit = 0x1000. The un-normalised
/// dot lands in Q6.24 space but we scale to a bounded intensity
/// before accumulating, so no 32-bit overflow risk.
fn light_vertex(
    lights: &[PointLight; NUM_LIGHTS],
    world_pos: Vec3I16,
    world_normal: Vec3I16,
) -> (u8, u8, u8) {
    let mut r: i32 = AMBIENT.0;
    let mut g: i32 = AMBIENT.1;
    let mut b: i32 = AMBIENT.2;

    for light in lights.iter() {
        // Vector from vertex to light.
        let dx = (light.pos.x as i32) - (world_pos.x as i32);
        let dy = (light.pos.y as i32) - (world_pos.y as i32);
        let dz = (light.pos.z as i32) - (world_pos.z as i32);

        if dx < -LIGHT_DELTA_LIMIT
            || dx > LIGHT_DELTA_LIMIT
            || dy < -LIGHT_DELTA_LIMIT
            || dy > LIGHT_DELTA_LIMIT
            || dz < -LIGHT_DELTA_LIMIT
            || dz > LIGHT_DELTA_LIMIT
        {
            continue;
        }

        let dist_sq = dx * dx + dy * dy + dz * dz;

        if dist_sq > LIGHT_RADIUS_SQ {
            continue;
        }

        // Quadratic attenuation: 1 at dist=0, 0 at dist²=RADIUS_SQ.
        // att in [0, 0x1000]
        let att_q12 = ((LIGHT_RADIUS_SQ - dist_sq) / LIGHT_ATT_DENOM_Q12).clamp(0, 0x1000);

        // Un-normalised N · D. Components of N are Q3.12, D's are
        // raw world units. Result is in Q3.12 × world-units.
        let dot_unnorm = (world_normal.x as i32) * dx
            + (world_normal.y as i32) * dy
            + (world_normal.z as i32) * dz;

        // Facing-away? Skip.
        if dot_unnorm <= 0 {
            continue;
        }

        // To get a unit-less cosine scalar we'd want dot_unnorm /
        // (|N| × |D|). |N| = 1.0 = 0x1000. |D| = sqrt(dist_sq).
        // We cheat: divide by sqrt(RADIUS_SQ) -- a constant -- which
        // sacrifices strict physical accuracy but gives a clean
        // "nearness + orientation" scalar in a known range.
        //
        // Dividing the Q3.12 dot (which carries distance in its
        // magnitude) by `RADIUS_LINEAR` rescales it into a rough
        // Q3.12 direction-intensity in `[0, ~0x1000]`.
        let intensity_q12 = dot_unnorm / RADIUS_LINEAR;

        // Modulate by quadratic attenuation and normal magnitude.
        // `lit_scalar` is in Q3.12, roughly [0, 0x1000].
        let lit_scalar = (intensity_q12 * att_q12) >> 12;
        // Clamp before accumulating -- single lights shouldn't
        // saturate the channel on their own.
        let lit_scalar = lit_scalar.clamp(0, 0x1000);

        // Add colour × lit_scalar to accumulator. Colour is 0..255,
        // lit_scalar is 0..0x1000 = 0..4096. Product up to ~1M
        // -- divide by 4096 to bring back into 0..255 range.
        r += (light.colour.0 as i32 * lit_scalar) >> 12;
        g += (light.colour.1 as i32 * lit_scalar) >> 12;
        b += (light.colour.2 as i32 * lit_scalar) >> 12;
    }

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

// ----------------------------------------------------------------------
// Scene impl
// ----------------------------------------------------------------------

impl Scene for Lighting {
    fn init(&mut self, _ctx: &mut Ctx) {
        scene::set_screen_offset((SCREEN_W as i32 / 2) << 16, (SCREEN_H as i32 / 2) << 16);
        scene::set_projection_plane(PROJ_H);
        scene::set_avsz_weights(0x155, 0xAA);
        self.font = Some(FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT));
    }

    fn update(&mut self, ctx: &mut Ctx) {
        self.tri_count = 0;

        // Each light orbits in the XZ plane at a slightly different
        // radius + speed + Y height + phase. All four together give
        // the sense of "lights dancing" through the room. Orbits
        // sized to sweep past all 4 cubes over their cycle so every
        // cube sees every colour at some point.
        let phases: [(i16, u16, i16, u16); NUM_LIGHTS] = [
            // (orbit_radius, frames_per_rev_scale, y_height, phase_deg)
            (0x2400, 16, 0x0400, 0),
            (0x2800, 20, 0x0C00, 1024),  // 90°
            (0x1E00, 24, -0x0400, 2048), // 180°
            (0x2C00, 18, 0x0800, 3072),  // 270°
        ];
        for i in 0..NUM_LIGHTS {
            let (r, speed, y, phase) = phases[i];
            let angle = (ctx.frame.wrapping_mul(speed as u32) as u16).wrapping_add(phase);
            // Q1.12 sin/cos scaled by Q3.12 radius → Q4.24 position,
            // shift right 12 to bring back to Q3.12.
            let x = ((sincos::cos_q12(angle) * r as i32) >> 12) as i16;
            let z = ((sincos::sin_q12(angle) * r as i32) >> 12) as i16;
            self.lights[i].pos = Vec3I16::new(x, y, (WORLD_Z as i16).wrapping_add(z));
        }
    }

    fn render(&mut self, ctx: &mut Ctx) {
        self.build_frame_ot(ctx.frame);
        let font = self.font.as_ref().expect("font uploaded in init");
        self.draw_hud(font, ctx.frame);
    }
}

impl Lighting {
    fn build_frame_ot(&mut self, frame: u32) {
        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut gouraud = unsafe { PrimitiveArena::new(&mut GOURAUD_TRIS) };
        let mut markers = unsafe { PrimitiveArena::new(&mut LIGHT_MARKERS) };
        let mut backgrounds = unsafe { PrimitiveArena::new(core::slice::from_mut(&mut BG_QUAD)) };

        let cube = Mesh::from_bytes(CUBE_BLOB).expect("cube blob");

        // Backmost slot -- dark-room gradient backdrop.
        let Some(bg) = backgrounds.push(QuadGouraud::new(
            [(0, 0), (SCREEN_W, 0), (0, SCREEN_H), (SCREEN_W, SCREEN_H)],
            [(16, 10, 24), (16, 10, 24), (4, 2, 8), (4, 2, 8)],
        )) else {
            return;
        };
        ot.add_packet(BG_SLOT, bg);

        let mesh_options = GouraudMeshOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE);
        let mut world_pass =
            unsafe { GouraudRenderPass::new(&mut ot, &mut gouraud, &mut GOURAUD_COMMANDS) };

        // World band -- all cube instances, CPU-lit + GTE-projected.
        for instance in &CUBE_LAYOUT {
            // Per-instance: compose Y-spin rotation × identity (no
            // other rotation). `frame * spin` gives the current angle.
            let angle = (frame.wrapping_mul(instance.y_spin_per_frame as u32) & 0xFFFF) as u16;
            let rot = Mat3I16::rotate_y(angle);

            // Build the full actor pose and upload -- one call
            // replaces the old hand-rolled "scale rotation → load
            // rotation → load translation" three-step. The scale
            // folds into the rotation matrix so the GTE's RTPS
            // pass does scale + rotate + translate in one go.
            let actor = ActorTransform::at(instance.position)
                .with_rotation(rot)
                .with_scale_q12(instance.scale);
            actor.load_gte();

            // CPU path also wants the *same* scaled rotation matrix
            // the GTE is using -- `actor.scaled_rotation()` returns
            // exactly what `load_gte` uploaded, so the CPU world-
            // space positions line up bit-for-bit with the GTE's
            // projected screen-space output.
            let rot_scaled = actor.scaled_rotation();
            let cube_proj = unsafe { &mut CUBE_PROJ };
            for vi in 0..cube.vert_count() {
                let vl = cube.vertex(vi);
                let nl = cube.vertex_normal(vi).unwrap_or(Vec3I16::new(0, 0x1000, 0));

                // World-space position: rot_scaled × local + translation.
                let wx = mat_row_dot(&rot_scaled, 0, vl) + instance.position.x;
                let wy = mat_row_dot(&rot_scaled, 1, vl) + instance.position.y;
                let wz = mat_row_dot(&rot_scaled, 2, vl) + instance.position.z;
                let world_pos = Vec3I16::new(
                    (wx.clamp(i16::MIN as i32, i16::MAX as i32)) as i16,
                    (wy.clamp(i16::MIN as i32, i16::MAX as i32)) as i16,
                    (wz.clamp(i16::MIN as i32, i16::MAX as i32)) as i16,
                );

                // World-space normal: rotation only (no translation,
                // no scale -- scale would stretch the normal length).
                let nrot = Vec3I16::new(
                    mat_row_dot(&rot, 0, nl).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    mat_row_dot(&rot, 1, nl).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    mat_row_dot(&rot, 2, nl).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                );

                let (r, g, b) = light_vertex(&self.lights, world_pos, nrot);

                // Project via GTE (actor transform already loaded).
                let p = scene::project_vertex(vl);
                cube_proj[vi as usize] = ProjectedLit {
                    sx: p.sx,
                    sy: p.sy,
                    sz: p.sz,
                    r,
                    g,
                    b,
                };
            }

            let cube_stats = world_pass.submit_projected_mesh(&cube, cube_proj, mesh_options);
            self.tri_count = self
                .tri_count
                .saturating_add(cube_stats.submitted_triangles);
        }
        world_pass.flush();

        // Foreground -- visible light position markers (small bright
        // squares at each light's projected screen position, colour
        // matches the light's colour). Use a direct-to-screen path --
        // identity rotation + zero translation -- to project light
        // positions cleanly.
        scene::load_rotation(&Mat3I16::IDENTITY);
        scene::load_translation(Vec3I32::new(0, 0, 0));
        for light in &self.lights {
            let p = scene::project_vertex(light.pos);
            if p.sx < 4 || p.sx > SCREEN_W - 4 || p.sy < 4 || p.sy > SCREEN_H - 4 {
                continue;
            }
            let Some(marker) = markers.push(RectFlat::new(
                p.sx - 2,
                p.sy - 2,
                4,
                4,
                light.colour.0,
                light.colour.1,
                light.colour.2,
            )) else {
                break;
            };
            ot.add_packet(LIGHT_MARKER_SLOT, marker);
        }

        ot.submit();
    }

    fn draw_hud(&self, font: &FontAtlas, frame: u32) {
        font.draw_text(4, 4, "SHOWCASE-LIGHTS", (220, 220, 250));
        // "4 lights" = 8 chars × 8 = 64 px. Anchor at W - 64 - 4.
        font.draw_text(SCREEN_W - 68, 4, "4 lights", (180, 180, 220));

        font.draw_text(4, SCREEN_H - 20, "frame", (160, 160, 200));
        let frame_hex = u16_hex((frame & 0xFFFF) as u16);
        font.draw_text(
            4 + 8 * 6,
            SCREEN_H - 20,
            frame_hex.as_str(),
            (200, 240, 160),
        );

        font.draw_text(SCREEN_W / 2 - 8 * 3, SCREEN_H - 20, "tri", (160, 160, 200));
        let tri = u16_hex(self.tri_count);
        font.draw_text(
            SCREEN_W / 2 + 8,
            SCREEN_H - 20,
            tri.as_str(),
            (240, 200, 160),
        );

        font.draw_text(SCREEN_W - 100, SCREEN_H - 20, "cubes", (160, 160, 200));
        let cubes = u16_hex(NUM_CUBES as u16);
        font.draw_text(
            SCREEN_W - 52,
            SCREEN_H - 20,
            cubes.as_str(),
            (160, 200, 240),
        );
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

// The uniform-scale matrix helper that used to live here is now
// engine-owned as `ActorTransform::scaled_rotation()` -- call sites
// building their own scaled rotation should go through the actor
// type, which guarantees the CPU and GTE paths see bit-identical
// matrices.

/// Dot product of matrix row `r` with vector `v` in Q3.12; result
/// in i32 (un-truncated range).
#[inline]
fn mat_row_dot(m: &Mat3I16, r: usize, v: Vec3I16) -> i32 {
    let sum = (m.m[r][0] as i32) * (v.x as i32)
        + (m.m[r][1] as i32) * (v.y as i32)
        + (m.m[r][2] as i32) * (v.z as i32);
    sum >> 12
}

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() -> ! {
    let mut scene = Lighting {
        tri_count: 0,
        // Initial positions filled in on the first `update`. Colours
        // are highly saturated so a cube sitting near one light reads
        // as clearly "that colour" rather than a neutral blend.
        lights: [
            PointLight::new(Vec3I16::ZERO, (255, 40, 40)),  // red
            PointLight::new(Vec3I16::ZERO, (40, 255, 60)),  // green
            PointLight::new(Vec3I16::ZERO, (60, 100, 255)), // blue
            PointLight::new(Vec3I16::ZERO, (255, 220, 40)), // yellow
        ],
        font: None,
    };
    let config = Config {
        screen_w: SCREEN_W as u16,
        screen_h: SCREEN_H as u16,
        clear_color: (0, 0, 0),
        ..Config::default()
    };
    App::run(config, &mut scene);
}

// ----------------------------------------------------------------------
// no_std hex formatter
// ----------------------------------------------------------------------

fn u16_hex(v: u16) -> HexU16 {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = [0u8; 6];
    out[0] = b'0';
    out[1] = b'x';
    out[2] = HEX[((v >> 12) & 0xF) as usize];
    out[3] = HEX[((v >> 8) & 0xF) as usize];
    out[4] = HEX[((v >> 4) & 0xF) as usize];
    out[5] = HEX[(v & 0xF) as usize];
    HexU16(out)
}

struct HexU16([u8; 6]);
impl HexU16 {
    fn as_str(&self) -> &str {
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}
