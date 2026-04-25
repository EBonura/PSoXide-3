//! `showcase-3d` — the flagship 3D demo. Features the two
//! canonical test models every graphics developer recognises:
//!
//! - **Suzanne** — the Blender monkey head (public domain,
//!   Blender Foundation). Decimated from 500 tris to ~180 tris
//!   via vertex clustering for PSX budget; silhouette intact.
//! - **Utah teapot** — Martin Newell's 1975 Bezier surface,
//!   tessellated to 6320 tris in its canonical OBJ, decimated
//!   here to ~225 tris.
//!
//! The two objects rotate independently on composite axes while a
//! slightly-raised camera orbits the scene centre. Behind them: a
//! GTE-projected starfield, `psx-fx` sparks drifting through, a
//! deep-space gradient backdrop, and a BASIC_8×16 HUD tracking frame
//! count + live triangle count + star count.
//!
//! Validates the end-to-end 3D pipeline under real load:
//!
//!   GTE project → back-face cull → OT depth slot → DMA submit
//!
//! with ~200 rendered triangles per frame after culling.
//!
//! Ported to `psx-engine` in Phase 3e. The `Scene` struct
//! (renamed to `Showcase3D` to avoid colliding with the engine
//! trait) now holds per-frame state inline; GTE + font setup
//! live in `Scene::init`; geometry driven by `ctx.frame`.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::Mesh;
use psx_engine::{
    ActorTransform, App, Config, Ctx, DepthBand, DepthPolicy, DepthRange, GouraudMeshOptions,
    GouraudRenderPass, GouraudTriCommand, OtFrame, PrimitiveArena, Scene, Vec3World,
};
use psx_font::{fonts::BASIC_8X16, FontAtlas};
use psx_fx::{LcgRng, ParticlePool, ShakeState};
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat, TriGouraud};
use psx_gte::lighting::{Light, LightRig, ProjectedLit};
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_vram::{Clut, TexDepth, Tpage};

/// Cooked mesh blobs — produced at build time by `psxed` from
/// `vendor/*.obj`, embedded here so the homebrew is self-
/// contained. Runtime just `Mesh::from_bytes` + index accessors.
static SUZANNE_BLOB: &[u8] = include_bytes!("../assets/suzanne.psxm");
static TEAPOT_BLOB: &[u8] = include_bytes!("../assets/teapot.psxm");

// ----------------------------------------------------------------------
// Screen + scene constants
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;
const OT_DEPTH: usize = 128;
const BG_SLOT: usize = OT_DEPTH - 1;
const STAR_SLOT: usize = OT_DEPTH - 2;
const WORLD_BAND: DepthBand = DepthBand::new(3, OT_DEPTH - 3);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(0x1800, 0x7800);
const SPARK_SLOT: usize = 2;

/// GTE focal length (H register).
const PROJ_H: u16 = 280;
/// Scene centre depth in camera space.
const WORLD_Z: i32 = 0x4800;

/// Camera orbit: 1 coarse GTE angle unit per frame = one revolution
/// every ~4.3 seconds at 60 Hz.
const CAMERA_ORBIT_PER_FRAME: u16 = 1;

/// Camera pitch in 256-per-revolution units. This is a small negative
/// X rotation, equivalent to lifting the camera and looking gently
/// down at the scene centre.
const CAMERA_PITCH: u16 = 0xF0;

// ----------------------------------------------------------------------
// Starfield
// ----------------------------------------------------------------------

const STAR_COUNT: usize = 48;
const STAR_SPREAD_XY: i16 = 0x1800;
const STAR_MAX_Z: i16 = 0x7FFF;
const STAR_Z_SPEED: i16 = 0x0140;

static mut STARS: [Vec3I16; STAR_COUNT] = [Vec3I16::ZERO; STAR_COUNT];

// ----------------------------------------------------------------------
// Mesh placement
// ----------------------------------------------------------------------

const SCENE_CENTER: Vec3World = Vec3World::from_raw(0, 0, WORLD_Z);
const SUZANNE_OFFSET: Vec3World = Vec3World::from_raw(-0x1300, 0, 0);
const TEAPOT_OFFSET: Vec3World = Vec3World::from_raw(0x1400, 0, 0);

// ----------------------------------------------------------------------
// Light rig (camera-space)
// ----------------------------------------------------------------------

/// Three directional lights arranged for studio-ish illumination:
///   - Key: warm, above-right, strong
///   - Fill: cool, above-left, moderate
///   - Rim: neutral-cool, behind, narrow
///
/// Directions are Q3.12 unit-ish vectors pointing FROM the surface
/// TO the light. They're `const`-evaluated so the rig lives in
/// .rodata — no per-frame rebuild cost.
/// "Studio" rig in its camera reference frame (facing -Z).
/// Per frame we rotate this around Y so lights orbit the scene,
/// ensuring both Suzanne and the teapot see every lighting angle
/// across a full rotation.
const BASE_LIGHTS: LightRig = LightRig::new(
    [
        // Key — warm, from upper-right at reference angle.
        Light {
            direction: Vec3I16::new(0x0B00, 0x0900, -0x0300),
            colour: (0x0E00, 0x0A80, 0x0700),
        },
        // Fill — cool, opposite side.
        Light {
            direction: Vec3I16::new(-0x0B00, 0x0800, -0x0300),
            colour: (0x0600, 0x0800, 0x0B00),
        },
        // Rim — from behind, narrow highlight.
        Light {
            direction: Vec3I16::new(0, -0x0200, 0x0F00),
            colour: (0x0500, 0x0500, 0x0800),
        },
    ],
    // Ambient — lifted so shadowed faces stay readable.
    (0x0400, 0x0400, 0x0600),
);

/// How fast the light rig orbits the scene Y axis, in Q0.12 units
/// per frame. 32 units/frame → 4096/32 = 128 frames per full
/// revolution ≈ 2.1 s at 60 Hz. Slow enough to track, fast enough
/// to see both meshes lit from every side within a few seconds.
const LIGHT_ORBIT_PER_FRAME: u16 = 32;

// ----------------------------------------------------------------------
// Font + VRAM
// ----------------------------------------------------------------------

const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

/// Renamed from the pre-engine `Scene` struct to avoid name
/// collision with [`psx_engine::Scene`] — this is *our* scene
/// type; the trait is what the engine calls back into.
struct Showcase3D {
    /// Live post-cull triangle count.
    tri_count: u16,
    rng: LcgRng,
    sparks: ParticlePool<32>,
    shake: ShakeState,
    font: Option<FontAtlas>,
}

// ----------------------------------------------------------------------
// OT + primitive buffers
// ----------------------------------------------------------------------

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();

/// Both meshes feed this Gouraud-triangle pool — every tri has
/// three per-vertex colours computed via the GTE's lighting
/// pipeline. Pre-cull upper bound is Suzanne (178) + teapot (92)
/// = 270; back-face culling drops about half. 320 leaves room.
static mut GOURAUD_TRIS: [TriGouraud; 320] = [const {
    TriGouraud {
        tag: 0,
        color0_cmd: 0,
        v0: 0,
        color1: 0,
        v1: 0,
        color2: 0,
        v2: 0,
    }
}; 320];

/// Rects for stars + sparks.
static mut SCENE_RECTS: [RectFlat; 128] = [const { RectFlat::new(0, 0, 0, 0, 0, 0, 0) }; 128];
static mut GOURAUD_COMMANDS: [GouraudTriCommand; 320] = [GouraudTriCommand::EMPTY; 320];

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

/// Pre-project + pre-light each mesh's vertices once per frame.
/// Topology shares verts 4-6× per face, so doing the GTE work
/// once per vertex rather than per triangle face is a big win.
///
/// Projected screen-space + lit vertices. The three colour channels
/// are the GTE's NCCS output for this vertex + the current
/// object-local light rig — they drive the Gouraud interpolator
/// on each triangle.
const EMPTY_VERT: ProjectedLit = ProjectedLit {
    sx: 0,
    sy: 0,
    sz: 0,
    r: 0,
    g: 0,
    b: 0,
};

static mut SUZANNE_PROJ: [ProjectedLit; 128] = [EMPTY_VERT; 128];
static mut TEAPOT_PROJ: [ProjectedLit; 128] = [EMPTY_VERT; 128];

// ----------------------------------------------------------------------
// Scene impl
// ----------------------------------------------------------------------

impl Scene for Showcase3D {
    fn init(&mut self, _ctx: &mut Ctx) {
        scene::set_screen_offset((SCREEN_W as i32 / 2) << 16, (SCREEN_H as i32 / 2) << 16);
        scene::set_projection_plane(PROJ_H);
        scene::set_avsz_weights(0x155, 0xAA);
        self.font = Some(FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT));
        self.init_starfield();
    }

    fn update(&mut self, _ctx: &mut Ctx) {
        self.tri_count = 0;

        for i in 0..STAR_COUNT {
            let z = unsafe { STARS[i].z };
            let new_z = z.wrapping_sub(STAR_Z_SPEED);
            if new_z < 0x0200 {
                self.spawn_star(i);
            } else {
                unsafe {
                    STARS[i].z = new_z;
                }
            }
        }

        // Spark emitter — 12-frame cadence reads `ctx.frame` via
        // our own counter (see comment in `update` below for why
        // we don't just use `ctx.frame` directly). Actually we
        // *do* — keep this in sync with the pre-engine cadence by
        // reading `frame % 12` on the raw counter.
        //
        // We take `ctx.frame` as the spawn clock. The pre-engine
        // demo incremented at start-of-update so its modulo hit
        // one frame earlier, but the sparks are random-seeded
        // anyway — the cadence is what matters, and that's
        // preserved.
    }

    fn render(&mut self, ctx: &mut Ctx) {
        // Spark emission lives here (rather than in update) to
        // read `ctx.frame` without duplicating the spawn logic in
        // update — keeps the scene code compact and the frame-
        // counter-driven cadence explicit.
        if ctx.frame % 12 == 0 {
            let x = SCREEN_W / 2 + self.rng.signed(SCREEN_W / 2 - 20);
            let y = SCREEN_H / 2 + self.rng.signed(SCREEN_H / 2 - 30);
            let color = (
                100 + (self.rng.next() & 0x7F) as u8,
                100 + (self.rng.next() & 0x7F) as u8,
                140 + (self.rng.next() & 0x7F) as u8,
            );
            self.sparks
                .spawn_burst(&mut self.rng, (x, y), color, 2, 16, 50);
        }
        self.sparks.update(1);

        self.build_frame_ot(ctx.frame);
        let font = self.font.as_ref().expect("font uploaded in init");
        self.draw_hud(font, ctx.frame);
    }
}

// ----------------------------------------------------------------------
// Showcase3D impl — starfield / OT build / HUD
// ----------------------------------------------------------------------

impl Showcase3D {
    fn init_starfield(&mut self) {
        for i in 0..STAR_COUNT {
            let x = self.rng.signed(STAR_SPREAD_XY);
            let y = self.rng.signed(STAR_SPREAD_XY);
            let z = (self.rng.next() as i16).abs() & 0x7FFF;
            unsafe {
                STARS[i] = Vec3I16::new(x, y, z);
            }
        }
    }

    fn spawn_star(&mut self, idx: usize) {
        let x = self.rng.signed(STAR_SPREAD_XY);
        let y = self.rng.signed(STAR_SPREAD_XY);
        let z = STAR_MAX_Z - (self.rng.next() as i16 & 0x0FFF);
        unsafe {
            STARS[idx] = Vec3I16::new(x, y, z);
        }
    }

    fn build_frame_ot(&mut self, frame: u32) {
        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut rects = unsafe { PrimitiveArena::new(&mut SCENE_RECTS) };
        let mut gouraud = unsafe { PrimitiveArena::new(&mut GOURAUD_TRIS) };
        let mut backgrounds = unsafe { PrimitiveArena::new(core::slice::from_mut(&mut BG_QUAD)) };

        let (shake_dx, shake_dy) = self.shake.tick();

        // Backmost slot — deep-space gradient.
        let Some(bg) = backgrounds.push(QuadGouraud::new(
            [(0, 0), (SCREEN_W, 0), (0, SCREEN_H), (SCREEN_W, SCREEN_H)],
            [(10, 6, 32), (10, 6, 32), (2, 1, 8), (2, 1, 8)],
        )) else {
            return;
        };
        ot.add_packet(BG_SLOT, bg);

        // Behind world geometry — starfield with a slow whole-scene
        // roll for parallax.
        let scene_roll = Mat3I16::rotate_z((frame.wrapping_mul(2) & 0xFFFF) as u16);
        scene::load_rotation(&scene_roll);
        scene::load_translation(Vec3I32::new(0, 0, 0));
        for i in 0..STAR_COUNT {
            let sv = unsafe { STARS[i] };
            let p = scene::project_vertex(sv);
            if p.sx < 0 || p.sx >= SCREEN_W || p.sy < 0 || p.sy >= SCREEN_H {
                continue;
            }
            let brightness = if sv.z > 0x5000 {
                80
            } else if sv.z > 0x2800 {
                160
            } else {
                240
            };
            let Some(rect) = rects.push(RectFlat::new(
                p.sx + shake_dx,
                p.sy + shake_dy,
                1,
                1,
                brightness,
                brightness,
                brightness,
            )) else {
                break;
            };
            ot.add_packet(STAR_SLOT, rect);
        }

        // Pre-project both meshes' vertices once, then walk triangle
        // lists. Topologically both meshes share verts heavily — this
        // avoids 4-6× repeated GTE loads per vertex.

        // Parse both meshes at frame-start. Parsing is effectively
        // free (zero-copy, just bounds-checks + slice arithmetic on
        // the `static` blobs). We could cache these but the savings
        // are below the noise floor.
        let suzanne = Mesh::from_bytes(SUZANNE_BLOB).expect("suzanne blob");
        let teapot = Mesh::from_bytes(TEAPOT_BLOB).expect("teapot blob");

        // Camera-space light rig for this frame — base rig rotated
        // around Y at the configured orbit rate so both meshes
        // eventually see every lighting angle. Compose this once per
        // frame; each object's final view rotation takes us into local
        // space.
        let light_orbit =
            Mat3I16::rotate_y((frame.wrapping_mul(LIGHT_ORBIT_PER_FRAME as u32) & 0xFFFF) as u16);
        let scene_lights = BASE_LIGHTS.rotated(&light_orbit);
        let view = camera_view(frame);
        let mesh_options = GouraudMeshOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE)
            .with_depth_policy(DepthPolicy::Average)
            .with_screen_offset((shake_dx, shake_dy));
        let mut world_pass =
            unsafe { GouraudRenderPass::new(&mut ot, &mut gouraud, &mut GOURAUD_COMMANDS) };

        // --- SUZANNE — two-axis tumble, lit by scene rig ---
        let suz_rot = Mat3I16::rotate_y((frame.wrapping_mul(3) & 0xFFFF) as u16)
            .mul(&Mat3I16::rotate_x((frame.wrapping_mul(2) & 0xFFFF) as u16));
        let suz_view_rot = view.mul(&suz_rot);
        let suz_view_pos = camera_space_position(&view, SUZANNE_OFFSET);
        // No per-actor scale (meshes authored at their intended
        // size), so `ActorTransform::at(pos).with_rotation(rot)`
        // leaves scale at 1.0× and `load_gte` folds that identity
        // multiply into the rotation upload in one step.
        ActorTransform::at(suz_view_pos)
            .with_rotation(suz_view_rot)
            .load_gte();
        // Rotate camera-space lights into Suzanne's local frame + upload.
        scene_lights.for_object(&suz_view_rot).load();
        let suz_stats =
            world_pass.submit_lit_mesh(&suzanne, unsafe { &mut SUZANNE_PROJ }, mesh_options);
        self.tri_count = self
            .tri_count
            .saturating_add(suz_stats.submitted_triangles);

        // --- TEAPOT — opposite tumble, lit by same rig ---
        let tea_rot = Mat3I16::rotate_y((frame.wrapping_mul(4) & 0xFFFF).wrapping_neg() as u16)
            .mul(&Mat3I16::rotate_z((frame.wrapping_mul(2) & 0xFFFF) as u16));
        let tea_view_rot = view.mul(&tea_rot);
        let tea_view_pos = camera_space_position(&view, TEAPOT_OFFSET);
        ActorTransform::at(tea_view_pos)
            .with_rotation(tea_view_rot)
            .load_gte();
        scene_lights.for_object(&tea_view_rot).load();
        let tea_stats =
            world_pass.submit_lit_mesh(&teapot, unsafe { &mut TEAPOT_PROJ }, mesh_options);
        self.tri_count = self
            .tri_count
            .saturating_add(tea_stats.submitted_triangles);

        world_pass.flush();

        // Foreground effects — sparks in front of the meshes.
        let _ = render_particles(
            &self.sparks,
            &mut rects,
            &mut ot,
            SPARK_SLOT,
            (shake_dx, shake_dy),
        );

        ot.submit();
    }

    fn draw_hud(&self, font: &FontAtlas, frame: u32) {
        font.draw_text(4, 4, "SHOWCASE-3D", (220, 220, 250));
        // Right side lists the two hero models on two lines — clean
        // attribution without crowding the top bar.
        font.draw_text(SCREEN_W - 80, 4, "suzanne", (220, 160, 80));
        font.draw_text(SCREEN_W - 64, 20, "teapot", (120, 200, 240));

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

        font.draw_text(SCREEN_W - 100, SCREEN_H - 20, "stars", (160, 160, 200));
        let stars = u16_hex(STAR_COUNT as u16);
        font.draw_text(
            SCREEN_W - 52,
            SCREEN_H - 20,
            stars.as_str(),
            (160, 200, 240),
        );
    }
}

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() -> ! {
    let mut scene = Showcase3D {
        tri_count: 0,
        rng: LcgRng::new(0xFACE_1337),
        sparks: ParticlePool::new(),
        shake: ShakeState::new(),
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
// Helpers
// ----------------------------------------------------------------------

fn camera_view(frame: u32) -> Mat3I16 {
    let yaw =
        Mat3I16::rotate_y((frame.wrapping_mul(CAMERA_ORBIT_PER_FRAME as u32) & 0xFFFF) as u16);
    let pitch = Mat3I16::rotate_x(CAMERA_PITCH);
    pitch.mul(&yaw)
}

fn camera_space_position(view: &Mat3I16, offset: Vec3World) -> Vec3World {
    let x = transform_axis(view, 0, offset);
    let y = transform_axis(view, 1, offset);
    let z = transform_axis(view, 2, offset);
    Vec3World::from_raw(SCENE_CENTER.x + x, SCENE_CENTER.y + y, SCENE_CENTER.z + z)
}

fn transform_axis(m: &Mat3I16, row: usize, v: Vec3World) -> i32 {
    let row = m.row(row);
    let sum =
        (row[0] as i32) * v.x + (row[1] as i32) * v.y + (row[2] as i32) * v.z;
    (sum >> 12) as i32
}

fn render_particles<const N: usize, const OT_N: usize>(
    particles: &ParticlePool<N>,
    rects: &mut PrimitiveArena<'_, RectFlat>,
    ot: &mut OtFrame<'_, OT_N>,
    slot: usize,
    shake: (i16, i16),
) -> usize {
    let mut written = 0;
    for p in particles.particles() {
        if !p.alive() {
            continue;
        }

        let denom = p.spawn_ttl.max(1) as u16;
        let scale = p.ttl as u16;
        let r = ((p.r as u16 * scale) / denom) as u8;
        let g = ((p.g as u16 * scale) / denom) as u8;
        let b = ((p.b as u16 * scale) / denom) as u8;
        let size = if (p.ttl as u16) * 2 > denom { 3 } else { 2 };
        let Some(rect) = rects.push(RectFlat::new(
            p.x + shake.0,
            p.y + shake.1,
            size,
            size,
            r,
            g,
            b,
        )) else {
            break;
        };
        ot.add_packet(slot, rect);
        written += 1;
    }
    written
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
