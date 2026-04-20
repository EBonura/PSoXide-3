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
//! The two objects rotate independently on composite axes with
//! a GTE-projected starfield behind them, `psx-fx` sparks
//! drifting through, a deep-space gradient backdrop, and a
//! BASIC_8×16 HUD tracking frame count + live triangle count +
//! star count.
//!
//! Validates the end-to-end 3D pipeline under real load:
//!
//!   GTE project → back-face cull → OT depth slot → DMA submit
//!
//! with ~200 rendered triangles per frame after culling.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::Mesh;
use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_fx::{LcgRng, ParticlePool, ShakeState};
use psx_gpu::framebuf::FrameBuffer;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat, TriGouraud};
use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_gte::lighting::{Light, LightRig};
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
const OT_DEPTH: usize = 8;

/// GTE focal length (H register).
const PROJ_H: u16 = 280;
/// Default camera setback.
const WORLD_Z: i32 = 0x3400;

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

const SUZANNE_POS: Vec3I32 = Vec3I32::new(-0x1300, 0, WORLD_Z);
const TEAPOT_POS: Vec3I32 = Vec3I32::new(0x1400, 0, WORLD_Z);

// ----------------------------------------------------------------------
// Light rig (world-space)
// ----------------------------------------------------------------------

/// Three directional lights arranged for studio-ish illumination:
///   - Key: warm, above-right, strong
///   - Fill: cool, above-left, moderate
///   - Rim: neutral-cool, behind, narrow
///
/// Directions are Q3.12 unit-ish vectors pointing FROM the surface
/// TO the light. They're `const`-evaluated so the rig lives in
/// .rodata — no per-frame rebuild cost.
/// "Studio" rig in its REFERENCE frame (facing -Z, camera-front).
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

struct Scene {
    frame: u32,
    /// Live post-cull triangle count.
    tri_count: u16,
    rng: LcgRng,
    sparks: ParticlePool<32>,
    shake: ShakeState,
}

static mut SCENE: Scene = Scene {
    frame: 0,
    tri_count: 0,
    rng: LcgRng::new(0xFACE_1337),
    sparks: ParticlePool::new(),
    shake: ShakeState::new(),
};

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
static mut SCENE_RECTS: [RectFlat; 128] = [const {
    RectFlat::new(0, 0, 0, 0, 0, 0, 0)
}; 128];

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
/// Tuple layout: (sx, sy, sz, r, g, b). The three colour channels
/// are the GTE's NCCS output for this vertex + the current
/// object-local light rig — they drive the Gouraud interpolator
/// on each triangle.
type VertProj = (i16, i16, u16, u8, u8, u8);
const EMPTY_VERT: VertProj = (0, 0, 0, 0, 0, 0);

static mut SUZANNE_PROJ: [VertProj; 128] = [EMPTY_VERT; 128];
static mut TEAPOT_PROJ: [VertProj; 128] = [EMPTY_VERT; 128];

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    let mut fb = FrameBuffer::new(SCREEN_W as u16, SCREEN_H as u16);
    gpu::set_draw_area(0, 0, (SCREEN_W - 1) as u16, (SCREEN_H - 1) as u16);
    gpu::set_draw_offset(0, 0);

    scene::set_screen_offset((SCREEN_W as i32 / 2) << 16, (SCREEN_H as i32 / 2) << 16);
    scene::set_projection_plane(PROJ_H);
    scene::set_avsz_weights(0x155, 0xAA);

    let font = FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT);

    init_starfield();

    loop {
        update_scene();

        fb.clear(0, 0, 0);

        build_frame_ot();
        submit_frame_ot();

        draw_hud(&font);

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
    }
}

// ----------------------------------------------------------------------
// Update
// ----------------------------------------------------------------------

fn spawn_star(rng: &mut LcgRng, idx: usize) {
    let x = rng.signed(STAR_SPREAD_XY);
    let y = rng.signed(STAR_SPREAD_XY);
    let z = STAR_MAX_Z - (rng.next() as i16 & 0x0FFF);
    unsafe {
        STARS[idx] = Vec3I16::new(x, y, z);
    }
}

fn init_starfield() {
    let s = unsafe { &mut SCENE };
    for i in 0..STAR_COUNT {
        let x = s.rng.signed(STAR_SPREAD_XY);
        let y = s.rng.signed(STAR_SPREAD_XY);
        let z = (s.rng.next() as i16).abs() & 0x7FFF;
        unsafe {
            STARS[i] = Vec3I16::new(x, y, z);
        }
    }
}

fn update_scene() {
    let s = unsafe { &mut SCENE };
    s.frame = s.frame.wrapping_add(1);
    s.tri_count = 0;

    for i in 0..STAR_COUNT {
        let z = unsafe { STARS[i].z };
        let new_z = z.wrapping_sub(STAR_Z_SPEED);
        if new_z < 0x0200 {
            spawn_star(&mut s.rng, i);
        } else {
            unsafe {
                STARS[i].z = new_z;
            }
        }
    }

    if s.frame % 12 == 0 {
        let x = SCREEN_W / 2 + s.rng.signed(SCREEN_W / 2 - 20);
        let y = SCREEN_H / 2 + s.rng.signed(SCREEN_H / 2 - 30);
        let color = (
            100 + (s.rng.next() & 0x7F) as u8,
            100 + (s.rng.next() & 0x7F) as u8,
            140 + (s.rng.next() & 0x7F) as u8,
        );
        s.sparks.spawn_burst(&mut s.rng, (x, y), color, 2, 16, 50);
    }
    s.sparks.update(1);
}

// ----------------------------------------------------------------------
// Render — build OT
// ----------------------------------------------------------------------

fn build_frame_ot() {
    let ot = unsafe { &mut OT };
    let rects = unsafe { &mut SCENE_RECTS };
    let gouraud = unsafe { &mut GOURAUD_TRIS };
    let bg = unsafe { &mut BG_QUAD };
    ot.clear();

    let s = unsafe { &mut SCENE };

    let (shake_dx, shake_dy) = s.shake.tick();

    // Slot 7 (back) — deep-space gradient.
    *bg = QuadGouraud::new(
        [
            (0, 0),
            (SCREEN_W, 0),
            (0, SCREEN_H),
            (SCREEN_W, SCREEN_H),
        ],
        [
            (10, 6, 32),
            (10, 6, 32),
            (2, 1, 8),
            (2, 1, 8),
        ],
    );
    ot.add(7, bg, QuadGouraud::WORDS);

    // Slot 6 — starfield with a slow whole-scene roll for parallax.
    let scene_roll = Mat3I16::rotate_z((s.frame.wrapping_mul(2) & 0xFFFF) as u16);
    scene::load_rotation(&scene_roll);
    scene::load_translation(Vec3I32::new(0, 0, 0));
    let mut rect_idx = 0;
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
        if rect_idx >= rects.len() {
            break;
        }
        rects[rect_idx] = RectFlat::new(
            p.sx + shake_dx,
            p.sy + shake_dy,
            1,
            1,
            brightness,
            brightness,
            brightness,
        );
        ot.add(6, &mut rects[rect_idx], RectFlat::WORDS);
        rect_idx += 1;
    }

    // Pre-project both meshes' vertices once, then walk triangle
    // lists. Topologically both meshes share verts heavily — this
    // avoids 4-6× repeated GTE loads per vertex.
    let mut tri_idx = 0;

    // Parse both meshes at frame-start. Parsing is effectively
    // free (zero-copy, just bounds-checks + slice arithmetic on
    // the `static` blobs). We could cache these but the savings
    // are below the noise floor.
    let suzanne = Mesh::from_bytes(SUZANNE_BLOB).expect("suzanne blob");
    let teapot = Mesh::from_bytes(TEAPOT_BLOB).expect("teapot blob");

    // World-space light rig for this frame — base rig rotated
    // around Y at the configured orbit rate so both meshes
    // eventually see every lighting angle. Compose this once per
    // frame; each object's `for_object` applies on top to take us
    // into local space.
    let light_orbit = Mat3I16::rotate_y(
        ((s.frame as u32).wrapping_mul(LIGHT_ORBIT_PER_FRAME as u32) & 0xFFFF) as u16,
    );
    let scene_lights = BASE_LIGHTS.rotated(&light_orbit);

    // --- SUZANNE (slot 5) — two-axis tumble, lit by scene rig ---
    let suz_rot = Mat3I16::rotate_y((s.frame.wrapping_mul(3) & 0xFFFF) as u16)
        .mul(&Mat3I16::rotate_x((s.frame.wrapping_mul(2) & 0xFFFF) as u16));
    scene::load_rotation(&suz_rot);
    scene::load_translation(SUZANNE_POS);
    // Rotate world-space lights into Suzanne's local frame + upload.
    scene_lights.for_object(&suz_rot).load();
    let suz_proj = unsafe { &mut SUZANNE_PROJ };
    for i in 0..suzanne.vert_count() {
        let v = suzanne.vertex(i as u8);
        let n = suzanne.vertex_normal(i as u8).unwrap_or(Vec3I16::ZERO);
        // Per-face palette still lives in the blob; we use it as
        // the material RGBC so the lit output carries the face's
        // tint. We average this per-vertex: assign vert colour
        // from the first face that references it (good enough —
        // the gouraud interpolator blurs any mismatch anyway).
        let material = suzanne_vertex_material(&suzanne, i as u8);
        let p = psx_gte::lighting::project_lit(v, n, material);
        suz_proj[i as usize] = (
            p.sx + shake_dx,
            p.sy + shake_dy,
            p.sz,
            p.r, p.g, p.b,
        );
    }
    for face_idx in 0..suzanne.face_count() {
        if tri_idx >= gouraud.len() {
            break;
        }
        let (ia, ib, ic) = suzanne.face(face_idx);
        let (v0, v1, v2) = (
            suz_proj[ia as usize],
            suz_proj[ib as usize],
            suz_proj[ic as usize],
        );
        if back_facing(v0, v1, v2) {
            continue;
        }
        gouraud[tri_idx] = TriGouraud::new(
            [(v0.0, v0.1), (v1.0, v1.1), (v2.0, v2.1)],
            [(v0.3, v0.4, v0.5), (v1.3, v1.4, v1.5), (v2.3, v2.4, v2.5)],
        );
        ot.add(5, &mut gouraud[tri_idx], TriGouraud::WORDS);
        tri_idx += 1;
        s.tri_count += 1;
    }

    // --- TEAPOT (slot 4) — opposite tumble, lit by same rig ---
    let tea_rot = Mat3I16::rotate_y((s.frame.wrapping_mul(4) & 0xFFFF).wrapping_neg() as u16)
        .mul(&Mat3I16::rotate_z((s.frame.wrapping_mul(2) & 0xFFFF) as u16));
    scene::load_rotation(&tea_rot);
    scene::load_translation(TEAPOT_POS);
    scene_lights.for_object(&tea_rot).load();
    let tea_proj = unsafe { &mut TEAPOT_PROJ };
    for i in 0..teapot.vert_count() {
        let v = teapot.vertex(i as u8);
        let n = teapot.vertex_normal(i as u8).unwrap_or(Vec3I16::ZERO);
        let material = suzanne_vertex_material(&teapot, i as u8);
        let p = psx_gte::lighting::project_lit(v, n, material);
        tea_proj[i as usize] = (
            p.sx + shake_dx,
            p.sy + shake_dy,
            p.sz,
            p.r, p.g, p.b,
        );
    }
    for face_idx in 0..teapot.face_count() {
        if tri_idx >= gouraud.len() {
            break;
        }
        let (ia, ib, ic) = teapot.face(face_idx);
        let (v0, v1, v2) = (
            tea_proj[ia as usize],
            tea_proj[ib as usize],
            tea_proj[ic as usize],
        );
        if back_facing(v0, v1, v2) {
            continue;
        }
        gouraud[tri_idx] = TriGouraud::new(
            [(v0.0, v0.1), (v1.0, v1.1), (v2.0, v2.1)],
            [(v0.3, v0.4, v0.5), (v1.3, v1.4, v1.5), (v2.3, v2.4, v2.5)],
        );
        ot.add(4, &mut gouraud[tri_idx], TriGouraud::WORDS);
        tri_idx += 1;
        s.tri_count += 1;
    }

    // Slot 3 — sparks (in front of meshes).
    let spark_budget = rects.len().saturating_sub(rect_idx);
    let _ = s.sparks.render_into_ot(
        ot,
        &mut rects[rect_idx..rect_idx + spark_budget],
        3,
        (shake_dx, shake_dy),
    );
}

fn submit_frame_ot() {
    unsafe {
        OT.submit();
    }
}

/// 2D cross-product test — triangles wound clockwise on screen
/// are back-facing and get culled.
fn back_facing(v0: VertProj, v1: VertProj, v2: VertProj) -> bool {
    let ax = (v1.0 as i32) - (v0.0 as i32);
    let ay = (v1.1 as i32) - (v0.1 as i32);
    let bx = (v2.0 as i32) - (v0.0 as i32);
    let by = (v2.1 as i32) - (v0.1 as i32);
    (ax * by - ay * bx) <= 0
}

/// Sample a per-vertex material colour for a lit-mesh vertex.
///
/// The cooked mesh stores per-*face* colours, not per-vertex.
/// For a Gouraud-lit render we need a per-vertex material — we
/// walk the triangle list and take the colour from the first
/// face that references this vertex. Vertices shared between
/// differently-coloured faces get the first-seen face's tint;
/// the gouraud interpolator then blurs the result so it reads
/// as a smooth palette.
fn suzanne_vertex_material(mesh: &Mesh, vert: u8) -> (u8, u8, u8) {
    for f in 0..mesh.face_count() {
        let (a, b, c) = mesh.face(f);
        if a == vert || b == vert || c == vert {
            return mesh.face_color(f).unwrap_or((128, 128, 128));
        }
    }
    (128, 128, 128)
}

// ----------------------------------------------------------------------
// HUD
// ----------------------------------------------------------------------

fn draw_hud(font: &FontAtlas) {
    let s = unsafe { &SCENE };

    font.draw_text(4, 4, "SHOWCASE-3D", (220, 220, 250));
    // Right side lists the two hero models on two lines — clean
    // attribution without crowding the top bar.
    font.draw_text(SCREEN_W - 80, 4, "suzanne", (220, 160, 80));
    font.draw_text(SCREEN_W - 64, 20, "teapot", (120, 200, 240));

    font.draw_text(4, SCREEN_H - 20, "frame", (160, 160, 200));
    let frame = u16_hex((s.frame & 0xFFFF) as u16);
    font.draw_text(4 + 8 * 6, SCREEN_H - 20, frame.as_str(), (200, 240, 160));

    font.draw_text(SCREEN_W / 2 - 8 * 3, SCREEN_H - 20, "tri", (160, 160, 200));
    let tri = u16_hex(s.tri_count);
    font.draw_text(SCREEN_W / 2 + 8, SCREEN_H - 20, tri.as_str(), (240, 200, 160));

    font.draw_text(SCREEN_W - 100, SCREEN_H - 20, "stars", (160, 160, 200));
    let stars = u16_hex(STAR_COUNT as u16);
    font.draw_text(
        SCREEN_W - 52,
        SCREEN_H - 20,
        stars.as_str(),
        (160, 200, 240),
    );
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
