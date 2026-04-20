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

mod meshes;

use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_fx::{LcgRng, ParticlePool, ShakeState};
use psx_gpu::framebuf::FrameBuffer;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat, TriFlat};
use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_vram::{Clut, TexDepth, Tpage};

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

/// Both meshes feed this flat-triangle pool. Pre-cull upper bound
/// is Suzanne (178) + teapot (225) = 403; culling drops ~half.
/// 256 gives generous headroom.
static mut FLAT_TRIS: [TriFlat; 256] = [const {
    TriFlat {
        tag: 0,
        color_cmd: 0,
        v0: 0,
        v1: 0,
        v2: 0,
    }
}; 256];

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

/// Pre-project each mesh's vertices once per frame. Suzanne has
/// ~84 verts (shared across 178 tris), teapot ~110. Sized to 128
/// each for headroom.
static mut SUZANNE_PROJ: [(i16, i16, u16); 128] = [(0, 0, 0); 128];
static mut TEAPOT_PROJ: [(i16, i16, u16); 128] = [(0, 0, 0); 128];

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
    let flats = unsafe { &mut FLAT_TRIS };
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
    let mut flat_idx = 0;

    // --- SUZANNE (slot 5) — two-axis tumble ---
    let suz_rot = Mat3I16::rotate_y((s.frame.wrapping_mul(3) & 0xFFFF) as u16)
        .mul(&Mat3I16::rotate_x((s.frame.wrapping_mul(2) & 0xFFFF) as u16));
    scene::load_rotation(&suz_rot);
    scene::load_translation(SUZANNE_POS);
    let suz_proj = unsafe { &mut SUZANNE_PROJ };
    for i in 0..meshes::SUZANNE_VERTS.len() {
        let p = scene::project_vertex(meshes::SUZANNE_VERTS[i]);
        suz_proj[i] = (p.sx + shake_dx, p.sy + shake_dy, p.sz);
    }
    for (face_idx, tri) in meshes::SUZANNE_TRIS.iter().enumerate() {
        if flat_idx >= flats.len() {
            break;
        }
        let (v0, v1, v2) = (
            suz_proj[tri[0] as usize],
            suz_proj[tri[1] as usize],
            suz_proj[tri[2] as usize],
        );
        if back_facing(v0, v1, v2) {
            continue;
        }
        let (r, g, b) = meshes::SUZANNE_FACE_COLORS[face_idx];
        flats[flat_idx] = TriFlat::new([(v0.0, v0.1), (v1.0, v1.1), (v2.0, v2.1)], r, g, b);
        ot.add(5, &mut flats[flat_idx], TriFlat::WORDS);
        flat_idx += 1;
        s.tri_count += 1;
    }

    // --- TEAPOT (slot 4) — opposite tumble direction ---
    let tea_rot = Mat3I16::rotate_y((s.frame.wrapping_mul(4) & 0xFFFF).wrapping_neg() as u16)
        .mul(&Mat3I16::rotate_z((s.frame.wrapping_mul(2) & 0xFFFF) as u16));
    scene::load_rotation(&tea_rot);
    scene::load_translation(TEAPOT_POS);
    let tea_proj = unsafe { &mut TEAPOT_PROJ };
    for i in 0..meshes::TEAPOT_VERTS.len() {
        let p = scene::project_vertex(meshes::TEAPOT_VERTS[i]);
        tea_proj[i] = (p.sx + shake_dx, p.sy + shake_dy, p.sz);
    }
    for (face_idx, tri) in meshes::TEAPOT_TRIS.iter().enumerate() {
        if flat_idx >= flats.len() {
            break;
        }
        let (v0, v1, v2) = (
            tea_proj[tri[0] as usize],
            tea_proj[tri[1] as usize],
            tea_proj[tri[2] as usize],
        );
        if back_facing(v0, v1, v2) {
            continue;
        }
        let (r, g, b) = meshes::TEAPOT_FACE_COLORS[face_idx];
        flats[flat_idx] = TriFlat::new([(v0.0, v0.1), (v1.0, v1.1), (v2.0, v2.1)], r, g, b);
        ot.add(4, &mut flats[flat_idx], TriFlat::WORDS);
        flat_idx += 1;
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
fn back_facing(v0: (i16, i16, u16), v1: (i16, i16, u16), v2: (i16, i16, u16)) -> bool {
    let ax = (v1.0 as i32) - (v0.0 as i32);
    let ay = (v1.1 as i32) - (v0.1 as i32);
    let bx = (v2.0 as i32) - (v0.0 as i32);
    let by = (v2.1 as i32) - (v0.1 as i32);
    (ax * by - ay * bx) <= 0
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
