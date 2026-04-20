//! `scene-lab` — the flagship 3D showcase, combining every SDK
//! graphics capability into a single animated scene.
//!
//! What's on screen:
//!
//! 1. **Starfield** — 48 points in world space receding toward the
//!    camera. When a star crosses the near plane it respawns at
//!    max depth. Projected through the same GTE pipeline as the
//!    solid meshes, just drawn as 1×1 rects at slot-7.
//! 2. **Three tumbling meshes** arranged in a triangular layout:
//!    - **Cube** (gouraud-shaded, 12 triangles). Per-vertex rainbow
//!      colours — the GPU gouraud interpolator produces the
//!      smooth surface shading.
//!    - **Pyramid** (4 flat-shaded triangles). Warm palette.
//!    - **Octahedron** (8 flat-shaded triangles). Cool palette.
//!
//!    Each mesh has its own rotation matrix tumbling at a unique
//!    rate so the scene stays lively.
//! 3. **OT depth sorting** — every triangle's GTE-computed average
//!    Z drives its OT slot, so meshes painted over each other
//!    resolve front-to-back correctly.
//! 4. **Floating sparks** — `psx_fx::ParticlePool` drifts coloured
//!    dots through the scene, spawned in subtle bursts from random
//!    screen positions.
//! 5. **HUD** — `BASIC_8X16` title, frame counter, primitive
//!    counter.
//!
//! Validates the end-to-end 3D pipeline:
//!
//!   GTE project → per-triangle depth → OT insert → DMA submit
//!
//! with psx-fx + psx-font composing on top for polish.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_fx::{LcgRng, ParticlePool, ShakeState};
use psx_gpu::framebuf::FrameBuffer;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat, TriFlat, TriGouraud};
use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_vram::{Clut, TexDepth, Tpage};

// ----------------------------------------------------------------------
// Screen + scene constants
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;

/// Depth slots used by the OT (0 = nearest / drawn last).
const OT_DEPTH: usize = 8;

/// Projection focal length for GTE (H register). 200 at z≈0x3000
/// gives a 1:1 pixel-per-object-unit feel at the centre of the
/// scene.
const PROJ_H: u16 = 260;

/// Camera setback — every vertex is translated by this amount
/// (negative so objects live at positive Z looking at the camera).
const WORLD_Z: i32 = 0x3400;

// ----------------------------------------------------------------------
// Starfield
// ----------------------------------------------------------------------

const STAR_COUNT: usize = 48;
/// Spread radius of stars in world space (Q3.12-ish). Bigger =
/// more angular coverage of the sky.
const STAR_SPREAD_XY: i16 = 0x1800;
/// Maximum star Z (how far out the field stretches).
const STAR_MAX_Z: i16 = 0x7FFF;
/// Per-frame Z approach speed — higher = faster flow.
const STAR_Z_SPEED: i16 = 0x0180;

/// Live stars stored as Vec3I16. `z` decrements each frame; when
/// it goes below a minimum, we respawn the star at max Z with a
/// fresh XY position.
static mut STARS: [Vec3I16; STAR_COUNT] = [Vec3I16::ZERO; STAR_COUNT];

// ----------------------------------------------------------------------
// Mesh: cube — 8 vertices, 12 gouraud triangles
// ----------------------------------------------------------------------

/// Half-side of the cube in Q3.12. `0x0800` = 0.5 units.
const CUBE_HALF: i16 = 0x0800;

const CUBE_VERTS: [Vec3I16; 8] = [
    Vec3I16::new(-CUBE_HALF, -CUBE_HALF, -CUBE_HALF),
    Vec3I16::new(CUBE_HALF, -CUBE_HALF, -CUBE_HALF),
    Vec3I16::new(CUBE_HALF, CUBE_HALF, -CUBE_HALF),
    Vec3I16::new(-CUBE_HALF, CUBE_HALF, -CUBE_HALF),
    Vec3I16::new(-CUBE_HALF, -CUBE_HALF, CUBE_HALF),
    Vec3I16::new(CUBE_HALF, -CUBE_HALF, CUBE_HALF),
    Vec3I16::new(CUBE_HALF, CUBE_HALF, CUBE_HALF),
    Vec3I16::new(-CUBE_HALF, CUBE_HALF, CUBE_HALF),
];

/// 12 triangles (2 per face × 6 faces). Vertex indices into
/// [`CUBE_VERTS`]. Winding is counter-clockwise when looking at
/// the outside of each face.
const CUBE_TRIS: [[u8; 3]; 12] = [
    // -Z face (front)
    [0, 1, 2], [0, 2, 3],
    // +Z face (back)
    [4, 6, 5], [4, 7, 6],
    // -X face (left)
    [0, 3, 7], [0, 7, 4],
    // +X face (right)
    [1, 5, 6], [1, 6, 2],
    // -Y face (top — Y up-negative here, matches GTE screen)
    [0, 4, 5], [0, 5, 1],
    // +Y face (bottom)
    [3, 2, 6], [3, 6, 7],
];

/// Per-vertex colours for the gouraud cube. Walking the vertex
/// indices gives a rainbow around the solid.
const CUBE_VERT_COLORS: [(u8, u8, u8); 8] = [
    (220, 40, 40),   // 0 — red
    (220, 180, 40),  // 1 — orange
    (220, 220, 60),  // 2 — yellow
    (60, 220, 60),   // 3 — green
    (60, 220, 220),  // 4 — cyan
    (80, 80, 220),   // 5 — blue
    (220, 60, 220),  // 6 — magenta
    (240, 240, 240), // 7 — white
];

// ----------------------------------------------------------------------
// Mesh: pyramid (tetrahedron) — 4 vertices, 4 flat triangles
// ----------------------------------------------------------------------

/// Tetrahedron with vertices approximately on the unit sphere,
/// one apex up, three base vertices around.
const PYRAMID_VERTS: [Vec3I16; 4] = [
    Vec3I16::new(0, 0x0C00, 0),               // apex
    Vec3I16::new(0x0B00, -0x0500, 0),         // bottom-right
    Vec3I16::new(-0x0580, -0x0500, 0x0990),   // bottom-back-left
    Vec3I16::new(-0x0580, -0x0500, -0x0990),  // bottom-front-left
];

const PYRAMID_TRIS: [[u8; 3]; 4] = [
    [0, 2, 1], // right face
    [0, 3, 2], // back face
    [0, 1, 3], // front face
    [1, 2, 3], // bottom
];

/// Per-triangle flat colours — warm palette.
const PYRAMID_FACE_COLORS: [(u8, u8, u8); 4] = [
    (240, 150, 60),  // amber
    (240, 110, 80),  // coral
    (200, 80, 60),   // rust
    (150, 50, 50),   // maroon
];

// ----------------------------------------------------------------------
// Mesh: octahedron — 6 vertices, 8 flat triangles
// ----------------------------------------------------------------------

const OCTA_R: i16 = 0x0A00; // inscribed radius ≈ 0.625

const OCTA_VERTS: [Vec3I16; 6] = [
    Vec3I16::new(0, OCTA_R, 0),   // +Y top
    Vec3I16::new(0, -OCTA_R, 0),  // -Y bottom
    Vec3I16::new(OCTA_R, 0, 0),   // +X right
    Vec3I16::new(-OCTA_R, 0, 0),  // -X left
    Vec3I16::new(0, 0, OCTA_R),   // +Z back
    Vec3I16::new(0, 0, -OCTA_R),  // -Z front
];

const OCTA_TRIS: [[u8; 3]; 8] = [
    // top cap
    [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
    // bottom cap
    [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
];

/// Cool palette for the octahedron faces.
const OCTA_FACE_COLORS: [(u8, u8, u8); 8] = [
    (60, 180, 220),
    (80, 200, 240),
    (100, 160, 220),
    (60, 140, 200),
    (40, 100, 180),
    (60, 120, 200),
    (80, 140, 220),
    (100, 160, 240),
];

// ----------------------------------------------------------------------
// World placement of each mesh
// ----------------------------------------------------------------------

/// World translations for the three meshes — arranged in an
/// equilateral-ish triangle pattern.
const CUBE_POS: Vec3I32 = Vec3I32::new(-0x1400, 0x0400, WORLD_Z);
const PYRAMID_POS: Vec3I32 = Vec3I32::new(0, -0x1000, WORLD_Z);
const OCTAHEDRON_POS: Vec3I32 = Vec3I32::new(0x1400, 0x0400, WORLD_Z);

// ----------------------------------------------------------------------
// Font + VRAM layout (HUD only)
// ----------------------------------------------------------------------

const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

struct Scene {
    frame: u32,
    /// Total triangle count rendered this frame — reported in the
    /// HUD to prove the OT is carrying real load.
    tri_count: u16,
    /// Deterministic PRNG from psx-fx.
    rng: LcgRng,
    /// Ambient sparks drifting through the scene.
    sparks: ParticlePool<32>,
    /// Screen shake — triggered by spark bursts for subtle camera
    /// movement.
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

/// Flat-shaded triangles — pyramid + octahedron = 4 + 8 = 12
/// per object, × some number of frames worth of headroom. Total:
/// pyramid(4) + octa(8) + misc = 16, rounded to 24.
static mut FLAT_TRIS: [TriFlat; 24] = [const {
    TriFlat {
        tag: 0,
        color_cmd: 0,
        v0: 0,
        v1: 0,
        v2: 0,
    }
}; 24];

/// Gouraud-shaded triangles — cube has 12. Sized to 16 for
/// headroom.
static mut GOURAUD_TRIS: [TriGouraud; 16] = [const {
    TriGouraud {
        tag: 0,
        color0_cmd: 0,
        v0: 0,
        color1: 0,
        v1: 0,
        color2: 0,
        v2: 0,
    }
}; 16];

/// Stars (1×1 rects) + sparks reserve + HUD backdrop rects.
static mut SCENE_RECTS: [RectFlat; 96] = [const {
    RectFlat::new(0, 0, 0, 0, 0, 0, 0)
}; 96];

/// One gradient-background quad.
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

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    let mut fb = FrameBuffer::new(SCREEN_W as u16, SCREEN_H as u16);
    gpu::set_draw_area(0, 0, (SCREEN_W - 1) as u16, (SCREEN_H - 1) as u16);
    gpu::set_draw_offset(0, 0);

    // GTE scene setup — screen offset centres projection, focal
    // length picks the zoom, depth cue + AVSZ weights get sane
    // defaults. Everything else (rotation / translation) is set
    // per-object each frame.
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
// Update phase
// ----------------------------------------------------------------------

/// Seed each star at a deterministic position based on its index
/// and the RNG. Reusable — a star that crosses the camera also
/// calls back here with just its slot.
fn spawn_star(rng: &mut LcgRng, idx: usize) {
    let x = rng.signed(STAR_SPREAD_XY);
    let y = rng.signed(STAR_SPREAD_XY);
    // New star spawns near the far plane.
    let z = STAR_MAX_Z - (rng.next() as i16 & 0x0FFF);
    // SAFETY: single-threaded bare-metal access to STARS.
    unsafe {
        STARS[idx] = Vec3I16::new(x, y, z);
    }
}

fn init_starfield() {
    let s = unsafe { &mut SCENE };
    for i in 0..STAR_COUNT {
        // Pre-distribute Z across the full range so the field looks
        // populated immediately rather than all stars starting at
        // max Z + approaching together.
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

    // Advance the starfield — every star moves toward the camera.
    for i in 0..STAR_COUNT {
        // SAFETY: serialised single-thread access.
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

    // Ambient sparks — occasional single-particle burst at a
    // random screen position. Keeps the scene from ever feeling
    // still.
    if s.frame % 8 == 0 {
        let x = SCREEN_W / 2 + s.rng.signed(SCREEN_W / 2 - 20);
        let y = SCREEN_H / 2 + s.rng.signed(SCREEN_H / 2 - 30);
        let color = (
            80 + (s.rng.next() & 0x7F) as u8,
            80 + (s.rng.next() & 0x7F) as u8,
            120 + (s.rng.next() & 0x7F) as u8,
        );
        s.sparks.spawn_burst(&mut s.rng, (x, y), color, 3, 20, 40);
    }
    s.sparks.update(1);

    // Trigger a soft shake every 4 seconds for a subtle "the
    // scene exists in space" feel.
    if s.frame % 240 == 0 && s.frame > 0 {
        s.shake.trigger(4);
    }
}

// ----------------------------------------------------------------------
// Render — build OT
// ----------------------------------------------------------------------

fn build_frame_ot() {
    let ot = unsafe { &mut OT };
    let rects = unsafe { &mut SCENE_RECTS };
    let flats = unsafe { &mut FLAT_TRIS };
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
            (10, 6, 28),     // indigo-black
            (10, 6, 28),
            (2, 1, 8),       // near-black
            (2, 1, 8),
        ],
    );
    ot.add(7, bg, QuadGouraud::WORDS);

    // Slot 6 — starfield. No GTE transform; a star's local
    // position IS its world position. We rotate the whole
    // starfield by the scene roll so parallax looks right.
    let scene_roll = Mat3I16::rotate_z((s.frame.wrapping_mul(2) & 0xFFFF) as u16);
    scene::load_rotation(&scene_roll);
    scene::load_translation(Vec3I32::new(0, 0, 0));
    let mut rect_idx = 0;
    for i in 0..STAR_COUNT {
        let sv = unsafe { STARS[i] };
        let p = scene::project_vertex(sv);
        // Out-of-bounds stars: skip drawing.
        if p.sx < 0 || p.sx >= SCREEN_W || p.sy < 0 || p.sy >= SCREEN_H {
            continue;
        }
        // Brightness fades with depth so new-spawn far stars are dim.
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

    // Now render each tumbling mesh. We pick its world position,
    // build its local rotation, project each triangle, and insert
    // into OT at a slot derived from the triangle's average Z so
    // overlapping meshes resolve correctly.
    let mut flat_idx = 0;
    let mut gouraud_idx = 0;

    // --- CUBE (gouraud) ---
    let cube_rot = Mat3I16::rotate_y((s.frame.wrapping_mul(5) & 0xFFFF) as u16)
        .mul(&Mat3I16::rotate_x((s.frame.wrapping_mul(3) & 0xFFFF) as u16));
    scene::load_rotation(&cube_rot);
    scene::load_translation(CUBE_POS);
    // Pre-project all 8 vertices once — cheaper than per-triangle
    // re-projection since the cube shares verts across faces.
    let mut cube_proj = [(0i16, 0i16, 0u16); 8];
    for i in 0..CUBE_VERTS.len() {
        let p = scene::project_vertex(CUBE_VERTS[i]);
        cube_proj[i] = (p.sx + shake_dx, p.sy + shake_dy, p.sz);
    }
    for tri in &CUBE_TRIS {
        if gouraud_idx >= gouraud.len() {
            break;
        }
        let (v0, v1, v2) = (
            cube_proj[tri[0] as usize],
            cube_proj[tri[1] as usize],
            cube_proj[tri[2] as usize],
        );
        // Back-face cull: if the 2D triangle is wound clockwise,
        // we're seeing the back face — skip. (cross_z < 0 for CW
        // in screen coords.)
        if back_facing(v0, v1, v2) {
            continue;
        }
        gouraud[gouraud_idx] = TriGouraud::new(
            [(v0.0, v0.1), (v1.0, v1.1), (v2.0, v2.1)],
            [
                CUBE_VERT_COLORS[tri[0] as usize],
                CUBE_VERT_COLORS[tri[1] as usize],
                CUBE_VERT_COLORS[tri[2] as usize],
            ],
        );
        let z_avg = (v0.2 as u32 + v1.2 as u32 + v2.2 as u32) / 3;
        let slot = ot_slot_for_depth(z_avg);
        ot.add(slot, &mut gouraud[gouraud_idx], TriGouraud::WORDS);
        gouraud_idx += 1;
        s.tri_count += 1;
    }

    // --- PYRAMID (flat) ---
    let pyr_rot = Mat3I16::rotate_y((s.frame.wrapping_mul(4) & 0xFFFF).wrapping_neg() as u16)
        .mul(&Mat3I16::rotate_z((s.frame.wrapping_mul(2) & 0xFFFF) as u16));
    scene::load_rotation(&pyr_rot);
    scene::load_translation(PYRAMID_POS);
    let mut pyr_proj = [(0i16, 0i16, 0u16); 4];
    for i in 0..PYRAMID_VERTS.len() {
        let p = scene::project_vertex(PYRAMID_VERTS[i]);
        pyr_proj[i] = (p.sx + shake_dx, p.sy + shake_dy, p.sz);
    }
    for (face_idx, tri) in PYRAMID_TRIS.iter().enumerate() {
        if flat_idx >= flats.len() {
            break;
        }
        let (v0, v1, v2) = (
            pyr_proj[tri[0] as usize],
            pyr_proj[tri[1] as usize],
            pyr_proj[tri[2] as usize],
        );
        if back_facing(v0, v1, v2) {
            continue;
        }
        let (r, g, b) = PYRAMID_FACE_COLORS[face_idx];
        flats[flat_idx] = TriFlat::new([(v0.0, v0.1), (v1.0, v1.1), (v2.0, v2.1)], r, g, b);
        let z_avg = (v0.2 as u32 + v1.2 as u32 + v2.2 as u32) / 3;
        let slot = ot_slot_for_depth(z_avg);
        ot.add(slot, &mut flats[flat_idx], TriFlat::WORDS);
        flat_idx += 1;
        s.tri_count += 1;
    }

    // --- OCTAHEDRON (flat) ---
    let oct_rot = Mat3I16::rotate_x((s.frame.wrapping_mul(2) & 0xFFFF) as u16)
        .mul(&Mat3I16::rotate_z((s.frame.wrapping_mul(4) & 0xFFFF).wrapping_neg() as u16));
    scene::load_rotation(&oct_rot);
    scene::load_translation(OCTAHEDRON_POS);
    let mut oct_proj = [(0i16, 0i16, 0u16); 6];
    for i in 0..OCTA_VERTS.len() {
        let p = scene::project_vertex(OCTA_VERTS[i]);
        oct_proj[i] = (p.sx + shake_dx, p.sy + shake_dy, p.sz);
    }
    for (face_idx, tri) in OCTA_TRIS.iter().enumerate() {
        if flat_idx >= flats.len() {
            break;
        }
        let (v0, v1, v2) = (
            oct_proj[tri[0] as usize],
            oct_proj[tri[1] as usize],
            oct_proj[tri[2] as usize],
        );
        if back_facing(v0, v1, v2) {
            continue;
        }
        let (r, g, b) = OCTA_FACE_COLORS[face_idx];
        flats[flat_idx] = TriFlat::new([(v0.0, v0.1), (v1.0, v1.1), (v2.0, v2.1)], r, g, b);
        let z_avg = (v0.2 as u32 + v1.2 as u32 + v2.2 as u32) / 3;
        let slot = ot_slot_for_depth(z_avg);
        ot.add(slot, &mut flats[flat_idx], TriFlat::WORDS);
        flat_idx += 1;
        s.tri_count += 1;
    }

    // Slot 3 — sparks (in front of meshes).
    let spark_budget = rects.len().saturating_sub(rect_idx);
    let wrote = s.sparks.render_into_ot(
        ot,
        &mut rects[rect_idx..rect_idx + spark_budget],
        3,
        (shake_dx, shake_dy),
    );
    let _ = wrote;
}

fn submit_frame_ot() {
    unsafe {
        OT.submit();
    }
}

/// True if the screen-space triangle winds clockwise — meaning
/// we're seeing its back face and should skip drawing.
///
/// Cross product of the 2D edges `(v1-v0) × (v2-v0)`. Sign tells
/// us winding. With "Y grows downward" screen coords, CW in math
/// terms looks CCW on screen — we treat `cross <= 0` as the back
/// face.
fn back_facing(v0: (i16, i16, u16), v1: (i16, i16, u16), v2: (i16, i16, u16)) -> bool {
    let ax = (v1.0 as i32) - (v0.0 as i32);
    let ay = (v1.1 as i32) - (v0.1 as i32);
    let bx = (v2.0 as i32) - (v0.0 as i32);
    let by = (v2.1 as i32) - (v0.1 as i32);
    (ax * by - ay * bx) <= 0
}

/// Map a projected Z value to an OT slot.
///
/// Meshes we want layered live in slots 4..=5 (slot 6 is the
/// starfield backdrop, slot 7 is the gradient). Each slot covers
/// roughly half the mesh depth range — finer granularity would
/// need more slots.
fn ot_slot_for_depth(z: u32) -> usize {
    // Mesh Z (after RTPS division) lands roughly in 0x1C00..0x2000
    // at our WORLD_Z. Split into two slots around the midpoint.
    if z > 0x1D00 { 5 } else { 4 }
}

// ----------------------------------------------------------------------
// HUD
// ----------------------------------------------------------------------

fn draw_hud(font: &FontAtlas) {
    let s = unsafe { &SCENE };

    font.draw_text(4, 4, "SCENE-LAB", (220, 220, 250));
    font.draw_text(
        SCREEN_W - 8 * 10,
        4,
        "psoxide",
        (140, 140, 180),
    );

    // Bottom HUD: frame + tri count.
    font.draw_text(4, SCREEN_H - 20, "frame", (160, 160, 200));
    let frame = u16_hex((s.frame & 0xFFFF) as u16);
    font.draw_text(4 + 8 * 6, SCREEN_H - 20, frame.as_str(), (200, 240, 160));

    font.draw_text(SCREEN_W / 2 - 8 * 3, SCREEN_H - 20, "tri", (160, 160, 200));
    let tri = u16_hex(s.tri_count);
    font.draw_text(SCREEN_W / 2 + 8, SCREEN_H - 20, tri.as_str(), (240, 200, 160));

    // Right-side label + value. "stars" = 5 chars × 8 = 40 px,
    // 8 px gap, 6-char hex = 48 px, 4 px right-margin → 100 px.
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
