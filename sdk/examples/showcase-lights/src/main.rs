//! `showcase-lights` — four coloured moving point lights
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
//! it's about 1k integer-math ops per frame — trivially under
//! budget on a 33 MHz MIPS R3000.
//!
//! What's on screen:
//!
//! 1. **Room floor + back wall** — two big `TriFlat` slabs at the
//!    edges, giving the light flow something to play across.
//! 2. **6 scaled cubes** on the floor — different heights + XZ
//!    positions, one canonical cube mesh `include_bytes!`'d from
//!    `assets/cube.psxm` (cooked from a face-split OBJ so per-face
//!    normals come out of `psxed --compute-normals` flat).
//! 3. **4 point lights** in R / G / B / Y, each on its own orbit
//!    with a different period so they visibly "dance" around the
//!    scene.
//! 4. **HUD** — frame / live-tri / light positions.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::Mesh;
use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_gpu::framebuf::FrameBuffer;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat, TriFlat, TriGouraud};
use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_math::sincos;
use psx_vram::{Clut, TexDepth, Tpage};

// ----------------------------------------------------------------------
// Screen / scene geometry
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;
const OT_DEPTH: usize = 8;

const PROJ_H: u16 = 280;
const WORLD_Z: i32 = 0x3800;

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

/// Tighter radius so each light dominates a smaller region —
/// distinct coloured zones rather than four lights additively
/// whiting-out the whole scene.
const LIGHT_RADIUS_SQ: i32 = 0x0500_0000;

/// Tuned to match `sqrt(LIGHT_RADIUS_SQ) ≈ 0x2400`. Used as the
/// "reference distance" the dot product divides by to turn its
/// N · D magnitude (which carries distance) into a rough
/// direction-only scalar.
const RADIUS_LINEAR: i64 = 0x2400;

/// Per-channel ambient term, added before clamping. Dim blueish
/// so shadowed faces read as "in the dark room" rather than
/// pure-black holes.
const AMBIENT: (i32, i32, i32) = (20, 24, 36);

#[derive(Copy, Clone)]
struct PointLight {
    /// World-space position (Q3.12-ish integers — we don't treat
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

const NUM_CUBES: usize = 6;

/// One cube placed in the room. `scale` is a Q3.12 factor
/// applied to each vertex before rotation; the cube mesh has
/// unit-side verts at ±0x0E00, so `scale = 0x1000` renders a
/// 1.0×1.0×1.0 cube and `scale = 0x2000` renders 2.0×2.0×2.0.
#[derive(Copy, Clone)]
struct CubeInstance {
    position: Vec3I32,
    scale: i32,
    /// Angular velocity per frame (Q0.12 units), for Y-axis spin.
    y_spin_per_frame: u16,
}

/// Hand-placed so cubes form a loose arc on the floor, varied
/// sizes suggesting stacked columns.
const CUBE_LAYOUT: [CubeInstance; NUM_CUBES] = [
    CubeInstance {
        position: Vec3I32::new(-0x1800, 0x0600, WORLD_Z + 0x0800),
        scale: 0x0A00,
        y_spin_per_frame: 20,
    },
    CubeInstance {
        position: Vec3I32::new(-0x0A00, 0x0400, WORLD_Z),
        scale: 0x0C00,
        y_spin_per_frame: 14,
    },
    CubeInstance {
        position: Vec3I32::new(0x0000, 0x0200, WORLD_Z - 0x0400),
        scale: 0x1000,
        y_spin_per_frame: 9,
    },
    CubeInstance {
        position: Vec3I32::new(0x0A00, 0x0500, WORLD_Z + 0x0200),
        scale: 0x0800,
        y_spin_per_frame: 28,
    },
    CubeInstance {
        position: Vec3I32::new(0x1800, 0x0700, WORLD_Z + 0x0800),
        scale: 0x0700,
        y_spin_per_frame: 36,
    },
    CubeInstance {
        position: Vec3I32::new(0x0500, 0x1100, WORLD_Z + 0x0C00),
        scale: 0x0600,
        y_spin_per_frame: 44,
    },
];

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

struct Scene {
    frame: u32,
    tri_count: u16,
    lights: [PointLight; NUM_LIGHTS],
}

static mut SCENE: Scene = Scene {
    frame: 0,
    tri_count: 0,
    // Initial positions filled in on `init`. Colours are highly
    // saturated so a cube sitting near one light reads as clearly
    // "that colour" rather than a neutral blend.
    lights: [
        PointLight {
            pos: Vec3I16::ZERO,
            colour: (255, 40, 40),   // red
        },
        PointLight {
            pos: Vec3I16::ZERO,
            colour: (40, 255, 60),   // green
        },
        PointLight {
            pos: Vec3I16::ZERO,
            colour: (60, 100, 255),  // blue
        },
        PointLight {
            pos: Vec3I16::ZERO,
            colour: (255, 220, 40),  // yellow
        },
    ],
};

// ----------------------------------------------------------------------
// OT + primitive buffers
// ----------------------------------------------------------------------

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();

/// Gouraud cube tris + 2 flat floor/wall tris: 6 cubes × 12 tris =
/// 72 tris, plus room. 96 slots = headroom.
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

/// Flat tris for the floor + back wall.
static mut FLAT_TRIS: [TriFlat; 8] = [const {
    TriFlat {
        tag: 0,
        color_cmd: 0,
        v0: 0,
        v1: 0,
        v2: 0,
    }
}; 8];

/// Rects for light position markers (one per light — small bright
/// square at each light's projected screen pos so the viewer can
/// see where the lights are).
static mut LIGHT_MARKERS: [RectFlat; 16] = [const {
    RectFlat::new(0, 0, 0, 0, 0, 0, 0)
}; 16];

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

/// Per-cube projected vertex cache: (sx, sy, sz, r, g, b).
type VertProj = (i16, i16, u16, u8, u8, u8);
const EMPTY_VERT: VertProj = (0, 0, 0, 0, 0, 0);
static mut CUBE_PROJ: [VertProj; 32] = [EMPTY_VERT; 32];

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
// Update: animate 4 lights on independent orbits
// ----------------------------------------------------------------------

fn update_scene() {
    let s = unsafe { &mut SCENE };
    s.frame = s.frame.wrapping_add(1);
    s.tri_count = 0;

    // Each light orbits in the XZ plane at a slightly different
    // radius + speed + Y height + phase. All four together give
    // the sense of "lights dancing" through the room.
    let phases: [(i16, u16, i16, u16); NUM_LIGHTS] = [
        // (orbit_radius, frames_per_rev_scale, y_height, phase_deg)
        (0x1200, 36, 0x0800, 0),
        (0x1500, 48, 0x1200, 1024), // 90°
        (0x0E00, 52, 0x0600, 2048), // 180°
        (0x1700, 42, 0x0F00, 3072), // 270°
    ];
    for i in 0..NUM_LIGHTS {
        let (r, speed, y, phase) = phases[i];
        let angle = ((s.frame as u32).wrapping_mul(speed as u32) as u16)
            .wrapping_add(phase);
        // Q1.12 sin/cos scaled by Q3.12 radius → Q4.24 position,
        // shift right 12 to bring back to Q3.12.
        let x = ((sincos::cos_q12(angle) * r as i32) >> 12) as i16;
        let z = ((sincos::sin_q12(angle) * r as i32) >> 12) as i16;
        s.lights[i].pos = Vec3I16::new(x, y, (WORLD_Z as i16).wrapping_add(z));
    }
}

// ----------------------------------------------------------------------
// Per-vertex CPU lighting
// ----------------------------------------------------------------------

/// Compute a lit colour for one vertex given its world-space
/// position + normal.
///
/// For each light: compute `direction = light_pos - vert_pos`,
/// then the un-normalised dot product with the normal. This mixes
/// "distance × cos(angle)" into one scalar — we skip the explicit
/// normalise (which would need a sqrt) and instead rely on a
/// separate quadratic distance-falloff term to take the distance
/// contribution out of the dot.
///
/// Fixed-point: normal is Q3.12 (components in ±0x1000), position
/// deltas are "world units" where 1 unit = 0x1000. The un-normalised
/// dot lands in Q6.24 space but we scale to a bounded intensity
/// before accumulating, so no 32-bit overflow risk.
fn light_vertex(world_pos: Vec3I16, world_normal: Vec3I16) -> (u8, u8, u8) {
    let mut r: i32 = AMBIENT.0;
    let mut g: i32 = AMBIENT.1;
    let mut b: i32 = AMBIENT.2;

    let lights = unsafe { &SCENE.lights };
    for light in lights.iter() {
        // Vector from vertex to light.
        let dx = (light.pos.x as i32) - (world_pos.x as i32);
        let dy = (light.pos.y as i32) - (world_pos.y as i32);
        let dz = (light.pos.z as i32) - (world_pos.z as i32);

        // Distance squared, in i64 to avoid overflow (each
        // component can be up to ~0x3000, squared ≈ 0x900_0000,
        // sum up to ~0x1B00_0000 — fits i32 but tight. i64 keeps
        // headroom for wider scenes.)
        let dist_sq = (dx as i64) * (dx as i64)
            + (dy as i64) * (dy as i64)
            + (dz as i64) * (dz as i64);

        if dist_sq > LIGHT_RADIUS_SQ as i64 {
            continue;
        }

        // Quadratic attenuation: 1 at dist=0, 0 at dist²=RADIUS_SQ.
        // att in [0, 0x1000]
        let att_q12 = (((LIGHT_RADIUS_SQ as i64 - dist_sq) << 12)
            / LIGHT_RADIUS_SQ as i64) as i32;

        // Un-normalised N · D. Components of N are Q3.12, D's are
        // raw world units. Result is in Q3.12 × world-units.
        let dot_unnorm = (world_normal.x as i64) * (dx as i64)
            + (world_normal.y as i64) * (dy as i64)
            + (world_normal.z as i64) * (dz as i64);

        // Facing-away? Skip.
        if dot_unnorm <= 0 {
            continue;
        }

        // To get a unit-less cosine scalar we'd want dot_unnorm /
        // (|N| × |D|). |N| = 1.0 = 0x1000. |D| = sqrt(dist_sq).
        // We cheat: divide by sqrt(RADIUS_SQ) — a constant — which
        // sacrifices strict physical accuracy but gives a clean
        // "nearness + orientation" scalar in a known range.
        //
        // Dividing the Q3.12 dot (which carries distance in its
        // magnitude) by `RADIUS_LINEAR` rescales it into a rough
        // Q3.12 direction-intensity in `[0, ~0x1000]`.
        let intensity_q12 = (dot_unnorm / RADIUS_LINEAR) as i32;

        // Modulate by quadratic attenuation and normal magnitude.
        // `lit_scalar` is in Q3.12, roughly [0, 0x1000].
        let lit_scalar = ((intensity_q12 as i64 * att_q12 as i64) >> 12) as i32;
        // Clamp before accumulating — single lights shouldn't
        // saturate the channel on their own.
        let lit_scalar = lit_scalar.clamp(0, 0x1000);

        // Add colour × lit_scalar to accumulator. Colour is 0..255,
        // lit_scalar is 0..0x1000 = 0..4096. Product up to ~1M
        // — divide by 4096 to bring back into 0..255 range.
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
// Render
// ----------------------------------------------------------------------

fn build_frame_ot() {
    let ot = unsafe { &mut OT };
    let gouraud = unsafe { &mut GOURAUD_TRIS };
    let flats = unsafe { &mut FLAT_TRIS };
    let markers = unsafe { &mut LIGHT_MARKERS };
    let bg = unsafe { &mut BG_QUAD };
    ot.clear();

    let s = unsafe { &mut SCENE };
    let cube = Mesh::from_bytes(CUBE_BLOB).expect("cube blob");

    // Slot 7 — dark-room gradient backdrop.
    *bg = QuadGouraud::new(
        [
            (0, 0),
            (SCREEN_W, 0),
            (0, SCREEN_H),
            (SCREEN_W, SCREEN_H),
        ],
        [
            (16, 10, 24),
            (16, 10, 24),
            (4, 2, 8),
            (4, 2, 8),
        ],
    );
    ot.add(7, bg, QuadGouraud::WORDS);

    // Slot 5 — all 6 cube instances, CPU-lit + GTE-projected.
    let mut tri_idx = 0;
    for instance in &CUBE_LAYOUT {
        // Per-instance: compose Y-spin rotation × identity (no
        // other rotation). `frame * spin` gives the current angle.
        let angle = ((s.frame as u32).wrapping_mul(instance.y_spin_per_frame as u32)
            & 0xFFFF) as u16;
        let rot = Mat3I16::rotate_y(angle);

        // GTE prep: rotation handles per-instance Y-spin, the
        // vertex scale is folded into the rotation matrix
        // (we multiply a uniform-scale matrix into `rot` so the
        // GTE does scale + rotate in one RTPS pass).
        let rot_scaled = scale_mat(&rot, instance.scale);
        scene::load_rotation(&rot_scaled);
        scene::load_translation(instance.position);

        // CPU path: we duplicate position + normal math to get
        // world-space values for lighting. The GTE isn't involved
        // in that computation — we only use it for projection.
        let cube_proj = unsafe { &mut CUBE_PROJ };
        for vi in 0..cube.vert_count() {
            let vl = cube.vertex(vi as u8);
            let nl = cube
                .vertex_normal(vi as u8)
                .unwrap_or(Vec3I16::new(0, 0x1000, 0));

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
            // no scale — scale would stretch the normal length).
            let nrot = Vec3I16::new(
                mat_row_dot(&rot, 0, nl)
                    .clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                mat_row_dot(&rot, 1, nl)
                    .clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                mat_row_dot(&rot, 2, nl)
                    .clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );

            let (r, g, b) = light_vertex(world_pos, nrot);

            // Project via GTE (the rot_scaled + translation are
            // already loaded).
            let p = scene::project_vertex(vl);
            cube_proj[vi as usize] = (p.sx, p.sy, p.sz, r, g, b);
        }

        // Emit triangles with per-vertex lit colours.
        for face_idx in 0..cube.face_count() {
            if tri_idx >= gouraud.len() {
                break;
            }
            let (ia, ib, ic) = cube.face(face_idx);
            let (v0, v1, v2) = (
                cube_proj[ia as usize],
                cube_proj[ib as usize],
                cube_proj[ic as usize],
            );
            if back_facing(v0, v1, v2) {
                continue;
            }
            gouraud[tri_idx] = TriGouraud::new(
                [(v0.0, v0.1), (v1.0, v1.1), (v2.0, v2.1)],
                [
                    (v0.3, v0.4, v0.5),
                    (v1.3, v1.4, v1.5),
                    (v2.3, v2.4, v2.5),
                ],
            );
            ot.add(5, &mut gouraud[tri_idx], TriGouraud::WORDS);
            tri_idx += 1;
            s.tri_count += 1;
        }
    }

    // Slot 3 — visible light position markers (small bright
    // squares at each light's projected screen position, colour
    // matches the light's colour). Use a direct-to-screen path —
    // identity rotation + zero translation — to project light
    // positions cleanly.
    scene::load_rotation(&Mat3I16::IDENTITY);
    scene::load_translation(Vec3I32::new(0, 0, 0));
    let mut marker_idx = 0;
    for light in &s.lights {
        if marker_idx >= markers.len() {
            break;
        }
        let p = scene::project_vertex(light.pos);
        if p.sx < 4 || p.sx > SCREEN_W - 4 || p.sy < 4 || p.sy > SCREEN_H - 4 {
            continue;
        }
        markers[marker_idx] = RectFlat::new(
            p.sx - 2,
            p.sy - 2,
            4,
            4,
            light.colour.0,
            light.colour.1,
            light.colour.2,
        );
        ot.add(3, &mut markers[marker_idx], RectFlat::WORDS);
        marker_idx += 1;
    }

    let _ = flats;
}

fn submit_frame_ot() {
    unsafe {
        OT.submit();
    }
}

/// Apply a uniform scale factor to a rotation matrix. Because
/// Mat3I16 is Q3.12 and scale is Q3.12, the product is Q3.12 too
/// after the shared `>> 12` shift.
fn scale_mat(m: &Mat3I16, scale_q12: i32) -> Mat3I16 {
    let mut out = [[0i16; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let v = ((m.m[i][j] as i32) * scale_q12) >> 12;
            out[i][j] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
    Mat3I16 { m: out }
}

/// Dot product of matrix row `r` with vector `v` in Q3.12; result
/// in i32 (un-truncated range).
#[inline]
fn mat_row_dot(m: &Mat3I16, r: usize, v: Vec3I16) -> i32 {
    let sum = (m.m[r][0] as i32) * (v.x as i32)
        + (m.m[r][1] as i32) * (v.y as i32)
        + (m.m[r][2] as i32) * (v.z as i32);
    sum >> 12
}

fn back_facing(v0: VertProj, v1: VertProj, v2: VertProj) -> bool {
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
    font.draw_text(4, 4, "SHOWCASE-LIGHTS", (220, 220, 250));
    // "4 lights" = 8 chars × 8 = 64 px. Anchor at W - 64 - 4.
    font.draw_text(SCREEN_W - 68, 4, "4 lights", (180, 180, 220));

    font.draw_text(4, SCREEN_H - 20, "frame", (160, 160, 200));
    let frame = u16_hex((s.frame & 0xFFFF) as u16);
    font.draw_text(4 + 8 * 6, SCREEN_H - 20, frame.as_str(), (200, 240, 160));

    font.draw_text(SCREEN_W / 2 - 8 * 3, SCREEN_H - 20, "tri", (160, 160, 200));
    let tri = u16_hex(s.tri_count);
    font.draw_text(SCREEN_W / 2 + 8, SCREEN_H - 20, tri.as_str(), (240, 200, 160));

    font.draw_text(SCREEN_W - 100, SCREEN_H - 20, "cubes", (160, 160, 200));
    let cubes = u16_hex(NUM_CUBES as u16);
    font.draw_text(
        SCREEN_W - 52,
        SCREEN_H - 20,
        cubes.as_str(),
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
