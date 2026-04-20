//! `showcase-fog` — the full PS1-commercial GTE + textured-poly
//! pipeline in one demo.
//!
//! Every frame, for every triangle, the GTE executes:
//!
//!   RTPT  → project 3 vertices in one op
//!   NCLIP → hardware back-face cull
//!   AVSZ3 → average-Z key for ordering-table insertion
//!   NCDT  → lit + depth-cue colour for all 3 vertices
//!
//! The three NCDT-produced per-vertex colours become the per-vertex
//! *tint* on a **textured Gouraud** triangle (GP0 0x34), so the GPU
//! multiplies each sampled texel by its interpolated NCDT colour.
//! Result: textures that properly light and fog — near walls show
//! crisp brick, far walls dissolve into the fog colour — the
//! Silent-Hill-era look achieved through hardware ops only.
//!
//! # Scene
//!
//! A square corridor receding into the distance. 16 rings of wall
//! segments — ceiling / floor / left / right — scroll toward the
//! camera; when a ring crosses the near plane, it wraps to the far
//! end. A single warm directional light orbits slowly around the
//! Y axis so each wall sees the highlight sweep past.
//!
//! # Textures
//!
//! Two 64×64 4bpp CLUT textures live in a shared tpage at VRAM
//! (640, 0):
//!
//!   - **brick-wall** at U = 0..63. Ceiling + left + right walls.
//!   - **floor**      at U = 64..127. Floor only.
//!
//! Each has its own 16-entry CLUT. Both are cooked by `psxed tex`
//! from a `vendor/*.jpg` source (see `vendor/README.md`). Every
//! wall segment between two rings maps the full 64×64 texture, so
//! the pattern tiles once per segment along the corridor length.
//!
//! # Triangle budget
//!
//! 15 ring-gaps × 4 walls × 2 tris = **120 triangles / frame** —
//! each a `TriTexturedGouraud` (9 data words per primitive).

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::Texture;
use psx_font::{FontAtlas, fonts::BASIC_8X16};
use psx_gpu::framebuf::FrameBuffer;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, TriTexturedGouraud};
use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_gte::lighting::{Light, LightRig, project_triangle_fogged};
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_vram::{Clut, TexDepth, Tpage, VramRect, upload_bytes};

// ----------------------------------------------------------------------
// Screen + projection constants
// ----------------------------------------------------------------------

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;

/// GTE projection plane (focal length). Larger = narrower FOV.
const PROJ_H: u16 = 280;

/// Ordering-table depth. Corridor OTZ values span roughly a 3-bit
/// range so 32 slots is more than enough to avoid far-ring
/// triangles overlapping near-ring ones.
const OT_DEPTH: usize = 32;

// ----------------------------------------------------------------------
// Corridor geometry
// ----------------------------------------------------------------------

const NUM_RINGS: usize = 16;

/// World-space half-dimensions of the corridor cross-section.
/// Chosen so near-ring walls extend off-screen (the camera is
/// inside the corridor), while far-ring walls shrink to a small
/// rectangle in the centre.
const CORR_HALF_W: i16 = 0x0780;
const CORR_HALF_H: i16 = 0x0500;

const RING_SPACING: i32 = 0x0500;
const NEAR_CAM_Z: i32 = 0x0400;
const FIRST_RING_Z: i32 = NEAR_CAM_Z + RING_SPACING;
const SCROLL_SPEED: i32 = 0x0020;

// ----------------------------------------------------------------------
// Fog parameters
// ----------------------------------------------------------------------

const FOG_FC: Vec3I32 = Vec3I32::new(0x0200, 0x0400, 0x0A00);

const FOG_CLEAR: (u8, u8, u8) = (
    (FOG_FC.x >> 4) as u8,
    (FOG_FC.y >> 4) as u8,
    (FOG_FC.z >> 4) as u8,
);

/// Fog gradient tuning.
///
/// With PROJ_H=280, corridor Z in [0x0900..0x5400]:
///   divisor_near = (H << 16) / 0x0900 ≈ 0x1F1C
///   divisor_far  = (H << 16) / 0x5400 ≈ 0x0350
///
/// We want:
///   IR0 at near = 0x0200  (12%  — atmospheric haze even on near walls,
///                          so the fog reads everywhere on screen
///                          rather than only in the central vanishing
///                          point strip)
///   IR0 at far  = 0x1000  (100% — far walls converge exactly to FC,
///                          blending seamlessly into the fog-clear BG)
///
/// Solving:
///   0x1F1C * DQA + DQB = 0x200_000
///   0x0350 * DQA + DQB = 0x1000_000
///   → DQA ≈ -0x0800, DQB = 0x118_E000
const DQA: i16 = -0x0800;
const DQB: i32 = 0x0118_E000;

const ZSF3: i16 = 0x0555;
const ZSF4: i16 = 0x0400;

// ----------------------------------------------------------------------
// Lighting
// ----------------------------------------------------------------------

const BASE_RIG: LightRig = LightRig::new(
    [
        Light {
            direction: Vec3I16::new(0x0E00, -0x0300, 0x0000),
            colour: (0x1C00, 0x1600, 0x0C00),
        },
        Light::OFF,
        Light::OFF,
    ],
    (0x0200, 0x0200, 0x0280),
);

/// `rotate_y`'s angle is 256-per-revolution and it masks `& 0xFF`
/// internally. `frame >> 2` gives 1 angle unit every 4 frames =
/// 1024 frames per revolution ≈ 17 s at 60 fps.
const LIGHT_ORBIT_SHIFT: u32 = 2;

// ----------------------------------------------------------------------
// Textures
// ----------------------------------------------------------------------

/// Shared tpage for both textures — X=640 sits past the 640-wide
/// double-buffered framebuffer region (two 320×240 buffers at
/// Y=0..240 and 240..480). One 4bpp tpage holds up to 256 texels
/// per row, so both 64×64 textures fit side-by-side.
const TEX_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);

/// Brick wall — ceilings + left/right walls.
static BRICK_BLOB: &[u8] = include_bytes!("../assets/brick-wall.psxt");
/// Cobblestone floor.
static FLOOR_BLOB: &[u8] = include_bytes!("../assets/floor.psxt");

/// CLUT slots. X multiples of 16; Y=480/481 keeps them out of the
/// font CLUT's row (256) and the framebuffer's rows (0..479).
const BRICK_CLUT: Clut = Clut::new(0, 480);
const FLOOR_CLUT: Clut = Clut::new(0, 481);

/// U origin of each texture inside [`TEX_TPAGE`] (pixels). A 4bpp
/// tpage is 256 texels wide; brick sits at U=0..63, floor at
/// U=64..127, leaving 128..255 free for future additions.
const BRICK_U: u8 = 0;
const FLOOR_U: u8 = 64;

/// Which texture a wall uses. Maps to the two U origins above.
#[derive(Copy, Clone)]
enum WallTex {
    Brick,
    Floor,
}

impl WallTex {
    const fn u_origin(self) -> u8 {
        match self {
            WallTex::Brick => BRICK_U,
            WallTex::Floor => FLOOR_U,
        }
    }
    const fn clut_word(self) -> u16 {
        match self {
            WallTex::Brick => BRICK_CLUT.uv_clut_word(),
            WallTex::Floor => FLOOR_CLUT.uv_clut_word(),
        }
    }
}

// ----------------------------------------------------------------------
// Wall definitions
// ----------------------------------------------------------------------

#[derive(Copy, Clone)]
struct Wall {
    /// XY position at either end of the wall's cross-section line.
    corners: [(i16, i16); 2],
    /// Inward-facing normal in world space, Q3.12.
    normal: Vec3I16,
    /// Texture slot for this wall.
    tex: WallTex,
}

const WALLS: [Wall; 4] = [
    // Ceiling — brick overhead.
    Wall {
        corners: [(-CORR_HALF_W, -CORR_HALF_H), (CORR_HALF_W, -CORR_HALF_H)],
        normal: Vec3I16::new(0, 0x1000, 0),
        tex: WallTex::Brick,
    },
    // Floor — cobblestone.
    Wall {
        corners: [(CORR_HALF_W, CORR_HALF_H), (-CORR_HALF_W, CORR_HALF_H)],
        normal: Vec3I16::new(0, -0x1000, 0),
        tex: WallTex::Floor,
    },
    // Left wall — brick.
    Wall {
        corners: [(-CORR_HALF_W, CORR_HALF_H), (-CORR_HALF_W, -CORR_HALF_H)],
        normal: Vec3I16::new(0x1000, 0, 0),
        tex: WallTex::Brick,
    },
    // Right wall — brick.
    Wall {
        corners: [(CORR_HALF_W, -CORR_HALF_H), (CORR_HALF_W, CORR_HALF_H)],
        normal: Vec3I16::new(-0x1000, 0, 0),
        tex: WallTex::Brick,
    },
];

// ----------------------------------------------------------------------
// Primitive arena
// ----------------------------------------------------------------------

const MAX_TRIS: usize = (NUM_RINGS - 1) * 4 * 2;

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();

const TRI_ZERO: TriTexturedGouraud = TriTexturedGouraud {
    tag: 0,
    color0_cmd: 0,
    v0: 0,
    uv0_clut: 0,
    color1: 0,
    v1: 0,
    uv1_tpage: 0,
    color2: 0,
    v2: 0,
    uv2: 0,
};
static mut TRIS: [TriTexturedGouraud; MAX_TRIS] = [const { TRI_ZERO }; MAX_TRIS];

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
// HUD
// ----------------------------------------------------------------------

const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

struct Scene {
    frame: u32,
    camera_offset: i32,
    tri_count: u16,
    culled_count: u16,
}

static mut SCENE: Scene = Scene {
    frame: 0,
    camera_offset: 0,
    tri_count: 0,
    culled_count: 0,
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

    scene::set_screen_offset((SCREEN_W as i32 / 2) << 16, (SCREEN_H as i32 / 2) << 16);
    scene::set_projection_plane(PROJ_H);
    scene::set_avsz_weights(ZSF3, ZSF4);
    scene::load_far_colour(FOG_FC);
    scene::set_depth_cue(DQA, DQB);

    // --- Upload both wall textures + CLUTs ---
    let brick = Texture::from_bytes(BRICK_BLOB).expect("brick.psxt");
    let floor = Texture::from_bytes(FLOOR_BLOB).expect("floor.psxt");
    // Brick at U=0..63 (halfwords 0..16 of the tpage).
    upload_bytes(
        VramRect::new(
            TEX_TPAGE.x(),
            TEX_TPAGE.y(),
            brick.halfwords_per_row(),
            brick.height(),
        ),
        brick.pixel_bytes(),
    );
    upload_bytes(
        VramRect::new(BRICK_CLUT.x(), BRICK_CLUT.y(), brick.clut_entries(), 1),
        brick.clut_bytes(),
    );
    // Floor at U=64..127 (halfwords 16..32).
    upload_bytes(
        VramRect::new(
            TEX_TPAGE.x() + brick.halfwords_per_row(),
            TEX_TPAGE.y(),
            floor.halfwords_per_row(),
            floor.height(),
        ),
        floor.pixel_bytes(),
    );
    upload_bytes(
        VramRect::new(FLOOR_CLUT.x(), FLOOR_CLUT.y(), floor.clut_entries(), 1),
        floor.clut_bytes(),
    );

    let font = FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT);

    loop {
        update_scene();
        fb.clear(FOG_CLEAR.0, FOG_CLEAR.1, FOG_CLEAR.2);
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

fn update_scene() {
    let s = unsafe { &mut SCENE };
    s.frame = s.frame.wrapping_add(1);
    s.camera_offset += SCROLL_SPEED;
    if s.camera_offset >= RING_SPACING {
        s.camera_offset -= RING_SPACING;
    }
    s.tri_count = 0;
    s.culled_count = 0;
}

// ----------------------------------------------------------------------
// Render — build OT
// ----------------------------------------------------------------------

#[inline]
fn ring_z(i: usize, camera_offset: i32) -> i32 {
    FIRST_RING_Z + (i as i32) * RING_SPACING - camera_offset
}

#[inline]
fn ot_slot(otz: u16) -> usize {
    let slot = (otz as usize) >> 10;
    if slot >= OT_DEPTH - 1 {
        OT_DEPTH - 2
    } else {
        slot
    }
}

fn build_frame_ot() {
    let ot = unsafe { &mut OT };
    let tris = unsafe { &mut TRIS };
    let bg = unsafe { &mut BG_QUAD };
    let s = unsafe { &mut SCENE };
    ot.clear();

    // --- Background — solid fog-colour quad behind everything. ---
    *bg = QuadGouraud::new(
        [
            (0, 0),
            (SCREEN_W, 0),
            (0, SCREEN_H),
            (SCREEN_W, SCREEN_H),
        ],
        [FOG_CLEAR, FOG_CLEAR, FOG_CLEAR, FOG_CLEAR],
    );
    ot.add(OT_DEPTH - 1, bg, QuadGouraud::WORDS);

    // --- Scene setup for this frame ---
    scene::load_rotation(&Mat3I16::IDENTITY);
    scene::load_translation(Vec3I32::new(0, 0, 0));
    let light_rot = Mat3I16::rotate_y((s.frame >> LIGHT_ORBIT_SHIFT) as u16);
    BASE_RIG.rotated(&light_rot).load();

    // Tpage word embeds in every textured primitive's vertex-1 UV
    // slot. Semi-transparency bit = 0 (opaque).
    let tpage_word = TEX_TPAGE.uv_tpage_word(0);

    // Material `(128, 128, 128)` makes NCDT produce "identity" vertex
    // tints at full light — the textured-Gouraud primitive's
    // tint-multiply then passes texels unmodulated. At deep fog the
    // NCDT vertex tint approaches FC (fog colour), darkening and
    // blue-shifting texels so the far end fades into the clear.
    const MAT: (u8, u8, u8) = (0x80, 0x80, 0x80);

    let mut tri_idx = 0;
    for ring in 0..NUM_RINGS - 1 {
        let z_near = ring_z(ring, s.camera_offset);
        let z_far = ring_z(ring + 1, s.camera_offset);

        for wall in &WALLS {
            let (x0, y0) = wall.corners[0];
            let (x1, y1) = wall.corners[1];

            // Four corners:
            //   a -------- b   (at z_near)
            //   |          |
            //   d -------- c   (at z_far)
            let a = Vec3I16::new(x0, y0, z_near as i16);
            let b = Vec3I16::new(x1, y1, z_near as i16);
            let c = Vec3I16::new(x1, y1, z_far as i16);
            let d = Vec3I16::new(x0, y0, z_far as i16);

            let u_lo = wall.tex.u_origin();
            let u_hi = u_lo + 63;
            let clut_word = wall.tex.clut_word();

            // UV per corner — same mapping regardless of wall
            // orientation, because the cooker's centre-square
            // crop means the texture has no "up" direction:
            //   a → (u_lo, 0)     near left/top
            //   b → (u_hi, 0)     near right/top
            //   c → (u_hi, 63)    far right/bottom
            //   d → (u_lo, 63)    far left/bottom

            let tri_a = project_triangle_fogged([a, b, c], [wall.normal; 3], MAT);
            let tri_b = project_triangle_fogged([a, c, d], [wall.normal; 3], MAT);

            let uvs_a = [(u_lo, 0), (u_hi, 0), (u_hi, 63)];
            let uvs_b = [(u_lo, 0), (u_hi, 63), (u_lo, 63)];

            for (ft, uvs) in [(tri_a, uvs_a), (tri_b, uvs_b)].iter() {
                if !ft.front_facing {
                    s.culled_count = s.culled_count.wrapping_add(1);
                    continue;
                }
                if tri_idx >= tris.len() {
                    break;
                }
                let v = ft.verts;
                tris[tri_idx] = TriTexturedGouraud::new(
                    [(v[0].sx, v[0].sy), (v[1].sx, v[1].sy), (v[2].sx, v[2].sy)],
                    *uvs,
                    [
                        (v[0].r, v[0].g, v[0].b),
                        (v[1].r, v[1].g, v[1].b),
                        (v[2].r, v[2].g, v[2].b),
                    ],
                    clut_word,
                    tpage_word,
                );
                ot.add(ot_slot(ft.otz), &mut tris[tri_idx], TriTexturedGouraud::WORDS);
                tri_idx += 1;
                s.tri_count = s.tri_count.wrapping_add(1);
            }
        }
    }
}

fn submit_frame_ot() {
    unsafe { &OT }.submit();
}

// ----------------------------------------------------------------------
// HUD
// ----------------------------------------------------------------------

fn draw_hud(font: &FontAtlas) {
    let s = unsafe { &SCENE };

    font.draw_text(4, 4, "SHOWCASE-FOG", (220, 220, 250));
    font.draw_text(
        SCREEN_W - 168,
        4,
        "RTPT NCLIP AVSZ3 NCDT",
        (160, 200, 240),
    );

    font.draw_text(4, SCREEN_H - 20, "frame", (160, 160, 200));
    let frame = u16_hex((s.frame & 0xFFFF) as u16);
    font.draw_text(4 + 8 * 6, SCREEN_H - 20, frame.as_str(), (200, 240, 160));

    font.draw_text(SCREEN_W / 2 - 8 * 3, SCREEN_H - 20, "tri", (160, 160, 200));
    let tri = u16_hex(s.tri_count);
    font.draw_text(SCREEN_W / 2 + 8, SCREEN_H - 20, tri.as_str(), (240, 200, 160));

    font.draw_text(SCREEN_W - 104, SCREEN_H - 20, "cull", (160, 160, 200));
    let cull = u16_hex(s.culled_count);
    font.draw_text(SCREEN_W - 56, SCREEN_H - 20, cull.as_str(), (240, 160, 200));
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
