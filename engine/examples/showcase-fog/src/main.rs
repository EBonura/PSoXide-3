//! `showcase-fog` -- the full PS1-commercial GTE + textured-poly
//! pipeline in one demo.
//!
//! Every frame, for every triangle, the GTE executes:
//!
//!   RTPS  → project 1 vertex, producing its own depth-cue weight
//!   NCLIP → hardware back-face cull
//!   AVSZ3 → average-Z key for ordering-table insertion
//!   NCDS  → lit + depth-cue colour for that vertex
//!
//! The three NCDS-produced per-vertex colours become the per-vertex
//! *tint* on a **textured Gouraud** triangle (GP0 0x34), so the GPU
//! multiplies each sampled texel by its interpolated NCDS colour.
//! Result: textures that properly light and fog -- near walls show
//! crisp brick, far walls dissolve into the fog colour -- the
//! Silent-Hill-era look achieved through hardware ops only.
//!
//! # Scene
//!
//! A square corridor receding into the distance. 16 rings of wall
//! segments -- ceiling / floor / left / right -- scroll slowly toward
//! the camera and wrap once per segment. Lighting is ambient-only so
//! the texture tiling and fog gradient stay easy to judge.
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
//! 15 ring-gaps × 4 walls × 2 tris = **120 triangles / frame** --
//! each a `TriTexturedGouraud` (9 data words per primitive).
//!
//! Ported to `psx-engine` in Phase 3e. The `Scene` struct (renamed
//! to `Corridor` to avoid collision with the engine trait) now
//! holds the per-frame counters inline instead of a `static mut`
//! god-object. Ring-Z / primitive-arena statics remain -- they need
//! fixed bus addresses for DMA, which is exactly what `static mut`
//! gives for free.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::Texture;
use psx_engine::{App, Config, Ctx, DepthBand, DepthRange, OtFrame, PrimitiveArena, Scene};
use psx_font::{fonts::BASIC_8X16, FontAtlas};
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, TriTexturedGouraud};
use psx_gte::lighting::{project_triangle_fogged, Light, LightRig};
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

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
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 2);
const GTE_OTZ_RANGE: DepthRange = DepthRange::new(0, ((OT_DEPTH - 2) as i32) << 10);

// ----------------------------------------------------------------------
// Corridor geometry
// ----------------------------------------------------------------------

/// Sixteen rings, fifteen segment gaps. Each gap maps the full
/// 64x64 wall texture once instead of asking the GPU to repeat UVs
/// across a huge polygon. This is the PS1-friendly tradeoff: more
/// polygons, much less affine texture stretch.
const NUM_RINGS: usize = 16;
const RING_GAPS: usize = NUM_RINGS - 1;

/// World units per visible frame. One full tile pass takes roughly
/// 2.7 seconds at 60 Hz, slow enough to inspect segment seams.
const SCROLL_SPEED: i32 = 0x08;

/// World-space half-dimensions of the corridor cross-section.
/// Chosen so near-ring walls extend off-screen (the camera is
/// inside the corridor), while far-ring walls shrink to a small
/// rectangle in the centre.
const CORR_HALF_W: i16 = 0x0780;
const CORR_HALF_H: i16 = 0x0500;

/// Z of the corridor's near wall edge and its far wall edge. Rings
/// are distributed evenly across this span; the final ring lands one
/// world unit short because the range does not divide cleanly by 15,
/// which is visually irrelevant and keeps the intended far-fog tuning.
const NEAR_CAM_Z: i32 = 0x0800;
const FAR_CAM_Z: i32 = 0x5400;
const FIRST_RING_Z: i32 = NEAR_CAM_Z;
const RING_SPACING: i32 = (FAR_CAM_Z - NEAR_CAM_Z) / (RING_GAPS as i32);

// ----------------------------------------------------------------------
// Fog parameters
// ----------------------------------------------------------------------

/// Fog colour. Near-black -- critical, because `TriTexturedGouraud`
/// uses **multiplicative** tint (`final = texel × tint / 128`), not
/// additive blending toward a target colour. If FC were a bright
/// blue, full-fog tint `(0x20, 0x40, 0xA0)` multiplied with a
/// bright brick texel `(0xE0, 0x60, 0x20)` would give a **dark
/// brown-grey**, not fog blue -- so the BG quad (at pure fog blue)
/// and the far-wall pixels wouldn't match, and a hard seam
/// appeared at the vanishing point.
///
/// With FC near black, "full fog" = "near black", and brick × near
/// black = also near black -- so far walls blend seamlessly into
/// the black BG. Classic "tunnel fades into darkness" fog, as used
/// in Silent Hill caves, Wipeout night tunnels, etc.
///
/// A tiny blue tint keeps the darkness from reading as "scene
/// turned off" -- gives it atmospheric depth.
const FOG_FC: Vec3I32 = Vec3I32::new(0x0040, 0x0060, 0x00C0);

const FOG_CLEAR: (u8, u8, u8) = (
    (FOG_FC.x >> 4) as u8,
    (FOG_FC.y >> 4) as u8,
    (FOG_FC.z >> 4) as u8,
);

/// Fog gradient tuning.
///
/// With PROJ_H=280, corridor Z in [0x0800..0x5400]:
///   divisor_near = (H << 16) / 0x0800 ≈ 0x2300
///   divisor_far  = (H << 16) / 0x5400 ≈ 0x0350
///
/// Target: **0% fog at the near plane, 100% at the far plane.**
/// Each ring vertex gets its own depth-cue weight, so neighbouring
/// segments agree at shared Z boundaries. The texture intentionally
/// restarts once per segment, while the fog gradient remains smooth.
///
/// Solving:
///   0x2300 * DQA + DQB = 0          (IR0 = 0 at near)
///   0x0350 * DQA + DQB = 0x1000_000 (IR0 = 0x1000 at far)
///   → DQA = -0x0840, DQB = 0x0122_2000 (approximately)
const DQA: i16 = -0x0840;
const DQB: i32 = 0x0122_2000;

const ZSF3: i16 = 0x0555;
const ZSF4: i16 = 0x0400;

// ----------------------------------------------------------------------
// Lighting
// ----------------------------------------------------------------------

/// Purely-ambient light rig. No directional lights, no orbit.
///
/// The demo's point is to isolate the **fog** effect -- every
/// pixel's colour is driven by (material × ambient × fog), with
/// no orbiting highlight to compete for the eye's attention.
/// Adding directional lighting made the scene read as "randomly
/// lit walls" rather than "foggy corridor". Pure ambient + fog
/// reads unambiguously as atmospheric depth.
///
/// NCDS still runs on every vertex -- it just computes the
/// "lit" term as ambient-only (no directional contribution),
/// then layers the depth-cue blend on top. The full
/// RTPS → NCDS → NCLIP → AVSZ3 pipeline is still exercised.
///
/// BK is in **Q19.12**; `0x1000` is the "identity-brightness"
/// value that leaves a `(128, 128, 128)` RGBC unmodulated at
/// zero fog. Anything larger saturates; anything smaller dims.
const BASE_RIG: LightRig = LightRig::new(
    [Light::OFF, Light::OFF, Light::OFF],
    (0x1000, 0x1000, 0x1000),
);

// ----------------------------------------------------------------------
// Textures
// ----------------------------------------------------------------------

/// Shared tpage for both textures -- X=640 sits past the 640-wide
/// double-buffered framebuffer region (two 320×240 buffers at
/// Y=0..240 and 240..480). One 4bpp tpage holds up to 256 texels
/// per row, so both 64×64 textures fit side-by-side.
const TEX_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);

/// Brick wall -- ceilings + left/right walls.
static BRICK_BLOB: &[u8] = include_bytes!("../../../../assets/textures/brick-wall.psxt");
/// Cobblestone floor.
static FLOOR_BLOB: &[u8] = include_bytes!("../../../../assets/textures/floor.psxt");

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
    // Ceiling -- brick overhead.
    Wall {
        corners: [(-CORR_HALF_W, -CORR_HALF_H), (CORR_HALF_W, -CORR_HALF_H)],
        normal: Vec3I16::new(0, 0x1000, 0),
        tex: WallTex::Brick,
    },
    // Floor -- cobblestone.
    Wall {
        corners: [(CORR_HALF_W, CORR_HALF_H), (-CORR_HALF_W, CORR_HALF_H)],
        normal: Vec3I16::new(0, -0x1000, 0),
        tex: WallTex::Floor,
    },
    // Left wall -- brick.
    Wall {
        corners: [(-CORR_HALF_W, CORR_HALF_H), (-CORR_HALF_W, -CORR_HALF_H)],
        normal: Vec3I16::new(0x1000, 0, 0),
        tex: WallTex::Brick,
    },
    // Right wall -- brick.
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

/// OT + primitive backing storage in `.bss`. These stay `static mut`
/// because DMA walks them by physical bus address -- the linker
/// picks a stable address once, and the walker follows
/// tag->next-addr pointers the scene builds up each frame.
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

/// Renamed from the pre-engine `Scene` struct to avoid name
/// collision with [`psx_engine::Scene`] -- this is *our* scene
/// type; the trait is what the engine calls back into.
struct Corridor {
    font: Option<FontAtlas>,
    tri_count: u16,
    culled_count: u16,
}

/// Absolute Z of each ring, sorted ascending (index 0 = nearest the
/// camera). Updated every frame from the scroll phase, and kept as
/// `static mut` because the old demo shape expects stable storage.
static mut RING_Z: [i32; NUM_RINGS] = init_ring_z();

const fn init_ring_z() -> [i32; NUM_RINGS] {
    let mut z = [0i32; NUM_RINGS];
    let mut i = 0;
    while i < NUM_RINGS {
        z[i] = FIRST_RING_Z + (i as i32) * RING_SPACING;
        i += 1;
    }
    z
}

// ----------------------------------------------------------------------
// Scene impl
// ----------------------------------------------------------------------

impl Scene for Corridor {
    fn init(&mut self, _ctx: &mut Ctx) {
        // GTE one-time setup. The scene-wide screen offset, proj
        // plane, AVSZ weights, far-colour, and depth-cue constants
        // never change across frames -- load them once.
        scene::set_screen_offset((SCREEN_W as i32 / 2) << 16, (SCREEN_H as i32 / 2) << 16);
        scene::set_projection_plane(PROJ_H);
        scene::set_avsz_weights(ZSF3, ZSF4);
        scene::load_far_colour(FOG_FC);
        scene::set_depth_cue(DQA, DQB);

        // Upload both wall textures + CLUTs.
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

        // HUD font atlas.
        self.font = Some(FontAtlas::upload(&BASIC_8X16, FONT_TPAGE, FONT_CLUT));
    }

    fn update(&mut self, ctx: &mut Ctx) {
        update_ring_z(ctx.frame);

        // Reset per-frame counters. `build_frame_ot` will repopulate
        // them during render.
        self.tri_count = 0;
        self.culled_count = 0;
    }

    fn render(&mut self, ctx: &mut Ctx) {
        self.build_frame_ot();
        let font = self.font.as_ref().expect("font uploaded in init");
        self.draw_hud(font, ctx.frame);
    }
}

fn update_ring_z(frame: u32) {
    let spacing = RING_SPACING as u32;
    let phase = (((frame % spacing) * (SCROLL_SPEED as u32)) % spacing) as i32;
    let rings = unsafe { &mut RING_Z };
    let mut i = 0;
    while i < NUM_RINGS {
        rings[i] = FIRST_RING_Z + (i as i32) * RING_SPACING - phase;
        i += 1;
    }
}

// ----------------------------------------------------------------------
// OT build
// ----------------------------------------------------------------------

impl Corridor {
    fn build_frame_ot(&mut self) {
        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut tris = unsafe { PrimitiveArena::new(&mut TRIS) };
        let mut backgrounds = unsafe { PrimitiveArena::new(core::slice::from_mut(&mut BG_QUAD)) };

        // --- Background -- solid fog-colour quad behind everything. ---
        let Some(bg) = backgrounds.push(QuadGouraud::new(
            [(0, 0), (SCREEN_W, 0), (0, SCREEN_H), (SCREEN_W, SCREEN_H)],
            [FOG_CLEAR, FOG_CLEAR, FOG_CLEAR, FOG_CLEAR],
        )) else {
            return;
        };
        ot.add_packet(OT_DEPTH - 1, bg);

        // --- Scene setup for this frame ---
        // Static light rig -- no orbit, no directional lighting. Just
        // ambient + fog produces the entire colour variation on screen.
        scene::load_rotation(&Mat3I16::IDENTITY);
        scene::load_translation(Vec3I32::new(0, 0, 0));
        BASE_RIG.load();

        // Tpage word embeds in every textured primitive's vertex-1 UV
        // slot. Semi-transparency bit = 0 (opaque).
        let tpage_word = TEX_TPAGE.uv_tpage_word(0);

        // Material `(128, 128, 128)` makes NCDS produce "identity" vertex
        // tints at full light -- the textured-Gouraud primitive's
        // tint-multiply then passes texels unmodulated. At deep fog the
        // NCDS vertex tint approaches FC (fog colour), darkening and
        // blue-shifting texels so the far end fades into the clear.
        const MAT: (u8, u8, u8) = (0x80, 0x80, 0x80);

        let rings = unsafe { &RING_Z };
        for ring in 0..NUM_RINGS - 1 {
            let z_near = rings[ring];
            let z_far = rings[ring + 1];

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

                // UV per corner -- same mapping regardless of wall
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
                        self.culled_count = self.culled_count.wrapping_add(1);
                        continue;
                    }
                    let v = ft.verts;
                    let Some(tri) = tris.push(TriTexturedGouraud::new(
                        [(v[0].sx, v[0].sy), (v[1].sx, v[1].sy), (v[2].sx, v[2].sy)],
                        *uvs,
                        [
                            (v[0].r, v[0].g, v[0].b),
                            (v[1].r, v[1].g, v[1].b),
                            (v[2].r, v[2].g, v[2].b),
                        ],
                        clut_word,
                        tpage_word,
                    )) else {
                        break;
                    };

                    let slot = WORLD_BAND.slot::<OT_DEPTH>(GTE_OTZ_RANGE, ft.otz as i32);
                    ot.add_packet_slot(slot, tri);
                    self.tri_count = self.tri_count.wrapping_add(1);
                }
            }
        }

        ot.submit();
    }

    fn draw_hud(&self, font: &FontAtlas, frame: u32) {
        font.draw_text(4, 4, "SHOWCASE-FOG", (220, 220, 250));
        font.draw_text(SCREEN_W - 168, 4, "RTPS NCDS NCLIP AVSZ3", (160, 200, 240));

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

        font.draw_text(SCREEN_W - 104, SCREEN_H - 20, "cull", (160, 160, 200));
        let cull = u16_hex(self.culled_count);
        font.draw_text(SCREEN_W - 56, SCREEN_H - 20, cull.as_str(), (240, 160, 200));
    }
}

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() -> ! {
    let mut scene = Corridor {
        font: None,
        tri_count: 0,
        culled_count: 0,
    };
    let config = Config {
        screen_w: SCREEN_W as u16,
        screen_h: SCREEN_H as u16,
        clear_color: FOG_CLEAR,
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
