//! `showcase-room` -- render a cooked `.psxw` on PS1 hardware.
//!
//! End-to-end validation of the cooker → asset → runtime path:
//! `build.rs` cooks the editor's starter room (3×3 stone room
//! with floor + brick textures) into `OUT_DIR/room.psxw`, this
//! binary parses it through `psx_engine::RuntimeRoom`, and
//! `psx_engine::draw_room` walks every populated sector through
//! `WorldRenderPass::submit_textured_quad`.
//!
//! No character, no input-driven movement, no portals -- D-pad
//! orbits the camera around the room centre.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::{Texture, World as AssetWorld};
use psx_engine::{
    button, draw_room, Angle, App, Config, Ctx, DepthBand, DepthRange, OtFrame, PrimitiveArena,
    RuntimeRoom, Scene, WorldCamera, WorldProjection, WorldRenderMaterial, WorldRenderPass,
    WorldSurfaceOptions, WorldTriCommand, WorldVertex,
};
use psx_gpu::{material::TextureMaterial, ot::OrderingTable, prim::TriTextured};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

// `room.psxw` is produced by build.rs from the starter project.
// Slot 0 = floor.psxt, slot 1 = brick-wall.psxt -- pinned by the
// build-time assertion next to the cook call.
static ROOM_PSXW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/room.psxw"));
static FLOOR_BLOB: &[u8] = include_bytes!("../../../../assets/textures/floor.psxt");
static BRICK_BLOB: &[u8] = include_bytes!("../../../../assets/textures/brick-wall.psxt");

// VRAM layout: each 64x64 4bpp material owns a full tpage page.
// `draw_room`'s v1 UVs are page-relative and start at (0,0) for
// every material, so material selection must happen via tpage word,
// not by side-by-side packing inside one page.
const FLOOR_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);
const BRICK_TPAGE: Tpage = Tpage::new(704, 0, TexDepth::Bit4);
const BRICK_CLUT: Clut = Clut::new(0, 480);
const FLOOR_CLUT: Clut = Clut::new(0, 481);

const FLOOR_TPAGE_WORD: u16 = FLOOR_TPAGE.uv_tpage_word(0);
const BRICK_TPAGE_WORD: u16 = BRICK_TPAGE.uv_tpage_word(0);
const FLOOR_CLUT_WORD: u16 = FLOOR_CLUT.uv_clut_word();
const BRICK_CLUT_WORD: u16 = BRICK_CLUT.uv_clut_word();
const NEUTRAL_TINT: (u8, u8, u8) = (0x80, 0x80, 0x80);

// Projection: 320×240 PSX framebuffer, focal 320 → half-FOV ≈ 26°.
const SCREEN_CX: i16 = 160;
const SCREEN_CY: i16 = 120;
const FOCAL: i32 = 320;
const NEAR_Z: i32 = 64;
const FAR_Z: i32 = 8192;
const PROJECTION: WorldProjection = WorldProjection::new(SCREEN_CX, SCREEN_CY, FOCAL, NEAR_Z);

// Room centre for a 3×3 sector_size=1024 room -- half_w * S = 1536.
// Cells are corner-rooted at world (0, 0) by `draw_room`, so the
// orbit target sits at the geometric centre.
const ROOM_HALF_X: i32 = 1536;
const ROOM_HALF_Z: i32 = 1536;
const CAMERA_TARGET: WorldVertex = WorldVertex::new(ROOM_HALF_X, 0, ROOM_HALF_Z);

const CAMERA_Y: i32 = 1100;
const CAMERA_START_RADIUS: i32 = 2400;
const CAMERA_RADIUS_MIN: i32 = 1200;
const CAMERA_RADIUS_MAX: i32 = 4800;
const CAMERA_RADIUS_STEP: i32 = 64;
const CAMERA_START_YAW: Angle = Angle::from_q12(220);
const CAMERA_YAW_STEP_Q12: i16 = 12;

const OT_DEPTH: usize = 64;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 1);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(NEAR_Z, FAR_Z);

// 3×3 floors + 12 perimeter walls = 21 quads = 42 triangles.
// Round up to 256 for HUD overlay headroom and any future
// authored geometry past the perimeter.
const MAX_TEXTURED_TRIS: usize = 256;

const TRI_ZERO: TriTextured = TriTextured::new(
    [(0, 0), (0, 0), (0, 0)],
    [(0, 0), (0, 0), (0, 0)],
    0,
    0,
    (0, 0, 0),
);

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();
static mut TEXTURED_TRIS: [TriTextured; MAX_TEXTURED_TRIS] =
    [const { TRI_ZERO }; MAX_TEXTURED_TRIS];
static mut WORLD_COMMANDS: [WorldTriCommand; MAX_TEXTURED_TRIS] =
    [WorldTriCommand::EMPTY; MAX_TEXTURED_TRIS];

struct Showcase {
    /// Parsed at `init` time; `None` until then so the const
    /// constructor stays legal. Render falls through silently
    /// if parsing failed -- no panic on hardware.
    room: Option<RuntimeRoom<'static>>,
    camera_yaw: Angle,
    camera_radius: i32,
}

impl Showcase {
    const fn new() -> Self {
        Self {
            room: None,
            camera_yaw: CAMERA_START_YAW,
            camera_radius: CAMERA_START_RADIUS,
        }
    }
}

impl Scene for Showcase {
    fn init(&mut self, _ctx: &mut Ctx) {
        upload_textures();
        // Parse once at boot. If the blob is malformed the field
        // stays `None` and `render` skips the geometry pass --
        // beats panicking and locking the console.
        if let Ok(world) = AssetWorld::from_bytes(ROOM_PSXW) {
            self.room = Some(RuntimeRoom::from_world(world));
        }
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.is_held(button::RIGHT) {
            self.camera_yaw = self.camera_yaw.add_signed_q12(CAMERA_YAW_STEP_Q12);
        }
        if ctx.is_held(button::LEFT) {
            self.camera_yaw = self.camera_yaw.add_signed_q12(-CAMERA_YAW_STEP_Q12);
        }
        if ctx.is_held(button::UP) {
            self.camera_radius = (self.camera_radius - CAMERA_RADIUS_STEP).max(CAMERA_RADIUS_MIN);
        }
        if ctx.is_held(button::DOWN) {
            self.camera_radius = (self.camera_radius + CAMERA_RADIUS_STEP).min(CAMERA_RADIUS_MAX);
        }
    }

    fn render(&mut self, _ctx: &mut Ctx) {
        let camera = WorldCamera::orbit_yaw(
            PROJECTION,
            CAMERA_TARGET,
            CAMERA_Y,
            self.camera_radius,
            self.camera_yaw,
        );

        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_TRIS) };
        let mut world = unsafe { WorldRenderPass::new(&mut ot, &mut WORLD_COMMANDS) };

        if let Some(room) = self.room {
            let materials = [
                WorldRenderMaterial::both(TextureMaterial::opaque(
                    FLOOR_CLUT_WORD,
                    FLOOR_TPAGE_WORD,
                    NEUTRAL_TINT,
                )),
                WorldRenderMaterial::both(TextureMaterial::opaque(
                    BRICK_CLUT_WORD,
                    BRICK_TPAGE_WORD,
                    NEUTRAL_TINT,
                )),
            ];
            let options = WorldSurfaceOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE);
            draw_room(
                room.render(),
                &materials,
                &camera,
                options,
                &mut triangles,
                &mut world,
            );
        }

        world.flush();
        ot.submit();
    }
}

#[no_mangle]
fn main() -> ! {
    let mut scene = Showcase::new();
    let config = Config {
        clear_color: (5, 7, 12),
        ..Config::default()
    };
    App::run(config, &mut scene);
}

/// Upload `floor.psxt` and `brick-wall.psxt` to separate 4-bit
/// tpages with one CLUT row each.
fn upload_textures() {
    let floor = Texture::from_bytes(FLOOR_BLOB).expect("floor.psxt");
    let brick = Texture::from_bytes(BRICK_BLOB).expect("brick-wall.psxt");

    // The future compact `.psxw` format will pack each material's
    // tpage / clut straight from an embedded material table and
    // remove this hand-wired mapping. See
    // `docs/world-format-roadmap.md`.
    let floor_pix_rect = VramRect::new(
        FLOOR_TPAGE.x(),
        FLOOR_TPAGE.y(),
        floor.halfwords_per_row(),
        floor.height(),
    );
    upload_bytes(floor_pix_rect, floor.pixel_bytes());
    let floor_clut_rect = VramRect::new(FLOOR_CLUT.x(), FLOOR_CLUT.y(), floor.clut_entries(), 1);
    upload_clut(floor_clut_rect, floor.clut_bytes());

    let brick_pix_rect = VramRect::new(
        BRICK_TPAGE.x(),
        BRICK_TPAGE.y(),
        brick.halfwords_per_row(),
        brick.height(),
    );
    upload_bytes(brick_pix_rect, brick.pixel_bytes());
    let brick_clut_rect = VramRect::new(BRICK_CLUT.x(), BRICK_CLUT.y(), brick.clut_entries(), 1);
    upload_clut(brick_clut_rect, brick.clut_bytes());
}

/// Mirrors `upload_blend_clut` in showcase-textured-sprite -- sets
/// the 0x8000 (semi-transparency-disable) bit on every non-zero
/// CLUT entry so opaque textures don't accidentally trigger the
/// PSX's STP-bit blending path.
fn upload_clut(rect: VramRect, bytes: &[u8]) {
    let mut marked = [0u8; 512];
    assert!(bytes.len() <= marked.len());
    assert!(bytes.len().is_multiple_of(2));

    let mut i = 0;
    while i < bytes.len() {
        let raw = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let stamped = if raw == 0 { 0 } else { raw | 0x8000 };
        let pair = stamped.to_le_bytes();
        marked[i] = pair[0];
        marked[i + 1] = pair[1];
        i += 2;
    }

    upload_bytes(rect, &marked[..bytes.len()]);
}
