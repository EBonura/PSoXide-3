//! `editor-playtest` — render a level cooked from the editor.
//!
//! Loads `generated/level_manifest.rs` (a Rust source file the
//! editor produces via `psxed_project::playtest::write_package`)
//! containing one or more cooked `.psxw` rooms, a player spawn,
//! and optional entity markers. Parses the first room with
//! `psx_asset::World::from_bytes` and renders it through
//! `psx_engine::draw_room`.
//!
//! The placeholder manifest committed alongside this crate has
//! zero rooms — the example boots to a clear-coloured screen
//! until you run "Cook & Play" in the editor.
//!
//! Controls:
//! - D-pad LEFT / RIGHT — yaw camera around spawn.
//! - D-pad UP / DOWN    — pull camera in / push out.
//!
//! No collision, no movement past camera orbit. P5 wires the
//! renderer; later passes layer in player movement and entity
//! markers.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::{Texture, World as AssetWorld};
use psx_engine::{
    button, draw_room, App, Config, Ctx, DepthBand, DepthRange, OtFrame, PrimitiveArena,
    RuntimeRoom, Scene, WorldCamera, WorldProjection, WorldRenderPass, WorldSurfaceOptions,
    WorldTriCommand, WorldVertex,
};
use psx_gpu::{material::TextureMaterial, ot::OrderingTable, prim::TriTextured};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

mod generated {
    // Placeholder-state warnings: the manifest is committed with
    // an empty ROOMS list, so its `world_bytes` constants and
    // `ENTITIES` static look unused until the editor cooks real
    // data. Silence the cosmetic warnings without touching the
    // generated source (which has to round-trip cleanly).
    #![allow(dead_code)]
    include!("../generated/level_manifest.rs");
}

use generated::{PLAYER_SPAWN, ROOMS};

// Same texture VRAM layout as showcase-room. The cooker emits
// slot 0 = floor.psxt, slot 1 = brick-wall.psxt for the starter
// project (pinned by `starter_cook_pins_floor_to_slot_zero_and_brick_to_slot_one`
// in psxed-project tests).
static FLOOR_BLOB: &[u8] = include_bytes!("../../../../assets/textures/floor.psxt");
static BRICK_BLOB: &[u8] = include_bytes!("../../../../assets/textures/brick-wall.psxt");

const SHARED_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);
const FLOOR_CLUT: Clut = Clut::new(0, 481);
const BRICK_CLUT: Clut = Clut::new(0, 480);

const TPAGE_WORD: u16 = SHARED_TPAGE.uv_tpage_word(0);
const FLOOR_CLUT_WORD: u16 = FLOOR_CLUT.uv_clut_word();
const BRICK_CLUT_WORD: u16 = BRICK_CLUT.uv_clut_word();
const NEUTRAL_TINT: (u8, u8, u8) = (0x80, 0x80, 0x80);

const SCREEN_CX: i16 = 160;
const SCREEN_CY: i16 = 120;
const FOCAL: i32 = 320;
const NEAR_Z: i32 = 64;
const FAR_Z: i32 = 8192;
const PROJECTION: WorldProjection = WorldProjection::new(SCREEN_CX, SCREEN_CY, FOCAL, NEAR_Z);

const CAMERA_Y_OFFSET: i32 = 1100;
const CAMERA_START_RADIUS: i32 = 2400;
const CAMERA_RADIUS_MIN: i32 = 800;
const CAMERA_RADIUS_MAX: i32 = 5200;
const CAMERA_RADIUS_STEP: i32 = 64;
const CAMERA_START_YAW: u16 = 220;
const CAMERA_YAW_STEP: u16 = 12;

const OT_DEPTH: usize = 64;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 1);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(NEAR_Z, FAR_Z);

// 32×32 max room budget: 1024 floors + 1024 ceilings + walls.
// Cap at 2048 to match `MAX_ROOM_TRIANGLES`. 4096 tri buffer
// covers two passes plus HUD headroom.
const MAX_TEXTURED_TRIS: usize = 4096;

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

struct Playtest {
    /// Parsed room blob. `None` until `init` runs and only
    /// `Some` when the cooked bytes were valid. The placeholder
    /// manifest leaves this `None` and renders a clear screen.
    room: Option<RuntimeRoom<'static>>,
    /// Camera target — the spawn position from the manifest.
    target: WorldVertex,
    camera_yaw: u16,
    camera_radius: i32,
}

impl Playtest {
    const fn new() -> Self {
        Self {
            room: None,
            target: WorldVertex::ZERO,
            camera_yaw: CAMERA_START_YAW,
            camera_radius: CAMERA_START_RADIUS,
        }
    }
}

impl Scene for Playtest {
    fn init(&mut self, _ctx: &mut Ctx) {
        // Texture upload runs unconditionally — even with no
        // rooms cooked, the textures sit harmlessly in VRAM and
        // a later cook can reference them without re-uploading.
        upload_textures();

        if let Some(room_record) = ROOMS.first() {
            if let Ok(world) = AssetWorld::from_bytes(room_record.world_bytes) {
                self.room = Some(RuntimeRoom::from_world(world));
            }
            // Spawn → camera target. PLAYER_SPAWN is in the same
            // engine-unit room-local space the renderer uses.
            self.target = WorldVertex::new(PLAYER_SPAWN.x, PLAYER_SPAWN.y, PLAYER_SPAWN.z);
        }
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.is_held(button::RIGHT) {
            self.camera_yaw = self.camera_yaw.wrapping_add(CAMERA_YAW_STEP);
        }
        if ctx.is_held(button::LEFT) {
            self.camera_yaw = self.camera_yaw.wrapping_sub(CAMERA_YAW_STEP);
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
            self.target,
            CAMERA_Y_OFFSET,
            self.camera_radius,
            self.camera_yaw,
        );

        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_TRIS) };
        let mut world = unsafe { WorldRenderPass::new(&mut ot, &mut WORLD_COMMANDS) };

        if let Some(room) = self.room {
            let materials = [
                TextureMaterial::opaque(FLOOR_CLUT_WORD, TPAGE_WORD, NEUTRAL_TINT),
                TextureMaterial::opaque(BRICK_CLUT_WORD, TPAGE_WORD, NEUTRAL_TINT),
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
    let mut scene = Playtest::new();
    let config = Config {
        clear_color: (5, 7, 12),
        ..Config::default()
    };
    App::run(config, &mut scene);
}

/// Upload `floor.psxt` and `brick-wall.psxt` to the shared
/// 4-bit tpage with one CLUT row each. Layout matches
/// `showcase-room` so the cooker's slot ordering (slot 0 =
/// floor, slot 1 = brick) lines up by construction.
fn upload_textures() {
    let floor = Texture::from_bytes(FLOOR_BLOB).expect("floor.psxt");
    let brick = Texture::from_bytes(BRICK_BLOB).expect("brick-wall.psxt");

    let floor_pix_rect = VramRect::new(
        SHARED_TPAGE.x(),
        SHARED_TPAGE.y(),
        floor.halfwords_per_row(),
        floor.height(),
    );
    upload_bytes(floor_pix_rect, floor.pixel_bytes());
    let floor_clut_rect = VramRect::new(FLOOR_CLUT.x(), FLOOR_CLUT.y(), floor.clut_entries(), 1);
    upload_clut(floor_clut_rect, floor.clut_bytes());

    let brick_pix_rect = VramRect::new(
        SHARED_TPAGE.x() + floor.halfwords_per_row(),
        SHARED_TPAGE.y(),
        brick.halfwords_per_row(),
        brick.height(),
    );
    upload_bytes(brick_pix_rect, brick.pixel_bytes());
    let brick_clut_rect = VramRect::new(BRICK_CLUT.x(), BRICK_CLUT.y(), brick.clut_entries(), 1);
    upload_clut(brick_clut_rect, brick.clut_bytes());
}

/// Mirrors the CLUT-stamping helper in `showcase-room`: sets
/// the 0x8000 (semi-transparency-disable) bit on every non-zero
/// entry so opaque textures don't accidentally trigger STP-bit
/// blending.
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
