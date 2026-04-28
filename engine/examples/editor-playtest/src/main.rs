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
use psx_gte::transform::{cos_1_3_12, sin_1_3_12};
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

use generated::{PlaytestEntityRecord, ENTITIES, PLAYER_SPAWN, ROOMS};

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

// Tank-controls follow camera. Half-turn in PSX angle units —
// the follow camera sits opposite the player's facing so the
// player faces away from the camera, which is what tank
// controls want.
const HALF_TURN_Q12: u16 = 2048;
const FOLLOW_RADIUS: i32 = 1400;
const FOLLOW_HEIGHT: i32 = 700;
/// Player linear speed (engine units per frame at 60 Hz).
const PLAYER_SPEED: i32 = 32;
/// Player turn rate (PSX angle units per frame).
const PLAYER_YAW_STEP: u16 = 32;

const OT_DEPTH: usize = 64;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 1);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(NEAR_Z, FAR_Z);

// 32×32 max room budget: 1024 floors + 1024 ceilings + walls.
// Cap at 2048 to match `MAX_ROOM_TRIANGLES`. 4096 tri buffer
// covers room + entity markers (12 tris each) + HUD headroom.
const MAX_TEXTURED_TRIS: usize = 4096;

/// Half-extent of an entity marker cube, in engine units.
/// Visible at orbit-cam radius without dominating the room.
const MARKER_HALF: i32 = 96;
/// Y-up offset so a marker authored at room-floor height
/// (`entity.y == 0`) draws as a cube sitting on the floor with
/// its base at y=0 and top at y=2*MARKER_HALF. Lifts the
/// marker out of any z-fight with the floor.
const MARKER_LIFT: i32 = MARKER_HALF;
/// Tint applied to the brick texture when drawing markers.
/// Yellow-orange, distinctly different from the brick's natural
/// reddish tone, so markers stand out at a glance.
const MARKER_TINT: (u8, u8, u8) = (0xff, 0xa8, 0x40);

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
    /// Player position in room-local engine units. Initialised
    /// from `PLAYER_SPAWN`.
    player_x: i32,
    player_y: i32,
    player_z: i32,
    player_yaw: u16,
    /// `true` toggles a free-orbit camera around the spawn for
    /// debug inspection. Default = follow.
    free_orbit: bool,
    orbit_yaw: u16,
    orbit_radius: i32,
    /// Spawn position retained for orbit-mode targeting.
    spawn: WorldVertex,
}

impl Playtest {
    const fn new() -> Self {
        Self {
            room: None,
            player_x: 0,
            player_y: 0,
            player_z: 0,
            player_yaw: 0,
            free_orbit: false,
            orbit_yaw: CAMERA_START_YAW,
            orbit_radius: CAMERA_START_RADIUS,
            spawn: WorldVertex::ZERO,
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
        }
        // Spawn → player + retained orbit target. PLAYER_SPAWN is
        // in the same engine-unit room-local space the renderer
        // uses, so no further transformation is needed.
        self.player_x = PLAYER_SPAWN.x;
        self.player_y = PLAYER_SPAWN.y;
        self.player_z = PLAYER_SPAWN.z;
        self.player_yaw = PLAYER_SPAWN.yaw as u16;
        self.spawn = WorldVertex::new(PLAYER_SPAWN.x, PLAYER_SPAWN.y, PLAYER_SPAWN.z);
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.just_pressed(button::SELECT) {
            self.free_orbit = !self.free_orbit;
        }
        if self.free_orbit {
            // Orbit mode: D-pad yaws the camera + zooms.
            if ctx.is_held(button::RIGHT) {
                self.orbit_yaw = self.orbit_yaw.wrapping_add(CAMERA_YAW_STEP);
            }
            if ctx.is_held(button::LEFT) {
                self.orbit_yaw = self.orbit_yaw.wrapping_sub(CAMERA_YAW_STEP);
            }
            if ctx.is_held(button::UP) {
                self.orbit_radius =
                    (self.orbit_radius - CAMERA_RADIUS_STEP).max(CAMERA_RADIUS_MIN);
            }
            if ctx.is_held(button::DOWN) {
                self.orbit_radius =
                    (self.orbit_radius + CAMERA_RADIUS_STEP).min(CAMERA_RADIUS_MAX);
            }
        } else {
            // Tank-controls mode: D-pad LEFT / RIGHT yaws the
            // player; UP / DOWN walks forward / back. Camera
            // follows from behind. No collision yet.
            if ctx.is_held(button::RIGHT) {
                self.player_yaw = self.player_yaw.wrapping_add(PLAYER_YAW_STEP);
            }
            if ctx.is_held(button::LEFT) {
                self.player_yaw = self.player_yaw.wrapping_sub(PLAYER_YAW_STEP);
            }
            // PSX trig: angle 0 → +Z, angle 1024 (90°) → +X. The
            // player's forward direction is (sin(yaw), cos(yaw))
            // as a Q1.3.12 unit vector.
            let sin_y = sin_1_3_12(self.player_yaw) as i32;
            let cos_y = cos_1_3_12(self.player_yaw) as i32;
            if ctx.is_held(button::UP) {
                self.player_x += (sin_y * PLAYER_SPEED) >> 12;
                self.player_z += (cos_y * PLAYER_SPEED) >> 12;
            }
            if ctx.is_held(button::DOWN) {
                self.player_x -= (sin_y * PLAYER_SPEED) >> 12;
                self.player_z -= (cos_y * PLAYER_SPEED) >> 12;
            }
        }
    }

    fn render(&mut self, _ctx: &mut Ctx) {
        let camera = if self.free_orbit {
            WorldCamera::orbit_yaw(
                PROJECTION,
                self.spawn,
                CAMERA_Y_OFFSET,
                self.orbit_radius,
                self.orbit_yaw,
            )
        } else {
            // Camera target = player; orbit yaw = player_yaw + π
            // so the camera sits behind the player along the
            // forward axis. `orbit_yaw` places the camera at
            // `target + (sin*r, _, cos*r)` and looks back at
            // `target`, so the +π flip puts it on the opposite
            // side of player from the forward direction.
            let target = WorldVertex::new(self.player_x, self.player_y, self.player_z);
            WorldCamera::orbit_yaw(
                PROJECTION,
                target,
                self.player_y + FOLLOW_HEIGHT,
                FOLLOW_RADIUS,
                self.player_yaw.wrapping_add(HALF_TURN_Q12),
            )
        };

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
            draw_entity_markers(ENTITIES, &camera, options, &mut triangles, &mut world);
        }

        world.flush();
        ot.submit();
    }
}

/// Draw one tinted cube per generated entity record. Cubes
/// share the brick texture with a yellow-orange tint so they
/// stand out from the cobblestone floor + reddish brick walls
/// without needing a dedicated texture upload.
///
/// No-alloc: emits at most `entities.len() * 6` quads through
/// the same `WorldRenderPass` the room uses.
fn draw_entity_markers(
    entities: &[PlaytestEntityRecord],
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
) {
    if entities.is_empty() {
        return;
    }
    let material = TextureMaterial::opaque(BRICK_CLUT_WORD, TPAGE_WORD, MARKER_TINT);
    let opts = options.with_material_layer(material);
    // Whole-cube UVs match the floor's 64×64 tile so the brick
    // pattern wraps cleanly across each face. Reuse for every
    // face to keep the helper trivial.
    const UVS: [(u8, u8); 4] = [(0, 0), (64, 0), (64, 64), (0, 64)];

    for entity in entities {
        let cx = entity.x;
        let cy = entity.y - MARKER_LIFT - MARKER_HALF;
        let cz = entity.z;
        let h = MARKER_HALF;

        // Six faces. Each `[a, b, c, d]` picks four corners in
        // the (NW, NE, SE, SW)-style winding `submit_textured_quad`
        // expects (split into triangles 0-1-2 and 2-1-3).
        // Top (y - h, "lower" if +Y is down):
        let top = [
            WorldVertex::new(cx - h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy - h, cz + h),
        ];
        // Bottom:
        let bottom = [
            WorldVertex::new(cx - h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz - h),
        ];
        // North (z = cz - h):
        let north = [
            WorldVertex::new(cx - h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy + h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz - h),
        ];
        // South (z = cz + h):
        let south = [
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
        ];
        // East (x = cx + h):
        let east = [
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz - h),
        ];
        // West (x = cx - h):
        let west = [
            WorldVertex::new(cx - h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy - h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz + h),
        ];

        for face in [top, bottom, north, south, east, west] {
            if let Some(projected) = camera.project_world_quad(face) {
                let _ = world.submit_textured_quad(triangles, projected, UVS, material, opts);
            }
        }
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
