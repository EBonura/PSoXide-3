//! `editor-playtest` -- render a level cooked from the editor.
//!
//! Loads a Rust manifest selected by `build.rs`: the ignored
//! `generated/level_manifest.cooked.rs` when the editor has
//! cooked a project, otherwise the tracked placeholder
//! `generated/level_manifest.rs`. The cooked manifest contains:
//!
//! * a master [`LevelAssetRecord`] table -- every cooked
//!   `.psxw` room blob and `.psxt` texture blob is a record;
//! * per-room [`LevelMaterialRecord`]s mapping each cooked
//!   local material slot to a texture asset id;
//! * per-room [`RoomResidencyRecord`]s declaring required
//!   RAM/VRAM assets;
//! * a [`PlayerSpawnRecord`] and [`EntityRecord`]s.
//!
//! The runtime resolves the active room by walking `ASSETS`,
//! uploads its texture assets through a tiny no-alloc
//! [`ResidencyManager`], builds a `TextureMaterial` table from
//! the room's material slice, and renders. No hardcoded starter
//! textures -- the asset table is the source of truth.
//!
//! Controls (free-orbit toggled with SELECT):
//! * Left stick / D-pad -- camera-relative movement.
//! * Right stick        -- camera yaw; vertical adjusts camera height.
//! * CIRCLE            -- run while moving.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::{Animation, Model, Texture};
#[cfg(not(feature = "world-grid-visible"))]
use psx_engine::draw_room_vertex_lit;
use psx_engine::{
    button, compute_joint_world_transform, telemetry, Angle, App, CharacterMotorAnim,
    CharacterMotorConfig, CharacterMotorInput, CharacterMotorState, Config, Ctx, CullMode,
    DepthBand, DepthPolicy, DepthRange, JointViewTransform, JointWorldTransform, LocalToWorldScale,
    Mat3I16, MaterialTint, OtFrame, PointLightSample, PrimitiveArena, ProjectedVertex, Rgb8,
    RoomPoint, RuntimeRoom, Scene, TexturedModelRenderStats, ThirdPersonCameraConfig,
    ThirdPersonCameraInput, ThirdPersonCameraState, ThirdPersonCameraTarget, WorldCamera,
    WorldProjection, WorldRenderMaterial, WorldRenderPass, WorldSurfaceLighting,
    WorldSurfaceOptions, WorldSurfaceSample, WorldTriCommand, WorldVertex, Q8,
};
#[cfg(feature = "world-grid-visible")]
use psx_engine::{draw_room_vertex_lit_grid_visible, GridVisibility};
use psx_font::{fonts::BASIC, FontAtlas};
use psx_gpu::{
    draw_quad_flat,
    material::{TextureMaterial, TextureWindow},
    ot::OrderingTable,
    prim::{TriTextured, TriTexturedGouraud},
};
use psx_level::{
    equipment_flags, find_asset_of_kind, room_flags, AssetId, AssetKind, EntityRecord,
    LevelCharacterRecord, LevelMaterialRecord, LevelMaterialSidedness, LevelModelRecord,
    LevelModelSocketRecord, LevelRoomRecord, ModelClipIndex, ModelClipTableIndex, ModelIndex,
    ModelSocketIndex, OptionalModelClipIndex, ResidencyManager, RoomIndex, WeaponHitShapeRecord,
};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

mod input;
mod overlay;
mod vram_upload;

use input::*;
use overlay::*;
use vram_upload::*;

// Placeholder manifests reference unused statics; populated
// manifests reference all of them. Quiet either side here.
#[allow(dead_code, unused_imports)]
mod generated {
    include!(env!("PSXED_PLAYTEST_MANIFEST"));
}

use generated::{
    ASSETS, CHARACTERS, ENTITIES, EQUIPMENT, LIGHTS, MATERIALS, MODELS, MODEL_CLIPS,
    MODEL_INSTANCES, MODEL_SOCKETS, PLAYER_CONTROLLER, PLAYER_SPAWN, ROOMS, ROOM_RESIDENCY,
    WEAPONS, WEAPON_HITBOXES,
};

// VRAM layout. Room materials and model atlases live in
// disjoint regions so a model atlas upload never overwrites a
// room texture (and vice versa).
//
// Room materials: 4bpp pages starting at (640, 0), packed as 64x64
// tiles inside each tpage. Each material carries GP0(E2)
// texture-window state so authored UV repetition samples only its tile
// instead of requiring physically repeated texels.
//
// Model atlases: 8bpp tpage at (384, 256); stripe atlases
// left-to-right (each atlas occupies its own halfword stride);
// one CLUT row per atlas at y starting at 484 (below the
// material CLUT band so the two never collide).
const SHARED_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);
const TPAGE_WORD: u16 = SHARED_TPAGE.uv_tpage_word(0);
const ROOM_TPAGE_STRIDE_HW: u16 = 64;
const ROOM_TPAGE_LIMIT_X: u16 = 1024;
const ROOM_TILE_TEXELS: u16 = 64;
const ROOM_TILE_HALFWORDS: u16 = ROOM_TILE_TEXELS / 4;
const ROOM_TILE_ROWS: u16 = 4;
const ROOM_TILE_COLUMNS: u16 = 4;
const ROOM_TILES_PER_PAGE: u16 = ROOM_TILE_ROWS * ROOM_TILE_COLUMNS;
/// CLUT strip used by room material textures. Keep it outside the
/// 320-pixel-wide double-buffered framebuffer (`x=0..319`,
/// `y=0..479`) so frame clears cannot overwrite palettes.
const ROOM_CLUT_BASE_X: u16 = 320;
const ROOM_CLUT_STRIDE: u16 = 16;
const ROOM_CLUT_Y: u16 = 480;

const MODEL_TPAGE: Tpage = Tpage::new(384, 256, TexDepth::Bit8);
const MODEL_TPAGE_WORD: u16 = MODEL_TPAGE.uv_tpage_word(0);
/// First CLUT row used by model atlases. 256-entry CLUTs span
/// a single row; we step one row down per uploaded atlas, so
/// `MODEL_CLUT_BASE_Y + n` is the row for the n-th atlas.
const MODEL_CLUT_BASE_Y: u16 = 484;

/// 4bpp 8x8 BIOS-style font atlas for the analog-mode gate prompt.
const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
const FONT_CLUT: Clut = Clut::new(320, 256);

const SCREEN_W: i16 = 320;
const SCREEN_H: i16 = 240;
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
const CAMERA_START_YAW: Angle = Angle::from_q12(220);
const CAMERA_YAW_STEP: Angle = Angle::from_q12(12);
const MOVE_STICK_DEADZONE: i16 = 18;
const STICK_MAX: i16 = 127;
const CAMERA_STICK_DEADZONE: i16 = 18;
const CAMERA_STICK_YAW_STEP: i16 = 64;
const CAMERA_STICK_PITCH_STEP: i16 = 48;
const CAMERA_SOFT_LOCK_BREAK_STICK: i16 = 72;
const LOCK_RANGE: i32 = 4096;
const LOCK_BREAK_RANGE: i32 = 5120;
const SOFT_LOCK_RANGE: i32 = 3072;
const SOFT_LOCK_BREAK_RANGE: i32 = 3840;
const CAMERA_COLLISION_ENABLED: bool = true;
const SOFT_LOCK_ENABLED: bool = false;

/// Fallback follow camera params used when no PLAYER_CONTROLLER
/// was authored -- matches the prior debug behaviour.
const FOLLOW_RADIUS_DEFAULT: i32 = 1400;
const FOLLOW_HEIGHT_DEFAULT: i32 = 700;
const FOLLOW_TARGET_HEIGHT_DEFAULT: i32 = 0;
/// Quanta-per-frame turn rate when the runtime can't resolve a
/// Character (no PLAYER_CONTROLLER). Mirrors the pre-character
/// debug value.
const FALLBACK_PLAYER_YAW_STEP: Angle = Angle::from_q12(32);
const FALLBACK_PLAYER_SPEED: i32 = 32;
const RUN_BUTTON: u16 = button::CIRCLE;

#[cfg(feature = "ot-2048")]
const OT_DEPTH: usize = 2048;
#[cfg(all(not(feature = "ot-2048"), feature = "ot-1024"))]
const OT_DEPTH: usize = 1024;
#[cfg(all(not(feature = "ot-2048"), not(feature = "ot-1024")))]
const OT_DEPTH: usize = 512;
/// Keep dynamic actors in the nearest ordering-table band so large
/// split room quads cannot overpaint characters in the no-Z-buffer
/// runtime. Room geometry starts after this reserved band.
const ACTOR_BAND_BACK: usize = 63;
const ROOM_BAND: DepthBand = DepthBand::new(ACTOR_BAND_BACK + 1, OT_DEPTH - 1);
const ACTOR_BAND: DepthBand = DepthBand::new(0, ACTOR_BAND_BACK);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(NEAR_Z, FAR_Z);
#[cfg(feature = "world-grid-visible")]
const ROOM_GRID_VISIBILITY_RADIUS: u16 = 4;

const MAX_TEXTURED_TRIS: usize = 4096;

/// Cap on the per-room material slot count. Picked to comfortably
/// exceed the cooker's currently-emitted material count without
/// over-reserving VRAM or RAM. If a future room exceeds this,
/// the runtime fails graceful (skips the over-cap material) and
/// the cook report should also flag.
const MAX_ROOM_MATERIALS: usize = 32;
/// Current generated chunk plus the eight chunks touching it.
/// Later visibility/portal logic can shrink this set; the first
/// streaming pass keeps the active window deliberately simple.
const MAX_ACTIVE_ROOMS: usize = 9;

/// Capacity of the residency manager's RAM table. Holds room
/// world + model meshes + animation clips.
const MAX_RESIDENT_RAM_ASSETS: usize = 128;
/// Capacity of the residency manager's VRAM table. Holds room
/// material atlases + model atlases.
const MAX_RESIDENT_VRAM_ASSETS: usize = 32;

/// Per-frame projected-vertex scratch for the model renderer.
/// Sized to the largest part vertex count we expect; instances
/// over this cap drop their over-budget triangles graceful.
const MODEL_VERTEX_CAP: usize = 1024;
/// Joint-transform scratch -- all biped rigs we currently cook
/// fit comfortably in 32.
const JOINT_CAP: usize = 32;
/// Cap on placed model instances rendered per frame.
const MAX_MODEL_INSTANCES: usize = 16;
/// Cap on attached weapon/equipment visuals rendered per frame.
const MAX_EQUIPMENT_DRAWS: usize = 8;
/// Runtime model cache capacity. The current playtest package only
/// needs one player model, but this keeps a little headroom for
/// lightweight NPC experiments without introducing heap allocation.
const MAX_RUNTIME_MODELS: usize = 8;
/// Runtime animation cache capacity. Matches the residency table
/// scale and avoids reparsing `.psxanim` headers per frame.
const MAX_RUNTIME_MODEL_CLIPS: usize = 32;

/// Marker visualization tuning. Markers are debug stubs -- keep
/// them visible at orbit-camera scales without dominating the
/// scene.
const MARKER_HALF: i32 = 96;
const MARKER_LIFT: i32 = MARKER_HALF;
const MARKER_TINT: (u8, u8, u8) = (0xff, 0xa8, 0x40);
const TRI_ZERO: TriTextured = TriTextured::new(
    [(0, 0), (0, 0), (0, 0)],
    [(0, 0), (0, 0), (0, 0)],
    0,
    0,
    (0, 0, 0),
);
const GOURAUD_TRI_ZERO: TriTexturedGouraud = TriTexturedGouraud {
    tag: 0,
    tex_window: 0,
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

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();
static mut TEXTURED_TRIS: [TriTextured; MAX_TEXTURED_TRIS] =
    [const { TRI_ZERO }; MAX_TEXTURED_TRIS];
static mut TEXTURED_GOURAUD_TRIS: [TriTexturedGouraud; MAX_TEXTURED_TRIS] =
    [const { GOURAUD_TRI_ZERO }; MAX_TEXTURED_TRIS];
static mut WORLD_COMMANDS: [WorldTriCommand; MAX_TEXTURED_TRIS] =
    [WorldTriCommand::EMPTY; MAX_TEXTURED_TRIS];
static mut MODEL_VERTICES: [ProjectedVertex; MODEL_VERTEX_CAP] =
    [ProjectedVertex::new(0, 0, 0); MODEL_VERTEX_CAP];
static mut JOINT_VIEW_TRANSFORMS: [JointViewTransform; JOINT_CAP] =
    [JointViewTransform::ZERO; JOINT_CAP];

/// Residency manager -- tracks which AssetIds are RAM/VRAM
/// resident across frames. Static so it survives across the
/// `Scene::init` → `Scene::render` boundary.
static mut RESIDENCY: ResidencyManager<MAX_RESIDENT_RAM_ASSETS, MAX_RESIDENT_VRAM_ASSETS> =
    ResidencyManager::new();

/// Per-asset upload bookkeeping. When a texture asset becomes
/// VRAM-resident we record its CLUT word, tpage word, and texture
/// window so the per-frame material build can reconstruct its
/// `TextureMaterial` without re-walking the upload code.
#[derive(Copy, Clone)]
struct VramSlot {
    asset: AssetId,
    clut_word: u16,
    tpage_word: u16,
    texture_window: TextureWindow,
}

const VRAM_SLOT_EMPTY: Option<VramSlot> = None;
static mut VRAM_SLOTS: [Option<VramSlot>; MAX_RESIDENT_VRAM_ASSETS] =
    [VRAM_SLOT_EMPTY; MAX_RESIDENT_VRAM_ASSETS];
/// Number of VRAM slots used so far across room textures and model atlases.
static mut VRAM_SLOT_COUNT: usize = 0;
/// Number of room material textures uploaded. Drives the per-material
/// tpage page and CLUT row; kept separate from `VRAM_SLOT_COUNT` so
/// model atlas uploads cannot shift room texture addressing.
static mut ROOM_TEXTURE_COUNT: usize = 0;

/// Tpage X cursor (in halfwords) for the model-atlas 8bpp
/// region. Distinct cursor so room-material uploads don't shift
/// model atlas positions and vice versa.
static mut MODEL_TPAGE_X_CURSOR: u16 = 0;
/// Number of model atlases uploaded so far. Doubles as the
/// CLUT row offset: each 8bpp atlas needs a fresh 256-entry
/// CLUT row.
static mut MODEL_ATLAS_COUNT: usize = 0;

/// Animation state machine for the player: idle with no movement,
/// walking for normal movement, running while Circle is held.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PlayerAnim {
    Idle,
    Walk,
    Run,
}

/// Runtime view of the cooked LevelCharacterRecord -- the same
/// fields, decoded into runtime-friendly types. Resolved once
/// at init time so per-frame movement / animation / camera code
/// doesn't keep re-resolving the manifest.
#[derive(Copy, Clone, Debug)]
struct RuntimeCharacter {
    /// Index into `MODELS`.
    model: ModelIndex,
    idle_clip: ModelClipIndex,
    walk_clip: ModelClipIndex,
    /// Optional run clip -- `CHARACTER_CLIP_NONE` when unset.
    /// Runtime falls back to `walk_clip` for run input.
    run_clip: OptionalModelClipIndex,
    /// Optional turn clip (currently unused at runtime -- turn
    /// is folded into idle with yaw input).
    _turn_clip: OptionalModelClipIndex,
    /// Capsule radius for collision. Engine units.
    radius: i32,
    walk_speed: i32,
    run_speed: i32,
    /// Yaw rate translated from degrees/second to PSX angle
    /// units / 60 Hz frame at init time.
    yaw_step: Angle,
    camera_distance: i32,
    camera_height: i32,
    camera_target_height: i32,
}

impl RuntimeCharacter {
    /// Resolve the cooked record into the runtime's preferred
    /// units. Yaw is converted from degrees/second to per-frame
    /// quanta (`4096 quanta = full turn`, runtime targets 60 Hz)
    /// up-front so the per-frame update path is just a wrapping
    /// add.
    const fn from_record(c: &LevelCharacterRecord) -> Self {
        // 4096 q12 / 360 deg = 11 q12 per deg, divided by
        // 60 Hz target ≈ 0.19 q12 per deg/frame. We approximate
        // as `(deg * 4096) / (360 * 60)` which is exact for the
        // 180 deg/s default (= 34 quanta/frame).
        let yaw_step_q12 = ((c.turn_speed_degrees_per_second as u32 * 4096) / (360 * 60)) as u16;
        Self {
            model: c.model,
            idle_clip: c.idle_clip,
            walk_clip: c.walk_clip,
            run_clip: c.run_clip,
            _turn_clip: c.turn_clip,
            radius: c.radius as i32,
            walk_speed: c.walk_speed,
            run_speed: c.run_speed,
            yaw_step: Angle::from_q12(yaw_step_q12),
            camera_distance: c.camera_distance,
            camera_height: c.camera_height,
            camera_target_height: c.camera_target_height,
        }
    }

    /// Pick the clip index for an animation state, with the
    /// "run falls back to walk when unassigned" rule.
    fn clip_for(&self, anim: PlayerAnim) -> ModelClipIndex {
        match anim {
            PlayerAnim::Idle => self.idle_clip,
            PlayerAnim::Walk => self.walk_clip,
            PlayerAnim::Run => self.run_clip.unwrap_or(self.walk_clip),
        }
    }

    fn motor_config(&self) -> CharacterMotorConfig {
        CharacterMotorConfig::character(self.radius, self.walk_speed, self.run_speed, self.yaw_step)
    }
}

/// Parsed, VRAM-bound model payload ready for the hot render path.
#[derive(Copy, Clone)]
struct RuntimeModelAsset {
    model: Model<'static>,
    material: TextureMaterial,
    clip_first: ModelClipTableIndex,
    clip_count: u16,
    default_clip: ModelClipIndex,
    socket_first: ModelSocketIndex,
    socket_count: u16,
    world_height: u16,
    local_to_world: LocalToWorldScale,
}

impl RuntimeModelAsset {
    fn from_record(record: &LevelModelRecord) -> Option<Self> {
        let mesh_asset = find_asset_of_kind(ASSETS, record.mesh_asset, AssetKind::ModelMesh)?;
        let model = Model::from_bytes(mesh_asset.bytes).ok()?;
        let texture_asset = record.texture_asset?;
        let atlas_asset = find_asset_of_kind(ASSETS, texture_asset, AssetKind::Texture)?;
        let atlas_slot = ensure_model_atlas_uploaded(atlas_asset.id, atlas_asset.bytes)?;
        Some(Self {
            model,
            material: TextureMaterial::opaque(
                atlas_slot.clut_word,
                atlas_slot.tpage_word,
                (0x80, 0x80, 0x80),
            ),
            clip_first: record.clip_first,
            clip_count: record.clip_count,
            default_clip: record.default_clip,
            socket_first: record.socket_first,
            socket_count: record.socket_count,
            world_height: record.world_height,
            local_to_world: LocalToWorldScale::from_q12(model.local_to_world_q12()),
        })
    }

    fn clip(
        self,
        clips: &[Option<Animation<'static>>; MAX_RUNTIME_MODEL_CLIPS],
        local_clip: ModelClipIndex,
    ) -> Option<Animation<'static>> {
        if local_clip.raw() >= self.clip_count {
            return None;
        }
        let index = self.clip_first.to_usize() + local_clip.to_usize();
        clips.get(index).copied().flatten()
    }
}

#[derive(Copy, Clone)]
struct ActiveRuntimeRoom {
    index: RoomIndex,
    room: RuntimeRoom<'static>,
    materials: [Option<WorldRenderMaterial>; MAX_ROOM_MATERIALS],
    material_count: usize,
    /// Offset from the current chunk's origin to this chunk's
    /// origin, in engine units.
    offset_x: i32,
    offset_z: i32,
}

struct Playtest {
    /// Active room. `None` until `init` runs and only `Some`
    /// when the manifest had at least one room and its bytes
    /// parsed.
    room: Option<RuntimeRoom<'static>>,
    /// Current generated chunk plus immediately touching chunks,
    /// all expressed relative to `room_index`.
    active_rooms: [Option<ActiveRuntimeRoom>; MAX_ACTIVE_ROOMS],
    /// Index in ROOMS the player is currently in. Used to scope
    /// model-instance + light queries.
    room_index: RoomIndex,
    /// Active room's material table, ordered by `local_slot`.
    /// Indexed directly by the slot value the cooked `.psxw`
    /// stores per face.
    materials: [Option<WorldRenderMaterial>; MAX_ROOM_MATERIALS],
    /// `materials[..material_count]` is the in-use slice; rest
    /// is `None`.
    material_count: usize,
    /// Player locomotion state: position, yaw, stamina, and evade actions.
    motor: CharacterMotorState,
    /// Resolved Character driving the player -- `None` when no
    /// `PLAYER_CONTROLLER` was authored. Falls back to the
    /// pre-character debug controls in that case.
    character: Option<RuntimeCharacter>,
    /// Current animation state. Source of truth for which clip
    /// `draw_player` plays each frame.
    anim_state: PlayerAnim,
    /// Tick the current animation started at -- used to phase
    /// the loop relative to clip switches so transitions don't
    /// pop into the middle of the new clip.
    anim_start_tick: u32,
    /// `true` toggles a free-orbit camera around the spawn for
    /// debug inspection. Default = follow.
    free_orbit: bool,
    orbit_yaw: Angle,
    orbit_radius: i32,
    /// Runtime third-person camera rig. Updated from render so it
    /// can consume the same room collision view used for drawing.
    camera: ThirdPersonCameraState,
    /// Index into `ENTITIES` for the current lock-on target. In this
    /// vertical-slice pass generic entity markers stand in for
    /// enemies until enemy records exist.
    lock_target: Option<usize>,
    /// Automatic camera-only target. Suppressed after strong
    /// manual camera input until the player leaves target range.
    soft_lock_target: Option<usize>,
    soft_lock_suppressed: bool,
    /// Spawn position retained for orbit-mode targeting.
    spawn: RoomPoint,
    /// Font atlas used for the analog-mode required prompt.
    font: Option<FontAtlas>,
    /// Parsed models/materials, built once at init.
    models: [Option<RuntimeModelAsset>; MAX_RUNTIME_MODELS],
    /// Parsed animations, indexed like `MODEL_CLIPS`.
    clips: [Option<Animation<'static>>; MAX_RUNTIME_MODEL_CLIPS],
}

impl Playtest {
    const fn new() -> Self {
        Self {
            room: None,
            active_rooms: [const { None }; MAX_ACTIVE_ROOMS],
            room_index: RoomIndex::ZERO,
            materials: [const { None }; MAX_ROOM_MATERIALS],
            material_count: 0,
            motor: CharacterMotorState::new(RoomPoint::ZERO, Angle::ZERO),
            character: None,
            anim_state: PlayerAnim::Idle,
            anim_start_tick: 0,
            free_orbit: false,
            orbit_yaw: CAMERA_START_YAW,
            orbit_radius: CAMERA_START_RADIUS,
            camera: ThirdPersonCameraState::new(CAMERA_START_YAW),
            lock_target: None,
            soft_lock_target: None,
            soft_lock_suppressed: false,
            spawn: RoomPoint::ZERO,
            font: None,
            models: [const { None }; MAX_RUNTIME_MODELS],
            clips: [const { None }; MAX_RUNTIME_MODEL_CLIPS],
        }
    }
}

impl Scene for Playtest {
    fn init(&mut self, _ctx: &mut Ctx) {
        self.font = Some(FontAtlas::upload(&BASIC, FONT_TPAGE, FONT_CLUT));

        // Empty manifest? Boot to a clear-coloured screen.
        if ROOMS.is_empty() {
            return;
        };

        // Player init: prefer PLAYER_CONTROLLER (cook output)
        // for spawn + character; fall back to the bare
        // PLAYER_SPAWN for placeholder manifests. The spawn room
        // may be a generated chunk rather than room zero.
        let (spawn, character) = match PLAYER_CONTROLLER {
            Some(pc) => {
                let character = CHARACTERS
                    .get(pc.character.to_usize())
                    .map(RuntimeCharacter::from_record);
                (pc.spawn, character)
            }
            None => (PLAYER_SPAWN, None),
        };
        if ROOMS.get(spawn.room.to_usize()).is_none() {
            return;
        };
        self.load_runtime_models();
        self.spawn = RoomPoint::new(spawn.x, spawn.y, spawn.z);
        self.character = character;
        self.motor
            .snap_to(self.spawn, Angle::from_q12(spawn.yaw as u16));
        self.room_index = spawn.room;
        self.load_active_room_window();
        self.anim_state = PlayerAnim::Idle;
        self.anim_start_tick = 0;
        self.camera
            .snap_to_player(self.camera_target(None, false), self.camera_config());
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.just_pressed(button::R3) {
            self.lock_target = match self.lock_target {
                Some(_) => None,
                None => self.find_best_lock_target(LOCK_RANGE),
            };
            self.soft_lock_target = None;
        }

        if !ctx.pad.is_analog() {
            return;
        }

        if ctx.just_pressed(button::SELECT) {
            self.free_orbit = !self.free_orbit;
        }
        if self.free_orbit {
            let (right_x, right_y) = ctx.pad.sticks.right_centered();
            self.orbit_yaw =
                self.orbit_yaw
                    .add_signed_q12(stick_to_yaw_delta(psx_engine::InputAxis::new(
                        right_x.saturating_neg(),
                    )));
            self.orbit_radius = (self.orbit_radius
                + stick_to_radius_delta(psx_engine::InputAxis::new(right_y)))
            .clamp(CAMERA_RADIUS_MIN, CAMERA_RADIUS_MAX);
            if ctx.is_held(button::RIGHT) {
                self.orbit_yaw = self.orbit_yaw.add(CAMERA_YAW_STEP);
            }
            if ctx.is_held(button::LEFT) {
                self.orbit_yaw = self.orbit_yaw.sub(CAMERA_YAW_STEP);
            }
            if ctx.is_held(button::UP) {
                self.orbit_radius = (self.orbit_radius - CAMERA_RADIUS_STEP).max(CAMERA_RADIUS_MIN);
            }
            if ctx.is_held(button::DOWN) {
                self.orbit_radius = (self.orbit_radius + CAMERA_RADIUS_STEP).min(CAMERA_RADIUS_MAX);
            }
            return;
        }

        let input = motor_input(ctx, self.camera.yaw());
        let config = self.motor_config();
        let collision = if self.chunked_level() {
            None
        } else {
            self.room.as_ref().map(|room| room.collision())
        };
        let motor_frame = self.motor.update(collision, input, config);
        self.update_current_room_from_player();

        // Animation state comes from the reusable motor, but the
        // playtest intentionally exposes only the core locomotion
        // trio for now: idle, walking, running.
        let new_state = player_anim_from_motor(motor_frame.anim);
        if new_state != self.anim_state {
            self.anim_state = new_state;
            self.anim_start_tick = ctx.time.elapsed_vblanks();
        }

        if self.lock_target.is_some() {
            if !self.lock_target_valid(LOCK_BREAK_RANGE) {
                self.lock_target = None;
            } else if ctx.just_pressed(button::R2) {
                self.switch_lock_target(1);
            } else if ctx.just_pressed(button::L2) {
                self.switch_lock_target(-1);
            }
        }
        if SOFT_LOCK_ENABLED {
            self.update_soft_lock(ctx);
        } else {
            self.soft_lock_target = None;
            self.soft_lock_suppressed = false;
        }
    }

    fn render(&mut self, ctx: &mut Ctx) {
        if !ctx.pad.is_analog() {
            if let Some(font) = self.font.as_ref() {
                draw_analog_required_prompt(font);
            }
            return;
        }

        telemetry::stage_begin(telemetry::stage::CAMERA);
        let camera = if self.free_orbit {
            WorldCamera::orbit_yaw(
                PROJECTION,
                self.spawn.to_world_vertex(),
                CAMERA_Y_OFFSET,
                self.orbit_radius,
                self.orbit_yaw,
            )
        } else {
            self.update_follow_camera(ctx)
        };
        telemetry::stage_end(telemetry::stage::CAMERA);

        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_TRIS) };
        let mut gouraud_triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_GOURAUD_TRIS) };
        // The vertical slice mixes room quads, model instances, and the
        // player in the same playable space. Coarse bucket ordering is
        // faster, but it produces visible doorway/wall artifacts when
        // character triangles land in the same OT slot as room faces.
        let mut world = unsafe { begin_world_render_pass(&mut ot, &mut WORLD_COMMANDS) };

        if self.room.is_some() {
            let room_options = WorldSurfaceOptions::new(ROOM_BAND, WORLD_DEPTH_RANGE);
            let actor_options = WorldSurfaceOptions::new(ACTOR_BAND, WORLD_DEPTH_RANGE);
            let mut total_instance_stats = ModelInstanceDrawStats::default();

            for active in self.active_rooms.iter().flatten().copied() {
                // Pack this chunk's material slice down to a
                // contiguous `&[WorldRenderMaterial]` indexed by
                // local_slot. Slots that didn't resolve become a
                // sentinel material -- visually obvious without
                // crashing the renderer.
                let mut bound: [WorldRenderMaterial; MAX_ROOM_MATERIALS] =
                    [WorldRenderMaterial::both(TextureMaterial::opaque(
                        0,
                        TPAGE_WORD,
                        (0x80, 0x80, 0x80),
                    )); MAX_ROOM_MATERIALS];
                for i in 0..active.material_count {
                    if let Some(m) = active.materials[i] {
                        bound[i] = m;
                    }
                }
                let materials = &bound[..active.material_count];
                let Some(room_record) = ROOMS.get(active.index.to_usize()) else {
                    continue;
                };
                let room_camera = camera_for_room(camera, active);
                let lighting = RuntimeRoomLighting {
                    room_index: active.index,
                    ambient: Rgb8::from_array(active.room.render().ambient_color()),
                    camera: room_camera,
                    fog_enabled: room_record.flags & room_flags::FOG_ENABLED != 0,
                    fog_rgb: Rgb8::from_array(room_record.fog_rgb),
                    fog_near: room_record.fog_near,
                    fog_far: room_record.fog_far,
                };
                telemetry::stage_begin(telemetry::stage::ROOM);
                #[cfg(feature = "world-grid-visible")]
                {
                    let player = self.motor.position();
                    let visibility_anchor = RoomPoint::new(
                        player.x.saturating_sub(active.offset_x),
                        player.y,
                        player.z.saturating_sub(active.offset_z),
                    );
                    let stats = draw_room_vertex_lit_grid_visible(
                        active.room.render(),
                        materials,
                        &lighting,
                        &room_camera,
                        room_options,
                        GridVisibility::around(visibility_anchor, ROOM_GRID_VISIBILITY_RADIUS),
                        &mut gouraud_triangles,
                        &mut world,
                    );
                    telemetry::counter(
                        telemetry::counter::ROOM_CELLS_CONSIDERED,
                        stats.cells_considered as u32,
                    );
                    telemetry::counter(
                        telemetry::counter::ROOM_CELLS_DRAWN,
                        stats.cells_drawn as u32,
                    );
                    telemetry::counter(
                        telemetry::counter::ROOM_CELLS_CULLED,
                        stats.cells_frustum_culled as u32,
                    );
                    telemetry::counter(
                        telemetry::counter::ROOM_SURFACES_CONSIDERED,
                        stats.surfaces_considered as u32,
                    );
                }
                #[cfg(not(feature = "world-grid-visible"))]
                {
                    draw_room_vertex_lit(
                        active.room.render(),
                        materials,
                        &lighting,
                        &room_camera,
                        room_options,
                        &mut gouraud_triangles,
                        &mut world,
                    );
                }
                telemetry::stage_end(telemetry::stage::ROOM);
                telemetry::stage_begin(telemetry::stage::ENTITY_MARKERS);
                draw_entity_markers(
                    ENTITIES,
                    active.index,
                    materials,
                    &room_camera,
                    room_options,
                    &mut triangles,
                    &mut world,
                );
                telemetry::stage_end(telemetry::stage::ENTITY_MARKERS);
                telemetry::stage_begin(telemetry::stage::MODEL_INSTANCES);
                let instance_stats = draw_model_instances(
                    active.index,
                    ctx.time.elapsed_vblanks(),
                    ctx.time.video_hz(),
                    &room_camera,
                    room_options,
                    &lighting,
                    &self.models,
                    &self.clips,
                    &mut triangles,
                    &mut world,
                );
                telemetry::stage_end(telemetry::stage::MODEL_INSTANCES);
                total_instance_stats.draws = total_instance_stats
                    .draws
                    .saturating_add(instance_stats.draws);
                accumulate_model_stats(&mut total_instance_stats.stats, instance_stats.stats);
            }

            telemetry::counter(
                telemetry::counter::MODEL_INSTANCE_DRAWS,
                total_instance_stats.draws as u32,
            );
            emit_model_counters(
                total_instance_stats.stats,
                telemetry::counter::MODEL_INSTANCE_PROJECTED_VERTICES,
                telemetry::counter::MODEL_INSTANCE_SUBMITTED_TRIS,
                telemetry::counter::MODEL_INSTANCE_CULLED_TRIS,
                telemetry::counter::MODEL_INSTANCE_DROPPED_TRIS,
            );
            // Player draws through the same compact model path as
            // placed model instances.
            if let Some(character) = self.character {
                let player = self.motor.position();
                let player_lighting = self.current_room_lighting(camera);
                telemetry::stage_begin(telemetry::stage::PLAYER);
                let player_stats =
                    player_lighting.map_or(TexturedModelRenderStats::default(), |lighting| {
                        draw_player(
                            character,
                            &self.models,
                            &self.clips,
                            player.x,
                            player.y,
                            player.z,
                            self.motor.yaw(),
                            character.clip_for(self.anim_state),
                            self.anim_start_tick,
                            ctx.time.elapsed_vblanks(),
                            ctx.time.video_hz(),
                            &camera,
                            actor_options,
                            &lighting,
                            &mut triangles,
                            &mut world,
                        )
                    });
                telemetry::stage_end(telemetry::stage::PLAYER);
                emit_model_counters(
                    player_stats,
                    telemetry::counter::PLAYER_PROJECTED_VERTICES,
                    telemetry::counter::PLAYER_SUBMITTED_TRIS,
                    telemetry::counter::PLAYER_CULLED_TRIS,
                    telemetry::counter::PLAYER_DROPPED_TRIS,
                );
                telemetry::stage_begin(telemetry::stage::EQUIPMENT);
                let equipment_stats =
                    player_lighting.map_or(EquipmentDrawStats::default(), |lighting| {
                        draw_player_equipment(
                            self.room_index,
                            character,
                            &self.models,
                            &self.clips,
                            player.x,
                            player.y,
                            player.z,
                            self.motor.yaw(),
                            character.clip_for(self.anim_state),
                            self.anim_start_tick,
                            ctx.time.elapsed_vblanks(),
                            ctx.time.video_hz(),
                            &camera,
                            actor_options,
                            &lighting,
                            &mut triangles,
                            &mut world,
                        )
                    });
                telemetry::stage_end(telemetry::stage::EQUIPMENT);
                telemetry::counter(
                    telemetry::counter::EQUIPMENT_DRAWS,
                    equipment_stats.draws as u32,
                );
                telemetry::counter(
                    telemetry::counter::EQUIPMENT_ACTIVE_HITBOXES,
                    equipment_stats.active_hitboxes as u32,
                );
                telemetry::counter(
                    telemetry::counter::EQUIPMENT_TARGET_HITS,
                    equipment_stats.target_hits as u32,
                );
                emit_model_counters(
                    equipment_stats.stats,
                    telemetry::counter::EQUIPMENT_PROJECTED_VERTICES,
                    telemetry::counter::EQUIPMENT_SUBMITTED_TRIS,
                    telemetry::counter::EQUIPMENT_CULLED_TRIS,
                    telemetry::counter::EQUIPMENT_DROPPED_TRIS,
                );
            }
        }

        telemetry::counter(
            telemetry::counter::TRI_PRIMITIVES,
            triangles.len().saturating_add(gouraud_triangles.len()) as u32,
        );
        telemetry::counter(
            telemetry::counter::WORLD_COMMANDS,
            world.command_len() as u32,
        );
        telemetry::stage_begin(telemetry::stage::WORLD_FLUSH);
        world.flush();
        telemetry::stage_end(telemetry::stage::WORLD_FLUSH);
        telemetry::stage_begin(telemetry::stage::OT_SUBMIT);
        ot.submit();
        telemetry::stage_end(telemetry::stage::OT_SUBMIT);
    }
}

#[cfg(all(
    feature = "world-order-global",
    any(
        feature = "world-order-slot",
        feature = "world-order-linked",
        feature = "world-order-bucketed"
    )
))]
compile_error!("choose only one world-order-* feature");
#[cfg(all(
    feature = "world-order-slot",
    any(feature = "world-order-linked", feature = "world-order-bucketed")
))]
compile_error!("choose only one world-order-* feature");
#[cfg(all(feature = "world-order-linked", feature = "world-order-bucketed"))]
compile_error!("choose only one world-order-* feature");

fn begin_world_render_pass<'a, 'ot>(
    ot: &'a mut OtFrame<'ot, OT_DEPTH>,
    commands: &'a mut [WorldTriCommand],
) -> WorldRenderPass<'a, 'ot, OT_DEPTH> {
    #[cfg(feature = "world-order-slot")]
    {
        return WorldRenderPass::new_deferred_slot_sorted(ot, commands);
    }
    #[cfg(feature = "world-order-linked")]
    {
        return WorldRenderPass::new(ot, commands);
    }
    #[cfg(feature = "world-order-bucketed")]
    {
        return WorldRenderPass::new_bucketed(ot, commands);
    }
    #[cfg(not(any(
        feature = "world-order-slot",
        feature = "world-order-linked",
        feature = "world-order-bucketed"
    )))]
    {
        WorldRenderPass::new_deferred_sorted(ot, commands)
    }
}

impl Playtest {
    fn load_runtime_models(&mut self) {
        let mut i = 0;
        while i < MAX_RUNTIME_MODELS {
            self.models[i] = None;
            i += 1;
        }
        i = 0;
        while i < MAX_RUNTIME_MODEL_CLIPS {
            self.clips[i] = None;
            i += 1;
        }

        for (index, clip) in MODEL_CLIPS.iter().enumerate() {
            if index >= MAX_RUNTIME_MODEL_CLIPS {
                break;
            }
            let Some(asset) =
                find_asset_of_kind(ASSETS, clip.animation_asset, AssetKind::ModelAnimation)
            else {
                continue;
            };
            self.clips[index] = Animation::from_bytes(asset.bytes).ok();
        }

        for (index, record) in MODELS.iter().enumerate() {
            if index >= MAX_RUNTIME_MODELS {
                break;
            }
            self.models[index] = RuntimeModelAsset::from_record(record);
        }
    }

    fn motor_config(&self) -> CharacterMotorConfig {
        match self.character {
            Some(c) => c.motor_config(),
            None => CharacterMotorConfig::character(
                0,
                FALLBACK_PLAYER_SPEED,
                FALLBACK_PLAYER_SPEED,
                FALLBACK_PLAYER_YAW_STEP,
            ),
        }
    }

    fn camera_config(&self) -> ThirdPersonCameraConfig {
        let mut config = match self.character {
            Some(c) => ThirdPersonCameraConfig::character(
                c.camera_distance,
                c.camera_height,
                c.camera_target_height,
            ),
            None => ThirdPersonCameraConfig::character(
                FOLLOW_RADIUS_DEFAULT,
                FOLLOW_HEIGHT_DEFAULT,
                FOLLOW_TARGET_HEIGHT_DEFAULT,
            ),
        };
        config.height = config.height.max(256);
        config
    }

    fn camera_target(
        &self,
        lock_target: Option<RoomPoint>,
        moving: bool,
    ) -> ThirdPersonCameraTarget {
        ThirdPersonCameraTarget {
            player: self.motor.position(),
            player_yaw: self.motor.yaw(),
            moving,
            lock_target,
        }
    }

    fn current_room_lighting(&self, camera: WorldCamera) -> Option<RuntimeRoomLighting> {
        let room = self.room?;
        let room_record = ROOMS.get(self.room_index.to_usize())?;
        Some(RuntimeRoomLighting {
            room_index: self.room_index,
            ambient: Rgb8::from_array(room.render().ambient_color()),
            camera,
            fog_enabled: room_record.flags & room_flags::FOG_ENABLED != 0,
            fog_rgb: Rgb8::from_array(room_record.fog_rgb),
            fog_near: room_record.fog_near,
            fog_far: room_record.fog_far,
        })
    }

    fn update_follow_camera(&mut self, ctx: &Ctx) -> WorldCamera {
        let input = camera_input(ctx);
        let lock_target = self
            .lock_target_position()
            .or_else(|| self.soft_lock_target_position());
        let target = self.camera_target(lock_target, self.anim_state != PlayerAnim::Idle);
        let config = self.camera_config();
        let collision = if CAMERA_COLLISION_ENABLED && !self.chunked_level() {
            self.room.as_ref().map(|room| room.collision())
        } else {
            None
        };
        self.camera
            .update(PROJECTION, collision, target, input, config)
            .camera
    }

    fn chunked_level(&self) -> bool {
        self.active_rooms
            .iter()
            .flatten()
            .any(|room| room.index != self.room_index)
    }

    fn load_active_room_window(&mut self) {
        self.room = None;
        self.materials = [const { None }; MAX_ROOM_MATERIALS];
        self.material_count = 0;
        self.active_rooms = [const { None }; MAX_ACTIVE_ROOMS];

        let current_index = self.room_index;
        let Some(current_record) = ROOMS.get(current_index.to_usize()) else {
            return;
        };
        let Some(current_room) = parse_runtime_room(current_record) else {
            return;
        };

        let mut next_slot = 0usize;
        if let Some(active) = build_active_room(current_index, current_record, current_record) {
            self.room = Some(active.room);
            self.materials = active.materials;
            self.material_count = active.material_count;
            self.active_rooms[next_slot] = Some(active);
            next_slot += 1;
        }

        for (raw_index, record) in ROOMS.iter().enumerate() {
            if raw_index == current_index.to_usize() || next_slot >= MAX_ACTIVE_ROOMS {
                continue;
            }
            let Some(room) = parse_runtime_room(record) else {
                continue;
            };
            if !rooms_touch(current_record, current_room, record, room) {
                continue;
            }
            let index = RoomIndex::new(raw_index as u16);
            if let Some(active) = build_active_room(index, record, current_record) {
                self.active_rooms[next_slot] = Some(active);
                next_slot += 1;
            }
        }
    }

    fn update_current_room_from_player(&mut self) {
        if !self.chunked_level() {
            return;
        }
        let global = local_to_global_room_point(self.room_index, self.motor.position());
        let Some(next_room) = room_index_containing_global(global) else {
            return;
        };
        if next_room == self.room_index {
            return;
        }
        let local = global_to_local_room_point(next_room, global);
        self.room_index = next_room;
        self.motor.relocate(local);
        self.lock_target = None;
        self.soft_lock_target = None;
        self.load_active_room_window();
    }

    fn lock_target_position(&self) -> Option<RoomPoint> {
        self.target_position(self.lock_target?)
    }

    fn soft_lock_target_position(&self) -> Option<RoomPoint> {
        self.target_position(self.soft_lock_target?)
    }

    fn target_position(&self, index: usize) -> Option<RoomPoint> {
        let target = ENTITIES.get(index)?;
        if target.room != self.room_index {
            return None;
        }
        Some(RoomPoint::new(target.x, target.y, target.z))
    }

    fn lock_target_valid(&self, range: i32) -> bool {
        self.lock_target
            .is_some_and(|index| self.target_index_valid(index, range))
    }

    fn target_index_valid(&self, index: usize, range: i32) -> bool {
        let Some(target) = self.target_position(index) else {
            return false;
        };
        distance_xz_sq(self.motor.position(), target) <= square_i32_saturating(range)
    }

    fn find_best_lock_target(&self, range: i32) -> Option<usize> {
        let player = self.motor.position();
        let view_yaw = self.camera.yaw().add(Angle::HALF);
        let sin_yaw = view_yaw.sin();
        let cos_yaw = view_yaw.cos();
        let range_sq = square_i32_saturating(range);
        let mut best: Option<(usize, i32)> = None;
        for (index, entity) in ENTITIES.iter().enumerate() {
            if entity.room != self.room_index {
                continue;
            }
            let target = RoomPoint::new(entity.x, entity.y, entity.z);
            let dx = target.x.saturating_sub(player.x);
            let dz = target.z.saturating_sub(player.z);
            let dist_sq = square_i32_saturating(dx).saturating_add(square_i32_saturating(dz));
            if dist_sq == 0 || dist_sq > range_sq {
                continue;
            }
            let dot = dx
                .saturating_mul(sin_yaw.raw())
                .saturating_add(dz.saturating_mul(cos_yaw.raw()));
            if dot <= 0 {
                continue;
            }
            let score = (dot >> 4).saturating_sub(dist_sq >> 12);
            match best {
                Some((_, best_score)) if best_score >= score => {}
                _ => best = Some((index, score)),
            }
        }
        best.map(|(index, _)| index)
    }

    fn update_soft_lock(&mut self, ctx: &Ctx) {
        if self.lock_target.is_some() {
            self.soft_lock_target = None;
            self.soft_lock_suppressed = false;
            return;
        }
        let (right_x, _) = ctx.pad.sticks.right_centered();
        if abs_i16(right_x) >= CAMERA_SOFT_LOCK_BREAK_STICK {
            self.soft_lock_target = None;
            self.soft_lock_suppressed = true;
            return;
        }
        if self.soft_lock_suppressed {
            if self.find_best_lock_target(SOFT_LOCK_BREAK_RANGE).is_none() {
                self.soft_lock_suppressed = false;
            }
            return;
        }
        match self.soft_lock_target {
            Some(index) if self.target_index_valid(index, SOFT_LOCK_BREAK_RANGE) => {}
            _ => self.soft_lock_target = self.find_best_lock_target(SOFT_LOCK_RANGE),
        }
    }

    fn switch_lock_target(&mut self, direction: i32) {
        let Some(current_index) = self.lock_target else {
            return;
        };
        let Some(current) = ENTITIES.get(current_index) else {
            self.lock_target = None;
            return;
        };
        let player = self.motor.position();
        let current_dx = current.x.saturating_sub(player.x);
        let current_dz = current.z.saturating_sub(player.z);
        if current_dx == 0 && current_dz == 0 {
            return;
        }
        let range_sq = square_i32_saturating(LOCK_RANGE);
        let mut best: Option<(usize, i32)> = None;
        for (index, entity) in ENTITIES.iter().enumerate() {
            if index == current_index || entity.room != self.room_index {
                continue;
            }
            let dx = entity.x.saturating_sub(player.x);
            let dz = entity.z.saturating_sub(player.z);
            let dist_sq = square_i32_saturating(dx).saturating_add(square_i32_saturating(dz));
            if dist_sq == 0 || dist_sq > range_sq {
                continue;
            }
            let cross = current_dx
                .saturating_mul(dz)
                .saturating_sub(current_dz.saturating_mul(dx));
            if direction > 0 {
                if cross >= 0 {
                    continue;
                }
            } else if cross <= 0 {
                continue;
            }
            let dot = current_dx
                .saturating_mul(dx)
                .saturating_add(current_dz.saturating_mul(dz));
            let score = ratio_q8_i32(dot.max(0), dist_sq.max(1)).saturating_sub(dist_sq >> 14);
            match best {
                Some((_, best_score)) if best_score >= score => {}
                _ => best = Some((index, score)),
            }
        }
        if let Some((index, _)) = best {
            self.lock_target = Some(index);
        }
    }
}

fn ratio_q8_i32(numerator: i32, denominator: i32) -> i32 {
    if numerator <= 0 || denominator <= 0 {
        return 0;
    }
    let numerator = numerator as u32;
    let denominator = denominator as u32;
    let whole = numerator / denominator;
    let remainder = numerator % denominator;
    let scaled_whole = if whole > (i32::MAX as u32 / 256) {
        return i32::MAX;
    } else {
        whole * 256
    };
    let scaled_remainder = remainder.saturating_mul(256) / denominator;
    scaled_whole
        .saturating_add(scaled_remainder)
        .min(i32::MAX as u32) as i32
}

fn draw_player(
    character: RuntimeCharacter,
    models: &[Option<RuntimeModelAsset>; MAX_RUNTIME_MODELS],
    clips: &[Option<Animation<'static>>; MAX_RUNTIME_MODEL_CLIPS],
    x: i32,
    y: i32,
    z: i32,
    yaw: Angle,
    clip_local: ModelClipIndex,
    anim_start_tick: u32,
    elapsed_vblanks: u32,
    video_hz: u16,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    lighting: &RuntimeRoomLighting,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
) -> TexturedModelRenderStats {
    let Some(runtime_model) = models.get(character.model.to_usize()).copied().flatten() else {
        return TexturedModelRenderStats::default();
    };

    let Some(anim) = runtime_model.clip(clips, clip_local) else {
        return TexturedModelRenderStats::default();
    };
    // Phase the animation relative to the clip-start tick so
    // state changes don't pop into the middle of a new clip.
    let local_tick = elapsed_vblanks.saturating_sub(anim_start_tick);
    let phase = anim.phase_at_tick_q12(local_tick, video_hz);

    let origin = floor_anchored_model_origin(x, y, z, runtime_model.world_height);
    let instance_rotation = yaw_rotation_matrix(yaw);
    let material = lighting.shade_model_material(origin, runtime_model.material);
    let model_options = options
        .with_depth_policy(DepthPolicy::Average)
        .with_cull_mode(CullMode::Back)
        .with_material_layer(material)
        .with_textured_triangle_splitting(false);

    world.submit_textured_model(
        triangles,
        runtime_model.model,
        anim,
        phase,
        *camera,
        origin,
        instance_rotation,
        unsafe { &mut MODEL_VERTICES },
        unsafe { &mut JOINT_VIEW_TRANSFORMS },
        material,
        model_options,
    )
}

#[derive(Copy, Clone, Debug, Default)]
struct EquipmentDrawStats {
    draws: u16,
    active_hitboxes: u16,
    target_hits: u16,
    stats: TexturedModelRenderStats,
}

#[derive(Copy, Clone)]
struct AttachmentPose {
    origin: WorldVertex,
    rotation: Mat3I16,
}

#[allow(clippy::too_many_arguments)]
fn draw_player_equipment(
    current_room: RoomIndex,
    character: RuntimeCharacter,
    models: &[Option<RuntimeModelAsset>; MAX_RUNTIME_MODELS],
    clips: &[Option<Animation<'static>>; MAX_RUNTIME_MODEL_CLIPS],
    x: i32,
    y: i32,
    z: i32,
    yaw: Angle,
    clip_local: ModelClipIndex,
    anim_start_tick: u32,
    elapsed_vblanks: u32,
    video_hz: u16,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    lighting: &RuntimeRoomLighting,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
) -> EquipmentDrawStats {
    let mut out = EquipmentDrawStats::default();
    let Some(character_model) = models.get(character.model.to_usize()).copied().flatten() else {
        return out;
    };
    let Some(character_anim) = character_model.clip(clips, clip_local) else {
        return out;
    };
    let local_tick = elapsed_vblanks.saturating_sub(anim_start_tick);
    let character_phase = character_anim.phase_at_tick_q12(local_tick, video_hz);
    let character_frame = (character_phase >> 12) as u16;
    let character_origin = floor_anchored_model_origin(x, y, z, character_model.world_height);
    let character_rotation = yaw_rotation_matrix(yaw);

    let mut drawn = 0usize;
    for equipment in EQUIPMENT {
        if equipment.room != current_room
            || equipment.flags & equipment_flags::PLAYER == 0
            || drawn >= MAX_EQUIPMENT_DRAWS
        {
            continue;
        }
        let Some(weapon) = WEAPONS.get(equipment.weapon.to_usize()) else {
            continue;
        };
        let Some(socket) = find_model_socket(character_model, equipment.character_socket)
            .or_else(|| find_model_socket(character_model, weapon.default_character_socket))
        else {
            continue;
        };
        let Some(socket_pose) = attachment_socket_pose(
            character_model,
            character_anim,
            character_phase,
            character_origin,
            character_rotation,
            socket,
        ) else {
            continue;
        };
        let weapon_rotation = socket_pose
            .rotation
            .mul(&euler_q12_rotation_inverse(weapon.grip_rotation_q12));

        match weapon.model {
            Some(model_index) => {
                let Some(weapon_model) = models.get(model_index.to_usize()).copied().flatten()
                else {
                    continue;
                };
                let grip = scaled_offset(weapon_model.local_to_world, weapon.grip_translation);
                let grip_world = rotate_offset_q12(&weapon_rotation, grip);
                let origin = WorldVertex::new(
                    socket_pose.origin.x.saturating_sub(grip_world[0]),
                    socket_pose.origin.y.saturating_sub(grip_world[1]),
                    socket_pose.origin.z.saturating_sub(grip_world[2]),
                );
                if let Some(anim) = weapon_model.clip(clips, weapon_model.default_clip) {
                    let phase = anim.phase_at_tick_q12(elapsed_vblanks, video_hz);
                    let material = lighting.shade_model_material(origin, weapon_model.material);
                    let model_options = options
                        .with_depth_policy(DepthPolicy::Average)
                        .with_cull_mode(CullMode::Back)
                        .with_material_layer(material)
                        .with_textured_triangle_splitting(false);
                    let stats = world.submit_textured_model_primary_joints(
                        triangles,
                        weapon_model.model,
                        anim,
                        phase,
                        *camera,
                        origin,
                        weapon_rotation,
                        unsafe { &mut MODEL_VERTICES },
                        unsafe { &mut JOINT_VIEW_TRANSFORMS },
                        material,
                        model_options,
                    );
                    accumulate_model_stats(&mut out.stats, stats);
                    if stats.primitive_overflow || stats.command_overflow {
                        out.draws = drawn as u16;
                        return out;
                    }
                    drawn += 1;
                    out.draws = drawn as u16;
                }
            }
            None => {}
        };

        let (active, hits) = evaluate_weapon_hitboxes(
            current_room,
            weapon.hitbox_first.to_usize(),
            weapon.hitbox_count,
            character_frame,
            socket_pose.origin,
            socket_pose.rotation,
        );
        out.active_hitboxes = out.active_hitboxes.saturating_add(active);
        out.target_hits = out.target_hits.saturating_add(hits);
    }
    out
}

fn find_model_socket(
    model: RuntimeModelAsset,
    name: &str,
) -> Option<&'static LevelModelSocketRecord> {
    let first = model.socket_first.to_usize();
    let count = model.socket_count as usize;
    let sockets = MODEL_SOCKETS.get(first..first.saturating_add(count))?;
    sockets.iter().find(|socket| socket.name == name)
}

fn attachment_socket_pose(
    model: RuntimeModelAsset,
    animation: Animation<'static>,
    phase_q12: u32,
    origin: WorldVertex,
    instance_rotation: Mat3I16,
    socket: &LevelModelSocketRecord,
) -> Option<AttachmentPose> {
    let pose = animation.pose_looped_q12(phase_q12, socket.joint)?;
    let joint =
        compute_joint_world_transform(pose, instance_rotation, model.local_to_world, origin);
    Some(compose_socket_pose(
        joint,
        socket.translation,
        socket.rotation_q12,
    ))
}

fn compose_socket_pose(
    joint: JointWorldTransform,
    translation: [i32; 3],
    rotation_q12: [i16; 3],
) -> AttachmentPose {
    let offset = rotate_offset_q12(&joint.rotation, translation);
    let local_rotation = euler_q12_rotation(rotation_q12);
    AttachmentPose {
        origin: WorldVertex::new(
            joint.translation.x.saturating_add(offset[0]),
            joint.translation.y.saturating_add(offset[1]),
            joint.translation.z.saturating_add(offset[2]),
        ),
        rotation: joint.rotation.mul(&local_rotation),
    }
}

fn evaluate_weapon_hitboxes(
    current_room: RoomIndex,
    first: usize,
    count: u16,
    frame: u16,
    origin: WorldVertex,
    rotation: Mat3I16,
) -> (u16, u16) {
    let mut active = 0u16;
    let mut hits = 0u16;
    let Some(hitboxes) = WEAPON_HITBOXES.get(first..first.saturating_add(count as usize)) else {
        return (0, 0);
    };
    for hitbox in hitboxes {
        if frame < hitbox.active_start_frame || frame > hitbox.active_end_frame {
            continue;
        }
        active = active.saturating_add(1);
        for entity in ENTITIES {
            if entity.room != current_room {
                continue;
            }
            if weapon_hit_shape_hits_point(hitbox.shape, origin, rotation, entity.x, entity.z) {
                hits = hits.saturating_add(1);
            }
        }
    }
    (active, hits)
}

fn weapon_hit_shape_hits_point(
    shape: WeaponHitShapeRecord,
    origin: WorldVertex,
    rotation: Mat3I16,
    px: i32,
    pz: i32,
) -> bool {
    match shape {
        WeaponHitShapeRecord::Box {
            center,
            half_extents,
        } => {
            let c = transform_local_point(origin, rotation, center);
            let radius = half_extents[0].max(half_extents[2]) as i32;
            distance_xz_sq(RoomPoint::new(px, 0, pz), RoomPoint::new(c.x, 0, c.z))
                <= square_i32_saturating(radius)
        }
        WeaponHitShapeRecord::Capsule { start, end, radius } => {
            let a = transform_local_point(origin, rotation, start);
            let b = transform_local_point(origin, rotation, end);
            point_segment_xz_distance_sq(px, pz, a.x, a.z, b.x, b.z)
                <= square_i32_saturating(radius as i32)
        }
    }
}

fn transform_local_point(origin: WorldVertex, rotation: Mat3I16, point: [i32; 3]) -> WorldVertex {
    let offset = rotate_offset_q12(&rotation, point);
    WorldVertex::new(
        origin.x.saturating_add(offset[0]),
        origin.y.saturating_add(offset[1]),
        origin.z.saturating_add(offset[2]),
    )
}

fn scaled_offset(scale: LocalToWorldScale, offset: [i32; 3]) -> [i32; 3] {
    [
        scale.apply(offset[0]),
        scale.apply(offset[1]),
        scale.apply(offset[2]),
    ]
}

fn rotate_offset_q12(rotation: &Mat3I16, offset: [i32; 3]) -> [i32; 3] {
    let row = |r: [i16; 3]| -> i32 {
        let x = (r[0] as i32).saturating_mul(offset[0]);
        let y = (r[1] as i32).saturating_mul(offset[1]);
        let z = (r[2] as i32).saturating_mul(offset[2]);
        x.saturating_add(y).saturating_add(z) >> 12
    };
    [row(rotation.m[0]), row(rotation.m[1]), row(rotation.m[2])]
}

fn euler_q12_rotation(rotation_q12: [i16; 3]) -> Mat3I16 {
    let rx = Mat3I16::rotate_x(Angle::from_q12(rotation_q12[0] as u16).rotate_y_arg());
    let ry = Mat3I16::rotate_y(Angle::from_q12(rotation_q12[1] as u16).rotate_y_arg());
    let rz = Mat3I16::rotate_z(Angle::from_q12(rotation_q12[2] as u16).rotate_y_arg());
    rz.mul(&ry).mul(&rx)
}

fn euler_q12_rotation_inverse(rotation_q12: [i16; 3]) -> Mat3I16 {
    let inv_x = (-(rotation_q12[0] as i32)) as u16;
    let inv_y = (-(rotation_q12[1] as i32)) as u16;
    let inv_z = (-(rotation_q12[2] as i32)) as u16;
    let rx = Mat3I16::rotate_x(Angle::from_q12(inv_x).rotate_y_arg());
    let ry = Mat3I16::rotate_y(Angle::from_q12(inv_y).rotate_y_arg());
    let rz = Mat3I16::rotate_z(Angle::from_q12(inv_z).rotate_y_arg());
    rx.mul(&ry).mul(&rz)
}

fn point_segment_xz_distance_sq(px: i32, pz: i32, ax: i32, az: i32, bx: i32, bz: i32) -> i32 {
    let abx = bx.saturating_sub(ax);
    let abz = bz.saturating_sub(az);
    let apx = px.saturating_sub(ax);
    let apz = pz.saturating_sub(az);
    let denom = square_i32_saturating(abx).saturating_add(square_i32_saturating(abz));
    if denom <= 0 {
        return square_i32_saturating(apx).saturating_add(square_i32_saturating(apz));
    }
    let dot = apx
        .saturating_mul(abx)
        .saturating_add(apz.saturating_mul(abz));
    let t_q8 = ratio_q8_i32(dot.clamp(0, denom), denom);
    let cx = ax.saturating_add((abx.saturating_mul(t_q8)) >> 8);
    let cz = az.saturating_add((abz.saturating_mul(t_q8)) >> 8);
    square_i32_saturating(px.saturating_sub(cx))
        .saturating_add(square_i32_saturating(pz.saturating_sub(cz)))
}

fn emit_model_counters(
    stats: TexturedModelRenderStats,
    projected_counter: u16,
    submitted_counter: u16,
    culled_counter: u16,
    dropped_counter: u16,
) {
    telemetry::counter(projected_counter, stats.projected_vertices as u32);
    telemetry::counter(submitted_counter, stats.submitted_triangles as u32);
    telemetry::counter(culled_counter, stats.culled_triangles as u32);
    telemetry::counter(dropped_counter, stats.dropped_triangles as u32);

    let mut overflow = 0u32;
    if stats.vertex_overflow {
        overflow |= 1;
    }
    if stats.primitive_overflow {
        overflow |= 1 << 1;
    }
    if stats.command_overflow {
        overflow |= 1 << 2;
    }
    if overflow != 0 {
        telemetry::counter(telemetry::counter::MODEL_OVERFLOW_FLAGS, overflow);
    }
}

fn parse_runtime_room(record: &LevelRoomRecord) -> Option<RuntimeRoom<'static>> {
    let asset = find_asset_of_kind(ASSETS, record.world_asset, AssetKind::RoomWorld)?;
    RuntimeRoom::from_bytes(asset.bytes).ok()
}

fn build_active_room(
    index: RoomIndex,
    record: &LevelRoomRecord,
    current_record: &LevelRoomRecord,
) -> Option<ActiveRuntimeRoom> {
    if let Some(residency) = ROOM_RESIDENCY.iter().find(|r| r.room == index) {
        let _ = unsafe { RESIDENCY.ensure_room_resident(residency) };
    }
    let room = parse_runtime_room(record)?;
    let mut materials = [const { None }; MAX_ROOM_MATERIALS];
    let material_count = build_room_materials(record, &mut materials);
    Some(ActiveRuntimeRoom {
        index,
        room,
        materials,
        material_count,
        offset_x: room_origin_x(record).saturating_sub(room_origin_x(current_record)),
        offset_z: room_origin_z(record).saturating_sub(room_origin_z(current_record)),
    })
}

fn room_origin_x(record: &LevelRoomRecord) -> i32 {
    record.origin_x.saturating_mul(record.sector_size)
}

fn room_origin_z(record: &LevelRoomRecord) -> i32 {
    record.origin_z.saturating_mul(record.sector_size)
}

fn room_bounds(record: &LevelRoomRecord, room: RuntimeRoom<'_>) -> (i32, i32, i32, i32) {
    let x0 = room_origin_x(record);
    let z0 = room_origin_z(record);
    let x1 = x0.saturating_add((room.width() as i32).saturating_mul(record.sector_size));
    let z1 = z0.saturating_add((room.depth() as i32).saturating_mul(record.sector_size));
    (x0, x1, z0, z1)
}

fn rooms_touch(
    a_record: &LevelRoomRecord,
    a_room: RuntimeRoom<'_>,
    b_record: &LevelRoomRecord,
    b_room: RuntimeRoom<'_>,
) -> bool {
    let (ax0, ax1, az0, az1) = room_bounds(a_record, a_room);
    let (bx0, bx1, bz0, bz1) = room_bounds(b_record, b_room);
    bx0 <= ax1 && bx1 >= ax0 && bz0 <= az1 && bz1 >= az0
}

fn room_index_containing_global(point: RoomPoint) -> Option<RoomIndex> {
    for (raw_index, record) in ROOMS.iter().enumerate() {
        let Some(room) = parse_runtime_room(record) else {
            continue;
        };
        let (x0, x1, z0, z1) = room_bounds(record, room);
        if point.x >= x0 && point.x < x1 && point.z >= z0 && point.z < z1 {
            return Some(RoomIndex::new(raw_index as u16));
        }
    }
    None
}

fn local_to_global_room_point(room: RoomIndex, point: RoomPoint) -> RoomPoint {
    let Some(record) = ROOMS.get(room.to_usize()) else {
        return point;
    };
    RoomPoint::new(
        point.x.saturating_add(room_origin_x(record)),
        point.y,
        point.z.saturating_add(room_origin_z(record)),
    )
}

fn global_to_local_room_point(room: RoomIndex, point: RoomPoint) -> RoomPoint {
    let Some(record) = ROOMS.get(room.to_usize()) else {
        return point;
    };
    RoomPoint::new(
        point.x.saturating_sub(room_origin_x(record)),
        point.y,
        point.z.saturating_sub(room_origin_z(record)),
    )
}

fn camera_for_room(camera: WorldCamera, active: ActiveRuntimeRoom) -> WorldCamera {
    WorldCamera::from_basis(
        camera.projection,
        WorldVertex::new(
            camera.position.x.saturating_sub(active.offset_x),
            camera.position.y,
            camera.position.z.saturating_sub(active.offset_z),
        ),
        camera.sin_yaw,
        camera.cos_yaw,
        camera.sin_pitch,
        camera.cos_pitch,
    )
}

/// Walk `room.material_first..material_first + material_count`,
/// resolve each material's texture asset, and build a
/// TextureMaterial in `out` indexed by `local_slot`. Each
/// texture asset is uploaded at most once across the program
/// lifetime -- the residency manager + VRAM_SLOTS tracks who's
/// already up.
///
/// Returns the highest `local_slot + 1` so the caller knows the
/// in-use prefix length.
fn build_room_materials(
    room: &LevelRoomRecord,
    out: &mut [Option<WorldRenderMaterial>; MAX_ROOM_MATERIALS],
) -> usize {
    let first = room.material_first.to_usize();
    let count = room.material_count as usize;
    let slice: &[LevelMaterialRecord] = &MATERIALS[first..first + count];

    let mut max_slot: usize = 0;
    for material in slice {
        let slot = material.local_slot.to_usize();
        if slot >= MAX_ROOM_MATERIALS {
            continue;
        }
        let Some(asset) = find_asset_of_kind(ASSETS, material.texture_asset, AssetKind::Texture)
        else {
            continue;
        };
        let Some(slot_record) = ensure_texture_uploaded(asset.id, asset.bytes) else {
            continue;
        };
        let texture = TextureMaterial::opaque(
            slot_record.clut_word,
            slot_record.tpage_word,
            rgb_tuple(material.tint_rgb),
        )
        .with_texture_window(slot_record.texture_window);
        out[slot] = Some(match material.sidedness() {
            LevelMaterialSidedness::Front => WorldRenderMaterial::front(texture),
            LevelMaterialSidedness::Back => WorldRenderMaterial::back(texture),
            LevelMaterialSidedness::Both => WorldRenderMaterial::both(texture),
        });
        if slot + 1 > max_slot {
            max_slot = slot + 1;
        }
    }
    max_slot
}

#[derive(Copy, Clone)]
struct RuntimeRoomLighting {
    room_index: RoomIndex,
    ambient: Rgb8,
    camera: WorldCamera,
    fog_enabled: bool,
    fog_rgb: Rgb8,
    fog_near: i32,
    fog_far: i32,
}

impl RuntimeRoomLighting {
    fn shade_model_material(
        &self,
        point: WorldVertex,
        material: TextureMaterial,
    ) -> TextureMaterial {
        material.with_tint(self.shade_tint_at(RoomPoint::from_world_vertex(point), material.tint()))
    }

    fn shade_tint_at(&self, point: RoomPoint, base: (u8, u8, u8)) -> (u8, u8, u8) {
        let lights = LIGHTS
            .iter()
            .filter(|light| light.room == self.room_index)
            .map(|light| {
                PointLightSample::from_rgb_intensity(
                    [light.x, light.y, light.z],
                    light.radius as i32,
                    Rgb8::from_array(light.color),
                    Q8::from_raw_u16(light.intensity_q8),
                )
            });
        let tint = psx_engine::shade_material_tint_with_lights(
            MaterialTint::from_tuple(base),
            point.to_array(),
            self.ambient,
            lights,
        )
        .to_tuple();
        let depth = self.camera.view_vertex(point.to_world_vertex()).z;
        apply_room_fog(
            tint,
            depth,
            self.fog_enabled,
            self.fog_rgb,
            self.fog_near,
            self.fog_far,
        )
    }

    fn apply_vertex_fog(&self, rgb: [u8; 3], vertex: WorldVertex) -> (u8, u8, u8) {
        let depth = self.camera.view_vertex(vertex).z;
        apply_room_fog(
            (rgb[0], rgb[1], rgb[2]),
            depth,
            self.fog_enabled,
            self.fog_rgb,
            self.fog_near,
            self.fog_far,
        )
    }
}

impl WorldSurfaceLighting for RuntimeRoomLighting {
    fn shade(
        &self,
        sample: WorldSurfaceSample,
        material: WorldRenderMaterial,
    ) -> WorldRenderMaterial {
        material.with_tint(self.shade_tint_at(sample.center, material.texture.tint()))
    }

    fn shade_vertex(
        &self,
        _sample: WorldSurfaceSample,
        vertex: RoomPoint,
        material: WorldRenderMaterial,
    ) -> (u8, u8, u8) {
        self.shade_tint_at(vertex, material.texture.tint())
    }

    fn shade_vertices(
        &self,
        sample: WorldSurfaceSample,
        vertices: [WorldVertex; 4],
        material: WorldRenderMaterial,
    ) -> [(u8, u8, u8); 4] {
        if let Some(vertex_rgb) = sample.baked_vertex_rgb {
            return [
                self.apply_vertex_fog(vertex_rgb[0], vertices[0]),
                self.apply_vertex_fog(vertex_rgb[1], vertices[1]),
                self.apply_vertex_fog(vertex_rgb[2], vertices[2]),
                self.apply_vertex_fog(vertex_rgb[3], vertices[3]),
            ];
        }
        [
            self.shade_vertex(sample, RoomPoint::from_world_vertex(vertices[0]), material),
            self.shade_vertex(sample, RoomPoint::from_world_vertex(vertices[1]), material),
            self.shade_vertex(sample, RoomPoint::from_world_vertex(vertices[2]), material),
            self.shade_vertex(sample, RoomPoint::from_world_vertex(vertices[3]), material),
        ]
    }
}

fn apply_room_fog(
    tint: (u8, u8, u8),
    depth: i32,
    enabled: bool,
    fog_rgb: Rgb8,
    fog_near: i32,
    fog_far: i32,
) -> (u8, u8, u8) {
    if !enabled || fog_far <= fog_near || depth <= fog_near {
        return tint;
    }
    let weight = (((depth - fog_near).saturating_mul(256)) / (fog_far - fog_near)).clamp(0, 256);
    let keep = 256 - weight;
    (
        blend_channel(tint.0, fog_rgb.r, keep, weight),
        blend_channel(tint.1, fog_rgb.g, keep, weight),
        blend_channel(tint.2, fog_rgb.b, keep, weight),
    )
}

fn blend_channel(src: u8, fog: u8, keep: i32, weight: i32) -> u8 {
    (((src as i32) * keep + (fog as i32) * weight) >> 8).clamp(0, 255) as u8
}

const fn rgb_tuple(rgb: [u8; 3]) -> (u8, u8, u8) {
    (rgb[0], rgb[1], rgb[2])
}

/// Upload `asset_bytes` to VRAM if not already resident; return
/// the slot record so the caller can build a TextureMaterial.
/// Returns `None` if the texture parse fails or the VRAM table
/// is full.
/// Look up the VRAM slot a previously-uploaded asset occupies.
/// VRAM_SLOTS is the source of truth -- `RESIDENCY` only tracks
/// the *contract*, which is pre-marked by `ensure_room_resident`
/// before any actual upload runs.
fn find_vram_slot(asset_id: AssetId) -> Option<VramSlot> {
    unsafe {
        VRAM_SLOTS
            .iter()
            .filter_map(|s| *s)
            .find(|s| s.asset == asset_id)
    }
}

fn ensure_texture_uploaded(asset_id: AssetId, asset_bytes: &[u8]) -> Option<VramSlot> {
    // VRAM_SLOTS is the source of truth for "have we actually
    // uploaded this asset". `RESIDENCY` is the *contract* -- it's
    // pre-marked by `ensure_room_resident` before any upload runs,
    // so reading it here would falsely report assets as uploaded
    // and skip the upload entirely.
    if let Some(slot) = find_vram_slot(asset_id) {
        return Some(slot);
    }

    let texture = Texture::from_bytes(asset_bytes).ok()?;
    if texture.clut_entries() != 16 {
        return None;
    }

    // Capacity check before we touch any VRAM state.
    let count = unsafe { VRAM_SLOT_COUNT };
    let room_count = unsafe { ROOM_TEXTURE_COUNT };
    if count >= MAX_RESIDENT_VRAM_ASSETS {
        return None;
    }

    if texture.width() > ROOM_TILE_TEXELS || texture.height() > ROOM_TILE_TEXELS {
        return None;
    }

    // Pack room materials into 64x64 cells inside 4bpp tpages. The
    // material's GP0(E2) texture window makes authored UV repetition
    // wrap inside this cell.
    let room_index = u16::try_from(room_count).ok()?;
    let page_index = room_index / ROOM_TILES_PER_PAGE;
    let tile_index = room_index % ROOM_TILES_PER_PAGE;
    let tile_x = tile_index % ROOM_TILE_COLUMNS;
    let tile_y = tile_index / ROOM_TILE_COLUMNS;
    let tpage_x = SHARED_TPAGE
        .x()
        .checked_add(page_index.checked_mul(ROOM_TPAGE_STRIDE_HW)?)?;
    let end_x = tpage_x.checked_add(ROOM_TPAGE_STRIDE_HW)?;
    if end_x > ROOM_TPAGE_LIMIT_X {
        return None;
    }
    let tile_x_hw = tile_x.checked_mul(ROOM_TILE_HALFWORDS)?;
    let tile_y_px = tile_y.checked_mul(ROOM_TILE_TEXELS)?;
    let tile_origin_u = u8::try_from(tile_x.checked_mul(ROOM_TILE_TEXELS)?).ok()?;
    let tile_origin_v = u8::try_from(tile_y.checked_mul(ROOM_TILE_TEXELS)?).ok()?;
    let clut_x = ROOM_CLUT_BASE_X.checked_add(room_index.checked_mul(ROOM_CLUT_STRIDE)?)?;
    if clut_x.checked_add(texture.clut_entries())? > 1024 {
        return None;
    }
    let tpage = Tpage::new(tpage_x, SHARED_TPAGE.y(), TexDepth::Bit4);

    if !upload_4bpp_tile(
        tpage_x.checked_add(tile_x_hw)?,
        SHARED_TPAGE.y().checked_add(tile_y_px)?,
        ROOM_TILE_HALFWORDS,
        ROOM_TILE_TEXELS,
        &texture,
    ) {
        return None;
    }

    let clut_rect = VramRect::new(clut_x, ROOM_CLUT_Y, texture.clut_entries(), 1);
    upload_opaque_clut(clut_rect, texture.clut_bytes());

    let clut = Clut::new(clut_x, ROOM_CLUT_Y);
    let slot = VramSlot {
        asset: asset_id,
        clut_word: clut.uv_clut_word(),
        tpage_word: tpage.uv_tpage_word(0),
        texture_window: room_texture_window(
            tile_origin_u,
            tile_origin_v,
            texture.width(),
            texture.height(),
        )?,
    };

    unsafe {
        VRAM_SLOTS[count] = Some(slot);
        VRAM_SLOT_COUNT = count + 1;
        ROOM_TEXTURE_COUNT = room_count + 1;
        // Mirror VRAM into the residency tracker. mark_vram_resident
        // returns false if it overflows; we already reserved a
        // slot so this should always succeed.
        let _ = RESIDENCY.mark_vram_resident(asset_id);
    }

    Some(slot)
}

fn room_texture_window(
    origin_u: u8,
    origin_v: u8,
    width: u16,
    height: u16,
) -> Option<TextureWindow> {
    Some(TextureWindow::power_of_two_tile(
        origin_u,
        origin_v,
        room_texture_window_size(width)?,
        room_texture_window_size(height)?,
    ))
}

fn room_texture_window_size(size: u16) -> Option<u8> {
    if size < 8 || size > ROOM_TILE_TEXELS || !size.is_power_of_two() || size % 8 != 0 {
        return None;
    }
    u8::try_from(size).ok()
}

/// Upload an 8bpp model atlas to the dedicated model VRAM
/// region. Returns a `VramSlot` carrying the 8bpp tpage word
/// and the atlas's CLUT word. Reuses an existing slot when the
/// asset's already resident.
///
/// Caller is responsible for confirming `asset_bytes` parses as
/// a `Texture` whose CLUT carries 256 entries (8bpp). Anything
/// else returns `None`.
fn ensure_model_atlas_uploaded(asset_id: AssetId, asset_bytes: &[u8]) -> Option<VramSlot> {
    // Same caveat as `ensure_texture_uploaded`: VRAM_SLOTS is
    // the source of truth, not the residency tracker.
    if let Some(slot) = find_vram_slot(asset_id) {
        return Some(slot);
    }
    let texture = Texture::from_bytes(asset_bytes).ok()?;
    if texture.clut_entries() != 256 {
        // Only 8bpp atlases supported -- 4bpp model atlases
        // would round-trip through `ensure_texture_uploaded`.
        return None;
    }

    let count = unsafe { VRAM_SLOT_COUNT };
    let atlas_count = unsafe { MODEL_ATLAS_COUNT };
    if count >= MAX_RESIDENT_VRAM_ASSETS {
        return None;
    }

    let tpage_x = MODEL_TPAGE.x() + unsafe { MODEL_TPAGE_X_CURSOR };
    let pix_rect = VramRect::new(
        tpage_x,
        MODEL_TPAGE.y(),
        texture.halfwords_per_row(),
        texture.height(),
    );
    upload_bytes(pix_rect, texture.pixel_bytes());

    // 256-entry CLUT: 256 halfwords on a single row.
    let clut_y = MODEL_CLUT_BASE_Y + atlas_count as u16;
    let clut_rect = VramRect::new(0, clut_y, texture.clut_entries(), 1);
    upload_clut(clut_rect, texture.clut_bytes());

    let slot = VramSlot {
        asset: asset_id,
        clut_word: Clut::new(0, clut_y).uv_clut_word(),
        tpage_word: MODEL_TPAGE_WORD,
        texture_window: TextureWindow::NONE,
    };

    unsafe {
        VRAM_SLOTS[count] = Some(slot);
        VRAM_SLOT_COUNT = count + 1;
        MODEL_TPAGE_X_CURSOR += texture.halfwords_per_row();
        MODEL_ATLAS_COUNT = atlas_count + 1;
        let _ = RESIDENCY.mark_vram_resident(asset_id);
    }
    Some(slot)
}

/// Animate + render placed model instances whose owning room matches
/// `current_room`. Meshes, clips, and atlas materials are resolved by
/// `load_runtime_models` once at init; the frame path only chooses
/// phase + transform and submits packets.
///
/// Errors (parse failure, missing asset) skip the instance
/// rather than crashing.
#[derive(Copy, Clone, Debug, Default)]
struct ModelInstanceDrawStats {
    draws: u16,
    stats: TexturedModelRenderStats,
}

fn draw_model_instances(
    current_room: RoomIndex,
    elapsed_vblanks: u32,
    video_hz: u16,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    lighting: &RuntimeRoomLighting,
    models: &[Option<RuntimeModelAsset>; MAX_RUNTIME_MODELS],
    clips: &[Option<Animation<'static>>; MAX_RUNTIME_MODEL_CLIPS],
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
) -> ModelInstanceDrawStats {
    let mut drawn = 0usize;
    let mut out = ModelInstanceDrawStats::default();
    for inst in MODEL_INSTANCES {
        if inst.room != current_room || drawn >= MAX_MODEL_INSTANCES {
            continue;
        }
        let Some(runtime_model) = models.get(inst.model.to_usize()).copied().flatten() else {
            continue;
        };

        // Clip resolution: per-instance override → model default.
        // The cooker validates that both end up `< clip_count`,
        // so by the time we get here `clip_local` is in-range.
        let clip_local = inst.clip.unwrap_or(runtime_model.default_clip);
        let Some(anim) = runtime_model.clip(clips, clip_local) else {
            continue;
        };
        let phase = anim.phase_at_tick_q12(elapsed_vblanks, video_hz);

        // Authored instance positions are floor anchors; cooked
        // model vertices are centred around their bounds.
        let origin =
            floor_anchored_model_origin(inst.x, inst.y, inst.z, runtime_model.world_height);
        let material = lighting.shade_model_material(origin, runtime_model.material);
        let model_options = options
            .with_depth_policy(DepthPolicy::Average)
            .with_cull_mode(CullMode::Back)
            .with_material_layer(material)
            .with_textured_triangle_splitting(false);
        // Instance Y-axis rotation from authored yaw. PSX angle
        // units (4096 per turn) → Q12 sin/cos via the existing
        // GTE shim, then composed into a rotation matrix.
        let instance_rotation = yaw_rotation_matrix(Angle::from_q12(inst.yaw as u16));

        let stats = world.submit_textured_model_primary_joints(
            triangles,
            runtime_model.model,
            anim,
            phase,
            *camera,
            origin,
            instance_rotation,
            unsafe { &mut MODEL_VERTICES },
            unsafe { &mut JOINT_VIEW_TRANSFORMS },
            material,
            model_options,
        );
        accumulate_model_stats(&mut out.stats, stats);
        if stats.primitive_overflow || stats.command_overflow {
            out.draws = drawn as u16;
            return out;
        }
        drawn += 1;
        out.draws = drawn as u16;
    }
    out
}

fn accumulate_model_stats(total: &mut TexturedModelRenderStats, next: TexturedModelRenderStats) {
    total.projected_vertices = total
        .projected_vertices
        .saturating_add(next.projected_vertices);
    total.submitted_triangles = total
        .submitted_triangles
        .saturating_add(next.submitted_triangles);
    total.culled_triangles = total.culled_triangles.saturating_add(next.culled_triangles);
    total.split_triangles = total.split_triangles.saturating_add(next.split_triangles);
    total.skipped_triangles = total
        .skipped_triangles
        .saturating_add(next.skipped_triangles);
    total.dropped_triangles = total
        .dropped_triangles
        .saturating_add(next.dropped_triangles);
    total.vertex_overflow |= next.vertex_overflow;
    total.primitive_overflow |= next.primitive_overflow;
    total.command_overflow |= next.command_overflow;
}

/// Rotation matrix around the world Y axis.
fn yaw_rotation_matrix(yaw: Angle) -> Mat3I16 {
    let s = clamp_i16(yaw.sin().raw());
    let c = clamp_i16(yaw.cos().raw());
    Mat3I16 {
        m: [[c, 0, s], [0, 0x1000, 0], [-s, 0, c]],
    }
}

fn floor_anchored_model_origin(x: i32, y: i32, z: i32, world_height: u16) -> WorldVertex {
    WorldVertex::new(
        x,
        y.saturating_add(model_origin_floor_lift(world_height)),
        z,
    )
}

fn model_origin_floor_lift(world_height: u16) -> i32 {
    (world_height as i32) / 2
}

/// Draw one tinted cube per generated entity record. Cubes
/// reuse the room's first material with an override tint so
/// markers stand out from the surrounding geometry without
/// needing a dedicated texture upload.
fn draw_entity_markers(
    entities: &[EntityRecord],
    current_room: RoomIndex,
    materials: &[WorldRenderMaterial],
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
) {
    if entities.is_empty() || materials.is_empty() {
        return;
    }
    // Reuse the room's first material so we don't need a
    // dedicated marker texture. Tint override picks up the
    // existing CLUT + tpage but recolours.
    let material = materials[0].texture.with_tint(MARKER_TINT);
    let opts = options.with_material_layer(material);
    const UVS: [(u8, u8); 4] = [(0, 0), (64, 0), (64, 64), (0, 64)];

    for entity in entities {
        if entity.room != current_room {
            continue;
        }
        let cx = entity.x;
        let cy = entity.y - MARKER_LIFT - MARKER_HALF;
        let cz = entity.z;
        let h = MARKER_HALF;

        let top = [
            WorldVertex::new(cx - h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy - h, cz + h),
        ];
        let bottom = [
            WorldVertex::new(cx - h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz - h),
        ];
        let north = [
            WorldVertex::new(cx - h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy + h, cz - h),
            WorldVertex::new(cx - h, cy + h, cz - h),
        ];
        let south = [
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy - h, cz + h),
            WorldVertex::new(cx - h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
        ];
        let east = [
            WorldVertex::new(cx + h, cy - h, cz - h),
            WorldVertex::new(cx + h, cy - h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz + h),
            WorldVertex::new(cx + h, cy + h, cz - h),
        ];
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
