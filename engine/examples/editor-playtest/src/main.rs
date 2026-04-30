//! `editor-playtest` -- render a level cooked from the editor.
//!
//! Loads `generated/level_manifest.rs` (a Rust source file the
//! editor's playtest compiler produces via
//! [`psxed_project::playtest::write_package`]) containing:
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

use psx_asset::{Animation, Model, Texture, World as AssetWorld};
use psx_engine::{
    button, draw_room, App, CharacterMotorAnim, CharacterMotorConfig, CharacterMotorInput,
    CharacterMotorState, Config, Ctx, CullMode, DepthBand, DepthPolicy, DepthRange,
    JointViewTransform, Mat3I16, OtFrame, PrimitiveArena, ProjectedVertex, RuntimeRoom, Scene,
    ThirdPersonCameraConfig, ThirdPersonCameraInput, ThirdPersonCameraState, ThirdPersonCameraTarget,
    WorldCamera, WorldProjection, WorldRenderPass, WorldSurfaceOptions, WorldTriCommand,
    WorldVertex,
};
use psx_font::{fonts::BASIC, FontAtlas};
use psx_gpu::{draw_quad_flat, material::TextureMaterial, ot::OrderingTable, prim::TriTextured};
use psx_gte::transform::{cos_1_3_12, sin_1_3_12};
use psx_level::{
    find_asset_of_kind, AssetId, AssetKind, EntityRecord, LevelCharacterRecord,
    LevelMaterialRecord, LevelRoomRecord, ResidencyManager, CHARACTER_CLIP_NONE,
    MODEL_CLIP_INHERIT,
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
    include!("../generated/level_manifest.rs");
}

use generated::{
    ASSETS, CHARACTERS, ENTITIES, LIGHTS, MATERIALS, MODELS, MODEL_CLIPS, MODEL_INSTANCES,
    PLAYER_CONTROLLER, PLAYER_SPAWN, ROOMS, ROOM_RESIDENCY,
};

// VRAM layout. Room materials and model atlases live in
// disjoint regions so a model atlas upload never overwrites a
// room texture (and vice versa).
//
// Room materials: 4bpp pages starting at (640, 0), one tpage per
// material. `draw_room` v1 UVs always start at (0,0), so packing
// multiple 64x64 textures side-by-side inside one tpage would make
// every material sample the first texture with a different CLUT.
//
// Model atlases: 8bpp tpage at (384, 256); stripe atlases
// left-to-right (each atlas occupies its own halfword stride);
// one CLUT row per atlas at y starting at 484 (below the
// material CLUT band so the two never collide).
const SHARED_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);
const TPAGE_WORD: u16 = SHARED_TPAGE.uv_tpage_word(0);
const ROOM_TPAGE_STRIDE_HW: u16 = 64;
const ROOM_TPAGE_LIMIT_X: u16 = 1024;
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
const CAMERA_START_YAW: u16 = 220;
const CAMERA_YAW_STEP: u16 = 12;
const MOVE_STICK_DEADZONE: i16 = 18;
const STICK_MAX: i16 = 127;
const CAMERA_STICK_DEADZONE: i16 = 18;
const CAMERA_STICK_YAW_STEP: i16 = 42;
const CAMERA_HEIGHT_STICK_STEP: i32 = 18;
const CAMERA_HEIGHT_OFFSET_MIN: i32 = -512;
const CAMERA_HEIGHT_OFFSET_MAX: i32 = 768;
const CAMERA_SOFT_LOCK_BREAK_STICK: i16 = 72;
const LOCK_RANGE: i32 = 4096;
const LOCK_BREAK_RANGE: i32 = 5120;
const SOFT_LOCK_RANGE: i32 = 3072;
const SOFT_LOCK_BREAK_RANGE: i32 = 3840;

const HALF_TURN_Q12: u16 = 2048;
/// Fallback follow camera params used when no PLAYER_CONTROLLER
/// was authored -- matches the prior debug behaviour.
const FOLLOW_RADIUS_DEFAULT: i32 = 1400;
const FOLLOW_HEIGHT_DEFAULT: i32 = 700;
const FOLLOW_TARGET_HEIGHT_DEFAULT: i32 = 0;
/// Quanta-per-frame turn rate when the runtime can't resolve a
/// Character (no PLAYER_CONTROLLER). Mirrors the pre-character
/// debug value.
const FALLBACK_PLAYER_YAW_STEP: u16 = 32;
const FALLBACK_PLAYER_SPEED: i32 = 32;
const RUN_BUTTON: u16 = button::CIRCLE;

const OT_DEPTH: usize = 64;
const WORLD_BAND: DepthBand = DepthBand::new(0, OT_DEPTH - 1);
const WORLD_DEPTH_RANGE: DepthRange = DepthRange::new(NEAR_Z, FAR_Z);

const MAX_TEXTURED_TRIS: usize = 4096;

/// Cap on the per-room material slot count. Picked to comfortably
/// exceed the cooker's currently-emitted material count without
/// over-reserving VRAM or RAM. If a future room exceeds this,
/// the runtime fails graceful (skips the over-cap material) and
/// the cook report should also flag.
const MAX_ROOM_MATERIALS: usize = 32;

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

/// Marker visualization tuning. Markers are debug stubs -- keep
/// them visible at orbit-camera scales without dominating the
/// scene.
const MARKER_HALF: i32 = 96;
const MARKER_LIFT: i32 = MARKER_HALF;
const MARKER_TINT: (u8, u8, u8) = (0xff, 0xa8, 0x40);
const ROOM_LIGHT_MAX_Q8: u32 = 144;

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
/// VRAM-resident we record its CLUT word and tpage half-x stride
/// so the per-frame material build can reconstruct its
/// `TextureMaterial` without re-walking the upload code.
#[derive(Copy, Clone)]
struct VramSlot {
    asset: AssetId,
    clut_word: u16,
    tpage_word: u16,
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
    model: u16,
    idle_clip: u16,
    walk_clip: u16,
    /// Optional run clip -- `CHARACTER_CLIP_NONE` when unset.
    /// Runtime falls back to `walk_clip` for run input.
    run_clip: u16,
    /// Optional turn clip (currently unused at runtime -- turn
    /// is folded into idle with yaw input).
    _turn_clip: u16,
    /// Capsule radius for collision. Engine units.
    radius: i32,
    walk_speed: i32,
    run_speed: i32,
    /// Yaw rate translated from degrees/second to PSX angle
    /// units / 60 Hz frame at init time.
    yaw_step_q12: u16,
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
            yaw_step_q12,
            camera_distance: c.camera_distance,
            camera_height: c.camera_height,
            camera_target_height: c.camera_target_height,
        }
    }

    /// Pick the clip index for an animation state, with the
    /// "run falls back to walk when unassigned" rule.
    fn clip_for(&self, anim: PlayerAnim) -> u16 {
        match anim {
            PlayerAnim::Idle => self.idle_clip,
            PlayerAnim::Walk => self.walk_clip,
            PlayerAnim::Run => {
                if self.run_clip == CHARACTER_CLIP_NONE {
                    self.walk_clip
                } else {
                    self.run_clip
                }
            }
        }
    }

    fn motor_config(&self) -> CharacterMotorConfig {
        CharacterMotorConfig::character(
            self.radius,
            self.walk_speed,
            self.run_speed,
            self.yaw_step_q12,
        )
    }
}

struct Playtest {
    /// Active room. `None` until `init` runs and only `Some`
    /// when the manifest had at least one room and its bytes
    /// parsed.
    room: Option<RuntimeRoom<'static>>,
    /// Index in ROOMS the player is currently in. Used to scope
    /// model-instance + light queries.
    room_index: u16,
    /// Active room's material table, ordered by `local_slot`.
    /// Indexed directly by the slot value the cooked `.psxw`
    /// stores per face.
    materials: [Option<TextureMaterial>; MAX_ROOM_MATERIALS],
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
    orbit_yaw: u16,
    orbit_radius: i32,
    /// Runtime third-person camera rig. Updated from render so it
    /// can consume the same room collision view used for drawing.
    camera: ThirdPersonCameraState,
    /// Manual right-stick vertical offset layered on top of the
    /// authored camera height.
    camera_height_offset: i32,
    /// Index into `ENTITIES` for the current lock-on target. In this
    /// vertical-slice pass generic entity markers stand in for
    /// enemies until enemy records exist.
    lock_target: Option<usize>,
    /// Automatic camera-only target. Suppressed after strong
    /// manual camera input until the player leaves target range.
    soft_lock_target: Option<usize>,
    soft_lock_suppressed: bool,
    /// Spawn position retained for orbit-mode targeting.
    spawn: WorldVertex,
    /// Font atlas used for the analog-mode required prompt.
    font: Option<FontAtlas>,
}

impl Playtest {
    const fn new() -> Self {
        Self {
            room: None,
            room_index: 0,
            materials: [const { None }; MAX_ROOM_MATERIALS],
            material_count: 0,
            motor: CharacterMotorState::new(WorldVertex::ZERO, 0),
            character: None,
            anim_state: PlayerAnim::Idle,
            anim_start_tick: 0,
            free_orbit: false,
            orbit_yaw: CAMERA_START_YAW,
            orbit_radius: CAMERA_START_RADIUS,
            camera: ThirdPersonCameraState::new(CAMERA_START_YAW),
            camera_height_offset: 0,
            lock_target: None,
            soft_lock_target: None,
            soft_lock_suppressed: false,
            spawn: WorldVertex::ZERO,
            font: None,
        }
    }
}

impl Scene for Playtest {
    fn init(&mut self, _ctx: &mut Ctx) {
        self.font = Some(FontAtlas::upload(&BASIC, FONT_TPAGE, FONT_CLUT));

        // Empty manifest? Boot to a clear-coloured screen.
        let Some(room_record) = ROOMS.first() else {
            return;
        };

        // Walk the residency contract for this room. Required
        // RAM assets are logical-only (every asset is
        // include_bytes!-resident from process start), but we
        // still tick them through the manager so the change-set
        // counts are honest. Required VRAM assets we'll need
        // textures for -- actual uploads happen below.
        let residency_record = ROOM_RESIDENCY
            .iter()
            .find(|r| r.room == 0)
            .expect("starter room has a residency record");
        let _ = unsafe { RESIDENCY.ensure_room_resident(residency_record) };

        // Resolve and parse the room's world bytes.
        let world_asset = find_asset_of_kind(ASSETS, room_record.world_asset, AssetKind::RoomWorld);
        if let Some(asset) = world_asset {
            if let Ok(world) = AssetWorld::from_bytes(asset.bytes) {
                self.room = Some(RuntimeRoom::from_world(world));
            }
        }

        // Build the material table by walking this room's slice
        // of MATERIALS. For each entry: ensure VRAM-resident
        // (uploading on first sight), then build the
        // TextureMaterial referencing the slot's CLUT/tpage.
        // Pass parsed room dimensions through so room-level
        // lighting samples the *actual* room centre -- not the
        // authored origin, which would land outside the room
        // on any non-1×1 grid.
        let room_dims = self
            .room
            .map(|r| {
                let render = r.render();
                RoomDims {
                    width: render.width(),
                    depth: render.depth(),
                    sector_size: render.sector_size(),
                }
            })
            .unwrap_or(RoomDims::ZERO);
        self.material_count = build_room_materials(room_record, room_dims, 0, &mut self.materials);

        // Player init: prefer PLAYER_CONTROLLER (cook output)
        // for spawn + character; fall back to the bare
        // PLAYER_SPAWN for placeholder manifests.
        let (spawn, character) = match PLAYER_CONTROLLER {
            Some(pc) => {
                let character = CHARACTERS
                    .get(pc.character as usize)
                    .map(RuntimeCharacter::from_record);
                (pc.spawn, character)
            }
            None => (PLAYER_SPAWN, None),
        };
        self.spawn = WorldVertex::new(spawn.x, spawn.y, spawn.z);
        self.character = character;
        self.motor.snap_to(self.spawn, spawn.yaw as u16);
        self.room_index = spawn.room;
        self.anim_state = PlayerAnim::Idle;
        self.anim_start_tick = 0;
        self.camera
            .snap_to_player(self.camera_target(None, false), self.camera_config());
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if !ctx.pad.is_analog() {
            return;
        }

        if ctx.just_pressed(button::SELECT) {
            self.free_orbit = !self.free_orbit;
        }
        if self.free_orbit {
            let (right_x, right_y) = ctx.pad.sticks.right_centered();
            self.orbit_yaw = add_signed_q12(self.orbit_yaw, stick_to_yaw_delta(right_x));
            self.orbit_radius = (self.orbit_radius + stick_to_radius_delta(right_y))
                .clamp(CAMERA_RADIUS_MIN, CAMERA_RADIUS_MAX);
            if ctx.is_held(button::RIGHT) {
                self.orbit_yaw = self.orbit_yaw.wrapping_add(CAMERA_YAW_STEP);
            }
            if ctx.is_held(button::LEFT) {
                self.orbit_yaw = self.orbit_yaw.wrapping_sub(CAMERA_YAW_STEP);
            }
            if ctx.is_held(button::UP) {
                self.orbit_radius = (self.orbit_radius - CAMERA_RADIUS_STEP).max(CAMERA_RADIUS_MIN);
            }
            if ctx.is_held(button::DOWN) {
                self.orbit_radius = (self.orbit_radius + CAMERA_RADIUS_STEP).min(CAMERA_RADIUS_MAX);
            }
            return;
        }

        self.update_camera_height_offset(ctx);

        let input = motor_input(ctx, self.camera.yaw_q12());
        let config = self.motor_config();
        let collision = self.room.as_ref().map(|room| room.collision());
        let motor_frame = self.motor.update(collision, input, config);

        // Animation state comes from the reusable motor, but the
        // playtest intentionally exposes only the core locomotion
        // trio for now: idle, walking, running.
        let new_state = player_anim_from_motor(motor_frame.anim);
        if new_state != self.anim_state {
            self.anim_state = new_state;
            self.anim_start_tick = ctx.time.elapsed_vblanks();
        }

        if ctx.just_pressed(button::R3) {
            self.lock_target = match self.lock_target {
                Some(_) => None,
                None => self.find_best_lock_target(LOCK_RANGE),
            };
            self.soft_lock_target = None;
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
        self.update_soft_lock(ctx);
    }

    fn render(&mut self, ctx: &mut Ctx) {
        if !ctx.pad.is_analog() {
            if let Some(font) = self.font.as_ref() {
                draw_analog_required_prompt(font);
            }
            return;
        }

        let camera = if self.free_orbit {
            WorldCamera::orbit_yaw(
                PROJECTION,
                self.spawn,
                CAMERA_Y_OFFSET,
                self.orbit_radius,
                self.orbit_yaw,
            )
        } else {
            self.update_follow_camera(ctx)
        };

        let mut ot = unsafe { OtFrame::begin(&mut OT) };
        let mut triangles = unsafe { PrimitiveArena::new(&mut TEXTURED_TRIS) };
        let mut world = unsafe { WorldRenderPass::new(&mut ot, &mut WORLD_COMMANDS) };

        if let Some(room) = self.room {
            // Pack the materials slice down to a contiguous
            // `&[TextureMaterial]` indexed by local_slot. Slots
            // that didn't resolve become a sentinel material --
            // visually obvious without crashing the renderer.
            let mut bound: [TextureMaterial; MAX_ROOM_MATERIALS] =
                [TextureMaterial::opaque(0, TPAGE_WORD, (0x80, 0x80, 0x80)); MAX_ROOM_MATERIALS];
            for i in 0..self.material_count {
                if let Some(m) = self.materials[i] {
                    bound[i] = m;
                }
            }
            let materials = &bound[..self.material_count];
            let options = WorldSurfaceOptions::new(WORLD_BAND, WORLD_DEPTH_RANGE);
            draw_room(
                room.render(),
                materials,
                &camera,
                options,
                &mut triangles,
                &mut world,
            );
            draw_entity_markers(
                ENTITIES,
                materials,
                &camera,
                options,
                &mut triangles,
                &mut world,
            );
            // Currently just room 0; future passes wire room
            // switching to a level-graph traversal.
            draw_model_instances(
                self.room_index,
                ctx.time.elapsed_vblanks(),
                ctx.time.video_hz(),
                &camera,
                options,
                &mut triangles,
                &mut world,
            );
            // Player draws on top of model instances. Same
            // `submit_textured_model` path; the scene-tree
            // duplicate (Wraith MeshInstance at room centre)
            // is the *placement preview* -- the active player
            // is the one driven by PLAYER_CONTROLLER and lives
            // at the player position.
            if let Some(character) = self.character {
                let player = self.motor.position();
                draw_player(
                    character,
                    player.x,
                    player.y,
                    player.z,
                    self.motor.yaw_q12(),
                    character.clip_for(self.anim_state),
                    self.anim_start_tick,
                    ctx.time.elapsed_vblanks(),
                    ctx.time.video_hz(),
                    &camera,
                    options,
                    &mut triangles,
                    &mut world,
                );
            }
        }

        world.flush();
        ot.submit();
    }
}

impl Playtest {
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
        config.height = config
            .height
            .saturating_add(self.camera_height_offset)
            .max(256);
        config
    }

    fn camera_target(
        &self,
        lock_target: Option<WorldVertex>,
        moving: bool,
    ) -> ThirdPersonCameraTarget {
        ThirdPersonCameraTarget {
            player: self.motor.position(),
            player_yaw_q12: self.motor.yaw_q12(),
            moving,
            lock_target,
        }
    }

    fn update_follow_camera(&mut self, ctx: &Ctx) -> WorldCamera {
        let input = camera_input(ctx);
        let lock_target = self
            .lock_target_position()
            .or_else(|| self.soft_lock_target_position());
        let target = self.camera_target(lock_target, self.anim_state != PlayerAnim::Idle);
        let config = self.camera_config();
        let collision = self.room.as_ref().map(|room| room.collision());
        self.camera
            .update(PROJECTION, collision, target, input, config)
            .camera
    }

    fn update_camera_height_offset(&mut self, ctx: &Ctx) {
        let (_, right_y) = ctx.pad.sticks.right_centered();
        self.camera_height_offset = self
            .camera_height_offset
            .saturating_add(stick_to_height_delta(right_y))
            .clamp(CAMERA_HEIGHT_OFFSET_MIN, CAMERA_HEIGHT_OFFSET_MAX);
    }

    fn lock_target_position(&self) -> Option<WorldVertex> {
        self.target_position(self.lock_target?)
    }

    fn soft_lock_target_position(&self) -> Option<WorldVertex> {
        self.target_position(self.soft_lock_target?)
    }

    fn target_position(&self, index: usize) -> Option<WorldVertex> {
        let target = ENTITIES.get(index)?;
        if target.room != self.room_index {
            return None;
        }
        Some(WorldVertex::new(target.x, target.y, target.z))
    }

    fn lock_target_valid(&self, range: i32) -> bool {
        self.lock_target
            .is_some_and(|index| self.target_index_valid(index, range))
    }

    fn target_index_valid(&self, index: usize, range: i32) -> bool {
        let Some(target) = self.target_position(index) else {
            return false;
        };
        distance_xz_sq(self.motor.position(), target) <= (range as i64 * range as i64)
    }

    fn find_best_lock_target(&self, range: i32) -> Option<usize> {
        let player = self.motor.position();
        let view_yaw = self.camera.yaw_q12().wrapping_add(HALF_TURN_Q12);
        let sin_yaw = sin_1_3_12(view_yaw) as i64;
        let cos_yaw = cos_1_3_12(view_yaw) as i64;
        let range_sq = range as i64 * range as i64;
        let mut best: Option<(usize, i64)> = None;
        for (index, entity) in ENTITIES.iter().enumerate() {
            if entity.room != self.room_index {
                continue;
            }
            let target = WorldVertex::new(entity.x, entity.y, entity.z);
            let dx = (target.x - player.x) as i64;
            let dz = (target.z - player.z) as i64;
            let dist_sq = dx * dx + dz * dz;
            if dist_sq == 0 || dist_sq > range_sq {
                continue;
            }
            let dot = dx * sin_yaw + dz * cos_yaw;
            if dot <= 0 {
                continue;
            }
            let score = (dot >> 4) - (dist_sq >> 12);
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
        let current_dx = (current.x - player.x) as i64;
        let current_dz = (current.z - player.z) as i64;
        if current_dx == 0 && current_dz == 0 {
            return;
        }
        let range_sq = LOCK_RANGE as i64 * LOCK_RANGE as i64;
        let mut best: Option<(usize, i64)> = None;
        for (index, entity) in ENTITIES.iter().enumerate() {
            if index == current_index || entity.room != self.room_index {
                continue;
            }
            let dx = (entity.x - player.x) as i64;
            let dz = (entity.z - player.z) as i64;
            let dist_sq = dx * dx + dz * dz;
            if dist_sq == 0 || dist_sq > range_sq {
                continue;
            }
            let cross = current_dx * dz - current_dz * dx;
            if direction > 0 {
                if cross >= 0 {
                    continue;
                }
            } else if cross <= 0 {
                continue;
            }
            let dot = current_dx * dx + current_dz * dz;
            let score = ((dot.max(0) << 8) / dist_sq.max(1)) - (dist_sq >> 14);
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

fn draw_player(
    character: RuntimeCharacter,
    x: i32,
    y: i32,
    z: i32,
    yaw: u16,
    clip_local: u16,
    anim_start_tick: u32,
    elapsed_vblanks: u32,
    video_hz: u16,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
) {
    let model_record = match MODELS.get(character.model as usize) {
        Some(m) => m,
        None => return,
    };
    let mesh_asset = match find_asset_of_kind(ASSETS, model_record.mesh_asset, AssetKind::ModelMesh)
    {
        Some(a) => a,
        None => return,
    };
    let model = match Model::from_bytes(mesh_asset.bytes) {
        Ok(m) => m,
        Err(_) => return,
    };

    let atlas_slot = match model_record.texture_asset {
        Some(id) => match find_asset_of_kind(ASSETS, id, AssetKind::Texture) {
            Some(asset) => match ensure_model_atlas_uploaded(asset.id, asset.bytes) {
                Some(slot) => slot,
                None => return,
            },
            None => return,
        },
        None => return,
    };
    let material = TextureMaterial::opaque(
        atlas_slot.clut_word,
        atlas_slot.tpage_word,
        (0x80, 0x80, 0x80),
    );
    let model_options = options
        .with_depth_policy(DepthPolicy::Average)
        .with_cull_mode(CullMode::Back)
        .with_material_layer(material);

    if clip_local >= model_record.clip_count {
        return;
    }
    let global = (model_record.clip_first + clip_local) as usize;
    let clip_record = match MODEL_CLIPS.get(global) {
        Some(c) => c,
        None => return,
    };
    let anim_asset = match find_asset_of_kind(
        ASSETS,
        clip_record.animation_asset,
        AssetKind::ModelAnimation,
    ) {
        Some(a) => a,
        None => return,
    };
    let anim = match Animation::from_bytes(anim_asset.bytes) {
        Ok(a) => a,
        Err(_) => return,
    };
    // Phase the animation relative to the clip-start tick so
    // state changes don't pop into the middle of a new clip.
    let local_tick = elapsed_vblanks.saturating_sub(anim_start_tick);
    let phase = anim.phase_at_tick_q12(local_tick, video_hz);

    let origin = floor_anchored_model_origin(x, y, z, model_record.world_height);
    let instance_rotation = yaw_rotation_matrix(yaw);

    let _ = world.submit_textured_model(
        triangles,
        model,
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
/// Parsed room dimensions in the runtime's preferred shape.
/// Pulled off `RuntimeRoom` once at init so the per-frame light
/// sampler can compute the room centre without re-parsing the
/// `.psxw` blob.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct RoomDims {
    width: u16,
    depth: u16,
    sector_size: i32,
}

impl RoomDims {
    /// Sentinel used when the room couldn't be parsed.
    /// `room_center_world` falls back to the cooked origin in
    /// that case so an unparseable manifest still picks *some*
    /// sample point.
    const ZERO: Self = Self {
        width: 0,
        depth: 0,
        sector_size: 0,
    };
}

fn build_room_materials(
    room: &LevelRoomRecord,
    room_dims: RoomDims,
    room_index: u16,
    out: &mut [Option<TextureMaterial>; MAX_ROOM_MATERIALS],
) -> usize {
    let first = room.material_first as usize;
    let count = room.material_count as usize;
    let slice: &[LevelMaterialRecord] = &MATERIALS[first..first + count];

    // Compute a single room-center light contribution and
    // modulate every material tint by it. Per-face lighting
    // would need a draw_room change -- for this pass we ship
    // room-level lighting that honours light colour /
    // intensity / radius without a renderer rewrite. The
    // editor preview does per-face lighting so authors still
    // see spatial variation while authoring.
    let room_center = room_center_world(room, room_dims);
    let lit_tint_factor = accumulate_room_light(room_center, room_index);

    let mut max_slot: usize = 0;
    for material in slice {
        let slot = material.local_slot as usize;
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
        let lit = modulate_tint(material.tint_rgb, lit_tint_factor);
        out[slot] = Some(TextureMaterial::opaque(
            slot_record.clut_word,
            slot_record.tpage_word,
            lit,
        ));
        if slot + 1 > max_slot {
            max_slot = slot + 1;
        }
    }
    max_slot
}

/// World coords of the room centre. Cooker emits geometry
/// array-rooted at world `(0, 0)`, so the centre is
/// `(width * sector_size / 2, 0, depth * sector_size / 2)`.
/// Without parsed dimensions (placeholder manifest, parse
/// failure) we fall back to the cooked origin so the sampler
/// still has *some* sample point.
fn room_center_world(room: &LevelRoomRecord, dims: RoomDims) -> [i32; 3] {
    if dims.sector_size <= 0 || (dims.width == 0 && dims.depth == 0) {
        return [room.origin_x, 0, room.origin_z];
    }
    [
        (dims.width as i32) * dims.sector_size / 2,
        0,
        (dims.depth as i32) * dims.sector_size / 2,
    ]
}

/// Walk every `LIGHTS` record for `room_index` and accumulate
/// `(R, G, B)` brightness contributions at `world_center`. Each
/// channel returned is in 0..=ROOM_LIGHT_MAX_Q8 -- neutral light
/// produces 128 (matching `tint_rgb`'s neutral value); bright
/// lights get a little headroom without crushing authored textures
/// to white.
fn accumulate_room_light(world_center: [i32; 3], room_index: u16) -> (u32, u32, u32) {
    // Start at neutral 128 so an unlit room is not pitch black.
    let mut accum: [u32; 3] = [128, 128, 128];
    for light in LIGHTS {
        if light.room != room_index {
            continue;
        }
        let dx = (world_center[0] - light.x) as i64;
        let dy = (world_center[1] - light.y) as i64;
        let dz = (world_center[2] - light.z) as i64;
        let d2 = dx * dx + dy * dy + dz * dz;
        let r = light.radius as i64;
        if r <= 0 {
            continue;
        }
        let r2 = r * r;
        if d2 >= r2 {
            continue;
        }
        let d = isqrt(d2);
        // Linear falloff in Q8: weight = 256 * (r - d) / r.
        let weight_q8 = (((r - d) << 8) / r) as u32;
        // intensity_q8 already = intensity × 256.
        let intensity_q8 = light.intensity_q8 as u32;
        for c in 0..3 {
            let contrib = (light.color[c] as u32) * intensity_q8 / 256 * weight_q8 / 256;
            accum[c] = accum[c].saturating_add(contrib);
        }
    }
    (
        accum[0].min(ROOM_LIGHT_MAX_Q8),
        accum[1].min(ROOM_LIGHT_MAX_Q8),
        accum[2].min(ROOM_LIGHT_MAX_Q8),
    )
}

/// Multiply a base tint (`128 = neutral`) by a Q8 lighting
/// factor, clamping to 8-bit. Output goes straight into
/// `TextureMaterial::opaque`.
fn modulate_tint(base: [u8; 3], factor: (u32, u32, u32)) -> (u8, u8, u8) {
    let mod_channel = |b: u8, f: u32| -> u8 {
        let scaled = (b as u32) * f / 128;
        scaled.min(255) as u8
    };
    (
        mod_channel(base[0], factor.0),
        mod_channel(base[1], factor.1),
        mod_channel(base[2], factor.2),
    )
}

/// Simple integer square root for `i64` -- same shape as the
/// host editor's helper; needed because `f32` would drag in
/// libm on the no_std target.
fn isqrt(value: i64) -> i64 {
    if value <= 0 {
        return 0;
    }
    let mut x = value as u64;
    let mut r: u64 = 0;
    let mut bit: u64 = 1u64 << 62;
    while bit > x {
        bit >>= 2;
    }
    while bit != 0 {
        if x >= r + bit {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    r as i64
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

    // Pick a full 4bpp tpage page per room material. v1 world UVs
    // are page-relative and always start at u=0, so page isolation is
    // the material selector; the CLUT only selects the palette.
    let room_index = u16::try_from(room_count).ok()?;
    let tpage_x = SHARED_TPAGE
        .x()
        .checked_add(room_index.checked_mul(ROOM_TPAGE_STRIDE_HW)?)?;
    let end_x = tpage_x.checked_add(texture.halfwords_per_row())?;
    if end_x > ROOM_TPAGE_LIMIT_X {
        return None;
    }
    let clut_x = ROOM_CLUT_BASE_X.checked_add(room_index.checked_mul(ROOM_CLUT_STRIDE)?)?;
    if clut_x.checked_add(texture.clut_entries())? > 1024 {
        return None;
    }
    let tpage = Tpage::new(tpage_x, SHARED_TPAGE.y(), TexDepth::Bit4);

    let pix_rect = VramRect::new(
        tpage_x,
        SHARED_TPAGE.y(),
        texture.halfwords_per_row(),
        texture.height(),
    );
    upload_bytes(pix_rect, texture.pixel_bytes());

    let clut_rect = VramRect::new(clut_x, ROOM_CLUT_Y, texture.clut_entries(), 1);
    upload_clut(clut_rect, texture.clut_bytes());

    let clut = Clut::new(clut_x, ROOM_CLUT_Y);
    let slot = VramSlot {
        asset: asset_id,
        clut_word: clut.uv_clut_word(),
        tpage_word: tpage.uv_tpage_word(0),
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

/// Animate + render every placed model instance whose owning
/// room matches `current_room`. Each instance:
/// 1. parses its `LevelModelRecord` mesh asset bytes;
/// 2. parses its current clip's `.psxanim` bytes (or holds bind
///    pose if the model has no clips / clip out of range);
/// 3. ensures the atlas is VRAM-resident;
/// 4. runs the GTE-driven `submit_textured_model` path with a
///    transform anchored at `(instance.x, instance.y, instance.z)`.
///
/// Errors (parse failure, missing asset) skip the instance
/// rather than crashing.
fn draw_model_instances(
    current_room: u16,
    elapsed_vblanks: u32,
    video_hz: u16,
    camera: &WorldCamera,
    options: WorldSurfaceOptions,
    triangles: &mut PrimitiveArena<'_, TriTextured>,
    world: &mut WorldRenderPass<'_, '_, OT_DEPTH>,
) {
    let mut drawn = 0usize;
    for inst in MODEL_INSTANCES {
        if inst.room != current_room || drawn >= MAX_MODEL_INSTANCES {
            continue;
        }
        let model_record = match MODELS.get(inst.model as usize) {
            Some(m) => m,
            None => continue,
        };
        let mesh_asset =
            match find_asset_of_kind(ASSETS, model_record.mesh_asset, AssetKind::ModelMesh) {
                Some(a) => a,
                None => continue,
            };
        let model = match Model::from_bytes(mesh_asset.bytes) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Atlas: cooker validation guarantees `texture_asset` is
        // `Some` for every placed model -- runtime treats a `None`
        // as a stale manifest and skips rather than crashing.
        let atlas_slot = match model_record.texture_asset {
            Some(id) => match find_asset_of_kind(ASSETS, id, AssetKind::Texture) {
                Some(asset) => match ensure_model_atlas_uploaded(asset.id, asset.bytes) {
                    Some(slot) => slot,
                    None => continue,
                },
                None => continue,
            },
            None => continue,
        };
        let material = TextureMaterial::opaque(
            atlas_slot.clut_word,
            atlas_slot.tpage_word,
            (0x80, 0x80, 0x80),
        );
        let model_options = options
            .with_depth_policy(DepthPolicy::Average)
            .with_cull_mode(CullMode::Back)
            .with_material_layer(material);

        // Clip resolution: per-instance override → model default.
        // The cooker validates that both end up `< clip_count`,
        // so by the time we get here `clip_local` is in-range.
        let clip_local = if inst.clip == MODEL_CLIP_INHERIT {
            model_record.default_clip
        } else {
            inst.clip
        };
        if clip_local >= model_record.clip_count {
            // Defensive -- cooker guarantees this won't happen.
            continue;
        }
        let global = (model_record.clip_first + clip_local) as usize;
        let clip_record = &MODEL_CLIPS[global];
        let Some(anim_asset) = find_asset_of_kind(
            ASSETS,
            clip_record.animation_asset,
            AssetKind::ModelAnimation,
        ) else {
            continue;
        };
        let Ok(anim) = Animation::from_bytes(anim_asset.bytes) else {
            continue;
        };
        let phase = anim.phase_at_tick_q12(elapsed_vblanks, video_hz);

        // Authored instance positions are floor anchors; cooked
        // model vertices are centred around their bounds.
        let origin = floor_anchored_model_origin(inst.x, inst.y, inst.z, model_record.world_height);
        // Instance Y-axis rotation from authored yaw. PSX angle
        // units (4096 per turn) → Q12 sin/cos via the existing
        // GTE shim, then composed into a rotation matrix.
        let instance_rotation = yaw_rotation_matrix(inst.yaw as u16);

        let stats = world.submit_textured_model(
            triangles,
            model,
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
        if stats.primitive_overflow || stats.command_overflow {
            return;
        }
        drawn += 1;
    }
}

/// Rotation matrix around the world Y axis for `yaw` in PSX
/// angle units (`0..4096` = full turn). Q12 fixed-point -- drop
/// straight into `submit_textured_model`'s `instance_rotation`.
fn yaw_rotation_matrix(yaw: u16) -> Mat3I16 {
    let s = sin_1_3_12(yaw);
    let c = cos_1_3_12(yaw);
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
    materials: &[TextureMaterial],
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
    let base = materials[0];
    let material = TextureMaterial::opaque(base.clut_word(), base.tpage_word(), MARKER_TINT);
    let opts = options.with_material_layer(material);
    const UVS: [(u8, u8); 4] = [(0, 0), (64, 0), (64, 64), (0, 64)];

    for entity in entities {
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
