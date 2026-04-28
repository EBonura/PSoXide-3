//! `editor-playtest` — render a level cooked from the editor.
//!
//! Loads `generated/level_manifest.rs` (a Rust source file the
//! editor's playtest compiler produces via
//! [`psxed_project::playtest::write_package`]) containing:
//!
//! * a master [`LevelAssetRecord`] table — every cooked
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
//! textures — the asset table is the source of truth.
//!
//! Controls (free-orbit toggled with SELECT):
//! * D-pad LEFT / RIGHT — yaw camera (orbit) or player (tank).
//! * D-pad UP / DOWN    — zoom (orbit) or walk (tank).

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_asset::{Animation, Model, Texture, World as AssetWorld};
use psx_engine::{
    button, draw_room, App, Config, Ctx, CullMode, DepthBand, DepthPolicy, DepthRange,
    JointViewTransform, Mat3I16, OtFrame, PrimitiveArena, ProjectedTexturedVertex,
    ProjectedVertex, RuntimeRoom, Scene, WorldCamera, WorldProjection, WorldRenderPass,
    WorldSurfaceOptions, WorldTriCommand, WorldVertex,
};
use psx_gpu::{material::TextureMaterial, ot::OrderingTable, prim::TriTextured};
use psx_gte::transform::{cos_1_3_12, sin_1_3_12};
use psx_level::{
    find_asset_of_kind, AssetId, AssetKind, EntityRecord, LevelMaterialRecord, LevelRoomRecord,
    ResidencyManager, MODEL_CLIP_INHERIT,
};
use psx_vram::{upload_bytes, Clut, TexDepth, Tpage, VramRect};

// Placeholder manifests reference unused statics; populated
// manifests reference all of them. Quiet either side here.
#[allow(dead_code, unused_imports)]
mod generated {
    include!("../generated/level_manifest.rs");
}

use generated::{
    ASSETS, ENTITIES, MATERIALS, MODELS, MODEL_CLIPS, MODEL_INSTANCES, PLAYER_SPAWN, ROOMS,
    ROOM_RESIDENCY,
};

// VRAM layout. Room materials and model atlases live in
// disjoint regions so a model atlas upload never overwrites a
// room texture (and vice versa).
//
// Room materials: 4bpp tpage at (640, 0); stripe textures
// left-to-right; one CLUT row per material at y >= 480.
//
// Model atlases: 8bpp tpage at (384, 256); stripe atlases
// left-to-right (each atlas occupies its own halfword stride);
// one CLUT row per atlas at y starting at 484 (below the
// material CLUT band so the two never collide).
const SHARED_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);
const TPAGE_WORD: u16 = SHARED_TPAGE.uv_tpage_word(0);
/// First CLUT row used by room material textures. Row N below
/// this belongs to material N's CLUT.
const CLUT_BASE_Y: u16 = 480;

const MODEL_TPAGE: Tpage = Tpage::new(384, 256, TexDepth::Bit8);
const MODEL_TPAGE_WORD: u16 = MODEL_TPAGE.uv_tpage_word(0);
/// First CLUT row used by model atlases. 256-entry CLUTs span
/// y..y+1 only; we step one row down per uploaded atlas, so
/// `MODEL_CLUT_BASE_Y - n` is the row for the n-th atlas.
const MODEL_CLUT_BASE_Y: u16 = 484;

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

const HALF_TURN_Q12: u16 = 2048;
const FOLLOW_RADIUS: i32 = 1400;
const FOLLOW_HEIGHT: i32 = 700;
const PLAYER_SPEED: i32 = 32;
const PLAYER_YAW_STEP: u16 = 32;

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
/// Joint-transform scratch — all biped rigs we currently cook
/// fit comfortably in 32.
const JOINT_CAP: usize = 32;
/// Cap on placed model instances rendered per frame.
const MAX_MODEL_INSTANCES: usize = 16;

/// Marker visualization tuning. Markers are debug stubs — keep
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

static mut OT: OrderingTable<OT_DEPTH> = OrderingTable::new();
static mut TEXTURED_TRIS: [TriTextured; MAX_TEXTURED_TRIS] =
    [const { TRI_ZERO }; MAX_TEXTURED_TRIS];
static mut WORLD_COMMANDS: [WorldTriCommand; MAX_TEXTURED_TRIS] =
    [WorldTriCommand::EMPTY; MAX_TEXTURED_TRIS];
static mut MODEL_VERTICES: [ProjectedTexturedVertex; MODEL_VERTEX_CAP] =
    [ProjectedTexturedVertex::new(ProjectedVertex::new(0, 0, 0), 0, 0); MODEL_VERTEX_CAP];
static mut JOINT_VIEW_TRANSFORMS: [JointViewTransform; JOINT_CAP] =
    [JointViewTransform::ZERO; JOINT_CAP];

/// Residency manager — tracks which AssetIds are RAM/VRAM
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
/// Number of VRAM slots used so far (next CLUT row + tpage cursor).
static mut VRAM_SLOT_COUNT: usize = 0;
/// Tpage X cursor (in halfwords) for the room-material 4bpp
/// region. Each uploaded room texture advances it by
/// `halfwords_per_row`.
static mut TPAGE_X_CURSOR: u16 = 0;

/// Tpage X cursor (in halfwords) for the model-atlas 8bpp
/// region. Distinct cursor so room-material uploads don't shift
/// model atlas positions and vice versa.
static mut MODEL_TPAGE_X_CURSOR: u16 = 0;
/// Number of model atlases uploaded so far. Doubles as the
/// CLUT row offset: each 8bpp atlas needs a fresh 256-entry
/// CLUT row.
static mut MODEL_ATLAS_COUNT: usize = 0;

struct Playtest {
    /// Active room. `None` until `init` runs and only `Some`
    /// when the manifest had at least one room and its bytes
    /// parsed.
    room: Option<RuntimeRoom<'static>>,
    /// Active room's material table, ordered by `local_slot`.
    /// Indexed directly by the slot value the cooked `.psxw`
    /// stores per face.
    materials: [Option<TextureMaterial>; MAX_ROOM_MATERIALS],
    /// `materials[..material_count]` is the in-use slice; rest
    /// is `None`.
    material_count: usize,
    /// Player position in room-local engine units.
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
            materials: [const { None }; MAX_ROOM_MATERIALS],
            material_count: 0,
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
        // Empty manifest? Boot to a clear-coloured screen.
        let Some(room_record) = ROOMS.first() else {
            return;
        };

        // Walk the residency contract for this room. Required
        // RAM assets are logical-only (every asset is
        // include_bytes!-resident from process start), but we
        // still tick them through the manager so the change-set
        // counts are honest. Required VRAM assets we'll need
        // textures for — actual uploads happen below.
        let residency_record = ROOM_RESIDENCY
            .iter()
            .find(|r| r.room == 0)
            .expect("starter room has a residency record");
        let _ = unsafe { RESIDENCY.ensure_room_resident(residency_record) };

        // Resolve and parse the room's world bytes.
        let world_asset =
            find_asset_of_kind(ASSETS, room_record.world_asset, AssetKind::RoomWorld);
        if let Some(asset) = world_asset {
            if let Ok(world) = AssetWorld::from_bytes(asset.bytes) {
                self.room = Some(RuntimeRoom::from_world(world));
            }
        }

        // Build the material table by walking this room's slice
        // of MATERIALS. For each entry: ensure VRAM-resident
        // (uploading on first sight), then build the
        // TextureMaterial referencing the slot's CLUT/tpage.
        self.material_count = build_room_materials(room_record, &mut self.materials);

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
            if ctx.is_held(button::RIGHT) {
                self.player_yaw = self.player_yaw.wrapping_add(PLAYER_YAW_STEP);
            }
            if ctx.is_held(button::LEFT) {
                self.player_yaw = self.player_yaw.wrapping_sub(PLAYER_YAW_STEP);
            }
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

    fn render(&mut self, ctx: &mut Ctx) {
        let camera = if self.free_orbit {
            WorldCamera::orbit_yaw(
                PROJECTION,
                self.spawn,
                CAMERA_Y_OFFSET,
                self.orbit_radius,
                self.orbit_yaw,
            )
        } else {
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
            // Pack the materials slice down to a contiguous
            // `&[TextureMaterial]` indexed by local_slot. Slots
            // that didn't resolve become a sentinel material —
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
            draw_entity_markers(ENTITIES, materials, &camera, options, &mut triangles, &mut world);
            // Currently just room 0; future passes wire room
            // switching to a level-graph traversal.
            draw_model_instances(
                0,
                ctx.time.elapsed_vblanks(),
                ctx.time.video_hz(),
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

/// Walk `room.material_first..material_first + material_count`,
/// resolve each material's texture asset, and build a
/// TextureMaterial in `out` indexed by `local_slot`. Each
/// texture asset is uploaded at most once across the program
/// lifetime — the residency manager + VRAM_SLOTS tracks who's
/// already up.
///
/// Returns the highest `local_slot + 1` so the caller knows the
/// in-use prefix length.
fn build_room_materials(
    room: &LevelRoomRecord,
    out: &mut [Option<TextureMaterial>; MAX_ROOM_MATERIALS],
) -> usize {
    let first = room.material_first as usize;
    let count = room.material_count as usize;
    let slice: &[LevelMaterialRecord] = &MATERIALS[first..first + count];

    let mut max_slot: usize = 0;
    for material in slice {
        let slot = material.local_slot as usize;
        if slot >= MAX_ROOM_MATERIALS {
            // Cooker should already have failed validation;
            // skip rather than crash if it slips through.
            continue;
        }
        let Some(asset) =
            find_asset_of_kind(ASSETS, material.texture_asset, AssetKind::Texture)
        else {
            continue;
        };
        let Some(slot_record) = ensure_texture_uploaded(asset.id, asset.bytes) else {
            continue;
        };
        out[slot] = Some(TextureMaterial::opaque(
            slot_record.clut_word,
            slot_record.tpage_word,
            (
                material.tint_rgb[0],
                material.tint_rgb[1],
                material.tint_rgb[2],
            ),
        ));
        if slot + 1 > max_slot {
            max_slot = slot + 1;
        }
    }
    max_slot
}

/// Upload `asset_bytes` to VRAM if not already resident; return
/// the slot record so the caller can build a TextureMaterial.
/// Returns `None` if the texture parse fails or the VRAM table
/// is full.
/// Look up the VRAM slot a previously-uploaded asset occupies.
/// VRAM_SLOTS is the source of truth — `RESIDENCY` only tracks
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
    // uploaded this asset". `RESIDENCY` is the *contract* — it's
    // pre-marked by `ensure_room_resident` before any upload runs,
    // so reading it here would falsely report assets as uploaded
    // and skip the upload entirely.
    if let Some(slot) = find_vram_slot(asset_id) {
        return Some(slot);
    }

    let texture = Texture::from_bytes(asset_bytes).ok()?;

    // Capacity check before we touch any VRAM state.
    let count = unsafe { VRAM_SLOT_COUNT };
    if count >= MAX_RESIDENT_VRAM_ASSETS {
        return None;
    }

    // Pick the next CLUT row and tpage stride offset, then
    // upload pixels + CLUT.
    let clut_y = CLUT_BASE_Y - count as u16;
    let tpage_x = SHARED_TPAGE.x() + unsafe { TPAGE_X_CURSOR };

    let pix_rect = VramRect::new(tpage_x, SHARED_TPAGE.y(), texture.halfwords_per_row(), texture.height());
    upload_bytes(pix_rect, texture.pixel_bytes());

    let clut_rect = VramRect::new(0, clut_y, texture.clut_entries(), 1);
    upload_clut(clut_rect, texture.clut_bytes());

    let clut = Clut::new(0, clut_y);
    let slot = VramSlot {
        asset: asset_id,
        clut_word: clut.uv_clut_word(),
        tpage_word: TPAGE_WORD,
    };

    unsafe {
        VRAM_SLOTS[count] = Some(slot);
        VRAM_SLOT_COUNT = count + 1;
        TPAGE_X_CURSOR += texture.halfwords_per_row();
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
        // Only 8bpp atlases supported — 4bpp model atlases
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
        let mesh_asset = match find_asset_of_kind(
            ASSETS,
            model_record.mesh_asset,
            AssetKind::ModelMesh,
        ) {
            Some(a) => a,
            None => continue,
        };
        let model = match Model::from_bytes(mesh_asset.bytes) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Atlas: required for textured rendering. Fall back to
        // skip when missing rather than rendering untextured —
        // the editor's validation should have already complained.
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

        // Clip resolution: per-instance override → model
        // default → bind pose.
        let clip_local = if inst.clip == MODEL_CLIP_INHERIT {
            model_record.default_clip
        } else {
            inst.clip
        };
        let frame_q12 = if (clip_local as u16) < model_record.clip_count {
            let global = (model_record.clip_first + clip_local) as usize;
            let clip_record = &MODEL_CLIPS[global];
            match find_asset_of_kind(ASSETS, clip_record.animation_asset, AssetKind::ModelAnimation)
            {
                Some(asset) => match Animation::from_bytes(asset.bytes) {
                    Ok(anim) => Some((anim, anim.phase_at_tick_q12(elapsed_vblanks, video_hz))),
                    Err(_) => None,
                },
                None => None,
            }
        } else {
            None
        };

        // Origin: room-local instance position.
        let origin = WorldVertex::new(inst.x, inst.y, inst.z);
        // Instance Y-axis rotation from authored yaw. PSX angle
        // units (4096 per turn) → Q12 sin/cos via the existing
        // GTE shim, then composed into a rotation matrix.
        let instance_rotation = yaw_rotation_matrix(inst.yaw as u16);

        // Render: animated when we have a clip, else hold the
        // first frame of any clip we *can* find as a static
        // pose (showcase-model parity for bind-pose preview).
        if let Some((anim, phase)) = frame_q12 {
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
        }
        drawn += 1;
    }
}

/// Rotation matrix around the world Y axis for `yaw` in PSX
/// angle units (`0..4096` = full turn). Q12 fixed-point — drop
/// straight into `submit_textured_model`'s `instance_rotation`.
fn yaw_rotation_matrix(yaw: u16) -> Mat3I16 {
    let s = sin_1_3_12(yaw);
    let c = cos_1_3_12(yaw);
    Mat3I16 {
        m: [
            [c, 0, s],
            [0, 0x1000, 0],
            [-s, 0, c],
        ],
    }
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
    let material = TextureMaterial::opaque(
        material_clut_word(base),
        material_tpage_word(base),
        MARKER_TINT,
    );
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

/// Extract a `TextureMaterial`'s CLUT word for re-tinting.
/// `TextureMaterial`'s fields are private, so we reconstruct
/// via a helper that round-trips through the public API.
fn material_clut_word(_m: TextureMaterial) -> u16 {
    // The marker reuses material[0]'s CLUT/tpage. We don't have
    // a public getter, so the cleanest workaround is to use the
    // first uploaded slot's CLUT word from VRAM_SLOTS, which is
    // stable across the program's lifetime. Marker code only
    // runs after `init` has populated at least one slot.
    unsafe {
        VRAM_SLOTS
            .iter()
            .find_map(|s| *s)
            .map(|s| s.clut_word)
            .unwrap_or(0)
    }
}

fn material_tpage_word(_m: TextureMaterial) -> u16 {
    TPAGE_WORD
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

/// Stamp the 0x8000 (semi-transparency-disable) bit on every
/// non-zero CLUT entry so opaque textures don't accidentally
/// trigger STP-bit blending.
fn upload_clut(rect: VramRect, bytes: &[u8]) {
    let mut marked = [0u8; 512];
    if bytes.len() > marked.len() || !bytes.len().is_multiple_of(2) {
        return;
    }

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
