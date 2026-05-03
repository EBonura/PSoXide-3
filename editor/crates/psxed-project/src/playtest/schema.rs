//! Host-side package schema for embedded editor play mode.

use crate::{MaterialFaceSidedness, ResourceId};

/// Generated subdirectory inside the playtest example that
/// receives manifest source + `rooms/` + `textures/`. Stable
/// so the example's `include!` paths don't move.
pub const GENERATED_DIRNAME: &str = "generated";

/// Tracked placeholder manifest. Kept buildable in source
/// control so a fresh clone can compile before any project cook.
pub const MANIFEST_FILENAME: &str = "level_manifest.rs";

/// Ignored cooked Rust-source manifest written by editor Play /
/// `make cook-playtest`.
pub const COOKED_MANIFEST_FILENAME: &str = "level_manifest.cooked.rs";

/// Subdirectory inside `generated/` that holds cooked `.psxw`
/// blobs.
pub const ROOMS_DIRNAME: &str = "rooms";

/// Subdirectory inside `generated/` that holds copied `.psxt`
/// texture blobs.
pub const TEXTURES_DIRNAME: &str = "textures";

/// Subdirectory inside `generated/` that holds per-model
/// folders (`model_NNN/`) carrying mesh + atlas + animation
/// blobs. One subfolder per unique [`ResourceData::Model`]
/// referenced by any placed [`NodeKind::MeshInstance`].
pub const MODELS_DIRNAME: &str = "models";

/// Coarse asset class -- mirrors [`psx_level::AssetKind`] but
/// stays host-side `String`/`Vec` friendly. Converted to the
/// runtime enum at write time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaytestAssetKind {
    /// Cooked `.psxw` room blob.
    RoomWorld,
    /// Cooked `.psxt` texture blob (room atlas or model atlas).
    Texture,
    /// Cooked `.psxmdl` mesh blob.
    ModelMesh,
    /// Cooked `.psxanim` skeletal animation clip.
    ModelAnimation,
}

/// One asset destined for the master table. Owns its bytes so
/// callers can write them out to the generated tree without
/// reaching back into the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestAsset {
    /// Asset class -- drives extension + loader.
    pub kind: PlaytestAssetKind,
    /// Backing payload.
    pub bytes: Vec<u8>,
    /// Filename inside the kind's subdirectory (e.g.
    /// `room_000.psxw`). Stable across runs because asset order
    /// is deterministic.
    pub filename: String,
    /// Diagnostic label -- display name of the source resource
    /// or room. Surfaces in cook reports and stays out of the
    /// runtime contract.
    pub source_label: String,
}

/// One room's residency-aware record. Carries indices into
/// [`PlaytestPackage::assets`] and [`PlaytestPackage::materials`]
/// so the writer can resolve `AssetId`s deterministically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestRoom {
    /// Display name lifted from the editor scene tree.
    pub name: String,
    /// Index into [`PlaytestPackage::assets`] of the room's
    /// `RoomWorld` asset.
    pub world_asset_index: usize,
    /// Editor-side `WorldGrid::origin[0]` (diagnostic only).
    pub origin_x: i32,
    /// Editor-side `WorldGrid::origin[1]`.
    pub origin_z: i32,
    /// Engine units per sector.
    pub sector_size: i32,
    /// First index into [`PlaytestPackage::materials`] for this
    /// room's slice.
    pub material_first: u16,
    /// Number of material records in the slice. Matches the
    /// cooked `.psxw`'s material count exactly.
    pub material_count: u16,
    /// Fog/depth-cue far colour.
    pub fog_rgb: [u8; 3],
    /// Fog start distance in engine units.
    pub fog_near: i32,
    /// Fog end distance in engine units.
    pub fog_far: i32,
    /// Room flags mirrored into the runtime manifest.
    pub flags: u16,
}

/// One material slot binding. Lifted from
/// [`CookedWorldMaterial`] and pinned to its owning room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestMaterial {
    /// Owning room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Cooked-world local material slot -- matches the slot value
    /// stored in the `.psxw`.
    pub local_slot: u16,
    /// Index into [`PlaytestPackage::assets`] of the texture
    /// asset bound at this slot.
    pub texture_asset_index: usize,
    /// Per-material modulation tint.
    pub tint_rgb: [u8; 3],
    /// Which side(s) of faces using this material should render.
    pub face_sidedness: MaterialFaceSidedness,
}

/// One animation clip bound to a [`PlaytestModel`]. Carries
/// pre-resolved indices into the master asset table so the
/// writer can emit `LevelModelClipRecord`s without re-walking
/// the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestModelClip {
    /// Owning model index in [`PlaytestPackage::models`].
    pub model: u16,
    /// Display name surfaced in debug HUDs.
    pub name: String,
    /// Index into [`PlaytestPackage::assets`] of the cooked
    /// `.psxanim` blob.
    pub animation_asset_index: usize,
}

/// One named model attachment socket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestModelSocket {
    /// Owning model index.
    pub model: u16,
    /// Socket name used by Equipment records.
    pub name: String,
    /// Joint index in the cooked model skeleton.
    pub joint: u16,
    /// Local translation relative to the joint pose.
    pub translation: [i32; 3],
    /// Local Euler rotation in Q12 turns: X/Y/Z.
    pub rotation_q12: [i16; 3],
}

/// One cooked PSX model included in the playtest package. A
/// [`ResourceData::Model`] referenced by any placed instance is
/// promoted into one `PlaytestModel`; multiple instances share
/// the same record (deduplicated by source `ResourceId`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestModel {
    /// Display name lifted from the editor resource.
    pub name: String,
    /// Source resource id -- used to deduplicate instances and
    /// to resolve per-instance clip overrides back to clip
    /// indices within this model's slice.
    pub source_resource: ResourceId,
    /// Index into [`PlaytestPackage::assets`] of the cooked
    /// `.psxmdl` blob.
    pub mesh_asset_index: usize,
    /// Index into [`PlaytestPackage::assets`] of the atlas
    /// `.psxt` blob. Always `Some` for placed models -- the
    /// playtest cooker rejects instances of models without an
    /// atlas. Kept as `Option` so the schema can later carry
    /// untextured author-time bundles unchanged.
    pub texture_asset_index: Option<usize>,
    /// First index into [`PlaytestPackage::model_clips`] for
    /// this model's clip slice.
    pub clip_first: u16,
    /// Number of clips in this model's slice. Matches the
    /// editor resource's clip count exactly.
    pub clip_count: u16,
    /// Default clip index *within this model's slice*.
    /// Cooker validation guarantees this is `< clip_count`,
    /// so the runtime always has a clip to play.
    pub default_clip: u16,
    /// First index in [`PlaytestPackage::model_sockets`].
    pub socket_first: u16,
    /// Number of sockets on this model.
    pub socket_count: u16,
    /// World-space height (engine units) -- propagated from the
    /// editor resource.
    pub world_height: u16,
}

/// One placed model instance. Coordinates are room-local
/// engine units (the same space cooked rooms live in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestModelInstance {
    /// Owning room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Model index in [`PlaytestPackage::models`].
    pub model: u16,
    /// Per-instance clip override, or [`MODEL_CLIP_INHERIT`]
    /// to use the model's `default_clip`.
    pub clip: u16,
    /// Room-local X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Room-local Z.
    pub z: i32,
    /// Yaw, PSX angle units.
    pub yaw: i16,
    /// Reserved.
    pub flags: u16,
}

/// Sentinel for [`PlaytestModelInstance::clip`] meaning
/// "inherit model default" -- same value as
/// [`psx_level::MODEL_CLIP_INHERIT`].
pub const MODEL_CLIP_INHERIT: u16 = 0xFFFF;

/// Weapon-local hit shape, ready for manifest emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaytestWeaponHitShape {
    /// Box hit volume.
    Box {
        /// Local center.
        center: [i32; 3],
        /// Half extents.
        half_extents: [u16; 3],
    },
    /// Capsule hit volume.
    Capsule {
        /// Local start.
        start: [i32; 3],
        /// Local end.
        end: [i32; 3],
        /// Radius.
        radius: u16,
    },
}

/// One weapon hitbox and active animation-frame window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestWeaponHitbox {
    /// Display name.
    pub name: String,
    /// Local shape.
    pub shape: PlaytestWeaponHitShape,
    /// First active frame.
    pub active_start_frame: u16,
    /// Last active frame.
    pub active_end_frame: u16,
}

/// Cooked weapon resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestWeapon {
    /// Display name.
    pub name: String,
    /// Source resource id.
    pub source_resource: ResourceId,
    /// Optional visual model index.
    pub model: Option<u16>,
    /// Character socket this weapon expects by default.
    pub default_character_socket: String,
    /// Weapon-local grip/pivot name.
    pub grip_name: String,
    /// Weapon-local grip translation.
    pub grip_translation: [i32; 3],
    /// Weapon-local grip rotation, Q12 turns.
    pub grip_rotation_q12: [i16; 3],
    /// First index in [`PlaytestPackage::weapon_hitboxes`].
    pub hitbox_first: u16,
    /// Number of hitboxes.
    pub hitbox_count: u16,
}

/// Cooked Equipment component on an Entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestEquipment {
    /// Owning room.
    pub room: u16,
    /// Weapon index.
    pub weapon: u16,
    /// Parent entity room-local X.
    pub x: i32,
    /// Parent entity room-local Y.
    pub y: i32,
    /// Parent entity room-local Z.
    pub z: i32,
    /// Parent entity yaw.
    pub yaw: i16,
    /// Character socket to follow.
    pub character_socket: String,
    /// Weapon grip/pivot to align.
    pub weapon_grip: String,
    /// Runtime flags. Bit 0 = follows the live player controller.
    pub flags: u16,
}

/// One placed point light, room-local engine units. Mirrors
/// [`psx_level::PointLightRecord`] one-for-one -- intensity is
/// already quantised to Q8.8 so the cook output is a direct
/// copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestLight {
    /// Room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Room-local X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Room-local Z.
    pub z: i32,
    /// Cutoff distance in engine units. Cooker rejects `0`.
    pub radius: u16,
    /// Brightness multiplier in Q8.8 (`256` = 1.0). Derived
    /// from the editor's `f32` intensity at cook time.
    pub intensity_q8: u16,
    /// 8-bit RGB tint.
    pub color: [u8; 3],
}

/// One baked static-light sample for a cooked room surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestSurfaceLight {
    /// Room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Sector X coordinate.
    pub sx: u16,
    /// Sector Z coordinate.
    pub sz: u16,
    /// One of [`psx_level::surface_light_kind`].
    pub kind: u8,
    /// Runtime wall direction id for walls, `0` otherwise.
    pub direction: u8,
    /// Surface ordinal inside the cooked sector.
    pub ordinal: u16,
    /// Baked RGB tint for the emitted quad's four vertices.
    pub vertex_rgb: [[u8; 3]; 4],
}

/// Player spawn record. Coordinates are room-local engine units
/// (the same space the cooked `.psxw` lives in -- array-rooted at
/// world `(0, 0)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestSpawn {
    /// Room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Room-local X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Room-local Z.
    pub z: i32,
    /// Yaw in PSX angle units.
    pub yaw: i16,
    /// Reserved flags. Bit 0 = "spawn enabled".
    pub flags: u16,
}

/// Coarse runtime kind for [`PlaytestEntity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaytestEntityKind {
    /// Visual marker (debug cube).
    Marker,
    /// Static mesh instance pinned by `resource_slot`.
    StaticMesh,
}

/// Cooked character record. Mirrors
/// [`psx_level::LevelCharacterRecord`] one-to-one -- the writer
/// emits the static slice from this struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestCharacter {
    /// Source resource id (so the writer can dedupe + cross-link).
    pub source_resource: ResourceId,
    /// Index into [`PlaytestPackage::models`].
    pub model: u16,
    /// Idle clip index *within the model's clip slice*.
    pub idle_clip: u16,
    /// Walk clip index within the model's clip slice.
    pub walk_clip: u16,
    /// Optional run clip; `u16::MAX` (= `CHARACTER_CLIP_NONE`)
    /// means "no run clip authored".
    pub run_clip: u16,
    /// Optional turn clip; same sentinel as `run_clip`.
    pub turn_clip: u16,
    /// Capsule radius in engine units.
    pub radius: u16,
    /// Capsule height in engine units.
    pub height: u16,
    /// Walk speed (engine units / 60 Hz frame).
    pub walk_speed: i32,
    /// Run speed (engine units / 60 Hz frame).
    pub run_speed: i32,
    /// Turn speed (degrees / second).
    pub turn_speed_degrees_per_second: u16,
    /// Camera follow distance (engine units).
    pub camera_distance: i32,
    /// Camera vertical offset above the character origin.
    pub camera_height: i32,
    /// Vertical offset of the camera's look-at target.
    pub camera_target_height: i32,
}

/// Cooked player-controller record. Always paired with a
/// [`PlaytestSpawn`] in the same package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestPlayerController {
    /// Resolved spawn -- same data the manifest's `PLAYER_SPAWN`
    /// carries.
    pub spawn: PlaytestSpawn,
    /// Character index in [`PlaytestPackage::characters`].
    pub character: u16,
}

/// Sentinel used in [`PlaytestCharacter::run_clip`] /
/// [`PlaytestCharacter::turn_clip`] when the role wasn't
/// authored. Mirrors [`psx_level::CHARACTER_CLIP_NONE`].
pub const CHARACTER_CLIP_NONE: u16 = u16::MAX;

/// One non-spawn entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestEntity {
    /// Owning room index.
    pub room: u16,
    /// Entity kind.
    pub kind: PlaytestEntityKind,
    /// Room-local X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Room-local Z.
    pub z: i32,
    /// Yaw, PSX angle units.
    pub yaw: i16,
    /// Resource slot (0 if unused).
    pub resource_slot: u16,
    /// Reserved flags.
    pub flags: u16,
}

/// Cooked playtest scene, ready to write to disk. Holds
/// everything the generated manifest needs to render: assets,
/// per-room metadata, per-room material slices, models, model
/// instances, and residency.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlaytestPackage {
    /// Master asset table -- rooms first, then room textures,
    /// then per-model assets (mesh + atlas + clips), in
    /// deterministic order.
    pub assets: Vec<PlaytestAsset>,
    /// Cooked rooms with material-slice metadata.
    pub rooms: Vec<PlaytestRoom>,
    /// Material records ordered as `(room, local_slot)`.
    pub materials: Vec<PlaytestMaterial>,
    /// Cooked model bundles, deduplicated across instances.
    pub models: Vec<PlaytestModel>,
    /// Per-model clip records ordered as `(model, clip_index)`.
    pub model_clips: Vec<PlaytestModelClip>,
    /// Per-model socket records ordered as `(model, socket_index)`.
    pub model_sockets: Vec<PlaytestModelSocket>,
    /// Placed model instances, room-local coordinates.
    pub model_instances: Vec<PlaytestModelInstance>,
    /// Weapon hitboxes, shared by [`Self::weapons`].
    pub weapon_hitboxes: Vec<PlaytestWeaponHitbox>,
    /// Cooked Weapon resources, deduplicated by source resource id.
    pub weapons: Vec<PlaytestWeapon>,
    /// Equipment components placed in rooms.
    pub equipment: Vec<PlaytestEquipment>,
    /// Placed point lights, room-local coordinates.
    pub lights: Vec<PlaytestLight>,
    /// Static per-vertex room lighting baked from [`Self::lights`].
    pub surface_lights: Vec<PlaytestSurfaceLight>,
    /// Single player spawn -- required.
    pub spawn: Option<PlaytestSpawn>,
    /// Cooked Character resources used by player / future
    /// gameplay. Currently only the player spawn references
    /// these, but the slice ships in the manifest unconditionally
    /// so the runtime can table-drive any future controllers.
    pub characters: Vec<PlaytestCharacter>,
    /// Resolved player controller -- `Some` when a player spawn
    /// was authored *and* a Character was assigned (or
    /// auto-picked). The runtime falls back to a debug camera
    /// when this is `None`.
    pub player_controller: Option<PlaytestPlayerController>,
    /// Optional entity markers (legacy, non-Model MeshInstance).
    pub entities: Vec<PlaytestEntity>,
}

impl PlaytestPackage {
    /// Number of `RoomWorld` entries in [`Self::assets`].
    pub fn room_asset_count(&self) -> usize {
        self.assets
            .iter()
            .filter(|a| a.kind == PlaytestAssetKind::RoomWorld)
            .count()
    }

    /// Number of `Texture` entries in [`Self::assets`].
    pub fn texture_asset_count(&self) -> usize {
        self.assets
            .iter()
            .filter(|a| a.kind == PlaytestAssetKind::Texture)
            .count()
    }

    /// Number of `ModelMesh` entries in [`Self::assets`].
    pub fn model_mesh_asset_count(&self) -> usize {
        self.assets
            .iter()
            .filter(|a| a.kind == PlaytestAssetKind::ModelMesh)
            .count()
    }

    /// Number of `ModelAnimation` entries in [`Self::assets`].
    pub fn model_animation_asset_count(&self) -> usize {
        self.assets
            .iter()
            .filter(|a| a.kind == PlaytestAssetKind::ModelAnimation)
            .count()
    }
}

/// Outcome of validating a project for playtest. Errors block
/// cooking; warnings are surfaced but not fatal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlaytestValidationReport {
    /// Hard errors. Embedded Play must refuse to launch when this
    /// list is non-empty.
    pub errors: Vec<String>,
    /// Soft warnings. Surface in the editor status line but
    /// don't block cooking.
    pub warnings: Vec<String>,
}

impl PlaytestValidationReport {
    /// `true` when there are zero hard errors.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub(super) fn error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    pub(super) fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}
