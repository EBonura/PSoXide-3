// Tracked placeholder for clean-tree builds.
//
// The editor Play action and `make cook-playtest` replace this file
// in the working tree with a cooked manifest that points at ignored
// generated blobs under `rooms/`, `textures/`, and `models/`.

use psx_level::{
    EntityRecord, LevelAssetRecord, LevelCharacterRecord, LevelMaterialRecord,
    LevelModelClipRecord, LevelModelInstanceRecord, LevelModelRecord, LevelRoomRecord,
    PlayerControllerRecord, PlayerSpawnRecord, PointLightRecord, RoomIndex, RoomResidencyRecord,
};

/// Master asset table. Empty in the tracked placeholder.
pub static ASSETS: &[LevelAssetRecord] = &[];

/// Rooms with material-slice metadata. Empty in the tracked placeholder.
pub static ROOMS: &[LevelRoomRecord] = &[];

/// Per-room material bindings. Empty in the tracked placeholder.
pub static MATERIALS: &[LevelMaterialRecord] = &[];

/// Per-room residency contract. Empty in the tracked placeholder.
pub static ROOM_RESIDENCY: &[RoomResidencyRecord] = &[];

/// Cooked models. Empty in the tracked placeholder.
pub static MODELS: &[LevelModelRecord] = &[];

/// Per-model clip records. Empty in the tracked placeholder.
pub static MODEL_CLIPS: &[LevelModelClipRecord] = &[];

/// Placed model instances. Empty in the tracked placeholder.
pub static MODEL_INSTANCES: &[LevelModelInstanceRecord] = &[];

/// Placed point lights. Empty in the tracked placeholder.
pub static LIGHTS: &[PointLightRecord] = &[];

/// Cooked character resources. Empty in the tracked placeholder.
pub static CHARACTERS: &[LevelCharacterRecord] = &[];

/// Entity markers. Empty in the tracked placeholder.
pub static ENTITIES: &[EntityRecord] = &[];

/// Legacy spawn fallback for placeholder manifests.
pub static PLAYER_SPAWN: PlayerSpawnRecord = PlayerSpawnRecord {
    room: RoomIndex::ZERO,
    x: 0,
    y: 0,
    z: 0,
    yaw: 0,
    flags: 0,
};

/// Optional player controller. Empty in the tracked placeholder.
pub static PLAYER_CONTROLLER: Option<PlayerControllerRecord> = None;
