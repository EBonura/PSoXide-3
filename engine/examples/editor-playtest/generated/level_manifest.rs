// Placeholder checked into source control so the editor-playtest
// example can compile before the editor cooks a project. Runtime
// builds overwrite this file with cooked asset records.

use psx_level::{
    EntityRecord, EquipmentRecord, LevelAssetRecord, LevelCharacterRecord, LevelMaterialRecord,
    LevelModelClipRecord, LevelModelInstanceRecord, LevelModelRecord, LevelModelSocketRecord,
    LevelRoomRecord, LevelRoomVisibilityRecord, LevelVisibilityCellRecord,
    LevelVisibleCellRecord, LevelWeaponRecord, PlayerControllerRecord, PlayerSpawnRecord,
    PointLightRecord, RoomIndex, RoomResidencyRecord, WeaponHitboxRecord,
};

pub static ASSETS: &[LevelAssetRecord] = &[];
pub static MATERIALS: &[LevelMaterialRecord] = &[];
pub static ROOMS: &[LevelRoomRecord] = &[];
pub static ROOM_VISIBILITY: &[LevelRoomVisibilityRecord] = &[];
pub static VISIBILITY_CELLS: &[LevelVisibilityCellRecord] = &[];
pub static VISIBLE_CELLS: &[LevelVisibleCellRecord] = &[];
pub static ROOM_RESIDENCY: &[RoomResidencyRecord] = &[];

pub static PLAYER_SPAWN: PlayerSpawnRecord = PlayerSpawnRecord {
    room: RoomIndex(0),
    x: 0,
    y: 0,
    z: 0,
    yaw: 0,
    flags: 0,
};

pub static MODEL_CLIPS: &[LevelModelClipRecord] = &[];
pub static MODEL_SOCKETS: &[LevelModelSocketRecord] = &[];
pub static MODELS: &[LevelModelRecord] = &[];
pub static MODEL_INSTANCES: &[LevelModelInstanceRecord] = &[];
pub static WEAPON_HITBOXES: &[WeaponHitboxRecord] = &[];
pub static WEAPONS: &[LevelWeaponRecord] = &[];
pub static EQUIPMENT: &[EquipmentRecord] = &[];
pub static LIGHTS: &[PointLightRecord] = &[];
pub static CHARACTERS: &[LevelCharacterRecord] = &[];
pub static PLAYER_CONTROLLER: Option<PlayerControllerRecord> = None;
pub static ENTITIES: &[EntityRecord] = &[];
