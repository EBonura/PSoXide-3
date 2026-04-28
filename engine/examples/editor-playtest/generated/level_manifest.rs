// Placeholder manifest — committed so the example compiles
// before any cook has run. **Don't edit by hand**: the editor's
// "Cook & Play" action and the `cook-playtest` CLI both
// overwrite this file. Run `make run-starter-playtest` from a
// fresh checkout to cook the bundled starter project, or use
// the editor for a custom scene.
//
// The committed state has zero rooms / zero textures, so the
// EXE boots to a clear-coloured screen until you cook.

use psx_level::{
    AssetId,
    AssetKind,
    EntityKind,
    EntityRecord,
    LevelAssetRecord,
    LevelMaterialRecord,
    LevelRoomRecord,
    PlayerSpawnRecord,
    RoomResidencyRecord,
};

/// Master asset table.
pub static ASSETS: &[LevelAssetRecord] = &[];

/// Per-room material bindings.
pub static MATERIALS: &[LevelMaterialRecord] = &[];

/// Rooms with material-slice metadata.
pub static ROOMS: &[LevelRoomRecord] = &[];

/// Per-room residency contract.
pub static ROOM_RESIDENCY: &[RoomResidencyRecord] = &[];

/// Player spawn.
pub static PLAYER_SPAWN: PlayerSpawnRecord = PlayerSpawnRecord {
    room: 0,
    x: 0,
    y: 0,
    z: 0,
    yaw: 0,
    flags: 0,
};

/// Entity markers (debug cubes for now).
pub static ENTITIES: &[EntityRecord] = &[];

// Keep imports in scope so generated and placeholder tracks
// stay close to identical (helps diffs after a real cook).
const _: Option<AssetId> = None;
const _: Option<AssetKind> = None;
const _: Option<EntityKind> = None;
