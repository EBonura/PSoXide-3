// Placeholder manifest — committed so the example compiles
// before any cook has run. **Don't edit by hand**: the editor's
// "Cook & Play" action and the `cook-playtest` CLI both
// overwrite this file. Run `make run-starter-playtest` from a
// fresh checkout to cook the bundled starter project, or use
// the editor for a custom scene.
//
// The committed state has zero rooms / zero entities, so the
// EXE boots to a clear-coloured screen until you cook.

#[derive(Debug, Clone, Copy)]
pub struct PlaytestRoomRecord {
    pub name: &'static str,
    pub world_bytes: &'static [u8],
    pub origin_x: i32,
    pub origin_z: i32,
    pub sector_size: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct PlaytestSpawnRecord {
    pub room: u16,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub yaw: i16,
    pub flags: u16,
}

#[derive(Debug, Clone, Copy)]
pub enum PlaytestEntityKind {
    Marker,
    StaticMesh,
}

#[derive(Debug, Clone, Copy)]
pub struct PlaytestEntityRecord {
    pub room: u16,
    pub kind: PlaytestEntityKind,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub yaw: i16,
    pub resource_slot: u16,
    pub flags: u16,
}

pub static ROOMS: &[PlaytestRoomRecord] = &[];

pub static PLAYER_SPAWN: PlaytestSpawnRecord = PlaytestSpawnRecord {
    room: 0,
    x: 0,
    y: 0,
    z: 0,
    yaw: 0,
    flags: 0,
};

pub static ENTITIES: &[PlaytestEntityRecord] = &[];
