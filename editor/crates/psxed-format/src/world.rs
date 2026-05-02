//! Shared contract for cooked grid-world assets.
//!
//! Each `.psxw` blob is a fixed-size grid room: a world header, one
//! sector record per grid cell, then a compact wall table referenced
//! by sectors. Only `VERSION = 1` is emitted by the cooker and
//! accepted by the runtime parser -- every public record below is
//! part of that single live contract.
//!
//! # Active format (VERSION = 1)
//!
//! - `[i32; 4]` heights per face -- 16 B per height set
//! - 44 B sector record, 24 B wall record
//! - No embedded material table; slot ids resolve via an external
//!   bank that the caller (engine / playtest manifest) supplies
//! - No sector logic stream, no portal records
//!
//! # Future compact format
//!
//! A more compact runtime format is sketched in
//! `docs/world-format-roadmap.md`. It does not live in this crate
//! as Rust types: `psxed-format` is the producer/consumer contract,
//! and a record only belongs here once both the cooker emits it
//! and the runtime parser accepts it. Until then the design stays
//! in docs.

/// ASCII magic for the `.psxw` grid-world format.
pub const MAGIC: [u8; 4] = *b"PSXW";

/// Current (and only) world format revision.
pub const VERSION: u16 = 1;

/// Canonical/default engine units per grid sector.
pub const SECTOR_SIZE: i32 = 1024;

/// Material sentinel used by missing optional floor/ceiling records.
pub const NO_MATERIAL: u16 = u16::MAX;

/// North-west horizontal face corner index.
pub const CORNER_NW: usize = 0;

/// North-east horizontal face corner index.
pub const CORNER_NE: usize = 1;

/// South-east horizontal face corner index.
pub const CORNER_SE: usize = 2;

/// South-west horizontal face corner index.
pub const CORNER_SW: usize = 3;

/// Bottom-left vertical wall corner index.
pub const WALL_BOTTOM_LEFT: usize = 0;

/// Bottom-right vertical wall corner index.
pub const WALL_BOTTOM_RIGHT: usize = 1;

/// Top-right vertical wall corner index.
pub const WALL_TOP_RIGHT: usize = 2;

/// Top-left vertical wall corner index.
pub const WALL_TOP_LEFT: usize = 3;

/// Stored values for diagonal split directions.
pub mod split {
    /// Split from north-west to south-east.
    pub const NORTH_WEST_SOUTH_EAST: u8 = 0;

    /// Split from north-east to south-west.
    pub const NORTH_EAST_SOUTH_WEST: u8 = 1;
}

/// Stored values for sector wall directions.
pub mod direction {
    /// North edge, negative Z.
    pub const NORTH: u8 = 0;

    /// East edge, positive X.
    pub const EAST: u8 = 1;

    /// South edge, positive Z.
    pub const SOUTH: u8 = 2;

    /// West edge, negative X.
    pub const WEST: u8 = 3;

    /// Diagonal edge from north-west to south-east.
    pub const NORTH_WEST_SOUTH_EAST: u8 = 4;

    /// Diagonal edge from north-east to south-west.
    pub const NORTH_EAST_SOUTH_WEST: u8 = 5;
}

/// World-level feature flags stored in `AssetHeader::flags`.
pub mod flags {
    /// Reserved for future multi-room payloads.
    pub const RESERVED: u16 = 0;
}

/// World payload flags stored in [`WorldHeader::flags`].
pub mod world_flags {
    /// PS1 depth cue/fog is enabled for this grid.
    pub const FOG_ENABLED: u8 = 1 << 0;
}

/// Sector flags stored in [`SectorRecord::flags`].
pub mod sector_flags {
    /// Sector has a floor face.
    pub const HAS_FLOOR: u8 = 1 << 0;

    /// Sector has a ceiling face.
    pub const HAS_CEILING: u8 = 1 << 1;

    /// Floor face is walkable.
    pub const FLOOR_WALKABLE: u8 = 1 << 2;

    /// Ceiling face is walkable.
    pub const CEILING_WALKABLE: u8 = 1 << 3;
}

/// Wall flags stored in [`WallRecord::flags`].
pub mod wall_flags {
    /// Wall blocks collision.
    pub const SOLID: u8 = 1 << 0;
}

/// Payload header that follows the common `AssetHeader`.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct WorldHeader {
    /// Width in sectors.
    pub width: u16,
    /// Depth in sectors.
    pub depth: u16,
    /// Engine units per sector.
    pub sector_size: i32,
    /// Number of sector records following this header.
    pub sector_count: u16,
    /// Number of material slots referenced by faces.
    pub material_count: u16,
    /// Number of wall records after the sector table.
    pub wall_count: u16,
    /// Room ambient RGB color.
    pub ambient_color: [u8; 3],
    /// World payload flags, see [`world_flags`].
    pub flags: u8,
    /// Reserved. Writers store zero; readers ignore.
    pub _reserved: u16,
}

impl WorldHeader {
    /// Size of the world header in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}

/// Fixed sector record in flat `[x * depth + z]` order.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct SectorRecord {
    /// Sector flags, see [`sector_flags`].
    pub flags: u8,
    /// Floor split id, see [`split`].
    pub floor_split: u8,
    /// Ceiling split id, see [`split`].
    pub ceiling_split: u8,
    /// Reserved padding. Writers store zero; readers ignore.
    pub _pad: u8,
    /// Floor material slot or [`NO_MATERIAL`].
    pub floor_material: u16,
    /// Ceiling material slot or [`NO_MATERIAL`].
    pub ceiling_material: u16,
    /// First wall record index for this sector.
    pub first_wall: u16,
    /// Number of wall records belonging to this sector.
    pub wall_count: u16,
    /// Floor heights `[NW, NE, SE, SW]`.
    pub floor_heights: [i32; 4],
    /// Ceiling heights `[NW, NE, SE, SW]`.
    pub ceiling_heights: [i32; 4],
}

impl SectorRecord {
    /// Size of one sector record in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}

/// Variable wall record referenced by a sector.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct WallRecord {
    /// Wall direction id, see [`direction`].
    pub direction: u8,
    /// Wall flags, see [`wall_flags`].
    pub flags: u8,
    /// Reserved padding. Writers store zero; readers ignore.
    pub _pad: u16,
    /// Material slot.
    pub material: u16,
    /// Reserved. Writers store zero; readers ignore.
    pub _reserved: u16,
    /// Wall heights `[bottom-left, bottom-right, top-right, top-left]`.
    pub heights: [i32; 4],
}

impl WallRecord {
    /// Size of one wall record in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}
