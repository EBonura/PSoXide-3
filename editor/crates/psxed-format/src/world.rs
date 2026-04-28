//! Shared contract for cooked grid-world assets.
//!
//! Each `.psxw` blob is a fixed-size grid room: a world header, one
//! sector record per grid cell, then a compact wall table referenced
//! by sectors.
//!
//! # Versions
//!
//! `VERSION = 1` (current shipping):
//!   - Heights are `[i32; 4]` per face → 16 B per height set
//!   - Sector record = 44 B, wall record = 24 B
//!   - No material table — slot ids resolve via an external bank
//!   - No sector logic stream
//!
//! `VERSION = 2` (designed, not yet wired through the cooker):
//!   - Heights are room-local `[i16; 4]` → 8 B per set
//!   - Sector record = **28 B**, wall record = **12 B** (a 36 % / 50 %
//!     reduction on the two records that dominate the wire size)
//!   - Embedded material table so a `.psxw` is self-resolving
//!   - `sector_logic_offset` referencing a sparse byte stream for
//!     triggers / slopes / portals (Tomb Raider's FloorData idea)
//!
//! See `WorldHeaderV2`, `SectorRecordV2`, `WallRecordV2`,
//! `MaterialRecordV2` below — each asserts its own size at compile
//! time so layout drift fails the build.

/// ASCII magic reserved for the future `.psxw` grid-world format.
pub const MAGIC: [u8; 4] = *b"PSXW";

/// Current world format revision (shipping).
pub const VERSION: u16 = 1;

/// Compact second-revision world format.
///
/// Not yet emitted by the cooker / consumed by the runtime parser
/// — this constant exists so `static_assert`-equivalent size
/// checks live alongside the v2 record types and a future cooker
/// version bump is a single-token change.
pub const VERSION_V2: u16 = 2;

/// Engine units per grid sector.
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

// ============================================================
// v2 records — compact, room-local, self-resolving
// ============================================================
//
// Designed but not yet wired through the cooker / runtime parser.
// Living next to v1 keeps the size deltas inspectable and lets
// the budget UI surface "v2 estimate" against the real v1 blob.

/// Sentinel for "no sector logic" in [`SectorRecordV2::logic_offset`].
pub const NO_SECTOR_LOGIC: u16 = u16::MAX;

/// Sentinel for "no portal target" in portal records.
pub const NO_ROOM: u16 = u16::MAX;

/// World payload header for v2.
///
/// Same 20 B footprint as v1, but the field meanings change:
/// `material_count` now counts records embedded in the v2 material
/// table instead of slots resolved against an external bank, and
/// `logic_stream_bytes` carries the FloorData-style sparse stream
/// length so the parser can bound-check `logic_offset` in sectors.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct WorldHeaderV2 {
    /// Width in sectors.
    pub width: u16,
    /// Depth in sectors.
    pub depth: u16,
    /// Engine units per sector.
    pub sector_size: i32,
    /// Number of [`MaterialRecordV2`] entries between this header
    /// and the sector table.
    pub material_count: u16,
    /// Number of [`SectorRecordV2`] entries.
    pub sector_count: u16,
    /// Number of [`WallRecordV2`] entries after the sector table.
    pub wall_count: u16,
    /// Bytes of `SectorLogicStream` data after the wall table.
    pub logic_stream_bytes: u16,
    /// Room ambient RGB color.
    pub ambient_color: [u8; 3],
    /// World payload flags, see [`world_flags`].
    pub flags: u8,
}

impl WorldHeaderV2 {
    /// Size of the v2 world header in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}
const _: () = assert!(WorldHeaderV2::SIZE == 20);

/// One material entry embedded in the v2 world.
///
/// Carries everything `psx_gpu::material::TextureMaterial` needs at
/// runtime so a `.psxw` is self-resolving — no external material
/// bank lookup. Tuple is laid out for natural alignment so the
/// packed attribute doesn't actually rearrange anything.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct MaterialRecordV2 {
    /// Tpage word for the texture page (PSX-GP0 packed form).
    pub tpage_word: u16,
    /// CLUT word for the palette.
    pub clut_word: u16,
    /// Texture modulation tint. `0x80` is neutral.
    pub tint: [u8; 3],
    /// PSX semi-transparency mode. `0` = opaque, `1..=4` = blends.
    pub blend_mode: u8,
    /// See [`material_flags`].
    pub flags: u8,
    /// Reserved padding. Writers store zero; readers ignore. Lifts
    /// the record to 12 B so the embedded table is naturally aligned.
    pub _pad: [u8; 3],
}

impl MaterialRecordV2 {
    /// Size of one material record in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}
const _: () = assert!(MaterialRecordV2::SIZE == 12);

/// Material-record flags stored in [`MaterialRecordV2::flags`].
pub mod material_flags {
    /// Pass-through texel; skip tint modulate.
    pub const RAW_TEXTURE: u8 = 1 << 0;
    /// Render double-sided (no backface cull).
    pub const DOUBLE_SIDED: u8 = 1 << 1;
}

/// Compact sector record for the v2 layout.
///
/// 28 B vs v1's 44 B — heights drop to room-local `i16`, the
/// `flags` field widens to `u16` for per-sector marker headroom,
/// and the two split bytes collapse into a single `split_bits`
/// bitfield so `logic_offset` can be a real `u16` index into the
/// FloorData-style logic stream rather than the previous
/// "kind tag" placeholder.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct SectorRecordV2 {
    /// See [`sector_flags`]. Widened from `u8` for headroom.
    pub flags: u16,
    /// Floor material slot or [`NO_MATERIAL`].
    pub floor_material: u16,
    /// Ceiling material slot or [`NO_MATERIAL`].
    pub ceiling_material: u16,
    /// First wall record index for this sector.
    pub first_wall: u16,
    /// Byte offset into the world's logic stream, or
    /// [`NO_SECTOR_LOGIC`] when this sector has no triggers /
    /// slopes / portals attached. The runtime parser bound-
    /// checks this against `WorldHeaderV2::logic_stream_bytes`.
    pub logic_offset: u16,
    /// Number of wall records belonging to this sector.
    /// `u8` because [`MAX_WALL_STACK`] caps any one edge low and
    /// the per-sector total stays small enough for a byte.
    pub wall_count: u8,
    /// Packed floor + ceiling diagonal split bits, see
    /// [`split_bits`]. Two booleans share one byte so that
    /// expanding `logic_offset_kind` to `logic_offset: u16`
    /// fits inside the same 28 B record.
    pub split_bits: u8,
    /// Floor heights `[NW, NE, SE, SW]`, room-local `i16`.
    pub floor_heights: [i16; 4],
    /// Ceiling heights `[NW, NE, SE, SW]`, room-local `i16`.
    pub ceiling_heights: [i16; 4],
}

impl SectorRecordV2 {
    /// Size of one v2 sector record in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}
const _: () = assert!(SectorRecordV2::SIZE == 28);

/// Bit positions for [`SectorRecordV2::split_bits`]. A clear bit
/// means the surface uses the default NW-SE diagonal; a set bit
/// flips the split to NE-SW.
pub mod split_bits {
    /// Floor surface split is NE-SW (else NW-SE).
    pub const FLOOR_NE_SW: u8 = 1 << 0;
    /// Ceiling surface split is NE-SW (else NW-SE).
    pub const CEILING_NE_SW: u8 = 1 << 1;
}

/// Compact wall record for the v2 layout.
///
/// 12 B vs v1's 24 B — heights drop to room-local `i16`, two
/// reserved padding fields are gone.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct WallRecordV2 {
    /// Wall direction id, see [`direction`].
    pub direction: u8,
    /// Wall flags, see [`wall_flags`].
    pub flags: u8,
    /// Material slot index into [`MaterialRecordV2`] table, or
    /// [`NO_MATERIAL`].
    pub material: u16,
    /// Wall heights `[bottom-left, bottom-right, top-right, top-left]`,
    /// room-local `i16`.
    pub heights: [i16; 4],
}

impl WallRecordV2 {
    /// Size of one v2 wall record in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}
const _: () = assert!(WallRecordV2::SIZE == 12);

// ============================================================
// Portal records — Tomb Raider style
// ============================================================
//
// Two distinct concepts:
//   * `RoomPortalRecord` — visibility / culling between rooms.
//   * `SectorPortalRecord` — collision / traversal a single
//     sector exposes (wall room, pit room, sky room).

/// Visibility portal between two rooms.
///
/// Quad in world space whose `target_room` becomes a candidate
/// for rendering when the portal lies in front of the camera.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct RoomPortalRecord {
    /// Target room id, or [`NO_ROOM`] if the portal is unwired.
    pub target_room: u16,
    /// Reserved padding for alignment.
    pub _pad: u16,
    /// Outward normal in room-local i16 coords.
    pub normal: [i16; 3],
    /// Four corner vertices in winding order.
    pub vertices: [[i16; 3]; 4],
}

impl RoomPortalRecord {
    /// Size of one room portal record in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}

/// Per-sector traversal portals — wall (horizontal escape),
/// pit (down through floor), sky (up through ceiling). Mirrors
/// Tomb Raider's `room_below` / `room_above` / wall-portal
/// triple stashed in `tr_room_sector`.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct SectorPortalRecord {
    /// Room you walk into when exiting through a wall portal,
    /// or [`NO_ROOM`].
    pub wall_room: u16,
    /// Room you fall into through the floor, or [`NO_ROOM`].
    pub pit_room: u16,
    /// Room you rise into through the ceiling, or [`NO_ROOM`].
    pub sky_room: u16,
    /// Reserved padding so the record aligns to 8 B.
    pub _pad: u16,
}

impl SectorPortalRecord {
    /// Size of one sector portal record in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();
}
const _: () = assert!(SectorPortalRecord::SIZE == 8);
