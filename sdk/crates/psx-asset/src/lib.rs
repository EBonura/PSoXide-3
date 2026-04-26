//! Runtime parsers for PSoXide cooked-asset blobs.
//!
//! Pairs with `editor/crates/psxed`, the host-side tool that
//! produces these files. Format structs live in `psxed-format`
//! (shared by both sides so drift is impossible).
//!
//! Usage pattern:
//!
//! ```ignore
//! // At compile time, embed the cooked blob into the MIPS binary.
//! static TEAPOT: &[u8] = include_bytes!("assets/teapot.psxm");
//!
//! // At runtime, parse into a typed view. Zero-copy — the view
//! // just borrows slices into the original byte stream.
//! let mesh = psx_asset::Mesh::from_bytes(TEAPOT).expect("cooked mesh");
//! for tri_idx in 0..mesh.face_count() {
//!     let (ia, ib, ic) = mesh.face(tri_idx);
//!     let v0 = mesh.vertex(ia);
//!     let v1 = mesh.vertex(ib);
//!     let v2 = mesh.vertex(ic);
//!     let (r, g, b) = mesh.face_color(tri_idx).unwrap_or((128, 128, 128));
//!     // project + draw …
//! }
//! ```
//!
//! Design:
//!
//! - **Zero-copy**: `Mesh::from_bytes` borrows into the caller's
//!   byte slice. No allocation, no memcpy; the static blob stays
//!   where it was embedded.
//! - **`no_std`-clean**: all parsing is manual LE decode, no
//!   `std::io`.
//! - **Bounds-checked**: every accessor validates its index
//!   against the counts in the header. Malformed blobs produce
//!   `None` rather than panics at integer level, so consumers
//!   see errors at load time, not in the render loop.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use psx_gte::math::Vec3I16;

/// Errors `Mesh::from_bytes` can return for a malformed blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// Blob shorter than the minimum header sizes.
    Truncated,
    /// `AssetHeader::magic` doesn't match the expected asset kind.
    WrongMagic,
    /// Format version newer than this parser supports.
    UnsupportedVersion(u16),
    /// Declared payload_len disagrees with the actual byte length.
    InvalidPayloadLen {
        /// Payload length declared in the common asset header.
        declared: u32,
        /// Actual payload byte count after the common asset header.
        actual: usize,
    },
    /// Vertex or face table wouldn't fit in the remaining payload.
    TableOverflow,
    /// World-grid header fields are inconsistent.
    InvalidWorldLayout,
}

/// A parsed 3D mesh backed by slices into the caller's cooked blob.
///
/// Cheap to construct (just bounds-checks the header + computes
/// sub-slice offsets). Cheap to pass around — four `&[u8]` slices
/// + a flags `u16` + the counts.
#[derive(Copy, Clone, Debug)]
pub struct Mesh<'a> {
    verts: &'a [u8],
    indices: &'a [u8],
    face_colors: Option<&'a [u8]>,
    vertex_normals: Option<&'a [u8]>,
    vert_count: u16,
    face_count: u16,
    flags: u16,
}

impl<'a> Mesh<'a> {
    /// Parse a cooked `.psxm` blob. Returns a `Mesh` view that
    /// borrows into `bytes`.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        // AssetHeader.
        if bytes.len() < psxed_format::AssetHeader::SIZE {
            return Err(ParseError::Truncated);
        }
        let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != psxed_format::mesh::MAGIC {
            return Err(ParseError::WrongMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != psxed_format::mesh::VERSION {
            return Err(ParseError::UnsupportedVersion(version));
        }
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let payload_start = psxed_format::AssetHeader::SIZE;
        let actual_payload = bytes.len().saturating_sub(payload_start);
        if (payload_len as usize) != actual_payload {
            return Err(ParseError::InvalidPayloadLen {
                declared: payload_len,
                actual: actual_payload,
            });
        }

        // MeshHeader.
        if actual_payload < psxed_format::mesh::MeshHeader::SIZE {
            return Err(ParseError::Truncated);
        }
        let mh = &bytes[payload_start..];
        let vert_count = u16::from_le_bytes([mh[0], mh[1]]);
        let face_count = u16::from_le_bytes([mh[2], mh[3]]);
        // Skip _reserved (4 bytes).

        // Slice the vertex + index + optional colour tables out.
        let mut off = payload_start + psxed_format::mesh::MeshHeader::SIZE;
        let vert_bytes = (vert_count as usize) * 6;
        if off + vert_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let verts = &bytes[off..off + vert_bytes];
        off += vert_bytes;

        let index_bytes = (face_count as usize) * 3;
        if off + index_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let indices = &bytes[off..off + index_bytes];
        off += index_bytes;

        let face_colors = if flags & psxed_format::mesh::flags::HAS_FACE_COLORS != 0 {
            let color_bytes = (face_count as usize) * 3;
            if off + color_bytes > bytes.len() {
                return Err(ParseError::TableOverflow);
            }
            let slice = &bytes[off..off + color_bytes];
            off += color_bytes;
            Some(slice)
        } else {
            None
        };

        let vertex_normals = if flags & psxed_format::mesh::flags::HAS_NORMALS != 0 {
            let normal_bytes = (vert_count as usize) * 6;
            if off + normal_bytes > bytes.len() {
                return Err(ParseError::TableOverflow);
            }
            let slice = &bytes[off..off + normal_bytes];
            Some(slice)
        } else {
            None
        };

        Ok(Self {
            verts,
            indices,
            face_colors,
            vertex_normals,
            vert_count,
            face_count,
            flags,
        })
    }

    /// Vertex count.
    #[inline]
    pub fn vert_count(&self) -> u16 {
        self.vert_count
    }

    /// Triangle count.
    #[inline]
    pub fn face_count(&self) -> u16 {
        self.face_count
    }

    /// Mesh feature flags (see [`psxed_format::mesh::flags`]).
    #[inline]
    pub fn flags(&self) -> u16 {
        self.flags
    }

    /// Does the blob carry a face-colour table?
    #[inline]
    pub fn has_face_colors(&self) -> bool {
        self.flags & psxed_format::mesh::flags::HAS_FACE_COLORS != 0
    }

    /// Does the blob carry per-vertex normals? Required for any
    /// GTE- or CPU-lit rendering path.
    #[inline]
    pub fn has_normals(&self) -> bool {
        self.flags & psxed_format::mesh::flags::HAS_NORMALS != 0
    }

    /// Decode vertex `i` as a Q3.12 [`Vec3I16`]. Returns
    /// [`Vec3I16::ZERO`] if the index is out of range — keeps
    /// the render path branch-free, callers who care can
    /// check against [`Self::vert_count`] first.
    #[inline]
    pub fn vertex(&self, i: u8) -> Vec3I16 {
        let idx = i as usize;
        if idx >= self.vert_count as usize {
            return Vec3I16::ZERO;
        }
        let base = idx * 6;
        let x = i16::from_le_bytes([self.verts[base], self.verts[base + 1]]);
        let y = i16::from_le_bytes([self.verts[base + 2], self.verts[base + 3]]);
        let z = i16::from_le_bytes([self.verts[base + 4], self.verts[base + 5]]);
        Vec3I16::new(x, y, z)
    }

    /// Triangle `i`'s three vertex indices. Returns `(0, 0, 0)`
    /// for an out-of-range index.
    #[inline]
    pub fn face(&self, i: u16) -> (u8, u8, u8) {
        let idx = i as usize;
        if idx >= self.face_count as usize {
            return (0, 0, 0);
        }
        let base = idx * 3;
        (
            self.indices[base],
            self.indices[base + 1],
            self.indices[base + 2],
        )
    }

    /// Triangle `i`'s flat colour, or `None` if the blob doesn't
    /// carry a face-colour table.
    #[inline]
    pub fn face_color(&self, i: u16) -> Option<(u8, u8, u8)> {
        let colors = self.face_colors?;
        let idx = i as usize;
        if idx >= self.face_count as usize {
            return None;
        }
        let base = idx * 3;
        Some((colors[base], colors[base + 1], colors[base + 2]))
    }

    /// Per-vertex Q3.12 normal. Returns `None` if the blob lacks
    /// a normal table (`HAS_NORMALS` flag clear) or `i` is out of
    /// range. Components are unit-length-ish (Q3.12 quantisation
    /// introduces sub-ULP error but the GTE lighting path is
    /// tolerant).
    #[inline]
    pub fn vertex_normal(&self, i: u8) -> Option<Vec3I16> {
        let normals = self.vertex_normals?;
        let idx = i as usize;
        if idx >= self.vert_count as usize {
            return None;
        }
        let base = idx * 6;
        let x = i16::from_le_bytes([normals[base], normals[base + 1]]);
        let y = i16::from_le_bytes([normals[base + 2], normals[base + 3]]);
        let z = i16::from_le_bytes([normals[base + 4], normals[base + 5]]);
        Some(Vec3I16::new(x, y, z))
    }
}

/// A parsed 2D texture backed by slices into the caller's cooked
/// blob. Pixel data is already packed into the halfword-nibble
/// layout the PSX GPU reads; the CLUT (if any) is a slice of
/// RGB555 halfwords.
///
/// Construct by `Texture::from_bytes(&blob)`; upload to VRAM via
/// [`Texture::upload`], which returns the matching `Tpage` + `Clut`
/// handles ready to feed into primitive constructors.
#[derive(Copy, Clone, Debug)]
pub struct Texture<'a> {
    /// Packed pixel halfwords — 4 texels per u16 at 4bpp,
    /// 2 at 8bpp, 1 Color555 at 15bpp.
    pixel_data: &'a [u8],
    /// CLUT halfwords, or empty for 15bpp.
    clut_data: &'a [u8],
    width_px: u16,
    height_px: u16,
    depth: psxed_format::texture::Depth,
    clut_entries: u16,
}

impl<'a> Texture<'a> {
    /// Parse a cooked `.psxt` blob. Returns a `Texture` view that
    /// borrows into `bytes`. Cheap — header decode + two slice
    /// computations.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        use psxed_format::texture::{Depth, TextureHeader, MAGIC, VERSION};

        // AssetHeader.
        if bytes.len() < psxed_format::AssetHeader::SIZE {
            return Err(ParseError::Truncated);
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(ParseError::WrongMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(ParseError::UnsupportedVersion(version));
        }
        let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let payload_start = psxed_format::AssetHeader::SIZE;
        let actual_payload = bytes.len().saturating_sub(payload_start);
        if (payload_len as usize) != actual_payload {
            return Err(ParseError::InvalidPayloadLen {
                declared: payload_len,
                actual: actual_payload,
            });
        }

        // TextureHeader.
        if actual_payload < TextureHeader::SIZE {
            return Err(ParseError::Truncated);
        }
        let th = &bytes[payload_start..];
        let depth = Depth::from_byte(th[0]).ok_or(ParseError::TableOverflow)?;
        // th[1] is _pad; skip.
        let width_px = u16::from_le_bytes([th[2], th[3]]);
        let height_px = u16::from_le_bytes([th[4], th[5]]);
        let clut_entries = u16::from_le_bytes([th[6], th[7]]);
        let pixel_bytes = u32::from_le_bytes([th[8], th[9], th[10], th[11]]) as usize;
        let clut_bytes = u32::from_le_bytes([th[12], th[13], th[14], th[15]]) as usize;

        let mut off = payload_start + TextureHeader::SIZE;
        if off + pixel_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let pixel_data = &bytes[off..off + pixel_bytes];
        off += pixel_bytes;

        if off + clut_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let clut_data = &bytes[off..off + clut_bytes];

        Ok(Self {
            pixel_data,
            clut_data,
            width_px,
            height_px,
            depth,
            clut_entries,
        })
    }

    /// Width in texels.
    #[inline]
    pub fn width(&self) -> u16 {
        self.width_px
    }

    /// Height in texels.
    #[inline]
    pub fn height(&self) -> u16 {
        self.height_px
    }

    /// Colour depth.
    #[inline]
    pub fn depth(&self) -> psxed_format::texture::Depth {
        self.depth
    }

    /// Number of CLUT entries (16, 256, or 0 for 15bpp).
    #[inline]
    pub fn clut_entries(&self) -> u16 {
        self.clut_entries
    }

    /// Raw packed pixel halfwords, as bytes. Suitable for
    /// halfword-level DMA upload; caller pairs this with a `VramRect`
    /// describing the *halfword footprint*, not the texel width.
    #[inline]
    pub fn pixel_bytes(&self) -> &'a [u8] {
        self.pixel_data
    }

    /// Raw CLUT halfwords, as bytes. Empty for 15bpp.
    #[inline]
    pub fn clut_bytes(&self) -> &'a [u8] {
        self.clut_data
    }

    /// Halfwords-per-row at this texture's depth (rows padded up to
    /// a full halfword). Needed when computing the VRAM rect for
    /// upload: VRAM measures in halfwords regardless of the texel
    /// depth the GPU will fetch them at.
    #[inline]
    pub fn halfwords_per_row(&self) -> u16 {
        psxed_format::texture::TextureHeader::halfwords_per_row(self.depth, self.width_px)
    }
}

/// A parsed `.psxw` grid-world backed by slices into the cooked blob.
///
/// This is the runtime view of the binary format, not the higher-level engine
/// `psx_engine::GridWorld` model. It keeps parsing zero-copy and lets engine
/// code pull sectors and walls by value as needed.
#[derive(Copy, Clone, Debug)]
pub struct World<'a> {
    sectors: &'a [u8],
    walls: &'a [u8],
    width: u16,
    depth: u16,
    sector_size: i32,
    material_count: u16,
    wall_count: u16,
    ambient_color: [u8; 3],
    flags: u8,
}

impl<'a> World<'a> {
    /// Parse a cooked `.psxw` blob.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        use psxed_format::world::{WorldHeader, MAGIC, VERSION};

        if bytes.len() < psxed_format::AssetHeader::SIZE {
            return Err(ParseError::Truncated);
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(ParseError::WrongMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(ParseError::UnsupportedVersion(version));
        }
        let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let payload_start = psxed_format::AssetHeader::SIZE;
        let actual_payload = bytes.len().saturating_sub(payload_start);
        if payload_len as usize != actual_payload {
            return Err(ParseError::InvalidPayloadLen {
                declared: payload_len,
                actual: actual_payload,
            });
        }
        if actual_payload < WorldHeader::SIZE {
            return Err(ParseError::Truncated);
        }

        let wh = &bytes[payload_start..payload_start + WorldHeader::SIZE];
        let width = read_u16(wh, 0);
        let depth = read_u16(wh, 2);
        let sector_size = read_i32(wh, 4);
        let sector_count = read_u16(wh, 8);
        let material_count = read_u16(wh, 10);
        let wall_count = read_u16(wh, 12);
        let ambient_color = [wh[14], wh[15], wh[16]];
        let flags = wh[17];

        let expected_sectors = (width as usize)
            .checked_mul(depth as usize)
            .ok_or(ParseError::InvalidWorldLayout)?;
        if sector_count as usize != expected_sectors {
            return Err(ParseError::InvalidWorldLayout);
        }

        let mut off = payload_start + WorldHeader::SIZE;
        let sector_bytes = (sector_count as usize)
            .checked_mul(psxed_format::world::SectorRecord::SIZE)
            .ok_or(ParseError::TableOverflow)?;
        if off + sector_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let sectors = &bytes[off..off + sector_bytes];
        off += sector_bytes;

        let wall_bytes = (wall_count as usize)
            .checked_mul(psxed_format::world::WallRecord::SIZE)
            .ok_or(ParseError::TableOverflow)?;
        if off + wall_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let walls = &bytes[off..off + wall_bytes];
        off += wall_bytes;
        if off != bytes.len() {
            return Err(ParseError::InvalidWorldLayout);
        }

        validate_sector_wall_ranges(sectors, wall_count)?;

        Ok(Self {
            sectors,
            walls,
            width,
            depth,
            sector_size,
            material_count,
            wall_count,
            ambient_color,
            flags,
        })
    }

    /// Width in grid sectors.
    #[inline]
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Depth in grid sectors.
    #[inline]
    pub fn depth(&self) -> u16 {
        self.depth
    }

    /// Engine units per sector.
    #[inline]
    pub fn sector_size(&self) -> i32 {
        self.sector_size
    }

    /// Number of material slots referenced by the world.
    #[inline]
    pub fn material_count(&self) -> u16 {
        self.material_count
    }

    /// Number of wall records in the world.
    #[inline]
    pub fn wall_count(&self) -> u16 {
        self.wall_count
    }

    /// Ambient RGB color.
    #[inline]
    pub fn ambient_color(&self) -> [u8; 3] {
        self.ambient_color
    }

    /// Whether fog/depth cue is enabled for this world.
    #[inline]
    pub fn fog_enabled(&self) -> bool {
        self.flags & psxed_format::world::world_flags::FOG_ENABLED != 0
    }

    /// Sector at a coordinate, returning `None` for empty cells or out of range.
    pub fn sector(&self, x: u16, z: u16) -> Option<WorldSector> {
        if x >= self.width || z >= self.depth {
            return None;
        }
        let index = x as usize * self.depth as usize + z as usize;
        let sector = self.sector_record(index)?;
        if sector.has_geometry() {
            Some(sector)
        } else {
            None
        }
    }

    /// Sector record by flat `[x * depth + z]` index, including empty cells.
    pub fn sector_record(&self, index: usize) -> Option<WorldSector> {
        let size = psxed_format::world::SectorRecord::SIZE;
        let base = index.checked_mul(size)?;
        let end = base.checked_add(size)?;
        let bytes = self.sectors.get(base..end)?;
        Some(WorldSector::decode(bytes))
    }

    /// Wall record by global wall index.
    pub fn wall(&self, index: u16) -> Option<WorldWall> {
        if index >= self.wall_count {
            return None;
        }
        let size = psxed_format::world::WallRecord::SIZE;
        let base = index as usize * size;
        let end = base.checked_add(size)?;
        let bytes = self.walls.get(base..end)?;
        Some(WorldWall::decode(bytes))
    }

    /// Wall record by sector-local wall index.
    pub fn sector_wall(&self, sector: WorldSector, local_index: u16) -> Option<WorldWall> {
        if local_index >= sector.wall_count {
            return None;
        }
        self.wall(sector.first_wall.checked_add(local_index)?)
    }
}

/// One decoded world sector record.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldSector {
    flags: u8,
    floor_split: u8,
    ceiling_split: u8,
    floor_material: u16,
    ceiling_material: u16,
    first_wall: u16,
    wall_count: u16,
    floor_heights: [i32; 4],
    ceiling_heights: [i32; 4],
}

impl WorldSector {
    fn decode(bytes: &[u8]) -> Self {
        Self {
            flags: bytes[0],
            floor_split: bytes[1],
            ceiling_split: bytes[2],
            floor_material: read_u16(bytes, 4),
            ceiling_material: read_u16(bytes, 6),
            first_wall: read_u16(bytes, 8),
            wall_count: read_u16(bytes, 10),
            floor_heights: read_i32x4(bytes, 12),
            ceiling_heights: read_i32x4(bytes, 28),
        }
    }

    /// True if this sector has any floor, ceiling, or wall geometry.
    #[inline]
    pub fn has_geometry(&self) -> bool {
        self.has_floor() || self.has_ceiling() || self.wall_count != 0
    }

    /// True if this sector has a floor face.
    #[inline]
    pub fn has_floor(&self) -> bool {
        self.flags & psxed_format::world::sector_flags::HAS_FLOOR != 0
    }

    /// True if this sector has a ceiling face.
    #[inline]
    pub fn has_ceiling(&self) -> bool {
        self.flags & psxed_format::world::sector_flags::HAS_CEILING != 0
    }

    /// True if the floor face is walkable.
    #[inline]
    pub fn floor_walkable(&self) -> bool {
        self.flags & psxed_format::world::sector_flags::FLOOR_WALKABLE != 0
    }

    /// Floor diagonal split id.
    #[inline]
    pub fn floor_split(&self) -> u8 {
        self.floor_split
    }

    /// Floor material slot.
    #[inline]
    pub fn floor_material(&self) -> Option<u16> {
        material_or_none(self.floor_material)
    }

    /// Floor corner heights `[NW, NE, SE, SW]`.
    #[inline]
    pub fn floor_heights(&self) -> [i32; 4] {
        self.floor_heights
    }

    /// Ceiling diagonal split id.
    #[inline]
    pub fn ceiling_split(&self) -> u8 {
        self.ceiling_split
    }

    /// Ceiling material slot.
    #[inline]
    pub fn ceiling_material(&self) -> Option<u16> {
        material_or_none(self.ceiling_material)
    }

    /// Ceiling corner heights `[NW, NE, SE, SW]`.
    #[inline]
    pub fn ceiling_heights(&self) -> [i32; 4] {
        self.ceiling_heights
    }

    /// First global wall index for this sector.
    #[inline]
    pub fn first_wall(&self) -> u16 {
        self.first_wall
    }

    /// Number of walls belonging to this sector.
    #[inline]
    pub fn wall_count(&self) -> u16 {
        self.wall_count
    }
}

/// One decoded world wall record.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldWall {
    direction: u8,
    flags: u8,
    material: u16,
    heights: [i32; 4],
}

impl WorldWall {
    fn decode(bytes: &[u8]) -> Self {
        Self {
            direction: bytes[0],
            flags: bytes[1],
            material: read_u16(bytes, 4),
            heights: read_i32x4(bytes, 8),
        }
    }

    /// Direction id, see `psxed_format::world::direction`.
    #[inline]
    pub fn direction(&self) -> u8 {
        self.direction
    }

    /// Whether this wall blocks collision.
    #[inline]
    pub fn solid(&self) -> bool {
        self.flags & psxed_format::world::wall_flags::SOLID != 0
    }

    /// Material slot.
    #[inline]
    pub fn material(&self) -> u16 {
        self.material
    }

    /// Wall heights `[bottom-left, bottom-right, top-right, top-left]`.
    #[inline]
    pub fn heights(&self) -> [i32; 4] {
        self.heights
    }
}

#[inline]
fn material_or_none(material: u16) -> Option<u16> {
    if material == psxed_format::world::NO_MATERIAL {
        None
    } else {
        Some(material)
    }
}

#[inline]
fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

#[inline]
fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[inline]
fn read_i32x4(bytes: &[u8], offset: usize) -> [i32; 4] {
    [
        read_i32(bytes, offset),
        read_i32(bytes, offset + 4),
        read_i32(bytes, offset + 8),
        read_i32(bytes, offset + 12),
    ]
}

fn validate_sector_wall_ranges(sectors: &[u8], wall_count: u16) -> Result<(), ParseError> {
    let size = psxed_format::world::SectorRecord::SIZE;
    let count = sectors.len() / size;
    for index in 0..count {
        let base = index * size;
        let first_wall = read_u16(sectors, base + 8);
        let sector_wall_count = read_u16(sectors, base + 10);
        let Some(end) = first_wall.checked_add(sector_wall_count) else {
            return Err(ParseError::InvalidWorldLayout);
        };
        if end > wall_count {
            return Err(ParseError::InvalidWorldLayout);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_magic() {
        let bad = [b'N', b'O', b'P', b'E', 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(matches!(
            Mesh::from_bytes(&bad),
            Err(ParseError::WrongMagic)
        ));
    }

    #[test]
    fn rejects_truncated() {
        let too_short = [0u8; 4];
        assert!(matches!(
            Mesh::from_bytes(&too_short),
            Err(ParseError::Truncated)
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bad = [0u8; 12];
        bad[0..4].copy_from_slice(&psxed_format::mesh::MAGIC);
        bad[4..6].copy_from_slice(&999u16.to_le_bytes());
        assert!(matches!(
            Mesh::from_bytes(&bad),
            Err(ParseError::UnsupportedVersion(999))
        ));
    }

    #[test]
    fn texture_round_trip_4bpp() {
        use psxed_format::texture::Depth;
        // 12 AssetHeader + 16 TextureHeader + 8 pixels + 32 CLUT = 68 bytes.
        let pixel_bytes: u32 = 8;
        let clut_bytes: u32 = 32;
        let payload_len = 16 + pixel_bytes + clut_bytes;
        let mut buf = [0u8; 68];
        buf[0..4].copy_from_slice(b"PSXT");
        buf[4..6].copy_from_slice(&1u16.to_le_bytes());
        buf[6..8].copy_from_slice(&0u16.to_le_bytes());
        buf[8..12].copy_from_slice(&payload_len.to_le_bytes());
        // TextureHeader @ offset 12.
        buf[12] = 4; // depth
        buf[13] = 0;
        buf[14..16].copy_from_slice(&4u16.to_le_bytes()); // width
        buf[16..18].copy_from_slice(&4u16.to_le_bytes()); // height
        buf[18..20].copy_from_slice(&16u16.to_le_bytes()); // clut_entries
        buf[20..24].copy_from_slice(&pixel_bytes.to_le_bytes());
        buf[24..28].copy_from_slice(&clut_bytes.to_le_bytes());
        // 4 rows × 1 halfword = 4 halfwords = 8 bytes @ offset 28.
        for row in 0..4u16 {
            let off = 28 + (row as usize) * 2;
            buf[off..off + 2].copy_from_slice(&(row * 0x1111).to_le_bytes());
        }
        // 16 CLUT entries @ offset 36.
        for i in 0..16u16 {
            let off = 36 + (i as usize) * 2;
            buf[off..off + 2].copy_from_slice(&(i * 0x0123).to_le_bytes());
        }

        let t = Texture::from_bytes(&buf).expect("parse");
        assert_eq!(t.width(), 4);
        assert_eq!(t.height(), 4);
        assert_eq!(t.depth(), Depth::Bit4);
        assert_eq!(t.clut_entries(), 16);
        assert_eq!(t.halfwords_per_row(), 1);
        assert_eq!(t.pixel_bytes().len(), 8);
        assert_eq!(t.clut_bytes().len(), 32);
    }

    #[test]
    fn texture_rejects_wrong_magic() {
        let bad = [b'N', b'O', b'P', b'E', 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(matches!(
            Texture::from_bytes(&bad),
            Err(ParseError::WrongMagic)
        ));
    }

    #[test]
    fn world_round_trip_1x1_with_wall() {
        use psxed_format::world;

        let payload_len =
            (world::WorldHeader::SIZE + world::SectorRecord::SIZE + world::WallRecord::SIZE) as u32;
        let mut buf = [0u8; 12 + 20 + 44 + 24];
        buf[0..4].copy_from_slice(&world::MAGIC);
        buf[4..6].copy_from_slice(&world::VERSION.to_le_bytes());
        buf[6..8].copy_from_slice(&0u16.to_le_bytes());
        buf[8..12].copy_from_slice(&payload_len.to_le_bytes());

        buf[12..14].copy_from_slice(&1u16.to_le_bytes()); // width
        buf[14..16].copy_from_slice(&1u16.to_le_bytes()); // depth
        buf[16..20].copy_from_slice(&world::SECTOR_SIZE.to_le_bytes());
        buf[20..22].copy_from_slice(&1u16.to_le_bytes()); // sectors
        buf[22..24].copy_from_slice(&2u16.to_le_bytes()); // materials
        buf[24..26].copy_from_slice(&1u16.to_le_bytes()); // walls
        buf[26..29].copy_from_slice(&[32, 32, 40]);
        buf[29] = world::world_flags::FOG_ENABLED;

        let sector = 12 + world::WorldHeader::SIZE;
        buf[sector] = world::sector_flags::HAS_FLOOR | world::sector_flags::FLOOR_WALKABLE;
        buf[sector + 1] = world::split::NORTH_WEST_SOUTH_EAST;
        buf[sector + 4..sector + 6].copy_from_slice(&0u16.to_le_bytes());
        buf[sector + 6..sector + 8].copy_from_slice(&world::NO_MATERIAL.to_le_bytes());
        buf[sector + 8..sector + 10].copy_from_slice(&0u16.to_le_bytes());
        buf[sector + 10..sector + 12].copy_from_slice(&1u16.to_le_bytes());

        let wall = sector + world::SectorRecord::SIZE;
        buf[wall] = world::direction::NORTH;
        buf[wall + 1] = world::wall_flags::SOLID;
        buf[wall + 4..wall + 6].copy_from_slice(&1u16.to_le_bytes());
        buf[wall + 8..wall + 12].copy_from_slice(&0i32.to_le_bytes());
        buf[wall + 12..wall + 16].copy_from_slice(&0i32.to_le_bytes());
        buf[wall + 16..wall + 20].copy_from_slice(&1024i32.to_le_bytes());
        buf[wall + 20..wall + 24].copy_from_slice(&1024i32.to_le_bytes());

        let world = World::from_bytes(&buf).expect("parse world");
        assert_eq!(world.width(), 1);
        assert_eq!(world.depth(), 1);
        assert_eq!(world.sector_size(), psxed_format::world::SECTOR_SIZE);
        assert_eq!(world.material_count(), 2);
        assert_eq!(world.wall_count(), 1);
        assert_eq!(world.ambient_color(), [32, 32, 40]);
        assert!(world.fog_enabled());

        let sector = world.sector(0, 0).unwrap();
        assert!(sector.has_floor());
        assert!(sector.floor_walkable());
        assert_eq!(sector.floor_material(), Some(0));
        assert_eq!(sector.ceiling_material(), None);
        assert_eq!(sector.wall_count(), 1);

        let wall = world.sector_wall(sector, 0).unwrap();
        assert_eq!(wall.direction(), psxed_format::world::direction::NORTH);
        assert!(wall.solid());
        assert_eq!(wall.material(), 1);
        assert_eq!(wall.heights(), [0, 0, 1024, 1024]);
    }

    #[test]
    fn world_rejects_bad_sector_count() {
        use psxed_format::world;

        let mut buf = [0u8; 12 + 20];
        buf[0..4].copy_from_slice(&world::MAGIC);
        buf[4..6].copy_from_slice(&world::VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&(world::WorldHeader::SIZE as u32).to_le_bytes());
        buf[12..14].copy_from_slice(&1u16.to_le_bytes());
        buf[14..16].copy_from_slice(&1u16.to_le_bytes());
        buf[16..20].copy_from_slice(&world::SECTOR_SIZE.to_le_bytes());
        buf[20..22].copy_from_slice(&2u16.to_le_bytes());

        assert!(matches!(
            World::from_bytes(&buf),
            Err(ParseError::InvalidWorldLayout)
        ));
    }

    #[test]
    fn world_rejects_wall_range_outside_table() {
        use psxed_format::world;

        let payload_len = (world::WorldHeader::SIZE + world::SectorRecord::SIZE) as u32;
        let mut buf = [0u8; 12 + 20 + 44];
        buf[0..4].copy_from_slice(&world::MAGIC);
        buf[4..6].copy_from_slice(&world::VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&payload_len.to_le_bytes());
        buf[12..14].copy_from_slice(&1u16.to_le_bytes());
        buf[14..16].copy_from_slice(&1u16.to_le_bytes());
        buf[16..20].copy_from_slice(&world::SECTOR_SIZE.to_le_bytes());
        buf[20..22].copy_from_slice(&1u16.to_le_bytes());

        let sector = 12 + world::WorldHeader::SIZE;
        buf[sector + 10..sector + 12].copy_from_slice(&1u16.to_le_bytes());

        assert!(matches!(
            World::from_bytes(&buf),
            Err(ParseError::InvalidWorldLayout)
        ));
    }

    #[test]
    fn round_trip_via_cooked_blob() {
        // Construct a proper blob programmatically: 2 verts, 0 faces.
        // (0 faces is legal; we just want to check header parses.)
        let mut buf: [u8; 12 + 8 + 2 * 6] = [0; 32];
        buf[0..4].copy_from_slice(&psxed_format::mesh::MAGIC);
        buf[4..6].copy_from_slice(&1u16.to_le_bytes());
        buf[6..8].copy_from_slice(&0u16.to_le_bytes());
        buf[8..12].copy_from_slice(&((8 + 2 * 6) as u32).to_le_bytes());
        buf[12..14].copy_from_slice(&2u16.to_le_bytes());
        buf[14..16].copy_from_slice(&0u16.to_le_bytes());
        // Reserved already zero.
        // Vert 0 = (0x0100, 0, 0) — X at offset 20..22.
        buf[20..22].copy_from_slice(&0x0100_i16.to_le_bytes());
        // Vert 1 = (0, 0x0200, 0) — Y at offset 28..30
        // (vert 1 starts at 26, Y is offset +2 into the vert).
        buf[28..30].copy_from_slice(&0x0200_i16.to_le_bytes());

        let m = Mesh::from_bytes(&buf).expect("parse");
        assert_eq!(m.vert_count(), 2);
        assert_eq!(m.face_count(), 0);
        let v0 = m.vertex(0);
        assert_eq!(v0.x, 0x0100);
        let v1 = m.vertex(1);
        assert_eq!(v1.y, 0x0200);
    }
}
