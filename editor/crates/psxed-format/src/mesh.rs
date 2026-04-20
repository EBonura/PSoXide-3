//! On-disk layout for cooked 3D meshes (`.psxm` files).
//!
//! # File layout
//!
//! ```text
//!   AssetHeader (12 bytes)
//!     magic   = b"PSXM"
//!     version = MESH_VERSION
//!     flags   = MeshFlags bits
//!     payload_len = everything after the header
//!
//!   MeshHeader (8 bytes)
//!     vert_count  u16 LE  — number of vertex entries
//!     face_count  u16 LE  — number of triangle entries
//!     _reserved   u32     — kept zero, for future use
//!
//!   Vertex table: vert_count × 6 bytes
//!     x: i16 LE  (Q3.12 fixed-point)
//!     y: i16 LE
//!     z: i16 LE
//!
//!   Index table: face_count × 3 bytes
//!     a, b, c: u8 each
//!
//!   Face-colour table: face_count × 3 bytes  (only if FLAG_HAS_FACE_COLORS set)
//!     r, g, b: u8 each
//! ```
//!
//! Total file size: `12 + 8 + (vert_count * 6) + (face_count * 3)
//! [ + (face_count * 3) if has_face_colors ]`.
//!
//! Every multi-byte integer is little-endian; bytes are tightly
//! packed (no alignment padding). That lets the runtime parser
//! just take slices into the blob without unpacking.

/// ASCII magic identifying the `.psxm` format.
pub const MAGIC: [u8; 4] = *b"PSXM";

/// Current mesh format revision. Runtime parser rejects values
/// it doesn't know; editor always writes this value.
pub const VERSION: u16 = 1;

/// Mesh-specific feature flags (stored in `AssetHeader::flags`).
pub mod flags {
    /// Face-colour table is present after the index table.
    pub const HAS_FACE_COLORS: u16 = 1 << 0;
    // Reserved bits:
    //   1 — HAS_NORMALS     (future: per-vertex normals for lighting)
    //   2 — HAS_UVS         (future: per-vertex UV for textured meshes)
    //   3 — HAS_VERT_COLORS (future: per-vertex Gouraud colours)
}

/// Byte layout of the mesh payload header that immediately
/// follows the 12-byte `AssetHeader`.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct MeshHeader {
    /// Number of vertex entries in the vertex table.
    pub vert_count: u16,
    /// Number of triangles in the index table.
    pub face_count: u16,
    /// Reserved. Writers must store zero; readers must ignore.
    pub _reserved: u32,
}

impl MeshHeader {
    /// Size of the mesh header in bytes (always 8).
    pub const SIZE: usize = 8;

    /// Build a header. The caller is responsible for ensuring
    /// `vert_count * 6 + face_count * 3 + optional_colors` matches
    /// the actual payload size.
    pub const fn new(vert_count: u16, face_count: u16) -> Self {
        Self { vert_count, face_count, _reserved: 0 }
    }
}
