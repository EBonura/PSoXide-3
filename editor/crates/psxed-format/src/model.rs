//! On-disk layout for cooked textured 3D models (`.psxmdl` files).
//!
//! This format is the next step beyond simple `.psxm` triangle soups:
//! it carries UVs, material slots, and a rigid-skeletal part table that
//! lets the runtime draw animated models one joint-owned part at a time.
//!
//! # File layout
//!
//! ```text
//!   AssetHeader (12 bytes)
//!     magic       = b"PSMD"
//!     version     = VERSION
//!     flags       = ModelFlags bits
//!     payload_len = everything after this header
//!
//!   ModelHeader (16 bytes)
//!     joint_count     u16
//!     part_count      u16
//!     vertex_count    u16
//!     face_count      u16
//!     material_count  u16
//!     texture_width   u16
//!     texture_height  u16
//!     local_to_world_q12 u16
//!
//!   Joint table: joint_count × 4 bytes
//!     parent: u16 LE, or NO_JOINT for a root joint
//!     _reserved: u16
//!
//!   Material table: material_count × 8 bytes
//!     texture_index: u16 LE
//!     flags: u16 LE
//!     base_color: rgba8
//!
//!   Part table: part_count × 16 bytes
//!     joint_index, first_vertex, vertex_count, first_face, face_count,
//!     material_index, _reserved
//!
//!   Vertex table: vertex_count × 16 bytes
//!     position: i16x3 model-local units
//!     normal:   i16x3 Q3.12
//!     uv:       u8x2
//!     joint1:   u8 secondary skin bone (`NO_JOINT8` when unused)
//!     blend:    u8 weight on `joint1`, 0..=255 (0 = single-bone)
//!
//!   Face table: face_count × 6 bytes
//!     a, b, c: u16 LE each, indexing the global vertex table
//! ```
//!
//! Skinning is a hybrid model. Each part still owns one rigid bone
//! (its `joint_index`), and most triangles transform rigidly through
//! that bone -- fast GTE path. A vertex near a joint seam can name a
//! second bone in `joint1` along with a `blend` weight: the runtime
//! transforms that vertex by both bones and lerps the view-space
//! results, which keeps elbow and shoulder creases continuous instead
//! of opening into chunky gaps. `blend == 0` opts a vertex out of the
//! slow path entirely. Model-local coordinates are deliberately
//! allowed to use a much denser scale than world/grid coordinates;
//! the header's `local_to_world_q12` is the content pipeline's
//! suggested conversion for previews and simple runtime draw paths.

/// ASCII magic identifying the `.psxmdl` model format.
pub const MAGIC: [u8; 4] = *b"PSMD";

/// Current model format revision.
pub const VERSION: u16 = 2;

/// Sentinel parent/joint value used when a record has no joint.
pub const NO_JOINT: u16 = u16::MAX;

/// 8-bit sentinel for the per-vertex secondary joint slot.
///
/// Stored in the `joint1` byte of a vertex record when the cooker
/// chose not to assign a secondary blend bone. Readers that see this
/// value should treat the vertex as single-bone regardless of `blend`.
pub const NO_JOINT8: u8 = u8::MAX;

/// Fallback model-local to world-space scale. `0x1000` is identity.
pub const DEFAULT_LOCAL_TO_WORLD_Q12: u16 = 0x1000;

/// Model-specific feature flags (stored in `AssetHeader::flags`).
pub mod flags {
    /// Vertex records carry Q3.12 normals.
    pub const HAS_NORMALS: u16 = 1 << 0;
    /// Vertex records carry 8-bit UV coordinates.
    pub const HAS_UVS: u16 = 1 << 1;
    /// Part records are associated with joint pose matrices.
    pub const RIGID_SKINNED: u16 = 1 << 2;
    /// Vertex records may carry a secondary joint and blend weight.
    ///
    /// Set when the cooker emitted any vertex with `blend > 0`. Pure
    /// rigid models clear this bit so the runtime can stay on the
    /// single-bone fast path without inspecting per-vertex blend.
    pub const BLEND_SKIN: u16 = 1 << 3;
}

/// Byte layout of the model payload header.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct ModelHeader {
    /// Number of joint records.
    pub joint_count: u16,
    /// Number of rigid part records.
    pub part_count: u16,
    /// Number of vertex records.
    pub vertex_count: u16,
    /// Number of triangle records.
    pub face_count: u16,
    /// Number of material records.
    pub material_count: u16,
    /// Primary texture width in texels.
    pub texture_width: u16,
    /// Primary texture height in texels.
    pub texture_height: u16,
    /// Suggested uniform scale from model-local units to engine world units.
    ///
    /// `0x1000` is identity. A value of zero means "unspecified" and
    /// readers should treat it as [`DEFAULT_LOCAL_TO_WORLD_Q12`].
    pub local_to_world_q12: u16,
}

impl ModelHeader {
    /// Size of the model header in bytes (always 16).
    pub const SIZE: usize = 16;

    /// Build a model header.
    pub const fn new(
        joint_count: u16,
        part_count: u16,
        vertex_count: u16,
        face_count: u16,
        material_count: u16,
        texture_width: u16,
        texture_height: u16,
    ) -> Self {
        Self {
            joint_count,
            part_count,
            vertex_count,
            face_count,
            material_count,
            texture_width,
            texture_height,
            local_to_world_q12: DEFAULT_LOCAL_TO_WORLD_Q12,
        }
    }

    /// Return a copy with the suggested model-local to world scale set.
    pub const fn with_local_to_world_q12(mut self, scale_q12: u16) -> Self {
        self.local_to_world_q12 = scale_q12;
        self
    }
}

/// Joint hierarchy record.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct JointRecord {
    /// Parent joint index, or [`NO_JOINT`] for a root joint.
    pub parent: u16,
    /// Reserved. Writers set to zero; readers ignore.
    pub _reserved: u16,
}

impl JointRecord {
    /// Size of one joint record in bytes.
    pub const SIZE: usize = 4;

    /// Build a joint record.
    pub const fn new(parent: u16) -> Self {
        Self {
            parent,
            _reserved: 0,
        }
    }
}

/// Material slot record.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct MaterialRecord {
    /// Texture slot index. Version 1 emits one external texture, index 0.
    pub texture_index: u16,
    /// Material flags, currently reserved.
    pub flags: u16,
    /// Base colour/tint, RGBA8. Current renderer can treat this as white.
    pub base_color: [u8; 4],
}

impl MaterialRecord {
    /// Size of one material record in bytes.
    pub const SIZE: usize = 8;

    /// Build a material record.
    pub const fn new(texture_index: u16, flags: u16, base_color: [u8; 4]) -> Self {
        Self {
            texture_index,
            flags,
            base_color,
        }
    }
}

/// Rigid render part record.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct PartRecord {
    /// Joint whose pose matrix animates this part.
    pub joint_index: u16,
    /// First vertex in the global vertex table.
    pub first_vertex: u16,
    /// Number of vertices in this part.
    pub vertex_count: u16,
    /// First face in the global face table.
    pub first_face: u16,
    /// Number of faces in this part.
    pub face_count: u16,
    /// Material slot used by this part.
    pub material_index: u16,
    /// Reserved. Writers set to zero; readers ignore.
    pub _reserved: u32,
}

impl PartRecord {
    /// Size of one part record in bytes.
    pub const SIZE: usize = 16;

    /// Build a part record.
    pub const fn new(
        joint_index: u16,
        first_vertex: u16,
        vertex_count: u16,
        first_face: u16,
        face_count: u16,
        material_index: u16,
    ) -> Self {
        Self {
            joint_index,
            first_vertex,
            vertex_count,
            first_face,
            face_count,
            material_index,
            _reserved: 0,
        }
    }
}

/// Size of one vertex record in bytes.
pub const VERTEX_RECORD_SIZE: usize = 16;

/// Size of one face record in bytes.
pub const FACE_RECORD_SIZE: usize = 6;
