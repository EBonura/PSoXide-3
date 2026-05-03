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
//! // At runtime, parse into a typed view. Zero-copy -- the view
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

use psx_gte::math::{Vec3I16, Vec3I32};

const MESH_VERSION_U8_INDICES: u16 = 1;
const MESH_U8_INDEX_STRIDE: usize = 3;
const MESH_U16_INDEX_STRIDE: usize = 6;

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
    /// Model header/table fields are inconsistent.
    InvalidModelLayout,
    /// Animation header/table fields are inconsistent.
    InvalidAnimationLayout,
}

/// A parsed 3D mesh backed by slices into the caller's cooked blob.
///
/// Cheap to construct (just bounds-checks the header + computes
/// sub-slice offsets). Cheap to pass around -- table `&[u8]` slices,
/// an index stride, a flags `u16`, and the counts.
#[derive(Copy, Clone, Debug)]
pub struct Mesh<'a> {
    verts: &'a [u8],
    indices: &'a [u8],
    face_colors: Option<&'a [u8]>,
    vertex_normals: Option<&'a [u8]>,
    index_stride: u8,
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
        let index_stride = match version {
            MESH_VERSION_U8_INDICES => MESH_U8_INDEX_STRIDE,
            psxed_format::mesh::VERSION => MESH_U16_INDEX_STRIDE,
            _ => return Err(ParseError::UnsupportedVersion(version)),
        };
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

        let index_bytes = (face_count as usize) * index_stride;
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
            index_stride: index_stride as u8,
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
    /// [`Vec3I16::ZERO`] if the index is out of range -- keeps
    /// the render path branch-free, callers who care can
    /// check against [`Self::vert_count`] first.
    #[inline]
    pub fn vertex(&self, i: u16) -> Vec3I16 {
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
    pub fn face(&self, i: u16) -> (u16, u16, u16) {
        let idx = i as usize;
        if idx >= self.face_count as usize {
            return (0, 0, 0);
        }
        let base = idx * self.index_stride as usize;
        if self.index_stride as usize == MESH_U16_INDEX_STRIDE {
            (
                u16::from_le_bytes([self.indices[base], self.indices[base + 1]]),
                u16::from_le_bytes([self.indices[base + 2], self.indices[base + 3]]),
                u16::from_le_bytes([self.indices[base + 4], self.indices[base + 5]]),
            )
        } else {
            (
                self.indices[base] as u16,
                self.indices[base + 1] as u16,
                self.indices[base + 2] as u16,
            )
        }
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
    pub fn vertex_normal(&self, i: u16) -> Option<Vec3I16> {
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

/// A parsed textured 3D model backed by slices into the caller's
/// cooked `.psxmdl` blob.
///
/// This is intentionally a low-level view. It exposes the tables the
/// renderer needs -- joints, materials, rigid parts, vertices, and
/// faces -- without allocating or converting the whole model at load
/// time.
#[derive(Copy, Clone, Debug)]
pub struct Model<'a> {
    joints: &'a [u8],
    materials: &'a [u8],
    parts: &'a [u8],
    vertices: &'a [u8],
    faces: &'a [u8],
    joint_count: u16,
    material_count: u16,
    part_count: u16,
    vertex_count: u16,
    face_count: u16,
    texture_width: u16,
    texture_height: u16,
    local_to_world_q12: u16,
    flags: u16,
}

impl<'a> Model<'a> {
    /// Parse a cooked `.psxmdl` blob.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        use psxed_format::model::{
            JointRecord, MaterialRecord, ModelHeader, PartRecord, FACE_RECORD_SIZE, MAGIC, VERSION,
            VERTEX_RECORD_SIZE,
        };

        if bytes.len() < psxed_format::AssetHeader::SIZE {
            return Err(ParseError::Truncated);
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(ParseError::WrongMagic);
        }
        let version = read_u16(bytes, 4);
        if version != VERSION {
            return Err(ParseError::UnsupportedVersion(version));
        }
        let flags = read_u16(bytes, 6);
        let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let payload_start = psxed_format::AssetHeader::SIZE;
        let actual_payload = bytes.len().saturating_sub(payload_start);
        if payload_len as usize != actual_payload {
            return Err(ParseError::InvalidPayloadLen {
                declared: payload_len,
                actual: actual_payload,
            });
        }
        if actual_payload < ModelHeader::SIZE {
            return Err(ParseError::Truncated);
        }

        let mh = &bytes[payload_start..payload_start + ModelHeader::SIZE];
        let joint_count = read_u16(mh, 0);
        let part_count = read_u16(mh, 2);
        let vertex_count = read_u16(mh, 4);
        let face_count = read_u16(mh, 6);
        let material_count = read_u16(mh, 8);
        let texture_width = read_u16(mh, 10);
        let texture_height = read_u16(mh, 12);
        let local_to_world_q12 = read_u16(mh, 14);

        let mut off = payload_start + ModelHeader::SIZE;

        let joint_bytes = checked_table_bytes(joint_count, JointRecord::SIZE)?;
        let joints = take_table(bytes, &mut off, joint_bytes)?;

        let material_bytes = checked_table_bytes(material_count, MaterialRecord::SIZE)?;
        let materials = take_table(bytes, &mut off, material_bytes)?;

        let part_bytes = checked_table_bytes(part_count, PartRecord::SIZE)?;
        let parts = take_table(bytes, &mut off, part_bytes)?;

        let vertex_bytes = checked_table_bytes(vertex_count, VERTEX_RECORD_SIZE)?;
        let vertices = take_table(bytes, &mut off, vertex_bytes)?;

        let face_bytes = checked_table_bytes(face_count, FACE_RECORD_SIZE)?;
        let faces = take_table(bytes, &mut off, face_bytes)?;

        if off != bytes.len() {
            return Err(ParseError::InvalidModelLayout);
        }
        validate_model_parts(parts, joint_count, material_count, vertex_count, face_count)?;
        validate_model_faces(faces, vertex_count)?;

        Ok(Self {
            joints,
            materials,
            parts,
            vertices,
            faces,
            joint_count,
            material_count,
            part_count,
            vertex_count,
            face_count,
            texture_width,
            texture_height,
            local_to_world_q12,
            flags,
        })
    }

    /// Model feature flags (see [`psxed_format::model::flags`]).
    #[inline]
    pub fn flags(&self) -> u16 {
        self.flags
    }

    /// Number of joint records.
    #[inline]
    pub fn joint_count(&self) -> u16 {
        self.joint_count
    }

    /// Number of material records.
    #[inline]
    pub fn material_count(&self) -> u16 {
        self.material_count
    }

    /// Number of rigid part records.
    #[inline]
    pub fn part_count(&self) -> u16 {
        self.part_count
    }

    /// Number of vertex records.
    #[inline]
    pub fn vertex_count(&self) -> u16 {
        self.vertex_count
    }

    /// Number of triangle records.
    #[inline]
    pub fn face_count(&self) -> u16 {
        self.face_count
    }

    /// Primary texture width in texels.
    #[inline]
    pub fn texture_width(&self) -> u16 {
        self.texture_width
    }

    /// Primary texture height in texels.
    #[inline]
    pub fn texture_height(&self) -> u16 {
        self.texture_height
    }

    /// Suggested uniform scale from model-local units to engine world units.
    ///
    /// `0x1000` is identity. Older blobs may store zero in the reserved
    /// header slot; those are treated as identity.
    #[inline]
    pub fn local_to_world_q12(&self) -> u16 {
        if self.local_to_world_q12 == 0 {
            psxed_format::model::DEFAULT_LOCAL_TO_WORLD_Q12
        } else {
            self.local_to_world_q12
        }
    }

    /// Joint record by index.
    pub fn joint(&self, index: u16) -> Option<ModelJoint> {
        if index >= self.joint_count {
            return None;
        }
        let base = index as usize * psxed_format::model::JointRecord::SIZE;
        let bytes = self
            .joints
            .get(base..base + psxed_format::model::JointRecord::SIZE)?;
        Some(ModelJoint {
            parent: read_u16(bytes, 0),
        })
    }

    /// Material record by index.
    pub fn material(&self, index: u16) -> Option<ModelMaterial> {
        if index >= self.material_count {
            return None;
        }
        let base = index as usize * psxed_format::model::MaterialRecord::SIZE;
        let bytes = self
            .materials
            .get(base..base + psxed_format::model::MaterialRecord::SIZE)?;
        Some(ModelMaterial {
            texture_index: read_u16(bytes, 0),
            flags: read_u16(bytes, 2),
            base_color: [bytes[4], bytes[5], bytes[6], bytes[7]],
        })
    }

    /// Rigid part record by index.
    pub fn part(&self, index: u16) -> Option<ModelPart> {
        if index >= self.part_count {
            return None;
        }
        let base = index as usize * psxed_format::model::PartRecord::SIZE;
        let bytes = self
            .parts
            .get(base..base + psxed_format::model::PartRecord::SIZE)?;
        Some(ModelPart {
            joint_index: read_u16(bytes, 0),
            first_vertex: read_u16(bytes, 2),
            vertex_count: read_u16(bytes, 4),
            first_face: read_u16(bytes, 6),
            face_count: read_u16(bytes, 8),
            material_index: read_u16(bytes, 10),
        })
    }

    /// Vertex record by global vertex index.
    pub fn vertex(&self, index: u16) -> Option<ModelVertex> {
        if index >= self.vertex_count {
            return None;
        }
        let base = index as usize * psxed_format::model::VERTEX_RECORD_SIZE;
        let bytes = self
            .vertices
            .get(base..base + psxed_format::model::VERTEX_RECORD_SIZE)?;
        Some(ModelVertex {
            position: Vec3I16::new(read_i16(bytes, 0), read_i16(bytes, 2), read_i16(bytes, 4)),
            joint1: bytes[6],
            blend: bytes[7],
        })
    }

    /// Textured triangle by global face index.
    pub fn face(&self, index: u16) -> Option<ModelFace> {
        if index >= self.face_count {
            return None;
        }
        let base = index as usize * psxed_format::model::FACE_RECORD_SIZE;
        let bytes = self
            .faces
            .get(base..base + psxed_format::model::FACE_RECORD_SIZE)?;
        Some(ModelFace {
            corners: [
                ModelFaceCorner {
                    vertex_index: read_u16(bytes, 0),
                    uv: (bytes[2], bytes[3]),
                },
                ModelFaceCorner {
                    vertex_index: read_u16(bytes, 4),
                    uv: (bytes[6], bytes[7]),
                },
                ModelFaceCorner {
                    vertex_index: read_u16(bytes, 8),
                    uv: (bytes[10], bytes[11]),
                },
            ],
        })
    }
}

/// Decoded model joint record.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ModelJoint {
    parent: u16,
}

impl ModelJoint {
    /// Parent joint index, or `None` for a root joint.
    #[inline]
    pub fn parent(&self) -> Option<u16> {
        if self.parent == psxed_format::model::NO_JOINT {
            None
        } else {
            Some(self.parent)
        }
    }
}

/// Decoded model material record.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ModelMaterial {
    texture_index: u16,
    flags: u16,
    base_color: [u8; 4],
}

impl ModelMaterial {
    /// Texture slot index.
    #[inline]
    pub fn texture_index(&self) -> u16 {
        self.texture_index
    }

    /// Material flags.
    #[inline]
    pub fn flags(&self) -> u16 {
        self.flags
    }

    /// Base colour/tint as RGBA8.
    #[inline]
    pub fn base_color(&self) -> [u8; 4] {
        self.base_color
    }
}

/// Decoded rigid part record.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ModelPart {
    joint_index: u16,
    first_vertex: u16,
    vertex_count: u16,
    first_face: u16,
    face_count: u16,
    material_index: u16,
}

impl ModelPart {
    /// Joint whose animation pose applies to this part.
    #[inline]
    pub fn joint_index(&self) -> u16 {
        self.joint_index
    }

    /// First global vertex owned by this part.
    #[inline]
    pub fn first_vertex(&self) -> u16 {
        self.first_vertex
    }

    /// Number of vertices owned by this part.
    #[inline]
    pub fn vertex_count(&self) -> u16 {
        self.vertex_count
    }

    /// First global triangle owned by this part.
    #[inline]
    pub fn first_face(&self) -> u16 {
        self.first_face
    }

    /// Number of triangles owned by this part.
    #[inline]
    pub fn face_count(&self) -> u16 {
        self.face_count
    }

    /// Material slot used by this part.
    #[inline]
    pub fn material_index(&self) -> u16 {
        self.material_index
    }
}

/// Sentinel value for [`ModelVertex::joint1`] meaning "no secondary
/// blend bone". Re-exported from `psxed_format::model` so runtime
/// code does not need to depend on the editor crate.
pub const NO_JOINT8: u8 = psxed_format::model::NO_JOINT8;

/// Decoded textured model vertex.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ModelVertex {
    /// Model-local position. The cooked importer may use a much denser
    /// scale here than world/grid units; use [`Model::local_to_world_q12`]
    /// when placing the model directly into world space.
    pub position: Vec3I16,
    /// Secondary blend joint, or [`NO_JOINT8`] when this vertex is
    /// single-bone.
    pub joint1: u8,
    /// Weight of `joint1` for view-space blending (0..=255). Zero
    /// signals the renderer to stay on the single-bone GTE fast path.
    pub blend: u8,
}

impl ModelVertex {
    /// `true` when this vertex needs the two-bone blend render path.
    #[inline]
    pub fn is_blend(&self) -> bool {
        self.blend != 0 && self.joint1 != NO_JOINT8
    }
}

/// One textured triangle corner in a cooked model face.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ModelFaceCorner {
    /// Global skinned vertex index.
    pub vertex_index: u16,
    /// 8-bit texture coordinate for this face corner.
    pub uv: (u8, u8),
}

/// Decoded textured model face.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ModelFace {
    /// Three triangle corners in packet order.
    pub corners: [ModelFaceCorner; 3],
}

/// A parsed rigid-skeletal animation backed by slices into the
/// caller's cooked `.psxanim` blob.
#[derive(Copy, Clone, Debug)]
pub struct Animation<'a> {
    poses: &'a [u8],
    joint_count: u16,
    frame_count: u16,
    sample_rate_hz: u16,
}

impl<'a> Animation<'a> {
    /// Parse a cooked `.psxanim` blob.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        use psxed_format::animation::{AnimationHeader, MAGIC, POSE_RECORD_SIZE, VERSION};

        if bytes.len() < psxed_format::AssetHeader::SIZE {
            return Err(ParseError::Truncated);
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(ParseError::WrongMagic);
        }
        let version = read_u16(bytes, 4);
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
        if actual_payload < AnimationHeader::SIZE {
            return Err(ParseError::Truncated);
        }

        let ah = &bytes[payload_start..payload_start + AnimationHeader::SIZE];
        let joint_count = read_u16(ah, 0);
        let frame_count = read_u16(ah, 2);
        let sample_rate_hz = read_u16(ah, 4);
        if joint_count == 0 || frame_count == 0 || sample_rate_hz == 0 {
            return Err(ParseError::InvalidAnimationLayout);
        }

        let mut off = payload_start + AnimationHeader::SIZE;
        let pose_count = (joint_count as usize)
            .checked_mul(frame_count as usize)
            .ok_or(ParseError::TableOverflow)?;
        let pose_bytes = pose_count
            .checked_mul(POSE_RECORD_SIZE)
            .ok_or(ParseError::TableOverflow)?;
        let poses = take_table(bytes, &mut off, pose_bytes)?;
        if off != bytes.len() {
            return Err(ParseError::InvalidAnimationLayout);
        }

        Ok(Self {
            poses,
            joint_count,
            frame_count,
            sample_rate_hz,
        })
    }

    /// Number of joint poses per frame.
    #[inline]
    pub fn joint_count(&self) -> u16 {
        self.joint_count
    }

    /// Number of sampled frames.
    #[inline]
    pub fn frame_count(&self) -> u16 {
        self.frame_count
    }

    /// Integer sample rate in Hz.
    #[inline]
    pub fn sample_rate_hz(&self) -> u16 {
        self.sample_rate_hz
    }

    /// Q12 sampled-frame phase advance for one playback tick.
    ///
    /// `playback_hz` is the caller's update cadence. For example,
    /// a 15 Hz cooked clip played by a 30 Hz update loop advances by
    /// half a sampled frame per tick (`0x0800`).
    #[inline]
    pub fn phase_step_q12(&self, playback_hz: u16) -> u32 {
        ((self.sample_rate_hz as u32) << 12) / playback_hz.max(1) as u32
    }

    /// Convert a fixed-rate playback tick to a Q12 sampled-frame phase.
    ///
    /// This is a convenience for frame-locked demos. More advanced
    /// scenes can accumulate [`Animation::phase_step_q12`] themselves
    /// when playback speed, pausing, or elapsed-time correction matters.
    #[inline]
    pub fn phase_at_tick_q12(&self, playback_tick: u32, playback_hz: u16) -> u32 {
        playback_tick.wrapping_mul(self.phase_step_q12(playback_hz))
    }

    /// Joint pose at `frame_index`, `joint_index`.
    pub fn pose(&self, frame_index: u16, joint_index: u16) -> Option<JointPose> {
        if frame_index >= self.frame_count || joint_index >= self.joint_count {
            return None;
        }
        let flat = frame_index as usize * self.joint_count as usize + joint_index as usize;
        let base = flat * psxed_format::animation::POSE_RECORD_SIZE;
        let bytes = self
            .poses
            .get(base..base + psxed_format::animation::POSE_RECORD_SIZE)?;
        let mut matrix = [[0i16; 3]; 3];
        let mut off = 0;
        for col in matrix.iter_mut() {
            for cell in col.iter_mut() {
                *cell = read_i16(bytes, off);
                off += 2;
            }
        }
        let translation = Vec3I32::new(
            read_i32(bytes, off),
            read_i32(bytes, off + 4),
            read_i32(bytes, off + 8),
        );
        Some(JointPose {
            matrix,
            translation,
        })
    }

    /// Interpolated looping joint pose at a Q12 fixed-point frame phase.
    ///
    /// `frame_q12` uses sampled animation frames as the integer unit
    /// and 12 fractional bits. For example, `1 << 12` samples frame
    /// 1 exactly, while `1 << 11` samples halfway between frames 0
    /// and 1. The cooker writes endpoint-inclusive clips, so looping
    /// playback treats the final stored frame as the duplicate of
    /// frame 0 and blends `frame_count - 2` back to frame 0.
    pub fn pose_looped_q12(&self, frame_q12: u32, joint_index: u16) -> Option<JointPose> {
        if self.frame_count == 0 {
            return None;
        }

        let cycle_frames = self.frame_count.saturating_sub(1).max(1);
        let base_frame = ((frame_q12 >> 12) % cycle_frames as u32) as u16;
        let next_frame = if cycle_frames <= 1 || base_frame + 1 >= cycle_frames {
            0
        } else {
            base_frame + 1
        };
        let alpha_q12 = (frame_q12 & 0x0fff) as u16;
        if alpha_q12 == 0 || base_frame == next_frame {
            return self.pose(base_frame, joint_index);
        }

        let a = self.pose(base_frame, joint_index)?;
        let b = self.pose(next_frame, joint_index)?;
        Some(lerp_pose_q12(a, b, alpha_q12))
    }
}

/// Decoded joint pose matrix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JointPose {
    /// Q3.12 column-major 3×3 transform.
    pub matrix: [[i16; 3]; 3],
    /// Q3.12 translation vector.
    pub translation: Vec3I32,
}

fn lerp_pose_q12(a: JointPose, b: JointPose, alpha_q12: u16) -> JointPose {
    let mut matrix = [[0i16; 3]; 3];
    let mut col = 0;
    while col < 3 {
        let mut row = 0;
        while row < 3 {
            matrix[col][row] = lerp_i16_q12(a.matrix[col][row], b.matrix[col][row], alpha_q12);
            row += 1;
        }
        col += 1;
    }

    JointPose {
        matrix,
        translation: Vec3I32::new(
            lerp_i32_q12(a.translation.x, b.translation.x, alpha_q12),
            lerp_i32_q12(a.translation.y, b.translation.y, alpha_q12),
            lerp_i32_q12(a.translation.z, b.translation.z, alpha_q12),
        ),
    }
}

fn lerp_i16_q12(a: i16, b: i16, alpha_q12: u16) -> i16 {
    let value = a as i32 + (((b as i32 - a as i32) * alpha_q12 as i32) >> 12);
    clamp_i32_to_i16(value)
}

fn lerp_i32_q12(a: i32, b: i32, alpha_q12: u16) -> i32 {
    let delta = b.saturating_sub(a);
    a.saturating_add(scale_i32_q12(delta, alpha_q12 as i32))
}

fn scale_i32_q12(value: i32, scale_q12: i32) -> i32 {
    let whole = value >> 12;
    let frac = value - (whole << 12);
    whole.saturating_mul(scale_q12) + ((frac * scale_q12) >> 12)
}

fn clamp_i32_to_i16(value: i32) -> i16 {
    if value < i16::MIN as i32 {
        i16::MIN
    } else if value > i16::MAX as i32 {
        i16::MAX
    } else {
        value as i16
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
    /// Packed pixel halfwords -- 4 texels per u16 at 4bpp,
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
    /// borrows into `bytes`. Cheap -- header decode + two slice
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
    surface_lights: &'a [u8],
    sector_record_size: usize,
    wall_record_size: usize,
    width: u16,
    depth: u16,
    sector_size: i32,
    material_count: u16,
    wall_count: u16,
    surface_light_count: u16,
    ambient_color: [u8; 3],
    flags: u8,
    static_vertex_lighting: bool,
}

/// Four PS1 UV coordinates for one world quad.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldQuadUvs {
    corners: [(u8, u8); 4],
}

impl WorldQuadUvs {
    /// Build a quad UV record from face-corner coordinates.
    pub const fn new(corners: [(u8, u8); 4]) -> Self {
        Self { corners }
    }

    /// Return UVs as face-corner coordinates.
    pub const fn corners(self) -> [(u8, u8); 4] {
        self.corners
    }
}

/// Four RGB vertex colours for one world quad.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorldSurfaceLight {
    vertex_rgb: [[u8; 3]; 4],
}

impl WorldSurfaceLight {
    /// Full-bright neutral lighting for legacy/unlit rooms.
    pub const fn white() -> Self {
        Self {
            vertex_rgb: [[255, 255, 255]; 4],
        }
    }

    /// Build from per-corner RGB values.
    pub const fn new(vertex_rgb: [[u8; 3]; 4]) -> Self {
        Self { vertex_rgb }
    }

    /// Return RGB values in face-corner order.
    pub const fn vertex_rgb(self) -> [[u8; 3]; 4] {
        self.vertex_rgb
    }
}

const WORLD_V1_SECTOR_RECORD_SIZE: usize = 44;
const WORLD_V1_WALL_RECORD_SIZE: usize = 24;
const WORLD_V2_SECTOR_RECORD_SIZE: usize = 60;
const WORLD_V2_WALL_RECORD_SIZE: usize = 32;

impl<'a> World<'a> {
    /// Parse a cooked `.psxw` blob.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        use psxed_format::world::{WorldHeader, MAGIC, VERSION, VERSION_V1, VERSION_V2};

        if bytes.len() < psxed_format::AssetHeader::SIZE {
            return Err(ParseError::Truncated);
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(ParseError::WrongMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION && version != VERSION_V2 && version != VERSION_V1 {
            return Err(ParseError::UnsupportedVersion(version));
        }
        let (sector_record_size, wall_record_size) = match version {
            VERSION_V1 => (WORLD_V1_SECTOR_RECORD_SIZE, WORLD_V1_WALL_RECORD_SIZE),
            VERSION_V2 => (WORLD_V2_SECTOR_RECORD_SIZE, WORLD_V2_WALL_RECORD_SIZE),
            _ => (
                psxed_format::world::SectorRecord::SIZE,
                psxed_format::world::WallRecord::SIZE,
            ),
        };
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
        let surface_light_count = if version == VERSION {
            read_u16(wh, 18)
        } else {
            0
        };
        let static_vertex_lighting = version == VERSION
            && flags & psxed_format::world::world_flags::STATIC_VERTEX_LIGHTING != 0;

        let expected_sectors = (width as usize)
            .checked_mul(depth as usize)
            .ok_or(ParseError::InvalidWorldLayout)?;
        if sector_count as usize != expected_sectors {
            return Err(ParseError::InvalidWorldLayout);
        }
        if static_vertex_lighting {
            let expected_surface_lights = sector_count
                .checked_mul(2)
                .and_then(|count| count.checked_add(wall_count))
                .ok_or(ParseError::InvalidWorldLayout)?;
            if surface_light_count != expected_surface_lights {
                return Err(ParseError::InvalidWorldLayout);
            }
        } else if surface_light_count != 0 {
            return Err(ParseError::InvalidWorldLayout);
        }

        let mut off = payload_start + WorldHeader::SIZE;
        let sector_bytes = (sector_count as usize)
            .checked_mul(sector_record_size)
            .ok_or(ParseError::TableOverflow)?;
        if off + sector_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let sectors = &bytes[off..off + sector_bytes];
        off += sector_bytes;

        let wall_bytes = (wall_count as usize)
            .checked_mul(wall_record_size)
            .ok_or(ParseError::TableOverflow)?;
        if off + wall_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let walls = &bytes[off..off + wall_bytes];
        off += wall_bytes;
        let surface_light_bytes = (surface_light_count as usize)
            .checked_mul(psxed_format::world::SurfaceLightRecord::SIZE)
            .ok_or(ParseError::TableOverflow)?;
        if off + surface_light_bytes > bytes.len() {
            return Err(ParseError::TableOverflow);
        }
        let surface_lights = &bytes[off..off + surface_light_bytes];
        off += surface_light_bytes;
        if off != bytes.len() {
            return Err(ParseError::InvalidWorldLayout);
        }

        validate_sector_wall_ranges(sectors, sector_record_size, wall_count)?;

        Ok(Self {
            sectors,
            walls,
            surface_lights,
            sector_record_size,
            wall_record_size,
            width,
            depth,
            sector_size,
            material_count,
            wall_count,
            surface_light_count,
            ambient_color,
            flags,
            static_vertex_lighting,
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

    /// Number of appended static surface-light records.
    #[inline]
    pub fn surface_light_count(&self) -> u16 {
        self.surface_light_count
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

    /// Whether face records carry baked static vertex lighting.
    #[inline]
    pub fn static_vertex_lighting(&self) -> bool {
        self.static_vertex_lighting
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
        let size = self.sector_record_size;
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
        let size = self.wall_record_size;
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

    /// Static surface-light record by direct table index.
    pub fn surface_light(&self, index: u16) -> Option<WorldSurfaceLight> {
        if !self.static_vertex_lighting || index >= self.surface_light_count {
            return None;
        }
        let size = psxed_format::world::SurfaceLightRecord::SIZE;
        let base = index as usize * size;
        let end = base.checked_add(size)?;
        let bytes = self.surface_lights.get(base..end)?;
        Some(read_world_surface_light(bytes, 0))
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
    floor_uvs: WorldQuadUvs,
    ceiling_uvs: WorldQuadUvs,
}

impl WorldSector {
    fn decode(bytes: &[u8]) -> Self {
        let has_v2_uvs = bytes.len() >= WORLD_V2_SECTOR_RECORD_SIZE;
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
            floor_uvs: if has_v2_uvs {
                read_world_uvs(bytes, 44)
            } else {
                WorldQuadUvs::new(psxed_format::world::FLOOR_UVS)
            },
            ceiling_uvs: if has_v2_uvs {
                read_world_uvs(bytes, 52)
            } else {
                WorldQuadUvs::new(psxed_format::world::FLOOR_UVS)
            },
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

    /// Floor UVs `[NW, NE, SE, SW]`.
    #[inline]
    pub fn floor_uvs(&self) -> WorldQuadUvs {
        self.floor_uvs
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

    /// Ceiling UVs `[NW, NE, SE, SW]`.
    #[inline]
    pub fn ceiling_uvs(&self) -> WorldQuadUvs {
        self.ceiling_uvs
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
    uvs: WorldQuadUvs,
}

impl WorldWall {
    fn decode(bytes: &[u8]) -> Self {
        let has_v2_uvs = bytes.len() >= WORLD_V2_WALL_RECORD_SIZE;
        Self {
            direction: bytes[0],
            flags: bytes[1],
            material: read_u16(bytes, 4),
            heights: read_i32x4(bytes, 8),
            uvs: if has_v2_uvs {
                read_world_uvs(bytes, 24)
            } else {
                WorldQuadUvs::new(psxed_format::world::WALL_UVS)
            },
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

    /// Wall UVs `[bottom-left, bottom-right, top-right, top-left]`.
    #[inline]
    pub fn uvs(&self) -> WorldQuadUvs {
        self.uvs
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

fn checked_table_bytes(count: u16, stride: usize) -> Result<usize, ParseError> {
    (count as usize)
        .checked_mul(stride)
        .ok_or(ParseError::TableOverflow)
}

fn take_table<'a>(bytes: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8], ParseError> {
    let end = offset.checked_add(len).ok_or(ParseError::TableOverflow)?;
    if end > bytes.len() {
        return Err(ParseError::TableOverflow);
    }
    let table = &bytes[*offset..end];
    *offset = end;
    Ok(table)
}

fn validate_model_parts(
    parts: &[u8],
    joint_count: u16,
    material_count: u16,
    vertex_count: u16,
    face_count: u16,
) -> Result<(), ParseError> {
    let size = psxed_format::model::PartRecord::SIZE;
    let count = parts.len() / size;
    for index in 0..count {
        let base = index * size;
        let joint_index = read_u16(parts, base);
        let first_vertex = read_u16(parts, base + 2);
        let part_vertex_count = read_u16(parts, base + 4);
        let first_face = read_u16(parts, base + 6);
        let part_face_count = read_u16(parts, base + 8);
        let material_index = read_u16(parts, base + 10);

        if joint_index != psxed_format::model::NO_JOINT && joint_index >= joint_count {
            return Err(ParseError::InvalidModelLayout);
        }
        if material_index >= material_count {
            return Err(ParseError::InvalidModelLayout);
        }
        let Some(vertex_end) = first_vertex.checked_add(part_vertex_count) else {
            return Err(ParseError::InvalidModelLayout);
        };
        if vertex_end > vertex_count {
            return Err(ParseError::InvalidModelLayout);
        }
        let Some(face_end) = first_face.checked_add(part_face_count) else {
            return Err(ParseError::InvalidModelLayout);
        };
        if face_end > face_count {
            return Err(ParseError::InvalidModelLayout);
        }
    }
    Ok(())
}

fn validate_model_faces(faces: &[u8], vertex_count: u16) -> Result<(), ParseError> {
    let size = psxed_format::model::FACE_RECORD_SIZE;
    let count = faces.len() / size;
    for index in 0..count {
        let base = index * size;
        let a = read_u16(faces, base);
        let b = read_u16(faces, base + 4);
        let c = read_u16(faces, base + 8);
        if a >= vertex_count || b >= vertex_count || c >= vertex_count {
            return Err(ParseError::InvalidModelLayout);
        }
    }
    Ok(())
}

#[inline]
fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

#[inline]
fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([bytes[offset], bytes[offset + 1]])
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

#[inline]
fn read_world_uvs(bytes: &[u8], offset: usize) -> WorldQuadUvs {
    WorldQuadUvs::new([
        (bytes[offset], bytes[offset + 1]),
        (bytes[offset + 2], bytes[offset + 3]),
        (bytes[offset + 4], bytes[offset + 5]),
        (bytes[offset + 6], bytes[offset + 7]),
    ])
}

#[inline]
fn read_world_surface_light(bytes: &[u8], offset: usize) -> WorldSurfaceLight {
    WorldSurfaceLight::new([
        [bytes[offset], bytes[offset + 1], bytes[offset + 2]],
        [bytes[offset + 3], bytes[offset + 4], bytes[offset + 5]],
        [bytes[offset + 6], bytes[offset + 7], bytes[offset + 8]],
        [bytes[offset + 9], bytes[offset + 10], bytes[offset + 11]],
    ])
}

fn validate_sector_wall_ranges(
    sectors: &[u8],
    sector_record_size: usize,
    wall_count: u16,
) -> Result<(), ParseError> {
    let size = sector_record_size;
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
extern crate std;

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

    /// Pin the `.psxw` parser to the versions it intentionally
    /// supports. v1 is legacy compatibility and v2 is current; any
    /// newer blob must be rejected.
    #[test]
    fn world_rejects_unknown_version() {
        let mut bad = [0u8; 12];
        bad[0..4].copy_from_slice(&psxed_format::world::MAGIC);
        bad[4..6].copy_from_slice(&4u16.to_le_bytes());
        // Payload length 0 -- won't matter; version check fires first.
        bad[8..12].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            World::from_bytes(&bad),
            Err(ParseError::UnsupportedVersion(4))
        ));
    }

    /// Sizes the cooker / runtime have agreed on for world formats.
    /// Drift would invalidate every committed `.psxw` blob, so
    /// pin them at the format crate's records, not the wire.
    #[test]
    fn world_record_sizes_match_contract() {
        assert_eq!(psxed_format::world::WorldHeader::SIZE, 20);
        assert_eq!(psxed_format::world::QuadUvRecord::SIZE, 8);
        assert_eq!(psxed_format::world::SurfaceLightRecord::SIZE, 12);
        assert_eq!(WORLD_V2_SECTOR_RECORD_SIZE, 60);
        assert_eq!(WORLD_V2_WALL_RECORD_SIZE, 32);
        assert_eq!(psxed_format::world::SectorRecord::SIZE, 60);
        assert_eq!(psxed_format::world::WallRecord::SIZE, 32);
    }

    #[test]
    fn world_v1_synthesizes_default_uvs() {
        use psxed_format::world;

        const WORLD_V1_LEN: usize = psxed_format::AssetHeader::SIZE
            + psxed_format::world::WorldHeader::SIZE
            + WORLD_V1_SECTOR_RECORD_SIZE;
        let payload_len = (world::WorldHeader::SIZE + WORLD_V1_SECTOR_RECORD_SIZE) as u32;
        let mut buf = [0u8; WORLD_V1_LEN];
        buf[0..4].copy_from_slice(&world::MAGIC);
        buf[4..6].copy_from_slice(&world::VERSION_V1.to_le_bytes());
        buf[8..12].copy_from_slice(&payload_len.to_le_bytes());
        buf[12..14].copy_from_slice(&1u16.to_le_bytes());
        buf[14..16].copy_from_slice(&1u16.to_le_bytes());
        buf[16..20].copy_from_slice(&world::SECTOR_SIZE.to_le_bytes());
        buf[20..22].copy_from_slice(&1u16.to_le_bytes());

        let sector = psxed_format::AssetHeader::SIZE + world::WorldHeader::SIZE;
        buf[sector] = world::sector_flags::HAS_FLOOR;
        buf[sector + 4..sector + 6].copy_from_slice(&0u16.to_le_bytes());
        buf[sector + 6..sector + 8].copy_from_slice(&world::NO_MATERIAL.to_le_bytes());

        let world = World::from_bytes(&buf).expect("legacy world parses");
        let sector = world.sector(0, 0).unwrap();
        assert_eq!(sector.floor_uvs().corners(), psxed_format::world::FLOOR_UVS);
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
    fn model_round_trip_minimal_textured_part() {
        use psxed_format::model;

        let payload_len = model::ModelHeader::SIZE
            + model::JointRecord::SIZE
            + model::MaterialRecord::SIZE
            + model::PartRecord::SIZE
            + 3 * model::VERTEX_RECORD_SIZE
            + model::FACE_RECORD_SIZE;
        let mut buf = std::vec::Vec::new();
        buf.extend_from_slice(&model::MAGIC);
        buf.extend_from_slice(&model::VERSION.to_le_bytes());
        buf.extend_from_slice(&(model::flags::HAS_UVS | model::flags::RIGID_SKINNED).to_le_bytes());
        buf.extend_from_slice(&(payload_len as u32).to_le_bytes());

        buf.extend_from_slice(&1u16.to_le_bytes()); // joints
        buf.extend_from_slice(&1u16.to_le_bytes()); // parts
        buf.extend_from_slice(&3u16.to_le_bytes()); // vertices
        buf.extend_from_slice(&1u16.to_le_bytes()); // faces
        buf.extend_from_slice(&1u16.to_le_bytes()); // materials
        buf.extend_from_slice(&128u16.to_le_bytes());
        buf.extend_from_slice(&128u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());

        buf.extend_from_slice(&model::NO_JOINT.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&[255, 255, 255, 255]);
        for value in [0u16, 0, 3, 0, 1, 0] {
            buf.extend_from_slice(&value.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes());

        for (x, y) in [(0i16, 0i16), (4096, 0), (0, 4096)] {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&0i16.to_le_bytes());
            // joint1 sentinel + blend=0 means single-bone vertex.
            buf.push(psxed_format::model::NO_JOINT8);
            buf.push(0);
        }
        for (index, u, v) in [(0u16, 0u8, 0u8), (1, 127, 0), (2, 0, 127)] {
            buf.extend_from_slice(&index.to_le_bytes());
            buf.push(u);
            buf.push(v);
        }

        let model = Model::from_bytes(&buf).expect("parse model");
        assert_eq!(model.joint_count(), 1);
        assert_eq!(model.part_count(), 1);
        assert_eq!(model.vertex_count(), 3);
        assert_eq!(model.face_count(), 1);
        assert_eq!(model.texture_width(), 128);
        assert_eq!(
            model.local_to_world_q12(),
            psxed_format::model::DEFAULT_LOCAL_TO_WORLD_Q12
        );
        assert_eq!(model.joint(0).unwrap().parent(), None);
        assert_eq!(
            model.material(0).unwrap().base_color(),
            [255, 255, 255, 255]
        );
        assert_eq!(model.part(0).unwrap().face_count(), 1);
        assert_eq!(model.vertex(1).unwrap().position, Vec3I16::new(4096, 0, 0));
        let face = model.face(0).unwrap();
        assert_eq!(face.corners[0].vertex_index, 0);
        assert_eq!(face.corners[1].vertex_index, 1);
        assert_eq!(face.corners[1].uv, (127, 0));
        assert_eq!(face.corners[2].vertex_index, 2);
        assert_eq!(face.corners[2].uv, (0, 127));
    }

    #[test]
    fn animation_round_trip_pose_table() {
        use psxed_format::animation;

        let payload_len = animation::AnimationHeader::SIZE + 2 * animation::POSE_RECORD_SIZE;
        let mut buf = std::vec::Vec::new();
        buf.extend_from_slice(&animation::MAGIC);
        buf.extend_from_slice(&animation::VERSION.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // joints
        buf.extend_from_slice(&2u16.to_le_bytes()); // frames
        buf.extend_from_slice(&15u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());

        for frame in 0..2i32 {
            for value in [4096i16, 0, 0, 0, 4096, 0, 0, 0, 4096] {
                buf.extend_from_slice(&value.to_le_bytes());
            }
            for value in [frame * 100, frame * 200, frame * 300] {
                buf.extend_from_slice(&value.to_le_bytes());
            }
        }

        let animation = Animation::from_bytes(&buf).expect("parse animation");
        assert_eq!(animation.joint_count(), 1);
        assert_eq!(animation.frame_count(), 2);
        assert_eq!(animation.sample_rate_hz(), 15);
        assert_eq!(animation.phase_step_q12(30), 0x0800);
        assert_eq!(animation.phase_at_tick_q12(2, 30), 0x1000);
        let pose = animation.pose(1, 0).unwrap();
        assert_eq!(pose.matrix[0][0], 4096);
        assert_eq!(pose.translation, Vec3I32::new(100, 200, 300));
    }

    #[test]
    fn animation_looped_pose_interpolates_q12_phase() {
        use psxed_format::animation;

        let payload_len = animation::AnimationHeader::SIZE + 3 * animation::POSE_RECORD_SIZE;
        let mut buf = std::vec::Vec::new();
        buf.extend_from_slice(&animation::MAGIC);
        buf.extend_from_slice(&animation::VERSION.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // joints
        buf.extend_from_slice(&3u16.to_le_bytes()); // frames: first, middle, duplicate first
        buf.extend_from_slice(&15u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());

        for (m00, tx, ty, tz) in [
            (4096i16, 0i32, 0i32, 0i32),
            (2048i16, 100i32, -100i32, 50i32),
            (4096i16, 0i32, 0i32, 0i32),
        ] {
            for value in [m00, 0, 0, 0, 4096, 0, 0, 0, 4096] {
                buf.extend_from_slice(&value.to_le_bytes());
            }
            for value in [tx, ty, tz] {
                buf.extend_from_slice(&value.to_le_bytes());
            }
        }

        let animation = Animation::from_bytes(&buf).expect("parse animation");
        let halfway = animation.pose_looped_q12(0x0800, 0).unwrap();
        assert_eq!(halfway.matrix[0][0], 3072);
        assert_eq!(halfway.translation, Vec3I32::new(50, -50, 25));

        let wrapped = animation.pose_looped_q12(0x1800, 0).unwrap();
        assert_eq!(wrapped.matrix[0][0], 3072);
        assert_eq!(wrapped.translation, Vec3I32::new(50, -50, 25));
    }

    #[test]
    fn world_round_trip_1x1_with_wall() {
        use psxed_format::world;

        const SURFACE_LIGHT_RECORD_COUNT: usize = 3;
        const WORLD_ROUND_TRIP_LEN: usize = psxed_format::AssetHeader::SIZE
            + psxed_format::world::WorldHeader::SIZE
            + psxed_format::world::SectorRecord::SIZE
            + psxed_format::world::WallRecord::SIZE
            + SURFACE_LIGHT_RECORD_COUNT * psxed_format::world::SurfaceLightRecord::SIZE;
        let payload_len = (world::WorldHeader::SIZE
            + world::SectorRecord::SIZE
            + world::WallRecord::SIZE
            + SURFACE_LIGHT_RECORD_COUNT * world::SurfaceLightRecord::SIZE)
            as u32;
        let mut buf = [0u8; WORLD_ROUND_TRIP_LEN];
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
        buf[29] = world::world_flags::FOG_ENABLED | world::world_flags::STATIC_VERTEX_LIGHTING;
        buf[30..32].copy_from_slice(&(SURFACE_LIGHT_RECORD_COUNT as u16).to_le_bytes());

        let sector = 12 + world::WorldHeader::SIZE;
        buf[sector] = world::sector_flags::HAS_FLOOR | world::sector_flags::FLOOR_WALKABLE;
        buf[sector + 1] = world::split::NORTH_WEST_SOUTH_EAST;
        buf[sector + 4..sector + 6].copy_from_slice(&0u16.to_le_bytes());
        buf[sector + 6..sector + 8].copy_from_slice(&world::NO_MATERIAL.to_le_bytes());
        buf[sector + 8..sector + 10].copy_from_slice(&0u16.to_le_bytes());
        buf[sector + 10..sector + 12].copy_from_slice(&1u16.to_le_bytes());
        for (i, (u, v)) in world::FLOOR_UVS.iter().copied().enumerate() {
            buf[sector + 44 + i * 2] = u;
            buf[sector + 45 + i * 2] = v;
        }
        let wall = sector + world::SectorRecord::SIZE;
        buf[wall] = world::direction::NORTH;
        buf[wall + 1] = world::wall_flags::SOLID;
        buf[wall + 4..wall + 6].copy_from_slice(&1u16.to_le_bytes());
        buf[wall + 8..wall + 12].copy_from_slice(&0i32.to_le_bytes());
        buf[wall + 12..wall + 16].copy_from_slice(&0i32.to_le_bytes());
        buf[wall + 16..wall + 20].copy_from_slice(&1024i32.to_le_bytes());
        buf[wall + 20..wall + 24].copy_from_slice(&1024i32.to_le_bytes());
        for (i, (u, v)) in world::WALL_UVS.iter().copied().enumerate() {
            buf[wall + 24 + i * 2] = u;
            buf[wall + 25 + i * 2] = v;
        }
        let floor_light = [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]];
        let ceiling_light = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
        let wall_light = [[12, 22, 32], [42, 52, 62], [72, 82, 92], [102, 112, 122]];
        let lights = wall + world::WallRecord::SIZE;
        for (record_index, light) in [floor_light, ceiling_light, wall_light].iter().enumerate() {
            for (corner_index, rgb) in light.iter().enumerate() {
                let off =
                    lights + record_index * world::SurfaceLightRecord::SIZE + corner_index * 3;
                buf[off..off + 3].copy_from_slice(rgb);
            }
        }

        let world = World::from_bytes(&buf).expect("parse world");
        assert_eq!(world.width(), 1);
        assert_eq!(world.depth(), 1);
        assert_eq!(world.sector_size(), psxed_format::world::SECTOR_SIZE);
        assert_eq!(world.material_count(), 2);
        assert_eq!(world.wall_count(), 1);
        assert_eq!(
            world.surface_light_count(),
            SURFACE_LIGHT_RECORD_COUNT as u16
        );
        assert_eq!(world.ambient_color(), [32, 32, 40]);
        assert!(world.fog_enabled());
        assert!(world.static_vertex_lighting());

        let sector = world.sector(0, 0).unwrap();
        assert!(sector.has_floor());
        assert!(sector.floor_walkable());
        assert_eq!(sector.floor_material(), Some(0));
        assert_eq!(sector.ceiling_material(), None);
        assert_eq!(sector.wall_count(), 1);
        assert_eq!(sector.floor_uvs().corners(), world::FLOOR_UVS);
        assert_eq!(world.surface_light(0).unwrap().vertex_rgb(), floor_light);
        assert_eq!(world.surface_light(1).unwrap().vertex_rgb(), ceiling_light);

        let wall = world.sector_wall(sector, 0).unwrap();
        assert_eq!(wall.direction(), psxed_format::world::direction::NORTH);
        assert!(wall.solid());
        assert_eq!(wall.material(), 1);
        assert_eq!(wall.heights(), [0, 0, 1024, 1024]);
        assert_eq!(wall.uvs().corners(), world::WALL_UVS);
        assert_eq!(world.surface_light(2).unwrap().vertex_rgb(), wall_light);
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
        let mut buf = [0u8; psxed_format::AssetHeader::SIZE
            + psxed_format::world::WorldHeader::SIZE
            + psxed_format::world::SectorRecord::SIZE];
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
    fn parses_legacy_v1_cooked_blob() {
        // Construct a proper v1 blob programmatically: 2 verts, 0 faces.
        // (0 faces is legal; we just want to check header parses.)
        let mut buf: [u8; 12 + 8 + 2 * 6] = [0; 32];
        buf[0..4].copy_from_slice(&psxed_format::mesh::MAGIC);
        buf[4..6].copy_from_slice(&1u16.to_le_bytes());
        buf[6..8].copy_from_slice(&0u16.to_le_bytes());
        buf[8..12].copy_from_slice(&((8 + 2 * 6) as u32).to_le_bytes());
        buf[12..14].copy_from_slice(&2u16.to_le_bytes());
        buf[14..16].copy_from_slice(&0u16.to_le_bytes());
        // Reserved already zero.
        // Vert 0 = (0x0100, 0, 0) -- X at offset 20..22.
        buf[20..22].copy_from_slice(&0x0100_i16.to_le_bytes());
        // Vert 1 = (0, 0x0200, 0) -- Y at offset 28..30
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

    #[test]
    fn parses_v2_u16_indices() {
        const VERTS: usize = 260;
        const FACES: usize = 1;
        const PAYLOAD_LEN: usize = psxed_format::mesh::MeshHeader::SIZE + VERTS * 6 + FACES * 6;
        const INDEX_OFFSET: usize =
            psxed_format::AssetHeader::SIZE + psxed_format::mesh::MeshHeader::SIZE + VERTS * 6;

        let mut buf = [0u8; psxed_format::AssetHeader::SIZE + PAYLOAD_LEN];
        buf[0..4].copy_from_slice(&psxed_format::mesh::MAGIC);
        buf[4..6].copy_from_slice(&psxed_format::mesh::VERSION.to_le_bytes());
        buf[6..8].copy_from_slice(&0u16.to_le_bytes());
        buf[8..12].copy_from_slice(&(PAYLOAD_LEN as u32).to_le_bytes());
        buf[12..14].copy_from_slice(&(VERTS as u16).to_le_bytes());
        buf[14..16].copy_from_slice(&(FACES as u16).to_le_bytes());

        buf[INDEX_OFFSET..INDEX_OFFSET + 2].copy_from_slice(&0u16.to_le_bytes());
        buf[INDEX_OFFSET + 2..INDEX_OFFSET + 4].copy_from_slice(&255u16.to_le_bytes());
        buf[INDEX_OFFSET + 4..INDEX_OFFSET + 6].copy_from_slice(&259u16.to_le_bytes());

        let mesh = Mesh::from_bytes(&buf).expect("parse v2 mesh");
        assert_eq!(mesh.vert_count(), VERTS as u16);
        assert_eq!(mesh.face(0), (0, 255, 259));
    }
}
