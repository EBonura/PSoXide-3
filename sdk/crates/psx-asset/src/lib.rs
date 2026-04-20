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
    /// `AssetHeader::magic` doesn't match `PSXM`.
    WrongMagic,
    /// Format version newer than this parser supports.
    UnsupportedVersion(u16),
    /// Declared payload_len disagrees with the actual byte length.
    InvalidPayloadLen { declared: u32, actual: usize },
    /// Vertex or face table wouldn't fit in the remaining payload.
    TableOverflow,
}

/// A parsed 3D mesh backed by slices into the caller's cooked blob.
///
/// Cheap to construct (just bounds-checks the header + computes
/// sub-slice offsets). Cheap to pass around — three `&[u8]` slices
/// + a flags `u16` + the face count.
#[derive(Copy, Clone, Debug)]
pub struct Mesh<'a> {
    verts: &'a [u8],
    indices: &'a [u8],
    face_colors: Option<&'a [u8]>,
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
            Some(slice)
        } else {
            None
        };

        Ok(Self {
            verts,
            indices,
            face_colors,
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
