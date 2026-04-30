//! On-disk layout for cooked 2D textures (`.psxt` files).
//!
//! A PSXT blob carries a VRAM-ready texture plus (for indexed
//! formats) its CLUT. The runtime parser pulls out two slices --
//! pixel halfwords and CLUT halfwords -- and hands them to
//! [`upload_16bpp`][psx-vram::upload_16bpp] / [`upload_clut`]
//! without copying. The cooking step (host-side in
//! `psxed-tex`) has already packed the indices into the exact
//! nibble-order the GPU expects.
//!
//! # File layout
//!
//! ```text
//!   AssetHeader (12 bytes)
//!     magic       = b"PSXT"
//!     version     = VERSION
//!     flags       = reserved (always 0 in v1)
//!     payload_len = everything after this header
//!
//!   TextureHeader (16 bytes)
//!     depth         u8   -- 4, 8, or 15 bit per texel
//!     _pad          u8   -- zero
//!     width_px      u16  -- texture width in TEXELS
//!     height_px     u16  -- texture height in TEXELS
//!     clut_entries  u16  -- 16 (4bpp), 256 (8bpp), or 0 (15bpp)
//!     pixel_bytes   u32  -- byte count of the pixel block
//!     clut_bytes    u32  -- byte count of the CLUT block (0 for 15bpp)
//!
//!   Pixel data (pixel_bytes bytes)
//!     Halfword-packed LE. At 4bpp, each u16 holds 4 texels with
//!     nibble 0 = leftmost pixel (x & 3 == 0). At 8bpp, 2 texels
//!     per u16, low byte first. At 15bpp, one Color555 per u16.
//!     Rows are padded up to the 4/2/1-texel boundary so every
//!     row starts on a halfword.
//!
//!   CLUT data (clut_bytes bytes, absent if depth == 15)
//!     clut_entries × u16 LE, each an RGB555+M entry.
//! ```
//!
//! All multi-byte integers little-endian. Bytes tightly packed.
//! Pixel data comes before CLUT data so a streaming DMA upload
//! can write the pixels into the tpage slot first and then drop
//! the CLUT into its own slot without seeking back.

/// ASCII magic identifying the `.psxt` format.
pub const MAGIC: [u8; 4] = *b"PSXT";

/// Current texture format revision.
pub const VERSION: u16 = 1;

/// The three bit-depths the PSX GPU supports natively.
///
/// Stored in [`TextureHeader::depth`] as the exact integer (4, 8,
/// or 15) so the byte is self-describing -- no need to cross-
/// reference an enum table to interpret a blob.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Depth {
    /// 4 bits per texel → 16-colour CLUT. The smallest, most
    /// common format for tiled walls / sprites / UI.
    Bit4 = 4,
    /// 8 bits per texel → 256-colour CLUT. Used for
    /// characters / higher-detail surfaces.
    Bit8 = 8,
    /// 15 bits per texel, direct RGB555 + mask bit. No CLUT.
    /// Used sparingly because of the 4× VRAM cost vs 4bpp.
    Bit15 = 15,
}

impl Depth {
    /// Parse a raw byte from a cooked blob. Returns `None` for
    /// any value other than 4, 8, 15.
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            4 => Some(Depth::Bit4),
            8 => Some(Depth::Bit8),
            15 => Some(Depth::Bit15),
            _ => None,
        }
    }

    /// Expected CLUT entry count for this depth, or `None` for 15bpp.
    pub const fn clut_entries(self) -> Option<u16> {
        match self {
            Depth::Bit4 => Some(16),
            Depth::Bit8 => Some(256),
            Depth::Bit15 => None,
        }
    }

    /// Texels per 16-bit halfword.
    pub const fn texels_per_halfword(self) -> u16 {
        match self {
            Depth::Bit4 => 4,
            Depth::Bit8 => 2,
            Depth::Bit15 => 1,
        }
    }
}

/// Byte layout of the texture payload header that follows the
/// common 12-byte `AssetHeader`.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct TextureHeader {
    /// 4, 8, or 15 -- matches [`Depth`] discriminants.
    pub depth: u8,
    /// Alignment padding; writers set to zero, readers ignore.
    pub _pad: u8,
    /// Width in texels (not halfwords).
    pub width_px: u16,
    /// Height in texels.
    pub height_px: u16,
    /// 16, 256, or 0 (for 15bpp).
    pub clut_entries: u16,
    /// Byte count of the pixel block immediately after this header.
    pub pixel_bytes: u32,
    /// Byte count of the CLUT block immediately after the pixels.
    /// Zero for 15bpp.
    pub clut_bytes: u32,
}

impl TextureHeader {
    /// Size of the texture header in bytes (always 16).
    pub const SIZE: usize = 16;

    /// Halfwords per row at the given depth and width. Rows are
    /// padded up to a full halfword -- a 5-texel-wide 4bpp texture
    /// still uses 2 halfwords per row.
    pub const fn halfwords_per_row(depth: Depth, width_px: u16) -> u16 {
        let per_hw = depth.texels_per_halfword();
        width_px.div_ceil(per_hw)
    }
}
