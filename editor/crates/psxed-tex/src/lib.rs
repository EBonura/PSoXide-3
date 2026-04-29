//! PNG/JPG → PSXT texture converter.
//!
//! Runs the standard PSX-era texture-cook pipeline:
//!
//! 1. **Decode** the source image (any format the `image` crate
//!    supports; we compile in PNG + JPG + BMP).
//! 2. **Crop** to an optional sub-rectangle of the source.
//! 3. **Resample** to the target texel dimensions. Uses Lanczos3
//!    for downscale — the PSX-era "bilinear" look but with less
//!    ringing.
//! 4. **Quantise** to `N` colours via median-cut, producing a
//!    palette + per-pixel index table.
//! 5. **Pack** the indices into the nibble-order the PSX GPU reads
//!    (4 texels per halfword for 4bpp, 2 for 8bpp).
//! 6. **Emit** the PSXT binary blob matching
//!    [`psxed_format::texture`].
//!
//! # Usage
//!
//! ```ignore
//! use psxed_tex::{Config, CropMode, Depth, Resampler, convert};
//!
//! let src = std::fs::read("brick-wall.jpg").unwrap();
//! let cfg = Config {
//!     width: 64,
//!     height: 64,
//!     depth: Depth::Bit4,
//!     crop: CropMode::CentreSquare,
//!     resampler: Resampler::Lanczos3,
//! };
//! let psxt = convert(&src, &cfg).unwrap();
//! std::fs::write("brick-wall.psxt", psxt).unwrap();
//! ```

use image::{imageops, DynamicImage, GenericImageView};
use psxed_format::texture::{Depth, TextureHeader, MAGIC, VERSION};
use psxed_format::AssetHeader;

/// Rectangle into the source image, pre-resize.
#[derive(Copy, Clone, Debug)]
pub struct CropRect {
    /// X offset from the left of the source, in source pixels.
    pub x: u32,
    /// Y offset from the top.
    pub y: u32,
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
}

impl CropRect {
    /// Compute the centred square crop of a `src_w × src_h` image
    /// — the largest square that fits, positioned in the middle.
    /// Used as the default when the caller hasn't specified an
    /// explicit crop, so arbitrary-aspect sources don't get
    /// distorted by the resize step.
    pub const fn centred_square(src_w: u32, src_h: u32) -> Self {
        let size = if src_w < src_h { src_w } else { src_h };
        Self {
            x: (src_w - size) / 2,
            y: (src_h - size) / 2,
            w: size,
            h: size,
        }
    }
}

/// Resampling kernel used for the resize step.
#[derive(Copy, Clone, Debug)]
pub enum Resampler {
    /// Nearest-neighbour. Blocky, fast. Good for already-pixel-art
    /// sources or when you want a crisp aliased look.
    Nearest,
    /// Lanczos3. Smooth, handles downscale well. Default.
    Lanczos3,
    /// Triangle filter (bilinear). Middle ground.
    Triangle,
}

/// Configuration for one texture conversion.
#[derive(Clone, Debug)]
pub struct Config {
    /// Target texture width in texels. Should be a power of two
    /// for tileable textures; the cooker doesn't enforce this so
    /// UI sprites can be any size.
    pub width: u16,
    /// Target height in texels.
    pub height: u16,
    /// PSX colour depth. 4bpp is the default; 8bpp for detail;
    /// 15bpp for true-colour reference images.
    pub depth: Depth,
    /// Source-side crop behaviour. Default
    /// [`CropMode::CentreSquare`] centre-crops to a square so
    /// arbitrary-aspect sources don't get distorted by the resize
    /// step; [`CropMode::Explicit`] locks to a specific rect;
    /// [`CropMode::None`] stretches the full source to the target
    /// aspect ratio (rarely what you want, but useful for
    /// previously-cropped sources and UI sprites).
    pub crop: CropMode,
    /// Resampling kernel. `Lanczos3` is a good default.
    pub resampler: Resampler,
}

/// How the cooker should handle source aspect vs target aspect.
#[derive(Copy, Clone, Debug, Default)]
pub enum CropMode {
    /// Centre-crop to the largest square that fits — the default.
    /// Guarantees no aspect distortion for arbitrary-aspect sources.
    #[default]
    CentreSquare,
    /// Use the caller-specified rect.
    Explicit(CropRect),
    /// No crop — resize-stretch the full source. Produces distorted
    /// output for non-square sources.
    None,
}

/// Errors the cooker can surface.
#[derive(Debug)]
pub enum Error {
    /// Source image couldn't be decoded.
    Decode(image::ImageError),
    /// Crop rectangle extends past source bounds.
    CropOutOfBounds {
        /// The offending crop.
        crop: CropRect,
        /// Source image dimensions.
        source: (u32, u32),
    },
    /// Target dimensions are zero.
    ZeroSize,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Decode(e) => write!(f, "decode source image: {e}"),
            Error::CropOutOfBounds { crop, source } => write!(
                f,
                "crop rect {:?}+{}×{} extends past {}×{} source",
                (crop.x, crop.y),
                crop.w,
                crop.h,
                source.0,
                source.1,
            ),
            Error::ZeroSize => write!(f, "target width/height is zero"),
        }
    }
}

impl std::error::Error for Error {}

/// End-to-end convert: bytes of a PNG/JPG/BMP → bytes of a PSXT
/// blob. The returned `Vec<u8>` is ready to write to disk and
/// `include_bytes!` into a game.
pub fn convert(src: &[u8], cfg: &Config) -> Result<Vec<u8>, Error> {
    if cfg.width == 0 || cfg.height == 0 {
        return Err(Error::ZeroSize);
    }
    let img = image::load_from_memory(src).map_err(Error::Decode)?;
    let (sw, sh) = img.dimensions();
    let crop_rect = match cfg.crop {
        CropMode::Explicit(c) => {
            if c.x + c.w > sw || c.y + c.h > sh {
                return Err(Error::CropOutOfBounds {
                    crop: c,
                    source: (sw, sh),
                });
            }
            Some(c)
        }
        CropMode::CentreSquare => Some(CropRect::centred_square(sw, sh)),
        CropMode::None => None,
    };
    let cropped = if let Some(c) = crop_rect {
        DynamicImage::ImageRgba8(imageops::crop_imm(&img, c.x, c.y, c.w, c.h).to_image())
    } else {
        img
    };
    let resized = resize(&cropped, cfg.width as u32, cfg.height as u32, cfg.resampler);

    encode_psxt(&resized, cfg.width, cfg.height, cfg.depth)
}

/// Resize `img` to `w × h` using the chosen resampling kernel.
fn resize(img: &DynamicImage, w: u32, h: u32, r: Resampler) -> image::RgbaImage {
    let filter = match r {
        Resampler::Nearest => imageops::FilterType::Nearest,
        Resampler::Lanczos3 => imageops::FilterType::Lanczos3,
        Resampler::Triangle => imageops::FilterType::Triangle,
    };
    img.resize_exact(w, h, filter).to_rgba8()
}

/// Heart of the pipeline — take a resized RgbaImage and emit the
/// full PSXT blob (12-byte AssetHeader + 16-byte TextureHeader +
/// pixel block + CLUT block).
fn encode_psxt(
    img: &image::RgbaImage,
    width: u16,
    height: u16,
    depth: Depth,
) -> Result<Vec<u8>, Error> {
    // Collect pixels as RGB triples (alpha is dropped; the PSX mask
    // bit is a separate concern we don't expose yet).
    let pixels: Vec<[u8; 3]> = img.pixels().map(|p| [p[0], p[1], p[2]]).collect();

    match depth {
        Depth::Bit4 | Depth::Bit8 => {
            let n_entries = depth.clut_entries().unwrap() as usize;
            let (palette, indices) = median_cut_quantize(&pixels, n_entries);
            let pixel_halfwords = pack_indices(&indices, width, height, depth);
            let clut_halfwords = encode_clut(&palette, n_entries);
            Ok(assemble_blob(
                width,
                height,
                depth,
                n_entries as u16,
                &pixel_halfwords,
                &clut_halfwords,
            ))
        }
        Depth::Bit15 => {
            // Direct colour — each pixel becomes one Color555 halfword.
            let pixel_halfwords: Vec<u16> = pixels
                .iter()
                .map(|rgb| rgb_to_555(rgb[0], rgb[1], rgb[2]))
                .collect();
            Ok(assemble_blob(
                width,
                height,
                depth,
                0,
                &pixel_halfwords,
                &[],
            ))
        }
    }
}

/// Median-cut colour quantisation. Input: per-pixel RGB. Output:
/// (palette, indices) where each index is `< n_entries`.
///
/// Algorithm: start with one box holding every unique colour.
/// Repeatedly split the box with the largest RGB range along its
/// longest channel's median until we have `n_entries` boxes. Each
/// box contributes its per-channel mean as the final palette entry.
///
/// Output is deterministic for a given input — `BTreeMap` + stable
/// sorts — so repeated builds produce byte-identical PSXT blobs.
fn median_cut_quantize(pixels: &[[u8; 3]], n_entries: usize) -> (Vec<[u8; 3]>, Vec<u8>) {
    assert!(
        n_entries.is_power_of_two() && (2..=256).contains(&n_entries),
        "n_entries must be a power of two in [2, 256]"
    );

    // Start with one box containing all pixels (as owned Vec so we
    // can sort in place per split).
    let mut boxes: Vec<Vec<[u8; 3]>> = vec![pixels.to_vec()];

    while boxes.len() < n_entries {
        // Find the box with the largest range along any channel.
        let (idx, axis) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() >= 2)
            .map(|(i, b)| {
                let (axis, range) = widest_axis(b);
                (i, axis, range)
            })
            .max_by_key(|&(_, _, r)| r)
            .map(|(i, a, _)| (i, a))
            .unwrap_or_else(|| {
                // Can't split any further (every box has <2 colours).
                // Pad with duplicates of existing entries to hit
                // n_entries — quantiser caller expects exactly that count.
                (0, 0)
            });

        let b = &mut boxes[idx];
        if b.len() < 2 {
            // Pad by duplicating. We only hit this when the input
            // has fewer than n_entries unique colours.
            let dup = b.clone();
            boxes.push(dup);
            continue;
        }

        // Sort along the chosen axis, split at the median.
        b.sort_by_key(|c| c[axis]);
        let mid = b.len() / 2;
        let hi = b.split_off(mid);
        boxes.push(hi);
    }

    // Compute each box's mean → palette entry.
    let palette: Vec<[u8; 3]> = boxes
        .iter()
        .map(|b| {
            let (mut sr, mut sg, mut sb) = (0u32, 0u32, 0u32);
            for c in b {
                sr += c[0] as u32;
                sg += c[1] as u32;
                sb += c[2] as u32;
            }
            let n = b.len().max(1) as u32;
            [(sr / n) as u8, (sg / n) as u8, (sb / n) as u8]
        })
        .collect();

    // Map each source pixel to nearest palette entry.
    let indices: Vec<u8> = pixels.iter().map(|p| nearest_index(p, &palette)).collect();

    (palette, indices)
}

/// Return the (axis_index, range) of the colour channel with the
/// widest spread in the given box.
fn widest_axis(b: &[[u8; 3]]) -> (usize, u32) {
    let mut lo = [u8::MAX; 3];
    let mut hi = [0u8; 3];
    for c in b {
        for ch in 0..3 {
            lo[ch] = lo[ch].min(c[ch]);
            hi[ch] = hi[ch].max(c[ch]);
        }
    }
    let ranges = [
        hi[0] as u32 - lo[0] as u32,
        hi[1] as u32 - lo[1] as u32,
        hi[2] as u32 - lo[2] as u32,
    ];
    let (i, &r) = ranges.iter().enumerate().max_by_key(|(_, r)| *r).unwrap();
    (i, r)
}

/// Linear scan of palette, squared-distance metric. `256 * 64 ≈
/// 16K ops per image — fast enough that we don't need a kd-tree
/// for texture sizes the PSX can actually use.
fn nearest_index(p: &[u8; 3], palette: &[[u8; 3]]) -> u8 {
    let mut best = 0u8;
    let mut best_d = u32::MAX;
    for (i, c) in palette.iter().enumerate() {
        let dr = p[0] as i32 - c[0] as i32;
        let dg = p[1] as i32 - c[1] as i32;
        let db = p[2] as i32 - c[2] as i32;
        let d = (dr * dr + dg * dg + db * db) as u32;
        if d < best_d {
            best_d = d;
            best = i as u8;
        }
    }
    best
}

/// Pack a linear `indices` buffer into halfword-aligned PSX
/// texture memory. Each output u16 holds N texels where N matches
/// the depth's `texels_per_halfword`. Row ends pad up to a full
/// halfword so every row starts on a halfword boundary.
fn pack_indices(indices: &[u8], width: u16, height: u16, depth: Depth) -> Vec<u16> {
    let hw_per_row = TextureHeader::halfwords_per_row(depth, width);
    let total = (hw_per_row as usize) * (height as usize);
    let mut out = vec![0u16; total];

    match depth {
        Depth::Bit4 => {
            for y in 0..height {
                for x in 0..width {
                    let idx = indices[(y as usize) * (width as usize) + (x as usize)] & 0x0F;
                    let hw = (y as usize) * (hw_per_row as usize) + (x as usize) / 4;
                    let shift = (x & 3) * 4;
                    out[hw] |= (idx as u16) << shift;
                }
            }
        }
        Depth::Bit8 => {
            for y in 0..height {
                for x in 0..width {
                    let idx = indices[(y as usize) * (width as usize) + (x as usize)];
                    let hw = (y as usize) * (hw_per_row as usize) + (x as usize) / 2;
                    let shift = (x & 1) * 8;
                    out[hw] |= (idx as u16) << shift;
                }
            }
        }
        Depth::Bit15 => unreachable!("indexed packer called with 15bpp depth"),
    }
    out
}

/// Encode a palette as RGB555 halfwords. If `entries < n_entries`
/// (rare — only on inputs with <16 unique colours), pad with zero
/// entries to hit the required CLUT row width.
fn encode_clut(palette: &[[u8; 3]], n_entries: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n_entries);
    for c in palette {
        out.push(rgb_to_555(c[0], c[1], c[2]));
    }
    while out.len() < n_entries {
        out.push(0);
    }
    out
}

/// 8-bit per channel RGB → PSX 5-5-5 + mask bit. Low 5 bits = red,
/// next 5 = green, next 5 = blue, bit 15 = mask (kept zero here —
/// masks are a render-time concern, not asset-time).
fn rgb_to_555(r: u8, g: u8, b: u8) -> u16 {
    let r5 = (r as u16 >> 3) & 0x1F;
    let g5 = (g as u16 >> 3) & 0x1F;
    let b5 = (b as u16 >> 3) & 0x1F;
    r5 | (g5 << 5) | (b5 << 10)
}

/// Concatenate the header + pixel + CLUT blocks into the final
/// PSXT byte sequence.
fn assemble_blob(
    width: u16,
    height: u16,
    depth: Depth,
    clut_entries: u16,
    pixel_hw: &[u16],
    clut_hw: &[u16],
) -> Vec<u8> {
    let pixel_bytes = (pixel_hw.len() * 2) as u32;
    let clut_bytes = (clut_hw.len() * 2) as u32;
    let payload_len = (TextureHeader::SIZE as u32) + pixel_bytes + clut_bytes;

    let mut out = Vec::with_capacity(AssetHeader::SIZE + payload_len as usize);
    // AssetHeader.
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // flags — reserved
    out.extend_from_slice(&payload_len.to_le_bytes());
    // TextureHeader.
    out.push(depth as u8);
    out.push(0); // _pad
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&clut_entries.to_le_bytes());
    out.extend_from_slice(&pixel_bytes.to_le_bytes());
    out.extend_from_slice(&clut_bytes.to_le_bytes());
    // Pixel halfwords.
    for hw in pixel_hw {
        out.extend_from_slice(&hw.to_le_bytes());
    }
    // CLUT halfwords.
    for hw in clut_hw {
        out.extend_from_slice(&hw.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centred_square_landscape() {
        // 3:2 landscape source → square is the short side (height),
        // centred horizontally.
        let c = CropRect::centred_square(6000, 4000);
        assert_eq!(c.w, 4000);
        assert_eq!(c.h, 4000);
        assert_eq!(c.x, 1000);
        assert_eq!(c.y, 0);
    }

    #[test]
    fn centred_square_portrait() {
        // 3:4 portrait → square is the short side (width), centred
        // vertically.
        let c = CropRect::centred_square(3000, 4000);
        assert_eq!(c.w, 3000);
        assert_eq!(c.h, 3000);
        assert_eq!(c.x, 0);
        assert_eq!(c.y, 500);
    }

    #[test]
    fn centred_square_already_square_is_identity() {
        let c = CropRect::centred_square(512, 512);
        assert_eq!(c.w, 512);
        assert_eq!(c.h, 512);
        assert_eq!(c.x, 0);
        assert_eq!(c.y, 0);
    }

    #[test]
    fn rgb_conversion_full_scale() {
        assert_eq!(rgb_to_555(0, 0, 0), 0);
        assert_eq!(rgb_to_555(255, 255, 255), 0x7FFF);
        assert_eq!(rgb_to_555(255, 0, 0), 0x001F);
        assert_eq!(rgb_to_555(0, 255, 0), 0x03E0);
        assert_eq!(rgb_to_555(0, 0, 255), 0x7C00);
    }

    #[test]
    fn pack_4bpp_nibble_order() {
        // 4 texels packed into one halfword: nibble 0 = leftmost.
        let indices = vec![0x1, 0x2, 0x3, 0x4];
        let packed = pack_indices(&indices, 4, 1, Depth::Bit4);
        assert_eq!(packed, vec![0x4321]);
    }

    #[test]
    fn pack_8bpp_byte_order() {
        let indices = vec![0x12, 0x34];
        let packed = pack_indices(&indices, 2, 1, Depth::Bit8);
        assert_eq!(packed, vec![0x3412]);
    }

    #[test]
    fn quantiser_exactly_n_entries_when_enough_colours() {
        // 16 distinct greys → median-cut should produce ~16 entries.
        let pixels: Vec<[u8; 3]> = (0..16).map(|i| [i * 16, i * 16, i * 16]).collect();
        let (palette, indices) = median_cut_quantize(&pixels, 16);
        assert_eq!(palette.len(), 16);
        assert_eq!(indices.len(), 16);
        for &idx in &indices {
            assert!((idx as usize) < palette.len());
        }
    }

    #[test]
    fn end_to_end_encoded_blob_roundtrips_via_header_offsets() {
        // Small synthetic PNG — solid red 4×4.
        let mut buf = Vec::new();
        {
            let img = image::RgbaImage::from_fn(4, 4, |_, _| image::Rgba([255, 0, 0, 255]));
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .unwrap();
        }
        let cfg = Config {
            width: 4,
            height: 4,
            depth: Depth::Bit4,
            crop: CropMode::None,
            resampler: Resampler::Nearest,
        };
        let psxt = convert(&buf, &cfg).unwrap();

        // Header sanity.
        assert_eq!(&psxt[0..4], b"PSXT");
        assert_eq!(u16::from_le_bytes([psxt[4], psxt[5]]), 1);
        // TextureHeader starts at offset 12.
        assert_eq!(psxt[12], 4); // depth
        assert_eq!(u16::from_le_bytes([psxt[14], psxt[15]]), 4); // width
        assert_eq!(u16::from_le_bytes([psxt[16], psxt[17]]), 4); // height
        assert_eq!(u16::from_le_bytes([psxt[18], psxt[19]]), 16); // clut_entries
                                                                  // pixel_bytes = halfwords_per_row(4bpp, 4) × height × 2 = 1 × 4 × 2 = 8
        assert_eq!(
            u32::from_le_bytes([psxt[20], psxt[21], psxt[22], psxt[23]]),
            8
        );
        // clut_bytes = 16 × 2 = 32
        assert_eq!(
            u32::from_le_bytes([psxt[24], psxt[25], psxt[26], psxt[27]]),
            32
        );
    }
}
