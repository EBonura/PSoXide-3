//! Bitmap-font atlases for the PS1.
//!
//! One crate, two types, three steps:
//!
//! 1. Declare (or pick) a [`BitmapFont`] — a `static` descriptor
//!    that carries glyph dimensions, a 1-bit-per-pixel bitmap, and
//!    layout metadata (advance, line height, bit order). The
//!    built-in fonts in [`fonts`] cover Public-Domain IBM-VGA-style
//!    8×8 for ASCII / Latin-1 / box-drawing.
//!
//! 2. Upload it into VRAM once with [`FontAtlas::upload`] — the
//!    crate picks a sensible atlas layout, expands the 1bpp source
//!    into a 4bpp CLUT texture, uploads that + a two-entry CLUT
//!    (transparent + white), and returns a handle.
//!
//! 3. Call [`FontAtlas::draw_text`] every frame. One GP0 0x64
//!    (textured rectangle) per glyph. The `tint` argument
//!    modulates the white glyph via the PSX's per-texel tint
//!    multiplier, so you can render the same font atlas in any
//!    colour without re-uploading.
//!
//! ## Generic over glyph dimensions
//!
//! Nothing in [`BitmapFont`] or [`FontAtlas`] hard-codes 8×8.
//! Declare a second font at 6×12 or 8×16, drop it into VRAM at a
//! different tpage, and you can mix sizes on the same screen (see
//! the ladder in `sdk/examples/showcase-reference-scene` once it
//! ships). What's size-dependent:
//!
//! - **Atlas layout**: the uploader picks `glyphs_per_row` so the
//!   atlas width stays within one tpage's effective 4bpp span
//!   (64 pixels at 4bpp). Tall fonts just use more rows.
//! - **Upload buffer**: sized at compile time by the caller via
//!   [`FontAtlas::upload`]'s fixed-capacity `packed` argument. The
//!   stack-only nature of the SDK's no-alloc profile means a font
//!   whose atlas doesn't fit in the caller's buffer won't even
//!   compile (with const-generics) — but since each font is a
//!   one-shot upload at game boot, the working set is small.
//!
//! ## Why 4bpp and not 15bpp direct
//!
//! 4bpp uses 1/4 the VRAM of 15bpp and opens the standard PSX
//! "colour a monochrome glyph via tint" trick. A 96-glyph ASCII
//! 8×8 atlas fits in 6×8 halfwords = 96 halfwords = 192 bytes of
//! VRAM. Direct-15bpp would be 768 bytes, and you'd lose the free
//! recolouring.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use psx_hw::gpu::{pack_color, pack_texcoord, pack_vertex, pack_xy};
use psx_io::gpu::{wait_cmd_ready, write_gp0};
use psx_vram::{Clut, Color555, TexDepth, Tpage, VramRect, upload_16bpp, upload_clut};

pub mod fonts;

// ======================================================================
// BitmapFont — the static descriptor
// ======================================================================

/// Bit-packing convention within each bitmap byte.
///
/// - [`BitOrder::Lsb`]: bit 0 of each byte is the **leftmost** pixel.
///   Matches the dhepper/font8x8 convention our built-in fonts use.
/// - [`BitOrder::Msb`]: bit 7 is the leftmost pixel. Matches the IBM
///   VGA BIOS ROM / Linux `fbcon` `font_8x8.c` / GRUB conventions.
///
/// Fonts imported from different sources carry different bit orders;
/// the uploader handles both transparently.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BitOrder {
    /// Bit 0 is the leftmost pixel.
    Lsb,
    /// Bit 7 is the leftmost pixel.
    Msb,
}

/// A bitmap font as a static data descriptor.
///
/// All fields are compile-time constants so fonts can live in
/// `.rodata` and a `BitmapFont` value can be declared as a `const`
/// right next to its bitmap. The type stays size-agnostic — 6×8,
/// 8×8, 8×16, 12×16 all work; the uploader reads `glyph_w` /
/// `glyph_h` and lays out the atlas from there.
///
/// Glyph `i` in the `bitmap` slice occupies
/// `bitmap[i * row_bytes * glyph_h .. i * row_bytes * glyph_h +
/// row_bytes * glyph_h]`, where `row_bytes = ceil(glyph_w / 8)`.
/// For 8-wide glyphs that's one byte per row. Wider glyphs (12,
/// 16 px) use multiple bytes per row, MSB-first within the row
/// (i.e., byte 0 covers columns 0..=7, byte 1 covers columns
/// 8..=15, regardless of [`BitOrder`] within each byte).
///
/// The `first_char` / `glyph_count` window is a codepoint range —
/// code `c` is looked up at offset `c - first_char` in the bitmap
/// as long as `first_char <= c < first_char + glyph_count`.
/// Anything outside falls back to the missing-glyph box.
#[derive(Copy, Clone, Debug)]
pub struct BitmapFont {
    /// Glyph cell width in pixels.
    pub glyph_w: u8,
    /// Glyph cell height in pixels.
    pub glyph_h: u8,
    /// First codepoint covered. For a basic-Latin font this is
    /// usually `0x00` (null) or `0x20` (space).
    pub first_char: u16,
    /// Number of glyphs in this font. The range
    /// `first_char..first_char + glyph_count` is the supported set.
    pub glyph_count: u16,
    /// Row-major bitmap bytes. `glyph_count * glyph_h *
    /// ceil(glyph_w / 8)` bytes total.
    pub bitmap: &'static [u8],
    /// Pixel step between adjacent characters on a text line.
    /// Usually `== glyph_w` for fixed-width fonts.
    pub advance_x: u8,
    /// Pixel step between text lines. Usually `== glyph_h`.
    pub line_height: u8,
    /// Bit packing within each bitmap byte.
    pub bit_order: BitOrder,
}

impl BitmapFont {
    /// Bytes per row of a single glyph in the source bitmap.
    /// Derived from [`BitmapFont::glyph_w`] — 1 byte for ≤ 8-wide
    /// fonts, 2 bytes for 9..16-wide, etc.
    pub const fn row_bytes(&self) -> usize {
        ((self.glyph_w as usize) + 7) / 8
    }

    /// Total bitmap bytes per glyph.
    pub const fn glyph_stride(&self) -> usize {
        self.row_bytes() * self.glyph_h as usize
    }

    /// Fetch row `r` of glyph `i` as 4 bytes LSB-packed little-
    /// endian — the row comes out as a `u32` with pixel 0 in bit
    /// 0, pixel 1 in bit 1, … up to pixel 31 (enough for any
    /// realistic cell width). Handles [`BitOrder`] normalisation
    /// in one place so callers don't have to branch.
    pub fn glyph_row_packed(&self, i: u16, r: u8) -> u32 {
        let base = i as usize * self.glyph_stride() + r as usize * self.row_bytes();
        let mut out: u32 = 0;
        for byte_idx in 0..self.row_bytes() {
            let raw = self.bitmap[base + byte_idx] as u32;
            let normalised = match self.bit_order {
                BitOrder::Lsb => raw,
                BitOrder::Msb => raw.reverse_bits() >> 24,
            };
            out |= normalised << (byte_idx * 8);
        }
        out
    }
}

// ======================================================================
// FontAtlas — the VRAM handle
// ======================================================================

/// A [`BitmapFont`] installed in VRAM and ready to draw from.
///
/// Hold on to the returned handle for the lifetime you want glyphs
/// to stay drawable. Uploading costs ~1 GP0 0xA0 transfer of
/// `glyph_count × glyph_w × glyph_h / 4` bytes (plus 2 halfwords
/// for the CLUT); draw calls cost 4 GP0 words per glyph.
#[derive(Copy, Clone, Debug)]
pub struct FontAtlas {
    font: &'static BitmapFont,
    tpage: Tpage,
    clut: Clut,
    /// How many glyphs per row of the atlas texture. Picked at
    /// upload time so the texture fits within a single 4bpp tpage.
    glyphs_per_row: u16,
}

impl FontAtlas {
    /// Maximum texels a 4bpp tpage covers horizontally. 64 pixels
    /// is the PSX-SPX value (tpage is 256×256 in texel units, but
    /// in 4bpp mode the effective horizontal texel range per page
    /// maps onto 64 VRAM halfwords = 64 × 4 = 256 texels).
    const MAX_ATLAS_W_TEXELS: u16 = 256;
    /// Stack buffer for the packed 4bpp atlas. 2048 halfwords =
    /// 4 KiB — big enough for a 256×32 4bpp atlas, which covers
    /// 128 × 8×8 glyphs with room to spare. Larger fonts will need
    /// a matching bump here.
    const MAX_PACK_HALFWORDS: usize = 2048;

    /// Upload `font` as a 4bpp CLUT texture at `tpage`, with a
    /// 2-entry CLUT (transparent, white) at `clut`.
    ///
    /// The caller picks the tpage / clut locations — typically
    /// `Tpage::new(768, 0, TexDepth::Bit4)` for the standard
    /// off-display region, and a `Clut::new(768, 480)` for the
    /// CLUT row. Both must live inside VRAM and not overlap the
    /// active framebuffer.
    ///
    /// Atlas layout picks `glyphs_per_row` so the texture's pixel
    /// width stays within `MAX_ATLAS_W_TEXELS`. For 8-wide fonts
    /// that's 32 glyphs per row; 16-wide fonts get 16 per row.
    pub fn upload(font: &'static BitmapFont, tpage: Tpage, clut: Clut) -> Self {
        assert!(
            matches!(tpage.depth(), TexDepth::Bit4),
            "FontAtlas::upload requires a 4bpp tpage",
        );

        let glyph_w = font.glyph_w as u16;
        let glyph_h = font.glyph_h as u16;
        let glyph_count = font.glyph_count;

        // Atlas width = glyphs_per_row × glyph_w, capped at the
        // MAX to stay within a single 4bpp tpage.
        let max_cols = Self::MAX_ATLAS_W_TEXELS / glyph_w;
        let glyphs_per_row = glyph_count.min(max_cols);
        let glyph_rows = glyph_count.div_ceil(glyphs_per_row);
        let atlas_w = glyphs_per_row * glyph_w;
        let atlas_h = glyph_rows * glyph_h;

        // Pack 1bpp source → 4bpp VRAM texture into a stack buffer.
        // Each 16-bit halfword holds 4 texels (nibble 0 = leftmost).
        let halfwords_per_row = (atlas_w + 3) / 4;
        let total_halfwords = halfwords_per_row as usize * atlas_h as usize;
        assert!(
            total_halfwords <= Self::MAX_PACK_HALFWORDS,
            "font atlas too large for stack buffer",
        );
        let mut packed = [0u16; Self::MAX_PACK_HALFWORDS];

        for gi in 0..glyph_count {
            let atlas_col = gi % glyphs_per_row;
            let atlas_row = gi / glyphs_per_row;
            let base_x = atlas_col * glyph_w;
            let base_y = atlas_row * glyph_h;
            for row in 0..glyph_h as u8 {
                let row_bits = font.glyph_row_packed(gi, row);
                for col in 0..glyph_w {
                    let bit = (row_bits >> col) & 1;
                    let x = base_x + col;
                    let y = base_y + row as u16;
                    // 4 texels per halfword; pick nibble by x & 3.
                    let hw_idx = y as usize * halfwords_per_row as usize
                        + (x as usize / 4);
                    let nibble_shift = (x & 3) * 4;
                    packed[hw_idx] |= (bit as u16) << nibble_shift;
                }
            }
        }

        let tex_rect = VramRect::new(tpage.x(), tpage.y(), atlas_w, atlas_h);
        // upload_16bpp takes pixel count = w*h, which for a 4bpp
        // texture uploaded as a 16bpp rect works out to
        // halfwords_per_row × atlas_h halfwords = the 4bpp pixel
        // count. Upload is by halfwords regardless of bit-depth;
        // the GPU doesn't inspect bits inside words.
        //
        // VRAM rect semantics expose the *halfword* footprint when
        // depth is 4bpp — so width is `atlas_w / 4`, not `atlas_w`.
        let vram_rect = VramRect::new(tpage.x(), tpage.y(), halfwords_per_row, atlas_h);
        upload_16bpp(vram_rect, &packed[..total_halfwords]);

        // Upload 2-entry CLUT: idx 0 = transparent, idx 1 = white.
        // The white texel will be tinted per-draw_call via the
        // sprite's per-vertex colour (GP0 0x64 tint byte).
        let clut_entries = [
            Color555::TRANSPARENT,
            Color555::rgb5(31, 31, 31),
        ];
        upload_clut(clut, &clut_entries);

        let _ = tex_rect;
        Self {
            font,
            tpage,
            clut,
            glyphs_per_row,
        }
    }

    /// Pixel width of `text` when rendered with this atlas. For a
    /// fixed-width font that's `advance_x × text.len()`.
    pub fn text_width(&self, text: &str) -> u16 {
        (text.len() as u16) * self.font.advance_x as u16
    }

    /// Pixel height of a single text line. Usually `== glyph_h`.
    pub fn line_height(&self) -> u16 {
        self.font.line_height as u16
    }

    /// Draw `text` at screen-space `(x, y)` with the given tint.
    /// `(0x80, 0x80, 0x80)` = unmodulated white; any other value
    /// recolours every glyph via the PSX per-texel multiplier
    /// (`output = texel * tint / 128`).
    ///
    /// Characters outside the font's codepoint range are skipped —
    /// they still advance the cursor so the rest of the string
    /// lines up as the caller intended.
    ///
    /// Iteration is per-`char`, so any `&str` is valid input. A
    /// Latin-1 atlas with `first_char = 0xA0` picks up `é`, `ü`, etc.
    /// automatically through the codepoint-offset lookup.
    ///
    /// Sets the GP0(0xE1) draw-mode tpage to our atlas once at the
    /// start of the call — if the caller was rendering with a
    /// different tpage, they'll need to re-apply theirs after
    /// draw_text returns.
    pub fn draw_text(&self, x: i16, y: i16, text: &str, tint: (u8, u8, u8)) {
        // Our atlas tpage takes over the current draw-mode slot; the
        // per-glyph GP0 0x64 rectangles all sample from it.
        self.tpage.apply_as_draw_mode();

        let font = self.font;
        let advance = font.advance_x as i16;
        let clut_word = self.clut.uv_clut_word();
        let mut cursor_x = x;
        for ch in text.chars() {
            let cp = ch as u32;
            let first = font.first_char as u32;
            let idx_in_font = if cp >= first && cp < first + font.glyph_count as u32 {
                (cp - first) as u16
            } else {
                cursor_x = cursor_x.wrapping_add(advance);
                continue;
            };
            // Atlas UV for this glyph — top-left texel within the
            // atlas texture.
            let atlas_col = idx_in_font % self.glyphs_per_row;
            let atlas_row = idx_in_font / self.glyphs_per_row;
            let u = (atlas_col as u16) * font.glyph_w as u16;
            let v = (atlas_row as u16) * font.glyph_h as u16;

            wait_cmd_ready();
            // GP0 0x64 = variable-size textured rectangle, no blend,
            // opaque. First word: 0x64_BB_GG_RR (color is the tint
            // multiplier — NOT a fill colour: CLUT index 1's white
            // texel gets modulated by this).
            write_gp0(0x6400_0000 | pack_color(tint.0, tint.1, tint.2));
            write_gp0(pack_vertex(cursor_x, y));
            // Second word packs (U, V, CLUT) — our `pack_texcoord`
            // takes (u, v, extra) where `extra` is the CLUT field
            // (high halfword). The tpage is implied by the current
            // draw mode.
            write_gp0(pack_texcoord(u as u8, v as u8, clut_word));
            // Third word: rectangle size.
            write_gp0(pack_xy(font.glyph_w as u16, font.glyph_h as u16));

            cursor_x = cursor_x.wrapping_add(advance);
        }
    }

    /// Access the underlying [`BitmapFont`]. Useful for UI code
    /// that needs glyph dimensions for its own layout math.
    pub fn font(&self) -> &'static BitmapFont {
        self.font
    }

    /// Tpage the atlas is installed at — useful if the caller wants
    /// to restore it after drawing with a different tpage. Always
    /// 4bpp, always inside a valid VRAM page-aligned slot.
    pub fn tpage(&self) -> Tpage {
        self.tpage
    }
}

// ======================================================================
// Tests (host-side — pure data-transform checks)
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic font: two 8×2 glyphs, LSB-first, for bit-unpacking
    /// verification.
    const TEST_FONT: BitmapFont = BitmapFont {
        glyph_w: 8,
        glyph_h: 2,
        first_char: b'A' as u16,
        glyph_count: 2,
        bitmap: &[
            // 'A': row 0 = all pixels, row 1 = leftmost only
            0xFF, 0x01,
            // 'B': row 0 = alternating, row 1 = rightmost
            0x55, 0x80,
        ],
        advance_x: 8,
        line_height: 2,
        bit_order: BitOrder::Lsb,
    };

    #[test]
    fn row_bytes_handles_widths_under_and_over_8() {
        let mut f = TEST_FONT;
        assert_eq!(f.row_bytes(), 1);
        f.glyph_w = 12;
        assert_eq!(f.row_bytes(), 2);
        f.glyph_w = 16;
        assert_eq!(f.row_bytes(), 2);
        f.glyph_w = 24;
        assert_eq!(f.row_bytes(), 3);
    }

    #[test]
    fn lsb_row_unpacks_as_expected() {
        let row_a0 = TEST_FONT.glyph_row_packed(0, 0);
        assert_eq!(row_a0, 0xFF);
        let row_a1 = TEST_FONT.glyph_row_packed(0, 1);
        assert_eq!(row_a1, 0x01);
        let row_b0 = TEST_FONT.glyph_row_packed(1, 0);
        assert_eq!(row_b0, 0x55);
        let row_b1 = TEST_FONT.glyph_row_packed(1, 1);
        assert_eq!(row_b1, 0x80);
    }

    #[test]
    fn msb_order_mirrors_the_byte() {
        let msb_font = BitmapFont {
            bit_order: BitOrder::Msb,
            ..TEST_FONT
        };
        // 0xFF mirrors to 0xFF.
        assert_eq!(msb_font.glyph_row_packed(0, 0), 0xFF);
        // 0x01 in MSB-first becomes 0x80 after reversing.
        assert_eq!(msb_font.glyph_row_packed(0, 1), 0x80);
        // 0x55 (0b01010101) mirrors to 0xAA (0b10101010).
        assert_eq!(msb_font.glyph_row_packed(1, 0), 0xAA);
    }
}
