//! Typed VRAM primitives.
//!
//! PS1 VRAM is a 1024×512 halfword frame, and getting a sprite
//! on screen is three interlocking decisions: *where* the texel
//! bytes live, *where* the CLUT lives, and *what* format you're
//! in. PsyQ leaves those as `short` variables you pass around by
//! convention. This crate promotes them to types.
//!
//! ## What's here
//!
//! - [`Color555`] -- a 15bpp BGR pixel. Constructed via
//!   [`Color555::rgb8`] (`u8` RGB → 5-bit truncation), or
//!   [`Color555::rgb5`] (already 5-bit), or the `const` raw
//!   constructor [`Color555::raw`].
//! - [`VramRect`] -- an `(x, y, w, h)` in VRAM pixels with const
//!   validation of the bounds. Can't construct one that would
//!   overflow VRAM.
//! - [`Tpage`] -- a texture-page handle. Const constructor
//!   enforces the PSX alignment rules (`x % 64 == 0`, `y ∈ {0, 256}`),
//!   knows its bit-depth, and emits the GP0(E1h) draw-mode word
//!   or the 16-bit tpage field embedded in textured-rect UV words.
//! - [`Clut`] -- a CLUT handle. `x % 16 == 0`, `y ∈ 0..512`. Emits
//!   the 16-bit clut field for UV words. 4bpp and 8bpp CLUTs are
//!   the same underlying type -- the calling primitive's tpage
//!   picks which entry count it uses.
//! - [`upload_16bpp`] -- safe upload wrapper: checks `pixels.len()`
//!   matches `rect.w * rect.h`, packs to GP0 0xA0 + word stream.
//!
//! ## What's NOT here (yet)
//!
//! - Compile-time tpage-overlap detection. A later pass can wrap
//!   allocations in a const-generic `Layout<...>` that enforces
//!   non-overlap across a fixed set of Tpage/Clut declarations;
//!   kept out for now because the rules around the 16-pixel CLUT
//!   stride + tpage-row sharing get messy to express in bare
//!   Rust const generics, and a build-time allocator tool is a
//!   cleaner path when we're ready.
//! - TIM parsing. Runtime / proc-macro loading lives in
//!   `psx-asset` when it lands.
//! - 4bpp / 8bpp upload helpers. PsyQ packs those as 16-bit
//!   halfwords too (4 / 2 texels per halfword); the same
//!   `upload_16bpp` works once the caller has pre-packed the
//!   indices. A typed `upload_4bpp(rect, &[u8], clut)` would be
//!   nicer; it's one of the first obvious follow-ups.
//!
//! ## Why these types over PsyQ's shorts
//!
//! Every field has rules PsyQ leaves to the programmer. Tpage X
//! must be a multiple of 64; Y must be 0 or 256. CLUT X must be
//! a multiple of 16. Upload sizes are clamped on hardware but
//! wrap (wrap, not truncate) on the real DMA controller if you
//! oversize them. Getting any of these wrong in PsyQ silently
//! corrupts VRAM. Const-validating them at construction means
//! the bug can't compile -- which is the whole point of writing
//! a Rust SDK instead of a C one.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use psx_hw::gpu::{gp0, pack_xy};
use psx_io::gpu::{wait_cmd_ready, write_gp0};

/// VRAM framebuffer width in pixels.
pub const VRAM_WIDTH: u16 = 1024;
/// VRAM framebuffer height in pixels.
pub const VRAM_HEIGHT: u16 = 512;

// ======================================================================
// Color
// ======================================================================

/// A 15-bit BGR pixel as stored in VRAM.
///
/// PSX VRAM stores every halfword in BGR-555-with-mask-bit layout:
///
/// ```text
///   bit  15   14..10   9..5   4..0
///        m    B       G      R
/// ```
///
/// The mask bit is bit 15; [`Color555::rgb8`] constructs with the
/// mask bit clear (most common). The "transparency" that games rely
/// on for CLUT sprites is `Color555::raw(0)` -- PSX treats an
/// all-zero texel as transparent in direct-color mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Color555(u16);

impl Color555 {
    /// Fully-transparent texel (value 0). PSX treats this as "skip"
    /// when sampling a textured primitive, regardless of mode.
    pub const TRANSPARENT: Self = Self(0);
    /// Opaque black with mask bit clear.
    pub const BLACK: Self = Self(0);
    /// Opaque white -- all five bits set in each channel.
    pub const WHITE: Self = Self(0x7FFF);

    /// Build a color from 8-bit channels, truncating the low 3 bits
    /// of each. `rgb8(255, 255, 255) == WHITE`.
    ///
    /// Matches PsyQ's `getTPage(0, r, g, b)` and
    /// `setRGB0(prim, r, g, b)` convention where the caller writes
    /// 8-bit values; hardware sees 5 bits per channel.
    pub const fn rgb8(r: u8, g: u8, b: u8) -> Self {
        let r = (r as u16) >> 3;
        let g = (g as u16) >> 3;
        let b = (b as u16) >> 3;
        Self(r | (g << 5) | (b << 10))
    }

    /// Build a color from already-5-bit channels. Asserts each
    /// fits in 5 bits. `rgb5(31, 31, 31) == WHITE`.
    pub const fn rgb5(r: u8, g: u8, b: u8) -> Self {
        assert!(r < 32, "rgb5: red must be < 32");
        assert!(g < 32, "rgb5: green must be < 32");
        assert!(b < 32, "rgb5: blue must be < 32");
        Self((r as u16) | ((g as u16) << 5) | ((b as u16) << 10))
    }

    /// Construct from the raw VRAM halfword. Useful when decoding
    /// CLUT bytes or loading pre-packed asset data.
    pub const fn raw(value: u16) -> Self {
        Self(value)
    }

    /// Underlying VRAM halfword.
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    /// Set the mask bit (bit 15). Matters when `GP0 0xE6` has
    /// "check mask on draw" enabled -- writes are skipped where
    /// existing mask is set. Games use this for UI overlays.
    pub const fn with_mask_bit(self) -> Self {
        Self(self.0 | 0x8000)
    }
}

// ======================================================================
// Rect
// ======================================================================

/// An `(x, y, w, h)` rectangle in VRAM coordinates.
///
/// Construction asserts the rect fits in VRAM. Subsequent code can
/// pass `VramRect` around without re-checking -- it's by-construction
/// safe.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VramRect {
    /// Top-left X in VRAM pixels (0..VRAM_WIDTH).
    pub x: u16,
    /// Top-left Y in VRAM pixels (0..VRAM_HEIGHT).
    pub y: u16,
    /// Width in pixels. `x + w <= VRAM_WIDTH`.
    pub w: u16,
    /// Height in pixels. `y + h <= VRAM_HEIGHT`.
    pub h: u16,
}

impl VramRect {
    /// Build a rect, asserting it fits in VRAM.
    pub const fn new(x: u16, y: u16, w: u16, h: u16) -> Self {
        assert!(w > 0, "VramRect: width must be > 0");
        assert!(h > 0, "VramRect: height must be > 0");
        assert!(x <= VRAM_WIDTH, "VramRect: x past VRAM right edge");
        assert!(y <= VRAM_HEIGHT, "VramRect: y past VRAM bottom edge");
        assert!(
            (x as u32) + (w as u32) <= VRAM_WIDTH as u32,
            "VramRect: (x + w) overflows VRAM width",
        );
        assert!(
            (y as u32) + (h as u32) <= VRAM_HEIGHT as u32,
            "VramRect: (y + h) overflows VRAM height",
        );
        Self { x, y, w, h }
    }

    /// Total pixel count. `w * h`.
    pub const fn pixel_count(self) -> u32 {
        (self.w as u32) * (self.h as u32)
    }
}

// ======================================================================
// Tpage + Clut
// ======================================================================

/// Texture color depth -- the last field of GP0(E1h) and bits 7..8
/// of a primitive's tpage word.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TexDepth {
    /// 4-bit CLUT-indexed. 4 texels per 16-bit halfword.
    Bit4 = 0,
    /// 8-bit CLUT-indexed. 2 texels per halfword.
    Bit8 = 1,
    /// 15-bit direct color. 1 texel per halfword.
    Bit15 = 2,
}

impl TexDepth {
    const fn as_u16(self) -> u16 {
        self as u16
    }

    /// Texels per 16-bit halfword at this depth. 4 for 4bpp,
    /// 2 for 8bpp, 1 for 15bpp.
    pub const fn texels_per_halfword(self) -> u16 {
        match self {
            TexDepth::Bit4 => 4,
            TexDepth::Bit8 => 2,
            TexDepth::Bit15 => 1,
        }
    }
}

/// A texture page: a 256×256 region at a specific VRAM origin,
/// plus a depth.
///
/// Hardware constraints (PSX-SPX "GP0 0xE1"):
/// - `x` is multiples of 64 pixels → one of 0, 64, 128, …, 960.
/// - `y` is either 0 or 256.
/// - The effective page width shrinks with depth: 64 pixels for
///   4bpp, 128 for 8bpp, 256 for 15bpp. (Because the texel-fetch
///   address is `tpage_x + u / texels_per_halfword`, so smaller
///   texels pack into fewer halfwords and hence a narrower span.)
///
/// The const constructor catches misaligned X/Y at compile time.
/// A const check across multiple Tpage consts (to enforce non-
/// overlap) needs const generics that are noisy to express here;
/// runtime collision tooling lives in a follow-up.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Tpage {
    x: u16,
    y: u16,
    depth: TexDepth,
}

impl Tpage {
    /// Build a Tpage. Compile-time assertion that `x` is a
    /// multiple of 64 and `y` is 0 or 256.
    #[allow(clippy::manual_is_multiple_of)]
    pub const fn new(x: u16, y: u16, depth: TexDepth) -> Self {
        assert!(x % 64 == 0, "Tpage: x must be a multiple of 64");
        assert!(x < VRAM_WIDTH, "Tpage: x past VRAM");
        assert!(y == 0 || y == 256, "Tpage: y must be 0 or 256");
        Self { x, y, depth }
    }

    /// VRAM X origin (0, 64, …, 960).
    pub const fn x(self) -> u16 {
        self.x
    }
    /// VRAM Y origin (0 or 256).
    pub const fn y(self) -> u16 {
        self.y
    }
    /// Color depth.
    pub const fn depth(self) -> TexDepth {
        self.depth
    }

    /// VRAM rect this tpage would actually cover at its depth.
    /// Useful for sanity-checking that uploaded texture data fits.
    pub const fn covering_rect(self) -> VramRect {
        let w = match self.depth {
            TexDepth::Bit4 => 64,
            TexDepth::Bit8 => 128,
            TexDepth::Bit15 => 256,
        };
        // covering_rect is for upload sizing; height is always the
        // 256-row page height. The assert in VramRect::new guards
        // the actual bound at compile time.
        VramRect::new(self.x, self.y, w, 256)
    }

    /// Encode as the low-16-bits of a textured-primitive UV word
    /// (the "tpage" embedded field). Same byte layout as GP0(E1h)'s
    /// low 10 bits minus display-disable / dither bits, which we
    /// leave to the draw-mode setter. Bits:
    ///
    /// ```text
    ///   bits 0..3   : tpage X / 64
    ///   bit  4      : tpage Y / 256
    ///   bits 5..6   : semi-transparency mode (0 = half + half)
    ///   bits 7..8   : depth (0=4bpp, 1=8bpp, 2=15bpp)
    /// ```
    ///
    /// `semi_trans` picks the GPU blend mode when a texel's mask
    /// bit is set. 0 is "0.5·bg + 0.5·fg" -- the most common.
    pub const fn uv_tpage_word(self, semi_trans: u8) -> u16 {
        assert!(semi_trans < 4, "semi_trans must be 0..4");
        let tpx = (self.x / 64) & 0xF;
        let tpy = if self.y == 256 { 1 } else { 0 };
        let depth = self.depth.as_u16();
        tpx | (tpy << 4) | ((semi_trans as u16) << 5) | (depth << 7)
    }

    /// The full GP0(E1h) draw-mode word. Use this to set the
    /// "current" tpage for sprite / rect primitives that don't
    /// embed a tpage of their own.
    pub fn apply_as_draw_mode(self) {
        wait_cmd_ready();
        write_gp0(gp0::draw_mode(
            (self.x / 64) as u32,
            if self.y == 256 { 1 } else { 0 },
            0,
            self.depth as u32,
            false,
            true,
        ));
        // Plain tpage application means "sample the page directly".
        // Material-aware helpers re-apply their own texture window after
        // setting draw mode.
        wait_cmd_ready();
        write_gp0(gp0::tex_window(0, 0, 0, 0));
    }
}

/// A CLUT slot -- 16 consecutive halfwords at `(x, y)` for 4bpp, or
/// 256 consecutive halfwords for 8bpp. Const constructor asserts
/// the 16-pixel X alignment hardware requires.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Clut {
    x: u16,
    y: u16,
}

impl Clut {
    /// Build a CLUT. `x` must be a multiple of 16; `y` is any row.
    #[allow(clippy::manual_is_multiple_of)]
    pub const fn new(x: u16, y: u16) -> Self {
        assert!(x % 16 == 0, "Clut: x must be a multiple of 16");
        assert!(x < VRAM_WIDTH, "Clut: x past VRAM");
        assert!(y < VRAM_HEIGHT, "Clut: y past VRAM");
        Self { x, y }
    }

    /// VRAM X origin (0, 16, 32, …).
    pub const fn x(self) -> u16 {
        self.x
    }
    /// VRAM Y origin (0..512).
    pub const fn y(self) -> u16 {
        self.y
    }

    /// Encode as the high-16-bits of a textured-primitive's first
    /// UV word. Bits:
    ///
    /// ```text
    ///   bits 0..5   : CLUT X / 16
    ///   bits 6..14  : CLUT Y
    /// ```
    pub const fn uv_clut_word(self) -> u16 {
        let cx = (self.x / 16) & 0x3F;
        let cy = self.y & 0x1FF;
        cx | (cy << 6)
    }
}

// ======================================================================
// Uploads
// ======================================================================

/// Upload raw 16bpp halfwords into a VRAM rect via GP0 0xA0 + word
/// stream. Checks that `pixels.len() * 2 == rect.w * rect.h` -- one
/// halfword per pixel, and the FIFO ships 32-bit words containing
/// two halfwords each. Odd pixel counts (which round up on
/// hardware) panic because they're rarely what the caller wanted.
///
/// For a 15bpp direct-color upload, each halfword is a
/// [`Color555`]. For pre-packed 4bpp / 8bpp index data, each
/// halfword packs 4 or 2 indices respectively.
pub fn upload_16bpp(rect: VramRect, pixels: &[u16]) {
    let expected = rect.pixel_count();
    assert_eq!(
        pixels.len() as u32,
        expected,
        "upload_16bpp: pixels.len() ({}) != rect.w*rect.h ({})",
        pixels.len(),
        expected,
    );
    assert!(
        expected.is_multiple_of(2),
        "upload_16bpp: odd pixel count ({expected}) not supported — caller should round up",
    );

    wait_cmd_ready();
    write_gp0(gp0::COPY_CPU_TO_VRAM);
    write_gp0(pack_xy(rect.x, rect.y));
    write_gp0(pack_xy(rect.w, rect.h));
    // Two halfwords per 32-bit FIFO word, low half first.
    let mut i = 0;
    while i + 1 < pixels.len() {
        let lo = pixels[i] as u32;
        let hi = pixels[i + 1] as u32;
        wait_cmd_ready();
        write_gp0(lo | (hi << 16));
        i += 2;
    }
}

/// Upload raw byte-stream pixel data, interpreted as halfwords in
/// little-endian order. Same semantics as [`upload_16bpp`] but
/// accepts a `&[u8]` -- useful when the data comes from
/// `include_bytes!` of a cooked asset blob, where the returned
/// byte array has alignment 1 and a direct `&[u16]` reinterpret
/// would be undefined behaviour.
///
/// `rect.w` still measures *halfwords* (the VRAM native unit).
/// The byte length must be exactly `2 × rect.w × rect.h`.
pub fn upload_bytes(rect: VramRect, bytes: &[u8]) {
    let expected = (rect.pixel_count() as usize) * 2;
    assert_eq!(
        bytes.len(),
        expected,
        "upload_bytes: bytes.len() ({}) != 2 × rect.w × rect.h ({})",
        bytes.len(),
        expected,
    );
    assert!(
        bytes.len() >= 4 && bytes.len().is_multiple_of(4),
        "upload_bytes: byte count must be a positive multiple of 4"
    );

    wait_cmd_ready();
    write_gp0(gp0::COPY_CPU_TO_VRAM);
    write_gp0(pack_xy(rect.x, rect.y));
    write_gp0(pack_xy(rect.w, rect.h));
    // Two halfwords per FIFO word, low half first -- matches
    // upload_16bpp's packing convention.
    let mut i = 0;
    while i + 3 < bytes.len() {
        let w = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        wait_cmd_ready();
        write_gp0(w);
        i += 4;
    }
}

/// Upload typed [`Color555`] pixels -- sugar over [`upload_16bpp`]
/// for direct-color textures.
pub fn upload_15bpp(rect: VramRect, pixels: &[Color555]) {
    let as_u16: &[u16] = unsafe {
        // SAFETY: Color555 is `#[repr(transparent)] u16`, so the
        // slice cast is layout-safe.
        core::slice::from_raw_parts(pixels.as_ptr() as *const u16, pixels.len())
    };
    upload_16bpp(rect, as_u16);
}

/// Upload a CLUT -- a row of [`Color555`]s at `clut`. The caller
/// picks the width: 16 entries for 4bpp, 256 for 8bpp. Asserts
/// the CLUT fits in VRAM width and the slice length matches.
pub fn upload_clut(clut: Clut, entries: &[Color555]) {
    let rect = VramRect::new(clut.x(), clut.y(), entries.len() as u16, 1);
    upload_15bpp(rect, entries);
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_rgb8_truncates_low_bits() {
        assert_eq!(Color555::rgb8(0, 0, 0).as_u16(), 0);
        assert_eq!(Color555::rgb8(255, 255, 255).as_u16(), 0x7FFF);
        // 0x18 = 24 → 24 >> 3 = 3.
        assert_eq!(Color555::rgb8(24, 0, 0).as_u16(), 3);
    }

    #[test]
    fn color_rgb5_rejects_out_of_range_via_panic() {
        // Compile-time assert check happens in `const fn`; the panic
        // path at runtime is the same.
        assert_eq!(Color555::rgb5(31, 0, 0).as_u16(), 31);
        assert_eq!(Color555::rgb5(0, 31, 0).as_u16(), 31 << 5);
        assert_eq!(Color555::rgb5(0, 0, 31).as_u16(), 31 << 10);
    }

    #[test]
    #[should_panic = "red must be < 32"]
    fn color_rgb5_panics_on_32() {
        let _ = Color555::rgb5(32, 0, 0);
    }

    #[test]
    fn vram_rect_new_accepts_valid_positions() {
        let _ = VramRect::new(0, 0, 64, 64);
        let _ = VramRect::new(VRAM_WIDTH - 1, 0, 1, 1);
        let _ = VramRect::new(0, VRAM_HEIGHT - 1, 1, 1);
    }

    #[test]
    #[should_panic = "overflows VRAM width"]
    fn vram_rect_new_rejects_horizontal_overflow() {
        let _ = VramRect::new(1000, 0, 100, 10);
    }

    #[test]
    fn tpage_encodes_uv_word_correctly() {
        let tp = Tpage::new(640, 0, TexDepth::Bit15);
        // x=640 → tpx=10, y=0 → tpy=0, depth=15bpp → 2.
        let word = tp.uv_tpage_word(0);
        assert_eq!(word & 0xF, 10, "tpx");
        assert_eq!((word >> 4) & 1, 0, "tpy");
        assert_eq!((word >> 5) & 3, 0, "semi_trans");
        assert_eq!((word >> 7) & 3, 2, "depth");
    }

    #[test]
    #[should_panic = "x must be a multiple of 64"]
    fn tpage_rejects_misaligned_x() {
        let _ = Tpage::new(32, 0, TexDepth::Bit8);
    }

    #[test]
    #[should_panic = "y must be 0 or 256"]
    fn tpage_rejects_bad_y() {
        let _ = Tpage::new(0, 128, TexDepth::Bit4);
    }

    #[test]
    fn clut_encodes_uv_word_correctly() {
        let cl = Clut::new(640, 240);
        // x=640 → cx=40, y=240 → cy=240.
        let word = cl.uv_clut_word();
        assert_eq!(word & 0x3F, 40, "cx");
        assert_eq!((word >> 6) & 0x1FF, 240, "cy");
    }

    #[test]
    #[should_panic = "x must be a multiple of 16"]
    fn clut_rejects_misaligned_x() {
        let _ = Clut::new(8, 0);
    }

    #[test]
    fn tex_depth_texels_per_halfword() {
        assert_eq!(TexDepth::Bit4.texels_per_halfword(), 4);
        assert_eq!(TexDepth::Bit8.texels_per_halfword(), 2);
        assert_eq!(TexDepth::Bit15.texels_per_halfword(), 1);
    }
}
