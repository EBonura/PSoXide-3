//! `showcase-text` — tour of every text-rendering capability the
//! `psx-font` crate exposes. One frame demonstrates:
//!
//! 1. **Gradient title** — 2× scaled "PSOXIDE" in the 8×16 IBM
//!    VGA font with a vertical yellow→red colour sweep (quad-
//!    gouraud path, GP0 0x3C).
//! 2. **Multi-font comparison** — same word rendered at the 8×8
//!    and 8×16 sizes side-by-side. Two independent atlases
//!    uploaded into different tpage slots; zero per-glyph overhead
//!    to mix them.
//! 3. **Size ladder** — "Hello" at 1×, 2×, 3× scale (rect path +
//!    quad path combined in one frame).
//! 4. **Tint palette** — same word at six tints, showing the
//!    per-texel multiplier working cleanly on the white CLUT-1
//!    glyph (rect path).
//! 5. **Gradient varieties** — five short strings with different
//!    top/bottom gradient pairs, back-to-back (quad-gouraud path).
//! 6. **Rotating text** — "SPIN!" rotating around a screen-space
//!    pivot, animated via the frame counter (quad path + Q0.12
//!    sin/cos).
//! 7. **Affine transforms** — a shear + an anisotropic squash
//!    drawn through [`psx_font::FontAtlas::draw_text_affine`].
//!
//! Also demonstrates double-buffering via
//! [`psx_gpu::framebuf::FrameBuffer`] — essential for dense text
//! scenes that would otherwise outrun the TV's scanout and flicker.
//!
//! Visually dense on purpose — one .exe that proves every code
//! path ships end-to-end in release mode on real hardware (or the
//! emulator, which is what CI runs against).

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_font::{
    FontAtlas,
    fonts::{BASIC, BASIC_8X16},
};
use psx_gpu::{self as gpu, Resolution, VideoMode, framebuf::FrameBuffer};
use psx_vram::{Clut, TexDepth, Tpage};

/// 8×8 atlas tpage. `x=320` is a multiple of 64 ✓ and clear of
/// both display buffers (A at 0..320, B at 0..320 + vertical
/// offset). 4bpp × 32 glyphs/row × 8-wide = 64 halfwords; atlas
/// height is 32 halfwords for 128 glyphs — sits inside tpage.
const FONT_8X8_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
/// 2-entry CLUT for the 8×8 atlas at (320, 256). X is a multiple
/// of 16 ✓, clear of the display buffers ✓, clear of both atlases ✓.
const FONT_8X8_CLUT: Clut = Clut::new(320, 256);

/// 8×16 atlas tpage. Next slot along at `x=384` — another multiple
/// of 64. Same 64-halfword width (8-wide × 32 glyphs/row) but
/// taller: 64 halfwords of atlas height for 128 × 8×16 glyphs.
const FONT_8X16_TPAGE: Tpage = Tpage::new(384, 0, TexDepth::Bit4);
/// 2-entry CLUT for the 8×16 atlas at (384, 256). Separate from
/// the 8×8 CLUT so the two atlases are fully independent.
const FONT_8X16_CLUT: Clut = Clut::new(384, 256);

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);

    // Double-buffer the display to avoid tearing during the dense
    // text draw. Buffer A at (0, 0, 320, 240); B at (0, 240, 320,
    // 240). Without this, drawing ~870 GP0 words per frame can
    // race scanout and flicker to black when the rasteriser is
    // still filling text while the TV scans out.
    let mut fb = FrameBuffer::new(320, 240);
    // Prime the draw area + offset to point at the initial back
    // buffer (buffer A). `FrameBuffer::swap` resets them every
    // flip, but the first frame needs an explicit bootstrap.
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // Upload both atlases once at boot. They're fully independent
    // VRAM objects — the draw methods pick which to sample from.
    let font8 = FontAtlas::upload(&BASIC, FONT_8X8_TPAGE, FONT_8X8_CLUT);
    let font16 = FontAtlas::upload(&BASIC_8X16, FONT_8X16_TPAGE, FONT_8X16_CLUT);

    let mut frame_idx: u32 = 0;
    loop {
        // Clear the back buffer (the one we're about to draw
        // into). `FrameBuffer::clear` knows which buffer is the
        // back at the current moment.
        fb.clear(6, 8, 24);

        gradient_title(&font16);
        multi_font_compare(&font8, &font16);
        size_ladder(&font8);
        tint_palette(&font8);
        gradient_varieties(&font8);
        rotation_demo(&font16, frame_idx);
        affine_skew_demo(&font16);
        footer(&font8, frame_idx);

        gpu::draw_sync();
        gpu::vsync();
        // Flip: publish the freshly-drawn back buffer and start
        // drawing into the other one next frame.
        fb.swap();
        frame_idx = frame_idx.wrapping_add(1);
    }
}

/// Section 1: big 2×-scaled gradient title "PSOXIDE" at the top,
/// using the taller 8×16 BIOS-style font.
///
/// 7 chars × 8 px × 2× = 112 px wide; centred at
/// `(320 - 112) / 2 = 104`. Output is 32 px tall (16 × 2), same
/// rough visual weight as the old 8×8 × 3× title but crisper
/// because the glyph shapes come from a true 8×16 bitmap rather
/// than an upscaled 8×8.
fn gradient_title(font16: &FontAtlas) {
    font16.draw_text_scaled_gradient(
        104, 4, "PSOXIDE", 2, 2,
        (255, 220, 80),  // bright yellow at the top
        (200, 40, 20),   // deep red at the bottom
    );
}

/// Section 2: same word drawn in both fonts at 1×, side-by-side.
/// Proves two atlases can be installed at different tpages and
/// sampled per-draw without re-uploading.
///
/// Layout:
/// ```text
///   "8x8:  PSoXide-3"     <- 8 px tall
///   "8x16: PSoXide-3"     <- 16 px tall (below the 8×8 row)
/// ```
fn multi_font_compare(font8: &FontAtlas, font16: &FontAtlas) {
    font8.draw_text(8, 42, "8x8:  PSoXide-3", (200, 200, 200));
    font16.draw_text(8, 52, "8x16: PSoXide-3", (200, 240, 200));
}

/// Section 3: size ladder showing the scale parameter across 1×,
/// 2×, 3× of the 8×8 font — demonstrates the `_scaled` path.
/// Compare to the 8×16 row above to see the difference between
/// "scale an 8×8" (blocky) and "use a native 8×16" (crisp).
fn size_ladder(font: &FontAtlas) {
    font.draw_text(8, 72, "H 1x", (220, 220, 220));
    font.draw_text_scaled(56, 70, "H 2x", 2, 2, (120, 220, 120));
    font.draw_text_scaled(128, 68, "H 3x", 3, 3, (120, 160, 255));
}

/// Section 4: 6 tints of the same glyph. Proves the per-draw
/// tint is applied independently — no atlas re-upload needed.
fn tint_palette(font: &FontAtlas) {
    // Six primary-ish colours swept across the band.
    const PALETTE: &[(u8, u8, u8)] = &[
        (255, 80, 80),   // red
        (255, 180, 40),  // orange
        (240, 240, 80),  // yellow
        (120, 240, 120), // green
        (100, 180, 255), // cyan
        (200, 120, 255), // magenta
    ];
    let mut x: i16 = 8;
    for &tint in PALETTE {
        font.draw_text(x, 100, "PSX", tint);
        x = x.wrapping_add(8 * 4); // 3 chars + 1 space @ 8px advance
    }
}

/// Section 5: five gradient variations side-by-side. Each shows a
/// different top/bottom tint pair, revealing how the gouraud-
/// textured quad path interpolates vertically.
///
/// **Tint design note.** PSX tint math is `output = texel * tint /
/// 128`. Any channel with `tint >= 128` and a pure-white texel
/// saturates to max — so if all three top channels are ≥ 128, the
/// top of the glyph reads as near-white regardless of hue intent.
/// To keep every gradient visibly coloured top AND bottom, we
/// drop at least one channel below 128 on each tier. FIRE's top
/// (255, 120, 20) shows orange because G and B don't clamp; ICE's
/// (80, 200, 255) shows cyan because R is low; and so on.
fn gradient_varieties(font: &FontAtlas) {
    // Fire: saturated orange → deep red.
    font.draw_text_gradient(8, 116, "FIRE", (255, 120, 20), (200, 20, 10));
    // Ice: bright cyan → deep blue.
    font.draw_text_gradient(64, 116, "ICE", (80, 200, 255), (20, 40, 180));
    // Toxic: lime → dark green (poison/radioactive). The reference
    // gradient — both top and bottom have clearly-dominant green.
    font.draw_text_gradient(104, 116, "TOXIC", (180, 255, 80), (30, 100, 20));
    // Royal: gold → violet.
    font.draw_text_gradient(160, 116, "ROYAL", (255, 100, 20), (80, 20, 200));
    // Sunset: coral pink → dusk navy.
    font.draw_text_gradient(216, 116, "SUNSET", (255, 100, 120), (60, 20, 120));
}

/// Section 6: rotating "SPIN!" around a left-side pivot in the
/// 8×16 font. `frame_idx` drives the angle at ~1 revolution every
/// 2.5 s (4096 / 96 ≈ 42 frames at 60 fps). Demonstrates the
/// transform path works with any font size, not just 8×8.
fn rotation_demo(font16: &FontAtlas, frame_idx: u32) {
    // 96 Q0.12 units per frame = 4096/96 ≈ 42.7 frames per rev.
    // u16 wraps automatically at 0x10000 so modulo 4096 handled
    // implicitly via the table lookup.
    let angle = (frame_idx.wrapping_mul(96) & 0xFFF) as u16;
    font16.draw_text_rotated(78, 170, "SPIN!", angle, (255, 255, 140));
}

/// Section 7: horizontal shear + anisotropic squash via the
/// affine path, also on the 8×16 font for extra drama.
///
/// Q3.12 encoding: 0.6 × 4096 ≈ 2458; 0.8 × 4096 = 3277;
/// 1.5 × 4096 = 6144. Identity would be `[[4096, 0], [0, 4096]]`.
fn affine_skew_demo(font16: &FontAtlas) {
    // Shear: push x proportional to y. 0.6 × 4096 = 2458.
    let shear = [[4096, 2458], [0, 4096]];
    font16.draw_text_affine((190, 158), "SKEW", shear, (180, 220, 255));
    // Anisotropic squash — 0.8× x, 1.5× y.
    let squash = [[3277, 0], [0, 6144]];
    font16.draw_text_affine((262, 152), "TALL", squash, (255, 200, 240));
}

/// Footer — frame counter at bottom-left so we can see the demo
/// is live, plus the word "psx-font" as a credit.
fn footer(font: &FontAtlas, frame_idx: u32) {
    font.draw_text(8, 200, "rotated         affine", (140, 140, 140));
    font.draw_text(8, 228, "psx-font showcase", (180, 180, 180));
    // Frame counter: show the low 4 hex digits (enough for ~18
    // minutes at 60 fps before wrapping, plenty for a demo).
    let hex = hex_u16((frame_idx & 0xFFFF) as u16);
    font.draw_text(320 - 8 * 6 - 4, 228, &hex, (100, 140, 100));
}

/// Format a u16 as `"0xABCD"` into a stack buffer. no_std,
/// no_alloc — same helper used in hello-input but duplicated here
/// to keep the showcase standalone.
fn hex_u16(v: u16) -> HexU16 {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = [0u8; 6];
    out[0] = b'0';
    out[1] = b'x';
    out[2] = HEX[((v >> 12) & 0xF) as usize];
    out[3] = HEX[((v >> 8) & 0xF) as usize];
    out[4] = HEX[((v >> 4) & 0xF) as usize];
    out[5] = HEX[(v & 0xF) as usize];
    HexU16(out)
}

struct HexU16([u8; 6]);
impl core::ops::Deref for HexU16 {
    type Target = str;
    fn deref(&self) -> &str {
        // SAFETY: only ASCII digits + '0' / 'x' are written into
        // the buffer by hex_u16.
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}
