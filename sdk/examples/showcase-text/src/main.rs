//! `showcase-text` — tour of every text-rendering capability the
//! `psx-font` crate exposes. One frame demonstrates:
//!
//! 1. **Gradient title** — 3× scaled "PSOXIDE" with a vertical
//!    yellow→red colour sweep (quad-gouraud path, GP0 0x3C).
//! 2. **Size ladder** — "Hello" rendered at 1×, 2×, and 3× scale
//!    (rect path + quad path combined in one frame).
//! 3. **Tint palette** — same word repeated at eight different
//!    tints, showing the per-texel multiplier working cleanly on
//!    the white CLUT-1 glyph (rect path).
//! 4. **Gradient varieties** — three short strings with different
//!    top/bottom gradient pairs, back-to-back (quad-gouraud path).
//! 5. **Rotating text** — "SPIN!" rotating around a screen-space
//!    pivot, animated via the frame counter (quad path + Q0.12
//!    sin/cos).
//! 6. **Affine skew** — a static shear transform drawn through
//!    [`psx_font::FontAtlas::draw_text_affine`].
//!
//! Visually dense on purpose — one .exe that proves every code
//! path ships end-to-end in release mode on real hardware (or the
//! emulator, which is what CI runs against).

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_font::{FontAtlas, fonts::BASIC};
use psx_gpu::{self as gpu, Resolution, VideoMode, framebuf::FrameBuffer};
use psx_vram::{Clut, TexDepth, Tpage};

/// Font atlas tpage. `x=320` (multiple of 64) is the lowest slot
/// clear of the 320-pixel framebuffer; 4bpp makes the atlas 64
/// halfwords × 32 halfword rows — sits entirely inside one tpage.
/// With double-buffering, buffer A lives at (0, 0) and B at (0,
/// 240) — the atlas at (320, 0..32) is still clear of both.
const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
/// 2-entry CLUT at (320, 256). X is a multiple of 16 ✓, past the
/// right edge of buffer B (which ends at x=320) ✓.
const FONT_CLUT: Clut = Clut::new(320, 256);

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

    let font = FontAtlas::upload(&BASIC, FONT_TPAGE, FONT_CLUT);

    let mut frame_idx: u32 = 0;
    loop {
        // Clear the back buffer (the one we're about to draw
        // into). `FrameBuffer::clear` knows which buffer is the
        // back at the current moment.
        fb.clear(6, 8, 24);

        gradient_title(&font);
        size_ladder(&font);
        tint_palette(&font);
        gradient_varieties(&font);
        rotation_demo(&font, frame_idx);
        affine_skew_demo(&font);
        footer(&font, frame_idx);

        gpu::draw_sync();
        gpu::vsync();
        // Flip: publish the freshly-drawn back buffer and start
        // drawing into the other one next frame.
        fb.swap();
        frame_idx = frame_idx.wrapping_add(1);
    }
}

/// Section 1: big 3×-scaled gradient title "PSOXIDE" at the top.
///
/// 7 chars × 8 px × 3× = 168 px wide; centred horizontally at
/// `(320 - 168) / 2 = 76`. Vertical gradient yellow→red reads as
/// "hot molten metal," a staple of PS1-era title cards.
fn gradient_title(font: &FontAtlas) {
    font.draw_text_scaled_gradient(
        76, 4, "PSOXIDE", 3, 3,
        (255, 220, 80),  // bright yellow at the top
        (200, 40, 20),   // deep red at the bottom
    );
}

/// Section 2: size ladder showing the scale parameter across 1×,
/// 2×, 3×. Same string at each, stacked so the viewer sees the
/// relative proportions.
fn size_ladder(font: &FontAtlas) {
    font.draw_text(8, 34, "Hello 1x", (220, 220, 220));
    font.draw_text_scaled(8, 44, "Hello 2x", 2, 2, (120, 220, 120));
    font.draw_text_scaled(8, 62, "Hello 3x", 3, 3, (120, 160, 255));
}

/// Section 3: 6 tints of the same glyph. Proves the per-draw
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
        font.draw_text(x, 94, "PSX", tint);
        x = x.wrapping_add(8 * 4); // 3 chars + 1 space @ 8px advance
    }
}

/// Section 4: three gradient variations side-by-side. Each pair
/// shows a different top/bottom tint combination, revealing how
/// the gouraud-textured quad path interpolates vertically.
fn gradient_varieties(font: &FontAtlas) {
    // Fire: bright yellow → deep red (classic)
    font.draw_text_gradient(8, 108, "FIRE", (255, 240, 100), (180, 30, 20));
    // Ice: white → blue (cool chromatic)
    font.draw_text_gradient(64, 108, "ICE", (240, 240, 255), (60, 100, 220));
    // Toxic: lime → dark green (poison/radioactive)
    font.draw_text_gradient(104, 108, "TOXIC", (180, 255, 80), (30, 100, 20));
    // Royal: gold → purple (richness)
    font.draw_text_gradient(160, 108, "ROYAL", (255, 220, 120), (100, 40, 160));
    // Sunset: pink → navy (dusk sky)
    font.draw_text_gradient(216, 108, "SUNSET", (255, 160, 160), (60, 40, 120));
}

/// Section 5: rotating "SPIN!" around a left-side pivot. The
/// `frame_idx` drives the angle at ~1 revolution every 2.5 s
/// (4096 / 96 ≈ 42 frames at 60 fps).
fn rotation_demo(font: &FontAtlas, frame_idx: u32) {
    // 96 Q0.12 units per frame = 4096/96 ≈ 42.7 frames per rev.
    // u16 wraps automatically at 0x10000 so modulo 4096 handled
    // implicitly via the table lookup.
    let angle = (frame_idx.wrapping_mul(96) & 0xFFF) as u16;
    font.draw_text_rotated(74, 170, "SPIN!", angle, (255, 255, 140));
    // Label so viewers know what the spinning thing is.
    font.draw_text(48, 196, "rotated", (140, 140, 140));
}

/// Section 6: horizontal shear via the affine path. `x' = x +
/// 0.6·y` pushes the top of each glyph to the right relative to
/// its base — classic "fast-motion italic" look.
///
/// Q3.12 encoding: 0.6 × 4096 ≈ 2458. Identity's `[[4096, 0], [0,
/// 4096]]` becomes `[[4096, 2458], [0, 4096]]` for this shear.
fn affine_skew_demo(font: &FontAtlas) {
    // Shear: push x proportional to y. 0.6 × 4096 = 2457.6 ≈ 2458.
    let shear = [[4096, 2458], [0, 4096]];
    font.draw_text_affine((196, 164), "SKEW", shear, (180, 220, 255));
    // Also show an anisotropic squash — scale y by 1.5× and x by
    // 0.8×, no shear. Q3.12: 0.8 = 3277, 1.5 = 6144.
    let squash = [[3277, 0], [0, 6144]];
    font.draw_text_affine((250, 162), "TALL", squash, (255, 200, 240));
    // Label row tying both effects to the "affine" name.
    font.draw_text(196, 196, "affine", (140, 140, 140));
}

/// Footer — frame counter at bottom-left so we can see the demo
/// is live, plus the word "psx-font" as a credit.
fn footer(font: &FontAtlas, frame_idx: u32) {
    font.draw_text(8, 224, "psx-font showcase", (180, 180, 180));
    // Frame counter: show the low 4 hex digits (enough for ~18
    // minutes at 60 fps before wrapping, plenty for a demo).
    let hex = hex_u16((frame_idx & 0xFFFF) as u16);
    font.draw_text(320 - 8 * 6 - 4, 224, &hex, (100, 140, 100));
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
