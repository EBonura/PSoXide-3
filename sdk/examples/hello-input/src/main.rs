//! `hello-input` — poll the port-1 digital pad every frame and
//! paint the screen with feedback that reacts to the user's input.
//!
//! Proves two pipelines end-to-end:
//! 1. SIO0 + pad (keyboard event → winit → emulator bus → SIO0 pad
//!    → homebrew poll).
//! 2. `psx-font`: ASCII text rendering through a 4bpp CLUT atlas
//!    in VRAM. The label row at the top spells out every held button
//!    so the visual matches whatever we polled.
//!
//! Bindings (mapped by the frontend's keyboard handler):
//! - D-pad UP    → brighten red channel
//! - D-pad DOWN  → brighten green channel
//! - D-pad LEFT  → brighten blue channel
//! - D-pad RIGHT → reset to black
//! - CROSS / CIRCLE / TRIANGLE / SQUARE draw a small coloured
//!   triangle in the centre whose orientation rotates with the
//!   face-button pressed.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_font::{FontAtlas, fonts::BASIC};
use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_pad::{ButtonState, button, poll_port1};
use psx_vram::{Clut, TexDepth, Tpage};

/// Font atlas VRAM slot.
///
/// Framebuffer is 320×240 at (0, 0); everything else is free real
/// estate. Tpage X must be a multiple of 64 → 320 is the lowest slot
/// that clears the framebuffer. At 4bpp, a BASIC 128-glyph atlas
/// uses 64 halfwords × 32 halfword rows → (320..384) × (0..32), well
/// inside the tpage.
const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
/// 2-entry CLUT (transparent + white) at (320, 256). X is a multiple
/// of 16 ✓, past the 240-pixel framebuffer ✓, clear of the atlas.
const FONT_CLUT: Clut = Clut::new(320, 256);

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // One-shot: expand the 1bpp BASIC font into a 4bpp CLUT texture
    // + a tiny transparent/white CLUT, both parked off the
    // framebuffer.
    let font = FontAtlas::upload(&BASIC, FONT_TPAGE, FONT_CLUT);

    let mut r: u8 = 0;
    let mut g: u8 = 0;
    let mut b: u8 = 32;

    loop {
        let pad = poll_port1();

        // Direction buttons nudge the background color.
        if pad.is_held(button::UP) {
            r = r.saturating_add(4);
        }
        if pad.is_held(button::DOWN) {
            g = g.saturating_add(4);
        }
        if pad.is_held(button::LEFT) {
            b = b.saturating_add(4);
        }
        if pad.is_held(button::RIGHT) {
            r = 0;
            g = 0;
            b = 32;
        }

        // Background.
        gpu::fill_rect(0, 0, 320, 240, r, g, b);

        // Face buttons paint a coloured triangle in the centre of
        // the screen. Each direction rotates 90° so "Triangle"
        // literally points up, etc.
        face_button_tri(pad);

        // Label row: show every held button as its uppercase name.
        // `(0x80, 0x80, 0x80)` = unmodulated white (PSX texture tint
        // is `output = texel * tint / 128`).
        draw_button_labels(&font, pad);

        gpu::draw_sync();
        gpu::vsync();
    }
}

/// Paint the name of every held button down the top-left of the
/// screen. One line per button — 8 px tall, so 14 buttons fits
/// comfortably inside 240 px of vertical room with a 2px gap.
fn draw_button_labels(font: &FontAtlas, pad: ButtonState) {
    // Header, unconditional — sanity check that the font is alive
    // even when no key is pressed.
    font.draw_text(4, 4, "HELD:", (0x80, 0x80, 0x80));

    // (button mask, label, tint) — tints match the triangle colours
    // where applicable so the label lines up visually with the
    // on-screen primitive.
    let rows: &[(u16, &str, (u8, u8, u8))] = &[
        (button::UP,       "UP",       (0xC0, 0x30, 0x30)),
        (button::DOWN,     "DOWN",     (0x30, 0xC0, 0x30)),
        (button::LEFT,     "LEFT",     (0x30, 0x30, 0xC0)),
        (button::RIGHT,    "RIGHT",    (0xA0, 0xA0, 0xA0)),
        (button::TRIANGLE, "TRIANGLE", (0x30, 0xC0, 0x30)),
        (button::CIRCLE,   "CIRCLE",   (0xC0, 0x30, 0x30)),
        (button::CROSS,    "CROSS",    (0x30, 0x80, 0xC0)),
        (button::SQUARE,   "SQUARE",   (0xC0, 0x30, 0xC0)),
        (button::L1,       "L1",       (0xA0, 0xA0, 0x30)),
        (button::R1,       "R1",       (0xA0, 0xA0, 0x30)),
        (button::L2,       "L2",       (0x60, 0xA0, 0x60)),
        (button::R2,       "R2",       (0x60, 0xA0, 0x60)),
        (button::START,    "START",    (0xE0, 0xE0, 0xE0)),
        (button::SELECT,   "SELECT",   (0xE0, 0xE0, 0xE0)),
    ];

    // Label area: start at y=16 (below the HELD: header), advance 10
    // px per line (8 px glyph + 2 px gap). Skip unpressed rows.
    let mut y: i16 = 16;
    for &(mask, label, tint) in rows {
        if pad.is_held(mask) {
            font.draw_text(4, y, label, tint);
            y = y.wrapping_add(10);
        }
    }

    // Bottom-right: show the raw bitmask in hex so the parity tests
    // can latch onto something stable regardless of which buttons
    // were held.
    let hex = hex_u16(pad.bits());
    font.draw_text(320 - 8 * 6 - 4, 240 - 10, &hex, (0x70, 0x70, 0x70));
}

/// `0x1234` as a 6-char string — `'0'`, `'x'`, then 4 uppercase hex
/// digits, written into a stack buffer. no_std-friendly (no alloc).
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

/// Owning newtype around a 6-byte hex buffer so `as_str` borrows
/// from `self` — avoids dangling-slice pitfalls with stack arrays.
struct HexU16([u8; 6]);
impl HexU16 {
    fn as_str(&self) -> &str {
        // SAFETY: only ASCII hex digits + '0', 'x' are ever written
        // to the buffer.
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}
impl core::ops::Deref for HexU16 {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

fn face_button_tri(pad: ButtonState) {
    let (verts, color) = if pad.is_held(button::TRIANGLE) {
        (
            [(160, 100), (140, 140), (180, 140)],
            (80, 220, 80), // green
        )
    } else if pad.is_held(button::CIRCLE) {
        (
            [(180, 120), (140, 100), (140, 140)],
            (220, 80, 80), // red
        )
    } else if pad.is_held(button::CROSS) {
        (
            [(160, 140), (140, 100), (180, 100)],
            (80, 160, 220), // blue
        )
    } else if pad.is_held(button::SQUARE) {
        (
            [(140, 120), (180, 100), (180, 140)],
            (220, 80, 220), // magenta
        )
    } else {
        // No face button held: tiny white triangle so the app still
        // shows signs of life.
        (
            [(160, 118), (156, 124), (164, 124)],
            (200, 200, 200),
        )
    };
    gpu::draw_tri_flat(verts, color.0, color.1, color.2);
}
