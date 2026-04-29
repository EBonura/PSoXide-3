//! `hello-input` — poll the port-1 pad every frame and
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
//! - The DualShock Analog button toggles analog mode. Frontend
//!   default: F9 on keyboard, or gamepad Mode/Guide when the OS
//!   exposes it. When analog mode is active, the bottom-left status
//!   panel shows raw stick bytes.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_font::{fonts::BASIC, FontAtlas};
use psx_gpu::{self as gpu, framebuf::FrameBuffer, Resolution, VideoMode};
use psx_pad::{button, poll_port1, ButtonState, PadMode, PadState};
use psx_vram::{Clut, TexDepth, Tpage};

/// Font atlas VRAM slot.
///
/// With double-buffering, buffer A is at (0..320, 0..240) and B
/// at (0..320, 240..480). Tpage X must be a multiple of 64 → 320
/// is the lowest slot that clears both buffers. At 4bpp, a BASIC
/// 128-glyph atlas uses 64 halfwords × 32 halfword rows →
/// (320..384) × (0..32), inside tpage (320, 0, Bit4).
const FONT_TPAGE: Tpage = Tpage::new(320, 0, TexDepth::Bit4);
/// 2-entry CLUT (transparent + white) at (320, 256). X is a
/// multiple of 16 ✓, past the right edge of buffer B at X=320 ✓.
const FONT_CLUT: Clut = Clut::new(320, 256);

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    let mut fb = FrameBuffer::new(320, 240);
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
        let state = poll_port1();
        let pad = state.buttons;

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

        // Background — clear the back buffer.
        fb.clear(r, g, b);

        // Face buttons paint a coloured triangle in the centre of
        // the screen. Each direction rotates 90° so "Triangle"
        // literally points up, etc.
        face_button_tri(pad);

        // Label row: show every held button as its uppercase name.
        // `(0x80, 0x80, 0x80)` = unmodulated white (PSX texture tint
        // is `output = texel * tint / 128`).
        draw_button_labels(&font, pad);
        draw_pad_status(&font, state);

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
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
        (button::UP, "UP", (0xC0, 0x30, 0x30)),
        (button::DOWN, "DOWN", (0x30, 0xC0, 0x30)),
        (button::LEFT, "LEFT", (0x30, 0x30, 0xC0)),
        (button::RIGHT, "RIGHT", (0xA0, 0xA0, 0xA0)),
        (button::TRIANGLE, "TRIANGLE", (0x30, 0xC0, 0x30)),
        (button::CIRCLE, "CIRCLE", (0xC0, 0x30, 0x30)),
        (button::CROSS, "CROSS", (0x30, 0x80, 0xC0)),
        (button::SQUARE, "SQUARE", (0xC0, 0x30, 0xC0)),
        (button::L1, "L1", (0xA0, 0xA0, 0x30)),
        (button::R1, "R1", (0xA0, 0xA0, 0x30)),
        (button::L2, "L2", (0x60, 0xA0, 0x60)),
        (button::R2, "R2", (0x60, 0xA0, 0x60)),
        (button::L3, "L3", (0x60, 0xC0, 0xC0)),
        (button::R3, "R3", (0x60, 0xC0, 0xC0)),
        (button::START, "START", (0xE0, 0xE0, 0xE0)),
        (button::SELECT, "SELECT", (0xE0, 0xE0, 0xE0)),
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

fn draw_pad_status(font: &FontAtlas, state: PadState) {
    let (mode, tint) = match state.mode {
        PadMode::Disconnected => ("NO PAD", (0xE0, 0x50, 0x50)),
        PadMode::Digital => ("DIGITAL", (0xE0, 0x90, 0x40)),
        PadMode::Analog => ("ANALOG", (0x60, 0xE0, 0x80)),
        PadMode::Config => ("CONFIG", (0x90, 0xA0, 0xE0)),
        PadMode::Unknown => ("UNKNOWN", (0xE0, 0xE0, 0x60)),
    };

    font.draw_text(4, 188, "MODE:", (0x80, 0x80, 0x80));
    font.draw_text(52, 188, mode, tint);

    let id = hex_u8(state.id_low);
    font.draw_text(124, 188, "ID:", (0x60, 0x60, 0x60));
    font.draw_text(152, 188, &id, (0x70, 0x70, 0x70));

    font.draw_text(4, 200, "ANALOG:", (0x80, 0x80, 0x80));
    if state.is_analog() {
        font.draw_text(68, 200, "ON", (0x60, 0xE0, 0x80));
        let lx = hex_u8(state.sticks.left_x);
        let ly = hex_u8(state.sticks.left_y);
        let rx = hex_u8(state.sticks.right_x);
        let ry = hex_u8(state.sticks.right_y);
        font.draw_text(4, 212, "L:", (0x80, 0x80, 0x80));
        font.draw_text(28, 212, &lx, (0x70, 0xC0, 0xE0));
        font.draw_text(68, 212, &ly, (0x70, 0xC0, 0xE0));
        font.draw_text(4, 224, "R:", (0x80, 0x80, 0x80));
        font.draw_text(28, 224, &rx, (0xC0, 0x90, 0xE0));
        font.draw_text(68, 224, &ry, (0xC0, 0x90, 0xE0));
    } else if state.is_connected() {
        font.draw_text(68, 200, "OFF", (0xE0, 0x90, 0x40));
        font.draw_text(4, 212, "PRESS ANALOG BUTTON", (0x90, 0x90, 0x90));
    } else {
        font.draw_text(68, 200, "N/A", (0xE0, 0x50, 0x50));
        font.draw_text(4, 212, "CONNECT A CONTROLLER", (0xE0, 0x90, 0x40));
    }
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

/// `0x12` as a 4-char string.
fn hex_u8(v: u8) -> HexU8 {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = [0u8; 4];
    out[0] = b'0';
    out[1] = b'x';
    out[2] = HEX[((v >> 4) & 0xF) as usize];
    out[3] = HEX[(v & 0xF) as usize];
    HexU8(out)
}

/// Owning newtype around a 4-byte hex buffer.
struct HexU8([u8; 4]);
impl HexU8 {
    fn as_str(&self) -> &str {
        // SAFETY: only ASCII hex digits + '0', 'x' are ever written
        // to the buffer.
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}
impl core::ops::Deref for HexU8 {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
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
        ([(160, 118), (156, 124), (164, 124)], (200, 200, 200))
    };
    gpu::draw_tri_flat(verts, color.0, color.1, color.2);
}
