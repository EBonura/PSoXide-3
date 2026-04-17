//! `hello-input` — poll the port-1 digital pad every frame and
//! paint the screen with a color that shifts as the user holds
//! direction buttons. Proves the SIO0 + pad pipeline end-to-end.
//!
//! Bindings (mapped by the frontend's keyboard handler):
//! - D-pad UP    → brighten red channel
//! - D-pad DOWN  → brighten green channel
//! - D-pad LEFT  → brighten blue channel
//! - D-pad RIGHT → reset to black
//! - CROSS / CIRCLE / TRIANGLE / SQUARE draw a small coloured
//!   triangle in the centre whose orientation rotates with the
//!   face-button pressed.
//!
//! The visible feedback is the cheapest way to confirm the full
//! path (key event → winit → emulator bus → SIO0 pad → homebrew
//! poll → frame render) is connected.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_pad::{button, poll_port1, ButtonState};

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

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

        gpu::draw_sync();
        gpu::vsync();
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
