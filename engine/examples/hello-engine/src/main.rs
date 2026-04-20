//! `hello-engine` — the smallest possible `psx-engine` demo.
//!
//! A single Gouraud-shaded square bounces horizontally, driven by
//! a sine wave on the [`Angle`] type. Proves end-to-end that the
//! engine's `Scene` / `App::run` plumbing works:
//!
//! - `App::run` sets up the GPU + framebuffer + main loop.
//! - `Game` implements `Scene::update` and `Scene::render`.
//! - `Ctx` carries the per-frame state (frame counter, pad,
//!   framebuffer) between the two.
//! - `Angle` (Q0.16) feeds the SDK's 4096-per-rev sincos LUT
//!   through its converter, with no unit-mismatch footguns.
//!
//! Scene geometry: one drifting quad + one "hello, engine" banner
//! near the top. Cross-pad moves the quad vertically so the pad
//! plumbing is visible too.

#![no_std]
#![no_main]
#![allow(static_mut_refs)]

extern crate psx_rt;

use psx_engine::{Angle, App, Config, Ctx, Scene, button};
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::{QuadGouraud, RectFlat};
use psx_math::sincos;

// ----------------------------------------------------------------------
// Motion parameters
// ----------------------------------------------------------------------

/// The quad oscillates horizontally over this many frames — one
/// full sine cycle every ~2 seconds at 60 fps.
const OSCILLATION_FRAMES: u32 = 120;

/// Peak horizontal displacement in pixels from the screen centre.
const AMPLITUDE_PX: i16 = 80;

/// Vertical Y the pad can shift the quad by on a single press.
const PAD_Y_STEP: i16 = 4;

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

struct Game {
    /// Vertical offset from the screen centre, adjusted by the D-pad.
    y_offset: i16,
}

/// OT + primitive backing storage lives in `.bss`. Two slots:
/// background and foreground. Every frame overwrites these in
/// place before DMA sees them.
static mut OT: OrderingTable<4> = OrderingTable::new();

const RECT_ZERO: RectFlat = RectFlat::new(0, 0, 0, 0, 0, 0, 0);
static mut QUAD: RectFlat = RECT_ZERO;
static mut BG: QuadGouraud = QuadGouraud {
    tag: 0,
    color0_cmd: 0,
    v0: 0,
    color1: 0,
    v1: 0,
    color2: 0,
    v2: 0,
    color3: 0,
    v3: 0,
};

// ----------------------------------------------------------------------
// Scene impl
// ----------------------------------------------------------------------

impl Scene for Game {
    fn update(&mut self, ctx: &mut Ctx) {
        // Pad drives vertical offset. `just_pressed` so a held
        // D-pad doesn't scroll off-screen in a single frame — the
        // user releases + re-presses to step. Shows the edge-
        // detection helper.
        if ctx.just_pressed(button::UP) {
            self.y_offset = self.y_offset.saturating_sub(PAD_Y_STEP);
        }
        if ctx.just_pressed(button::DOWN) {
            self.y_offset = self.y_offset.saturating_add(PAD_Y_STEP);
        }
    }

    fn render(&mut self, ctx: &mut Ctx) {
        let ot = unsafe { &mut OT };
        let quad = unsafe { &mut QUAD };
        let bg = unsafe { &mut BG };
        ot.clear();

        // Background gradient quad — dim blue top, near-black
        // bottom. Gouraud across four vertices; showcases the OT
        // + QuadGouraud pipeline alongside the engine plumbing.
        *bg = QuadGouraud::new(
            [(0, 0), (320, 0), (0, 240), (320, 240)],
            [(8, 16, 48), (8, 16, 48), (0, 0, 8), (0, 0, 8)],
        );
        ot.add(3, bg, QuadGouraud::WORDS);

        // Foreground: a 24×24 solid quad oscillating horizontally
        // via `Angle`. `per_frames(N).mul_frame(frame)` is the
        // canonical way to drive a periodic motion — no modulo
        // snap-back, no unit mismatch.
        let phase = Angle::per_frames(OSCILLATION_FRAMES).mul_frame(ctx.frame);
        let dx = (sincos::sin_q12(phase.sin_q12_arg()) * AMPLITUDE_PX as i32) >> 12;
        let cx = 160 - 12 + dx as i16;
        let cy = 120 - 12 + self.y_offset;

        *quad = RectFlat::new(cx, cy, 24, 24, 0xE0, 0x60, 0x20);
        ot.add(1, quad, RectFlat::WORDS);

        ot.submit();
    }
}

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() -> ! {
    let mut game = Game { y_offset: 0 };
    App::run(Config::default(), &mut game);
}
