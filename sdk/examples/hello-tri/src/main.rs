//! `hello-tri` — the smallest interesting PSoXide homebrew.
//!
//! Initialises the GPU, clears the framebuffer to dark blue, and
//! draws one Gouraud-shaded triangle per frame with a slight
//! time-based wobble so you can tell the render loop is alive.
//!
//! Running it end-to-end proves:
//! - The PSX-EXE loader correctly seeded PC / SP / GP.
//! - `_start` cleared BSS and reached `main()`.
//! - The GPU's GP1 commands configure the display.
//! - Triangle rasterisation + vsync behave.

#![no_std]
#![no_main]

// Pull psx-rt in so the linker keeps `_start`, the panic handler,
// and (if enabled) the heap allocator. Re-exports via psx-sdk
// don't force the link because we don't call anything from them.
extern crate psx_rt;

use psx_gpu::{self as gpu, Resolution, VideoMode, framebuf::FrameBuffer};
use psx_rt::tty;

#[no_mangle]
fn main() {
    tty::println("hello-tri: booted via HLE BIOS");

    gpu::init(VideoMode::Ntsc, Resolution::R320X240);

    // Double-buffer: buffer A at VRAM (0, 0), buffer B at (0, 240).
    // The TV always scans out a stable completed frame while we
    // draw into the other buffer. PS1 games do this universally —
    // workload-light demos like this one wouldn't flicker even
    // single-buffered, but adopting the pattern here keeps every
    // example on the same footing.
    let mut fb = FrameBuffer::new(320, 240);
    // Bootstrap draw-area / offset to the first back buffer (A).
    // `FrameBuffer::swap` re-applies them every flip.
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    tty::println("hello-tri: entering render loop");
    let mut frame: u16 = 0;
    loop {
        // Clear the back buffer (the one we're about to draw into).
        fb.clear(0, 0, 64);

        // Wobble the triangle: ±30 px vertical bounce.
        let wobble = (((frame % 60) as i16) - 30).abs();
        let verts = [
            (160, 40 + wobble),
            (60, 200 - wobble),
            (260, 200 - wobble),
        ];
        gpu::draw_tri_gouraud(verts, [(255, 64, 64), (64, 255, 64), (64, 64, 255)]);

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();

        frame = frame.wrapping_add(1);
    }
}
