//! `hello-ot` — submit GPU primitives via an ordering table and
//! DMA channel 2 in linked-list mode, the same path a commercial
//! game uses.
//!
//! Draws three overlapping Gouraud triangles back-to-front; the
//! OT's depth sort determines the draw order. Running this
//! exercises the full commercial-game rendering pipe:
//!
//! 1. Clear the OT (sets up inter-slot chain).
//! 2. Build primitive packets in a static RAM arena.
//! 3. Prepend into the OT at the primitive's Z slot.
//! 4. Fill a background rect.
//! 5. Submit the OT to GP0 via DMA linked-list.
//! 6. VSync.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::TriGouraud;
use psx_gpu::{self as gpu, Resolution, VideoMode};

/// Primitives live in `.bss` so the DMA walker hits stable
/// addresses. Mutable static is fine on single-threaded bare metal.
static mut OT: OrderingTable<16> = OrderingTable::new();
static mut TRIS: [TriGouraud; 3] = [
    TriGouraud {
        tag: 0,
        color0_cmd: 0,
        v0: 0,
        color1: 0,
        v1: 0,
        color2: 0,
        v2: 0,
    },
    TriGouraud {
        tag: 0,
        color0_cmd: 0,
        v0: 0,
        color1: 0,
        v1: 0,
        color2: 0,
        v2: 0,
    },
    TriGouraud {
        tag: 0,
        color0_cmd: 0,
        v0: 0,
        color1: 0,
        v1: 0,
        color2: 0,
        v2: 0,
    },
];

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    let mut frame: u16 = 0;
    loop {
        let t = frame as i16;

        // Three overlapping triangles at different depth slots.
        // Slot 0 = back, slot 15 = front.
        let red = TriGouraud::new(
            [(80 + (t % 40), 60), (40, 160), (200, 160)],
            [(220, 80, 80), (80, 220, 80), (80, 80, 220)],
        );
        let green = TriGouraud::new(
            [(160, 40), (80, 180), (240, 160)],
            [(80, 200, 120), (200, 80, 120), (120, 80, 200)],
        );
        let blue = TriGouraud::new(
            [(200 - (t % 30), 80), (120, 200), (260, 170)],
            [(80, 160, 220), (220, 120, 80), (160, 220, 80)],
        );

        // Safety: the OT + TRIS statics are the only mutation site
        // and we touch them sequentially on one thread.
        unsafe {
            TRIS[0] = red;
            TRIS[1] = green;
            TRIS[2] = blue;

            OT.clear();
            OT.add(10, &mut TRIS[0], TriGouraud::WORDS);
            OT.add(8, &mut TRIS[1], TriGouraud::WORDS);
            OT.add(6, &mut TRIS[2], TriGouraud::WORDS);

            gpu::fill_rect(0, 0, 320, 240, 0, 0, 48);
            OT.submit();
        }

        gpu::vsync();
        frame = frame.wrapping_add(1);
    }
}
