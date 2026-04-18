//! `hello-gte` — spin a 3D cube via the GTE, using the SDK math layer.
//!
//! Before the `transform` + `scene` helpers existed this file had to
//! hand-roll a sine table and poke each control register individually.
//! Now it reads like a normal 3D engine: build a rotation, load it,
//! project vertices, draw lines.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_rt::tty;

/// Eight corners of a unit cube, centred at the origin, in 1.3.12
/// fixed point — `0x0800` = 0.5.
const CUBE_VERTS: [Vec3I16; 8] = [
    Vec3I16::new(-0x0800, -0x0800, -0x0800),
    Vec3I16::new(0x0800, -0x0800, -0x0800),
    Vec3I16::new(0x0800, 0x0800, -0x0800),
    Vec3I16::new(-0x0800, 0x0800, -0x0800),
    Vec3I16::new(-0x0800, -0x0800, 0x0800),
    Vec3I16::new(0x0800, -0x0800, 0x0800),
    Vec3I16::new(0x0800, 0x0800, 0x0800),
    Vec3I16::new(-0x0800, 0x0800, 0x0800),
];

/// Index pairs that form cube edges.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0), // back face
    (4, 5), (5, 6), (6, 7), (7, 4), // front face
    (0, 4), (1, 5), (2, 6), (3, 7), // connectors
];

#[no_mangle]
fn main() {
    tty::println("hello-gte: booted");

    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // One-shot scene setup: everything the GTE needs that doesn't
    // change per-frame. 200 is a reasonable focal length for a small
    // scene at z≈4.0; the translation pushes the cube into the
    // screen so RTPS's perspective divide stays well-defined.
    scene::set_screen_offset(160 << 16, 120 << 16);
    scene::set_projection_plane(200);
    scene::load_translation(Vec3I32::new(0, 0, 0x4000));

    let mut frame: u16 = 0;
    loop {
        gpu::fill_rect(0, 0, 320, 240, 0, 0, 32);

        // Compose the frame's rotation on the CPU, then push to GTE
        // once. Multiple rotations compose via `mul`:
        //   let spin = Mat3I16::rotate_y(frame).mul(&Mat3I16::rotate_x(frame / 2));
        // Single-axis spin here keeps the math obvious.
        let rot = Mat3I16::rotate_y(frame);
        scene::load_rotation(&rot);

        // Project every vertex to screen-space. `scene::project_vertex`
        // bundles MTC2 + RTPS + MFC2 into one call.
        let mut projected: [(i16, i16); 8] = [(0, 0); 8];
        for (i, v) in CUBE_VERTS.iter().enumerate() {
            let p = scene::project_vertex(*v);
            projected[i] = (p.sx, p.sy);
        }

        // Rasterise the twelve edges as coarse lines.
        for &(a, b) in &CUBE_EDGES {
            let (ax, ay) = projected[a];
            let (bx, by) = projected[b];
            draw_line(ax, ay, bx, by, (255, 255, 255));
        }

        // Diagnostic: TTY-dump GTE FLAG once at startup to catch a
        // mis-configured scene before wondering why nothing renders.
        if frame == 0 {
            let flag = scene::read_flag();
            if flag != 0 {
                tty::println("hello-gte: GTE flag non-zero at first frame");
            }
        }

        gpu::draw_sync();
        gpu::vsync();
        frame = frame.wrapping_add(1);
    }
}

/// Poor-man's line draw — rasterise a 2×2 block at each interpolated
/// point. 64 steps along the line keeps it cheap. A real engine would
/// build a GPU line primitive via the GP0 command list.
fn draw_line(x0: i16, y0: i16, x1: i16, y1: i16, rgb: (u8, u8, u8)) {
    let steps = 64i16;
    for i in 0..=steps {
        let x = x0 as i32 + ((x1 as i32 - x0 as i32) * i as i32) / steps as i32;
        let y = y0 as i32 + ((y1 as i32 - y0 as i32) * i as i32) / steps as i32;
        if (0..320).contains(&x) && (0..240).contains(&y) {
            gpu::fill_rect(x as u16, y as u16, 2, 2, rgb.0, rgb.1, rgb.2);
        }
    }
}
