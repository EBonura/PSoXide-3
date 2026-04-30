//! `hello-gte` -- spin a 3D cube via the GTE, using the SDK math layer.
//!
//! Reads like a normal 3D engine: build a rotation, load it, project
//! vertices, draw edges. Rotation spins yaw × pitch so the cube's
//! depth is obvious. Angles are `Mat3I16::rotate_*` 256-per-revolution
//! units; a ×4 step on the frame counter gives one full Y-rev per 64
//! frames (≈1 s at 60 fps).
//!
//! Edges drawn via `gpu::draw_line_mono` (GP0 0x40, real diagonal
//! rasteriser). The previous version of this example used
//! `fill_rect`-per-pixel, which the PSX's GP0 0x02 fill primitive
//! rounds to 16-pixel X boundaries -- the result was blocky noise
//! rather than a visible wireframe.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_gpu::{self as gpu, Resolution, VideoMode, framebuf::FrameBuffer};
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene;
use psx_rt::tty;

/// Eight corners of a unit cube, centred at the origin, in 1.3.12
/// fixed point -- `0x0800` = 0.5.
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

/// Index pairs that form cube edges. Four back-face edges, four
/// front-face edges, four connectors.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

/// Angle step per frame (256 = one full revolution). 4 → 64 frames
/// per rev. Different multipliers on yaw vs pitch so the cube
/// precesses as well as spins.
const YAW_STEP: u16 = 4;
const PITCH_STEP: u16 = 3;

#[no_mangle]
fn main() {
    tty::println("hello-gte: booted");

    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    // Double-buffer: tear-free cube spinning, even under the
    // mono-line rasteriser's per-pixel cost.
    let mut fb = FrameBuffer::new(320, 240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // One-shot scene setup. OFX/OFY in 15.16 fixed-point centre the
    // projection at the screen middle. H=200 is a reasonable focal
    // length at z≈3.0; the Z translation pushes the cube into the
    // scene so RTPS's perspective divide stays well-defined.
    scene::set_screen_offset(160 << 16, 120 << 16);
    scene::set_projection_plane(200);
    scene::load_translation(Vec3I32::new(0, 0, 0x3000));

    let mut frame: u16 = 0;
    loop {
        fb.clear(0, 0, 32);

        // Compose yaw × pitch on the CPU, upload once to GTE R0..R4.
        let yaw = Mat3I16::rotate_y(frame.wrapping_mul(YAW_STEP));
        let pitch = Mat3I16::rotate_x(frame.wrapping_mul(PITCH_STEP));
        let rot = yaw.mul(&pitch);
        scene::load_rotation(&rot);

        // Project every vertex via RTPS → screen-space (sx, sy).
        let mut projected: [(i16, i16); 8] = [(0, 0); 8];
        for (i, v) in CUBE_VERTS.iter().enumerate() {
            let p = scene::project_vertex(*v);
            projected[i] = (p.sx, p.sy);
        }

        // Twelve real GPU lines -- GP0 0x40.
        for &(a, b) in &CUBE_EDGES {
            let (ax, ay) = projected[a];
            let (bx, by) = projected[b];
            gpu::draw_line_mono(ax, ay, bx, by, 255, 255, 255);
        }

        if frame == 0 {
            let flag = scene::read_flag();
            if flag != 0 {
                tty::println("hello-gte: GTE flag non-zero at first frame");
            }
        }

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
        frame = frame.wrapping_add(1);
    }
}
