//! `hello-gte` — spin a 3D cube via the GTE, using the SDK math layer.
//!
//! Reads like a normal 3D engine: build a rotation, load it, project
//! vertices, draw edges. Rotation spins around Y and X simultaneously
//! so the cube's depth is obvious (pure-Y from a straight-on camera
//! reads as a 2D rocking rectangle). Angles are
//! `Mat3I16::rotate_*` 256-per-revolution units, so a ×4 step on the
//! frame counter gives a full revolution every 64 frames (~1 second
//! at 60 fps) — fast enough to be visible, slow enough to watch.
//!
//! Previously the edges were "drawn" with `fill_rect` at 2×2 pixels
//! per step along each edge. GP0 0x02 (fill_rect) ignores draw-area
//! AND rounds X to 16-pixel boundaries, so every "pixel" actually
//! painted a 16×2 horizontal strip aligned to the nearest
//! multiple-of-16 column. The resulting output looked like random
//! blocky noise rather than a cube. Fixed by switching to
//! `gpu::draw_line_mono` (GP0 0x40, proper diagonal-capable line
//! rasteriser).

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

/// Index pairs that form cube edges. Four back-face edges, four
/// front-face edges, four connectors between them. Twelve total —
/// `make examples && make run-gte` and you should see twelve lines
/// tumbling over a blue background.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0), // back face
    (4, 5), (5, 6), (6, 7), (7, 4), // front face
    (0, 4), (1, 5), (2, 6), (3, 7), // connectors
];

/// How much to advance the angle per frame. 4 units at 256 per rev
/// → 64 frames per rev → ~1.07 s/rev at 60 fps. Different multipliers
/// on X vs Y so the cube precesses as well as spins — makes the
/// 3D-ness unmistakable.
const YAW_STEP: u16 = 4;
const PITCH_STEP: u16 = 3;

#[no_mangle]
fn main() {
    tty::println("hello-gte: booted");

    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // One-shot scene setup: everything the GTE needs that doesn't
    // change per-frame. Focal length 200 is a reasonable choice
    // for a small cube at z≈4.0; the translation pushes the cube
    // into the scene so RTPS's perspective divide stays well-
    // defined.
    scene::set_screen_offset(160 << 16, 120 << 16);
    scene::set_projection_plane(200);
    scene::load_translation(Vec3I32::new(0, 0, 0x4000));

    let mut frame: u16 = 0;
    loop {
        gpu::fill_rect(0, 0, 320, 240, 0, 0, 32);

        // Compose yaw (Y) * pitch (X). Both rotations are cheap and
        // `Mat3I16::mul` composes them on the CPU before a single
        // `load_rotation` upload to GTE R0..R5.
        let yaw = Mat3I16::rotate_y(frame.wrapping_mul(YAW_STEP));
        let pitch = Mat3I16::rotate_x(frame.wrapping_mul(PITCH_STEP));
        let rot = yaw.mul(&pitch);
        scene::load_rotation(&rot);

        // Project every vertex to screen-space. `scene::project_vertex`
        // bundles MTC2 + RTPS + MFC2 into one call.
        let mut projected: [(i16, i16); 8] = [(0, 0); 8];
        for (i, v) in CUBE_VERTS.iter().enumerate() {
            let p = scene::project_vertex(*v);
            projected[i] = (p.sx, p.sy);
        }

        // Twelve real GPU lines — GP0 0x40, diagonals rasterised
        // correctly by the hardware.
        for &(a, b) in &CUBE_EDGES {
            let (ax, ay) = projected[a];
            let (bx, by) = projected[b];
            gpu::draw_line_mono(ax, ay, bx, by, 255, 255, 255);
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
