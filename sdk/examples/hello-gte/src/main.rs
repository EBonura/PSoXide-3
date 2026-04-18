//! `hello-gte` — spin a 3D cube via the GTE.
//!
//! Exercises the psx-gte SDK crate end-to-end: loads an identity-
//! scaled rotation matrix, applies a tiny per-frame Y rotation, runs
//! RTPS against the eight cube vertices, and reads back the projected
//! screen-space XY to draw the cube's edges as 2D lines.
//!
//! The rotation math lives on the CPU — we update `RT` each frame
//! rather than accumulating GTE state across frames — so this example
//! is really about proving the SDK wiring: `ctc2!` to load control
//! regs, `mtc2!` to load vertex data, `rtps()` to run the projection,
//! `mfc2!` to read the result.

#![no_std]
#![no_main]
// Required so the `asm!` macros expanded from psx-gte compile at the
// call site. The feature gate is per-crate, not per-dependency, so
// every crate that actually invokes the GTE register macros must opt
// in here.
#![cfg_attr(target_arch = "mips", feature(asm_experimental_arch))]

extern crate psx_rt;

use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_gte::{cfc2, ctc2, math::Vec3I16, mfc2, mtc2, regs::pack_xy, rtps};
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

/// Small cosine/sine lookup — 64 entries covering 2π. Values in
/// 1.3.12 fixed point so they can be stored directly in rotation
/// matrix cells. Hand-computed (no libm dep).
const COS_TABLE: [i16; 64] = [
    4096, 4086, 4056, 4006, 3935, 3845, 3736, 3608, 3462, 3299, 3119, 2925, 2716, 2494, 2259, 2014,
    1759, 1495, 1224, 948, 668, 386, 103, -180, -461, -739, -1010, -1275, -1530, -1775, -2007,
    -2225, -2428, -2614, -2782, -2931, -3060, -3168, -3253, -3316, -3356, -3372, -3365, -3334,
    -3280, -3202, -3102, -2981, -2840, -2681, -2506, -2317, -2115, -1902, -1681, -1455, -1225,
    -993, -761, -533, -310, -94, 112, 306,
];

#[inline(always)]
fn cos_1_3_12(index: u16) -> i16 {
    COS_TABLE[(index & 63) as usize]
}

#[inline(always)]
fn sin_1_3_12(index: u16) -> i16 {
    COS_TABLE[((index.wrapping_sub(16)) & 63) as usize]
}

#[no_mangle]
fn main() {
    tty::println("hello-gte: booted");

    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // GTE control setup — one-shot at boot. The ctc2!/mtc2!/cfc2!/mfc2!
    // macros are self-contained safe wrappers; no `unsafe {}` needed.
    // OFX = screen centre, 160.0 in 15.16. OFY = 120.0.
    ctc2!(24, 160 << 16);
    ctc2!(25, 120 << 16);
    // H = projection plane. 200 → FOV that makes the unit cube
    // look reasonable at z=2.0 (SZ3 ≈ 0x2000 after perspective).
    ctc2!(26, 200);
    // TR (translation) — push the cube into the screen along +Z.
    ctc2!(5, 0);
    ctc2!(6, 0);
    // TR.z = 0x4000 (4.0 in 1.3.12). Keeps SZ positive so RTPS
    // doesn't trip DIV_OVERFLOW.
    ctc2!(7, 0x4000);
    // Depth-cue disabled — DQA = 0, DQB = 0.
    ctc2!(27, 0);
    ctc2!(28, 0);

    let mut frame: u16 = 0;
    loop {
        gpu::fill_rect(0, 0, 320, 240, 0, 0, 32);

        // Build RT = Y-rotation by `frame` / 64 of a revolution. Pure
        // Y rotation is diag(cos, 1, cos) with off-diagonal ±sin.
        let c = cos_1_3_12(frame);
        let s = sin_1_3_12(frame);
        ctc2!(0, pack_xy(c, 0));
        ctc2!(1, pack_xy(s, 0));
        ctc2!(2, pack_xy(0x1000, 0));
        ctc2!(3, pack_xy(-s, 0));
        ctc2!(4, c as i32 as u32);

        // Project every vertex to screen-space and collect the SXY
        // results into a scratch array. Only `rtps()` needs the
        // `unsafe {}` — it's an `unsafe fn` because running it
        // without first loading V0 is undefined behaviour.
        let mut projected: [(i16, i16); 8] = [(0, 0); 8];
        for (i, v) in CUBE_VERTS.iter().enumerate() {
            mtc2!(0, v.xy_packed());
            mtc2!(1, v.z_packed());
            unsafe { rtps() };
            let sxy: u32 = mfc2!(14);
            projected[i] = (sxy as i16, (sxy >> 16) as i16);
        }

        // Rasterise the twelve edges as thin lines — GPU doesn't have
        // a native "line" prim, so we fake it with long thin triangles
        // if needed. For now, 1×n filled rects approximate straight
        // horizontals / verticals. Good enough as a smoke test.
        for &(a, b) in &CUBE_EDGES {
            let (ax, ay) = projected[a];
            let (bx, by) = projected[b];
            draw_line(ax, ay, bx, by, (255, 255, 255));
        }

        // Optional: dump FLAG after every frame so the TTY shows
        // DIV_OVERFLOW / saturation events.
        if frame == 0 {
            let flag: u32 = cfc2!(31);
            if flag != 0 {
                tty::println("hello-gte: GTE flag non-zero");
            }
        }

        gpu::draw_sync();
        gpu::vsync();
        frame = frame.wrapping_add(1);
    }
}

/// Poor-man's line draw — rasterise the convex hull of a 1×1
/// square at each interpolated point. 64 steps keeps it cheap.
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
