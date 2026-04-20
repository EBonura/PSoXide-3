//! `hello-tex` — upload two 4bpp CLUT textures cooked from source
//! photographs (a brick wall + a cobblestone floor, produced by
//! `psxed tex` from their `vendor/*.jpg` originals) and draw them
//! as animated bouncing sprites.
//!
//! Exercises the full host-side + runtime texture pipeline:
//!
//!   editor `psxed tex`       host: PNG/JPG → crop → resample →
//!                             median-cut quantise → pack 4bpp →
//!                             `.psxt` blob
//!
//!   `include_bytes!`         compile: blob embedded in the MIPS
//!                             binary, no runtime disc IO needed
//!
//!   `Texture::from_bytes`    runtime: zero-copy header parse,
//!                             two slices (pixels + CLUT)
//!
//!   `upload_bytes`           runtime: pixels + CLUT to VRAM via
//!                             GP0 0xA0, once per texture + CLUT
//!
//!   GP0 0x64                 runtime: textured-rect primitive
//!                             samples the tpage + CLUT
//!
//! Two sprites bouncing at different speeds prove that multiple
//! cooked textures can share a tpage and each hits its own CLUT.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_asset::Texture;
use psx_gpu::{self as gpu, Resolution, VideoMode, framebuf::FrameBuffer};
use psx_hw::gpu::{pack_color, pack_texcoord, pack_vertex, pack_xy};
use psx_io::gpu::{wait_cmd_ready, write_gp0};
use psx_math::sincos;
use psx_vram::{Clut, TexDepth, Tpage, VramRect, upload_bytes};

/// Wall (brick) — cooked by `make assets` from
/// `vendor/brick-wall.jpg`.
static BRICK_BLOB: &[u8] = include_bytes!("../assets/brick-wall.psxt");

/// Floor (cobblestone / batako) — cooked from `vendor/floor.jpg`.
static FLOOR_BLOB: &[u8] = include_bytes!("../assets/floor.psxt");

/// Shared tpage — both 64-texel-wide 4bpp textures fit in one
/// 256-wide tpage with room to spare. The `apply_as_draw_mode`
/// call at startup sets this as the current tpage; every textured
/// sprite we draw samples from it.
const SHARED_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);

/// CLUT for the brick texture. X aligned to 16; Y=480 sits below
/// both framebuffer halves (0..240 and 240..480).
const BRICK_CLUT: Clut = Clut::new(0, 480);

/// CLUT for the floor texture. One row down — 16 halfwords ≠ a
/// full VRAM row width, so we just step Y by 1 to keep CLUTs
/// compact.
const FLOOR_CLUT: Clut = Clut::new(0, 481);

/// Both textures are 64×64. The floor sits at U=64 inside the
/// tpage, offset from the brick at U=0.
const TEX_W: u16 = 64;
const TEX_H: u16 = 64;
const BRICK_U: u8 = 0;
const FLOOR_U: u8 = 64;

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    let mut fb = FrameBuffer::new(320, 240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // --- Parse + upload both cooked textures ---
    let brick = Texture::from_bytes(BRICK_BLOB).expect("brick.psxt");
    let floor = Texture::from_bytes(FLOOR_BLOB).expect("floor.psxt");

    // Brick occupies the left 16 halfwords of the tpage (U 0..64).
    // VRAM rect width is in HALFWORDS — 4 texels per halfword at
    // 4bpp — so 64 texels wide = 16 halfwords wide.
    let brick_pix_rect = VramRect::new(
        SHARED_TPAGE.x(),
        SHARED_TPAGE.y(),
        brick.halfwords_per_row(),
        brick.height(),
    );
    upload_bytes(brick_pix_rect, brick.pixel_bytes());
    let brick_clut_rect =
        VramRect::new(BRICK_CLUT.x(), BRICK_CLUT.y(), brick.clut_entries(), 1);
    upload_bytes(brick_clut_rect, brick.clut_bytes());

    // Floor slotted right next to the brick, still in the same
    // tpage. X offset = brick's halfword count.
    let floor_pix_rect = VramRect::new(
        SHARED_TPAGE.x() + brick.halfwords_per_row(),
        SHARED_TPAGE.y(),
        floor.halfwords_per_row(),
        floor.height(),
    );
    upload_bytes(floor_pix_rect, floor.pixel_bytes());
    let floor_clut_rect =
        VramRect::new(FLOOR_CLUT.x(), FLOOR_CLUT.y(), floor.clut_entries(), 1);
    upload_bytes(floor_clut_rect, floor.clut_bytes());

    // --- Pre-encode the per-texture UV words ---
    SHARED_TPAGE.apply_as_draw_mode();
    let brick_clut_word = BRICK_CLUT.uv_clut_word();
    let floor_clut_word = FLOOR_CLUT.uv_clut_word();

    // Sine-oscillating centres — same Q0.12 phase pattern as
    // hello-ot's triangles. Modulo arithmetic would create a
    // sawtooth snap-back every N frames; `sincos::sin_q12`
    // gives a smooth back-and-forth drift.
    let mut frame: u32 = 0;
    loop {
        fb.clear(16, 16, 32);

        // Brick in the upper half: anchor at (100, 40), drifts
        // horizontally ±80 px and vertically ±20 px at slightly
        // different rates so the motion is a Lissajous curve.
        let brick_phase_x = (frame.wrapping_mul(48) & 0xFFF) as u16;
        let brick_phase_y = (frame.wrapping_mul(64) & 0xFFF) as u16;
        let bx = 100 + drift(brick_phase_x, 80);
        let by = 40 + drift(brick_phase_y, 20);
        draw_sprite(bx, by, TEX_W, TEX_H, (BRICK_U, 0), brick_clut_word);

        // Floor in the lower half: faster, opposite phase, so the
        // two sprites cross each other without colliding.
        let floor_phase_x = (frame.wrapping_mul(72).wrapping_add(2048) & 0xFFF) as u16;
        let floor_phase_y = (frame.wrapping_mul(56).wrapping_add(1024) & 0xFFF) as u16;
        let fx = 160 + drift(floor_phase_x, 80);
        let fy = 140 + drift(floor_phase_y, 20);
        draw_sprite(fx, fy, TEX_W, TEX_H, (FLOOR_U, 0), floor_clut_word);

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
        frame = frame.wrapping_add(1);
    }
}

/// Pixel displacement from a Q0.12 phase. `sin_q12` returns Q1.12
/// in `[-0x1000, 0x1000]`; multiplying by `amp_px` and shifting 12
/// rescales back to pixels — smooth ±`amp_px` drift with no
/// sawtooth wrap.
#[inline]
fn drift(phase_q12: u16, amp_px: i16) -> i16 {
    ((sincos::sin_q12(phase_q12) * amp_px as i32) >> 12) as i16
}

/// Issue GP0 0x64 (variable-size textured rectangle) with the
/// 4-word packet. Tint `(0x80, 0x80, 0x80)` passes texels through
/// unmodulated.
fn draw_sprite(x: i16, y: i16, w: u16, h: u16, uv: (u8, u8), clut: u16) {
    wait_cmd_ready();
    write_gp0(0x6400_0000 | pack_color(0x80, 0x80, 0x80));
    write_gp0(pack_vertex(x, y));
    write_gp0(pack_texcoord(uv.0, uv.1, clut));
    write_gp0(pack_xy(w, h));
}
