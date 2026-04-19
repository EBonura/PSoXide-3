//! `hello-tex` — upload a 16×16 15-bit direct-color checkerboard
//! texture and draw it as an animated bouncing sprite.
//!
//! Exercises:
//! - CPU→VRAM upload via [`psx_vram::upload_15bpp`] (GP0 0xA0 under
//!   the hood).
//! - [`Tpage`] encoding — compile-time-validated page origin + depth.
//! - Textured-rect primitive (GP0 0x64) fed the tpage word from
//!   [`Tpage::uv_tpage_word`].
//!
//! Compared to the pre-refactor version, everything related to VRAM
//! layout is now typed: colors construct with `Color555::rgb8`,
//! the texture lives at a `Tpage::new(…)` that refuses misaligned
//! origins at compile time, and the upload helper verifies
//! `pixels.len() == rect.w * rect.h` before kicking the FIFO.
//! The pixel-packing and tpage-word hand-math are gone.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_hw::gpu::{pack_color, pack_texcoord, pack_vertex, pack_xy};
use psx_io::gpu::{wait_cmd_ready, write_gp0};
use psx_vram::{Color555, Tpage, TexDepth, VramRect, upload_15bpp};

/// Where the texture lives in VRAM. X≥640 is the traditional tpage
/// region (past the 640-wide visible framebuffer).
const TEX_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit15);
/// Texture dimensions.
const TEX_W: u16 = 16;
const TEX_H: u16 = 16;
/// Full rect in VRAM covered by the texture upload.
const TEX_RECT: VramRect = VramRect::new(
    TEX_TPAGE.x(),
    TEX_TPAGE.y(),
    TEX_W,
    TEX_H,
);

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // Build a 16×16 checkerboard in 15-bit BGR. Orange / teal makes
    // the "is my texture being sampled correctly?" question easy to
    // answer by eye. Using `rgb5` with the exact 5-bit values the
    // pre-refactor code used so the milestone-C golden stays valid
    // across the psx-vram refactor — bit-identical VRAM output.
    let mut pixels = [Color555::raw(0); (TEX_W as usize) * (TEX_H as usize)];
    let orange = Color555::rgb5(31, 16, 4);
    let teal = Color555::rgb5(4, 24, 28);
    for y in 0..TEX_H {
        for x in 0..TEX_W {
            let checker = ((x / 4) ^ (y / 4)) & 1 != 0;
            pixels[(y * TEX_W + x) as usize] = if checker { orange } else { teal };
        }
    }

    upload_15bpp(TEX_RECT, &pixels);
    TEX_TPAGE.apply_as_draw_mode();

    // Tpage word embedded in the sprite's UV field. `semi_trans = 0`
    // means opaque (blend bits still get respected on the per-texel
    // mask-bit basis for 15bpp; 0 is a safe default here).
    let tpage_word = TEX_TPAGE.uv_tpage_word(0);

    let mut frame: u16 = 0;
    loop {
        gpu::fill_rect(0, 0, 320, 240, 16, 16, 32);

        // Bouncing motion across the visible area.
        let t = frame as i16;
        let x = 40 + (t * 2) % 240;
        let y = 80 + ((t / 3) % 60);
        draw_sprite(x, y, TEX_W, TEX_H, (0, 0), tpage_word);

        gpu::draw_sync();
        gpu::vsync();
        frame = frame.wrapping_add(1);
    }
}

/// Issue GP0 0x64 (variable-size textured rectangle) with the
/// 4-word packet. Tint = `(0x80, 0x80, 0x80)` passes the texel
/// through unmodulated.
fn draw_sprite(x: i16, y: i16, w: u16, h: u16, uv: (u8, u8), tpage: u16) {
    wait_cmd_ready();
    write_gp0(0x6400_0000 | pack_color(0x80, 0x80, 0x80));
    write_gp0(pack_vertex(x, y));
    write_gp0(pack_texcoord(uv.0, uv.1, tpage));
    write_gp0(pack_xy(w, h));
}
