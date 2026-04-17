//! `hello-tex` — upload a tiny 15-bit direct-color texture to VRAM,
//! then draw an animated textured sprite. Exercises:
//!
//! - CPU→VRAM upload via GP0 0xA0 (our emulator tracks this in
//!   the `cpu->vram` opcode bucket).
//! - Draw-mode command (GP0 0xE1) that selects the texture page.
//! - Textured-rect primitive (GP0 0x64) with tpage / color bytes.
//!
//! The texture is a 16×16 checkerboard generated at start; the
//! sprite bounces across the screen so you can see it move.
//!
//! Running this confirms the emulator's full texture path works:
//! VRAM upload, texture-page lookup, UV sampling, sprite raster.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_gpu::{self as gpu, Resolution, TextureDepth, VideoMode};
use psx_hw::gpu::{gp0, pack_color, pack_texcoord, pack_vertex, pack_xy};
use psx_io::gpu::{wait_cmd_ready, write_gp0};

/// VRAM coordinates for our texture. Drop it in the off-display
/// region (X ≥ 640 is traditional for tpages), 16×16 pixels.
const TEX_X: u16 = 640;
const TEX_Y: u16 = 0;
const TEX_W: u16 = 16;
const TEX_H: u16 = 16;

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // Build a 16×16 checkerboard in 15-bit BGR. Two pixels per
    // u32 word, row-major. Orange / teal for easy identification.
    let mut pixels = [0u32; (TEX_W * TEX_H / 2) as usize];
    for y in 0..TEX_H {
        for x in 0..TEX_W {
            let checker = ((x / 4) ^ (y / 4)) & 1 != 0;
            let color15 = if checker {
                // Orange-ish: R31 G16 B4
                31 | (16 << 5) | (4 << 10)
            } else {
                // Teal-ish: R4 G24 B28
                4 | (24 << 5) | (28 << 10)
            };
            let i = (y * TEX_W + x) as usize;
            let word = &mut pixels[i / 2];
            if i & 1 == 0 {
                *word = (*word & 0xFFFF_0000) | color15;
            } else {
                *word = (*word & 0x0000_FFFF) | (color15 << 16);
            }
        }
    }

    gpu::upload_rect_raw(TEX_X, TEX_Y, TEX_W, TEX_H, &pixels);
    gpu::set_texture_page(TEX_X, TEX_Y, TextureDepth::Bit15);

    // Tpage byte the sprite's header word needs (encodes tpage_x
    // quarter + tpage_y bit + depth=2 = 15-bit).
    let tpage_x_q = (TEX_X / 64) as u16; // 0..=15
    let tpage_y_b = (TEX_Y / 256) as u16; // 0..=1
    let tpage = tpage_x_q | (tpage_y_b << 4) | (2 << 7);

    let mut frame: u16 = 0;
    loop {
        gpu::fill_rect(0, 0, 320, 240, 16, 16, 32);

        // Animate the sprite across the screen on a sine-ish bounce.
        let t = frame as i16;
        let x = 40 + (t * 2) % 240;
        let y = 80 + ((t / 3) % 60);

        draw_sprite(x, y, TEX_W, TEX_H, (0, 0), tpage);

        gpu::draw_sync();
        gpu::vsync();
        frame = frame.wrapping_add(1);
    }
}

/// Issue a textured-rect (GP0 0x64) at `(x, y)` of size `w × h`,
/// reading from texture page `tpage` at offset `uv`. Colour is
/// the blend tint (`(0x80, 0x80, 0x80)` = "use texel as-is").
fn draw_sprite(x: i16, y: i16, w: u16, h: u16, uv: (u8, u8), tpage: u16) {
    wait_cmd_ready();
    // 0x64 = variable-size textured rectangle, no blend, opaque.
    write_gp0(0x6400_0000 | pack_color(0x80, 0x80, 0x80));
    write_gp0(pack_vertex(x, y));
    write_gp0(pack_texcoord(uv.0, uv.1, tpage));
    write_gp0(pack_xy(w, h));

    // pack_xy is used but rust warnings if unused elsewhere. Silence.
    let _ = gp0::CLEAR_CACHE;
}
