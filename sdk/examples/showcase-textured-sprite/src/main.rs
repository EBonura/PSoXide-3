//! `showcase-textured-sprite` — Phase-1a polished demo.
//!
//! A 32×32 texture with four visually-distinct regions (quadrant
//! colors + diagonal accents) uploaded into a 15bpp tpage, then two
//! sprites bounce around the screen on different periods:
//! a full 32×32 rendering of the texture, plus a 16×16 detail
//! crop that shows UV-offset sampling of the same page.
//!
//! Compared to `hello-tex` this exercises:
//! - a larger, visually rich texture built via [`psx_vram::Color555`]
//! - two simultaneous sprites with different UVs into the same page
//! - a non-uniform bouncing motion that gives the pixel-pinned
//!   milestone test a non-trivial frame to hash
//!
//! Intentionally NOT yet exercised (lands in later phases):
//! - text / font rendering — Phase 1c
//! - semi-transparency blend modes — follow-up
//! - CLUT-indexed textures — follow-up when 4bpp/8bpp helpers land
//!
//! This is the second live consumer of [`psx_vram`] (after
//! `hello-tex`) — two independent uses proving the API isn't just
//! tailored to one caller's shape.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_gpu::{self as gpu, Resolution, VideoMode, framebuf::FrameBuffer};
use psx_hw::gpu::{pack_color, pack_texcoord, pack_vertex, pack_xy};
use psx_io::gpu::{wait_cmd_ready, write_gp0};
use psx_vram::{Color555, Tpage, TexDepth, VramRect, upload_15bpp};

/// Texture page — picked at the standard off-display X=768 to
/// stay clear of a potential 640-wide framebuffer.
const TEX_TPAGE: Tpage = Tpage::new(768, 0, TexDepth::Bit15);
/// 32×32 demo texture.
const TEX_W: u16 = 32;
const TEX_H: u16 = 32;
const TEX_RECT: VramRect = VramRect::new(TEX_TPAGE.x(), TEX_TPAGE.y(), TEX_W, TEX_H);

/// Screen resolution.
const SCREEN_W: u16 = 320;
const SCREEN_H: u16 = 240;

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    // Double-buffer: buffer A at (0, 0) and B at (0, 240). Texture
    // at Tpage(768, 0) is past both buffers' X extent.
    let mut fb = FrameBuffer::new(SCREEN_W, SCREEN_H);
    gpu::set_draw_area(0, 0, SCREEN_W - 1, SCREEN_H - 1);
    gpu::set_draw_offset(0, 0);

    upload_demo_texture();
    TEX_TPAGE.apply_as_draw_mode();
    let tpage_word = TEX_TPAGE.uv_tpage_word(0);

    let mut frame: u16 = 0;
    loop {
        fb.clear(8, 14, 40);

        // Full 32×32 sprite bouncing left/right + down/up.
        let x_full = 24 + bounce(frame, 200);
        let y_full = 48 + bounce(frame.wrapping_mul(2), 120);
        draw_sprite(x_full, y_full, TEX_W, TEX_H, (0, 0), tpage_word);

        // 16×16 detail crop starting at UV (8, 8) — the centre of
        // the texture. Independent bounce periods so the two sprites
        // trace different paths.
        let x_detail = 220 - bounce(frame.wrapping_mul(3), 120);
        let y_detail = 132 + bounce(frame.wrapping_mul(5), 56);
        draw_sprite(x_detail, y_detail, 16, 16, (8, 8), tpage_word);

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
        frame = frame.wrapping_add(1);
    }
}

/// Build the 32×32 showcase texture:
/// - background quadrants (orange / cyan / deep-blue)
/// - diagonal white accents
/// - checker detail at the 4-pixel grid
fn upload_demo_texture() {
    const SIZE: usize = (TEX_W as usize) * (TEX_H as usize);
    let mut pixels = [Color555::BLACK; SIZE];
    // Palette chosen for visual distinctness and within the 5-bit
    // channel range that rgb5 takes directly.
    let orange = Color555::rgb5(31, 17, 4);
    let cyan = Color555::rgb5(0, 27, 30);
    let deep_blue = Color555::rgb5(2, 4, 15);
    let white = Color555::rgb5(31, 31, 31);

    for y in 0..TEX_H {
        for x in 0..TEX_W {
            let checker = ((x / 4) + (y / 4)) % 2 == 0;
            let on_main_diag = (x as i16 - y as i16).abs() < 3;
            let on_anti_diag = (x as i16 + y as i16 - (TEX_W as i16 - 1)).abs() < 3;
            let color = if on_main_diag || on_anti_diag {
                white
            } else if checker {
                orange
            } else if y < TEX_H / 2 {
                cyan
            } else {
                deep_blue
            };
            pixels[y as usize * TEX_W as usize + x as usize] = color;
        }
    }

    upload_15bpp(TEX_RECT, &pixels);
}

/// Issue GP0 0x64 (variable-size textured rectangle).
fn draw_sprite(x: i16, y: i16, w: u16, h: u16, uv: (u8, u8), tpage: u16) {
    wait_cmd_ready();
    write_gp0(0x6400_0000 | pack_color(0x80, 0x80, 0x80));
    write_gp0(pack_vertex(x, y));
    write_gp0(pack_texcoord(uv.0, uv.1, tpage));
    write_gp0(pack_xy(w, h));
}

/// Triangle wave: input ∈ u16 phase, output ∈ 0..span pixels,
/// period = 2·span. Simpler than a sine but enough motion for
/// a visual-debug demo.
fn bounce(frame: u16, span: i16) -> i16 {
    let cycle = (span as u16).saturating_mul(2);
    let phase = if cycle == 0 { 0 } else { frame % cycle };
    if phase < span as u16 {
        phase as i16
    } else {
        (cycle - phase) as i16
    }
}
