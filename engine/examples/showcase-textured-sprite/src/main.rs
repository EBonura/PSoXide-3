//! `showcase-textured-sprite` — polished textured-sprite demo.
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
//! Ported to `psx-engine` in Phase 3e: the init (gpu::init,
//! framebuffer, draw-area, texture upload, tpage/apply_as_draw_mode)
//! lives in `Scene::init`; render submits per-frame GP0 writes for
//! the two sprites. `ctx.frame` drives the bounce cadence.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_engine::{App, Config, Ctx, Scene};
use psx_hw::gpu::{pack_color, pack_texcoord, pack_vertex, pack_xy};
use psx_io::gpu::{wait_cmd_ready, write_gp0};
use psx_vram::{Color555, TexDepth, Tpage, VramRect, upload_15bpp};

/// Texture page — picked at the standard off-display X=768 to
/// stay clear of a potential 640-wide framebuffer.
const TEX_TPAGE: Tpage = Tpage::new(768, 0, TexDepth::Bit15);
/// 32×32 demo texture.
const TEX_W: u16 = 32;
const TEX_H: u16 = 32;
const TEX_RECT: VramRect = VramRect::new(TEX_TPAGE.x(), TEX_TPAGE.y(), TEX_W, TEX_H);

// ----------------------------------------------------------------------
// Scene state
// ----------------------------------------------------------------------

/// Scene state. `tpage_word` is baked once in `init` because
/// [`Tpage::uv_tpage_word`] can be const-folded at that point and
/// held cheap for the render loop.
struct Showcase {
    tpage_word: u16,
}

impl Scene for Showcase {
    fn init(&mut self, _ctx: &mut Ctx) {
        // Upload the demo texture, then activate its tpage as the
        // draw mode (the GP0 0xE1 write that sets tex-page + semi-
        // transparency for subsequent primitives).
        upload_demo_texture();
        TEX_TPAGE.apply_as_draw_mode();
        self.tpage_word = TEX_TPAGE.uv_tpage_word(0);
    }

    fn update(&mut self, _ctx: &mut Ctx) {
        // Pure animation demo; no state mutates outside the
        // frame counter that `App` owns.
    }

    fn render(&mut self, ctx: &mut Ctx) {
        // The bounce cadence reads a `u16` so it matches the
        // pre-engine golden exactly — wrapping at 65536 gives the
        // same phase arithmetic.
        let frame = ctx.frame as u16;

        // Full 32×32 sprite bouncing left/right + down/up.
        let x_full = 24 + bounce(frame, 200);
        let y_full = 48 + bounce(frame.wrapping_mul(2), 120);
        draw_sprite(x_full, y_full, TEX_W, TEX_H, (0, 0), self.tpage_word);

        // 16×16 detail crop starting at UV (8, 8) — the centre of
        // the texture. Independent bounce periods so the two sprites
        // trace different paths.
        let x_detail = 220 - bounce(frame.wrapping_mul(3), 120);
        let y_detail = 132 + bounce(frame.wrapping_mul(5), 56);
        draw_sprite(x_detail, y_detail, 16, 16, (8, 8), self.tpage_word);
    }
}

// ----------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------

#[no_mangle]
fn main() -> ! {
    let mut scene = Showcase { tpage_word: 0 };
    // Deep navy clear — matches the pre-engine (8, 14, 40) backdrop.
    let config = Config {
        clear_color: (8, 14, 40),
        ..Config::default()
    };
    App::run(config, &mut scene);
}

// ----------------------------------------------------------------------
// Texture + sprite helpers — unchanged from the pre-engine showcase.
// ----------------------------------------------------------------------

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
