//! `hello-tex` — upload a cooked 4bpp CLUT texture (a brick wall,
//! produced by `psxed tex` from a source JPG) and draw it as an
//! animated bouncing sprite.
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
//!   `upload_bytes` ×2        runtime: pixels + CLUT to VRAM via
//!                             GP0 0xA0
//!
//!   GP0 0x64                 runtime: textured-rect primitive
//!                             samples the tpage + CLUT
//!
//! This replaces the previous hand-baked 16×16 15bpp checkerboard —
//! every byte of the texture now comes from the content pipeline.

#![no_std]
#![no_main]

extern crate psx_rt;

use psx_asset::Texture;
use psx_gpu::{self as gpu, Resolution, VideoMode, framebuf::FrameBuffer};
use psx_hw::gpu::{pack_color, pack_texcoord, pack_vertex, pack_xy};
use psx_io::gpu::{wait_cmd_ready, write_gp0};
use psx_vram::{Clut, TexDepth, Tpage, VramRect, upload_bytes};

/// Cooked brick-wall blob, produced at build time by
/// `psxed tex vendor/brick-wall.jpg …` and embedded here so the
/// homebrew is self-contained — no disc, no filesystem, no
/// runtime deps beyond the emulator + BIOS.
static BRICK_BLOB: &[u8] = include_bytes!("../assets/brick-wall.psxt");

/// Where the texture lives in VRAM. X ≥ 640 sits past the
/// double-buffered visible region (two 320×240 buffers occupy
/// X 0..640, Y 0..480). X=640 aligns to the 64-halfword tpage grid.
const TEX_TPAGE: Tpage = Tpage::new(640, 0, TexDepth::Bit4);

/// CLUT slot. X must be multiple of 16; Y is any row. (0, 480)
/// sits below both back buffers (which use Y 0..240 and 240..480
/// for the double-buffer), safely out of the framebuffer's way.
const TEX_CLUT: Clut = Clut::new(0, 480);

#[no_mangle]
fn main() {
    gpu::init(VideoMode::Ntsc, Resolution::R320X240);
    let mut fb = FrameBuffer::new(320, 240);
    gpu::set_draw_area(0, 0, 319, 239);
    gpu::set_draw_offset(0, 0);

    // --- Parse + upload the cooked texture ---
    let tex = Texture::from_bytes(BRICK_BLOB).expect("cooked brick-wall.psxt");

    // VRAM rect width is in HALFWORDS (VRAM's native unit) — not
    // texels. For 4bpp, 4 texels pack into 1 halfword, so a 64-
    // texel-wide texture occupies 16 halfwords per row.
    let pix_rect = VramRect::new(
        TEX_TPAGE.x(),
        TEX_TPAGE.y(),
        tex.halfwords_per_row(),
        tex.height(),
    );
    upload_bytes(pix_rect, tex.pixel_bytes());

    // CLUT is a single row of `clut_entries` halfwords.
    let clut_rect = VramRect::new(TEX_CLUT.x(), TEX_CLUT.y(), tex.clut_entries(), 1);
    upload_bytes(clut_rect, tex.clut_bytes());

    // --- Pre-encode the UV words so the render loop stays tight ---
    TEX_TPAGE.apply_as_draw_mode();
    // For textured rects (GP0 0x64) the second word packs U+V+CLUT.
    // Tpage info comes from the current draw-mode register, set
    // once by `apply_as_draw_mode` above — it doesn't embed in the
    // primitive like it does for textured polygons.
    let clut_word = TEX_CLUT.uv_clut_word();

    let mut frame: u16 = 0;
    loop {
        // Dark-blue clear so the wall reads bright against it.
        fb.clear(16, 16, 32);

        // Bouncing across the visible area. Same motion the previous
        // checkerboard version used so the milestone's motion
        // signature is preserved; only the texture changes.
        let t = frame as i16;
        let x = 40 + (t * 2) % 240;
        let y = 80 + ((t / 3) % 60);
        draw_sprite(x, y, tex.width(), tex.height(), (0, 0), clut_word);

        gpu::draw_sync();
        gpu::vsync();
        fb.swap();
        frame = frame.wrapping_add(1);
    }
}

/// Issue GP0 0x64 (variable-size textured rectangle) with the
/// 4-word packet. Tint `(0x80, 0x80, 0x80)` passes texels through
/// unmodulated — we see the brick colours directly rather than
/// them being multiplied by a colour mask.
fn draw_sprite(x: i16, y: i16, w: u16, h: u16, uv: (u8, u8), clut: u16) {
    wait_cmd_ready();
    write_gp0(0x6400_0000 | pack_color(0x80, 0x80, 0x80));
    write_gp0(pack_vertex(x, y));
    write_gp0(pack_texcoord(uv.0, uv.1, clut));
    write_gp0(pack_xy(w, h));
}
