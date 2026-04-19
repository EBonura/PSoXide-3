//! High-level PS1 GPU interface.
//!
//! Sits on top of `psx-io::gpu` + `psx-hw::gpu` constructors to expose
//! a friendlier API: `init()` to set up display mode, primitives
//! (`draw_tri_flat`, `draw_tri_gouraud`, `fill_rect`), and
//! synchronisation (`wait_cmd_ready`, `vsync`).
//!
//! This crate is where the engine's rendering pipeline will plug in.
//! Keeping the low-level constructors in `psx-hw` means the same
//! encoding is shared with the emulator's GPU decoder — both sides
//! can't drift out of sync on command layout.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod framebuf;
pub mod ot;
pub mod prim;

use psx_hw::gpu::{GpuStat, gp0, gp1, pack_color, pack_vertex, pack_xy};
use psx_io::dma::{self, Channel};
use psx_io::gpu::{gpustat, wait_cmd_ready, write_gp0, write_gp1};
use psx_io::timers;

/// Video standard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoMode {
    /// 60 Hz NTSC.
    Ntsc,
    /// 50 Hz PAL.
    Pal,
}

/// Display resolution. Arbitrary combinations aren't valid on hardware;
/// stick to the preset constants below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolution {
    /// Width in pixels.
    pub width: u16,
    /// Height in pixels (240 or 480).
    pub height: u16,
}

impl Resolution {
    /// 320×240 — the default for most PS1 games.
    pub const R320X240: Self = Self { width: 320, height: 240 };
    /// 256×240.
    pub const R256X240: Self = Self { width: 256, height: 240 };
    /// 512×240.
    pub const R512X240: Self = Self { width: 512, height: 240 };
    /// 640×240.
    pub const R640X240: Self = Self { width: 640, height: 240 };
    /// 320×256 — PAL's natural vertical resolution.
    pub const R320X256: Self = Self { width: 320, height: 256 };
}

/// Initialise the GPU: reset, set display mode, set display ranges,
/// configure DMA direction, enable display output.
pub fn init(mode: VideoMode, res: Resolution) {
    write_gp1(gp1::RESET);

    let hres_field = match res.width {
        256 => 0,
        320 => 1,
        512 => 2,
        640 => 3,
        _ => 1,
    };
    let vres_field = if res.height >= 480 { 1 } else { 0 };
    let pal = matches!(mode, VideoMode::Pal);

    write_gp1(gp1::display_mode(hres_field, vres_field, pal, false, false));

    // Horizontal & vertical display windows. Values below match the
    // standard PSX output (NTSC 260h..C60h, PAL similar) — tweaking
    // them shifts the picture on the TV but not the VRAM layout.
    let h_start = 0x260;
    let h_end = h_start + (res.width as u32) * 8;
    write_gp1(gp1::h_display_range(h_start, h_end));

    let (v_start, v_end) = match mode {
        VideoMode::Ntsc => (0x10, 0x10 + res.height as u32),
        VideoMode::Pal => (0x23, 0x23 + res.height as u32),
    };
    write_gp1(gp1::v_display_range(v_start, v_end));

    write_gp1(gp1::dma_direction(2)); // CPU → GP0
    write_gp1(gp1::display_enable(true));
}

/// Block until the GPU has drained its command queue.
#[inline]
pub fn draw_sync() {
    while !gpustat().contains(GpuStat::READY_DMA_RECV) {}
}

/// Wait for the next vertical blank by polling Timer 1 in HBlank-source
/// mode. The timer auto-resets at VBlank start, so we wait for it to
/// reach the VBlank scanline region, then wait for the reset.
pub fn vsync() {
    // Mode: bit0=sync enable, bits1-2=01 (reset at VBlank), bit8=1
    // (clock source = HBlank).
    timers::set_mode(timers::Timer::Timer1, 0x0103);
    while timers::counter(timers::Timer::Timer1) < 242 {}
}

/// Set the drawing-area rectangle. Pixels outside this rect are
/// clipped by the rasteriser.
pub fn set_draw_area(x0: u16, y0: u16, x1: u16, y1: u16) {
    wait_cmd_ready();
    write_gp0(gp0::draw_area_top_left(x0 as u32, y0 as u32));
    write_gp0(gp0::draw_area_bottom_right(x1 as u32, y1 as u32));
}

/// Set the drawing offset — added to every vertex by the GPU.
/// Use this to position a coordinate system at the top-left of your
/// back-buffer.
pub fn set_draw_offset(x: i16, y: i16) {
    wait_cmd_ready();
    write_gp0(gp0::draw_offset(x as i32, y as i32));
}

/// Fill a VRAM rectangle with a solid color. Ignores draw area / offset.
/// Useful for clearing a back buffer.
pub fn fill_rect(x: u16, y: u16, w: u16, h: u16, r: u8, g: u8, b: u8) {
    wait_cmd_ready();
    write_gp0(gp0::fill_rect(r, g, b));
    write_gp0(pack_xy(x, y));
    write_gp0(pack_xy(w, h));
}

/// Draw a flat-shaded (single-color) triangle.
pub fn draw_tri_flat(verts: [(i16, i16); 3], r: u8, g: u8, b: u8) {
    wait_cmd_ready();
    write_gp0(gp0::polygon_opcode(false, false, false, false, false) | pack_color(r, g, b));
    write_gp0(pack_vertex(verts[0].0, verts[0].1));
    write_gp0(pack_vertex(verts[1].0, verts[1].1));
    write_gp0(pack_vertex(verts[2].0, verts[2].1));
}

/// Draw a Gouraud-shaded triangle. `colors[i]` is the color at `verts[i]`;
/// the GPU interpolates across the triangle.
pub fn draw_tri_gouraud(verts: [(i16, i16); 3], colors: [(u8, u8, u8); 3]) {
    wait_cmd_ready();
    let op = gp0::polygon_opcode(true, false, false, false, false);
    let (r0, g0, b0) = colors[0];
    write_gp0(op | pack_color(r0, g0, b0));
    write_gp0(pack_vertex(verts[0].0, verts[0].1));
    let (r1, g1, b1) = colors[1];
    write_gp0(pack_color(r1, g1, b1));
    write_gp0(pack_vertex(verts[1].0, verts[1].1));
    let (r2, g2, b2) = colors[2];
    write_gp0(pack_color(r2, g2, b2));
    write_gp0(pack_vertex(verts[2].0, verts[2].1));
}

/// Draw a single monochrome line from `(x0, y0)` to `(x1, y1)`
/// via GP0 0x40 (single mono line, 3 words). The GPU's line
/// rasteriser handles diagonal paths correctly — unlike building
/// a line out of `fill_rect` calls, which the PSX fill-rect
/// primitive (GP0 0x02) rounds to 16-pixel X boundaries and
/// produces blocky staircase output.
///
/// Packet: `[cmd+color, v0, v1]`.
pub fn draw_line_mono(x0: i16, y0: i16, x1: i16, y1: i16, r: u8, g: u8, b: u8) {
    wait_cmd_ready();
    // 0x40 = single mono line, opaque. Color in the low 24 bits
    // of the first word (same as other monochrome primitives).
    write_gp0(0x4000_0000 | pack_color(r, g, b));
    write_gp0(pack_vertex(x0, y0));
    write_gp0(pack_vertex(x1, y1));
}

/// Draw a Gouraud-shaded line from `(x0, y0, c0)` to `(x1, y1, c1)`.
/// The GPU interpolates RGB across the segment. Packet (GP0 0x50,
/// 4 words): `[cmd+c0, v0, c1, v1]`.
pub fn draw_line_gouraud(
    x0: i16, y0: i16, c0: (u8, u8, u8),
    x1: i16, y1: i16, c1: (u8, u8, u8),
) {
    wait_cmd_ready();
    write_gp0(0x5000_0000 | pack_color(c0.0, c0.1, c0.2));
    write_gp0(pack_vertex(x0, y0));
    write_gp0(pack_color(c1.0, c1.1, c1.2));
    write_gp0(pack_vertex(x1, y1));
}

/// Draw a flat-shaded quad (two triangles sharing the v1-v2 edge).
pub fn draw_quad_flat(verts: [(i16, i16); 4], r: u8, g: u8, b: u8) {
    wait_cmd_ready();
    write_gp0(gp0::polygon_opcode(false, true, false, false, false) | pack_color(r, g, b));
    write_gp0(pack_vertex(verts[0].0, verts[0].1));
    write_gp0(pack_vertex(verts[1].0, verts[1].1));
    write_gp0(pack_vertex(verts[2].0, verts[2].1));
    write_gp0(pack_vertex(verts[3].0, verts[3].1));
}

/// Upload raw 16bpp pixels from CPU memory into a VRAM rectangle.
/// Used for font glyphs, sprites, and CLUTs — the standard
/// "CPU→VRAM transfer" pipe (GP0 0xA0 + pixel words).
///
/// Length of `pixels` must equal `w * h / 2` words (two 16bpp
/// pixels packed per word). Alignment of `pixels` doesn't matter
/// here because we push one word at a time via the FIFO; games
/// doing DMA uploads get an order-of-magnitude speedup but need
/// extra care with addresses, which we skip for this simple path.
pub fn upload_rect_raw(x: u16, y: u16, w: u16, h: u16, pixels: &[u32]) {
    wait_cmd_ready();
    write_gp0(gp0::COPY_CPU_TO_VRAM);
    write_gp0(pack_xy(x, y));
    write_gp0(pack_xy(w, h));
    for word in pixels {
        wait_cmd_ready();
        write_gp0(*word);
    }
}

/// Set the texture page + CLUT + color depth used by subsequent
/// textured primitives. Textured-rect commands (0x64..=0x7F) read
/// the texpage from the last GP0(E1h); textured polygons embed
/// the texpage in one of their UV words. Setting it via E1h is
/// a good default for sprites.
pub fn set_texture_page(tpage_x: u16, tpage_y: u16, depth: TextureDepth) {
    wait_cmd_ready();
    write_gp0(gp0::draw_mode(
        (tpage_x / 64) as u32,
        (tpage_y / 256) as u32,
        0,
        depth as u32,
        false,
        true,
    ));
}

/// Texture color depth passed to [`set_texture_page`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TextureDepth {
    /// 4-bit CLUT-indexed.
    Bit4 = 0,
    /// 8-bit CLUT-indexed.
    Bit8 = 1,
    /// 15-bit direct color.
    Bit15 = 2,
}

/// Submit a linked-list chain starting at `head` to GPU GP0 via
/// DMA channel 2 in linked-list mode. Blocks until the walker hits
/// the `0x00FFFFFF` terminator.
///
/// `head` must point at a 4-byte-aligned RAM address; the DMA
/// controller clocks bits 23..=0 of the 32-bit tag as the next-
/// node address and bits 31..=24 as that packet's data-word count.
pub fn submit_linked_list(head: *const u32) {
    draw_sync();
    // Make sure the GPU's DMA direction is CPU→GP0 before we kick
    // off the walker. `gpu::init` sets this, but games occasionally
    // re-route DMA for VRAM readback and forget to reset it.
    write_gp1(gp1::dma_direction(2));
    dma::enable_channel(Channel::Gpu);
    dma::set_madr(Channel::Gpu, head as u32);
    // BCR is ignored in linked-list mode but must be written to
    // some value on real hardware; zero is conventional.
    dma::set_bcr_manual(Channel::Gpu, 0);
    dma::set_chcr(
        Channel::Gpu,
        dma::CHCR_TO_DEVICE | dma::CHCR_SYNC_LINKED | dma::CHCR_START,
    );
    while dma::is_busy(Channel::Gpu) {}
}
