//! GPU — minimum viable surface for BIOS init + VRAM display.
//!
//! **Phase 2h scope:** the GPU owns VRAM (migrated here from the
//! top-level `vram` module — re-exported for compatibility), exposes
//! `GPUSTAT` reads with the ready-bit pattern the BIOS polls for, and
//! accepts GP0 / GP1 writes. No command processing, no rasterization,
//! no display output yet — those arrive in follow-up milestones once
//! DMA actually ships command lists.
//!
//! Register map (single-cycle MMIO, 32-bit):
//! - `0x1F80_1810` GP0 write  / `GPUREAD` read
//! - `0x1F80_1814` GP1 write / `GPUSTAT` read

use crate::vram::{Vram, VRAM_HEIGHT, VRAM_WIDTH};

/// Physical address of the GP0 / GPUREAD port.
pub const GP0_ADDR: u32 = 0x1F80_1810;
/// Physical address of the GP1 / GPUSTAT port.
pub const GP1_ADDR: u32 = 0x1F80_1814;

/// GPU state.
pub struct Gpu {
    /// Video memory — 1 MiB, 1024×512 at 16 bpp. The VRAM viewer in
    /// the frontend decodes this each frame.
    pub vram: Vram,
    status: GpuStatus,
    /// Packet assembler for GP0 commands that span multiple words.
    /// Holds words from the start of the current command; once the
    /// full packet has arrived, [`Gpu::execute_gp0_packet`] dispatches
    /// on the opcode and clears the buffer.
    gp0_fifo: Vec<u32>,
    /// Number of words the current packet expects in total (including
    /// the first/opcode word). `0` means "no packet in progress".
    gp0_expected: usize,
    /// Total GP0 writes the GPU has received since reset — diagnostic
    /// for `examples/smoke_draw` and the frontend HUD, tells us whether
    /// software has actually started shipping commands.
    gp0_write_count: u64,
    /// X offset (signed 11-bit) added to every primitive vertex —
    /// set by GP0 0xE5. Usually zero on BIOS boot, non-zero once the
    /// kernel sets up a display-list origin.
    draw_offset_x: i32,
    /// Y offset (signed 11-bit) added to every primitive vertex.
    draw_offset_y: i32,
    /// Drawing area clipping rectangle. All primitive pixels outside
    /// `[left..=right] × [top..=bottom]` are discarded. Set by
    /// GP0 0xE3 (top-left) and 0xE4 (bottom-right).
    draw_area_left: u16,
    draw_area_top: u16,
    draw_area_right: u16,
    draw_area_bottom: u16,
    /// Active CPU→VRAM transfer state. `Some` when GP0 0xA0 has set
    /// up a destination rect and is now expecting pixel-data words.
    vram_upload: Option<VramTransfer>,
    /// Active VRAM→CPU transfer state. `Some` when GP0 0xC0 set up a
    /// source rect and is now supplying pixel-data words via GPUREAD.
    vram_download: Option<VramTransfer>,
    /// Most-recent single-word response to a GP1 0x10 (Get GPU Info)
    /// request — returned on the next GPUREAD while a VRAM download
    /// isn't active. Matches hardware's "GPU info" latch.
    gpuread_latch: u32,

    // --- Texture-page state (GP0 0xE1 draw mode) ---
    /// VRAM X base of the current texture page (pixels, 0..=960,
    /// multiples of 64).
    tex_page_x: u16,
    /// VRAM Y base of the current texture page (0 or 256).
    tex_page_y: u16,
    /// Texture colour depth: 0 = 4bpp (CLUT), 1 = 8bpp (CLUT),
    /// 2 = 15bpp (direct).
    tex_depth: u8,

    // --- Display area (GP1 0x05 / 0x06 / 0x07 / 0x08) ---
    /// VRAM X of the top-left pixel of the displayed framebuffer.
    display_start_x: u16,
    /// VRAM Y of the top-left pixel of the displayed framebuffer.
    display_start_y: u16,
    /// Horizontal display resolution from GP1 0x08 (pixels). One of
    /// 256, 320, 368, 512, 640.
    display_width: u16,
    /// Vertical display resolution from GP1 0x08 (pixels). 240 or 480
    /// depending on interlace.
    display_height: u16,
    /// 24bpp colour depth flag from GP1 0x08 bit 4. For now we always
    /// decode VRAM as 15bpp; when this flag comes into play the
    /// frontend's framebuffer view can respect it.
    display_24bpp: bool,

    /// Count of executed GP0 packets by opcode byte (the high 8 bits
    /// of the header word). Diagnostic only — lets `smoke_draw` see at
    /// a glance which primitive types the BIOS is issuing.
    gp0_opcode_hist: [u32; 256],
}

/// Public snapshot of the GPU's display configuration, read by the
/// frontend's framebuffer panel. Updated by the GP1 0x05 (display
/// start) and GP1 0x08 (display mode) handlers.
#[derive(Debug, Clone, Copy)]
pub struct DisplayArea {
    /// VRAM X of the top-left displayed pixel.
    pub x: u16,
    /// VRAM Y of the top-left displayed pixel.
    pub y: u16,
    /// Horizontal resolution in pixels (one of 256/320/384/512/640).
    pub width: u16,
    /// Vertical resolution in pixels (240 or 480 interlaced).
    pub height: u16,
    /// `true` when the GP1 0x08 colour-depth bit selected 24bpp mode.
    /// The frontend framebuffer panel still decodes VRAM as 15bpp;
    /// respecting this flag is a future refinement.
    pub bpp24: bool,
}

/// In-flight CPU→VRAM transfer state — 2 pixels per incoming GP0 word,
/// written in row-major order across the destination rect. Completes
/// when `remaining == 0`, and then the GPU goes back to accepting
/// command packets on GP0.
#[derive(Clone, Copy)]
struct VramTransfer {
    x: u16,
    y: u16,
    w: u16,
    h: u16,
    /// Row of the next pixel to write (0 = top of the rect).
    row: u16,
    /// Column of the next pixel to write (0 = left of the rect).
    col: u16,
    /// Words still expected (= ceil(w*h / 2)).
    remaining: u32,
}

impl Gpu {
    /// Construct a fresh GPU — VRAM zeroed, status at the soft-GPU
    /// always-ready pattern the BIOS expects.
    pub fn new() -> Self {
        Self {
            vram: Vram::new(),
            status: GpuStatus::new(),
            gp0_fifo: Vec::with_capacity(12),
            gp0_expected: 0,
            gp0_write_count: 0,
            draw_offset_x: 0,
            draw_offset_y: 0,
            draw_area_left: 0,
            draw_area_top: 0,
            draw_area_right: VRAM_WIDTH as u16 - 1,
            draw_area_bottom: VRAM_HEIGHT as u16 - 1,
            vram_upload: None,
            tex_page_x: 0,
            tex_page_y: 0,
            tex_depth: 0,
            display_start_x: 0,
            display_start_y: 0,
            display_width: 320,
            display_height: 240,
            display_24bpp: false,
            vram_download: None,
            gpuread_latch: 0,
            gp0_opcode_hist: [0; 256],
        }
    }

    /// Snapshot of the GP0 opcode histogram — per-byte count of
    /// executed packets keyed by high-byte of word 0. Diagnostic.
    pub fn gp0_opcode_histogram(&self) -> [u32; 256] {
        self.gp0_opcode_hist
    }

    /// Snapshot of the currently-configured display area, for the
    /// frontend's framebuffer panel. Cheap to call each frame.
    pub fn display_area(&self) -> DisplayArea {
        DisplayArea {
            x: self.display_start_x,
            y: self.display_start_y,
            width: self.display_width,
            height: self.display_height,
            bpp24: self.display_24bpp,
        }
    }

    /// Total GP0 writes received since reset. Diagnostic counter.
    pub fn gp0_write_count(&self) -> u64 {
        self.gp0_write_count
    }

    /// Dispatch an MMIO read inside the GPU window. Returns `Some` for
    /// the two valid ports; `None` means the caller should fall through
    /// to a different region.
    ///
    /// `&mut self` because the GP0 (GPUREAD) port drains an in-flight
    /// VRAM→CPU transfer one word at a time; the GP1 (GPUSTAT) port
    /// stays side-effect-free.
    pub fn read32(&mut self, phys: u32) -> Option<u32> {
        match phys {
            GP0_ADDR => Some(self.read_gpuread()),
            GP1_ADDR => Some(self.status.read()),
            _ => None,
        }
    }

    /// GPUREAD — two paths:
    /// - If a VRAM→CPU transfer is active, return the next 2 packed
    ///   16bpp pixels from the source rect.
    /// - Otherwise return the latch written by the last GP1 0x10.
    fn read_gpuread(&mut self) -> u32 {
        if self.vram_download.is_some() {
            self.download_next_word()
        } else {
            self.gpuread_latch
        }
    }

    /// Toggle GPUSTAT bit 31 (interlace / even-odd line flag). Called
    /// once per VBlank by `Bus::run_vblank_scheduler`. BIOS-side code
    /// often polls this bit to tell that a new frame has started,
    /// independent of the VBlank IRQ.
    pub fn toggle_vblank_field(&mut self) {
        self.status.toggle_field();
    }

    /// Dispatch an MMIO write inside the GPU window. Returns `true` if
    /// the address belonged to the GPU.
    pub fn write32(&mut self, phys: u32, value: u32) -> bool {
        match phys {
            GP0_ADDR => {
                self.gp0_write(value);
                true
            }
            GP1_ADDR => {
                self.apply_gp1_display(value);
                self.status.gp1_write(value);
                true
            }
            _ => false,
        }
    }

    /// GP0 0xC0 — VRAM→CPU transfer. `[cmd, xy, wh]` header; pixel
    /// words are then drained by GPUREAD. Two 16bpp pixels per word
    /// in row-major order across the source rect.
    fn begin_vram_download(&mut self) {
        let xy = self.gp0_fifo[1];
        let wh = self.gp0_fifo[2];
        let x = (xy & 0x3FF) as u16;
        let y = ((xy >> 16) & 0x1FF) as u16;
        let w = {
            let raw = (wh & 0x3FF) as u16;
            if raw == 0 {
                1024
            } else {
                raw
            }
        };
        let h = {
            let raw = ((wh >> 16) & 0x1FF) as u16;
            if raw == 0 {
                512
            } else {
                raw
            }
        };
        let pixels = w as u32 * h as u32;
        let remaining = pixels.div_ceil(2);
        self.vram_download = Some(VramTransfer {
            x,
            y,
            w,
            h,
            row: 0,
            col: 0,
            remaining,
        });
    }

    /// Pop two pixels from the active VRAM→CPU transfer, packed into
    /// a u32 (low 16 = first pixel, high 16 = second). When the
    /// transfer completes, the download slot clears and subsequent
    /// GPUREAD reads return the GP1 0x10 latch.
    fn download_next_word(&mut self) -> u32 {
        let Some(t) = self.vram_download.as_mut() else {
            return self.gpuread_latch;
        };
        let pix_a = Self::read_download_pixel(t, &self.vram);
        let pix_b = Self::read_download_pixel(t, &self.vram);
        t.remaining = t.remaining.saturating_sub(1);
        let word = (pix_a as u32) | ((pix_b as u32) << 16);
        if t.remaining == 0 {
            self.vram_download = None;
        }
        word
    }

    /// Fetch the next pixel from the source rect for a VRAM→CPU
    /// download. Advances row/col; over-draws past the final row
    /// return zero (paired-halving at odd widths).
    fn read_download_pixel(t: &mut VramTransfer, vram: &Vram) -> u16 {
        if t.row >= t.h {
            return 0;
        }
        let px = t.x.wrapping_add(t.col);
        let py = t.y.wrapping_add(t.row);
        let texel = vram.get_pixel(px, py);
        t.col += 1;
        if t.col >= t.w {
            t.col = 0;
            t.row += 1;
        }
        texel
    }

    /// Handle GP1 commands that update the display-area state
    /// (0x05 / 0x06 / 0x07 / 0x08) or the GPU-info latch (0x10).
    /// The status-bit updates stay in `GpuStatus::gp1_write`; this
    /// function captures the geometry + latch the frontend + CPU need.
    fn apply_gp1_display(&mut self, value: u32) {
        let cmd = (value >> 24) & 0xFF;
        match cmd {
            // GP1 0x10 — Get GPU Info. Sub-op selects what latches
            // into GPUREAD. See nocash PSX-SPX "GPU Memory Transfer
            // Commands / GP1(10h)". Common sub-ops:
            //   0x02 — texture window (E2 readback)
            //   0x03 — draw area top-left  (E3 readback)
            //   0x04 — draw area bottom-right (E4)
            //   0x05 — draw offset (E5)
            //   0x07 — GPU version / misc (returns 0x0000_0002)
            //   0x08 — unknown / returns 0
            0x10 => {
                let sub_op = value & 0x0F;
                self.gpuread_latch = match sub_op {
                    0x03 => (self.draw_area_left as u32) | ((self.draw_area_top as u32) << 10),
                    0x04 => (self.draw_area_right as u32) | ((self.draw_area_bottom as u32) << 10),
                    0x05 => {
                        let x = (self.draw_offset_x as u32) & 0x7FF;
                        let y = (self.draw_offset_y as u32) & 0x7FF;
                        x | (y << 11)
                    }
                    0x07 => 0x0000_0002,
                    _ => 0,
                };
            }
            // GP1 0x00 — GPU reset — also resets the display area.
            0x00 => {
                self.display_start_x = 0;
                self.display_start_y = 0;
                self.display_width = 320;
                self.display_height = 240;
                self.display_24bpp = false;
            }
            // GP1 0x05 — display area start (top-left corner in VRAM).
            //   bits 9:0  = X (pixels)
            //   bits 18:10 = Y (pixels)
            0x05 => {
                self.display_start_x = (value & 0x3FF) as u16;
                self.display_start_y = ((value >> 10) & 0x1FF) as u16;
            }
            // GP1 0x08 — display mode. Resolve to pixel W/H.
            0x08 => {
                let hres = match value & 0x3 {
                    0 => 256,
                    1 => 320,
                    2 => 512,
                    3 => 640,
                    _ => unreachable!(),
                };
                let hres = if value & (1 << 6) != 0 { 384 } else { hres };
                self.display_width = hres;
                self.display_height = if value & (1 << 2) != 0 { 480 } else { 240 };
                self.display_24bpp = value & (1 << 4) != 0;
            }
            _ => {}
        }
    }

    /// Feed one 32-bit word to the GP0 packet assembler. Public so DMA
    /// channel 2 can ship words through the same path CPU-direct writes
    /// take.
    pub fn gp0_push(&mut self, word: u32) {
        self.gp0_write(word);
    }

    /// Feed one 32-bit word to the GP0 packet assembler. If this word
    /// completes a packet, the packet is executed and the FIFO clears.
    fn gp0_write(&mut self, word: u32) {
        self.gp0_write_count += 1;

        // CPU→VRAM transfer consumes pixel words ahead of the packet
        // assembler — it's a mode, not a packet.
        if self.vram_upload.is_some() {
            self.ingest_vram_upload_word(word);
            return;
        }

        if self.gp0_expected == 0 {
            let op = (word >> 24) & 0xFF;
            self.gp0_expected = gp0_packet_size(op as u8);
            // Single-word commands execute immediately without buffering.
            if self.gp0_expected == 1 {
                self.execute_gp0_single(word);
                self.gp0_expected = 0;
                return;
            }
        }
        self.gp0_fifo.push(word);
        if self.gp0_fifo.len() == self.gp0_expected {
            self.execute_gp0_packet();
            self.gp0_fifo.clear();
            self.gp0_expected = 0;
        }
    }

    /// Execute a command whose packet size is exactly 1. Draw-mode
    /// setters (GP0 0xE1..=0xE6) live here; we track drawing-area
    /// and drawing-offset because the rasterizer needs them.
    fn execute_gp0_single(&mut self, word: u32) {
        let op = (word >> 24) & 0xFF;
        match op {
            // GP0 0xE3 — drawing area top-left. X bits 9:0, Y bits 18:10.
            0xE3 => {
                self.draw_area_left = (word & 0x3FF) as u16;
                self.draw_area_top = ((word >> 10) & 0x1FF) as u16;
            }
            // GP0 0xE4 — drawing area bottom-right.
            0xE4 => {
                self.draw_area_right = (word & 0x3FF) as u16;
                self.draw_area_bottom = ((word >> 10) & 0x1FF) as u16;
            }
            // GP0 0xE5 — drawing offset. X / Y are both signed 11-bit.
            0xE5 => {
                self.draw_offset_x = sign_extend_11((word & 0x7FF) as i32);
                self.draw_offset_y = sign_extend_11(((word >> 11) & 0x7FF) as i32);
            }
            // GP0 0xE1 — draw mode: texture page base + colour depth
            // + dither/display/transparency flags. We pick up the
            // bits the texture rasterizer consults; the rest are for
            // future work.
            0xE1 => {
                self.tex_page_x = ((word & 0x0F) as u16) * 64;
                self.tex_page_y = if (word >> 4) & 1 != 0 { 256 } else { 0 };
                self.tex_depth = ((word >> 7) & 0x3) as u8;
            }
            // 0xE2 (texture window), 0xE6 (mask bit): not wired up yet.
            _ => {}
        }
    }

    /// Execute a multi-word packet that has just been fully assembled
    /// in `gp0_fifo`. Dispatches on the opcode in word 0.
    fn execute_gp0_packet(&mut self) {
        let op = (self.gp0_fifo[0] >> 24) & 0xFF;
        self.gp0_opcode_hist[op as usize] =
            self.gp0_opcode_hist[op as usize].saturating_add(1);
        match op {
            // Monochrome fill rect (ignores draw area / offset).
            0x02 => self.fill_rect(),
            // Monochrome triangle / quad. Bit 3 distinguishes 3-vs-4
            // vertices; bit 1 is opaque-vs-semi-transparent (we treat
            // both as opaque for now).
            0x20..=0x23 => self.draw_monochrome_tri(),
            0x28..=0x2B => self.draw_monochrome_quad(),
            // Gouraud-shaded triangle / quad — per-vertex colour
            // interpolated across the primitive via barycentrics.
            0x30..=0x33 => self.draw_shaded_tri(),
            0x38..=0x3B => self.draw_shaded_quad(),
            // Textured (flat-shade) triangle / quad — per-vertex UV,
            // texture-page and CLUT pulled from the UV words.
            0x24..=0x27 => self.draw_textured_tri(),
            0x2C..=0x2F => self.draw_textured_quad(),
            // Monochrome rectangles — bit 3 set selects variable size
            // (followed by a W/H word), else 1×1/8×8/16×16 by bits 5:4.
            0x60..=0x63 => self.draw_monochrome_rect_variable(),
            0x68..=0x6B => self.draw_monochrome_rect_sized(1, 1),
            0x70..=0x73 => self.draw_monochrome_rect_sized(8, 8),
            0x78..=0x7B => self.draw_monochrome_rect_sized(16, 16),
            // Textured rectangles — same geometry as the monochrome
            // variants plus a UV/CLUT word between pos and size.
            0x64..=0x67 => self.draw_textured_rect_variable(),
            0x6C..=0x6F => self.draw_textured_rect_sized(1, 1),
            0x74..=0x77 => self.draw_textured_rect_sized(8, 8),
            0x7C..=0x7F => self.draw_textured_rect_sized(16, 16),
            // CPU→VRAM transfer — 3 words of setup, then `w*h/2`
            // words of pixel data follow as a separate mode.
            0xA0 => self.begin_vram_upload(),
            // VRAM→CPU transfer — 3 words of setup, pixel words are
            // then pulled by the CPU via GPUREAD.
            0xC0 => self.begin_vram_download(),
            // VRAM→VRAM copy — source rect blitted to dest rect.
            0x80..=0x9F => self.vram_to_vram_copy(),
            _ => {}
        }
    }

    /// GP0 0x80 — copy a rectangle of VRAM to another VRAM location.
    /// Packet: `[cmd, src_xy, dst_xy, wh]`. PS1 hardware masks the
    /// coordinates to VRAM size and wraps at the edges; our `get_pixel`
    /// / `set_pixel` already do that, so a simple double loop suffices.
    /// We buffer the source row into a temp so that overlapping
    /// src/dst rects blit correctly.
    fn vram_to_vram_copy(&mut self) {
        let src_word = self.gp0_fifo[1];
        let dst_word = self.gp0_fifo[2];
        let wh_word = self.gp0_fifo[3];
        let sx = (src_word & 0xFFFF) as u16;
        let sy = ((src_word >> 16) & 0xFFFF) as u16;
        let dx = (dst_word & 0xFFFF) as u16;
        let dy = ((dst_word >> 16) & 0xFFFF) as u16;
        // Width / height: 0 → 1024 / 512 (mask-and-wrap convention).
        let raw_w = (wh_word & 0xFFFF) as u16;
        let raw_h = ((wh_word >> 16) & 0xFFFF) as u16;
        let w = if raw_w == 0 { 1024 } else { raw_w };
        let h = if raw_h == 0 { 512 } else { raw_h };
        let mut row = vec![0u16; w as usize];
        for dy_off in 0..h {
            for dx_off in 0..w {
                row[dx_off as usize] = self.vram.get_pixel(sx + dx_off, sy + dy_off);
            }
            for dx_off in 0..w {
                self.vram
                    .set_pixel(dx + dx_off, dy + dy_off, row[dx_off as usize]);
            }
        }
    }

    // --- Primitive rasterization ---

    /// Parse a polygon vertex word: low 16 bits X, next 16 bits Y,
    /// both signed 11-bit. The drawing-offset is added here so callers
    /// get screen-space coordinates ready to rasterize.
    fn decode_vertex(&self, word: u32) -> (i32, i32) {
        let x = sign_extend_11((word & 0x7FF) as i32) + self.draw_offset_x;
        let y = sign_extend_11(((word >> 16) & 0x7FF) as i32) + self.draw_offset_y;
        (x, y)
    }

    /// GP0 0x20..=0x23 — monochrome 3-vertex triangle.
    /// Words: `[cmd+color, v0, v1, v2]`.
    fn draw_monochrome_tri(&mut self) {
        let color = rgb24_to_bgr15(self.gp0_fifo[0] & 0x00FF_FFFF);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let v1 = self.decode_vertex(self.gp0_fifo[2]);
        let v2 = self.decode_vertex(self.gp0_fifo[3]);
        self.rasterize_triangle(v0, v1, v2, color);
    }

    /// GP0 0x28..=0x2B — monochrome 4-vertex quad, split into two
    /// triangles `(v0, v1, v2)` + `(v1, v2, v3)`.
    fn draw_monochrome_quad(&mut self) {
        let color = rgb24_to_bgr15(self.gp0_fifo[0] & 0x00FF_FFFF);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let v1 = self.decode_vertex(self.gp0_fifo[2]);
        let v2 = self.decode_vertex(self.gp0_fifo[3]);
        let v3 = self.decode_vertex(self.gp0_fifo[4]);
        self.rasterize_triangle(v0, v1, v2, color);
        self.rasterize_triangle(v1, v2, v3, color);
    }

    /// GP0 0x30..=0x33 — Gouraud triangle. Per-vertex RGB24 colours,
    /// interpolated across the triangle via barycentric weights.
    /// Words: `[cmd+c0, v0, c1, v1, c2, v2]`.
    fn draw_shaded_tri(&mut self) {
        let c0 = self.gp0_fifo[0] & 0x00FF_FFFF;
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let c1 = self.gp0_fifo[2] & 0x00FF_FFFF;
        let v1 = self.decode_vertex(self.gp0_fifo[3]);
        let c2 = self.gp0_fifo[4] & 0x00FF_FFFF;
        let v2 = self.decode_vertex(self.gp0_fifo[5]);
        self.rasterize_shaded_triangle(v0, v1, v2, c0, c1, c2);
    }

    /// GP0 0x38..=0x3B — Gouraud quad. 4 × (colour+vertex) =
    /// 8 words, split into two shaded triangles sharing the middle
    /// edge (v1–v2).
    fn draw_shaded_quad(&mut self) {
        let c0 = self.gp0_fifo[0] & 0x00FF_FFFF;
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let c1 = self.gp0_fifo[2] & 0x00FF_FFFF;
        let v1 = self.decode_vertex(self.gp0_fifo[3]);
        let c2 = self.gp0_fifo[4] & 0x00FF_FFFF;
        let v2 = self.decode_vertex(self.gp0_fifo[5]);
        let c3 = self.gp0_fifo[6] & 0x00FF_FFFF;
        let v3 = self.decode_vertex(self.gp0_fifo[7]);
        self.rasterize_shaded_triangle(v0, v1, v2, c0, c1, c2);
        self.rasterize_shaded_triangle(v1, v2, v3, c1, c2, c3);
    }

    /// GP0 0x60..=0x63 — monochrome variable-size rectangle.
    /// Words: `[cmd+color, xy, wh]`.
    fn draw_monochrome_rect_variable(&mut self) {
        let color = rgb24_to_bgr15(self.gp0_fifo[0] & 0x00FF_FFFF);
        let pos = self.gp0_fifo[1];
        let size = self.gp0_fifo[2];
        let x = sign_extend_11((pos & 0x7FF) as i32) + self.draw_offset_x;
        let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32) + self.draw_offset_y;
        let w = (size & 0xFFFF) as i32;
        let h = ((size >> 16) & 0xFFFF) as i32;
        self.paint_rect(x, y, w, h, color);
    }

    /// GP0 0x68/0x70/0x78 — fixed-size monochrome rectangles.
    fn draw_monochrome_rect_sized(&mut self, w: i32, h: i32) {
        let color = rgb24_to_bgr15(self.gp0_fifo[0] & 0x00FF_FFFF);
        let pos = self.gp0_fifo[1];
        let x = sign_extend_11((pos & 0x7FF) as i32) + self.draw_offset_x;
        let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32) + self.draw_offset_y;
        self.paint_rect(x, y, w, h, color);
    }

    /// Variable-size textured rect. Words: `[cmd+tint, xy, clut+uv, wh]`.
    fn draw_textured_rect_variable(&mut self) {
        let pos = self.gp0_fifo[1];
        let uv_clut = self.gp0_fifo[2];
        let size = self.gp0_fifo[3];
        let x = sign_extend_11((pos & 0x7FF) as i32) + self.draw_offset_x;
        let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32) + self.draw_offset_y;
        let w = (size & 0xFFFF) as i32;
        let h = ((size >> 16) & 0xFFFF) as i32;
        let u0 = (uv_clut & 0xFF) as u16;
        let v0 = ((uv_clut >> 8) & 0xFF) as u16;
        let clut_word = ((uv_clut >> 16) & 0xFFFF) as u16;
        self.paint_textured_rect(x, y, w, h, u0, v0, clut_word);
    }

    /// Fixed-size textured rect (1×1, 8×8, 16×16).
    /// Words: `[cmd+tint, xy, clut+uv]`.
    fn draw_textured_rect_sized(&mut self, w: i32, h: i32) {
        let pos = self.gp0_fifo[1];
        let uv_clut = self.gp0_fifo[2];
        let x = sign_extend_11((pos & 0x7FF) as i32) + self.draw_offset_x;
        let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32) + self.draw_offset_y;
        let u0 = (uv_clut & 0xFF) as u16;
        let v0 = ((uv_clut >> 8) & 0xFF) as u16;
        let clut_word = ((uv_clut >> 16) & 0xFFFF) as u16;
        self.paint_textured_rect(x, y, w, h, u0, v0, clut_word);
    }

    /// Plot a textured rectangle. Each destination pixel samples a
    /// 1:1 texel from the current texture page, CLUT-indexed for
    /// 4bpp / 8bpp modes, direct for 15bpp. Texels of value 0 are
    /// transparent (standard PS1 convention).
    #[allow(clippy::too_many_arguments)]
    fn paint_textured_rect(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        u0: u16,
        v0: u16,
        clut_word: u16,
    ) {
        if w <= 0 || h <= 0 {
            return;
        }
        let clut_x = (clut_word & 0x3F) * 16;
        let clut_y = (clut_word >> 6) & 0x1FF;

        let left = x.max(self.draw_area_left as i32);
        let top = y.max(self.draw_area_top as i32);
        let right = (x + w - 1).min(self.draw_area_right as i32);
        let bottom = (y + h - 1).min(self.draw_area_bottom as i32);
        if left > right || top > bottom {
            return;
        }

        for py in top..=bottom {
            for px in left..=right {
                let tex_u = u0.wrapping_add((px - x) as u16);
                let tex_v = v0.wrapping_add((py - y) as u16);
                if let Some(texel) = self.sample_texture(tex_u, tex_v, clut_x, clut_y) {
                    self.vram.set_pixel(px as u16, py as u16, texel);
                }
            }
        }
    }

    /// Fetch a single texel from the active texture page. Returns
    /// `None` for transparent (texel value 0 in CLUT modes, or 0x0000
    /// with mask bit clear in direct mode).
    fn sample_texture(&self, u: u16, v: u16, clut_x: u16, clut_y: u16) -> Option<u16> {
        let tpy = self.tex_page_y.wrapping_add(v);
        match self.tex_depth {
            0 => {
                // 4bpp: 4 texels per VRAM word; select by (u & 3).
                let tpx = self.tex_page_x.wrapping_add(u / 4);
                let word = self.vram.get_pixel(tpx, tpy);
                let idx = (word >> ((u & 3) * 4)) & 0xF;
                if idx == 0 {
                    None
                } else {
                    Some(self.vram.get_pixel(clut_x + idx, clut_y))
                }
            }
            1 => {
                // 8bpp: 2 texels per VRAM word.
                let tpx = self.tex_page_x.wrapping_add(u / 2);
                let word = self.vram.get_pixel(tpx, tpy);
                let idx = (word >> ((u & 1) * 8)) & 0xFF;
                if idx == 0 {
                    None
                } else {
                    Some(self.vram.get_pixel(clut_x + idx, clut_y))
                }
            }
            _ => {
                // 15bpp: direct colour, 1 texel per word.
                let tpx = self.tex_page_x.wrapping_add(u);
                let texel = self.vram.get_pixel(tpx, tpy);
                if texel == 0 {
                    None
                } else {
                    Some(texel)
                }
            }
        }
    }

    /// Plot a rectangle of `color` in screen-space, clipped to the
    /// GPU's drawing area.
    fn paint_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u16) {
        if w <= 0 || h <= 0 {
            return;
        }
        let left = x.max(self.draw_area_left as i32);
        let top = y.max(self.draw_area_top as i32);
        let right = (x + w - 1).min(self.draw_area_right as i32);
        let bottom = (y + h - 1).min(self.draw_area_bottom as i32);
        if left > right || top > bottom {
            return;
        }
        for py in top..=bottom {
            for px in left..=right {
                self.vram.set_pixel(px as u16, py as u16, color);
            }
        }
    }

    /// Scanline-ish triangle rasterizer using the edge-function test.
    /// For each pixel in the bounding box we evaluate three edge
    /// equations; a pixel is inside iff all three have the same sign.
    /// Works regardless of triangle winding. Clipped to both VRAM
    /// bounds and the active drawing area.
    fn rasterize_triangle(&mut self, v0: (i32, i32), v1: (i32, i32), v2: (i32, i32), color: u16) {
        let min_x = v0.0.min(v1.0).min(v2.0).max(self.draw_area_left as i32);
        let max_x = v0.0.max(v1.0).max(v2.0).min(self.draw_area_right as i32);
        let min_y = v0.1.min(v1.1).min(v2.1).max(self.draw_area_top as i32);
        let max_y = v0.1.max(v1.1).max(v2.1).min(self.draw_area_bottom as i32);
        if min_x > max_x || min_y > max_y {
            return;
        }

        // Precompute fixed area; we just need its sign to normalize.
        let area = edge(v0, v1, v2);
        if area == 0 {
            return; // degenerate
        }
        let area_sign = area.signum();

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = (x, y);
                let w0 = edge(v1, v2, p) * area_sign;
                let w1 = edge(v2, v0, p) * area_sign;
                let w2 = edge(v0, v1, p) * area_sign;
                if (w0 | w1 | w2) >= 0 {
                    self.vram.set_pixel(x as u16, y as u16, color);
                }
            }
        }
    }

    /// GP0 0x24..=0x27 — textured triangle. 7 words:
    /// `[cmd+tint, v0, clut+uv0, v1, tpage+uv1, v2, uv2]`.
    fn draw_textured_tri(&mut self) {
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let uv0 = self.gp0_fifo[2];
        let v1 = self.decode_vertex(self.gp0_fifo[3]);
        let uv1 = self.gp0_fifo[4];
        let v2 = self.decode_vertex(self.gp0_fifo[5]);
        let uv2 = self.gp0_fifo[6];
        let clut_word = ((uv0 >> 16) & 0xFFFF) as u16;
        // The tpage word in UV1 overrides the current draw-mode tpage
        // for the duration of this primitive.
        self.apply_primitive_tpage(uv1);
        let t0 = ((uv0 & 0xFF) as u16, ((uv0 >> 8) & 0xFF) as u16);
        let t1 = ((uv1 & 0xFF) as u16, ((uv1 >> 8) & 0xFF) as u16);
        let t2 = ((uv2 & 0xFF) as u16, ((uv2 >> 8) & 0xFF) as u16);
        self.rasterize_textured_triangle(v0, v1, v2, t0, t1, t2, clut_word);
    }

    /// GP0 0x2C..=0x2F — textured quad. 9 words; split into two
    /// textured triangles sharing v1–v2.
    fn draw_textured_quad(&mut self) {
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let uv0 = self.gp0_fifo[2];
        let v1 = self.decode_vertex(self.gp0_fifo[3]);
        let uv1 = self.gp0_fifo[4];
        let v2 = self.decode_vertex(self.gp0_fifo[5]);
        let uv2 = self.gp0_fifo[6];
        let v3 = self.decode_vertex(self.gp0_fifo[7]);
        let uv3 = self.gp0_fifo[8];
        let clut_word = ((uv0 >> 16) & 0xFFFF) as u16;
        self.apply_primitive_tpage(uv1);
        let t0 = ((uv0 & 0xFF) as u16, ((uv0 >> 8) & 0xFF) as u16);
        let t1 = ((uv1 & 0xFF) as u16, ((uv1 >> 8) & 0xFF) as u16);
        let t2 = ((uv2 & 0xFF) as u16, ((uv2 >> 8) & 0xFF) as u16);
        let t3 = ((uv3 & 0xFF) as u16, ((uv3 >> 8) & 0xFF) as u16);
        self.rasterize_textured_triangle(v0, v1, v2, t0, t1, t2, clut_word);
        self.rasterize_textured_triangle(v1, v2, v3, t1, t2, t3, clut_word);
    }

    /// Apply the tpage bits embedded in a textured-primitive UV word
    /// (they override the draw-mode tpage for this primitive onward).
    fn apply_primitive_tpage(&mut self, uv_word: u32) {
        let tpage = (uv_word >> 16) & 0xFFFF;
        self.tex_page_x = ((tpage & 0x0F) as u16) * 64;
        self.tex_page_y = if (tpage >> 4) & 1 != 0 { 256 } else { 0 };
        self.tex_depth = ((tpage >> 7) & 0x3) as u8;
    }

    /// Rasterize a textured triangle — same edge-function test as the
    /// other triangle paths, with nearest-neighbor texture sampling
    /// via barycentric-interpolated UV.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_textured_triangle(
        &mut self,
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        t0: (u16, u16),
        t1: (u16, u16),
        t2: (u16, u16),
        clut_word: u16,
    ) {
        let min_x = v0.0.min(v1.0).min(v2.0).max(self.draw_area_left as i32);
        let max_x = v0.0.max(v1.0).max(v2.0).min(self.draw_area_right as i32);
        let min_y = v0.1.min(v1.1).min(v2.1).max(self.draw_area_top as i32);
        let max_y = v0.1.max(v1.1).max(v2.1).min(self.draw_area_bottom as i32);
        if min_x > max_x || min_y > max_y {
            return;
        }

        let area = edge(v0, v1, v2);
        if area == 0 {
            return;
        }
        let area_sign = area.signum();
        let area_abs = area.unsigned_abs() as i64;

        let clut_x = (clut_word & 0x3F) * 16;
        let clut_y = (clut_word >> 6) & 0x1FF;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = (x, y);
                let w0 = (edge(v1, v2, p) * area_sign) as i64;
                let w1 = (edge(v2, v0, p) * area_sign) as i64;
                let w2 = (edge(v0, v1, p) * area_sign) as i64;
                if (w0 | w1 | w2) >= 0 {
                    let u = (w0 * t0.0 as i64 + w1 * t1.0 as i64 + w2 * t2.0 as i64) / area_abs;
                    let v = (w0 * t0.1 as i64 + w1 * t1.1 as i64 + w2 * t2.1 as i64) / area_abs;
                    if let Some(texel) = self.sample_texture(u as u16, v as u16, clut_x, clut_y) {
                        self.vram.set_pixel(x as u16, y as u16, texel);
                    }
                }
            }
        }
    }

    /// Rasterize a triangle with per-vertex colours — Gouraud shading.
    /// Same edge-function inside test as the flat path, but interpolates
    /// RGB using normalized barycentric weights `(w0, w1, w2)` per pixel
    /// and packs the result back into a 15-bit BGR VRAM word.
    fn rasterize_shaded_triangle(
        &mut self,
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        c0: u32,
        c1: u32,
        c2: u32,
    ) {
        let min_x = v0.0.min(v1.0).min(v2.0).max(self.draw_area_left as i32);
        let max_x = v0.0.max(v1.0).max(v2.0).min(self.draw_area_right as i32);
        let min_y = v0.1.min(v1.1).min(v2.1).max(self.draw_area_top as i32);
        let max_y = v0.1.max(v1.1).max(v2.1).min(self.draw_area_bottom as i32);
        if min_x > max_x || min_y > max_y {
            return;
        }

        let area = edge(v0, v1, v2);
        if area == 0 {
            return;
        }
        let area_sign = area.signum();
        let area_abs = area.unsigned_abs() as i64;

        // Channel-extract closures — r/g/b are low/mid/high bytes of the
        // 24-bit word written in the command.
        let r = |c: u32| (c & 0xFF) as i64;
        let g = |c: u32| ((c >> 8) & 0xFF) as i64;
        let b = |c: u32| ((c >> 16) & 0xFF) as i64;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = (x, y);
                let w0 = (edge(v1, v2, p) * area_sign) as i64;
                let w1 = (edge(v2, v0, p) * area_sign) as i64;
                let w2 = (edge(v0, v1, p) * area_sign) as i64;
                if (w0 | w1 | w2) >= 0 {
                    let ri = (w0 * r(c0) + w1 * r(c1) + w2 * r(c2)) / area_abs;
                    let gi = (w0 * g(c0) + w1 * g(c1) + w2 * g(c2)) / area_abs;
                    let bi = (w0 * b(c0) + w1 * b(c1) + w2 * b(c2)) / area_abs;
                    let colour = rgb24_to_bgr15(
                        (ri.clamp(0, 255) as u32)
                            | ((gi.clamp(0, 255) as u32) << 8)
                            | ((bi.clamp(0, 255) as u32) << 16),
                    );
                    self.vram.set_pixel(x as u16, y as u16, colour);
                }
            }
        }
    }

    // --- CPU→VRAM transfer (GP0 0xA0) ---

    /// GP0 0xA0 — start a CPU-to-VRAM transfer. `[cmd, xy, wh]` is
    /// followed (in subsequent GP0 writes) by `ceil(w*h / 2)` words
    /// of 16bpp pixel data, 2 pixels per word. Transfer state lives
    /// in [`Gpu::vram_upload`] until every pixel has been ingested.
    fn begin_vram_upload(&mut self) {
        let xy = self.gp0_fifo[1];
        let wh = self.gp0_fifo[2];
        let x = (xy & 0x3FF) as u16;
        let y = ((xy >> 16) & 0x1FF) as u16;
        // Hardware uses a wrap-around convention: width/height of 0
        // means 1024 / 512 respectively. Matches Redux.
        let w = {
            let raw = (wh & 0x3FF) as u16;
            if raw == 0 {
                1024
            } else {
                raw
            }
        };
        let h = {
            let raw = ((wh >> 16) & 0x1FF) as u16;
            if raw == 0 {
                512
            } else {
                raw
            }
        };
        let pixels = w as u32 * h as u32;
        // Two 16bpp pixels per 32-bit word, round up.
        let remaining = pixels.div_ceil(2);
        self.vram_upload = Some(VramTransfer {
            x,
            y,
            w,
            h,
            row: 0,
            col: 0,
            remaining,
        });
    }

    /// Consume one word of pixel data for the active CPU→VRAM
    /// transfer. When `remaining` hits zero, the transfer closes and
    /// the next GP0 write is interpreted as a new command.
    fn ingest_vram_upload_word(&mut self, word: u32) {
        let done = {
            let Some(t) = self.vram_upload.as_mut() else {
                return;
            };
            let pix_a = word as u16;
            let pix_b = (word >> 16) as u16;
            Self::write_upload_pixel(t, pix_a, &mut self.vram);
            Self::write_upload_pixel(t, pix_b, &mut self.vram);
            t.remaining = t.remaining.saturating_sub(1);
            t.remaining == 0
        };
        if done {
            self.vram_upload = None;
        }
    }

    /// Place the next pixel in an active upload. Advances `col`; at
    /// the right edge wraps to the next `row`. Pixels past the final
    /// row are silently dropped (VRAM wrap on the destination only
    /// applies to coordinates, not to an over-long upload payload).
    fn write_upload_pixel(t: &mut VramTransfer, pixel: u16, vram: &mut Vram) {
        if t.row >= t.h {
            return;
        }
        let px = t.x.wrapping_add(t.col);
        let py = t.y.wrapping_add(t.row);
        vram.set_pixel(px, py, pixel);
        t.col += 1;
        if t.col >= t.w {
            t.col = 0;
            t.row += 1;
        }
    }

    /// GP0 0x02 — monochrome fill rectangle, ignores draw mode /
    /// clipping / blending. Writes `color` directly into VRAM.
    ///
    /// Packet layout (Redux `GPU::cmdFillRect`):
    ///   word 0: `0x02RRGGBB`      — opcode + 24-bit RGB
    ///   word 1: `0xYYYYXXXX`      — top-left: X is 16-pixel-aligned
    ///   word 2: `0xHHHHWWWW`      — width is rounded up to 16 pixels
    ///
    /// Both coordinates and sizes wrap mod VRAM dimensions.
    fn fill_rect(&mut self) {
        let color24 = self.gp0_fifo[0] & 0x00FF_FFFF;
        let (x, y) = {
            let w = self.gp0_fifo[1];
            // X is aligned to 16-pixel boundaries; low 4 bits ignored.
            let x = (w & 0x3F0) as u16;
            let y = ((w >> 16) & 0x1FF) as u16;
            (x, y)
        };
        let (w, h) = {
            let s = self.gp0_fifo[2];
            // Width rounded up to next multiple of 16.
            let w = (((s & 0x3FF) + 0x0F) & !0x0F) as u16;
            let h = ((s >> 16) & 0x1FF) as u16;
            (w, h)
        };

        let color15 = rgb24_to_bgr15(color24);
        for row in 0..h {
            for col in 0..w {
                let px = (x + col) as usize % VRAM_WIDTH;
                let py = (y + row) as usize % VRAM_HEIGHT;
                self.vram.set_pixel(px as u16, py as u16, color15);
            }
        }
    }
}

/// Expected total word count for a GP0 command starting with opcode `op`.
///
/// For commands we don't decode yet (textured / shaded primitives,
/// VRAM-to-VRAM blits, VRAM-to-CPU transfers), returning `1` means
/// we consume just the first word and drop subsequent data on the
/// floor. That's usually harmless because nothing reads VRAM from
/// those paths yet — when we add them, the packet size here grows
/// to match.
fn gp0_packet_size(op: u8) -> usize {
    match op {
        // NOP / clear cache / misc — single word.
        0x00 | 0x01 | 0x03..=0x1E => 1,
        // Quick fill — RGB + (X,Y) + (W,H) = 3 words.
        0x02 => 3,
        // Monochrome flat triangle: color + 3 vertices = 4 words.
        0x20..=0x23 => 4,
        // Shaded flat triangle: 3 × (color+vertex) = 6 words.
        0x30..=0x33 => 6,
        // Monochrome flat quad: color + 4 vertices = 5 words.
        0x28..=0x2B => 5,
        // Shaded flat quad: 4 × (color+vertex) = 8 words.
        0x38..=0x3B => 8,
        // Textured primitives (flat + textured): 3-vert = 7, 4-vert = 9.
        0x24..=0x27 => 7,
        0x2C..=0x2F => 9,
        // Textured + shaded: 3-vert = 9, 4-vert = 12.
        0x34..=0x37 => 9,
        0x3C..=0x3F => 12,
        // Lines: 3 words (color + 2 vertices) for monochrome,
        // 4 words for shaded.
        0x40..=0x43 => 3,
        0x50..=0x53 => 4,
        // Monochrome rectangles: variable (3), 1×1 (2), 8×8 (2), 16×16 (2).
        0x60..=0x63 => 3,
        0x64..=0x67 => 4, // variable + textured
        0x68..=0x6B => 2,
        0x6C..=0x6F => 3, // 1×1 textured
        0x70..=0x73 => 2,
        0x74..=0x77 => 3, // 8×8 textured
        0x78..=0x7B => 2,
        0x7C..=0x7F => 3, // 16×16 textured
        // VRAM-to-VRAM copy: opcode + src + dst + size = 4 words.
        0x80..=0x9F => 4,
        // CPU-to-VRAM transfer: opcode + xy + wh = 3 words, then
        // pixel data is consumed in a separate "upload mode".
        0xA0..=0xBF => 3,
        // VRAM-to-CPU: same 3-word header.
        0xC0..=0xDF => 3,
        // Draw-mode settings (E1..=E6) — single word each.
        0xE1..=0xE6 => 1,
        _ => 1,
    }
}

/// Triangle edge function: signed doubled area of △(a, b, c). Positive
/// when the winding is counter-clockwise. Used by `rasterize_triangle`
/// for per-pixel inside/outside tests.
fn edge(a: (i32, i32), b: (i32, i32), c: (i32, i32)) -> i32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Sign-extend an 11-bit integer (PS1 vertex coords + drawing offset
/// are 11-bit signed).
fn sign_extend_11(v: i32) -> i32 {
    if v & 0x400 != 0 {
        v | !0x7FF
    } else {
        v & 0x7FF
    }
}

/// Convert a 24-bit RGB value (as written by the CPU in GP0 packets)
/// into the 15-bit BGR word VRAM stores. Matches Redux / PS1
/// hardware: the 3 high bits of each channel are discarded.
fn rgb24_to_bgr15(rgb24: u32) -> u16 {
    let r = ((rgb24 >> 3) & 0x1F) as u16;
    let g = (((rgb24 >> 8) >> 3) & 0x1F) as u16;
    let b = (((rgb24 >> 16) >> 3) & 0x1F) as u16;
    r | (g << 5) | (b << 10)
}

impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

/// GPUSTAT — the status register the CPU polls to check whether the
/// GPU is ready for commands, ready to upload VRAM, etc.
///
/// Value model matches a "always-idle soft GPU" (same convention as
/// PCSX-Redux and PSoXide-2): bits 26–28 are forced ready on every
/// read, bit 25 (DMA request) is computed from the DMA direction
/// bits 29:30, and bit 31 (interlace/field) toggles at VBlank.
struct GpuStatus {
    raw: u32,
}

impl GpuStatus {
    fn new() -> Self {
        // Reset: display disabled (bit 23), DMA direction 0,
        // interlace odd field, ready bits cleared (filled in on read).
        Self { raw: 0x1480_2000 }
    }

    fn read(&self) -> u32 {
        let mut ret = self.raw;

        // Always-ready: bits 26 (cmd FIFO ready), 27 (VRAM→CPU ready),
        // 28 (DMA block ready). Always on for a soft GPU.
        ret |= 0x1C00_0000;

        // Bit 25 (DMA data request) is derived from direction (bits 29:30).
        //   Direction 0 (Off):       bit 25 = 0
        //   Direction 1 (FIFO):      bit 25 = 1
        //   Direction 2 (CPU→GPU):   bit 25 = copy of bit 28
        //   Direction 3 (GPU→CPU):   bit 25 = copy of bit 27
        ret &= !0x0200_0000;
        match (ret >> 29) & 3 {
            1 => ret |= 0x0200_0000,
            2 => ret |= (ret & 0x1000_0000) >> 3,
            3 => ret |= (ret & 0x0800_0000) >> 2,
            _ => {}
        }
        ret
    }

    /// GP1 command dispatch. GP1 commands are simple enough to inline
    /// here; GP0 is more elaborate (packet assembly, texture upload,
    /// primitives) and gets a dedicated module when we actually render.
    fn toggle_field(&mut self) {
        self.raw ^= 0x8000_0000;
    }

    fn gp1_write(&mut self, value: u32) {
        // GP1 writes that affect GpuStatus itself are handled here.
        // Display-area writes (0x05 / 0x06 / 0x07 / 0x08) are routed
        // through `Gpu::apply_gp1_display` in the outer impl so they
        // can update the display-area fields. We still apply the
        // subset that touches GPUSTAT bits (0x08).
        let cmd = (value >> 24) & 0xFF;
        match cmd {
            0x00 => *self = Self::new(),
            0x03 => {
                if value & 1 != 0 {
                    self.raw |= 1 << 23;
                } else {
                    self.raw &= !(1 << 23);
                }
            }
            0x04 => {
                self.raw = (self.raw & !0x6000_0000) | ((value & 3) << 29);
            }
            0x08 => {
                // GP1 0x08: display mode. Update the GPUSTAT bits that
                // mirror it (16:17 for h-res-1, 18 for v-res, 19 for
                // mode, 20 for 24bpp, 21 for v-interlace, 22 for h-res-2).
                let hres1 = value & 0x3;
                let vres = (value >> 2) & 0x1;
                let mode = (value >> 3) & 0x1;
                let bpp = (value >> 4) & 0x1;
                let vinter = (value >> 5) & 0x1;
                let hres2 = (value >> 6) & 0x1;
                let status_bits = (hres1 << 17)
                    | (vres << 19)
                    | (mode << 20)
                    | (bpp << 21)
                    | (vinter << 22)
                    | (hres2 << 16);
                self.raw = (self.raw & !0x007F_0000) | status_bits;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_status_has_always_ready_bits() {
        let mut gpu = Gpu::new();
        let stat = gpu.read32(GP1_ADDR).unwrap();
        // Bits 26, 27, 28 are the "ready" bits we force on every read.
        assert_eq!(stat & 0x1C00_0000, 0x1C00_0000);
    }

    #[test]
    fn gp1_reset_clears_dma_direction() {
        let mut gpu = Gpu::new();
        gpu.write32(GP1_ADDR, 0x0400_0002); // set DMA direction to 2
        let stat = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!((stat >> 29) & 3, 2);

        gpu.write32(GP1_ADDR, 0x0000_0000); // reset
        let stat = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!((stat >> 29) & 3, 0);
    }

    #[test]
    fn gp1_display_disable_toggles_bit_23() {
        let mut gpu = Gpu::new();
        // Start disabled (reset state has bit 23 set).
        let stat_before = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat_before & (1 << 23), 1 << 23);

        // GP1(0x03) with bit 0 = 0: enable display.
        gpu.write32(GP1_ADDR, 0x0300_0000);
        let stat_enabled = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat_enabled & (1 << 23), 0);
    }

    #[test]
    fn gp0_writes_are_accepted_without_effect() {
        let mut gpu = Gpu::new();
        let stat_before = gpu.read32(GP1_ADDR).unwrap();
        gpu.write32(GP0_ADDR, 0xE100_0000);
        let stat_after = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat_before, stat_after);
    }

    #[test]
    fn rgb24_to_bgr15_white_reaches_0x7fff() {
        assert_eq!(rgb24_to_bgr15(0x00FFFFFF), 0x7FFF);
    }

    #[test]
    fn rgb24_to_bgr15_primary_channels() {
        // Pure red (FF in low byte): bottom 5 bits set.
        assert_eq!(rgb24_to_bgr15(0x000000FF), 0x001F);
        // Pure green: middle 5.
        assert_eq!(rgb24_to_bgr15(0x0000FF00), 0x03E0);
        // Pure blue: top 5.
        assert_eq!(rgb24_to_bgr15(0x00FF0000), 0x7C00);
    }

    #[test]
    fn gp0_fill_rect_writes_vram() {
        let mut gpu = Gpu::new();
        // Fill a 16×16 red rect at (0, 0).
        gpu.write32(GP0_ADDR, 0x0200_00FF); // 0x02 + red
        gpu.write32(GP0_ADDR, 0x0000_0000); // y=0, x=0
        gpu.write32(GP0_ADDR, 0x0010_0010); // h=16, w=16

        assert_eq!(gpu.vram.get_pixel(0, 0), 0x001F);
        assert_eq!(gpu.vram.get_pixel(15, 15), 0x001F);
        // Just outside stays zero.
        assert_eq!(gpu.vram.get_pixel(16, 0), 0);
        assert_eq!(gpu.vram.get_pixel(0, 16), 0);
    }

    #[test]
    fn gp0_draw_mode_commands_are_accepted() {
        let mut gpu = Gpu::new();
        // Four draw-mode setters back-to-back, each 1 word.
        gpu.write32(GP0_ADDR, 0xE100_0000); // draw mode
        gpu.write32(GP0_ADDR, 0xE200_0000); // texture window
        gpu.write32(GP0_ADDR, 0xE300_0000); // drawing area TL
        gpu.write32(GP0_ADDR, 0xE400_0000); // drawing area BR
                                            // None of these should have stuck a packet in the FIFO.
                                            // (Implementation detail, but worth guarding against.)
    }

    #[test]
    fn gpu_address_match_returns_none_off_port() {
        let mut gpu = Gpu::new();
        assert!(gpu.read32(0x1F80_1800).is_none());
        assert!(gpu.read32(0x1F80_1818).is_none());
    }
}
