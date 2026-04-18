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
    /// Semi-transparency mode from the current texpage (bits 5-6 of
    /// GP0 0xE1 or of a textured-primitive's tpage override). Kept
    /// as a [`BlendMode`] so primitives can plug it straight into
    /// the rasterizer without re-parsing the bits. Only matters when
    /// a primitive's cmd-bit-1 selects semi-transparent; opaque
    /// prims ignore this field.
    tex_blend_mode: BlendMode,
    /// Texture-window mask X in pixels (multiple of 8). Set by
    /// GP0 0xE2 bits 0..=4, left-shifted by 3. Used at UV-lookup
    /// time to force the high bits of the effective U to a constant
    /// pattern — typically 0 (no mask) so UV passes through, but
    /// games that tile a texture via this feature set specific bits.
    tex_window_mask_x: u8,
    /// Texture-window mask Y, same shape.
    tex_window_mask_y: u8,
    /// Texture-window offset X (OR'd into the masked-out U bits).
    tex_window_offset_x: u8,
    /// Texture-window offset Y.
    tex_window_offset_y: u8,
    /// When true, every plotted pixel gets its mask bit (VRAM bit 15)
    /// forced to 1. Set by GP0 0xE6 bit 0. Mirrors GPUSTAT bit 11.
    /// Used by games that want to protect pixels from being
    /// overwritten by later primitives (combined with
    /// `mask_check_before_draw`).
    mask_set_on_draw: bool,
    /// When true, the rasterizer skips plotting over existing pixels
    /// whose mask bit (bit 15) is already set. Set by GP0 0xE6 bit
    /// 1. Mirrors GPUSTAT bit 12. Games commonly pair this with
    /// `mask_set_on_draw`: first pass sets the mask on important
    /// sprites (HUD, transparent edges), later prims can't stomp
    /// them. Without this check, HUDs flicker under overlapping
    /// backgrounds.
    mask_check_before_draw: bool,
    /// When true, apply the PSX 4×4 Bayer dither matrix to every
    /// 24-bit → 15-bit channel reduction. Set by GP0 0xE1 bit 9.
    /// Active on Gouraud-shaded primitives + textured primitives
    /// with tint modulation; flat colour + raw-texture prims are
    /// unaffected (their source is already 15-bit).
    ///
    /// Games rarely enable this — it's a conservative choice that
    /// trades crispness for banding reduction. When it's on, the
    /// 24-bit shaded intermediate gets a per-pixel offset in the
    /// range -4..=+3 added before the `>> 3` truncation, producing
    /// ordered dither patterns instead of visible 5-bit staircases.
    dither_enabled: bool,

    // --- Display area (GP1 0x05 / 0x06 / 0x07 / 0x08) ---
    /// VRAM X of the top-left pixel of the displayed framebuffer.
    display_start_x: u16,
    /// VRAM Y of the top-left pixel of the displayed framebuffer.
    display_start_y: u16,
    /// Horizontal display resolution from GP1 0x08 (pixels). One of
    /// 256, 320, 368, 512, 640.
    display_width: u16,
    /// Vertical resolution flag from GP1(08h) bit 2 — `true` means
    /// 480-line interlaced (each V-range line doubles). The actual
    /// displayed row count is computed from the V-range (Y1..Y2)
    /// and this flag in [`Gpu::effective_display_height`].
    display_height_480: bool,
    /// V-range Y1 from GP1(07h) bits 0..=9 — top scanline of the
    /// visible window in the video output. Default ~16.
    v_range_y1: u16,
    /// V-range Y2 from GP1(07h) bits 10..=19 — bottom scanline of
    /// the visible window. Default ~256, giving 240 visible rows.
    v_range_y2: u16,
    /// H-range X1 from GP1(06h) bits 0..=11 — left-edge GPU clock
    /// of the visible window. Stored for display-area reporting but
    /// not (yet) used to derive the width.
    h_range_x1: u16,
    /// H-range X2 from GP1(06h) bits 12..=23 — right-edge GPU clock.
    h_range_x2: u16,
    /// 24bpp colour depth flag from GP1 0x08 bit 4. For now we always
    /// decode VRAM as 15bpp; when this flag comes into play the
    /// frontend's framebuffer view can respect it.
    display_24bpp: bool,
    /// `true` after the BIOS / game has written GP1 0x07 (V-range) or
    /// GP1 0x08 (display mode). Before that, `display_area` reports
    /// (0, 0) — matching Redux's `takeScreenShot`, which also hands
    /// back a zero-sized image until its internal
    /// `updateDisplayIfChanged` runs (triggered by those same two
    /// GP1 writes). Parity tools rely on this to avoid seeing a
    /// spurious "dimension mismatch" before the first configured
    /// frame even exists.
    display_configured: bool,

    /// Count of executed GP0 packets by opcode byte (the high 8 bits
    /// of the header word). Diagnostic only — lets `smoke_draw` see at
    /// a glance which primitive types the BIOS is issuing.
    gp0_opcode_hist: [u32; 256],
    /// Count of GP1 writes by opcode byte. Same diagnostic role as
    /// gp0_opcode_hist but for the display / control port.
    gp1_opcode_hist: [u32; 256],
    /// Distinct (x, y) pairs written to GP1 0x05 (display-start). Lets
    /// diagnostics see whether the BIOS is flipping buffers or just
    /// repeatedly re-writing the same location.
    display_start_history: std::collections::BTreeSet<(u16, u16)>,
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
            tex_blend_mode: BlendMode::Average,
            tex_window_mask_x: 0,
            tex_window_mask_y: 0,
            tex_window_offset_x: 0,
            tex_window_offset_y: 0,
            mask_set_on_draw: false,
            mask_check_before_draw: false,
            dither_enabled: false,
            display_start_x: 0,
            display_start_y: 0,
            display_width: 320,
            display_height_480: false,
            // Power-on V- and H-range defaults, matching Redux's
            // `SoftGPU::impl::initBackend` which zeroes `Range.x0 =
            // Range.x1 = Range.y0 = Range.y1 = 0`. Crucially the
            // BIOS writes GP1 0x08 (display mode) *before* GP1 0x07
            // (v-range), and because Redux derives Height from
            // `y1 - y0` — both zero — its `takeScreenShot` height
            // is 0 during that window. Earlier we defaulted these
            // to 0x10/0x100, which made our screenshot height 240
            // during the same window and broke lockstep parity at
            // step 19.3 M on Crash's BIOS handoff.
            v_range_y1: 0,
            v_range_y2: 0,
            h_range_x1: 0,
            h_range_x2: 0,
            display_24bpp: false,
            display_configured: false,
            vram_download: None,
            gpuread_latch: 0,
            gp0_opcode_hist: [0; 256],
            gp1_opcode_hist: [0; 256],
            display_start_history: std::collections::BTreeSet::new(),
        }
    }

    /// Distinct display-start corners the BIOS has written to. Useful
    /// for telling a re-write loop from a front/back-buffer flip.
    pub fn display_start_history(&self) -> impl Iterator<Item = (u16, u16)> + '_ {
        self.display_start_history.iter().copied()
    }

    /// Snapshot of the GP0 opcode histogram — per-byte count of
    /// executed packets keyed by high-byte of word 0. Diagnostic.
    pub fn gp0_opcode_histogram(&self) -> [u32; 256] {
        self.gp0_opcode_hist
    }

    /// Snapshot of the GP1 opcode histogram. Diagnostic.
    pub fn gp1_opcode_histogram(&self) -> [u32; 256] {
        self.gp1_opcode_hist
    }

    /// Snapshot of the currently-configured display area, for the
    /// frontend's framebuffer panel. Cheap to call each frame. The
    /// `height` is derived from the V-range + 480-mode flag (see
    /// [`Gpu::effective_display_height`]) so it matches what Redux's
    /// screenshot path reports — letting milestone parity tests
    /// compare byte-for-byte.
    pub fn display_area(&self) -> DisplayArea {
        if !self.display_configured {
            // Match Redux's `takeScreenShot`: zero-sized image until
            // GP1 0x07 or 0x08 has been written. That's what lets
            // `display_hash` compare apples to apples from the very
            // first instruction onward, instead of us reporting our
            // `GP1 0x00` reset defaults while Redux still reports 0×0.
            return DisplayArea {
                x: self.display_start_x,
                y: self.display_start_y,
                width: 0,
                height: 0,
                bpp24: self.display_24bpp,
            };
        }
        DisplayArea {
            x: self.display_start_x,
            y: self.display_start_y,
            width: self.display_width,
            height: self.effective_display_height(),
            bpp24: self.display_24bpp,
        }
    }

    /// FNV-1a-64 over the visible display area's 15bpp pixel bytes,
    /// for Redux-parity comparisons. Rows are packed tightly (no
    /// stride padding) so a given (width, height, bpp) maps to a
    /// specific byte sequence — identical to what Redux's
    /// `PCSX.GPU.takeScreenShot()` produces server-side on the
    /// oracle path.
    ///
    /// Returns `(hash, width, height, byte_len)`. If the display
    /// area extends past VRAM the rows are clipped at the VRAM
    /// edge and the row count is reduced — matching Redux's
    /// behaviour.
    pub fn display_hash(&self) -> (u64, u32, u32, usize) {
        let da = self.display_area();
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        let mut byte_len = 0usize;
        let vram_w = crate::VRAM_WIDTH as u16;
        let vram_h = crate::VRAM_HEIGHT as u16;
        let effective_h = da.height.min(vram_h.saturating_sub(da.y));
        let effective_w = da.width.min(vram_w.saturating_sub(da.x));
        let mut fold = |b: u8, byte_len: &mut usize| {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01B3);
            *byte_len += 1;
        };
        if da.bpp24 {
            // 24-bit mode: each pixel is 3 bytes packed in VRAM. A row
            // of W 24-bit pixels occupies W*3 bytes = 1.5 * W 16-bit
            // words. We read per-byte to span the straddles.
            for dy in 0..effective_h {
                for dx in 0..effective_w {
                    let (r, g, b) = self.read_pixel_rgb24(da.x + dx, da.y + dy);
                    fold(r, &mut byte_len);
                    fold(g, &mut byte_len);
                    fold(b, &mut byte_len);
                }
            }
        } else {
            for dy in 0..effective_h {
                for dx in 0..effective_w {
                    let pixel = self.vram.get_pixel(da.x + dx, da.y + dy);
                    for b in pixel.to_le_bytes() {
                        fold(b, &mut byte_len);
                    }
                }
            }
        }
        (h, effective_w as u32, effective_h as u32, byte_len)
    }

    /// Read one 24-bit display pixel. VRAM bytes are packed: pixel
    /// N lives at byte offsets `3*N..3*N+2` within a row, and each
    /// row is 2048 bytes (1024 × 16-bit). The three bytes may
    /// straddle two VRAM halfwords — we read them individually.
    fn read_pixel_rgb24(&self, x: u16, y: u16) -> (u8, u8, u8) {
        let byte_x = (x as u32) * 3;
        let word_x = (byte_x / 2) as u16;
        let even = byte_x & 1 == 0;
        let w0 = self.vram.get_pixel(word_x, y);
        let w1 = self.vram.get_pixel(word_x.wrapping_add(1), y);
        if even {
            let r = (w0 & 0xFF) as u8;
            let g = (w0 >> 8) as u8;
            let b = (w1 & 0xFF) as u8;
            (r, g, b)
        } else {
            let r = (w0 >> 8) as u8;
            let g = (w1 & 0xFF) as u8;
            let b = (w1 >> 8) as u8;
            (r, g, b)
        }
    }

    /// Produce a row-major `RGBA8` buffer of the current display area.
    /// In 16-bit mode the 5-bit channels are bit-replicated to 8-bit;
    /// in 24-bit mode the packed RGB888 triplets are used directly.
    /// Alpha is always 0xFF. Size = `width * height * 4` bytes.
    ///
    /// Used by the frontend to upload a display texture — a single
    /// format regardless of the PS1's current bpp, so the wgpu
    /// path doesn't need to branch.
    pub fn display_rgba8(&self) -> (Vec<u8>, u32, u32) {
        let da = self.display_area();
        let vram_w = crate::VRAM_WIDTH as u16;
        let vram_h = crate::VRAM_HEIGHT as u16;
        let eff_h = da.height.min(vram_h.saturating_sub(da.y));
        let eff_w = da.width.min(vram_w.saturating_sub(da.x));
        let mut out = Vec::with_capacity((eff_w as usize) * (eff_h as usize) * 4);
        for dy in 0..eff_h {
            for dx in 0..eff_w {
                if da.bpp24 {
                    let (r, g, b) = self.read_pixel_rgb24(da.x + dx, da.y + dy);
                    out.extend_from_slice(&[r, g, b, 0xFF]);
                } else {
                    let pixel = self.vram.get_pixel(da.x + dx, da.y + dy);
                    let r = ((pixel & 0x1F) as u8) << 3;
                    let g = (((pixel >> 5) & 0x1F) as u8) << 3;
                    let b = (((pixel >> 10) & 0x1F) as u8) << 3;
                    // Replicate high 3 bits into low 3 for fuller range.
                    out.extend_from_slice(&[
                        r | (r >> 5),
                        g | (g >> 5),
                        b | (b >> 5),
                        0xFF,
                    ]);
                }
            }
        }
        (out, eff_w as u32, eff_h as u32)
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
            GP1_ADDR => Some(self.status.read(self.vram_download.is_some())),
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
                let op = ((value >> 24) & 0xFF) as usize;
                self.gp1_opcode_hist[op] = self.gp1_opcode_hist[op].saturating_add(1);
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
            // GP1 0x00 — GPU reset. Matches Redux's `CtrlReset`:
            // clears the display-enable flag + RGB24/interlace bits
            // and resets DrawOffset, but **does not** touch the
            // V/H-ranges or DisplayPosition. The BIOS writes those
            // via the explicit GP1 0x05 / 0x06 / 0x07 commands
            // later, so reset-persisting them matches hardware.
            0x00 => {
                self.display_start_x = 0;
                self.display_start_y = 0;
                self.display_width = 320;
                self.display_height_480 = false;
                self.display_24bpp = false;
                self.display_configured = false;
                // Mask flags reset per PSX-SPX (GP1 0x00 clears both).
                self.mask_set_on_draw = false;
                self.mask_check_before_draw = false;
                self.status.raw &= !0x1800;
            }
            // GP1 0x05 — display area start (top-left corner in VRAM).
            //   bits 9:0  = X (pixels)
            //   bits 18:10 = Y (pixels)
            0x05 => {
                self.display_start_x = (value & 0x3FF) as u16;
                self.display_start_y = ((value >> 10) & 0x1FF) as u16;
                self.display_start_history
                    .insert((self.display_start_x, self.display_start_y));
            }
            // GP1 0x06 — Horizontal display range (on screen, in GPU
            // clocks — not pixels). Used for centering the active
            // display inside the video signal; doesn't change the
            // VRAM read window's width. Stored for completeness.
            0x06 => {
                self.h_range_x1 = (value & 0xFFF) as u16;
                self.h_range_x2 = ((value >> 12) & 0xFFF) as u16;
            }
            // GP1 0x07 — Vertical display range. Bits 0..=9 = top
            // scanline, bits 10..=19 = bottom scanline. Effective
            // rendered rows = (y2 - y1), doubled in 480-interlaced
            // mode. Redux's `takeScreenShot` dimensions come from
            // this, not from the GP1(08h) mode bit — matching it is
            // what gets us 640×478 instead of 640×480 at boot.
            0x07 => {
                self.v_range_y1 = (value & 0x3FF) as u16;
                self.v_range_y2 = ((value >> 10) & 0x3FF) as u16;
                self.display_configured = true;
            }
            // GP1 0x08 — display mode. Height is the interlace flag;
            // actual pixel count is derived together with V-range in
            // [`Gpu::effective_display_height`].
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
                self.display_height_480 = value & (1 << 2) != 0;
                self.display_24bpp = value & (1 << 4) != 0;
                self.display_configured = true;
            }
            _ => {}
        }
    }

    /// Current dot-clock divisor: system clocks per pixel-clock tick.
    /// Indexed by the current display resolution. Values match Redux's
    /// `HDotClock` array in `src/core/psxcounters.cc`.
    ///
    /// Used by Timer 0 when its source is set to "dot clock" (mode
    /// bits 8..9 = 1 or 3). Games that sync to horizontal raster
    /// (rare — dot-clock timing is usually a bit granular) key off
    /// this.
    pub fn dot_clock_divisor(&self) -> u64 {
        match self.display_width {
            256 => 10,
            320 => 8,
            368 | 384 => 7,
            512 => 5,
            640 => 4,
            _ => 10, // Safe fallback.
        }
    }

    /// Effective vertical pixel count shown on the video output —
    /// derived from V-range (`GP1(07h)`) and the 480-mode flag
    /// (`GP1(08h)` bit 2). Matches Redux's
    /// `PCSX.GPU.takeScreenShot()` height, so using this value for
    /// pixel-parity regression tests lines up byte-for-byte.
    ///
    /// Formula:
    /// ```text
    ///   rows_per_field = max(y2 - y1, 0)
    ///   visible        = rows_per_field * (480-mode ? 2 : 1)
    /// ```
    pub fn effective_display_height(&self) -> u16 {
        let rows = self.v_range_y2.saturating_sub(self.v_range_y1);
        if self.display_height_480 {
            rows.saturating_mul(2)
        } else {
            rows
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
            // + dither/display/transparency flags. We extract the
            // subset the texture rasterizer needs AND mirror bits
            // 0..=10 into `GpuStatus::raw`, since those are
            // observable via GPUSTAT reads. Redux's softgpu does
            // the equivalent in `gpuWriteStatus` / `sCommand0xE1`,
            // and the BIOS polls GPUSTAT right after each E1h to
            // verify the command took effect. Leaving the status
            // bits stale produces a GPUSTAT divergence that doesn't
            // surface until the poll.
            //
            // E1h layout:
            //   bits 0-3: texture page base X (each unit = 64 pix)
            //   bit  4:   texture page base Y (0=0, 1=256)
            //   bits 5-6: semi-transparency
            //   bits 7-8: texture page colour depth
            //   bit  9:   dither 24→15
            //   bit  10:  drawing to display area
            //   bit  11:  texture disable (requires GP1 09h unlock)
            //   bit  12:  textured rectangle X flip
            //   bit  13:  textured rectangle Y flip
            // These map 1:1 to GPUSTAT bits 0..=10 (plus rect-flip
            // bits that aren't visible in GPUSTAT).
            0xE1 => {
                self.tex_page_x = ((word & 0x0F) as u16) * 64;
                self.tex_page_y = if (word >> 4) & 1 != 0 { 256 } else { 0 };
                self.tex_depth = ((word >> 7) & 0x3) as u8;
                self.tex_blend_mode = BlendMode::from_tpage_bits(word >> 5);
                self.dither_enabled = (word >> 9) & 1 != 0;
                // GPUSTAT bits 0..=10 come from E1h bits 0..=10.
                let stat_bits = word & 0x07FF;
                self.status.raw = (self.status.raw & !0x07FF) | stat_bits;
            }
            // GP0 0xE6 — mask-bit setting.
            //   bit 0 = `mask_set_on_draw`: force bit 15 of every
            //           plotted pixel to 1 (protect it against
            //           later draws that check the mask).
            //   bit 1 = `mask_check_before_draw`: skip pixels whose
            //           existing VRAM bit 15 is already 1.
            // Both also surface in GPUSTAT at bits 11 / 12 so
            // software polls see the updated setting.
            0xE6 => {
                self.mask_set_on_draw = word & 1 != 0;
                self.mask_check_before_draw = word & 2 != 0;
                let stat_bits = (word & 0x3) << 11;
                self.status.raw = (self.status.raw & !0x1800) | stat_bits;
            }
            // GP0 0xE2 — texture window. Lets a textured primitive
            // AND-mask its U/V into a smaller patch of the tpage,
            // effectively tiling a sub-rectangle of texture across
            // the prim. Format:
            //   bits 0-4  : mask X (U high bits forced; in 8-pixel steps)
            //   bits 5-9  : mask Y
            //   bits 10-14: offset X (U low bits OR'd)
            //   bits 15-19: offset Y
            // Per PSX-SPX, the effective texture coordinate is
            //     U' = (U & ~(mask_x << 3)) | ((offset_x & mask_x) << 3)
            //     V' = (V & ~(mask_y << 3)) | ((offset_y & mask_y) << 3)
            // (mask is in 8-pixel units; left-shift by 3 gives the
            // pixel-space mask.) Games that use palettes laid out in
            // small sub-rectangles rely on this to save VRAM — the
            // same 128×128 tile gets reused for many prims with
            // different offsets.
            0xE2 => {
                self.tex_window_mask_x = ((word & 0x1F) as u8) << 3;
                self.tex_window_mask_y = (((word >> 5) & 0x1F) as u8) << 3;
                self.tex_window_offset_x = (((word >> 10) & 0x1F) as u8) << 3;
                self.tex_window_offset_y = (((word >> 15) & 0x1F) as u8) << 3;
            }
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
            // Textured + Gouraud shaded — both per-vertex tint colours
            // AND per-vertex UV. The tint modulates every sampled
            // texel (per PSX-SPX tint formula). Triangle = 9 words,
            // quad = 12 words.
            0x34..=0x37 => self.draw_textured_shaded_tri(),
            0x3C..=0x3F => self.draw_textured_shaded_quad(),
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
        let cmd = self.gp0_fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let v1 = self.decode_vertex(self.gp0_fifo[2]);
        let v2 = self.decode_vertex(self.gp0_fifo[3]);
        self.rasterize_triangle(v0, v1, v2, color, mode);
    }

    /// GP0 0x28..=0x2B — monochrome 4-vertex quad, split into two
    /// triangles `(v0, v1, v2)` + `(v1, v2, v3)`.
    fn draw_monochrome_quad(&mut self) {
        let cmd = self.gp0_fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let v1 = self.decode_vertex(self.gp0_fifo[2]);
        let v2 = self.decode_vertex(self.gp0_fifo[3]);
        let v3 = self.decode_vertex(self.gp0_fifo[4]);
        self.rasterize_triangle(v0, v1, v2, color, mode);
        self.rasterize_triangle(v1, v2, v3, color, mode);
    }

    /// GP0 0x30..=0x33 — Gouraud triangle. Per-vertex RGB24 colours,
    /// interpolated across the triangle via barycentric weights.
    /// Words: `[cmd+c0, v0, c1, v1, c2, v2]`.
    fn draw_shaded_tri(&mut self) {
        let cmd = self.gp0_fifo[0];
        let c0 = cmd & 0x00FF_FFFF;
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let c1 = self.gp0_fifo[2] & 0x00FF_FFFF;
        let v1 = self.decode_vertex(self.gp0_fifo[3]);
        let c2 = self.gp0_fifo[4] & 0x00FF_FFFF;
        let v2 = self.decode_vertex(self.gp0_fifo[5]);
        self.rasterize_shaded_triangle(v0, v1, v2, c0, c1, c2, mode);
    }

    /// GP0 0x38..=0x3B — Gouraud quad. 4 × (colour+vertex) =
    /// 8 words, split into two shaded triangles sharing the middle
    /// edge (v1–v2).
    fn draw_shaded_quad(&mut self) {
        let cmd = self.gp0_fifo[0];
        let c0 = cmd & 0x00FF_FFFF;
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let c1 = self.gp0_fifo[2] & 0x00FF_FFFF;
        let v1 = self.decode_vertex(self.gp0_fifo[3]);
        let c2 = self.gp0_fifo[4] & 0x00FF_FFFF;
        let v2 = self.decode_vertex(self.gp0_fifo[5]);
        let c3 = self.gp0_fifo[6] & 0x00FF_FFFF;
        let v3 = self.decode_vertex(self.gp0_fifo[7]);
        self.rasterize_shaded_triangle(v0, v1, v2, c0, c1, c2, mode);
        self.rasterize_shaded_triangle(v1, v2, v3, c1, c2, c3, mode);
    }

    /// GP0 0x60..=0x63 — monochrome variable-size rectangle.
    /// Words: `[cmd+color, xy, wh]`.
    fn draw_monochrome_rect_variable(&mut self) {
        let cmd = self.gp0_fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let pos = self.gp0_fifo[1];
        let size = self.gp0_fifo[2];
        let x = sign_extend_11((pos & 0x7FF) as i32) + self.draw_offset_x;
        let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32) + self.draw_offset_y;
        let w = (size & 0xFFFF) as i32;
        let h = ((size >> 16) & 0xFFFF) as i32;
        self.paint_rect(x, y, w, h, color, mode);
    }

    /// GP0 0x68/0x70/0x78 — fixed-size monochrome rectangles.
    fn draw_monochrome_rect_sized(&mut self, w: i32, h: i32) {
        let cmd = self.gp0_fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let pos = self.gp0_fifo[1];
        let x = sign_extend_11((pos & 0x7FF) as i32) + self.draw_offset_x;
        let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32) + self.draw_offset_y;
        self.paint_rect(x, y, w, h, color, mode);
    }

    /// Variable-size textured rect. Words: `[cmd+tint, xy, clut+uv, wh]`.
    fn draw_textured_rect_variable(&mut self) {
        let cmd = self.gp0_fifo[0];
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
        let tint = if cmd & 1 != 0 {
            RAW_TEXTURE_TINT
        } else {
            split_tint(cmd & 0x00FF_FFFF)
        };
        self.paint_textured_rect(
            x, y, w, h, u0, v0, clut_word,
            prim_is_semi_trans(cmd),
            tint,
        );
    }

    /// Fixed-size textured rect (1×1, 8×8, 16×16).
    /// Words: `[cmd+tint, xy, clut+uv]`.
    fn draw_textured_rect_sized(&mut self, w: i32, h: i32) {
        let cmd = self.gp0_fifo[0];
        let pos = self.gp0_fifo[1];
        let uv_clut = self.gp0_fifo[2];
        let x = sign_extend_11((pos & 0x7FF) as i32) + self.draw_offset_x;
        let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32) + self.draw_offset_y;
        let u0 = (uv_clut & 0xFF) as u16;
        let v0 = ((uv_clut >> 8) & 0xFF) as u16;
        let clut_word = ((uv_clut >> 16) & 0xFFFF) as u16;
        let tint = if cmd & 1 != 0 {
            RAW_TEXTURE_TINT
        } else {
            split_tint(cmd & 0x00FF_FFFF)
        };
        self.paint_textured_rect(
            x, y, w, h, u0, v0, clut_word,
            prim_is_semi_trans(cmd),
            tint,
        );
    }

    /// Plot a textured rectangle. Each destination pixel samples a
    /// 1:1 texel from the current texture page, CLUT-indexed for
    /// 4bpp / 8bpp modes, direct for 15bpp. Texels of value 0 are
    /// transparent (standard PS1 convention).
    ///
    /// `semi_trans` — cmd-bit-1. Texels with bit 15 high blend via
    /// `self.tex_blend_mode` when it's set; texels with bit 15 clear
    /// always draw opaque.
    ///
    /// `tint` — 24-bit vertex colour that modulates each texel (see
    /// [`modulate_tint`]). Raw-texture rectangles pass
    /// `(0x80, 0x80, 0x80)` so modulation is a no-op.
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
        semi_trans: bool,
        tint: (u32, u32, u32),
    ) {
        if w <= 0 || h <= 0 {
            return;
        }
        let clut_x = (clut_word & 0x3F) * 16;
        let clut_y = (clut_word >> 6) & 0x1FF;
        let tpage_mode = self.tex_blend_mode;

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
                    let shaded = modulate_tint(texel, tint.0, tint.1, tint.2);
                    let mode = if semi_trans && (texel & 0x8000) != 0 {
                        tpage_mode
                    } else {
                        BlendMode::Opaque
                    };
                    self.plot_pixel(px as u16, py as u16, shaded, mode);
                }
            }
        }
    }

    /// Fetch a single texel from the active texture page. Returns
    /// `None` for transparent (texel value 0 in CLUT modes, or 0x0000
    /// with mask bit clear in direct mode).
    ///
    /// The incoming `u` / `v` are run through the GP0 0xE2 texture
    /// window first: `U' = (U & ~mask) | (offset & mask)` per axis.
    /// With the default (all zeroes) that's a no-op; games that use
    /// tiling set non-zero mask/offset to reuse a sub-rectangle of
    /// the tpage across multiple primitives.
    fn sample_texture(&self, u: u16, v: u16, clut_x: u16, clut_y: u16) -> Option<u16> {
        // Apply the texture window — PSX-SPX:
        //   U' = (U AND NOT(mask_x * 8)) OR ((offset_x * 8) AND (mask_x * 8))
        // but both `mask_*` and `offset_*` are already pre-shifted (×8)
        // when we stored them in the GP0 0xE2 handler.
        let mask_x = self.tex_window_mask_x as u16;
        let mask_y = self.tex_window_mask_y as u16;
        let off_x = self.tex_window_offset_x as u16;
        let off_y = self.tex_window_offset_y as u16;
        let u = (u & !mask_x) | (off_x & mask_x);
        let v = (v & !mask_y) | (off_y & mask_y);

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
    /// GPU's drawing area. `mode` lets the caller pass the
    /// primitive's semi-transparency mode — opaque for the common
    /// case, one of the blend variants when the GP0 command's
    /// cmd-bit-1 is set.
    fn paint_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u16, mode: BlendMode) {
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
                self.plot_pixel(px as u16, py as u16, color, mode);
            }
        }
    }

    /// Scanline-ish triangle rasterizer using the edge-function test.
    /// For each pixel in the bounding box we evaluate three edge
    /// equations; a pixel is inside iff all three have the same sign.
    /// Works regardless of triangle winding. Clipped to both VRAM
    /// bounds and the active drawing area.
    fn rasterize_triangle(
        &mut self,
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        color: u16,
        mode: BlendMode,
    ) {
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
                    self.plot_pixel(x as u16, y as u16, color, mode);
                }
            }
        }
    }

    /// GP0 0x24..=0x27 — textured triangle. 7 words:
    /// `[cmd+tint, v0, clut+uv0, v1, tpage+uv1, v2, uv2]`.
    ///
    /// Command bit 0 chooses raw-texture (tint ignored) vs
    /// texture-blended (vertex tint modulates each texel).
    fn draw_textured_tri(&mut self) {
        let cmd = self.gp0_fifo[0];
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
        let tint = if cmd & 1 != 0 {
            RAW_TEXTURE_TINT
        } else {
            split_tint(cmd & 0x00FF_FFFF)
        };
        self.rasterize_textured_triangle(
            v0,
            v1,
            v2,
            t0,
            t1,
            t2,
            clut_word,
            prim_is_semi_trans(cmd),
            tint,
        );
    }

    /// GP0 0x2C..=0x2F — textured quad. 9 words; split into two
    /// textured triangles sharing v1–v2.
    fn draw_textured_quad(&mut self) {
        let cmd = self.gp0_fifo[0];
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
        let semi = prim_is_semi_trans(cmd);
        let tint = if cmd & 1 != 0 {
            RAW_TEXTURE_TINT
        } else {
            split_tint(cmd & 0x00FF_FFFF)
        };
        self.rasterize_textured_triangle(v0, v1, v2, t0, t1, t2, clut_word, semi, tint);
        self.rasterize_textured_triangle(v1, v2, v3, t1, t2, t3, clut_word, semi, tint);
    }

    /// GP0 0x34..=0x37 — textured + Gouraud-shaded triangle.
    /// Words: `[cmd+c0, v0, uv0+clut, c1, v1, uv1+texpage, c2, v2, uv2]`.
    /// Per-vertex tint interpolation is barycentric just like flat
    /// Gouraud, but the tint modulates the sampled texel instead of
    /// being the final pixel colour. Raw-texture mode (bit 0 set)
    /// zeros the tint effect per PSX-SPX.
    fn draw_textured_shaded_tri(&mut self) {
        let cmd = self.gp0_fifo[0];
        let c0 = cmd & 0x00FF_FFFF;
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let uv0 = self.gp0_fifo[2];
        let c1 = self.gp0_fifo[3] & 0x00FF_FFFF;
        let v1 = self.decode_vertex(self.gp0_fifo[4]);
        let uv1 = self.gp0_fifo[5];
        let c2 = self.gp0_fifo[6] & 0x00FF_FFFF;
        let v2 = self.decode_vertex(self.gp0_fifo[7]);
        let uv2 = self.gp0_fifo[8];
        let clut_word = ((uv0 >> 16) & 0xFFFF) as u16;
        self.apply_primitive_tpage(uv1);
        let t0 = ((uv0 & 0xFF) as u16, ((uv0 >> 8) & 0xFF) as u16);
        let t1 = ((uv1 & 0xFF) as u16, ((uv1 >> 8) & 0xFF) as u16);
        let t2 = ((uv2 & 0xFF) as u16, ((uv2 >> 8) & 0xFF) as u16);
        let raw = cmd & 1 != 0;
        self.rasterize_textured_shaded_triangle(
            v0,
            v1,
            v2,
            t0,
            t1,
            t2,
            c0,
            c1,
            c2,
            clut_word,
            prim_is_semi_trans(cmd),
            raw,
        );
    }

    /// GP0 0x3C..=0x3F — textured + Gouraud-shaded quad. 12 words;
    /// split into two textured-shaded triangles sharing edge v1–v2.
    /// Words: `[cmd+c0, v0, uv0+clut, c1, v1, uv1+texpage, c2, v2, uv2,
    ///          c3, v3, uv3]`.
    fn draw_textured_shaded_quad(&mut self) {
        let cmd = self.gp0_fifo[0];
        let c0 = cmd & 0x00FF_FFFF;
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let uv0 = self.gp0_fifo[2];
        let c1 = self.gp0_fifo[3] & 0x00FF_FFFF;
        let v1 = self.decode_vertex(self.gp0_fifo[4]);
        let uv1 = self.gp0_fifo[5];
        let c2 = self.gp0_fifo[6] & 0x00FF_FFFF;
        let v2 = self.decode_vertex(self.gp0_fifo[7]);
        let uv2 = self.gp0_fifo[8];
        let c3 = self.gp0_fifo[9] & 0x00FF_FFFF;
        let v3 = self.decode_vertex(self.gp0_fifo[10]);
        let uv3 = self.gp0_fifo[11];
        let clut_word = ((uv0 >> 16) & 0xFFFF) as u16;
        self.apply_primitive_tpage(uv1);
        let t0 = ((uv0 & 0xFF) as u16, ((uv0 >> 8) & 0xFF) as u16);
        let t1 = ((uv1 & 0xFF) as u16, ((uv1 >> 8) & 0xFF) as u16);
        let t2 = ((uv2 & 0xFF) as u16, ((uv2 >> 8) & 0xFF) as u16);
        let t3 = ((uv3 & 0xFF) as u16, ((uv3 >> 8) & 0xFF) as u16);
        let semi = prim_is_semi_trans(cmd);
        let raw = cmd & 1 != 0;
        self.rasterize_textured_shaded_triangle(
            v0, v1, v2, t0, t1, t2, c0, c1, c2, clut_word, semi, raw,
        );
        self.rasterize_textured_shaded_triangle(
            v1, v2, v3, t1, t2, t3, c1, c2, c3, clut_word, semi, raw,
        );
    }

    /// Rasterize a triangle with per-vertex tint colours AND per-vertex
    /// UVs. Combines the math from `rasterize_shaded_triangle` (three
    /// barycentric-weighted colours) and `rasterize_textured_triangle`
    /// (three barycentric-weighted UVs). The interpolated tint
    /// modulates the sampled texel via [`modulate_tint`]; raw-texture
    /// mode passes `0x80, 0x80, 0x80` which is the identity.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_textured_shaded_triangle(
        &mut self,
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        t0: (u16, u16),
        t1: (u16, u16),
        t2: (u16, u16),
        c0: u32,
        c1: u32,
        c2: u32,
        clut_word: u16,
        semi_trans: bool,
        raw_texture: bool,
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
        let tpage_mode = self.tex_blend_mode;

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
                    let u = (w0 * t0.0 as i64 + w1 * t1.0 as i64 + w2 * t2.0 as i64) / area_abs;
                    let v = (w0 * t0.1 as i64 + w1 * t1.1 as i64 + w2 * t2.1 as i64) / area_abs;
                    if let Some(texel) = self.sample_texture(u as u16, v as u16, clut_x, clut_y) {
                        let (tint_r, tint_g, tint_b) = if raw_texture {
                            RAW_TEXTURE_TINT
                        } else {
                            // Interpolated per-pixel tint.
                            let ri = (w0 * r(c0) + w1 * r(c1) + w2 * r(c2)) / area_abs;
                            let gi = (w0 * g(c0) + w1 * g(c1) + w2 * g(c2)) / area_abs;
                            let bi = (w0 * b(c0) + w1 * b(c1) + w2 * b(c2)) / area_abs;
                            (
                                ri.clamp(0, 255) as u32,
                                gi.clamp(0, 255) as u32,
                                bi.clamp(0, 255) as u32,
                            )
                        };
                        let shaded = modulate_tint(texel, tint_r, tint_g, tint_b);
                        let mode = if semi_trans && (texel & 0x8000) != 0 {
                            tpage_mode
                        } else {
                            BlendMode::Opaque
                        };
                        self.plot_pixel(x as u16, y as u16, shaded, mode);
                    }
                }
            }
        }
    }

    /// Apply the tpage bits embedded in a textured-primitive UV word
    /// (they override the draw-mode tpage for this primitive onward).
    fn apply_primitive_tpage(&mut self, uv_word: u32) {
        let tpage = (uv_word >> 16) & 0xFFFF;
        self.tex_page_x = ((tpage & 0x0F) as u16) * 64;
        self.tex_page_y = if (tpage >> 4) & 1 != 0 { 256 } else { 0 };
        self.tex_depth = ((tpage >> 7) & 0x3) as u8;
        self.tex_blend_mode = BlendMode::from_tpage_bits(tpage >> 5);
    }

    /// Plot a single 15bpp pixel at `(x, y)`. When `mode == Opaque`
    /// this is a plain VRAM write; otherwise we fetch the existing
    /// pixel and run the semi-transparency blend.
    ///
    /// Also respects the GP0 0xE6 mask-bit flags:
    /// - If `mask_check_before_draw` is on and the existing VRAM
    ///   pixel's bit 15 is already set, the plot is dropped (the
    ///   protected pixel survives).
    /// - If `mask_set_on_draw` is on, the new pixel is OR'd with
    ///   bit 15 so subsequent mask checks protect it.
    ///
    /// Callers do their own draw-area clipping before calling this
    /// — it's the hot per-pixel path and shouldn't re-check bounds.
    fn plot_pixel(&mut self, x: u16, y: u16, fg: u16, mode: BlendMode) {
        let existing = self.vram.get_pixel(x, y);
        if self.mask_check_before_draw && existing & 0x8000 != 0 {
            return;
        }
        let mut pixel = if mode == BlendMode::Opaque {
            fg
        } else {
            blend_pixel(existing, fg, mode)
        };
        if self.mask_set_on_draw {
            pixel |= 0x8000;
        }
        self.vram.set_pixel(x, y, pixel);
    }

    /// Rasterize a textured triangle — same edge-function test as the
    /// other triangle paths, with nearest-neighbor texture sampling
    /// via barycentric-interpolated UV.
    ///
    /// `semi_trans` is the primitive's command-bit-1 state. When set,
    /// texels with bit 15 high blend via `self.tex_blend_mode`; texels
    /// with bit 15 clear still draw opaquely. When clear, every texel
    /// draws opaque regardless of its bit 15 — matching PSX-SPX's
    /// per-texel semi-transparency rule.
    ///
    /// `tint` is the 24-bit vertex colour that modulates each texel
    /// (see [`modulate_tint`]). Raw-texture primitives (cmd bit 0 set)
    /// pass `(0x80, 0x80, 0x80)` so modulation is a no-op.
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
        semi_trans: bool,
        tint: (u32, u32, u32),
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
        let tpage_mode = self.tex_blend_mode;

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
                        let shaded = modulate_tint(texel, tint.0, tint.1, tint.2);
                        let mode = if semi_trans && (texel & 0x8000) != 0 {
                            tpage_mode
                        } else {
                            BlendMode::Opaque
                        };
                        self.plot_pixel(x as u16, y as u16, shaded, mode);
                    }
                }
            }
        }
    }

    /// Rasterize a triangle with per-vertex colours — Gouraud shading.
    /// Same edge-function inside test as the flat path, but interpolates
    /// RGB using normalized barycentric weights `(w0, w1, w2)` per pixel
    /// and packs the result back into a 15-bit BGR VRAM word.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_shaded_triangle(
        &mut self,
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        c0: u32,
        c1: u32,
        c2: u32,
        mode: BlendMode,
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
                    let colour = if self.dither_enabled {
                        dither_rgb(ri as i32, gi as i32, bi as i32, x, y)
                    } else {
                        rgb24_to_bgr15(
                            (ri.clamp(0, 255) as u32)
                                | ((gi.clamp(0, 255) as u32) << 8)
                                | ((bi.clamp(0, 255) as u32) << 16),
                        )
                    };
                    self.plot_pixel(x as u16, y as u16, colour, mode);
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

/// The PSX 4×4 Bayer dither matrix — signed 8-bit offsets applied to
/// each of R/G/B in 24-bit space before truncating to 5 bits per
/// channel. Indexed by `(y & 3) * 4 + (x & 3)`. Values are the PSX
/// hardware's published constants (PSX-SPX, "GPU Render Commands"):
///
/// ```text
///   -4  +0  -3  +1
///   +2  -2  +3  -1
///   -3  +1  -4  +0
///   +3  -1  +2  -2
/// ```
const DITHER_MATRIX: [i32; 16] = [
    -4, 0, -3, 1, 2, -2, 3, -1, -3, 1, -4, 0, 3, -1, 2, -2,
];

/// Apply the Bayer dither offset to an 8-bit RGB triple at screen
/// position `(x, y)`, clamp to 0..=255, then convert to the 15-bit
/// BGR VRAM word. Used by shaded + textured-shaded rasterizers when
/// `Gpu::dither_enabled` is on.
fn dither_rgb(r: i32, g: i32, b: i32, x: i32, y: i32) -> u16 {
    let offset = DITHER_MATRIX[((y & 3) * 4 + (x & 3)) as usize];
    let dr = (r + offset).clamp(0, 255) as u32;
    let dg = (g + offset).clamp(0, 255) as u32;
    let db = (b + offset).clamp(0, 255) as u32;
    rgb24_to_bgr15(dr | (dg << 8) | (db << 16))
}

/// PSX semi-transparency mode. The four non-`Opaque` variants map
/// directly to the four encodings in GP0 0xE1 / tpage bits 5-6.
/// `Opaque` is our shortcut for "don't touch the destination — just
/// overwrite" so primitives share one rasterizer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Write the foreground pixel directly, ignoring the background.
    Opaque,
    /// `(B + F) / 2` — 50% average. Smoke, translucent glass.
    Average,
    /// `B + F`, channel-clamped to 31. Additive blending — fire, lights.
    Add,
    /// `B - F`, channel-clamped to 0. Subtractive — shadows.
    Sub,
    /// `B + F/4`, channel-clamped to 31. Low-intensity additive —
    /// subtle glow / haze.
    AddQuarter,
}

impl BlendMode {
    /// Decode the 2-bit tpage/E1-command "semi-transparency" field
    /// (bits 5-6 of GP0 0xE1, or of a textured-primitive tpage word).
    /// Always returns a non-`Opaque` variant — whether the primitive
    /// actually blends is determined by the caller's "is this prim
    /// semi-transparent?" flag.
    fn from_tpage_bits(bits: u32) -> Self {
        match bits & 0x3 {
            0 => Self::Average,
            1 => Self::Add,
            2 => Self::Sub,
            _ => Self::AddQuarter,
        }
    }
}

/// `true` if the primitive-command word's cmd-bit-1 ("semi-trans
/// flag") is set. GP0 primitive opcodes are laid out as
/// `0b001XXPTC` where bit 1 (= `P`) is the semi-trans flag: 0 means
/// opaque, 1 means the primitive blends per the active tpage mode.
#[inline]
fn prim_is_semi_trans(cmd_word: u32) -> bool {
    (cmd_word >> 25) & 1 != 0
}

/// Resolve the blend mode for a non-textured primitive: opaque if
/// the cmd-bit-1 flag is clear, otherwise the current tpage's
/// active semi-transparency mode. Textured primitives don't use
/// this helper — per-texel bit-15 controls blending there.
#[inline]
fn prim_blend_mode(cmd_word: u32, tpage_mode: BlendMode) -> BlendMode {
    if prim_is_semi_trans(cmd_word) {
        tpage_mode
    } else {
        BlendMode::Opaque
    }
}

/// Modulate a 15bpp BGR texel by a 24-bit RGB tint. PSX formula:
/// `result_channel = (tint_8bit * texel_5bit * 2) / 0x100`, which
/// makes tint value `0x80` per channel act as identity (no-change)
/// and `0xFF` act as double-brightness (clamped to 31 per channel).
///
/// Called with `tint = 0x80_80_80` when the primitive is a "raw
/// texture" (cmd bit 0 set) — that passes the texel through
/// unchanged. Callers of flat-tint textured primitives derive the
/// tint from the cmd word's low 24 bits; Gouraud-textured primitives
/// interpolate per pixel and call us with the per-pixel colour.
///
/// The texel's mask bit (bit 15) is preserved so downstream
/// semi-transparency detection still sees it.
fn modulate_tint(texel: u16, tint_r: u32, tint_g: u32, tint_b: u32) -> u16 {
    let tr = (texel & 0x1F) as u32;
    let tg = ((texel >> 5) & 0x1F) as u32;
    let tb = ((texel >> 10) & 0x1F) as u32;
    let r = (tint_r * tr / 0x80).min(0x1F) as u16;
    let g = (tint_g * tg / 0x80).min(0x1F) as u16;
    let b = (tint_b * tb / 0x80).min(0x1F) as u16;
    r | (g << 5) | (b << 10) | (texel & 0x8000)
}

/// Split a 24-bit RGB tint word (from the low 24 bits of a textured
/// primitive's command) into the three channels the modulator
/// expects. Returns `(tint_r, tint_g, tint_b)` with each in 0..=255.
/// For "raw texture" primitives the caller substitutes `(128, 128,
/// 128)` directly — one code path through [`modulate_tint`].
#[inline]
fn split_tint(tint24: u32) -> (u32, u32, u32) {
    (
        tint24 & 0xFF,
        (tint24 >> 8) & 0xFF,
        (tint24 >> 16) & 0xFF,
    )
}

/// Identity tint — pass-through for raw-texture primitives. Each
/// channel at `0x80` means modulation returns the texel unchanged.
const RAW_TEXTURE_TINT: (u32, u32, u32) = (0x80, 0x80, 0x80);

/// Blend a foreground pixel over a background pixel per `mode`.
/// Both pixels are 15-bit BGR with a mask bit at bit 15. The mask
/// bit of the result comes from the foreground so semi-transparent
/// texels keep marking themselves.
fn blend_pixel(bg: u16, fg: u16, mode: BlendMode) -> u16 {
    if mode == BlendMode::Opaque {
        return fg;
    }
    // Channel extraction — 5 bits each, i16 so we can subtract without
    // wrapping and compare signed for the saturating clamp.
    let br = (bg & 0x1F) as i16;
    let bgg = ((bg >> 5) & 0x1F) as i16;
    let bb = ((bg >> 10) & 0x1F) as i16;
    let fr = (fg & 0x1F) as i16;
    let fgg = ((fg >> 5) & 0x1F) as i16;
    let fb = ((fg >> 10) & 0x1F) as i16;
    let (r, g, b) = match mode {
        BlendMode::Opaque => unreachable!(),
        BlendMode::Average => ((br + fr) / 2, (bgg + fgg) / 2, (bb + fb) / 2),
        BlendMode::Add => (
            (br + fr).min(31),
            (bgg + fgg).min(31),
            (bb + fb).min(31),
        ),
        BlendMode::Sub => ((br - fr).max(0), (bgg - fgg).max(0), (bb - fb).max(0)),
        BlendMode::AddQuarter => (
            (br + fr / 4).min(31),
            (bgg + fgg / 4).min(31),
            (bb + fb / 4).min(31),
        ),
    };
    (r as u16) | ((g as u16) << 5) | ((b as u16) << 10) | (fg & 0x8000)
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
        // Reset defaults matching Redux's `SoftGPU::impl::open` /
        // `softReset`, both of which initialise `m_statusRet` to
        // `0x14802000`:
        //   bit 23 (DISPLAY_DISABLE) = 1
        //   bit 21 = 1 (reserved, Redux sets on reset)
        //   bit 13 (INTERLACE_FIELD) = 1
        //   bit 31 (DRAWING_ODD) = 0 at power-on; first VBlank XORs it
        //          to 1, second back to 0, etc. Earlier we initialised
        //          this to 1 (matching the "odd field looks ready"
        //          intuition), but parity divergence at step 19259778
        //          showed Redux is the authority and starts at 0.
        // Ready bits 26/28 are filled in by `read`; VRAM-ready (27) is
        // gated on an active VRAM→CPU transfer.
        Self {
            raw: 0x1480_2000,
        }
    }

    /// Compose the observable GPUSTAT word. `vram_send_ready` is the
    /// live "is a VRAM→CPU transfer in progress and has pixels
    /// waiting" flag owned by the GPU proper — we pass it in so the
    /// status register doesn't duplicate state that lives elsewhere.
    fn read(&self, vram_send_ready: bool) -> u32 {
        let mut ret = self.raw;

        // Always-ready: bits 26 (cmd FIFO ready) + 28 (DMA block ready).
        // Bit 27 (VRAM→CPU ready) is only set while software is
        // actually pulling pixels from a GPUREAD; Redux's default is 0
        // and the BIOS polls this bit to detect transfer completion.
        ret |= 0x1400_0000;
        if vram_send_ready {
            ret |= 0x0800_0000;
        } else {
            ret &= !0x0800_0000;
        }

        // Bit 25 (DMA data request) — Redux-observed semantics, not
        // PSX-SPX's. PSX-SPX says dir=2 copies bit 28 and dir=3 copies
        // bit 27; Redux only sets bit 25 when direction == 1 (FIFO)
        // and leaves it clear otherwise. Parity at step 19,474,030
        // pinned this: the BIOS polls GPUSTAT during its post-Sony
        // intro init with DMA direction 2 and expects bit 25 = 0.
        // See `pcsx-redux/src/core/gpu.cc::readStatus`.
        ret &= !0x0200_0000;
        if (ret & 0x6000_0000) == 0x2000_0000 {
            ret |= 0x0200_0000;
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
        // Bits 26 (cmd ready) + 28 (DMA block ready) are always set.
        // Bit 27 (VRAM→CPU ready) is gated on an active transfer;
        // we don't have one here, so it's clear.
        assert_eq!(stat & 0x1400_0000, 0x1400_0000);
        assert_eq!(stat & 0x0800_0000, 0, "VRAM-send ready clear when idle");
    }

    #[test]
    fn read_status_sets_vram_send_ready_during_download() {
        let mut gpu = Gpu::new();
        // Start a VRAM→CPU download via GP0 0xC0.
        gpu.write32(GP0_ADDR, 0xC000_0000); // header
        gpu.write32(GP0_ADDR, 0); // xy
        gpu.write32(GP0_ADDR, 0x0001_0001); // 1x1 size
        let stat = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat & 0x0800_0000, 0x0800_0000, "bit 27 set during transfer");
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

    #[test]
    fn blend_mode_decodes_tpage_bits() {
        assert_eq!(BlendMode::from_tpage_bits(0), BlendMode::Average);
        assert_eq!(BlendMode::from_tpage_bits(1), BlendMode::Add);
        assert_eq!(BlendMode::from_tpage_bits(2), BlendMode::Sub);
        assert_eq!(BlendMode::from_tpage_bits(3), BlendMode::AddQuarter);
        // Higher bits are masked off.
        assert_eq!(BlendMode::from_tpage_bits(0b100), BlendMode::Average);
    }

    #[test]
    fn blend_opaque_returns_foreground_unchanged() {
        assert_eq!(blend_pixel(0x1234, 0x5678, BlendMode::Opaque), 0x5678);
    }

    #[test]
    fn blend_average_halves_then_sums() {
        // BG = (10, 10, 10), FG = (20, 20, 20) → avg = (15, 15, 15).
        let bg = 10 | (10 << 5) | (10 << 10);
        let fg = 20 | (20 << 5) | (20 << 10);
        let out = blend_pixel(bg, fg, BlendMode::Average);
        assert_eq!(out & 0x1F, 15);
        assert_eq!((out >> 5) & 0x1F, 15);
        assert_eq!((out >> 10) & 0x1F, 15);
    }

    #[test]
    fn blend_add_saturates_at_31() {
        let bg = 20 | (20 << 5) | (20 << 10);
        let fg = 20 | (20 << 5) | (20 << 10);
        let out = blend_pixel(bg, fg, BlendMode::Add);
        // 20+20 = 40 → clamps to 31 per channel.
        assert_eq!(out & 0x1F, 31);
        assert_eq!((out >> 5) & 0x1F, 31);
        assert_eq!((out >> 10) & 0x1F, 31);
    }

    #[test]
    fn blend_sub_saturates_at_zero() {
        let bg = 5 | (10 << 5) | (15 << 10);
        let fg = 10 | (10 << 5) | (10 << 10);
        let out = blend_pixel(bg, fg, BlendMode::Sub);
        // R: 5-10 = -5 → 0. G: 10-10 = 0. B: 15-10 = 5.
        assert_eq!(out & 0x1F, 0);
        assert_eq!((out >> 5) & 0x1F, 0);
        assert_eq!((out >> 10) & 0x1F, 5);
    }

    #[test]
    fn blend_add_quarter_adds_fractional_foreground() {
        let bg = 10 | (10 << 5) | (10 << 10);
        let fg = 20 | (20 << 5) | (20 << 10);
        let out = blend_pixel(bg, fg, BlendMode::AddQuarter);
        // BG + FG/4 → 10 + 5 = 15 per channel.
        assert_eq!(out & 0x1F, 15);
        assert_eq!((out >> 5) & 0x1F, 15);
        assert_eq!((out >> 10) & 0x1F, 15);
    }

    #[test]
    fn blend_preserves_foreground_mask_bit() {
        // Mask bit (bit 15) must come from the foreground so semi-
        // transparent texels keep marking themselves.
        let bg = 0x0000;
        let fg = 0x8000 | 10;
        let out = blend_pixel(bg, fg, BlendMode::Average);
        assert_eq!(out & 0x8000, 0x8000);
    }

    #[test]
    fn prim_helpers_decode_semi_trans_bit() {
        // 0x20 = opaque monochrome tri. 0x22 = semi-trans monochrome tri.
        // The opcode is in bits 24..=31; bit 25 of the word = bit 1 of op.
        assert!(!prim_is_semi_trans(0x2000_0000));
        assert!(prim_is_semi_trans(0x2200_0000));
        assert_eq!(
            prim_blend_mode(0x2000_0000, BlendMode::Add),
            BlendMode::Opaque
        );
        assert_eq!(
            prim_blend_mode(0x2200_0000, BlendMode::Add),
            BlendMode::Add
        );
    }

    #[test]
    fn draw_mode_e1_extracts_blend_mode() {
        // GP0 0xE1: bits 5-6 select semi-transparency mode.
        let mut gpu = Gpu::new();
        gpu.write32(GP0_ADDR, 0xE100_0020); // bits 5-6 = 01 → Add
        assert_eq!(gpu.tex_blend_mode, BlendMode::Add);
        gpu.write32(GP0_ADDR, 0xE100_0060); // bits 5-6 = 11 → AddQuarter
        assert_eq!(gpu.tex_blend_mode, BlendMode::AddQuarter);
    }

    #[test]
    fn modulate_tint_identity_at_0x80() {
        // tint 0x80 per channel = identity. Any texel passes unchanged.
        let texel = 0x1234; // arbitrary 15bpp
        let out = modulate_tint(texel, 0x80, 0x80, 0x80);
        assert_eq!(out, texel);
    }

    #[test]
    fn modulate_tint_scales_each_channel() {
        // texel = (R=16, G=10, B=5) at bits (0..5), (5..10), (10..15).
        let texel: u16 = 16 | (10 << 5) | (5 << 10);
        // tint R=0xC0 (1.5×), G=0x40 (0.5×), B=0x80 (1.0×).
        let out = modulate_tint(texel, 0xC0, 0x40, 0x80);
        // Expected:
        //   R = 0xC0 * 16 / 0x80 = 192 * 16 / 128 = 24 → clamp 31 → 24
        //   G = 0x40 * 10 / 0x80 = 64 * 10 / 128 = 5
        //   B = 0x80 * 5 / 0x80 = 5
        assert_eq!(out & 0x1F, 24);
        assert_eq!((out >> 5) & 0x1F, 5);
        assert_eq!((out >> 10) & 0x1F, 5);
    }

    #[test]
    fn modulate_tint_clamps_to_31_on_overbright() {
        // texel at max (31 each), tint at max (0xFF each) → should
        // clamp to 31 per channel.
        let texel: u16 = 31 | (31 << 5) | (31 << 10);
        let out = modulate_tint(texel, 0xFF, 0xFF, 0xFF);
        assert_eq!(out & 0x1F, 31);
        assert_eq!((out >> 5) & 0x1F, 31);
        assert_eq!((out >> 10) & 0x1F, 31);
    }

    #[test]
    fn modulate_tint_preserves_mask_bit() {
        // Semi-transparent texel (bit 15 set) must keep bit 15 after
        // modulation so downstream blend logic still fires.
        let texel: u16 = 0x8000 | 10;
        let out = modulate_tint(texel, 0x80, 0x80, 0x80);
        assert_eq!(out & 0x8000, 0x8000);
    }

    #[test]
    fn split_tint_extracts_rgb_channels() {
        // PSX tint word = 0xBBGGRR (the low 24 bits of a
        // textured-primitive cmd).
        assert_eq!(split_tint(0x00123456), (0x56, 0x34, 0x12));
        assert_eq!(split_tint(0x00FFFFFF), (0xFF, 0xFF, 0xFF));
        assert_eq!(split_tint(0x0080_8080), RAW_TEXTURE_TINT);
    }

    #[test]
    fn gp0_e6_sets_mask_flags() {
        let mut gpu = Gpu::new();
        gpu.write32(GP0_ADDR, 0xE600_0000); // both clear
        assert!(!gpu.mask_set_on_draw);
        assert!(!gpu.mask_check_before_draw);
        gpu.write32(GP0_ADDR, 0xE600_0001); // set-on-draw only
        assert!(gpu.mask_set_on_draw);
        assert!(!gpu.mask_check_before_draw);
        gpu.write32(GP0_ADDR, 0xE600_0003); // both on
        assert!(gpu.mask_set_on_draw);
        assert!(gpu.mask_check_before_draw);
        // GPUSTAT bits 11 (set-on-draw) + 12 (check-before-draw)
        // mirror the flag state.
        let stat = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat & 0x1800, 0x1800);
    }

    #[test]
    fn plot_pixel_respects_set_mask_on_draw() {
        let mut gpu = Gpu::new();
        gpu.write32(GP0_ADDR, 0xE600_0001); // set-on-draw
        gpu.plot_pixel(10, 10, 0x1234, BlendMode::Opaque);
        // The drawn pixel should have bit 15 forced to 1.
        assert_eq!(gpu.vram.get_pixel(10, 10) & 0x8000, 0x8000);
    }

    #[test]
    fn plot_pixel_skips_when_mask_check_sees_masked_pixel() {
        let mut gpu = Gpu::new();
        // Pre-mark (20, 20) as masked.
        gpu.vram.set_pixel(20, 20, 0x8000 | 0x1234);
        gpu.write32(GP0_ADDR, 0xE600_0002); // check-before-draw
        gpu.plot_pixel(20, 20, 0x5678, BlendMode::Opaque);
        // Drop: original pixel survives.
        assert_eq!(gpu.vram.get_pixel(20, 20), 0x8000 | 0x1234);
    }

    #[test]
    fn plot_pixel_draws_when_mask_check_sees_unmasked_pixel() {
        let mut gpu = Gpu::new();
        gpu.vram.set_pixel(30, 30, 0x1234); // mask bit clear
        gpu.write32(GP0_ADDR, 0xE600_0002); // check-before-draw
        gpu.plot_pixel(30, 30, 0x5678, BlendMode::Opaque);
        assert_eq!(gpu.vram.get_pixel(30, 30), 0x5678);
    }

    #[test]
    fn gp1_reset_clears_mask_flags() {
        let mut gpu = Gpu::new();
        gpu.write32(GP0_ADDR, 0xE600_0003); // both on
        gpu.write32(GP1_ADDR, 0x0000_0000); // GP1 reset
        assert!(!gpu.mask_set_on_draw);
        assert!(!gpu.mask_check_before_draw);
        let stat = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat & 0x1800, 0);
    }

    #[test]
    fn gp0_e2_parses_texture_window_fields() {
        let mut gpu = Gpu::new();
        // mask_x = 3 (24 px), mask_y = 5 (40 px), off_x = 1 (8 px),
        // off_y = 2 (16 px).
        //
        //   bits 0..=4  : mask_x  = 3
        //   bits 5..=9  : mask_y  = 5
        //   bits 10..=14: off_x   = 1
        //   bits 15..=19: off_y   = 2
        let word = 0xE200_0000u32 | 3 | (5 << 5) | (1 << 10) | (2 << 15);
        gpu.write32(GP0_ADDR, word);
        assert_eq!(gpu.tex_window_mask_x, 24);
        assert_eq!(gpu.tex_window_mask_y, 40);
        assert_eq!(gpu.tex_window_offset_x, 8);
        assert_eq!(gpu.tex_window_offset_y, 16);
    }

    #[test]
    fn texture_window_default_is_passthrough() {
        // Default window (all zeroes) must leave UV unchanged when
        // sampled; otherwise we'd break every game that doesn't
        // touch GP0 0xE2.
        let gpu = Gpu::new();
        assert_eq!(gpu.tex_window_mask_x, 0);
        assert_eq!(gpu.tex_window_mask_y, 0);
        assert_eq!(gpu.tex_window_offset_x, 0);
        assert_eq!(gpu.tex_window_offset_y, 0);
        // The sample-time formula is `u & !mask | offset & mask`;
        // with mask=0 every u passes through.
        let u: u16 = 0x5A;
        let mask: u16 = 0;
        let off: u16 = 0;
        assert_eq!((u & !mask) | (off & mask), u);
    }

    #[test]
    fn textured_shaded_tri_packet_size_is_nine() {
        // 0x34..=0x37 is "textured + Gouraud-shaded triangle" —
        // 3 vertices × (colour+vertex+uv) = 9 words total.
        assert_eq!(gp0_packet_size(0x34), 9);
        assert_eq!(gp0_packet_size(0x37), 9);
    }

    #[test]
    fn textured_shaded_quad_packet_size_is_twelve() {
        // 0x3C..=0x3F is "textured + Gouraud-shaded quad" —
        // 4 vertices × (colour+vertex+uv) = 12 words.
        assert_eq!(gp0_packet_size(0x3C), 12);
        assert_eq!(gp0_packet_size(0x3F), 12);
    }

    #[test]
    fn gp0_e1_bit_9_toggles_dither_flag() {
        let mut gpu = Gpu::new();
        assert!(!gpu.dither_enabled);
        // E1h with bit 9 set → dither on.
        gpu.write32(GP0_ADDR, 0xE100_0000 | (1 << 9));
        assert!(gpu.dither_enabled);
        // E1h with bit 9 clear → dither off.
        gpu.write32(GP0_ADDR, 0xE100_0000);
        assert!(!gpu.dither_enabled);
    }

    #[test]
    fn dither_rgb_produces_offsets_within_clamp() {
        // Mid-value input (128) with the most negative offset (-4)
        // still lands in the valid 0..=255 range.
        let v = dither_rgb(128, 128, 128, 0, 0); // offset = -4
        // Top 5 bits of 124 (128 - 4 = 124), 124 >> 3 = 15.
        assert_eq!(v & 0x1F, 15);
        // Full-saturation input with +3 offset stays clamped at 255.
        let v = dither_rgb(255, 255, 255, 0, 3); // matrix[12] = +3
        assert_eq!(v & 0x1F, 31);
        assert_eq!((v >> 5) & 0x1F, 31);
        assert_eq!((v >> 10) & 0x1F, 31);
    }

    #[test]
    fn textured_shaded_tri_consumes_full_packet_without_panic() {
        // Smoke test: feeding a complete 9-word textured-shaded tri
        // packet must not panic or leave the FIFO partially full.
        let mut gpu = Gpu::new();
        // All vertices inside draw area, degenerate (zero-area) triangle
        // so we don't need to chase pixel output — the dispatch path is
        // what we're testing.
        gpu.write32(GP0_ADDR, 0xE3_00_00_00); // draw area top-left 0,0
        gpu.write32(GP0_ADDR, 0xE4_00_03_FF); // draw area bottom-right 1023,0
        let words = [
            0x34_FF_FF_FFu32, // cmd + c0 = white
            0x0000_0000,      // v0 = (0, 0)
            0x0000_1020,      // uv0 + clut
            0x00FF_00FF,      // c1 = cyan
            0x0000_0000,      // v1 = (0, 0) (degenerate)
            0x0040_0000,      // uv1 + texpage
            0x00_00_FF_00,    // c2 = green
            0x0000_0000,      // v2 = (0, 0)
            0x0000_1020,      // uv2
        ];
        for w in words {
            gpu.write32(GP0_ADDR, w);
        }
        // FIFO must be empty — the 9-word packet consumed cleanly.
        assert_eq!(gpu.gp0_expected, 0);
    }
}
