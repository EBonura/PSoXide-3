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

    /// Horizontal-flip flag for textured rectangles (GP0 0xE1 bit 12).
    /// When true, texture U coordinates are mirrored across the
    /// rectangle's midline. Common for sprite animations that reuse
    /// a single source image with direction-dependent facing.
    tex_rect_flip_x: bool,
    /// Vertical-flip flag for textured rectangles (GP0 0xE1 bit 13).
    tex_rect_flip_y: bool,

    /// Active polyline receive state. `None` when no polyline is in
    /// flight; `Some(...)` between a polyline start packet and its
    /// terminator word. While `Some`, every GP0 write is
    /// interpreted as polyline continuation data, bypassing the
    /// regular packet assembler.
    polyline: Option<PolylineState>,

    /// Wireframe mode — replaces filled triangles with their
    /// three edges, rendered as lines at the triangle's primary
    /// colour. Rectangles and already-line primitives are
    /// unchanged. Off by default; toggled from the frontend's
    /// debug toolbar for visualising the geometry a game is
    /// submitting.
    pub wireframe_enabled: bool,

    /// Pixel-owner trace — when `Some`, every `plot_pixel` records
    /// the index of the currently-executing GPU command into
    /// `pixel_owner[y*VRAM_WIDTH + x]`. Paired with `cmd_log`
    /// this lets us answer "which command drew the pixel at
    /// (x, y)?" after a run, the essential first step in
    /// diagnosing per-pixel parity divergences against Redux.
    ///
    /// Allocating is opt-in because the buffer is 2 MiB — tiny
    /// in absolute terms but enough to want control over when it
    /// appears in core state.
    pub pixel_owner: Option<Vec<u32>>,
    /// Command log — one entry per GP0 packet executed since this
    /// tracer was enabled. Each entry captures the opcode plus
    /// the raw fifo words the packet consumed, so we can replay
    /// the exact inputs to a single draw in isolation.
    pub cmd_log: Vec<GpuCmdLogEntry>,
    /// Master gate for `cmd_log` pushes. Set by both
    /// `enable_pixel_tracer` and the lighter `enable_cmd_log`.
    /// Decoupled from `pixel_owner.is_some()` so the HW renderer can
    /// capture the GP0 stream without paying for the 2 MiB owner Vec
    /// or its per-pixel stamping cost.
    cmd_log_enabled: bool,
    /// The index that will be written into `pixel_owner` for the
    /// NEXT pixel plotted — i.e., the index of the currently-
    /// executing command. Bumped just before each packet dispatch.
    current_cmd_index: u32,

    /// Cumulative diagnostic "pseudo-busy" credit for expensive
    /// primitives. This deliberately does *not* affect GPUSTAT:
    /// PCSX-Redux's soft GPU keeps command/DMA-ready bits set even
    /// while software has just kicked a large copy/fill, and games
    /// like WipEout 2097 poll those bits during boot.
    busy_credit: u64,

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
    /// Distinct raw GP1 0x08 display-mode values seen since reset.
    display_mode_history: std::collections::BTreeSet<u32>,
    /// Recent GP1 writes in chronological order. Diagnostic only; capped
    /// so long FMV probes do not grow without bound.
    gp1_write_history: Vec<u32>,
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
    /// Horizontal resolution in pixels (one of 256/320/368/384/512/640).
    pub width: u16,
    /// Vertical resolution in pixels (240 or 480 interlaced).
    pub height: u16,
    /// `true` when the GP1 0x08 colour-depth bit selected 24bpp mode.
    /// The frontend framebuffer panel still decodes VRAM as 15bpp;
    /// respecting this flag is a future refinement.
    pub bpp24: bool,
}

/// One captured GP0 packet in the pixel-tracer's command log.
/// `index` matches the value stored in [`Gpu::pixel_owner`] for every
/// pixel this packet plotted, so `pixel_owner[y*W+x]` → look up the
/// corresponding `cmd_log` entry to see what primitive drew that
/// pixel.
#[derive(Debug, Clone)]
pub struct GpuCmdLogEntry {
    /// Monotonic command index, starting at 0. Wraps via saturation
    /// at u32::MAX; not a concern for typical debug runs (a few
    /// hundred thousand draw calls at most).
    pub index: u32,
    /// Opcode byte — top 8 bits of the first FIFO word.
    pub opcode: u8,
    /// Full FIFO contents at dispatch time. Short slices (3..=12 words)
    /// so cloning per command is cheap.
    pub fifo: Vec<u32>,
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
            tex_rect_flip_x: false,
            tex_rect_flip_y: false,
            polyline: None,
            wireframe_enabled: false,
            pixel_owner: None,
            cmd_log: Vec::new(),
            cmd_log_enabled: false,
            current_cmd_index: 0,
            busy_credit: 0,
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
            display_mode_history: std::collections::BTreeSet::new(),
            gp1_write_history: Vec::new(),
        }
    }

    /// Distinct display-start corners the BIOS has written to. Useful
    /// for telling a re-write loop from a front/back-buffer flip.
    pub fn display_start_history(&self) -> impl Iterator<Item = (u16, u16)> + '_ {
        self.display_start_history.iter().copied()
    }

    /// Distinct raw GP1 0x08 display-mode values seen since reset.
    pub fn display_mode_history(&self) -> impl Iterator<Item = u32> + '_ {
        self.display_mode_history.iter().copied()
    }

    /// Recent raw GP1 writes in chronological order. Diagnostic.
    pub fn gp1_write_history(&self) -> &[u32] {
        &self.gp1_write_history
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

    /// Enable per-pixel command tracing. Allocates the 2 MiB owner
    /// buffer (one u32 per VRAM pixel). Every subsequent
    /// `plot_pixel` stamps the currently-executing command's index
    /// into the buffer; every subsequent `execute_gp0_packet`
    /// pushes a `GpuCmdLogEntry` into `cmd_log`.
    ///
    /// Idempotent: re-enabling resets the tracer to empty.
    pub fn enable_pixel_tracer(&mut self) {
        const SENTINEL_NO_OWNER: u32 = u32::MAX;
        self.pixel_owner = Some(vec![SENTINEL_NO_OWNER; VRAM_WIDTH * VRAM_HEIGHT]);
        self.cmd_log.clear();
        self.current_cmd_index = 0;
        self.cmd_log_enabled = true;
    }

    /// Enable cmd_log capture WITHOUT allocating the per-pixel owner
    /// buffer. The HW (wgpu render-pipeline) renderer needs the GP0
    /// packet stream to drive its draw calls each frame, but doesn't
    /// need pixel-level provenance — saves the 2 MiB owner Vec and
    /// the per-`plot_pixel` stamp cost. The bench probes that DO
    /// want owner tracking still call `enable_pixel_tracer`.
    ///
    /// Idempotent: re-enabling clears cmd_log.
    pub fn enable_cmd_log(&mut self) {
        self.cmd_log.clear();
        self.current_cmd_index = 0;
        self.cmd_log_enabled = true;
    }

    /// Whether `cmd_log` capture is currently armed (either via
    /// `enable_cmd_log` or `enable_pixel_tracer`). Lets the
    /// frontend call `enable_cmd_log` at most once per Bus
    /// lifetime instead of clobbering the log every frame.
    pub fn cmd_log_enabled(&self) -> bool {
        self.cmd_log_enabled
    }

    /// Look up which command drew the pixel at (x, y), returning
    /// `None` if no command has touched that pixel since the tracer
    /// was enabled (or if the tracer is off). The returned entry
    /// carries the opcode + raw FIFO words, enough to replay the
    /// single command in isolation.
    pub fn pixel_owner_at(&self, x: u16, y: u16) -> Option<&GpuCmdLogEntry> {
        let pixel_owner = self.pixel_owner.as_ref()?;
        let idx = pixel_owner
            .get(y as usize * VRAM_WIDTH + x as usize)
            .copied()?;
        if idx == u32::MAX {
            return None;
        }
        self.cmd_log.get(idx as usize)
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
                    out.extend_from_slice(&[r | (r >> 5), g | (g >> 5), b | (b >> 5), 0xFF]);
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

    /// Charge "busy credit" that clears the GPU's ready-bits until
    /// it's spent. Called by the rasteriser after each expensive
    /// primitive (VRAM-to-VRAM copy, full-screen fill) to expose
    /// a realistic 0→1 transition on GPUSTAT bit 26 / 28 for games
    /// that spin-wait on GPU idle before kicking the next DMA.
    ///
    /// Small primitives (triangles, lines, short rectangles) don't
    /// charge any credit — they're fast enough on real hardware
    /// that software rarely polls ready between them, and charging
    /// busy here would just add delays where none are needed.
    pub fn charge_busy(&mut self, cost: u64) {
        self.busy_credit = self.busy_credit.saturating_add(cost);
    }

    /// Drain busy credit over time. Called by the bus each tick
    /// so the busy flag settles back to "ready" as cycles advance.
    /// One cycle of real time decays one unit of credit — with
    /// primitives charging in the tens of units, the ready flag
    /// goes high again within a few hundred cycles.
    pub fn decay_busy(&mut self, cycles: u64) {
        self.busy_credit = self.busy_credit.saturating_sub(cycles);
    }

    /// Is the GPU currently "busy"? Used to gate GPUSTAT ready
    /// bits 26 + 28.
    pub fn is_busy(&self) -> bool {
        self.busy_credit > 0
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
                if self.gp1_write_history.len() == 512 {
                    self.gp1_write_history.remove(0);
                }
                self.gp1_write_history.push(value);
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
                self.display_mode_history.insert(value);
                let hres = if value & (1 << 6) != 0 {
                    match value & 0x3 {
                        0 => 368,
                        1 => 384,
                        2 => 512,
                        3 => 640,
                        _ => unreachable!(),
                    }
                } else {
                    match value & 0x3 {
                        0 => 256,
                        1 => 320,
                        2 => 512,
                        3 => 640,
                        _ => unreachable!(),
                    }
                };
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

        // Polyline receive — every word is either a vertex / colour
        // or the terminator sentinel until the list ends.
        if self.polyline.is_some() {
            self.ingest_polyline_word(word);
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
        // Pixel tracer also wants to see state-modifying single-word
        // packets (0xE1..=0xE6 draw-mode / tex-window / draw-area /
        // draw-offset / mask). These don't plot pixels but they shift
        // the state that subsequent draws interpret — useful to see
        // in the log when chasing a parity divergence.
        if self.cmd_log_enabled {
            let index = self.cmd_log.len() as u32;
            self.current_cmd_index = index;
            self.cmd_log.push(GpuCmdLogEntry {
                index,
                opcode: op as u8,
                fifo: vec![word],
            });
        }
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
                self.tex_rect_flip_x = (word >> 12) & 1 != 0;
                self.tex_rect_flip_y = (word >> 13) & 1 != 0;
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
        self.gp0_opcode_hist[op as usize] = self.gp0_opcode_hist[op as usize].saturating_add(1);
        // If the pixel tracer is armed, stamp this packet into the
        // command log *before* dispatching — `plot_pixel` uses
        // `current_cmd_index` to tag every write, so it must point
        // at the index of the entry we're about to push.
        if self.cmd_log_enabled {
            let index = self.cmd_log.len() as u32;
            self.current_cmd_index = index;
            self.cmd_log.push(GpuCmdLogEntry {
                index,
                opcode: op as u8,
                fifo: self.gp0_fifo.clone(),
            });
        }
        match op {
            // Monochrome fill rect (ignores draw area / offset).
            0x02 => self.fill_rect(),
            // Monochrome triangle / quad. Bit 3 distinguishes 3-vs-4
            // vertices; bit 1 is opaque-vs-semi-transparent (we treat
            // both as opaque for now).
            0x20..=0x23 => self.draw_monochrome_tri(),
            0x28..=0x2B => self.draw_monochrome_quad(),
            // Single monochrome line — 3 words.
            0x40..=0x43 => self.draw_line_mono_single(),
            // Polyline monochrome start — 3 words. After this packet
            // the FIFO enters a streaming mode that accepts vertex
            // words until the 0x55555555 / 0x50005000 terminator.
            0x48..=0x4B => self.draw_line_mono_start_polyline(),
            // Single shaded line — 4 words.
            0x50..=0x53 => self.draw_line_shaded_single(),
            // Polyline shaded start — 4 words.
            0x58..=0x5B => self.draw_line_shaded_start_polyline(),
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
        // Charge busy: VRAM↔VRAM copies cost one cycle per pixel
        // on hardware; games may poll ready right after kicking.
        // Use ~1 cycle/pixel so bit 26 clears for the duration of
        // the copy as software-visible time advances.
        self.charge_busy((w as u64) * (h as u64));
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

    /// GP0 0x28..=0x2B — monochrome 4-vertex quad. Redux draws the
    /// lower/right half first, then the upper/left half, so pixels on
    /// the shared diagonal are owned by `(v0, v1, v2)`.
    fn draw_monochrome_quad(&mut self) {
        let cmd = self.gp0_fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let v1 = self.decode_vertex(self.gp0_fifo[2]);
        let v2 = self.decode_vertex(self.gp0_fifo[3]);
        let v3 = self.decode_vertex(self.gp0_fifo[4]);
        self.rasterize_triangle(v1, v3, v2, color, mode);
        self.rasterize_triangle(v0, v1, v2, color, mode);
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
    /// 8 words, split in Redux order so the first half wins the
    /// shared diagonal.
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
        self.rasterize_shaded_triangle(v1, v3, v2, c1, c3, c2, mode);
        self.rasterize_shaded_triangle(v0, v1, v2, c0, c1, c2, mode);
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
        // Raw-texture flag is bit 0 of the *opcode byte* (bit 24 of
        // the full cmd word), per PSX-SPX. Testing `cmd & 1` reads
        // bit 0 of the R channel of the embedded colour instead —
        // so odd-R tints like 0xFF would be mis-flagged raw.
        let tint = if (cmd >> 24) & 1 != 0 {
            RAW_TEXTURE_TINT
        } else {
            split_tint(cmd & 0x00FF_FFFF)
        };
        self.paint_textured_rect(x, y, w, h, u0, v0, clut_word, prim_is_semi_trans(cmd), tint);
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
        // Raw-texture flag is bit 0 of the *opcode byte* (bit 24 of
        // the full cmd word), per PSX-SPX. Testing `cmd & 1` reads
        // bit 0 of the R channel of the embedded colour instead —
        // so odd-R tints like 0xFF would be mis-flagged raw.
        let tint = if (cmd >> 24) & 1 != 0 {
            RAW_TEXTURE_TINT
        } else {
            split_tint(cmd & 0x00FF_FFFF)
        };
        self.paint_textured_rect(x, y, w, h, u0, v0, clut_word, prim_is_semi_trans(cmd), tint);
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

        let flip_x = self.tex_rect_flip_x;
        let flip_y = self.tex_rect_flip_y;
        let last_col = (w - 1) as u16;
        let last_row = (h - 1) as u16;
        for py in top..=bottom {
            for px in left..=right {
                let dx = (px - x) as u16;
                let dy = (py - y) as u16;
                let u_off = if flip_x { last_col - dx } else { dx };
                let v_off = if flip_y { last_row - dy } else { dy };
                let tex_u = u0.wrapping_add(u_off);
                let tex_v = v0.wrapping_add(v_off);
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
    /// `None` for transparent — PSX convention is **the resolved
    /// 16-bit colour == 0x0000**, regardless of mode. For 4bpp/8bpp,
    /// that means `CLUT[idx] == 0` (not `idx == 0`); games routinely
    /// place `0x0000` at non-zero CLUT entries to punch transparency
    /// into sprites (e.g. the BIOS TM glyph: background uses a CLUT
    /// index whose entry is 0 → transparent). Checking `idx == 0`
    /// instead is a common simplification that renders those pixels
    /// opaque black, producing the infamous "TM on a black box"
    /// regression. Matches Redux's `getTextureTransCol*` which all
    /// start with `if (color == 0) return;`.
    ///
    /// The incoming `u` / `v` are run through the GP0 0xE2 texture
    /// window first: `U' = (U & ~mask) | (offset & mask)` per axis.
    /// With the default (all zeroes) that's a no-op; games that use
    /// tiling set non-zero mask/offset to reuse a sub-rectangle of
    /// the tpage across multiple primitives.
    fn sample_texture(&self, u: u16, v: u16, clut_x: u16, clut_y: u16) -> Option<u16> {
        // PSX-SPX: the GPU's U/V counters are 8 bits — texture pages
        // wrap every 256 texels horizontally and vertically. Callers
        // pass u16 because rasterizer interpolation works in a wider
        // domain, so we mask down to 8 bits *before* the texture
        // window. Without this, a sprite or polygon whose `U + dx`
        // exceeds 255 reads VRAM PAST the tpage edge — typically the
        // neighbouring tpage's data, garbage texels, or a different
        // CLUT-driven byte. Visible as smeared / corrupted 2D sprites
        // (Tekken 3 character portraits, BIOS dialog frames).
        let u = u & 0xFF;
        let v = v & 0xFF;
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
        let texel = match self.tex_depth {
            0 => {
                // 4bpp: 4 texels per VRAM word; select by (u & 3).
                let tpx = self.tex_page_x.wrapping_add(u / 4);
                let word = self.vram.get_pixel(tpx, tpy);
                let idx = (word >> ((u & 3) * 4)) & 0xF;
                self.vram.get_pixel(clut_x + idx, clut_y)
            }
            1 => {
                // 8bpp: 2 texels per VRAM word.
                let tpx = self.tex_page_x.wrapping_add(u / 2);
                let word = self.vram.get_pixel(tpx, tpy);
                let idx = (word >> ((u & 1) * 8)) & 0xFF;
                self.vram.get_pixel(clut_x + idx, clut_y)
            }
            _ => {
                // 15bpp: direct colour, 1 texel per word.
                let tpx = self.tex_page_x.wrapping_add(u);
                self.vram.get_pixel(tpx, tpy)
            }
        };
        if texel == 0 {
            None
        } else {
            Some(texel)
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
        // Wireframe debug mode: replace the filled fill with three
        // edge lines. Useful for visualising the geometry a game
        // actually submits, independent of shading / texturing —
        // including over-sized triangles that would normally be
        // dropped by the hardware extent check below.
        if self.wireframe_enabled {
            self.rasterize_line(v0, v1, color, color, mode, false);
            self.rasterize_line(v1, v2, color, color, mode, false);
            self.rasterize_line(v2, v0, color, color, mode, false);
            return;
        }
        if triangle_exceeds_hw_extent(v0, v1, v2) {
            return;
        }
        // Flat triangles don't need Gouraud or UV interpolation, but
        // the scanline walk itself is identical — reuse `setup_sections`
        // with zeroed colour/UV and just plot `color` for every pixel
        // between leftX and rightX per scanline.
        let Some(mut setup) = setup_sections(
            [v0.0, v1.0, v2.0],
            [v0.1, v1.1, v2.1],
            [(0, 0, 0); 3],
            [(0, 0); 3],
        ) else {
            return;
        };

        let draw_top = self.draw_area_top as i32;
        let draw_bottom = self.draw_area_bottom as i32;
        let draw_left = self.draw_area_left as i32;
        let draw_right = self.draw_area_right as i32;
        let mut y = setup.y_min;
        while y < draw_top {
            if setup.next_row().is_err() {
                return;
            }
            y += 1;
        }
        let y_max = setup.y_max.min(draw_bottom);
        while y <= y_max {
            let xmin = (setup.left_x >> 16) as i32;
            let xmax = ((setup.right_x >> 16) as i32 - 1).min(draw_right);
            let xmin_clipped = xmin.max(draw_left);
            let mut j = xmin_clipped;
            while j <= xmax {
                self.plot_pixel(j as u16, y as u16, color, mode);
                j += 1;
            }
            if setup.next_row().is_err() {
                return;
            }
            y += 1;
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
        // Raw-texture flag is bit 0 of the *opcode byte* (bit 24 of
        // the full cmd word), per PSX-SPX. Testing `cmd & 1` reads
        // bit 0 of the R channel of the embedded colour instead —
        // so odd-R tints like 0xFF would be mis-flagged raw.
        let tint = if (cmd >> 24) & 1 != 0 {
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

    /// GP0 0x2C..=0x2F — textured quad. 9 words; split in Redux order.
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
        // Raw-texture flag is bit 0 of the *opcode byte* (bit 24 of
        // the full cmd word), per PSX-SPX. Testing `cmd & 1` reads
        // bit 0 of the R channel of the embedded colour instead —
        // so odd-R tints like 0xFF would be mis-flagged raw.
        let tint = if (cmd >> 24) & 1 != 0 {
            RAW_TEXTURE_TINT
        } else {
            split_tint(cmd & 0x00FF_FFFF)
        };
        if self.rasterize_axis_aligned_textured_quad(
            v0, v1, v2, v3, t0, t1, t2, t3, clut_word, semi, tint,
        ) {
            return;
        }
        self.rasterize_textured_triangle(v1, v3, v2, t1, t3, t2, clut_word, semi, tint);
        self.rasterize_textured_triangle(v0, v1, v2, t0, t1, t2, clut_word, semi, tint);
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
        // See comment on the other textured primitives: raw-texture
        // flag is bit 0 of the opcode byte (= bit 24 of the cmd word),
        // not bit 0 of the full cmd word.
        let raw = (cmd >> 24) & 1 != 0;
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
    /// split in Redux order.
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
        // See comment on the other textured primitives: raw-texture
        // flag is bit 0 of the opcode byte (= bit 24 of the cmd word),
        // not bit 0 of the full cmd word.
        let raw = (cmd >> 24) & 1 != 0;
        self.rasterize_textured_shaded_triangle(
            v1, v3, v2, t1, t3, t2, c1, c3, c2, clut_word, semi, raw,
        );
        self.rasterize_textured_shaded_triangle(
            v0, v1, v2, t0, t1, t2, c0, c1, c2, clut_word, semi, raw,
        );
    }

    /// Fast path for the common 2D-sprite case: a flat textured quad
    /// whose vertex order is top-left, top-right, bottom-left,
    /// bottom-right. Redux renders flat textured quads with a true
    /// four-edge scanline walker, not by splitting the primitive into
    /// two triangles; using the same row-wide UV interpolation removes
    /// diagonal sampling seams in BIOS text and loading-screen sprites.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_axis_aligned_textured_quad(
        &mut self,
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        v3: (i32, i32),
        t0: (u16, u16),
        t1: (u16, u16),
        t2: (u16, u16),
        t3: (u16, u16),
        clut_word: u16,
        semi_trans: bool,
        tint: (u32, u32, u32),
    ) -> bool {
        if self.wireframe_enabled {
            return false;
        }
        if v0.1 != v1.1 || v2.1 != v3.1 || v0.0 != v2.0 || v1.0 != v3.0 {
            return false;
        }
        if triangle_exceeds_hw_extent(v1, v3, v2) || triangle_exceeds_hw_extent(v0, v1, v2) {
            return true;
        }
        let left = v0.0;
        let right = v1.0;
        let top = v0.1;
        let bottom = v2.1;
        let width = right - left;
        let height = bottom - top;
        if width <= 0 || height <= 0 {
            return true;
        }

        let draw_left = self.draw_area_left as i32;
        let draw_right = self.draw_area_right as i32;
        let draw_top = self.draw_area_top as i32;
        let draw_bottom = self.draw_area_bottom as i32;
        let y_start = top.max(draw_top);
        let y_end = (bottom - 1).min(draw_bottom);
        let x_start = left.max(draw_left);
        let x_end = (right - 1).min(draw_right);
        if y_start > y_end || x_start > x_end {
            return true;
        }

        let clut_x = (clut_word & 0x3F) * 16;
        let clut_y = (clut_word >> 6) & 0x1FF;
        let tpage_mode = self.tex_blend_mode;
        let left_u0 = (t0.0 as i64) << 16;
        let left_v0 = (t0.1 as i64) << 16;
        let right_u0 = (t1.0 as i64) << 16;
        let right_v0 = (t1.1 as i64) << 16;
        let delta_left_u = (((t2.0 as i64) << 16) - left_u0) / height as i64;
        let delta_left_v = (((t2.1 as i64) << 16) - left_v0) / height as i64;
        let delta_right_u = (((t3.0 as i64) << 16) - right_u0) / height as i64;
        let delta_right_v = (((t3.1 as i64) << 16) - right_v0) / height as i64;

        for py in y_start..=y_end {
            let row = (py - top) as i64;
            let mut pos_u = left_u0 + row * delta_left_u;
            let mut pos_v = left_v0 + row * delta_left_v;
            let right_u = right_u0 + row * delta_right_u;
            let right_v = right_v0 + row * delta_right_v;
            let delta_u = (right_u - pos_u) / width as i64;
            let delta_v = (right_v - pos_v) / width as i64;
            if x_start > left {
                let skip = (x_start - left) as i64;
                pos_u += skip * delta_u;
                pos_v += skip * delta_v;
            }
            for px in x_start..=x_end {
                let u = (pos_u >> 16) as u16;
                let v = (pos_v >> 16) as u16;
                if let Some(texel) = self.sample_texture(u, v, clut_x, clut_y) {
                    let shaded = modulate_tint(texel, tint.0, tint.1, tint.2);
                    let mode = if semi_trans && (texel & 0x8000) != 0 {
                        tpage_mode
                    } else {
                        BlendMode::Opaque
                    };
                    self.plot_pixel(px as u16, py as u16, shaded, mode);
                }
                pos_u += delta_u;
                pos_v += delta_v;
            }
        }
        true
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
        if self.wireframe_enabled {
            self.rasterize_line_shaded(v0, v1, c0, c1, BlendMode::Opaque);
            self.rasterize_line_shaded(v1, v2, c1, c2, BlendMode::Opaque);
            self.rasterize_line_shaded(v2, v0, c2, c0, BlendMode::Opaque);
            // Silence unused-var warnings for the texture args we
            // intentionally drop in wireframe mode.
            let _ = (t0, t1, t2, clut_word, semi_trans, raw_texture);
            return;
        }
        if triangle_exceeds_hw_extent(v0, v1, v2) {
            return;
        }
        let clut_x = (clut_word & 0x3F) * 16;
        let clut_y = (clut_word >> 6) & 0x1FF;
        let tpage_mode = self.tex_blend_mode;

        let r = |c: u32| (c & 0xFF) as i32;
        let g = |c: u32| ((c >> 8) & 0xFF) as i32;
        let b = |c: u32| ((c >> 16) & 0xFF) as i32;
        let v_rgb = [
            (r(c0), g(c0), b(c0)),
            (r(c1), g(c1), b(c1)),
            (r(c2), g(c2), b(c2)),
        ];
        let v_uv = [
            (t0.0 as i32, t0.1 as i32),
            (t1.0 as i32, t1.1 as i32),
            (t2.0 as i32, t2.1 as i32),
        ];
        let Some(mut setup) = setup_sections([v0.0, v1.0, v2.0], [v0.1, v1.1, v2.1], v_rgb, v_uv)
        else {
            return;
        };

        let dif_r = setup.delta_col_r;
        let dif_g = setup.delta_col_g;
        let dif_b = setup.delta_col_b;
        let dif_u = setup.delta_col_u;
        let dif_v = setup.delta_col_v;

        let draw_top = self.draw_area_top as i32;
        let draw_bottom = self.draw_area_bottom as i32;
        let draw_left = self.draw_area_left as i32;
        let draw_right = self.draw_area_right as i32;
        let mut y = setup.y_min;
        while y < draw_top {
            if setup.next_row().is_err() {
                return;
            }
            y += 1;
        }
        let y_max = setup.y_max.min(draw_bottom);

        while y <= y_max {
            let xmin = (setup.left_x >> 16) as i32;
            let xmax_raw = (setup.right_x >> 16) as i32 - 1;
            let xmax = xmax_raw.min(draw_right);
            if xmax >= xmin {
                let mut c_r = setup.left_r;
                let mut c_g = setup.left_g;
                let mut c_b = setup.left_b;
                let mut c_u = setup.left_u;
                let mut c_v = setup.left_v;
                let xmin_clipped = if xmin < draw_left {
                    let skip = (draw_left - xmin) as i64;
                    c_r += skip * dif_r;
                    c_g += skip * dif_g;
                    c_b += skip * dif_b;
                    c_u += skip * dif_u;
                    c_v += skip * dif_v;
                    draw_left
                } else {
                    xmin
                };
                let mut j = xmin_clipped;
                while j <= xmax {
                    let u = (c_u >> 16) as u16;
                    let v = (c_v >> 16) as u16;
                    if let Some(texel) = self.sample_texture(u, v, clut_x, clut_y) {
                        let (tint_r, tint_g, tint_b) = if raw_texture {
                            RAW_TEXTURE_TINT
                        } else {
                            (
                                ((c_r >> 16).clamp(0, 255)) as u32,
                                ((c_g >> 16).clamp(0, 255)) as u32,
                                ((c_b >> 16).clamp(0, 255)) as u32,
                            )
                        };
                        let shaded = if !raw_texture && self.dither_enabled {
                            modulate_tint_dithered(texel, tint_r, tint_g, tint_b, j, y)
                        } else {
                            modulate_tint(texel, tint_r, tint_g, tint_b)
                        };
                        let mode = if semi_trans && (texel & 0x8000) != 0 {
                            tpage_mode
                        } else {
                            BlendMode::Opaque
                        };
                        self.plot_pixel(j as u16, y as u16, shaded, mode);
                    }
                    c_r += dif_r;
                    c_g += dif_g;
                    c_b += dif_b;
                    c_u += dif_u;
                    c_v += dif_v;
                    j += 1;
                }
            }
            if setup.next_row().is_err() {
                return;
            }
            y += 1;
        }
    }

    /// Apply the tpage bits embedded in a textured-primitive UV word
    /// (they override the draw-mode tpage for this primitive onward).
    ///
    /// Mirrors Redux's poly-path behaviour (`gpu.cc:347/396`):
    /// before calling `texturePage`, the drawPoly handler OR-s bit 9
    /// of the incoming tpage with the current `m_ditherMode`. That
    /// gives a primitive the power to *enable* dither but not to
    /// disable it — in effect, dither is sticky across primitives
    /// until an explicit GP0 0xE1 turns it off. Then `texturePage`
    /// copies bits 0..=10 of the (possibly OR-fixed) tpage into
    /// `m_statusRet`.
    ///
    /// Missing this sync surfaced at parity step 60,041,097 as a
    /// GPUSTAT load that differed in the low byte: our GPUSTAT
    /// kept reflecting the last E1's tpage-X even after a textured
    /// polygon's embedded tpage re-pointed it.
    fn apply_primitive_tpage(&mut self, uv_word: u32) {
        let mut tpage = (uv_word >> 16) & 0xFFFF;
        if self.dither_enabled {
            tpage |= 0x200;
        }
        self.tex_page_x = ((tpage & 0x0F) as u16) * 64;
        self.tex_page_y = if (tpage >> 4) & 1 != 0 { 256 } else { 0 };
        self.tex_depth = ((tpage >> 7) & 0x3) as u8;
        self.tex_blend_mode = BlendMode::from_tpage_bits(tpage >> 5);
        self.dither_enabled = (tpage >> 9) & 1 != 0;
        let stat_bits = (tpage & 0x07FF) as u32;
        self.status.raw = (self.status.raw & !0x07FF) | stat_bits;
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
        // Stamp ownership for the pixel tracer if enabled. We hit
        // this every time a primitive writes a pixel, but the cost
        // is a single array write behind an Option check — cheap
        // enough to keep on even in release diagnostic builds.
        if let Some(ref mut owner) = self.pixel_owner {
            owner[y as usize * VRAM_WIDTH + x as usize] = self.current_cmd_index;
        }
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
        if self.wireframe_enabled {
            // Wireframe uses the first tint channel triple directly
            // for the outline colour (or white for raw-texture prims).
            let edge_rgb = (tint.0 | (tint.1 << 8) | (tint.2 << 16)) as u32;
            let colour = rgb24_to_bgr15(edge_rgb);
            self.rasterize_line(v0, v1, colour, colour, BlendMode::Opaque, false);
            self.rasterize_line(v1, v2, colour, colour, BlendMode::Opaque, false);
            self.rasterize_line(v2, v0, colour, colour, BlendMode::Opaque, false);
            let _ = (t0, t1, t2, clut_word, semi_trans);
            return;
        }
        if triangle_exceeds_hw_extent(v0, v1, v2) {
            return;
        }
        let clut_x = (clut_word & 0x3F) * 16;
        let clut_y = (clut_word >> 6) & 0x1FF;
        let tpage_mode = self.tex_blend_mode;

        let v_uv = [
            (t0.0 as i32, t0.1 as i32),
            (t1.0 as i32, t1.1 as i32),
            (t2.0 as i32, t2.1 as i32),
        ];
        let Some(mut setup) =
            setup_sections([v0.0, v1.0, v2.0], [v0.1, v1.1, v2.1], [(0, 0, 0); 3], v_uv)
        else {
            return;
        };

        let dif_u = setup.delta_col_u;
        let dif_v = setup.delta_col_v;

        let draw_top = self.draw_area_top as i32;
        let draw_bottom = self.draw_area_bottom as i32;
        let draw_left = self.draw_area_left as i32;
        let draw_right = self.draw_area_right as i32;
        let mut y = setup.y_min;
        while y < draw_top {
            if setup.next_row().is_err() {
                return;
            }
            y += 1;
        }
        let y_max = setup.y_max.min(draw_bottom);

        while y <= y_max {
            let xmin = (setup.left_x >> 16) as i32;
            let xmax = ((setup.right_x >> 16) as i32 - 1).min(draw_right);
            if xmax >= xmin {
                let mut c_u = setup.left_u;
                let mut c_v = setup.left_v;
                let xmin_clipped = if xmin < draw_left {
                    let skip = (draw_left - xmin) as i64;
                    c_u += skip * dif_u;
                    c_v += skip * dif_v;
                    draw_left
                } else {
                    xmin
                };
                let mut j = xmin_clipped;
                while j <= xmax {
                    let u = (c_u >> 16) as u16;
                    let v = (c_v >> 16) as u16;
                    if let Some(texel) = self.sample_texture(u, v, clut_x, clut_y) {
                        let shaded = modulate_tint(texel, tint.0, tint.1, tint.2);
                        let mode = if semi_trans && (texel & 0x8000) != 0 {
                            tpage_mode
                        } else {
                            BlendMode::Opaque
                        };
                        self.plot_pixel(j as u16, y as u16, shaded, mode);
                    }
                    c_u += dif_u;
                    c_v += dif_v;
                    j += 1;
                }
            }
            if setup.next_row().is_err() {
                return;
            }
            y += 1;
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
        if self.wireframe_enabled {
            self.rasterize_line_shaded(v0, v1, c0, c1, mode);
            self.rasterize_line_shaded(v1, v2, c1, c2, mode);
            self.rasterize_line_shaded(v2, v0, c2, c0, mode);
            return;
        }
        if triangle_exceeds_hw_extent(v0, v1, v2) {
            return;
        }
        let min_x = v0.0.min(v1.0).min(v2.0).max(self.draw_area_left as i32);
        let max_x = v0.0.max(v1.0).max(v2.0).min(self.draw_area_right as i32);
        let min_y = v0.1.min(v1.1).min(v2.1).max(self.draw_area_top as i32);
        let max_y = v0.1.max(v1.1).max(v2.1).min(self.draw_area_bottom as i32);
        if min_x > max_x || min_y > max_y {
            return;
        }

        // Channel-extract closures — r/g/b are low/mid/high bytes of the
        // 24-bit word written in the command.
        let r = |c: u32| (c & 0xFF) as i32;
        let g = |c: u32| ((c >> 8) & 0xFF) as i32;
        let b = |c: u32| ((c >> 16) & 0xFF) as i32;
        let v_rgb = [
            (r(c0), g(c0), b(c0)),
            (r(c1), g(c1), b(c1)),
            (r(c2), g(c2), b(c2)),
        ];
        let Some(mut setup) =
            setup_sections([v0.0, v1.0, v2.0], [v0.1, v1.1, v2.1], v_rgb, [(0, 0); 3])
        else {
            return;
        };

        // Per-column (per-pixel horizontal) deltas — computed once at
        // setup, applied on every step within a scanline.
        let dif_r = setup.delta_col_r;
        let dif_g = setup.delta_col_g;
        let dif_b = setup.delta_col_b;

        // Clip the top of the triangle against the draw-area top: step
        // the sections down until `y_min` reaches `draw_area_top`. If we
        // exit early, the triangle is entirely above the drawable region.
        let draw_top = self.draw_area_top as i32;
        let draw_bottom = self.draw_area_bottom as i32;
        let draw_left = self.draw_area_left as i32;
        let draw_right = self.draw_area_right as i32;
        let mut y = setup.y_min;
        while y < draw_top {
            if setup.next_row().is_err() {
                return;
            }
            y += 1;
        }
        let y_max = setup.y_max.min(draw_bottom);

        while y <= y_max {
            let xmin = (setup.left_x >> 16) as i32;
            let xmax_raw = (setup.right_x >> 16) as i32 - 1;
            let xmax = xmax_raw.min(draw_right);
            if xmax >= xmin {
                // Starting attributes at the left edge of this scanline.
                let mut c_r = setup.left_r;
                let mut c_g = setup.left_g;
                let mut c_b = setup.left_b;
                // Clip left to draw_area_left: step the per-column deltas
                // forward by `(draw_left - xmin)` pixels to skip the
                // hidden left portion.
                let xmin_clipped = if xmin < draw_left {
                    let skip = (draw_left - xmin) as i64;
                    c_r += skip * dif_r;
                    c_g += skip * dif_g;
                    c_b += skip * dif_b;
                    draw_left
                } else {
                    xmin
                };
                let mut j = xmin_clipped;
                while j <= xmax {
                    // Redux packs the 8-bit channels into the top byte
                    // of each Q16.16 accumulator — recover them with
                    // `>> 16`. Clamp isn't necessary for well-formed
                    // triangles (vertex colours are 0..=255 and the
                    // per-column deltas keep them in range), but we
                    // clamp to match hardware's saturation anyway.
                    let ri = (c_r >> 16) as i32;
                    let gi = (c_g >> 16) as i32;
                    let bi = (c_b >> 16) as i32;
                    let colour = if self.dither_enabled {
                        dither_rgb(ri, gi, bi, j, y)
                    } else {
                        rgb24_to_bgr15(
                            (ri.clamp(0, 255) as u32)
                                | ((gi.clamp(0, 255) as u32) << 8)
                                | ((bi.clamp(0, 255) as u32) << 16),
                        )
                    };
                    self.plot_pixel(j as u16, y as u16, colour, mode);
                    c_r += dif_r;
                    c_g += dif_g;
                    c_b += dif_b;
                    j += 1;
                }
            }
            if setup.next_row().is_err() {
                return;
            }
            y += 1;
        }
    }

    // --- Lines (GP0 0x40..=0x5F) ---

    /// GP0 0x40..=0x43 — single monochrome line. Packet: `[cmd+color, v0, v1]`.
    fn draw_line_mono_single(&mut self) {
        let cmd = self.gp0_fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let v1 = self.decode_vertex(self.gp0_fifo[2]);
        self.rasterize_line(v0, v1, color, color, mode, false);
    }

    /// GP0 0x50..=0x53 — single Gouraud-shaded line. Packet:
    /// `[cmd+c0, v0, c1, v1]` — each endpoint carries its own colour
    /// word.
    fn draw_line_shaded_single(&mut self) {
        let cmd = self.gp0_fifo[0];
        let c0 = cmd & 0x00FF_FFFF;
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let c1 = self.gp0_fifo[2] & 0x00FF_FFFF;
        let v1 = self.decode_vertex(self.gp0_fifo[3]);
        self.rasterize_line_shaded(v0, v1, c0, c1, mode);
    }

    /// GP0 0x48..=0x4B — start a monochrome polyline. The initial
    /// packet has the same shape as a single line (cmd+color, v0,
    /// v1); after executing it we switch to receive mode.
    fn draw_line_mono_start_polyline(&mut self) {
        let cmd = self.gp0_fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let v1 = self.decode_vertex(self.gp0_fifo[2]);
        self.rasterize_line(v0, v1, color, color, mode, false);
        // Enter receive mode with `v1` as the starting point for
        // the next segment.
        self.polyline = Some(PolylineState::Mono {
            color,
            mode,
            last_vertex: v1,
        });
    }

    /// GP0 0x58..=0x5B — start a Gouraud polyline. Initial packet
    /// is `[cmd+c0, v0, c1, v1]`; after the first segment we
    /// enter receive mode waiting for alternating (color, vertex)
    /// pairs.
    fn draw_line_shaded_start_polyline(&mut self) {
        let cmd = self.gp0_fifo[0];
        let c0 = cmd & 0x00FF_FFFF;
        let mode = prim_blend_mode(cmd, self.tex_blend_mode);
        let v0 = self.decode_vertex(self.gp0_fifo[1]);
        let c1 = self.gp0_fifo[2] & 0x00FF_FFFF;
        let v1 = self.decode_vertex(self.gp0_fifo[3]);
        self.rasterize_line_shaded(v0, v1, c0, c1, mode);
        self.polyline = Some(PolylineState::Shaded {
            mode,
            last_color: c1,
            last_vertex: v1,
            awaiting_color: true,
            pending_color: 0,
        });
    }

    /// Consume one GP0 word while in polyline mode. Terminator
    /// pattern per PSX-SPX is `0x50005000` / `0x55555555` — any
    /// word whose top bits match `0x5000_5000 >> 28 == 0x5` in
    /// both high and low halves means "end". We accept the
    /// canonical sentinels.
    fn ingest_polyline_word(&mut self, word: u32) {
        // Sentinel check — both halves have the terminator pattern.
        // Redux uses `(word & 0xF000F000) == 0x50005000`.
        let is_term = (word & 0xF000_F000) == 0x5000_5000;
        if is_term {
            self.polyline = None;
            return;
        }
        match self.polyline.as_mut().unwrap() {
            PolylineState::Mono {
                color,
                mode,
                last_vertex,
            } => {
                let c = *color;
                let m = *mode;
                let v0 = *last_vertex;
                let v1 = self.decode_vertex(word);
                self.rasterize_line(v0, v1, c, c, m, false);
                if let Some(PolylineState::Mono { last_vertex, .. }) = self.polyline.as_mut() {
                    *last_vertex = v1;
                }
            }
            PolylineState::Shaded {
                mode,
                last_color,
                last_vertex,
                awaiting_color,
                pending_color,
            } => {
                if *awaiting_color {
                    *pending_color = word & 0x00FF_FFFF;
                    *awaiting_color = false;
                } else {
                    let c0 = *last_color;
                    let c1 = *pending_color;
                    let m = *mode;
                    let v0 = *last_vertex;
                    let v1 = self.decode_vertex(word);
                    self.rasterize_line_shaded(v0, v1, c0, c1, m);
                    if let Some(PolylineState::Shaded {
                        last_color,
                        last_vertex,
                        awaiting_color,
                        ..
                    }) = self.polyline.as_mut()
                    {
                        *last_color = c1;
                        *last_vertex = v1;
                        *awaiting_color = true;
                    }
                }
            }
        }
    }

    /// Rasterize a line from `v0` to `v1` using Bresenham's
    /// algorithm. Clips to the draw area and respects mask-bit
    /// flags. `c0` and `c1` are the same for monochrome lines;
    /// Gouraud lines use [`Gpu::rasterize_line_shaded`] which
    /// interpolates per pixel.
    ///
    /// `_interpolate` is reserved for future shaded mode but kept
    /// here so the signature is stable.
    fn rasterize_line(
        &mut self,
        v0: (i32, i32),
        v1: (i32, i32),
        c0: u16,
        _c1: u16,
        mode: BlendMode,
        _interpolate: bool,
    ) {
        let (mut x, mut y) = v0;
        let (x1, y1) = v1;
        let dx = (x1 - x).abs();
        let dy = -(y1 - y).abs();
        let sx: i32 = if x < x1 { 1 } else { -1 };
        let sy: i32 = if y < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (min_x, max_x) = (self.draw_area_left as i32, self.draw_area_right as i32);
        let (min_y, max_y) = (self.draw_area_top as i32, self.draw_area_bottom as i32);
        loop {
            if (min_x..=max_x).contains(&x) && (min_y..=max_y).contains(&y) {
                self.plot_pixel(x as u16, y as u16, c0, mode);
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Rasterize a Gouraud-shaded line — interpolates RGB between
    /// `c0` and `c1` linearly in screen space. Uses the same
    /// Bresenham walk as the mono path; the colour parameter is
    /// re-evaluated each step from a normalised distance-along-
    /// line metric.
    fn rasterize_line_shaded(
        &mut self,
        v0: (i32, i32),
        v1: (i32, i32),
        c0: u32,
        c1: u32,
        mode: BlendMode,
    ) {
        let (mut x, mut y) = v0;
        let (x1, y1) = v1;
        let dx_abs = (x1 - x).abs();
        let dy_abs = (y1 - y).abs();
        let steps = dx_abs.max(dy_abs).max(1);
        let r0 = (c0 & 0xFF) as i32;
        let g0 = ((c0 >> 8) & 0xFF) as i32;
        let b0 = ((c0 >> 16) & 0xFF) as i32;
        let r1 = (c1 & 0xFF) as i32;
        let g1 = ((c1 >> 8) & 0xFF) as i32;
        let b1 = ((c1 >> 16) & 0xFF) as i32;
        let dx = (x1 - x).abs();
        let dy = -(y1 - y).abs();
        let sx: i32 = if x < x1 { 1 } else { -1 };
        let sy: i32 = if y < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (min_x, max_x) = (self.draw_area_left as i32, self.draw_area_right as i32);
        let (min_y, max_y) = (self.draw_area_top as i32, self.draw_area_bottom as i32);
        let mut step = 0i32;
        loop {
            if (min_x..=max_x).contains(&x) && (min_y..=max_y).contains(&y) {
                // Linear interpolate each channel.
                let r = r0 + ((r1 - r0) * step) / steps;
                let g = g0 + ((g1 - g0) * step) / steps;
                let b = b0 + ((b1 - b0) * step) / steps;
                let colour = if self.dither_enabled {
                    dither_rgb(r, g, b, x, y)
                } else {
                    rgb24_to_bgr15(
                        (r.clamp(0, 255) as u32)
                            | ((g.clamp(0, 255) as u32) << 8)
                            | ((b.clamp(0, 255) as u32) << 16),
                    )
                };
                self.plot_pixel(x as u16, y as u16, colour, mode);
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
            step += 1;
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
        // Fill rect costs ~0.5 cycles / pixel on hardware.
        self.charge_busy(((w as u64) * (h as u64)) / 2);
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
        // Single lines: 3 words (color + 2 vertices) for monochrome,
        // 4 words for shaded.
        0x40..=0x43 => 3,
        0x50..=0x53 => 4,
        // Polyline starts: same initial shape, but after the first
        // endpoint the FIFO enters a streaming receive mode until
        // the terminator sentinel is seen.
        0x48..=0x4B => 3,
        0x58..=0x5B => 4,
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

// ======================================================================
// Scanline-delta triangle rasterizer
// ======================================================================
//
// Direct port of Redux's `drawPoly3Gi` / `drawPoly3TGEx8i` family from
// `pcsx-redux/src/gpu/soft/soft.cc`. The scanline-delta approach is what
// PSX hardware actually does: sort vertices by Y, walk each of the three
// edges as scanline-advancing sections, and for each scanline plot pixels
// from `leftX` to `rightX - 1` inclusive with attribute interpolation
// driven by precomputed per-pixel (column) deltas.
//
// Why this beats barycentric division-per-pixel: rounding. Two equivalent
// formulas for interpolated colour or UV produce subtly different integer
// results at edge-adjacent pixels. Matching Redux's exact algorithm
// (including its fixed-point shift sizes) is the only way to hit pixel-
// exact parity on game content; our old barycentric path was 14% off on
// the Crash title screen even with identical primitive inputs.
//
// Naming convention: Redux's `m_deltaRightR` / `m_deltaRightU` etc are
// actually **per-column** (per-X) deltas despite being named "right".
// This port keeps the Redux names so a side-by-side diff with
// `soft.cc` stays readable — even where "right" looks wrong in isolation.
//
// Fixed-point layout: X / U / V are Q16.16. Colour channels are stored
// in bits 16..23 of the state (Q16.8 relative to the 8-bit vertex
// colours, i.e. the fraction lives in bits 0..15).

/// One vertex as seen by the scanline setup. All attributes — position,
/// colour, UV — are already shifted into the fixed-point domain the
/// rasterizer uses. `y` stays as a plain integer because scanlines step
/// by 1 on hardware; only horizontal attributes need sub-pixel precision.
#[derive(Copy, Clone, Debug, Default)]
struct SlVertex {
    /// X coordinate in Q16.16 fixed-point.
    x: i64,
    /// Y coordinate as a plain pixel integer.
    y: i32,
    /// Red channel in Q16.16 per-8bit: `vtx_r << 16`. Shifted the same
    /// way Redux does so its `v->R = rgb & 0x00ff0000` constant stays
    /// structurally identical.
    r: i64,
    g: i64,
    b: i64,
    /// U coordinate, Q16.16.
    u: i64,
    /// V coordinate, Q16.16.
    v: i64,
}

/// Per-triangle scanline walk state. Built once by `setup_sections_*`,
/// then `next_row_*` advances it by one scanline between rasterizer
/// iterations.
///
/// Redux uses two *arrays of 3 pointers* (one for the left walk, one
/// for the right) plus a section index that decrements on pop. A
/// single-edge side (the long v1→v3) has 2 entries (section count 1);
/// a two-edge side (v1→v2 + v2→v3) has 3 entries (section count 2).
/// We mirror that exactly.
#[derive(Clone, Debug)]
struct SlTriSetup {
    /// Sorted vertices: `[v1 (top), v2 (middle), v3 (bottom)]`, by Y
    /// ascending. Stored as owned values because we shuffle pointers
    /// into the left/right arrays below.
    sorted: [SlVertex; 3],
    /// Left-edge walk: `[bottom, maybe_middle, top]` — `left_section`
    /// indexes the highest-unvisited entry; section descends toward 0.
    left_array: [usize; 3],
    right_array: [usize; 3],
    /// Number of edge segments remaining on each side. Starts at 1
    /// (single-edge long side) or 2 (pivot-at-middle), decrements in
    /// `next_row_*` when a section exhausts.
    left_section: i32,
    right_section: i32,

    // --- Current scanline state (updated every row). ---
    /// Left X on the current scanline, Q16.16.
    left_x: i64,
    /// Right X on the current scanline, Q16.16. Redux stores this pre-
    /// shifted by 16; the rasterizer reads `rightX >> 16 - 1` as the
    /// inclusive right edge.
    right_x: i64,
    /// Pre-step-to-add for left_x each scanline (Q16.16). Changes
    /// whenever the left section pops.
    delta_left_x: i64,
    delta_right_x: i64,
    /// Rows remaining in the currently-active section. Hits zero →
    /// pop to the next section, recompute deltas.
    left_section_height: i32,
    right_section_height: i32,

    // --- Gouraud colour at left edge, current scanline (Q16.16). ---
    left_r: i64,
    left_g: i64,
    left_b: i64,
    delta_left_r: i64,
    delta_left_g: i64,
    delta_left_b: i64,

    // --- UV at left edge, current scanline (Q16.16). ---
    left_u: i64,
    left_v: i64,
    delta_left_u: i64,
    delta_left_v: i64,

    // --- Per-column (per-X) deltas, computed once at setup time. ---
    //
    // Named "delta_right_*" to match Redux's `m_deltaRightR` / `m_deltaRightU`
    // (which are also mis-named — they're per-column, not per-edge).
    delta_col_r: i64,
    delta_col_g: i64,
    delta_col_b: i64,
    delta_col_u: i64,
    delta_col_v: i64,

    // --- Scanline bounds ---
    y_min: i32,
    y_max: i32,
}

impl SlTriSetup {
    /// Pop the active left section to the next shorter one. Returns
    /// `Err` when the pop runs out of sections (signals the triangle
    /// walk is done).
    fn pop_left_section(&mut self) -> Result<(), ()> {
        self.left_section -= 1;
        if self.left_section <= 0 {
            return Err(());
        }
        self.compute_left_section()
    }

    /// Pop the active right section; same contract as `pop_left_section`.
    fn pop_right_section(&mut self) -> Result<(), ()> {
        self.right_section -= 1;
        if self.right_section <= 0 {
            return Err(());
        }
        self.compute_right_section()
    }

    /// Recompute `left_x` / `delta_left_x` / colour + UV start + deltas
    /// from the pair of vertices defining the currently-active left
    /// section.
    fn compute_left_section(&mut self) -> Result<(), ()> {
        let idx1 = self.left_array[self.left_section as usize];
        let idx2 = self.left_array[(self.left_section - 1) as usize];
        let v1 = self.sorted[idx1];
        let v2 = self.sorted[idx2];
        let height = v2.y - v1.y;
        if height == 0 {
            return Err(());
        }
        let h = height as i64;
        self.delta_left_x = (v2.x - v1.x) / h;
        self.left_x = v1.x;
        // Gouraud + UV tracking are only meaningful when the caller
        // populated them; zero-divides can't happen because h != 0.
        self.delta_left_r = (v2.r - v1.r) / h;
        self.delta_left_g = (v2.g - v1.g) / h;
        self.delta_left_b = (v2.b - v1.b) / h;
        self.left_r = v1.r;
        self.left_g = v1.g;
        self.left_b = v1.b;
        self.delta_left_u = (v2.u - v1.u) / h;
        self.delta_left_v = (v2.v - v1.v) / h;
        self.left_u = v1.u;
        self.left_v = v1.v;
        self.left_section_height = height;
        Ok(())
    }

    fn compute_right_section(&mut self) -> Result<(), ()> {
        let idx1 = self.right_array[self.right_section as usize];
        let idx2 = self.right_array[(self.right_section - 1) as usize];
        let v1 = self.sorted[idx1];
        let v2 = self.sorted[idx2];
        let height = v2.y - v1.y;
        if height == 0 {
            return Err(());
        }
        let h = height as i64;
        self.delta_right_x = (v2.x - v1.x) / h;
        self.right_x = v1.x;
        self.right_section_height = height;
        Ok(())
    }

    /// Advance one scanline. Returns `Err` when the triangle's bottom
    /// edge is past.
    fn next_row(&mut self) -> Result<(), ()> {
        self.left_section_height -= 1;
        if self.left_section_height <= 0 {
            self.pop_left_section()?;
        } else {
            self.left_x += self.delta_left_x;
            self.left_r += self.delta_left_r;
            self.left_g += self.delta_left_g;
            self.left_b += self.delta_left_b;
            self.left_u += self.delta_left_u;
            self.left_v += self.delta_left_v;
        }
        self.right_section_height -= 1;
        if self.right_section_height <= 0 {
            self.pop_right_section()?;
        } else {
            self.right_x += self.delta_right_x;
        }
        Ok(())
    }
}

/// `(x << 10) / y` with i64 intermediate, matching Redux's
/// `shl10idiv` helper at `soft.h:276`.
fn shl10_idiv(x: i64, y: i64) -> i64 {
    (x << 10) / y
}

/// Core setup: sort 3 vertices by Y, pick which side has the pivot
/// Hardware extent rule: any triangle whose vertex pairs span more
/// than 1023 pixels horizontally or 511 vertically is silently
/// dropped on real PS1 hardware. Off-screen geometry coming out of
/// projection lands here constantly — without this gate it
/// rasterises as a giant garbage smear instead of being culled.
///
/// The check is per-edge, not bounding-box: hardware compares each
/// pair of vertices independently. Quads are already split into
/// two triangles by the caller, so each half is gated separately —
/// matching hardware behaviour where one half of a quad can survive
/// while the other gets dropped.
fn triangle_exceeds_hw_extent(v0: (i32, i32), v1: (i32, i32), v2: (i32, i32)) -> bool {
    const MAX_DX: i32 = 1023;
    const MAX_DY: i32 = 511;
    let edges = [(v0, v1), (v1, v2), (v2, v0)];
    edges
        .iter()
        .any(|(a, b)| (a.0 - b.0).abs() > MAX_DX || (a.1 - b.1).abs() > MAX_DY)
}

/// (the "middle" vertex v2), and seed left / right walks. Colour + UV
/// are optional — pass zeros for the ones a particular primitive
/// doesn't use. Returns the setup ready for the scanline loop, or
/// `None` when the triangle has zero height or zero "longest" width
/// (both degenerate).
///
/// "longest" meaning: Redux computes the signed horizontal distance
/// from v2.x to where the long v1→v3 edge crosses y=v2.y. Positive →
/// the long edge is to the RIGHT of v2 → v1 is on the left side and
/// the two-edge walk lives on the left. Negative → the inverse.
fn setup_sections(
    v_x: [i32; 3],
    v_y: [i32; 3],
    v_rgb: [(i32, i32, i32); 3],
    v_uv: [(i32, i32); 3],
) -> Option<SlTriSetup> {
    // Build unsorted vertex structs. X/U/V are shifted into Q16.16
    // up front; colour channels get `<< 16` to match Redux's
    // `v->R = rgb & 0x00ff0000` convention.
    let mut verts = [SlVertex::default(); 3];
    for i in 0..3 {
        verts[i] = SlVertex {
            x: (v_x[i] as i64) << 16,
            y: v_y[i],
            r: (v_rgb[i].0 as i64) << 16,
            g: (v_rgb[i].1 as i64) << 16,
            b: (v_rgb[i].2 as i64) << 16,
            u: (v_uv[i].0 as i64) << 16,
            v: (v_uv[i].1 as i64) << 16,
        };
    }
    // Sort by y ascending: bubble sort is fine for n=3.
    if verts[0].y > verts[1].y {
        verts.swap(0, 1);
    }
    if verts[0].y > verts[2].y {
        verts.swap(0, 2);
    }
    if verts[1].y > verts[2].y {
        verts.swap(1, 2);
    }

    let v1 = &verts[0]; // top
    let v2 = &verts[1]; // middle
    let v3 = &verts[2]; // bottom

    let height = v3.y - v1.y;
    if height == 0 {
        return None;
    }
    // `temp = (v2.y - v1.y) / height` in Q16.16.
    let temp = ((v2.y - v1.y) as i64) << 16;
    let temp = temp / (height as i64);
    // longest = temp * (v3.x - v1.x) / (2^16) + (v1.x - v2.x)
    //   — i.e. extrapolate the v1→v3 edge to y=v2.y, subtract v2.x.
    // Both factors of `temp` are already in Q16.16, so `(v3.x - v1.x) >> 16`
    // drops the fixed fraction before multiply (matches Redux).
    let longest = temp * ((v3.x - v1.x) >> 16) + (v1.x - v2.x);
    if longest == 0 {
        return None;
    }

    let mut setup = SlTriSetup {
        sorted: [verts[0], verts[1], verts[2]],
        left_array: [0; 3],
        right_array: [0; 3],
        left_section: 0,
        right_section: 0,
        left_x: 0,
        right_x: 0,
        delta_left_x: 0,
        delta_right_x: 0,
        left_section_height: 0,
        right_section_height: 0,
        left_r: 0,
        left_g: 0,
        left_b: 0,
        delta_left_r: 0,
        delta_left_g: 0,
        delta_left_b: 0,
        left_u: 0,
        left_v: 0,
        delta_left_u: 0,
        delta_left_v: 0,
        delta_col_r: 0,
        delta_col_g: 0,
        delta_col_b: 0,
        delta_col_u: 0,
        delta_col_v: 0,
        y_min: v1.y,
        y_max: v3.y - 1, // top-left rule: bottom row excluded
    };

    // Layout the left/right arrays depending on which side has the pivot.
    // sorted[] is indexed: 0=v1(top), 1=v2(middle), 2=v3(bottom).
    if longest < 0 {
        // Long edge v1→v3 is on the RIGHT. Left = single edge v1→v3.
        // Right walks v3 → v2 → v1 (two sections).
        setup.right_array = [2, 1, 0];
        setup.right_section = 2;
        setup.left_array = [2, 0, 0];
        setup.left_section = 1;
        setup.compute_left_section().ok()?;
        // Redux: if the first right section degenerates (height 0),
        // pop once and try again. Handles triangles where v1 == v2 in Y.
        if setup.compute_right_section().is_err() {
            setup.right_section -= 1;
            setup.compute_right_section().ok()?;
        }
    } else {
        // Long edge v1→v3 is on the LEFT. Left walks v3 → v2 → v1.
        // Right = single edge v1→v3.
        setup.left_array = [2, 1, 0];
        setup.left_section = 2;
        setup.right_array = [2, 0, 0];
        setup.right_section = 1;
        setup.compute_right_section().ok()?;
        if setup.compute_left_section().is_err() {
            setup.left_section -= 1;
            setup.compute_left_section().ok()?;
        }
    }

    // Clamp `longest` to ±0x1000 (Redux does this as `if (longest <
    // 0x1000) longest = 0x1000` and symmetric for the other sign —
    // prevents pathological per-column deltas when the triangle is
    // degenerately thin horizontally).
    let longest_clamped: i64 = if longest < 0 {
        longest.min(-0x1000)
    } else {
        longest.max(0x1000)
    };

    // Per-column deltas. The formula is Redux's `shl10idiv(temp * ((v3->X
    // - v1->X) >> 10) + ((v1->X - v2->X) << 6), longest)` for each of
    // R/G/B/U/V. The >> 10 and << 6 line up with `temp`'s Q16.16 so
    // the final shl10idiv produces a Q16.16 per-column delta.
    let compute_col_delta = |a3: i64, a1: i64, a2: i64| -> i64 {
        shl10_idiv(
            (temp * ((a3 - a1) >> 10)) + ((a1 - a2) << 6),
            longest_clamped,
        )
    };
    setup.delta_col_r = compute_col_delta(v3.r, v1.r, v2.r);
    setup.delta_col_g = compute_col_delta(v3.g, v1.g, v2.g);
    setup.delta_col_b = compute_col_delta(v3.b, v1.b, v2.b);
    setup.delta_col_u = compute_col_delta(v3.u, v1.u, v2.u);
    setup.delta_col_v = compute_col_delta(v3.v, v1.v, v2.v);

    Some(setup)
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

/// Redux's 4×4 dither coefficient table, indexed by
/// `(y & 3) * 4 + (x & 3)`. See `s_dithertable` in
/// `pcsx-redux/src/gpu/soft/soft.cc`. Note these are NOT the signed
/// Bayer offsets you'll see quoted in some PSX-SPX derivatives —
/// they're threshold coefficients for Redux's "conditional round-up"
/// dither model, which produces the exact bit pattern PSX hardware
/// uses.
const DITHER_COEFFS: [u8; 16] = [7, 0, 6, 1, 2, 5, 3, 4, 1, 6, 0, 7, 4, 3, 5, 2];

/// Dither an 8-bit RGB triple to 15bpp, matching Redux's
/// `prepareDitherLut` / `applyDither` byte-for-byte.
///
/// The algorithm: for each channel split into a 5-bit quotient and a
/// 3-bit remainder; if the remainder beats the coefficient for this
/// pixel AND the quotient isn't already saturated (0x1F), round the
/// quotient up by one. That produces the characteristic PSX 4×4
/// dither pattern — fundamentally different from the additive
/// `-4..+3` offset model PSX-SPX sometimes describes, and
/// producing different bit patterns. Matching Redux's algorithm
/// exactly is the only way to hit pixel-exact parity on Gouraud
/// gradients (which is most of what the Sony logo is).
fn dither_rgb(r: i32, g: i32, b: i32, x: i32, y: i32) -> u16 {
    let coeff = DITHER_COEFFS[((y & 3) * 4 + (x & 3)) as usize] as u32;
    let r = r.clamp(0, 255) as u32;
    let g = g.clamp(0, 255) as u32;
    let b = b.clamp(0, 255) as u32;
    let mut rc = r >> 3;
    let mut gc = g >> 3;
    let mut bc = b >> 3;
    // Round-up rule: if the low 3 bits exceed the coefficient AND we
    // have headroom, increment. The saturation guard is essential —
    // without it pure-white pixels would get stuck rounding up past
    // 0x1F and wrapping (or in Redux's case, indexing past the
    // precomputed LUT).
    if rc < 0x1F && (r & 7) > coeff {
        rc += 1;
    }
    if gc < 0x1F && (g & 7) > coeff {
        gc += 1;
    }
    if bc < 0x1F && (b & 7) > coeff {
        bc += 1;
    }
    (bc << 10) as u16 | ((gc << 5) as u16) | rc as u16
}

/// Transient state held between the start of a polyline primitive
/// (GP0 0x48..=0x4B or 0x58..=0x5B) and its terminator word. Each
/// variant carries the most recently-rasterised endpoint so the
/// next segment can chain from it.
#[derive(Copy, Clone, Debug)]
enum PolylineState {
    /// Monochrome polyline — all segments use the same color.
    Mono {
        color: u16,
        mode: BlendMode,
        last_vertex: (i32, i32),
    },
    /// Gouraud polyline — each segment interpolates between the
    /// prior vertex's colour and the next colour word. Polyline
    /// receive mode alternates between (color word, vertex word)
    /// pairs; `awaiting_color` tracks which half we're on.
    Shaded {
        mode: BlendMode,
        last_color: u32,
        last_vertex: (i32, i32),
        awaiting_color: bool,
        pending_color: u32,
    },
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

/// Dithered variant of [`modulate_tint`] — computes the modulated
/// RGB in 8-bit space, applies the 4×4 Bayer dither offset for the
/// pixel position, then truncates to 5 bits per channel. Used by
/// textured-Gouraud primitives when GP0 0xE1 bit 9 is on.
fn modulate_tint_dithered(
    texel: u16,
    tint_r: u32,
    tint_g: u32,
    tint_b: u32,
    x: i32,
    y: i32,
) -> u16 {
    // Scale 5-bit texel channels to 8-bit, apply the tint (which
    // is 0x80 = identity at 8-bit scale), then dither + truncate.
    let tr = ((texel & 0x1F) as u32) << 3;
    let tg = (((texel >> 5) & 0x1F) as u32) << 3;
    let tb = (((texel >> 10) & 0x1F) as u32) << 3;
    let r = (tint_r * tr / 0x80).min(0xFF) as i32;
    let g = (tint_g * tg / 0x80).min(0xFF) as i32;
    let b = (tint_b * tb / 0x80).min(0xFF) as i32;
    dither_rgb(r, g, b, x, y) | (texel & 0x8000)
}

/// Split a 24-bit RGB tint word (from the low 24 bits of a textured
/// primitive's command) into the three channels the modulator
/// expects. Returns `(tint_r, tint_g, tint_b)` with each in 0..=255.
/// For "raw texture" primitives the caller substitutes `(128, 128,
/// 128)` directly — one code path through [`modulate_tint`].
#[inline]
fn split_tint(tint24: u32) -> (u32, u32, u32) {
    (tint24 & 0xFF, (tint24 >> 8) & 0xFF, (tint24 >> 16) & 0xFF)
}

/// Identity tint — pass-through for raw-texture primitives. Each
/// channel at `0x80` means modulation returns the texel unchanged.
const RAW_TEXTURE_TINT: (u32, u32, u32) = (0x80, 0x80, 0x80);

/// Blend a foreground pixel over a background pixel per `mode`.
/// Both pixels are 15-bit BGR with a mask bit at bit 15. The mask
/// bit of the result comes from the foreground so semi-transparent
/// texels keep marking themselves.
///
/// Matches Redux's per-channel arithmetic in
/// `pcsx-redux/src/gpu/soft/soft.cc` byte-for-byte. The subtle one
/// is **Average**: Redux computes `(bg >> 1) + (fg >> 1)` independent
/// per-channel, dropping each operand's LSB *before* summing. The
/// naive `(bg + fg) / 2` rounds differently when both inputs are
/// odd — e.g. `(3 + 3) / 2 = 3` vs Redux's `1 + 1 = 2`. That bug
/// alone produces off-by-1 diffs on the Sony logo's semi-
/// transparent gradient edges.
fn blend_pixel(bg: u16, fg: u16, mode: BlendMode) -> u16 {
    if mode == BlendMode::Opaque {
        return fg;
    }
    let br = (bg & 0x1F) as i16;
    let bgg = ((bg >> 5) & 0x1F) as i16;
    let bb = ((bg >> 10) & 0x1F) as i16;
    let fr = (fg & 0x1F) as i16;
    let fgg = ((fg >> 5) & 0x1F) as i16;
    let fb = ((fg >> 10) & 0x1F) as i16;
    let (r, g, b) = match mode {
        BlendMode::Opaque => unreachable!(),
        // Half-back + half-front — per-channel right-shift before
        // summing, matching Redux's `& 0x7bde >> 1` pattern.
        BlendMode::Average => (
            (br >> 1) + (fr >> 1),
            (bgg >> 1) + (fgg >> 1),
            (bb >> 1) + (fb >> 1),
        ),
        BlendMode::Add => ((br + fr).min(31), (bgg + fgg).min(31), (bb + fb).min(31)),
        BlendMode::Sub => ((br - fr).max(0), (bgg - fgg).max(0), (bb - fb).max(0)),
        // Full-back + quarter-front — `fg / 4` via integer division
        // is the same as Redux's `(fg & 0x1c) >> 2` for 5-bit
        // channels: both truncate the low 2 bits then shift.
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
        Self { raw: 0x1480_2000 }
    }

    /// Compose the observable GPUSTAT word. `vram_send_ready` is the
    /// live "is a VRAM→CPU transfer in progress and has pixels
    /// waiting" flag owned by the GPU proper.
    fn read(&self, vram_send_ready: bool) -> u32 {
        let mut ret = self.raw;

        // Ready bits: 26 (cmd FIFO ready) + 28 (DMA block ready).
        // Redux's soft GPU keeps both set on every status read.
        // Bit 27 (VRAM→CPU ready) is only set while software is
        // actually pulling pixels from GPUREAD; Redux's default is
        // 0 and BIOS/game code polls this bit to detect transfer
        // completion.
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
        gpu.charge_busy(10_000);
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
        assert_eq!(
            stat & 0x0800_0000,
            0x0800_0000,
            "bit 27 set during transfer"
        );
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
    fn gp1_extended_hres_zero_is_368_pixels() {
        let mut gpu = Gpu::new();
        gpu.write32(GP1_ADDR, 0x0703_c000); // v-range 0..240
        gpu.write32(GP1_ADDR, 0x0800_0060); // hres2=1, hres1=0
        assert_eq!(gpu.display_area().width, 368);

        gpu.write32(GP1_ADDR, 0x0800_0061); // hres2=1, hres1=1
        assert_eq!(gpu.display_area().width, 384);
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
    fn blend_average_halves_each_operand_then_sums() {
        // Even+even case: per-channel `(bg >> 1) + (fg >> 1)`.
        // BG = (10, 10, 10), FG = (20, 20, 20) → (5+10, 5+10, 5+10) = (15, 15, 15).
        let bg = 10 | (10 << 5) | (10 << 10);
        let fg = 20 | (20 << 5) | (20 << 10);
        let out = blend_pixel(bg, fg, BlendMode::Average);
        assert_eq!(out & 0x1F, 15);
        assert_eq!((out >> 5) & 0x1F, 15);
        assert_eq!((out >> 10) & 0x1F, 15);
    }

    #[test]
    fn blend_average_odd_plus_odd_matches_redux() {
        // The bug that cost pixel parity on Sony-logo letter edges:
        // naive `(bg + fg) / 2` gives `(3 + 3) / 2 = 3`; Redux does
        // `(3 >> 1) + (3 >> 1) = 1 + 1 = 2`. Assert the Redux result.
        let bg = 3 | (3 << 5) | (3 << 10);
        let fg = 3 | (3 << 5) | (3 << 10);
        let out = blend_pixel(bg, fg, BlendMode::Average);
        assert_eq!(out & 0x1F, 2, "R: odd+odd must lose LSB per-operand");
        assert_eq!((out >> 5) & 0x1F, 2, "G: same");
        assert_eq!((out >> 10) & 0x1F, 2, "B: same");

        // Asymmetric: BG=5 (odd), FG=3 (odd) → (5>>1)+(3>>1) = 2+1 = 3.
        // Naive would give (5+3)/2 = 4.
        let bg = 5u16;
        let fg = 3u16;
        let out = blend_pixel(bg, fg, BlendMode::Average);
        assert_eq!(out & 0x1F, 3);
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
        assert_eq!(prim_blend_mode(0x2200_0000, BlendMode::Add), BlendMode::Add);
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
    fn dither_rgb_matches_redux_truth_table() {
        // Hand-computed expected outputs from Redux's `prepareDitherLut`
        // at `pcsx-redux/src/gpu/soft/soft.cc:277`. Each row is a
        // different (input, x, y) combination; each column the
        // expected R/G/B 5-bit quotient after the conditional round-up.
        //
        //   input r,g,b | (x,y)  | coeff | expected_r expected_g expected_b
        //
        //   128,128,128 | (0,0)  |   7   |    16         16         16
        //     (128 >> 3 = 16; 128 & 7 = 0; 0 > 7? no → keep 16)
        //   128,128,128 | (1,0)  |   0   |    17         17         17
        //     (128 >> 3 = 16; 128 & 7 = 0; 0 > 0? no → keep 16)
        //     (wait: 128 & 7 = 0. Redux needs low > coeff, 0 > 0 is false.)
        //   255,255,255 | (3,3)  |   2   |    31         31         31  (saturated)
        //   0,0,0       | (0,0)  |   7   |     0          0          0  (round-up guarded)
        //   7,7,7       | (0,0)  |   7   |     0          0          0   (7 > 7? no)
        //   7,7,7       | (1,0)  |   0   |     1          1          1   (7 > 0? yes)
        //   5,5,5       | (1,0)  |   0   |     1          1          1   (5 > 0? yes)
        //   5,5,5       | (2,0)  |   6   |     0          0          0   (5 > 6? no)

        let check = |r: i32, g: i32, b: i32, x: i32, y: i32, er: u16, eg: u16, eb: u16| {
            let v = dither_rgb(r, g, b, x, y);
            assert_eq!(v & 0x1F, er, "R mismatch for ({r},{g},{b})@({x},{y})");
            assert_eq!((v >> 5) & 0x1F, eg, "G mismatch");
            assert_eq!((v >> 10) & 0x1F, eb, "B mismatch");
        };
        // coeff[0]=7 → only strictly-greater-than-7 rounds up; 128&7=0 keeps 16.
        check(128, 128, 128, 0, 0, 16, 16, 16);
        // coeff[1]=0 → any non-zero low bits round up. But 128&7=0, so stays 16.
        check(128, 128, 128, 1, 0, 16, 16, 16);
        // coeff[1]=0, 7&7=7 > 0 → round up 0→1.
        check(7, 7, 7, 1, 0, 1, 1, 1);
        // coeff[0]=7, 7&7=7 not > 7 → stays 0.
        check(7, 7, 7, 0, 0, 0, 0, 0);
        // Saturation guard: 255 >> 3 = 31 already, can't increment.
        check(255, 255, 255, 3, 3, 31, 31, 31);
        // Zero stays zero.
        check(0, 0, 0, 0, 0, 0, 0, 0);
        // coeff[2]=6, 5&7=5, 5 > 6 is false → stays 0.
        check(5, 5, 5, 2, 0, 0, 0, 0);
    }

    #[test]
    fn dither_rgb_saturates_at_255() {
        // Pure 255 must never wrap — tests the `rc < 0x1F` guard
        // across every coefficient position.
        for x in 0..4 {
            for y in 0..4 {
                let v = dither_rgb(255, 255, 255, x, y);
                assert_eq!(v & 0x1F, 31);
                assert_eq!((v >> 5) & 0x1F, 31);
                assert_eq!((v >> 10) & 0x1F, 31);
            }
        }
    }

    #[test]
    fn mono_line_packet_size_is_three() {
        for op in 0x40..=0x43 {
            assert_eq!(gp0_packet_size(op), 3, "opcode 0x{op:02X}");
        }
    }

    #[test]
    fn shaded_line_packet_size_is_four() {
        for op in 0x50..=0x53 {
            assert_eq!(gp0_packet_size(op), 4, "opcode 0x{op:02X}");
        }
    }

    #[test]
    fn polyline_start_packet_sizes_match_single() {
        for op in 0x48..=0x4B {
            assert_eq!(gp0_packet_size(op), 3);
        }
        for op in 0x58..=0x5B {
            assert_eq!(gp0_packet_size(op), 4);
        }
    }

    #[test]
    fn wireframe_toggle_makes_tri_draw_edges_only() {
        let mut gpu = Gpu::new();
        // Draw area: top-left (0, 0), bottom-right (1023, 511).
        // GP0 0xE3: x (bits 0..=9), y (bits 10..=18).
        gpu.write32(GP0_ADDR, 0xE3_00_00_00);
        // GP0 0xE4: right (bits 0..=9), bottom (bits 10..=18).
        gpu.write32(GP0_ADDR, 0xE4_00_00_00 | 0x3FF | (0x1FF << 10));
        gpu.wireframe_enabled = true;
        // Tiny triangle at (0,0), (4,0), (2,2). With wireframe
        // on, edges get drawn; interior stays zero.
        gpu.write32(GP0_ADDR, 0x20_FF_FF_FF);
        gpu.write32(GP0_ADDR, 0x0000_0000);
        gpu.write32(GP0_ADDR, 0x0000_0004);
        gpu.write32(GP0_ADDR, 0x0002_0002);
        // Corner pixels sit on edges — must be lit.
        assert_ne!(gpu.vram.get_pixel(0, 0), 0, "corner (0,0)");
        assert_ne!(gpu.vram.get_pixel(4, 0), 0, "corner (4,0)");
        assert_ne!(gpu.vram.get_pixel(2, 2), 0, "corner (2,2)");
        // A fully-interior pixel at (2, 1) sits just inside the
        // triangle and on no edge — must stay zero.
        assert_eq!(gpu.vram.get_pixel(2, 1), 0, "interior should be empty");
    }

    #[test]
    fn mono_line_horizontal_plots_one_row() {
        let mut gpu = Gpu::new();
        // Draw area: full VRAM.
        gpu.write32(GP0_ADDR, 0xE3_00_00_00); // top-left 0,0
        gpu.write32(GP0_ADDR, 0xE4_00_03_FF); // bot-right 1023,0 (one row)
                                              // Mono line: white, from (0,0) to (9,0).
        gpu.write32(GP0_ADDR, 0x40_FF_FF_FF); // cmd + white
        gpu.write32(GP0_ADDR, 0x0000_0000); // v0 = (0, 0)
        gpu.write32(GP0_ADDR, 0x0000_0009); // v1 = (9, 0)
        for x in 0..=9u16 {
            let px = gpu.vram.get_pixel(x, 0);
            assert_ne!(px, 0, "pixel ({x}, 0) should be set");
        }
        assert_eq!(gpu.vram.get_pixel(10, 0), 0);
    }

    #[test]
    fn mono_polyline_end_sentinel_exits_receive_mode() {
        let mut gpu = Gpu::new();
        gpu.write32(GP0_ADDR, 0xE3_00_00_00);
        gpu.write32(GP0_ADDR, 0xE4_00_03_FF);
        // Start polyline.
        gpu.write32(GP0_ADDR, 0x48_FF_FF_FF);
        gpu.write32(GP0_ADDR, 0x0000_0000); // v0
        gpu.write32(GP0_ADDR, 0x0000_0005); // v1
        assert!(gpu.polyline.is_some());
        // Another vertex.
        gpu.write32(GP0_ADDR, 0x0000_000A);
        assert!(gpu.polyline.is_some());
        // Terminator.
        gpu.write32(GP0_ADDR, 0x5000_5000);
        assert!(gpu.polyline.is_none());
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

    // --- sample_texture transparency rules ---
    //
    // PSX convention: a texel is transparent when the **resolved
    // 16-bit colour** is 0x0000. For 4bpp/8bpp that means
    // `CLUT[idx] == 0`, not `idx == 0`. The BIOS TM-glyph regression
    // was caused by the simpler `idx == 0` check rendering opaque
    // black where the CLUT had deliberately-zero entries at non-zero
    // indices to punch the letter cutouts.

    #[test]
    fn sample_texture_4bpp_idx0_resolves_to_clut_entry() {
        // CLUT entry at index 0 is non-zero → idx==0 is NOT transparent.
        // Matches Redux: only the resolved colour's 0x0000 is skipped.
        let mut gpu = Gpu::new();
        gpu.tex_depth = 0; // 4bpp
        gpu.tex_page_x = 0;
        gpu.tex_page_y = 0;
        // Texture word at (0, 0) — all four texels point to CLUT idx 0.
        gpu.vram.set_pixel(0, 0, 0x0000);
        // CLUT row at (0x100, 0), entry 0 = red (0x001F).
        gpu.vram.set_pixel(0x100, 0, 0x001F);
        // u=0..3 all sample idx=0 → all must resolve to CLUT[0] = 0x001F.
        for u in 0..4u16 {
            assert_eq!(
                gpu.sample_texture(u, 0, 0x100, 0),
                Some(0x001F),
                "u={u}: CLUT[0]=0x001F should be opaque, not transparent",
            );
        }
    }

    #[test]
    fn sample_texture_4bpp_nonzero_idx_with_zero_clut_is_transparent() {
        // The TM-glyph bug: CLUT entry at non-zero index is 0x0000
        // (deliberate punch-through). Hardware skips (transparent);
        // the bugged emulator would draw opaque black.
        let mut gpu = Gpu::new();
        gpu.tex_depth = 0;
        gpu.tex_page_x = 0;
        gpu.tex_page_y = 0;
        // Texture word at (0, 0): idx[0]=5, idx[1]=3, idx[2]=2, idx[3]=1.
        // We want to hit idx=5 — that means bits 0..3 should be 5.
        // (word >> 0) & 0xF = 5. So word low nibble = 5.
        gpu.vram.set_pixel(0, 0, 0x0005);
        // CLUT at (0x200, 0): entry 5 is 0x0000 (punch-through).
        // All others set to non-zero so we can be sure the transparency
        // comes from CLUT[5]=0 and not from a wrong index.
        for e in 0..16u16 {
            gpu.vram
                .set_pixel(0x200 + e, 0, if e == 5 { 0x0000 } else { 0x7FFF });
        }
        assert_eq!(
            gpu.sample_texture(0, 0, 0x200, 0),
            None,
            "CLUT[5]=0x0000 should be transparent even though idx=5, not 0",
        );
    }

    #[test]
    fn sample_texture_8bpp_zero_clut_is_transparent_regardless_of_idx() {
        // Same rule for 8bpp mode.
        let mut gpu = Gpu::new();
        gpu.tex_depth = 1; // 8bpp
        gpu.tex_page_x = 0;
        gpu.tex_page_y = 0;
        // Texture word at (0, 0) low byte = 42.
        gpu.vram.set_pixel(0, 0, 42);
        // CLUT[42] = 0x0000 → should be transparent.
        gpu.vram.set_pixel(0x100 + 42, 0, 0x0000);
        assert_eq!(gpu.sample_texture(0, 0, 0x100, 0), None);
        // Flip CLUT[42] to non-zero → must draw opaque.
        gpu.vram.set_pixel(0x100 + 42, 0, 0x1234);
        assert_eq!(gpu.sample_texture(0, 0, 0x100, 0), Some(0x1234));
    }

    #[test]
    fn sample_texture_15bpp_zero_is_transparent() {
        // Direct-colour mode: 0x0000 is transparent, anything else opaque.
        let mut gpu = Gpu::new();
        gpu.tex_depth = 2; // 15bpp
        gpu.tex_page_x = 0;
        gpu.tex_page_y = 0;
        gpu.vram.set_pixel(0, 0, 0x0000);
        assert_eq!(gpu.sample_texture(0, 0, 0, 0), None);
        gpu.vram.set_pixel(1, 0, 0x1234);
        assert_eq!(gpu.sample_texture(1, 0, 0, 0), Some(0x1234));
    }

    #[test]
    fn sample_texture_uv_wrap_at_256_per_psx_spx() {
        // Regression: the PS1's GPU walks U/V through an 8-bit counter,
        // so a tpage wraps every 256 texels horizontally and vertically.
        // The rasterizer adds a width-bounded `dx` to the base U/V and
        // can pass values >= 256 here; without the explicit `& 0xFF`
        // we read VRAM PAST the tpage edge and pull garbage from the
        // neighbouring tpage. Visible as smeared 2D sprites in pre-fight
        // loading screens (Tekken 3 character portraits) and corrupt
        // BIOS dialog frames.
        //
        // Test: in 15bpp mode, place a recognisable colour at tpage
        // origin (u,v)=(0,0) and a different colour at host VRAM
        // (256,0) — outside the tpage. Sampling at u=256 must return
        // the FIRST colour (wrap), not the second.
        let mut gpu = Gpu::new();
        gpu.tex_depth = 2; // 15bpp — one VRAM word per texel
        gpu.tex_page_x = 0;
        gpu.tex_page_y = 0;
        gpu.vram.set_pixel(0, 0, 0x1111); // tpage origin
        gpu.vram.set_pixel(256, 0, 0x2222); // outside the tpage
        assert_eq!(
            gpu.sample_texture(256, 0, 0, 0),
            Some(0x1111),
            "u=256 should wrap to u=0 within the tpage"
        );
        assert_eq!(
            gpu.sample_texture(257, 0, 0, 0),
            None,
            "u=257 should wrap to u=1 → vram(1,0)=0 → transparent"
        );
        // V wrap: v=256 → v&0xFF=0 → vram(0,0)=0x1111. v=257 →
        // v&0xFF=1 → vram(0,1) (a different row), confirms v wraps too.
        gpu.vram.set_pixel(0, 1, 0x3333);
        assert_eq!(
            gpu.sample_texture(0, 256, 0, 0),
            Some(0x1111),
            "v=256 wraps to v=0"
        );
        assert_eq!(
            gpu.sample_texture(0, 257, 0, 0),
            Some(0x3333),
            "v=257 wraps to v=1"
        );
    }

    fn pack_test_vertex(x: i16, y: i16) -> u32 {
        ((x as u16) as u32) | (((y as u16) as u32) << 16)
    }

    fn pack_test_uv(u: u8, v: u8, extra: u16) -> u32 {
        (u as u32) | ((v as u32) << 8) | ((extra as u32) << 16)
    }

    fn prepare_opaque_15bpp_texture(gpu: &mut Gpu) -> u16 {
        // 15bpp texture page at x=0, y=256. Keeping texture source
        // away from the draw area makes these tests easier to reason
        // about: any low-screen pixel change came from the primitive.
        const TPAGE_15BPP_Y256: u16 = (1 << 4) | (2 << 7);
        gpu.write32(GP0_ADDR, 0xE3_00_00_00);
        gpu.write32(GP0_ADDR, 0xE4_00_00_00 | 0x3FF | (0x1FF << 10));
        gpu.vram.set_pixel(0, 256, 0x7FFF);
        TPAGE_15BPP_Y256
    }

    #[test]
    fn extent_boundary_inclusive_keeps_triangle() {
        // Exactly 1023 / 511 deltas are *kept* — only strictly greater
        // is dropped. This matches the hardware spec that the punch
        // list quotes ("|Δx| > 1023 or |Δy| > 511").
        assert!(!triangle_exceeds_hw_extent((0, 0), (1023, 0), (0, 511),));
    }

    #[test]
    fn extent_one_pixel_over_horizontal_drops() {
        assert!(triangle_exceeds_hw_extent((0, 0), (1024, 0), (0, 0)));
    }

    #[test]
    fn extent_one_pixel_over_vertical_drops() {
        assert!(triangle_exceeds_hw_extent((0, 0), (0, 512), (0, 0)));
    }

    #[test]
    fn extent_check_uses_absolute_value() {
        // Negative deltas must trip the check too — typically a vertex
        // at (-2000, 0) paired with (0, 0).
        assert!(triangle_exceeds_hw_extent((-2000, 0), (0, 0), (0, 0)));
        assert!(triangle_exceeds_hw_extent((0, 0), (0, 0), (0, -700)));
    }

    #[test]
    fn extent_check_compares_every_edge() {
        // First two vertices coincide (|Δ|=0), but v0→v2 is huge.
        // A naive bounding-box check would still be 1024 wide and
        // catch this; what we want to confirm is that edge pairs
        // are visited even when one of them looks small.
        assert!(triangle_exceeds_hw_extent((0, 0), (0, 0), (1024, 0)));
        assert!(triangle_exceeds_hw_extent((1024, 0), (0, 0), (0, 0)));
    }

    #[test]
    fn oversize_textured_triangle_is_dropped() {
        // This is the material-viewer lesson in miniature: projected
        // textured triangles can look mostly sane on screen but still
        // cross the PS1's per-edge extent limit. Hardware drops the
        // whole primitive; engines should split before submitting.
        let mut gpu = Gpu::new();
        let tpage = prepare_opaque_15bpp_texture(&mut gpu);

        gpu.write32(GP0_ADDR, 0x2500_0000); // raw textured triangle
        gpu.write32(GP0_ADDR, pack_test_vertex(-1000, 0));
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, 0));
        gpu.write32(GP0_ADDR, pack_test_vertex(24, 0)); // dx = 1024 -> drop
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, tpage));
        gpu.write32(GP0_ADDR, pack_test_vertex(24, 64));
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, 0));

        assert_eq!(
            gpu.vram.get_pixel(20, 8),
            0,
            "oversize textured triangle should be skipped, not partially drawn",
        );
    }

    #[test]
    fn legal_textured_triangle_one_pixel_under_extent_draws() {
        let mut gpu = Gpu::new();
        let tpage = prepare_opaque_15bpp_texture(&mut gpu);

        gpu.write32(GP0_ADDR, 0x2500_0000); // raw textured triangle
        gpu.write32(GP0_ADDR, pack_test_vertex(-999, 0));
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, 0));
        gpu.write32(GP0_ADDR, pack_test_vertex(24, 0)); // dx = 1023 -> keep
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, tpage));
        gpu.write32(GP0_ADDR, pack_test_vertex(24, 64));
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, 0));

        assert_ne!(
            gpu.vram.get_pixel(20, 8),
            0,
            "triangle at the exact legal extent should still rasterise",
        );
    }

    #[test]
    fn textured_quad_drops_only_the_oversize_split_half() {
        // Non-axis-aligned textured quads are split into two triangles.
        // As with real hardware, the extent rule applies to each half
        // independently; a bad second half must not erase the good one.
        let mut gpu = Gpu::new();
        let tpage = prepare_opaque_15bpp_texture(&mut gpu);

        gpu.write32(GP0_ADDR, 0x2D00_0000); // raw textured quad
        gpu.write32(GP0_ADDR, pack_test_vertex(10, 10)); // v0
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, 0));
        gpu.write32(GP0_ADDR, pack_test_vertex(30, 10)); // v1
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, tpage));
        gpu.write32(GP0_ADDR, pack_test_vertex(10, 30)); // v2
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, 0));
        gpu.write32(GP0_ADDR, pack_test_vertex(30, 522)); // v3: |dy v1->v3| = 512
        gpu.write32(GP0_ADDR, pack_test_uv(0, 0, 0));

        assert_ne!(gpu.vram.get_pixel(14, 14), 0, "legal half should draw");
        assert_eq!(
            gpu.vram.get_pixel(28, 120),
            0,
            "oversize split half should be skipped independently",
        );
    }

    #[test]
    fn oversize_monochrome_triangle_is_dropped() {
        // Submit a triangle whose v0→v1 edge is 1500px wide via the
        // GP0 monochrome-tri command. Hardware drops it; we should
        // too — VRAM stays untouched.
        let mut gpu = Gpu::new();
        gpu.write32(GP0_ADDR, 0x2000_00FF); // 0x20 cmd + red
        gpu.write32(GP0_ADDR, 0x0000_0000); // v0 = (0, 0)
        gpu.write32(GP0_ADDR, 0x0000_05DC); // v1 = (1500, 0) — 1500 > 1023
        gpu.write32(GP0_ADDR, 0x0064_0064); // v2 = (100, 100)
                                            // No pixel anywhere along the would-be triangle should be set.
        for x in [0u16, 50, 100, 500, 1000, 1500] {
            assert_eq!(
                gpu.vram.get_pixel(x, 0),
                0,
                "pixel ({x}, 0) was written despite oversize triangle",
            );
        }
    }

    #[test]
    fn oversize_quad_drops_only_the_oversize_half() {
        // A four-vertex monochrome quad (GP0 0x28) splits into two
        // triangles: (v0,v1,v2) and (v1,v2,v3). Build one where the
        // first half is sane and the second half has v3 placed so
        // its Δy from v1 exceeds 511 — only the bad half should be
        // culled.
        let mut gpu = Gpu::new();
        gpu.write32(GP0_ADDR, 0x2800_00FF); // 0x28 + red
        gpu.write32(GP0_ADDR, 0x0000_0000); // v0 = (0, 0)
        gpu.write32(GP0_ADDR, 0x0000_0010); // v1 = (16, 0)
        gpu.write32(GP0_ADDR, 0x0010_0000); // v2 = (0, 16)
                                            // v3 = (16, 600) — |v3.y - v1.y| = 600 > 511, second triangle drops.
        gpu.write32(GP0_ADDR, 0x0258_0010);
        // Sane half wrote pixels.
        assert_ne!(gpu.vram.get_pixel(1, 1), 0, "first half should rasterise");
        // Oversize half left no pixels in the only place its
        // bounding box could have reached (a row well below the
        // sane half).
        assert_eq!(
            gpu.vram.get_pixel(8, 300),
            0,
            "oversize half should not rasterise",
        );
    }
}
