//! Phase C — wire `psx-gpu-compute` into the frontend as a parallel
//! rendering backend.
//!
//! Strategy: each frame the frontend drains the CPU rasterizer's
//! `cmd_log` (already populated by `enable_pixel_tracer`), replays
//! every GP0 packet through this backend's compute dispatchers, and
//! downloads the resulting VRAM for display. The CPU rasterizer
//! remains the source of truth — its VRAM is uploaded into the
//! compute backend at frame start so VRAM uploads / VRAM-to-VRAM
//! copies / FMV writes are reflected. The compute path then redraws
//! the frame's GP0 packets on top.
//!
//! This is intentionally a SHADOW renderer for now: if the compute
//! output diverges from the CPU's, the user-visible result is wrong
//! pixels but the next frame the VRAM gets re-synced, so divergences
//! don't accumulate. Behind the runtime `--gpu-compute` flag.
//!
//! What's NOT handled here yet
//!   - Lines / polylines (`0x40..=0x5F`) — rare in real games; skip.
//!   - GP1 commands — display-mode state, not rendering.
//!   - VRAM-to-CPU readback (`0xC0..=0xDF`) — game-side reads, no
//!     visible output.

use std::sync::Arc;

use emulator_core::gpu::GpuCmdLogEntry;

use crate::primitive::{
    BlendMode, DrawArea, Fill, MonoRect, MonoTri, PrimFlags, ShadedTexTri, ShadedTri,
    TexQuadBilinear, TexRect, TexTri, Tpage,
};
use crate::rasterizer::Rasterizer;
use crate::vram::{self, VramGpu};

/// GP0 state we have to track in lockstep with the CPU rasterizer
/// so triangle / rect dispatches see the right tpage, drawing area,
/// drawing offset, mask flags, and per-primitive blend mode.
#[derive(Debug, Clone)]
pub struct ReplayState {
    pub draw_area: DrawArea,
    pub draw_offset_x: i32,
    pub draw_offset_y: i32,
    pub tpage: Tpage,
    /// Active tpage blend mode (decoded from GP0 0xE1 bits 5-6 or
    /// per-primitive tpage word). Used by mono / shaded primitives
    /// when their cmd-bit-1 (semi-trans) is set.
    pub tex_blend_mode: BlendMode,
    /// `mask_set_on_draw` (GP0 0xE6 bit 0).
    pub mask_set: bool,
    /// `mask_check_before_draw` (GP0 0xE6 bit 1).
    pub mask_check: bool,
    /// `dither` (GP0 0xE1 bit 9). Affects shaded / textured-shaded.
    pub dither: bool,
    /// `tex_rect_flip_x` / `_y` (GP0 0xE1 bits 12 / 13).
    pub flip_x: bool,
    pub flip_y: bool,
}

impl ReplayState {
    fn new() -> Self {
        Self {
            draw_area: DrawArea {
                left: 0,
                top: 0,
                right: 1023,
                bottom: 511,
            },
            draw_offset_x: 0,
            draw_offset_y: 0,
            tpage: Tpage::new(0, 0, 0),
            tex_blend_mode: BlendMode::Average,
            mask_set: false,
            mask_check: false,
            dither: false,
            flip_x: false,
            flip_y: false,
        }
    }

    fn base_flags(&self) -> PrimFlags {
        let mut f = PrimFlags::empty();
        if self.mask_set {
            f |= PrimFlags::MASK_SET;
        }
        if self.mask_check {
            f |= PrimFlags::MASK_CHECK;
        }
        if self.dither {
            f |= PrimFlags::DITHER;
        }
        f
    }

    fn rect_flip_flags(&self) -> PrimFlags {
        let mut f = self.base_flags();
        if self.flip_x {
            f |= PrimFlags::FLIP_X;
        }
        if self.flip_y {
            f |= PrimFlags::FLIP_Y;
        }
        f
    }
}

/// Decode helper for "is this primitive raw-textured?". Tested
/// against bit 0 of the OPCODE byte (= bit 24 of the cmd word) per
/// PSX-SPX, matching `emulator-core::Gpu::draw_textured_*`.
fn is_raw_texture(cmd: u32) -> bool {
    (cmd >> 24) & 1 != 0
}

/// `prim_is_semi_trans` — bit 1 of the opcode byte (cmd word bit 25).
fn is_semi_trans(cmd: u32) -> bool {
    (cmd >> 25) & 1 != 0
}

fn decode_vertex(state: &ReplayState, word: u32) -> (i32, i32) {
    // 11-bit signed for both axes, then add the drawing offset.
    let x = sign_extend_11((word & 0x7FF) as i32) + state.draw_offset_x;
    let y = sign_extend_11(((word >> 16) & 0x7FF) as i32) + state.draw_offset_y;
    (x, y)
}

fn sign_extend_11(v: i32) -> i32 {
    if v & 0x400 != 0 {
        v | !0x7FF
    } else {
        v & 0x7FF
    }
}

fn decode_uv(word: u32) -> (u8, u8) {
    ((word & 0xFF) as u8, ((word >> 8) & 0xFF) as u8)
}

fn decode_tint(cmd: u32) -> (u8, u8, u8) {
    (
        (cmd & 0xFF) as u8,
        ((cmd >> 8) & 0xFF) as u8,
        ((cmd >> 16) & 0xFF) as u8,
    )
}

fn decode_clut(word: u32) -> (u32, u32) {
    let clut_word = (word >> 16) & 0xFFFF;
    let cx = (clut_word & 0x3F) * 16;
    let cy = (clut_word >> 6) & 0x1FF;
    (cx, cy)
}

fn rgb24_to_bgr15(rgb: u32) -> u16 {
    let r = ((rgb >> 3) & 0x1F) as u16;
    let g = (((rgb >> 8) >> 3) & 0x1F) as u16;
    let b = (((rgb >> 16) >> 3) & 0x1F) as u16;
    r | (g << 5) | (b << 10)
}

/// Apply a primitive's tpage word (the high half of UV1 in the GP0
/// packet) to the replay state, mirroring
/// `emulator-core::Gpu::apply_primitive_tpage` byte-for-byte.
fn apply_primitive_tpage(state: &mut ReplayState, uv_word: u32) {
    let tpage = (uv_word >> 16) & 0xFFFF;
    let tpage_x = (tpage & 0x0F) * 64;
    let tpage_y: u32 = if (tpage >> 4) & 1 != 0 { 256 } else { 0 };
    let tex_depth = (tpage >> 7) & 0x3;
    state.tpage = Tpage::new(tpage_x, tpage_y, tex_depth);
    state.tex_blend_mode = decode_blend_mode(tpage >> 5);
    // CPU `apply_primitive_tpage` also OR-folds dither_enabled
    // (`tpage |= 0x200` if dither was on globally). We mirror by
    // leaving `state.dither` as-is — dither only flips off via a
    // GP0 0xE1, not a primitive's tpage word.
}

fn decode_blend_mode(bits: u32) -> BlendMode {
    match bits & 0x3 {
        0 => BlendMode::Average,
        1 => BlendMode::Add,
        2 => BlendMode::Sub,
        _ => BlendMode::AddQuarter,
    }
}

/// Compute backend — owns `VramGpu` + `Rasterizer` plus the replay
/// state needed to interpret each GP0 packet.
pub struct ComputeBackend {
    vram: VramGpu,
    rasterizer: Rasterizer,
    state: ReplayState,
    /// Counts of unhandled opcodes so the frontend can surface
    /// "compute backend doesn't yet know how to draw X" warnings.
    pub unhandled: std::collections::BTreeMap<u8, u64>,
}

impl ComputeBackend {
    /// Build the backend on a fresh headless wgpu adapter. The
    /// frontend uses this when `--gpu-compute` is enabled — sharing
    /// the main `Graphics` device would need an `Arc<Device>`
    /// refactor through the whole gfx layer, and the per-frame VRAM
    /// bounce already goes through CPU memory so a separate adapter
    /// costs nothing extra in the steady state.
    pub fn new_headless() -> Self {
        let vram = VramGpu::new_headless();
        let rasterizer = Rasterizer::new(&vram);
        Self {
            vram,
            rasterizer,
            state: ReplayState::new(),
            unhandled: std::collections::BTreeMap::new(),
        }
    }

    /// Build on top of an existing wgpu device — useful for tests
    /// or for a future zero-bounce display path where compute and
    /// display share the same adapter.
    #[allow(dead_code)]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let vram = VramGpu::new(device, queue);
        let rasterizer = Rasterizer::new(&vram);
        Self {
            vram,
            rasterizer,
            state: ReplayState::new(),
            unhandled: std::collections::BTreeMap::new(),
        }
    }

    /// Replace the GPU-side VRAM with the CPU rasterizer's current
    /// VRAM. Called at frame start so the compute path sees the
    /// same texture / framebuffer state the CPU is about to render
    /// against. Flushes any pending dispatches first — VRAM uploads
    /// happen on the queue-write side, but writes are ordered with
    /// submitted command buffers, so we want all prior draws to
    /// land before this overwrites their output.
    pub fn sync_vram_from_cpu(&self, cpu_words: &[u16]) {
        self.rasterizer.flush();
        let _ = self.vram.upload_full(cpu_words);
    }

    /// Read back the GPU VRAM for display. Slow per-frame
    /// (1 MiB GPU→CPU bounce) — acceptable for an opt-in shadow
    /// renderer. A future optimisation would render directly into
    /// the egui texture without the CPU round-trip. Flushes pending
    /// dispatches so the readback sees their output.
    pub fn download_vram(&self) -> Vec<u16> {
        self.rasterizer.flush();
        self.vram.download_full().unwrap_or_default()
    }

    /// Lift a sub-rectangle of CPU VRAM into GPU VRAM. The bisector
    /// uses this to apply CPU-to-VRAM uploads and FillRects whose
    /// pixel data isn't in the cmd_log proper — it streams via
    /// `ingest_vram_upload_word` on the bus side. Production replay
    /// (frontend / replay_disc) doesn't need this because it
    /// `sync_vram_from_cpu`s the full VRAM at frame boundaries.
    pub fn upload_rect_from_cpu(&self, cpu_words: &[u16], x: u32, y: u32, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.rasterizer.flush();
        // Honour VRAM wrap (hardware wraps both axes mod 1024 / 512).
        // We slice CPU words row-by-row so partial-row wraps work.
        let mut buf = Vec::with_capacity((w * h) as usize);
        for row in 0..h {
            let py = (y + row) & (vram::VRAM_HEIGHT as u32 - 1);
            for col in 0..w {
                let px = (x + col) & (vram::VRAM_WIDTH as u32 - 1);
                buf.push(cpu_words[(py * vram::VRAM_WIDTH as u32 + px) as usize]);
            }
        }
        let _ = self.vram.upload_rect(x & (vram::VRAM_WIDTH as u32 - 1),
                                       y & (vram::VRAM_HEIGHT as u32 - 1),
                                       w.min(vram::VRAM_WIDTH as u32),
                                       h.min(vram::VRAM_HEIGHT as u32),
                                       &buf);
    }

    /// Replay one GP0 packet captured by `enable_pixel_tracer` on
    /// the CPU side. Updates state for `0xE1..=0xE6`, dispatches a
    /// compute primitive for draw / fill / copy commands.
    pub fn replay_packet(&mut self, entry: &GpuCmdLogEntry) {
        let op = entry.opcode;
        let fifo = &entry.fifo[..];
        // Map cmd_log opcodes to the right path. Mirrors the
        // dispatch table in `emulator-core::Gpu::execute_gp0_packet`
        // / `execute_gp0_single`.
        match op {
            // ---------- State changes ----------
            0xE1 => self.handle_e1(fifo),
            0xE2 => self.handle_e2(fifo),
            0xE3 => self.handle_e3(fifo),
            0xE4 => self.handle_e4(fifo),
            0xE5 => self.handle_e5(fifo),
            0xE6 => self.handle_e6(fifo),

            // ---------- Fill ----------
            0x02 => self.handle_fill(fifo),

            // ---------- Mono triangle / quad ----------
            0x20..=0x23 => self.handle_mono_tri(fifo),
            0x28..=0x2B => self.handle_mono_quad(fifo),

            // ---------- Tex triangle / quad ----------
            0x24..=0x27 => self.handle_tex_tri(fifo),
            0x2C..=0x2F => self.handle_tex_quad(fifo),

            // ---------- Shaded triangle / quad ----------
            0x30..=0x33 => self.handle_shaded_tri(fifo),
            0x38..=0x3B => self.handle_shaded_quad(fifo),

            // ---------- Shaded textured triangle / quad ----------
            0x34..=0x37 => self.handle_shaded_tex_tri(fifo),
            0x3C..=0x3F => self.handle_shaded_tex_quad(fifo),

            // ---------- Rectangle variants ----------
            0x60..=0x63 => self.handle_mono_rect_variable(fifo),
            0x64..=0x67 => self.handle_tex_rect_variable(fifo),
            0x68..=0x6B => self.handle_mono_rect_fixed(fifo, 1, 1),
            0x6C..=0x6F => self.handle_tex_rect_fixed(fifo, 1, 1),
            0x70..=0x73 => self.handle_mono_rect_fixed(fifo, 8, 8),
            0x74..=0x77 => self.handle_tex_rect_fixed(fifo, 8, 8),
            0x78..=0x7B => self.handle_mono_rect_fixed(fifo, 16, 16),
            0x7C..=0x7F => self.handle_tex_rect_fixed(fifo, 16, 16),

            // ---------- VRAM-to-VRAM copy ----------
            0x80..=0x9F => self.handle_vram_copy(fifo),

            // CPU-to-VRAM upload (0xA0..=0xBF): the cmd packet is
            // 3 words but the pixel data isn't in cmd_log — it
            // streams via `ingest_vram_upload_word` which doesn't
            // record. We rely on `sync_vram_from_cpu` at frame
            // start to pick up the data. Nothing to dispatch here.
            0xA0..=0xBF => {}
            // VRAM-to-CPU readback: only writes back to a host
            // buffer the game polls; no VRAM mutation, nothing to
            // dispatch on the compute side.
            0xC0..=0xDF => {}

            // NOPs and clear-cache are also no-ops for rendering.
            0x00 | 0x01 | 0x03..=0x1E => {}

            // Lines / polylines: not yet ported. Track count.
            other => {
                *self.unhandled.entry(other).or_insert(0) += 1;
            }
        }
    }

    // ========== State change handlers ==========

    fn handle_e1(&mut self, fifo: &[u32]) {
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        let tpage_x = (word & 0x0F) * 64;
        let tpage_y: u32 = if (word >> 4) & 1 != 0 { 256 } else { 0 };
        let tex_depth = (word >> 7) & 0x3;
        // Texture-window state is preserved across primitives but
        // our `Tpage::new` reset it. Re-apply.
        let prev_tw = (
            self.state.tpage.tex_window_mask_x,
            self.state.tpage.tex_window_mask_y,
            self.state.tpage.tex_window_off_x,
            self.state.tpage.tex_window_off_y,
        );
        self.state.tpage = Tpage::new(tpage_x, tpage_y, tex_depth);
        self.state.tpage.tex_window_mask_x = prev_tw.0;
        self.state.tpage.tex_window_mask_y = prev_tw.1;
        self.state.tpage.tex_window_off_x = prev_tw.2;
        self.state.tpage.tex_window_off_y = prev_tw.3;
        self.state.tex_blend_mode = decode_blend_mode(word >> 5);
        self.state.dither = (word >> 9) & 1 != 0;
        self.state.flip_x = (word >> 12) & 1 != 0;
        self.state.flip_y = (word >> 13) & 1 != 0;
    }

    fn handle_e2(&mut self, fifo: &[u32]) {
        // GP0 0xE2 — texture window. Per PSX-SPX, the host stores
        // mask×8 / offset×8 (pre-multiplied for the rasterizer).
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        let mask_x = (word & 0x1F) * 8;
        let mask_y = ((word >> 5) & 0x1F) * 8;
        let off_x = ((word >> 10) & 0x1F) * 8;
        let off_y = ((word >> 15) & 0x1F) * 8;
        self.state.tpage.tex_window_mask_x = mask_x;
        self.state.tpage.tex_window_mask_y = mask_y;
        self.state.tpage.tex_window_off_x = off_x;
        self.state.tpage.tex_window_off_y = off_y;
    }

    fn handle_e3(&mut self, fifo: &[u32]) {
        // Drawing area top-left.
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        self.state.draw_area.left = (word & 0x3FF) as i32;
        self.state.draw_area.top = ((word >> 10) & 0x1FF) as i32;
    }

    fn handle_e4(&mut self, fifo: &[u32]) {
        // Drawing area bottom-right.
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        self.state.draw_area.right = (word & 0x3FF) as i32;
        self.state.draw_area.bottom = ((word >> 10) & 0x1FF) as i32;
    }

    fn handle_e5(&mut self, fifo: &[u32]) {
        // Drawing offset.
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        self.state.draw_offset_x = sign_extend_11((word & 0x7FF) as i32);
        self.state.draw_offset_y = sign_extend_11(((word >> 11) & 0x7FF) as i32);
    }

    fn handle_e6(&mut self, fifo: &[u32]) {
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        self.state.mask_set = word & 1 != 0;
        self.state.mask_check = word & 2 != 0;
    }

    // ========== Fill ==========

    fn handle_fill(&mut self, fifo: &[u32]) {
        if fifo.len() < 3 {
            return;
        }
        let cmd = fifo[0];
        let xy = fifo[1];
        let wh = fifo[2];
        let x = xy & 0x3FF;
        let y = (xy >> 16) & 0x1FF;
        let w = wh & 0x3FF;
        let h = (wh >> 16) & 0x1FF;
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let fill = Fill::new((x, y), (w, h), color);
        self.rasterizer.dispatch_fill(&self.vram, &fill);
    }

    // ========== Mono triangle / quad ==========

    fn mono_blend_mode_and_flags(&self, cmd: u32) -> (PrimFlags, BlendMode) {
        let mut flags = self.state.base_flags();
        if is_semi_trans(cmd) {
            flags |= PrimFlags::SEMI_TRANS;
        }
        (flags, self.state.tex_blend_mode)
    }

    fn handle_mono_tri(&mut self, fifo: &[u32]) {
        if fifo.len() < 4 {
            return;
        }
        let cmd = fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let v1 = decode_vertex(&self.state, fifo[2]);
        let v2 = decode_vertex(&self.state, fifo[3]);
        let (flags, mode) = self.mono_blend_mode_and_flags(cmd);
        let tri = MonoTri::new(v0, v1, v2, color, flags, mode);
        self.rasterizer
            .dispatch_mono_tri_scanline(&self.vram, &tri, &self.state.draw_area);
    }

    fn handle_mono_quad(&mut self, fifo: &[u32]) {
        if fifo.len() < 5 {
            return;
        }
        let cmd = fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let v1 = decode_vertex(&self.state, fifo[2]);
        let v2 = decode_vertex(&self.state, fifo[3]);
        let v3 = decode_vertex(&self.state, fifo[4]);
        let (flags, mode) = self.mono_blend_mode_and_flags(cmd);
        // Quad → 2 triangles: (v0, v1, v2) + (v1, v3, v2). Same
        // split order the CPU rasterizer uses.
        let t1 = MonoTri::new(v0, v1, v2, color, flags, mode);
        let t2 = MonoTri::new(v1, v3, v2, color, flags, mode);
        self.rasterizer
            .dispatch_mono_tri_scanline(&self.vram, &t1, &self.state.draw_area);
        self.rasterizer
            .dispatch_mono_tri_scanline(&self.vram, &t2, &self.state.draw_area);
    }

    // ========== Tex triangle / quad ==========

    fn tex_flags_and_mode(&self, cmd: u32) -> (PrimFlags, BlendMode) {
        let mut flags = self.state.base_flags();
        if is_raw_texture(cmd) {
            flags |= PrimFlags::RAW_TEXTURE;
        }
        if is_semi_trans(cmd) {
            flags |= PrimFlags::SEMI_TRANS;
        }
        (flags, self.state.tex_blend_mode)
    }

    fn handle_tex_tri(&mut self, fifo: &[u32]) {
        if fifo.len() < 7 {
            return;
        }
        let cmd = fifo[0];
        let v0 = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let (clut_x, clut_y) = decode_clut(fifo[2]);
        let v1 = decode_vertex(&self.state, fifo[3]);
        let uv1 = decode_uv(fifo[4]);
        // The tpage word in UV1's high half overrides the global
        // tpage state for THIS primitive only. CPU mirrors it via
        // `apply_primitive_tpage`, which permanently updates state.
        apply_primitive_tpage(&mut self.state, fifo[4]);
        let v2 = decode_vertex(&self.state, fifo[5]);
        let uv2 = decode_uv(fifo[6]);
        let tint = decode_tint(cmd & 0x00FF_FFFF);
        let (flags, mode) = self.tex_flags_and_mode(cmd);
        let tri = TexTri::new(
            v0, v1, v2, uv0, uv1, uv2, clut_x, clut_y, tint, flags, mode,
        );
        self.rasterizer
            .dispatch_tex_tri_scanline(&self.vram, &tri, &self.state.tpage, &self.state.draw_area);
    }

    fn handle_tex_quad(&mut self, fifo: &[u32]) {
        if fifo.len() < 9 {
            return;
        }
        let cmd = fifo[0];
        let v0 = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let (clut_x, clut_y) = decode_clut(fifo[2]);
        let v1 = decode_vertex(&self.state, fifo[3]);
        let uv1 = decode_uv(fifo[4]);
        apply_primitive_tpage(&mut self.state, fifo[4]);
        let v2 = decode_vertex(&self.state, fifo[5]);
        let uv2 = decode_uv(fifo[6]);
        let v3 = decode_vertex(&self.state, fifo[7]);
        let uv3 = decode_uv(fifo[8]);
        let tint = decode_tint(cmd & 0x00FF_FFFF);
        let (flags, mode) = self.tex_flags_and_mode(cmd);

        // Phase C bug fix: when the quad is axis-aligned the CPU
        // rasterizer skips the triangle split and runs a bilinear
        // UV walk over all four corners. Triangle-split + bary
        // interpolation produces different pixels for non-affine
        // UV layouts (Tekken 3 character draws hit this). Mirror
        // the CPU's fast path here so VRAM stays in sync.
        if TexQuadBilinear::is_axis_aligned(v0, v1, v2, v3) {
            let q = TexQuadBilinear::new(
                v0, v1, v2, v3, uv0, uv1, uv2, uv3, clut_x, clut_y, tint, flags, mode,
            );
            self.rasterizer.dispatch_tex_quad_bilinear(
                &self.vram,
                &q,
                &self.state.tpage,
                &self.state.draw_area,
            );
            return;
        }

        // Non-axis-aligned: fall back to the same triangle split
        // the CPU uses (v1, v3, v2) then (v0, v1, v2).
        let t1 = TexTri::new(
            v1, v3, v2, uv1, uv3, uv2, clut_x, clut_y, tint, flags, mode,
        );
        let t2 = TexTri::new(
            v0, v1, v2, uv0, uv1, uv2, clut_x, clut_y, tint, flags, mode,
        );
        self.rasterizer
            .dispatch_tex_tri_scanline(&self.vram, &t1, &self.state.tpage, &self.state.draw_area);
        self.rasterizer
            .dispatch_tex_tri_scanline(&self.vram, &t2, &self.state.tpage, &self.state.draw_area);
    }

    // ========== Shaded triangle / quad ==========

    fn handle_shaded_tri(&mut self, fifo: &[u32]) {
        if fifo.len() < 6 {
            return;
        }
        let c0 = decode_tint(fifo[0] & 0x00FF_FFFF);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let c1 = decode_tint(fifo[2] & 0x00FF_FFFF);
        let v1 = decode_vertex(&self.state, fifo[3]);
        let c2 = decode_tint(fifo[4] & 0x00FF_FFFF);
        let v2 = decode_vertex(&self.state, fifo[5]);
        let (flags, mode) = self.mono_blend_mode_and_flags(fifo[0]);
        let tri = ShadedTri::new(v0, v1, v2, c0, c1, c2, flags, mode);
        self.rasterizer
            .dispatch_shaded_tri_scanline(&self.vram, &tri, &self.state.draw_area);
    }

    fn handle_shaded_quad(&mut self, fifo: &[u32]) {
        if fifo.len() < 8 {
            return;
        }
        let c0 = decode_tint(fifo[0] & 0x00FF_FFFF);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let c1 = decode_tint(fifo[2] & 0x00FF_FFFF);
        let v1 = decode_vertex(&self.state, fifo[3]);
        let c2 = decode_tint(fifo[4] & 0x00FF_FFFF);
        let v2 = decode_vertex(&self.state, fifo[5]);
        let c3 = decode_tint(fifo[6] & 0x00FF_FFFF);
        let v3 = decode_vertex(&self.state, fifo[7]);
        let (flags, mode) = self.mono_blend_mode_and_flags(fifo[0]);
        let t1 = ShadedTri::new(v0, v1, v2, c0, c1, c2, flags, mode);
        let t2 = ShadedTri::new(v1, v3, v2, c1, c3, c2, flags, mode);
        self.rasterizer
            .dispatch_shaded_tri_scanline(&self.vram, &t1, &self.state.draw_area);
        self.rasterizer
            .dispatch_shaded_tri_scanline(&self.vram, &t2, &self.state.draw_area);
    }

    // ========== Shaded textured triangle / quad ==========

    fn handle_shaded_tex_tri(&mut self, fifo: &[u32]) {
        if fifo.len() < 9 {
            return;
        }
        let c0 = decode_tint(fifo[0] & 0x00FF_FFFF);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let (clut_x, clut_y) = decode_clut(fifo[2]);
        let c1 = decode_tint(fifo[3] & 0x00FF_FFFF);
        let v1 = decode_vertex(&self.state, fifo[4]);
        let uv1 = decode_uv(fifo[5]);
        apply_primitive_tpage(&mut self.state, fifo[5]);
        let c2 = decode_tint(fifo[6] & 0x00FF_FFFF);
        let v2 = decode_vertex(&self.state, fifo[7]);
        let uv2 = decode_uv(fifo[8]);
        let (flags, mode) = self.tex_flags_and_mode(fifo[0]);
        let tri = ShadedTexTri::new(
            v0, v1, v2, c0, c1, c2, uv0, uv1, uv2, clut_x, clut_y, flags, mode,
        );
        self.rasterizer.dispatch_shaded_tex_tri_scanline(
            &self.vram,
            &tri,
            &self.state.tpage,
            &self.state.draw_area,
        );
    }

    fn handle_shaded_tex_quad(&mut self, fifo: &[u32]) {
        if fifo.len() < 12 {
            return;
        }
        let c0 = decode_tint(fifo[0] & 0x00FF_FFFF);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let (clut_x, clut_y) = decode_clut(fifo[2]);
        let c1 = decode_tint(fifo[3] & 0x00FF_FFFF);
        let v1 = decode_vertex(&self.state, fifo[4]);
        let uv1 = decode_uv(fifo[5]);
        apply_primitive_tpage(&mut self.state, fifo[5]);
        let c2 = decode_tint(fifo[6] & 0x00FF_FFFF);
        let v2 = decode_vertex(&self.state, fifo[7]);
        let uv2 = decode_uv(fifo[8]);
        let c3 = decode_tint(fifo[9] & 0x00FF_FFFF);
        let v3 = decode_vertex(&self.state, fifo[10]);
        let uv3 = decode_uv(fifo[11]);
        let (flags, mode) = self.tex_flags_and_mode(fifo[0]);
        let t1 = ShadedTexTri::new(
            v1, v3, v2, c1, c3, c2, uv1, uv3, uv2, clut_x, clut_y, flags, mode,
        );
        let t2 = ShadedTexTri::new(
            v0, v1, v2, c0, c1, c2, uv0, uv1, uv2, clut_x, clut_y, flags, mode,
        );
        self.rasterizer.dispatch_shaded_tex_tri_scanline(
            &self.vram,
            &t1,
            &self.state.tpage,
            &self.state.draw_area,
        );
        self.rasterizer.dispatch_shaded_tex_tri_scanline(
            &self.vram,
            &t2,
            &self.state.tpage,
            &self.state.draw_area,
        );
    }

    // ========== Rectangle handlers ==========

    fn rect_flags_and_mode(&self, cmd: u32) -> (PrimFlags, BlendMode) {
        let mut flags = self.state.base_flags();
        if is_semi_trans(cmd) {
            flags |= PrimFlags::SEMI_TRANS;
        }
        (flags, self.state.tex_blend_mode)
    }

    fn handle_mono_rect_variable(&mut self, fifo: &[u32]) {
        if fifo.len() < 3 {
            return;
        }
        let cmd = fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let pos = fifo[1];
        let xy = decode_vertex(&self.state, pos);
        let size = fifo[2];
        let w = (size & 0xFFFF) as u32;
        let h = ((size >> 16) & 0xFFFF) as u32;
        let (flags, mode) = self.rect_flags_and_mode(cmd);
        let rect = MonoRect::new(xy, (w, h), color, flags, mode);
        self.rasterizer
            .dispatch_mono_rect(&self.vram, &rect, &self.state.draw_area);
    }

    fn handle_mono_rect_fixed(&mut self, fifo: &[u32], w: u32, h: u32) {
        if fifo.len() < 2 {
            return;
        }
        let cmd = fifo[0];
        let color = rgb24_to_bgr15(cmd & 0x00FF_FFFF);
        let xy = decode_vertex(&self.state, fifo[1]);
        let (flags, mode) = self.rect_flags_and_mode(cmd);
        let rect = MonoRect::new(xy, (w, h), color, flags, mode);
        self.rasterizer
            .dispatch_mono_rect(&self.vram, &rect, &self.state.draw_area);
    }

    fn tex_rect_flags_and_mode(&self, cmd: u32) -> (PrimFlags, BlendMode) {
        let mut flags = self.state.rect_flip_flags();
        if is_raw_texture(cmd) {
            flags |= PrimFlags::RAW_TEXTURE;
        }
        if is_semi_trans(cmd) {
            flags |= PrimFlags::SEMI_TRANS;
        }
        (flags, self.state.tex_blend_mode)
    }

    fn handle_tex_rect_variable(&mut self, fifo: &[u32]) {
        if fifo.len() < 4 {
            return;
        }
        let cmd = fifo[0];
        let xy = decode_vertex(&self.state, fifo[1]);
        let uv = decode_uv(fifo[2]);
        let (clut_x, clut_y) = decode_clut(fifo[2]);
        let size = fifo[3];
        let w = (size & 0xFFFF) as u32;
        let h = ((size >> 16) & 0xFFFF) as u32;
        let tint = decode_tint(cmd & 0x00FF_FFFF);
        let (flags, mode) = self.tex_rect_flags_and_mode(cmd);
        let rect = TexRect::new(xy, (w, h), uv, clut_x, clut_y, tint, flags, mode);
        self.rasterizer
            .dispatch_tex_rect(&self.vram, &rect, &self.state.tpage, &self.state.draw_area);
    }

    fn handle_tex_rect_fixed(&mut self, fifo: &[u32], w: u32, h: u32) {
        if fifo.len() < 3 {
            return;
        }
        let cmd = fifo[0];
        let xy = decode_vertex(&self.state, fifo[1]);
        let uv = decode_uv(fifo[2]);
        let (clut_x, clut_y) = decode_clut(fifo[2]);
        let tint = decode_tint(cmd & 0x00FF_FFFF);
        let (flags, mode) = self.tex_rect_flags_and_mode(cmd);
        let rect = TexRect::new(xy, (w, h), uv, clut_x, clut_y, tint, flags, mode);
        self.rasterizer
            .dispatch_tex_rect(&self.vram, &rect, &self.state.tpage, &self.state.draw_area);
    }

    // ========== VRAM-to-VRAM copy ==========

    fn handle_vram_copy(&mut self, fifo: &[u32]) {
        if fifo.len() < 4 {
            return;
        }
        let src = fifo[1];
        let dst = fifo[2];
        let wh = fifo[3];
        let sx = src & 0xFFFF;
        let sy = (src >> 16) & 0xFFFF;
        let dx = dst & 0xFFFF;
        let dy = (dst >> 16) & 0xFFFF;
        let raw_w = wh & 0xFFFF;
        let raw_h = (wh >> 16) & 0xFFFF;
        let w = if raw_w == 0 { 1024 } else { raw_w };
        let h = if raw_h == 0 { 512 } else { raw_h };
        self.rasterizer
            .dispatch_vram_copy(&self.vram, (sx, sy), (dx, dy), (w, h));
    }
}
