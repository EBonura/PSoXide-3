//! `cmd_log` → `Vec<HwVertex>` translator.
//!
//! Walks each frame's GP0 packet stream, tracks `ReplayState` for
//! state setters (`0xE1..=0xE6`), and emits triangles for the
//! primitive opcodes the renderer currently supports. Phase 1
//! supports flat-color mono tris (`0x20..=0x23`) and mono quads
//! (`0x28..=0x2B`, decomposed to two tris with the same winding
//! the CPU rasterizer uses).
//!
//! Other primitive opcodes are silently skipped — their state
//! setters still update `ReplayState` so that when later phases
//! enable them the tpage / draw_offset / draw_area they observe
//! is correct.

use emulator_core::gpu::GpuCmdLogEntry;
use psx_gpu_compute::decode::{
    apply_primitive_tpage, decode_tint, decode_vertex, rgb24_to_bgr15, ReplayState,
};

use crate::pipeline::HwVertex;

pub struct Translator {
    state: ReplayState,
    /// Reused per call so we don't allocate every frame.
    vertices: Vec<HwVertex>,
}

impl Translator {
    pub fn new() -> Self {
        Self {
            state: ReplayState::new(),
            vertices: Vec::with_capacity(4 * 1024),
        }
    }

    /// Walk `cmd_log`, return the vertex stream for this frame.
    /// The returned slice borrows from `self`; copy out before the
    /// next call.
    pub fn translate(&mut self, log: &[GpuCmdLogEntry]) -> &[HwVertex] {
        self.vertices.clear();
        for entry in log {
            self.process(entry);
        }
        &self.vertices
    }

    fn process(&mut self, entry: &GpuCmdLogEntry) {
        let fifo = &entry.fifo[..];
        match entry.opcode {
            // ---------- State setters (always applied) ----------
            0xE1 => self.handle_e1(fifo),
            0xE2 => self.handle_e2(fifo),
            0xE3 => self.handle_e3(fifo),
            0xE4 => self.handle_e4(fifo),
            0xE5 => self.handle_e5(fifo),
            0xE6 => self.handle_e6(fifo),

            // ---------- Phase 1 primitives ----------
            0x20..=0x23 => self.emit_mono_tri(fifo),
            0x28..=0x2B => self.emit_mono_quad(fifo),

            // ---------- Phase 2+ ---------- (silently skipped)
            // 0x02         => fill rect
            // 0x24..=0x27  => tex tri
            // 0x2C..=0x2F  => tex quad
            // 0x30..=0x33  => shaded tri
            // 0x34..=0x37  => shaded-tex tri
            // 0x38..=0x3B  => shaded quad
            // 0x3C..=0x3F  => shaded-tex quad
            // 0x40..=0x5F  => lines / polylines
            // 0x60..=0x7F  => rectangles
            // 0x80..=0x9F  => VRAM-VRAM copy
            // 0xA0..=0xBF  => CPU-VRAM upload
            // 0xC0..=0xDF  => VRAM-CPU readback (no draws)
            _ => {}
        }
    }

    // -------- State-setter handlers (mirror psx-gpu-compute::replay) --------

    fn handle_e1(&mut self, fifo: &[u32]) {
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        // Re-applying the tpage word also resets `tex_blend_mode`;
        // synthesise a "fake" UV1-high-half word that just carries
        // the low 16 bits of E1 in its high half.
        apply_primitive_tpage(&mut self.state, word << 16);
        self.state.dither = (word >> 9) & 1 != 0;
        self.state.flip_x = (word >> 12) & 1 != 0;
        self.state.flip_y = (word >> 13) & 1 != 0;
    }

    fn handle_e2(&mut self, fifo: &[u32]) {
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
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        self.state.draw_area.left = (word & 0x3FF) as i32;
        self.state.draw_area.top = ((word >> 10) & 0x1FF) as i32;
    }

    fn handle_e4(&mut self, fifo: &[u32]) {
        if fifo.is_empty() {
            return;
        }
        let word = fifo[0];
        self.state.draw_area.right = (word & 0x3FF) as i32;
        self.state.draw_area.bottom = ((word >> 10) & 0x1FF) as i32;
    }

    fn handle_e5(&mut self, fifo: &[u32]) {
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
        self.state.mask_set = (word & 1) != 0;
        self.state.mask_check = (word & 2) != 0;
    }

    // -------- Primitive emitters --------

    fn emit_mono_tri(&mut self, fifo: &[u32]) {
        if fifo.len() < 4 {
            return;
        }
        let cmd = fifo[0];
        let color = mono_color_rgba8(cmd);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let v1 = decode_vertex(&self.state, fifo[2]);
        let v2 = decode_vertex(&self.state, fifo[3]);
        self.push_tri(v0, v1, v2, color);
    }

    fn emit_mono_quad(&mut self, fifo: &[u32]) {
        if fifo.len() < 5 {
            return;
        }
        let cmd = fifo[0];
        let color = mono_color_rgba8(cmd);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let v1 = decode_vertex(&self.state, fifo[2]);
        let v2 = decode_vertex(&self.state, fifo[3]);
        let v3 = decode_vertex(&self.state, fifo[4]);
        // Match the CPU rasterizer's split order
        // (`Gpu::draw_monochrome_quad`): lower/right half first, then
        // upper/left, so pixels on the shared diagonal are owned by
        // (v0, v1, v2). Doesn't matter visually with opaque flat
        // color but keeps Phase 4 semi-trans behaviour identical.
        self.push_tri(v1, v3, v2, color);
        self.push_tri(v0, v1, v2, color);
    }

    fn push_tri(&mut self, v0: (i32, i32), v1: (i32, i32), v2: (i32, i32), color: [u8; 4]) {
        for v in [v0, v1, v2] {
            self.vertices.push(HwVertex {
                pos: [v.0 as i16, v.1 as i16],
                color,
                uv: [0, 0],
                flags: 0,
            });
        }
    }
}

impl Default for Translator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a GP0 cmd word's low 24 bits into an RGBA8 tuple. PSX
/// writes BGR15 to VRAM; the CPU rasterizer goes via
/// `rgb24_to_bgr15` then `plot_pixel`, which is what shows up on
/// screen. For the HW renderer we keep RGB8 because the output
/// texture is `Rgba8UnormSrgb` — the channel reduction to 5 bits
/// happens on the CPU side, the HW side renders the full-precision
/// RGB. Phase 7 may add a knob to clamp to PSX 5-bit precision for
/// strict-look games.
fn mono_color_rgba8(cmd: u32) -> [u8; 4] {
    let (r, g, b) = decode_tint(cmd & 0x00FF_FFFF);
    let _ = rgb24_to_bgr15; // imported for future BGR15 round-trip
    [r, g, b, 0xFF]
}

fn sign_extend_11(v: i32) -> i32 {
    if v & 0x400 != 0 {
        v | !0x7FF
    } else {
        v & 0x7FF
    }
}
