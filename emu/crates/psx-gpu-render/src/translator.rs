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
    apply_primitive_tpage, decode_clut, decode_tint, decode_uv, decode_vertex, is_raw_texture,
    is_semi_trans, rgb24_to_bgr15, ReplayState,
};

use crate::pipeline::{flags as fbits, HwVertex};

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

            // ---------- Phase 2 primitives ----------
            0x24..=0x27 => self.emit_tex_tri(fifo),
            0x2C..=0x2F => self.emit_tex_quad(fifo),

            // ---------- Phase 3 primitives ----------
            // Shaded (Gouraud) tris + quads.
            0x30..=0x33 => self.emit_shaded_tri(fifo),
            0x38..=0x3B => self.emit_shaded_quad(fifo),
            // Shaded + textured.
            0x34..=0x37 => self.emit_shaded_tex_tri(fifo),
            0x3C..=0x3F => self.emit_shaded_tex_quad(fifo),
            // Mono rectangles (variable + 1x1 / 8x8 / 16x16 fixed).
            0x60..=0x63 => self.emit_mono_rect_variable(fifo),
            0x68..=0x6B => self.emit_mono_rect_fixed(fifo, 1, 1),
            0x70..=0x73 => self.emit_mono_rect_fixed(fifo, 8, 8),
            0x78..=0x7B => self.emit_mono_rect_fixed(fifo, 16, 16),
            // Textured rectangles (variable + fixed sizes).
            0x64..=0x67 => self.emit_tex_rect_variable(fifo),
            0x6C..=0x6F => self.emit_tex_rect_fixed(fifo, 1, 1),
            0x74..=0x77 => self.emit_tex_rect_fixed(fifo, 8, 8),
            0x7C..=0x7F => self.emit_tex_rect_fixed(fifo, 16, 16),

            // ---------- Phase 4+ ---------- (silently skipped)
            // 0x02         => fill rect
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

    // ----- Phase 2: textured tris + quads -----

    /// `0x24..=0x27` — textured triangle. Packet:
    ///   `[cmd+tint, v0, uv0+clut, v1, uv1+tpage, v2, uv2]`
    /// uv1's high half is the active tpage word; consume it via
    /// `apply_primitive_tpage` so subsequent primitives see the
    /// updated tpage.
    fn emit_tex_tri(&mut self, fifo: &[u32]) {
        if fifo.len() < 7 {
            return;
        }
        let cmd = fifo[0];
        let v0 = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let clut = decode_clut(fifo[2]);
        let v1 = decode_vertex(&self.state, fifo[3]);
        let uv1 = decode_uv(fifo[4]);
        apply_primitive_tpage(&mut self.state, fifo[4]);
        let v2 = decode_vertex(&self.state, fifo[5]);
        let uv2 = decode_uv(fifo[6]);

        let prim_flags = self.tex_prim_flags(cmd, clut);
        let color = tex_tint(cmd);
        self.push_tex_tri(v0, uv0, v1, uv1, v2, uv2, color, prim_flags);
    }

    /// `0x2C..=0x2F` — textured quad. Packet:
    ///   `[cmd+tint, v0, uv0+clut, v1, uv1+tpage, v2, uv2, v3, uv3]`
    /// Decomposes to two triangles using the same winding the CPU
    /// rasterizer's `draw_textured_quad` uses (`v0,v1,v2` then
    /// `v1,v3,v2`), so semi-trans / mask behaviour stays
    /// pixel-equivalent in later phases.
    fn emit_tex_quad(&mut self, fifo: &[u32]) {
        if fifo.len() < 9 {
            return;
        }
        let cmd = fifo[0];
        let v0 = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let clut = decode_clut(fifo[2]);
        let v1 = decode_vertex(&self.state, fifo[3]);
        let uv1 = decode_uv(fifo[4]);
        apply_primitive_tpage(&mut self.state, fifo[4]);
        let v2 = decode_vertex(&self.state, fifo[5]);
        let uv2 = decode_uv(fifo[6]);
        let v3 = decode_vertex(&self.state, fifo[7]);
        let uv3 = decode_uv(fifo[8]);

        let prim_flags = self.tex_prim_flags(cmd, clut);
        let color = tex_tint(cmd);

        self.push_tex_tri(v0, uv0, v1, uv1, v2, uv2, color, prim_flags);
        self.push_tex_tri(v1, uv1, v3, uv3, v2, uv2, color, prim_flags);
    }

    /// Pack the per-primitive state setter bits + vertex flags
    /// `bits` into the format the shader expects. `clut` is in
    /// PSX VRAM pixels.
    fn tex_prim_flags(&self, cmd: u32, clut: (u32, u32)) -> u32 {
        let tp = &self.state.tpage;
        let depth = tp.tex_depth;
        let mut flags = fbits::TEXTURED;
        flags |= fbits::pack_tpage(tp.tpage_x, tp.tpage_y, depth);
        flags |= fbits::pack_clut(clut.0, clut.1);
        if is_raw_texture(cmd) {
            flags |= fbits::RAW_TEXTURE;
        }
        if is_semi_trans(cmd) {
            // Phase 4 wires this up to a blend-state branch. For
            // now it's tracked but the pipeline's blend state is
            // still REPLACE — semi-trans textured pixels render
            // as opaque. Visible in some FMVs / particle
            // effects; will land with Phase 4.
            flags |= fbits::SEMI_TRANS;
        }
        flags
    }

    fn push_tex_tri(
        &mut self,
        v0: (i32, i32),
        uv0: (u8, u8),
        v1: (i32, i32),
        uv1: (u8, u8),
        v2: (i32, i32),
        uv2: (u8, u8),
        color: [u8; 4],
        prim_flags: u32,
    ) {
        let make = |v: (i32, i32), uv: (u8, u8)| HwVertex {
            pos: [v.0 as i16, v.1 as i16],
            color,
            uv: [uv.0 as u16, uv.1 as u16],
            flags: prim_flags,
        };
        self.vertices.push(make(v0, uv0));
        self.vertices.push(make(v1, uv1));
        self.vertices.push(make(v2, uv2));
    }

    // ----- Phase 3: shaded (Gouraud) tris + quads -----

    /// `0x30..=0x33` — Gouraud-shaded triangle.
    /// Words: `[cmd+c0, v0, c1, v1, c2, v2]`.
    /// The fragment shader interpolates `color` linearly across
    /// the tri; we just push three different vertex colours.
    fn emit_shaded_tri(&mut self, fifo: &[u32]) {
        if fifo.len() < 6 {
            return;
        }
        let c0 = mono_color_rgba8(fifo[0]);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let c1 = mono_color_rgba8(fifo[2]);
        let v1 = decode_vertex(&self.state, fifo[3]);
        let c2 = mono_color_rgba8(fifo[4]);
        let v2 = decode_vertex(&self.state, fifo[5]);
        self.push_shaded_tri(v0, c0, v1, c1, v2, c2, 0);
    }

    /// `0x38..=0x3B` — Gouraud-shaded quad.
    /// Words: `[cmd+c0, v0, c1, v1, c2, v2, c3, v3]`.
    fn emit_shaded_quad(&mut self, fifo: &[u32]) {
        if fifo.len() < 8 {
            return;
        }
        let c0 = mono_color_rgba8(fifo[0]);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let c1 = mono_color_rgba8(fifo[2]);
        let v1 = decode_vertex(&self.state, fifo[3]);
        let c2 = mono_color_rgba8(fifo[4]);
        let v2 = decode_vertex(&self.state, fifo[5]);
        let c3 = mono_color_rgba8(fifo[6]);
        let v3 = decode_vertex(&self.state, fifo[7]);
        // Same split as `draw_monochrome_quad` for diagonal-pixel
        // ownership consistency.
        self.push_shaded_tri(v1, c1, v3, c3, v2, c2, 0);
        self.push_shaded_tri(v0, c0, v1, c1, v2, c2, 0);
    }

    /// `0x34..=0x37` — Gouraud + textured triangle.
    /// Words: `[cmd+c0, v0, uv0+clut, c1, v1, uv1+tpage, c2, v2, uv2]`.
    fn emit_shaded_tex_tri(&mut self, fifo: &[u32]) {
        if fifo.len() < 9 {
            return;
        }
        let cmd = fifo[0];
        let c0 = mono_color_rgba8(cmd);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let clut = decode_clut(fifo[2]);
        let c1 = mono_color_rgba8(fifo[3]);
        let v1 = decode_vertex(&self.state, fifo[4]);
        let uv1 = decode_uv(fifo[5]);
        apply_primitive_tpage(&mut self.state, fifo[5]);
        let c2 = mono_color_rgba8(fifo[6]);
        let v2 = decode_vertex(&self.state, fifo[7]);
        let uv2 = decode_uv(fifo[8]);

        let prim_flags = self.tex_prim_flags(cmd, clut);
        self.push_tex_tri_shaded(v0, uv0, c0, v1, uv1, c1, v2, uv2, c2, prim_flags);
    }

    /// `0x3C..=0x3F` — Gouraud + textured quad. Words:
    /// `[cmd+c0, v0, uv0+clut, c1, v1, uv1+tpage, c2, v2, uv2,
    ///   c3, v3, uv3]`.
    fn emit_shaded_tex_quad(&mut self, fifo: &[u32]) {
        if fifo.len() < 12 {
            return;
        }
        let cmd = fifo[0];
        let c0 = mono_color_rgba8(cmd);
        let v0 = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let clut = decode_clut(fifo[2]);
        let c1 = mono_color_rgba8(fifo[3]);
        let v1 = decode_vertex(&self.state, fifo[4]);
        let uv1 = decode_uv(fifo[5]);
        apply_primitive_tpage(&mut self.state, fifo[5]);
        let c2 = mono_color_rgba8(fifo[6]);
        let v2 = decode_vertex(&self.state, fifo[7]);
        let uv2 = decode_uv(fifo[8]);
        let c3 = mono_color_rgba8(fifo[9]);
        let v3 = decode_vertex(&self.state, fifo[10]);
        let uv3 = decode_uv(fifo[11]);

        let prim_flags = self.tex_prim_flags(cmd, clut);
        self.push_tex_tri_shaded(v0, uv0, c0, v1, uv1, c1, v2, uv2, c2, prim_flags);
        self.push_tex_tri_shaded(v1, uv1, c1, v3, uv3, c3, v2, uv2, c2, prim_flags);
    }

    #[allow(clippy::too_many_arguments)]
    fn push_shaded_tri(
        &mut self,
        v0: (i32, i32),
        c0: [u8; 4],
        v1: (i32, i32),
        c1: [u8; 4],
        v2: (i32, i32),
        c2: [u8; 4],
        prim_flags: u32,
    ) {
        let make = |v: (i32, i32), c: [u8; 4]| HwVertex {
            pos: [v.0 as i16, v.1 as i16],
            color: c,
            uv: [0, 0],
            flags: prim_flags,
        };
        self.vertices.push(make(v0, c0));
        self.vertices.push(make(v1, c1));
        self.vertices.push(make(v2, c2));
    }

    #[allow(clippy::too_many_arguments)]
    fn push_tex_tri_shaded(
        &mut self,
        v0: (i32, i32),
        uv0: (u8, u8),
        c0: [u8; 4],
        v1: (i32, i32),
        uv1: (u8, u8),
        c1: [u8; 4],
        v2: (i32, i32),
        uv2: (u8, u8),
        c2: [u8; 4],
        prim_flags: u32,
    ) {
        let make = |v: (i32, i32), uv: (u8, u8), c: [u8; 4]| HwVertex {
            pos: [v.0 as i16, v.1 as i16],
            color: c,
            uv: [uv.0 as u16, uv.1 as u16],
            flags: prim_flags,
        };
        self.vertices.push(make(v0, uv0, c0));
        self.vertices.push(make(v1, uv1, c1));
        self.vertices.push(make(v2, uv2, c2));
    }

    // ----- Phase 3: rectangles -----

    /// `0x60..=0x63` — variable-size mono rect.
    /// Words: `[cmd+rgb24, xy, wh]`. Decomposes to two tris.
    fn emit_mono_rect_variable(&mut self, fifo: &[u32]) {
        if fifo.len() < 3 {
            return;
        }
        let cmd = fifo[0];
        let (x, y) = decode_vertex(&self.state, fifo[1]);
        let (w, h) = decode_wh(fifo[2]);
        self.push_mono_rect(cmd, x, y, w, h);
    }

    /// `0x68..=0x6B` (1×1), `0x70..=0x73` (8×8), `0x78..=0x7B` (16×16)
    /// — fixed-size mono rect. Words: `[cmd+rgb24, xy]`.
    fn emit_mono_rect_fixed(&mut self, fifo: &[u32], w: i32, h: i32) {
        if fifo.len() < 2 {
            return;
        }
        let cmd = fifo[0];
        let (x, y) = decode_vertex(&self.state, fifo[1]);
        self.push_mono_rect(cmd, x, y, w, h);
    }

    /// `0x64..=0x67` — variable-size textured rect.
    /// Words: `[cmd+tint, xy, uv+clut, wh]`. Tpage is the active
    /// state setter's; rectangles do NOT update tpage on their
    /// own (unlike textured polys' uv1).
    fn emit_tex_rect_variable(&mut self, fifo: &[u32]) {
        if fifo.len() < 4 {
            return;
        }
        let cmd = fifo[0];
        let (x, y) = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let clut = decode_clut(fifo[2]);
        let (w, h) = decode_wh(fifo[3]);
        self.push_tex_rect(cmd, x, y, w, h, uv0, clut);
    }

    /// `0x6C..=0x6F` (1×1), `0x74..=0x77` (8×8), `0x7C..=0x7F` (16×16).
    /// Words: `[cmd+tint, xy, uv+clut]`.
    fn emit_tex_rect_fixed(&mut self, fifo: &[u32], w: i32, h: i32) {
        if fifo.len() < 3 {
            return;
        }
        let cmd = fifo[0];
        let (x, y) = decode_vertex(&self.state, fifo[1]);
        let uv0 = decode_uv(fifo[2]);
        let clut = decode_clut(fifo[2]);
        self.push_tex_rect(cmd, x, y, w, h, uv0, clut);
    }

    fn push_mono_rect(&mut self, cmd: u32, x: i32, y: i32, w: i32, h: i32) {
        if w <= 0 || h <= 0 {
            return;
        }
        let color = mono_color_rgba8(cmd);
        let v00 = (x, y);
        let v10 = (x + w, y);
        let v01 = (x, y + h);
        let v11 = (x + w, y + h);
        // Two tris that cover the rect: (v00, v10, v01) + (v10, v11, v01).
        self.push_tri(v00, v10, v01, color);
        self.push_tri(v10, v11, v01, color);
    }

    fn push_tex_rect(
        &mut self,
        cmd: u32,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        uv0: (u8, u8),
        clut: (u32, u32),
    ) {
        if w <= 0 || h <= 0 {
            return;
        }
        let color = tex_tint(cmd);
        let prim_flags = self.tex_prim_flags(cmd, clut);
        // Sprite UVs step 1 texel per pixel — top-left at uv0,
        // top-right at uv0 + (w, 0), bottom-left at uv0 + (0, h),
        // bottom-right at uv0 + (w, h). Real PSX uses (w-1, h-1)
        // because UVs are inclusive; the GPU rasterizer interpolates
        // continuously across the quad, which gives subpixel UV at
        // high res — good for fractional scaling. The integer UV
        // ends up identical at native scale.
        // Wrap u8 max 255 — handled in the shader's `page_uv`.
        let u0 = uv0.0 as i32;
        let v0 = uv0.1 as i32;
        let uw = w;
        let uh = h;
        let uv_a = (u0 as u8, v0 as u8);
        let uv_b = ((u0 + uw) as u8, v0 as u8);
        let uv_c = (u0 as u8, (v0 + uh) as u8);
        let uv_d = ((u0 + uw) as u8, (v0 + uh) as u8);
        let p_a = (x, y);
        let p_b = (x + w, y);
        let p_c = (x, y + h);
        let p_d = (x + w, y + h);
        // Two tris covering the rect.
        self.push_tex_tri(p_a, uv_a, p_b, uv_b, p_c, uv_c, color, prim_flags);
        self.push_tex_tri(p_b, uv_b, p_d, uv_d, p_c, uv_c, color, prim_flags);
    }
}

/// Decode a `WH` word into `(w, h)`. Layout matches the CPU
/// rasterizer: low 16 bits = width, high 16 bits = height.
fn decode_wh(word: u32) -> (i32, i32) {
    let w = (word & 0xFFFF) as i32;
    let h = ((word >> 16) & 0xFFFF) as i32;
    (w, h)
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

/// Tint colour for a textured primitive. Same layout as the mono
/// case (low 24 bits of `cmd`). Used as the modulator on top of
/// the sampled texel; raw-texture mode skips this and uses
/// `(0x80, 0x80, 0x80)` semantics, but we leave the original
/// tint here and let the shader decide whether to modulate
/// based on the `RAW_TEXTURE` flag bit. Keeps the vertex format
/// uniform across raw / non-raw primitives.
fn tex_tint(cmd: u32) -> [u8; 4] {
    let (r, g, b) = decode_tint(cmd & 0x00FF_FFFF);
    [r, g, b, 0xFF]
}

fn sign_extend_11(v: i32) -> i32 {
    if v & 0x400 != 0 {
        v | !0x7FF
    } else {
        v & 0x7FF
    }
}
