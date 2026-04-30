//! PSX hardware renderer -- wgpu render pipeline that draws each
//! GP0 primitive at the internal-resolution multiple of native PSX
//! VRAM, producing fractional upscaling for free.
//!
//! ## Architecture
//!
//! The HW target is **VRAM-shaped**: a `(1024 · S) × (512 · S)`
//! texture (S = internal-resolution multiplier). PSX vertex coords
//! map directly into this VRAM space; the display shown to the user
//! is just a sub-rect read of the texture (the PSX `display_area`,
//! scaled by S). This mirrors real hardware -- the GPU draws into
//! persistent VRAM and the CRT scans out a window of it.
//!
//! Consequences:
//! - **VRAM persistence works for free.** A demo that draws once at
//!   boot and then just present-flips keeps its pixels because we
//!   never clear the target between frames.
//! - **Native↔Window is a single knob (S).** S=1 → 1024×512 texture
//!   → tiny PSX-native rasterisation; the display sub-rect (e.g.
//!   320×240) gets scaled up by Nearest at egui paint = "big crisp
//!   PSX pixels". S=N → N× rasterisation density; the display
//!   sub-rect is N× larger and paints at ~1:1 = "sharp host edges".
//! - **One pipeline for everything.** The vertex shader divides by
//!   constant `(1024, 512)` regardless of S. The wgpu viewport
//!   tracks the texture's pixel dims, so density follows S
//!   automatically -- no shader maths changes when S does.
//!
//! ## Sibling crates
//!
//! - `psx-gpu-compute` -- compute-shader rasterizer that matches the
//!   CPU rasterizer pixel-for-pixel; the parity oracle for the
//!   shared decode path (`psx-gpu-compute::decode`).
//! - `emulator-core` -- owns `Gpu` (CPU rasterizer + VRAM + cmd_log)
//!   and `Bus`.

pub mod from_ot;
pub mod pipeline;
pub mod target;
pub mod translator;

pub use from_ot::build_cmd_log;
pub use pipeline::{BlendKind, HwPipeline, HwVertex};
pub use target::{RenderTarget, MAX_SCALE, VRAM_HEIGHT, VRAM_WIDTH};
pub use translator::{DrawRun, TranslatedFrame, Translator};

use emulator_core::gpu::GpuCmdLogEntry;
use emulator_core::Gpu;

/// Top-level HW renderer. Owns the wgpu pipeline + the VRAM-shaped
/// target. The frontend creates one, calls [`HwRenderer::render_frame`]
/// each frame, and the egui central panel reads
/// [`HwRenderer::texture_id`] (paired with the display-area UV
/// sub-rect -- the renderer doesn't care about display size, only
/// the PSX-VRAM space the primitives live in).
///
/// `wgpu::Device` and `wgpu::Queue` are internally reference-counted
/// in wgpu 24, so cloning them into the renderer is cheap.
pub struct HwRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: HwPipeline,
    target: RenderTarget,
    translator: Translator,
}

/// Toolbar Native↔Window selector. Mirrors the frontend's
/// `app::ScaleMode` so this crate doesn't depend on `frontend`.
/// The renderer translates this + a panel-size hint into an
/// internal-resolution multiplier S in [`HwRenderer::scale_for`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ScaleMode {
    /// Internal scale = 1. Display sub-rect rendered at PSX-native
    /// pixel density; egui Nearest scales up at paint = chunky
    /// retro pixels.
    Native,
    /// Internal scale chosen from the current presentation/display
    /// pixel budget, clamped to `[1, MAX_SCALE]`. Display sub-rect
    /// rasterised near host density = sharp edges at any window size.
    Window,
}

impl HwRenderer {
    /// Live constructor -- registers the target with `egui_renderer`
    /// so the central panel can paint it. Initial scale = 1; bump
    /// it via [`HwRenderer::set_internal_scale`] on toggle/resize.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        egui_renderer: &mut egui_wgpu::Renderer,
    ) -> Self {
        let pipeline = HwPipeline::new(&device);
        let target = RenderTarget::new(&device, &queue, egui_renderer);
        Self {
            device,
            queue,
            pipeline,
            target,
            translator: Translator::new(),
        }
    }

    /// Headless constructor -- no surface, no egui registration.
    /// Used by the parity harness and any CLI dump path.
    pub fn new_headless(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let pipeline = HwPipeline::new(&device);
        let target = RenderTarget::new_headless(&device, &queue);
        Self {
            device,
            queue,
            pipeline,
            target,
            translator: Translator::new(),
        }
    }

    /// Stable `egui::TextureId` of the VRAM-shaped target. The
    /// frontend's central panel reads this every frame, paired with
    /// a UV sub-rect = `display_area / VRAM_DIMS`.
    pub fn texture_id(&self) -> egui::TextureId {
        self.target.texture_id()
    }

    /// Current internal-resolution multiplier in effect.
    pub fn internal_scale(&self) -> u32 {
        self.target.scale()
    }

    /// Map [`ScaleMode`] + presentation/display dims to an internal
    /// scale. `Native` → 1; `Window` → the smallest integer scale
    /// that covers the current framebuffer presentation budget,
    /// capped at [`MAX_SCALE`]. Pure function so the frontend can
    /// test scaling decisions without wgpu state.
    pub fn scale_for(mode: ScaleMode, panel_size_px: (u32, u32), display_size: (u32, u32)) -> u32 {
        match mode {
            ScaleMode::Native => 1,
            ScaleMode::Window => {
                let sx = panel_size_px.0.max(1) as f32 / display_size.0.max(1) as f32;
                let sy = panel_size_px.1.max(1) as f32 / display_size.1.max(1) as f32;
                (sx.max(sy).ceil() as u32).clamp(1, MAX_SCALE)
            }
        }
    }

    /// Reallocate the target to the requested internal scale. Cheap
    /// when unchanged. Reallocation clears the new texture to opaque
    /// black; in-flight VRAM contents are lost -- frontend may want
    /// to flush a fresh full-frame redraw after toggling.
    pub fn set_internal_scale(
        &mut self,
        scale: u32,
        egui_renderer: Option<&mut egui_wgpu::Renderer>,
    ) {
        self.target
            .ensure_scale(&self.device, &self.queue, egui_renderer, scale);
    }

    /// Render one frame's `cmd_log` into the persistent VRAM target.
    /// Always loads the existing texture -- never clears -- so PSX
    /// VRAM-style persistence holds across frames. `vram_words` is
    /// the CPU rasterizer's VRAM (post-frame) which the fragment
    /// shader reads for textured primitives.
    pub fn render_frame(&mut self, gpu: &Gpu, cmd_log: &[GpuCmdLogEntry], vram_words: &[u16]) {
        self.pipeline.upload_vram(&self.queue, vram_words);

        let mut segment_start = 0;
        for (i, entry) in cmd_log.iter().enumerate() {
            if is_vram_image_op(entry) {
                self.render_draw_segment(&cmd_log[segment_start..i], gpu.wireframe_enabled);
                self.mirror_vram_image_op(entry, vram_words);
                segment_start = i + 1;
            }
        }
        self.render_draw_segment(&cmd_log[segment_start..], gpu.wireframe_enabled);
    }

    fn render_draw_segment(&mut self, cmd_log: &[GpuCmdLogEntry], wireframe: bool) {
        if cmd_log.is_empty() {
            return;
        }

        let frame = self.translator.translate_with_wireframe(cmd_log, wireframe);
        if frame.total() > 0 {
            let vertices = frame.vertices.to_vec();
            let runs = frame.runs.to_vec();
            self.pipeline
                .upload_vertices(&self.queue, bytemuck::cast_slice(&vertices));
            self.draw_runs(&runs);
        }
    }

    fn draw_runs(&mut self, runs: &[DrawRun]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-hw-renderer-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("psx-hw-renderer-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.target.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // PSX VRAM is persistent -- never clear.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, self.pipeline.bind_group(), &[]);
            pass.set_vertex_buffer(0, self.pipeline.vertex_buffer().slice(..));

            let scale = self.target.scale();
            let (target_w, target_h) = self.target.size();
            for run in runs {
                if run.count == 0 || run.clip[0] > run.clip[2] || run.clip[1] > run.clip[3] {
                    continue;
                }
                let x = run.clip[0] as u32 * scale;
                let y = run.clip[1] as u32 * scale;
                let right = ((run.clip[2] as u32 + 1) * scale).min(target_w);
                let bottom = ((run.clip[3] as u32 + 1) * scale).min(target_h);
                if right <= x || bottom <= y {
                    continue;
                }
                pass.set_scissor_rect(x, y, right - x, bottom - y);
                pass.set_pipeline(self.pipeline.pipeline(run.kind));
                pass.set_blend_constant(self.pipeline.blend_constant(run.kind));
                pass.draw(run.start..(run.start + run.count), 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn mirror_vram_image_op(&self, entry: &GpuCmdLogEntry, vram_words: &[u16]) {
        match entry.opcode {
            0x80..=0x9F => self.mirror_vram_copy(entry),
            0xA0..=0xBF => self.mirror_vram_upload(entry, vram_words),
            _ => {}
        }
    }

    fn mirror_vram_copy(&self, entry: &GpuCmdLogEntry) {
        if entry.fifo.len() < 4 {
            return;
        }
        let src = entry.fifo[1];
        let dst = entry.fifo[2];
        let wh = entry.fifo[3];
        let sx = src & (VRAM_WIDTH - 1);
        let sy = (src >> 16) & (VRAM_HEIGHT - 1);
        let dx = dst & (VRAM_WIDTH - 1);
        let dy = (dst >> 16) & (VRAM_HEIGHT - 1);
        let raw_w = wh & 0xFFFF;
        let raw_h = (wh >> 16) & 0xFFFF;
        let w = if raw_w == 0 { VRAM_WIDTH } else { raw_w };
        let h = if raw_h == 0 { VRAM_HEIGHT } else { raw_h };
        if w == 0 || h == 0 {
            return;
        }

        let scale = self.target.scale();
        let out_w = w * scale;
        let temp = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("psx-hw-vram-copy-temp"),
            size: wgpu::Extent3d {
                width: out_w,
                height: scale,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::target::TARGET_FORMAT,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-hw-vram-copy-encoder"),
            });

        // Match the CPU rasterizer's row-buffer copy semantics:
        // read one complete wrapped source row, then write that row
        // to the wrapped destination. This keeps overlapping copies
        // and edge-wrapping image ops in command order.
        for row in 0..h {
            let src_y = (sy + row) & (VRAM_HEIGHT - 1);
            let dst_y = (dy + row) & (VRAM_HEIGHT - 1);

            let mut copied = 0;
            while copied < w {
                let src_x = (sx + copied) & (VRAM_WIDTH - 1);
                let chunk_w = (w - copied).min(VRAM_WIDTH - src_x);
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: self.target.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: src_x * scale,
                            y: src_y * scale,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &temp,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copied * scale,
                            y: 0,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: chunk_w * scale,
                        height: scale,
                        depth_or_array_layers: 1,
                    },
                );
                copied += chunk_w;
            }

            copied = 0;
            while copied < w {
                let dst_x = (dx + copied) & (VRAM_WIDTH - 1);
                let chunk_w = (w - copied).min(VRAM_WIDTH - dst_x);
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &temp,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copied * scale,
                            y: 0,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: self.target.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: dst_x * scale,
                            y: dst_y * scale,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: chunk_w * scale,
                        height: scale,
                        depth_or_array_layers: 1,
                    },
                );
                copied += chunk_w;
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn mirror_vram_upload(&self, entry: &GpuCmdLogEntry, vram_words: &[u16]) {
        if entry.fifo.len() < 3 {
            return;
        }
        let xy = entry.fifo[1];
        let wh = entry.fifo[2];
        let x = xy & 0x3FF;
        let y = (xy >> 16) & 0x1FF;
        let raw_w = wh & 0x3FF;
        let raw_h = (wh >> 16) & 0x1FF;
        let w = if raw_w == 0 { VRAM_WIDTH } else { raw_w };
        let h = if raw_h == 0 { VRAM_HEIGHT } else { raw_h };
        let payload = entry.fifo.get(3..).unwrap_or(&[]);
        if payload.is_empty() {
            self.write_scaled_vram_rect_wrapped(x, y, w, h, |col, row| {
                let xx = (x + col) & (VRAM_WIDTH - 1);
                let yy = (y + row) & (VRAM_HEIGHT - 1);
                vram_words[(yy * VRAM_WIDTH + xx) as usize]
            });
            return;
        }

        self.write_scaled_vram_rect_wrapped(x, y, w, h, |col, row| {
            let pixel_index = row * w + col;
            let Some(&word) = payload.get((pixel_index / 2) as usize) else {
                return 0;
            };
            if pixel_index & 1 == 0 {
                word as u16
            } else {
                (word >> 16) as u16
            }
        });
    }

    fn write_scaled_vram_rect_wrapped(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        pixel_at: impl Fn(u32, u32) -> u16,
    ) {
        if w == 0 || h == 0 {
            return;
        }

        let scale = self.target.scale();
        let mut row = 0;
        while row < h {
            let dst_y = (y + row) & (VRAM_HEIGHT - 1);
            let chunk_h = (h - row).min(VRAM_HEIGHT - dst_y);
            let mut col = 0;
            while col < w {
                let dst_x = (x + col) & (VRAM_WIDTH - 1);
                let chunk_w = (w - col).min(VRAM_WIDTH - dst_x);

                let out_w = chunk_w * scale;
                let out_h = chunk_h * scale;
                let row_bytes = out_w * 4;
                let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                let padded_row_bytes = row_bytes.div_ceil(align) * align;
                let mut rgba = vec![0u8; (padded_row_bytes * out_h) as usize];

                for src_y in 0..chunk_h {
                    for src_x in 0..chunk_w {
                        let color = bgr15_to_rgba8(pixel_at(col + src_x, row + src_y));
                        for sy in 0..scale {
                            let out_y = src_y * scale + sy;
                            let row_start = (out_y * padded_row_bytes) as usize;
                            for sx in 0..scale {
                                let out_x = src_x * scale + sx;
                                let off = row_start + (out_x * 4) as usize;
                                rgba[off..off + 4].copy_from_slice(&color);
                            }
                        }
                    }
                }

                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: self.target.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: dst_x * scale,
                            y: dst_y * scale,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row_bytes),
                        rows_per_image: Some(out_h),
                    },
                    wgpu::Extent3d {
                        width: out_w,
                        height: out_h,
                        depth_or_array_layers: 1,
                    },
                );

                col += chunk_w;
            }
            row += chunk_h;
        }
    }

    /// Synchronously read back the entire VRAM-shaped target as
    /// tightly-packed RGBA8 (sRGB color space, since `TARGET_FORMAT`
    /// is `Rgba8UnormSrgb`). Issues a blocking `device.poll(Wait)`
    /// -- for headless/parity use, never the per-frame loop.
    pub fn read_pixels_rgba8(&self) -> (u32, u32, Vec<u8>) {
        let (w, h) = self.target.size();
        self.read_subrect_rgba8(0, 0, w, h)
    }

    /// Read back a `(w × h)` sub-rect of the target starting at
    /// `(x, y)` in target-pixel coordinates (i.e. PSX VRAM coords ×
    /// internal scale). Designed for parity-style display-sub-rect
    /// extraction: pass `(display.x * S, display.y * S, display.w *
    /// S, display.h * S)` to grab exactly the user-visible region.
    pub fn read_subrect_rgba8(&self, x: u32, y: u32, w: u32, h: u32) -> (u32, u32, Vec<u8>) {
        let (tw, th) = self.target.size();
        let x = x.min(tw);
        let y = y.min(th);
        let w = w.min(tw - x);
        let h = h.min(th - y);
        if w == 0 || h == 0 {
            return (0, 0, Vec::new());
        }
        let unpadded_bpr = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;
        let buffer_size = (padded_bpr * h) as wgpu::BufferAddress;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-hw-readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-hw-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self.target.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((unpadded_bpr * h) as usize);
        for row in 0..h {
            let start = (row * padded_bpr) as usize;
            let end = start + unpadded_bpr as usize;
            out.extend_from_slice(&data[start..end]);
        }
        drop(data);
        buffer.unmap();
        (w, h, out)
    }
}

fn is_vram_image_op(entry: &GpuCmdLogEntry) -> bool {
    matches!(entry.opcode, 0x80..=0xBF)
}

fn bgr15_to_rgba8(pixel: u16) -> [u8; 4] {
    let r5 = (pixel & 0x1F) as u8;
    let g5 = ((pixel >> 5) & 0x1F) as u8;
    let b5 = ((pixel >> 10) & 0x1F) as u8;
    [
        (r5 << 3) | (r5 >> 2),
        (g5 << 3) | (g5 >> 2),
        (b5 << 3) | (b5 >> 2),
        0xFF,
    ]
}
