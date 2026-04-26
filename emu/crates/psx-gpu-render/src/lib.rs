//! PSX hardware renderer — wgpu render pipeline that draws each
//! GP0 primitive as one or more triangles at the host resolution,
//! producing internal-resolution upscaling for free.
//!
//! Sibling to `psx-gpu-compute`. The compute backend exists for
//! parity testing (it matches the CPU rasterizer pixel-for-pixel
//! at native resolution). This crate is for *display*: it does
//! NOT need to match VRAM byte-for-byte; it needs to look right
//! on screen at any window size.
//!
//! Architecture:
//!
//! - One render pipeline (vertex + fragment shaders), one shader
//!   file. Native vs Window scaling is a *render-target sizing*
//!   choice — the same pipeline runs in both modes, only the
//!   output texture's dimensions differ. This honours the user's
//!   "one flexible pipeline" constraint.
//! - The translator walks `Gpu::cmd_log` exactly like
//!   `psx-gpu-compute::ComputeBackend::replay_packet`, sharing
//!   `psx-gpu-compute::decode::ReplayState` so state setters
//!   (`0xE0..=0xE6`) advance identically across backends.
//! - Phase 1: only flat-color mono triangles / quads (`0x20..=0x2B`).
//!   Other primitive types are silently dropped (their state still
//!   updates `ReplayState`).
//!
//! See `/Users/ebonura/.claude/plans/toasty-snacking-noodle.md` for
//! the staged plan.

pub mod pipeline;
pub mod target;
pub mod translator;

pub use pipeline::{BlendKind, HwPipeline, HwVertex};
pub use target::RenderTarget;
pub use translator::{TranslatedFrame, Translator};

use emulator_core::Gpu;

/// Top-level HW renderer. Owns the wgpu pipeline + the output
/// texture. The frontend creates one, feeds it `cmd_log` each
/// frame, and reads back an `egui::TextureId` for the central
/// panel to paint.
///
/// `wgpu::Device` and `wgpu::Queue` are internally reference-
/// counted in wgpu 24, so cloning them into the renderer is cheap.
pub struct HwRenderer {
    device: wgpu::Device,
    queue:  wgpu::Queue,
    pipeline: HwPipeline,
    target: RenderTarget,
    translator: Translator,
    /// Most recent global uniforms — written once per frame after
    /// the target is resized.
    globals: pipeline::Globals,
}

/// What the toolbar's `Native ↔ Window` toggle picks. Mirrors the
/// frontend's `app::ScaleMode`; redeclared here so this crate
/// doesn't have to depend on `frontend`. The frontend converts
/// from its own enum to ours when it calls `render_frame`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ScaleMode {
    /// 1:1 PSX pixels — output texture sized exactly to the active
    /// display rect.
    Native,
    /// Output texture sized to the largest 4:3 box that fits the
    /// available central-panel rect, so primitives rasterize at
    /// host resolution with sharp edges at any window size.
    Window,
}

impl HwRenderer {
    /// Build the renderer on top of an existing `wgpu::Device` /
    /// `Queue` (typically the frontend's swap-chain device, so the
    /// output texture can be sampled by egui without bouncing
    /// through CPU memory). The target texture is registered with
    /// the supplied `egui_wgpu::Renderer`; calling
    /// `RenderTarget::texture_id` later returns the stable handle
    /// the central panel paints.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        egui_renderer: &mut egui_wgpu::Renderer,
    ) -> Self {
        let pipeline = HwPipeline::new(&device);
        let target = RenderTarget::new(&device, egui_renderer);
        Self {
            device,
            queue,
            pipeline,
            target,
            translator: Translator::new(),
            globals: pipeline::Globals::zero(),
        }
    }

    /// `egui::TextureId` of the output texture. Stable for the
    /// life of the `HwRenderer` (re-registered on resize, but
    /// the id is updated in place).
    pub fn texture_id(&self) -> egui::TextureId {
        self.target.texture_id()
    }

    /// Render one frame.
    ///
    /// `panel_rect` is the egui central-panel rect IN PIXELS that
    /// will receive the output (so we can size the render target
    /// to it for `Window` mode). `gpu` is read for the active
    /// display area + the drained `cmd_log`. `vram_words` is the
    /// CPU rasterizer's VRAM after this frame's draws — uploaded
    /// to the GPU-side `R16Uint` texture for the fragment shader
    /// to sample on textured primitives.
    pub fn render_frame(
        &mut self,
        gpu: &Gpu,
        scale_mode: ScaleMode,
        panel_size_px: (u32, u32),
        egui_renderer: &mut egui_wgpu::Renderer,
        cmd_log: &[emulator_core::gpu::GpuCmdLogEntry],
        vram_words: &[u16],
    ) {
        let display = gpu.display_area();
        let (display_w, display_h) = (
            display.width.max(1) as u32,
            display.height.max(1) as u32,
        );
        let (target_w, target_h) = match scale_mode {
            ScaleMode::Native => (display_w, display_h),
            ScaleMode::Window => Self::compute_window_target_size(panel_size_px, display_w, display_h),
        };

        self.target
            .ensure(&self.device, egui_renderer, target_w, target_h);

        self.globals = pipeline::Globals {
            display_origin: [display.x as f32, display.y as f32],
            display_size: [display_w as f32, display_h as f32],
            target_size: [target_w as f32, target_h as f32],
            _pad: [0.0, 0.0],
        };
        self.queue.write_buffer(
            self.pipeline.globals_buffer(),
            0,
            bytemuck::bytes_of(&self.globals),
        );

        // Mirror CPU VRAM into the GPU `R16Uint` so textured
        // primitives can sample texels + CLUTs correctly. Phase 1
        // didn't need this; Phase 2's textured primitives do.
        self.pipeline.upload_vram(&self.queue, vram_words);

        // Translate cmd_log → per-blend-kind sorted vertex stream.
        let frame = self.translator.translate(cmd_log);
        if frame.total() > 0 {
            self.pipeline
                .upload_vertices(&self.queue, bytemuck::cast_slice(frame.vertices));
        }

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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, self.pipeline.bind_group(), &[]);
            pass.set_vertex_buffer(0, self.pipeline.vertex_buffer().slice(..));

            // One draw per non-empty blend batch. Average and
            // AddQuarter use the blend constant — bind the right
            // value before their pipelines run. The other modes
            // ignore the constant so binding it is a no-op.
            let mut start: u32 = 0;
            for i in 0..pipeline::BlendKind::COUNT {
                let count = frame.counts[i];
                if count == 0 {
                    start += count;
                    continue;
                }
                let kind = match i {
                    0 => pipeline::BlendKind::Opaque,
                    1 => pipeline::BlendKind::Average,
                    2 => pipeline::BlendKind::Add,
                    3 => pipeline::BlendKind::Sub,
                    _ => pipeline::BlendKind::AddQuarter,
                };
                pass.set_pipeline(self.pipeline.pipeline(kind));
                pass.set_blend_constant(self.pipeline.blend_constant(kind));
                pass.draw(start..(start + count), 0..1);
                start += count;
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Largest 4:3 rect that fits inside the available panel rect,
    /// rounded to integer pixels. Same formula the old
    /// `paint_image_window` used at paint-time, just lifted into
    /// render-target sizing so the rasterizer produces exactly that
    /// many pixels (no paint-time stretch).
    fn compute_window_target_size(
        panel: (u32, u32),
        _display_w: u32,
        _display_h: u32,
    ) -> (u32, u32) {
        const CRT_ASPECT: f32 = 4.0 / 3.0;
        let pw = panel.0.max(1) as f32;
        let ph = panel.1.max(1) as f32;
        let h = ph.min(pw / CRT_ASPECT);
        let w = h * CRT_ASPECT;
        (w.round().max(1.0) as u32, h.round().max(1.0) as u32)
    }
}
