//! Windowing + wgpu + egui plumbing.
//!
//! Owns the surface, device, queue, and the egui-wgpu renderer. Kept
//! separate from `App` so `App` can focus on emulator state and UI logic
//! without wgpu types leaking into it.

use std::sync::Arc;

use emulator_core::{Gpu, Vram, VRAM_HEIGHT, VRAM_WIDTH};
use psx_gpu_render::HwRenderer;
use winit::window::Window;

/// All GPU / windowing state. Built lazily in `App::resumed` because
/// winit 0.30 + macOS requires the window to be created inside the
/// `resumed()` callback, not before.
pub struct Graphics {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,

    /// Persistent 1024×512 RGBA8 texture that mirrors VRAM for display.
    /// Uploaded once per frame in `prepare_vram`.
    vram_texture: wgpu::Texture,
    /// Egui-side handle for the VRAM texture; panels reference it via
    /// [`Graphics::vram_texture_id`].
    vram_texture_id: egui::TextureId,
    /// Hardware renderer — issues per-primitive wgpu draw calls
    /// from the frame's `cmd_log` into a window-resolution texture.
    /// Replaces the old `prepare_display` CPU-blit path entirely.
    hw_renderer: HwRenderer,
}

impl Graphics {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("request adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("psoxide3-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("request device");

        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface config");
        surface.configure(&device, &config);

        let egui_ctx = egui::Context::default();
        crate::theme::apply(&egui_ctx);

        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let mut egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);

        let vram_texture = create_rgba_texture(&device, "psoxide3-vram");
        let vram_view = vram_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let vram_texture_id =
            egui_renderer.register_native_texture(&device, &vram_view, wgpu::FilterMode::Nearest);

        let hw_renderer = HwRenderer::new(device.clone(), queue.clone(), &mut egui_renderer);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            egui_ctx,
            egui_winit,
            egui_renderer,
            vram_texture,
            vram_texture_id,
            hw_renderer,
        }
    }

    /// Egui handle for the VRAM texture. Stable for the life of the
    /// window; safe to pass to panels once per frame.
    pub fn vram_texture_id(&self) -> egui::TextureId {
        self.vram_texture_id
    }

    /// Egui handle for the hardware renderer's output texture.
    /// The id is stable for the life of the window even though the
    /// underlying wgpu texture is re-allocated whenever the
    /// requested target size changes (window resize, display-rect
    /// change, scale-mode toggle).
    pub fn hw_texture_id(&self) -> egui::TextureId {
        self.hw_renderer.texture_id()
    }

    /// Drive the hardware renderer for one frame. Walks `cmd_log`,
    /// issues wgpu draw calls for the supported primitive types,
    /// and leaves the result in the texture exposed by
    /// [`Graphics::hw_texture_id`]. The central panel paints that
    /// texture verbatim; no paint-time stretch.
    pub fn render_hw_frame(
        &mut self,
        gpu: &Gpu,
        scale_mode: psx_gpu_render::ScaleMode,
        panel_size_px: (u32, u32),
        cmd_log: &[emulator_core::gpu::GpuCmdLogEntry],
    ) {
        self.hw_renderer.render_frame(
            gpu,
            scale_mode,
            panel_size_px,
            &mut self.egui_renderer,
            cmd_log,
        );
    }

    /// Upload a full VRAM snapshot to the GPU-side texture. Called once
    /// per frame from `App` before `render`. `None` means "no Bus yet"
    /// — we leave the last texture contents alone (typically zeros).
    pub fn prepare_vram(&self, vram: Option<&Vram>) {
        let Some(vram) = vram else {
            return;
        };
        let rgba = vram.to_rgba8(0, 0, VRAM_WIDTH as u16, VRAM_HEIGHT as u16);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.vram_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(VRAM_WIDTH as u32 * 4),
                rows_per_image: Some(VRAM_HEIGHT as u32),
            },
            wgpu::Extent3d {
                width: VRAM_WIDTH as u32,
                height: VRAM_HEIGHT as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Paint one frame. `build_ui` runs inside the egui context and builds
    /// all panels/overlays for this frame. Split out so `App` controls UI
    /// content without `Graphics` knowing what panels exist.
    pub fn render(&mut self, build_ui: impl FnMut(&egui::Context)) {
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let full_output = self.egui_ctx.run(raw_input, build_ui);

        self.egui_winit
            .handle_platform_output(&self.window, full_output.platform_output);

        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let output = match self.surface.get_current_texture() {
            Ok(out) => out,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(e) => {
                eprintln!("surface error: {e:?}");
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psoxide3-frame"),
            });

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );

        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("psoxide3-egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.04,
                                g: 0.04,
                                b: 0.06,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                })
                .forget_lifetime();

            self.egui_renderer
                .render(&mut rpass, &paint_jobs, &screen_desc);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        output.present();
    }
}

fn create_rgba_texture(device: &wgpu::Device, label: &'static str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: VRAM_WIDTH as u32,
            height: VRAM_HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
