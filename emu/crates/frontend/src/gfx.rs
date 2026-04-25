//! Windowing + wgpu + egui plumbing.
//!
//! Owns the surface, device, queue, and the egui-wgpu renderer. Kept
//! separate from `App` so `App` can focus on emulator state and UI logic
//! without wgpu types leaking into it.

use std::sync::Arc;

use emulator_core::{Gpu, VRAM_HEIGHT, VRAM_WIDTH};
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
    /// Top-left-packed RGBA8 texture of the active display area. Unlike
    /// the VRAM texture, this respects 24-bit framebuffer mode.
    display_texture: wgpu::Texture,
    /// Egui handle for the active display texture.
    display_texture_id: egui::TextureId,
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
        let display_texture = create_rgba_texture(&device, "psoxide3-display");
        let display_view = display_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let display_texture_id = egui_renderer.register_native_texture(
            &device,
            &display_view,
            wgpu::FilterMode::Nearest,
        );

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
            display_texture,
            display_texture_id,
        }
    }

    /// Egui handle for the VRAM texture. Stable for the life of the
    /// window; safe to pass to panels once per frame.
    pub fn vram_texture_id(&self) -> egui::TextureId {
        self.vram_texture_id
    }

    /// Egui handle for the active display texture. Stable for the life of
    /// the window.
    pub fn display_texture_id(&self) -> egui::TextureId {
        self.display_texture_id
    }

    /// Upload a full VRAM snapshot to the GPU-side texture. Called once
    /// per frame from `App` before `render`. `None` means "no Bus yet"
    /// — we leave the last texture contents alone (typically zeros).
    /// Upload a raw 1024×512 BGR15 buffer into the VRAM viewer
    /// texture. The GPU compute backend feeds this with its
    /// downloaded VRAM each frame.
    pub fn prepare_vram_from_words(&self, words: &[u16]) {
        if words.len() != VRAM_WIDTH * VRAM_HEIGHT {
            return;
        }
        let mut rgba = Vec::with_capacity(VRAM_WIDTH * VRAM_HEIGHT * 4);
        for &pixel in words {
            let r5 = pixel & 0x1F;
            let g5 = (pixel >> 5) & 0x1F;
            let b5 = (pixel >> 10) & 0x1F;
            rgba.push(((r5 << 3) | (r5 >> 2)) as u8);
            rgba.push(((g5 << 3) | (g5 >> 2)) as u8);
            rgba.push(((b5 << 3) | (b5 >> 2)) as u8);
            rgba.push(0xFF);
        }
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

    /// Upload the GPU's active display rectangle into a top-left-packed
    /// texture. The source pixels come from a downloaded GPU compute
    /// VRAM buffer; the display rect / 24-bit-mode flag is still
    /// pulled from `gpu` (the same state the framebuffer panel reads).
    pub fn prepare_display_from_words(&self, words: Vec<u16>, gpu: Option<&Gpu>) {
        let Some(gpu) = gpu else {
            return;
        };
        if words.len() != VRAM_WIDTH * VRAM_HEIGHT {
            return;
        }
        let da = gpu.display_area();
        if da.width == 0 || da.height == 0 {
            return;
        }
        // 16bpp BGR15 readback, bit-replicated to RGB888 — same
        // expansion `Vram::to_rgba8` does on the CPU side.
        let eff_w = da.width.min(VRAM_WIDTH as u16 - da.x) as usize;
        let eff_h = da.height.min(VRAM_HEIGHT as u16 - da.y) as usize;
        let mut packed = vec![0u8; VRAM_WIDTH * VRAM_HEIGHT * 4];
        let dst_stride = VRAM_WIDTH * 4;
        for row in 0..eff_h {
            let src_row = (da.y as usize + row) * VRAM_WIDTH + da.x as usize;
            let dst_off = row * dst_stride;
            for col in 0..eff_w {
                let pix = words[src_row + col];
                let r5 = (pix & 0x1F) as u8;
                let g5 = ((pix >> 5) & 0x1F) as u8;
                let b5 = ((pix >> 10) & 0x1F) as u8;
                let i = dst_off + col * 4;
                packed[i] = (r5 << 3) | (r5 >> 2);
                packed[i + 1] = (g5 << 3) | (g5 >> 2);
                packed[i + 2] = (b5 << 3) | (b5 >> 2);
                packed[i + 3] = 0xFF;
            }
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.display_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &packed,
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
