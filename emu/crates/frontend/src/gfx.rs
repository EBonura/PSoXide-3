//! Windowing + wgpu + egui plumbing.
//!
//! Owns the surface, device, queue, and the egui-wgpu renderer. Kept
//! separate from `App` so `App` can focus on emulator state and UI logic
//! without wgpu types leaking into it.

use std::sync::Arc;
use std::time::Instant;

use emulator_core::{Gpu, Vram, VRAM_HEIGHT, VRAM_WIDTH};
use psx_gpu_render::HwRenderer;
use winit::window::Window;

use crate::ui::profiler::EguiRenderProfile;

/// Largest logical display window the PSX exposes. The 24bpp fallback
/// texture is allocated once at this size and each frame uploads the
/// active display area into its top-left corner.
pub const MAX_DISPLAY_WIDTH: u32 = 640;
pub const MAX_DISPLAY_HEIGHT: u32 = 480;

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
    /// Packed visible framebuffer used only for 24bpp display modes.
    /// The HW target is VRAM-cell-shaped, but 24bpp video packs RGB
    /// bytes across cells, so presentation needs this CPU decode path.
    display_texture: wgpu::Texture,
    /// Egui-side handle for the 24bpp display fallback texture.
    display_texture_id: egui::TextureId,
    /// Hardware renderer — issues per-primitive wgpu draw calls
    /// from the frame's `cmd_log` into a VRAM-shaped texture. Both
    /// Native and Window framebuffer modes use this path; the only
    /// difference is the internal resolution multiplier. 24bpp video
    /// presentation falls back to [`Gpu::display_rgba8`] until the HW
    /// presenter has a packed-RGB decode shader.
    hw_renderer: HwRenderer,
    /// Editor-side `HwRenderer` instance. Independent VRAM target,
    /// rendered from the editor's authored scene rather than the
    /// running game; the editor's egui panel paints its texture as
    /// an Image so the preview is bit-faithful to what the PS1 would
    /// draw given the same input data.
    editor_hw_renderer: HwRenderer,
    /// Stub `Gpu` consumed by [`HwRenderer::render_frame`]. The
    /// renderer only reads `wireframe_enabled` from it; the editor
    /// preview always renders solid, so we hand it a freshly-zeroed
    /// instance.
    editor_gpu_stub: Gpu,
    /// Per-material procedural textures + the VRAM mirror they're
    /// stamped into. Refreshed each frame against the project; new
    /// materials get a tpage allocated lazily.
    editor_textures: crate::editor_textures::EditorTextures,
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

        let display_texture = create_display_texture(&device);
        let display_view = display_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let display_texture_id = egui_renderer.register_native_texture(
            &device,
            &display_view,
            wgpu::FilterMode::Nearest,
        );

        let hw_renderer = HwRenderer::new(device.clone(), queue.clone(), &mut egui_renderer);
        let mut editor_hw_renderer =
            HwRenderer::new(device.clone(), queue.clone(), &mut egui_renderer);
        // Higher internal scale for the editor preview — the game's
        // renderer follows user/scale-mode preference, but the editor
        // viewport is always overlay-friendly host resolution.
        editor_hw_renderer.set_internal_scale(2, Some(&mut egui_renderer));

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
            hw_renderer,
            editor_hw_renderer,
            editor_gpu_stub: Gpu::new(),
            editor_textures: crate::editor_textures::EditorTextures::new(),
        }
    }

    /// Egui handle for the VRAM texture. Stable for the life of the
    /// window; safe to pass to panels once per frame.
    pub fn vram_texture_id(&self) -> egui::TextureId {
        self.vram_texture_id
    }

    /// Egui handle for the packed 24bpp display fallback texture.
    pub fn display_texture_id(&self) -> egui::TextureId {
        self.display_texture_id
    }

    /// Egui handle for the hardware-renderer target.
    pub fn hw_texture_id(&self) -> egui::TextureId {
        self.hw_renderer.texture_id()
    }

    /// Egui handle for the editor's preview HW target. Stable across
    /// frames; the editor's 3D viewport panel paints it as an Image.
    pub fn editor_hw_texture_id(&self) -> egui::TextureId {
        self.editor_hw_renderer.texture_id()
    }

    /// Render one frame of the editor preview through the second
    /// `HwRenderer` instance.
    ///
    /// Phase 1: walks the editor project's first Room, projects each
    /// floor through the host GTE shim, and feeds the resulting
    /// `TriFlat` packets to the renderer. The path is intentionally
    /// the same one PS1 runtime code follows — only the final DMA
    /// step is replaced by `build_cmd_log` + `render_frame`.
    pub fn render_editor_preview(
        &mut self,
        project: &psxed_project::ProjectDocument,
        project_root: &std::path::Path,
        camera: psxed_ui::ViewportCameraState,
        selected: psxed_project::NodeId,
        hovered_cell: Option<(u16, u16)>,
        hovered_edge: Option<(u16, u16, u8)>,
        hovered_face: Option<psxed_ui::FaceRef>,
        selected_face: Option<psxed_ui::FaceRef>,
        wall_paint_preview: Option<psxed_ui::FaceRef>,
    ) {
        self.editor_textures.refresh(project, project_root);
        let cmd_log = crate::editor_preview::build_phase1_cmd_log(
            project,
            camera,
            selected,
            hovered_cell,
            hovered_edge,
            hovered_face,
            selected_face,
            wall_paint_preview,
            &self.editor_textures,
        );
        self.editor_hw_renderer.render_frame(
            &self.editor_gpu_stub,
            &cmd_log,
            self.editor_textures.vram_words(),
        );
    }

    /// Current internal scale used by the hardware-renderer target.
    pub fn hw_internal_scale(&self) -> u32 {
        self.hw_renderer.internal_scale()
    }

    /// Drive the hardware renderer for one frame. Walks `cmd_log`,
    /// issues wgpu draw calls for the supported primitive types,
    /// and leaves the result in the renderer's VRAM-shaped target.
    /// `vram_words` is the CPU rasterizer's VRAM (post-frame),
    /// uploaded into the GPU-side `R16Uint` so textured primitives
    /// can sample the right texels + CLUT entries.
    pub fn render_hw_frame(
        &mut self,
        gpu: &Gpu,
        cmd_log: &[emulator_core::gpu::GpuCmdLogEntry],
        vram_words: &[u16],
    ) {
        self.hw_renderer.render_frame(gpu, cmd_log, vram_words);
    }

    /// Pick the right internal-resolution multiplier for the current
    /// scale-mode + panel size and reallocate the HW target to it
    /// if changed. Cheap when the size is stable; reallocation
    /// clears the target so the frontend may want to flush a fresh
    /// frame after toggling.
    pub fn update_hw_scale(
        &mut self,
        scale_mode: psx_gpu_render::ScaleMode,
        panel_size_px: (u32, u32),
        display_size: (u32, u32),
    ) {
        let s = psx_gpu_render::HwRenderer::scale_for(scale_mode, panel_size_px, display_size);
        self.hw_renderer
            .set_internal_scale(s, Some(&mut self.egui_renderer));
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

    /// Upload the CPU-decoded display area for 24bpp presentation.
    /// Normal 15bpp/CLUT rendering stays on the HW target; this path
    /// exists because PS1 24bpp display pixels are packed as byte
    /// triplets across 16-bit VRAM cells.
    pub fn prepare_24bpp_display(&self, gpu: Option<&Gpu>) {
        let Some(gpu) = gpu else {
            self.clear_display_texture();
            return;
        };
        if !gpu.display_area().bpp24 {
            return;
        }
        let (rgba, width, height) = gpu.display_rgba8();
        if width == 0 || height == 0 || rgba.is_empty() {
            self.clear_display_texture();
            return;
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.display_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn clear_display_texture(&self) {
        let rgba = vec![0u8; (MAX_DISPLAY_WIDTH * MAX_DISPLAY_HEIGHT * 4) as usize];
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.display_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(MAX_DISPLAY_WIDTH * 4),
                rows_per_image: Some(MAX_DISPLAY_HEIGHT),
            },
            wgpu::Extent3d {
                width: MAX_DISPLAY_WIDTH,
                height: MAX_DISPLAY_HEIGHT,
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
    pub fn render(&mut self, build_ui: impl FnMut(&egui::Context)) -> EguiRenderProfile {
        let total_start = Instant::now();
        let mut profile = EguiRenderProfile::default();

        let t = Instant::now();
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        profile.input_ms = elapsed_ms(t);

        let t = Instant::now();
        let full_output = self.egui_ctx.run(raw_input, build_ui);
        profile.ui_ms = elapsed_ms(t);

        let t = Instant::now();
        self.egui_winit
            .handle_platform_output(&self.window, full_output.platform_output);
        profile.platform_output_ms = elapsed_ms(t);

        let t = Instant::now();
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        profile.tessellate_ms = elapsed_ms(t);

        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let t = Instant::now();
        let output = match self.surface.get_current_texture() {
            Ok(out) => out,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                profile.surface_ms = elapsed_ms(t);
                profile.total_ms = elapsed_ms(total_start);
                return profile;
            }
            Err(e) => {
                eprintln!("surface error: {e:?}");
                profile.surface_ms = elapsed_ms(t);
                profile.total_ms = elapsed_ms(total_start);
                return profile;
            }
        };
        profile.surface_ms = elapsed_ms(t);

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psoxide3-frame"),
            });

        let t = Instant::now();
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        profile.texture_update_ms = elapsed_ms(t);

        let t = Instant::now();
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );
        profile.buffer_update_ms = elapsed_ms(t);

        let t = Instant::now();
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
        profile.paint_ms = elapsed_ms(t);

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        let t = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        output.present();
        profile.submit_present_ms = elapsed_ms(t);
        profile.total_ms = elapsed_ms(total_start);
        profile
    }
}

fn elapsed_ms(start: Instant) -> f32 {
    start.elapsed().as_secs_f32() * 1000.0
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

fn create_display_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("psoxide3-display-24bpp"),
        size: wgpu::Extent3d {
            width: MAX_DISPLAY_WIDTH,
            height: MAX_DISPLAY_HEIGHT,
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
