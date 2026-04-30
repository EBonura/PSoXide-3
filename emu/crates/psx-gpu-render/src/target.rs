//! HW-renderer render target.
//!
//! Owns a single VRAM-shaped texture: `(VRAM_WIDTH * S) × (VRAM_HEIGHT * S)`
//! where S is the internal-resolution multiplier (1 for Native, ≥1 for
//! Window). The PSX's GPU draws into a persistent 1024×512 VRAM and the
//! display is just a sub-rect read of it; this target mirrors that, so
//! Native↔Window stops being "different texture sizes" and becomes
//! "different internal scales of the same VRAM-shaped buffer".
//!
//! The target is cleared to opaque black at construction (and again on
//! every scale change, which reallocates). Steady-state per-frame draws
//! always `LoadOp::Load` because PSX VRAM is persistent -- a primitive
//! drawn last frame stays until something overwrites it. The CPU
//! rasterizer follows the same convention; the HW path now matches.

use wgpu::TextureFormat;

/// Format of the HW renderer's output. `Rgba8UnormSrgb` matches
/// the surface format used elsewhere (egui's central panel
/// composes everything in linear-from-sRGB), so egui's blit
/// requires no extra colorspace conversion.
pub const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

/// PSX VRAM dimensions in 16-bit cells. The HW target is a multiple
/// of these by the internal scale.
pub const VRAM_WIDTH: u32 = 1024;
pub const VRAM_HEIGHT: u32 = 512;

/// Cap on the internal-resolution multiplier. 4× → 4096×2048 RGBA8 ≈
/// 32 MiB. 8× would be 128 MiB -- out of reasonable range for a
/// retro-emulator default. The frontend can request lower; higher
/// requires raising this cap deliberately.
pub const MAX_SCALE: u32 = 4;

pub struct RenderTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// `None` in headless mode (parity harness, dump CLI). The live
    /// frontend always supplies a registered id.
    egui_id: Option<egui::TextureId>,
    /// Active internal-resolution multiplier. Texture dims are this
    /// times `VRAM_{WIDTH,HEIGHT}`.
    scale: u32,
}

impl RenderTarget {
    /// Live constructor -- registers with egui for in-app paint.
    /// Initial scale = 1 (Native equivalent); the frontend bumps it
    /// via `ensure_scale` on every Native↔Window toggle / resize.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_renderer: &mut egui_wgpu::Renderer,
    ) -> Self {
        let scale = 1;
        let (texture, view) = create_target(device, scale);
        clear_to_black(device, queue, &view);
        let egui_id =
            egui_renderer.register_native_texture(device, &view, wgpu::FilterMode::Nearest);
        Self {
            texture,
            view,
            egui_id: Some(egui_id),
            scale,
        }
    }

    /// Headless constructor -- used by the parity harness / dump CLI.
    pub fn new_headless(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let scale = 1;
        let (texture, view) = create_target(device, scale);
        clear_to_black(device, queue, &view);
        Self {
            texture,
            view,
            egui_id: None,
            scale,
        }
    }

    /// Resize to a new internal-resolution multiplier. Cheap when
    /// `new_scale` is unchanged; reallocates + re-registers + clears
    /// to black otherwise. Cap at [`MAX_SCALE`].
    pub fn ensure_scale(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_renderer: Option<&mut egui_wgpu::Renderer>,
        new_scale: u32,
    ) {
        let s = new_scale.clamp(1, MAX_SCALE);
        if s == self.scale {
            return;
        }
        let (texture, view) = create_target(device, s);
        clear_to_black(device, queue, &view);
        if let (Some(id), Some(r)) = (self.egui_id, egui_renderer) {
            r.update_egui_texture_from_wgpu_texture(device, &view, wgpu::FilterMode::Nearest, id);
        }
        self.texture = texture;
        self.view = view;
        self.scale = s;
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn texture_id(&self) -> egui::TextureId {
        self.egui_id.unwrap_or_default()
    }

    pub fn scale(&self) -> u32 {
        self.scale
    }

    pub fn size(&self) -> (u32, u32) {
        (VRAM_WIDTH * self.scale, VRAM_HEIGHT * self.scale)
    }
}

fn create_target(device: &wgpu::Device, scale: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let s = scale.clamp(1, MAX_SCALE);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("psx-hw-vram-target"),
        size: wgpu::Extent3d {
            width: VRAM_WIDTH * s,
            height: VRAM_HEIGHT * s,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// One-shot clear to opaque black. Called when the target is
/// (re)allocated so subsequent steady-state passes can `LoadOp::Load`
/// without sampling undefined memory.
fn clear_to_black(device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("psx-hw-vram-target-init-clear"),
    });
    {
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("psx-hw-vram-target-init-clear-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
    }
    queue.submit(Some(encoder.finish()));
}
