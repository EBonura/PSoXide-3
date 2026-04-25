//! Render target for the HW renderer.
//!
//! Owns the output `wgpu::Texture` + its `egui::TextureId`. The
//! frontend gets the id once at startup, but the underlying
//! texture is re-allocated whenever the requested size changes
//! (display rect change → resize, panel resize → resize). Each
//! re-allocation re-registers with `egui_wgpu::Renderer` so the
//! id stays valid for paint-time consumers.

use wgpu::TextureFormat;

/// Format of the HW renderer's output. `Rgba8UnormSrgb` matches
/// the surface format used elsewhere (egui's central panel
/// composes everything in linear-from-sRGB), so egui's blit
/// requires no extra colorspace conversion.
pub const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

pub struct RenderTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    egui_id: egui::TextureId,
    width: u32,
    height: u32,
}

impl RenderTarget {
    /// Allocate a 1×1 placeholder so the target is in a valid state
    /// before the first frame's `ensure` call sizes it properly.
    /// 1×1 is the smallest non-zero size wgpu accepts; egui
    /// won't paint a meaningful image until `ensure` resizes it.
    pub fn new(device: &wgpu::Device, egui_renderer: &mut egui_wgpu::Renderer) -> Self {
        let (texture, view) = create_target(device, 1, 1);
        let egui_id =
            egui_renderer.register_native_texture(device, &view, wgpu::FilterMode::Nearest);
        Self {
            texture,
            view,
            egui_id,
            width: 1,
            height: 1,
        }
    }

    /// Re-allocate the texture if the requested size has changed.
    /// Cheap when the size is stable.
    pub fn ensure(
        &mut self,
        device: &wgpu::Device,
        egui_renderer: &mut egui_wgpu::Renderer,
        new_w: u32,
        new_h: u32,
    ) {
        if new_w == self.width && new_h == self.height {
            return;
        }
        let (texture, view) = create_target(device, new_w, new_h);
        // Re-register with egui in place — same `TextureId`, new
        // backing view. Keeps the central panel's reference stable.
        egui_renderer.update_egui_texture_from_wgpu_texture(
            device,
            &view,
            wgpu::FilterMode::Nearest,
            self.egui_id,
        );
        self.texture = texture;
        self.view = view;
        self.width = new_w;
        self.height = new_h;
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn texture_id(&self) -> egui::TextureId {
        self.egui_id
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

fn create_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("psx-hw-render-target"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
