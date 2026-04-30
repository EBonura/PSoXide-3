//! VRAM viewer -- renders the full 1024×512 VRAM as an image panel.
//!
//! The texture upload happens in `gfx::Graphics::prepare_vram`; this
//! module is purely the egui layout that places the image inside a
//! themed frame. A later milestone will add overlays for framebuffer
//! regions, texture pages, and CLUT rows.

use emulator_core::{VRAM_HEIGHT, VRAM_WIDTH};

use crate::theme;

/// Draw the VRAM panel anchored to the bottom of the window.
pub fn draw(ctx: &egui::Context, tex: egui::TextureId) {
    egui::TopBottomPanel::bottom("vram")
        .resizable(true)
        .default_height(ctx.screen_rect().height() * 0.32)
        .min_height(160.0)
        .show(ctx, |ui| {
            theme::viz_frame(ui, "VRAM (1024×512)", |ui| {
                paint_image_fit(
                    ui,
                    tex,
                    egui::vec2(VRAM_WIDTH as f32, VRAM_HEIGHT as f32),
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                );
            });
        });
}

/// Fit `tex_size` into the remaining space, preserving aspect ratio,
/// and paint the given texture (with `uv`) centered.
pub fn paint_image_fit(
    ui: &mut egui::Ui,
    tex: egui::TextureId,
    tex_size: egui::Vec2,
    uv: egui::Rect,
) {
    let avail = ui.available_rect_before_wrap();
    if avail.width() <= 0.0 || avail.height() <= 0.0 || tex_size.x <= 0.0 || tex_size.y <= 0.0 {
        return;
    }
    let scale = (avail.width() / tex_size.x).min(avail.height() / tex_size.y);
    let size = tex_size * scale;
    let rect = egui::Rect::from_center_size(avail.center(), size);
    ui.allocate_rect(avail, egui::Sense::hover());
    egui::Image::new((tex, size)).uv(uv).paint_at(ui, rect);
}
