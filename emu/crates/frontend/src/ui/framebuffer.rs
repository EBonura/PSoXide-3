//! Framebuffer view for the central panel.
//!
//! The central panel presents the hardware renderer's VRAM-shaped
//! target. Native and high-resolution modes share this path and only
//! differ by the renderer's internal scale.

use emulator_core::{Bus, DisplayArea};

use crate::theme;

const DEFAULT_FB_WIDTH: f32 = 320.0;
const DEFAULT_FB_HEIGHT: f32 = 240.0;

/// CRT display aspect ratio for NTSC. The visible area is 4:3
/// regardless of which horizontal-resolution mode the game picks
/// (256/320/368/384/512/640) — a 512×240 frame is supposed to
/// squash horizontally on a real CRT, not stretch into 16:9.
const CRT_ASPECT: f32 = 4.0 / 3.0;

pub fn draw(
    ui: &mut egui::Ui,
    display_tex: egui::TextureId,
    bus: Option<&Bus>,
    present_size_px: &mut (u32, u32),
) {
    let reported = bus.map(|b| b.gpu.display_area()).unwrap_or(DisplayArea {
        x: 0,
        y: 0,
        width: DEFAULT_FB_WIDTH as u16,
        height: DEFAULT_FB_HEIGHT as u16,
        bpp24: false,
    });
    let area = if reported.width == 0 || reported.height == 0 {
        DisplayArea {
            x: reported.x,
            y: reported.y,
            width: DEFAULT_FB_WIDTH as u16,
            height: DEFAULT_FB_HEIGHT as u16,
            bpp24: reported.bpp24,
        }
    } else {
        reported
    };

    theme::viz_frame(ui, "", |ui| {
        let avail = ui.available_rect_before_wrap();
        if avail.width() <= 0.0 || avail.height() <= 0.0 {
            return;
        }
        let h = avail.height().min(avail.width() / CRT_ASPECT);
        let w = h * CRT_ASPECT;
        let rect = egui::Rect::from_center_size(avail.center(), egui::vec2(w, h));

        let pixels_per_point = ui.ctx().pixels_per_point().max(1.0);
        *present_size_px = (
            (rect.width() * pixels_per_point).round().max(1.0) as u32,
            (rect.height() * pixels_per_point).round().max(1.0) as u32,
        );

        let uv = egui::Rect::from_min_max(
            egui::pos2(
                area.x as f32 / psx_gpu_render::VRAM_WIDTH as f32,
                area.y as f32 / psx_gpu_render::VRAM_HEIGHT as f32,
            ),
            egui::pos2(
                (area.x + area.width) as f32 / psx_gpu_render::VRAM_WIDTH as f32,
                (area.y + area.height) as f32 / psx_gpu_render::VRAM_HEIGHT as f32,
            ),
        );

        ui.allocate_rect(avail, egui::Sense::hover());
        egui::Image::new((display_tex, rect.size()))
            .uv(uv)
            .paint_at(ui, rect);
    });
}
