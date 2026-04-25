//! Framebuffer view for the central panel.
//!
//! The PS1 GPU displays a rectangle of VRAM as the active framebuffer.
//! The renderer uploads that active rectangle into a top-left-packed
//! RGBA texture before this panel draws it. Keeping the central view on
//! that decoded display texture matters for FMV: 24-bit MDEC frames are
//! stored in VRAM word-packed form and look wrong if sampled from the raw
//! 15-bit VRAM viewer texture.

use emulator_core::{Bus, DisplayArea, VRAM_HEIGHT, VRAM_WIDTH};

use crate::app::ScaleMode;
use crate::theme;

const DEFAULT_FB_WIDTH: f32 = 320.0;
const DEFAULT_FB_HEIGHT: f32 = 240.0;

pub fn draw(
    ui: &mut egui::Ui,
    display_tex: egui::TextureId,
    bus: Option<&Bus>,
    scale_mode: ScaleMode,
) {
    // Pull the GPU's configured display area when a Bus is live;
    // otherwise fall back to the canonical 320×240 @ (0, 0) layout.
    let reported = bus.map(|b| b.gpu.display_area()).unwrap_or(DisplayArea {
        x: 0,
        y: 0,
        width: DEFAULT_FB_WIDTH as u16,
        height: DEFAULT_FB_HEIGHT as u16,
        bpp24: false,
    });
    let using_fallback = reported.width == 0 || reported.height == 0;
    let area = if using_fallback {
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
        let uv = egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(
                area.width as f32 / VRAM_WIDTH as f32,
                area.height as f32 / VRAM_HEIGHT as f32,
            ),
        );
        let tex_size = egui::vec2(area.width as f32, area.height as f32);
        match scale_mode {
            ScaleMode::Window => paint_image_window(ui, display_tex, tex_size, uv),
            ScaleMode::Native => paint_image_native(ui, display_tex, tex_size, uv),
        }
    });
}

/// CRT display aspect ratio for NTSC — the visible area is 4:3
/// regardless of the PSX horizontal-resolution mode the game picks
/// (256/320/368/384/512/640). A 512×240 game frame is NOT supposed
/// to display with square pixels; real hardware squashes it
/// horizontally to hit the 4:3 CRT window. Emulating this squash in
/// Window mode is what stops Crash Bandicoot (which runs 512×240)
/// from looking comically wide-screen.
const CRT_ASPECT: f32 = 4.0 / 3.0;

/// Fit the framebuffer into the available space AT CRT ASPECT (4:3),
/// not at 1:1 texture pixels. This matches what a real PSX on a CRT
/// produces regardless of which horizontal resolution the game
/// picked — 256, 320, 512, and 640 all fill the same horizontal
/// span, with differently-shaped pixels.
///
/// Native mode (see `paint_image_native`) keeps exact 1× pixels for
/// inspection; this "Window" mode is for users who want the game to
/// fill the host window.
fn paint_image_window(
    ui: &mut egui::Ui,
    tex: egui::TextureId,
    tex_size: egui::Vec2,
    uv: egui::Rect,
) {
    let avail = ui.available_rect_before_wrap();
    if avail.width() <= 0.0 || avail.height() <= 0.0 || tex_size.x <= 0.0 || tex_size.y <= 0.0 {
        return;
    }
    // Largest 4:3 rectangle that fits inside `avail`. The horizontal
    // resolution of the framebuffer (512, 640, etc.) doesn't appear
    // here — the UV already maps the right sub-rectangle of VRAM and
    // egui handles the horizontal squash during sampling.
    let h = avail.height().min(avail.width() / CRT_ASPECT);
    let w = h * CRT_ASPECT;
    let rect = egui::Rect::from_center_size(avail.center(), egui::vec2(w, h));
    ui.allocate_rect(avail, egui::Sense::hover());
    egui::Image::new((tex, rect.size()))
        .uv(uv)
        .paint_at(ui, rect);
}

/// Native 1× presentation. Always shows the display rectangle at its
/// original framebuffer size, centered in the available space.
fn paint_image_native(
    ui: &mut egui::Ui,
    tex: egui::TextureId,
    tex_size: egui::Vec2,
    uv: egui::Rect,
) {
    let avail = ui.available_rect_before_wrap();
    if avail.width() <= 0.0 || avail.height() <= 0.0 || tex_size.x <= 0.0 || tex_size.y <= 0.0 {
        return;
    }
    let size = egui::vec2(
        tex_size.x.min(avail.width()),
        tex_size.y.min(avail.height()),
    );
    let rect = egui::Rect::from_center_size(avail.center(), size);
    ui.allocate_rect(avail, egui::Sense::hover());
    egui::Image::new((tex, size)).uv(uv).paint_at(ui, rect);
}
