//! Framebuffer view for the central panel.
//!
//! The PS1 GPU displays a rectangle of VRAM as the active framebuffer.
//! Full fidelity needs the GP1-0x05 (display start) and GP1-0x08
//! (display mode) commands decoded into x / y / width / height, plus
//! 15-vs-24-bit colour depth selection. Phase 4 ships with a fixed
//! 320×240 view at `(0, 0)` in VRAM — the default BIOS layout — so
//! anything the emulator writes into the top-left quadrant of VRAM
//! (fill-rect commands, DMA uploads, the frontend's test pattern)
//! becomes visible immediately.
//!
//! Once the real GPU command decoder + display-start tracking lands
//! we pull the rectangle from `bus.gpu`. Until then, `_bus` is
//! accepted as a hook for that migration.

use emulator_core::{Bus, DisplayArea, VRAM_HEIGHT, VRAM_WIDTH};

use crate::app::ScaleMode;
use crate::theme;

const DEFAULT_FB_WIDTH: f32 = 320.0;
const DEFAULT_FB_HEIGHT: f32 = 240.0;

pub fn draw(
    ui: &mut egui::Ui,
    vram_tex: egui::TextureId,
    bus: Option<&Bus>,
    scale_mode: ScaleMode,
) {
    // Pull the GPU's configured display area when a Bus is live;
    // otherwise fall back to the canonical 320×240 @ (0, 0) layout.
    let area = bus.map(|b| b.gpu.display_area()).unwrap_or(DisplayArea {
        x: 0,
        y: 0,
        width: DEFAULT_FB_WIDTH as u16,
        height: DEFAULT_FB_HEIGHT as u16,
        bpp24: false,
    });

    let mode_label = match scale_mode {
        ScaleMode::Fit => "fit",
        ScaleMode::Integer => "native",
    };
    let title = format!(
        "Framebuffer ({}×{} at {},{}){}  [{}]",
        area.width,
        area.height,
        area.x,
        area.y,
        if area.bpp24 { " · 24bpp" } else { "" },
        mode_label,
    );

    theme::viz_frame(ui, &title, |ui| {
        let uv = egui::Rect::from_min_size(
            egui::pos2(
                area.x as f32 / VRAM_WIDTH as f32,
                area.y as f32 / VRAM_HEIGHT as f32,
            ),
            egui::vec2(
                area.width as f32 / VRAM_WIDTH as f32,
                area.height as f32 / VRAM_HEIGHT as f32,
            ),
        );
        let tex_size = egui::vec2(area.width as f32, area.height as f32);
        match scale_mode {
            ScaleMode::Fit => paint_image_fit(ui, vram_tex, tex_size, uv),
            ScaleMode::Integer => paint_image_integer(ui, vram_tex, tex_size, uv),
        }
    });
}

/// CRT display aspect ratio for NTSC — the visible area is 4:3
/// regardless of the PSX horizontal-resolution mode the game picks
/// (256/320/368/384/512/640). A 512×240 game frame is NOT supposed
/// to display with square pixels; real hardware squashes it
/// horizontally to hit the 4:3 CRT window. Emulating this squash in
/// Fit mode is what stops Crash Bandicoot (which runs 512×240)
/// from looking comically wide-screen.
const CRT_ASPECT: f32 = 4.0 / 3.0;

/// Fit the framebuffer into the available space AT CRT ASPECT (4:3),
/// not at 1:1 texture pixels. This matches what a real PSX on a CRT
/// produces regardless of which horizontal resolution the game
/// picked — 256, 320, 512, and 640 all fill the same horizontal
/// span, with differently-shaped pixels.
///
/// Integer mode (see `paint_image_integer`) keeps 1:1 pixels for
/// pixel-art fans; this "Fit" mode is for users who want the game
/// to look like it did on a TV.
fn paint_image_fit(
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
    egui::Image::new((tex, rect.size())).uv(uv).paint_at(ui, rect);
}

/// Integer-scale (nearest-neighbour multiple) fit. Always shows
/// the texture at an exact 1× / 2× / 3× / … factor — letterboxes
/// with the background when the window isn't a clean multiple.
/// Pixel-perfect, which is what the user means by "PS1 original
/// resolution" — no interpolation stretch blur.
fn paint_image_integer(
    ui: &mut egui::Ui,
    tex: egui::TextureId,
    tex_size: egui::Vec2,
    uv: egui::Rect,
) {
    let avail = ui.available_rect_before_wrap();
    if avail.width() <= 0.0 || avail.height() <= 0.0 || tex_size.x <= 0.0 || tex_size.y <= 0.0 {
        return;
    }
    let max_scale_x = (avail.width() / tex_size.x).floor().max(1.0);
    let max_scale_y = (avail.height() / tex_size.y).floor().max(1.0);
    let scale = max_scale_x.min(max_scale_y);
    let size = tex_size * scale;
    let rect = egui::Rect::from_center_size(avail.center(), size);
    ui.allocate_rect(avail, egui::Sense::hover());
    egui::Image::new((tex, size)).uv(uv).paint_at(ui, rect);
}
