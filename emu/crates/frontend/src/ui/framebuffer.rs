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

use crate::theme;
use crate::ui::vram as vram_panel;

const DEFAULT_FB_WIDTH: f32 = 320.0;
const DEFAULT_FB_HEIGHT: f32 = 240.0;

pub fn draw(ui: &mut egui::Ui, vram_tex: egui::TextureId, bus: Option<&Bus>) {
    // Pull the GPU's configured display area when a Bus is live;
    // otherwise fall back to the canonical 320×240 @ (0, 0) layout.
    let area = bus
        .map(|b| b.gpu.display_area())
        .unwrap_or(DisplayArea {
            x: 0,
            y: 0,
            width: DEFAULT_FB_WIDTH as u16,
            height: DEFAULT_FB_HEIGHT as u16,
            bpp24: false,
        });

    let title = format!(
        "Framebuffer ({}×{} at {},{}){}",
        area.width,
        area.height,
        area.x,
        area.y,
        if area.bpp24 { " · 24bpp" } else { "" }
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
        vram_panel::paint_image_fit(
            ui,
            vram_tex,
            egui::vec2(area.width as f32, area.height as f32),
            uv,
        );
    });
}
