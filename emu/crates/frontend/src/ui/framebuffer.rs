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

use emulator_core::{Bus, VRAM_HEIGHT, VRAM_WIDTH};

use crate::theme;
use crate::ui::vram as vram_panel;

const FB_WIDTH: f32 = 320.0;
const FB_HEIGHT: f32 = 240.0;

pub fn draw(ui: &mut egui::Ui, vram_tex: egui::TextureId, _bus: Option<&Bus>) {
    // When the GPU tracks display-start for real, replace these with
    // values from `_bus.as_ref().map(|b| b.gpu.display_start)`.
    let fb_x = 0.0_f32;
    let fb_y = 0.0_f32;

    theme::viz_frame(ui, "Framebuffer (320×240 at 0,0)", |ui| {
        let uv = egui::Rect::from_min_size(
            egui::pos2(fb_x / VRAM_WIDTH as f32, fb_y / VRAM_HEIGHT as f32),
            egui::vec2(FB_WIDTH / VRAM_WIDTH as f32, FB_HEIGHT / VRAM_HEIGHT as f32),
        );
        vram_panel::paint_image_fit(
            ui,
            vram_tex,
            egui::vec2(FB_WIDTH, FB_HEIGHT),
            uv,
        );
    });
}
