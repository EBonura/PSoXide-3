//! UI panel orchestration.
//!
//! Individual panels live in submodules; `draw_layout` composes them in
//! the order that makes the visual layering work: docked panels first
//! (bottom/side get clipped to remaining space), then the central area,
//! then free-floating overlays (Menu, HUD) on top.

pub mod vram;

use crate::app::AppState;

/// Paint every panel for this frame, in layering order.
pub fn draw_layout(ctx: &egui::Context, state: &mut AppState, vram_tex: egui::TextureId) {
    if state.panels.vram {
        vram::draw(ctx, vram_tex);
    }

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("PSoXide");
        ui.label("Phase 1c — VRAM viewer live. Central area hosts the framebuffer once GPU lands.");
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Panels:");
            ui.checkbox(&mut state.panels.registers, "Registers");
            ui.checkbox(&mut state.panels.vram, "VRAM");
            ui.checkbox(&mut state.panels.hud, "HUD");
        });
        ui.add_space(8.0);
        ui.label("VRAM poke test:");
        if ui.button("Fill with test pattern").clicked() {
            fill_vram_test_pattern(&mut state.vram);
        }
        if ui.button("Clear VRAM").clicked() {
            state.vram.clear();
        }
    });
}

fn fill_vram_test_pattern(vram: &mut emulator_core::Vram) {
    use emulator_core::{VRAM_HEIGHT, VRAM_WIDTH};
    // Gradient: R = x/32, G = y/16, B = (x^y) & 0x1F. Makes the viewer
    // instantly recognisable as "working" — distinct from the noise a
    // buggy upload would produce.
    for y in 0..VRAM_HEIGHT {
        for x in 0..VRAM_WIDTH {
            let r = ((x * 31 / VRAM_WIDTH) & 0x1F) as u16;
            let g = ((y * 31 / VRAM_HEIGHT) & 0x1F) as u16;
            let b = ((x ^ y) as u16) & 0x1F;
            let color = r | (g << 5) | (b << 10);
            vram.set_pixel(x as u16, y as u16, color);
        }
    }
}
