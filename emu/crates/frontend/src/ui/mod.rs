//! UI panel orchestration.
//!
//! Individual panels live in submodules; `draw_layout` composes them.

use crate::app::AppState;

/// Paint every panel for this frame, in layering order.
pub fn draw_layout(ctx: &egui::Context, state: &mut AppState) {
    // Central area — will later host the PS1 framebuffer display.
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("PSoXide");
        ui.label("Phase 1b — theme applied, panels pending.");
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Panels:");
            ui.checkbox(&mut state.panels.registers, "Registers");
            ui.checkbox(&mut state.panels.vram, "VRAM");
            ui.checkbox(&mut state.panels.hud, "HUD");
        });
    });
}
