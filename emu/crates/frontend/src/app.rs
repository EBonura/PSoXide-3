//! Top-level application state and UI orchestration.
//!
//! Owns the emulator state (currently just a `Cpu` + `Bus` — VRAM will
//! join once the GPU subsystem lands) and drives the per-frame UI build.

use crate::ui;

/// Panels that can be shown/hidden via the Menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelVisibility {
    pub registers: bool,
    pub vram: bool,
    pub hud: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            registers: true,
            vram: true,
            hud: true,
        }
    }
}

/// Minimal app state — will grow as phases land.
#[derive(Default)]
pub struct AppState {
    pub panels: PanelVisibility,
}

/// Build all panels/overlays for one frame. Called from `gfx::Graphics::render`
/// inside the egui context.
pub fn build_ui(ctx: &egui::Context, state: &mut AppState) {
    ui::draw_layout(ctx, state);
}
