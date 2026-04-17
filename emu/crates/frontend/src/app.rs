//! Top-level application state and UI orchestration.
//!
//! Owns the emulator state (currently just a `Cpu` + `Bus` — VRAM will
//! join once the GPU subsystem lands) and drives the per-frame UI build.

use emulator_core::Vram;

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

/// Top-level app state. Owns the emulator state directly — no Arc/Mutex,
/// single-threaded, UI reads state in-place per frame.
pub struct AppState {
    pub panels: PanelVisibility,
    /// VRAM viewer target. Once the GPU subsystem lands, `Gpu` will own
    /// this and expose `&Vram`; for now the frontend owns it so the
    /// viewer can render something (even if it's just zeros).
    pub vram: Vram,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            panels: PanelVisibility::default(),
            vram: Vram::new(),
        }
    }
}

/// Build all panels/overlays for one frame. Called from `gfx::Graphics::render`
/// inside the egui context.
pub fn build_ui(ctx: &egui::Context, state: &mut AppState, vram_tex: egui::TextureId) {
    ui::draw_layout(ctx, state, vram_tex);
}
