//! Top-level application state and UI orchestration.
//!
//! Owns the emulator state (currently just a `Cpu` + `Bus` — VRAM will
//! join once the GPU subsystem lands) and drives the per-frame UI build.

use std::path::PathBuf;

use emulator_core::{Bus, Cpu, Vram};

use crate::ui;

/// Default BIOS location. Matches the parity-test default so both
/// tooling converges on the same image in a fresh checkout.
const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

/// Panels that can be shown/hidden via the XMB.
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
    pub cpu: Cpu,
    /// Optional because we let the frontend run without a BIOS for UI
    /// development. If absent, register panels show the reset-state CPU
    /// but no instruction stepping is possible. Unused until the step
    /// button lands alongside the XMB.
    #[allow(dead_code)]
    pub bus: Option<Bus>,
    pub vram: Vram,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            panels: PanelVisibility::default(),
            cpu: Cpu::new(),
            bus: load_bus(),
            vram: Vram::new(),
        }
    }
}

fn load_bus() -> Option<Bus> {
    let path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS));
    match std::fs::read(&path) {
        Ok(bytes) => match Bus::new(bytes) {
            Ok(bus) => Some(bus),
            Err(e) => {
                eprintln!("[frontend] BIOS at {} rejected: {e}", path.display());
                None
            }
        },
        Err(e) => {
            eprintln!("[frontend] no BIOS at {}: {e}", path.display());
            None
        }
    }
}

/// Build all panels/overlays for one frame. Called from `gfx::Graphics::render`
/// inside the egui context.
pub fn build_ui(ctx: &egui::Context, state: &mut AppState, vram_tex: egui::TextureId) {
    ui::draw_layout(ctx, state, vram_tex);
}
