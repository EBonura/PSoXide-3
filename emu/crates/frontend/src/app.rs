//! Top-level application state and UI orchestration.
//!
//! Owns the emulator state (currently just a `Cpu` + `Bus` — VRAM will
//! join once the GPU subsystem lands) and drives the per-frame UI build.

use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;

use emulator_core::{Bus, Cpu};
use psx_trace::InstructionRecord;

use crate::ui;
use crate::ui::hud::HudState;
use crate::ui::memory::MemoryView;
use crate::ui::menu::MenuState;

/// Ring-buffer capacity for the execution-history panel. 64 rows fits
/// comfortably in the register side panel and is enough to recognise
/// most inner loops by eye.
pub const EXEC_HISTORY_CAP: usize = 64;

/// Default BIOS location. Matches the parity-test default so both
/// tooling converges on the same image in a fresh checkout.
const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

/// Panels that can be shown/hidden via the Menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelVisibility {
    pub registers: bool,
    pub memory: bool,
    pub vram: bool,
    pub hud: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            registers: true,
            memory: false,
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
    /// button lands alongside the Menu.
    pub bus: Option<Bus>,
    pub menu: MenuState,
    pub hud: HudState,
    pub memory_view: MemoryView,
    /// When true, the shell drives `cpu.step` at `run_steps_per_frame`
    /// instructions per redraw. Toggled via the Menu's Run/Pause item.
    pub running: bool,
    /// How many CPU instructions the run loop retires per frame when
    /// `running` is true. Tuned to stay real-time-ish on a modern host
    /// without overshooting VBlank granularity once timers land.
    pub run_steps_per_frame: u32,
    /// Rolling window of the last [`EXEC_HISTORY_CAP`] retired
    /// instructions, newest at the back. Driven by both single-step
    /// and continuous-run paths.
    pub exec_history: VecDeque<InstructionRecord>,
    /// PC addresses at which the run loop pauses. Toggled from the
    /// memory viewer; displayed in the register panel.
    pub breakpoints: BTreeSet<u32>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            panels: PanelVisibility::default(),
            cpu: Cpu::new(),
            bus: load_bus(),
            menu: MenuState::new(),
            hud: HudState::default(),
            memory_view: MemoryView::default(),
            running: false,
            run_steps_per_frame: 100_000,
            exec_history: VecDeque::with_capacity(EXEC_HISTORY_CAP),
            breakpoints: BTreeSet::new(),
        }
    }
}

/// Record a retired instruction into the ring buffer, evicting the
/// oldest entry when capacity is reached.
///
/// Free-function rather than a method so callers can borrow `AppState`
/// fields disjointly: `state.bus`, `state.cpu`, and
/// `state.exec_history` often need to be held mutably at once inside
/// the step loop, which a `&mut self` method would block.
pub fn push_history(history: &mut VecDeque<InstructionRecord>, record: InstructionRecord) {
    if history.len() >= EXEC_HISTORY_CAP {
        history.pop_front();
    }
    history.push_back(record);
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
/// inside the egui context. `dt` drives Menu animations.
pub fn build_ui(ctx: &egui::Context, state: &mut AppState, vram_tex: egui::TextureId, dt: f32) {
    ui::draw_layout(ctx, state, vram_tex, dt);
}
