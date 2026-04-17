//! Top-level application state and UI orchestration.
//!
//! Owns the emulator state (currently just a `Cpu` + `Bus` — VRAM will
//! join once the GPU subsystem lands) and drives the per-frame UI build.

use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;

use emulator_core::{Bus, Cpu};
use psx_iso::{Disc, Exe, SECTOR_BYTES};
use psx_trace::InstructionRecord;

use crate::ui;
use crate::ui::hud::HudState;
use crate::ui::memory::MemoryView;
use crate::ui::xmb::XmbState;

/// Ring-buffer capacity for the execution-history panel. 64 rows fits
/// comfortably in the register side panel and is enough to recognise
/// most inner loops by eye.
pub const EXEC_HISTORY_CAP: usize = 64;

/// Default BIOS location. Matches the parity-test default so both
/// tooling converges on the same image in a fresh checkout.
const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

/// Panels that can be shown/hidden via the XMB.
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
    /// button lands alongside the XMB.
    pub bus: Option<Bus>,
    pub xmb: XmbState,
    pub hud: HudState,
    pub memory_view: MemoryView,
    /// When true, the shell drives `cpu.step` at `run_steps_per_frame`
    /// instructions per redraw. Toggled via the XMB's Run/Pause item.
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
    /// Snapshot of `cpu.gprs()` at some point the user chose (via the
    /// register panel's "Snapshot" button). The panel highlights GPRs
    /// whose current value differs from the snapshot. Reset clears
    /// this along with the rest of the emulator state.
    pub gpr_snapshot: Option<[u32; 32]>,
}

impl Default for AppState {
    fn default() -> Self {
        let mut cpu = Cpu::new();
        let bus = load_bus().map(|mut bus| {
            // PSOXIDE_EXE side-loading. Copies the payload into RAM
            // and seeds the CPU to jump straight into the homebrew,
            // bypassing the BIOS reset vector. BIOS ROM stays
            // resident so SDK calls (FlushCache, putchar, …) keep
            // working.
            if let Some(exe) = load_exe() {
                bus.load_exe_payload(exe.load_addr, &exe.payload);
                cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());
                bus.enable_hle_bios();
                eprintln!(
                    "[frontend] side-loaded EXE: entry=0x{:08x} payload={}B (hle-bios on)",
                    exe.initial_pc,
                    exe.payload.len()
                );
            }
            bus
        });
        Self {
            panels: PanelVisibility::default(),
            cpu,
            bus,
            xmb: XmbState::new(),
            hud: HudState::default(),
            memory_view: MemoryView::default(),
            running: false,
            run_steps_per_frame: 100_000,
            exec_history: VecDeque::with_capacity(EXEC_HISTORY_CAP),
            breakpoints: BTreeSet::new(),
            gpr_snapshot: None,
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
    let mut bus = match std::fs::read(&path) {
        Ok(bytes) => match Bus::new(bytes) {
            Ok(bus) => bus,
            Err(e) => {
                eprintln!("[frontend] BIOS at {} rejected: {e}", path.display());
                return None;
            }
        },
        Err(e) => {
            eprintln!("[frontend] no BIOS at {}: {e}", path.display());
            return None;
        }
    };

    // Optional disc. Absence is not an error — BIOS boots fine without
    // one and just sits on the "insert disc" screen. Presence wires the
    // bytes into the CD-ROM controller's tray so `CdlGetID` / `CdlReadN`
    // return real data once the BIOS/game asks.
    if let Some(disc) = load_disc() {
        bus.cdrom.insert_disc(Some(disc));
    }

    Some(bus)
}

/// Read `PSOXIDE_EXE` → PSX-EXE file → parsed `Exe`. Logs and returns
/// `None` on any trouble so a misconfigured path doesn't wedge boot.
fn load_exe() -> Option<Exe> {
    let var = std::env::var("PSOXIDE_EXE").ok()?;
    let path = PathBuf::from(&var);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[frontend] PSOXIDE_EXE={} unreadable: {e}", path.display());
            return None;
        }
    };
    match Exe::parse(&bytes) {
        Ok(exe) => Some(exe),
        Err(e) => {
            eprintln!("[frontend] PSOXIDE_EXE={} malformed: {e:?}", path.display());
            None
        }
    }
}

/// Read `PSOXIDE_DISC` → BIN file → `Disc`. Logs and returns `None` on
/// any trouble so a misconfigured path doesn't wedge the frontend.
fn load_disc() -> Option<Disc> {
    let var = std::env::var("PSOXIDE_DISC").ok()?;
    let path = PathBuf::from(&var);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[frontend] PSOXIDE_DISC={} unreadable: {e}", path.display());
            return None;
        }
    };
    if bytes.len() < SECTOR_BYTES {
        eprintln!(
            "[frontend] PSOXIDE_DISC={} too small ({} bytes, need at least {SECTOR_BYTES})",
            path.display(),
            bytes.len()
        );
        return None;
    }
    let disc = Disc::from_bin(bytes);
    eprintln!(
        "[frontend] mounted disc {} ({} sectors)",
        path.display(),
        disc.sector_count()
    );
    Some(disc)
}

/// Build all panels/overlays for one frame. Called from `gfx::Graphics::render`
/// inside the egui context. `dt` drives XMB animations.
pub fn build_ui(ctx: &egui::Context, state: &mut AppState, vram_tex: egui::TextureId, dt: f32) {
    ui::draw_layout(ctx, state, vram_tex, dt);
}
