//! Top-level application state and UI orchestration.
//!
//! Owns the emulator state (currently just a `Cpu` + `Bus` — VRAM will
//! join once the GPU subsystem lands) and drives the per-frame UI build.

use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;

use emulator_core::{Bus, Cpu};
use psoxide_settings::{ConfigPaths, Library, LibraryEntry, Settings};
use psoxide_settings::library::{GameKind, Region};
use psx_iso::{Disc, Exe, SECTOR_BYTES};
use psx_trace::InstructionRecord;

use crate::ui;
use crate::ui::hud::HudState;
use crate::ui::memory::MemoryView;
use crate::ui::xmb::{LibraryItem as XmbLibraryItem, XmbState};

/// Ring-buffer capacity for the execution-history panel. 64 rows fits
/// comfortably in the register side panel and is enough to recognise
/// most inner loops by eye.
pub const EXEC_HISTORY_CAP: usize = 64;

/// Default BIOS location. Matches the parity-test default so both
/// tooling converges on the same image in a fresh checkout.
const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

/// Panels that can be shown/hidden via the XMB. The XMB *is* the
/// library browser (Games / Examples columns), so we don't have
/// a separate "library" panel — it's integrated into the shell
/// the PSX way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelVisibility {
    /// CPU registers + exec history side panel.
    pub registers: bool,
    /// Memory / disassembly dual-pane viewer.
    pub memory: bool,
    /// VRAM thumbnail at the bottom of the screen.
    pub vram: bool,
    /// Cycles / PC / FPS overlay in a corner.
    pub hud: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            // Debug panels collapsed on first run — the XMB is the
            // primary surface a fresh user sees, not the internals.
            registers: false,
            memory: false,
            vram: false,
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
    /// Persisted user preferences (BIOS path, library root, input
    /// mappings, video tweaks). Read at startup, re-saved when the
    /// settings panel commits changes. The frontend mutates this
    /// directly; the filesystem is written via
    /// [`AppState::save_settings`].
    pub settings: Settings,
    /// Cached library scan results. Populated from
    /// `<config>/library.ron` at startup, refreshed by
    /// [`AppState::rescan_library`] (triggered from the XMB's
    /// Games / Examples "Refresh library" row).
    pub library: Library,
    /// Resolved on-disk paths (settings.ron, library.ron, per-game
    /// subtree). Set once from the platform default or a
    /// `--config-dir` override and never mutated afterwards.
    pub paths: ConfigPaths,
    /// What the BIOS was asked to boot at the last launch. `None`
    /// = no game loaded yet (initial state on first run, also after
    /// "Reset" with no last-loaded game).
    pub current_game: Option<LibraryEntry>,
    /// Short-lived status line — shows "Launched Crash Bandicoot",
    /// "Scan complete: 54 games", etc. Displayed beneath the
    /// library panel; cleared after a few frames.
    pub status_message: Option<(String, f32)>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::with_config_dir(None)
    }
}

impl AppState {
    /// Build app state honouring an optional `--config-dir`
    /// override. `None` means "use the platform default" (the
    /// normal user path); `Some(p)` means "use this directory"
    /// (testing / portable installs).
    pub fn with_config_dir(override_dir: Option<PathBuf>) -> Self {
        // Resolve the config directory up-front. In production this
        // lives under ~/Library/Application Support/PSoXide-3
        // (macOS) etc; if the OS won't give us one we degrade to a
        // tempdir-rooted view so the app still runs — just without
        // persistence.
        let paths = match override_dir {
            Some(p) => ConfigPaths::rooted(p),
            None => ConfigPaths::platform_default().unwrap_or_else(|e| {
                eprintln!("[frontend] no platform config dir ({e}); persistence disabled");
                ConfigPaths::rooted(std::env::temp_dir().join("PSoXide-3-ephemeral"))
            }),
        };
        let _ = paths.ensure_dir(paths.root());

        let settings = match Settings::load(&paths.settings_file()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[frontend] settings load: {e}; using defaults");
                Settings::default()
            }
        };
        let library = Library::load_or_empty(&paths.library_file());

        // Legacy env-var side-load path: if PSOXIDE_EXE or
        // PSOXIDE_DISC is set, honour it so existing developer
        // workflows keep working. The library UI is the
        // forward path for everyone else.
        let mut cpu = Cpu::new();
        let bus = load_bus(&settings).map(|mut bus| {
            if let Some(exe) = load_exe() {
                bus.load_exe_payload(exe.load_addr, &exe.payload);
                cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());
                bus.enable_hle_bios();
                bus.attach_digital_pad_port1();
                eprintln!(
                    "[frontend] side-loaded EXE: entry=0x{:08x} payload={}B (hle-bios + pad1)",
                    exe.initial_pc,
                    exe.payload.len()
                );
            }
            bus
        });

        let mut out = Self {
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
            settings,
            library,
            paths,
            current_game: None,
            status_message: None,
        };
        // Seed the XMB's Games + Examples columns from the loaded
        // library so the user sees entries immediately instead of
        // a "No games found" placeholder.
        out.refresh_xmb_library();
        out
    }
}

impl AppState {
    /// Rebuild the emulator state around `entry`. Same flow the
    /// headless `launch` CLI runs: load BIOS, mount the disc or
    /// side-load the EXE, plug a pad into port 1. On success the
    /// emulator is paused at the reset vector (or the EXE entry
    /// point); the user clicks Run to start stepping.
    pub fn launch_entry(&mut self, entry: &LibraryEntry) -> Result<(), String> {
        let bios_path = resolve_bios_path(&self.settings);
        let bios = std::fs::read(&bios_path)
            .map_err(|e| format!("BIOS {}: {e}", bios_path.display()))?;
        let mut bus = Bus::new(bios).map_err(|e| format!("BIOS rejected: {e}"))?;
        let mut cpu = Cpu::new();

        match entry.kind {
            GameKind::Exe => {
                let bytes = std::fs::read(&entry.path)
                    .map_err(|e| format!("{}: {e}", entry.path.display()))?;
                let exe = Exe::parse(&bytes).map_err(|e| format!("parse EXE: {e:?}"))?;
                bus.load_exe_payload(exe.load_addr, &exe.payload);
                cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());
                if self.settings.emulator.hle_bios_for_side_load {
                    bus.enable_hle_bios();
                }
                bus.attach_digital_pad_port1();
            }
            GameKind::DiscBin | GameKind::DiscIso => {
                let bytes = std::fs::read(&entry.path)
                    .map_err(|e| format!("{}: {e}", entry.path.display()))?;
                if bytes.len() < SECTOR_BYTES {
                    return Err(format!(
                        "{} is too small to be a valid disc image",
                        entry.path.display()
                    ));
                }
                bus.cdrom.insert_disc(Some(Disc::from_bin(bytes)));
                bus.attach_digital_pad_port1();
            }
            GameKind::DiscCue => {
                return Err("CUE handling is pending — point me at the BIN directly".into());
            }
            GameKind::Unknown => {
                return Err(format!("unsupported game kind for {}", entry.path.display()));
            }
        }

        // Swap everything at once — no half-loaded state.
        self.bus = Some(bus);
        self.cpu = cpu;
        self.running = false;
        self.exec_history.clear();
        self.gpr_snapshot = None;
        self.current_game = Some(entry.clone());
        self.xmb.sync_run_label(false);
        self.status_message = Some((
            format!("Launched: {}", entry.title),
            STATUS_MESSAGE_TTL_SECS,
        ));
        Ok(())
    }

    /// Convenience: look up an entry by its stable ID and launch
    /// it. The XMB dispatches [`MenuAction::LaunchGame`] with only
    /// the ID, and we resolve it here so the XMB never needs a
    /// reference to the full library.
    pub fn launch_by_id(&mut self, id: &str) -> Result<(), String> {
        let entry = self
            .library
            .entries
            .iter()
            .find(|e| e.id == id)
            .cloned()
            .ok_or_else(|| format!("no library entry with id={id}"))?;
        self.launch_entry(&entry)
    }

    /// Walk the configured library root and update the cache. Uses
    /// `self.settings.paths.game_library` — if unset, returns an
    /// error so the UI can surface a "choose a library root"
    /// prompt. Also refreshes the XMB's Games + Examples columns
    /// so the newly-scanned entries appear immediately.
    pub fn rescan_library(&mut self) -> Result<usize, String> {
        if self.settings.paths.game_library.is_empty() {
            return Err(
                "Library root is not set. Configure paths.game_library in settings.ron."
                    .to_string(),
            );
        }
        let root = PathBuf::from(&self.settings.paths.game_library);
        if !root.exists() {
            return Err(format!("Library root does not exist: {}", root.display()));
        }
        let changed = self
            .library
            .scan(&root)
            .map_err(|e| format!("scan failed: {e}"))?;
        self.library
            .save(&self.paths.library_file())
            .map_err(|e| format!("save library.ron: {e}"))?;
        self.refresh_xmb_library();
        self.status_message = Some((
            format!("Scan complete: {} entries", self.library.entries.len()),
            STATUS_MESSAGE_TTL_SECS,
        ));
        Ok(changed)
    }

    /// Project the current library into the XMB's Games + Examples
    /// columns. Games get sorted by title (humans scan alphabetised
    /// lists faster); Examples get sorted by title too but live in
    /// their own column so commercial discs + homebrew don't share
    /// a list. Multi-track rips (audio tracks 2..N) are hidden —
    /// they're not independently bootable, just listing them
    /// pollutes the grid.
    pub fn refresh_xmb_library(&mut self) {
        let mut games: Vec<XmbLibraryItem> = Vec::new();
        let mut examples: Vec<XmbLibraryItem> = Vec::new();
        for e in &self.library.entries {
            let label = e
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("<unknown>");
            // Skip audio tracks — any filename with "(Track N)"
            // where N >= 2 (track 1 is always the data track).
            if label.contains("(Track ") && !label.contains("(Track 1)") {
                continue;
            }
            let subtitle = format_subtitle(e);
            let item = XmbLibraryItem {
                id: e.id.clone(),
                title: e.title.clone(),
                subtitle,
            };
            match e.kind {
                GameKind::Exe => examples.push(item),
                GameKind::DiscBin | GameKind::DiscIso | GameKind::DiscCue => games.push(item),
                GameKind::Unknown => {}
            }
        }
        games.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
        examples.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
        self.xmb.set_library(&games, &examples);
    }

    /// Persist the current `Settings` to `settings.ron`. Called
    /// when a settings-panel control commits a change.
    pub fn save_settings(&self) -> Result<(), String> {
        self.settings
            .save(&self.paths.settings_file())
            .map_err(|e| format!("save settings.ron: {e}"))
    }

    /// Decay the short-lived status message. Called once per frame
    /// with the frame's dt.
    pub fn tick_status(&mut self, dt: f32) {
        if let Some((_, ref mut ttl)) = self.status_message {
            *ttl -= dt;
            if *ttl <= 0.0 {
                self.status_message = None;
            }
        }
    }

    /// Show `msg` in the status toast for the standard TTL. Used
    /// by action handlers to surface success / failure from the
    /// XMB without allocating a whole notification subsystem.
    pub fn status_message_set(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), STATUS_MESSAGE_TTL_SECS));
    }
}

/// Seconds a status toast stays visible.
const STATUS_MESSAGE_TTL_SECS: f32 = 3.5;

/// Pick the BIOS path the launcher should read, honouring
/// precedence: explicit settings field > env var > compiled-in
/// fallback. Centralised so every caller agrees.
/// Format the right-aligned subtitle the XMB shows next to a
/// game's title. Keeps everything in one place so the Games and
/// Examples columns stay visually consistent.
fn format_subtitle(e: &LibraryEntry) -> String {
    let region = match e.region {
        Region::NtscU => "NTSC-U",
        Region::Pal => "PAL",
        Region::NtscJ => "NTSC-J",
        Region::Unknown => "",
    };
    let size_mib = e.size / (1024 * 1024);
    match (region.is_empty(), e.kind) {
        (false, GameKind::DiscBin | GameKind::DiscIso | GameKind::DiscCue) => {
            format!("{region} · {size_mib} MiB")
        }
        (true, GameKind::DiscBin | GameKind::DiscIso | GameKind::DiscCue) => {
            format!("{size_mib} MiB")
        }
        (_, GameKind::Exe) => {
            if e.size < 1024 {
                format!("{} B", e.size)
            } else if e.size < 1024 * 1024 {
                format!("{} KiB", e.size / 1024)
            } else {
                format!("{size_mib} MiB")
            }
        }
        _ => String::new(),
    }
}

fn resolve_bios_path(settings: &Settings) -> PathBuf {
    if !settings.paths.bios.is_empty() {
        PathBuf::from(&settings.paths.bios)
    } else if let Ok(p) = std::env::var("PSOXIDE_BIOS") {
        PathBuf::from(p)
    } else {
        PathBuf::from(DEFAULT_BIOS)
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

fn load_bus(settings: &Settings) -> Option<Bus> {
    let path = resolve_bios_path(settings);
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
