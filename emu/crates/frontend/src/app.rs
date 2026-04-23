//! Top-level application state and UI orchestration.
//!
//! Owns the emulator state (currently just a `Cpu` + `Bus` — VRAM will
//! join once the GPU subsystem lands) and drives the per-frame UI build.

use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;

use emulator_core::{
    fast_boot_disc_with_hle, warm_bios_for_disc_fast_boot, Bus, Cpu, DISC_FAST_BOOT_WARMUP_STEPS,
};
use psoxide_settings::library::{GameKind, Region};
use psoxide_settings::{ConfigPaths, Library, LibraryEntry, Settings};
use psx_iso::{Disc, Exe, SECTOR_BYTES};
use psx_trace::InstructionRecord;

use crate::ui;
use crate::ui::hud::HudState;
use crate::ui::memory::MemoryView;
use crate::ui::menu::{LibraryItem as MenuLibraryItem, MenuState};

/// Ring-buffer capacity for the execution-history panel. 16 rows is
/// the "what just ran" context window — enough to spot a tight loop
/// or trace a branch without the history section taking over the
/// registers side panel vertically.
pub const EXEC_HISTORY_CAP: usize = 16;

/// Default BIOS location. Matches the parity-test default so both
/// tooling converges on the same image in a fresh checkout.
const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

/// Panels that can be shown/hidden via the Menu. The Menu *is* the
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
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            // Debug panels collapsed on first run — the Menu is the
            // primary surface a fresh user sees, not the internals.
            registers: false,
            memory: false,
            vram: false,
        }
    }
}

/// How the framebuffer panel scales the PSX display into the host
/// window. Toggled from the debug toolbar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    /// Stretch to fit the available area while preserving aspect
    /// ratio. Fractional scale factors — smooth but may visibly
    /// interpolate on integer-pixel art.
    Fit,
    /// Integer-multiple scale (1×, 2×, 3×, …) fitting inside the
    /// available area. Pixel-perfect — the PSX's native res shows
    /// with no interpolation or stretch. Letterboxes as needed.
    Integer,
}

impl Default for ScaleMode {
    fn default() -> Self {
        Self::Fit
    }
}

/// Top-level app state. Owns the emulator state directly — no Arc/Mutex,
/// single-threaded, UI reads state in-place per frame.
pub struct AppState {
    pub panels: PanelVisibility,
    /// How the framebuffer scales — `Fit` (stretch with aspect
    /// preserved) vs `Integer` (1×/2×/3× pixel-perfect, letterbox
    /// as needed). Toggled via the debug toolbar.
    pub scale_mode: ScaleMode,
    pub cpu: Cpu,
    /// Optional because we let the frontend run without a BIOS for UI
    /// development. If absent, register panels show the reset-state CPU
    /// but no instruction stepping is possible. Unused until the step
    /// button lands alongside the Menu.
    pub bus: Option<Bus>,
    pub menu: MenuState,
    pub hud: HudState,
    pub memory_view: MemoryView,
    /// When true, the shell advances emulation on each redraw. Toggled
    /// via the Menu's Run/Pause item.
    pub running: bool,
    /// Safety cap for one frontend frame. The run loop targets PSX
    /// master-clock cycles, not this many instructions, but the cap
    /// prevents a broken guest from spinning forever in one redraw.
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
    /// [`AppState::rescan_library`] (triggered from the Menu's
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
        // lives under ~/Library/Application Support/PSoXide
        // (macOS) etc; if the OS won't give us one we degrade to a
        // tempdir-rooted view so the app still runs — just without
        // persistence.
        let paths = match override_dir {
            Some(p) => ConfigPaths::rooted(p),
            None => ConfigPaths::platform_default().unwrap_or_else(|e| {
                eprintln!("[frontend] no platform config dir ({e}); persistence disabled");
                ConfigPaths::rooted(std::env::temp_dir().join("PSoXide-ephemeral"))
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
                bus.clear_exe_bss(exe.bss_addr, exe.bss_size);
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
            scale_mode: ScaleMode::default(),
            cpu,
            bus,
            menu: MenuState::new(),
            hud: HudState::default(),
            memory_view: MemoryView::default(),
            running: false,
            run_steps_per_frame: 1_000_000,
            exec_history: VecDeque::with_capacity(EXEC_HISTORY_CAP),
            breakpoints: BTreeSet::new(),
            gpr_snapshot: None,
            settings,
            library,
            paths,
            current_game: None,
            status_message: None,
        };
        // Startup auto-rescan: always run when the SDK-examples dir
        // exists so stale `library.ron` entries (e.g. cargo
        // `deps/<name>-<hash>.exe` intermediates picked up by an
        // earlier version of the scanner before the deps/ filter
        // landed) get purged. `scan_roots` is mtime-cached for
        // already-seen files, so the cost is bounded by
        // "number of files that changed since last scan" — cheap
        // on every boot.
        //
        // Scoped to "SDK dir exists" so an end-user install without
        // the MIPS toolchain doesn't pay the cost every startup.
        if let Some(sdk_dir) = out.resolve_sdk_examples_dir() {
            if sdk_dir.exists() {
                if let Err(e) = out.rescan_library() {
                    eprintln!("[frontend] startup auto-rescan skipped: {e}");
                }
            }
        }
        // Seed the Menu's Games + Examples columns from the (now
        // possibly-rescanned) library so the user sees entries
        // immediately instead of a "No games found" placeholder.
        out.refresh_menu_library();
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
        // Flush the outgoing game's memcard before we discard its
        // Bus state. Silently log on failure — we'd rather launch
        // the new game than refuse because of a stale save.
        if let Err(e) = self.flush_memcard_port1() {
            eprintln!("[frontend] memcard flush before launch: {e}");
        }
        let bios_path = resolve_bios_path(&self.settings);
        let bios =
            std::fs::read(&bios_path).map_err(|e| format!("BIOS {}: {e}", bios_path.display()))?;
        let mut bus = Bus::new(bios).map_err(|e| format!("BIOS rejected: {e}"))?;
        let mut cpu = Cpu::new();

        match entry.kind {
            GameKind::Exe => {
                let bytes = std::fs::read(&entry.path)
                    .map_err(|e| format!("{}: {e}", entry.path.display()))?;
                let exe = Exe::parse(&bytes).map_err(|e| format!("parse EXE: {e:?}"))?;
                bus.load_exe_payload(exe.load_addr, &exe.payload);
                bus.clear_exe_bss(exe.bss_addr, exe.bss_size);
                cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());
                // HLE BIOS is effectively mandatory for side-loaded
                // EXEs: the kernel's syscall tables (A0 / B0 / C0)
                // + cold-init state aren't populated when we jump
                // straight to the EXE entry instead of the reset
                // vector. Previously gated on
                // `settings.emulator.hle_bios_for_side_load` — the
                // gate stayed on `false` (derived Default) for
                // users with a pre-existing settings.ron, which
                // made EXEs launched from the Menu render blank
                // while the env-var path `PSOXIDE_EXE=…` (HLE
                // unconditional) worked. Both paths now match.
                bus.enable_hle_bios();
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
                let disc = Disc::from_bin(bytes);
                maybe_fast_boot_disc(&mut bus, &mut cpu, &disc, entry, self.settings.emulator.fast_boot_disc);
                bus.cdrom.insert_disc(Some(disc));
                bus.attach_digital_pad_port1();
                // Load + attach the per-game memory card on port 1.
                // File lives under `<config>/games/<id>/memcard-1.mcd`;
                // first launch of any game gets a fresh 128 KiB blank.
                self.paths
                    .ensure_game_tree(&entry.id)
                    .map_err(|e| e.to_string())?;
                let mc_path = self.paths.memcard_file(&entry.id, 1);
                let mc_bytes = std::fs::read(&mc_path).unwrap_or_default();
                bus.attach_memcard_port1(mc_bytes);
            }
            GameKind::DiscCue => {
                let disc = psoxide_settings::library::load_disc_from_cue(&entry.path)?;
                maybe_fast_boot_disc(&mut bus, &mut cpu, &disc, entry, self.settings.emulator.fast_boot_disc);
                bus.cdrom.insert_disc(Some(disc));
                bus.attach_digital_pad_port1();
                self.paths
                    .ensure_game_tree(&entry.id)
                    .map_err(|e| e.to_string())?;
                let mc_path = self.paths.memcard_file(&entry.id, 1);
                let mc_bytes = std::fs::read(&mc_path).unwrap_or_default();
                bus.attach_memcard_port1(mc_bytes);
            }
            GameKind::Unknown => {
                return Err(format!(
                    "unsupported game kind for {}",
                    entry.path.display()
                ));
            }
        }

        // Swap everything at once — no half-loaded state. Start in
        // the running state so the user sees the game boot
        // immediately when they hit Enter in the Menu — matches a real
        // PS1 where selecting a disc and pressing X fires it right up.
        // The Menu's caller (`apply_menu_action::LaunchGame`) closes
        // the overlay so the game is actually visible.
        self.bus = Some(bus);
        self.cpu = cpu;
        self.running = true;
        self.exec_history.clear();
        self.gpr_snapshot = None;
        self.current_game = Some(entry.clone());
        self.menu.sync_run_label(true);
        self.status_message = Some((
            format!("Launched: {}", entry.title),
            STATUS_MESSAGE_TTL_SECS,
        ));
        Ok(())
    }

    /// Convenience: look up an entry by its stable ID and launch
    /// it. The Menu dispatches [`MenuAction::LaunchGame`] with only
    /// the ID, and we resolve it here so the Menu never needs a
    /// reference to the full library.
    pub fn launch_by_id(&mut self, id: &str) -> Result<(), String> {
        let Some(entry) = self.library.entries.iter().find(|e| e.id == id).cloned() else {
            return Err(format!("no library entry with id={id}"));
        };
        self.launch_entry(&entry)
    }

    /// Walk the configured library root(s) and update the cache.
    /// Scans TWO roots in one pass:
    ///
    /// 1. `settings.paths.game_library` — user's retail-disc folder.
    /// 2. `settings.paths.sdk_examples` (or auto-detected
    ///    `build/examples/mipsel-sony-psx/release/` under the repo
    ///    root) — `.exe` homebrew built by `make examples`.
    ///
    /// Either can be missing without erroring. If neither yields
    /// entries, the Menu's columns show the "No … found" placeholder
    /// instead of blowing up.
    ///
    /// Also refreshes the Menu's Games + Examples columns so the
    /// newly-scanned entries appear immediately.
    pub fn rescan_library(&mut self) -> Result<usize, String> {
        let game_root = if self.settings.paths.game_library.is_empty() {
            None
        } else {
            Some(PathBuf::from(&self.settings.paths.game_library))
        };
        let sdk_root = self.resolve_sdk_examples_dir();

        // No roots → still not an error; the UI shows empty columns.
        // Matches the "fresh clone, user hasn't set a library yet"
        // state rather than punishing it with a dialog.
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Some(g) = game_root.clone() {
            if g.exists() {
                roots.push(g);
            } else {
                return Err(format!("Library root does not exist: {}", g.display()));
            }
        }
        if let Some(s) = sdk_root.clone() {
            // sdk_root from auto-detect may not exist (e.g. on an
            // end-user install that never built the examples); that
            // doesn't deserve an error. `scan_roots` silently skips
            // missing roots for exactly this reason.
            roots.push(s);
        }

        let root_refs: Vec<&std::path::Path> = roots.iter().map(|p| p.as_path()).collect();
        let changed = self
            .library
            .scan_roots(&root_refs)
            .map_err(|e| format!("scan failed: {e}"))?;
        self.library
            .save(&self.paths.library_file())
            .map_err(|e| format!("save library.ron: {e}"))?;
        self.refresh_menu_library();
        let sdk_hint = match &sdk_root {
            Some(p) if p.exists() => format!(" (SDK: {})", p.display()),
            _ => String::new(),
        };
        self.status_message = Some((
            format!(
                "Scan complete: {} entries{sdk_hint}",
                self.library.entries.len()
            ),
            STATUS_MESSAGE_TTL_SECS,
        ));
        Ok(changed)
    }

    /// Resolve where to look for SDK-built example `.exe`s. Honours
    /// the explicit `settings.paths.sdk_examples` if the user set
    /// one; otherwise walks up from the frontend crate's source
    /// directory (`CARGO_MANIFEST_DIR`) to the repo root and joins
    /// the canonical build-output path. Returns `None` when the
    /// resolver can't place the repo root — in which case scanning
    /// proceeds with only the game-library root.
    fn resolve_sdk_examples_dir(&self) -> Option<PathBuf> {
        if !self.settings.paths.sdk_examples.is_empty() {
            return Some(PathBuf::from(&self.settings.paths.sdk_examples));
        }
        // `emu/crates/frontend/` → four `..`s land at the repo root.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest.parent()?.parent()?.parent()?;
        let candidate = repo_root.join("build/examples/mipsel-sony-psx/release");
        Some(candidate)
    }

    /// Project the current library into the Menu's Games + Examples
    /// columns. Three passes:
    ///
    /// 1. Walk every CUE entry and parse it to find its primary
    ///    (data-track) BIN. Build a map
    ///    `absolute_bin_path → (cue_title, cue_id)` so each BIN
    ///    the CUE owns shows up with the CUE's friendly filename
    ///    as its title (e.g. "Crash Bandicoot (USA)" instead of
    ///    the raw PVD ID "SCUS-94900"), and under the CUE's stable
    ///    game ID so savestates key off the disc identity rather
    ///    than the BIN byte hash alone.
    /// 2. Walk every entry. For BIN entries: drop multi-track
    ///    audio rips (Track 2..N), prefer the CUE-linked title if
    ///    one exists, and skip BINs that map to the *same* CUE as
    ///    an earlier BIN (dedup). For CUE entries: hidden from the
    ///    games list because the owning BIN already appears there
    ///    under the CUE's title/ID. For EXE entries: into Examples.
    /// 3. Alphabetise each column.
    ///
    /// Result: Crash Bandicoot shows once, under its friendly
    /// title, and clicking it launches the BIN.
    pub fn refresh_menu_library(&mut self) {
        use std::collections::HashMap;

        // Pass 1: map "BIN path" → (CUE-derived title, CUE id).
        let mut cue_owns_bin: HashMap<PathBuf, (String, String)> = HashMap::new();
        for e in &self.library.entries {
            if e.kind != GameKind::DiscCue {
                continue;
            }
            if let Some(bin) = psoxide_settings::library::primary_bin_from_cue(&e.path) {
                cue_owns_bin.insert(bin, (e.title.clone(), e.id.clone()));
            }
        }

        // Pass 2: project entries, applying dedup + title overrides.
        let mut games: Vec<MenuLibraryItem> = Vec::new();
        let mut examples: Vec<MenuLibraryItem> = Vec::new();
        let mut cue_already_listed: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for e in &self.library.entries {
            let label = e
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("<unknown>");

            // Audio tracks: any "(Track N)" filename where N != 1.
            // Multi-track CUE rips leave each audio track as a
            // standalone BIN; none of those boot, so hide them.
            if label.contains("(Track ")
                && !label.contains("(Track 01)")
                && !label.contains("(Track 1)")
            {
                continue;
            }

            match e.kind {
                // CUEs are never shown directly — their BIN is.
                GameKind::DiscCue => continue,
                GameKind::DiscBin | GameKind::DiscIso => {
                    // If a CUE owns this BIN, use the CUE's
                    // friendly title + stable ID. Also dedup: the
                    // *first* BIN of a CUE wins; subsequent BINs
                    // (multi-disc sets not yet modelled) are
                    // hidden to keep the list clean.
                    let (title, id) = if let Some((cue_title, cue_id)) = cue_owns_bin.get(&e.path) {
                        if !cue_already_listed.insert(cue_id.clone()) {
                            continue;
                        }
                        (cue_title.clone(), cue_id.clone())
                    } else {
                        (e.title.clone(), e.id.clone())
                    };
                    games.push(MenuLibraryItem {
                        id,
                        title,
                        subtitle: format_subtitle(e),
                    });
                }
                GameKind::Exe => examples.push(MenuLibraryItem {
                    id: e.id.clone(),
                    title: e.title.clone(),
                    subtitle: format_subtitle(e),
                }),
                GameKind::Unknown => {}
            }
        }

        // Pass 3: stable alphabetical order per column.
        games.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
        examples.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
        self.menu.set_library(&games, &examples);
    }

    /// Persist the current `Settings` to `settings.ron`. Called
    /// when a settings-panel control commits a change.
    pub fn save_settings(&self) -> Result<(), String> {
        self.settings
            .save(&self.paths.settings_file())
            .map_err(|e| format!("save settings.ron: {e}"))
    }

    /// Flush any dirty memory-card state on port 1 back to its
    /// `<config>/games/<id>/memcard-1.mcd` file. A no-op when no
    /// card is attached or when no writes have landed since load.
    /// Called from the shell's exit path and periodically during
    /// run so a hard crash doesn't lose save progress.
    pub fn flush_memcard_port1(&mut self) -> Result<(), String> {
        let Some(game) = self.current_game.as_ref().map(|g| g.id.clone()) else {
            return Ok(()); // no game loaded → nothing to persist
        };
        let Some(bus) = self.bus.as_mut() else {
            return Ok(());
        };
        if let Some(bytes) = bus.memcard_port1_snapshot() {
            let path = self.paths.memcard_file(&game, 1);
            self.paths
                .ensure_game_tree(&game)
                .map_err(|e| e.to_string())?;
            std::fs::write(&path, &bytes)
                .map_err(|e| format!("save memcard {}: {e}", path.display()))?;
            eprintln!(
                "[frontend] persisted port-1 memcard → {} ({} bytes)",
                path.display(),
                bytes.len()
            );
        }
        Ok(())
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
    /// Menu without allocating a whole notification subsystem.
    pub fn status_message_set(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), STATUS_MESSAGE_TTL_SECS));
    }
}

/// Seconds a status toast stays visible.
const STATUS_MESSAGE_TTL_SECS: f32 = 3.5;

/// Pick the BIOS path the launcher should read, honouring
/// precedence: explicit settings field > env var > compiled-in
/// fallback. Centralised so every caller agrees.
/// Format the right-aligned subtitle the Menu shows next to a
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

/// Retire enough instructions to cover one NTSC frame's worth of PSX
/// master-clock cycles. Any execution error auto-pauses, reopens the
/// Menu, and surfaces the stopped state via the register panel. Hitting
/// a breakpoint does the same. Split out here (rather than living in
/// the shell loop) so both the shell's per-frame run path and the
/// toolbar's "advance one frame" button can invoke the same logic.
pub fn step_one_frame(state: &mut AppState) {
    const PSX_MASTER_CLOCK_HZ: u64 = 33_868_800;
    const NTSC_FRAMES_PER_SECOND: u64 = 60;
    const CYCLES_PER_FRAME: u64 = PSX_MASTER_CLOCK_HZ / NTSC_FRAMES_PER_SECOND;

    let max_steps = state.run_steps_per_frame.max(1);
    let Some(bus) = state.bus.as_mut() else {
        state.running = false;
        state.menu.sync_run_label(false);
        return;
    };

    let target_cycles = bus.cycles().saturating_add(CYCLES_PER_FRAME);
    for _ in 0..max_steps {
        if bus.cycles() >= target_cycles {
            break;
        }
        // Breakpoint check happens BEFORE stepping so the paused PC
        // is the BP address itself — the instruction at that PC has
        // not yet executed.
        if state.breakpoints.contains(&state.cpu.pc()) {
            state.running = false;
            state.menu.sync_run_label(false);
            state.menu.open = true;
            break;
        }

        match state.cpu.step(bus) {
            Ok(record) => {
                push_history(&mut state.exec_history, record);
            }
            Err(_) => {
                state.running = false;
                state.menu.sync_run_label(false);
                state.menu.open = true;
                break;
            }
        }
    }
}

fn maybe_fast_boot_disc(
    bus: &mut Bus,
    cpu: &mut Cpu,
    disc: &Disc,
    entry: &LibraryEntry,
    enabled: bool,
) {
    if !enabled {
        return;
    }
    if let Err(e) = warm_bios_for_disc_fast_boot(bus, cpu, DISC_FAST_BOOT_WARMUP_STEPS) {
        eprintln!(
            "[frontend] BIOS warmup failed for {} ({e:?}); falling back to BIOS boot",
            entry.path.display()
        );
        return;
    }
    match fast_boot_disc_with_hle(bus, cpu, disc, false) {
        Ok(info) => eprintln!(
            "[frontend] warm-fast-booted {} via {} entry=0x{:08x} load=0x{:08x} payload={}B",
            entry.path.display(),
            info.boot_path,
            info.initial_pc,
            info.load_addr,
            info.payload_len
        ),
        Err(e) => eprintln!(
            "[frontend] fast boot unavailable for {} ({e:?}); falling back to BIOS boot",
            entry.path.display()
        ),
    }
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

/// Read `PSOXIDE_DISC` → disc image → `Disc`. Accepts raw BIN/ISO and
/// CUE-backed multitrack images. Logs and returns `None` on any trouble
/// so a misconfigured path doesn't wedge the frontend.
fn load_disc() -> Option<Disc> {
    let var = std::env::var("PSOXIDE_DISC").ok()?;
    let path = PathBuf::from(&var);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let disc = if ext == "cue" {
        match psoxide_settings::library::load_disc_from_cue(&path) {
            Ok(disc) => disc,
            Err(e) => {
                eprintln!(
                    "[frontend] PSOXIDE_DISC={} unreadable CUE: {e}",
                    path.display()
                );
                return None;
            }
        }
    } else {
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
        Disc::from_bin(bytes)
    };
    eprintln!(
        "[frontend] mounted disc {} ({} sectors)",
        path.display(),
        disc.sector_count()
    );
    Some(disc)
}

/// Build all panels/overlays for one frame. Called from `gfx::Graphics::render`
/// inside the egui context. `dt` drives Menu animations.
pub fn build_ui(ctx: &egui::Context, state: &mut AppState, vram_tex: egui::TextureId, dt: f32) {
    ui::draw_layout(ctx, state, vram_tex, dt);
}
