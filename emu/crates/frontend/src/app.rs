//! Top-level application state and UI orchestration.
//!
//! Owns the emulator state (currently just a `Cpu` + `Bus` -- VRAM will
//! join once the GPU subsystem lands) and drives the per-frame UI build.

use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use emulator_core::{
    fast_boot_disc_with_hle, warm_bios_for_disc_fast_boot, Bus, Cpu, DISC_FAST_BOOT_WARMUP_STEPS,
};
use psoxide_settings::library::{GameKind, Region};
use psoxide_settings::{ConfigPaths, Library, LibraryEntry, Settings};
use psx_iso::{Disc, Exe, SECTOR_BYTES};
use psx_trace::InstructionRecord;
use psxed_ui::{EditorPlaytestStatus, EditorWorkspace};

use crate::embedded_playtest::EmbeddedPlaytestState;
use crate::ui;
use crate::ui::hud::HudState;
use crate::ui::memory::MemoryView;
use crate::ui::menu::{LibraryItem as MenuLibraryItem, MenuState};

/// Ring-buffer capacity for the execution-history panel. 16 rows is
/// the "what just ran" context window -- enough to spot a tight loop
/// or trace a branch without the history section taking over the
/// registers side panel vertically.
pub const EXEC_HISTORY_CAP: usize = 16;

/// Panels that can be shown/hidden via the Menu. The Menu *is* the
/// library browser (Games / Examples columns), so we don't have
/// a separate "library" panel -- it's integrated into the shell
/// the PSX way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PanelVisibility {
    /// CPU registers + exec history side panel.
    pub registers: bool,
    /// Memory / disassembly dual-pane viewer.
    pub memory: bool,
    /// VRAM thumbnail at the bottom of the screen.
    pub vram: bool,
    /// Floating frame-profiler window.
    pub profiler: bool,
}

/// Hardware-renderer internal scale mode. Both modes use the same
/// renderer; Native forces scale 1, Window chooses a larger scale
/// from the framebuffer panel size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleMode {
    /// Internal scale chosen from the available framebuffer area.
    #[default]
    Window,
    /// Internal scale 1, presented in the same framebuffer area.
    Native,
}

/// Active host workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workspace {
    /// Emulator/debugger workspace.
    Emulator,
    /// Mouse/keyboard editor workspace.
    Editor,
}

impl Workspace {
    /// True when editor panels own the central UI.
    pub const fn is_editor(self) -> bool {
        matches!(self, Self::Editor)
    }
}

/// Work to perform after the shared editor-playtest MIPS build exits.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EditorBuildCompletion {
    /// Load the built EXE into the embedded editor viewport.
    RunEmbedded,
    /// Copy the built EXE into the active project's baked output folder.
    ExportProject { dest_path: PathBuf },
}

/// Top-level app state. Owns the emulator state directly -- no Arc/Mutex,
/// single-threaded, UI reads state in-place per frame.
pub struct AppState {
    /// Active host workspace.
    pub workspace: Workspace,
    /// Embedded editor workspace. Kept alive while hidden so editor
    /// state survives a quick trip back to the Menu/emulator.
    pub editor: EditorWorkspace,
    /// In-process playtest launched from the editor viewport.
    pub embedded_playtest: EmbeddedPlaytestState,
    /// Deferred action attached to the currently running editor build.
    editor_build_completion: Option<EditorBuildCompletion>,
    pub panels: PanelVisibility,
    /// Framebuffer mode -- shared HW renderer at native scale vs
    /// window-fitted high resolution. Toggled via the debug toolbar.
    pub scale_mode: ScaleMode,
    /// Physical pixel size used by the central framebuffer on the
    /// previous UI frame. The renderer uses this as its internal
    /// resolution budget; one-frame latency is fine because it only
    /// changes when resizing/toggling scale mode.
    pub framebuffer_present_size_px: (u32, u32),
    pub cpu: Cpu,
    /// Optional because we let the frontend run without a BIOS for UI
    /// development. If absent, register panels show the reset-state CPU
    /// but no instruction stepping is possible. Unused until the step
    /// button lands alongside the Menu.
    pub bus: Option<Bus>,
    pub menu: MenuState,
    pub hud: HudState,
    /// Rolling frame-time breakdown, visible from the profiler toolbar button.
    pub profiler: ui::profiler::FrameProfiler,
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
    /// mappings, video tweaks). Read at startup, re-saved when Menu
    /// settings actions commit changes. The frontend mutates this
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
    /// Short-lived status line -- shows "Launched Crash Bandicoot",
    /// "Scan complete: 54 games", etc. Displayed beneath the
    /// library panel; cleared after a few frames.
    pub status_message: Option<(String, f32)>,
    /// Host-audio gain controlled from the toolbar. `1.0` is unity.
    pub audio_volume: f32,
    /// Toolbar mute latch. Kept separate from `audio_volume` so
    /// unmuting restores the prior level.
    pub audio_muted: bool,
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
        // tempdir-rooted view so the app still runs -- just without
        // persistence.
        let paths = match override_dir {
            Some(p) => ConfigPaths::rooted(p),
            None => ConfigPaths::platform_default().unwrap_or_else(|e| {
                eprintln!("[frontend] no platform config dir ({e}); persistence disabled");
                ConfigPaths::rooted(std::env::temp_dir().join("PSoXide-ephemeral"))
            }),
        };
        let _ = paths.ensure_dir(paths.root());

        // Legacy file-based workspace: surface once, then ignore.
        // The new model is project = directory under
        // editor/projects/. No automated migration; a stale
        // workspace.ron is just a starter snapshot.
        let legacy_workspace = paths.editor_dir().join("workspace.ron");
        if legacy_workspace.is_file() {
            eprintln!(
                "[frontend] legacy editor/workspace.ron at {} ignored — projects now live under editor/projects/",
                legacy_workspace.display()
            );
        }

        let settings = match Settings::load(&paths.settings_file()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[frontend] settings load: {e}; using defaults");
                Settings::default()
            }
        };

        let preferred_dir = settings
            .editor
            .last_project_dir
            .clone()
            .unwrap_or_else(psxed_project::default_project_dir);
        let editor = EditorWorkspace::open_directory(&preferred_dir)
            .or_else(|first_err| {
                eprintln!(
                    "[frontend] open editor project at {} failed: {first_err}; falling back to default",
                    preferred_dir.display()
                );
                EditorWorkspace::open_directory(psxed_project::default_project_dir())
            })
            .unwrap_or_else(|err| {
                panic!("open default editor project: {err}");
        });
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
            workspace: Workspace::Emulator,
            editor,
            embedded_playtest: EmbeddedPlaytestState::default(),
            editor_build_completion: None,
            panels: PanelVisibility::default(),
            scale_mode: ScaleMode::default(),
            framebuffer_present_size_px: (320, 240),
            cpu,
            bus,
            menu: MenuState::new(),
            hud: HudState::default(),
            profiler: ui::profiler::FrameProfiler::default(),
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
            audio_volume: 1.0,
            audio_muted: false,
        };
        // Startup auto-rescan: always run when a developer-facing build dir
        // exists so stale `library.ron` entries (e.g. cargo
        // `deps/<name>-<hash>.exe` intermediates picked up by an
        // earlier version of the scanner before the deps/ filter
        // landed) get purged. `scan_roots` is mtime-cached for
        // already-seen files, so the cost is bounded by
        // "number of files that changed since last scan" -- cheap
        // on every boot.
        //
        // Scoped to "SDK/project dirs exist" so an end-user install
        // without local builds doesn't pay the cost every startup.
        let sdk_exists = out
            .resolve_sdk_examples_dir()
            .is_some_and(|sdk_dir| sdk_dir.exists());
        let projects_exist = out
            .resolve_editor_projects_dir()
            .is_some_and(|projects_dir| projects_dir.exists());
        if sdk_exists || projects_exist {
            if let Err(e) = out.rescan_library() {
                eprintln!("[frontend] startup auto-rescan skipped: {e}");
            }
        }
        // Seed the Menu's Games + Examples columns from the (now
        // possibly-rescanned) library so the user sees entries
        // immediately instead of a "No games found" placeholder.
        out.refresh_menu_library();
        out.menu
            .sync_fast_boot_label(out.settings.emulator.fast_boot_disc);
        out.menu.sync_editor_label(out.workspace.is_editor());
        out.sync_menu_settings_paths();
        if settings_setup_incomplete(&out.settings) {
            out.select_settings_category();
        }
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
        // Bus state. Silently log on failure -- we'd rather launch
        // the new game than refuse because of a stale save.
        if let Err(e) = self.flush_memcard_port1() {
            eprintln!("[frontend] memcard flush before launch: {e}");
        }
        let bios_path = resolve_bios_path(&self.settings)?;
        let bios =
            std::fs::read(&bios_path).map_err(|e| format!("BIOS {}: {e}", bios_path.display()))?;
        let mut bus = Bus::new(bios).map_err(|e| format!("BIOS rejected: {e}"))?;
        let mut cpu = Cpu::new();
        let mut boot_mode = "EXE";

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
                // `settings.emulator.hle_bios_for_side_load` -- the
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
                boot_mode = maybe_fast_boot_disc(
                    &mut bus,
                    &mut cpu,
                    &disc,
                    entry,
                    self.settings.emulator.fast_boot_disc,
                );
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
                boot_mode = maybe_fast_boot_disc(
                    &mut bus,
                    &mut cpu,
                    &disc,
                    entry,
                    self.settings.emulator.fast_boot_disc,
                );
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

        // Swap everything at once -- no half-loaded state. Start in
        // the running state so the user sees the game boot
        // immediately when they hit Enter in the Menu -- matches a real
        // PS1 where selecting a disc and pressing X fires it right up.
        // The Menu's caller (`apply_menu_action::LaunchGame`) closes
        // the overlay so the game is actually visible.
        self.bus = Some(bus);
        self.cpu = cpu;
        self.running = true;
        self.workspace = Workspace::Emulator;
        self.exec_history.clear();
        self.gpr_snapshot = None;
        self.current_game = Some(entry.clone());
        self.menu.sync_run_label(true);
        self.menu.sync_editor_label(false);
        self.status_message = Some((
            format!("Launched: {} ({boot_mode})", entry.title),
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
    /// Scans roots in one pass:
    ///
    /// 1. `settings.paths.game_library` -- user's retail-disc folder.
    /// 2. `settings.paths.sdk_examples` (or auto-detected
    ///    `build/examples/mipsel-sony-psx/release/` under the repo
    ///    root) -- `.exe` homebrew built by `make examples`.
    /// 3. Auto-detected `editor/projects/` under the repo root --
    ///    project-baked `.exe`s surfaced in the Projects category.
    ///
    /// Either can be missing without erroring. If neither yields
    /// entries, the Menu's columns show the "No … found" placeholder
    /// instead of blowing up.
    ///
    /// Also refreshes the Menu's Games + Examples columns so the
    /// newly-scanned entries appear immediately.
    pub fn rescan_library(&mut self) -> Result<usize, String> {
        let game_library = self.settings.paths.game_library.trim();
        let game_root = if game_library.is_empty() {
            None
        } else {
            Some(PathBuf::from(game_library))
        };
        let sdk_root = self.resolve_sdk_examples_dir();
        let projects_root = self.resolve_editor_projects_dir();

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
        if let Some(p) = projects_root.clone() {
            roots.push(p);
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
    /// resolver can't place the repo root -- in which case scanning
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

    /// Resolve the editor projects root used for launchable project
    /// builds. The editor owns the project folders; the frontend only
    /// scans them for `.exe` outputs so baked builds can be launched
    /// without opening the editor first.
    fn resolve_editor_projects_dir(&self) -> Option<PathBuf> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest.parent()?.parent()?.parent()?;
        Some(repo_root.join("editor/projects"))
    }

    /// Project the current library into the Menu's Games + Examples +
    /// Projects columns. Three passes:
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
    ///    under the CUE's title/ID. For EXE entries: into Examples,
    ///    except internal runtime templates owned by editor Play.
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
        let mut projects: Vec<MenuLibraryItem> = Vec::new();
        let mut cue_already_listed: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let project_root = self.resolve_editor_projects_dir().filter(|p| p.exists());

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
                // CUEs are never shown directly -- their BIN is.
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
                GameKind::Exe if is_internal_example_exe(&e.path) => continue,
                GameKind::Exe => {
                    let item = MenuLibraryItem {
                        id: e.id.clone(),
                        title: e.title.clone(),
                        subtitle: format_subtitle(e),
                    };
                    if project_root
                        .as_ref()
                        .is_some_and(|root| path_is_under(&e.path, root))
                    {
                        projects.push(item);
                    } else {
                        examples.push(item);
                    }
                }
                GameKind::Unknown => {}
            }
        }

        // Pass 3: stable alphabetical order per column.
        games.sort_by_key(|a| a.title.to_lowercase());
        examples.sort_by_key(|a| a.title.to_lowercase());
        projects.sort_by_key(|a| a.title.to_lowercase());
        self.menu.set_library(&games, &examples, &projects);
    }

    /// Persist the current `Settings` to `settings.ron`. Called
    /// when a settings-panel control commits a change.
    pub fn save_settings(&self) -> Result<(), String> {
        self.settings
            .save(&self.paths.settings_file())
            .map_err(|e| format!("save settings.ron: {e}"))
    }

    /// True when no BIOS can be resolved from settings or env.
    pub fn bios_path_missing(&self) -> bool {
        !effective_bios_configured(&self.settings)
    }

    /// True when the user game-library path is blank.
    pub fn games_path_missing(&self) -> bool {
        self.settings.paths.game_library.trim().is_empty()
    }

    /// Warning banner to show at the top of the Menu, if any.
    pub fn menu_setup_warning(&self) -> Option<&'static str> {
        if self.bios_path_missing() {
            Some("please chose a bios path")
        } else if self.games_path_missing() {
            Some("please chose a games path")
        } else {
            None
        }
    }

    /// Move Menu selection to Settings and ensure the overlay is open.
    pub fn select_settings_category(&mut self) {
        self.menu.open = true;
        self.menu.select_category("Settings");
    }

    /// Choose and persist a BIOS image from the Menu Settings column.
    pub fn choose_bios_path(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Choose PlayStation BIOS")
            .add_filter("PlayStation BIOS", &["bin", "rom"]);
        if let Some(dir) = path_parent_or_self(self.settings.paths.bios.trim()) {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.pick_file() else {
            return;
        };
        self.settings.paths.bios = path.to_string_lossy().into_owned();
        match self.save_settings() {
            Ok(()) => {
                self.sync_menu_settings_paths();
                self.status_message_set(format!("BIOS path saved: {}", path_label(&path)));
            }
            Err(e) => {
                eprintln!("[frontend] {e}");
                self.status_message_set(e);
            }
        }
    }

    /// Choose and persist the games folder from the Menu Settings column.
    pub fn choose_games_path(&mut self) {
        let mut dialog = rfd::FileDialog::new().set_title("Choose games folder");
        if let Some(dir) = path_parent_or_self(self.settings.paths.game_library.trim()) {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.pick_folder() else {
            return;
        };
        self.settings.paths.game_library = path.to_string_lossy().into_owned();
        match self.save_settings() {
            Ok(()) => {
                self.sync_menu_settings_paths();
                match self.rescan_library() {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("[frontend] rescan after games path change failed: {e}");
                        self.status_message_set(format!("Games path saved; rescan failed: {e}"));
                    }
                }
            }
            Err(e) => {
                eprintln!("[frontend] {e}");
                self.status_message_set(e);
            }
        }
    }

    /// Refresh the Settings Menu row values from persisted path state.
    pub fn sync_menu_settings_paths(&mut self) {
        self.menu
            .sync_settings_paths(self.bios_path_label(), self.games_path_label());
    }

    fn bios_path_label(&self) -> String {
        let configured = self.settings.paths.bios.trim();
        if !configured.is_empty() {
            return path_label(PathBuf::from(configured));
        }
        if let Some(env) = std::env::var_os("PSOXIDE_BIOS") {
            return format!("env: {}", path_label(PathBuf::from(env)));
        }
        "Missing".into()
    }

    fn games_path_label(&self) -> String {
        let configured = self.settings.paths.game_library.trim();
        if configured.is_empty() {
            "Missing".into()
        } else {
            path_label(PathBuf::from(configured))
        }
    }

    /// Persist the embedded editor project if it has unsaved edits,
    /// and remember which project directory is active so the next
    /// launch reopens it.
    pub fn save_editor_project(&mut self) -> Result<bool, String> {
        let saved = self
            .editor
            .save_if_dirty()
            .map_err(|e| format!("save editor project: {e}"))?;
        let current = Some(self.editor.project_dir().to_path_buf());
        if self.settings.editor.last_project_dir != current {
            self.settings.editor.last_project_dir = current;
            if let Err(e) = self.save_settings() {
                eprintln!("[frontend] {e}");
            }
        }
        Ok(saved)
    }

    /// Flip the disc fast-boot preference, keep the Menu label in
    /// sync, and persist immediately so the next launch uses the
    /// requested path even if the app exits abruptly.
    pub fn toggle_fast_boot_disc(&mut self) {
        let enabled = !self.settings.emulator.fast_boot_disc;
        self.settings.emulator.fast_boot_disc = enabled;
        self.menu.sync_fast_boot_label(enabled);

        let msg = if enabled {
            "Fast boot enabled: PS logo skipped on disc launch"
        } else {
            "Fast boot disabled: BIOS logo shown on disc launch"
        };

        match self.save_settings() {
            Ok(()) => self.status_message_set(msg),
            Err(e) => {
                eprintln!("[frontend] {e}");
                self.status_message_set(format!("{msg} (settings save failed)"));
            }
        }
    }

    /// Enter the embedded editor workspace.
    pub fn open_editor_workspace(&mut self) {
        self.running = false;
        self.workspace = Workspace::Editor;
        self.menu.sync_run_label(false);
        self.menu.sync_editor_label(true);
        self.status_message_set("Editor workspace open");
    }

    /// Return from the editor workspace to the emulator view.
    pub fn close_editor_workspace(&mut self) {
        self.stop_embedded_playtest();
        let save_result = self.save_editor_project();
        self.workspace = Workspace::Emulator;
        self.menu.sync_editor_label(false);
        match save_result {
            Ok(true) => self.status_message_set("Returned to emulator workspace (editor saved)"),
            Ok(false) => self.status_message_set("Returned to emulator workspace"),
            Err(e) => {
                eprintln!("[frontend] {e}");
                self.status_message_set("Returned to emulator workspace (editor save failed)");
            }
        }
    }

    /// Toggle the embedded editor workspace.
    pub fn toggle_editor_workspace(&mut self) {
        if self.workspace.is_editor() {
            self.close_editor_workspace();
        } else {
            self.open_editor_workspace();
        }
    }

    /// Editor-facing status mirror for the embedded play controls.
    pub fn editor_playtest_status(&self) -> EditorPlaytestStatus {
        self.embedded_playtest.editor_status()
    }

    /// True when the editor viewport is currently the live game.
    pub fn embedded_playtest_running(&self) -> bool {
        self.embedded_playtest.is_running()
    }

    /// True when keyboard/gamepad input should be routed to the
    /// embedded game even though the editor workspace is visible.
    pub fn embedded_playtest_input_captured(&self) -> bool {
        self.embedded_playtest.input_captured()
    }

    /// Build and run the active editor project: cook assets, spawn
    /// the existing MIPS build target, then side-load the EXE. The
    /// build is asynchronous; call
    /// [`Self::poll_embedded_playtest_build`] once per frame to load
    /// the resulting EXE when it exits successfully.
    pub fn start_embedded_playtest(&mut self) {
        self.stop_embedded_playtest();
        self.editor.set_status("Play: cooking assets...");
        if let Err(error) = self.save_editor_project() {
            let message = format!("Embedded Play failed: {error}");
            self.editor.set_status(message.clone());
            self.embedded_playtest.fail();
            return;
        }
        let cook_status = match self.editor.cook_playtest_to_disk() {
            Ok(status) => status,
            Err(error) => {
                let message = format!("Embedded Play failed while cooking assets: {error}");
                self.editor.set_status(message.clone());
                self.embedded_playtest.fail();
                return;
            }
        };
        self.editor
            .set_status(format!("{cook_status}; compiling runtime..."));

        if let Err(error) = self.spawn_editor_playtest_build(
            EditorBuildCompletion::RunEmbedded,
            "Building embedded playtest",
        ) {
            let message = format!("Embedded Play build failed: {error}");
            self.editor.set_status(message.clone());
            self.embedded_playtest.fail();
        }
    }

    /// Build the active project by cooking assets, compiling the runtime,
    /// and copying the resulting PSX EXE into the project folder so the
    /// launcher Projects category can run it without opening the editor.
    pub fn build_current_project_for_launcher(&mut self) {
        self.stop_embedded_playtest();
        self.editor
            .set_status("Building project: cooking assets...");
        if let Err(error) = self.save_editor_project() {
            let message = format!("Project build failed: {error}");
            self.editor.set_status(message.clone());
            self.embedded_playtest.fail();
            return;
        }
        let dest_path =
            project_baked_exe_path(self.editor.project_dir(), &self.editor.project().name);
        let cook_status = match self.editor.cook_playtest_to_disk() {
            Ok(status) => status,
            Err(error) => {
                let message = format!("Project build failed while cooking assets: {error}");
                self.editor.set_status(message.clone());
                self.embedded_playtest.fail();
                return;
            }
        };
        self.editor.set_status(format!(
            "{cook_status}; compiling project EXE for {}...",
            dest_path.display()
        ));

        if let Err(error) = self.spawn_editor_playtest_build(
            EditorBuildCompletion::ExportProject { dest_path },
            "Building project EXE",
        ) {
            let message = format!("Project build failed: {error}");
            self.editor.set_status(message.clone());
            self.embedded_playtest.fail();
        }
    }

    fn spawn_editor_playtest_build(
        &mut self,
        completion: EditorBuildCompletion,
        toast: &'static str,
    ) -> Result<(), String> {
        let workspace_root = repo_root_dir();
        let mut command = Command::new("make");
        command
            .arg("build-editor-playtest")
            .current_dir(&workspace_root)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let child = command
            .spawn()
            .map_err(|error| format!("spawn make: {error}"))?;
        self.editor_build_completion = Some(completion);
        self.embedded_playtest.start_building(child);
        self.status_message_set(toast);
        Ok(())
    }

    /// Poll the background build child, then either side-load the
    /// resulting editor-playtest EXE or export it as a project build.
    pub fn poll_embedded_playtest_build(&mut self) {
        let Some(child) = self.embedded_playtest.building_child_mut() else {
            return;
        };
        let status = match child.try_wait() {
            Ok(Some(status)) => status,
            Ok(None) => return,
            Err(error) => {
                let message = format!("{} poll failed: {error}", self.editor_build_label());
                self.editor.set_status(message.clone());
                self.editor_build_completion = None;
                self.embedded_playtest.fail();
                return;
            }
        };

        if !status.success() {
            let message = format!("{} failed: {status}", self.editor_build_label());
            self.editor.set_status(message.clone());
            self.editor_build_completion = None;
            self.embedded_playtest.fail();
            return;
        }

        let completion = self
            .editor_build_completion
            .take()
            .unwrap_or(EditorBuildCompletion::RunEmbedded);
        match completion {
            EditorBuildCompletion::RunEmbedded => match self.load_embedded_playtest_exe() {
                Ok(()) => {
                    self.embedded_playtest.start_running(true);
                    self.running = true;
                    self.menu.open = false;
                    self.menu.sync_run_label(true);
                    self.editor
                        .set_status("Embedded Play running in the 3D viewport");
                    self.status_message_set("Embedded Play running");
                }
                Err(error) => {
                    let message = format!("Embedded Play load failed: {error}");
                    self.editor.set_status(message.clone());
                    self.embedded_playtest.fail();
                }
            },
            EditorBuildCompletion::ExportProject { dest_path } => {
                match self.export_project_build(dest_path) {
                    Ok(message) => {
                        self.embedded_playtest.stop();
                        self.editor.set_status(message.clone());
                        self.status_message_set(message);
                    }
                    Err(error) => {
                        let message = format!("Project build export failed: {error}");
                        self.editor.set_status(message.clone());
                        self.embedded_playtest.fail();
                    }
                }
            }
        }
    }

    fn editor_build_label(&self) -> &'static str {
        match self.editor_build_completion.as_ref() {
            Some(EditorBuildCompletion::ExportProject { .. }) => "Project build",
            _ => "Embedded Play build",
        }
    }

    /// Stop embedded play mode and return the editor viewport to the
    /// authored 3D preview.
    pub fn stop_embedded_playtest(&mut self) {
        if let Some(mut child) = self.embedded_playtest.take_build_child() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.editor_build_completion = None;
        self.embedded_playtest.stop();
        self.running = false;
        self.menu.sync_run_label(false);
    }

    /// Capture input for the embedded game and resume emulation.
    pub fn capture_embedded_playtest_input(&mut self) {
        if self.embedded_playtest.capture_input() {
            self.running = true;
            self.menu.open = false;
            self.menu.sync_run_label(true);
            self.editor.set_status("Embedded Play input captured");
        }
    }

    /// Release input capture from the embedded game and pause it.
    pub fn release_embedded_playtest_input(&mut self) {
        if self.embedded_playtest.release_input() {
            self.running = false;
            self.menu.open = true;
            self.menu.sync_run_label(false);
            self.editor
                .set_status("Embedded Play paused; click viewport to resume");
        }
    }

    /// Handle one request emitted by the editor UI.
    pub fn handle_editor_playtest_request(&mut self, request: psxed_ui::EditorPlaytestRequest) {
        match request {
            psxed_ui::EditorPlaytestRequest::Play | psxed_ui::EditorPlaytestRequest::Rebuild => {
                self.start_embedded_playtest();
            }
            psxed_ui::EditorPlaytestRequest::BuildProject => {
                self.build_current_project_for_launcher();
            }
            psxed_ui::EditorPlaytestRequest::Stop => {
                self.stop_embedded_playtest();
                self.editor
                    .set_status("Embedded Play stopped; returned to edit preview");
            }
            psxed_ui::EditorPlaytestRequest::CaptureInput => {
                self.capture_embedded_playtest_input();
            }
        }
    }

    fn load_embedded_playtest_exe(&mut self) -> Result<(), String> {
        let bios_path = resolve_bios_path(&self.settings)?;
        let bios =
            std::fs::read(&bios_path).map_err(|e| format!("BIOS {}: {e}", bios_path.display()))?;
        let mut bus = Bus::new(bios).map_err(|e| format!("BIOS rejected: {e}"))?;
        let exe_path = editor_playtest_exe_path();
        let bytes = std::fs::read(&exe_path).map_err(|e| format!("{}: {e}", exe_path.display()))?;
        let exe = Exe::parse(&bytes).map_err(|e| format!("parse EXE: {e:?}"))?;
        let mut cpu = Cpu::new();
        bus.load_exe_payload(exe.load_addr, &exe.payload);
        bus.clear_exe_bss(exe.bss_addr, exe.bss_size);
        cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());
        bus.enable_hle_bios();
        bus.attach_digital_pad_port1();
        let _ = bus.force_port1_analog_mode();

        self.bus = Some(bus);
        self.cpu = cpu;
        self.exec_history.clear();
        self.gpr_snapshot = None;
        self.current_game = None;
        Ok(())
    }

    fn export_project_build(&mut self, dest_path: PathBuf) -> Result<String, String> {
        let source_path = editor_playtest_exe_path();
        let dest_dir = dest_path
            .parent()
            .ok_or_else(|| format!("invalid build output path: {}", dest_path.display()))?;
        std::fs::create_dir_all(dest_dir)
            .map_err(|error| format!("mkdir {}: {error}", dest_dir.display()))?;
        let bytes = std::fs::copy(&source_path, &dest_path).map_err(|error| {
            format!(
                "copy {} to {}: {error}",
                source_path.display(),
                dest_path.display()
            )
        })?;

        let rescan_error = self.rescan_library().err();
        let mut message = format!(
            "Project build exported -> {} ({} KiB)",
            dest_path.display(),
            bytes / 1024
        );
        if let Some(error) = rescan_error {
            message.push_str(&format!("; launcher rescan failed: {error}"));
        }
        Ok(message)
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

    /// Current output gain after the mute latch is applied.
    pub fn effective_audio_volume(&self) -> f32 {
        if self.audio_muted {
            0.0
        } else {
            self.audio_volume.clamp(0.0, 1.5)
        }
    }
}

/// Seconds a status toast stays visible.
const STATUS_MESSAGE_TTL_SECS: f32 = 3.5;

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

fn is_internal_example_exe(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n == "editor-playtest.exe")
}

fn path_is_under(path: &Path, root: &Path) -> bool {
    match (path.canonicalize(), root.canonicalize()) {
        (Ok(path), Ok(root)) => path.starts_with(root),
        _ => path.starts_with(root),
    }
}

/// Pick the BIOS path the launcher should read, honouring
/// precedence: explicit settings field > env var. Centralised so
/// every normal frontend caller agrees and no local path leaks into
/// app defaults.
pub(crate) fn resolve_bios_path(settings: &Settings) -> Result<PathBuf, String> {
    let configured = settings.paths.bios.trim();
    if !configured.is_empty() {
        Ok(PathBuf::from(configured))
    } else if let Ok(p) = std::env::var("PSOXIDE_BIOS") {
        Ok(PathBuf::from(p))
    } else {
        Err("BIOS path is not configured. Open Settings and choose a BIOS image, or export PSOXIDE_BIOS.".to_string())
    }
}

fn effective_bios_configured(settings: &Settings) -> bool {
    !settings.paths.bios.trim().is_empty() || std::env::var_os("PSOXIDE_BIOS").is_some()
}

fn settings_setup_incomplete(settings: &Settings) -> bool {
    !effective_bios_configured(settings) || settings.paths.game_library.trim().is_empty()
}

fn path_parent_or_self(value: &str) -> Option<PathBuf> {
    if value.is_empty() {
        return None;
    }
    let path = PathBuf::from(value);
    if path.is_dir() {
        Some(path)
    } else {
        path.parent().map(Path::to_path_buf)
    }
}

fn path_label(path: impl AsRef<Path>) -> String {
    let path = path.as_ref();
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| path.display().to_string())
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

/// Guest-side work performed while advancing one video frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepFrameReport {
    /// Cycle budget for the target video frame.
    pub target_cycles: u64,
    /// Bus cycles actually advanced.
    pub cycles: u64,
    /// CPU instructions retired.
    pub instructions: u64,
    /// VBlank IRQ raises observed while stepping.
    pub vblanks: u64,
    /// True when the safety instruction cap stopped the frame early.
    pub hit_step_cap: bool,
}

/// Retire enough instructions to cover one PSX video frame's worth of
/// master-clock cycles. Any execution error auto-pauses, reopens the
/// Menu, and surfaces the stopped state via the register panel. Hitting
/// a breakpoint does the same. Split out here (rather than living in
/// the shell loop) so both the shell's per-frame run path and the
/// toolbar's "advance one frame" button can invoke the same logic.
pub fn step_one_frame(state: &mut AppState) -> StepFrameReport {
    let max_steps = state.run_steps_per_frame.max(1);
    let Some(bus) = state.bus.as_mut() else {
        state.running = false;
        state.menu.sync_run_label(false);
        return StepFrameReport::default();
    };

    // Only fill `exec_history` when the registers panel is open --
    // otherwise the 404-byte `InstructionRecord` per step is pure
    // overhead. The panel is closed by default on first run, so
    // most users get the fast `step()` path automatically.
    let trace = state.panels.registers;
    let cycles_before = bus.cycles();
    let tick_before = state.cpu.tick();
    let vblank_before = bus.irq().raise_counts()[0];
    let frame_budget = bus.vblank_period().max(1);
    let target_cycles = cycles_before.saturating_add(frame_budget);
    let mut steps_run = 0;
    for _ in 0..max_steps {
        if bus.cycles() >= target_cycles {
            break;
        }
        // Breakpoint check happens BEFORE stepping so the paused PC
        // is the BP address itself -- the instruction at that PC has
        // not yet executed.
        if state.breakpoints.contains(&state.cpu.pc()) {
            state.running = false;
            state.menu.sync_run_label(false);
            state.menu.open = true;
            break;
        }
        steps_run += 1;

        let result = if trace {
            match state.cpu.step_traced(bus) {
                Ok(record) => {
                    push_history(&mut state.exec_history, record);
                    Ok(())
                }
                Err(e) => Err(e),
            }
        } else {
            state.cpu.step(bus)
        };
        if result.is_err() {
            state.running = false;
            state.menu.sync_run_label(false);
            state.menu.open = true;
            break;
        }
    }

    let cycles_after = bus.cycles();
    let vblank_after = bus.irq().raise_counts()[0];
    StepFrameReport {
        target_cycles: frame_budget,
        cycles: cycles_after.saturating_sub(cycles_before),
        instructions: state.cpu.tick().saturating_sub(tick_before),
        vblanks: vblank_after.saturating_sub(vblank_before),
        hit_step_cap: steps_run >= max_steps && cycles_after < target_cycles && state.running,
    }
}

fn maybe_fast_boot_disc(
    bus: &mut Bus,
    cpu: &mut Cpu,
    disc: &Disc,
    entry: &LibraryEntry,
    enabled: bool,
) -> &'static str {
    if !enabled {
        return "BIOS boot";
    }
    if let Err(e) = warm_bios_for_disc_fast_boot(bus, cpu, DISC_FAST_BOOT_WARMUP_STEPS) {
        eprintln!(
            "[frontend] BIOS warmup failed for {} ({e:?}); falling back to BIOS boot",
            entry.path.display()
        );
        return "BIOS boot";
    }
    match fast_boot_disc_with_hle(bus, cpu, disc, false) {
        Ok(info) => {
            eprintln!(
                "[frontend] warm-fast-booted {} via {} entry=0x{:08x} load=0x{:08x} payload={}B",
                entry.path.display(),
                info.boot_path,
                info.initial_pc,
                info.load_addr,
                info.payload_len
            );
            "fast boot"
        }
        Err(e) => {
            eprintln!(
                "[frontend] fast boot unavailable for {} ({e:?}); falling back to BIOS boot",
                entry.path.display()
            );
            "BIOS boot"
        }
    }
}

fn load_bus(settings: &Settings) -> Option<Bus> {
    let path = match resolve_bios_path(settings) {
        Ok(path) => path,
        Err(e) => {
            eprintln!("[frontend] {e}");
            return None;
        }
    };
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

    // Optional disc. Absence is not an error -- BIOS boots fine without
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

fn repo_root_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn editor_playtest_exe_path() -> PathBuf {
    repo_root_dir()
        .join("build")
        .join("examples")
        .join("mipsel-sony-psx")
        .join("release")
        .join("editor-playtest.exe")
}

fn project_baked_exe_path(project_dir: &Path, project_name: &str) -> PathBuf {
    project_dir
        .join("baked")
        .join(format!("{}.exe", safe_project_exe_stem(project_name)))
}

fn safe_project_exe_stem(name: &str) -> String {
    psxed_project::project_file_stem(name)
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
pub fn build_ui(
    ctx: &egui::Context,
    state: &mut AppState,
    vram_tex: egui::TextureId,
    display_tex: egui::TextureId,
    editor_viewport: psxed_ui::EditorViewport3dPresentation,
    framebuffer_source: ui::framebuffer::FramebufferSource,
    dt: f32,
) {
    ui::draw_layout(
        ctx,
        state,
        vram_tex,
        display_tex,
        editor_viewport,
        framebuffer_source,
        dt,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_editor_playtest_exe_is_hidden_from_menu_examples() {
        assert!(is_internal_example_exe(Path::new(
            "build/examples/mipsel-sony-psx/release/editor-playtest.exe"
        )));
        assert!(!is_internal_example_exe(Path::new(
            "build/examples/mipsel-sony-psx/release/showcase-room.exe"
        )));
    }

    #[test]
    fn project_build_exe_name_is_filesystem_safe() {
        assert_eq!(
            safe_project_exe_stem("Stone Room: Vertical Slice!"),
            "stone_room_vertical_slice"
        );
        assert_eq!(safe_project_exe_stem("..."), "project");
        assert_eq!(
            project_baked_exe_path(Path::new("editor/projects/default"), "Demo Project"),
            Path::new("editor/projects/default")
                .join("baked")
                .join("demo_project.exe")
        );
    }
}
