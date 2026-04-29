//! User-editable preferences, persisted as `settings.ron`.
//!
//! The on-disk file is meant to be read — and sometimes edited — by
//! humans. That shapes two decisions in this module:
//!
//! 1. **RON instead of TOML/JSON.** Our settings are enum-heavy
//!    (key bindings, peripheral choices); RON round-trips them
//!    losslessly. TOML forces stringly-typed encodings, JSON lacks
//!    comments and is noisy.
//! 2. **Pretty-printed output, leading comment banner.** We write
//!    a welcoming comment at the top of the file so someone who
//!    opens it knows what it is, where to put new values, and
//!    that missing fields fall back to defaults.
//!
//! The loader is deliberately permissive:
//!
//! - Missing file → default settings (first run).
//! - Parse error → logged via [`SettingsError::Parse`] and the
//!   caller can choose to keep defaults or surface the error.
//! - Unknown fields → ignored (RON's serde respects
//!   `#[serde(default)]`).
//! - Version bumps → migration hook [`Settings::migrate`] runs
//!   on load; callers re-save to pin the upgrade.
//!
//! Writes are atomic: we serialise to a temp file next to the
//! target and rename on success. A crash mid-write leaves the old
//! settings intact.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::SETTINGS_VERSION;

/// Errors returned by [`Settings::load`] / [`Settings::save`].
#[derive(Debug, Error)]
pub enum SettingsError {
    /// The settings file exists but couldn't be opened / read.
    #[error("settings I/O error at {path}: {source}")]
    Io {
        /// The file we were reading / writing.
        path: std::path::PathBuf,
        /// The underlying `io::Error`.
        #[source]
        source: io::Error,
    },
    /// The settings file is present but doesn't parse as RON.
    /// Default policy: log and fall back to defaults; the caller
    /// can choose to surface this if the user ran with a strict
    /// flag.
    #[error("settings parse error at {path}: {source}")]
    Parse {
        /// The file we were parsing.
        path: std::path::PathBuf,
        /// The RON parser's error.
        #[source]
        source: ron::error::SpannedError,
    },
    /// RON serialisation failed — unlikely at this scale, but
    /// propagated for completeness.
    #[error("settings serialization error: {0}")]
    Serialize(#[from] ron::Error),
}

/// Video output preferences. Kept small — we'll grow it as the
/// renderer picks up more knobs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoSettings {
    /// Snap the output to integer scales only. Keeps the image
    /// pixel-sharp; defaults to on because that's the PSX
    /// aesthetic most users want.
    pub integer_scale: bool,
    /// Optional CRT-scanline shader overlay. Off by default.
    pub scanline_filter: bool,
}

impl Default for VideoSettings {
    fn default() -> Self {
        Self {
            integer_scale: true,
            scanline_filter: false,
        }
    }
}

/// Where on the filesystem the app pulls assets from at startup.
/// Empty strings mean "no preference — use built-in defaults or
/// explicit env-var fallbacks." All paths are stored as strings so the
/// RON file stays platform-portable (Path on disk uses forward
/// slashes even on Windows — we join via `PathBuf` at read time).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paths {
    /// Preferred BIOS image (`SCPH1001.BIN` & friends). Empty =
    /// use the `PSOXIDE_BIOS` env var. Normal frontend paths do
    /// not use a hardcoded BIOS fallback.
    pub bios: String,
    /// Root directory for the game library scanner. Empty =
    /// library feature inactive until the user sets one.
    pub game_library: String,
    /// Optional second root for the library scanner that contains
    /// SDK-built homebrew example `.exe` files. Empty = frontend
    /// auto-detects the standard `build/examples/mipsel-sony-psx/release/`
    /// directory relative to the repo root (compile-time path from
    /// `CARGO_MANIFEST_DIR`). Set explicitly to override — e.g.
    /// pointing at a co-developer's alternate SDK build tree.
    ///
    /// Kept separate from `game_library` so users can ship the
    /// emulator without disturbing their retail-game directory and
    /// still get the SDK demos populating the Examples column.
    #[serde(default)]
    pub sdk_examples: String,
    /// Optional directory for the parity-trace cache. Empty =
    /// use the default under `target/parity-cache`.
    pub parity_cache_dir: String,
}

/// A single key → button mapping. `InputBinding` is a tagged enum
/// (rather than a stringly-typed name) so RON preserves exact
/// variant information and we don't pay a string-match cost every
/// keypress.
///
/// Named variants carry an owned `String` (not a `&'static str`)
/// because the loader materialises bindings from the RON file at
/// runtime — a borrow couldn't satisfy serde's lifetime. In-flight
/// cost at keypress is one `String::eq` per binding; negligible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InputBinding {
    /// A named key (ArrowUp, Enter, Escape, …). The frontend
    /// translates this into its key enum at input-event time.
    Named(String),
    /// A character key, lowercased (`'x'`, `'a'`, `'1'`, …).
    Character(char),
    /// Explicitly unbound — renders as blank in the UI. Useful for
    /// defaulting some pad buttons to "not yet assigned" without
    /// forcing a real key.
    #[default]
    Unbound,
}

impl InputBinding {
    /// Convenience constructor for a named-key binding — keeps
    /// call-site noise down (`named("Enter")` vs
    /// `Named("Enter".into())`).
    pub fn named(name: impl Into<String>) -> Self {
        InputBinding::Named(name.into())
    }

    /// A short human-readable label for this binding — used by the
    /// settings UI and the HUD.
    pub fn label(&self) -> String {
        match self {
            InputBinding::Named(name) => name.clone(),
            InputBinding::Character(c) => c.to_uppercase().next().unwrap_or(*c).to_string(),
            InputBinding::Unbound => "—".to_string(),
        }
    }
}

/// Controller key bindings for a single port.
///
/// Button naming follows the PSX convention: `cross` (X),
/// `circle` (O), `square`, `triangle`. Those are the real button
/// labels printed on every DualShock; renaming them to
/// Nintendo-style "A/B/X/Y" causes muscle-memory confusion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortBindings {
    /// Up on the d-pad.
    pub up: InputBinding,
    /// Down on the d-pad.
    pub down: InputBinding,
    /// Left on the d-pad.
    pub left: InputBinding,
    /// Right on the d-pad.
    pub right: InputBinding,
    /// Cross (X) button.
    pub cross: InputBinding,
    /// Circle (O) button.
    pub circle: InputBinding,
    /// Square button.
    pub square: InputBinding,
    /// Triangle button.
    pub triangle: InputBinding,
    /// L1 shoulder.
    pub l1: InputBinding,
    /// R1 shoulder.
    pub r1: InputBinding,
    /// L2 shoulder.
    pub l2: InputBinding,
    /// R2 shoulder.
    pub r2: InputBinding,
    /// Start.
    pub start: InputBinding,
    /// Select.
    pub select: InputBinding,
    /// DualShock Analog button. This is not part of the normal
    /// button bitmask; it toggles whether the controller reports
    /// Digital (`0x41`) or Analog (`0x73`) poll IDs.
    #[serde(default)]
    pub analog: InputBinding,
}

impl Default for PortBindings {
    /// Default keyboard mapping. Chosen so a right-hander can hold
    /// the d-pad on the left hand (arrow keys) and the action
    /// cluster on the right (X / C / V / D forms a rough diamond).
    ///
    /// Not claiming this is optimal — users will want to rebind.
    /// It's just "works immediately on a fresh install."
    fn default() -> Self {
        Self {
            up: InputBinding::named("ArrowUp"),
            down: InputBinding::named("ArrowDown"),
            left: InputBinding::named("ArrowLeft"),
            right: InputBinding::named("ArrowRight"),
            cross: InputBinding::Character('x'),
            circle: InputBinding::Character('c'),
            square: InputBinding::Character('z'),
            triangle: InputBinding::Character('s'),
            l1: InputBinding::Character('q'),
            r1: InputBinding::Character('e'),
            l2: InputBinding::Character('1'),
            r2: InputBinding::Character('3'),
            start: InputBinding::named("Enter"),
            select: InputBinding::named("Backspace"),
            analog: default_port1_analog_binding(),
        }
    }
}

fn default_port1_analog_binding() -> InputBinding {
    InputBinding::named("F9")
}

/// Input-related settings across all ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSettings {
    /// Port 1 keyboard bindings.
    pub port1: PortBindings,
    /// Port 2 bindings. Defaults to all-unbound so a fresh install
    /// doesn't accidentally capture random keys for an unattached
    /// controller.
    pub port2: PortBindings,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            port1: PortBindings::default(),
            port2: PortBindings {
                up: InputBinding::Unbound,
                down: InputBinding::Unbound,
                left: InputBinding::Unbound,
                right: InputBinding::Unbound,
                cross: InputBinding::Unbound,
                circle: InputBinding::Unbound,
                square: InputBinding::Unbound,
                triangle: InputBinding::Unbound,
                l1: InputBinding::Unbound,
                r1: InputBinding::Unbound,
                l2: InputBinding::Unbound,
                r2: InputBinding::Unbound,
                start: InputBinding::Unbound,
                select: InputBinding::Unbound,
                analog: InputBinding::Unbound,
            },
        }
    }
}

/// Top-level grouping of all input-related settings. Factored so
/// the settings panel can focus its "Input" tab on this subtree
/// without touching anything else.
impl InputSettings {
    /// Return port N's bindings (1 or 2). Out-of-range returns
    /// port1 as a safe default.
    pub fn port(&self, port: u8) -> &PortBindings {
        if port == 2 {
            &self.port2
        } else {
            &self.port1
        }
    }
}

/// Emulator-level toggles. `hle_bios` is the big one — everything
/// else can grow as we add features.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmulatorSettings {
    /// Side-loaded EXEs rely on HLE BIOS by default; fully-booted
    /// commercial games don't. This toggle governs the *default*
    /// that gets applied when launching a game; per-game overrides
    /// in the library cache can flip it.
    ///
    /// Defaults to **true** — matches what `PSOXIDE_EXE=…` side-load
    /// does unconditionally (see `app::load_exe`). Previously this
    /// derived `Default`, giving `false`, which made the library-
    /// launch path ship a half-initialised BIOS kernel state to the
    /// EXE. Every SDK example's SYSCALL to `InstallISR` / `FlushCache`
    /// etc. landed in a BIOS kernel that hadn't been cold-booted,
    /// silently fell off a wild PC, and produced a blank screen
    /// while `make run-tri` (env-var path, HLE unconditional) worked
    /// fine. Flipping the default brings the two paths into agreement.
    #[serde(default = "default_hle_for_side_load")]
    pub hle_bios_for_side_load: bool,
    /// Boot discs by first letting the real BIOS initialize its RAM
    /// kernel state, then loading the `SYSTEM.CNF` PSX-EXE directly
    /// and leaving the disc mounted for normal CD-ROM commands. This
    /// is the default while BIOS-disc handoff parity is still under
    /// investigation; set to `false` to force the real BIOS logo path.
    #[serde(default = "default_fast_boot_disc")]
    pub fast_boot_disc: bool,
    /// If set, the run loop paces itself to real-time instead of
    /// running flat-out. Defaults to false — we want flat-out
    /// speed for parity / debugging. A future audio feature will
    /// flip this on so the SPU stays in sync.
    #[serde(default)]
    pub real_time_pacing: bool,
}

fn default_hle_for_side_load() -> bool {
    true
}

fn default_fast_boot_disc() -> bool {
    true
}

impl Default for EmulatorSettings {
    fn default() -> Self {
        Self {
            hle_bios_for_side_load: default_hle_for_side_load(),
            fast_boot_disc: default_fast_boot_disc(),
            real_time_pacing: false,
        }
    }
}

/// Editor-side preferences that persist across sessions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditorSettings {
    /// Last-opened project directory. `None` on first launch — the
    /// frontend then opens `psxed_project::default_project_dir()`.
    /// Updated by the frontend after every successful
    /// `open_directory` / `create_and_open_project` so re-launches
    /// resume on the project the user was last editing.
    #[serde(default)]
    pub last_project_dir: Option<PathBuf>,
}

/// The full settings root. Shape is intentionally flat — no deep
/// nesting — so that someone skimming `settings.ron` can see every
/// section at once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// Schema version of this file on disk. Bumped when the struct
    /// changes in a breaking way; `Settings::migrate` handles the
    /// upgrade at load time.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Filesystem locations the app reads from.
    #[serde(default)]
    pub paths: Paths,
    /// Video-output preferences.
    #[serde(default)]
    pub video: VideoSettings,
    /// Keyboard → pad mappings.
    #[serde(default)]
    pub input: InputSettings,
    /// Emulator-level toggles.
    #[serde(default)]
    pub emulator: EmulatorSettings,
    /// Editor preferences.
    #[serde(default)]
    pub editor: EditorSettings,
}

fn default_version() -> u32 {
    SETTINGS_VERSION
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: SETTINGS_VERSION,
            paths: Paths::default(),
            video: VideoSettings::default(),
            input: InputSettings::default(),
            emulator: EmulatorSettings::default(),
            editor: EditorSettings::default(),
        }
    }
}

impl Settings {
    /// Read `path` and parse as RON. If the file doesn't exist, the
    /// returned settings are the defaults — *not* an error, because
    /// first-run shouldn't fail. If the file exists but is corrupt,
    /// returns [`SettingsError::Parse`] so the caller can decide
    /// whether to log + default or bail.
    pub fn load(path: &Path) -> Result<Self, SettingsError> {
        match fs::read_to_string(path) {
            Ok(contents) => {
                let mut settings: Settings =
                    ron::from_str(&contents).map_err(|source| SettingsError::Parse {
                        path: path.to_path_buf(),
                        source,
                    })?;
                settings.migrate();
                Ok(settings)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(SettingsError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    /// Serialise to RON and write atomically. The content is written
    /// to `<path>.tmp` first, then renamed to `path` — so a crash
    /// mid-write can never leave a half-written settings file
    /// behind.
    ///
    /// The output starts with a short comment banner so the first
    /// thing a user sees when they open the file is context, not
    /// raw RON.
    pub fn save(&self, path: &Path) -> Result<(), SettingsError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|source| SettingsError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }
        let ron_body = ron::ser::to_string_pretty(
            self,
            ron::ser::PrettyConfig::new()
                .depth_limit(6)
                .indentor("    ".to_string()),
        )?;
        let banner = concat!(
            "// PSoXide user settings.\n",
            "//\n",
            "// RON (Rusty Object Notation). Comments and whitespace are\n",
            "// preserved on load but *not* on save — if you edit this file\n",
            "// and the app re-saves it, your comments will go away. Best\n",
            "// practice: edit, don't re-save (or treat this as read-only).\n",
            "//\n",
            "// Missing fields fall back to their compiled-in defaults, so\n",
            "// you can delete anything you don't care about overriding.\n",
            "\n"
        );
        let content = format!("{banner}{ron_body}\n");
        let tmp = path.with_extension("ron.tmp");
        fs::write(&tmp, content).map_err(|source| SettingsError::Io {
            path: tmp.clone(),
            source,
        })?;
        fs::rename(&tmp, path).map_err(|source| SettingsError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }

    /// Hook for cross-version migrations. Today it only pins the
    /// version field to the current value — as the schema evolves,
    /// this branches on `self.version` to upgrade.
    pub fn migrate(&mut self) {
        if self.version < 2 {
            self.input.port1.analog = default_port1_analog_binding();
            self.input.port2.analog = InputBinding::Unbound;
        }
        self.version = SETTINGS_VERSION;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn default_round_trips() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.ron");
        let original = Settings::default();
        original.save(&path).unwrap();
        let loaded = Settings::load(&path).unwrap();
        assert_eq!(original, loaded);
    }

    #[test]
    fn missing_file_returns_defaults() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("doesnotexist.ron");
        let loaded = Settings::load(&path).unwrap();
        assert_eq!(loaded, Settings::default());
    }

    #[test]
    fn corrupt_file_returns_parse_error() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bad.ron");
        std::fs::write(&path, "{{{ this is not ron").unwrap();
        let err = Settings::load(&path).unwrap_err();
        assert!(matches!(err, SettingsError::Parse { .. }));
    }

    #[test]
    fn save_is_atomic_on_existing_file() {
        // The tmp->rename dance must not corrupt an existing file.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.ron");
        let mut s = Settings::default();
        s.paths.bios = "/first/path".to_string();
        s.save(&path).unwrap();
        s.paths.bios = "/second/path".to_string();
        s.save(&path).unwrap();
        let loaded = Settings::load(&path).unwrap();
        assert_eq!(loaded.paths.bios, "/second/path");
    }

    #[test]
    fn save_includes_banner_comment() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.ron");
        Settings::default().save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("// PSoXide user settings."));
    }

    #[test]
    fn missing_fields_use_defaults() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("partial.ron");
        // Only set `video.integer_scale = false`, rely on serde
        // defaults for everything else.
        std::fs::write(
            &path,
            "(\n    version: 1,\n    video: (integer_scale: false, scanline_filter: false),\n)",
        )
        .unwrap();
        let loaded = Settings::load(&path).unwrap();
        assert!(!loaded.video.integer_scale);
        assert_eq!(loaded.input, InputSettings::default());
    }

    #[test]
    fn enum_bindings_round_trip_losslessly() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.ron");
        let mut s = Settings::default();
        s.input.port1.cross = InputBinding::Character('j');
        s.input.port1.circle = InputBinding::named("Space");
        s.input.port2.start = InputBinding::Unbound;
        s.save(&path).unwrap();
        let loaded = Settings::load(&path).unwrap();
        assert_eq!(loaded.input.port1.cross, InputBinding::Character('j'));
        assert_eq!(loaded.input.port1.circle, InputBinding::named("Space"));
        assert_eq!(loaded.input.port2.start, InputBinding::Unbound);
        assert_eq!(loaded.input.port1.analog, default_port1_analog_binding());
    }

    #[test]
    fn older_settings_gain_analog_button_binding() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.ron");
        std::fs::write(
            &path,
            "(
                version: 1,
                input: (
                    port1: (
                        up: Named(\"ArrowUp\"),
                        down: Named(\"ArrowDown\"),
                        left: Named(\"ArrowLeft\"),
                        right: Named(\"ArrowRight\"),
                        cross: Character('x'),
                        circle: Character('c'),
                        square: Character('z'),
                        triangle: Character('s'),
                        l1: Character('q'),
                        r1: Character('e'),
                        l2: Character('1'),
                        r2: Character('3'),
                        start: Named(\"Enter\"),
                        select: Named(\"Backspace\"),
                    ),
                    port2: (
                        up: Unbound,
                        down: Unbound,
                        left: Unbound,
                        right: Unbound,
                        cross: Unbound,
                        circle: Unbound,
                        square: Unbound,
                        triangle: Unbound,
                        l1: Unbound,
                        r1: Unbound,
                        l2: Unbound,
                        r2: Unbound,
                        start: Unbound,
                        select: Unbound,
                    ),
                ),
            )",
        )
        .unwrap();

        let loaded = Settings::load(&path).unwrap();
        assert_eq!(loaded.version, crate::SETTINGS_VERSION);
        assert_eq!(loaded.input.port1.analog, default_port1_analog_binding());
        assert_eq!(loaded.input.port2.analog, InputBinding::Unbound);
    }

    #[test]
    fn binding_labels_are_sensible() {
        assert_eq!(InputBinding::named("Enter").label(), "Enter");
        assert_eq!(InputBinding::Character('j').label(), "J");
        assert_eq!(InputBinding::Unbound.label(), "—");
    }
}
