//! Filesystem layout + path resolution.
//!
//! Everything PSoXide writes on disk lives under one
//! platform-appropriate directory:
//!
//! ```text
//! <config-dir>/PSoXide/
//! ├── settings.ron
//! ├── library.ron
//! ├── editor/
//! │   └── workspace.ron
//! ├── games/
//! │   └── <game-id>/
//! │       ├── thumbnail.png
//! │       ├── memcard-1.mcd
//! │       ├── memcard-2.mcd
//! │       └── savestates/
//! │           ├── slot0.psx
//! │           └── slot1.psx
//! └── logs/
//!     └── last-crash.log
//! ```
//!
//! `<game-id>` is a stable 16-hex-char hash derived from the
//! disc's license text + PVD title, so renaming a BIN doesn't
//! orphan its savestates.
//!
//! `ConfigPaths` is constructed once at startup and threaded
//! through the app -- no code outside this module should build
//! paths by hand. Tests construct an instance rooted at a
//! tempdir instead of the OS config dir.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Application identifier used for the platform config-dir lookup.
/// Baked here so every subsystem agrees on the directory name.
const APP_NAME: &str = "PSoXide";

/// Resolved absolute paths to every on-disk artifact the app owns.
///
/// Construct with [`ConfigPaths::default`] in production (uses the
/// platform config dir) or [`ConfigPaths::rooted`] in tests (uses a
/// caller-supplied directory -- typically a tempdir).
#[derive(Debug, Clone)]
pub struct ConfigPaths {
    /// Root directory -- everything else hangs off this.
    root: PathBuf,
}

/// Errors from path resolution / directory creation.
#[derive(Debug, Error)]
pub enum PathError {
    /// The OS didn't give us a config directory (headless containers,
    /// broken home, etc.). Callers can fall back to `rooted` with an
    /// explicit path, or simply log and give up on persistence.
    #[error("could not resolve a platform config directory for {APP_NAME}")]
    NoConfigDir,
    /// A filesystem operation failed while trying to ensure a
    /// directory exists.
    #[error("filesystem error at {path}: {source}")]
    Io {
        /// The path we were trying to create.
        path: PathBuf,
        /// The underlying `io::Error`.
        #[source]
        source: std::io::Error,
    },
}

impl ConfigPaths {
    /// Resolve paths rooted at the platform config directory -- the
    /// normal production path. Returns [`PathError::NoConfigDir`] if
    /// the OS won't hand us one (e.g. inside some restrictive
    /// container).
    pub fn platform_default() -> Result<Self, PathError> {
        let dirs = directories::ProjectDirs::from("com", "psoxide", APP_NAME)
            .ok_or(PathError::NoConfigDir)?;
        Ok(Self {
            root: dirs.config_dir().to_path_buf(),
        })
    }

    /// Resolve paths rooted at `root`. Used by tests (so they don't
    /// touch the real config dir) and by the CLI `--config-dir` flag
    /// (so people can point the app at a portable install).
    pub fn rooted(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Absolute path to the root config directory (where
    /// `settings.ron` lives).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to `settings.ron`. Always under [`ConfigPaths::root`];
    /// callers shouldn't try to override it (keeps every
    /// PSoXide install self-consistent).
    pub fn settings_file(&self) -> PathBuf {
        self.root.join("settings.ron")
    }

    /// Path to `library.ron` (machine-generated scan cache).
    pub fn library_file(&self) -> PathBuf {
        self.root.join("library.ron")
    }

    /// Directory for host-side editor documents and autosaves.
    pub fn editor_dir(&self) -> PathBuf {
        self.root.join("editor")
    }

    /// Per-game directory under `games/<id>/`. Nothing is created --
    /// callers use `ensure_dir` when they actually need to write.
    pub fn game_dir(&self, game_id: &str) -> PathBuf {
        self.root.join("games").join(game_id)
    }

    /// Where this game's thumbnail lives (PNG).
    pub fn thumbnail_file(&self, game_id: &str) -> PathBuf {
        self.game_dir(game_id).join("thumbnail.png")
    }

    /// Where this game's save states live. Slot files go inside:
    /// `slot0.psx`, `slot1.psx`, ….
    pub fn savestates_dir(&self, game_id: &str) -> PathBuf {
        self.game_dir(game_id).join("savestates")
    }

    /// Path to a specific save-state slot. Slots are numbered from 0
    /// upward; the convention is that slot 0 = "quick save," higher
    /// slots = user-chosen.
    pub fn savestate_file(&self, game_id: &str, slot: u8) -> PathBuf {
        self.savestates_dir(game_id).join(format!("slot{slot}.psx"))
    }

    /// Memory-card file for the given port (1 or 2). Stored raw (128
    /// KiB), same format every PS1 emulator reads -- that's the one
    /// bit of community-standard interop we inherit for free.
    pub fn memcard_file(&self, game_id: &str, port: u8) -> PathBuf {
        let clamped = port.clamp(1, 2);
        self.game_dir(game_id)
            .join(format!("memcard-{clamped}.mcd"))
    }

    /// Rolling crash-log file. Appended to from panic hooks.
    pub fn crash_log(&self) -> PathBuf {
        self.root.join("logs").join("last-crash.log")
    }

    /// Ensure `dir` exists as a directory, creating parents as
    /// needed. Idempotent. Errors are wrapped with the path so
    /// callers can include it in user messages.
    pub fn ensure_dir(&self, dir: &Path) -> Result<(), PathError> {
        std::fs::create_dir_all(dir).map_err(|source| PathError::Io {
            path: dir.to_path_buf(),
            source,
        })
    }

    /// Convenience wrapper around [`ensure_dir`] for the per-game
    /// tree -- the usual thing the caller wants before writing a
    /// save state or thumbnail.
    pub fn ensure_game_tree(&self, game_id: &str) -> Result<(), PathError> {
        self.ensure_dir(&self.game_dir(game_id))?;
        self.ensure_dir(&self.savestates_dir(game_id))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn rooted_paths_nest_correctly() {
        let tmp = TempDir::new().unwrap();
        let p = ConfigPaths::rooted(tmp.path());
        let game = "abc123";
        assert_eq!(p.settings_file(), tmp.path().join("settings.ron"));
        assert_eq!(p.library_file(), tmp.path().join("library.ron"));
        assert_eq!(p.editor_dir(), tmp.path().join("editor"));
        assert_eq!(p.game_dir(game), tmp.path().join("games/abc123"));
        assert_eq!(
            p.savestate_file(game, 3),
            tmp.path().join("games/abc123/savestates/slot3.psx")
        );
        assert_eq!(
            p.memcard_file(game, 2),
            tmp.path().join("games/abc123/memcard-2.mcd")
        );
        assert_eq!(
            p.thumbnail_file(game),
            tmp.path().join("games/abc123/thumbnail.png")
        );
    }

    #[test]
    fn ensure_game_tree_creates_directories() {
        let tmp = TempDir::new().unwrap();
        let p = ConfigPaths::rooted(tmp.path());
        p.ensure_game_tree("mygame").unwrap();
        assert!(p.game_dir("mygame").is_dir());
        assert!(p.savestates_dir("mygame").is_dir());
    }

    #[test]
    fn ensure_dir_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let p = ConfigPaths::rooted(tmp.path());
        p.ensure_game_tree("g1").unwrap();
        p.ensure_game_tree("g1").unwrap(); // second call must succeed
    }

    #[test]
    fn memcard_port_is_clamped() {
        let tmp = TempDir::new().unwrap();
        let p = ConfigPaths::rooted(tmp.path());
        // Ports 0 and 3 both clamp to 1/2 -- no surprising out-of-range files.
        assert!(p
            .memcard_file("g", 0)
            .to_string_lossy()
            .ends_with("memcard-1.mcd"));
        assert!(p
            .memcard_file("g", 7)
            .to_string_lossy()
            .ends_with("memcard-2.mcd"));
    }
}
