//! Persistence layer for PSoXide.
//!
//! Three kinds of data land on disk:
//!
//! - **`settings.ron`** — user-editable preferences. BIOS paths,
//!   input bindings, UI prefs. Written in RON (Rusty Object
//!   Notation) so enums round-trip losslessly and humans can
//!   still hand-edit it with comments.
//! - **`library.ron`** — machine-generated cache of the game
//!   library. Regenerable by re-scanning, but caching it avoids
//!   re-parsing every BIN's PVD on every startup.
//! - **`games/<id>/*`** — per-game data tree. Thumbnails (PNG),
//!   save states (postcard), memory cards (raw 128 KiB in the
//!   standard PS1 format every emulator reads).
//!
//! Everything lives under a platform-appropriate config dir (via
//! the `directories` crate):
//!
//! - macOS:  `~/Library/Application Support/PSoXide/`
//! - Linux:  `$XDG_CONFIG_HOME/PSoXide/`
//! - Windows: `%APPDATA%\PSoXide\`
//!
//! The module intentionally has no GUI / windowing / audio
//! dependencies so it's equally usable from the frontend and from
//! any headless CLI tool we build (scan/launch/savestate-probe),
//! and so its tests run in milliseconds without a window.

#![warn(missing_docs)]

pub mod library;
pub mod paths;
pub mod savestate;
pub mod settings;

pub use library::{Library, LibraryEntry, LibraryError};
pub use paths::{ConfigPaths, PathError};
pub use savestate::{SaveStateError, SaveStateV1, SAVESTATE_MAGIC};
pub use settings::{EditorSettings, InputBinding, Settings, SettingsError};

/// Current on-disk format version for `settings.ron`. Bumped when
/// a breaking change lands; the loader migrates older versions in
/// place rather than failing outright.
pub const SETTINGS_VERSION: u32 = 1;

/// Current version for `library.ron`. Cache files from older
/// versions are discarded silently — the scanner regenerates them.
pub const LIBRARY_VERSION: u32 = 1;
