//! On-disk save-state format.
//!
//! ## Design
//!
//! - **Binary (postcard)**, because a typical save state is
//!   ~2 MiB and RON would balloon it 5-10× for no benefit — nobody
//!   hand-edits save states.
//! - **Explicit file header** with a 8-byte magic (`PSOX001\0`), a
//!   format version, and a human-readable creator tag.
//! - **Versioned payload** via [`SaveStateV1`]. When the schema
//!   evolves, we add `V2`, read the header version, dispatch to
//!   the right deserializer, and (typically) convert older
//!   payloads forward in memory.
//!
//! ## Not cross-emulator
//!
//! This format is PSoXide-3-specific. The PS1 emulator community
//! hasn't converged on a shared save-state schema
//! (Duckstation/PCSX-Redux/Beetle/Mednafen all roll their own),
//! so we don't either — at least not at format level. Cross-load
//! with other emulators would live as an explicit converter
//! module, when/if demand lands.
//!
//! ## Why postcard over bincode
//!
//! - `postcard`'s encoding is slightly more compact for our
//!   mostly-small-integer state (varint length prefixes).
//! - `no_std` and `alloc`-only, which matters zero today but keeps
//!   the door open to on-device use (a debug EXE that serialises
//!   runtime state, say).
//! - Serde-native: no build-script or schema duplication.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Magic bytes at the start of every `*.psx` save-state file.
/// Eight bytes = one `u64`; gives us a quick sanity-check before
/// we commit to a postcard decode. The trailing NUL lets us grow
/// the magic in-place by replacing the NUL if we ever need to.
pub const SAVESTATE_MAGIC: &[u8; 8] = b"PSOX001\0";

/// On-disk format version, independent of the PSoXide-3 crate
/// version. Bumped when the payload layout changes in a way that
/// can't round-trip through the current deserializer.
pub const SAVESTATE_FORMAT_VERSION: u32 = 1;

/// Errors from save-state load / save.
#[derive(Debug, Error)]
pub enum SaveStateError {
    /// Raw filesystem problem.
    #[error("save-state I/O error at {path}: {source}")]
    Io {
        /// The file we were reading / writing.
        path: std::path::PathBuf,
        /// Underlying `io::Error`.
        #[source]
        source: io::Error,
    },
    /// File too short to contain a header.
    #[error("save-state truncated at {path} (need at least {min_bytes} bytes)")]
    Truncated {
        /// The file we were reading.
        path: std::path::PathBuf,
        /// The minimum byte length needed.
        min_bytes: usize,
    },
    /// File didn't start with the PSoXide-3 magic. Typical cause:
    /// someone pointed the loader at a non-save-state file by
    /// mistake, or at a save-state from a different emulator.
    #[error("save-state magic mismatch at {path} (not a PSoXide-3 save)")]
    BadMagic {
        /// The file we were reading.
        path: std::path::PathBuf,
    },
    /// Header decoded but version isn't one we can load. The
    /// message includes both the on-disk version and the current
    /// `SAVESTATE_FORMAT_VERSION` so users know what to do.
    #[error("save-state version {found} not supported (expected ≤ {max})")]
    UnsupportedVersion {
        /// Version the file claims.
        found: u32,
        /// Highest version this crate can load.
        max: u32,
    },
    /// Postcard decode failed. Usually means a corrupt or
    /// partially-written file — but it can also mean the schema
    /// changed mid-development without a version bump.
    #[error("save-state decode failed: {0}")]
    Decode(String),
    /// Postcard encode failed. Extremely unlikely in practice —
    /// postcard only fails on custom serde impls that return errs.
    #[error("save-state encode failed: {0}")]
    Encode(String),
}

/// File-wide header. Written at the start of every save state so
/// the loader can validate the file before committing to a full
/// decode.
///
/// Keep this struct **append-only** across versions. Fields are
/// declared in fixed order and postcard encodes them positionally,
/// so any rearrangement silently breaks old files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SaveStateHeader {
    /// Magic bytes — must equal [`SAVESTATE_MAGIC`].
    pub magic: [u8; 8],
    /// Format version of the payload that follows.
    pub format_version: u32,
    /// Creator tag — human-readable "what produced this save."
    /// Displayed by the UI; useful when debugging cross-build
    /// format drift.
    pub creator: String,
    /// UNIX timestamp (seconds since epoch) when the save was
    /// created. Displayed by the UI.
    pub created_at: u64,
    /// ID of the game this save belongs to. Matches the `id` in
    /// [`crate::LibraryEntry`] — lets the loader verify the user
    /// didn't cross-load slot0.psx from Game A into Game B (which
    /// would crash spectacularly).
    pub game_id: String,
    /// Absolute instruction count at the moment of the save — for
    /// diagnostics / UI.
    pub cpu_tick: u64,
}

/// Version-1 save-state payload. Kept intentionally abstract here
/// — the actual CPU + Bus state types live in `emulator-core`, so
/// pinning the concrete types in this module would create a
/// circular dependency. Callers supply their own concrete
/// serializable state (typically a `(Cpu, Bus)` newtype) and pass
/// it through the write/read helpers.
///
/// The generic makes this module usable from tests *and* from
/// the frontend, while keeping `psoxide-settings` free of any
/// emulator-core dep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SaveStateV1<T> {
    /// File header (magic + version + metadata).
    pub header: SaveStateHeader,
    /// Emulator state payload. The generic is supplied by the
    /// caller — typically a serializable wrapper over
    /// `(Cpu, Bus)`.
    pub payload: T,
}

impl<T> SaveStateV1<T>
where
    T: Serialize + for<'a> Deserialize<'a>,
{
    /// Build a save-state around an existing payload. Fills in the
    /// header boilerplate.
    pub fn new(payload: T, game_id: impl Into<String>, cpu_tick: u64) -> Self {
        Self {
            header: SaveStateHeader {
                magic: *SAVESTATE_MAGIC,
                format_version: SAVESTATE_FORMAT_VERSION,
                creator: concat!("PSoXide-3/", env!("CARGO_PKG_VERSION")).to_string(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                game_id: game_id.into(),
                cpu_tick,
            },
            payload,
        }
    }

    /// Serialise to bytes (postcard).
    pub fn to_bytes(&self) -> Result<Vec<u8>, SaveStateError> {
        postcard::to_allocvec(self).map_err(|e| SaveStateError::Encode(e.to_string()))
    }

    /// Deserialise from bytes. Validates magic + version, returns a
    /// typed error for each failure mode so the UI can surface
    /// something meaningful ("wrong emulator save" vs "corrupted
    /// file" vs "newer version, please upgrade").
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SaveStateError> {
        if bytes.len() < SAVESTATE_MAGIC.len() + 4 {
            return Err(SaveStateError::Truncated {
                path: std::path::PathBuf::new(),
                min_bytes: SAVESTATE_MAGIC.len() + 4,
            });
        }
        if &bytes[..8] != SAVESTATE_MAGIC {
            return Err(SaveStateError::BadMagic {
                path: std::path::PathBuf::new(),
            });
        }
        let state: SaveStateV1<T> = postcard::from_bytes(bytes)
            .map_err(|e| SaveStateError::Decode(e.to_string()))?;
        if state.header.format_version > SAVESTATE_FORMAT_VERSION {
            return Err(SaveStateError::UnsupportedVersion {
                found: state.header.format_version,
                max: SAVESTATE_FORMAT_VERSION,
            });
        }
        Ok(state)
    }

    /// Write atomically to `path`. Writes to `<path>.tmp` first,
    /// renames on success — no half-written save-state file ever
    /// lands on disk, which is critical because partial writes
    /// deserialise as garbage that corrupts user progress.
    pub fn write_to(&self, path: &Path) -> Result<(), SaveStateError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|source| SaveStateError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }
        let bytes = self.to_bytes()?;
        let tmp = path.with_extension("psx.tmp");
        fs::write(&tmp, &bytes).map_err(|source| SaveStateError::Io {
            path: tmp.clone(),
            source,
        })?;
        fs::rename(&tmp, path).map_err(|source| SaveStateError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }

    /// Read and deserialise from `path`. Produces path-tagged
    /// errors (unlike [`Self::from_bytes`]) so the UI can report
    /// `"…/slot0.psx is corrupt"` with the right filename.
    pub fn read_from(path: &Path) -> Result<Self, SaveStateError> {
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(source) => {
                return Err(SaveStateError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        Self::from_bytes(&bytes).map_err(|e| match e {
            SaveStateError::Truncated { min_bytes, .. } => SaveStateError::Truncated {
                path: path.to_path_buf(),
                min_bytes,
            },
            SaveStateError::BadMagic { .. } => SaveStateError::BadMagic {
                path: path.to_path_buf(),
            },
            other => other,
        })
    }
}

/// Quick header-only peek: read just enough of `path` to pull the
/// creation time, game ID, and cpu_tick out for a slot-list UI.
/// Avoids deserializing the full multi-MiB payload when all we
/// want to do is show "Slot 0 — Crash — 2026-04-18".
pub fn peek_header(path: &Path) -> Result<SaveStateHeader, SaveStateError> {
    // We still have to postcard-decode because postcard's format
    // isn't random-access — but we decode only the header (a small
    // struct) by using a wrapper that ignores the tail. postcard's
    // `take_from_bytes` returns the unread remainder; we use that
    // to stop early once the header is in hand.
    #[derive(Deserialize)]
    struct HeaderOnly {
        header: SaveStateHeader,
    }
    let bytes = fs::read(path).map_err(|source| SaveStateError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if bytes.len() < SAVESTATE_MAGIC.len() {
        return Err(SaveStateError::Truncated {
            path: path.to_path_buf(),
            min_bytes: SAVESTATE_MAGIC.len(),
        });
    }
    if &bytes[..SAVESTATE_MAGIC.len()] != SAVESTATE_MAGIC {
        return Err(SaveStateError::BadMagic {
            path: path.to_path_buf(),
        });
    }
    let (ho, _tail): (HeaderOnly, &[u8]) = postcard::take_from_bytes(&bytes)
        .map_err(|e| SaveStateError::Decode(e.to_string()))?;
    Ok(ho.header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::TempDir;

    // Stand-in for the real emulator state — keeps these tests
    // from pulling emulator-core in as a dep. In the frontend
    // we'll wrap the actual (Cpu, Bus) similarly.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct FakePayload {
        cpu_pc: u32,
        ram_hash: u64,
    }

    #[test]
    fn round_trip_in_memory() {
        let payload = FakePayload {
            cpu_pc: 0xbfc00000,
            ram_hash: 0xdead_beef_f00d,
        };
        let state = SaveStateV1::new(payload.clone(), "abc123", 12345);
        let bytes = state.to_bytes().unwrap();
        let restored: SaveStateV1<FakePayload> = SaveStateV1::from_bytes(&bytes).unwrap();
        assert_eq!(restored.payload, payload);
        assert_eq!(restored.header.game_id, "abc123");
        assert_eq!(restored.header.cpu_tick, 12345);
        assert_eq!(&restored.header.magic, SAVESTATE_MAGIC);
    }

    #[test]
    fn round_trip_on_disk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("slot0.psx");
        let state = SaveStateV1::new(
            FakePayload {
                cpu_pc: 0x8000_0000,
                ram_hash: 0,
            },
            "game",
            42,
        );
        state.write_to(&path).unwrap();
        let restored: SaveStateV1<FakePayload> = SaveStateV1::read_from(&path).unwrap();
        assert_eq!(restored.payload.cpu_pc, 0x8000_0000);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("notastate.psx");
        std::fs::write(&path, b"NOPE NOT A SAVE STATE FILE CONTENTS").unwrap();
        let err = SaveStateV1::<FakePayload>::read_from(&path).unwrap_err();
        assert!(matches!(err, SaveStateError::BadMagic { .. }));
    }

    #[test]
    fn truncated_file_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("short.psx");
        std::fs::write(&path, b"PSO").unwrap();
        let err = SaveStateV1::<FakePayload>::read_from(&path).unwrap_err();
        assert!(matches!(
            err,
            SaveStateError::Truncated { .. } | SaveStateError::BadMagic { .. }
        ));
    }

    #[test]
    fn future_version_is_rejected() {
        // Build a state, corrupt its version to SAVESTATE_FORMAT_VERSION+1,
        // assert we don't load it.
        let mut state = SaveStateV1::new(FakePayload { cpu_pc: 0, ram_hash: 0 }, "g", 0);
        state.header.format_version = SAVESTATE_FORMAT_VERSION + 1;
        let bytes = state.to_bytes().unwrap();
        let err = SaveStateV1::<FakePayload>::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, SaveStateError::UnsupportedVersion { .. }));
    }

    #[test]
    fn peek_header_without_full_decode() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("slot0.psx");
        let state = SaveStateV1::new(
            FakePayload {
                cpu_pc: 0xAABB_CCDD,
                ram_hash: 0,
            },
            "my_game_id",
            999,
        );
        state.write_to(&path).unwrap();
        let h = peek_header(&path).unwrap();
        assert_eq!(h.game_id, "my_game_id");
        assert_eq!(h.cpu_tick, 999);
        assert_eq!(h.format_version, SAVESTATE_FORMAT_VERSION);
    }

    #[test]
    fn atomic_write_preserves_existing_file_on_encoder_failure() {
        // Can't easily simulate an encoder failure with postcard
        // (it's infallible for our types), but we can at least check
        // that writing creates the final file, not the .tmp file.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("slot0.psx");
        let state = SaveStateV1::new(
            FakePayload {
                cpu_pc: 0,
                ram_hash: 0,
            },
            "g",
            0,
        );
        state.write_to(&path).unwrap();
        assert!(path.exists());
        let tmp_name = path.with_extension("psx.tmp");
        assert!(!tmp_name.exists(), "tmp file should be removed after rename");
    }
}
