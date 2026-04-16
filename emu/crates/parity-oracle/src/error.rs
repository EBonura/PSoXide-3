//! Error type for the harness.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Errors from launching or driving a Redux subprocess.
#[derive(Error, Debug)]
pub enum OracleError {
    /// The Redux binary could not be found at the configured or fallback
    /// paths.
    #[error("Redux binary not found (tried {tried:?}). Set PSOXIDE_REDUX_BIN or build pcsx-redux.")]
    BinaryNotFound {
        /// The list of paths that were searched, in order.
        tried: Vec<PathBuf>,
    },

    /// A required file (BIOS, Lua script) was missing or unreadable.
    #[error("required file missing at {path}: {source}")]
    MissingFile {
        /// Path that could not be accessed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },

    /// Spawning the subprocess failed.
    #[error("failed to spawn Redux: {0}")]
    Spawn(#[source] io::Error),

    /// Timed out waiting for a stdout marker.
    #[error("timed out after {timeout_ms}ms waiting for marker {marker:?} (captured stdout tail: {tail:?})")]
    Timeout {
        /// Marker string that never appeared.
        marker: String,
        /// Timeout in milliseconds.
        timeout_ms: u64,
        /// Last ~1 KiB of captured stdout for diagnosis.
        tail: String,
    },

    /// The Redux process exited before producing the expected output.
    #[error("Redux exited early with status {status:?}; stdout tail: {stdout_tail:?}; stderr tail: {stderr_tail:?}")]
    EarlyExit {
        /// Exit status reported by the child.
        status: Option<i32>,
        /// Last ~1 KiB of captured stdout.
        stdout_tail: String,
        /// Last ~1 KiB of captured stderr.
        stderr_tail: String,
    },

    /// Redux returned something other than the expected protocol response.
    #[error("protocol mismatch: expected {expected:?}, got {got:?}")]
    Protocol {
        /// Response the harness was expecting.
        expected: String,
        /// Response that actually arrived.
        got: String,
    },

    /// The command could not be sent because stdin is closed.
    #[error("Redux stdin is closed; can no longer send commands")]
    StdinClosed,

    /// Generic I/O error encountered during harness operation.
    #[error("harness I/O error: {0}")]
    Io(#[from] io::Error),
}
