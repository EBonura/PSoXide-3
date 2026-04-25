//! PCSX-Redux harness.
//!
//! This crate owns the lifecycle of a headless Redux subprocess that
//! we drive to produce reference traces. It encapsulates every painful
//! lesson from PSoXide-2's harness:
//!
//! - Use `-no-ui`, **not** `-cli` (the latter silently skips config).
//! - Drain stdout/stderr in background threads: Redux will deadlock if
//!   pipe backpressure stalls.
//! - `try_wait` after each poll to surface child death promptly
//!   (Redux can SIGSEGV mid-run; we must not block forever on a dead
//!   process).
//! - The run directory lives under `std::env::temp_dir()`, which on
//!   macOS is `/var/folders/.../T/`, **not** `/tmp`.
//!
//! The public surface is intentionally minimal right now: configure,
//! launch, wait for a stdout marker, terminate. HTTP/stepping APIs
//! land in a subsequent milestone once the lifecycle is trusted.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod cache;
mod config;
mod error;
mod launch;

pub use config::OracleConfig;
pub use error::OracleError;
pub use launch::{ReduxProcess, StateCheckpoint};
