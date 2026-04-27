//! Composable 3D transform math.
//!
//! Re-exports from [`psx_gte_core::transform`] so callers continue to
//! find `Mat3I16::rotate_y(...)` / `sin_1_3_12` / `cos_1_3_12` at
//! their familiar `psx_gte::transform::*` paths. The actual `impl`s
//! live in `psx-gte-core` next to the [`Mat3I16`] / [`Vec3I16`]
//! definitions so we satisfy Rust's orphan rules without a wrapper
//! type.

pub use psx_gte_core::transform::{cos_1_3_12, sin_1_3_12};
