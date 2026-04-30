//! Cooked-asset binary formats for PSoXide.
//!
//! This crate defines the on-disk layouts produced by the host
//! editor (`editor/crates/psxed`) and consumed by the MIPS-side
//! runtime parser (`sdk/crates/psx-asset`). Keeping the format
//! constants in a single crate that both sides depend on makes
//! it structurally impossible for producer + consumer to drift.
//!
//! The crate is **`no_std`** -- it compiles on every target we
//! ship to, host or PS1. It defines only:
//!
//! - Magic byte identifiers for each asset type
//! - Format-version constants
//! - `#[repr(C, packed)]` header structs laid out byte-exact to
//!   the file contents
//!
//! Encoders (host side) and decoders (PS1 side) live in
//! separate crates. This crate knows nothing about IO, serde,
//! or allocation -- it's a layout-only contract.
//!
//! ## Wire format
//!
//! Every cooked asset starts with a 12-byte common header:
//!
//! ```text
//!   offset  bytes  field
//!        0      4  magic    ASCII 4-char identifier (e.g. b"PSXM")
//!        4      2  version  u16 LE -- format revision, bumped on breakage
//!        6      2  flags    u16 LE -- asset-type-specific feature bits
//!        8      4  payload_len u32 LE -- size of the rest of the file
//! ```
//!
//! Each asset type appends its own payload after the header.
//! Payload layouts live in the per-type modules ([`mesh`]).
//!
//! ## Byte order
//!
//! All multi-byte integers are **little-endian**, matching PS1
//! / MIPS cooked order. Consumers on big-endian hosts would need
//! byte-swapping -- not a concern for our all-LE toolchain today.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod animation;
pub mod mesh;
pub mod model;
pub mod texture;
pub mod world;

/// Shared header that prefixes every cooked asset file.
///
/// Storing this as a fixed 12-byte block simplifies the runtime
/// parser -- one read, check `magic`, check `version`, then
/// dispatch on the known payload layout for that `magic`.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct AssetHeader {
    /// 4-byte ASCII magic identifying the asset type.
    pub magic: [u8; 4],
    /// Format version. Bumped on layout breakage.
    pub version: u16,
    /// Type-specific feature flags.
    pub flags: u16,
    /// Total payload byte count after this header.
    pub payload_len: u32,
}

impl AssetHeader {
    /// Size of the header in bytes (always 12).
    pub const SIZE: usize = 12;

    /// Build a header with the given fields, no encoding happens.
    pub const fn new(magic: [u8; 4], version: u16, flags: u16, payload_len: u32) -> Self {
        Self {
            magic,
            version,
            flags,
            payload_len,
        }
    }
}
