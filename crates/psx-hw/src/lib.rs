//! PlayStation 1 hardware model.
//!
//! This crate is the single source of truth for PS1 hardware details:
//! register addresses, bitfield layouts, and command packet formats.
//! Both the emulator and the SDK depend on this crate; they cannot drift
//! from each other because they read the same constants.
//!
//! **This crate defines data, not behavior.** Nothing here allocates,
//! performs I/O, or implements traits beyond basic `#[derive]`s. If a
//! symbol requires `#[cfg(feature = "std")]` or a runtime allocator, it
//! does not belong here.
//!
//! Hardware references used throughout:
//! - nocash PSX-SPX (<https://psx-spx.consoledev.net/>)
//! - PCSX-Redux source tree at `/Users/ebonura/Desktop/repos/pcsx-redux`

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod memory;
pub mod interrupt;
pub mod gpu;
pub mod gte;
pub mod spu;
pub mod cdrom;
pub mod mdec;
pub mod dma;
pub mod timer;
pub mod sio;
pub mod bios;
