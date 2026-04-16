//! PSoXide-3 emulator core.
//!
//! At this stage the core exposes just enough to load a BIOS, seat a
//! CPU at the reset vector, and fetch its first instruction. No
//! execution yet — this is the thin wire along which the rest of the
//! emulator will be strung.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod bus;
pub mod cpu;

pub use bus::{Bus, BusError};
pub use cpu::{Cpu, ExecutionError};
