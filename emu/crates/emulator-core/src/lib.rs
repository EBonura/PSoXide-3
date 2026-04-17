//! PSoXide emulator core.
//!
//! At this stage the core exposes just enough to load a BIOS, seat a
//! CPU at the reset vector, and fetch its first instruction. No
//! execution yet — this is the thin wire along which the rest of the
//! emulator will be strung.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod bus;
pub mod cpu;
pub mod dma;
pub mod gpu;
pub mod irq;
pub mod timers;
pub mod vram;

pub use bus::{Bus, BusError};
pub use cpu::{Cpu, ExecutionError};
pub use dma::{Dma, DmaChannel};
pub use gpu::Gpu;
pub use irq::{Irq, IrqSource};
pub use timers::{Timer, Timers};
pub use vram::{Vram, VRAM_HEIGHT, VRAM_WIDTH};
