//! PSoXide emulator core.
//!
//! At this stage the core exposes just enough to load a BIOS, seat a
//! CPU at the reset vector, and fetch its first instruction. No
//! execution yet — this is the thin wire along which the rest of the
//! emulator will be strung.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod bus;
pub mod cdrom;
pub mod cpu;
pub mod dma;
pub mod gpu;
pub mod gte;
pub mod hle_bios;
pub mod irq;
pub mod mdec;
pub mod mmio_trace;
pub mod pad;
pub mod scheduler;
pub mod sio;
pub mod spu;
pub mod timers;
pub mod vram;

pub use bus::{Bus, BusError};
pub use cdrom::CdRom;
pub use cpu::{Cpu, ExecutionError};
pub use dma::{Dma, DmaChannel};
pub use gpu::{DisplayArea, Gpu};
pub use gte::Gte;
pub use irq::{Irq, IrqSource};
pub use mmio_trace::{MmioKind, MmioTrace};
pub use pad::{button, ButtonState, DigitalPad, PortDevice};
pub use sio::Sio0;
pub use spu::{Spu, XaDecoderState};
pub use timers::{Timer, Timers};
pub use vram::{Vram, VRAM_HEIGHT, VRAM_WIDTH};
