//! SIO0 (controller / memory-card) MMIO.
//!
//! Keeping this thin for now: register reads/writes only. The full
//! controller protocol (address byte, polling sequence, response
//! decoding) lands in a higher-level `psx-pad` crate when we need it.

/// TX/RX FIFO port (byte).
pub const DATA: u32 = 0x1F80_1040;
/// Status register (32-bit).
pub const STAT: u32 = 0x1F80_1044;
/// Mode register (16-bit).
pub const MODE: u32 = 0x1F80_1048;
/// Control register (16-bit).
pub const CTRL: u32 = 0x1F80_104A;
/// Baud-rate divisor (16-bit).
pub const BAUD: u32 = 0x1F80_104E;
