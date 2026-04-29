//! SPU MMIO base.

/// Base address of the Sound Processing Unit's MMIO bank.
/// Voice / volume / reverb registers are stamped at fixed
/// offsets from this address; see the SPUCNT / SPUSTAT
/// constants below for the two non-voice registers the
/// emulator surfaces today.
pub const SPU_BASE: u32 = 0x1F80_1C00;
/// SPU control register (`SPUCNT`).
pub const SPUCNT: u32 = 0x1F80_1DAA;
/// SPU status register (`SPUSTAT`).
pub const SPUSTAT: u32 = 0x1F80_1DAE;
