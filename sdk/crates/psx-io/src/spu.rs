//! SPU MMIO base.
pub const SPU_BASE: u32 = 0x1F80_1C00;
/// SPU control register (`SPUCNT`).
pub const SPUCNT: u32 = 0x1F80_1DAA;
/// SPU status register (`SPUSTAT`).
pub const SPUSTAT: u32 = 0x1F80_1DAE;
