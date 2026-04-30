//! CD-ROM MMIO -- 4-register index-based.

/// Base address; all four registers share it and switch meaning
/// via the low-2 bits of register `0` (index register).
pub const BASE: u32 = 0x1F80_1800;
