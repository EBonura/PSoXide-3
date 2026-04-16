//! MDEC — Motion Decoder.
//!
//! IDCT + Huffman decoder used primarily for .STR video playback. Driven
//! via DMA channel 0; commands written to register 0, status read from
//! register 1.
//!
//! To be populated: command word layout, status-bit definitions,
//! output-format opcodes (4-bit/8-bit/24-bit/15-bit), quant-table / IDCT
//! setup command format.
//!
//! Reference: nocash PSX-SPX "MDEC" section.

/// MDEC command/data register.
pub const MDEC0: u32 = 0x1F80_1820;

/// MDEC control/response register.
pub const MDEC1: u32 = 0x1F80_1824;
