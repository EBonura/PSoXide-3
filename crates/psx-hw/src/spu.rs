//! SPU -- Sound Processing Unit.
//!
//! 24 voices, 512 KiB of dedicated SPU RAM, ADSR envelopes, reverb,
//! and XA-ADPCM stereo decoding for CD audio streaming.
//!
//! To be populated: register base, voice register block layout (16 bytes
//! per voice × 24 voices), main volume / reverb / key-on/off registers,
//! ADSR encoding format, XA-ADPCM block layout.
//!
//! Reference: nocash PSX-SPX "SPU" section.

/// SPU register base address. Voice registers and global controls are
/// laid out in a contiguous 640-byte block starting here.
pub const BASE: u32 = 0x1F80_1C00;
