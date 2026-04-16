//! CD-ROM controller registers and command format.
//!
//! Four registers at the same base, multiplexed by an index register.
//! Commands are submitted by writing opcodes to register 1 and parameters
//! to register 2; responses come back through the response FIFO in
//! register 1 (read).
//!
//! To be populated: index register layout, command opcodes (GetStat,
//! Setloc, ReadN, ReadS, Pause, Init, Mute, …), status-byte bitfield,
//! response-FIFO framing, sector-layout constants (Mode 2 Form 1/2).
//!
//! Reference: nocash PSX-SPX "CDROM" section.

/// CD-ROM register base. The four registers at `BASE+0..=BASE+3` change
/// role based on the current value of the index register (`BASE+0` bits 0..=1).
pub const BASE: u32 = 0x1F80_1800;

/// Sector size for Mode 2 Form 1 (data): 2048 user bytes per sector.
pub const SECTOR_SIZE_M2F1: usize = 2048;

/// Sector size for Mode 2 Form 2 (XA-ADPCM): 2324 user bytes per sector.
pub const SECTOR_SIZE_M2F2: usize = 2324;

/// Raw sector size on disc including sync, header, and ECC: 2352 bytes.
pub const SECTOR_SIZE_RAW: usize = 2352;
