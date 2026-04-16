//! BIOS function vectors and POST codes.
//!
//! The PS1 BIOS exposes three function-call tables accessed via `JAL` to
//! fixed entry points; the actual function is selected by the `t1`
//! register. Table A is the largest (≈130 entries for stdio, memcard,
//! file I/O); table B covers process/event/thread control; table C
//! handles interrupt vectors and install routines.
//!
//! To be populated: per-function enum for each of the three tables with
//! argument conventions and known quirks; POST code enumeration written
//! to `0x1F80_2041` (see [`crate::memory::expansion2::POST`]).
//!
//! Reference: nocash PSX-SPX "BIOS Function Summary" section.

/// Entry point for BIOS table A functions. `JAL` here with the desired
/// function number in `$t1`.
pub const VECTOR_A: u32 = 0xA000_00A0;

/// Entry point for BIOS table B functions.
pub const VECTOR_B: u32 = 0xB000_00B0;

/// Entry point for BIOS table C functions.
pub const VECTOR_C: u32 = 0xC000_00C0;
