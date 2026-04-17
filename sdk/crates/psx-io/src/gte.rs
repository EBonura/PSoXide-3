//! GTE (COP2) access.
//!
//! The GTE isn't MMIO — it's the MIPS coprocessor 2 and has its own
//! instruction class. This module is an empty stub for now; the real
//! access path is in `psx-gte` via inline assembly (`MFC2` / `MTC2` /
//! `CFC2` / `CTC2` / the GTE function ops).
