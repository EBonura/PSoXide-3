//! GTE (COP2) -- Geometry Transformation Engine.
//!
//! Not a memory-mapped device. Accessed via MIPS COP2 instructions
//! (`MTC2`, `MFC2`, `CTC2`, `CFC2`, and the `COP2` command opcodes).
//!
//! To be populated: data register (`cp2d`) indices 0..=31, control
//! register (`cp2c`) indices 0..=31, command opcodes (RTPS, RTPT,
//! MVMVA, NCDS, etc.) and their bitfield flags (`sf`, `lm`, `mvmva`).
//!
//! Reference: nocash PSX-SPX "GTE" section.
