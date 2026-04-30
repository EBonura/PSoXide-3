//! GTE (COP2) access -- PS1 Geometry Transformation Engine.
//!
//! The GTE isn't memory-mapped. It's MIPS coprocessor 2, accessed via
//! a dedicated instruction class (`MTC2`/`MFC2`/`CTC2`/`CFC2` to move
//! data between CPU and COP2 registers, `COP2 cofun` to run function
//! ops like RTPS/RTPT/MVMVA/NCDS).
//!
//! This crate exposes two layers:
//!
//! - **Low-level register macros**: [`mtc2!`], [`mfc2!`], [`ctc2!`],
//!   [`cfc2!`]. Each takes a literal register index (0..31) so the
//!   assembler emits the correct immediate field. Use these when you
//!   need direct access for performance-sensitive paths.
//!
//! - **High-level operation wrappers**: zero-argument inline functions
//!   for the common GTE commands with their typical options baked in
//!   (e.g. [`ops::rtps`] uses `sf=1, lm=0`). On MIPS each compiles to
//!   a single 4-byte `.word`. On host they dispatch to a per-thread
//!   software GTE living in [`host`].
//!
//! All function-op wrappers are `unsafe fn` -- they assume the caller
//! has loaded the required input registers via the register macros.
//!
//! # Same simulation, two backends
//!
//! - On `target_arch = "mips"` everything compiles down to direct
//!   coprocessor instructions, identical to writing the assembly by
//!   hand.
//! - On host the macros and ops route through [`psx_gte_core::Gte`]
//!   stored in a `thread_local!` cell. The simulation is the
//!   bit-faithful one the emulator already runs against PCSX-Redux's
//!   parity oracle, so editor previews match what the hardware draws.
//!
//! # Example
//!
//! ```ignore
//! use psx_gte::{ctc2, mtc2, mfc2, ops::rtps};
//!
//! unsafe {
//!     ctc2!(0, 0x0000_1000); // RT[0][0]=0x1000, RT[0][1]=0
//!     // … fill in the rest of the rotation matrix and TR …
//!     mtc2!(0, (10 & 0xFFFF) | (20 << 16));
//!     mtc2!(1, 30);
//!     rtps();
//!     let sxy2: u32 = mfc2!(14);
//! }
//! ```

#![cfg_attr(target_arch = "mips", no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![cfg_attr(target_arch = "mips", feature(asm_experimental_arch))]

pub mod lighting;
pub mod math;
pub mod ops;
pub mod regs;
pub mod scene;
pub mod transform;

#[cfg(not(target_arch = "mips"))]
pub mod host;
