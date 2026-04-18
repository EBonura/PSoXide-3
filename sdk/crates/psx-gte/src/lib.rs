//! GTE (COP2) access — PS1 Geometry Transformation Engine.
//!
//! The GTE isn't memory-mapped. It's MIPS coprocessor 2, accessed via
//! a dedicated instruction class (`MTC2`/`MFC2`/`CTC2`/`CFC2` to move
//! data between CPU and COP2 registers, `COP2 cofun` to run function
//! ops like RTPS/RTPT/MVMVA/NCDS). That means we can only reach it
//! through inline assembly — no register banks, no MMIO addresses.
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
//!   (e.g. [`rtps`] uses `sf=1, lm=0`). These compile down to a single
//!   4-byte `.word` instruction. Variants with other sf/lm settings
//!   are available as suffixed functions (`rtps_sf0`, `rtps_sf1_lm1`,
//!   etc.) where they're useful.
//!
//! All functions are `unsafe fn` — they assume the caller has loaded
//! the required input registers via `mtc2!`/`ctc2!` first.
//!
//! # Example
//!
//! ```ignore
//! use psx_gte::{ctc2, mtc2, mfc2, rtps};
//!
//! // Set up rotation matrix = identity (1.3.12 fixed point).
//! unsafe {
//!     ctc2!(0, 0x0000_1000); // RT[0][0]=0x1000, RT[0][1]=0
//!     ctc2!(1, 0x0000_0000); // RT[0][2]=0, RT[1][0]=0
//!     ctc2!(2, 0x0000_1000); // RT[1][1]=0x1000, RT[1][2]=0
//!     ctc2!(3, 0x0000_0000); // RT[2][0]=0, RT[2][1]=0
//!     ctc2!(4, 0x0000_1000); // RT[2][2]=0x1000
//!     // Load V0 = (10, 20, 30).
//!     mtc2!(0, (10 & 0xFFFF) | (20 << 16));
//!     mtc2!(1, 30);
//!     rtps(); // project V0
//!     // Read SX2/SY2.
//!     let sxy2: u32 = mfc2!(14);
//! }
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![cfg_attr(target_arch = "mips", feature(asm_experimental_arch))]

pub mod math;
pub mod regs;
pub mod scene;
pub mod transform;

#[cfg(target_arch = "mips")]
pub mod ops;

// Host-target fallback — lets the SDK `cargo check` on non-MIPS without
// fabricating fake GTE behaviour. Functions panic if called; they exist
// only so downstream code compiles when cross-checking on the host.
#[cfg(not(target_arch = "mips"))]
#[allow(missing_docs)]
pub mod ops {
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn rtps() {
        panic!("psx-gte::rtps only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn rtpt() {
        panic!("psx-gte::rtpt only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn nclip() {
        panic!("psx-gte::nclip only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn op_sf1() {
        panic!("psx-gte::op_sf1 only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn avsz3() {
        panic!("psx-gte::avsz3 only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn avsz4() {
        panic!("psx-gte::avsz4 only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn sqr() {
        panic!("psx-gte::sqr only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn ncds() {
        panic!("psx-gte::ncds only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn nccs() {
        panic!("psx-gte::nccs only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn ncs() {
        panic!("psx-gte::ncs only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn gpf() {
        panic!("psx-gte::gpf only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn gpl() {
        panic!("psx-gte::gpl only runs on target_arch = mips")
    }
    /// Host stub — real implementation targets MIPS only.
    #[cold]
    pub unsafe fn mvmva_rt_v0_tr_sf1() {
        panic!("psx-gte::mvmva_rt_v0_tr_sf1 only runs on target_arch = mips")
    }
}

pub use ops::*;
