//! Pure-Rust GTE state machine and shared math types.
//!
//! Both the PS1 SDK (`psx-gte`, host build) and the emulator
//! (`emulator-core`'s CPU COP2 dispatcher) consume this crate, so the
//! editor's 3D preview projects vertices through *the same* GTE
//! simulation that the running emulator does.
//!
//! No I/O, no inline assembly, no PS1 hardware bindings. Everything is
//! `core::cmp`-only arithmetic; safe to use from any host or target.
//!
//! - [`math`]: fixed-point types ([`Vec3I16`], [`Vec3I32`],
//!   [`Mat3I16`]) shared across the GTE / GPU layers.
//! - [`state::Gte`]: full register state + the 21 documented function
//!   opcodes (RTPS / RTPT / MVMVA / NCLIP / NCDS / AVSZ3 / etc.).
//!   Driven via [`Gte::execute`] with a 32-bit command word.

#![no_std]
#![warn(missing_docs)]

pub mod math;
pub mod state;
pub mod transform;

pub use math::{Mat3I16, Vec3I16, Vec3I32};
pub use state::{Gte, GteProfileSnapshot};
pub use transform::{cos_1_3_12, sin_1_3_12};
