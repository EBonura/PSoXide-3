//! PSoXide-3 SDK — umbrella crate.
//!
//! Re-exports the per-subsystem crates so homebrew doesn't have to
//! track each one individually:
//!
//! ```ignore
//! use psx_sdk::{gpu, rt, io};
//!
//! #[no_mangle]
//! fn main() {
//!     gpu::init(gpu::VideoMode::Ntsc, gpu::Resolution::R320X240);
//!     loop {
//!         gpu::fill_rect(0, 0, 320, 240, 255, 0, 0);
//!         gpu::vsync();
//!     }
//! }
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub use psx_gpu as gpu;
pub use psx_hw as hw;
pub use psx_io as io;
pub use psx_rt as rt;
