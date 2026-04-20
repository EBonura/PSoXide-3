//! Arcade-style visual effects for PS1 homebrew.
//!
//! Extracted from the duplicated particle / shake / RNG pattern
//! that breakout and invaders both grew independently. A *primitive*
//! layer, not an engine:
//!
//! - [`rng::LcgRng`] — deterministic, no_std, no_alloc integer
//!   PRNG. Every consumer uses this for particle velocity spread
//!   so replays + golden tests stay stable.
//! - [`particles::ParticlePool`] — fixed-size pool of short-lived
//!   coloured dots with Q4.4 sub-pixel velocity, gravity, TTL-
//!   based fade + size taper. Emits `RectFlat`s into an
//!   [`psx_gpu::ot::OrderingTable`] at the caller's chosen depth.
//! - [`shake::ShakeState`] — frame-decay counter that produces a
//!   triangle-wave `(dx, dy)` offset; caller folds it into vertex
//!   positions when building the frame.
//!
//! The particle pool is generic over size (const generic), so each
//! game picks a capacity that fits its budget — 32 for a casual
//! puzzler, 64 for a shoot-em-up, 256 for a bullet-hell.
//!
//! ## What's NOT here (on purpose)
//!
//! - Emitter objects, keyframed colour / size ramps, layer
//!   compositing, collision response — all engine-level. Those
//!   land when a future `psx-engine` crate needs them.
//! - Floats — the PS1 has no FPU and we stay honest about that.
//!   All math is fixed-point / integer.
//! - Textures for particles — particles are rasterised rects
//!   today; a textured variant is a small addition if someone
//!   needs sparkles later.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod particles;
pub mod rng;
pub mod shake;

pub use particles::{Particle, ParticlePool};
pub use rng::LcgRng;
pub use shake::ShakeState;
