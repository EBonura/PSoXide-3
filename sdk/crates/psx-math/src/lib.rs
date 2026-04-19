//! Fixed-point math primitives for the PS1 SDK.
//!
//! The PS1 has no FPU. Games that want sine-driven animation,
//! camera rotation, pseudo-3D, or smoothed HUDs all reach for the
//! same toolbox: a Q-format fixed-point type, a sin/cos LUT, and
//! a couple of vector/matrix helpers. This crate owns those
//! primitives once so every downstream crate (`psx-font`, a
//! future `psx-gte`-wrapper, a future `psx-ui`, gameplay code)
//! shares the same representation.
//!
//! ## What's here today
//!
//! - [`sincos`] — Q0.12 angle type + 256-entry Q1.12 sine LUT +
//!   `sin_q12` / `cos_q12` lookups. ~1.4° angular resolution.
//!
//! ## What's planned (same crate, future modules)
//!
//! - `fixed` — Q3.12 and Q16.16 wrappers with operator overloads
//!   and saturating / wrapping arithmetic.
//! - `vec` — `Vec2` / `Vec3` over integer and fixed-point types
//!   with dot / cross / length² / normalise.
//! - `mat` — 2×2 / 3×3 / GTE-shape matrices, chosen to match how
//!   the GTE's `RTPS` / `MVMVA` instructions consume data.
//! - `ease` — sinusoidal / quadratic / cubic easing helpers for
//!   timeline animation.
//! - `lerp` — generic linear interpolation.
//!
//! These modules land as concrete consumer code needs them — no
//! premature abstractions. The crate ships empty apart from the
//! documented scope so reviewers can see where new primitives
//! should go.
//!
//! ## Q-format refresher
//!
//! `Qm.n` means `m` integer bits + `n` fractional bits in a
//! signed integer. Arithmetic rules:
//!
//! - Add / sub: same as ordinary integer add / sub.
//! - Multiply: `(a * b) >> n` — two Q_._n values produce a Q_._2n
//!   intermediate that must be shifted back.
//! - Divide: `(a << n) / b` — scale up the numerator by `n` bits
//!   first so the quotient is in Qm.n.
//!
//! Conventions the PS1 SDK uses:
//!
//! - **Q0.12 angle**: `u16` in `[0, 4096)` maps to `[0°, 360°)`.
//!   Matches the GTE's angle resolution. `4096` = one full
//!   revolution. Overflow wraps naturally in u16.
//! - **Q1.12 trig values**: `i16` in `[-4096, 4096]`. `4096` =
//!   +1.0, `-4096` = -1.0. Multiplying a Q_._12 input by a Q1.12
//!   coefficient produces a Q_._24 intermediate; shift right 12
//!   to land back in the original format.
//! - **Q3.12 transforms** (future): `i16` with range `±7.999…`.
//!   Used by `psx-font`'s affine matrices today; planned as a
//!   proper `Mat2` type here.
//! - **Q16.16 positions** (future): `i32` in `±32767` with 16
//!   fractional bits — standard precision for sub-pixel
//!   positions, accumulated velocities, etc.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod sincos;

// Re-export the most common symbols at the crate root so users
// can `use psx_math::{sin_q12, cos_q12}` without reaching through
// the submodule — these are the hot-loop primitives.
pub use sincos::{cos_q12, sin_q12, SIN_TABLE};
