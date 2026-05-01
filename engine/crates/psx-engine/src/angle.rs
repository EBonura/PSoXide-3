//! Canonical angle type.
//!
//! # Why a newtype
//!
//! The SDK historically exposed angles in two different units, with
//! no compile-time distinction between them:
//!
//! - [`psx_gte::math::Mat3I16::rotate_y`] takes a `u16` in
//!   **256-per-revolution** units (the sin/cos LUT masks `& 0xFF`
//!   internally). A full turn is `256`; `0x4000` is *not* a quarter
//!   turn, it's 64 full turns aliased to 0.
//!
//! - [`psx_math::sincos::sin_q12`] takes a `u16` in **Q0.12**
//!   (4096-per-revolution). A full turn is `0x1000`.
//!
//! Both are `u16`. Passing one to the other compiles cleanly and
//! produces catastrophic flickering -- exactly the bug that sank
//! an early showcase-fog iteration. The fix there was an
//! after-the-fact audit; the fix *here* is to refuse to let
//! `u16`-shaped code hand angles to the SDK directly.
//!
//! # Canonical unit: Q0.16
//!
//! `Angle` wraps a `u16` representing fractions of a revolution
//! at Q0.16 precision: a full turn is `0x10000` (which wraps to
//! `0x0000`). This is the finest of the SDK's three flavours; it
//! converts down to 256-per-rev and Q0.12 by a simple right
//! shift, losing the low bits the coarser unit wouldn't have
//! stored anyway.
//!
//! # Use cases
//!
//! - **Constants**: [`Angle::from_degrees`] / [`Angle::from_turns_q16`]
//!   build compile-time values. [`Angle::ZERO`], [`Angle::QUARTER`],
//!   [`Angle::HALF`] cover the common ones.
//! - **Per-frame orbits**: [`Angle::per_frames`] computes the
//!   per-frame delta so a full rotation takes N frames. Typical
//!   use is `angle = Angle::per_frames(256).mul_frame(s.frame)`
//!   for a once-every-256-frames rotation.
//! - **Feeding the SDK**: [`Angle::rotate_y_arg`] gives the u16
//!   `rotate_y` expects; [`Angle::sin_q12_arg`] gives the u16
//!   `sin_q12` expects. Converting explicitly at the call site
//!   makes the unit hop visible in reviews.

use crate::fixed::Q12;

use psx_math::{cos_q12, sin_q12};

/// Fixed-point angle, Q0.16 -- one revolution = `0x10000`, which
/// wraps to `0x0000`. See [module docs](crate::angle) for the
/// rationale.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct Angle(u16);

impl Angle {
    /// Zero radians / degrees / turns. Also equivalent to one full
    /// revolution (it wraps).
    pub const ZERO: Angle = Angle(0);
    /// A quarter turn (90° / π/2 rad / 0.25 revolutions).
    pub const QUARTER: Angle = Angle(0x4000);
    /// A half turn (180°).
    pub const HALF: Angle = Angle(0x8000);
    /// Three-quarter turn (270°).
    pub const THREE_QUARTER: Angle = Angle(0xC000);

    /// Raw Q0.16 value constructor. Use [`Angle::from_degrees`] or
    /// [`Angle::from_turns_q16`] for readable call sites.
    pub const fn from_raw_q16(q16: u16) -> Angle {
        Angle(q16)
    }

    /// Build from a Q0.16 value -- alias of [`Angle::from_raw_q16`]
    /// that reads nicely when you already think in turns:
    /// `Angle::from_turns_q16(0x4000)` = quarter turn.
    pub const fn from_turns_q16(q16: u16) -> Angle {
        Angle(q16)
    }

    /// Build from a Q0.12 value (`4096` units per revolution), the
    /// unit consumed by [`psx_math::sincos::sin_q12`] /
    /// [`psx_math::sincos::cos_q12`].
    pub const fn from_q12(q12: u16) -> Angle {
        Angle((q12 & 0x0FFF) << 4)
    }

    /// Build from the 256-per-revolution unit consumed by
    /// [`psx_gte::math::Mat3I16::rotate_y`] and friends.
    pub const fn from_rotate_y_arg(arg: u16) -> Angle {
        Angle((arg & 0x00FF) << 8)
    }

    /// Build from an integer-degree value, `0..360`. Inputs ≥ 360
    /// wrap naturally (the Q0.16 representation also wraps, so
    /// `from_degrees(720)` = `ZERO`).
    pub const fn from_degrees(deg: u32) -> Angle {
        // A full turn is `0x10000` units, 360°; one degree =
        // `0x10000 / 360 ≈ 182.04` units. Use multiply-before-divide
        // so small angles round toward the intended unit count
        // rather than to zero.
        let q16 = (deg.wrapping_mul(0x1_0000) / 360) as u16;
        Angle(q16)
    }

    /// Per-frame delta for a rotation that completes in exactly
    /// `frames` frames. `frames = 60` at 60 fps = one turn per
    /// second; `frames = 256` = about four seconds.
    ///
    /// Returns [`Angle::ZERO`] if `frames == 0` -- no divide by zero,
    /// no rotation.
    pub const fn per_frames(frames: u32) -> Angle {
        if frames == 0 {
            return Angle(0);
        }
        Angle((0x1_0000 / frames) as u16)
    }

    /// The inner Q0.16 value. Mostly useful for serialising / testing;
    /// day-to-day code stays inside the type.
    pub const fn as_q16(self) -> u16 {
        self.0
    }

    /// Add another angle, wrapping on overflow (every operation on
    /// `Angle` wraps -- that's the whole point of a normalised
    /// representation).
    pub const fn add(self, other: Angle) -> Angle {
        Angle(self.0.wrapping_add(other.0))
    }

    /// Subtract another angle, wrapping.
    pub const fn sub(self, other: Angle) -> Angle {
        Angle(self.0.wrapping_sub(other.0))
    }

    /// Add a signed Q0.12 delta, wrapping at one revolution.
    pub const fn add_signed_q12(self, delta_q12: i16) -> Angle {
        Angle(((self.0 as i32 + ((delta_q12 as i32) << 4)) & 0xFFFF) as u16)
    }

    /// Multiply this per-frame delta by an integer frame count.
    /// Typical use: `Angle::per_frames(256).mul_frame(s.frame)` to
    /// get the current rotation angle from a monotonic frame
    /// counter.
    pub const fn mul_frame(self, frame: u32) -> Angle {
        Angle((self.0 as u32).wrapping_mul(frame) as u16)
    }

    /// Convert to the u16 that [`psx_gte::math::Mat3I16::rotate_y`]
    /// (and `rotate_x` / `rotate_z`) consume. Drops the low 8 bits
    /// of the Q0.16 value to land on 256-per-revolution units, which
    /// is all the sin/cos LUT uses anyway.
    ///
    /// Calling this is the only *correct* way to feed an `Angle` to
    /// a rotation-matrix constructor -- `angle.as_q16() as u16` looks
    /// like it should work and will produce chaotic flips instead.
    pub const fn rotate_y_arg(self) -> u16 {
        self.0 >> 8
    }

    /// Convert to the u16 that [`psx_math::sincos::sin_q12`] and
    /// [`psx_math::sincos::cos_q12`] consume. Drops the low 4 bits
    /// of the Q0.16 value to land on Q0.12
    /// (4096-per-revolution).
    pub const fn sin_q12_arg(self) -> u16 {
        (self.0 >> 4) & 0xFFF
    }

    /// Alias for [`Angle::sin_q12_arg`] when the caller is not
    /// feeding a trig function directly.
    pub const fn as_q12(self) -> u16 {
        self.sin_q12_arg()
    }

    /// Sine in Q12 fixed point.
    pub fn sin_q12(self) -> i32 {
        sin_q12(self.sin_q12_arg())
    }

    /// Sine as a typed Q12 scalar.
    pub fn sin(self) -> Q12 {
        Q12::from_raw(self.sin_q12())
    }

    /// Cosine in Q12 fixed point.
    pub fn cos_q12(self) -> i32 {
        cos_q12(self.sin_q12_arg())
    }

    /// Cosine as a typed Q12 scalar.
    pub fn cos(self) -> Q12 {
        Q12::from_raw(self.cos_q12())
    }

    /// Shortest signed delta to `target`, expressed in Q0.12 angle
    /// units. Positive means turn forward, negative means turn
    /// backward.
    pub const fn shortest_delta_q12(self, target: Angle) -> i16 {
        let mut delta = ((target.sin_q12_arg() as i32 - self.sin_q12_arg() as i32) & 0x0FFF) as i16;
        if delta > 2048 {
            delta -= 4096;
        }
        delta
    }

    /// Step from `self` toward `target` by at most `step_q12`
    /// Q0.12 units, taking the shortest wrapping path.
    pub const fn approach_q12(self, target: Angle, step_q12: u16) -> Angle {
        let delta = self.shortest_delta_q12(target);
        let step = step_q12 as i16;
        if abs_i16_const(delta) <= step {
            target
        } else if delta > 0 {
            self.add_signed_q12(step)
        } else {
            self.add_signed_q12(-step)
        }
    }
}

const fn abs_i16_const(value: i16) -> i16 {
    if value == i16::MIN {
        i16::MAX
    } else if value < 0 {
        -value
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(Angle::ZERO.as_q16(), 0);
        assert_eq!(Angle::QUARTER.as_q16(), 0x4000);
        assert_eq!(Angle::HALF.as_q16(), 0x8000);
        assert_eq!(Angle::THREE_QUARTER.as_q16(), 0xC000);
    }

    #[test]
    fn from_degrees_key_values() {
        assert_eq!(Angle::from_degrees(0), Angle::ZERO);
        assert_eq!(Angle::from_degrees(90), Angle::QUARTER);
        assert_eq!(Angle::from_degrees(180), Angle::HALF);
        assert_eq!(Angle::from_degrees(270), Angle::THREE_QUARTER);
        // 360 wraps to 0.
        assert_eq!(Angle::from_degrees(360), Angle::ZERO);
        assert_eq!(Angle::from_degrees(720), Angle::ZERO);
    }

    #[test]
    fn per_frames_round_trip() {
        // 256-frame rotation → per-frame delta × 256 = full turn (= 0).
        let d = Angle::per_frames(256);
        assert_eq!(d.mul_frame(256), Angle::ZERO);
        // Halfway → half turn.
        assert_eq!(d.mul_frame(128), Angle::HALF);
        // Quarter-way → quarter turn.
        assert_eq!(d.mul_frame(64), Angle::QUARTER);
    }

    #[test]
    fn per_frames_zero_is_safe() {
        assert_eq!(Angle::per_frames(0), Angle::ZERO);
    }

    #[test]
    fn rotate_y_arg_strips_low_byte() {
        // Q0.16 = 0x4000 (quarter turn) → rotate_y wants 256-per-rev.
        // 0x4000 >> 8 = 0x40 = quarter of 256.
        assert_eq!(Angle::QUARTER.rotate_y_arg(), 0x40);
        // Q0.16 = 0x0080 (tiny angle, below 256-per-rev granularity)
        // → rotate_y arg rounds to zero.
        assert_eq!(Angle::from_raw_q16(0x0080).rotate_y_arg(), 0);
    }

    #[test]
    fn sin_q12_arg_converts_to_q0_12() {
        // Q0.16 = 0x4000 → Q0.12 = 0x400 (quarter turn in 4096-per-rev).
        assert_eq!(Angle::QUARTER.sin_q12_arg(), 0x400);
        // Q0.16 = 0x8000 → Q0.12 = 0x800.
        assert_eq!(Angle::HALF.sin_q12_arg(), 0x800);
    }

    #[test]
    fn q12_constructor_round_trips() {
        assert_eq!(Angle::from_q12(0).sin_q12_arg(), 0);
        assert_eq!(Angle::from_q12(1024), Angle::QUARTER);
        assert_eq!(Angle::from_q12(4096), Angle::ZERO);
        assert_eq!(Angle::from_q12(5120), Angle::QUARTER);
    }

    #[test]
    fn signed_q12_add_wraps() {
        assert_eq!(Angle::from_q12(4090).add_signed_q12(8), Angle::from_q12(2));
        assert_eq!(Angle::from_q12(4).add_signed_q12(-8), Angle::from_q12(4092));
    }

    #[test]
    fn approach_q12_takes_shortest_wrapping_path() {
        assert_eq!(
            Angle::from_q12(4090).approach_q12(Angle::from_q12(8), 16),
            Angle::from_q12(8)
        );
        assert_eq!(
            Angle::from_q12(20).approach_q12(Angle::from_q12(4000), 16),
            Angle::from_q12(4)
        );
    }

    #[test]
    fn wrapping_semantics() {
        let a = Angle::from_raw_q16(0xF000);
        let b = Angle::from_raw_q16(0x2000);
        // 0xF000 + 0x2000 = 0x11000 → wraps to 0x1000.
        assert_eq!(a.add(b).as_q16(), 0x1000);
        // 0xF000 - 0x1000 = 0xE000, no wrap.
        assert_eq!(a.sub(Angle::from_raw_q16(0x1000)).as_q16(), 0xE000);
        // 0x2000 - 0xF000 wraps under zero.
        assert_eq!(b.sub(a).as_q16(), 0x3000);
    }
}
