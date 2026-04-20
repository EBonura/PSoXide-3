//! Frame- and tick-counter newtypes.
//!
//! # Why a newtype
//!
//! Games reach for `ctx.frame` for a grab-bag of reasons: as a
//! timer (`frame & N` for periodic strobes), as a seed for
//! per-frame variation, as input to the engine's [`Angle`][crate::Angle]
//! math. Each call site wants a `u32` back, but *what the counter
//! measures* should be explicit — is this a visible frame, a
//! simulation step, an SPU sample?
//!
//! Two distinct counters show up at different scales:
//!
//! - **[`Frames`]** — visible frames since boot. Ticks once per
//!   end-of-loop in `App::run`. Wraps at `u32::MAX` (≈828 days at
//!   60 fps — not a real concern, but well-defined).
//!
//! - **[`Ticks`]** — SPU-audio-rate counter (44100 Hz). Wraps at
//!   `u32::MAX` = ~27 hours of uptime, comfortably past any
//!   session length. Staying `u32` matches the R3000A's native
//!   word size — a `u64` would force every arithmetic op to be
//!   synthesised as two 32-bit half-ops with carry, roughly 3-4×
//!   cost with zero practical benefit at PSX timescales.
//!
//! Games that need raw arithmetic call [`Frames::as_u32`] to break
//! the newtype at the call site. Games that just want "is this a
//! strobe frame?" can use [`Frames::bit`] which reads a single
//! bit without exposing the raw integer.
//!
//! # Semantics: when does `Frames` advance?
//!
//! `App::run` calls [`Scene::update`][crate::scene::Scene::update]
//! and [`Scene::render`][crate::scene::Scene::render] *then*
//! increments. So on the very first frame, both callbacks see
//! `Frames::ZERO`; on the second, `Frames(1)`. This is distinct
//! from the "increment at start of update" convention some pre-
//! engine games used (notably `game-invaders`) — their frame-0
//! was our frame-1. Net behaviour is identical apart from a
//! one-frame phase shift.

/// Monotonic visible-frame counter.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct Frames(u32);

impl Frames {
    /// The counter's zero point. Both `update` and `render` see
    /// this value on the first iteration of the main loop.
    pub const ZERO: Frames = Frames(0);

    /// Raw `u32` constructor — use for test fixtures or when
    /// bridging to an external frame count.
    pub const fn from_u32(n: u32) -> Frames {
        Frames(n)
    }

    /// Unwrap to the underlying `u32`. Use at call sites that
    /// need bare arithmetic — `frame.as_u32() % 40`, etc.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Increment by one, wrapping at `u32::MAX`.
    #[inline]
    pub const fn advance(self) -> Frames {
        Frames(self.0.wrapping_add(1))
    }

    /// Add an arbitrary `u32` delta, wrapping on overflow.
    pub const fn wrapping_add(self, n: u32) -> Frames {
        Frames(self.0.wrapping_add(n))
    }

    /// Read a specific bit of the underlying counter — useful for
    /// strobe effects (`frames.bit(1)` flips every frame).
    #[inline]
    pub const fn bit(self, index: u8) -> bool {
        (self.0 >> index) & 1 != 0
    }

    /// `true` when the counter is divisible by `n` — shorthand
    /// for "every `n`th frame" cadences. `Frames::ZERO` satisfies
    /// this for any `n ≥ 1`.
    #[inline]
    pub const fn every(self, n: u32) -> bool {
        if n == 0 { false } else { self.0 % n == 0 }
    }
}

/// SPU-audio-rate tick counter.
///
/// `u32` at 44100 Hz wraps after ~27 hours of uptime — comfortably
/// past any real session. The R3000A has 32-bit native registers;
/// a `u64` counter would force every arithmetic op to be
/// synthesised as an add-with-carry pair, 3-4× the instruction
/// cost, to gain coverage we can't exhaust anyway.
///
/// Reserved for engine subsystems that need finer time resolution
/// than per-frame (audio scheduling, profiling). No `Ctx` field yet.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct Ticks(u32);

impl Ticks {
    /// Counter origin.
    pub const ZERO: Ticks = Ticks(0);

    /// Wrap a raw tick count.
    pub const fn from_u32(n: u32) -> Ticks {
        Ticks(n)
    }

    /// Unwrap to `u32`.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Add a delta, wrapping.
    pub const fn wrapping_add(self, n: u32) -> Ticks {
        Ticks(self.0.wrapping_add(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_advance_wraps() {
        let f = Frames::from_u32(u32::MAX);
        assert_eq!(f.advance(), Frames::ZERO);
    }

    #[test]
    fn frames_bit_strobes() {
        assert!(!Frames::ZERO.bit(0));
        assert!(Frames::from_u32(1).bit(0));
        assert!(Frames::from_u32(2).bit(1));
        assert!(!Frames::from_u32(3).bit(2));
        assert!(Frames::from_u32(4).bit(2));
    }

    #[test]
    fn frames_every() {
        assert!(Frames::ZERO.every(40));
        assert!(!Frames::from_u32(1).every(40));
        assert!(Frames::from_u32(40).every(40));
        assert!(Frames::from_u32(80).every(40));
        // Zero denominator is a safe "never".
        assert!(!Frames::from_u32(5).every(0));
    }

    #[test]
    fn ticks_round_trip() {
        let t = Ticks::from_u32(0xCAFE_F00D);
        assert_eq!(t.as_u32(), 0xCAFE_F00D);
    }

    #[test]
    fn ticks_wraps_at_u32_max() {
        let t = Ticks::from_u32(u32::MAX);
        assert_eq!(t.wrapping_add(1), Ticks::ZERO);
    }
}
