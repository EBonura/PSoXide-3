//! Deterministic linear-congruential PRNG.
//!
//! Not cryptographically interesting — perfect for sprinkling
//! variability across particle velocity / enemy-shot cadence /
//! any "looks random but must replay identically" effect. The
//! constants match the venerable `glibc` LCG, which has
//! good-enough statistical properties for game use and produces
//! the same output on every PS1 / host / emulator.

/// 32-bit integer LCG. Step once per `next()` / `signed()` call.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct LcgRng(u32);

impl LcgRng {
    /// Build with an explicit seed. Same seed → same sequence.
    pub const fn new(seed: u32) -> Self {
        Self(seed)
    }

    /// One LCG step. Multiplier + increment are `glibc`'s constants.
    /// Returns the fresh internal state.
    #[inline]
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12345);
        self.0
    }

    /// Signed integer in roughly `[-range, +range]`, sourced from
    /// five bits of the LCG. Bias is ≤ 1 unit at the extremes,
    /// good enough for cosmetic particle spread.
    #[inline]
    pub fn signed(&mut self, range: i16) -> i16 {
        let r = self.next();
        let raw = ((r >> 16) & 0x1F) as i16; // 0..=31
        (raw - 16) * range / 16
    }

    /// Current internal state — useful if a caller wants to save /
    /// restore the RNG across reset boundaries.
    pub const fn state(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = LcgRng::new(0xC0DE_F00D);
        let mut b = LcgRng::new(0xC0DE_F00D);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn signed_stays_in_range() {
        let mut rng = LcgRng::new(0xBEEF_0042);
        for _ in 0..10_000 {
            let v = rng.signed(40);
            assert!(v >= -42 && v <= 40, "out of range: {v}");
        }
    }

    #[test]
    fn signed_zero_range_is_zero() {
        let mut rng = LcgRng::new(1);
        for _ in 0..100 {
            assert_eq!(rng.signed(0), 0);
        }
    }
}
