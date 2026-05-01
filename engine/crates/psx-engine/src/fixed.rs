//! Small fixed-point scalar wrappers used by gameplay code.
//!
//! These types intentionally cover scalar values, not coordinates or
//! angles. Use [`crate::Angle`] for rotation and [`crate::RoomPoint`]
//! for room-local positions.

/// Signed Q20.12 scalar.
///
/// `Q12::ONE` is `1.0`, stored as raw `4096`. This is the natural
/// unit for GTE matrix cells, unit movement vectors, animation phase
/// fractions, and per-frame scalar speeds.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Q12(i32);

impl Q12 {
    /// Number of fractional bits.
    pub const FRACTIONAL_BITS: u32 = 12;
    /// Raw value representing `1.0`.
    pub const SCALE: i32 = 1 << Self::FRACTIONAL_BITS;
    /// Zero scalar.
    pub const ZERO: Self = Self(0);
    /// Scalar `0.5`.
    pub const HALF: Self = Self(Self::SCALE / 2);
    /// Scalar `1.0`.
    pub const ONE: Self = Self(Self::SCALE);
    /// Scalar `-1.0`.
    pub const NEG_ONE: Self = Self(-Self::SCALE);

    /// Build from raw Q12 storage.
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    /// Build from raw Q12 storage held in an `i16`.
    pub const fn from_raw_i16(raw: i16) -> Self {
        Self(raw as i32)
    }

    /// Build from a signed integer.
    pub const fn from_int(value: i32) -> Self {
        Self(value.saturating_mul(Self::SCALE))
    }

    /// Build `numerator / denominator` as Q12.
    ///
    /// A zero denominator returns [`Q12::ZERO`]. The operation stays
    /// in 32-bit arithmetic for predictable PS1 codegen.
    pub fn from_ratio(numerator: i32, denominator: i32) -> Self {
        if denominator == 0 {
            return Self::ZERO;
        }
        Self(numerator.saturating_mul(Self::SCALE) / denominator)
    }

    /// Return the raw Q12 storage.
    pub const fn raw(self) -> i32 {
        self.0
    }

    /// Return `true` when the scalar is exactly zero.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Saturating addition.
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Clamp between two Q12 bounds.
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }

    /// Multiply another Q12 scalar, returning Q12.
    pub fn mul_q12(self, rhs: Self) -> Self {
        Self(self.0.saturating_mul(rhs.0) >> Self::FRACTIONAL_BITS)
    }

    /// Multiply an integer by this Q12 scalar, returning an integer.
    pub fn mul_i32(self, value: i32) -> i32 {
        value.saturating_mul(self.0) >> Self::FRACTIONAL_BITS
    }

    /// Multiply by `numerator / denominator`, returning Q12.
    ///
    /// A zero denominator returns [`Q12::ZERO`].
    pub fn mul_ratio(self, numerator: i32, denominator: i32) -> Self {
        if denominator == 0 {
            return Self::ZERO;
        }
        Self(self.0.saturating_mul(numerator) / denominator)
    }

    /// Convert to raw `i16`, saturating at the destination limits.
    pub fn to_raw_i16_saturating(self) -> i16 {
        if self.0 > i16::MAX as i32 {
            i16::MAX
        } else if self.0 < i16::MIN as i32 {
            i16::MIN
        } else {
            self.0 as i16
        }
    }
}

/// Unsigned Q8.8 scalar.
///
/// `Q8::ONE` is `1.0`, stored as raw `256`. This is used for light
/// intensity and distance falloff, where values are naturally
/// non-negative and generated content stores them as `u16`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Q8(u16);

impl Q8 {
    /// Number of fractional bits.
    pub const FRACTIONAL_BITS: u32 = 8;
    /// Raw value representing `1.0`.
    pub const SCALE: u32 = 1 << Self::FRACTIONAL_BITS;
    /// Zero scalar.
    pub const ZERO: Self = Self(0);
    /// Scalar `0.5`.
    pub const HALF: Self = Self((Self::SCALE / 2) as u16);
    /// Scalar `1.0`.
    pub const ONE: Self = Self(Self::SCALE as u16);

    /// Build from raw Q8 storage, saturating to the stored range.
    pub const fn from_raw(raw: u32) -> Self {
        if raw > u16::MAX as u32 {
            Self(u16::MAX)
        } else {
            Self(raw as u16)
        }
    }

    /// Build from raw Q8 storage held in a `u16`.
    pub const fn from_raw_u16(raw: u16) -> Self {
        Self(raw)
    }

    /// Build from an unsigned integer.
    pub const fn from_int(value: u16) -> Self {
        Self(value.saturating_mul(Self::SCALE as u16))
    }

    /// Build `numerator / denominator` as Q8.
    ///
    /// A zero denominator returns [`Q8::ZERO`].
    pub fn from_ratio(numerator: u32, denominator: u32) -> Self {
        if denominator == 0 {
            return Self::ZERO;
        }
        Self::from_raw(numerator.saturating_mul(Self::SCALE) / denominator)
    }

    /// Return the raw Q8 storage as `u32` for arithmetic.
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }

    /// Return `true` when the scalar is exactly zero.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Multiply another Q8 scalar, returning Q8.
    pub fn mul_q8(self, rhs: Self) -> Self {
        Self::from_raw(self.raw().saturating_mul(rhs.raw()) >> Self::FRACTIONAL_BITS)
    }

    /// Multiply an integer by this Q8 scalar, returning an integer.
    pub fn mul_u32(self, value: u32) -> u32 {
        value.saturating_mul(self.raw()) >> Self::FRACTIONAL_BITS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q12_ratio_uses_4096_as_one() {
        assert_eq!(Q12::from_ratio(1, 2), Q12::HALF);
        assert_eq!(Q12::from_ratio(2, 1), Q12::from_raw(8192));
    }

    #[test]
    fn q12_mul_i32_keeps_integer_result() {
        assert_eq!(Q12::HALF.mul_i32(64), 32);
        assert_eq!(Q12::NEG_ONE.mul_i32(64), -64);
    }

    #[test]
    fn q12_mul_q12_preserves_fractional_scale() {
        assert_eq!(Q12::HALF.mul_q12(Q12::HALF), Q12::from_raw(1024));
    }

    #[test]
    fn q8_ratio_uses_256_as_one() {
        assert_eq!(Q8::from_ratio(1, 2), Q8::HALF);
        assert_eq!(Q8::from_ratio(1, 1), Q8::ONE);
    }

    #[test]
    fn q8_raw_input_saturates_to_generated_storage_range() {
        assert_eq!(Q8::from_raw(u32::MAX).raw(), u16::MAX as u32);
    }
}
