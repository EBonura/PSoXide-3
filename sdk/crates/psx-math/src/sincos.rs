//! Compact sin/cos lookup for Q0.12 angles.
//!
//! A Q0.12 angle represents one full revolution in a 12-bit
//! unsigned: `0..4096` maps linearly onto `0°..360°`. This
//! matches the PSX GTE's `ONE` / `4096` angle precision. The lookup
//! table stays compact at 256 entries, and `sin_q12` linearly
//! interpolates between entries so every Q0.12 angle step produces a
//! distinct result when the table slope permits it.
//!
//! Values are Q1.12 fixed-point: `4096` = 1.0, `-4096` = -1.0.
//!
//! # Example
//!
//! ```ignore
//! // Rotate point (x, y) by `angle` around origin.
//! let s = sin_q12(angle);
//! let c = cos_q12(angle);
//! let xr = (x * c - y * s) >> 12;
//! let yr = (x * s + y * c) >> 12;
//! ```
//!
//! Integer multiplication before the shift keeps the arithmetic
//! inside i32; `>> 12` drops the Q1.12 fractional bits after the
//! multiplies combine.

/// 256-entry sine table, Q1.12 fixed-point (one revolution).
///
/// Index = `(angle_q12 >> 4) & 0xFF`, where `angle_q12` is a u16
/// in `[0, 4096)` mapping linearly to `[0°, 360°)`. `sin_q12`
/// interpolates using the low 4 angle bits. Values are i16 in
/// `[-4096, 4096]`.
pub const SIN_TABLE: [i16; 256] = [
        0,   101,   201,   301,   401,   501,   601,   700,
      799,   897,   995,  1092,  1189,  1285,  1380,  1474,
     1567,  1660,  1751,  1842,  1931,  2019,  2106,  2191,
     2276,  2359,  2440,  2520,  2598,  2675,  2751,  2824,
     2896,  2967,  3035,  3102,  3166,  3229,  3290,  3349,
     3406,  3461,  3513,  3564,  3612,  3659,  3703,  3745,
     3784,  3822,  3857,  3889,  3920,  3948,  3973,  3996,
     4017,  4036,  4052,  4065,  4076,  4085,  4091,  4095,
     4096,  4095,  4091,  4085,  4076,  4065,  4052,  4036,
     4017,  3996,  3973,  3948,  3920,  3889,  3857,  3822,
     3784,  3745,  3703,  3659,  3612,  3564,  3513,  3461,
     3406,  3349,  3290,  3229,  3166,  3102,  3035,  2967,
     2896,  2824,  2751,  2675,  2598,  2520,  2440,  2359,
     2276,  2191,  2106,  2019,  1931,  1842,  1751,  1660,
     1567,  1474,  1380,  1285,  1189,  1092,   995,   897,
      799,   700,   601,   501,   401,   301,   201,   101,
        0,  -101,  -201,  -301,  -401,  -501,  -601,  -700,
     -799,  -897,  -995, -1092, -1189, -1285, -1380, -1474,
    -1567, -1660, -1751, -1842, -1931, -2019, -2106, -2191,
    -2276, -2359, -2440, -2520, -2598, -2675, -2751, -2824,
    -2896, -2967, -3035, -3102, -3166, -3229, -3290, -3349,
    -3406, -3461, -3513, -3564, -3612, -3659, -3703, -3745,
    -3784, -3822, -3857, -3889, -3920, -3948, -3973, -3996,
    -4017, -4036, -4052, -4065, -4076, -4085, -4091, -4095,
    -4096, -4095, -4091, -4085, -4076, -4065, -4052, -4036,
    -4017, -3996, -3973, -3948, -3920, -3889, -3857, -3822,
    -3784, -3745, -3703, -3659, -3612, -3564, -3513, -3461,
    -3406, -3349, -3290, -3229, -3166, -3102, -3035, -2967,
    -2896, -2824, -2751, -2675, -2598, -2520, -2440, -2359,
    -2276, -2191, -2106, -2019, -1931, -1842, -1751, -1660,
    -1567, -1474, -1380, -1285, -1189, -1092,  -995,  -897,
     -799,  -700,  -601,  -501,  -401,  -301,  -201,  -101,
];

/// `sin(angle_q12)` as a Q1.12 value. Input wraps modulo 4096.
#[inline]
pub fn sin_q12(angle_q12: u16) -> i32 {
    let angle = angle_q12 & 0x0FFF;
    let idx = (angle >> 4) as usize;
    let frac = (angle & 0x000F) as i32;
    let a = SIN_TABLE[idx] as i32;
    let b = SIN_TABLE[(idx + 1) & 0xFF] as i32;
    a + (((b - a) * frac) / 16)
}

/// `cos(angle_q12)` as a Q1.12 value, implemented as
/// `sin(angle + 90°)`. Input wraps modulo 4096.
#[inline]
pub fn cos_q12(angle_q12: u16) -> i32 {
    // 90° = 1024 in Q0.12 (4096 / 4). Wrap-add is automatic in u16.
    sin_q12(angle_q12.wrapping_add(1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_anchors() {
        assert_eq!(sin_q12(0), 0);
        assert_eq!(sin_q12(1024), 4096); // 90°
        assert_eq!(sin_q12(2048), 0); //    180°
        assert_eq!(sin_q12(3072), -4096); // 270°
    }

    #[test]
    fn cos_anchors() {
        assert_eq!(cos_q12(0), 4096);
        assert_eq!(cos_q12(1024), 0);
        assert_eq!(cos_q12(2048), -4096);
        assert_eq!(cos_q12(3072), 0);
    }

    #[test]
    fn wraps_modulo_4096() {
        // 4096 is one full revolution; the u16 wrap makes it 0.
        assert_eq!(sin_q12(4096), sin_q12(0));
        assert_eq!(sin_q12(5120), sin_q12(1024));
    }

    #[test]
    fn interpolates_low_angle_bits() {
        assert_eq!(sin_q12(0), SIN_TABLE[0] as i32);
        assert!(sin_q12(1) > sin_q12(0));
        assert!(sin_q12(1) < SIN_TABLE[1] as i32);
        assert_eq!(sin_q12(16), SIN_TABLE[1] as i32);
    }

    #[test]
    fn interpolates_across_table_wrap() {
        assert!(sin_q12(4095) < 0);
        assert!(sin_q12(4095) > SIN_TABLE[255] as i32);
        assert_eq!(sin_q12(4096), 0);
    }
}
