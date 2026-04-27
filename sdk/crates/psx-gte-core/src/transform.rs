//! Composable 3D transform math operating on [`Mat3I16`] / [`Vec3I16`].
//!
//! Lives next to the type definitions because the inherent `impl`s
//! belong in the same crate as the type. Higher-level GTE register
//! access (`mtc2!` / `ctc2!` macros) lives in `psx-gte`, which
//! re-exports this module so callers can keep importing it from there.
//!
//! The PS1 has no hardware sin/cos — we ship a 65-entry quarter-sine
//! lookup and derive the full range by reflection. Angles are in
//! **256-per-revolution** units (`u16` with implicit wrap at 256)
//! rather than radians or degrees, matching the convention most PSX
//! homebrew uses. A full turn fits in a single byte, and the
//! quarter-table lookup is one mask away.
//!
//! Fixed-point conventions (inherited from the GTE):
//! - Matrix cells: **1.3.12** signed — `0x1000` = 1.0.
//! - Translation / screen offset / H: integers in engine-chosen
//!   scaling; the math here just passes them through.
//! - Vectors: **1.3.12** `i16`; accumulator outputs are `i32` in MAC
//!   or `i16` in IR post-saturation.

use crate::math::{Mat3I16, Vec3I16};

/// Quarter-sine table — 65 entries (indices 0..=64) covering angles
/// `0..=π/2` in 1.3.12. Full-range [`sin_1_3_12`] / [`cos_1_3_12`] use
/// this plus quadrant reflection. Values were computed as
/// `round(sin(i * π / 128) * 4096)`.
#[rustfmt::skip]
static QUARTER_SIN: [i16; 65] = [
    0,    101,  201,  301,  401,  501,  601,  700,
    799,  898,  995,  1092, 1189, 1285, 1380, 1474,
    1567, 1660, 1751, 1841, 1930, 2019, 2106, 2191,
    2276, 2359, 2440, 2520, 2598, 2675, 2751, 2824,
    2896, 2967, 3035, 3102, 3166, 3229, 3290, 3349,
    3406, 3461, 3513, 3564, 3612, 3659, 3702, 3744,
    3784, 3821, 3856, 3889, 3919, 3948, 3973, 3996,
    4017, 4036, 4051, 4065, 4076, 4085, 4091, 4094,
    4096,
];

/// Sine of `angle` in 256-per-revolution units, returned in 1.3.12.
///
/// Angles wrap modulo 256. Values are piecewise-linear approximations
/// of `sin` — accurate to within ±1 LSB of the 1.3.12 representation,
/// which is good enough for real-time 3D but not for scientific use.
#[inline]
pub const fn sin_1_3_12(angle: u16) -> i16 {
    let i = (angle & 0xFF) as usize;
    // Four quadrants: Q0 [0..=63] rising 0→~1, Q1 [64..=127] falling
    // ~1→0, Q2 [128..=191] falling 0→~-1, Q3 [192..=255] rising
    // ~-1→0. Endpoint values (64, 128, 192) are handled by the
    // quarter table's 65th entry.
    if i <= 64 {
        QUARTER_SIN[i]
    } else if i <= 128 {
        QUARTER_SIN[128 - i]
    } else if i <= 192 {
        -QUARTER_SIN[i - 128]
    } else {
        -QUARTER_SIN[256 - i]
    }
}

/// Cosine of `angle` in 256-per-revolution units, returned in 1.3.12.
#[inline]
pub const fn cos_1_3_12(angle: u16) -> i16 {
    // cos(θ) = sin(θ + π/2). Our angle units put π/2 at 64.
    sin_1_3_12(angle.wrapping_add(64))
}

impl Mat3I16 {
    /// Matrix × matrix, with 1.3.12 fixed-point scaling applied so the
    /// output stays in 1.3.12. Saturates each cell to `i16` range —
    /// rotation compositions stay well within, but callers doing
    /// scale-by-100 before compose should watch for truncation.
    pub fn mul(&self, other: &Self) -> Self {
        let mut out = [[0i16; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let column = [other.m[0][j], other.m[1][j], other.m[2][j]];
                out[i][j] = clamp_i32_to_i16(dot_q12_i16(self.m[i], column));
            }
        }
        Self { m: out }
    }

    /// Matrix × vector, with 1.3.12 scaling. Returns `i32` per
    /// component so the caller can inspect the un-saturated product
    /// (useful for normal generation / light calculations before
    /// feeding back into the GTE as V0).
    pub fn transform(&self, v: Vec3I16) -> [i32; 3] {
        let mut out = [0i32; 3];
        for i in 0..3 {
            out[i] = dot_q12_i16(self.m[i], [v.x, v.y, v.z]);
        }
        out
    }

    /// Rotation around the X axis by `angle` (256-per-revolution units).
    ///
    /// ```text
    ///   | 1    0     0  |
    ///   | 0   cos  -sin |
    ///   | 0   sin   cos |
    /// ```
    pub const fn rotate_x(angle: u16) -> Self {
        let c = cos_1_3_12(angle);
        let s = sin_1_3_12(angle);
        Self {
            m: [[0x1000, 0, 0], [0, c, -s], [0, s, c]],
        }
    }

    /// Rotation around the Y axis.
    ///
    /// ```text
    ///   |  cos  0  sin |
    ///   |   0   1   0  |
    ///   | -sin  0  cos |
    /// ```
    pub const fn rotate_y(angle: u16) -> Self {
        let c = cos_1_3_12(angle);
        let s = sin_1_3_12(angle);
        Self {
            m: [[c, 0, s], [0, 0x1000, 0], [-s, 0, c]],
        }
    }

    /// Rotation around the Z axis.
    ///
    /// ```text
    ///   | cos  -sin  0 |
    ///   | sin   cos  0 |
    ///   |  0     0   1 |
    /// ```
    pub const fn rotate_z(angle: u16) -> Self {
        let c = cos_1_3_12(angle);
        let s = sin_1_3_12(angle);
        Self {
            m: [[c, -s, 0], [s, c, 0], [0, 0, 0x1000]],
        }
    }
}

#[inline]
fn dot_q12_i16(a: [i16; 3], b: [i16; 3]) -> i32 {
    let mut sum = 0i32;
    let mut i = 0;
    while i < 3 {
        sum = sum.saturating_add((a[i] as i32) * (b[i] as i32));
        i += 1;
    }
    sum >> 12
}

#[inline]
fn clamp_i32_to_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_cos_identity_at_cardinal_angles() {
        // Exact values at 0, π/2, π, 3π/2.
        assert_eq!(sin_1_3_12(0), 0);
        assert_eq!(cos_1_3_12(0), 0x1000);
        assert_eq!(sin_1_3_12(64), 0x1000);
        assert_eq!(cos_1_3_12(64), 0);
        assert_eq!(sin_1_3_12(128), 0);
        assert_eq!(cos_1_3_12(128), -0x1000);
        assert_eq!(sin_1_3_12(192), -0x1000);
        assert_eq!(cos_1_3_12(192), 0);
    }

    #[test]
    fn sin_wraps_modulo_256() {
        // Angles outside 0..256 must wrap.
        assert_eq!(sin_1_3_12(256), sin_1_3_12(0));
        assert_eq!(sin_1_3_12(320), sin_1_3_12(64));
        assert_eq!(sin_1_3_12(65535), sin_1_3_12(255));
    }

    #[test]
    fn sin_cos_pythagorean_identity_holds_approximately() {
        // sin²+cos² = 1.0² = 0x1000² = 0x100_0000.
        // Our values are approximate so allow a small tolerance.
        for angle in (0..256).step_by(7) {
            let s = sin_1_3_12(angle as u16) as i32;
            let c = cos_1_3_12(angle as u16) as i32;
            let sum = s * s + c * c;
            let expected = 0x0100_0000;
            // ±0.5% tolerance — quarter-sine table rounds each entry.
            let tolerance = expected / 200;
            assert!(
                (sum - expected).abs() <= tolerance,
                "angle {angle}: sin²+cos² = {sum:#x}, expected ≈ {expected:#x}"
            );
        }
    }

    #[test]
    fn identity_mul_identity_is_identity() {
        let id = Mat3I16::IDENTITY;
        let prod = id.mul(&id);
        assert_eq!(prod, id);
    }

    #[test]
    fn rotate_y_zero_is_identity() {
        assert_eq!(Mat3I16::rotate_y(0), Mat3I16::IDENTITY);
    }

    #[test]
    fn rotate_y_quarter_turn_swaps_axes() {
        // 90° Y rotation: (1,0,0) → (0,0,-1), (0,0,1) → (1,0,0).
        let m = Mat3I16::rotate_y(64);
        assert_eq!(m.m[0][0], 0);
        assert_eq!(m.m[0][2], 0x1000);
        assert_eq!(m.m[2][0], -0x1000);
        assert_eq!(m.m[2][2], 0);
        // Y axis untouched.
        assert_eq!(m.m[1][1], 0x1000);
    }

    #[test]
    fn rotate_x_then_inverse_round_trips() {
        // Rotating by θ then by -θ (256-θ in our units) should yield
        // the identity within rounding error.
        let r = Mat3I16::rotate_x(30);
        let r_inv = Mat3I16::rotate_x(226);
        let prod = r.mul(&r_inv);
        // Diagonal should be ≈ 0x1000, off-diagonal ≈ 0.
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 0x1000 } else { 0 };
                let diff = (prod.m[i][j] as i32 - expected).abs();
                assert!(
                    diff < 8,
                    "rotate_x ∘ rotate_x⁻¹ cell ({i},{j}) = {:#x}, expected ≈ {expected:#x}",
                    prod.m[i][j]
                );
            }
        }
    }

    #[test]
    fn transform_identity_leaves_vector_unchanged() {
        let v = Vec3I16::new(100, -200, 300);
        let r = Mat3I16::IDENTITY.transform(v);
        assert_eq!(r, [100, -200, 300]);
    }

    #[test]
    fn transform_rotate_y_quarter_turn() {
        // (1, 0, 0) after 90° Y = (0, 0, -1) in our convention.
        let v = Vec3I16::new(0x1000, 0, 0);
        let r = Mat3I16::rotate_y(64).transform(v);
        assert_eq!(r, [0, 0, -0x1000]);
    }
}
