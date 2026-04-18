//! Fixed-point math types for GTE inputs/outputs.
//!
//! The PS1 GTE works in two flavours of fixed point:
//!
//! - **1.3.12** for matrix entries, normals, small vectors — 16-bit
//!   signed, divide by 4096 for real value. Range: roughly ±8.0 with
//!   12 bits of fraction.
//! - **19.12 / 31.0** for translations, colour bias, far colour —
//!   32-bit signed stored raw.
//!
//! Types here are **POD** (`Copy`, `no_std`, no heap). The GTE register
//! wire format is always a packed `u32`, so the types mainly exist for
//! static correctness — the actual register I/O happens through the
//! [`mtc2!`][crate::mtc2!] / [`ctc2!`][crate::ctc2!] macros.

/// 3-component vector of `i16` in 1.3.12 fixed point.
/// Matches the GTE's V0/V1/V2 input slots and the IR accumulators.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Vec3I16 {
    /// X component.
    pub x: i16,
    /// Y component.
    pub y: i16,
    /// Z component.
    pub z: i16,
}

impl Vec3I16 {
    /// Zero vector.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    /// Construct from three components.
    #[inline(always)]
    pub const fn new(x: i16, y: i16, z: i16) -> Self {
        Self { x, y, z }
    }

    /// Pack the `(x, y)` pair into the 32-bit form expected by MTC2
    /// for V0/V1/V2 slots 0/2/4 — low half = `x`, high half = `y`.
    #[inline(always)]
    pub const fn xy_packed(self) -> u32 {
        ((self.y as u16 as u32) << 16) | (self.x as u16 as u32)
    }

    /// Sign-extended `z` for MTC2 of V0/V1/V2 slots 1/3/5.
    #[inline(always)]
    pub const fn z_packed(self) -> u32 {
        self.z as i32 as u32
    }
}

/// 3-component vector of `i32` — used for translation / colour bias
/// / far colour values (stored raw in 31.0 or 19.12 depending on which
/// register, see the GTE spec).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Vec3I32 {
    /// X / R component.
    pub x: i32,
    /// Y / G component.
    pub y: i32,
    /// Z / B component.
    pub z: i32,
}

impl Vec3I32 {
    /// Zero vector.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    /// Construct from three components.
    #[inline(always)]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// 3×3 rotation / light matrix of `i16` in 1.3.12. Laid out row-major
/// so `m[row][col]` matches PSX-SPX notation (`M_rc`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Mat3I16 {
    /// Row-major 3×3 cells.
    pub m: [[i16; 3]; 3],
}

impl Mat3I16 {
    /// Zero matrix.
    pub const ZERO: Self = Self { m: [[0; 3]; 3] };

    /// Identity matrix scaled 1.0 in 1.3.12 (diagonal = 0x1000).
    pub const IDENTITY: Self = Self {
        m: [
            [0x1000, 0, 0],
            [0, 0x1000, 0],
            [0, 0, 0x1000],
        ],
    };

    /// Access a row.
    #[inline(always)]
    pub const fn row(&self, i: usize) -> [i16; 3] {
        self.m[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_diagonal_is_one_in_1_3_12() {
        // 0x1000 = 4096 = 1.0 when divided by 4096.
        assert_eq!(Mat3I16::IDENTITY.m[0][0], 0x1000);
        assert_eq!(Mat3I16::IDENTITY.m[1][1], 0x1000);
        assert_eq!(Mat3I16::IDENTITY.m[2][2], 0x1000);
        // Off-diagonal zero.
        assert_eq!(Mat3I16::IDENTITY.m[0][1], 0);
        assert_eq!(Mat3I16::IDENTITY.m[1][2], 0);
    }

    #[test]
    fn vec3_i16_xy_packs_little_endian() {
        // x in low half, y in high half — matches MTC2 expectation.
        let v = Vec3I16::new(-1, 2, 3);
        assert_eq!(v.xy_packed(), 0x0002_FFFF);
    }

    #[test]
    fn vec3_i16_z_sign_extends() {
        let v = Vec3I16::new(0, 0, -1);
        assert_eq!(v.z_packed(), 0xFFFF_FFFF);
    }

    #[test]
    fn vec3_i32_construct_and_fields() {
        let t = Vec3I32::new(100, 200, 300);
        assert_eq!(t.x, 100);
        assert_eq!(t.y, 200);
        assert_eq!(t.z, 300);
    }

    #[test]
    fn zero_constants_are_zero() {
        assert_eq!(Vec3I16::ZERO.x, 0);
        assert_eq!(Vec3I16::ZERO.y, 0);
        assert_eq!(Vec3I16::ZERO.z, 0);
        assert_eq!(Vec3I32::ZERO.x, 0);
        assert_eq!(Mat3I16::ZERO.m[0][0], 0);
    }
}
