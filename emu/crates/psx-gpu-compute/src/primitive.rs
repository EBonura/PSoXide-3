//! Host-side primitive descriptions that the rasterizer dispatches.
//!
//! Layout is hand-tuned to match the WGSL `struct` definitions in
//! `shaders/*.wgsl` (16-byte alignment for storage buffers, fields
//! grouped into `vec*` chunks). When you add or reorder a field
//! here, update the matching WGSL struct in lockstep.

use bytemuck::{Pod, Zeroable};

/// Drawing-area clip rectangle that the rasterizer applies to every
/// primitive. Inclusive on all four sides — matches the CPU rasterizer
/// (`draw_area_left..=draw_area_right`, `draw_area_top..=draw_area_bottom`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DrawArea {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl DrawArea {
    pub fn full_vram() -> Self {
        Self {
            left: 0,
            top: 0,
            right: (super::VRAM_WIDTH - 1) as i32,
            bottom: (super::VRAM_HEIGHT - 1) as i32,
        }
    }
}

/// PSX semi-transparency mode — wire-compatible with the values the
/// CPU rasterizer's `BlendMode` enum decodes from GP0 0xE1 / tpage
/// bits 5-6. The shader expects the encoded `u32` form below; the
/// `as u32` cast on the enum gives the right value automatically.
///
/// `Opaque` is encoded by clearing the `SEMI_TRANS` flag in
/// `MonoTri::flags`; the `blend_mode` field is then ignored by
/// the shader.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// `(bg >> 1) + (fg >> 1)` per channel. Redux quirk: pre-shift
    /// both operands before summing, NOT `(bg + fg) / 2`. The two
    /// disagree when both inputs are odd: `(3+3)/2 = 3` vs `1+1 = 2`.
    Average = 0,
    /// `B + F`, channel-clamped to 31.
    Add = 1,
    /// `B - F`, channel-clamped to 0.
    Sub = 2,
    /// `B + F/4`, channel-clamped to 31. `F/4` is integer divide on
    /// the raw 5-bit channel, equivalent to `F >> 2`.
    AddQuarter = 3,
}

bitflags::bitflags! {
    /// Per-primitive flags packed into the WGSL `flags` field. The
    /// shader reads these bits to decide whether to do RMW. Bits 0..7
    /// are flags; bits 8..9 hold the [`BlendMode`] when `SEMI_TRANS`
    /// is set.
    #[repr(transparent)]
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct PrimFlags: u32 {
        /// Apply semi-transparency blending (cmd-bit-1). When clear,
        /// the foreground pixel is written opaque regardless of the
        /// `blend_mode` field. When set, the shader reads the back
        /// buffer and applies `blend_mode`.
        const SEMI_TRANS = 1 << 0;
        /// `mask_check_before_draw` (GP0 0xE6 bit 1). Skip the pixel
        /// entirely if the existing VRAM word has bit 15 set.
        const MASK_CHECK = 1 << 1;
        /// `mask_set_on_draw` (GP0 0xE6 bit 0). OR bit 15 into every
        /// written pixel.
        const MASK_SET = 1 << 2;
    }
}

/// Pack a [`BlendMode`] into the high bits of [`PrimFlags`] for
/// transport to the shader. The shader extracts via `(flags >> 8) & 3`.
#[inline]
pub fn pack_flags(flags: PrimFlags, mode: BlendMode) -> u32 {
    flags.bits() | ((mode as u32) << 8)
}

/// Monochrome flat triangle (`GP0 0x20..=0x23`). Three screen-space
/// vertices (already include drawing-offset) plus a single 15bpp BGR
/// color. The bounding box is precomputed by the host.
///
/// WGSL counterpart in `shaders/mono_tri.wgsl`:
/// ```wgsl
/// struct MonoTri {
///     v0: vec2<i32>,
///     v1: vec2<i32>,
///     v2: vec2<i32>,
///     bbox_min: vec2<i32>,
///     bbox_max: vec2<i32>,
///     color: u32,
///     flags: u32,  // PrimFlags | (BlendMode << 8)
/// }
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MonoTri {
    pub v0: [i32; 2],
    pub v1: [i32; 2],
    pub v2: [i32; 2],
    pub bbox_min: [i32; 2],
    pub bbox_max: [i32; 2],
    /// 15-bit BGR color in the low 16 bits; bit 15 is interpreted by
    /// the shader the same way the CPU rasterizer does — for an
    /// opaque mono primitive it ends up in the output pixel
    /// regardless (`(r as u16) | (g << 5) | (b << 10) | (fg & 0x8000)`).
    pub color: u32,
    /// `PrimFlags::bits() | ((BlendMode as u32) << 8)`. See
    /// [`pack_flags`] / [`PrimFlags`] / [`BlendMode`].
    pub flags: u32,
}

impl MonoTri {
    /// Opaque triangle — the simplest case. No mask handling, no
    /// blending. Bounding box is computed automatically.
    pub fn opaque(v0: (i32, i32), v1: (i32, i32), v2: (i32, i32), color_bgr15: u16) -> Self {
        Self::new(v0, v1, v2, color_bgr15, PrimFlags::empty(), BlendMode::Average)
    }

    /// Full constructor — set any combination of [`PrimFlags`] and
    /// pick a [`BlendMode`] (only used when `SEMI_TRANS` is set).
    pub fn new(
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        color_bgr15: u16,
        flags: PrimFlags,
        blend_mode: BlendMode,
    ) -> Self {
        let min_x = v0.0.min(v1.0).min(v2.0);
        let min_y = v0.1.min(v1.1).min(v2.1);
        let max_x = v0.0.max(v1.0).max(v2.0);
        let max_y = v0.1.max(v1.1).max(v2.1);
        Self {
            v0: [v0.0, v0.1],
            v1: [v1.0, v1.1],
            v2: [v2.0, v2.1],
            bbox_min: [min_x, min_y],
            bbox_max: [max_x, max_y],
            color: color_bgr15 as u32,
            flags: pack_flags(flags, blend_mode),
        }
    }

    /// Same hardware-extent rule as `triangle_exceeds_hw_extent` in
    /// `emulator-core`: any edge whose `|Δx| > 1023` or `|Δy| > 511`
    /// causes hardware to silently drop the primitive.
    pub fn exceeds_hw_extent(&self) -> bool {
        const MAX_DX: i32 = 1023;
        const MAX_DY: i32 = 511;
        let edges = [(self.v0, self.v1), (self.v1, self.v2), (self.v2, self.v0)];
        edges
            .iter()
            .any(|(a, b)| (a[0] - b[0]).abs() > MAX_DX || (a[1] - b[1]).abs() > MAX_DY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_tri_struct_size_is_16_byte_aligned() {
        // Storage-buffer arrays in WGSL require 16-byte stride.
        assert_eq!(std::mem::size_of::<MonoTri>() % 16, 0);
    }

    #[test]
    fn mono_tri_struct_size_is_48() {
        // Pinned: keeping the struct at exactly 48 bytes matches the
        // shader struct's natural layout (six i32×2 + two u32 = 48).
        // If we ever add fields, choose 64 next.
        assert_eq!(std::mem::size_of::<MonoTri>(), 48);
    }

    #[test]
    fn bounding_box_is_inclusive() {
        let t = MonoTri::opaque((10, 20), (30, 25), (15, 40), 0x7FFF);
        assert_eq!(t.bbox_min, [10, 20]);
        assert_eq!(t.bbox_max, [30, 40]);
    }

    #[test]
    fn extent_check_matches_cpu_rule() {
        let kept = MonoTri::opaque((0, 0), (1023, 0), (0, 511), 0);
        assert!(!kept.exceeds_hw_extent());
        let dropped = MonoTri::opaque((0, 0), (1024, 0), (0, 0), 0);
        assert!(dropped.exceeds_hw_extent());
    }

    #[test]
    fn pack_flags_layout() {
        // SEMI_TRANS in low bit, BlendMode in bits 8..9.
        let packed = pack_flags(PrimFlags::SEMI_TRANS, BlendMode::AddQuarter);
        assert_eq!(packed, 0x0000_0301);
        let no_blend = pack_flags(PrimFlags::MASK_SET, BlendMode::Average);
        assert_eq!(no_blend, 0x04);
    }
}
