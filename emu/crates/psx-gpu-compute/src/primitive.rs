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
    /// Default 640×480 area used by the CPU `Gpu::new`. Tests can
    /// pass this if they don't care about clipping.
    pub fn full_vram() -> Self {
        Self {
            left: 0,
            top: 0,
            right: (super::VRAM_WIDTH - 1) as i32,
            bottom: (super::VRAM_HEIGHT - 1) as i32,
        }
    }
}

/// Monochrome flat triangle (`GP0 0x20..=0x23`). Three screen-space
/// vertices (already include drawing-offset) plus a single 15bpp BGR
/// color. The bounding box is precomputed by the host so the shader
/// can dispatch over exactly the right pixel range.
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
///     _pad: u32,
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
    /// 15-bit BGR color (0bbbbbgggggrrrrr). Mask-bit (top bit) is
    /// preserved if the host wants to set it via `MASK_SET_ON_DRAW`,
    /// but for now we always write opaque without flipping bit 15.
    pub color: u32,
    /// Padding to keep the struct size a multiple of 16 bytes
    /// (storage buffer arrays in WGSL want 16-byte stride).
    pub _pad: u32,
}

impl MonoTri {
    /// Construct from three vertices + a 15-bit color. Bounding box
    /// is computed automatically; caller still has to pre-apply the
    /// drawing offset to the vertices (matching `decode_vertex` on
    /// the CPU side).
    pub fn new(v0: (i32, i32), v1: (i32, i32), v2: (i32, i32), color_bgr15: u16) -> Self {
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
            _pad: 0,
        }
    }

    /// Same hardware-extent rule as `triangle_exceeds_hw_extent` in
    /// `emulator-core`: any edge whose `|Δx| > 1023` or `|Δy| > 511`
    /// causes hardware to silently drop the primitive. Mirrors the
    /// rule into the host-side check so the shader doesn't have to
    /// chase a degenerate dispatch.
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
        // If we ever resize the struct, the runtime will silently
        // misindex array elements. Pin it.
        assert_eq!(std::mem::size_of::<MonoTri>() % 16, 0);
    }

    #[test]
    fn bounding_box_is_inclusive() {
        let t = MonoTri::new((10, 20), (30, 25), (15, 40), 0x7FFF);
        assert_eq!(t.bbox_min, [10, 20]);
        assert_eq!(t.bbox_max, [30, 40]);
    }

    #[test]
    fn extent_check_matches_cpu_rule() {
        // 1023 / 511 deltas are KEPT; one over drops.
        let kept = MonoTri::new((0, 0), (1023, 0), (0, 511), 0);
        assert!(!kept.exceeds_hw_extent());
        let dropped = MonoTri::new((0, 0), (1024, 0), (0, 0), 0);
        assert!(dropped.exceeds_hw_extent());
        let dropped_y = MonoTri::new((0, 0), (0, 512), (0, 0), 0);
        assert!(dropped_y.exceeds_hw_extent());
    }
}
