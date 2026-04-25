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
        /// Raw-texture primitive (cmd-bit-0 set). Skip the per-pixel
        /// `modulate_tint` step — the texel is written through as-is.
        /// Equivalent to passing `RAW_TEXTURE_TINT = (0x80,0x80,0x80)`
        /// to `modulate_tint`, which is the identity case, but the
        /// shader skips the multiply for clarity and a tiny perf win.
        const RAW_TEXTURE = 1 << 3;
        /// Texture-rect X flip (GP0 0xE1 bit 12). Only consulted by
        /// the textured-rect shader; ignored by triangles.
        const FLIP_X = 1 << 4;
        /// Texture-rect Y flip (GP0 0xE1 bit 13).
        const FLIP_Y = 1 << 5;
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

/// Texture page (the 256×256 sub-rectangle of VRAM that 4bpp/8bpp/
/// 15bpp texture sampling reads from). Plus per-primitive texture-
/// window override and bit-depth selector. Wire format mirrors the
/// CPU `Gpu` state derived from GP0 0xE1 / `apply_primitive_tpage`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Tpage {
    /// Tpage origin X (0/64/128/.../960). Multiple of 64.
    pub tpage_x: u32,
    /// Tpage origin Y (0 or 256).
    pub tpage_y: u32,
    /// 0 = 4bpp (CLUT, 16 cols / 64 entries), 1 = 8bpp (CLUT, 256
    /// cols / 256 entries), 2 = 15bpp direct colour.
    pub tex_depth: u32,
    pub _pad: u32,
    /// Texture window mask, **already pre-shifted ×8** (matches the
    /// CPU side which stores `mask_x * 8`). Default 0 = passthrough.
    pub tex_window_mask_x: u32,
    pub tex_window_mask_y: u32,
    pub tex_window_off_x: u32,
    pub tex_window_off_y: u32,
}

impl Tpage {
    /// Default tpage at VRAM origin, 4bpp, no texture window. The
    /// only fields tests usually need to override are `tpage_x`,
    /// `tpage_y`, `tex_depth`.
    pub fn new(tpage_x: u32, tpage_y: u32, tex_depth: u32) -> Self {
        Self {
            tpage_x,
            tpage_y,
            tex_depth,
            _pad: 0,
            tex_window_mask_x: 0,
            tex_window_mask_y: 0,
            tex_window_off_x: 0,
            tex_window_off_y: 0,
        }
    }
}

/// Textured flat-shaded triangle (`GP0 0x24..=0x27`). Three vertices
/// + three (U, V) texture coordinates + a per-primitive tint + a
/// CLUT location. The tpage state lives in the separate [`Tpage`]
/// uniform so multiple textured primitives in a batch can share it.
///
/// WGSL counterpart in `shaders/tex_tri.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TexTri {
    pub v0: [i32; 2],
    pub v1: [i32; 2],
    pub v2: [i32; 2],
    pub bbox_min: [i32; 2],
    pub bbox_max: [i32; 2],
    /// `u0 | (v0 << 8)`. Each axis is 8 bits; the shader extracts
    /// them then runs through the texture window + 8-bit wrap.
    pub uv0: u32,
    pub uv1: u32,
    pub uv2: u32,
    /// `R | (G << 8) | (B << 16)`. For raw-texture primitives the
    /// host should set `RAW_TEXTURE` in `flags` and pass any tint
    /// here (it'll be ignored).
    pub tint: u32,
    /// PrimFlags bits + (BlendMode << 8). See [`pack_flags`].
    pub flags: u32,
    /// CLUT origin in VRAM. Only consulted in 4bpp / 8bpp.
    pub clut_x: u32,
    pub clut_y: u32,
    /// Padding so the struct is 16-byte aligned for the uniform
    /// buffer (see B.1 layout note). Size is pinned to 80 bytes
    /// by the `tex_tri_struct_is_16_byte_aligned_and_pinned` test.
    pub _pad: [u32; 3],
}

impl TexTri {
    /// Builder: caller has decoded the GP0 packet's vertices, UVs,
    /// CLUT, and tint into typed fields. Tpage is passed separately.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        uv0: (u8, u8),
        uv1: (u8, u8),
        uv2: (u8, u8),
        clut_x: u32,
        clut_y: u32,
        tint_rgb: (u8, u8, u8),
        flags: PrimFlags,
        blend_mode: BlendMode,
    ) -> Self {
        let min_x = v0.0.min(v1.0).min(v2.0);
        let min_y = v0.1.min(v1.1).min(v2.1);
        let max_x = v0.0.max(v1.0).max(v2.0);
        let max_y = v0.1.max(v1.1).max(v2.1);
        let pack_uv = |u: u8, v: u8| (u as u32) | ((v as u32) << 8);
        let pack_tint =
            |r: u8, g: u8, b: u8| (r as u32) | ((g as u32) << 8) | ((b as u32) << 16);
        Self {
            v0: [v0.0, v0.1],
            v1: [v1.0, v1.1],
            v2: [v2.0, v2.1],
            bbox_min: [min_x, min_y],
            bbox_max: [max_x, max_y],
            uv0: pack_uv(uv0.0, uv0.1),
            uv1: pack_uv(uv1.0, uv1.1),
            uv2: pack_uv(uv2.0, uv2.1),
            tint: pack_tint(tint_rgb.0, tint_rgb.1, tint_rgb.2),
            flags: pack_flags(flags, blend_mode),
            clut_x,
            clut_y,
            _pad: [0; 3],
        }
    }

    /// Same hardware-extent rule as `MonoTri::exceeds_hw_extent`.
    pub fn exceeds_hw_extent(&self) -> bool {
        const MAX_DX: i32 = 1023;
        const MAX_DY: i32 = 511;
        let edges = [(self.v0, self.v1), (self.v1, self.v2), (self.v2, self.v0)];
        edges
            .iter()
            .any(|(a, b)| (a[0] - b[0]).abs() > MAX_DX || (a[1] - b[1]).abs() > MAX_DY)
    }
}

/// Monochrome rectangle (`GP0 0x60..=0x63`, plus the fixed-size
/// 1×1 / 8×8 / 16×16 variants). Top-left corner + `(w, h)` size
/// + a single 15bpp BGR colour. Same RMW semantics as `MonoTri`.
///
/// WGSL counterpart in `shaders/mono_rect.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MonoRect {
    pub xy: [i32; 2],
    pub wh: [u32; 2],
    pub color: u32,
    pub flags: u32,
    pub _pad: [u32; 2],
}

impl MonoRect {
    /// Opaque rectangle — no semi-trans, no mask. Width / height of
    /// zero are dropped (the dispatcher handles this).
    pub fn opaque(xy: (i32, i32), wh: (u32, u32), color_bgr15: u16) -> Self {
        Self::new(
            xy,
            wh,
            color_bgr15,
            PrimFlags::empty(),
            BlendMode::Average,
        )
    }

    pub fn new(
        xy: (i32, i32),
        wh: (u32, u32),
        color_bgr15: u16,
        flags: PrimFlags,
        blend_mode: BlendMode,
    ) -> Self {
        Self {
            xy: [xy.0, xy.1],
            wh: [wh.0, wh.1],
            color: color_bgr15 as u32,
            flags: pack_flags(flags, blend_mode),
            _pad: [0; 2],
        }
    }
}

/// Textured rectangle (`GP0 0x64..=0x67`, plus fixed-size variants).
/// Direct (U, V) blit from the active tpage with optional X/Y flip
/// from GP0 0xE1 bits 12/13. Tile coords step linearly — no UV
/// interpolation, so parity vs the CPU rasterizer is bit-exact.
///
/// WGSL counterpart in `shaders/tex_rect.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TexRect {
    pub xy: [i32; 2],
    pub wh: [u32; 2],
    /// `u_base | (v_base << 8)` — the top-left UV (or bottom-right
    /// if FLIP_X / FLIP_Y are set, since the flip happens *around*
    /// the base).
    pub uv: u32,
    pub clut_x: u32,
    pub clut_y: u32,
    pub tint: u32,
    /// PrimFlags + (BlendMode << 8). FLIP_X / FLIP_Y are extra bits
    /// in the same field — see [`PrimFlags`].
    pub flags: u32,
    /// Padding to bring the struct to 48 bytes (16-byte aligned for
    /// uniform buffers). Pinned by `tex_rect_struct_pinned_at_48`.
    pub _pad: [u32; 3],
}

impl TexRect {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        xy: (i32, i32),
        wh: (u32, u32),
        uv: (u8, u8),
        clut_x: u32,
        clut_y: u32,
        tint_rgb: (u8, u8, u8),
        flags: PrimFlags,
        blend_mode: BlendMode,
    ) -> Self {
        let pack_uv = |u: u8, v: u8| (u as u32) | ((v as u32) << 8);
        let pack_tint =
            |r: u8, g: u8, b: u8| (r as u32) | ((g as u32) << 8) | ((b as u32) << 16);
        Self {
            xy: [xy.0, xy.1],
            wh: [wh.0, wh.1],
            uv: pack_uv(uv.0, uv.1),
            clut_x,
            clut_y,
            tint: pack_tint(tint_rgb.0, tint_rgb.1, tint_rgb.2),
            flags: pack_flags(flags, blend_mode),
            _pad: [0; 3],
        }
    }
}

/// Quick fill (`GP0 0x02`). Writes a single 15bpp colour into a
/// rectangle, ignoring drawing-area, drawing-offset, mask-check,
/// mask-set, and semi-trans. The hardware clamps `x`/`w` to 16-pixel
/// alignment — the host should mask before constructing this.
///
/// WGSL counterpart in `shaders/fill.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Fill {
    /// Top-left, x in [0..1023] aligned to 16, y in [0..511].
    pub xy: [u32; 2],
    /// Size, w in [0..1023] aligned to 16, h in [0..511].
    pub wh: [u32; 2],
    /// 15bpp BGR colour. Bit 15 is written as-is — caller must
    /// mask if they need the bit clear.
    pub color: u32,
    pub _pad: [u32; 3],
}

impl Fill {
    /// Construct a fill primitive. Applies the same hardware-mask
    /// rules as the CPU rasterizer: x's low 4 bits are cleared,
    /// w is rounded UP to 16.
    pub fn new(xy: (u32, u32), wh: (u32, u32), color_bgr15: u16) -> Self {
        let x = xy.0 & 0x3F0;
        let y = xy.1 & 0x1FF;
        let w = (wh.0.saturating_add(0xF)) & 0x3F0;
        let h = wh.1 & 0x1FF;
        Self {
            xy: [x, y],
            wh: [w, h],
            color: color_bgr15 as u32,
            _pad: [0; 3],
        }
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

    #[test]
    fn tex_tri_struct_is_16_byte_aligned_and_pinned() {
        // Pin the struct size — if it ever changes, the WGSL struct
        // in tex_tri.wgsl needs an explicit update.
        assert_eq!(std::mem::size_of::<TexTri>() % 16, 0);
        assert_eq!(std::mem::size_of::<TexTri>(), 80);
    }

    #[test]
    fn tpage_struct_is_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<Tpage>() % 16, 0);
        assert_eq!(std::mem::size_of::<Tpage>(), 32);
    }

    #[test]
    fn mono_rect_struct_pinned_at_32() {
        assert_eq!(std::mem::size_of::<MonoRect>(), 32);
        assert_eq!(std::mem::size_of::<MonoRect>() % 16, 0);
    }

    #[test]
    fn tex_rect_struct_pinned_at_48() {
        assert_eq!(std::mem::size_of::<TexRect>(), 48);
        assert_eq!(std::mem::size_of::<TexRect>() % 16, 0);
    }

    #[test]
    fn fill_struct_pinned_at_32() {
        assert_eq!(std::mem::size_of::<Fill>(), 32);
        assert_eq!(std::mem::size_of::<Fill>() % 16, 0);
    }

    #[test]
    fn fill_applies_hardware_alignment() {
        // x low 4 bits cleared; w rounded up to 16. y / h pass through.
        let f = Fill::new((23, 100), (50, 30), 0x1234);
        assert_eq!(f.xy, [16, 100], "x masked to 16-pixel alignment");
        assert_eq!(f.wh, [64, 30], "w rounded UP to next 16");
        // Already-aligned values pass through.
        let g = Fill::new((32, 100), (48, 30), 0x1234);
        assert_eq!(g.xy, [32, 100]);
        assert_eq!(g.wh, [48, 30]);
    }

    #[test]
    fn tex_tri_packs_uv_in_low_high_bytes() {
        let t = TexTri::new(
            (0, 0), (10, 0), (0, 10),
            (0x12, 0x34), (0x56, 0x78), (0x9A, 0xBC),
            0, 0,
            (0x80, 0x80, 0x80),
            PrimFlags::empty(),
            BlendMode::Average,
        );
        assert_eq!(t.uv0, 0x3412);
        assert_eq!(t.uv1, 0x7856);
        assert_eq!(t.uv2, 0xBC9A);
    }
}
