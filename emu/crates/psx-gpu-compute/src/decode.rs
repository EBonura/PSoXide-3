//! GP0 packet decoders + replay-time state tracking.
//!
//! Lifted out of `replay.rs` so the new `psx-gpu-render` crate
//! (the wgpu render-pipeline backend that runs alongside the
//! compute backend) can reuse the exact same vertex/UV/tint/CLUT
//! decoding and the exact same `ReplayState` rules. Two backends
//! reading the same `cmd_log` MUST decode it identically — keeping
//! the helpers here is the single point that guarantees that.
//!
//! Public API:
//! - [`ReplayState`]: per-replay state tracker for tpage / draw-area
//!   / draw-offset / mask / dither / flip flags, mirroring the
//!   `emulator-core::Gpu` fields the CPU rasterizer reads.
//! - [`decode_vertex`], [`decode_uv`], [`decode_clut`], [`decode_tint`],
//!   [`rgb24_to_bgr15`]: pure decoders — the per-word arithmetic
//!   each opcode performs to extract its parameters.
//! - [`apply_primitive_tpage`]: per-primitive tpage word ingestion
//!   (UV1's high half on textured primitives).
//! - [`decode_blend_mode`]: the 2-bit semi-transparency selector.
//! - [`is_raw_texture`], [`is_semi_trans`]: cmd-bit predicates.
//! - [`sign_extend_11`]: vertex-coord sign-extension to i32.

use crate::primitive::{BlendMode, DrawArea, PrimFlags, Tpage};

/// GP0 state we have to track in lockstep with the CPU rasterizer
/// so triangle / rect dispatches see the right tpage, drawing area,
/// drawing offset, mask flags, and per-primitive blend mode.
#[derive(Debug, Clone)]
pub struct ReplayState {
    pub draw_area: DrawArea,
    pub draw_offset_x: i32,
    pub draw_offset_y: i32,
    pub tpage: Tpage,
    /// Active tpage blend mode (decoded from GP0 0xE1 bits 5-6 or
    /// per-primitive tpage word). Used by mono / shaded primitives
    /// when their cmd-bit-1 (semi-trans) is set.
    pub tex_blend_mode: BlendMode,
    /// `mask_set_on_draw` (GP0 0xE6 bit 0).
    pub mask_set: bool,
    /// `mask_check_before_draw` (GP0 0xE6 bit 1).
    pub mask_check: bool,
    /// `dither` (GP0 0xE1 bit 9). Affects shaded / textured-shaded.
    pub dither: bool,
    /// `tex_rect_flip_x` / `_y` (GP0 0xE1 bits 12 / 13).
    pub flip_x: bool,
    pub flip_y: bool,
}

impl ReplayState {
    pub fn new() -> Self {
        Self {
            draw_area: DrawArea {
                left: 0,
                top: 0,
                right: 1023,
                bottom: 511,
            },
            draw_offset_x: 0,
            draw_offset_y: 0,
            tpage: Tpage::new(0, 0, 0),
            tex_blend_mode: BlendMode::Average,
            mask_set: false,
            mask_check: false,
            dither: false,
            flip_x: false,
            flip_y: false,
        }
    }

    pub fn base_flags(&self) -> PrimFlags {
        let mut f = PrimFlags::empty();
        if self.mask_set {
            f |= PrimFlags::MASK_SET;
        }
        if self.mask_check {
            f |= PrimFlags::MASK_CHECK;
        }
        if self.dither {
            f |= PrimFlags::DITHER;
        }
        f
    }

    pub fn rect_flip_flags(&self) -> PrimFlags {
        let mut f = self.base_flags();
        if self.flip_x {
            f |= PrimFlags::FLIP_X;
        }
        if self.flip_y {
            f |= PrimFlags::FLIP_Y;
        }
        f
    }
}

impl Default for ReplayState {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode helper for "is this primitive raw-textured?". Tested
/// against bit 0 of the OPCODE byte (= bit 24 of the cmd word) per
/// PSX-SPX, matching `emulator-core::Gpu::draw_textured_*`.
#[inline]
pub fn is_raw_texture(cmd: u32) -> bool {
    (cmd >> 24) & 1 != 0
}

/// `prim_is_semi_trans` — bit 1 of the opcode byte (cmd word bit 25).
#[inline]
pub fn is_semi_trans(cmd: u32) -> bool {
    (cmd >> 25) & 1 != 0
}

/// Decode an `XY` vertex word: 11-bit signed components plus the
/// active drawing offset (already pre-applied here so callers see
/// screen-space integers).
#[inline]
pub fn decode_vertex(state: &ReplayState, word: u32) -> (i32, i32) {
    let x = sign_extend_11((word & 0x7FF) as i32) + state.draw_offset_x;
    let y = sign_extend_11(((word >> 16) & 0x7FF) as i32) + state.draw_offset_y;
    (x, y)
}

/// Sign-extend an 11-bit signed value held in the low 11 bits of an
/// `i32` to the full `i32` range.
#[inline]
pub fn sign_extend_11(v: i32) -> i32 {
    if v & 0x400 != 0 {
        v | !0x7FF
    } else {
        v & 0x7FF
    }
}

/// Decode a UV word's low half: `(u, v)` as 8-bit unsigned.
#[inline]
pub fn decode_uv(word: u32) -> (u8, u8) {
    ((word & 0xFF) as u8, ((word >> 8) & 0xFF) as u8)
}

/// Decode the `RGB` tint that's packed into the low 24 bits of an
/// opcode word.
#[inline]
pub fn decode_tint(cmd: u32) -> (u8, u8, u8) {
    (
        (cmd & 0xFF) as u8,
        ((cmd >> 8) & 0xFF) as u8,
        ((cmd >> 16) & 0xFF) as u8,
    )
}

/// Decode CLUT origin from the high half of UV0 on textured
/// primitives. Returns `(clut_x, clut_y)` in VRAM pixel coordinates.
#[inline]
pub fn decode_clut(word: u32) -> (u32, u32) {
    let clut_word = (word >> 16) & 0xFFFF;
    let cx = (clut_word & 0x3F) * 16;
    let cy = (clut_word >> 6) & 0x1FF;
    (cx, cy)
}

/// Convert a 24-bit RGB color (low 24 bits of a GP0 opcode word) to
/// the BGR15 representation stored in VRAM.
#[inline]
pub fn rgb24_to_bgr15(rgb: u32) -> u16 {
    let r = ((rgb >> 3) & 0x1F) as u16;
    let g = (((rgb >> 8) >> 3) & 0x1F) as u16;
    let b = (((rgb >> 16) >> 3) & 0x1F) as u16;
    r | (g << 5) | (b << 10)
}

/// Apply a primitive's tpage word (the high half of UV1 in the GP0
/// packet) to the replay state, mirroring
/// `emulator-core::Gpu::apply_primitive_tpage` byte-for-byte.
pub fn apply_primitive_tpage(state: &mut ReplayState, uv_word: u32) {
    let tpage = (uv_word >> 16) & 0xFFFF;
    let tpage_x = (tpage & 0x0F) * 64;
    let tpage_y: u32 = if (tpage >> 4) & 1 != 0 { 256 } else { 0 };
    let tex_depth = (tpage >> 7) & 0x3;
    state.tpage = Tpage::new(tpage_x, tpage_y, tex_depth);
    state.tex_blend_mode = decode_blend_mode(tpage >> 5);
    // CPU `apply_primitive_tpage` also OR-folds dither_enabled
    // (`tpage |= 0x200` if dither was on globally). We mirror by
    // leaving `state.dither` as-is — dither only flips off via a
    // GP0 0xE1, not a primitive's tpage word.
}

/// Decode the 2-bit semi-transparency selector from a tpage word's
/// bits 5-6.
#[inline]
pub fn decode_blend_mode(bits: u32) -> BlendMode {
    match bits & 0x3 {
        0 => BlendMode::Average,
        1 => BlendMode::Add,
        2 => BlendMode::Sub,
        _ => BlendMode::AddQuarter,
    }
}
