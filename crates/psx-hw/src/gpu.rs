//! GPU registers and status bits.
//!
//! The PS1 GPU has exactly two memory-mapped addresses. Software
//! communicates almost entirely through command packets pushed to `GP0`
//! (rendering and VRAM transfers) and `GP1` (display control and state).
//!
//! Command-packet layouts (triangles, rectangles, fills, CPU-to-VRAM,
//! etc.) live in [`packet`]; the register definitions and `GPUSTAT` bits
//! live in the top level of this module.
//!
//! Reference: nocash PSX-SPX "GPU I/O Ports" section.

use bitflags::bitflags;

/// `GP0` — write: rendering commands and VRAM transfer data.
///
/// Reading from this address returns `GPUREAD` (VRAM-to-CPU transfer data
/// or GP1(10h) register response).
pub const GP0: u32 = 0x1F80_1810;

/// `GP1` — write: display-control commands (reset, display enable, mode,
/// DMA direction, display area, horizontal/vertical range, etc.).
///
/// Reading from this address returns [`GPUSTAT`](GpuStat).
pub const GP1: u32 = 0x1F80_1814;

/// Alias for reads from `0x1F80_1810`: the VRAM-to-CPU / GP1(10h) read path.
pub const GPUREAD: u32 = GP0;

/// Alias for reads from `0x1F80_1814`: the status register.
pub const GPUSTAT: u32 = GP1;

bitflags! {
    /// `GPUSTAT` — 32-bit status register read via [`GPUSTAT`].
    ///
    /// Several fields are multi-bit (texture page base, horizontal
    /// resolution, DMA direction); those are exposed as masks with a
    /// helper accessor. Single-bit flags use the normal bitflags API.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GpuStat: u32 {
        /// Texture page X base: `N * 64` pixels in VRAM (bits 0..=3).
        const TEXPAGE_X_MASK     = 0b1111;
        /// Texture page Y base: `N * 256` pixels in VRAM (bit 4).
        const TEXPAGE_Y          = 1 << 4;
        /// Semi-transparency mode (bits 5..=6).
        const SEMI_TRANS_MASK    = 0b11 << 5;
        /// Texture page color depth (bits 7..=8). 0=4bit, 1=8bit, 2=15bit.
        const TEXPAGE_COLORS_MASK = 0b11 << 7;

        /// Dither 24-bit to 15-bit enable.
        const DITHER             = 1 << 9;
        /// Drawing to display area allowed.
        const DRAW_TO_DISPLAY    = 1 << 10;
        /// Set mask bit when drawing pixels.
        const SET_MASK_ON_DRAW   = 1 << 11;
        /// Draw pixels only when mask bit is clear.
        const DRAW_PIXELS_CHECK  = 1 << 12;
        /// Interlace field (always 1 when GP1(08h).5 is set).
        const INTERLACE_FIELD    = 1 << 13;
        /// Reverseflag (bit 14, unused on retail hardware).
        const REVERSE_FLAG       = 1 << 14;
        /// Texture disable (only when GP1(09h) has enabled the feature).
        const TEXTURE_DISABLE    = 1 << 15;

        /// Horizontal resolution bit 2 (bit 16). When set, H-res is 368.
        const HRES2              = 1 << 16;
        /// Horizontal resolution bits 0..=1 (bits 17..=18).
        const HRES1_MASK         = 0b11 << 17;
        /// Vertical resolution: 0=240, 1=480 (interlace only).
        const VRES               = 1 << 19;
        /// Video mode: 0=NTSC/60Hz, 1=PAL/50Hz.
        const VMODE_PAL          = 1 << 20;
        /// Display area color depth: 0=15bit, 1=24bit.
        const DISPLAY_24BIT      = 1 << 21;
        /// Vertical interlace enable.
        const VERTICAL_INTERLACE = 1 << 22;
        /// Display off (GP1(03h).0 = 1).
        const DISPLAY_DISABLE    = 1 << 23;

        /// Interrupt request (GP0(1Fh) sets this).
        const IRQ1               = 1 << 24;
        /// DMA / data request. Meaning depends on DMA direction (bits 29..=30).
        const DMA_REQUEST        = 1 << 25;
        /// Ready to receive command word.
        const READY_CMD          = 1 << 26;
        /// Ready to send VRAM → CPU data.
        const READY_VRAM_SEND    = 1 << 27;
        /// Ready to receive DMA block.
        const READY_DMA_RECV     = 1 << 28;
        /// DMA direction bits 29..=30.
        /// 0=off, 1=FIFO, 2=CPU→GP0, 3=GPUREAD→CPU.
        const DMA_DIRECTION_MASK = 0b11 << 29;
        /// Drawing odd line in interlace mode (0=even/vblank, 1=odd).
        const INTERLACE_ODD      = 1 << 31;
    }
}

impl GpuStat {
    /// Texture page X base in pixels (0, 64, 128, …, 960).
    #[inline]
    pub const fn texpage_x(self) -> u16 {
        (self.bits() as u16 & 0xF) * 64
    }

    /// Texture page Y base in pixels (0 or 256).
    #[inline]
    pub const fn texpage_y(self) -> u16 {
        if self.contains(Self::TEXPAGE_Y) {
            256
        } else {
            0
        }
    }

    /// Horizontal resolution in pixels (256, 320, 368, 512, 640).
    #[inline]
    pub const fn horizontal_resolution(self) -> u16 {
        if self.contains(Self::HRES2) {
            368
        } else {
            match (self.bits() >> 17) & 0b11 {
                0 => 256,
                1 => 320,
                2 => 512,
                3 => 640,
                _ => unreachable!(),
            }
        }
    }

    /// Vertical resolution in pixels (240 or 480).
    #[inline]
    pub const fn vertical_resolution(self) -> u16 {
        if self.contains(Self::VRES) {
            480
        } else {
            240
        }
    }

    /// DMA direction as a decoded enum.
    #[inline]
    pub const fn dma_direction(self) -> DmaDirection {
        match (self.bits() >> 29) & 0b11 {
            0 => DmaDirection::Off,
            1 => DmaDirection::Fifo,
            2 => DmaDirection::CpuToGp0,
            3 => DmaDirection::GpureadToCpu,
            _ => unreachable!(),
        }
    }
}

/// DMA direction encoded in `GPUSTAT` bits 29..=30.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DmaDirection {
    /// Transfer disabled.
    Off = 0,
    /// FIFO (unused by most software).
    Fifo = 1,
    /// CPU → GP0, used for rendering-command DMA.
    CpuToGp0 = 2,
    /// GPUREAD → CPU, used for VRAM readback.
    GpureadToCpu = 3,
}

/// GP1 command-word constructors.
///
/// GP1 commands are single 32-bit writes to [`GP1`]. The top byte is
/// the opcode; the low 24 bits are the parameters. Each constructor
/// encodes exactly one command — callers are free to use them
/// directly or wrap them in higher-level SDK calls.
pub mod gp1 {
    /// GP1(00h) — Reset GPU (clear FIFO, reset mode, disable display).
    pub const RESET: u32 = 0x0000_0000;

    /// GP1(01h) — Reset command buffer / FIFO.
    pub const RESET_CMD_BUFFER: u32 = 0x0100_0000;

    /// GP1(02h) — Acknowledge GPU IRQ1 (GP0 1Fh).
    pub const ACK_IRQ: u32 = 0x0200_0000;

    /// GP1(03h) — Display enable. `true` disables (screen black),
    /// `false` enables (normal output). The sense is inverted on
    /// hardware: bit 0 set = DISABLED.
    #[inline(always)]
    pub const fn display_enable(enable: bool) -> u32 {
        0x0300_0000 | ((!enable) as u32 & 1)
    }

    /// GP1(04h) — DMA direction.
    /// `0` = off, `1` = FIFO, `2` = CPU→GP0, `3` = GPUREAD→CPU.
    #[inline(always)]
    pub const fn dma_direction(dir: u32) -> u32 {
        0x0400_0000 | (dir & 3)
    }

    /// GP1(05h) — VRAM coordinates of the top-left displayed pixel.
    /// Used for double-buffering: write the back buffer's origin
    /// to flip.
    #[inline(always)]
    pub const fn display_start(x: u32, y: u32) -> u32 {
        0x0500_0000 | (x & 0x3FF) | ((y & 0x1FF) << 10)
    }

    /// GP1(06h) — Horizontal display range (GPU clock counts from
    /// start-of-line). Standard NTSC window: `0x260..0xC60`.
    #[inline(always)]
    pub const fn h_display_range(x1: u32, x2: u32) -> u32 {
        0x0600_0000 | (x1 & 0xFFF) | ((x2 & 0xFFF) << 12)
    }

    /// GP1(07h) — Vertical display range (scanline). Standard NTSC:
    /// `0x10..0x100` → 240 lines starting at line 16.
    #[inline(always)]
    pub const fn v_display_range(y1: u32, y2: u32) -> u32 {
        0x0700_0000 | (y1 & 0x3FF) | ((y2 & 0x3FF) << 10)
    }

    /// GP1(08h) — Display mode (resolution, video standard, depth).
    /// - `hres_1` (0..3): 256/320/512/640 (combine with `hres_2=true`
    ///   for 368).
    /// - `vres`: 0 = 240 lines, 1 = 480 interlaced.
    /// - `pal`: PAL 50 Hz when true, NTSC 60 Hz when false.
    /// - `depth_24`: 24-bit VRAM output.
    /// - `interlace`: enable vertical interlace.
    #[inline(always)]
    pub const fn display_mode(
        hres_1: u32,
        vres: u32,
        pal: bool,
        depth_24: bool,
        interlace: bool,
    ) -> u32 {
        0x0800_0000
            | (hres_1 & 3)
            | ((vres & 1) << 2)
            | ((pal as u32) << 3)
            | ((depth_24 as u32) << 4)
            | ((interlace as u32) << 5)
    }

    /// GP1(10h) — Get-GPU-info latch. The returned value can be read
    /// via `GPUREAD` (at `GP0`) on the next read.
    /// - `0x02`: texture-window setting
    /// - `0x03`: draw-area top-left
    /// - `0x04`: draw-area bottom-right
    /// - `0x05`: draw offset
    /// - `0x07`: GPU type (always `0x02` on retail).
    #[inline(always)]
    pub const fn get_gpu_info(sub_op: u32) -> u32 {
        0x1000_0000 | (sub_op & 0xF)
    }
}

/// GP0 command-word constructors.
///
/// GP0 commands come in two kinds:
/// - **Environment commands** (0xE1..=0xE6): single-word settings
///   like draw mode, draw area, draw offset, mask bit.
/// - **Primitive / transfer commands**: multi-word packets. The
///   command byte + first parameter is built here; the caller sends
///   additional vertex / color / size words on follow-up writes.
pub mod gp0 {
    use super::pack_color;

    /// GP0(00h) — NOP.
    pub const NOP: u32 = 0x0000_0000;
    /// GP0(01h) — Clear texture cache.
    pub const CLEAR_CACHE: u32 = 0x0100_0000;

    /// GP0(02h) — Fill rectangle in VRAM (not clipped, ignores draw
    /// area / offset). Header carries the color; follow-up words are
    /// `pack_xy(x, y)` then `pack_xy(w, h)`.
    #[inline(always)]
    pub const fn fill_rect(r: u8, g: u8, b: u8) -> u32 {
        0x0200_0000 | pack_color(r, g, b)
    }

    /// GP0(E1h) — Draw mode (texpage + semi-transparency + depth).
    #[inline(always)]
    pub const fn draw_mode(
        texpage_x: u32,
        texpage_y: u32,
        semi_tp: u32,
        tex_depth: u32,
        dither: bool,
        draw_to_display: bool,
    ) -> u32 {
        0xE100_0000
            | (texpage_x & 0xF)
            | ((texpage_y & 1) << 4)
            | ((semi_tp & 3) << 5)
            | ((tex_depth & 3) << 7)
            | ((dither as u32) << 9)
            | ((draw_to_display as u32) << 10)
    }

    /// GP0(E2h) — Texture window (wrap inside a rect).
    #[inline(always)]
    pub const fn tex_window(mask_x: u32, mask_y: u32, off_x: u32, off_y: u32) -> u32 {
        0xE200_0000
            | (mask_x & 0x1F)
            | ((mask_y & 0x1F) << 5)
            | ((off_x & 0x1F) << 10)
            | ((off_y & 0x1F) << 15)
    }

    /// GP0(E3h) — Draw area top-left corner (pixels clipped inside).
    #[inline(always)]
    pub const fn draw_area_top_left(x: u32, y: u32) -> u32 {
        0xE300_0000 | (x & 0x3FF) | ((y & 0x1FF) << 10)
    }

    /// GP0(E4h) — Draw area bottom-right corner.
    #[inline(always)]
    pub const fn draw_area_bottom_right(x: u32, y: u32) -> u32 {
        0xE400_0000 | (x & 0x3FF) | ((y & 0x1FF) << 10)
    }

    /// GP0(E5h) — Draw offset (added to every vertex before raster).
    /// Each component is 11-bit signed; pass negatives as two's-comp.
    #[inline(always)]
    pub const fn draw_offset(x: i32, y: i32) -> u32 {
        0xE500_0000 | ((x as u32) & 0x7FF) | (((y as u32) & 0x7FF) << 11)
    }

    /// GP0(E6h) — Mask-bit setting (stencil-style control).
    #[inline(always)]
    pub const fn mask_bit(set_on_draw: bool, check_before_draw: bool) -> u32 {
        0xE600_0000 | (set_on_draw as u32) | ((check_before_draw as u32) << 1)
    }

    /// Build the opcode byte for a polygon command. Bits:
    /// - `7..5`: fixed `001` (polygon class)
    /// - `4`:   gouraud-shaded
    /// - `3`:   4-vertex quad (else 3-vertex triangle)
    /// - `2`:   textured
    /// - `1`:   semi-transparent
    /// - `0`:   raw-texture (bypass texture blending)
    ///
    /// Returns a value suitable for OR-ing with the first-vertex color
    /// (via [`pack_color`]) to form the full header word.
    #[inline(always)]
    pub const fn polygon_opcode(
        gouraud: bool,
        quad: bool,
        textured: bool,
        semi_transparent: bool,
        raw_tex: bool,
    ) -> u32 {
        0x2000_0000
            | ((gouraud as u32) << 28)
            | ((quad as u32) << 27)
            | ((textured as u32) << 26)
            | ((semi_transparent as u32) << 25)
            | ((raw_tex as u32) << 24)
    }

    /// GP0(A0h) — Begin CPU→VRAM transfer. Header is this word; then
    /// two words of dest `xy` / `wh`; then `ceil(w*h/2)` pixel words.
    pub const COPY_CPU_TO_VRAM: u32 = 0xA000_0000;

    /// GP0(C0h) — Begin VRAM→CPU transfer. Same header layout as
    /// [`COPY_CPU_TO_VRAM`]; pixel words come back via `GPUREAD`.
    pub const COPY_VRAM_TO_CPU: u32 = 0xC000_0000;

    /// GP0(80h) — VRAM→VRAM copy (rect blit).
    pub const COPY_VRAM_TO_VRAM: u32 = 0x8000_0000;

    /// GP0(1Fh) — Request IRQ1.
    pub const REQUEST_IRQ: u32 = 0x1F00_0000;
}

/// Pack three 8-bit color channels into the low 24 bits of a GP0
/// command word. Several GP0 opcodes embed the first vertex's color
/// in their header this way.
#[inline(always)]
pub const fn pack_color(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16)
}

/// Pack a signed 11-bit XY pair into a GP0 vertex word. The GPU
/// sign-extends each half internally — we store them as unsigned
/// u16 bit patterns.
#[inline(always)]
pub const fn pack_vertex(x: i16, y: i16) -> u32 {
    ((x as u16) as u32) | (((y as u16) as u32) << 16)
}

/// Pack two 16-bit unsigned values (VRAM coord or rect size) into
/// one word. Used by CPU↔VRAM transfer headers.
#[inline(always)]
pub const fn pack_xy(x: u16, y: u16) -> u32 {
    (x as u32) | ((y as u32) << 16)
}

/// Pack a texcoord + CLUT/tpage extra into a GP0 data word.
/// `u` and `v` are 8-bit unsigned texture coords; `extra` is the
/// 16-bit CLUT handle or texpage selector that lives in the high half.
#[inline(always)]
pub const fn pack_texcoord(u: u8, v: u8, extra: u16) -> u32 {
    (u as u32) | ((v as u32) << 8) | ((extra as u32) << 16)
}

/// Command-packet layouts for `GP0` and `GP1`.
///
/// Populated incrementally as the emulator and SDK grow: triangles and
/// rectangles first (Milestone C), textured primitives next, then
/// VRAM-transfer packets.
pub mod packet {
    // Intentionally empty for now. Each packet type will live in its own
    // submodule (`triangle`, `rectangle`, `fill`, `copy`, …) with a
    // `#[repr(C)]` layout and a `static_assert_eq!` on its size.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_addresses_live_in_io_window() {
        use crate::memory::io;
        let end = io::BASE + io::SIZE as u32;
        assert!((io::BASE..end).contains(&GP0));
        assert!((io::BASE..end).contains(&GP1));
    }

    #[test]
    fn horizontal_resolution_decodes_all_modes() {
        // HRES1=0, HRES2=0 → 256
        assert_eq!(GpuStat::empty().horizontal_resolution(), 256);
        // HRES1=1, HRES2=0 → 320
        assert_eq!(
            GpuStat::from_bits_retain(1 << 17).horizontal_resolution(),
            320
        );
        // HRES1=2, HRES2=0 → 512
        assert_eq!(
            GpuStat::from_bits_retain(2 << 17).horizontal_resolution(),
            512
        );
        // HRES1=3, HRES2=0 → 640
        assert_eq!(
            GpuStat::from_bits_retain(3 << 17).horizontal_resolution(),
            640
        );
        // HRES2=1 → 368 regardless of HRES1
        assert_eq!(
            GpuStat::from_bits_retain(1 << 16).horizontal_resolution(),
            368
        );
    }

    #[test]
    fn dma_direction_round_trips() {
        for raw in 0..4u32 {
            let stat = GpuStat::from_bits_retain(raw << 29);
            assert_eq!(stat.dma_direction() as u32, raw);
        }
    }

    #[test]
    fn texpage_decoding() {
        // Texpage X=5 → 320, Y=1 → 256.
        let stat = GpuStat::from_bits_retain(5 | (1 << 4));
        assert_eq!(stat.texpage_x(), 320);
        assert_eq!(stat.texpage_y(), 256);
    }

    #[test]
    fn gp1_display_enable_inverts_sense() {
        // Hardware bit 0 means DISABLED; our constructor takes a
        // logical "enable" bool and flips it.
        assert_eq!(gp1::display_enable(true), 0x0300_0000);
        assert_eq!(gp1::display_enable(false), 0x0300_0001);
    }

    #[test]
    fn gp1_display_start_packs_coords() {
        assert_eq!(gp1::display_start(0, 0), 0x0500_0000);
        // X=640 (10 bits), Y=256 (9 bits): 640 | (256 << 10) = 640 + 262144 = 0x40280
        assert_eq!(gp1::display_start(640, 256), 0x0500_0000 | 0x40280);
    }

    #[test]
    fn gp1_display_mode_encodes_all_fields() {
        // hres=1 (320), vres=0 (240), NTSC, 15bit, no interlace.
        assert_eq!(gp1::display_mode(1, 0, false, false, false), 0x0800_0001);
        // 640x480 PAL, 24bit, interlaced.
        let m = gp1::display_mode(3, 1, true, true, true);
        assert_eq!(m & 0xFF, 0x3F, "low byte = 00111111");
    }

    #[test]
    fn gp0_fill_rect_embeds_color() {
        // R=0xAA G=0xBB B=0xCC → 0x02CCBBAA
        let word = gp0::fill_rect(0xAA, 0xBB, 0xCC);
        assert_eq!(word, 0x02CC_BBAA);
    }

    #[test]
    fn gp0_polygon_opcode_bits() {
        // Flat triangle, opaque: 0x20000000
        assert_eq!(
            gp0::polygon_opcode(false, false, false, false, false),
            0x2000_0000
        );
        // Gouraud quad, textured, semi-transparent, raw-tex: all bits set.
        assert_eq!(
            gp0::polygon_opcode(true, true, true, true, true),
            0x3F00_0000
        );
    }

    #[test]
    fn pack_vertex_handles_negative() {
        // (-1, -1) should produce 0xFFFFFFFF.
        assert_eq!(pack_vertex(-1, -1), 0xFFFF_FFFF);
        assert_eq!(pack_vertex(100, 200), 100 | (200 << 16));
    }

    #[test]
    fn gp0_draw_area_and_offset_symmetric() {
        // draw_area_top_left and bottom_right share packing; differ only
        // in opcode byte.
        let tl = gp0::draw_area_top_left(10, 20);
        let br = gp0::draw_area_bottom_right(10, 20);
        assert_eq!(tl, 0xE300_0000 | 10 | (20 << 10));
        assert_eq!(br, 0xE400_0000 | 10 | (20 << 10));
    }
}
