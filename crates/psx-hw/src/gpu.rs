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
        if self.contains(Self::TEXPAGE_Y) { 256 } else { 0 }
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
        if self.contains(Self::VRES) { 480 } else { 240 }
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
}
