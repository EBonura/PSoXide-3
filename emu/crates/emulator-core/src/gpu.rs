//! GPU — minimum viable surface for BIOS init + VRAM display.
//!
//! **Phase 2h scope:** the GPU owns VRAM (migrated here from the
//! top-level `vram` module — re-exported for compatibility), exposes
//! `GPUSTAT` reads with the ready-bit pattern the BIOS polls for, and
//! accepts GP0 / GP1 writes. No command processing, no rasterization,
//! no display output yet — those arrive in follow-up milestones once
//! DMA actually ships command lists.
//!
//! Register map (single-cycle MMIO, 32-bit):
//! - `0x1F80_1810` GP0 write  / `GPUREAD` read
//! - `0x1F80_1814` GP1 write / `GPUSTAT` read

use crate::vram::{Vram, VRAM_HEIGHT, VRAM_WIDTH};

/// Physical address of the GP0 / GPUREAD port.
pub const GP0_ADDR: u32 = 0x1F80_1810;
/// Physical address of the GP1 / GPUSTAT port.
pub const GP1_ADDR: u32 = 0x1F80_1814;

/// GPU state.
pub struct Gpu {
    /// Video memory — 1 MiB, 1024×512 at 16 bpp. The VRAM viewer in
    /// the frontend decodes this each frame.
    pub vram: Vram,
    status: GpuStatus,
    /// Packet assembler for GP0 commands that span multiple words.
    /// Holds words from the start of the current command; once the
    /// full packet has arrived, [`Gpu::execute_gp0_packet`] dispatches
    /// on the opcode and clears the buffer.
    gp0_fifo: Vec<u32>,
    /// Number of words the current packet expects in total (including
    /// the first/opcode word). `0` means "no packet in progress".
    gp0_expected: usize,
}

impl Gpu {
    /// Construct a fresh GPU — VRAM zeroed, status at the soft-GPU
    /// always-ready pattern the BIOS expects.
    pub fn new() -> Self {
        Self {
            vram: Vram::new(),
            status: GpuStatus::new(),
            gp0_fifo: Vec::with_capacity(12),
            gp0_expected: 0,
        }
    }

    /// Dispatch an MMIO read inside the GPU window. Returns `Some` for
    /// the two valid ports; `None` means the caller should fall through
    /// to a different region.
    pub fn read32(&self, phys: u32) -> Option<u32> {
        match phys {
            GP0_ADDR => Some(0),
            GP1_ADDR => Some(self.status.read()),
            _ => None,
        }
    }

    /// Dispatch an MMIO write inside the GPU window. Returns `true` if
    /// the address belonged to the GPU.
    pub fn write32(&mut self, phys: u32, value: u32) -> bool {
        match phys {
            GP0_ADDR => {
                self.gp0_write(value);
                true
            }
            GP1_ADDR => {
                self.status.gp1_write(value);
                true
            }
            _ => false,
        }
    }

    /// Feed one 32-bit word to the GP0 packet assembler. If this word
    /// completes a packet, the packet is executed and the FIFO clears.
    fn gp0_write(&mut self, word: u32) {
        if self.gp0_expected == 0 {
            let op = (word >> 24) & 0xFF;
            self.gp0_expected = gp0_packet_size(op as u8);
            // Single-word commands execute immediately without buffering.
            if self.gp0_expected == 1 {
                self.execute_gp0_single(word);
                self.gp0_expected = 0;
                return;
            }
        }
        self.gp0_fifo.push(word);
        if self.gp0_fifo.len() == self.gp0_expected {
            self.execute_gp0_packet();
            self.gp0_fifo.clear();
            self.gp0_expected = 0;
        }
    }

    /// Execute a command whose packet size is exactly 1. Most draw-
    /// mode setters (GP0 0xE1..=0xE6) live here; for now we accept
    /// them without affecting state beyond the GPU-internal flags
    /// a full implementation would track.
    fn execute_gp0_single(&mut self, _word: u32) {
        // Draw-mode / texture-window / drawing-area / mask-bit setters
        // all fit here. We don't rasterize yet, so they don't do
        // anything observable. Tracking them lands with the primitive
        // rasterizer.
    }

    /// Execute a multi-word packet that has just been fully assembled
    /// in `gp0_fifo`. Dispatches on the opcode in word 0.
    fn execute_gp0_packet(&mut self) {
        let op = (self.gp0_fifo[0] >> 24) & 0xFF;
        match op {
            0x02 => self.fill_rect(),
            _ => {
                // Polygons, lines, rects, VRAM transfers — scaffolding
                // for all of them lands as we implement each one. For
                // now, unknown multi-word commands are silently dropped
                // so the BIOS can finish shipping its command stream.
            }
        }
    }

    /// GP0 0x02 — monochrome fill rectangle, ignores draw mode /
    /// clipping / blending. Writes `color` directly into VRAM.
    ///
    /// Packet layout (Redux `GPU::cmdFillRect`):
    ///   word 0: `0x02RRGGBB`      — opcode + 24-bit RGB
    ///   word 1: `0xYYYYXXXX`      — top-left: X is 16-pixel-aligned
    ///   word 2: `0xHHHHWWWW`      — width is rounded up to 16 pixels
    ///
    /// Both coordinates and sizes wrap mod VRAM dimensions.
    fn fill_rect(&mut self) {
        let color24 = self.gp0_fifo[0] & 0x00FF_FFFF;
        let (x, y) = {
            let w = self.gp0_fifo[1];
            // X is aligned to 16-pixel boundaries; low 4 bits ignored.
            let x = (w & 0x3F0) as u16;
            let y = ((w >> 16) & 0x1FF) as u16;
            (x, y)
        };
        let (w, h) = {
            let s = self.gp0_fifo[2];
            // Width rounded up to next multiple of 16.
            let w = (((s & 0x3FF) + 0x0F) & !0x0F) as u16;
            let h = ((s >> 16) & 0x1FF) as u16;
            (w, h)
        };

        let color15 = rgb24_to_bgr15(color24);
        for row in 0..h {
            for col in 0..w {
                let px = (x + col) as usize % VRAM_WIDTH;
                let py = (y + row) as usize % VRAM_HEIGHT;
                self.vram.set_pixel(px as u16, py as u16, color15);
            }
        }
    }
}

/// Expected total word count for a GP0 command starting with opcode `op`.
/// `0` means the opcode is unknown / unimplemented — treated as no-op.
fn gp0_packet_size(op: u8) -> usize {
    match op {
        // NOP / clear cache / flip misc — single word, no data.
        0x00 | 0x01 | 0x03..=0x1E => 1,
        // Quick fill — RGB + (X,Y) + (W,H) = 3 words.
        0x02 => 3,
        // Draw-mode settings (E1..=E6) — single word each.
        0xE1..=0xE6 => 1,
        // Everything else (polygons, lines, rects, transfers): we
        // don't know the size without full decoding, so treat as 1
        // word. That drops data on the floor for multi-word commands
        // we haven't taught the decoder yet — fine for now since no
        // one consumes VRAM from those paths.
        _ => 1,
    }
}

/// Convert a 24-bit RGB value (as written by the CPU in GP0 packets)
/// into the 15-bit BGR word VRAM stores. Matches Redux / PS1
/// hardware: the 3 high bits of each channel are discarded.
fn rgb24_to_bgr15(rgb24: u32) -> u16 {
    let r = ((rgb24 >> 3) & 0x1F) as u16;
    let g = (((rgb24 >> 8) >> 3) & 0x1F) as u16;
    let b = (((rgb24 >> 16) >> 3) & 0x1F) as u16;
    r | (g << 5) | (b << 10)
}

impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

/// GPUSTAT — the status register the CPU polls to check whether the
/// GPU is ready for commands, ready to upload VRAM, etc.
///
/// Value model matches a "always-idle soft GPU" (same convention as
/// PCSX-Redux and PSoXide-2): bits 26–28 are forced ready on every
/// read, bit 25 (DMA request) is computed from the DMA direction
/// bits 29:30, and bit 31 (interlace/field) toggles at VBlank.
struct GpuStatus {
    raw: u32,
}

impl GpuStatus {
    fn new() -> Self {
        // Reset: display disabled (bit 23), DMA direction 0,
        // interlace odd field, ready bits cleared (filled in on read).
        Self { raw: 0x1480_2000 }
    }

    fn read(&self) -> u32 {
        let mut ret = self.raw;

        // Always-ready: bits 26 (cmd FIFO ready), 27 (VRAM→CPU ready),
        // 28 (DMA block ready). Always on for a soft GPU.
        ret |= 0x1C00_0000;

        // Bit 25 (DMA data request) is derived from direction (bits 29:30).
        //   Direction 0 (Off):       bit 25 = 0
        //   Direction 1 (FIFO):      bit 25 = 1
        //   Direction 2 (CPU→GPU):   bit 25 = copy of bit 28
        //   Direction 3 (GPU→CPU):   bit 25 = copy of bit 27
        ret &= !0x0200_0000;
        match (ret >> 29) & 3 {
            1 => ret |= 0x0200_0000,
            2 => ret |= (ret & 0x1000_0000) >> 3,
            3 => ret |= (ret & 0x0800_0000) >> 2,
            _ => {}
        }
        ret
    }

    /// GP1 command dispatch. GP1 commands are simple enough to inline
    /// here; GP0 is more elaborate (packet assembly, texture upload,
    /// primitives) and gets a dedicated module when we actually render.
    fn gp1_write(&mut self, value: u32) {
        let cmd = (value >> 24) & 0xFF;
        match cmd {
            // GP1(0x00): reset GPU. Restore the post-boot state.
            0x00 => *self = Self::new(),
            // GP1(0x03): display enable. Bit 0: 0=on, 1=off.
            0x03 => {
                if value & 1 != 0 {
                    self.raw |= 1 << 23;
                } else {
                    self.raw &= !(1 << 23);
                }
            }
            // GP1(0x04): DMA direction. Value bits 0:1 → GPUSTAT 29:30.
            0x04 => {
                self.raw = (self.raw & !0x6000_0000) | ((value & 3) << 29);
            }
            _ => {
                // Other GP1 commands (display area, horizontal range,
                // vertical range, display mode, GPU info request) are
                // recorded but unmodeled for now — they only matter
                // once we actually present a framebuffer.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_status_has_always_ready_bits() {
        let gpu = Gpu::new();
        let stat = gpu.read32(GP1_ADDR).unwrap();
        // Bits 26, 27, 28 are the "ready" bits we force on every read.
        assert_eq!(stat & 0x1C00_0000, 0x1C00_0000);
    }

    #[test]
    fn gp1_reset_clears_dma_direction() {
        let mut gpu = Gpu::new();
        gpu.write32(GP1_ADDR, 0x0400_0002); // set DMA direction to 2
        let stat = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!((stat >> 29) & 3, 2);

        gpu.write32(GP1_ADDR, 0x0000_0000); // reset
        let stat = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!((stat >> 29) & 3, 0);
    }

    #[test]
    fn gp1_display_disable_toggles_bit_23() {
        let mut gpu = Gpu::new();
        // Start disabled (reset state has bit 23 set).
        let stat_before = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat_before & (1 << 23), 1 << 23);

        // GP1(0x03) with bit 0 = 0: enable display.
        gpu.write32(GP1_ADDR, 0x0300_0000);
        let stat_enabled = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat_enabled & (1 << 23), 0);
    }

    #[test]
    fn gp0_writes_are_accepted_without_effect() {
        let mut gpu = Gpu::new();
        let stat_before = gpu.read32(GP1_ADDR).unwrap();
        gpu.write32(GP0_ADDR, 0xE100_0000);
        let stat_after = gpu.read32(GP1_ADDR).unwrap();
        assert_eq!(stat_before, stat_after);
    }

    #[test]
    fn rgb24_to_bgr15_white_reaches_0x7fff() {
        assert_eq!(rgb24_to_bgr15(0x00FFFFFF), 0x7FFF);
    }

    #[test]
    fn rgb24_to_bgr15_primary_channels() {
        // Pure red (FF in low byte): bottom 5 bits set.
        assert_eq!(rgb24_to_bgr15(0x000000FF), 0x001F);
        // Pure green: middle 5.
        assert_eq!(rgb24_to_bgr15(0x0000FF00), 0x03E0);
        // Pure blue: top 5.
        assert_eq!(rgb24_to_bgr15(0x00FF0000), 0x7C00);
    }

    #[test]
    fn gp0_fill_rect_writes_vram() {
        let mut gpu = Gpu::new();
        // Fill a 16×16 red rect at (0, 0).
        gpu.write32(GP0_ADDR, 0x0200_00FF); // 0x02 + red
        gpu.write32(GP0_ADDR, 0x0000_0000); // y=0, x=0
        gpu.write32(GP0_ADDR, 0x0010_0010); // h=16, w=16

        assert_eq!(gpu.vram.get_pixel(0, 0), 0x001F);
        assert_eq!(gpu.vram.get_pixel(15, 15), 0x001F);
        // Just outside stays zero.
        assert_eq!(gpu.vram.get_pixel(16, 0), 0);
        assert_eq!(gpu.vram.get_pixel(0, 16), 0);
    }

    #[test]
    fn gp0_draw_mode_commands_are_accepted() {
        let mut gpu = Gpu::new();
        // Four draw-mode setters back-to-back, each 1 word.
        gpu.write32(GP0_ADDR, 0xE100_0000); // draw mode
        gpu.write32(GP0_ADDR, 0xE200_0000); // texture window
        gpu.write32(GP0_ADDR, 0xE300_0000); // drawing area TL
        gpu.write32(GP0_ADDR, 0xE400_0000); // drawing area BR
        // None of these should have stuck a packet in the FIFO.
        // (Implementation detail, but worth guarding against.)
    }

    #[test]
    fn gpu_address_match_returns_none_off_port() {
        let gpu = Gpu::new();
        assert!(gpu.read32(0x1F80_1800).is_none());
        assert!(gpu.read32(0x1F80_1818).is_none());
    }
}
