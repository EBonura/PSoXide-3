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

use crate::vram::Vram;

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
}

impl Gpu {
    /// Construct a fresh GPU — VRAM zeroed, status at the soft-GPU
    /// always-ready pattern the BIOS expects.
    pub fn new() -> Self {
        Self {
            vram: Vram::new(),
            status: GpuStatus::new(),
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
                // GP0 command word — Phase 2h accepts and discards.
                // Real command decoding lands when DMA actually ships
                // lists (GPU channel 2 + Phase 2g's scaffolding).
                let _ = value;
                true
            }
            GP1_ADDR => {
                self.status.gp1_write(value);
                true
            }
            _ => false,
        }
    }
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
    fn gpu_address_match_returns_none_off_port() {
        let gpu = Gpu::new();
        assert!(gpu.read32(0x1F80_1800).is_none());
        assert!(gpu.read32(0x1F80_1818).is_none());
    }
}
