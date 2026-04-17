//! SPU — read-only status stub.
//!
//! **Phase 3a scope.** Just enough of the SPU to unblock the parity
//! harness past step 2,735,076, where Redux returns a non-zero
//! `SPUSTAT` (`0x1F80_1DAE`) because its lower 6 bits mirror the last
//! `SPUCNT` (`0x1F80_1DAA`) write. Our previous echo buffer had
//! `SPUSTAT` at zero because nothing ever wrote that specific address
//! — it's a read-only status register that tracks another register.
//!
//! Everything else in SPU register space keeps flowing through the
//! MMIO echo buffer. The full SPU model (24 voices, ADSR, reverb,
//! ADPCM decoder, DMA) lands with Milestone F / H when audio actually
//! matters.

/// Physical address of `SPUCNT` — 16-bit control register. Written by
/// BIOS during SPU init (typically `0x8010` — enable + mute reverb).
pub const SPUCNT_ADDR: u32 = 0x1F80_1DAA;
/// Physical address of `SPUSTAT` — 16-bit status register. Its lower
/// 6 bits mirror the current `SPUCNT` bits 0..=5.
pub const SPUSTAT_ADDR: u32 = 0x1F80_1DAE;

/// Minimal SPU state — just the control register. Other SPU MMIO
/// still routes through the bus echo buffer for now.
pub struct Spu {
    spucnt: u16,
}

impl Spu {
    /// Freshly-reset SPU: `SPUCNT = 0`. BIOS writes a real value
    /// during init — typically `0x8010` (enable + mute reverb).
    pub fn new() -> Self {
        Self { spucnt: 0 }
    }

    /// `true` when `phys` is one of the two register addresses this
    /// stub handles. Everything else keeps going to the echo buffer.
    pub fn contains(phys: u32) -> bool {
        phys == SPUCNT_ADDR || phys == SPUSTAT_ADDR
    }

    /// Current `SPUCNT` value (last write echoed back on read).
    pub fn spucnt(&self) -> u16 {
        self.spucnt
    }

    /// Current `SPUSTAT` value — lower 6 bits mirror `SPUCNT`. Other
    /// SPUSTAT bits (IRQ9, DMA busy, capture-half flags, …) aren't
    /// modelled yet; they read as zero, which matches "nothing is
    /// happening" — accurate while no voices are active.
    pub fn spustat(&self) -> u16 {
        self.spucnt & 0x3F
    }

    /// 16-bit read — the native width for SPU MMIO.
    pub fn read16(&self, phys: u32) -> u16 {
        match phys {
            SPUCNT_ADDR => self.spucnt(),
            SPUSTAT_ADDR => self.spustat(),
            _ => 0,
        }
    }

    /// 32-bit read — zero-extends the 16-bit register into the low
    /// half of the returned word. Upper half is 0 on hardware too.
    pub fn read32(&self, phys: u32) -> u32 {
        self.read16(phys) as u32
    }

    /// 16-bit write. `SPUCNT` is read/write; `SPUSTAT` is read-only
    /// and drops writes on real hardware.
    pub fn write16(&mut self, phys: u32, value: u16) {
        if phys == SPUCNT_ADDR {
            self.spucnt = value;
        }
    }

    /// 32-bit write — only the low half matters.
    pub fn write32(&mut self, phys: u32, value: u32) {
        self.write16(phys, value as u16);
    }
}

impl Default for Spu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_hits_both_registers() {
        assert!(Spu::contains(SPUCNT_ADDR));
        assert!(Spu::contains(SPUSTAT_ADDR));
        assert!(!Spu::contains(0x1F80_1DA8));
    }

    #[test]
    fn spustat_mirrors_low_six_bits_of_spucnt() {
        let mut spu = Spu::new();
        spu.write16(SPUCNT_ADDR, 0x8010);
        assert_eq!(spu.spucnt(), 0x8010);
        assert_eq!(spu.spustat(), 0x0010);
    }

    #[test]
    fn spustat_is_read_only() {
        let mut spu = Spu::new();
        spu.write16(SPUCNT_ADDR, 0x8010);
        spu.write16(SPUSTAT_ADDR, 0xFFFF); // dropped on real hw
        assert_eq!(spu.spustat(), 0x0010);
    }

    #[test]
    fn read32_zero_extends_spucnt() {
        let mut spu = Spu::new();
        spu.write16(SPUCNT_ADDR, 0x8010);
        assert_eq!(spu.read32(SPUCNT_ADDR), 0x0000_8010);
    }
}
