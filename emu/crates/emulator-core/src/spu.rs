//! SPU — register-bank stub with enough fidelity to match Redux.
//!
//! We model:
//! - `SPUCNT` / `SPUSTAT` (the main control / status pair).
//! - The 24 voice-register banks at `0x1F80_1C00..0x1F80_1D80`.
//!   Writes are stored verbatim; reads return the stored value
//!   except for the per-voice ADSR current-volume register (+0xC),
//!   which returns `0x0001` to match Redux's steady-state.
//!
//! The voice-register accuracy matters because the BIOS polls
//! voice 0's ADSR current volume at ~step 19M during SPU init —
//! a zero-return was the first parity divergence past 10M.
//!
//! Beyond this, the full SPU model (ADPCM decoder, envelope
//! stepping, reverb, DMA) still routes through `io[]` fallbacks.
//! Those land when audio playback actually matters.

/// Physical address of `SPUCNT` — 16-bit control register. Written by
/// BIOS during SPU init (typically `0x8010` — enable + mute reverb).
pub const SPUCNT_ADDR: u32 = 0x1F80_1DAA;
/// Physical address of `SPUSTAT` — 16-bit status register. Its lower
/// 6 bits mirror the current `SPUCNT` bits 0..=5.
pub const SPUSTAT_ADDR: u32 = 0x1F80_1DAE;

/// Base of the 24-voice register bank (16 bytes per voice).
pub const VOICE_BASE: u32 = 0x1F80_1C00;
/// One past the end of the voice bank. `VOICE_BASE + 24 * 16 = 0x1F80_1D80`.
pub const VOICE_END: u32 = 0x1F80_1D80;

/// Per-voice register offsets (0..=15 bytes within each voice block).
mod voice_offset {
    /// +0..1 volume left.
    pub const VOLUME_L: u32 = 0x0;
    /// +2..3 volume right.
    pub const VOLUME_R: u32 = 0x2;
    /// +4..5 ADPCM pitch.
    pub const PITCH: u32 = 0x4;
    /// +6..7 ADPCM start address (in 8-byte SPU words).
    pub const START_ADDR: u32 = 0x6;
    /// +8..9 ADSR config low half.
    pub const ADSR_LO: u32 = 0x8;
    /// +A..B ADSR config high half.
    pub const ADSR_HI: u32 = 0xA;
    /// +C..D ADSR current volume — computed by the SPU. We pin this
    /// to `0x0001` on reads so the BIOS's SPU-init verification sees
    /// a "voice is alive" signal matching Redux.
    pub const ADSR_CURRENT: u32 = 0xC;
    /// +E..F repeat address.
    pub const REPEAT_ADDR: u32 = 0xE;
}

/// Minimal SPU state — control + status + voice register bank.
pub struct Spu {
    spucnt: u16,
    /// 24 voices × 8 half-words = 192 u16s. Flat so we can index
    /// by `voice_index * 8 + half_word_index`.
    voice_regs: [u16; 24 * 8],
}

impl Spu {
    /// Freshly-reset SPU: all registers zero. BIOS writes real values
    /// during SPU init — typically `SPUCNT = 0x8010` (enable + mute
    /// reverb), followed by per-voice volume / pitch / ADSR writes.
    pub fn new() -> Self {
        Self {
            spucnt: 0,
            voice_regs: [0; 24 * 8],
        }
    }

    /// `true` when `phys` falls inside the SPU register region we
    /// model. Everything else still routes through the echo buffer.
    pub fn contains(phys: u32) -> bool {
        phys == SPUCNT_ADDR
            || phys == SPUSTAT_ADDR
            || (VOICE_BASE..VOICE_END).contains(&phys)
    }

    /// Decode a voice-bank address into `(voice_index, byte_offset)`.
    /// Returns `None` when `phys` is outside the voice region.
    fn decode_voice(phys: u32) -> Option<(usize, u32)> {
        if !(VOICE_BASE..VOICE_END).contains(&phys) {
            return None;
        }
        let rel = phys - VOICE_BASE;
        Some(((rel / 16) as usize, rel % 16))
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
        if phys == SPUCNT_ADDR {
            return self.spucnt();
        }
        if phys == SPUSTAT_ADDR {
            return self.spustat();
        }
        if let Some((voice, off)) = Self::decode_voice(phys) {
            // +0xC is ADSR CURRENT VOLUME — computed by the real SPU
            // envelope unit and read-only. Redux initialises it to 1
            // at reset and the BIOS polls it during init; returning 0
            // here was the first parity divergence past 10M.
            if off == voice_offset::ADSR_CURRENT {
                return 0x0001;
            }
            // All other voice registers read back what was last written.
            let idx = voice * 8 + (off as usize) / 2;
            return self.voice_regs[idx];
        }
        0
    }

    /// 32-bit read — zero-extends the 16-bit register into the low
    /// half of the returned word. Upper half is 0 on hardware too.
    pub fn read32(&self, phys: u32) -> u32 {
        self.read16(phys) as u32
    }

    /// 16-bit write. `SPUCNT` is read/write; `SPUSTAT` + the
    /// ADSR-current-volume register are read-only and drop writes.
    pub fn write16(&mut self, phys: u32, value: u16) {
        if phys == SPUCNT_ADDR {
            self.spucnt = value;
            return;
        }
        if let Some((voice, off)) = Self::decode_voice(phys) {
            // Drop writes to the read-only ADSR-current-volume slot.
            if off == voice_offset::ADSR_CURRENT {
                return;
            }
            let idx = voice * 8 + (off as usize) / 2;
            self.voice_regs[idx] = value;
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

    #[test]
    fn voice_bank_contains_24_voices() {
        for v in 0..24u32 {
            assert!(Spu::contains(VOICE_BASE + v * 16));
            assert!(Spu::contains(VOICE_BASE + v * 16 + 0xE));
        }
        assert!(!Spu::contains(VOICE_END));
        assert!(!Spu::contains(VOICE_BASE - 2));
    }

    #[test]
    fn voice_reads_write_through() {
        let mut spu = Spu::new();
        // Voice 3, volume left (+0).
        spu.write16(VOICE_BASE + 3 * 16, 0x1234);
        assert_eq!(spu.read16(VOICE_BASE + 3 * 16), 0x1234);
        // Voice 7, pitch (+4).
        spu.write16(VOICE_BASE + 7 * 16 + 4, 0x0800);
        assert_eq!(spu.read16(VOICE_BASE + 7 * 16 + 4), 0x0800);
    }

    #[test]
    fn voice_adsr_current_volume_reads_one() {
        let mut spu = Spu::new();
        // Every voice's +0xC must read as 0x0001 regardless of what
        // was "written" there (writes to that register are dropped).
        for v in 0..24u32 {
            let addr = VOICE_BASE + v * 16 + 0xC;
            spu.write16(addr, 0xDEAD); // should be dropped
            assert_eq!(spu.read16(addr), 0x0001, "voice {v} ADSR current");
        }
    }

    #[test]
    fn voice_decode_boundaries() {
        assert_eq!(Spu::decode_voice(VOICE_BASE), Some((0, 0)));
        assert_eq!(Spu::decode_voice(VOICE_BASE + 0xF), Some((0, 15)));
        assert_eq!(Spu::decode_voice(VOICE_BASE + 16), Some((1, 0)));
        assert_eq!(Spu::decode_voice(VOICE_BASE + 23 * 16 + 0xE), Some((23, 14)));
        assert_eq!(Spu::decode_voice(VOICE_END), None);
    }
}
