//! SPU — register-bank stub with voice tracking.
//!
//! This is **not** a real audio subsystem. No ADPCM decode, no
//! reverb, no output. What it *is*:
//!
//! - Full 512-byte MMIO register bank read/write, byte/halfword/
//!   word. Every addressable register stores what's written and
//!   reads it back — the only reason games would care about our
//!   accuracy at this layer is either "does the register bank
//!   echo correctly" or "does the voice-end bit (ENDX) flip on
//!   eventually".
//! - Per-voice ADSR-current-volume read (`+0xC`) pinned at `0x0001`
//!   to mirror Redux's reset state — the BIOS's SPU-init test
//!   checks this at ~step 19M.
//! - **Simulated ENDX** (voice-ended). Games like Tekken start a
//!   set of voices on KON, then poll ENDX to know when a
//!   one-shot sample finishes. Without this, they wait forever.
//!   We track the cycle count at which each voice was key-on'd
//!   and set its ENDX bit once a heuristic "long enough for a
//!   typical sample" has passed. Not sample-accurate, but
//!   unblocks the "poll ENDX until non-zero" wait.
//!
//! The envelope / ADPCM / reverb / output work stays deferred
//! until actual audio matters.

/// Physical address range [`Spu::BASE` .. `Spu::END`).
/// `0x1F80_1C00 .. 0x1F80_1E00` — the entire 512-byte SPU MMIO
/// block (voices + global regs + reverb config).
///
/// Voices sit at the start of this range; global registers
/// (KON / KOFF / ENDX / SPUCNT / …) live in the upper 128 bytes.

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

/// Key-on register — low half (voices 0..=15). 16-bit. Writing a 1
/// bit to voice N starts that voice.
pub const KON_LO_ADDR: u32 = 0x1F80_1D88;
/// Key-on register — high half (voices 16..=23 in bits 0..=7). 16-bit.
pub const KON_HI_ADDR: u32 = 0x1F80_1D8A;
/// Key-off register — low half. Writing a 1 to voice N triggers
/// its release phase; ADSR envelope drops to zero.
pub const KOFF_LO_ADDR: u32 = 0x1F80_1D8C;
/// Key-off register — high half.
pub const KOFF_HI_ADDR: u32 = 0x1F80_1D8E;
/// ENDX low half — per-voice "reached end of sample" flag. Set by
/// the SPU when a voice's ADPCM pointer hits an end block with
/// the loop-end flag set. Cleared by software writes.
pub const ENDX_LO_ADDR: u32 = 0x1F80_1D9C;
/// ENDX high half.
pub const ENDX_HI_ADDR: u32 = 0x1F80_1D9E;

/// Approximate system-clock cycles the SPU takes to play a
/// typical one-shot sample through its ADSR envelope. This is a
/// heuristic for our ENDX simulation — not tied to any
/// particular sample length. ~33.8 MHz / 44.1 kHz = 767 cycles
/// per sample; a mid-length sample (~4 KiB of ADPCM = ~7 kHz
/// samples = ~0.16s) → ~5.4M cycles.
///
/// Games that poll ENDX for "voice done" will eventually see
/// their bit flip after this many cycles. Games that don't care
/// about precise timing (most of them — the wait is usually "has
/// at least some time passed") will advance.
const VOICE_ENDX_LATENCY_CYCLES: u64 = 5_000_000;

/// Per-voice register offsets (0..=15 bytes within each voice block).
/// Only `ADSR_CURRENT` is referenced today (the BIOS polls it); the
/// rest are documentation placeholders that will become live entry
/// points once SPU audio synthesis lands.
#[allow(dead_code)]
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

/// SPU state — register bank + voice-on bookkeeping so ENDX can
/// simulate "long enough to be done."
pub struct Spu {
    /// 512-byte flat MMIO block. Every read / write hits this
    /// unless a specific address has custom behaviour (SPUCNT ↔
    /// SPUSTAT mirror, voice `+0xC` pinned at 0x0001, ENDX
    /// derived from `voice_on_cycle`).
    regs: [u8; 0x200],
    /// Cycle count at which each voice was last keyed on. `u64::MAX`
    /// means "never started" or "already finished" — in either case
    /// the voice's ENDX bit reads as 0 until another KON bumps it.
    voice_on_cycle: [u64; 24],
    /// ENDX bitmap software has observed and acknowledged. Real
    /// hardware clears ENDX bits when software writes 1s; we mirror
    /// that by tracking the latched bitmap here and OR-ing in any
    /// bits whose simulated latency has now elapsed.
    endx_latched: u32,
}

impl Spu {
    /// Low edge of the SPU MMIO range.
    pub const BASE: u32 = 0x1F80_1C00;
    /// High edge (exclusive). 0x200 bytes total.
    pub const END: u32 = 0x1F80_1E00;

    /// Freshly-reset SPU: all registers zero. BIOS writes real values
    /// during SPU init — typically `SPUCNT = 0x8010` (enable + mute
    /// reverb), followed by per-voice volume / pitch / ADSR writes.
    pub fn new() -> Self {
        Self {
            regs: [0; 0x200],
            voice_on_cycle: [u64::MAX; 24],
            endx_latched: 0,
        }
    }

    /// `true` when `phys` falls inside the SPU register region.
    pub fn contains(phys: u32) -> bool {
        (Self::BASE..Self::END).contains(&phys)
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

    /// Read-only register values that override the backing store.
    /// Returns `Some` for slots with custom semantics, `None` to
    /// fall through to the raw `regs[]` byte buffer.
    fn custom_read16(&self, phys: u32, now_cycles: u64) -> Option<u16> {
        // SPUCNT stored in regs; SPUSTAT is derived from SPUCNT.
        if phys == SPUSTAT_ADDR {
            return Some(self.spustat());
        }
        // ENDX halves: derive from voice_on_cycle + latency.
        if phys == ENDX_LO_ADDR {
            return Some(self.compute_endx(now_cycles) as u16);
        }
        if phys == ENDX_HI_ADDR {
            return Some((self.compute_endx(now_cycles) >> 16) as u16);
        }
        // Voice ADSR current (+0xC) — pinned at 0x0001 for parity.
        if let Some((_, off)) = Self::decode_voice(phys) {
            if off == voice_offset::ADSR_CURRENT {
                return Some(0x0001);
            }
        }
        None
    }

    /// Current `SPUCNT` value.
    pub fn spucnt(&self) -> u16 {
        self.read_raw_u16(SPUCNT_ADDR)
    }

    /// Current `SPUSTAT` value — lower 6 bits mirror `SPUCNT`. Other
    /// SPUSTAT bits (IRQ9, DMA busy, capture-half flags, …) aren't
    /// modelled yet; they read as zero, which matches "nothing is
    /// happening" — accurate while no voices are active.
    pub fn spustat(&self) -> u16 {
        self.spucnt() & 0x3F
    }

    /// Compute the ENDX 32-bit bitmap at the given cycle count.
    /// Every voice whose time since KON exceeds the latency threshold
    /// contributes a 1 bit; the latched bitmap carries over any bits
    /// from earlier ticks that software hasn't acknowledged yet.
    fn compute_endx(&self, now_cycles: u64) -> u32 {
        let mut bitmap = self.endx_latched;
        for (voice, &on_cycle) in self.voice_on_cycle.iter().enumerate() {
            if on_cycle == u64::MAX {
                continue;
            }
            if now_cycles.saturating_sub(on_cycle) >= VOICE_ENDX_LATENCY_CYCLES {
                bitmap |= 1u32 << voice;
            }
        }
        bitmap
    }

    /// 16-bit read, honouring custom registers. `now_cycles` is the
    /// current bus cycle count — only used for ENDX.
    pub fn read16_at(&self, phys: u32, now_cycles: u64) -> u16 {
        if let Some(v) = self.custom_read16(phys, now_cycles) {
            return v;
        }
        self.read_raw_u16(phys)
    }

    /// Back-compat shim: same as [`read16_at`] but with an implicit
    /// `now_cycles = 0`. Safe for anything except ENDX; callers that
    /// want correct ENDX must use [`read16_at`].
    pub fn read16(&self, phys: u32) -> u16 {
        self.read16_at(phys, 0)
    }

    /// 32-bit read — zero-extends the 16-bit register into the low
    /// half of the returned word. Upper half is 0 on hardware too.
    pub fn read32(&self, phys: u32) -> u32 {
        self.read16(phys) as u32
    }

    /// 32-bit read with cycle context — used for ENDX-sensitive paths.
    pub fn read32_at(&self, phys: u32, now_cycles: u64) -> u32 {
        self.read16_at(phys, now_cycles) as u32
    }

    /// 16-bit write. Most writes land in `regs[]` verbatim; a few
    /// registers have side effects:
    ///
    /// - **KON**: each 1 bit starts the corresponding voice
    ///   (records the KON cycle so ENDX eventually reports "done").
    /// - **KOFF**: each 1 bit ends the voice immediately (clears its
    ///   KON timer so ENDX no longer flips for it).
    /// - **ENDX**: write-1-to-clear. Any bit the software writes as 1
    ///   drops from the latched bitmap; our on-the-fly `compute_endx`
    ///   still reports "currently done" for voices that are still
    ///   past their latency threshold, matching hardware.
    pub fn write16_at(&mut self, phys: u32, value: u16, now_cycles: u64) {
        match phys {
            KON_LO_ADDR => self.handle_kon(value, 0, now_cycles),
            KON_HI_ADDR => self.handle_kon(value, 16, now_cycles),
            KOFF_LO_ADDR => self.handle_koff(value, 0),
            KOFF_HI_ADDR => self.handle_koff(value, 16),
            ENDX_LO_ADDR => {
                self.endx_latched &= !(value as u32);
            }
            ENDX_HI_ADDR => {
                self.endx_latched &= !((value as u32) << 16);
            }
            _ => {
                // Drop writes to the read-only voice ADSR current
                // volume slot so reads keep returning 0x0001.
                if let Some((_, off)) = Self::decode_voice(phys) {
                    if off == voice_offset::ADSR_CURRENT {
                        return;
                    }
                }
            }
        }
        // Whether or not it had side effects, store the value so
        // reads echo it back (KON / KOFF registers are readable).
        self.write_raw_u16(phys, value);
    }

    /// Back-compat shim.
    pub fn write16(&mut self, phys: u32, value: u16) {
        self.write16_at(phys, value, 0);
    }

    /// 32-bit write — only the low half matters.
    pub fn write32(&mut self, phys: u32, value: u32) {
        self.write16(phys, value as u16);
    }

    /// 32-bit write with cycle context.
    pub fn write32_at(&mut self, phys: u32, value: u32, now_cycles: u64) {
        self.write16_at(phys, value as u16, now_cycles);
    }

    /// 8-bit read for the rare byte-wide access. Splits on even/odd
    /// addresses and pulls the right half of the underlying word.
    pub fn read8(&self, phys: u32) -> u8 {
        if let Some(idx) = Self::reg_index(phys) {
            self.regs[idx]
        } else {
            0
        }
    }

    /// 8-bit write.
    pub fn write8(&mut self, phys: u32, value: u8) {
        if let Some(idx) = Self::reg_index(phys) {
            self.regs[idx] = value;
        }
    }

    fn reg_index(phys: u32) -> Option<usize> {
        if (Self::BASE..Self::END).contains(&phys) {
            Some((phys - Self::BASE) as usize)
        } else {
            None
        }
    }

    fn read_raw_u16(&self, phys: u32) -> u16 {
        let Some(idx) = Self::reg_index(phys) else {
            return 0;
        };
        if idx + 1 >= self.regs.len() {
            return 0;
        }
        u16::from_le_bytes([self.regs[idx], self.regs[idx + 1]])
    }

    fn write_raw_u16(&mut self, phys: u32, value: u16) {
        let Some(idx) = Self::reg_index(phys) else {
            return;
        };
        if idx + 1 >= self.regs.len() {
            return;
        }
        let bytes = value.to_le_bytes();
        self.regs[idx] = bytes[0];
        self.regs[idx + 1] = bytes[1];
    }

    fn handle_kon(&mut self, value: u16, base_voice: usize, now_cycles: u64) {
        for bit in 0..16usize {
            if value & (1 << bit) != 0 {
                let v = base_voice + bit;
                if v < 24 {
                    self.voice_on_cycle[v] = now_cycles;
                    // A fresh KON clears the previous ENDX for that
                    // voice — matches hardware behaviour where
                    // restarting playback resets the "done" flag.
                    self.endx_latched &= !(1u32 << v);
                }
            }
        }
    }

    fn handle_koff(&mut self, value: u16, base_voice: usize) {
        // KOFF immediately ends the voice; ENDX doesn't flip here
        // (hardware uses it for loop-end detection, not for KOFF),
        // but we clear voice_on_cycle so a later ENDX read doesn't
        // still report "not done yet" for this voice.
        for bit in 0..16usize {
            if value & (1 << bit) != 0 {
                let v = base_voice + bit;
                if v < 24 {
                    self.voice_on_cycle[v] = u64::MAX;
                }
            }
        }
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
    fn contains_covers_whole_range() {
        assert!(Spu::contains(Spu::BASE));
        assert!(Spu::contains(0x1F80_1DFF));
        assert!(!Spu::contains(Spu::END));
        assert!(!Spu::contains(Spu::BASE - 1));
    }

    #[test]
    fn spucnt_round_trip() {
        let mut s = Spu::new();
        s.write16(SPUCNT_ADDR, 0x8010);
        assert_eq!(s.spucnt(), 0x8010);
        assert_eq!(s.spustat(), 0x0010);
    }

    #[test]
    fn spustat_is_read_only() {
        let mut s = Spu::new();
        s.write16(SPUCNT_ADDR, 0x8010);
        s.write16(SPUSTAT_ADDR, 0xFFFF); // hardware drops writes
        assert_eq!(s.spustat(), 0x0010);
    }

    #[test]
    fn voice_bank_echoes() {
        let mut s = Spu::new();
        s.write16(VOICE_BASE + 3 * 16 + voice_offset::PITCH, 0x0800);
        assert_eq!(s.read16(VOICE_BASE + 3 * 16 + voice_offset::PITCH), 0x0800);
    }

    #[test]
    fn voice_adsr_current_volume_reads_one() {
        let mut s = Spu::new();
        for v in 0..24u32 {
            let addr = VOICE_BASE + v * 16 + voice_offset::ADSR_CURRENT;
            s.write16(addr, 0xDEAD);
            assert_eq!(s.read16(addr), 0x0001, "voice {v}");
        }
    }

    #[test]
    fn kon_sets_endx_after_latency() {
        let mut s = Spu::new();
        // Start voices 0 and 5.
        s.write16_at(KON_LO_ADDR, 0x0021, 0);
        // Right after KON, ENDX is 0 — voices haven't "finished" yet.
        assert_eq!(s.read16_at(ENDX_LO_ADDR, 100), 0);
        // Past the latency threshold they read as done.
        let past = VOICE_ENDX_LATENCY_CYCLES + 1;
        assert_eq!(s.read16_at(ENDX_LO_ADDR, past), 0x0021);
        // High half stays 0 — we only keyed on low voices.
        assert_eq!(s.read16_at(ENDX_HI_ADDR, past), 0);
    }

    #[test]
    fn endx_clears_on_write_one() {
        let mut s = Spu::new();
        s.write16_at(KON_LO_ADDR, 0x0001, 0);
        let past = VOICE_ENDX_LATENCY_CYCLES + 1;
        assert_eq!(s.read16_at(ENDX_LO_ADDR, past), 0x0001);
        // Software acks by writing 1.
        s.write16_at(ENDX_LO_ADDR, 0x0001, past);
        // Voice 0's KON timer is still active, so ENDX reports
        // done again — matches hardware (ENDX "stays set" as long
        // as voice is past end; software must KOFF to clear
        // permanently).
        assert_eq!(s.read16_at(ENDX_LO_ADDR, past), 0x0001);
    }

    #[test]
    fn koff_stops_endx_flipping() {
        let mut s = Spu::new();
        s.write16_at(KON_LO_ADDR, 0x0002, 0);
        s.write16_at(KOFF_LO_ADDR, 0x0002, 1_000);
        let past = VOICE_ENDX_LATENCY_CYCLES + 10_000;
        // Voice 1 was KOFF'd before latency elapsed — ENDX stays 0.
        assert_eq!(s.read16_at(ENDX_LO_ADDR, past) & 0x0002, 0);
    }

    #[test]
    fn kon_high_half_targets_voices_16_to_23() {
        let mut s = Spu::new();
        // Key on voice 23 (bit 7 of high KON = voice 23).
        s.write16_at(KON_HI_ADDR, 1 << 7, 0);
        let past = VOICE_ENDX_LATENCY_CYCLES + 1;
        assert_eq!(s.read16_at(ENDX_HI_ADDR, past), 1 << 7);
    }

    #[test]
    fn global_register_writes_echo() {
        let mut s = Spu::new();
        // Main volume L / R, reverb volume, etc. all just echo.
        s.write16(0x1F80_1D80, 0x3FFF);
        s.write16(0x1F80_1D82, 0x2FFF);
        assert_eq!(s.read16(0x1F80_1D80), 0x3FFF);
        assert_eq!(s.read16(0x1F80_1D82), 0x2FFF);
    }

    #[test]
    fn byte_access_halves_work() {
        let mut s = Spu::new();
        s.write16(SPUCNT_ADDR, 0x8010);
        assert_eq!(s.read8(SPUCNT_ADDR), 0x10);
        assert_eq!(s.read8(SPUCNT_ADDR + 1), 0x80);
    }
}
