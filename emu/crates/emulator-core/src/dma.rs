//! DMA controller — 7 channels plus global control registers.
//!
//! Channel layout (each is 16 bytes at `0x1F80_1080 + 0x10 * ch`):
//! - `+0x0` base address (RAM address the channel transfers to/from)
//! - `+0x4` block control (count + block size for block/list modes)
//! - `+0x8` channel control (direction, sync mode, start trigger, …)
//!
//! Global:
//! - `0x1F80_10F0` DPCR — per-channel enables + priority bits
//! - `0x1F80_10F4` DICR — IRQ enables + pending flags
//!
//! **Phase 2g scope:** register backing + MMIO dispatch only. No actual
//! transfers fire yet. When a channel's `channel_control` start bit is
//! written, we record the intent but don't move bytes — the channel's
//! owning subsystem (GPU for ch 2, SPU for ch 4, CD-ROM for ch 3, …)
//! isn't online. OTC (ch 6) is self-contained and will be the first
//! real transfer path; it lands in the follow-up commit that wires
//! ticking.
//!
//! Channel identity (for the `IrqSource::Dma` side):
//!
//!
//! | ch | consumer     |
//! |----|--------------|
//! | 0  | MDEC-in      |
//! | 1  | MDEC-out     |
//! | 2  | GPU          |
//! | 3  | CD-ROM       |
//! | 4  | SPU          |
//! | 5  | PIO          |
//! | 6  | OTC          |

/// Number of DMA channels.
pub const NUM_CHANNELS: usize = 7;

/// Per-channel register state.
#[derive(Default, Clone, Copy)]
pub struct DmaChannel {
    /// `MADR` — base address (destination for RAM-bound transfers,
    /// source otherwise). Lower 24 bits are the physical RAM offset;
    /// upper bits read back as written but aren't consulted during
    /// transfers.
    pub base: u32,
    /// `BCR` — block control. Format depends on sync mode; for block
    /// mode, lower 16 bits are block size and upper 16 bits are count.
    pub block_control: u32,
    /// `CHCR` — channel control (direction / step / sync / start).
    pub channel_control: u32,
}

/// Global controller state + 7 channels.
pub struct Dma {
    /// Per-channel register blocks.
    pub channels: [DmaChannel; NUM_CHANNELS],
    /// `DPCR` — DMA Primary Control Register (per-channel enables +
    /// priority). Reset value is `0x0765_4321` on hardware but the BIOS
    /// writes it early so the reset default doesn't matter for us.
    pub dpcr: u32,
    /// `DICR` — DMA Interrupt Register (IRQ enable + pending bits).
    pub dicr: u32,
    /// Per-channel count of CHCR writes with the start bit set.
    /// Diagnostic only — tells us which channels software is using
    /// and how often. Cleared on `new()`, never decremented.
    pub start_trigger_counts: [u64; NUM_CHANNELS],
    /// Mirror of `start_trigger_counts` that survives cargo-naming
    /// nuance — a second counter bumped in the same place so a
    /// probe can distinguish "channel never touched" from "channel
    /// touched but we haven't picked it up."
    pub chcr_write_count: [u64; NUM_CHANNELS],
}

impl Dma {
    /// Low edge of the controller's MMIO range.
    pub const BASE: u32 = 0x1F80_1080;
    /// High edge (exclusive). DPCR sits at `BASE + 0x70`, DICR at
    /// `BASE + 0x74`; we round the range up to the next 16-byte
    /// boundary so the dispatch check is a single `contains`.
    pub const END: u32 = 0x1F80_10F8;
    /// Channel stride within the DMA window.
    pub const STRIDE: u32 = 0x10;
    /// Offset of DPCR from [`Dma::BASE`].
    pub const DPCR_OFFSET: u32 = 0x70;
    /// Offset of DICR from [`Dma::BASE`].
    pub const DICR_OFFSET: u32 = 0x74;

    /// All channels + both global regs cleared.
    pub fn new() -> Self {
        Self {
            channels: [DmaChannel::default(); NUM_CHANNELS],
            dpcr: 0,
            dicr: 0,
            start_trigger_counts: [0; NUM_CHANNELS],
            chcr_write_count: [0; NUM_CHANNELS],
        }
    }

    /// `true` when `phys` falls inside `BASE..END`.
    pub fn contains(phys: u32) -> bool {
        (Self::BASE..Self::END).contains(&phys)
    }

    /// `true` when DPCR has channel `ch` enabled. Mirrors Redux's
    /// `isDMAEnabled<n>()` in `psxmem.h`: the enable bit for
    /// channel N is DPCR bit `N*4 + 3`. Software writes DPCR to
    /// selectively enable/disable channels before starting a
    /// transfer — if the channel is off, the CHCR start-bit write
    /// MUST NOT kick the DMA. Skipping this check lets Crash's
    /// intro issue CHCR writes that Redux ignores, producing extra
    /// DMA completions on our side and a +2488-cycle drift in the
    /// first fold after the Sony logo.
    pub fn is_channel_enabled(&self, ch: usize) -> bool {
        debug_assert!(ch < NUM_CHANNELS);
        let bit = 1u32 << (ch * 4 + 3);
        self.dpcr & bit != 0
    }

    /// Read a 32-bit word. `phys` must be inside [`Dma::BASE`]..[`Dma::END`].
    pub fn read32(&self, phys: u32) -> u32 {
        let offset = phys - Self::BASE;
        match offset {
            Self::DPCR_OFFSET => self.dpcr,
            Self::DICR_OFFSET => self.dicr_with_master_flag(),
            _ => {
                let (ch, field) = decode(phys);
                if ch >= NUM_CHANNELS {
                    return 0;
                }
                let c = &self.channels[ch];
                match field {
                    0x0 => c.base,
                    0x4 => c.block_control,
                    0x8 => c.channel_control,
                    _ => 0,
                }
            }
        }
    }

    /// Write a 32-bit word. `phys` must be inside [`Dma::BASE`]..[`Dma::END`].
    ///
    /// Returns `true` when a DICR write transitions the DMA master IRQ
    /// flag from clear to set, matching Redux's `psxhw.cc` side effect.
    /// Channel-control start bits are recorded here; transfer execution
    /// is still owned by the bus/peripheral layer.
    pub fn write32(&mut self, phys: u32, value: u32) -> bool {
        let offset = phys - Self::BASE;
        match offset {
            Self::DPCR_OFFSET => {
                self.dpcr = value;
                false
            }
            Self::DICR_OFFSET => self.write_dicr(value),
            _ => {
                let (ch, field) = decode(phys);
                if ch >= NUM_CHANNELS {
                    return false;
                }
                let c = &mut self.channels[ch];
                match field {
                    0x0 => c.base = value & 0x00FF_FFFF,
                    0x4 => c.block_control = value,
                    0x8 => {
                        c.channel_control = value;
                        if (value >> 24) & 1 != 0 {
                            self.start_trigger_counts[ch] += 1;
                            self.chcr_write_count[ch] = self.chcr_write_count[ch].saturating_add(1);
                        }
                    }
                    _ => {}
                }
                false
            }
        }
    }
}

impl Default for Dma {
    fn default() -> Self {
        Self::new()
    }
}

// --- DICR semantics ---
//
// Layout (Redux/PCSX-style):
//   bits  0..5  : Unknown (R/W)
//   bits  6..14 : Reserved (always 0)
//   bit   15    : Bus-error flag
//   bits 16..22 : Per-channel IRQ enable (R/W) — DMA0..DMA6
//   bit   23    : IRQ master enable (R/W)
//   bits 24..30 : Per-channel IRQ flag (R, W1C) — DMA0..DMA6
//   bit   31    : IRQ master flag (stored by Redux, not purely derived)
//
// Redux stores bit 31 explicitly and raises IRQ 8 from two places:
// DMA completion and CPU writes to DICR that make the master flag
// transition. Matching that detail matters because BIOS code can leave
// a per-channel flag pending while toggling the master enable later.
impl Dma {
    const DICR_RW_MASK: u32 = 0x00FF_003F;
    const DICR_FLAG_MASK: u32 = 0x7F00_0000;
    const DICR_MASTER_ENABLE: u32 = 1 << 23;
    const DICR_MASTER_FLAG: u32 = 1 << 31;
    const DICR_ERROR_OR_FLAGS: u32 = 0x7F00_8000;

    fn write_dicr(&mut self, value: u32) -> bool {
        // Direct port of Redux `psxhw.cc` DICR write handling:
        // - bits 24..30 are write-1-to-clear flags
        // - bits 0..5 and 16..23 are regular writable state
        // - writing master-enable while a flag is pending raises IRQ 8
        let mut icr = self.dicr;
        let ack_mask = (value & Self::DICR_FLAG_MASK) ^ Self::DICR_FLAG_MASK;
        let was_not_triggered = icr & Self::DICR_MASTER_FLAG == 0;
        let has_error = value & 0x0000_8000 != 0;
        let is_enabled = value & Self::DICR_MASTER_ENABLE != 0;

        icr &= ack_mask;
        icr |= value & Self::DICR_RW_MASK;

        let mut triggered = false;
        if (icr & Self::DICR_ERROR_OR_FLAGS) != 0 && (has_error || is_enabled) {
            icr |= Self::DICR_MASTER_FLAG;
            triggered = true;
        }
        self.dicr = icr;
        was_not_triggered && triggered
    }

    fn dicr_with_master_flag(&self) -> u32 {
        self.dicr
    }

    /// Notify the DICR that DMA channel `ch` has completed. Sets the
    /// channel's IRQ flag (bit `24+ch`) when the matching enable bit
    /// (`16+ch`) is set, and returns `true` when the master IRQ flag
    /// transitions from clear to set as a result — that's the edge the
    /// main IRQ controller should treat as a `Dma` raise.
    pub fn notify_channel_done(&mut self, ch: usize) -> bool {
        if ch >= NUM_CHANNELS {
            return false;
        }
        if self.dicr & Self::DICR_MASTER_ENABLE == 0 {
            return false;
        }
        let enable_bit = 1 << (16 + ch);
        if self.dicr & enable_bit == 0 {
            return false;
        }
        let prev_master = self.dicr & Self::DICR_MASTER_FLAG != 0;
        self.dicr |= 1 << (24 + ch);
        self.dicr |= Self::DICR_MASTER_FLAG;
        !prev_master
    }
}

// --- Transfer execution ---

impl Dma {
    /// Run the OTC (channel 6) transfer if its start bit is set.
    ///
    /// OTC builds an ordering-table linked list with `MADR` as the
    /// **head** (highest address) and the terminator at the tail
    /// (`MADR - (count-1)*4`, the lowest address). Each non-terminator
    /// word holds a 24-bit pointer to the next address, descending by
    /// 4 each step.
    ///
    /// Mirrors Redux's `dma6` (psxdma.cc:113-119):
    /// ```c
    /// while (bcr--) { *mem-- = (madr - 4) & 0xffffff; madr -= 4; }
    /// mem++;        *mem = 0xffffff;   // overwrites last chain ptr
    /// ```
    /// Crucially, the terminator overwrites the chain pointer the loop
    /// just wrote at the lowest address — so the structure ends up:
    /// `madr → madr-4 → … → madr-(count-2)*4 → terminator`.
    ///
    /// CHCR start (24) and busy (28) bits are NOT cleared here — the
    /// caller schedules a delayed completion (Redux's
    /// `scheduleGPUOTCDMAIRQ(size)`) so BIOS polls of CHCR keep
    /// observing the busy state for `size` cycles after kickoff.
    ///
    /// Returns the word count transferred (0 if start bit was clear).
    /// Called by [`crate::Bus`] after every CHCR write.
    pub fn run_otc(&mut self, ram: &mut [u8]) -> u32 {
        let ch = &self.channels[6];
        if (ch.channel_control >> 24) & 1 == 0 {
            return 0;
        }

        let base = ch.base & 0x001F_FFFC;
        let count = match ch.block_control & 0xFFFF {
            0 => 0x1_0000, // hardware: 0 means 65536
            n => n,
        };

        // Chain pointers: at address `base - i*4`, write a pointer
        // to `base - (i+1)*4` (the address one step further down).
        let mut addr = base;
        for _ in 0..count {
            let next = addr.wrapping_sub(4) & 0x00FF_FFFF;
            let offset = (addr & 0x001F_FFFF) as usize;
            if offset + 4 <= ram.len() {
                ram[offset..offset + 4].copy_from_slice(&next.to_le_bytes());
            }
            addr = addr.wrapping_sub(4);
        }
        // Overwrite the last chain entry (at the lowest address) with
        // the terminator. `addr` after the loop is one step past the
        // last write, so the tail is `addr + 4`.
        let tail = addr.wrapping_add(4) & 0x001F_FFFF;
        let offset = tail as usize;
        if offset + 4 <= ram.len() {
            ram[offset..offset + 4].copy_from_slice(&0x00FF_FFFFu32.to_le_bytes());
        }

        count
    }
}

fn decode(phys: u32) -> (usize, u32) {
    let rel = phys - Dma::BASE;
    let ch = (rel / Dma::STRIDE) as usize;
    let field = rel % Dma::STRIDE;
    (ch, field)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_matches_mmio_range() {
        assert!(Dma::contains(0x1F80_1080));
        assert!(Dma::contains(0x1F80_10F4));
        assert!(!Dma::contains(0x1F80_107F));
        assert!(!Dma::contains(0x1F80_10F8));
    }

    #[test]
    fn channel_roundtrips_all_three_regs() {
        let mut dma = Dma::new();
        dma.write32(0x1F80_1080, 0x0012_3456); // ch0 base
        dma.write32(0x1F80_1084, 0xCAFEBABE); // ch0 bcr
        dma.write32(0x1F80_1088, 0xDEADBEEF); // ch0 chcr
        assert_eq!(dma.read32(0x1F80_1080), 0x0012_3456);
        assert_eq!(dma.read32(0x1F80_1084), 0xCAFEBABE);
        assert_eq!(dma.read32(0x1F80_1088), 0xDEADBEEF);
    }

    #[test]
    fn base_address_is_masked_to_24_bits() {
        let mut dma = Dma::new();
        dma.write32(0x1F80_1080, 0xDEAD_BEEF);
        assert_eq!(dma.read32(0x1F80_1080), 0x00AD_BEEF);
    }

    #[test]
    fn dpcr_roundtrips_verbatim() {
        let mut dma = Dma::new();
        dma.write32(0x1F80_10F0, 0x0765_4321);
        assert_eq!(dma.read32(0x1F80_10F0), 0x0765_4321);
    }

    #[test]
    fn dicr_rw_bits_roundtrip_w1c_bits_dont() {
        // Configure all enables + master, leave flags clear. R/W bits
        // round-trip; W1C bits (15, 24..30) ignore writes-of-zero.
        let mut dma = Dma::new();
        dma.write32(0x1F80_10F4, 0x00FF_0000); // enables 16..23
                                               // Read-back includes the computed master flag in bit 31 — clear
                                               // here because no per-channel flag is set.
        assert_eq!(dma.read32(0x1F80_10F4), 0x00FF_0000);
    }

    #[test]
    fn dicr_channel_done_sets_pending_when_enabled() {
        let mut dma = Dma::new();
        // Enable channel 6 + master.
        dma.write32(0x1F80_10F4, (1 << (16 + 6)) | (1 << 23));
        let edge = dma.notify_channel_done(6);
        assert!(edge, "0->1 master flag transition expected");
        let dicr = dma.read32(0x1F80_10F4);
        assert!(dicr & (1 << (24 + 6)) != 0, "channel-6 flag set");
        assert!(dicr & (1 << 31) != 0, "master flag set");
    }

    #[test]
    fn dicr_channel_done_ignored_when_disabled() {
        let mut dma = Dma::new();
        // Master enable on but channel 2 enable off.
        dma.write32(0x1F80_10F4, 1 << 23);
        let edge = dma.notify_channel_done(2);
        assert!(!edge);
        assert_eq!(dma.read32(0x1F80_10F4) & (1 << (24 + 2)), 0);
    }

    #[test]
    fn dicr_w1c_clears_pending_flag() {
        let mut dma = Dma::new();
        dma.write32(0x1F80_10F4, (1 << (16 + 0)) | (1 << 23));
        dma.notify_channel_done(0);
        assert!(dma.read32(0x1F80_10F4) & (1 << 24) != 0);
        // BIOS acks by writing 1 to the flag bit (along with re-asserting
        // the R/W enables, which the BIOS has been managing all along).
        dma.write32(0x1F80_10F4, (1 << (16 + 0)) | (1 << 23) | (1 << 24));
        let dicr = dma.read32(0x1F80_10F4);
        assert_eq!(dicr & (1 << 24), 0, "flag cleared by W1C");
        assert_eq!(dicr & (1 << 31), 0, "master flag follows");
    }

    #[test]
    fn dicr_write_can_raise_master_irq_edge() {
        let mut dma = Dma::new();
        // Simulate a pending GPU-DMA flag with the channel enable set,
        // but master IRQ still clear. Redux raises IRQ 8 when software
        // writes DICR with master enable in this state.
        dma.dicr = (1 << (16 + 2)) | (1 << (24 + 2));
        let edge = dma.write32(0x1F80_10F4, (1 << (16 + 2)) | (1 << 23));
        assert!(edge, "DICR write should create a master IRQ edge");
        assert!(dma.read32(0x1F80_10F4) & (1 << 31) != 0);
    }

    #[test]
    fn all_seven_channels_are_addressable() {
        let mut dma = Dma::new();
        for ch in 0..NUM_CHANNELS as u32 {
            let addr = 0x1F80_1080 + ch * 0x10;
            dma.write32(addr, ch * 0x1000);
            assert_eq!(dma.read32(addr), ch * 0x1000);
        }
    }

    fn set_otc(dma: &mut Dma, base: u32, count: u32) {
        // CH6: base, block count, start (bit 24) + busy (bit 28)
        dma.write32(0x1F80_10E0, base);
        dma.write32(0x1F80_10E4, count);
        dma.write32(0x1F80_10E8, (1 << 24) | (1 << 28));
    }

    #[test]
    fn otc_madr_is_head_terminator_at_tail() {
        let mut dma = Dma::new();
        let mut ram = vec![0u8; 2 * 1024 * 1024];
        // Sentinel just below the OT range — must remain untouched
        // (validates that the loop's last chain-write to 0x3F4 is
        // overwritten by the terminator, not extended into 0x3F0).
        write_u32(&mut ram, 0x3F0, 0xDEAD_BEEF);

        // 4-entry OT with MADR (head) at 0x400; terminator lands at
        // 0x3F4 (= 0x400 - (4-1)*4).
        set_otc(&mut dma, 0x0000_0400, 4);
        assert_eq!(dma.run_otc(&mut ram), 4);

        // Head: MADR points to next-step-down.
        assert_eq!(read_u32(&ram, 0x400), 0x0000_03FC);
        assert_eq!(read_u32(&ram, 0x3FC), 0x0000_03F8);
        assert_eq!(read_u32(&ram, 0x3F8), 0x0000_03F4);
        // Tail: lowest address holds the terminator.
        assert_eq!(read_u32(&ram, 0x3F4), 0x00FF_FFFF);
        // Sentinel below the tail is untouched.
        assert_eq!(read_u32(&ram, 0x3F0), 0xDEAD_BEEF);
    }

    #[test]
    fn otc_does_not_clear_start_and_busy_bits_synchronously() {
        // Bus is responsible for clearing the busy bits at the
        // scheduled completion cycle (Redux's `gpuotcInterrupt`); the
        // DMA module itself just transfers data. Start AND busy bits
        // both stay set during the "virtual transfer window" so BIOS
        // polling of CHCR sees the same sequence Redux produces.
        //
        // Preventing duplicate runs is done at the bus level via the
        // CHCR-write-only trigger — only a CHCR write with bit 24 set
        // enters `maybe_run_dma`. See `bus.rs:write32` handling of the
        // DMA region.
        let mut dma = Dma::new();
        let mut ram = vec![0u8; 2 * 1024 * 1024];
        set_otc(&mut dma, 0x0000_0100, 1);
        assert_eq!(dma.run_otc(&mut ram), 1);

        let chcr = dma.channels[6].channel_control;
        assert_ne!(chcr & (1 << 24), 0, "start bit must remain set");
        assert_ne!(chcr & (1 << 28), 0, "busy bit must remain set");
    }

    #[test]
    fn otc_is_noop_when_start_bit_not_set() {
        let mut dma = Dma::new();
        let mut ram = vec![0u8; 2 * 1024 * 1024];
        // Write base/count but not start bit.
        dma.write32(0x1F80_10E0, 0x0000_0100);
        dma.write32(0x1F80_10E4, 1);
        assert_eq!(dma.run_otc(&mut ram), 0);
        assert_eq!(read_u32(&ram, 0x100), 0);
    }

    fn read_u32(ram: &[u8], offset: u32) -> u32 {
        let o = offset as usize;
        u32::from_le_bytes([ram[o], ram[o + 1], ram[o + 2], ram[o + 3]])
    }

    fn write_u32(ram: &mut [u8], offset: u32, value: u32) {
        let o = offset as usize;
        ram[o..o + 4].copy_from_slice(&value.to_le_bytes());
    }
}
