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
    /// Channel-control start bits are *recorded* (Phase 2g is scaffolding
    /// only) but no transfer actually fires — peripheral subsystems
    /// claim that work as they come online.
    pub fn write32(&mut self, phys: u32, value: u32) {
        let offset = phys - Self::BASE;
        match offset {
            Self::DPCR_OFFSET => self.dpcr = value,
            Self::DICR_OFFSET => self.write_dicr(value),
            _ => {
                let (ch, field) = decode(phys);
                if ch >= NUM_CHANNELS {
                    return;
                }
                let c = &mut self.channels[ch];
                match field {
                    0x0 => c.base = value & 0x00FF_FFFF,
                    0x4 => c.block_control = value,
                    0x8 => {
                        c.channel_control = value;
                        if (value >> 24) & 1 != 0 {
                            self.start_trigger_counts[ch] += 1;
                            self.chcr_write_count[ch] =
                                self.chcr_write_count[ch].saturating_add(1);
                        }
                    }
                    _ => {}
                }
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
// Layout (Nocash):
//   bits  0..5  : Unknown (R/W)
//   bits  6..14 : Reserved (always 0)
//   bit   15    : Bus-error flag (R, W1C)
//   bits 16..22 : Per-channel IRQ enable (R/W) — DMA0..DMA6
//   bit   23    : IRQ master enable (R/W)
//   bits 24..30 : Per-channel IRQ flag (R, W1C) — DMA0..DMA6
//   bit   31    : IRQ master flag (R) — derived:
//                 (bit15) | (bit23 & ((bits16..22) & (bits24..30) != 0))
//
// The R/W bits are stored verbatim. The W1C bits (15, 24..30) are
// cleared by writing a `1` (and preserved by writing `0`), which is the
// inverse of the natural u32 store. Bit 31 is read-only and computed.
impl Dma {
    const DICR_RW_MASK: u32 = 0x00FF_003F;
    const DICR_W1C_MASK: u32 = 0x7F00_8000;
    const DICR_MASTER_ENABLE: u32 = 1 << 23;
    const DICR_BUS_ERROR: u32 = 1 << 15;

    fn write_dicr(&mut self, value: u32) {
        let prev = self.dicr;
        let rw = value & Self::DICR_RW_MASK;
        let preserved_flags = prev & Self::DICR_W1C_MASK & !value;
        self.dicr = rw | preserved_flags;
    }

    fn dicr_with_master_flag(&self) -> u32 {
        let enabled = (self.dicr >> 16) & 0x7F;
        let pending = (self.dicr >> 24) & 0x7F;
        let any_irq = self.dicr & Self::DICR_MASTER_ENABLE != 0 && (enabled & pending) != 0;
        let master_flag = self.dicr & Self::DICR_BUS_ERROR != 0 || any_irq;
        (self.dicr & 0x7FFF_FFFF) | ((master_flag as u32) << 31)
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
        let enable_bit = 1 << (16 + ch);
        if self.dicr & enable_bit == 0 {
            return false;
        }
        let prev_master = self.dicr_with_master_flag() >> 31;
        self.dicr |= 1 << (24 + ch);
        let new_master = self.dicr_with_master_flag() >> 31;
        prev_master == 0 && new_master == 1
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
