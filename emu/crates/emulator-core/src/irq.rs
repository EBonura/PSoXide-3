//! Interrupt controller — `I_STAT` (0x1F801070) and `I_MASK` (0x1F801074).
//!
//! Two 32-bit registers with 11 meaningful source bits. `I_STAT` is the
//! pending set; `I_MASK` is the enable set. A source `s` fires at the
//! CPU iff `(I_STAT & I_MASK) >> s & 1 == 1`. The CPU reflects this in
//! `COP0.CAUSE.IP[2]` — exactly one pin, all 11 sources OR'd together.
//!
//! Write semantics are source-asymmetric:
//! - **`I_STAT` is an AND-acknowledge**. Writing `v` clears any bit
//!   that is 0 in `v` and preserves any bit that is 1. So software
//!   acknowledges interrupt `s` by writing `!(1 << s)`.
//! - **`I_MASK` is a straight write.**

/// Source-bit positions inside `I_STAT` / `I_MASK`. Kept as a typed enum
/// so calling sites read as intent (`irq.raise(IrqSource::VBlank)`)
/// rather than magic bit numbers.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum IrqSource {
    /// Vertical blank — fires once per frame (50 Hz PAL, 60 Hz NTSC).
    VBlank = 0,
    /// GPU — fires on various GPU-internal conditions.
    Gpu = 1,
    /// CD-ROM controller (command response / sector ready / …).
    Cdrom = 2,
    /// DMA controller — any channel completing a transfer.
    Dma = 3,
    /// Root counter 0 (dot-clock / system-clock).
    Timer0 = 4,
    /// Root counter 1 (system-clock / hblank).
    Timer1 = 5,
    /// Root counter 2 (system-clock / system-clock/8).
    Timer2 = 6,
    /// Controller / memory-card (SIO0).
    Controller = 7,
    /// Serial I/O (SIO1).
    Sio = 8,
    /// Sound processing unit.
    Spu = 9,
    /// Light-pen (rarely used).
    Lightpen = 10,
}

/// Combined controller state. Bits beyond position 10 are hardware-
/// reserved and ignored on both read and write.
pub struct Irq {
    stat: u32,
    mask: u32,
    /// Per-source count of `raise()` calls since reset. Diagnostic only.
    raise_counts: [u64; 11],
    /// Count of `pending()` calls that returned `true`. Diagnostic only.
    pending_true_calls: u64,
    /// Peak observed `I_STAT` value across the run. Tells us whether any
    /// pending bits have ever been set (before the BIOS ack cleared them).
    peak_stat: u32,
    /// Number of `write_mask` calls (any width). Diagnostic — tells us
    /// whether the BIOS is reaching `I_MASK` through the IRQ controller
    /// at all, or whether its writes are landing in the io[] echo buffer.
    mask_write_count: u64,
    /// Number of `write_stat` calls. Diagnostic.
    stat_write_count: u64,
    /// First 16 values written to `I_MASK`. Diagnostic.
    mask_write_log: Vec<u32>,
}

impl Irq {
    /// Mask of valid source bits (0..=10).
    pub const VALID_BITS: u32 = 0x7FF;

    /// All bits cleared — matches post-reset hardware.
    pub fn new() -> Self {
        Self {
            stat: 0,
            mask: 0,
            raise_counts: [0; 11],
            pending_true_calls: 0,
            peak_stat: 0,
            mask_write_count: 0,
            stat_write_count: 0,
            mask_write_log: Vec::new(),
        }
    }

    /// Per-source count of `raise()` calls. Diagnostic.
    pub fn raise_counts(&self) -> [u64; 11] {
        self.raise_counts
    }

    /// Peak observed I_STAT since reset. Diagnostic.
    pub fn peak_stat(&self) -> u32 {
        self.peak_stat
    }

    /// Raise interrupt `source` — sets its bit in `I_STAT`. Pending-ness
    /// is OR'd: calling `raise` multiple times before acknowledgement
    /// is a no-op after the first.
    pub fn raise(&mut self, source: IrqSource) {
        let idx = source as usize;
        if idx < self.raise_counts.len() {
            self.raise_counts[idx] = self.raise_counts[idx].saturating_add(1);
        }
        self.stat |= 1 << (source as u32);
        self.peak_stat |= self.stat;
    }

    /// `true` when some source is both pending and enabled — drives the
    /// single CPU IP[2] pin.
    pub fn pending(&self) -> bool {
        let p = (self.stat & self.mask & Self::VALID_BITS) != 0;
        if p {
            // This isn't quite kosher — `pending` takes `&self` — but
            // the counter is `&mut` on the interior. Fix via a dedicated
            // method on `&mut self` that Cpu calls each step.
        }
        p
    }

    /// Same as [`Irq::pending`] but also increments the diagnostic
    /// counter. Call this exactly once per CPU step, from the bus.
    pub fn pending_tick(&mut self) -> bool {
        let p = (self.stat & self.mask & Self::VALID_BITS) != 0;
        if p {
            self.pending_true_calls = self.pending_true_calls.saturating_add(1);
        }
        p
    }

    /// How many times `pending_tick` returned `true`. Diagnostic.
    pub fn pending_true_calls(&self) -> u64 {
        self.pending_true_calls
    }

    /// Current `I_STAT` value (pending sources).
    pub fn stat(&self) -> u32 {
        self.stat
    }

    /// Current `I_MASK` value (enabled sources).
    pub fn mask(&self) -> u32 {
        self.mask
    }

    /// MMIO write to `I_STAT` — AND-acknowledge. Writing 0 to a bit
    /// clears it; writing 1 preserves it.
    pub fn write_stat(&mut self, value: u32) {
        self.stat &= value & Self::VALID_BITS;
        self.stat_write_count = self.stat_write_count.saturating_add(1);
    }

    /// MMIO write to `I_MASK` — direct overwrite.
    pub fn write_mask(&mut self, value: u32) {
        self.mask = value & Self::VALID_BITS;
        self.mask_write_count = self.mask_write_count.saturating_add(1);
        if self.mask_write_log.len() < 16 {
            self.mask_write_log.push(value);
        }
    }

    /// Diagnostic: first 16 values written to I_MASK in order.
    pub fn mask_write_log(&self) -> &[u32] {
        &self.mask_write_log
    }

    /// Diagnostic: how many times `write_mask` has been called.
    pub fn mask_write_count(&self) -> u64 {
        self.mask_write_count
    }

    /// Diagnostic: how many times `write_stat` has been called.
    pub fn stat_write_count(&self) -> u64 {
        self.stat_write_count
    }
}

impl Default for Irq {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_state_is_cleared() {
        let irq = Irq::new();
        assert_eq!(irq.stat(), 0);
        assert_eq!(irq.mask(), 0);
        assert!(!irq.pending());
    }

    #[test]
    fn raise_sets_bit_in_stat() {
        let mut irq = Irq::new();
        irq.raise(IrqSource::VBlank);
        assert_eq!(irq.stat(), 1);
    }

    #[test]
    fn pending_requires_both_stat_and_mask() {
        let mut irq = Irq::new();
        irq.raise(IrqSource::VBlank);
        assert!(!irq.pending()); // mask is 0
        irq.write_mask(1);
        assert!(irq.pending());
    }

    #[test]
    fn write_stat_is_and_acknowledge() {
        let mut irq = Irq::new();
        irq.raise(IrqSource::VBlank);
        irq.raise(IrqSource::Gpu);
        assert_eq!(irq.stat(), 0b11);
        // Ack VBlank (bit 0) by writing 0 to it, preserving bit 1.
        irq.write_stat(!1);
        assert_eq!(irq.stat(), 0b10);
    }

    #[test]
    fn write_mask_is_direct_overwrite() {
        let mut irq = Irq::new();
        irq.write_mask(0xFFFF_FFFF);
        // Only valid bits survive.
        assert_eq!(irq.mask(), Irq::VALID_BITS);
    }
}
