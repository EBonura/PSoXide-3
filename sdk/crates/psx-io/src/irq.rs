//! Interrupt controller (I_STAT / I_MASK) MMIO.

/// `I_STAT` -- pending interrupt set. Write-to-ack: writing 0 to a bit
/// clears it (writing 1 preserves).
pub const I_STAT: u32 = 0x1F80_1070;

/// `I_MASK` -- interrupt enable set. Straight write.
pub const I_MASK: u32 = 0x1F80_1074;

/// Source bits inside `I_STAT` / `I_MASK`.
pub mod source {
    /// Bit position of the VBlank IRQ.
    pub const VBLANK: u32 = 0;
    /// Bit position of the GPU IRQ (GP0 1Fh).
    pub const GPU: u32 = 1;
    /// Bit position of the CD-ROM IRQ.
    pub const CDROM: u32 = 2;
    /// Bit position of the DMA-completion IRQ.
    pub const DMA: u32 = 3;
    /// Bit position of root-counter 0.
    pub const TIMER0: u32 = 4;
    /// Bit position of root-counter 1.
    pub const TIMER1: u32 = 5;
    /// Bit position of root-counter 2.
    pub const TIMER2: u32 = 6;
    /// Bit position of the controller / memory-card IRQ.
    pub const CONTROLLER: u32 = 7;
    /// Bit position of the SIO1 (debug-serial) IRQ.
    pub const SIO1: u32 = 8;
    /// Bit position of the SPU IRQ.
    pub const SPU: u32 = 9;
    /// Bit position of the lightpen / controller-IRQ10 line.
    pub const LIGHTPEN: u32 = 10;
}

/// Read `I_STAT`.
#[inline(always)]
pub fn stat() -> u32 {
    unsafe { crate::read32(I_STAT) }
}

/// Read `I_MASK`.
#[inline(always)]
pub fn mask() -> u32 {
    unsafe { crate::read32(I_MASK) }
}

/// Acknowledge pending bits by writing `!(bits)` -- the hardware
/// AND-accumulates, so any bit left 1 in the written value is
/// preserved, any bit that was 0 is cleared.
#[inline(always)]
pub fn ack(bits: u32) {
    unsafe { crate::write32(I_STAT, !bits) }
}

/// Set the mask register (who can interrupt the CPU).
#[inline(always)]
pub fn set_mask(bits: u32) {
    unsafe { crate::write32(I_MASK, bits) }
}
