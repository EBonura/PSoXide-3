//! Interrupt controller registers.
//!
//! The PS1 has a single on-chip interrupt controller with 11 sources.
//! `I_STAT` latches pending interrupts; `I_MASK` gates which ones fire
//! `IRQ2` on the CPU's COP0. Both are 16-bit registers memory-mapped in
//! the I/O region.
//!
//! `I_STAT` has write-one-to-*clear* semantics in reverse: writing `0`
//! to a bit clears the latched interrupt, writing `1` leaves it alone.
//!
//! Reference: nocash PSX-SPX "Interrupts" section.

use bitflags::bitflags;

/// Interrupt status register. Write-0-to-clear on latched bits.
pub const I_STAT: u32 = 0x1F80_1070;

/// Interrupt mask register. `1` = this source is allowed to raise `IRQ2`.
pub const I_MASK: u32 = 0x1F80_1074;

bitflags! {
    /// Bits shared by `I_STAT` and `I_MASK`. The 16-bit register layout is
    /// identical in both; only the write semantics differ.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Irq: u16 {
        /// VBlank: raised at the start of vertical blanking.
        const VBLANK   = 1 << 0;
        /// GPU: GP0 IRQ command (rarely used).
        const GPU      = 1 << 1;
        /// CD-ROM: command response or sector ready.
        const CDROM    = 1 << 2;
        /// DMA: any channel asserts in `DICR`.
        const DMA      = 1 << 3;
        /// Root counter 0 (dot-clock or system-clock timer).
        const TIMER0   = 1 << 4;
        /// Root counter 1 (hblank or system-clock timer).
        const TIMER1   = 1 << 5;
        /// Root counter 2 (system-clock / 8 timer).
        const TIMER2   = 1 << 6;
        /// Controller / memory-card byte received.
        const SIO0     = 1 << 7;
        /// Serial port byte received.
        const SIO1     = 1 << 8;
        /// SPU IRQ (voice or capture address match).
        const SPU      = 1 << 9;
        /// Controller: lightpen / PIO interrupt (expansion pin).
        const PIO      = 1 << 10;
    }
}

impl Irq {
    /// COP0 `Cause` register bit position for the consolidated interrupt
    /// signal raised by this controller. Firmware polls `Cause & (1 << 10)`.
    pub const COP0_CAUSE_BIT: u32 = 10;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_addresses_live_in_io_window() {
        use crate::memory::io;
        let end = io::BASE + io::SIZE as u32;
        assert!((io::BASE..end).contains(&I_STAT));
        assert!((io::BASE..end).contains(&I_MASK));
    }

    #[test]
    fn all_known_sources_fit_in_11_bits() {
        assert_eq!(Irq::all().bits() & !0x07FF, 0);
    }
}
