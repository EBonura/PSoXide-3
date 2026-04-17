//! SIO0 — controller / memory-card serial port.
//!
//! Register map at `0x1F80_1040`:
//!   +0  `SIO0_DATA`  TX/RX FIFO (8-bit; reads may be accessed as wider)
//!   +4  `SIO0_STAT`  status (32-bit, read-only)
//!   +8  `SIO0_MODE`  framing config (16-bit)
//!   +A  `SIO0_CTRL`  control (16-bit)
//!   +E  `SIO0_BAUD`  baud divisor (16-bit)
//!
//! This implementation models the common cold-boot case where no
//! controller and no memory-card are connected to either port. Every
//! byte the CPU writes to `DATA` transmits instantly, the TX-ready
//! status bits are always set, and no device ever answers, so the
//! RX FIFO stays empty and `/ACK` never goes low. The BIOS's
//! controller-poll path uses this to conclude "no pad present" after
//! one address byte and move on — enough to unblock the boot
//! spin-wait on `SIO0_STAT & 1`.
//!
//! Real controller / memcard protocols (packet framing, slot select,
//! IRQ7 on ACK, DSR timing) land when we emulate an attached device.

mod stat_bit {
    pub const TX_READY_1: u32 = 1 << 0;
    pub const RX_NOT_EMPTY: u32 = 1 << 1;
    pub const TX_READY_2: u32 = 1 << 2;
}

mod ctrl_bit {
    /// Write-1 to acknowledge pending IRQ bits in STAT.
    pub const ACK: u16 = 1 << 4;
    /// Write-1 to soft-reset the port.
    pub const RESET: u16 = 1 << 6;
}

mod offset {
    pub const DATA: u32 = 0x0;
    pub const STAT: u32 = 0x4;
    /// High half of the 32-bit STAT register; 16-bit reads land here
    /// when software does two half-reads. Returns zero on hardware.
    pub const STAT_HI: u32 = 0x6;
    pub const MODE: u32 = 0x8;
    pub const CTRL: u32 = 0xA;
    pub const BAUD: u32 = 0xE;
}

/// SIO0 state. Register-level accuracy for the "nothing plugged in"
/// path; no shift-clock simulation, but every byte-write pulses an
/// IRQ7 so the BIOS's pad-poll handler advances its descriptors.
pub struct Sio0 {
    mode: u16,
    ctrl: u16,
    baud: u16,
    /// One-slot RX buffer. The real chip has a small FIFO; the BIOS
    /// only reads one response per TX so a single slot is enough.
    /// `None` means RX empty (`STAT.RX_NOT_EMPTY` clears).
    rx: Option<u8>,
    /// Set by `write_data`, consumed by [`Sio0::take_pending_irq`].
    /// The Bus raises `IrqSource::Controller` when true. Hardware does
    /// this via /ACK low-to-high; with no device we approximate via
    /// "always IRQ after TX", which matches the DSR-timeout outcome.
    pending_irq: bool,
}

impl Sio0 {
    /// Physical base address of SIO0.
    pub const BASE: u32 = 0x1F80_1040;
    /// Size of the register window (`DATA..=BAUD` plus padding).
    pub const SIZE: u32 = 0x10;

    /// All registers zero — matches a cold power-on port.
    pub fn new() -> Self {
        Self {
            mode: 0,
            ctrl: 0,
            baud: 0,
            rx: None,
            pending_irq: false,
        }
    }

    /// Returns true and clears the flag when a DATA write has armed
    /// IRQ7. The Bus calls this after dispatching a write.
    pub fn take_pending_irq(&mut self) -> bool {
        let p = self.pending_irq;
        self.pending_irq = false;
        p
    }

    /// `true` when `phys` falls within the SIO0 register window.
    #[inline]
    pub fn contains(phys: u32) -> bool {
        (Self::BASE..Self::BASE + Self::SIZE).contains(&phys)
    }

    /// `SIO0_STAT`. TX is never busy (no clock simulation); RX-not-empty
    /// tracks whether a response byte is waiting.
    fn stat(&self) -> u32 {
        let mut s = stat_bit::TX_READY_1 | stat_bit::TX_READY_2;
        if self.rx.is_some() {
            s |= stat_bit::RX_NOT_EMPTY;
        }
        s
    }

    /// Pop the RX slot, returning 0xFF when empty. Used by all three
    /// widths of DATA read — the real chip zero-extends on the bus.
    fn pop_rx(&mut self) -> u8 {
        self.rx.take().unwrap_or(0xFF)
    }

    /// 32-bit read dispatch. `Some(value)` for every offset in the
    /// window; unrecognised offsets read as zero to match the general
    /// MMIO echo-buffer fallback. Reading `DATA` consumes the RX slot.
    pub fn read32(&mut self, phys: u32) -> Option<u32> {
        match phys - Self::BASE {
            offset::DATA => Some(self.pop_rx() as u32),
            offset::STAT => Some(self.stat()),
            offset::MODE => Some(self.mode as u32),
            offset::CTRL => Some(self.ctrl as u32),
            offset::BAUD => Some(self.baud as u32),
            _ => Some(0),
        }
    }

    /// 16-bit read dispatch.
    pub fn read16(&mut self, phys: u32) -> Option<u16> {
        match phys - Self::BASE {
            offset::DATA => Some(self.pop_rx() as u16),
            offset::STAT => Some(self.stat() as u16),
            offset::STAT_HI => Some(0),
            offset::MODE => Some(self.mode),
            offset::CTRL => Some(self.ctrl),
            offset::BAUD => Some(self.baud),
            _ => Some(0),
        }
    }

    /// 8-bit read dispatch.
    pub fn read8(&mut self, phys: u32) -> Option<u8> {
        match phys - Self::BASE {
            offset::DATA => Some(self.pop_rx()),
            offset::STAT => Some(self.stat() as u8),
            _ => Some(0),
        }
    }

    /// TX clocks a byte out on the (empty) port. Full-duplex: the
    /// shifter simultaneously reads in a byte, and with nothing
    /// driving MISO the line floats to 0xFF. RX slot fills and an
    /// IRQ7 is armed so the BIOS's pad handler runs and advances
    /// its per-port descriptor counter.
    fn write_data(&mut self, _value: u8) {
        self.rx = Some(0xFF);
        self.pending_irq = true;
    }

    fn write_ctrl(&mut self, value: u16) {
        if value & ctrl_bit::RESET != 0 {
            self.mode = 0;
            self.ctrl = 0;
            self.baud = 0;
            return;
        }
        // ACK bit is write-1-to-clear of STAT.IRQ9; we never set that,
        // so we strip the bit and store the rest.
        self.ctrl = value & !ctrl_bit::ACK;
    }

    /// 32-bit write dispatch. STAT is 32-bit on hardware but the
    /// writable registers all live in the 16-bit half, so we delegate.
    pub fn write32(&mut self, phys: u32, value: u32) -> bool {
        self.write16(phys, value as u16)
    }

    /// 16-bit write dispatch. Returns `true` when the address fell in
    /// a recognised slot (even if the write was semantically ignored).
    pub fn write16(&mut self, phys: u32, value: u16) -> bool {
        match phys - Self::BASE {
            offset::DATA => self.write_data(value as u8),
            offset::STAT => {}
            offset::MODE => self.mode = value,
            offset::CTRL => self.write_ctrl(value),
            offset::BAUD => self.baud = value,
            _ => return false,
        }
        true
    }

    /// 8-bit write dispatch. Only `DATA` accepts a byte write; every
    /// other register is 16-bit on the real chip.
    pub fn write8(&mut self, phys: u32, value: u8) -> bool {
        match phys - Self::BASE {
            offset::DATA => self.write_data(value),
            _ => return false,
        }
        true
    }
}

impl Default for Sio0 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_reports_tx_ready_bits() {
        let mut sio = Sio0::new();
        let s = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_eq!(s & 0x1, 0x1, "TX_READY_1 must be set");
        assert_eq!(s & 0x4, 0x4, "TX_READY_2 must be set");
        assert_eq!(s & 0x2, 0, "RX_NOT_EMPTY must be clear");
    }

    #[test]
    fn data_read_returns_ff_no_device() {
        let mut sio = Sio0::new();
        assert_eq!(sio.read8(Sio0::BASE).unwrap(), 0xFF);
        assert_eq!(sio.read16(Sio0::BASE).unwrap(), 0x00FF);
    }

    #[test]
    fn stat_hi_half_reads_zero() {
        let mut sio = Sio0::new();
        assert_eq!(sio.read16(Sio0::BASE + 0x6).unwrap(), 0);
    }

    #[test]
    fn tx_fills_rx_with_ff_and_sets_stat_bit() {
        let mut sio = Sio0::new();
        sio.write8(Sio0::BASE, 0x01);
        let s = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_eq!(s & 0x2, 0x2, "RX_NOT_EMPTY must be set after TX");
        assert_eq!(sio.read8(Sio0::BASE).unwrap(), 0xFF);
        // Reading DATA consumes the RX slot.
        let s = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_eq!(s & 0x2, 0, "RX_NOT_EMPTY must be clear after DATA read");
    }

    #[test]
    fn write_then_read_roundtrips_mode_baud() {
        let mut sio = Sio0::new();
        sio.write16(Sio0::BASE + 0x8, 0x1234);
        sio.write16(Sio0::BASE + 0xE, 0x5678);
        assert_eq!(sio.read16(Sio0::BASE + 0x8).unwrap(), 0x1234);
        assert_eq!(sio.read16(Sio0::BASE + 0xE).unwrap(), 0x5678);
    }

    #[test]
    fn ctrl_reset_bit_zeroes_everything() {
        let mut sio = Sio0::new();
        sio.write16(Sio0::BASE + 0x8, 0xFFFF);
        sio.write16(Sio0::BASE + 0xE, 0xFFFF);
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::RESET);
        assert_eq!(sio.mode, 0);
        assert_eq!(sio.baud, 0);
        assert_eq!(sio.ctrl, 0);
    }

    #[test]
    fn ctrl_ack_bit_is_stripped() {
        let mut sio = Sio0::new();
        // Every bit except RESET (0x40); ACK must not be kept.
        sio.write16(Sio0::BASE + 0xA, 0x001F);
        assert_eq!(sio.ctrl & ctrl_bit::ACK, 0);
        // Other low bits pass through (bits 0..3 are TXEN/DTR/RXEN/JOYN).
        assert_eq!(sio.ctrl & 0x000F, 0x000F);
    }

    #[test]
    fn contains_matches_full_window() {
        for off in 0..Sio0::SIZE {
            assert!(Sio0::contains(Sio0::BASE + off));
        }
        assert!(!Sio0::contains(Sio0::BASE - 1));
        assert!(!Sio0::contains(Sio0::BASE + Sio0::SIZE));
    }
}
