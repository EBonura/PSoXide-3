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
    /// bit 1 — drive `/JOYN` output. Transitioning high-to-low is how
    /// the CPU "selects" the device at the start of a transfer; going
    /// low-to-high deselects and resets the device state machine.
    pub const JOYN_OUTPUT: u16 = 1 << 1;
    /// Write-1 to acknowledge pending IRQ bits in STAT.
    pub const ACK: u16 = 1 << 4;
    /// Write-1 to soft-reset the port.
    pub const RESET: u16 = 1 << 6;
    /// bit 13 — port / slot select (0 = JOY1, 1 = JOY2).
    pub const SLOT: u16 = 1 << 13;
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
    /// The Bus raises `IrqSource::Controller` when true. Mirrors how
    /// real hardware pulses IRQ7 when `/DSR` ACKs a transfer (or
    /// DSR-timeout fires if nothing answers).
    pending_irq: bool,
    /// Device on port 1 (controller slot 1 + memory card 1).
    port1: crate::pad::PortDevice,
    /// Device on port 2 (controller slot 2 + memory card 2).
    port2: crate::pad::PortDevice,
    /// Last observed JOYN-output level. We use high-to-low
    /// transitions (deselect → select) to reset device state
    /// machines, matching hardware.
    last_joyn: bool,
}

impl Sio0 {
    /// Physical base address of SIO0.
    pub const BASE: u32 = 0x1F80_1040;
    /// Size of the register window (`DATA..=BAUD` plus padding).
    pub const SIZE: u32 = 0x10;

    /// All registers zero, port 1 pre-populated with a digital pad,
    /// port 2 left empty. Matches the "controller is plugged in,
    /// nothing in slot 2" default most games assume on boot — the
    /// same default our frontend wires up when the user runs a
    /// disc. Without this, `set_port1_buttons` was a silent no-op
    /// (no `DigitalPad` to route the button mask to) and every
    /// game's SIO poll returned 0xFF regardless of what the player
    /// pressed on the keyboard or host gamepad.
    pub fn new() -> Self {
        Self {
            mode: 0,
            ctrl: 0,
            baud: 0,
            rx: None,
            pending_irq: false,
            port1: crate::pad::PortDevice::empty()
                .with_pad(crate::pad::DigitalPad::new()),
            port2: crate::pad::PortDevice::empty(),
            last_joyn: false,
        }
    }

    /// Immutable access to port 1. Used by the frontend to read
    /// pad state (rumble motor) without having to mutate.
    pub fn port1(&self) -> &crate::pad::PortDevice {
        &self.port1
    }

    /// Mutable access to port 1 — lets higher layers swap a
    /// memory card in while keeping the pad attached.
    pub fn port1_mut(&mut self) -> &mut crate::pad::PortDevice {
        &mut self.port1
    }

    /// Immutable access to port 2.
    pub fn port2(&self) -> &crate::pad::PortDevice {
        &self.port2
    }

    /// Mutable access to port 2.
    pub fn port2_mut(&mut self) -> &mut crate::pad::PortDevice {
        &mut self.port2
    }

    /// Plug a device into port 1 (typical for "player 1" games).
    pub fn attach_port1(&mut self, device: crate::pad::PortDevice) {
        self.port1 = device;
    }

    /// Plug a device into port 2.
    pub fn attach_port2(&mut self, device: crate::pad::PortDevice) {
        self.port2 = device;
    }

    /// Update the button state held on port 1.
    pub fn set_port1_buttons(&mut self, buttons: crate::pad::ButtonState) {
        self.port1.set_buttons(buttons);
    }

    /// Update the button state held on port 2.
    pub fn set_port2_buttons(&mut self, buttons: crate::pad::ButtonState) {
        self.port2.set_buttons(buttons);
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

    /// TX clocks one byte across the active port's serial link. The
    /// selected device (if any) returns its RX byte and whether it
    /// wants another round (pulls `/DSR` low → IRQ7 armed).
    fn write_data(&mut self, value: u8) {
        let (rx, ack) = self.active_port().exchange(value);
        self.rx = Some(rx);
        self.pending_irq = ack;
    }

    /// Device selected by the current `CTRL.SLOT` bit.
    fn active_port(&mut self) -> &mut crate::pad::PortDevice {
        if self.ctrl & ctrl_bit::SLOT == 0 {
            &mut self.port1
        } else {
            &mut self.port2
        }
    }

    fn write_ctrl(&mut self, value: u16) {
        if value & ctrl_bit::RESET != 0 {
            self.mode = 0;
            self.ctrl = 0;
            self.baud = 0;
            self.port1.deselect();
            self.port2.deselect();
            self.last_joyn = false;
            return;
        }
        // ACK bit is write-1-to-clear of STAT.IRQ9; we never set that,
        // so we strip the bit and store the rest.
        let new_ctrl = value & !ctrl_bit::ACK;
        // Edge-detect JOYN: the high-to-low transition starts a new
        // transfer; the low-to-high transition deselects and the
        // device state machine resets.
        let old_joyn = self.last_joyn;
        let new_joyn = new_ctrl & ctrl_bit::JOYN_OUTPUT != 0;
        if old_joyn && !new_joyn {
            self.port1.deselect();
            self.port2.deselect();
        }
        self.last_joyn = new_joyn;
        self.ctrl = new_ctrl;
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
        // Unseat the default pad so we can verify the no-device path
        // stays correct (TX ignored, DATA reads 0xFF).
        let mut sio = Sio0::new();
        sio.attach_port1(crate::pad::PortDevice::empty());
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
        // Test the no-device path explicitly — with the default pad
        // attached, TX 0x01 now selects the pad and returns 0x41.
        let mut sio = Sio0::new();
        sio.attach_port1(crate::pad::PortDevice::empty());
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

    #[test]
    fn digital_pad_full_poll_via_mmio() {
        use crate::pad::{button, ButtonState, DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        // Hold Cross + Start.
        sio.set_port1_buttons(ButtonState::from_bits(button::START | button::CROSS));

        // Simulate the BIOS pad-poll sequence on port 1 (SLOT bit clear).
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT); // assert JOYN

        // TX 0x01 → RX 0x41
        sio.write8(Sio0::BASE, 0x01);
        assert_eq!(sio.pop_rx(), 0x41);

        // TX 0x42 → RX 0x5A
        sio.write8(Sio0::BASE, 0x42);
        assert_eq!(sio.pop_rx(), 0x5A);

        // TX 0x00 → RX buttons1 (START = bit 3 pressed → wire 0xF7)
        sio.write8(Sio0::BASE, 0x00);
        assert_eq!(sio.pop_rx(), 0xF7);

        // TX 0x00 → RX buttons2 (CROSS = bit 14 pressed → wire 0xBF)
        sio.write8(Sio0::BASE, 0x00);
        assert_eq!(sio.pop_rx(), 0xBF);
    }

    #[test]
    fn slot_bit_switches_ports() {
        use crate::pad::{DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        // Port 2 stays empty.

        // CTRL.SLOT = 1 → port 2 (empty).
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::SLOT | ctrl_bit::JOYN_OUTPUT);
        sio.write8(Sio0::BASE, 0x01);
        assert_eq!(sio.pop_rx(), 0xFF, "empty port 2 returns 0xFF");

        // CTRL.SLOT = 0 → port 1 (has pad).
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT);
        sio.write8(Sio0::BASE, 0x01);
        assert_eq!(sio.pop_rx(), 0x41, "port 1 pad responds");
    }
}
