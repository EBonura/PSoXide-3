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
    /// ACK/DSR input level from the selected device.
    pub const ACK_INPUT: u32 = 1 << 7;
    /// SIO0 interrupt latched in the controller's own STAT register.
    pub const IRQ: u32 = 1 << 9;
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
    /// bit 12 — enable IRQ7 generation from controller ACK pulses.
    pub const ACK_IRQ_ENABLE: u16 = 1 << 12;
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

/// Serial-transfer time for one byte when BAUD is zero. Matches the
/// BIOS's common `0x88 * 8 = 1088`-cycle setup.
const DEFAULT_TRANSFER_TICKS: u64 = 1088;
/// Delay from RX-byte delivery to `/ACK` for a controller.
const PAD_ACK_DELAY_TICKS: u64 = 450;
/// Memory cards answer faster than pads.
const MEMCARD_ACK_DELAY_TICKS: u64 = 170;
/// `/ACK` is a pulse, not a sticky level.
const ACK_PULSE_TICKS: u64 = 100;

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
    /// Sticky ACK input level exposed in `STAT` bit 7. Cleared when a
    /// new transfer starts, on deselect, or on reset.
    ack_input: bool,
    /// Sticky SIO IRQ bit exposed in `STAT` bit 9. Writing CTRL.ACK
    /// clears it; the bus-level interrupt controller is separate.
    irq_latched: bool,
    /// Byte currently in flight on the serial wire. Becomes visible
    /// in `rx` only once the transfer phase completes.
    pending_rx: u8,
    /// One-byte TX holding register. `STAT.TX_READY_1` reflects
    /// whether this slot is free, while `STAT.TX_READY_2` stays low
    /// until the currently active byte clears the shifter and any
    /// pending ACK delay for the previous byte has expired.
    queued_tx: Option<u8>,
    /// Whether the current byte will be followed by an ACK pulse.
    pending_ack: bool,
    /// Current byte transfer occupies the wire. A queued follow-up
    /// byte can launch as soon as this shifter phase completes even
    /// if the previous byte's ACK pulse has not fired yet.
    transfer_busy: bool,
    /// Transfer phase is done and we're waiting for the ACK delay to
    /// expire.
    awaiting_ack: bool,
    /// Absolute cycle at which the current byte transfer completes.
    transfer_deadline: Option<u64>,
    /// Absolute cycle at which the pending ACK pulse should fire.
    ack_deadline: Option<u64>,
    /// Absolute cycle at which the ACK pulse ends.
    ack_end_deadline: Option<u64>,
    /// Delay for the currently selected device kind.
    ack_delay_ticks: u64,
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
            ack_input: false,
            irq_latched: false,
            pending_rx: 0xFF,
            queued_tx: None,
            pending_ack: false,
            transfer_busy: false,
            awaiting_ack: false,
            transfer_deadline: None,
            ack_deadline: None,
            ack_end_deadline: None,
            ack_delay_ticks: PAD_ACK_DELAY_TICKS,
            port1: crate::pad::PortDevice::empty()
                .with_pad(crate::pad::DigitalPad::new())
                .with_memcard(crate::pad::MemoryCard::new()),
            port2: crate::pad::PortDevice::empty().with_memcard(crate::pad::MemoryCard::new()),
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

    /// Snapshot the software-visible STAT register without consuming
    /// any FIFO state. Diagnostic-only.
    pub fn debug_stat(&self) -> u32 {
        self.stat()
    }

    /// Diagnostic-only copy of the raw CTRL register.
    pub fn debug_ctrl(&self) -> u16 {
        self.ctrl
    }

    /// Whether `service_sio0` would currently raise a controller IRQ
    /// edge to the bus.
    pub fn debug_pending_irq(&self) -> bool {
        self.pending_irq
    }

    /// Sticky IRQ bit exposed in `STAT`.
    pub fn debug_irq_latched(&self) -> bool {
        self.irq_latched
    }

    /// Whether a TX byte is currently shifting.
    pub fn debug_transfer_busy(&self) -> bool {
        self.transfer_busy
    }

    /// Whether the transfer phase finished and the ACK delay is in
    /// flight.
    pub fn debug_awaiting_ack(&self) -> bool {
        self.awaiting_ack
    }

    /// Absolute cycle where the current byte transfer completes.
    pub fn debug_transfer_deadline(&self) -> Option<u64> {
        self.transfer_deadline
    }

    /// Absolute cycle where the next ACK pulse should begin.
    pub fn debug_ack_deadline(&self) -> Option<u64> {
        self.ack_deadline
    }

    /// Absolute cycle where the current ACK pulse ends.
    pub fn debug_ack_end_deadline(&self) -> Option<u64> {
        self.ack_end_deadline
    }

    /// Earliest pending state-machine deadline across the three
    /// timers, or `None` if SIO0 is idle. The bus uses this to
    /// (re)schedule [`EventSlot::Sio`] so the per-instruction
    /// `Bus::tick` poll can be retired in favour of one event-driven
    /// wake-up. Mirrors the role of `intCycle[PSXINT_SIO]` in
    /// PCSX-Redux's `branchTest`.
    pub fn next_deadline(&self) -> Option<u64> {
        let mut next: Option<u64> = None;
        let consider = |cur: &mut Option<u64>, candidate: Option<u64>| {
            if let Some(c) = candidate {
                *cur = Some(match cur {
                    Some(prev) => (*prev).min(c),
                    None => c,
                });
            }
        };
        consider(&mut next, self.transfer_deadline);
        consider(&mut next, self.ack_deadline);
        consider(&mut next, self.ack_end_deadline);
        next
    }

    /// RX slot contents, if a byte is waiting to be read.
    pub fn debug_rx(&self) -> Option<u8> {
        self.rx
    }

    /// Next TX byte queued behind the active transfer.
    pub fn debug_queued_tx(&self) -> Option<u8> {
        self.queued_tx
    }

    /// `true` when `phys` falls within the SIO0 register window.
    #[inline]
    pub fn contains(phys: u32) -> bool {
        (Self::BASE..Self::BASE + Self::SIZE).contains(&phys)
    }

    /// `SIO0_STAT`. TX is never busy (no clock simulation); RX-not-empty
    /// tracks whether a response byte is waiting.
    fn stat(&self) -> u32 {
        let mut s = 0;
        if self.queued_tx.is_none() {
            s |= stat_bit::TX_READY_1;
        }
        if !self.transfer_busy && !self.awaiting_ack {
            s |= stat_bit::TX_READY_2;
        }
        if self.rx.is_some() {
            s |= stat_bit::RX_NOT_EMPTY;
        }
        if self.ack_input {
            s |= stat_bit::ACK_INPUT;
        }
        if self.irq_latched {
            s |= stat_bit::IRQ;
        }
        s
    }

    /// Pop the RX slot, returning 0xFF when empty. Used by all three
    /// widths of DATA read — the real chip zero-extends on the bus.
    fn pop_rx(&mut self) -> u8 {
        self.rx.take().unwrap_or(0xFF)
    }

    /// Transfer time for one byte. Hardware uses `BAUD * 8`; when the
    /// BIOS hasn't set BAUD yet we still give software a realistic
    /// default delay instead of "instant byte".
    fn transfer_ticks(&self) -> u64 {
        let baud = self.baud as u64;
        if baud != 0 {
            baud.saturating_mul(8)
        } else {
            DEFAULT_TRANSFER_TICKS
        }
    }

    /// Advance any pending transfer / ACK timers to `now`.
    pub fn tick(&mut self, now: u64) {
        loop {
            if let Some(deadline) = self.transfer_deadline {
                if deadline <= now {
                    self.transfer_deadline = None;
                    self.rx = Some(self.pending_rx);
                    self.transfer_busy = false;
                    if self.pending_ack {
                        self.awaiting_ack = true;
                        self.ack_deadline = Some(deadline.saturating_add(self.ack_delay_ticks));
                    }
                    if let Some(value) = self.queued_tx.take() {
                        self.start_transfer_at(value, deadline, false);
                    } else if !self.awaiting_ack {
                        self.transfer_busy = false;
                    }
                    continue;
                }
            }

            if let Some(deadline) = self.ack_deadline {
                if deadline <= now {
                    self.ack_deadline = None;
                    self.awaiting_ack = false;
                    // Keep IRQ and visible ACK/DSR in phase. Crash's
                    // BIOS pad handler samples the ACK level while it
                    // services the interrupt; raising IRQ at RX-ready
                    // time makes digital polls stop before the high
                    // button byte.
                    self.ack_input = true;
                    self.ack_end_deadline = Some(deadline.saturating_add(ACK_PULSE_TICKS));
                    if self.ctrl & ctrl_bit::ACK_IRQ_ENABLE != 0 {
                        self.irq_latched = true;
                        self.pending_irq = true;
                    }
                    continue;
                }
            }

            if let Some(deadline) = self.ack_end_deadline {
                if deadline <= now {
                    self.ack_end_deadline = None;
                    self.ack_input = false;
                    continue;
                }
            }

            break;
        }
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
    fn write_data_at(&mut self, value: u8, now: u64) {
        if self.transfer_busy {
            if self.queued_tx.is_none() {
                self.queued_tx = Some(value);
            }
            return;
        }
        self.start_transfer_at(value, now, true);
    }

    fn start_transfer_at(&mut self, value: u8, now: u64, clear_ack_pulse: bool) {
        // A fresh byte clocks a new phase of the transfer, so the
        // previous ACK pulse is no longer visible.
        if clear_ack_pulse {
            self.ack_input = false;
            self.ack_end_deadline = None;
        }
        let (rx, ack, ack_delay_ticks) = if self.ctrl & ctrl_bit::JOYN_OUTPUT != 0 {
            let port = self.active_port();
            let (rx, ack) = port.exchange(value);
            let ack_delay_ticks = if port.selected_is_memcard() {
                MEMCARD_ACK_DELAY_TICKS
            } else {
                PAD_ACK_DELAY_TICKS
            };
            (rx, ack, ack_delay_ticks)
        } else {
            (0xFF, false, PAD_ACK_DELAY_TICKS)
        };
        self.pending_rx = rx;
        self.pending_ack = ack;
        self.ack_delay_ticks = ack_delay_ticks;
        self.transfer_busy = true;
        self.transfer_deadline = Some(now.saturating_add(self.transfer_ticks()));
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
            self.pending_irq = false;
            self.ack_input = false;
            self.irq_latched = false;
            self.pending_rx = 0xFF;
            self.queued_tx = None;
            self.pending_ack = false;
            self.transfer_busy = false;
            self.awaiting_ack = false;
            self.transfer_deadline = None;
            self.ack_deadline = None;
            self.ack_end_deadline = None;
            self.rx = None;
            self.port1.deselect();
            self.port2.deselect();
            self.last_joyn = false;
            return;
        }
        if value & ctrl_bit::ACK != 0 {
            self.irq_latched = false;
            self.pending_irq = false;
        }
        // ACK bit is write-1-to-clear of STAT.IRQ9; don't keep it in
        // the stored register value.
        let new_ctrl = value & !ctrl_bit::ACK;
        // Edge-detect JOYN: the high-to-low transition starts a new
        // transfer; the low-to-high transition deselects and the
        // device state machine resets.
        let old_joyn = self.last_joyn;
        let new_joyn = new_ctrl & ctrl_bit::JOYN_OUTPUT != 0;
        if old_joyn && !new_joyn {
            self.ack_input = false;
            self.queued_tx = None;
            self.pending_ack = false;
            self.transfer_busy = false;
            self.awaiting_ack = false;
            self.transfer_deadline = None;
            self.ack_deadline = None;
            self.ack_end_deadline = None;
            self.port1.deselect();
            self.port2.deselect();
        }
        self.last_joyn = new_joyn;
        self.ctrl = new_ctrl;
    }

    /// 32-bit write dispatch. STAT is 32-bit on hardware but the
    /// writable registers all live in the 16-bit half, so we delegate.
    pub fn write32(&mut self, phys: u32, value: u32) -> bool {
        self.write16_at(phys, value as u16, 0)
    }

    /// 16-bit write dispatch. Returns `true` when the address fell in
    /// a recognised slot (even if the write was semantically ignored).
    pub fn write16(&mut self, phys: u32, value: u16) -> bool {
        self.write16_at(phys, value, 0)
    }

    /// Cycle-aware 32-bit write dispatch.
    pub fn write32_at(&mut self, phys: u32, value: u32, now: u64) -> bool {
        self.write16_at(phys, value as u16, now)
    }

    /// Cycle-aware 16-bit write dispatch.
    pub fn write16_at(&mut self, phys: u32, value: u16, now: u64) -> bool {
        match phys - Self::BASE {
            offset::DATA => self.write_data_at(value as u8, now),
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
        self.write8_at(phys, value, 0)
    }

    /// Cycle-aware 8-bit write dispatch.
    pub fn write8_at(&mut self, phys: u32, value: u8, now: u64) -> bool {
        match phys - Self::BASE {
            offset::DATA => self.write_data_at(value, now),
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
        // attached, TX 0x01 would select the pad and return the
        // dummy select-byte response.
        let mut sio = Sio0::new();
        sio.attach_port1(crate::pad::PortDevice::empty());
        sio.write8(Sio0::BASE, 0x01);
        sio.tick(DEFAULT_TRANSFER_TICKS);
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
        sio.ack_input = true;
        sio.irq_latched = true;
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::RESET);
        assert_eq!(sio.mode, 0);
        assert_eq!(sio.baud, 0);
        assert_eq!(sio.ctrl, 0);
        assert!(!sio.ack_input);
        assert!(!sio.irq_latched);
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

        // TX 0x01 → RX 0xFF (dummy select-byte response)
        sio.write8(Sio0::BASE, 0x01);
        sio.tick(DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS);
        assert_eq!(sio.pop_rx(), 0xFF);

        // TX 0x42 → RX 0x41
        sio.write8(Sio0::BASE, 0x42);
        sio.tick(2 * (DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS));
        assert_eq!(sio.pop_rx(), 0x41);

        // TX 0x00 → RX 0x5A
        sio.write8(Sio0::BASE, 0x00);
        sio.tick(3 * (DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS));
        assert_eq!(sio.pop_rx(), 0x5A);

        // TX 0x00 → RX buttons1 (START = bit 3 pressed → wire 0xF7)
        sio.write8(Sio0::BASE, 0x00);
        sio.tick(4 * (DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS));
        assert_eq!(sio.pop_rx(), 0xF7);

        // TX 0x00 → RX buttons2 (CROSS = bit 14 pressed → wire 0xBF)
        sio.write8(Sio0::BASE, 0x00);
        sio.tick(4 * (DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS) + DEFAULT_TRANSFER_TICKS);
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
        sio.tick(DEFAULT_TRANSFER_TICKS);
        assert_eq!(sio.pop_rx(), 0xFF, "empty port 2 returns 0xFF");

        // CTRL.SLOT = 0 → port 1 (has pad).
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT);
        sio.write8(Sio0::BASE, 0x01);
        sio.tick(DEFAULT_TRANSFER_TICKS);
        assert_eq!(
            sio.pop_rx(),
            0xFF,
            "port 1 select byte gets the dummy response"
        );
        sio.write8(Sio0::BASE, 0x42);
        sio.tick(DEFAULT_TRANSFER_TICKS * 2);
        assert_eq!(
            sio.pop_rx(),
            0x41,
            "port 1 pad responds on the command byte"
        );
    }

    #[test]
    fn controller_ack_sets_stat_bits_and_irq_when_enabled() {
        use crate::pad::{DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        sio.write16(
            Sio0::BASE + 0xA,
            ctrl_bit::JOYN_OUTPUT | ctrl_bit::ACK_IRQ_ENABLE,
        );

        sio.write8(Sio0::BASE, 0x01);
        sio.tick(DEFAULT_TRANSFER_TICKS);

        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_eq!(
            stat & stat_bit::ACK_INPUT,
            0,
            "ACK must wait for the ACK phase"
        );
        assert_eq!(
            stat & stat_bit::IRQ,
            0,
            "IRQ bit must stay low until ACK"
        );
        assert!(
            !sio.take_pending_irq(),
            "bus IRQ should not fire before ACK"
        );

        sio.tick(DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS);
        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_ne!(
            stat & stat_bit::ACK_INPUT,
            0,
            "ACK input should be visible"
        );
        assert_ne!(stat & stat_bit::IRQ, 0, "STAT IRQ bit should latch");
        assert!(sio.take_pending_irq(), "bus should see one IRQ edge");
        assert!(!sio.take_pending_irq(), "IRQ edge should be single-shot");

        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT | ctrl_bit::ACK);
        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_eq!(stat & stat_bit::IRQ, 0, "CTRL.ACK should clear STAT IRQ");
    }

    #[test]
    fn controller_ack_without_irq_enable_does_not_raise_irq() {
        use crate::pad::{DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT);

        sio.write8(Sio0::BASE, 0x01);
        sio.tick(DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS);

        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_ne!(
            stat & stat_bit::ACK_INPUT,
            0,
            "ACK input should be visible"
        );
        assert_eq!(
            stat & stat_bit::IRQ,
            0,
            "IRQ bit should stay low without enable"
        );
        assert!(
            !sio.take_pending_irq(),
            "bus IRQ must stay quiet without enable"
        );
    }

    #[test]
    fn successive_controller_acks_raise_successive_bus_irqs_without_ctrl_ack() {
        use crate::pad::{DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        sio.write16(
            Sio0::BASE + 0xA,
            ctrl_bit::JOYN_OUTPUT | ctrl_bit::ACK_IRQ_ENABLE,
        );

        sio.write8_at(Sio0::BASE, 0x01, 10);
        sio.tick(10 + DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS);
        assert!(sio.take_pending_irq(), "first ACK should raise an IRQ edge");

        let second_start = 10 + DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS + 1;
        sio.write8_at(Sio0::BASE, 0x42, second_start);
        sio.tick(second_start + DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS);
        assert!(
            sio.take_pending_irq(),
            "second ACK should still raise an IRQ edge even if CTRL.ACK was never written"
        );
    }

    #[test]
    fn sdk_style_pad_poll_sequence_completes() {
        use crate::pad::{button, ButtonState, DigitalPad, PortDevice};

        fn wait_for_stat(sio: &mut Sio0, now: &mut u64, mask: u32) {
            const TIMEOUT_TICKS: u64 = 10_000;
            let start = *now;
            while sio.read32(Sio0::BASE + 0x4).unwrap() & mask == 0 {
                *now = now.saturating_add(1);
                sio.tick(*now);
                assert!(
                    *now - start < TIMEOUT_TICKS,
                    "timed out waiting for STAT mask {mask:#x}"
                );
            }
        }

        fn exchange_sdk_style(sio: &mut Sio0, now: &mut u64, tx: u8) -> u8 {
            wait_for_stat(sio, now, stat_bit::TX_READY_1);
            sio.write8(Sio0::BASE, tx);
            wait_for_stat(sio, now, stat_bit::RX_NOT_EMPTY);
            sio.read8(Sio0::BASE).unwrap()
        }

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        sio.set_port1_buttons(ButtonState::from_bits(button::START | button::CROSS));

        let mut now = 0u64;
        // Match psx-pad's init sequence: ACK any stale IRQ, then
        // assert JOYN with TX/RX enabled on port 1.
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::ACK);
        sio.write16(
            Sio0::BASE + 0xA,
            (1 << 0) | ctrl_bit::JOYN_OUTPUT | (1 << 2),
        );

        assert_eq!(exchange_sdk_style(&mut sio, &mut now, 0x01), 0xFF);
        assert_eq!(exchange_sdk_style(&mut sio, &mut now, 0x42), 0x41);
        assert_eq!(exchange_sdk_style(&mut sio, &mut now, 0x00), 0x5A);
        assert_eq!(exchange_sdk_style(&mut sio, &mut now, 0x00), 0xF7);
        assert_eq!(exchange_sdk_style(&mut sio, &mut now, 0x00), 0xBF);
    }

    #[test]
    fn transfer_busy_holds_tx_ready_low_until_ack_phase_finishes() {
        use crate::pad::{DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT);
        sio.write16(Sio0::BASE + 0xE, 0x0088);

        sio.write8_at(Sio0::BASE, 0x01, 10);
        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_eq!(
            stat & stat_bit::TX_READY_2,
            0,
            "transfer should drop TX_READY_2"
        );

        sio.tick(10 + 0x88 * 8);
        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_eq!(
            stat & stat_bit::TX_READY_2,
            0,
            "ACK wait should still look busy"
        );

        sio.tick(10 + 0x88 * 8 + PAD_ACK_DELAY_TICKS);
        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_ne!(
            stat & stat_bit::TX_READY_2,
            0,
            "port should go ready after ACK"
        );
    }

    #[test]
    fn tx_ready_1_stays_high_until_the_single_byte_queue_is_full() {
        use crate::pad::{DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT);

        sio.write8_at(Sio0::BASE, 0x01, 10);
        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_ne!(
            stat & stat_bit::TX_READY_1,
            0,
            "queue slot should stay free during first byte"
        );

        sio.write8_at(Sio0::BASE, 0x42, 11);
        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_eq!(
            stat & stat_bit::TX_READY_1,
            0,
            "second write should fill the one-byte queue"
        );

        sio.tick(10 + DEFAULT_TRANSFER_TICKS);
        let stat = sio.read32(Sio0::BASE + 0x4).unwrap();
        assert_ne!(
            stat & stat_bit::TX_READY_1,
            0,
            "queued byte should launch as soon as the shifter is free"
        );
    }

    #[test]
    fn queued_byte_starts_before_the_previous_ack_delay_finishes() {
        use crate::pad::{DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT);

        sio.write8_at(Sio0::BASE, 0x01, 10);
        sio.write8_at(Sio0::BASE, 0x42, 11);

        sio.tick(10 + DEFAULT_TRANSFER_TICKS);
        assert_eq!(sio.pop_rx(), 0xFF, "select byte should complete on time");
        assert!(
            sio.transfer_busy,
            "queued second byte should have started immediately after the first transfer"
        );
        assert!(
            sio.awaiting_ack,
            "the first byte should still be waiting to pulse ACK"
        );

        sio.tick(10 + DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS - 1);
        assert_eq!(
            sio.read32(Sio0::BASE + 0x4).unwrap() & stat_bit::ACK_INPUT,
            0,
            "ACK should still wait for its own delay even while the next byte is in flight"
        );
    }

    #[test]
    fn ack_input_is_a_pulse_not_a_sticky_level() {
        use crate::pad::{DigitalPad, PortDevice};

        let mut sio = Sio0::new();
        sio.attach_port1(PortDevice::empty().with_pad(DigitalPad::new()));
        sio.write16(Sio0::BASE + 0xA, ctrl_bit::JOYN_OUTPUT);

        sio.write8_at(Sio0::BASE, 0x01, 100);
        sio.tick(100 + DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS);
        assert_ne!(
            sio.read32(Sio0::BASE + 0x4).unwrap() & stat_bit::ACK_INPUT,
            0,
            "ACK pulse should become visible"
        );

        sio.tick(100 + DEFAULT_TRANSFER_TICKS + PAD_ACK_DELAY_TICKS + ACK_PULSE_TICKS);
        assert_eq!(
            sio.read32(Sio0::BASE + 0x4).unwrap() & stat_bit::ACK_INPUT,
            0,
            "ACK pulse should self-clear"
        );
    }
}
