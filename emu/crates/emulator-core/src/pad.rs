//! Controller / memory-card device models attached to SIO0.
//!
//! The PS1 SIO0 port is a simple synchronous serial link. The CPU
//! clocks out one byte at a time; the attached device clocks a byte
//! back simultaneously (full-duplex). Each transfer is one byte,
//! and the BIOS/game keeps clocking bytes until the device stops
//! acknowledging (DSR goes high permanently).
//!
//! For a digital controller the full poll sequence is four bytes:
//!
//! | `TX` | `RX` | Meaning                                      |
//! |------|------|----------------------------------------------|
//! | `01` | `41` | Address byte (select controller); ID low     |
//! | `42` | `5A` | Poll command; ID high                        |
//! | `00` | `b0` | Fill byte; buttons group 1 (active-low)      |
//! | `00` | `b1` | Fill byte; buttons group 2 (active-low)      |
//!
//! The `ack` flag is true for every exchange except the last, so the
//! BIOS's pad IRQ handler keeps clocking bytes until it sees no-ack.
//!
//! Any byte received when the state machine isn't expecting one
//! (for example, a stale TX after a failed poll) resets the device
//! back to idle and returns `0xFF` / no-ack, matching the DSR-timeout
//! behaviour real hardware exhibits.

/// Logical button bit positions. `ButtonState::bits()` returns a
/// `u16` where bit N = 1 means button N is currently held. The
/// device converts to wire-active-low at TX time.
#[derive(Copy, Clone, Debug, Default)]
pub struct ButtonState(u16);

impl ButtonState {
    /// All released.
    pub const NONE: Self = Self(0);

    /// Bit layout (same as the PSX wire format for easy mapping):
    /// - 0: SELECT
    /// - 3: START
    /// - 4: D-pad Up
    /// - 5: D-pad Right
    /// - 6: D-pad Down
    /// - 7: D-pad Left
    /// - 8: L2, 9: R2, 10: L1, 11: R1
    /// - 12: Triangle, 13: Circle, 14: Cross, 15: Square
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Raw bitmask.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Press one or more buttons (OR-in).
    pub fn press(&mut self, mask: u16) {
        self.0 |= mask;
    }

    /// Release one or more buttons.
    pub fn release(&mut self, mask: u16) {
        self.0 &= !mask;
    }

    /// Set the full button mask in one call.
    pub fn set(&mut self, mask: u16) {
        self.0 = mask;
    }
}

/// Named button bitmasks. All constants are single-bit.
pub mod button {
    /// SELECT.
    pub const SELECT: u16 = 1 << 0;
    /// START.
    pub const START: u16 = 1 << 3;
    /// D-pad up.
    pub const UP: u16 = 1 << 4;
    /// D-pad right.
    pub const RIGHT: u16 = 1 << 5;
    /// D-pad down.
    pub const DOWN: u16 = 1 << 6;
    /// D-pad left.
    pub const LEFT: u16 = 1 << 7;
    /// L2 shoulder.
    pub const L2: u16 = 1 << 8;
    /// R2 shoulder.
    pub const R2: u16 = 1 << 9;
    /// L1 shoulder.
    pub const L1: u16 = 1 << 10;
    /// R1 shoulder.
    pub const R1: u16 = 1 << 11;
    /// Triangle face button.
    pub const TRIANGLE: u16 = 1 << 12;
    /// Circle face button.
    pub const CIRCLE: u16 = 1 << 13;
    /// Cross (X) face button.
    pub const CROSS: u16 = 1 << 14;
    /// Square face button.
    pub const SQUARE: u16 = 1 << 15;
}

/// One port on SIO0. Either empty or carrying a digital pad for now;
/// memory cards and analog pads land as variants when needed.
pub enum PortDevice {
    /// Nothing plugged in — every TX returns `0xFF`, never ACKs.
    None,
    /// Standard 8-button digital pad (SCPH-1080 / the grey one).
    DigitalPad(DigitalPad),
}

impl PortDevice {
    /// Clock one byte across the serial link. Returns the device's
    /// RX byte and whether the device is holding `/DSR` low (ACK)
    /// to request another byte.
    pub fn exchange(&mut self, tx: u8) -> (u8, bool) {
        match self {
            PortDevice::None => (0xFF, false),
            PortDevice::DigitalPad(p) => p.exchange(tx),
        }
    }

    /// Deselect (JOYN goes high) — every device drops back to idle.
    pub fn deselect(&mut self) {
        if let PortDevice::DigitalPad(p) = self {
            p.reset();
        }
    }

    /// Update the buttons held by the attached pad, if any.
    /// A no-op when nothing is plugged in.
    pub fn set_buttons(&mut self, buttons: ButtonState) {
        if let PortDevice::DigitalPad(p) = self {
            p.buttons = buttons;
        }
    }
}

/// Standard PSX digital controller.
pub struct DigitalPad {
    buttons: ButtonState,
    state: PadState,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PadState {
    /// Waiting for the address byte (`0x01` = "talk to controller").
    Idle,
    /// Saw `0x01`, waiting for the poll command (`0x42`).
    GotAddr,
    /// Sent the pad-ID high byte, about to send buttons group 1.
    SendButtons1,
    /// Sent buttons 1; next TX pops buttons 2 and ends the transfer.
    SendButtons2,
}

impl DigitalPad {
    /// Build a pad with all buttons released.
    pub fn new() -> Self {
        Self {
            buttons: ButtonState::NONE,
            state: PadState::Idle,
        }
    }

    fn reset(&mut self) {
        self.state = PadState::Idle;
    }

    fn exchange(&mut self, tx: u8) -> (u8, bool) {
        match (self.state, tx) {
            // Start of a pad poll. The BIOS clocks `0x01` to the
            // addressed port; we respond with the digital-pad ID low
            // byte and hold ACK for the next exchange.
            (PadState::Idle, 0x01) => {
                self.state = PadState::GotAddr;
                (0x41, true)
            }
            // Poll command. Respond with ID high byte.
            (PadState::GotAddr, 0x42) => {
                self.state = PadState::SendButtons1;
                (0x5A, true)
            }
            // Return buttons group 1 (bits 0..=7 on the wire, active
            // low: `1` on the wire = released, `0` = pressed). Our
            // internal [`ButtonState`] stores active-high for easy
            // composition, so we invert here.
            (PadState::SendButtons1, _) => {
                self.state = PadState::SendButtons2;
                let b1 = !(self.buttons.bits() & 0xFF) as u8;
                (b1, true)
            }
            // Buttons group 2 (bits 8..=15). Last byte of the
            // transaction — drop ACK so the BIOS stops clocking.
            (PadState::SendButtons2, _) => {
                self.state = PadState::Idle;
                let b2 = !((self.buttons.bits() >> 8) & 0xFF) as u8;
                (b2, false)
            }
            // Any other combination (unexpected TX for current state)
            // aborts the transfer. Real hardware behaves similarly —
            // the pad times out DSR when the protocol gets off-rails.
            _ => {
                self.state = PadState::Idle;
                (0xFF, false)
            }
        }
    }
}

impl Default for DigitalPad {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_digital_poll_all_released() {
        let mut pad = DigitalPad::new();
        assert_eq!(pad.exchange(0x01), (0x41, true));
        assert_eq!(pad.exchange(0x42), (0x5A, true));
        assert_eq!(pad.exchange(0x00), (0xFF, true)); // all released → 0xFF
        assert_eq!(pad.exchange(0x00), (0xFF, false)); // last byte
        assert_eq!(pad.state, PadState::Idle);
    }

    #[test]
    fn pressing_start_shows_in_byte1() {
        let mut pad = DigitalPad::new();
        pad.buttons.press(button::START);
        let _ = pad.exchange(0x01);
        let _ = pad.exchange(0x42);
        let (b1, _) = pad.exchange(0x00);
        // START = bit 3; active-low on the wire: !0x08 = 0xF7.
        assert_eq!(b1, 0xF7);
    }

    #[test]
    fn pressing_cross_shows_in_byte2() {
        let mut pad = DigitalPad::new();
        pad.buttons.press(button::CROSS);
        let _ = pad.exchange(0x01);
        let _ = pad.exchange(0x42);
        let _ = pad.exchange(0x00);
        let (b2, _) = pad.exchange(0x00);
        // CROSS = bit 14; (bits >> 8) = bit 6 → !0x40 = 0xBF.
        assert_eq!(b2, 0xBF);
    }

    #[test]
    fn unexpected_byte_resets_state_machine() {
        let mut pad = DigitalPad::new();
        let _ = pad.exchange(0x01); // GotAddr
        // Wrong poll command — abort.
        assert_eq!(pad.exchange(0xFF), (0xFF, false));
        assert_eq!(pad.state, PadState::Idle);
    }

    #[test]
    fn no_device_always_returns_ff() {
        let mut port = PortDevice::None;
        assert_eq!(port.exchange(0x01), (0xFF, false));
        assert_eq!(port.exchange(0x42), (0xFF, false));
    }

    #[test]
    fn deselect_drops_state() {
        let mut port = PortDevice::DigitalPad(DigitalPad::new());
        let _ = port.exchange(0x01);
        port.deselect();
        // After deselect, a fresh 0x01 should start from Idle again
        // and succeed.
        let (rx, ack) = port.exchange(0x01);
        assert_eq!(rx, 0x41);
        assert!(ack);
    }
}
