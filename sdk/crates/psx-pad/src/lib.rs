//! Digital-pad polling via SIO0.
//!
//! Talks directly to the SIO0 hardware (no BIOS syscalls) so the
//! same code works whether we side-load a homebrew (HLE BIOS) or
//! boot through the real BIOS. The controller protocol is simple
//! enough that a hand-rolled select + four-byte exchange beats
//! opening events and waiting on them.
//!
//! Typical use from a game loop:
//!
//! ```ignore
//! use psx_pad::{poll_port1, ButtonState};
//!
//! let buttons = poll_port1();
//! if buttons.is_held(psx_pad::button::START) {
//!     // …
//! }
//! ```
//!
//! The protocol spec, reproduced from nocash PSX-SPX:
//!
//! | `TX` | `RX` | Meaning                            |
//! |------|------|------------------------------------|
//! | `01` | `FF` | Address byte / select controller   |
//! | `42` | `41` | Poll command / digital pad ID low  |
//! | `00` | `5A` | Fill byte / ID high                |
//! | `00` | `b0` | Buttons group 1 (active-low)       |
//! | `00` | `b1` | Buttons group 2 (active-low)       |
//!
//! Our [`ButtonState`] stores active-high so `buttons.is_held` feels
//! natural in game code.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use psx_io::sio;

/// Named button bitmasks (active-high in this representation).
/// Hardware's active-low wire format is hidden inside [`poll_port1`].
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

/// Result of one pad poll. `bits()` gives the raw active-high mask;
/// [`ButtonState::is_held`] is the ergonomic per-button check.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ButtonState(u16);

impl ButtonState {
    /// Empty — nothing held.
    pub const NONE: Self = Self(0);

    /// Construct from an active-high mask.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Raw bitmask.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// `true` when `mask` (any single-bit [`button`] constant or an
    /// OR of several) is currently pressed.
    #[inline]
    pub const fn is_held(self, mask: u16) -> bool {
        self.0 & mask != 0
    }
}

// --- SIO0 CTRL bits we care about ---

const CTRL_TXEN: u16 = 1 << 0;
const CTRL_JOYN: u16 = 1 << 1;
const CTRL_RXEN: u16 = 1 << 2;
const CTRL_ACK: u16 = 1 << 4;
const CTRL_SLOT_PORT2: u16 = 1 << 13;

/// Poll the controller in port 1 once. Returns the current button
/// state; unplugged ports (no device responds) come back as
/// [`ButtonState::NONE`].
pub fn poll_port1() -> ButtonState {
    poll(false)
}

/// Poll the controller in port 2.
pub fn poll_port2() -> ButtonState {
    poll(true)
}

fn poll(port2: bool) -> ButtonState {
    unsafe {
        // Reset port state: raise JOYN then drop it, so the attached
        // device's state machine starts from idle.
        let slot = if port2 { CTRL_SLOT_PORT2 } else { 0 };
        psx_io::write16(sio::CTRL, CTRL_ACK);
        psx_io::write16(sio::CTRL, slot | CTRL_TXEN | CTRL_RXEN | CTRL_JOYN);

        let _select = exchange(0x01);
        let id_lo = exchange(0x42);
        if id_lo != 0x41 {
            // No digital pad on this port.
            deselect();
            return ButtonState::NONE;
        }
        let _id_hi = exchange(0x00);
        let b0 = exchange(0x00);
        let b1 = exchange(0x00);
        deselect();

        // Wire bytes are active-low; invert to match our active-high
        // ButtonState convention.
        let bits = !((b0 as u16) | ((b1 as u16) << 8));
        ButtonState::from_bits(bits)
    }
}

/// Drop JOYN so the attached device's state machine resets before
/// the next poll.
#[inline]
unsafe fn deselect() {
    unsafe { psx_io::write16(sio::CTRL, 0) };
}

/// Clock one byte across the serial link. Waits for TX ready, writes
/// the byte, then waits for RX to fill before reading.
///
/// The STAT register layout (from PSX-SPX):
/// - bit 0: TX-ready-1 (FIFO can accept a byte)
/// - bit 1: RX FIFO not empty
/// - bit 2: TX-ready-2 (last byte has cleared the shifter)
#[inline]
unsafe fn exchange(tx: u8) -> u8 {
    unsafe {
        while psx_io::read32(sio::STAT) & 0x1 == 0 {}
        psx_io::write8(sio::DATA, tx);
        while psx_io::read32(sio::STAT) & 0x2 == 0 {}
        psx_io::read8(sio::DATA)
    }
}
