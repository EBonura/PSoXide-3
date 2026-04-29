//! Digital / DualShock pad polling via SIO0.
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
//! use psx_pad::poll_port1;
//!
//! let pad = poll_port1();
//! if pad.buttons.is_held(psx_pad::button::START) {
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
//! DualShock analog mode uses the same first four bytes but reports
//! ID low `0x73` and appends four stick bytes:
//! right X/Y, then left X/Y. Fresh DualShocks boot digital, so games
//! that require sticks should either call [`enable_analog_port1`] or
//! show an "enable analog mode" prompt when [`PadState::is_analog`]
//! is false.
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
    /// Left stick click (DualShock L3).
    pub const L3: u16 = 1 << 1;
    /// Right stick click (DualShock R3).
    pub const R3: u16 = 1 << 2;
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

/// Default analog-stick reading. `0x80` = centred, `0x00` = full
/// negative, `0xFF` = full positive.
pub const STICK_CENTER: u8 = 0x80;

/// Controller operating mode inferred from the poll ID byte.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PadMode {
    /// No controller answered the poll.
    Disconnected,
    /// SCPH-1080 digital-pad shape: ID `0x41`, buttons only.
    Digital,
    /// DualShock analog shape: ID `0x73`, buttons plus stick bytes.
    Analog,
    /// DualShock config/escape shape: ID `0xF3`.
    Config,
    /// A controller answered with an ID this SDK does not classify yet.
    Unknown,
}

impl PadMode {
    /// `true` when a controller answered the poll.
    #[inline]
    pub const fn is_connected(self) -> bool {
        !matches!(self, Self::Disconnected)
    }

    /// `true` when the controller is currently reporting stick bytes.
    #[inline]
    pub const fn has_sticks(self) -> bool {
        matches!(self, Self::Analog | Self::Config)
    }
}

/// Raw DualShock stick bytes from one poll.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AnalogSticks {
    /// Right stick horizontal axis.
    pub right_x: u8,
    /// Right stick vertical axis.
    pub right_y: u8,
    /// Left stick horizontal axis.
    pub left_x: u8,
    /// Left stick vertical axis.
    pub left_y: u8,
}

impl AnalogSticks {
    /// Centred sticks.
    pub const CENTERED: Self = Self {
        right_x: STICK_CENTER,
        right_y: STICK_CENTER,
        left_x: STICK_CENTER,
        left_y: STICK_CENTER,
    };

    /// Left stick as signed deltas from centre.
    #[inline]
    pub const fn left_centered(self) -> (i16, i16) {
        (
            self.left_x as i16 - STICK_CENTER as i16,
            self.left_y as i16 - STICK_CENTER as i16,
        )
    }

    /// Right stick as signed deltas from centre.
    #[inline]
    pub const fn right_centered(self) -> (i16, i16) {
        (
            self.right_x as i16 - STICK_CENTER as i16,
            self.right_y as i16 - STICK_CENTER as i16,
        )
    }
}

impl Default for AnalogSticks {
    fn default() -> Self {
        Self::CENTERED
    }
}

/// Result of one controller poll.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PadState {
    /// Active-high button state.
    pub buttons: ButtonState,
    /// Inferred controller mode.
    pub mode: PadMode,
    /// Stick bytes. Centred when the controller is not reporting sticks.
    pub sticks: AnalogSticks,
    /// Raw low ID byte returned by the controller.
    pub id_low: u8,
}

impl PadState {
    /// No controller connected / no response.
    pub const NONE: Self = Self {
        buttons: ButtonState::NONE,
        mode: PadMode::Disconnected,
        sticks: AnalogSticks::CENTERED,
        id_low: 0xFF,
    };

    /// `true` when a controller answered the poll.
    #[inline]
    pub const fn is_connected(self) -> bool {
        self.mode.is_connected()
    }

    /// `true` when the controller is in DualShock analog mode.
    #[inline]
    pub const fn is_analog(self) -> bool {
        matches!(self.mode, PadMode::Analog)
    }
}

// --- SIO0 CTRL bits we care about ---

const CTRL_TXEN: u16 = 1 << 0;
const CTRL_JOYN: u16 = 1 << 1;
const CTRL_RXEN: u16 = 1 << 2;
const CTRL_ACK: u16 = 1 << 4;
const CTRL_SLOT_PORT2: u16 = 1 << 13;

/// Poll the controller in port 1 once.
///
/// The returned [`PadState`] always contains active-high buttons; in
/// analog mode it also contains the four DualShock stick bytes.
pub fn poll_port1() -> PadState {
    poll_state(false)
}

/// Poll the controller in port 2 once.
///
/// The returned [`PadState`] always contains active-high buttons; in
/// analog mode it also contains the four DualShock stick bytes.
pub fn poll_port2() -> PadState {
    poll_state(true)
}

/// Ask the port-1 controller to enter DualShock analog mode. Returns
/// `true` when a follow-up poll reports analog mode.
///
/// Digital-only controllers simply keep reporting digital mode, so
/// callers should still gate analog-only controls on
/// [`PadState::is_analog`].
pub fn enable_analog_port1() -> bool {
    enable_analog(false)
}

/// Ask the port-2 controller to enter DualShock analog mode. Returns
/// `true` when a follow-up poll reports analog mode.
pub fn enable_analog_port2() -> bool {
    enable_analog(true)
}

fn poll_state(port2: bool) -> PadState {
    unsafe {
        // Reset port state: raise JOYN then drop it, so the attached
        // device's state machine starts from idle.
        select(port2);

        let _select = exchange(0x01);
        let id_lo = exchange(0x42);
        let mode = mode_from_id_low(id_lo);
        if !mode.is_connected() {
            deselect();
            return PadState {
                id_low: id_lo,
                ..PadState::NONE
            };
        }

        let id_hi = exchange(0x00);
        if id_hi != 0x5A {
            deselect();
            return PadState {
                mode: PadMode::Unknown,
                id_low: id_lo,
                ..PadState::NONE
            };
        }

        let b0 = exchange(0x00);
        let b1 = exchange(0x00);
        let buttons = decode_buttons(b0, b1);

        let sticks = if mode.has_sticks() {
            AnalogSticks {
                right_x: exchange(0x00),
                right_y: exchange(0x00),
                left_x: exchange(0x00),
                left_y: exchange(0x00),
            }
        } else {
            AnalogSticks::CENTERED
        };

        deselect();

        PadState {
            buttons,
            mode,
            sticks,
            id_low: id_lo,
        }
    }
}

fn enable_analog(port2: bool) -> bool {
    unsafe {
        // Enter config mode.
        transaction(port2, [0x43, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        // Request analog mode and lock it so the pad cannot toggle
        // back underneath analog-only game controls.
        transaction(port2, [0x44, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 0x00]);
        // Exit config mode, restoring the requested analog mode.
        transaction(port2, [0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    }
    poll_state(port2).is_analog()
}

#[inline]
fn mode_from_id_low(id_low: u8) -> PadMode {
    match id_low {
        0x41 => PadMode::Digital,
        0x73 => PadMode::Analog,
        0xF3 => PadMode::Config,
        0xFF => PadMode::Disconnected,
        _ => PadMode::Unknown,
    }
}

#[inline]
fn decode_buttons(b0: u8, b1: u8) -> ButtonState {
    // Wire bytes are active-low; invert to match our active-high
    // ButtonState convention.
    ButtonState::from_bits(!((b0 as u16) | ((b1 as u16) << 8)))
}

/// Select the requested controller port and prepare SIO0 for a new
/// transaction.
#[inline]
unsafe fn select(port2: bool) {
    unsafe {
        let slot = if port2 { CTRL_SLOT_PORT2 } else { 0 };
        psx_io::write16(sio::CTRL, CTRL_ACK);
        psx_io::write16(sio::CTRL, slot | CTRL_TXEN | CTRL_RXEN | CTRL_JOYN);
    }
}

/// Run a fixed eight-byte DualShock command after the port-level
/// controller select byte.
#[inline]
unsafe fn transaction(port2: bool, bytes: [u8; 8]) -> [u8; 8] {
    unsafe {
        select(port2);
        let _select = exchange(0x01);
        let mut out = [0u8; 8];
        let mut i = 0;
        while i < bytes.len() {
            out[i] = exchange(bytes[i]);
            i += 1;
        }
        deselect();
        out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_ids_match_dualshock_wire_values() {
        assert_eq!(mode_from_id_low(0x41), PadMode::Digital);
        assert_eq!(mode_from_id_low(0x73), PadMode::Analog);
        assert_eq!(mode_from_id_low(0xF3), PadMode::Config);
        assert_eq!(mode_from_id_low(0xFF), PadMode::Disconnected);
        assert_eq!(mode_from_id_low(0x12), PadMode::Unknown);
    }

    #[test]
    fn decode_buttons_converts_active_low_to_active_high() {
        let state = decode_buttons(0xEF, 0xBF);
        assert!(state.is_held(button::UP));
        assert!(state.is_held(button::CROSS));
        assert!(!state.is_held(button::DOWN));
    }

    #[test]
    fn centred_stick_helpers_return_zero() {
        assert_eq!(AnalogSticks::CENTERED.left_centered(), (0, 0));
        assert_eq!(AnalogSticks::CENTERED.right_centered(), (0, 0));
    }
}
