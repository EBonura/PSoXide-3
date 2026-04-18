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

/// One port on SIO0. Holds up to one pad **and** one memory card —
/// real hardware multiplexes them on the same port via different
/// leading address bytes (`0x01` = talk to controller, `0x81` =
/// talk to memory card). Both slots can be empty independently.
pub struct PortDevice {
    pad: Option<DigitalPad>,
    memcard: Option<MemoryCard>,
    /// Which slot is driving the current transaction. `None` on
    /// the first byte (we haven't decoded the address yet); set
    /// to `Pad` or `Memcard` once the first byte picks one;
    /// cleared by [`PortDevice::deselect`].
    selected: Option<Selected>,
}

/// Which of the two sub-devices is active in the current SIO
/// transaction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Selected {
    Pad,
    Memcard,
}

impl PortDevice {
    /// Build an empty port — no pad, no memcard.
    pub const fn empty() -> Self {
        Self {
            pad: None,
            memcard: None,
            selected: None,
        }
    }

    /// `true` when nothing is plugged in to either slot.
    pub fn is_empty(&self) -> bool {
        self.pad.is_none() && self.memcard.is_none()
    }

    /// Attach / replace the digital pad on this port.
    pub fn with_pad(mut self, pad: DigitalPad) -> Self {
        self.pad = Some(pad);
        self
    }

    /// Attach / replace the memory card on this port.
    pub fn with_memcard(mut self, card: MemoryCard) -> Self {
        self.memcard = Some(card);
        self
    }

    /// Clock one byte across the serial link. Returns the device's
    /// RX byte and whether the device is holding `/DSR` low (ACK)
    /// to request another byte.
    ///
    /// The first byte of a transaction (after deselect) selects
    /// which slot answers:
    ///
    /// - `0x01` → pad (if present)
    /// - `0x81` → memcard (if present)
    /// - anything else (or selected slot is empty) → `0xFF`, no ACK
    pub fn exchange(&mut self, tx: u8) -> (u8, bool) {
        match self.selected {
            Some(Selected::Pad) => match &mut self.pad {
                Some(pad) => pad.exchange(tx),
                None => (0xFF, false),
            },
            Some(Selected::Memcard) => match &mut self.memcard {
                Some(card) => card.exchange(tx),
                None => (0xFF, false),
            },
            None => {
                // First byte decides the target.
                match tx {
                    0x01 => {
                        if let Some(pad) = self.pad.as_mut() {
                            self.selected = Some(Selected::Pad);
                            pad.exchange(tx)
                        } else {
                            (0xFF, false)
                        }
                    }
                    0x81 => {
                        if let Some(card) = self.memcard.as_mut() {
                            self.selected = Some(Selected::Memcard);
                            card.exchange(tx)
                        } else {
                            (0xFF, false)
                        }
                    }
                    _ => (0xFF, false),
                }
            }
        }
    }

    /// Deselect (JOYN goes high) — every device drops back to idle
    /// and the port forgets which slot was active.
    pub fn deselect(&mut self) {
        if let Some(pad) = self.pad.as_mut() {
            pad.reset();
        }
        if let Some(card) = self.memcard.as_mut() {
            card.reset();
        }
        self.selected = None;
    }

    /// Update the buttons held by the attached pad, if any.
    /// A no-op when no pad is plugged in.
    pub fn set_buttons(&mut self, buttons: ButtonState) {
        if let Some(pad) = self.pad.as_mut() {
            pad.buttons = buttons;
        }
    }

    /// Immutable access to the attached memory card, if any. The
    /// frontend uses this to snapshot card bytes for persistence
    /// at shutdown.
    pub fn memcard(&self) -> Option<&MemoryCard> {
        self.memcard.as_ref()
    }

    /// Mutable access to the memory card.
    pub fn memcard_mut(&mut self) -> Option<&mut MemoryCard> {
        self.memcard.as_mut()
    }

    /// Consume the port, extracting its pad (if any). Used when
    /// swapping memcards in while keeping the pad attached.
    pub fn into_pad(self) -> Option<DigitalPad> {
        self.pad
    }
}

impl Default for PortDevice {
    fn default() -> Self {
        Self::empty()
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

// --- Memory card ---

/// PS1 memory card size: 128 KiB = 16 blocks × 8 KiB = 1024
/// frames of 128 bytes each.
pub const MEMCARD_SIZE: usize = 128 * 1024;
/// One "frame" (the unit of read/write on the serial protocol).
pub const MEMCARD_FRAME_SIZE: usize = 128;

/// PSX memory card emulator. 128 KiB byte buffer + a small
/// protocol state machine for the Read/Write/GetID commands the
/// BIOS issues via SIO0.
///
/// Game-specific persistence lives one level up (frontend side) —
/// this struct just holds the live bytes and the "NEW" (never
/// written) flag.
pub struct MemoryCard {
    /// Live contents. 128 KiB. Empty-card (fresh) is all `0x00`
    /// except for the directory and header frames, but games that
    /// use the BIOS helpers to initialise the card write those
    /// themselves — we don't prefill.
    bytes: Box<[u8; MEMCARD_SIZE]>,
    /// "New card" sticky flag. Hardware sets bit 3 of the flag
    /// byte after power-on until the first successful Write, then
    /// clears it. Games check it to know "do we need to prompt
    /// the user to format this card?" Our behaviour: true at
    /// construction, false once any Write frame lands.
    flag_new: bool,
    state: MemcardState,
    /// Stateful fields used across frames of a single transaction:
    /// frame-number selector (MSB + LSB halves), running checksum
    /// for Read (XOR of all bytes sent), buffer for Write data
    /// prior to committing it to `bytes`.
    frame_msb: u8,
    frame_lsb: u8,
    write_checksum: u8,
    write_buffer: [u8; MEMCARD_FRAME_SIZE],
    write_buffer_len: usize,
    /// Byte counter within the current command stream. Resets on
    /// every deselect.
    byte_index: usize,
    /// Whether any Write has landed since construction. Feeds the
    /// `flag_new` transition and lets the frontend know whether to
    /// persist the card on shutdown.
    dirty: bool,
}

/// Top-level state of the memory-card protocol. Each transaction
/// starts at `Idle`; the first received byte (`0x81`) moves it to
/// `GotAddr`, then the second byte (`R`/`W`/`S`) routes to a
/// command-specific substate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MemcardState {
    Idle,
    GotAddr,
    Read(MemcardReadState),
    Write(MemcardWriteState),
    GetId(MemcardGetIdState),
}

/// Sub-state for Read (command 0x52, "R").
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MemcardReadState {
    /// Just got the `R`. Next byte the host clocks is a dummy,
    /// the byte we return is `0x5A` (Memory Card ID 1).
    SendId1,
    /// Next return byte is `0x5D` (Memory Card ID 2).
    SendId2,
    /// Host sends frame MSB; we return zero.
    RxMsb,
    /// Host sends frame LSB; we return the MSB echo (hardware
    /// quirk — once we have both halves we can compute the address).
    RxLsb,
    /// Send the ACK-1 byte `0x5C`.
    SendAck1,
    /// Send ACK-2 `0x5D`.
    SendAck2,
    /// Echo back the frame MSB.
    EchoMsb,
    /// Echo back the frame LSB.
    EchoLsb,
    /// Send the 128 data bytes, one at a time.
    SendData(u16), // current byte index 0..128
    /// Send the XOR checksum.
    SendChecksum,
    /// Send the terminator `0x47` and drop ACK.
    SendEnd,
}

/// Sub-state for Write (command 0x57, "W").
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MemcardWriteState {
    SendId1,
    SendId2,
    RxMsb,
    RxLsb,
    /// Receive the 128 data bytes from the host.
    RxData(u16),
    /// Receive the host's checksum. We compare to our own; if it
    /// matches we commit the buffer to `bytes`.
    RxChecksum,
    /// Send ACK-1.
    SendAck1,
    SendAck2,
    /// Send end status: `0x47` on success, `0x4E` on checksum
    /// mismatch.
    SendEnd,
}

/// Sub-state for GetID (command 0x53, "S").
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MemcardGetIdState {
    SendId1,
    SendId2,
    SendCommandAck1,
    SendCommandAck2,
    // 4 info bytes: 04 00 00 80 (std. PSX memcard)
    SendInfo(u8),
    SendEnd,
}

impl MemoryCard {
    /// Build a fresh memory card — all bytes `0x00`, "new" flag set.
    pub fn new() -> Self {
        Self {
            bytes: Box::new([0u8; MEMCARD_SIZE]),
            flag_new: true,
            state: MemcardState::Idle,
            frame_msb: 0,
            frame_lsb: 0,
            write_checksum: 0,
            write_buffer: [0u8; MEMCARD_FRAME_SIZE],
            write_buffer_len: 0,
            byte_index: 0,
            dirty: false,
        }
    }

    /// Build a memory card from an existing backing buffer —
    /// typically the bytes loaded from disk. Panics if the buffer
    /// isn't exactly 128 KiB.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        assert_eq!(bytes.len(), MEMCARD_SIZE, "memcard buffer must be 128 KiB");
        let mut buf = Box::new([0u8; MEMCARD_SIZE]);
        buf.copy_from_slice(&bytes);
        Self {
            bytes: buf,
            // Loaded cards are, by definition, not new.
            flag_new: false,
            state: MemcardState::Idle,
            frame_msb: 0,
            frame_lsb: 0,
            write_checksum: 0,
            write_buffer: [0u8; MEMCARD_FRAME_SIZE],
            write_buffer_len: 0,
            byte_index: 0,
            dirty: false,
        }
    }

    /// Has the card been written since construction / load?
    /// Frontend checks this on shutdown to decide whether to flush
    /// back to disk.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the card as clean (caller has just persisted it).
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Snapshot of the full 128 KiB buffer for persistence.
    pub fn as_bytes(&self) -> &[u8; MEMCARD_SIZE] {
        &self.bytes
    }

    /// Reset the state machine to idle. Called when the SIO port
    /// deselects this device.
    fn reset(&mut self) {
        self.state = MemcardState::Idle;
        self.byte_index = 0;
        self.frame_msb = 0;
        self.frame_lsb = 0;
        self.write_checksum = 0;
        self.write_buffer_len = 0;
    }

    /// Compute the flag byte — `0x08` when new, `0x00` once at
    /// least one successful write has cleared the flag.
    fn flag_byte(&self) -> u8 {
        if self.flag_new {
            0x08
        } else {
            0x00
        }
    }

    /// Drive one byte of the SIO exchange. Mirrors the shape of
    /// [`DigitalPad::exchange`]: returns the RX byte + an `ack`
    /// bit asking SIO0 to keep clocking.
    pub fn exchange(&mut self, tx: u8) -> (u8, bool) {
        self.byte_index = self.byte_index.saturating_add(1);
        match self.state {
            MemcardState::Idle => {
                if tx == 0x81 {
                    self.state = MemcardState::GotAddr;
                    // Classic response: the flag byte (NEW=0x08 /
                    // OK=0x00).
                    (self.flag_byte(), true)
                } else {
                    (0xFF, false)
                }
            }
            MemcardState::GotAddr => match tx {
                0x52 => {
                    // "R" — read.
                    self.state = MemcardState::Read(MemcardReadState::SendId1);
                    (0x5A, true)
                }
                0x57 => {
                    // "W" — write.
                    self.state = MemcardState::Write(MemcardWriteState::SendId1);
                    (0x5A, true)
                }
                0x53 => {
                    // "S" — get ID.
                    self.state = MemcardState::GetId(MemcardGetIdState::SendId1);
                    (0x5A, true)
                }
                _ => {
                    self.reset();
                    (0xFF, false)
                }
            },
            MemcardState::Read(rs) => self.exchange_read(rs, tx),
            MemcardState::Write(ws) => self.exchange_write(ws, tx),
            MemcardState::GetId(gs) => self.exchange_get_id(gs, tx),
        }
    }

    fn exchange_read(&mut self, rs: MemcardReadState, tx: u8) -> (u8, bool) {
        use MemcardReadState::*;
        match rs {
            SendId1 => {
                self.state = MemcardState::Read(SendId2);
                (0x5D, true)
            }
            SendId2 => {
                self.state = MemcardState::Read(RxMsb);
                (0x00, true)
            }
            RxMsb => {
                self.frame_msb = tx;
                self.state = MemcardState::Read(RxLsb);
                (0x00, true)
            }
            RxLsb => {
                self.frame_lsb = tx;
                self.state = MemcardState::Read(SendAck1);
                (self.frame_msb, true)
            }
            SendAck1 => {
                self.state = MemcardState::Read(SendAck2);
                (0x5C, true)
            }
            SendAck2 => {
                self.state = MemcardState::Read(EchoMsb);
                (0x5D, true)
            }
            EchoMsb => {
                self.state = MemcardState::Read(EchoLsb);
                (self.frame_msb, true)
            }
            EchoLsb => {
                // Starting the 128 data bytes. Seed checksum.
                self.write_checksum = self.frame_msb ^ self.frame_lsb;
                self.state = MemcardState::Read(SendData(0));
                (self.frame_lsb, true)
            }
            SendData(i) => {
                let byte = self.read_frame_byte(i);
                self.write_checksum ^= byte;
                let next = i + 1;
                if next < MEMCARD_FRAME_SIZE as u16 {
                    self.state = MemcardState::Read(SendData(next));
                } else {
                    self.state = MemcardState::Read(SendChecksum);
                }
                (byte, true)
            }
            SendChecksum => {
                self.state = MemcardState::Read(SendEnd);
                (self.write_checksum, true)
            }
            SendEnd => {
                self.reset();
                (0x47, false)
            }
        }
    }

    fn exchange_write(&mut self, ws: MemcardWriteState, tx: u8) -> (u8, bool) {
        use MemcardWriteState::*;
        match ws {
            SendId1 => {
                self.state = MemcardState::Write(SendId2);
                (0x5D, true)
            }
            SendId2 => {
                self.state = MemcardState::Write(RxMsb);
                (0x00, true)
            }
            RxMsb => {
                self.frame_msb = tx;
                self.state = MemcardState::Write(RxLsb);
                (0x00, true)
            }
            RxLsb => {
                self.frame_lsb = tx;
                self.write_checksum = self.frame_msb ^ self.frame_lsb;
                self.write_buffer_len = 0;
                self.state = MemcardState::Write(RxData(0));
                (0x00, true)
            }
            RxData(i) => {
                self.write_buffer[i as usize] = tx;
                self.write_checksum ^= tx;
                let next = i + 1;
                if next < MEMCARD_FRAME_SIZE as u16 {
                    self.state = MemcardState::Write(RxData(next));
                } else {
                    self.state = MemcardState::Write(RxChecksum);
                }
                // Echo the host's previous byte back on the next
                // cycle — we simplify and just return 0x00.
                (0x00, true)
            }
            RxChecksum => {
                // Compare host checksum to our accumulator. If they
                // match, commit the frame buffer to the backing
                // bytes. Otherwise flag end-with-error.
                let ok = tx == self.write_checksum;
                if ok {
                    self.commit_write_buffer();
                }
                self.state = MemcardState::Write(if ok {
                    SendAck1
                } else {
                    SendAck2 // skip committing; still finish the protocol
                });
                (0x00, true)
            }
            SendAck1 => {
                self.state = MemcardState::Write(SendAck2);
                (0x5C, true)
            }
            SendAck2 => {
                self.state = MemcardState::Write(SendEnd);
                (0x5D, true)
            }
            SendEnd => {
                let end = if self.dirty { 0x47 } else { 0x4E };
                self.reset();
                (end, false)
            }
        }
    }

    fn exchange_get_id(&mut self, gs: MemcardGetIdState, _tx: u8) -> (u8, bool) {
        use MemcardGetIdState::*;
        match gs {
            SendId1 => {
                self.state = MemcardState::GetId(SendId2);
                (0x5D, true)
            }
            SendId2 => {
                self.state = MemcardState::GetId(SendCommandAck1);
                (0x00, true)
            }
            SendCommandAck1 => {
                self.state = MemcardState::GetId(SendCommandAck2);
                (0x5C, true)
            }
            SendCommandAck2 => {
                self.state = MemcardState::GetId(SendInfo(0));
                (0x5D, true)
            }
            SendInfo(i) => {
                // Standard PSX card ID bytes: 04 00 00 80
                let bytes = [0x04, 0x00, 0x00, 0x80];
                let byte = bytes[i as usize];
                let next = i + 1;
                if (next as usize) < bytes.len() {
                    self.state = MemcardState::GetId(SendInfo(next));
                    (byte, true)
                } else {
                    self.state = MemcardState::GetId(SendEnd);
                    (byte, false)
                }
            }
            SendEnd => {
                self.reset();
                (0x00, false)
            }
        }
    }

    fn read_frame_byte(&self, i: u16) -> u8 {
        let frame = ((self.frame_msb as usize) << 8) | (self.frame_lsb as usize);
        let offset = frame * MEMCARD_FRAME_SIZE + (i as usize);
        self.bytes.get(offset).copied().unwrap_or(0)
    }

    fn commit_write_buffer(&mut self) {
        let frame = ((self.frame_msb as usize) << 8) | (self.frame_lsb as usize);
        let start = frame * MEMCARD_FRAME_SIZE;
        let end = start + MEMCARD_FRAME_SIZE;
        if end <= self.bytes.len() {
            self.bytes[start..end].copy_from_slice(&self.write_buffer);
            self.dirty = true;
            self.flag_new = false;
        }
    }
}

impl Default for MemoryCard {
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
        let mut port = PortDevice::empty();
        assert_eq!(port.exchange(0x01), (0xFF, false));
        assert_eq!(port.exchange(0x42), (0xFF, false));
    }

    #[test]
    fn deselect_drops_state() {
        let mut port = PortDevice::empty().with_pad(DigitalPad::new());
        let _ = port.exchange(0x01);
        port.deselect();
        // After deselect, a fresh 0x01 should start from Idle again
        // and succeed.
        let (rx, ack) = port.exchange(0x01);
        assert_eq!(rx, 0x41);
        assert!(ack);
    }

    // --- MemoryCard tests ---

    #[test]
    fn memcard_get_id_response() {
        let mut card = MemoryCard::new();
        // First byte: 0x81 (address). Card responds with flag byte.
        assert_eq!(card.exchange(0x81), (0x08, true)); // NEW flag set
        // Second byte: 'S' (GetID).
        assert_eq!(card.exchange(0x53), (0x5A, true));
        // Then the 8-byte ID sequence.
        let mut seq = vec![];
        for _ in 0..10 {
            let (rx, ack) = card.exchange(0x00);
            seq.push((rx, ack));
        }
        // 5A(just above) then 5D, 00, 5C, 5D, 04, 00, 00, 80
        assert_eq!(seq[0], (0x5D, true));
    }

    #[test]
    fn memcard_write_then_read_round_trips() {
        let mut card = MemoryCard::new();
        // Write a frame of pattern data to frame 0.
        let pattern: Vec<u8> = (0..MEMCARD_FRAME_SIZE).map(|i| (i * 3) as u8).collect();
        let checksum: u8 = pattern.iter().fold(0u8, |a, b| a ^ b) ^ 0x00 ^ 0x00;
        // Protocol: 0x81, 'W' (0x57), 0x00, 0x00, MSB (0x00), LSB (0x00), data*128, checksum
        assert_eq!(card.exchange(0x81), (0x08, true));
        assert_eq!(card.exchange(0x57), (0x5A, true));
        card.exchange(0x00); // id2
        card.exchange(0x00); // pad
        card.exchange(0x00); // MSB
        card.exchange(0x00); // LSB
        for &b in &pattern {
            card.exchange(b);
        }
        // Checksum byte.
        card.exchange(checksum);
        // ACK + SendAck1 + SendAck2 + SendEnd.
        let _ = card.exchange(0x00);
        let _ = card.exchange(0x00);
        let (end, ack) = card.exchange(0x00);
        assert_eq!(end, 0x47, "expected OK terminator after write");
        assert!(!ack, "no ACK on terminator");
        assert!(card.is_dirty(), "card should be marked dirty after write");
        assert!(
            !card.flag_new,
            "NEW flag must clear after first successful write"
        );

        // Read frame 0 back and verify contents match.
        card.reset();
        assert_eq!(card.exchange(0x81), (0x00, true)); // flag = 0 now
        assert_eq!(card.exchange(0x52), (0x5A, true)); // 'R'
        card.exchange(0x00); // id2
        card.exchange(0x00); // pad
        card.exchange(0x00); // MSB
        card.exchange(0x00); // LSB
        card.exchange(0x00); // ack1
        card.exchange(0x00); // ack2
        card.exchange(0x00); // echo msb
        card.exchange(0x00); // echo lsb
        let mut read_back = Vec::with_capacity(MEMCARD_FRAME_SIZE);
        for _ in 0..MEMCARD_FRAME_SIZE {
            let (b, _) = card.exchange(0x00);
            read_back.push(b);
        }
        assert_eq!(read_back, pattern, "written data should round-trip");
    }
}
