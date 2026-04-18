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

    /// Immutable access to the attached pad, if any. Used by the
    /// frontend to read motor state + analog positions without
    /// going through the SIO transaction loop.
    pub fn pad(&self) -> Option<&DigitalPad> {
        self.pad.as_ref()
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

/// Default analog-stick reading — 0x80 (128) = centre, 0x00 = full
/// negative, 0xFF = full positive. Matches real DualShock output.
pub const STICK_CENTER: u8 = 0x80;

/// Pad operating mode.
///
/// Fresh-plugged DualShocks boot in Digital and stay there until the
/// game runs the config-mode dance (`0x43` + `0x44` commands) to
/// switch them to Analog. Games that expect specific analog
/// responses (racers, souls-likes, most modern ports) rely on the
/// mode transition working correctly — reporting Analog from the
/// start confuses games that assume a Digital → Analog sequence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PadMode {
    /// SCPH-1080 digital protocol. ID byte `0x41`, 4-byte poll.
    /// Default for fresh-plugged controllers.
    Digital,
    /// DualShock analog mode. ID byte `0x73`, 8-byte poll (4 extra
    /// bytes for right + left stick X/Y, each 0-255).
    Analog,
    /// Config / escape mode. ID byte `0xF3`, 8-byte transaction.
    /// Host enters via command `0x43`+`0x01`, uses in-mode commands
    /// to change pad state (switch between digital/analog, rumble
    /// mapping, etc), then exits via `0x43`+`0x00`.
    Config,
}

/// PS1 controller — digital by default, DualShock-capable. Tracks
/// the active operating mode, the 16 digital buttons, the four
/// analog axes, and DualShock vibration state.
pub struct DigitalPad {
    buttons: ButtonState,
    /// Right-stick horizontal axis, 0..=255 (centre `0x80`).
    right_x: u8,
    /// Right-stick vertical axis.
    right_y: u8,
    /// Left-stick horizontal axis.
    left_x: u8,
    /// Left-stick vertical axis.
    left_y: u8,
    /// Current operating mode. Switched by host via the `0x43` /
    /// `0x44` config dance.
    mode: PadMode,
    /// The mode we were in before entering config. On `0x43`+`0x00`
    /// (exit config) we restore this instead of defaulting back to
    /// Digital — matters because a game can enter config from
    /// Analog mode to change rumble mapping and expects Analog to
    /// still be active on exit.
    mode_before_config: PadMode,
    /// Which command byte the current transaction is running
    /// (`0x42` poll, `0x43` config-enter/exit, `0x44` set mode,
    /// etc). Set at the second exchange of the transaction.
    cmd: u8,
    /// Which byte of the current transaction we're on. 0 = host
    /// just sent the address byte; 1 = sent command; 2..=N = payload
    /// / response bytes.
    step: u8,
    /// DualShock vibration-motor byte mapping. Set by command
    /// `0x4D` in Config mode. Each entry maps a byte position in
    /// the `0x42` (poll) command's payload (steps 2..=7) to a
    /// motor:
    /// - `0x00` → small motor (on/off — `0xFF` = on).
    /// - `0x01` → big motor (strength `0..=0xFF`).
    /// - `0xFF` → no motor at this position (default).
    ///
    /// Default state is `[0xFF; 6]` — no motors mapped — matching
    /// a fresh DualShock before a game has configured rumble.
    motor_mapping: [u8; 6],
    /// Current small-motor state (binary on/off).
    motor_small: bool,
    /// Current big-motor strength (0..=255).
    motor_big: u8,
}

/// Zero-indexed step of the last byte in a transaction. Ack stays
/// low (true) for every step strictly less than this; on the last
/// step we drop ack so the host stops clocking.
///
/// A DualShock analog transaction runs 8 bytes (steps 0..=7). A
/// legacy digital poll is 4 bytes (0..=3). Config-mode commands
/// always use the 8-byte shape regardless of the pad's current
/// active mode.
const DIGITAL_POLL_LAST: u8 = 3;
const ANALOG_POLL_LAST: u8 = 7;

impl DigitalPad {
    /// Build a pad with all buttons released, sticks centred, in
    /// Digital mode (default). Mirrors a freshly-plugged DualShock
    /// before any game has issued the "switch to analog" config
    /// sequence.
    pub fn new() -> Self {
        Self {
            buttons: ButtonState::NONE,
            right_x: STICK_CENTER,
            right_y: STICK_CENTER,
            left_x: STICK_CENTER,
            left_y: STICK_CENTER,
            mode: PadMode::Digital,
            mode_before_config: PadMode::Digital,
            cmd: 0,
            step: 0,
            motor_mapping: [0xFF; 6],
            motor_small: false,
            motor_big: 0,
        }
    }

    /// Current vibration-motor state: `(small_on, big_strength)`.
    /// The frontend polls this once per frame to drive a host
    /// haptics device (gamepad rumble, phone vibrate, etc.).
    /// Values are latched — they persist across polls — so a
    /// reader that samples less often than the game polls still
    /// sees the most recently-commanded state.
    pub fn motor_state(&self) -> (bool, u8) {
        (self.motor_small, self.motor_big)
    }

    fn reset(&mut self) {
        self.cmd = 0;
        self.step = 0;
    }

    /// Current operating mode. Diagnostic; games don't read this
    /// directly (they infer it from the ID byte in the poll).
    pub fn mode(&self) -> PadMode {
        self.mode
    }

    /// Set the analog-stick state. `(right_x, right_y, left_x,
    /// left_y)` are each 0..=255 with `STICK_CENTER = 0x80`. Games
    /// only see these when the pad is in Analog (or Config) mode.
    pub fn set_sticks(&mut self, right_x: u8, right_y: u8, left_x: u8, left_y: u8) {
        self.right_x = right_x;
        self.right_y = right_y;
        self.left_x = left_x;
        self.left_y = left_y;
    }

    /// ID low byte reported on byte 0 of a transaction. Derived from
    /// the current mode.
    fn id_low_byte(&self) -> u8 {
        match self.mode {
            PadMode::Digital => 0x41,
            PadMode::Analog => 0x73,
            PadMode::Config => 0xF3,
        }
    }

    /// Get the byte to send at step `step` of a standard poll
    /// (command `0x42`). Step 1 = ID hi (0x5A), 2-3 = buttons, 4-7 =
    /// analog sticks when present.
    fn poll_byte(&self, step: u8) -> u8 {
        match step {
            1 => 0x5A,
            2 => !(self.buttons.bits() & 0xFF) as u8,
            3 => !((self.buttons.bits() >> 8) & 0xFF) as u8,
            4 => self.right_x,
            5 => self.right_y,
            6 => self.left_x,
            7 => self.left_y,
            _ => 0xFF,
        }
    }

    /// Process one byte of a transaction. Returns `(rx_byte,
    /// ack_is_low)` — ACK stays low (true) for all bytes except the
    /// final one of the transaction.
    fn exchange(&mut self, tx: u8) -> (u8, bool) {
        // Step 0: host sent the address byte `0x01`. Reply with
        // ID_low + hold ACK to invite the next byte.
        if self.step == 0 {
            if tx != 0x01 {
                // Spurious first byte — reset and report no-ack.
                self.reset();
                return (0xFF, false);
            }
            self.step = 1;
            return (self.id_low_byte(), true);
        }

        // Step 1: host sends the command byte. Store it and reply
        // with ID_high (`0x5A`), holding ACK to continue.
        if self.step == 1 {
            self.cmd = tx;
            self.step = 2;
            return (0x5A, true);
        }

        // Steps 2..N: payload bytes. `self.step` currently names
        // the step we're responding for (the payload index). Ack
        // stays low while that index is less than the transaction's
        // last-step; when we're answering the last byte, ack drops
        // so the host stops clocking.
        let pre_step = self.step;
        let byte = self.payload_byte(tx);
        self.step = self.step.saturating_add(1);

        let ack = pre_step < self.last_step_for_cmd(self.cmd);
        if !ack {
            self.reset();
        }
        (byte, ack)
    }

    /// Zero-indexed step of the last byte in a transaction for a
    /// given command. Ack is held (low) for every step strictly less
    /// than this value; on the last step ack drops so the host
    /// stops clocking. Poll (0x42) uses the mode's length; config
    /// + mode-set commands always use the 8-byte shape.
    fn last_step_for_cmd(&self, cmd: u8) -> u8 {
        match cmd {
            0x42 => match self.mode {
                PadMode::Digital => DIGITAL_POLL_LAST,
                PadMode::Analog | PadMode::Config => ANALOG_POLL_LAST,
            },
            0x43 | 0x44 | 0x45 | 0x46..=0x4F => ANALOG_POLL_LAST,
            // Unknown command — abort after the ID bytes (steps
            // 0 + 1) have been exchanged.
            _ => 1,
        }
    }

    /// Byte to return at the current payload step. Called AFTER the
    /// step's TX has been observed, BEFORE `self.step` advances.
    /// Dispatches on `self.cmd`.
    fn payload_byte(&mut self, tx: u8) -> u8 {
        match self.cmd {
            // 0x42 — standard button poll. 4 payload bytes in
            // Digital, 8 in Analog. `self.step` currently points at
            // the byte we're ABOUT to respond with (step 2 = buttons
            // group 1, step 3 = buttons group 2, etc).
            //
            // When the motor mapping has been set (command 0x4D),
            // this is also where rumble TX bytes land: the host
            // ships motor-command bytes alongside the poll, and we
            // dispatch by mapping slot.
            0x42 => {
                if self.mode == PadMode::Analog {
                    self.apply_rumble_tx(tx, self.step);
                }
                self.poll_byte(self.step)
            }

            // 0x43 — enter / exit config mode. Param at step 2:
            //   0x01 → enter config (save prior mode first)
            //   0x00 → exit config (restore saved mode)
            // Remaining bytes are buttons (digital or analog shape
            // depending on saved mode).
            0x43 => {
                if self.step == 2 {
                    if tx == 0x01 {
                        self.mode_before_config = self.mode;
                        self.mode = PadMode::Config;
                    } else if tx == 0x00 && self.mode == PadMode::Config {
                        self.mode = self.mode_before_config;
                    }
                }
                self.poll_byte(self.step)
            }

            // 0x44 — set mode (only valid in Config). Param at step
            // 2 chooses Digital (0x00) or Analog (0x01). We record
            // the target into `mode_before_config` so the subsequent
            // `0x43`+`0x00` exit lands us in the right mode.
            0x44 => {
                if self.mode == PadMode::Config && self.step == 2 {
                    match tx {
                        0x00 => self.mode_before_config = PadMode::Digital,
                        0x01 => self.mode_before_config = PadMode::Analog,
                        _ => {}
                    }
                }
                0x00
            }

            // 0x45 — query current mode. Returns a 6-byte status
            // block. Games probe this to confirm the pad accepted a
            // mode switch.
            0x45 => match self.step {
                2 => 0x01, // device class: DualShock
                3 => 0x02, // number of analog param words
                4 => {
                    if self.mode_before_config == PadMode::Analog {
                        0x01
                    } else {
                        0x00
                    }
                }
                5 => 0x02, // analog mode activated
                6 => 0x01, // rumble capable (stub)
                _ => 0x00,
            },

            // 0x4D — vibration-motor mapping. 6 param bytes at
            // steps 2..=7 each assign a motor to its corresponding
            // 0x42-poll slot (see `motor_mapping` docs). The
            // response is the PREVIOUS mapping byte at that slot —
            // games read the returned bytes to confirm their
            // mapping took effect.
            0x4D => {
                let slot = (self.step.saturating_sub(2)) as usize;
                if slot < self.motor_mapping.len() {
                    let prev = self.motor_mapping[slot];
                    self.motor_mapping[slot] = tx;
                    prev
                } else {
                    0x00
                }
            }

            // 0x46..=0x4F (except 0x4D handled above) — other
            // config commands (e.g. 0x46 = constant query,
            // 0x47 = extended query, 0x4C = unknown). Not modelled;
            // zero-fill matches a no-op reply that the BIOS
            // tolerates.
            0x46..=0x4C | 0x4E | 0x4F => 0x00,

            _ => 0xFF,
        }
    }

    /// Apply a rumble-control TX byte at the given step when a
    /// `0x42` poll is in progress. `step` is the same step index
    /// the response path uses — values 2..=7 map to motor slots
    /// 0..=5 (matching the 6 mapping bytes set by command 0x4D).
    fn apply_rumble_tx(&mut self, tx: u8, step: u8) {
        let slot = match step {
            2..=7 => (step - 2) as usize,
            _ => return,
        };
        if slot >= self.motor_mapping.len() {
            return;
        }
        match self.motor_mapping[slot] {
            0x00 => {
                // Small motor — binary on/off. Any value != 0xFF
                // means off; 0xFF means on. Redux matches this
                // in `pad/controller.cc`.
                self.motor_small = tx == 0xFF;
            }
            0x01 => {
                // Big motor — strength directly from the byte.
                self.motor_big = tx;
            }
            // Unmapped slot — ignore.
            _ => {}
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
        // Transaction state should have reset.
        assert_eq!(pad.step, 0);
        assert_eq!(pad.cmd, 0);
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
    fn spurious_first_byte_resets_immediately() {
        let mut pad = DigitalPad::new();
        // Anything other than 0x01 as the first byte should abort
        // without touching internal state.
        assert_eq!(pad.exchange(0xFF), (0xFF, false));
        assert_eq!(pad.step, 0);
    }

    #[test]
    fn unknown_command_aborts_after_id_bytes() {
        let mut pad = DigitalPad::new();
        assert_eq!(pad.exchange(0x01), (0x41, true));
        // Hardware acks the command byte even when the command's
        // unknown — it's the payload step that aborts.
        assert_eq!(pad.exchange(0x99), (0x5A, true));
        // Payload byte returns 0xFF + no ack.
        assert_eq!(pad.exchange(0x00), (0xFF, false));
        assert_eq!(pad.step, 0);
    }

    #[test]
    fn dualshock_config_sequence_switches_to_analog_mode() {
        let mut pad = DigitalPad::new();
        assert_eq!(pad.mode(), PadMode::Digital);
        // Enter config: 0x01, 0x43, 0x01, ... (8 bytes total).
        let _ = pad.exchange(0x01);
        let _ = pad.exchange(0x43);
        let _ = pad.exchange(0x01);
        for _ in 3..8 {
            let _ = pad.exchange(0x00);
        }
        assert_eq!(pad.mode(), PadMode::Config);

        // Switch to analog: 0x01, 0x44, 0x01, 0x02, 0, 0, 0, 0.
        assert_eq!(pad.exchange(0x01), (0xF3, true)); // config ID
        let _ = pad.exchange(0x44);
        let _ = pad.exchange(0x01); // analog
        for _ in 3..8 {
            let _ = pad.exchange(0x00);
        }
        // Still in Config mode until we exit it.
        assert_eq!(pad.mode(), PadMode::Config);

        // Exit config: 0x01, 0x43, 0x00, ...
        let _ = pad.exchange(0x01);
        let _ = pad.exchange(0x43);
        let _ = pad.exchange(0x00);
        for _ in 3..8 {
            let _ = pad.exchange(0x00);
        }
        assert_eq!(pad.mode(), PadMode::Analog);
    }

    #[test]
    fn analog_poll_returns_8_bytes_with_stick_axes() {
        let mut pad = DigitalPad::new();
        // Force pad into analog mode directly for the test.
        pad.mode = PadMode::Analog;
        pad.set_sticks(0x10, 0x20, 0x30, 0x40);
        // Poll in analog mode.
        assert_eq!(pad.exchange(0x01), (0x73, true)); // analog ID
        assert_eq!(pad.exchange(0x42), (0x5A, true));
        assert_eq!(pad.exchange(0x00).0, 0xFF); // buttons low — all released
        assert_eq!(pad.exchange(0x00).0, 0xFF); // buttons high — all released
        assert_eq!(pad.exchange(0x00).0, 0x10); // right X
        assert_eq!(pad.exchange(0x00).0, 0x20); // right Y
        assert_eq!(pad.exchange(0x00).0, 0x30); // left X
        // Last byte — no ack.
        assert_eq!(pad.exchange(0x00), (0x40, false));
    }

    #[test]
    fn sticks_default_to_center() {
        let pad = DigitalPad::new();
        assert_eq!(pad.right_x, STICK_CENTER);
        assert_eq!(pad.right_y, STICK_CENTER);
        assert_eq!(pad.left_x, STICK_CENTER);
        assert_eq!(pad.left_y, STICK_CENTER);
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

    #[test]
    fn rumble_mapping_defaults_to_unmapped() {
        let pad = DigitalPad::new();
        assert_eq!(pad.motor_mapping, [0xFF; 6]);
        assert_eq!(pad.motor_state(), (false, 0));
    }

    #[test]
    fn command_0x4d_stores_mapping_and_returns_previous() {
        let mut pad = DigitalPad::new();
        // Preload a distinctive mapping.
        pad.motor_mapping = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let mapping = [0x00u8, 0x01, 0xFF, 0xFF, 0xFF, 0xFF];
        // Run the 0x4D transaction.
        assert_eq!(pad.exchange(0x01).0, 0x41); // ID low (Digital is fine for test)
        assert_eq!(pad.exchange(0x4D).0, 0x5A);
        // Send 6 mapping bytes, one per payload slot; expect the
        // previous value in the response.
        for (i, &m) in mapping.iter().enumerate() {
            let expected_prev = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF][i];
            let (rx, _) = pad.exchange(m);
            assert_eq!(rx, expected_prev, "slot {i}");
        }
        assert_eq!(pad.motor_mapping, mapping);
    }

    #[test]
    fn analog_poll_with_mapping_drives_motors() {
        let mut pad = DigitalPad::new();
        pad.mode = PadMode::Analog;
        // slot 0 -> small motor, slot 1 -> big motor.
        pad.motor_mapping = [0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF];
        // Poll: 0x01, 0x42, then 6 TX bytes. First TX byte is the
        // small-motor on/off (0xFF = on); second is big-motor
        // strength.
        assert_eq!(pad.exchange(0x01).0, 0x73);
        assert_eq!(pad.exchange(0x42).0, 0x5A);
        // Payload step 2 — small motor slot. 0xFF turns it on.
        let _ = pad.exchange(0xFF);
        // Payload step 3 — big motor slot. 0x80 = half strength.
        let _ = pad.exchange(0x80);
        // Consume remaining payload bytes.
        for _ in 4..=7 {
            let _ = pad.exchange(0x00);
        }
        assert_eq!(pad.motor_state(), (true, 0x80));
    }

    #[test]
    fn small_motor_turns_off_with_non_ff_tx() {
        let mut pad = DigitalPad::new();
        pad.mode = PadMode::Analog;
        pad.motor_mapping = [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        pad.motor_small = true;
        // Poll.
        let _ = pad.exchange(0x01);
        let _ = pad.exchange(0x42);
        // Any non-0xFF value turns the small motor off.
        let _ = pad.exchange(0x00);
        for _ in 3..=7 {
            let _ = pad.exchange(0x00);
        }
        assert_eq!(pad.motor_state(), (false, 0));
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
