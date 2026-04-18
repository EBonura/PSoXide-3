//! CD-ROM controller.
//!
//! **Full implementation**, not a stub — the plan is to carry this
//! from BIOS-boot-without-disc (Milestone B) all the way through
//! real disc reads (Milestone D onward) and XA audio (Milestone F).
//! The file will grow by phase:
//!
//! - **6a (this file)**: register infrastructure — the index-based
//!   MMIO dispatch, the three FIFOs (parameter, response, data),
//!   the IRQ flag/mask register model, the raw status byte.
//! - **6b**: command dispatcher with async-response scheduling.
//!   Commands queue first- and second-response events at specific
//!   cycle offsets; `tick` fires them when their time comes.
//! - **6c**: core commands that appear in every BIOS boot —
//!   Sync / GetStat / Init / Demute / GetID / Pause.
//! - **6d**: disc-present commands — SetLoc, SeekL/SeekP, ReadN/ReadS,
//!   sector data streaming into the data FIFO + DMA channel 3.
//! - **6e**: ISO9660 filesystem hook from `psx-iso`.
//! - **6f**: audio plumbing — volume matrix, XA decode (deferred
//!   to Milestone F when it actually matters).
//!
//! **Reference**: nocash PSX-SPX "CDROM Drive" and
//! `pcsx-redux/src/core/cdrom.cc`. Status-byte bits + IRQ types +
//! index semantics all follow those two.
//!
//! This module is parity-safe at the register level — software that
//! reads status / queues parameters / pops responses sees the values
//! Redux would return. Command side effects (seeking, reading) start
//! landing in 6b onwards.

use std::collections::VecDeque;

use psx_iso::{msf_to_lba, Disc};

/// Base MMIO address — the whole controller fits in 4 bytes at
/// `0x1F80_1800..=0x1F80_1803`.
pub const BASE: u32 = 0x1F80_1800;
/// Range end (exclusive) — `BASE + 4`.
pub const END: u32 = 0x1F80_1804;

/// Status-byte bits (read from `0x1F80_1800` at any index).
#[allow(dead_code)]
pub mod status_bit {
    /// Index (low 2 bits) — selects which sub-register is visible at
    /// `0x1F80_1801..=0x1F80_1803`. Written via `0x1F80_1800`.
    pub const INDEX_MASK: u8 = 0b0000_0011;
    /// ADPCM-decoder busy.
    pub const ADPCM_BUSY: u8 = 1 << 2;
    /// Parameter FIFO is empty (room for writes).
    pub const PARAM_FIFO_EMPTY: u8 = 1 << 3;
    /// Parameter FIFO is *not* full (software checks this before push).
    pub const PARAM_FIFO_NOT_FULL: u8 = 1 << 4;
    /// Response FIFO is *not* empty (something to pop).
    pub const RESPONSE_FIFO_NOT_EMPTY: u8 = 1 << 5;
    /// Data FIFO is *not* empty (sector bytes available).
    pub const DATA_FIFO_NOT_EMPTY: u8 = 1 << 6;
    /// A command is in flight — cleared when the first response arrives.
    pub const TRANSMISSION_BUSY: u8 = 1 << 7;
}

/// Interrupt types (value written to `0x1F80_1803 idx=1` and visible
/// via the IRQ-flag register). Only values 1..=5 are meaningful;
/// value 0 means "no interrupt" and is what the BIOS writes to ack.
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IrqType {
    /// No interrupt — canonical cleared state.
    None = 0,
    /// Async sector-data-ready — fires for each sector during ReadN/S.
    DataReady = 1,
    /// Second response — completion of a command whose 1st response
    /// indicated the action would take time (seek, read, init).
    Complete = 2,
    /// First response — command accepted, ~50 k cycles after the write.
    Acknowledge = 3,
    /// Fourth response — end of data for Play/ReadS/ReadN on bounds.
    DataEnd = 4,
    /// Error response — command rejected, disc error, etc.
    Error = 5,
}

impl IrqType {
    #[allow(dead_code)]
    fn from_u8(v: u8) -> Self {
        match v & 0x7 {
            1 => IrqType::DataReady,
            2 => IrqType::Complete,
            3 => IrqType::Acknowledge,
            4 => IrqType::DataEnd,
            5 => IrqType::Error,
            _ => IrqType::None,
        }
    }
}

/// Drive-status byte (a.k.a. "stat" — the first byte of every
/// command response). Separate from the MMIO status byte above.
///
/// On cold boot with no disc, `SHELL_OPEN` is set and `MOTOR_ON`
/// is clear. Once the BIOS `Init` runs, the motor spins up; if a
/// disc is present, `SHELL_OPEN` clears and `MOTOR_ON` sets.
#[allow(dead_code)]
pub mod drive_status_bit {
    /// Error bit (set on invalid command, disc error).
    pub const ERROR: u8 = 1 << 0;
    /// Spindle motor is on (disc spinning).
    pub const MOTOR_ON: u8 = 1 << 1;
    /// Last seek failed.
    pub const SEEK_ERROR: u8 = 1 << 2;
    /// GetID detected a disc mismatch / unlicensed disc.
    pub const ID_ERROR: u8 = 1 << 3;
    /// Drive shell is open (no disc / cover lifted).
    pub const SHELL_OPEN: u8 = 1 << 4;
    /// Drive is currently reading sectors (ReadN / ReadS).
    pub const READING: u8 = 1 << 5;
    /// Drive is currently seeking.
    pub const SEEKING: u8 = 1 << 6;
    /// Drive is currently playing CD-DA audio.
    pub const PLAYING: u8 = 1 << 7;
}

/// Depth of both FIFOs in the controller. Real hardware is 16 bytes
/// for parameter / response, 2352 bytes for the sector data buffer.
const PARAM_FIFO_DEPTH: usize = 16;
const RESPONSE_FIFO_DEPTH: usize = 16;

/// Canonical cycle delays for command responses, transcribed from
/// Redux's `core/cdrom.cc`. Exact match is the difference between
/// our CDROM events landing on the same instructions as Redux's
/// and silently scheduling them thousands of cycles apart — which
/// compounds into full game-state divergence over tens of millions
/// of instructions.
///
/// Redux cross-references (line numbers from the upstream file):
///
/// - `AddIrqQueue(m_cmd, 0x800)` — universal first-response delay
///   (L1284). Every command's ack fires 2048 cycles after issue.
/// - `AddIrqQueue(CdlInit + 0x100, 20480)` — Init / GetID second
///   response, ~4.4 µs, observed across boot roms (L900, and the
///   shellopen path L938 scheduleCDLidIRQ(20480)).
/// - `cdReadTime = psxClockSpeed / 75` — one PSX CD-frame period
///   (L135). At double-speed (mode bit 7) a sector is ready every
///   `cdReadTime` cycles; at single-speed, `cdReadTime * 2`. We
///   don't track the mode bit yet — the BIOS's disc probe uses
///   double-speed, so pick that as the default.
/// - `scheduleCDPlayIRQ(SEEK_DONE ? 0x800 : cdReadTime * 4)` —
///   SeekL / SeekP second response (L875). If the target is already
///   seeked, quick ack; otherwise a full seek-time equivalent.
const FIRST_RESPONSE_CYCLES: u64 = 0x800; // 2048
const INIT_SECOND_RESPONSE_CYCLES: u64 = 20_480;
const GETID_SECOND_RESPONSE_CYCLES: u64 = 20_480;
const SEEK_SECOND_RESPONSE_CYCLES: u64 = CD_READ_TIME * 4; // ≈ 1,806,336
/// PSX system clock / CD frames per second. `33_868_800 / 75`.
/// Redux's `cdReadTime`.
const CD_READ_TIME: u64 = 451_584;
// Sector-read cycles are now derived per-instance from the current
// mode byte via [`CdRom::sector_read_cycles`]: double-speed =
// `CD_READ_TIME`, single-speed = `CD_READ_TIME * 2`.

/// A deferred response: when `bus.cycles` passes `deadline`, the
/// event's bytes land in the response FIFO and its IRQ type fires.
///
/// `deadline` has two interpretations depending on `rebased`:
/// - `rebased=false` → deadline is a *relative offset in cycles*
///   from the moment [`tick`] first sees the event. `tick` rebases
///   it to an absolute cycle and sets `rebased=true`.
/// - `rebased=true` → deadline is an absolute bus-cycle count.
///
/// Encoding relative-vs-absolute explicitly (instead of relying on
/// comparing against `cycles_now`) avoids a re-rebase ambiguity
/// when an event is re-queued due to IRQ-flag collision.
#[derive(Debug, Clone)]
struct PendingEvent {
    deadline: u64,
    rebased: bool,
    irq: IrqType,
    bytes: Vec<u8>,
}

/// CD-ROM controller state.
pub struct CdRom {
    /// Index register low 2 bits — selects the register visible at
    /// each sub-port for the next read/write.
    index: u8,
    /// Drive status byte returned by `GetStat` and embedded in the
    /// first byte of most responses. Read by phase-6c command handlers.
    #[allow(dead_code)]
    drive_status: u8,
    /// Parameter FIFO — software pushes here before invoking a
    /// command; the controller drains it when the command runs.
    params: VecDeque<u8>,
    /// Response FIFO — command responses arrive here for software
    /// to pop via `0x1F80_1801`.
    responses: VecDeque<u8>,
    /// IRQ enable mask — low 3 bits; an IRQ fires only if its type
    /// bit is enabled. BIOS writes via `0x1F80_1802 idx=1`.
    irq_mask: u8,
    /// Currently-raised IRQ type; cleared by writing 1s to the low
    /// 3 bits of `0x1F80_1803 idx=1`.
    irq_flag: u8,
    /// Scheduled events waiting for their cycle deadlines. Processed
    /// in order by [`CdRom::tick`].
    pending: VecDeque<PendingEvent>,
    /// `true` once the BIOS has sent an `Init` command and the
    /// motor has completed spinning up. Gates commands that need the
    /// motor (Seek, Read, Play).
    motor_on: bool,
    /// Whether a disc is currently inserted. For "no-disc" boots
    /// this stays `false`, and commands that expect a disc (GetID,
    /// ReadN) return error responses.
    disc_present: bool,
    /// Most-recent SetLoc BCD target (minute, second, frame). Used
    /// by SeekL / ReadN to know where to go.
    setloc_msf: (u8, u8, u8),
    /// Total commands dispatched since reset — diagnostic counter.
    commands_dispatched: u64,
    /// Diagnostic histogram of each command byte seen — `[0x00..=0x1F]`
    /// is enough to capture every BIOS command. Exposed via
    /// [`CdRom::command_histogram`] for `smoke_draw`.
    command_hist: [u32; 32],
    /// The most recently dispatched command byte. Diagnostic only
    /// — used by the `cdrom_probe` example to log exactly which
    /// command was just issued at each `commands_dispatched` bump.
    last_command: u8,
    /// Total bytes popped from the data FIFO via MMIO reads.
    /// Diagnostic. If this grows in lockstep with DataReady events
    /// the BIOS is actually consuming the sectors we delivered; if
    /// it's stuck, the BIOS's read path is blocked on something
    /// we're not signalling (BFRD / request-register / IRQ ack).
    data_fifo_pops: u64,
    /// Loaded disc image, if any. When `Some`, `disc_present` is also
    /// true and GetID / ReadN follow the disc-present paths; when
    /// `None`, they fall back to the "please insert disc" path.
    disc: Option<Disc>,
    /// Data FIFO — 2048 bytes of sector user data, drained by MMIO
    /// reads at `0x1F80_1802` or by DMA channel 3. Filled by each
    /// DataReady event during an active ReadN / ReadS.
    data_fifo: VecDeque<u8>,
    /// Set while a read is in progress; controls whether new
    /// DataReady events chain into further sectors.
    reading: bool,
    /// Next sector LBA to deliver during an active read.
    read_lba: u32,
    /// Last SetMode byte written by the CPU. Bit layout:
    ///   0: CD-DA enable (for Play command)
    ///   1: auto-pause on track boundary
    ///   2: play-report enable
    ///   3: XA filter enable
    ///   4: ignore-bit (internal)
    ///   5: sector size (0 = 2048 bytes / data only, 1 = 2340 bytes / full)
    ///   6: XA ADPCM enable
    ///   7: speed (0 = single-speed 1x, 1 = double-speed 2x)
    ///
    /// We act on bit 7 (speed) for sector pacing and bit 6 (XA
    /// ADPCM enable) for in-stream audio decode.
    mode: u8,
    /// Decoded stereo sample buffer — filled by XA ADPCM decode
    /// when an audio sector arrives. Drained by the bus each tick
    /// and pushed to the SPU's CD audio input.
    cd_audio: VecDeque<(i16, i16)>,
    /// XA ADPCM decoder left-channel filter history (y0, y1).
    /// Persists across blocks within a file; reset between XA
    /// files / on Pause.
    xa_left: crate::spu::XaDecoderState,
    /// XA right-channel history.
    xa_right: crate::spu::XaDecoderState,
}

impl CdRom {
    /// Fresh controller — shell open, motor off, all FIFOs empty,
    /// IRQ disabled. Matches hardware state a few cycles after reset,
    /// before the BIOS has had a chance to write anything.
    pub fn new() -> Self {
        Self {
            index: 0,
            // Cold boot: on a closed shell with no disc seated, we
            // want the BIOS to reach the "Please insert disc" shell —
            // that needs SHELL_OPEN clear (lid closed) so the Init
            // command spins the motor up without erroring.
            drive_status: 0,
            params: VecDeque::with_capacity(PARAM_FIFO_DEPTH),
            responses: VecDeque::with_capacity(RESPONSE_FIFO_DEPTH),
            irq_mask: 0,
            irq_flag: 0,
            pending: VecDeque::new(),
            motor_on: false,
            disc_present: false,
            setloc_msf: (0, 0, 0),
            commands_dispatched: 0,
            command_hist: [0; 32],
            last_command: 0,
            data_fifo_pops: 0,
            disc: None,
            data_fifo: VecDeque::new(),
            reading: false,
            read_lba: 0,
            // Power-on mode: double-speed, no XA, data-only 2048-byte
            // sectors. Matches the BIOS's probe-time expectation — it
            // issues SetMode 0x80 (double-speed) before its first
            // ReadN. A fresh emulator reset without an intervening
            // SetMode still uses double-speed, matching the prior
            // behaviour of always-CD_READ_TIME pacing.
            mode: 0x80,
            cd_audio: VecDeque::new(),
            xa_left: crate::spu::XaDecoderState::new(),
            xa_right: crate::spu::XaDecoderState::new(),
        }
    }

    /// Pull all pending CD audio samples — called by the bus once
    /// per frame, then forwarded to `Spu::feed_cd_audio`. Returns
    /// stereo pairs in playback order (oldest first). Empty when
    /// no XA / CD-DA decode has produced samples since the last
    /// drain.
    pub fn drain_cd_audio(&mut self) -> Vec<(i16, i16)> {
        self.cd_audio.drain(..).collect()
    }

    /// Queue depth of the CD audio buffer — diagnostic.
    pub fn cd_audio_queue_len(&self) -> usize {
        self.cd_audio.len()
    }

    /// Load a disc image. After this, GetID returns the licensed-disc
    /// response and ReadN streams real sector data through the
    /// DataReady event chain.
    ///
    /// Inserting a disc also spins up the motor. On real hardware the
    /// motor starts when the shell is closed with a disc seated, even
    /// before the BIOS issues `Init`. Without this, `GetID`'s stat
    /// byte reports motor-off (0x00), which the BIOS's shell poll
    /// treats as "drive not ready yet" and never advances past the
    /// disc-probe loop to issue `SetLoc + SeekL + ReadN`.
    ///
    /// `insert_disc(None)` "ejects" — disc_present flips false, the
    /// motor stops, any in-flight read is cancelled, and the next
    /// `GetID` returns the no-disc response again.
    pub fn insert_disc(&mut self, disc: Option<Disc>) {
        self.disc = disc;
        self.disc_present = self.disc.is_some();
        self.motor_on = self.disc_present;
        self.reading = false;
        self.data_fifo.clear();
    }

    /// `true` when `phys` is inside the CD-ROM MMIO range.
    pub fn contains(phys: u32) -> bool {
        (BASE..END).contains(&phys)
    }

    /// 8-bit read through the index-selected register at `phys`.
    pub fn read8(&mut self, phys: u32) -> u8 {
        let offset = (phys - BASE) as u8;
        match (offset, self.index) {
            // 0x1F80_1800 — status byte (same at every index).
            (0, _) => self.status_byte(),
            // 0x1F80_1801 — response FIFO (any index).
            (1, _) => self.pop_response(),
            // 0x1F80_1802 — data FIFO (any index).
            (2, _) => {
                self.data_fifo_pops = self.data_fifo_pops.saturating_add(1);
                self.data_fifo.pop_front().unwrap_or(0)
            }
            // 0x1F80_1803 — index-dependent:
            //   idx=0 → interrupt enable,
            //   idx=1 → interrupt flag,
            //   idx=2 → mirror of enable,
            //   idx=3 → mirror of flag.
            (3, 0) | (3, 2) => self.irq_mask | 0xE0,
            (3, 1) | (3, 3) => self.irq_flag | 0xE0,
            _ => 0,
        }
    }

    /// 8-bit write through the index-selected register at `phys`.
    pub fn write8(&mut self, phys: u32, value: u8) {
        let offset = (phys - BASE) as u8;
        match (offset, self.index) {
            // 0x1F80_1800 write — set the index.
            (0, _) => self.index = value & status_bit::INDEX_MASK,
            // 0x1F80_1801 idx=0 — command register. Queue for 6b.
            (1, 0) => self.queue_command(value),
            // 0x1F80_1801 idx=1/2/3 — audio sound-map / CD-to-SPU
            // volume. Accepted for now; XA audio arrives in 6f.
            (1, _) => {}
            // 0x1F80_1802 idx=0 — parameter FIFO push.
            (2, 0) => self.push_param(value),
            // 0x1F80_1802 idx=1 — interrupt enable.
            (2, 1) => self.irq_mask = value & 0x1F,
            // 0x1F80_1802 idx=2/3 — audio volume.
            (2, _) => {}
            // 0x1F80_1803 idx=0 — request register (data transfer on,
            // command-buffer reset, etc.). Bit 7 = want-data. Full
            // modelling arrives with sector reads.
            (3, 0) => {
                // Bit 6 = BFRD (reset). If set, clear parameter FIFO.
                if value & 0x40 != 0 {
                    self.params.clear();
                }
            }
            // 0x1F80_1803 idx=1 — acknowledge interrupts (write-1-to-
            // clear on the low 5 bits; bit 6 resets the param FIFO too).
            (3, 1) => {
                self.irq_flag &= !(value & 0x1F);
                if value & 0x40 != 0 {
                    self.params.clear();
                }
            }
            // 0x1F80_1803 idx=2/3 — audio volume matrix.
            (3, _) => {}
            _ => {}
        }
    }

    /// Compose the MMIO status byte from live FIFO + index state.
    fn status_byte(&self) -> u8 {
        let mut s = self.index;
        if self.params.is_empty() {
            s |= status_bit::PARAM_FIFO_EMPTY;
        }
        if self.params.len() < PARAM_FIFO_DEPTH {
            s |= status_bit::PARAM_FIFO_NOT_FULL;
        }
        if !self.responses.is_empty() {
            s |= status_bit::RESPONSE_FIFO_NOT_EMPTY;
        }
        if !self.data_fifo.is_empty() {
            s |= status_bit::DATA_FIFO_NOT_EMPTY;
        }
        // Transmission busy (bit 7) + ADPCM busy (bit 2) come from
        // subsystems not yet wired in; both zero for now.
        s
    }

    fn pop_response(&mut self) -> u8 {
        self.responses.pop_front().unwrap_or(0)
    }

    fn push_param(&mut self, value: u8) {
        if self.params.len() < PARAM_FIFO_DEPTH {
            self.params.push_back(value);
        }
    }

    /// Execute a command received on the command port. The command's
    /// handler synthesises its first (and optional second) response
    /// and schedules them into the pending-events queue.
    ///
    /// A handful of commands use the parameter FIFO for arguments
    /// (SetLoc MSF, SetMode, Test sub-op). The parameters are drained
    /// inline by the handler.
    fn queue_command(&mut self, command: u8) {
        self.commands_dispatched += 1;
        self.last_command = command;
        if (command as usize) < self.command_hist.len() {
            self.command_hist[command as usize] += 1;
        }

        // Drain parameters into a local vec — handlers need them and
        // pop-order matches push-order.
        let params: Vec<u8> = self.params.drain(..).collect();

        match command {
            // Sync / NOP — acts as GetStat.
            0x00 | 0x01 => self.cmd_getstat(),
            // SetLoc: 3 BCD bytes (minute, second, frame).
            0x02 => self.cmd_setloc(&params),
            // Play (CD-DA): 1 optional byte (track). Starts Red-Book
            // audio playback. We accept the command and drive the
            // reading state machine so the BIOS's follow-up polls see
            // the drive active; actual CD-DA sample data isn't
            // sourced yet (requires track-level data from the .cue).
            0x03 => self.cmd_play(&params),
            // ReadN (read with auto-retry). Without a disc, error.
            0x06 => self.cmd_read(),
            // Stop: halt motor.
            0x08 => self.cmd_stop(),
            // Pause: halt reads but keep motor on.
            0x09 => self.cmd_pause(),
            // Init: reset drive + spin motor + clear mode.
            0x0A => self.cmd_init(),
            // Mute / Demute.
            0x0B | 0x0C => self.cmd_getstat(),
            // SetMode: 1 byte (speed, CD-DA enable, filter, etc.).
            // We accept and store; full behaviour in 6d.
            0x0E => self.cmd_setmode(&params),
            // SeekL: seek to last-SetLoc position (logical sectors).
            0x15 => self.cmd_seek(),
            // Test: sub-op in param[0]. Most common: 0x20 = get
            // drive version / BIOS date (6-byte response).
            0x19 => self.cmd_test(&params),
            // GetID: "what kind of disc is this?"
            0x1A => self.cmd_getid(),
            _ => self.cmd_getstat(),
        }
    }

    // --- Command handlers ---

    /// Schedule a first-response IRQ containing the given bytes, at
    /// now + `FIRST_RESPONSE_CYCLES`. Used by almost every command's
    /// acknowledge path.
    ///
    /// Deadlines are stored as "relative cycles from scheduling time"
    /// and rebased into absolute bus cycles by the next [`tick`] call
    /// (which is the first place we know `now`). Sentinel `u64::MAX`
    /// means "already absolute, do not rebase."
    fn schedule_first_response(&mut self, bytes: Vec<u8>) {
        self.pending.push_back(PendingEvent {
            deadline: FIRST_RESPONSE_CYCLES,
            rebased: false,
            irq: IrqType::Acknowledge,
            bytes,
        });
    }

    /// Schedule a second-response IRQ. `additional_delay` is time
    /// *after* the first response — so the actual cycle delta from
    /// command issue is `FIRST_RESPONSE_CYCLES + additional_delay`.
    /// This matters: if you pass a small delay here without adding
    /// the first-response time, the second response fires BEFORE the
    /// first, and the BIOS sees an INT2 (Complete) with 8 data bytes
    /// before its expected INT3 (Acknowledge) — breaking the
    /// command-response chain completely and stranding the boot in
    /// the shell.
    fn schedule_second_response(&mut self, bytes: Vec<u8>, additional_delay: u64) {
        self.pending.push_back(PendingEvent {
            deadline: FIRST_RESPONSE_CYCLES.saturating_add(additional_delay),
            rebased: false,
            irq: IrqType::Complete,
            bytes,
        });
    }

    /// Like [`schedule_second_response`] but the IRQ type is Error
    /// (INT5). Used for the second reply of commands that fail because
    /// the disc isn't present — the first reply is still an ack (INT3)
    /// so the BIOS can distinguish "command received" from "command
    /// completed with error".
    fn schedule_second_error(&mut self, bytes: Vec<u8>, additional_delay: u64) {
        self.pending.push_back(PendingEvent {
            deadline: FIRST_RESPONSE_CYCLES.saturating_add(additional_delay),
            rebased: false,
            irq: IrqType::Error,
            bytes,
        });
    }

    fn schedule_error_response(&mut self, bytes: Vec<u8>) {
        self.pending.push_back(PendingEvent {
            deadline: FIRST_RESPONSE_CYCLES,
            rebased: false,
            irq: IrqType::Error,
            bytes,
        });
    }

    fn stat_byte(&self) -> u8 {
        let mut s = self.drive_status;
        if self.motor_on {
            s |= drive_status_bit::MOTOR_ON;
        }
        s
    }

    fn cmd_getstat(&mut self) {
        let stat = self.stat_byte();
        self.schedule_first_response(vec![stat]);
    }

    fn cmd_setloc(&mut self, params: &[u8]) {
        if params.len() >= 3 {
            self.setloc_msf = (params[0], params[1], params[2]);
        }
        let stat = self.stat_byte();
        self.schedule_first_response(vec![stat]);
    }

    fn cmd_setmode(&mut self, params: &[u8]) {
        if let Some(&m) = params.first() {
            self.mode = m;
        }
        let stat = self.stat_byte();
        self.schedule_first_response(vec![stat]);
    }

    /// Cycles between DataReady events for the current speed. Double-
    /// speed (mode bit 7 set) reads at 150 sectors/sec → `CD_READ_TIME`
    /// cycles per sector. Single-speed reads at half that rate →
    /// `CD_READ_TIME * 2` cycles per sector. Matches Redux's pacing in
    /// `core/cdriso.cc`.
    fn sector_read_cycles(&self) -> u64 {
        if self.mode & 0x80 != 0 {
            CD_READ_TIME
        } else {
            CD_READ_TIME * 2
        }
    }

    fn cmd_stop(&mut self) {
        self.motor_on = false;
        self.schedule_first_response(vec![self.stat_byte()]);
        let stat = self.stat_byte();
        self.schedule_second_response(vec![stat], SEEK_SECOND_RESPONSE_CYCLES);
    }

    fn cmd_pause(&mut self) {
        // Pause halts the sector-read chain but leaves the motor on.
        // Missing this flip meant DataReady events kept chaining
        // `load_next_sector + schedule_sector_event` indefinitely
        // after the BIOS asked us to pause, producing a runaway
        // pending queue that burned the entire CPU budget on
        // peripheral-scheduling overhead.
        self.reading = false;
        self.schedule_first_response(vec![self.stat_byte()]);
        let stat = self.stat_byte();
        self.schedule_second_response(vec![stat], SEEK_SECOND_RESPONSE_CYCLES);
    }

    fn cmd_init(&mut self) {
        // Init is a drive reset — also halt any in-flight read so
        // DataReady chains from a previous ReadN don't keep firing
        // across the reset boundary.
        self.reading = false;
        self.data_fifo.clear();
        // 1st response: current (pre-spin) stat.
        self.schedule_first_response(vec![self.stat_byte()]);
        // Spin up and post-init stat.
        self.motor_on = true;
        let stat = self.stat_byte();
        self.schedule_second_response(vec![stat], INIT_SECOND_RESPONSE_CYCLES);
    }

    fn cmd_seek(&mut self) {
        // Need a disc / motor. Without disc we still "seek" but it
        // succeeds immediately on the real drive — BIOS rarely calls
        // SeekL without a disc.
        self.schedule_first_response(vec![self.stat_byte()]);
        let stat = self.stat_byte() | drive_status_bit::SEEKING;
        self.schedule_second_response(vec![stat], SEEK_SECOND_RESPONSE_CYCLES);
    }

    fn cmd_read(&mut self) {
        if !self.disc_present {
            // No disc: two-phase response like the other error-returning
            // commands. First an ack (INT3) with stat so the BIOS
            // confirms we got the command, then an error (INT5) a bit
            // later with stat|ERROR + error code 0x80 (shell open /
            // no disc). Sending only the error IRQ confuses the BIOS's
            // command-state machine which expects the ack first.
            self.schedule_first_response(vec![self.stat_byte()]);
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_second_error(vec![stat, 0x80], FIRST_RESPONSE_CYCLES);
            return;
        }
        // First response: ack with READING set.
        self.schedule_first_response(vec![self.stat_byte() | drive_status_bit::READING]);
        // Kick off sector delivery.
        self.reading = true;
        let (m, s, f) = self.setloc_msf;
        self.read_lba = msf_to_lba(m, s, f);
        // Schedule the first DataReady event. Subsequent sectors are
        // chained in `tick` so the BIOS sees a steady stream.
        self.schedule_sector_event();
    }

    /// Enqueue the next DataReady event for the in-flight read. The
    /// event carries a single stat byte; the actual sector data is
    /// copied into `data_fifo` when the event fires (in `tick`).
    fn schedule_sector_event(&mut self) {
        let stat = self.stat_byte() | drive_status_bit::READING;
        self.pending.push_back(PendingEvent {
            deadline: self.sector_read_cycles(),
            rebased: false,
            irq: IrqType::DataReady,
            bytes: vec![stat],
        });
    }

    /// On DataReady event firing: populate the data FIFO with the
    /// next sector's user data and bump the read LBA. When the
    /// sector's subheader marks it as an XA audio block and mode
    /// bit 6 (XA ADPCM enable) is set, we ALSO decode the audio
    /// half into `cd_audio` for the SPU's CD input. Called from
    /// `tick` once per sector event.
    fn load_next_sector(&mut self) {
        let lba = self.read_lba;
        self.read_lba = self.read_lba.wrapping_add(1);
        if let Some(disc) = self.disc.as_ref() {
            // If XA mode is on, inspect the full raw sector's
            // subheader to decide "audio or data". Game games
            // with XA-streamed cutscenes use a single ReadN to
            // pull both sector kinds; audio sectors go to the SPU
            // and skip the CPU-visible data FIFO.
            if self.mode & 0x40 != 0 {
                if let Some(raw) = disc.read_sector_raw(lba) {
                    if let Some(samples) = decode_xa_audio_sector(
                        raw,
                        &mut self.xa_left,
                        &mut self.xa_right,
                    ) {
                        // XA audio sector — queue samples, leave
                        // data FIFO empty (games discard it anyway
                        // for audio sectors).
                        let cap = 44_100; // ~1 s at SPU rate
                        let overflow =
                            (self.cd_audio.len() + samples.len()).saturating_sub(cap);
                        for _ in 0..overflow {
                            self.cd_audio.pop_front();
                        }
                        self.cd_audio.extend(samples.iter().copied());
                        self.data_fifo.clear();
                        return;
                    }
                }
            }
            if let Some(user) = disc.read_sector_user(lba) {
                self.data_fifo.clear();
                self.data_fifo.extend(user.iter().copied());
                return;
            }
        }
        // Past end of disc — stop the read and leave the FIFO empty.
        self.reading = false;
    }

    /// Legacy helper kept for the `0x04` / `0x05` forward / backward
    /// commands which still use "ack or nodisc-error" semantics.
    #[allow(dead_code)]
    fn cmd_simple_stat_or_nodisc(&mut self) {
        if self.disc_present {
            self.schedule_first_response(vec![self.stat_byte()]);
        } else {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
        }
    }

    /// CdlPlay (0x03) — CD-DA playback. Parameter is an optional
    /// track number (BCD); when absent, playback continues from
    /// the last SetLoc position.
    ///
    /// We accept the command, mark the drive as playing, and
    /// respond with status. Real audio-sample delivery requires
    /// track-level CD-DA data from the `.cue` (most redump
    /// images interleave CD-DA tracks as separate `.bin`s); we
    /// don't have that plumbing yet, so `cd_audio` stays empty
    /// and the SPU sees silence from the CD source. The command
    /// itself completes successfully so the game's event loop
    /// doesn't hang.
    fn cmd_play(&mut self, _params: &[u8]) {
        if !self.disc_present {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
            return;
        }
        // Stat byte with PLAY bit (0x80) set so GetStat polls return
        // "playing" — match PSX-SPX drive_status_bit layout.
        self.motor_on = true;
        let stat = self.stat_byte() | drive_status_bit::PLAYING;
        self.schedule_first_response(vec![stat]);
    }

    fn cmd_test(&mut self, params: &[u8]) {
        // Only Test 0x20 (drive version / BIOS date) is commonly used
        // by the BIOS. Return the PSX SCPH-5502 canonical 6-byte
        // response; real responses vary by firmware but the BIOS
        // doesn't check.
        match params.first().copied() {
            Some(0x20) => {
                // 6-byte: YY MM DD VER
                self.schedule_first_response(vec![0x94, 0x09, 0x19, 0xC0]);
            }
            _ => self.cmd_getstat(),
        }
    }

    fn cmd_getid(&mut self) {
        if self.disc_present {
            // Licensed PS1 PAL disc placeholder. 8 bytes:
            //   stat, flags, disc_type, 00, 'S','C','E','A'
            let stat = self.stat_byte();
            self.schedule_first_response(vec![stat]);
            self.schedule_second_response(
                vec![0x02, 0x00, 0x20, 0x00, b'S', b'C', b'E', b'A'],
                GETID_SECOND_RESPONSE_CYCLES,
            );
        } else {
            // No disc: 1st response (INT3) with stat, 2nd response
            // (INT5, Error) with the shell-recognised "no disc"
            // pattern. The second response MUST be Error, not
            // Complete - the BIOS dispatches on irq_flag and runs a
            // different code path for INT5 that transitions the shell
            // state machine from "probing" to "show insert-disc screen".
            self.schedule_first_response(vec![self.stat_byte()]);
            self.schedule_second_error(
                vec![0x08, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                GETID_SECOND_RESPONSE_CYCLES,
            );
        }
    }

    /// Advance pending events by `cycles_now` (absolute bus cycle
    /// count). Events whose deadlines are in the past deliver their
    /// bytes into the response FIFO and raise their IRQ type (stored
    /// in `irq_flag`; actual CPU-facing wake-up happens via the IRQ
    /// controller, which the caller raises when `irq_flag` transitions
    /// non-zero).
    ///
    /// Returns `true` if this call raised an IRQ that was previously
    /// clear — the caller (Bus) uses that to poke `IrqSource::Cdrom`.
    pub fn tick(&mut self, cycles_now: u64) -> bool {
        let mut raised = false;
        // Rebase newly-scheduled events from relative cycles into
        // absolute bus-cycle deadlines. See `PendingEvent::rebased`.
        for ev in self.pending.iter_mut() {
            if !ev.rebased {
                ev.deadline = cycles_now.saturating_add(ev.deadline);
                ev.rebased = true;
            }
        }

        while let Some(front) = self.pending.front() {
            if front.deadline > cycles_now {
                break;
            }
            let ev = self.pending.pop_front().unwrap();

            // If the drive was paused/reset between this event's
            // scheduling and now, drop it silently. On real
            // hardware, Pause/Init kills in-flight sector reads;
            // letting a stale DataReady fire would clobber the
            // data FIFO that software was about to drain, or
            // deliver the wrong LBA entirely (we were burning
            // hours on exactly this, LBA 17's volume-descriptor
            // terminator landing in the slot where the BIOS
            // expected its PVD).
            if ev.irq == IrqType::DataReady && !self.reading {
                continue;
            }

            for b in ev.bytes.iter().copied() {
                if self.responses.len() < RESPONSE_FIFO_DEPTH {
                    self.responses.push_back(b);
                }
            }
            // DataReady events drive the sector-stream — load the next
            // sector's payload into the data FIFO as the event fires,
            // and chain the subsequent DataReady while we're still
            // reading.
            if ev.irq == IrqType::DataReady {
                self.load_next_sector();
                if self.reading {
                    self.schedule_sector_event();
                    // The event we just pushed has a *relative*
                    // deadline; promote it to absolute here since we
                    // already know `cycles_now`. (The per-event
                    // `rebased` flag would handle this on the next
                    // tick, but doing it inline avoids the DataReady
                    // chain firing twice in a single tick when
                    // SECTOR_READ_CYCLES is small.)
                    let delay = self.sector_read_cycles();
                    if let Some(last) = self.pending.back_mut() {
                        last.deadline = cycles_now.saturating_add(delay);
                        last.rebased = true;
                    }
                }
            }
            // Only fire IRQ if none is currently pending — hardware
            // holds the latched IRQ until software acks.
            if self.irq_flag == 0 {
                self.irq_flag = ev.irq as u8;
                raised = true;
            } else {
                // Re-enqueue at the front with a tiny delay; the BIOS
                // will ack eventually. The deadline we set here is
                // already absolute, so mark rebased.
                let mut re = ev;
                re.deadline = cycles_now + FIRST_RESPONSE_CYCLES / 10;
                re.rebased = true;
                self.pending.push_front(re);
                break;
            }
        }

        raised
    }

    /// Total commands received — used by `smoke_draw` to confirm BIOS
    /// is talking to the drive.
    pub fn commands_dispatched(&self) -> u64 {
        self.commands_dispatched
    }

    /// Per-command histogram (indexed by command byte) — same purpose.
    pub fn command_histogram(&self) -> &[u32; 32] {
        &self.command_hist
    }

    /// Current raw IRQ-flag (for diagnostics).
    pub fn irq_flag(&self) -> u8 {
        self.irq_flag
    }

    /// Current IRQ-enable mask (for diagnostics).
    pub fn irq_mask_raw(&self) -> u8 {
        self.irq_mask
    }

    /// The most-recently-dispatched command byte (for diagnostics).
    pub fn last_command(&self) -> u8 {
        self.last_command
    }

    /// Most-recent `SetLoc` MSF target, as a `(minute, second,
    /// frame)` BCD triple. Diagnostic-only — lets probes correlate
    /// `ReadN` events with the LBA the BIOS is asking for.
    pub fn debug_setloc_msf(&self) -> (u8, u8, u8) {
        self.setloc_msf
    }

    /// Total bytes popped from the data FIFO via MMIO reads since
    /// boot. Diagnostic.
    pub fn data_fifo_pops(&self) -> u64 {
        self.data_fifo_pops
    }

    /// Pull one byte from the data FIFO — used by DMA channel 3's
    /// block-read path to drain a sector into RAM. Returns `0` when
    /// the FIFO is empty (hardware returns stale-bus bytes; `0` is
    /// a safe stand-in).
    pub fn pop_data_byte(&mut self) -> u8 {
        self.data_fifo_pops = self.data_fifo_pops.saturating_add(1);
        self.data_fifo.pop_front().unwrap_or(0)
    }

    /// Number of bytes currently buffered in the data FIFO.
    pub fn data_fifo_len(&self) -> usize {
        self.data_fifo.len()
    }
}

impl Default for CdRom {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode one raw 2352-byte Mode 2 Form 2 XA ADPCM audio sector into
/// stereo PCM samples. Returns `None` when the sector isn't an XA
/// audio sector (subheader submode bit 2 / Form 2 bit 5 not set).
///
/// Sector layout (Mode 2):
/// - 0..=11   : sync pattern (0x00, 12× 0xFF, 0x00)
/// - 12..=14  : MSF header
/// - 15       : mode (02 for Mode 2)
/// - 16..=23  : 8-byte subheader (4 bytes × 2 copies)
/// - 24..=2347: 2324-byte user data (XA audio payload for Form 2)
/// - 2348+    : EDC (unused for Form 2)
///
/// Subheader byte 2 (submode): bit 2 = audio, bit 5 = Form 2.
/// Subheader byte 3 (coding info): bits 0-1 mono/stereo, bits 2-3
/// sample rate, bits 4-5 bits/sample.
///
/// For the common case (4-bit stereo, 37800 Hz) we decode the 18
/// sound groups of 128 bytes each, each producing 112 stereo
/// samples. Other forms return `None`.
///
/// Samples are nearest-neighbour resampled from the source rate up
/// to the SPU's 44.1 kHz rate on output.
fn decode_xa_audio_sector(
    raw: &[u8],
    left: &mut crate::spu::XaDecoderState,
    right: &mut crate::spu::XaDecoderState,
) -> Option<Vec<(i16, i16)>> {
    if raw.len() < 2352 {
        return None;
    }
    let submode = raw[18];
    let coding = raw[19];
    // Must be audio + Form 2.
    if submode & 0x04 == 0 || submode & 0x20 == 0 {
        return None;
    }
    // Only support 4-bit stereo 37800 Hz for now — the format PS1
    // FMV XA streams overwhelmingly use. Other configurations
    // (8-bit, mono, 18900 Hz) fall through to silence rather than
    // glitching with wrong-shape decodes.
    let stereo = (coding & 0x03) == 1;
    let hi_rate = (coding & 0x0C) == 0; // 00 → 37800, 01 → 18900
    let bits_8 = (coding & 0x30) != 0;
    if !stereo || !hi_rate || bits_8 {
        return None;
    }

    // XA payload starts at offset 24 (after 12+4+8 bytes of header).
    // 18 sound groups × 128 bytes.
    let payload = &raw[24..24 + 18 * 128];
    let mut decoded: Vec<(i16, i16)> = Vec::with_capacity(18 * 112);
    let head_table = [0usize, 2, 8, 10];
    for group_idx in 0..18 {
        let group = &payload[group_idx * 128..group_idx * 128 + 128];
        let headers = &group[0..16];
        let data = &group[16..128];

        // 4-bit stereo has 4 blocks per group: 2 interleaved stereo
        // pairs. Each pair shares 2 filter-range bytes from the
        // header (positions 0/1, 8/9 in the 16-byte header).
        for blk in 0..4 {
            let pair = blk / 2; // 0 or 1
            let channel = blk & 1; // 0 = L, 1 = R
            let filter_range = headers[head_table[pair * 2 + channel]];
            // Each block's 28 samples come from 14 bytes, nibble-
            // column-arranged across the 112-byte block. For 4-bit
            // stereo the columns interleave two blocks; we select
            // our 14 input words by walking `data[blk + k*4]` for
            // k in 0..28 — but XA packs 4 blocks per group at
            // byte stride 4. We build 14 16-bit words for the
            // decoder.
            let mut words = [0u16; 14];
            for k in 0..7 {
                // Each word is 4 nibbles: low nibble from byte
                // `data[pair*... + k*16 + blk]`, etc.
                // Mirroring Redux's level-B/C loop in decode_xa.cc:
                // nibble low from `sound_datap2[0]`, then +4, +8, +12
                // at stride 16.
                let base = k * 16;
                let b0 = data[base + blk] as u16;
                let b1 = data[base + 4 + blk] as u16;
                let b2 = data[base + 8 + blk] as u16;
                let b3 = data[base + 12 + blk] as u16;
                let (lo, hi) = if blk < 2 {
                    // First two blocks — low nibble of each byte.
                    (
                        (b0 & 0x0F) | ((b1 & 0x0F) << 4),
                        (b2 & 0x0F) | ((b3 & 0x0F) << 4),
                    )
                } else {
                    // Last two blocks — high nibble.
                    ((b0 >> 4) | ((b1 >> 4) << 4), (b2 >> 4) | ((b3 >> 4) << 4))
                };
                words[k * 2] = lo;
                words[k * 2 + 1] = hi;
            }
            let mut samples = [0i16; 28];
            let state = if channel == 0 { &mut *left } else { &mut *right };
            crate::spu::xa_decode_block(state, filter_range, &words, &mut samples, 1);

            // Interleave into `decoded`: block's 28 samples go
            // into 28 consecutive output slots at this pair's base.
            let base = pair * 28;
            for (i, &s) in samples.iter().enumerate() {
                while decoded.len() <= base + i {
                    decoded.push((0, 0));
                }
                let (l, r) = &mut decoded[base + i];
                if channel == 0 {
                    *l = s;
                } else {
                    *r = s;
                }
            }
        }
    }

    // Upsample 37800 → 44100 Hz by nearest-neighbour (ratio ≈
    // 1.167). Acceptable quality for a first pass; linear /
    // gaussian can land later.
    let mut resampled: Vec<(i16, i16)> = Vec::with_capacity(decoded.len() * 44100 / 37800 + 1);
    let src_n = decoded.len() as u32;
    let dst_n = (src_n as u64 * 44100 / 37800) as u32;
    for i in 0..dst_n {
        let src_idx = ((i as u64 * src_n as u64) / dst_n as u64) as usize;
        resampled.push(decoded[src_idx.min(decoded.len() - 1)]);
    }
    Some(resampled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_covers_4_bytes() {
        for off in 0..4 {
            assert!(CdRom::contains(BASE + off));
        }
        assert!(!CdRom::contains(BASE - 1));
        assert!(!CdRom::contains(BASE + 4));
    }

    #[test]
    fn index_write_is_masked() {
        let mut cd = CdRom::new();
        cd.write8(BASE, 0xFF);
        // The status byte readback has index in low 2 bits.
        assert_eq!(cd.read8(BASE) & 0x3, 3);
    }

    #[test]
    fn parameter_fifo_roundtrips() {
        let mut cd = CdRom::new();
        cd.write8(BASE + 2, 0xAB);
        cd.write8(BASE + 2, 0xCD);
        assert_eq!(cd.params.len(), 2);
        // FIFO not-empty bit cleared, not-full bit still set.
        let s = cd.read8(BASE);
        assert_eq!(s & status_bit::PARAM_FIFO_EMPTY, 0);
        assert!(s & status_bit::PARAM_FIFO_NOT_FULL != 0);
    }

    #[test]
    fn response_fifo_pop_returns_pushed_bytes() {
        let mut cd = CdRom::new();
        cd.responses.push_back(0x11);
        cd.responses.push_back(0x22);
        assert_eq!(cd.read8(BASE + 1), 0x11);
        assert_eq!(cd.read8(BASE + 1), 0x22);
        // Empty pop reads as zero — matches Redux.
        assert_eq!(cd.read8(BASE + 1), 0);
    }

    #[test]
    fn irq_ack_clears_only_written_bits() {
        let mut cd = CdRom::new();
        cd.irq_flag = 0x1F;
        // Select index 1 then write-1-to-clear bit 2.
        cd.write8(BASE, 1);
        cd.write8(BASE + 3, 0x04);
        assert_eq!(cd.irq_flag, 0x1B);
    }

    #[test]
    fn cold_boot_is_closed_lid_no_disc() {
        // Cold boot state for "Please insert disc" path: lid closed
        // (SHELL_OPEN cleared) so Init succeeds, motor off until
        // Init runs. No disc (GetID returns the no-disc response).
        let cd = CdRom::new();
        assert_eq!(cd.drive_status & drive_status_bit::SHELL_OPEN, 0);
        assert!(!cd.motor_on);
        assert!(!cd.disc_present);
    }

    #[test]
    fn sector_read_cycles_tracks_mode_bit_7() {
        let mut cd = CdRom::new();
        // Default mode = 0x80 → double-speed.
        assert_eq!(cd.sector_read_cycles(), CD_READ_TIME);
        // Flipping bit 7 off via SetMode gives single-speed (2×).
        cd.cmd_setmode(&[0x00]);
        assert_eq!(cd.sector_read_cycles(), CD_READ_TIME * 2);
        // Setting other bits without bit 7 stays single-speed.
        cd.cmd_setmode(&[0x60]);
        assert_eq!(cd.sector_read_cycles(), CD_READ_TIME * 2);
        // Back to double-speed when bit 7 returns.
        cd.cmd_setmode(&[0x80]);
        assert_eq!(cd.sector_read_cycles(), CD_READ_TIME);
    }
}
