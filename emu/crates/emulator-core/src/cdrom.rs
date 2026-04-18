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

/// Canonical cycle delays for command responses, matching the
/// ballpark Redux uses. Real hardware values depend on motor speed,
/// seek distance, head position, etc.; these are fine for "emulates
/// enough to boot" use.
const FIRST_RESPONSE_CYCLES: u64 = 50_000;
const INIT_SECOND_RESPONSE_CYCLES: u64 = 900_000;
const GETID_SECOND_RESPONSE_CYCLES: u64 = 33_000;
const SEEK_SECOND_RESPONSE_CYCLES: u64 = 500_000;
/// Cycles between sector reads at 2× drive speed (BIOS default).
/// Real hardware is ~33_868_800 / 150 sectors/s = 225k cycles;
/// Redux uses a closer approximation.
const SECTOR_READ_CYCLES: u64 = 225_000;

/// A deferred response: when `bus.cycles` passes `deadline`, the
/// event's bytes land in the response FIFO and its IRQ type fires.
#[derive(Debug, Clone)]
struct PendingEvent {
    deadline: u64,
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
            disc: None,
            data_fifo: VecDeque::new(),
            reading: false,
            read_lba: 0,
        }
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
            (2, _) => self.data_fifo.pop_front().unwrap_or(0),
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
            // Play (CD-DA): 1 optional byte (track). Treat as no-disc
            // error if no disc; otherwise we'd need audio plumbing.
            0x03 => self.cmd_simple_stat_or_nodisc(),
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
    fn schedule_first_response(&mut self, bytes: Vec<u8>) {
        self.pending.push_back(PendingEvent {
            deadline: 0, // filled in by `tick_at` when we know `now`
            irq: IrqType::Acknowledge,
            bytes,
        });
    }

    fn schedule_second_response(&mut self, bytes: Vec<u8>, delay: u64) {
        // We use deadline=1 as a marker for "relative delay"; `tick_at`
        // translates both 0 and non-zero markers correctly.
        let _ = delay;
        self.pending.push_back(PendingEvent {
            deadline: delay,
            irq: IrqType::Complete,
            bytes,
        });
    }

    /// Like [`schedule_second_response`] but the IRQ type is Error
    /// (INT5). Used for the second reply of commands that fail because
    /// the disc isn't present — the first reply is still an ack (INT3)
    /// so the BIOS can distinguish "command received" from "command
    /// completed with error".
    fn schedule_second_error(&mut self, bytes: Vec<u8>, delay: u64) {
        self.pending.push_back(PendingEvent {
            deadline: delay,
            irq: IrqType::Error,
            bytes,
        });
    }

    fn schedule_error_response(&mut self, bytes: Vec<u8>) {
        self.pending.push_back(PendingEvent {
            deadline: 0,
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

    fn cmd_setmode(&mut self, _params: &[u8]) {
        let stat = self.stat_byte();
        self.schedule_first_response(vec![stat]);
    }

    fn cmd_stop(&mut self) {
        self.motor_on = false;
        self.schedule_first_response(vec![self.stat_byte()]);
        let stat = self.stat_byte();
        self.schedule_second_response(vec![stat], SEEK_SECOND_RESPONSE_CYCLES);
    }

    fn cmd_pause(&mut self) {
        self.schedule_first_response(vec![self.stat_byte()]);
        let stat = self.stat_byte();
        self.schedule_second_response(vec![stat], SEEK_SECOND_RESPONSE_CYCLES);
    }

    fn cmd_init(&mut self) {
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
            deadline: SECTOR_READ_CYCLES,
            irq: IrqType::DataReady,
            bytes: vec![stat],
        });
    }

    /// On DataReady event firing: populate the data FIFO with the
    /// next sector's user data and bump the read LBA. Called from
    /// `tick` once per sector event.
    fn load_next_sector(&mut self) {
        let lba = self.read_lba;
        self.read_lba = self.read_lba.wrapping_add(1);
        if let Some(disc) = self.disc.as_ref() {
            if let Some(user) = disc.read_sector_user(lba) {
                self.data_fifo.clear();
                self.data_fifo.extend(user.iter().copied());
                return;
            }
        }
        // Past end of disc — stop the read and leave the FIFO empty.
        self.reading = false;
    }

    fn cmd_simple_stat_or_nodisc(&mut self) {
        if self.disc_present {
            self.schedule_first_response(vec![self.stat_byte()]);
        } else {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
        }
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
        // Fix up pending events with absolute deadlines. We used 0 and
        // small relative numbers as markers; translate them now.
        for ev in self.pending.iter_mut() {
            if ev.deadline < cycles_now {
                // Interpret 0 as "immediate = +FIRST_RESPONSE_CYCLES";
                // anything else non-zero as the relative offset itself.
                let offset = if ev.deadline == 0 {
                    FIRST_RESPONSE_CYCLES
                } else {
                    ev.deadline
                };
                ev.deadline = cycles_now.saturating_add(offset);
            }
        }

        while let Some(front) = self.pending.front() {
            if front.deadline > cycles_now {
                break;
            }
            let ev = self.pending.pop_front().unwrap();
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
                    // Convert the relative delay we just used into an
                    // absolute deadline against `cycles_now`.
                    if let Some(last) = self.pending.back_mut() {
                        last.deadline = cycles_now.saturating_add(SECTOR_READ_CYCLES);
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
                // will ack eventually.
                let mut re = ev;
                re.deadline = cycles_now + FIRST_RESPONSE_CYCLES / 10;
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

    /// Pull one byte from the data FIFO — used by DMA channel 3's
    /// block-read path to drain a sector into RAM. Returns `0` when
    /// the FIFO is empty (hardware returns stale-bus bytes; `0` is
    /// a safe stand-in).
    pub fn pop_data_byte(&mut self) -> u8 {
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
}
