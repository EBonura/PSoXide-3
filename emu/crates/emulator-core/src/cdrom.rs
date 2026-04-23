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

use psx_iso::{bcd_to_bin, msf_to_lba, Disc};

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
/// - `AddIrqQueue(CdlID + 0x100, 20480)` — GetID second response,
///   ~4.4 µs, observed across boot roms (L900). `CdlInit` uses the
///   separate lid/rescan path instead of a second CDROM IRQ.
/// - `cdReadTime = psxClockSpeed / 75` — one PSX CD-frame period
///   (L135). Redux schedules the first ReadN/ReadS sector at
///   `cdReadTime` in double-speed mode, then chains steady-state
///   sectors at `cdReadTime / 2` (single-speed uses 2x those delays).
/// - `scheduleCDPlayIRQ(SEEK_DONE ? 0x800 : cdReadTime * 4)` —
///   SeekL / SeekP second response (L875). If the target is already
///   seeked, quick ack; otherwise a full seek-time equivalent.
const FIRST_RESPONSE_CYCLES: u64 = 0x800; // 2048
const GETID_SECOND_RESPONSE_CYCLES: u64 = 20_480;
const SEEK_SECOND_RESPONSE_CYCLES: u64 = CD_READ_TIME * 4; // ≈ 1,806,336
const PAUSE_COMPLETE_CYCLES_STANDBY: u64 = 7_000;
const PAUSE_COMPLETE_CYCLES_ACTIVE: u64 = 1_000_000;
const LID_BOOTSTRAP_CYCLES: u64 = 20_480;
const LID_PREPARE_SPINUP_CYCLES: u64 = CD_READ_TIME * 150;
const LID_PREPARE_SEEK_CYCLES: u64 = CD_READ_TIME * 26;
/// PSX system clock / CD frames per second. `33_868_800 / 75`.
/// Redux's `cdReadTime`.
const CD_READ_TIME: u64 = 451_584;
// Sector-read cycles are derived per-instance from the current mode
// byte. The first sector after ReadN/ReadS is slower than the
// steady-state stream; see `initial_sector_read_cycles` and
// `sector_read_cycles`.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct XaCoding {
    stereo: bool,
    freq: u32,
    nbits: u8,
}

/// A chained response scheduled when this event fires. Redux enqueues
/// a command's long-running completion from inside the first-response
/// interrupt handler, so the second deadline is relative to the actual
/// first IRQ service cycle rather than the original command write.
#[derive(Debug, Clone)]
struct PendingFollowup {
    delay: u64,
    irq: IrqType,
    bytes: Vec<u8>,
}

/// A deferred response: when `bus.cycles` passes `deadline` (an
/// absolute bus-cycle count), the event's bytes land in the response
/// FIFO and its IRQ type fires.
#[derive(Debug, Clone)]
struct PendingEvent {
    deadline: u64,
    irq: IrqType,
    bytes: Vec<u8>,
    followup: Option<PendingFollowup>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DriveState {
    Stopped,
    Standby,
    LidOpen,
    RescanCd,
    PrepareCd,
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
    /// Command/parameter transmission busy latch (status bit 7).
    /// Redux sets this when a command byte is written, then clears it
    /// when the corresponding interrupt packet is materialized.
    command_busy: bool,
    /// IRQ enable mask — low 3 bits; an IRQ fires only if its type
    /// bit is enabled. BIOS writes via `0x1F80_1802 idx=1`.
    irq_mask: u8,
    /// Currently-raised IRQ type; cleared by writing 1s to the low
    /// 3 bits of `0x1F80_1803 idx=1`.
    irq_flag: u8,
    /// Scheduled events waiting for their cycle deadlines. Processed
    /// in order by [`CdRom::tick`].
    pending: VecDeque<PendingEvent>,
    /// Internal lid / rescan timer. Redux drives this via a separate
    /// interrupt slot (`PSXINT_CDRLID`) rather than a visible CDROM
    /// IRQ packet.
    lid_deadline: Option<u64>,
    /// `CdlInit` starts the lid/rescan path when its ACK is serviced,
    /// not at command-write time.
    lid_bootstrap_pending: bool,
    drive_state: DriveState,
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
    /// Whether a new SetLoc target is waiting to be applied to the
    /// live read/play head. Redux doesn't move immediately on SetLoc;
    /// it latches the target and consumes it on Seek/Read/Play.
    setloc_pending: bool,
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
    /// Bus cycle at which the currently-dispatching command was
    /// received. `queue_command` stashes the caller-supplied `now`
    /// here so `schedule_*_response` helpers can compute absolute
    /// deadlines without threading `now` through every command
    /// handler.
    scheduling_cycle: u64,
    /// Per-IrqType raise histogram — indexed by `IrqType`
    /// discriminant (0..=5). Probes read this to tell
    /// Acknowledge/DataReady/Complete/Error counts apart when the
    /// aggregate CDROM raise count looks suspicious.
    pub irq_type_counts: [u64; 6],
    /// Per-raise log of `(cycle_when_raised, irq_type_discriminant)`
    /// tuples. Populated only when `cdrom_irq_log_cap > 0`, capped
    /// at that length to keep memory bounded in long runs. Probes
    /// compare this sequence against Redux's silent-run CDROM-IRQ
    /// log to pinpoint which specific IRQ fires at a divergent
    /// cycle.
    pub cdrom_irq_log: Vec<(u64, u8)>,
    /// Max length of `cdrom_irq_log` — 0 disables logging (the
    /// default, to avoid the per-raise allocation in production
    /// runs). Probes set this via [`CdRom::enable_irq_log`].
    cdrom_irq_log_cap: usize,
    /// Count of `schedule_sector_event_at` calls. Diagnostic only —
    /// the BIOS should see one DataReady per sector read at the
    /// current CD speed's cadence, so this should match the sector
    /// count the game requested. A blown-out number means we're
    /// chaining extra events somewhere.
    pub sector_events_scheduled: u64,
    /// Loaded disc image, if any. When `Some`, `disc_present` is also
    /// true and GetID / ReadN follow the disc-present paths; when
    /// `None`, they fall back to the "please insert disc" path.
    disc: Option<Disc>,
    /// Data FIFO — 2048 bytes of sector user data, drained by MMIO
    /// reads at `0x1F80_1802` or by DMA channel 3. Filled by each
    /// DataReady event during an active ReadN / ReadS.
    data_fifo: VecDeque<u8>,
    /// Redux's DRQSTS/data-ready latch (status bit 6). A fresh sector
    /// sets this even before software has armed a transfer via the
    /// request register; stray reads with no transfer armed clear it
    /// back down without consuming the buffered bytes.
    data_fifo_ready: bool,
    /// Set by request-register bit 7 (`0x1F80_1803` index 0). MMIO
    /// reads and DMA may only drain the current sector while this is
    /// armed; otherwise the controller returns zeroes even if the FIFO
    /// already holds data.
    data_transfer_active: bool,
    /// Last read sector header (MM, SS, FF, mode) — returned by
    /// `GetlocL` after the drive has actually delivered a sector.
    last_sector_header: [u8; 4],
    /// Last read sector subheader (file, channel, submode, coding) —
    /// likewise returned by `GetlocL`.
    last_sector_subheader: [u8; 4],
    /// Whether `last_sector_header` / `last_sector_subheader`
    /// currently hold real sector data.
    last_sector_header_valid: bool,
    /// Set while a read is in progress; controls whether new
    /// DataReady events chain into further sectors.
    reading: bool,
    /// Next sector LBA to deliver during an active read.
    read_lba: u32,
    /// Redux tracks whether the drive has already completed a seek and
    /// uses that to pick the short 0x800-cycle SeekL/SeekP follow-up
    /// path on subsequent seeks.
    seek_done: bool,
    /// Redux inserts a long delay before the second sector after a
    /// relocated read starts (`m_locationChanged`). Without it we
    /// stream multiple sectors where the hardware only delivered one.
    location_changed: bool,
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
    /// CD-XA mute latch (`Mute` / `Demute` commands).
    muted: bool,
    /// XA filter state written by `Setfilter` and reported back by
    /// `Getparam`. XA-streaming games use this to confirm the drive
    /// latched their file/channel filter before they start reads.
    xa_filter_file: u8,
    xa_filter_channel: u8,
    /// XA stream state: `1` = next matching audio sector is the
    /// first one in a stream, `0` = continuing stream, `-1` =
    /// decode disabled until the next `Read*`.
    xa_first_sector: i8,
    /// Parsed XA coding for the active stream. Redux parses this on
    /// the first sector, resets decoder history if it changes, then
    /// reuses it for successive sectors instead of trusting every
    /// sector's coding byte.
    xa_coding: Option<XaCoding>,
    /// Live CD-XA volume matrix, applied before samples reach the SPU.
    attenuator_left_to_left: u8,
    attenuator_left_to_right: u8,
    attenuator_right_to_left: u8,
    attenuator_right_to_right: u8,
    /// Shadow volume matrix registers, committed when software writes
    /// bit 5 on `0x1F80_1803` with index 3.
    attenuator_left_to_left_t: u8,
    attenuator_left_to_right_t: u8,
    attenuator_right_to_left_t: u8,
    attenuator_right_to_right_t: u8,
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
            command_busy: false,
            // Redux initializes m_reg2 (the CDROM IRQ mask) to 0x1F on
            // reset (cdrom.cc:1562) so all five IRQ types are enabled
            // out of the gate. We used to start at 0, which blocked every
            // CDROM IRQ from reaching the CPU at boot. That was masked in
            // earlier runs because we also ignored the mask when raising
            // (see `should_wake_cpu`); after adding the mask gate, our 0
            // initial value caused CDROM IRQs to silently latch without
            // waking CPU — BIOS boot then polled the latched flag instead
            // of getting its usual ISR-driven ack, drifting from Redux.
            irq_mask: 0x1F,
            irq_flag: 0,
            pending: VecDeque::new(),
            lid_deadline: None,
            lid_bootstrap_pending: false,
            drive_state: DriveState::Stopped,
            motor_on: false,
            disc_present: false,
            setloc_msf: (0, 0, 0),
            setloc_pending: false,
            commands_dispatched: 0,
            command_hist: [0; 32],
            last_command: 0,
            data_fifo_pops: 0,
            scheduling_cycle: 0,
            irq_type_counts: [0; 6],
            cdrom_irq_log: Vec::new(),
            cdrom_irq_log_cap: 0,
            sector_events_scheduled: 0,
            disc: None,
            data_fifo: VecDeque::new(),
            data_fifo_ready: false,
            data_transfer_active: false,
            last_sector_header: [0; 4],
            last_sector_subheader: [0; 4],
            last_sector_header_valid: false,
            reading: false,
            read_lba: 0,
            seek_done: false,
            location_changed: false,
            // Power-on mode: double-speed, no XA, data-only 2048-byte
            // sectors. Matches the BIOS's probe-time expectation — it
            // issues SetMode 0x80 (double-speed) before its first
            // ReadN. A fresh emulator reset without an intervening
            // SetMode still uses double-speed, matching the prior
            // behaviour of always-CD_READ_TIME pacing.
            mode: 0x80,
            muted: false,
            xa_filter_file: 1,
            xa_filter_channel: 1,
            xa_first_sector: 0,
            xa_coding: None,
            attenuator_left_to_left: 0x80,
            attenuator_left_to_right: 0x00,
            attenuator_right_to_left: 0x00,
            attenuator_right_to_right: 0x80,
            attenuator_left_to_left_t: 0x00,
            attenuator_left_to_right_t: 0x00,
            attenuator_right_to_left_t: 0x00,
            attenuator_right_to_right_t: 0x00,
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

    fn commit_attenuator(&mut self) {
        self.attenuator_left_to_left = self.attenuator_left_to_left_t;
        self.attenuator_left_to_right = self.attenuator_left_to_right_t;
        self.attenuator_right_to_left = self.attenuator_right_to_left_t;
        self.attenuator_right_to_right = self.attenuator_right_to_right_t;
    }

    fn reset_xa_stream(&mut self) {
        self.xa_first_sector = 0;
        self.xa_coding = None;
        self.xa_left.reset();
        self.xa_right.reset();
        self.cd_audio.clear();
    }

    fn clear_data_fifo(&mut self) {
        self.data_fifo.clear();
        self.data_fifo_ready = false;
        self.data_transfer_active = false;
    }

    fn pop_data_fifo_byte(&mut self) -> u8 {
        if !self.data_transfer_active {
            self.data_fifo_ready = false;
            return 0;
        }

        let byte = self.data_fifo.pop_front().unwrap_or(0);
        if self.data_fifo.is_empty() {
            self.data_fifo_ready = false;
            self.data_transfer_active = false;
        }
        byte
    }

    fn attenuate_xa_samples(&self, samples: &mut [(i16, i16)]) {
        let ll = self.attenuator_left_to_left as i32;
        let lr = self.attenuator_left_to_right as i32;
        let rl = self.attenuator_right_to_left as i32;
        let rr = self.attenuator_right_to_right as i32;

        if lr == 0 && rl == 0 && (0x78..=0x88).contains(&ll) && (0x78..=0x88).contains(&rr) {
            return;
        }

        for (l, r) in samples.iter_mut() {
            let mixed_l = ((*l as i32) * ll + (*r as i32) * rl) >> 7;
            let mixed_r = ((*r as i32) * rr + mixed_l * lr) >> 7;
            *l = mixed_l.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            *r = mixed_r.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
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
        self.drive_state = if self.disc_present {
            DriveState::Standby
        } else {
            DriveState::Stopped
        };
        self.lid_deadline = None;
        self.lid_bootstrap_pending = false;
        self.reading = false;
        self.drive_status &= !drive_status_bit::PLAYING;
        self.clear_data_fifo();
        self.cd_audio.clear();
        self.last_sector_header = [0; 4];
        self.last_sector_subheader = [0; 4];
        self.last_sector_header_valid = false;
        self.xa_first_sector = 0;
        self.xa_left.reset();
        self.xa_right.reset();
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
                self.pop_data_fifo_byte()
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
        // Back-compat shim: tests that aren't cycle-aware call this
        // variant. For parity-correct scheduling, the bus uses
        // `write8_at` instead so command-port writes carry the
        // current bus cycle through to the CDROM scheduler.
        self.write8_at(phys, value, 0);
    }

    /// Like [`write8`], but threads the bus cycle through so
    /// `queue_command` can schedule first/second responses with
    /// absolute deadlines anchored on issue time. Matches Redux's
    /// `AddIrqQueue(cmd, delay)` which anchors on `m_regs.cycle` at
    /// the cmd-port write. The previous "relative then rebase on
    /// next tick" scheme lost the BIAS + memory-access cycles of
    /// the SB that issued the command — surfaced as a 5-cycle late
    /// IRQ dispatch at parity step 89,198,894.
    pub fn write8_at(&mut self, phys: u32, value: u8, now: u64) {
        let offset = (phys - BASE) as u8;
        match (offset, self.index) {
            // 0x1F80_1800 write — set the index.
            (0, _) => self.index = value & status_bit::INDEX_MASK,
            // 0x1F80_1801 idx=0 — command register. Queue for 6b.
            (1, 0) => self.queue_command(value, now),
            // 0x1F80_1801 idx=1/2/3 — audio sound-map / CD-to-SPU
            // volume.
            (1, 1 | 2) => {}
            (1, 3) => self.attenuator_right_to_right_t = value,
            // 0x1F80_1802 idx=0 — parameter FIFO push.
            (2, 0) => self.push_param(value),
            // 0x1F80_1802 idx=1 — interrupt enable.
            (2, 1) => self.irq_mask = value & 0x1F,
            // 0x1F80_1802 idx=2/3 — audio volume.
            (2, 2) => self.attenuator_left_to_left_t = value,
            (2, 3) => self.attenuator_right_to_left_t = value,
            // 0x1F80_1803 idx=0 — request register (data transfer on,
            // command-buffer reset, etc.). Bit 7 = want-data. Full
            // modelling arrives with sector reads.
            (3, 0) => {
                // Bit 6 = BFRD (reset). If set, clear parameter FIFO.
                if value & 0x40 != 0 {
                    self.params.clear();
                }
                // Bit 7 arms the sector-transfer buffer. Redux gates
                // both MMIO reads and DMA behind this request latch
                // instead of exposing any queued sector bytes
                // immediately when DataReady fires.
                if value & 0x80 != 0 && !self.data_transfer_active {
                    self.data_transfer_active = true;
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
            (3, 2) => self.attenuator_left_to_right_t = value,
            (3, 3) => {
                if value & 0x20 != 0 {
                    self.commit_attenuator();
                }
            }
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
        if self.data_fifo_ready {
            s |= status_bit::DATA_FIFO_NOT_EMPTY;
        }
        if self.command_busy {
            s |= status_bit::TRANSMISSION_BUSY;
        }
        // ADPCM busy (bit 2) comes from a subsystem we don't expose
        // yet; keep it clear for now.
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
    fn queue_command(&mut self, command: u8, now: u64) {
        self.commands_dispatched += 1;
        self.last_command = command;
        if (command as usize) < self.command_hist.len() {
            self.command_hist[command as usize] += 1;
        }
        // Real hardware exposes only one live response packet. A new
        // command drops any unread bytes from the prior packet instead
        // of appending another logical response stream behind them.
        self.responses.clear();
        self.command_busy = true;
        // Stash the issue-time cycle so each `schedule_*_response`
        // call below resolves its absolute deadline against the
        // right anchor, without threading `now` through every
        // command handler's signature.
        self.scheduling_cycle = now;

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
            // MotorOn — some retail loaders wake the spindle
            // explicitly before querying/reading.
            0x07 => self.cmd_motor_on(),
            // Stop: halt motor.
            0x08 => self.cmd_stop(),
            // Pause: halt reads but keep motor on.
            0x09 => self.cmd_pause(),
            // Init: reset drive + spin motor + clear mode.
            0x0A => self.cmd_init(),
            // Mute / Demute.
            0x0B => self.cmd_mute(true),
            0x0C => self.cmd_mute(false),
            // Setfilter — XA file/channel filter.
            0x0D => self.cmd_set_filter(&params),
            // Getparam — mode/filter query.
            0x0F => self.cmd_get_param(),
            // GetlocL — current logical MSF + mode + subheader.
            0x10 => self.cmd_get_loc_l(),
            // GetlocP — current play position (track, index, MSF).
            0x11 => self.cmd_get_loc_p(),
            // SetSession — current loader only models session 1, but
            // retail software still expects the completion IRQ.
            0x12 => self.cmd_set_session(&params),
            // GetTN — first/last track numbers.
            0x13 => self.cmd_get_tn(),
            // GetTD — start time of a track, or lead-out for track 0.
            0x14 => self.cmd_get_td(&params),
            // ReadS — read without auto-retry. Data arrives the same
            // way as ReadN for our purposes; games use ReadS for
            // audio/video streaming where a retry would cause hitching.
            0x1B => self.cmd_read(),
            // SetMode: 1 byte (speed, CD-DA enable, filter, etc.).
            // We accept and store; full behaviour in 6d.
            0x0E => self.cmd_setmode(&params),
            // SeekL: seek to last-SetLoc position (logical sectors).
            0x15 | 0x16 => self.cmd_seek(),
            // Test: sub-op in param[0]. Most common: 0x20 = get
            // drive version / BIOS date (6-byte response).
            0x19 => self.cmd_test(&params),
            // GetID: "what kind of disc is this?"
            0x1A => self.cmd_getid(),
            // ReadTOC: re-read the table of contents. The BIOS
            // issues this during disc-boot to learn the track
            // layout; without a response the BIOS hangs waiting
            // for INT2 (Complete), stranding MGS / Crash at the
            // Sony splash.
            0x1E => self.cmd_read_toc(),
            // Reset — abort in-flight drive activity.
            0x1C => self.cmd_reset(),
            _ => self.cmd_getstat(),
        }
    }

    // --- Command handlers ---

    /// Schedule a first-response IRQ at
    /// `scheduling_cycle + FIRST_RESPONSE_CYCLES` (absolute). Matches
    /// Redux's `AddIrqQueue` which anchors on `m_regs.cycle` at the
    /// moment of the cmd-port write.
    fn schedule_first_response(&mut self, bytes: Vec<u8>) {
        self.insert_pending_event(PendingEvent {
            deadline: self.scheduling_cycle.saturating_add(FIRST_RESPONSE_CYCLES),
            irq: IrqType::Acknowledge,
            bytes,
            followup: None,
        });
    }

    /// Schedule a second-response IRQ. `additional_delay` is time
    /// *after* the first response interrupt actually fires. Matches
    /// Redux's `AddIrqQueue(cmd + 0x100, delay)` path inside the
    /// first-response handler.
    fn schedule_second_response(&mut self, bytes: Vec<u8>, additional_delay: u64) {
        self.chain_followup(IrqType::Complete, bytes, additional_delay);
    }

    /// Like [`schedule_second_response`] but the IRQ type is Error
    /// (INT5). Used for the second reply of commands that fail because
    /// the disc isn't present.
    fn schedule_second_error(&mut self, bytes: Vec<u8>, additional_delay: u64) {
        self.chain_followup(IrqType::Error, bytes, additional_delay);
    }

    fn schedule_error_response(&mut self, bytes: Vec<u8>) {
        self.insert_pending_event(PendingEvent {
            deadline: self.scheduling_cycle.saturating_add(FIRST_RESPONSE_CYCLES),
            irq: IrqType::Error,
            bytes,
            followup: None,
        });
    }

    fn chain_followup(&mut self, irq: IrqType, bytes: Vec<u8>, delay: u64) {
        if let Some(idx) = self
            .pending
            .iter()
            .rposition(|ev| ev.irq == IrqType::Acknowledge && ev.followup.is_none())
        {
            self.pending[idx].followup = Some(PendingFollowup { delay, irq, bytes });
            return;
        }
        // Fallback for probes/tests that inject events directly.
        self.insert_pending_event(PendingEvent {
            deadline: self.scheduling_cycle.saturating_add(delay),
            irq,
            bytes,
            followup: None,
        });
    }

    fn insert_pending_event(&mut self, event: PendingEvent) {
        let idx = self
            .pending
            .iter()
            .position(|existing| existing.deadline > event.deadline)
            .unwrap_or(self.pending.len());
        self.pending.insert(idx, event);
    }

    fn schedule_lid_transition_at(&mut self, now: u64, delay: u64) {
        self.lid_deadline = Some(now.saturating_add(delay));
    }

    fn tick_lid_state_machine(&mut self, cycles_now: u64) {
        while let Some(deadline) = self.lid_deadline {
            if deadline >= cycles_now {
                break;
            }
            self.lid_deadline = None;
            match self.drive_state {
                DriveState::Stopped => {}
                DriveState::Standby => {
                    self.drive_status &= !drive_status_bit::SEEKING;
                    if !self.disc_present {
                        self.drive_status |= drive_status_bit::SHELL_OPEN;
                        self.drive_state = DriveState::LidOpen;
                    }
                }
                DriveState::LidOpen => {
                    if self.disc_present {
                        // SHELL_OPEN stays sticky until a subsequent
                        // GetStat consumes it.
                        self.drive_state = DriveState::RescanCd;
                        self.schedule_lid_transition_at(cycles_now, CD_READ_TIME * 105);
                    } else {
                        self.schedule_lid_transition_at(cycles_now, CD_READ_TIME * 3);
                    }
                }
                DriveState::RescanCd => {
                    self.motor_on = true;
                    self.drive_state = DriveState::PrepareCd;
                    self.schedule_lid_transition_at(cycles_now, LID_PREPARE_SPINUP_CYCLES);
                }
                DriveState::PrepareCd => {
                    self.drive_status |= drive_status_bit::SEEKING;
                    self.drive_state = DriveState::Standby;
                    self.schedule_lid_transition_at(cycles_now, LID_PREPARE_SEEK_CYCLES);
                }
            }
        }
    }

    /// Stop the live sector stream and strip any queued DataReady work
    /// from both the pending queue and ACK followups. Redux cancels the
    /// read interrupt source on ReadN/Pause/Seek/Init/Reset; without
    /// this, stale sectors from an older stream can leak into the next
    /// command sequence.
    fn cancel_pending_data_ready_events(&mut self) {
        self.pending.retain(|ev| ev.irq != IrqType::DataReady);
        for ev in self.pending.iter_mut() {
            if ev
                .followup
                .as_ref()
                .is_some_and(|f| f.irq == IrqType::DataReady)
            {
                ev.followup = None;
            }
        }
    }

    fn stat_byte(&self) -> u8 {
        let mut s = self.drive_status & !(drive_status_bit::MOTOR_ON | drive_status_bit::READING);
        if self.motor_on {
            s |= drive_status_bit::MOTOR_ON;
        }
        if self.reading {
            s |= drive_status_bit::READING;
        }
        s
    }

    fn cmd_getstat(&mut self) {
        let stat = self.stat_byte();
        self.schedule_first_response(vec![stat]);
        // Redux keeps STATUS_SHELLOPEN sticky until GetStat observes
        // it, then clears the latched bit after producing the reply
        // unless the lid is genuinely still open.
        if self.drive_state != DriveState::LidOpen {
            self.drive_status &= !drive_status_bit::SHELL_OPEN;
        }
    }

    fn cmd_setloc(&mut self, params: &[u8]) {
        if params.len() >= 3 {
            let next_msf = (params[0], params[1], params[2]);
            let next_lba = msf_to_lba(next_msf.0, next_msf.1, next_msf.2);
            let current_lba = if self.read_lba != 0 {
                self.read_lba
            } else {
                msf_to_lba(self.setloc_msf.0, self.setloc_msf.1, self.setloc_msf.2)
            };
            if next_lba.abs_diff(current_lba) > 16 {
                self.seek_done = false;
            }
            self.setloc_msf = next_msf;
            self.setloc_pending = true;
        }
        let stat = self.stat_byte();
        self.schedule_first_response(vec![stat]);
    }

    fn cmd_setmode(&mut self, params: &[u8]) {
        if let Some(&m) = params.first() {
            if self.mode & 0x40 == 0 && m & 0x40 != 0 {
                self.xa_left.reset();
                self.xa_right.reset();
            }
            self.mode = m;
        }
        let stat = self.stat_byte();
        self.schedule_first_response(vec![stat]);
    }

    fn cmd_mute(&mut self, muted: bool) {
        self.muted = muted;
        self.schedule_first_response(vec![self.stat_byte()]);
    }

    fn cmd_set_filter(&mut self, params: &[u8]) {
        self.xa_filter_file = params.first().copied().unwrap_or(0);
        self.xa_filter_channel = params.get(1).copied().unwrap_or(0);
        self.schedule_first_response(vec![self.stat_byte()]);
    }

    fn cmd_get_param(&mut self) {
        self.schedule_first_response(vec![
            self.stat_byte(),
            self.mode,
            0,
            self.xa_filter_file,
            self.xa_filter_channel,
        ]);
    }

    /// Delay from a ReadN/ReadS command to the first DataReady event.
    /// Redux uses one full CD frame at double-speed here, then switches
    /// to half-frame cadence for the chained stream.
    fn initial_sector_read_cycles(&self) -> u64 {
        if self.mode & 0x80 != 0 {
            CD_READ_TIME
        } else {
            CD_READ_TIME * 2
        }
    }

    /// Cycles between chained DataReady events once a sector stream is
    /// active. Redux's `readInterrupt()` schedules steady double-speed
    /// reads at `cdReadTime / 2`, not `cdReadTime`; the old value fed
    /// XA audio at half rate, which made long music streams underrun.
    fn sector_read_cycles(&self) -> u64 {
        if self.mode & 0x80 != 0 {
            CD_READ_TIME / 2
        } else {
            CD_READ_TIME
        }
    }

    fn cmd_stop(&mut self) {
        self.motor_on = false;
        self.drive_state = DriveState::Stopped;
        self.lid_deadline = None;
        self.lid_bootstrap_pending = false;
        self.reading = false;
        self.cancel_pending_data_ready_events();
        self.location_changed = false;
        self.drive_status &= !drive_status_bit::PLAYING;
        self.reset_xa_stream();
        self.schedule_first_response(vec![self.stat_byte()]);
        let stat = self.stat_byte();
        self.schedule_second_response(vec![stat], SEEK_SECOND_RESPONSE_CYCLES);
    }

    fn cmd_pause(&mut self) {
        let was_motor_on = self.motor_on;
        // Pause halts the sector-read chain but leaves the motor on.
        // Missing this flip meant DataReady events kept chaining
        // `load_next_sector + schedule_sector_event` indefinitely
        // after the BIOS asked us to pause, producing a runaway
        // pending queue that burned the entire CPU budget on
        // peripheral-scheduling overhead.
        self.reading = false;
        self.cancel_pending_data_ready_events();
        self.location_changed = false;
        self.drive_status &= !drive_status_bit::PLAYING;
        self.reset_xa_stream();
        self.schedule_first_response(vec![self.stat_byte()]);
        let stat = self.stat_byte();
        // Redux uses a short ~7000-cycle follow-up when the drive is
        // already spun up ("standby"), and a much longer completion
        // only when pausing from a stopped / not-ready state. MGS hits
        // the standby path: without the short follow-up, Redux raises a
        // general CDROM IRQ ~7k cycles later and we don't.
        let delay = if was_motor_on {
            PAUSE_COMPLETE_CYCLES_STANDBY
        } else if self.mode & 0x80 != 0 {
            PAUSE_COMPLETE_CYCLES_ACTIVE * 2
        } else {
            PAUSE_COMPLETE_CYCLES_ACTIVE
        };
        self.schedule_second_response(vec![stat], delay);
    }

    fn cmd_motor_on(&mut self) {
        self.motor_on = true;
        self.drive_state = DriveState::Standby;
        self.schedule_first_response(vec![self.stat_byte()]);
    }

    /// CdlReadToc (0x1E): re-scan the disc table-of-contents.
    /// Two-part response:
    /// - INT3 (Acknowledge) with stat, immediately.
    /// - INT2 (Complete) with stat, ~20 M cycles later (Redux:
    ///   `cdReadTime * 180 / 4 = 20_321_280`). No track data is
    ///   returned in either response — the BIOS queries individual
    ///   track info via GetTD after ReadTOC completes.
    ///
    /// The BIOS's disc-boot sequence blocks on the INT2 here; we
    /// used to fall through to `cmd_getstat` on 0x1E, which only
    /// emitted the INT3 and left the BIOS waiting forever on the
    /// INT2. Surfaced as MGS + Crash hanging on the Sony splash
    /// at step ~90 M.
    fn cmd_read_toc(&mut self) {
        let stat = self.stat_byte();
        self.schedule_first_response(vec![stat]);
        // Redux value: `cdReadTime * 180 / 4`. We inline the
        // literal to avoid introducing a new named constant
        // here.
        const READ_TOC_SECOND_RESPONSE_CYCLES: u64 = 451_584 * 180 / 4;
        self.schedule_second_response(vec![stat], READ_TOC_SECOND_RESPONSE_CYCLES);
    }

    fn cmd_init(&mut self) {
        // Init is a drive reset — also halt any in-flight read so
        // DataReady chains from a previous ReadN don't keep firing
        // across the reset boundary.
        self.reading = false;
        self.cancel_pending_data_ready_events();
        self.location_changed = false;
        self.drive_status &= !drive_status_bit::PLAYING;
        self.clear_data_fifo();
        self.reset_xa_stream();
        self.muted = false;
        self.last_sector_header = [0; 4];
        self.last_sector_subheader = [0; 4];
        self.last_sector_header_valid = false;
        // Redux returns only the pre-init ACK here; the later 20480
        // cycle work happens on the lid/rescan state machine, not as a
        // second CPU-visible CDROM completion IRQ.
        self.schedule_first_response(vec![self.stat_byte()]);
        self.seek_done = true;
        self.motor_on = true;
        self.drive_status |= drive_status_bit::SHELL_OPEN;
        self.drive_state = DriveState::RescanCd;
        self.lid_deadline = None;
        self.lid_bootstrap_pending = true;
    }

    fn cmd_set_session(&mut self, _params: &[u8]) {
        if !self.disc_present {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
            return;
        }
        self.schedule_first_response(vec![self.stat_byte()]);
        self.schedule_second_response(vec![self.stat_byte()], 33_868);
    }

    fn cmd_reset(&mut self) {
        self.pending.clear();
        self.responses.clear();
        self.reading = false;
        self.seek_done = false;
        self.setloc_pending = false;
        self.location_changed = false;
        self.drive_status &= !drive_status_bit::PLAYING;
        self.clear_data_fifo();
        self.reset_xa_stream();
        self.last_sector_header = [0; 4];
        self.last_sector_subheader = [0; 4];
        self.last_sector_header_valid = false;
        self.motor_on = false;
        self.drive_state = DriveState::Stopped;
        self.lid_deadline = None;
        self.lid_bootstrap_pending = false;
        self.drive_status = 0;
        self.muted = false;
        self.schedule_first_response(vec![self.stat_byte()]);
    }

    fn cmd_seek(&mut self) {
        // Need a disc / motor. Without disc we still "seek" but it
        // succeeds immediately on the real drive — BIOS rarely calls
        // SeekL without a disc.
        self.reading = false;
        self.cancel_pending_data_ready_events();
        self.location_changed = false;
        self.schedule_first_response(vec![self.stat_byte()]);
        let stat = self.stat_byte() | drive_status_bit::SEEKING;
        let delay = if self.seek_done {
            0x800
        } else {
            SEEK_SECOND_RESPONSE_CYCLES
        };
        self.schedule_second_response(vec![stat], delay);
        self.seek_done = true;
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
        self.cancel_pending_data_ready_events();
        self.drive_status &= !drive_status_bit::PLAYING;
        self.reading = true;
        self.seek_done = true;
        self.xa_first_sector = 1;
        self.schedule_first_response(vec![self.stat_byte()]);
        if self.setloc_pending {
            let (m, s, f) = self.setloc_msf;
            self.read_lba = msf_to_lba(m, s, f);
            self.setloc_pending = false;
            self.location_changed = true;
        } else if self.read_lba == 0 {
            let (m, s, f) = self.setloc_msf;
            self.read_lba = msf_to_lba(m, s, f);
        }
        // Redux arms the first ReadN/ReadS sector from inside the
        // command ACK handler (`interrupt()`), not from the original
        // command write. Chaining it off the ACK keeps the first
        // DataReady deadline anchored on the actual ACK service cycle
        // rather than `scheduling_cycle`, which otherwise lands the
        // first sector ~0x800 cycles too early and makes MGS service
        // a CDROM IRQ before Redux does.
        self.chain_followup(
            IrqType::DataReady,
            vec![self.stat_byte()],
            self.initial_sector_read_cycles(),
        );
    }

    fn schedule_sector_event_at(&mut self, base_cycle: u64, delay: u64) {
        let stat = self.stat_byte();
        self.insert_pending_event(PendingEvent {
            deadline: base_cycle.saturating_add(delay),
            irq: IrqType::DataReady,
            bytes: vec![stat],
            followup: None,
        });
        self.sector_events_scheduled = self.sector_events_scheduled.saturating_add(1);
    }

    /// On DataReady event firing: populate the data FIFO with the
    /// next sector's user data and bump the read LBA. When the
    /// sector's subheader marks it as an XA audio block and mode
    /// bit 6 (XA ADPCM enable) is set, we ALSO decode the audio
    /// half into `cd_audio` for the SPU's CD input. Called from
    /// `tick` once per sector event.
    ///
    /// Returns whether this sector should raise the CPU-visible
    /// DataReady IRQ. Redux suppresses DataReady for XA audio sectors
    /// while STRSND is enabled, but still schedules the next sector.
    fn load_next_sector(&mut self) -> bool {
        let lba = self.read_lba;
        self.read_lba = self.read_lba.wrapping_add(1);
        if let Some(disc) = self.disc.as_ref() {
            if let Some(raw) = disc.read_sector_raw(lba) {
                self.last_sector_header.copy_from_slice(&raw[12..16]);
                self.last_sector_subheader.copy_from_slice(&raw[16..20]);
                self.last_sector_header_valid = true;
                let submode = raw[18];
                let suppress_data_ready = self.mode & 0x40 != 0 && submode & 0x04 != 0;

                // If XA mode is on, only decode sectors that match the
                // Redux gate: unmuted, audio submode set, matching
                // file/channel filter, and a live first-sector state.
                // Games with XA-streamed cutscenes use a single ReadN
                // to pull both sector kinds; matching audio sectors go
                // to the SPU and skip the CPU-visible data FIFO.
                if !self.muted && self.mode & 0x40 != 0 && self.xa_first_sector != -1 {
                    let file = raw[16];
                    let channel = raw[17];

                    if self.xa_first_sector == 1 && self.mode & 0x08 == 0 {
                        self.xa_filter_file = file;
                        self.xa_filter_channel = channel;
                    }

                    if submode & 0x04 != 0
                        && file == self.xa_filter_file
                        && channel == self.xa_filter_channel
                        && channel != 0xFF
                    {
                        if self.xa_first_sector == 1 || self.xa_coding.is_none() {
                            let Some(coding) = parse_xa_coding(raw[19]) else {
                                self.xa_first_sector = -1;
                                return !suppress_data_ready;
                            };
                            if self.xa_coding != Some(coding) {
                                self.xa_left.reset();
                                self.xa_right.reset();
                                self.xa_coding = Some(coding);
                            }
                        }
                        let coding = self.xa_coding.expect("XA coding seeded above");
                        if let Some(mut samples) = decode_xa_audio_sector(
                            raw,
                            coding,
                            &mut self.xa_left,
                            &mut self.xa_right,
                        ) {
                            self.attenuate_xa_samples(&mut samples);
                            let cap = 44_100; // ~1 s at SPU rate
                            let overflow =
                                (self.cd_audio.len() + samples.len()).saturating_sub(cap);
                            for _ in 0..overflow {
                                self.cd_audio.pop_front();
                            }
                            self.cd_audio.extend(samples.iter().copied());
                            self.xa_first_sector = 0;
                            self.data_fifo.clear();
                            self.data_fifo_ready = false;
                            self.data_transfer_active = false;
                            return false;
                        }
                        self.xa_first_sector = -1;
                    }
                }

                self.data_fifo.clear();
                self.data_fifo_ready = false;
                self.data_transfer_active = false;
                let whole_sector = self.mode & 0x20 != 0;
                let sector_mode = raw[15];
                if whole_sector {
                    if sector_mode == 1 {
                        self.data_fifo.extend(raw[12..16].iter().copied());
                        self.data_fifo.extend([0; 8]);
                        self.data_fifo.extend(raw[16..16 + 2048].iter().copied());
                    } else {
                        self.data_fifo.extend(raw[12..12 + 2340].iter().copied());
                    }
                } else if sector_mode == 1 {
                    self.data_fifo.extend(raw[16..16 + 2048].iter().copied());
                } else {
                    self.data_fifo.extend(raw[24..24 + 2048].iter().copied());
                }
                self.data_fifo_ready = !self.data_fifo.is_empty();
                self.data_transfer_active = false;
                return !suppress_data_ready;
            }

            // Raw-sector miss means we ran off the end of the image.
        }
        // Past end of disc — stop the read and leave the FIFO empty.
        self.reading = false;
        self.location_changed = false;
        self.data_fifo.clear();
        self.data_fifo_ready = false;
        self.data_transfer_active = false;
        true
    }

    /// CdlGetlocL (0x10) — return the current logical position
    /// and sector-header info. 8-byte reply:
    /// `[MM, SS, FF, Mode, File, Channel, Submode, Coding]` from the
    /// last delivered sector's raw header/subheader.
    ///
    /// Without a disc, returns an INT5 error like GetID.
    fn cmd_get_loc_l(&mut self) {
        if !self.disc_present {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
            return;
        }
        if !self.last_sector_header_valid {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat]);
            return;
        }
        let mut resp = Vec::with_capacity(8);
        resp.extend_from_slice(&self.last_sector_header);
        resp.extend_from_slice(&self.last_sector_subheader);
        self.schedule_first_response(resp);
    }

    /// CdlGetlocP (0x11) — return the current physical play
    /// position. 8-byte reply: `[Track, Index, RMM, RSS, RSECT,
    /// AMM, ASS, ASECT]`. RMM/RSS/RSECT are relative to the
    /// track/index start; AMM/ASS/ASECT are absolute MSF.
    fn cmd_get_loc_p(&mut self) {
        if !self.disc_present {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
            return;
        }
        let Some(disc) = self.disc.as_ref() else {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat]);
            return;
        };
        let lba = if self.reading {
            self.read_lba.saturating_sub(1)
        } else {
            self.read_lba
        };
        let Some(pos) = disc.track_position_for_lba(lba) else {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat]);
            return;
        };
        let (rm, rs, rf) = pos.relative_msf;
        let (am, as_, af) = pos.absolute_msf;
        self.schedule_first_response(vec![
            bin_to_bcd(pos.track_number),
            pos.index_number,
            bin_to_bcd(rm),
            bin_to_bcd(rs),
            bin_to_bcd(rf),
            bin_to_bcd(am),
            bin_to_bcd(as_),
            bin_to_bcd(af),
        ]);
    }

    /// CdlGetTN (0x13) — first and last track numbers from the disc's
    /// track table.
    fn cmd_get_tn(&mut self) {
        if !self.disc_present {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
            return;
        }
        let Some(disc) = self.disc.as_ref() else {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat]);
            return;
        };
        let (Some(first), Some(last)) = (disc.first_track_number(), disc.last_track_number())
        else {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat]);
            return;
        };
        self.schedule_first_response(vec![self.stat_byte(), bin_to_bcd(first), bin_to_bcd(last)]);
    }

    /// CdlGetTD (0x14) — start time for a given track, or lead-out for
    /// track 0. Parameter is a BCD track number.
    fn cmd_get_td(&mut self, params: &[u8]) {
        if !self.disc_present {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
            return;
        }
        let Some(disc) = self.disc.as_ref() else {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat]);
            return;
        };
        let track = bcd_to_bin(params.first().copied().unwrap_or(0));
        if track == 0xFF {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat]);
            return;
        }
        let target_lba = if track == 0 {
            disc.leadout_lba()
        } else {
            let Some(start_lba) = disc.track_start_lba(track) else {
                let stat = self.stat_byte() | drive_status_bit::ERROR;
                self.schedule_error_response(vec![stat]);
                return;
            };
            start_lba
        };
        let (m, s, _f) = lba_to_msf(target_lba);
        self.schedule_first_response(vec![self.stat_byte(), bin_to_bcd(m), bin_to_bcd(s)]);
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
    fn cmd_play(&mut self, params: &[u8]) {
        if !self.disc_present {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat, 0x80]);
            return;
        }
        let Some(disc) = self.disc.as_ref() else {
            let stat = self.stat_byte() | drive_status_bit::ERROR;
            self.schedule_error_response(vec![stat]);
            return;
        };
        if let Some(&track_bcd) = params.first() {
            let track = bcd_to_bin(track_bcd);
            if track == 0xFF {
                let stat = self.stat_byte() | drive_status_bit::ERROR;
                self.schedule_error_response(vec![stat]);
                return;
            }
            if track != 0 {
                let Some(start_lba) = disc.track_start_lba(track) else {
                    let stat = self.stat_byte() | drive_status_bit::ERROR;
                    self.schedule_error_response(vec![stat]);
                    return;
                };
                self.read_lba = start_lba;
            }
        } else if self.setloc_msf != (0, 0, 0) {
            let (m, s, f) = self.setloc_msf;
            self.read_lba = msf_to_lba(m, s, f);
        } else if self.read_lba == 0 {
            self.read_lba = disc
                .track_start_lba(disc.first_track_number().unwrap_or(1))
                .unwrap_or(0);
        }
        self.motor_on = true;
        self.reading = false;
        self.drive_status |= drive_status_bit::PLAYING;
        self.schedule_first_response(vec![self.stat_byte()]);
    }

    fn cmd_test(&mut self, params: &[u8]) {
        // Only Test 0x20 (drive version / BIOS date) is commonly used
        // by the BIOS. Must match Redux byte-for-byte — the BIOS's
        // IRQ handler stores the 4-byte response into a kernel
        // buffer, and later code paths read those bytes back to
        // dispatch on firmware version. Parity step 89,184,517
        // diverged on a byte out of this buffer.
        //
        // Redux (cdrom.cc): `Test20[] = {0x98, 0x06, 0x10, 0xC3}`.
        // Format is YY MM DD VER — 1998-06-10 v0xC3, matching the
        // SCPH-550x / 700x firmware Redux targets by default.
        match params.first().copied() {
            Some(0x20) => {
                self.schedule_first_response(vec![0x98, 0x06, 0x10, 0xC3]);
            }
            _ => self.cmd_getstat(),
        }
    }

    fn cmd_getid(&mut self) {
        if self.disc_present {
            // Licensed PS1 disc response: 8 bytes.
            //   [0]: stat (0x02 = motor on)
            //   [1]: flags (0x00 = licensed)
            //   [2]: disc type (0x20 = mode-2 data)
            //   [3]: reserved 0
            //   [4..=7]: region code
            //
            // Region comes from the disc's license sector (LBA 4
            // user-data), which carries the same "Sony Computer
            // Entertainment Amer/Euro/Inc." strings the BIOS checks.
            // Falling back to SCEA keeps homebrew / synthetic discs on
            // the permissive US path instead of spuriously rejecting.
            let stat = self.stat_byte();
            let region = self.disc.as_ref().map(disc_region_code).unwrap_or(*b"SCEA");
            self.schedule_first_response(vec![stat]);
            self.schedule_second_response(
                vec![
                    0x02, 0x00, 0x20, 0x00, region[0], region[1], region[2], region[3],
                ],
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

        self.tick_lid_state_machine(cycles_now);

        while let Some(front) = self.pending.front() {
            // Redux's scheduled interrupt queue only dispatches when
            // `target < cycle` (see `R3000Acpu::branchTest`). CDROM
            // responses live on that queue, unlike root counters /
            // VBlank which update on equality. Keep CDROM strict so a
            // BIOS poll on the exact target cycle still sees no IRQ.
            if front.deadline >= cycles_now {
                break;
            }

            // Check IRQ-flag gate BEFORE popping or running the
            // event. Hardware holds a latched IRQ until software
            // acks via 0x1F801803 idx=1; if we already have an
            // unacked IRQ, leave the front event in the queue.
            //
            // If we popped first and then re-queued on flag-set
            // (as the previous implementation did), we'd also run
            // the event's side effects (notably
            // `load_next_sector` + chained
            // `schedule_sector_event_at`) on every failed attempt.
            // MGS triggered that tight loop: 46.9 M sector events
            // scheduled and 11.7 M DataReady pops — enough
            // load_next_sector re-entries to bury the emulator in
            // ISR dispatch cycles (16.7 cyc/step vs 2.4 baseline),
            // stranding MGS at the PlayStation splash.
            if self.irq_flag != 0 {
                // Bump the front event's deadline slightly so the
                // next tick re-checks, rather than spinning on an
                // already-due event every tick until the ack lands.
                // Matches the prior "check again in ~200 cycles"
                // cadence without ever re-running the event body.
                let delay = FIRST_RESPONSE_CYCLES / 10;
                if let Some(mut ev) = self.pending.pop_front() {
                    ev.deadline = cycles_now.saturating_add(delay);
                    self.insert_pending_event(ev);
                }
                break;
            }

            let ev = self.pending.pop_front().unwrap();

            // If the drive was paused/reset between this event's
            // scheduling and now, drop it silently. On real
            // hardware, Pause/Init kills in-flight sector reads;
            // letting a stale DataReady fire would clobber the
            // data FIFO that software was about to drain, or
            // deliver the wrong LBA entirely.
            if ev.irq == IrqType::DataReady && !self.reading {
                continue;
            }

            // DataReady events drive the sector-stream — load the
            // next sector's payload into the data FIFO as the event
            // fires, and chain the subsequent DataReady (anchored on
            // `cycles_now` so the next sector fires at the steady
            // stream cadence after the PREVIOUS one, not from the
            // ancient `cmd_read` issue time).
            let mut should_raise_irq = true;
            if ev.irq == IrqType::DataReady {
                should_raise_irq = self.load_next_sector();
                if self.reading {
                    let delay = if self.location_changed {
                        self.location_changed = false;
                        self.sector_read_cycles().saturating_mul(30)
                    } else {
                        self.sector_read_cycles()
                    };
                    self.schedule_sector_event_at(cycles_now, delay);
                }
            }
            if !should_raise_irq {
                continue;
            }
            // Like Redux's `m_result`, each IRQ publishes a fresh
            // packet rather than appending to any unread prior bytes.
            self.responses.clear();
            for b in ev.bytes.iter().copied() {
                if self.responses.len() < RESPONSE_FIFO_DEPTH {
                    self.responses.push_back(b);
                }
            }
            self.command_busy = false;
            // Raise IRQ. The flag-gate above already guaranteed
            // irq_flag was 0 on entry.
            self.irq_flag = ev.irq as u8;
            let ty = ev.irq as usize;
            if ty < self.irq_type_counts.len() {
                self.irq_type_counts[ty] = self.irq_type_counts[ty].saturating_add(1);
            }
            // Per-raise log for divergence probes. Cap-guarded so
            // long production runs don't bloat memory.
            if self.cdrom_irq_log.len() < self.cdrom_irq_log_cap {
                self.cdrom_irq_log.push((cycles_now, ev.irq as u8));
            }
            if let Some(followup) = ev.followup {
                self.insert_pending_event(PendingEvent {
                    deadline: cycles_now.saturating_add(followup.delay),
                    irq: followup.irq,
                    bytes: followup.bytes,
                    followup: None,
                });
            }
            if self.lid_bootstrap_pending && ev.irq == IrqType::Acknowledge {
                self.lid_bootstrap_pending = false;
                self.schedule_lid_transition_at(cycles_now, LID_BOOTSTRAP_CYCLES);
            }
            raised = true;
            // Hardware can only latch one IRQ at a time; subsequent
            // due events wait until this one is acked.
            break;
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

    /// Current CDROM controller INDEX (bits 0-1 of the status
    /// register at 0x1F801800). Low-level writes to 0x1F801801-3
    /// are routed through this index — reading 0x1F801803 with
    /// index=0 returns the IRQ mask, with index=1 returns the
    /// IRQ flag. Probes compare this against Redux's to catch
    /// index-tracking drift.
    pub fn index_value(&self) -> u8 {
        self.index
    }

    /// Current CDROM IRQ mask (the per-IRQ-type enable bits — the
    /// CPU-level I_MASK is separate). Written via 0x1F801802
    /// index=1. `setIrq` in Redux (and our raise-gate) checks
    /// `irq_flag & irq_mask` before waking the CPU.
    pub fn irq_mask_value(&self) -> u8 {
        self.irq_mask
    }

    /// Redux-equivalent `setIrq()` gate: the CDROM only escalates
    /// a latched IRQ to the PSX IRQ controller (I_STAT bit 2) when
    /// `irq_flag & irq_mask` is nonzero. When it's zero the
    /// response stays latched for polled access via 0x1F801803
    /// idx=1, but no CPU interrupt is dispatched. MGS (and other
    /// games) poll the flag with bits 0-2 of `irq_mask` cleared —
    /// relying on this gate to keep CDROM acks from firing the
    /// ISR while the BIOS's loader code walks the response
    /// manually. Skipping the gate (our pre-fix behaviour) caused
    /// the BIOS to run an ISR it didn't expect, stomping state
    /// the MGS boot loop needed.
    pub fn should_wake_cpu(&self) -> bool {
        (self.irq_flag & self.irq_mask) != 0
    }

    /// Number of pending events queued. 0 means the CDROM has
    /// nothing scheduled to fire.
    pub fn pending_queue_len(&self) -> usize {
        self.pending.len()
    }

    /// Absolute cycle used as the base for the most recent command's
    /// response scheduling. Diagnostic-only.
    pub fn scheduling_cycle(&self) -> u64 {
        self.scheduling_cycle
    }

    /// Front pending event as `(deadline, irq_type)`. Lets probes
    /// compare the next latched CDROM action against Redux without
    /// exposing the full private queue.
    pub fn next_pending_event(&self) -> Option<(u64, IrqType)> {
        self.pending.front().map(|ev| (ev.deadline, ev.irq))
    }

    /// Enable per-raise logging up to `cap` entries. Probes call
    /// this once before running; afterward, read `cdrom_irq_log`.
    pub fn enable_irq_log(&mut self, cap: usize) {
        self.cdrom_irq_log_cap = cap;
        self.cdrom_irq_log.reserve(cap);
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

    /// `true` when the request register's bit-7 latch has armed the
    /// current sector buffer for MMIO/DMA consumption.
    pub fn data_transfer_armed(&self) -> bool {
        self.data_transfer_active
    }

    /// Pull one byte from the data FIFO — used by DMA channel 3's
    /// block-read path to drain a sector into RAM. Returns `0` when
    /// the FIFO is empty (hardware returns stale-bus bytes; `0` is
    /// a safe stand-in).
    pub fn pop_data_byte(&mut self) -> u8 {
        self.data_fifo_pops = self.data_fifo_pops.saturating_add(1);
        self.pop_data_fifo_byte()
    }

    /// Number of bytes currently buffered in the data FIFO.
    pub fn data_fifo_len(&self) -> usize {
        self.data_fifo.len()
    }

    /// Current sector-buffer length expressed as DMA words. Used when
    /// software programs a zero-sized CDROM DMA and expects the drive
    /// to fall back to the active sector size (2048/2340 bytes).
    pub fn data_fifo_words(&self) -> u32 {
        self.data_fifo.len().div_ceil(4) as u32
    }

    #[cfg(test)]
    pub(crate) fn debug_seed_data_fifo(&mut self, bytes: &[u8], ready: bool, armed: bool) {
        self.data_fifo.clear();
        self.data_fifo.extend(bytes.iter().copied());
        self.data_fifo_ready = ready;
        self.data_transfer_active = armed;
    }
}

impl Default for CdRom {
    fn default() -> Self {
        Self::new()
    }
}

// Shared helpers — `lba_to_msf` + `bin_to_bcd` live in `psx-iso` so
// any crate that speaks the CDROM protocol (tools, test harnesses)
// can use them.
use psx_iso::{bin_to_bcd, lba_to_msf};

fn disc_region_code(disc: &Disc) -> [u8; 4] {
    let Some(user) = disc.read_sector_user(4) else {
        return *b"SCEA";
    };
    let text = String::from_utf8_lossy(user);
    if text.contains("Sony Computer Entertainment Amer") {
        *b"SCEA"
    } else if text.contains("Sony Computer Entertainment Euro")
        || text.contains("Sony Computer Entertainment Inc. for U.K.")
    {
        *b"SCEE"
    } else if text.contains("Sony Computer Entertainment Inc.") {
        *b"SCEI"
    } else {
        *b"SCEA"
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
/// Decodes the Redux-supported XA layouts: 4-bit/8-bit, mono/stereo,
/// at 37.8 kHz or 18.9 kHz. Unsupported coding nibbles return `None`.
///
/// Samples are decoded in Redux's sound-unit order, then resampled from
/// the XA source rate up to the SPU's 44.1 kHz rate on output.
fn parse_xa_coding(coding: u8) -> Option<XaCoding> {
    let stereo = match coding & 0x03 {
        1 => true,
        0 => false,
        _ => return None,
    };
    let freq = match (coding >> 2) & 0x03 {
        0 => 37_800u32,
        1 => 18_900u32,
        _ => return None,
    };
    let nbits = match (coding >> 4) & 0x03 {
        0 => 4u8,
        1 => 8u8,
        _ => return None,
    };
    Some(XaCoding {
        stereo,
        freq,
        nbits,
    })
}

fn decode_xa_audio_sector(
    raw: &[u8],
    coding: XaCoding,
    left: &mut crate::spu::XaDecoderState,
    right: &mut crate::spu::XaDecoderState,
) -> Option<Vec<(i16, i16)>> {
    if raw.len() < 2352 {
        return None;
    }
    let stereo = coding.stereo;
    let freq = coding.freq;
    let nbits = coding.nbits;

    // XA payload starts at offset 24 (after 12+4+8 bytes of header).
    // 18 sound groups × 128 bytes. 4-bit stereo yields 2016 source
    // frames, while 4-bit mono yields 4032. 8-bit modes carry fewer
    // sound units per group; keep the same Redux unpacking path rather
    // than rejecting the stream and prematurely killing playback.
    let payload = &raw[24..24 + 18 * 128];
    let units_per_group = if nbits == 4 { 4 } else { 2 };
    let mut decoded: Vec<(i16, i16)> =
        Vec::with_capacity(18 * units_per_group * 28 * if stereo { 1 } else { 2 });
    let head_table = [0usize, 2, 8, 10];
    for group_idx in 0..18 {
        let group = &payload[group_idx * 128..group_idx * 128 + 128];
        let headers = &group[0..16];
        let data = &group[16..128];

        for unit in 0..units_per_group {
            let decode_words = if nbits == 4 {
                let mut low_words = [0u16; 7];
                let mut high_words = [0u16; 7];
                for k in 0..7 {
                    let base = k * 16 + unit;
                    let b0 = data[base] as u16;
                    let b1 = data[base + 4] as u16;
                    let b2 = data[base + 8] as u16;
                    let b3 = data[base + 12] as u16;
                    low_words[k] =
                        (b0 & 0x0F) | ((b1 & 0x0F) << 4) | ((b2 & 0x0F) << 8) | ((b3 & 0x0F) << 12);
                    high_words[k] =
                        (b0 >> 4) | ((b1 >> 4) << 4) | ((b2 >> 4) << 8) | ((b3 >> 4) << 12);
                }
                (low_words, high_words)
            } else {
                let mut words = [0u16; 7];
                for k in 0..7 {
                    let base = k * 8 + unit;
                    words[k] = data[base] as u16 | ((data[base + 4] as u16) << 8);
                }
                (words, words)
            };

            let mut first_samples = [0i16; 28];
            crate::spu::xa_decode_block(
                left,
                headers[head_table[unit]],
                &decode_words.0,
                &mut first_samples,
                1,
            );

            if stereo {
                let mut second_samples = [0i16; 28];
                crate::spu::xa_decode_block(
                    right,
                    headers[head_table[unit] + 1],
                    &decode_words.1,
                    &mut second_samples,
                    1,
                );
                for i in 0..28 {
                    decoded.push((first_samples[i], second_samples[i]));
                }
            } else {
                for &sample in &first_samples {
                    decoded.push((sample, sample));
                }
                let mut second_samples = [0i16; 28];
                crate::spu::xa_decode_block(
                    left,
                    headers[head_table[unit] + 1],
                    &decode_words.1,
                    &mut second_samples,
                    1,
                );
                for &sample in &second_samples {
                    decoded.push((sample, sample));
                }
            }
        }
    }

    // Upsample to the SPU rate. This is still a simple resampler, but
    // the sector decode above now matches Redux's sound-unit ordering
    // and frame count.
    let mut resampled: Vec<(i16, i16)> =
        Vec::with_capacity(decoded.len() * 44_100 / freq as usize + 1);
    let src_n = decoded.len() as u32;
    let dst_n = (src_n as u64 * 44_100 / freq as u64) as u32;
    for i in 0..dst_n {
        let src_idx = ((i as u64 * src_n as u64) / dst_n as u64) as usize;
        resampled.push(decoded[src_idx.min(decoded.len() - 1)]);
    }
    Some(resampled)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_sector(header: [u8; 4], subheader: [u8; 4], payload_fill: u8) -> Vec<u8> {
        let mut raw = vec![0u8; psx_iso::SECTOR_BYTES];
        raw[12..16].copy_from_slice(&header);
        raw[16..20].copy_from_slice(&subheader);
        raw[20..24].copy_from_slice(&subheader);
        let payload_start = if header[3] == 1 { 16 } else { 24 };
        raw[payload_start..payload_start + 2048].fill(payload_fill);
        raw
    }

    fn multitrack_disc_with_pregap() -> Disc {
        Disc::from_tracks(vec![
            psx_iso::Track {
                number: 1,
                track_type: psx_iso::TrackType::Data,
                start_lba: 0,
                sector_count: 10,
                pregap: 0,
                file_pregap: 0,
                bytes: vec![0u8; psx_iso::SECTOR_BYTES * 10],
            },
            psx_iso::Track {
                number: 2,
                track_type: psx_iso::TrackType::Audio,
                start_lba: 12,
                sector_count: 4,
                pregap: 2,
                file_pregap: 0,
                bytes: vec![0u8; psx_iso::SECTOR_BYTES * 4],
            },
        ])
    }

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
    fn data_fifo_read_without_transfer_request_returns_zero_and_keeps_buffered_sector() {
        let mut cd = CdRom::new();
        cd.data_fifo.extend([0x12, 0x34]);
        cd.data_fifo_ready = true;

        assert_ne!(cd.read8(BASE) & status_bit::DATA_FIFO_NOT_EMPTY, 0);
        assert_eq!(cd.read8(BASE + 2), 0);
        assert_eq!(cd.data_fifo_len(), 2);
        assert_eq!(cd.data_fifo.front().copied(), Some(0x12));
        assert_eq!(cd.read8(BASE) & status_bit::DATA_FIFO_NOT_EMPTY, 0);
    }

    #[test]
    fn request_register_bit7_arms_transfer_until_sector_buffer_drains() {
        let mut cd = CdRom::new();
        cd.data_fifo.extend([0x12, 0x34]);
        cd.data_fifo_ready = true;

        cd.write8(BASE + 3, 0x80);
        assert_eq!(cd.read8(BASE + 2), 0x12);
        assert_ne!(cd.read8(BASE) & status_bit::DATA_FIFO_NOT_EMPTY, 0);
        assert_eq!(cd.read8(BASE + 2), 0x34);
        assert_eq!(cd.read8(BASE) & status_bit::DATA_FIFO_NOT_EMPTY, 0);
        assert!(!cd.data_transfer_active);
        assert_eq!(cd.read8(BASE + 2), 0);
    }

    #[test]
    fn new_command_discards_unread_response_and_sets_busy_bit() {
        let mut cd = CdRom::new();
        cd.responses.push_back(0x55);

        cd.queue_command(0x01, 123);

        assert!(
            cd.responses.is_empty(),
            "new command should replace old packet"
        );
        assert_ne!(
            cd.read8(BASE) & status_bit::TRANSMISSION_BUSY,
            0,
            "status bit 7 should latch while the command is pending"
        );
    }

    #[test]
    fn delivered_irq_packet_replaces_stale_bytes_and_clears_busy_bit() {
        let mut cd = CdRom::new();
        cd.command_busy = true;
        cd.responses.push_back(0x11);
        cd.insert_pending_event(PendingEvent {
            deadline: 100,
            irq: IrqType::Acknowledge,
            bytes: vec![0xAA, 0xBB],
            followup: None,
        });

        assert!(cd.tick(101));
        assert_eq!(cd.read8(BASE + 1), 0xAA);
        assert_eq!(cd.read8(BASE + 1), 0xBB);
        assert_eq!(cd.read8(BASE + 1), 0);
        assert_eq!(cd.read8(BASE) & status_bit::TRANSMISSION_BUSY, 0);
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
    fn init_only_queues_ack_and_latches_shell_open() {
        let mut cd = CdRom::new();
        cd.scheduling_cycle = 1_000;

        cd.cmd_init();

        assert_eq!(cd.pending.len(), 1);
        let ack = cd.pending.front().expect("Init ACK pending");
        assert_eq!(ack.irq, IrqType::Acknowledge);
        assert!(ack.followup.is_none(), "Init should not invent INT2");
        assert!(cd.motor_on, "Init should spin the motor up");
        assert_eq!(
            cd.drive_status & drive_status_bit::SHELL_OPEN,
            drive_status_bit::SHELL_OPEN
        );
        assert!(cd.seek_done, "Init resets the seek latch to DONE");
    }

    #[test]
    fn init_ack_bootstraps_lid_rescan_state_machine() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(Disc::from_bin(vec![0u8; psx_iso::SECTOR_BYTES])));
        cd.scheduling_cycle = 1_000;

        cd.cmd_init();
        assert!(cd.lid_bootstrap_pending);
        assert_eq!(cd.drive_state, DriveState::RescanCd);

        let ack_cycle = 1_000 + FIRST_RESPONSE_CYCLES + 1;
        assert!(cd.tick(ack_cycle));
        assert!(!cd.lid_bootstrap_pending);
        assert_eq!(cd.lid_deadline, Some(ack_cycle + LID_BOOTSTRAP_CYCLES));

        let rescan_cycle = ack_cycle + LID_BOOTSTRAP_CYCLES + 1;
        assert!(!cd.tick(rescan_cycle));
        assert_eq!(cd.drive_state, DriveState::PrepareCd);
        assert_eq!(
            cd.lid_deadline,
            Some(rescan_cycle + LID_PREPARE_SPINUP_CYCLES)
        );

        let prepare_cycle = rescan_cycle + LID_PREPARE_SPINUP_CYCLES + 1;
        assert!(!cd.tick(prepare_cycle));
        assert_eq!(cd.drive_state, DriveState::Standby);
        assert_ne!(cd.drive_status & drive_status_bit::SEEKING, 0);

        let settle_cycle = prepare_cycle + LID_PREPARE_SEEK_CYCLES + 1;
        assert!(!cd.tick(settle_cycle));
        assert_eq!(cd.drive_state, DriveState::Standby);
        assert_eq!(cd.drive_status & drive_status_bit::SEEKING, 0);
    }

    #[test]
    fn getstat_reports_then_clears_shell_open_sticky_bit() {
        let mut cd = CdRom::new();
        cd.drive_status |= drive_status_bit::SHELL_OPEN;
        cd.scheduling_cycle = 1_000;

        cd.cmd_getstat();

        assert_eq!(cd.drive_status & drive_status_bit::SHELL_OPEN, 0);
        assert!(cd.tick(1_000 + FIRST_RESPONSE_CYCLES + 1));
        let stat = cd.read8(BASE + 1);
        assert_eq!(
            stat & drive_status_bit::SHELL_OPEN,
            drive_status_bit::SHELL_OPEN
        );
        assert_eq!(cd.read8(BASE + 1), 0);
    }

    #[test]
    fn getlocl_without_disc_returns_error() {
        let mut cd = CdRom::new();
        // No disc.
        cd.cmd_get_loc_l();
        // First tick rebases the pending event's relative deadline
        // to absolute; second tick past the deadline delivers it.
        cd.tick(0);
        cd.tick(10_000_000);
        let first = cd.responses.front().copied().unwrap_or(0);
        assert_ne!(first & drive_status_bit::ERROR, 0);
    }

    #[test]
    fn getlocl_without_prior_sector_returns_error_even_with_disc() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(Disc::from_bin(vec![0u8; psx_iso::SECTOR_BYTES])));

        cd.cmd_get_loc_l();
        cd.tick(0);
        cd.tick(10_000_000);

        let first = cd.read8(BASE + 1);
        assert_ne!(first & drive_status_bit::ERROR, 0);
        assert_eq!(
            cd.read8(BASE + 1),
            0,
            "invalid-header error should be a 1-byte reply"
        );
    }

    #[test]
    fn getlocl_returns_last_sector_header_and_subheader() {
        let mut cd = CdRom::new();
        let header = [0x12, 0x34, 0x56, 0x02];
        let subheader = [0xAA, 0xBB, 0xCC, 0xDD];
        cd.insert_disc(Some(Disc::from_bin(raw_sector(header, subheader, 0x6C))));

        cd.load_next_sector();
        assert_eq!(cd.data_fifo_len(), 2048);
        assert_eq!(cd.data_fifo.front().copied(), Some(0x6C));

        cd.cmd_get_loc_l();
        cd.tick(0);
        cd.tick(10_000_000);

        for expected in header.into_iter().chain(subheader) {
            assert_eq!(cd.read8(BASE + 1), expected);
        }
    }

    #[test]
    fn load_next_sector_whole_sector_mode_returns_2340_bytes() {
        let mut cd = CdRom::new();
        let header = [0x01, 0x02, 0x03, 0x02];
        let subheader = [0x10, 0x20, 0x30, 0x40];
        let raw = raw_sector(header, subheader, 0xAB);
        cd.mode = 0x20;
        cd.insert_disc(Some(Disc::from_bin(raw.clone())));

        cd.load_next_sector();

        assert_eq!(cd.data_fifo_len(), 2340);
        let first_bytes: Vec<u8> = cd.data_fifo.iter().take(12).copied().collect();
        assert_eq!(first_bytes, raw[12..24].to_vec());
    }

    #[test]
    fn load_next_sector_mode1_uses_payload_after_header() {
        let mut cd = CdRom::new();
        let mut raw = raw_sector([0x00, 0x02, 0x00, 0x01], [0; 4], 0x00);
        raw[16..16 + 2048].fill(0x5D);
        raw[24..24 + 2048].fill(0xA7);
        cd.insert_disc(Some(Disc::from_bin(raw)));

        cd.load_next_sector();

        assert_eq!(cd.data_fifo_len(), 2048);
        assert_eq!(cd.data_fifo.front().copied(), Some(0x5D));
    }

    #[test]
    fn load_next_sector_sets_ready_latch_but_leaves_transfer_disarmed() {
        let mut cd = CdRom::new();
        let raw = raw_sector([0x00, 0x02, 0x00, 0x02], [0; 4], 0x6C);
        cd.insert_disc(Some(Disc::from_bin(raw)));

        cd.load_next_sector();

        assert!(cd.data_fifo_ready);
        assert!(!cd.data_transfer_active);
        assert_ne!(cd.read8(BASE) & status_bit::DATA_FIFO_NOT_EMPTY, 0);
        assert_eq!(cd.pop_data_byte(), 0);
        assert_eq!(cd.data_fifo_len(), 2048);
        assert_eq!(cd.data_fifo.front().copied(), Some(0x6C));
    }

    #[test]
    fn sector_read_cycles_match_redux_initial_and_stream_cadence() {
        let mut cd = CdRom::new();
        // Default mode = 0x80 → double-speed.
        assert_eq!(cd.initial_sector_read_cycles(), CD_READ_TIME);
        assert_eq!(cd.sector_read_cycles(), CD_READ_TIME / 2);
        // Flipping bit 7 off via SetMode gives single-speed (2×).
        cd.cmd_setmode(&[0x00]);
        assert_eq!(cd.initial_sector_read_cycles(), CD_READ_TIME * 2);
        assert_eq!(cd.sector_read_cycles(), CD_READ_TIME);
        // Setting other bits without bit 7 stays single-speed.
        cd.cmd_setmode(&[0x60]);
        assert_eq!(cd.initial_sector_read_cycles(), CD_READ_TIME * 2);
        assert_eq!(cd.sector_read_cycles(), CD_READ_TIME);
        // Back to double-speed when bit 7 returns.
        cd.cmd_setmode(&[0x80]);
        assert_eq!(cd.initial_sector_read_cycles(), CD_READ_TIME);
        assert_eq!(cd.sector_read_cycles(), CD_READ_TIME / 2);
    }

    #[test]
    fn read_command_uses_initial_delay_then_steady_stream_delay() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(Disc::from_bin(vec![0u8; psx_iso::SECTOR_BYTES * 4])));
        cd.scheduling_cycle = 1_000;

        cd.cmd_read();
        assert_eq!(cd.pending.len(), 1); // command ACK only; first DataReady is chained off it
        assert_eq!(
            cd.pending.front().map(|ev| ev.irq),
            Some(IrqType::Acknowledge)
        );

        assert!(cd.tick(1_000 + FIRST_RESPONSE_CYCLES + 1));
        let first_data_ready = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::DataReady)
            .expect("first DataReady scheduled from ACK fire time");
        assert_eq!(
            first_data_ready.deadline,
            1_000 + FIRST_RESPONSE_CYCLES + 1 + CD_READ_TIME
        );
        cd.irq_flag = 0;
        cd.responses.clear();

        assert!(cd.tick(1_000 + FIRST_RESPONSE_CYCLES + 1 + CD_READ_TIME + 1));
        let next_data_ready = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::DataReady)
            .expect("steady DataReady scheduled");
        assert_eq!(
            next_data_ready.deadline,
            1_000 + FIRST_RESPONSE_CYCLES + 1 + CD_READ_TIME + 1 + CD_READ_TIME / 2
        );
    }

    #[test]
    fn xa_audio_sector_suppresses_dataready_irq_but_keeps_streaming() {
        let mut cd = CdRom::new();
        let xa_sector = raw_sector([0x00, 0x02, 0x00, 0x02], [0x07, 0x02, 0x24, 0x01], 0);
        let data_sector = raw_sector([0x00, 0x02, 0x01, 0x02], [0x07, 0x02, 0x00, 0x00], 0x5A);
        let mut disc = xa_sector;
        disc.extend(data_sector);
        cd.insert_disc(Some(Disc::from_bin(disc)));
        cd.mode = 0xC0; // double-speed + STRSND/XA enable
        cd.setloc_msf = (0x00, 0x02, 0x00); // LBA 0
        cd.scheduling_cycle = 1_000;

        cd.cmd_read();
        assert!(cd.tick(1_000 + FIRST_RESPONSE_CYCLES + 1));
        cd.irq_flag = 0;
        cd.responses.clear();

        assert!(
            !cd.tick(1_000 + FIRST_RESPONSE_CYCLES + 1 + CD_READ_TIME + 1),
            "Redux suppresses DataReady IRQs for STRSND XA audio sectors"
        );
        assert_eq!(cd.irq_flag, 0);
        assert!(cd.responses.is_empty());
        assert!(
            cd.cd_audio_queue_len() > 0,
            "XA sector should still feed decoded samples to the SPU"
        );
        assert_eq!(cd.irq_type_counts[IrqType::DataReady as usize], 0);
        let next_data_ready = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::DataReady)
            .expect("suppressed audio sector should still chain the read stream");
        assert_eq!(
            next_data_ready.deadline,
            1_000 + FIRST_RESPONSE_CYCLES + 1 + CD_READ_TIME + 1 + CD_READ_TIME / 2
        );
    }

    #[test]
    fn pause_on_spun_up_drive_uses_short_followup_delay() {
        let mut cd = CdRom::new();
        cd.motor_on = true;
        cd.reading = true;
        cd.scheduling_cycle = 1_000;
        cd.insert_pending_event(PendingEvent {
            deadline: 50_000,
            irq: IrqType::DataReady,
            bytes: vec![0x20],
            followup: None,
        });

        cd.cmd_pause();
        assert_eq!(
            cd.pending.len(),
            1,
            "Pause should cancel the in-flight read chain"
        );

        let ack_deadline = 1_000 + FIRST_RESPONSE_CYCLES;
        assert!(cd.tick(ack_deadline + 1));
        let pause_complete = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::Complete)
            .expect("pause completion chained off ACK");
        assert_eq!(
            pause_complete.deadline,
            ack_deadline + 1 + PAUSE_COMPLETE_CYCLES_STANDBY
        );
    }

    #[test]
    fn seek_command_uses_short_followup_after_drive_has_seeked_once() {
        let mut cd = CdRom::new();
        cd.seek_done = true;
        cd.scheduling_cycle = 1_000;

        cd.cmd_seek();

        let ack_deadline = 1_000 + FIRST_RESPONSE_CYCLES;
        assert!(cd.tick(ack_deadline + 1));
        let seek_complete = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::Complete)
            .expect("seek completion chained off ACK");
        assert_eq!(seek_complete.deadline, ack_deadline + 1 + 0x800);
    }

    #[test]
    fn setloc_far_target_clears_seek_done_and_marks_pending() {
        let mut cd = CdRom::new();
        cd.seek_done = true;
        cd.read_lba = 200;

        cd.cmd_setloc(&[0x00, 0x02, 0x16]);

        assert!(
            !cd.seek_done,
            "far SetLoc should force the next seek slow-path"
        );
        assert!(
            cd.setloc_pending,
            "SetLoc should latch until Read/Play consumes it"
        );
    }

    #[test]
    fn read_command_cancels_old_stream_and_rearms_first_sector() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(Disc::from_bin(vec![0u8; psx_iso::SECTOR_BYTES * 32])));
        cd.reading = true;
        cd.scheduling_cycle = 1_000;
        cd.setloc_msf = (0x00, 0x02, 0x16);
        cd.setloc_pending = true;
        cd.insert_pending_event(PendingEvent {
            deadline: 50_000,
            irq: IrqType::DataReady,
            bytes: vec![0x20],
            followup: None,
        });

        cd.cmd_read();

        assert!(
            cd.pending.iter().all(|ev| ev.irq != IrqType::DataReady),
            "stale DataReady events from the previous stream must be cancelled"
        );
        let ack = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::Acknowledge)
            .expect("ReadN ACK present");
        let followup = ack.followup.as_ref().expect("first sector chained off ACK");
        assert_eq!(followup.irq, IrqType::DataReady);
        assert_eq!(followup.delay, cd.initial_sector_read_cycles());
    }

    #[test]
    fn relocated_read_stretches_second_sector_gap() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(Disc::from_bin(vec![0u8; psx_iso::SECTOR_BYTES * 64])));
        cd.scheduling_cycle = 1_000;
        cd.setloc_msf = (0x00, 0x02, 0x16);
        cd.setloc_pending = true;

        cd.cmd_read();

        let ack_deadline = 1_000 + FIRST_RESPONSE_CYCLES;
        assert!(cd.tick(ack_deadline + 1));
        cd.irq_flag = 0;

        let first_sector_deadline = ack_deadline + 1 + CD_READ_TIME;
        assert!(cd.tick(first_sector_deadline + 1));
        let next_sector = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::DataReady)
            .expect("read stream should continue after first sector");
        assert_eq!(
            next_sector.deadline,
            first_sector_deadline + 1 + (CD_READ_TIME / 2) * 30
        );
        assert!(
            !cd.location_changed,
            "the long-gap latch should clear once it has stretched one sector"
        );
    }

    #[test]
    fn gettn_single_track_disc_reports_one_to_one() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(Disc::from_bin(vec![0u8; psx_iso::SECTOR_BYTES * 10])));

        cd.cmd_get_tn();
        cd.tick(10_000_000);

        assert_eq!(cd.read8(BASE + 1), cd.stat_byte());
        assert_eq!(cd.read8(BASE + 1), 0x01);
        assert_eq!(cd.read8(BASE + 1), 0x01);
    }

    #[test]
    fn gettd_track_one_reports_data_start() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(Disc::from_bin(vec![0u8; psx_iso::SECTOR_BYTES * 10])));

        cd.cmd_get_td(&[0x01]);
        cd.tick(10_000_000);

        assert_eq!(cd.read8(BASE + 1), cd.stat_byte());
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x02);
    }

    #[test]
    fn gettd_track_zero_reports_leadout_minute_second() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(Disc::from_bin(vec![0u8; psx_iso::SECTOR_BYTES * 10])));

        cd.cmd_get_td(&[0x00]);
        cd.tick(10_000_000);

        let _stat = cd.read8(BASE + 1);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x02);
    }

    #[test]
    fn getlocp_reports_index0_and_index1_for_pregap_tracks() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(multitrack_disc_with_pregap()));

        cd.read_lba = 10;
        cd.cmd_get_loc_p();
        cd.tick(10_000_000);
        assert_eq!(cd.read8(BASE + 1), 0x02);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x01);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x02);
        assert_eq!(cd.read8(BASE + 1), 0x10);
        cd.irq_flag = 0;

        cd.read_lba = 12;
        cd.cmd_get_loc_p();
        cd.tick(20_000_000);
        assert_eq!(cd.read8(BASE + 1), 0x02);
        assert_eq!(cd.read8(BASE + 1), 0x01);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x02);
        assert_eq!(cd.read8(BASE + 1), 0x12);
    }

    #[test]
    fn gettn_reports_last_track_for_multitrack_disc() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(multitrack_disc_with_pregap()));

        cd.cmd_get_tn();
        cd.tick(10_000_000);

        assert_eq!(cd.read8(BASE + 1), cd.stat_byte());
        assert_eq!(cd.read8(BASE + 1), 0x01);
        assert_eq!(cd.read8(BASE + 1), 0x02);
    }

    #[test]
    fn gettd_reports_track_start_and_leadout_for_multitrack_disc() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(multitrack_disc_with_pregap()));

        cd.cmd_get_td(&[0x02]);
        cd.tick(10_000_000);
        assert_eq!(cd.read8(BASE + 1), cd.stat_byte());
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x02);
        cd.irq_flag = 0;

        cd.cmd_get_td(&[0x00]);
        cd.tick(20_000_000);
        assert_eq!(cd.read8(BASE + 1), cd.stat_byte());
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x02);
    }

    #[test]
    fn play_track_param_seeks_to_requested_track_start() {
        let mut cd = CdRom::new();
        cd.insert_disc(Some(multitrack_disc_with_pregap()));

        cd.cmd_play(&[0x02]);
        cd.tick(10_000_000);

        assert_eq!(cd.read_lba, 12);
        assert_eq!(
            cd.read8(BASE + 1) & drive_status_bit::PLAYING,
            drive_status_bit::PLAYING
        );
    }

    #[test]
    fn setfilter_and_getparam_roundtrip_filter_state() {
        let mut cd = CdRom::new();
        cd.mode = 0xE8;

        cd.cmd_set_filter(&[0x12, 0x34]);
        assert_eq!(cd.xa_filter_file, 0x12);
        assert_eq!(cd.xa_filter_channel, 0x34);
        cd.pending.clear();
        cd.responses.clear();
        cd.irq_flag = 0;

        cd.cmd_get_param();
        cd.tick(10_000_000);
        assert_eq!(cd.read8(BASE + 1), cd.stat_byte());
        assert_eq!(cd.read8(BASE + 1), 0xE8);
        assert_eq!(cd.read8(BASE + 1), 0x00);
        assert_eq!(cd.read8(BASE + 1), 0x12);
        assert_eq!(cd.read8(BASE + 1), 0x34);
    }

    #[test]
    fn mute_and_demute_commands_flip_latch() {
        let mut cd = CdRom::new();
        assert!(!cd.muted);

        cd.cmd_mute(true);
        cd.tick(10_000_000);
        assert!(cd.muted);

        cd.pending.clear();
        cd.responses.clear();
        cd.irq_flag = 0;

        cd.cmd_mute(false);
        cd.tick(20_000_000);
        assert!(!cd.muted);
    }

    #[test]
    fn xa_decode_silent_stereo_sector_has_full_frame_count() {
        let mut raw = vec![0u8; psx_iso::SECTOR_BYTES];
        raw[15] = 2;
        raw[18] = 0x24;
        raw[19] = 0x01; // 4-bit stereo, 37.8 kHz

        let mut left = crate::spu::XaDecoderState::new();
        let mut right = crate::spu::XaDecoderState::new();
        let coding = parse_xa_coding(raw[19]).expect("valid XA coding");
        let samples = decode_xa_audio_sector(&raw, coding, &mut left, &mut right)
            .expect("common 4-bit stereo XA should decode");

        assert_eq!(samples.len(), 2352);
        assert!(samples.iter().all(|&(l, r)| l == 0 && r == 0));
    }

    #[test]
    fn xa_decode_silent_mono_sector_has_full_frame_count() {
        let mut raw = vec![0u8; psx_iso::SECTOR_BYTES];
        raw[15] = 2;
        raw[18] = 0x24;
        raw[19] = 0x00; // 4-bit mono, 37.8 kHz

        let mut left = crate::spu::XaDecoderState::new();
        let mut right = crate::spu::XaDecoderState::new();
        let coding = parse_xa_coding(raw[19]).expect("valid XA coding");
        let samples = decode_xa_audio_sector(&raw, coding, &mut left, &mut right)
            .expect("4-bit mono XA should decode");

        assert_eq!(samples.len(), 4704);
        assert!(samples.iter().all(|&(l, r)| l == 0 && r == 0));
    }

    #[test]
    fn xa_decode_uses_stream_coding_not_each_sector_byte() {
        let mut raw = vec![0u8; psx_iso::SECTOR_BYTES];
        raw[15] = 2;
        raw[18] = 0x24;
        raw[19] = 0x0c; // invalid if reparsed; Redux ignores this mid-stream.

        let mut left = crate::spu::XaDecoderState::new();
        let mut right = crate::spu::XaDecoderState::new();
        let coding = XaCoding {
            stereo: true,
            freq: 37_800,
            nbits: 4,
        };
        let samples = decode_xa_audio_sector(&raw, coding, &mut left, &mut right)
            .expect("stream coding should drive decode after the first sector");

        assert_eq!(samples.len(), 2352);
        assert!(samples.iter().all(|&(l, r)| l == 0 && r == 0));
    }

    #[test]
    fn motor_on_command_sets_motor_flag() {
        let mut cd = CdRom::new();
        assert!(!cd.motor_on);

        cd.cmd_motor_on();
        cd.tick(10_000_000);

        assert!(cd.motor_on);
        assert_eq!(
            cd.read8(BASE + 1) & drive_status_bit::MOTOR_ON,
            drive_status_bit::MOTOR_ON
        );
    }

    #[test]
    fn reset_cancels_read_and_clears_fifo() {
        let mut cd = CdRom::new();
        cd.motor_on = true;
        cd.reading = true;
        cd.data_fifo.push_back(0xAB);
        cd.pending.push_back(PendingEvent {
            deadline: 123,
            irq: IrqType::DataReady,
            bytes: vec![0x20],
            followup: None,
        });

        cd.cmd_reset();
        cd.tick(10_000_000);

        assert!(!cd.motor_on);
        assert!(!cd.reading);
        assert!(cd.data_fifo.is_empty());
        assert_eq!(
            cd.pending.len(),
            0,
            "reset should leave only its own ack already delivered"
        );
        assert_eq!(cd.read8(BASE + 1), 0x00);
    }

    #[test]
    fn pending_events_are_kept_sorted_by_deadline() {
        let mut cd = CdRom::new();
        cd.insert_pending_event(PendingEvent {
            deadline: 300,
            irq: IrqType::DataReady,
            bytes: vec![0x20],
            followup: None,
        });
        cd.insert_pending_event(PendingEvent {
            deadline: 100,
            irq: IrqType::Acknowledge,
            bytes: vec![0x00],
            followup: None,
        });
        cd.insert_pending_event(PendingEvent {
            deadline: 200,
            irq: IrqType::Complete,
            bytes: vec![0x00],
            followup: None,
        });

        let deadlines = cd.pending.iter().map(|ev| ev.deadline).collect::<Vec<_>>();
        assert_eq!(deadlines, vec![100, 200, 300]);
    }

    #[test]
    fn followup_chains_to_latest_ack_even_with_later_events_present() {
        let mut cd = CdRom::new();
        cd.insert_pending_event(PendingEvent {
            deadline: 300,
            irq: IrqType::DataReady,
            bytes: vec![0x20],
            followup: None,
        });
        cd.insert_pending_event(PendingEvent {
            deadline: 100,
            irq: IrqType::Acknowledge,
            bytes: vec![0x00],
            followup: None,
        });

        cd.chain_followup(IrqType::Complete, vec![0x01], 77);

        let ack = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::Acknowledge)
            .expect("ack event present");
        let data_ready = cd
            .pending
            .iter()
            .find(|ev| ev.irq == IrqType::DataReady)
            .expect("later event present");
        assert_eq!(
            ack.followup
                .as_ref()
                .map(|f| (f.delay, f.irq, f.bytes.clone())),
            Some((77, IrqType::Complete, vec![0x01]))
        );
        assert!(
            data_ready.followup.is_none(),
            "later event must stay untouched"
        );
    }

    #[test]
    fn disc_region_code_uses_license_sector_text() {
        use psx_iso::{SECTOR_BYTES, SECTOR_USER_DATA_OFFSET};

        let mut bytes = vec![0u8; SECTOR_BYTES * 6];
        let license = b"Licensed by Sony Computer Entertainment Europe for PlayStation";
        let off = 4 * SECTOR_BYTES + SECTOR_USER_DATA_OFFSET;
        bytes[off..off + license.len()].copy_from_slice(license);

        let disc = Disc::from_bin(bytes);
        assert_eq!(disc_region_code(&disc), *b"SCEE");
    }
}
