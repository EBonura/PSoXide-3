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
    /// Total commands dispatched since reset — diagnostic counter.
    commands_dispatched: u64,
}

impl CdRom {
    /// Fresh controller — shell open, motor off, all FIFOs empty,
    /// IRQ disabled. Matches hardware state a few cycles after reset,
    /// before the BIOS has had a chance to write anything.
    pub fn new() -> Self {
        Self {
            index: 0,
            // Cold boot: shell "open" per hardware convention when no
            // disc has been seated yet. Motor off. Everything else
            // clear. The BIOS will see this via its first GetStat.
            drive_status: drive_status_bit::SHELL_OPEN,
            params: VecDeque::with_capacity(PARAM_FIFO_DEPTH),
            responses: VecDeque::with_capacity(RESPONSE_FIFO_DEPTH),
            irq_mask: 0,
            irq_flag: 0,
            commands_dispatched: 0,
        }
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
            //   Stubbed to zero until sector reads land in 6d.
            (2, _) => 0,
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
        // Parameter-FIFO-empty bit is set when the FIFO has room.
        if self.params.is_empty() {
            s |= status_bit::PARAM_FIFO_EMPTY;
        }
        if self.params.len() < PARAM_FIFO_DEPTH {
            s |= status_bit::PARAM_FIFO_NOT_FULL;
        }
        if !self.responses.is_empty() {
            s |= status_bit::RESPONSE_FIFO_NOT_EMPTY;
        }
        // Data FIFO state (bit 6) + transmission busy (bit 7) + ADPCM
        // busy (bit 2) come from subsystems not yet wired in; all zero
        // for the moment.
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

    /// Placeholder command-dispatch hook — Phase 6b wires commands to
    /// actual behaviour. Recording dispatches so the smoke-test can
    /// see what the BIOS is trying to do.
    fn queue_command(&mut self, command: u8) {
        self.commands_dispatched += 1;
        let _ = command;
        // Phase 6b: schedule 1st response at +50_000 cycles,
        // 2nd response (if any) at +1_000_000 cycles, etc.
        // For 6a we just drop the command and keep status-FIFO hygiene.
    }

    /// Diagnostic: total commands received.
    pub fn commands_dispatched(&self) -> u64 {
        self.commands_dispatched
    }

    /// Diagnostic: current IRQ-flag value.
    pub fn irq_flag(&self) -> u8 {
        self.irq_flag
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
    fn shell_open_is_set_on_cold_boot() {
        let cd = CdRom::new();
        assert_eq!(cd.drive_status & drive_status_bit::SHELL_OPEN, drive_status_bit::SHELL_OPEN);
        assert_eq!(cd.drive_status & drive_status_bit::MOTOR_ON, 0);
    }
}
