//! System bus: owns physical memory and dispatches loads to regions.
//!
//! Current coverage: RAM, BIOS, scratchpad. Everything else panics on
//! access — deliberately, because we want unmapped reads to be loud
//! until each region's owning module (GPU, SPU, CD-ROM, …) is wired up.

use psx_hw::memory::{self, to_physical};
use thiserror::Error;

use crate::cdrom::CdRom;
use crate::dma::Dma;
use crate::gpu::Gpu;
use crate::irq::{Irq, IrqSource};
use crate::mmio_trace::{MmioKind, MmioTrace};
use crate::sio::Sio0;
use crate::spu::Spu;
use crate::timers::Timers;

/// Physical address of `I_STAT` (interrupt status / ack register).
const IRQ_STAT_ADDR: u32 = 0x1F80_1070;
/// Physical address of `I_MASK` (interrupt enable register).
const IRQ_MASK_ADDR: u32 = 0x1F80_1074;

/// Errors constructing a [`Bus`].
#[derive(Error, Debug)]
pub enum BusError {
    /// BIOS image was not exactly 512 KiB.
    #[error("BIOS image must be exactly {expected} bytes, got {actual}")]
    BiosSize {
        /// Expected size in bytes.
        expected: usize,
        /// Size that was actually provided.
        actual: usize,
    },
}

/// The PS1 system bus.
pub struct Bus {
    ram: Box<[u8; memory::ram::SIZE]>,
    bios: Box<[u8; memory::bios::SIZE]>,
    scratchpad: Box<[u8; memory::scratchpad::SIZE]>,
    /// Write-echoes-on-read buffer for the MMIO window. **Placeholder.**
    /// Individual peripherals with real semantics (IRQ below, later GPU /
    /// SPU / CD-ROM / DMA / timers) intercept their own ranges ahead of
    /// this fallback; the rest of MMIO still round-trips writes to reads.
    io: Box<[u8; memory::io::SIZE]>,
    /// Interrupt controller (`I_STAT` / `I_MASK`). Accessed via the MMIO
    /// dispatch below and queried by the CPU each step to update
    /// `COP0.CAUSE.IP[2]`.
    irq: Irq,
    /// Root counters (Timer 0 / 1 / 2). Phase 2e is register-backing
    /// only; ticking lands with the cycle model.
    pub timers: Timers,
    /// DMA controller (7 channels + DPCR + DICR). Phase 2g is
    /// register-backing only; transfers land as subsystems come online.
    dma: Dma,
    /// GPU — owns VRAM and handles the GP0/GP1 MMIO ports. The
    /// frontend's VRAM viewer reads `bus.gpu.vram` directly.
    pub gpu: Gpu,
    /// SPU — full 24-voice ADPCM synthesis, ADSR envelopes, stereo
    /// mixing at 44.1 kHz. Output drains into `spu.audio_out`; the
    /// frontend pulls samples every frame via [`Spu::drain_audio`].
    /// Public so the frontend can access the audio queue + tests can
    /// inspect voice state directly.
    pub spu: Spu,
    /// SIO0 — controller / memory-card port. Currently models a cold
    /// port with nothing connected; enough to satisfy BIOS init polls.
    sio0: Sio0,
    /// CD-ROM controller — byte-granular MMIO at 0x1F80_1800..=0x1803.
    /// Exposed public so diagnostics can inspect FIFO / command state.
    pub cdrom: CdRom,
    /// Motion decoder. Defensive stub today — register shape is
    /// faithful (idle / empty status, DMA-enable latching, reset)
    /// but no real Huffman / IDCT / YUV→RGB. Games that poll MDEC
    /// status see plausible values instead of the unmapped 0xFFFF_FFFF.
    pub mdec: crate::mdec::Mdec,
    /// Cumulative CPU cycles retired since reset. Fed by `Cpu::step`
    /// via [`Bus::tick`]. Peripherals read this to schedule events
    /// (VBlank, timer ticks, DMA completion). Phase 4a just counts;
    /// Phase 4b starts firing IRQs off it.
    cycles: u64,
    // VBlank scheduling lives in `scheduler` under
    // [`EventSlot::VBlank`]. Seeded at `FIRST_VBLANK_CYCLE` by
    // `Bus::new`; every VBlank handler invocation re-schedules the
    // next one `VBLANK_PERIOD_CYCLES` later.
    /// Unified event scheduler — the 15-slot queue that owns
    /// DMA / CDROM / VBlank / SPU / MDEC / SIO timings, matching
    /// Redux's `m_regs.interrupt` + `intTargets`. See
    /// [`crate::scheduler`] for the model.
    ///
    /// Migration status: DMA channel completions (slots `GpuDma`,
    /// `GpuOtcDma`, `CdrDma`, `MdecInDma`, `MdecOutDma`, `SpuDma`)
    /// run through the scheduler. VBlank, CDROM command / read
    /// events, SPU async, SIO — still on their legacy per-subsystem
    /// timers; migrations land in follow-up commits.
    pub scheduler: crate::scheduler::Scheduler,
    /// Bounded ring buffer of recent MMIO accesses. Zero-sized and
    /// no-op at every call site unless the `trace-mmio` Cargo feature
    /// is enabled — see `mmio_trace.rs` for the rationale.
    pub mmio_trace: MmioTrace,
    /// When true, the CPU replaces fetches at `0xA0` / `0xB0` / `0xC0`
    /// with a host-Rust implementation of the BIOS syscall they
    /// dispatch to. Off by default so parity tests stay bit-exact
    /// against Redux (which does the real BIOS ROM dispatch).
    /// Turned on by [`Bus::enable_hle_bios`] — typically right after
    /// side-loading an EXE that wants BIOS services but skipped the
    /// BIOS's own init.
    pub hle_bios_enabled: bool,
    /// Per-(table, func) count of HLE BIOS calls. Diagnostic only.
    /// `[table][func]` where table is 0=A, 1=B, 2=C.
    hle_bios_calls: [[u32; 256]; 3],
    /// HSync cycles for the current video region (NTSC = 2146,
    /// PAL = 2157). Used by the timer bank's HBlank source and by
    /// the VBlank scheduler. Flipped by [`Bus::set_pal_mode`];
    /// defaults to NTSC for existing parity tests.
    hsync_cycles: u64,
    /// HSync cadence used by the VBlank scheduler. Redux changes
    /// the active PAL/NTSC thresholds on GP1 display-mode writes,
    /// but the base counter's target can retain its previous cadence;
    /// keeping this separate from Timer 1's HBlank source preserves
    /// that phase.
    vblank_hsync_cycles: u64,
    /// VBlank period in cycles — one full frame at the current
    /// video region. 564_398 for NTSC, 677_343 for PAL.
    vblank_period: u64,
    /// Addresses we've already logged as unmapped reads. Keeps
    /// log noise bounded when a buggy game pokes a bad pointer
    /// in a tight loop.
    unmapped_read_seen: std::collections::BTreeSet<u32>,
    /// Same, writes. Separate so read + write to the same bad
    /// address both log at least once.
    unmapped_write_seen: std::collections::BTreeSet<u32>,
    /// Diagnostic: when true, every DMA-completion schedule pushes
    /// a record to `dma_log`. Off by default — only the
    /// `probe_dma_schedules` example flips it on.
    dma_log_enabled: bool,
    dma_log: Vec<(String, u64, u64, u64)>,
}

// --- Video-timing constants ---
//
// Match Redux's `psxcounters.cc` math exactly so VBlank fires at the
// same cycle — and therefore at the same instruction — on both sides.
//
//   HSync period   = psxClockSpeed / (FrameRate × HSyncTotal)
//   NTSC: 33_868_800 / (60 × 263) = 2146 cycles/HSync,  564_398 cyc/frame
//   PAL : 33_868_800 / (50 × 314) = 2157 cycles/HSync,  677_343 cyc/frame
//
// `FIRST_VBLANK_CYCLE` is derived from the per-region VBlank-start
// scanline × HSync; kept as NTSC default to preserve existing parity
// tests. PAL builds call [`Bus::set_pal_mode`] before running, which
// re-seeds the VBlank scheduler and updates the tick-rate knobs.

const HSYNC_CYCLES_NTSC: u64 = 2146;
#[allow(dead_code)]
const HSYNC_TOTAL_NTSC: u64 = 263;
const VBLANK_START_SCANLINE_NTSC: u64 = 243;
const FIRST_VBLANK_CYCLE_NTSC: u64 = HSYNC_CYCLES_NTSC * VBLANK_START_SCANLINE_NTSC;
const VBLANK_PERIOD_CYCLES_NTSC: u64 = HSYNC_CYCLES_NTSC * HSYNC_TOTAL_NTSC;

const HSYNC_CYCLES_PAL: u64 = 2157;
#[allow(dead_code)]
const HSYNC_TOTAL_PAL: u64 = 314;
const VBLANK_START_SCANLINE_PAL: u64 = 256;
const FIRST_VBLANK_CYCLE_PAL: u64 = HSYNC_CYCLES_PAL * VBLANK_START_SCANLINE_PAL;
const VBLANK_PERIOD_CYCLES_PAL: u64 = HSYNC_CYCLES_PAL * HSYNC_TOTAL_PAL;

// NTSC first-VBlank constant kept for the default scheduler seed.
// PAL switch re-seeds via [`Bus::set_pal_mode`].
const FIRST_VBLANK_CYCLE: u64 = FIRST_VBLANK_CYCLE_NTSC;
#[allow(dead_code)]
const VBLANK_PERIOD_CYCLES: u64 = VBLANK_PERIOD_CYCLES_NTSC;

#[derive(Clone, Copy)]
struct VideoTiming {
    hsync: u64,
    period: u64,
    start_scanline: u64,
    total_scanlines: u64,
}

fn current_video_params(hsync: u64, period: u64) -> Option<VideoTiming> {
    if period == hsync.saturating_mul(HSYNC_TOTAL_NTSC) {
        Some(VideoTiming {
            hsync,
            period,
            start_scanline: VBLANK_START_SCANLINE_NTSC,
            total_scanlines: HSYNC_TOTAL_NTSC,
        })
    } else if period == hsync.saturating_mul(HSYNC_TOTAL_PAL) {
        Some(VideoTiming {
            hsync,
            period,
            start_scanline: VBLANK_START_SCANLINE_PAL,
            total_scanlines: HSYNC_TOTAL_PAL,
        })
    } else {
        None
    }
}

fn estimate_current_scanline(
    now: u64,
    next_vblank: u64,
    period: u64,
    hsync: u64,
    start_scanline: u64,
    total_scanlines: u64,
) -> u64 {
    let previous_vblank = previous_vblank_target(now, next_vblank, period);
    let since_previous = now.saturating_sub(previous_vblank);
    (start_scanline + since_previous / hsync.max(1)) % total_scanlines.max(1)
}

fn estimate_scanline_phase(now: u64, next_vblank: u64, period: u64, hsync: u64) -> u64 {
    let previous_vblank = previous_vblank_target(now, next_vblank, period);
    now.saturating_sub(previous_vblank) % hsync.max(1)
}

fn previous_vblank_target(now: u64, mut next_vblank: u64, period: u64) -> u64 {
    let period = period.max(1);
    while next_vblank > now {
        next_vblank = next_vblank.saturating_sub(period);
    }
    next_vblank
}

impl Bus {
    /// Build a bus with the given BIOS image. RAM and scratchpad are
    /// zero-initialised; hardware leaves them in an undefined state, but
    /// zeroing is deterministic and adequate for a cold-boot harness.
    pub fn new(bios: Vec<u8>) -> Result<Self, BusError> {
        if bios.len() != memory::bios::SIZE {
            return Err(BusError::BiosSize {
                expected: memory::bios::SIZE,
                actual: bios.len(),
            });
        }

        let bios_arr: Box<[u8; memory::bios::SIZE]> = bios
            .into_boxed_slice()
            .try_into()
            .expect("size was just checked");

        Ok(Self {
            ram: zeroed_box(),
            bios: bios_arr,
            scratchpad: zeroed_box(),
            io: zeroed_box(),
            irq: Irq::new(),
            timers: Timers::new(),
            dma: Dma::new(),
            gpu: Gpu::new(),
            spu: Spu::new(),
            sio0: Sio0::new(),
            cdrom: CdRom::new(),
            mdec: crate::mdec::Mdec::new(),
            cycles: 0,
            scheduler: {
                let mut s = crate::scheduler::Scheduler::new();
                // Seed the first VBlank at scanline 243. Every fire
                // of `EventSlot::VBlank` in `drain_scheduler_events`
                // reschedules the next one.
                s.schedule(crate::scheduler::EventSlot::VBlank, 0, FIRST_VBLANK_CYCLE);
                // NOTE: SPU scheduler seed is deliberately *not* here.
                // Reason: Redux's SPU runs in a detached `std::thread`
                // that doesn't run during the parity-oracle trace. If
                // we tick the SPU on every 768th cycle during the same
                // window, our ADSR advances (envelope non-zero,
                // voice_on_cycle bookkeeping) while Redux's stays
                // frozen — and downstream SPU reads diverge. Until
                // the parity oracle learns to pump Redux's SPU thread
                // synchronously, we leave SPU synthesis dormant during
                // CPU execution and pump it on demand from the
                // frontend's per-frame audio callback instead. See
                // `Spu::seed_scheduler` + `Bus::run_spu_samples`.
                s
            },
            mmio_trace: MmioTrace::new(),
            hle_bios_enabled: false,
            hle_bios_calls: [[0; 256]; 3],
            hsync_cycles: HSYNC_CYCLES_NTSC,
            vblank_hsync_cycles: HSYNC_CYCLES_NTSC,
            vblank_period: VBLANK_PERIOD_CYCLES_NTSC,
            unmapped_read_seen: std::collections::BTreeSet::new(),
            unmapped_write_seen: std::collections::BTreeSet::new(),
            dma_log_enabled: false,
            dma_log: Vec::new(),
        })
    }

    /// Switch to PAL video timing: 50 Hz refresh, 314-scanline
    /// frames, 2157 HSync cycles. Resets the VBlank scheduler to
    /// the PAL first-VBlank cycle + period, and reconfigures the
    /// HBlank tick rate for Timer 1. PAL retail games select this
    /// through GP1 display-mode writes; NTSC is the reset default.
    ///
    /// Calling this after any stepping has already happened leaves
    /// cumulative `cycles` in place — the next VBlank will still
    /// fire at the correct *frame* boundary even if mid-frame.
    pub fn set_pal_mode(&mut self) {
        self.set_video_region(true);
    }

    /// Switch to NTSC video timing: 60 Hz refresh, 263-scanline
    /// frames, 2146 HSync cycles.
    pub fn set_ntsc_mode(&mut self) {
        self.set_video_region(false);
    }

    fn set_video_region(&mut self, pal: bool) {
        let old = current_video_params(self.vblank_hsync_cycles, self.vblank_period);
        let (hsync, first_vblank, canonical_period, start_scanline, total_scanlines) = if pal {
            (
                HSYNC_CYCLES_PAL,
                FIRST_VBLANK_CYCLE_PAL,
                VBLANK_PERIOD_CYCLES_PAL,
                VBLANK_START_SCANLINE_PAL,
                HSYNC_TOTAL_PAL,
            )
        } else {
            (
                HSYNC_CYCLES_NTSC,
                FIRST_VBLANK_CYCLE_NTSC,
                VBLANK_PERIOD_CYCLES_NTSC,
                VBLANK_START_SCANLINE_NTSC,
                HSYNC_TOTAL_NTSC,
            )
        };
        if self.hsync_cycles == hsync {
            return;
        }
        let vblank_hsync = if self.cycles == 0 {
            hsync
        } else {
            old.map(|old| old.hsync).unwrap_or(hsync)
        };
        let period = if self.cycles == 0 {
            canonical_period
        } else {
            vblank_hsync.saturating_mul(total_scanlines)
        };
        let delay = if self.cycles == 0 {
            first_vblank
        } else {
            let current_scanline = old
                .and_then(|old| {
                    self.scheduler
                        .target(crate::scheduler::EventSlot::VBlank)
                        .map(|t| (old, t))
                })
                .map(|(old, next_vblank)| {
                    estimate_current_scanline(
                        self.cycles,
                        next_vblank,
                        old.period,
                        old.hsync,
                        old.start_scanline,
                        old.total_scanlines,
                    )
                })
                .unwrap_or(0);
            let line_phase = old
                .and_then(|old| {
                    self.scheduler
                        .target(crate::scheduler::EventSlot::VBlank)
                        .map(|t| (old, t))
                })
                .map(|(old, next_vblank)| {
                    estimate_scanline_phase(self.cycles, next_vblank, old.period, old.hsync)
                })
                .unwrap_or(0);
            let remaining_lines = if current_scanline < start_scanline {
                start_scanline - current_scanline
            } else {
                total_scanlines - current_scanline + start_scanline
            };
            // Redux's base counter target is not recalculated by the
            // display-mode write, so the next VBlank remains aligned
            // to the old hsync cadence while using the new region's
            // VBlank-start scanline.
            remaining_lines
                .saturating_mul(vblank_hsync)
                .saturating_sub(line_phase)
                .saturating_add(4)
                .max(1)
        };
        self.hsync_cycles = hsync;
        self.vblank_hsync_cycles = vblank_hsync;
        self.vblank_period = period;
        self.scheduler.cancel(crate::scheduler::EventSlot::VBlank);
        // Preserve current scanline phase across the region switch.
        // Redux's auto-video path changes the active video setting
        // from GP1 display-mode writes, but the counter phase keeps
        // marching; restarting from scanline 0 makes PAL games miss
        // the next VBlank by a large fraction of a frame.
        self.scheduler
            .schedule(crate::scheduler::EventSlot::VBlank, self.cycles, delay);
    }

    /// Current HSync period in cycles — NTSC = 2146, PAL = 2157.
    pub fn hsync_cycles(&self) -> u64 {
        self.hsync_cycles
    }

    /// Current VBlank period — one frame in cycles.
    pub fn vblank_period(&self) -> u64 {
        self.vblank_period
    }

    /// Turn on HLE BIOS interception. Call after side-loading an EXE
    /// that expects BIOS services to be live without running the real
    /// BIOS boot sequence. Never enable when validating parity — the
    /// oracle emulator runs the real BIOS ROM and will diverge.
    pub fn enable_hle_bios(&mut self) {
        self.hle_bios_enabled = true;
    }

    /// Plug a digital controller into port 1 so homebrew / commercial
    /// games can poll for button state. Convenience: most single-player
    /// games use port 1 only.
    pub fn attach_digital_pad_port1(&mut self) {
        let old = std::mem::replace(self.sio0.port1_mut(), crate::pad::PortDevice::empty());
        let memcard = old.into_memcard();
        let mut device = crate::pad::PortDevice::empty().with_pad(crate::pad::DigitalPad::new());
        if let Some(card) = memcard {
            device = device.with_memcard(card);
        }
        self.sio0.attach_port1(device);
    }

    /// Immutable access to SIO0 for diagnostics.
    pub fn sio0(&self) -> &Sio0 {
        &self.sio0
    }

    /// Plug a memory card into port 1 with the given backing
    /// contents (128 KiB buffer, typically loaded from a
    /// `.mcd` file). Pass an empty `Vec` to start with a fresh
    /// card. Keeps any pad already attached — real hardware
    /// multiplexes pad + memcard on the same port.
    pub fn attach_memcard_port1(&mut self, initial_bytes: Vec<u8>) {
        let card = if initial_bytes.len() == crate::pad::MEMCARD_SIZE {
            crate::pad::MemoryCard::from_bytes(initial_bytes)
        } else {
            crate::pad::MemoryCard::new()
        };
        // Preserve any existing pad.
        let pad =
            std::mem::replace(self.sio0.port1_mut(), crate::pad::PortDevice::empty()).into_pad();
        let mut device = crate::pad::PortDevice::empty().with_memcard(card);
        if let Some(p) = pad {
            device = device.with_pad(p);
        }
        self.sio0.attach_port1(device);
    }

    /// Snapshot the port-1 memcard bytes for persistence. `None`
    /// when there's no card on port 1 or the card hasn't been
    /// written since load.
    pub fn memcard_port1_snapshot(&mut self) -> Option<Vec<u8>> {
        let card = self.sio0.port1_mut().memcard_mut()?;
        if !card.is_dirty() {
            return None;
        }
        let bytes = card.as_bytes().to_vec();
        card.clear_dirty();
        Some(bytes)
    }

    /// Update the buttons currently held on the port-1 controller.
    /// Called by the frontend each frame from the keyboard state.
    pub fn set_port1_buttons(&mut self, buttons: crate::pad::ButtonState) {
        self.sio0.set_port1_buttons(buttons);
    }

    /// Update the analog-stick positions on the port-1
    /// controller. Each axis is `0..=255` with `0x80` = centre.
    /// No-op when no pad is attached to port 1. The stick values
    /// are only observed by games once the pad is in Analog mode
    /// (which they enter via the DualShock config protocol).
    pub fn set_port1_sticks(&mut self, right_x: u8, right_y: u8, left_x: u8, left_y: u8) {
        if let Some(pad) = self.sio0.port1_mut().pad_mut() {
            pad.set_sticks(right_x, right_y, left_x, left_y);
        }
    }

    /// Snapshot of the port-1 DualShock vibration-motor state:
    /// `(small_on, big_strength)` where `small_on` is a binary
    /// on/off and `big_strength` is 0..=255. Returns `(false, 0)`
    /// when no pad is attached to port 1. Frontend drives host
    /// haptics from this each frame.
    pub fn port1_motor_state(&self) -> (bool, u8) {
        self.sio0
            .port1()
            .pad()
            .map(|p| p.motor_state())
            .unwrap_or((false, 0))
    }

    /// Histogram of pad command bytes observed on port 1 since boot.
    /// `None` when no controller is attached.
    pub fn port1_pad_command_histogram(&self) -> Option<&[u32; 256]> {
        self.sio0.port1().pad().map(|p| p.command_histogram())
    }

    /// Recent pad command bytes seen on port 1, oldest first.
    pub fn port1_pad_recent_commands(&self) -> Vec<u8> {
        self.sio0
            .port1()
            .pad()
            .map(|p| p.recent_commands())
            .unwrap_or_default()
    }

    /// Histogram of transaction-leading bytes seen on SIO0 port 1.
    pub fn port1_first_byte_histogram(&self) -> &[u32; 256] {
        self.sio0.port1().first_byte_histogram()
    }

    /// Recent transaction-leading bytes seen on SIO0 port 1.
    pub fn port1_recent_first_bytes(&self) -> Vec<u8> {
        self.sio0.port1().recent_first_bytes()
    }

    /// Recent completed `0x42` poll transactions seen on port 1.
    pub fn port1_recent_polls(&self) -> Vec<crate::pad::PollSnapshot> {
        self.sio0
            .port1()
            .pad()
            .map(|p| p.recent_polls())
            .unwrap_or_default()
    }

    /// Plug a digital controller into port 2. Used by two-player
    /// games (Tekken VS, Bomberman, Street Fighter Alpha, etc.).
    /// SIO0 already multiplexes port 1 / port 2 internally via
    /// the CTRL.SLOT bit — games switch between them per poll.
    pub fn attach_digital_pad_port2(&mut self) {
        let old = std::mem::replace(self.sio0.port2_mut(), crate::pad::PortDevice::empty());
        let memcard = old.into_memcard();
        let mut device = crate::pad::PortDevice::empty().with_pad(crate::pad::DigitalPad::new());
        if let Some(card) = memcard {
            device = device.with_memcard(card);
        }
        self.sio0.attach_port2(device);
    }

    /// Plug a memory card into port 2. Same semantics as
    /// [`Bus::attach_memcard_port1`].
    pub fn attach_memcard_port2(&mut self, initial_bytes: Vec<u8>) {
        let card = if initial_bytes.len() == crate::pad::MEMCARD_SIZE {
            crate::pad::MemoryCard::from_bytes(initial_bytes)
        } else {
            crate::pad::MemoryCard::new()
        };
        let pad =
            std::mem::replace(self.sio0.port2_mut(), crate::pad::PortDevice::empty()).into_pad();
        let mut device = crate::pad::PortDevice::empty().with_memcard(card);
        if let Some(p) = pad {
            device = device.with_pad(p);
        }
        self.sio0.attach_port2(device);
    }

    /// Snapshot the port-2 memcard bytes for persistence. `None`
    /// when there's no card on port 2 or the card hasn't been
    /// written since load.
    pub fn memcard_port2_snapshot(&mut self) -> Option<Vec<u8>> {
        let card = self.sio0.port2_mut().memcard_mut()?;
        if !card.is_dirty() {
            return None;
        }
        let bytes = card.as_bytes().to_vec();
        card.clear_dirty();
        Some(bytes)
    }

    /// Update the buttons currently held on the port-2 controller.
    pub fn set_port2_buttons(&mut self, buttons: crate::pad::ButtonState) {
        self.sio0.set_port2_buttons(buttons);
    }

    /// Port-2 DualShock motor state. Mirrors
    /// [`Bus::port1_motor_state`].
    pub fn port2_motor_state(&self) -> (bool, u8) {
        self.sio0
            .port2()
            .pad()
            .map(|p| p.motor_state())
            .unwrap_or((false, 0))
    }

    /// Internal: log one HLE BIOS call. Called from the HLE dispatcher.
    pub(crate) fn hle_bios_log_call(&mut self, table: crate::hle_bios::Table, func: u8) {
        let idx = match table {
            crate::hle_bios::Table::A => 0,
            crate::hle_bios::Table::B => 1,
            crate::hle_bios::Table::C => 2,
        };
        self.hle_bios_calls[idx][func as usize] =
            self.hle_bios_calls[idx][func as usize].saturating_add(1);
    }

    /// Snapshot of HLE BIOS call counts: `[A, B, C]` tables × 256
    /// function slots. Diagnostic only.
    pub fn hle_bios_call_counts(&self) -> [[u32; 256]; 3] {
        self.hle_bios_calls
    }

    /// True when `phys` sits inside the MMIO window at `0x1F80_1000..0x1F80_2000`.
    /// Used to filter trace recording — RAM / BIOS fetches are out of scope.
    #[inline]
    fn is_mmio(phys: u32) -> bool {
        (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys)
    }

    /// Record an MMIO access in the ring buffer when tracing is enabled.
    /// Call sites stay cfg-free; the inner record() is a no-op otherwise.
    #[inline]
    fn trace_mmio(&mut self, kind: MmioKind, phys: u32, value: u32) {
        if Self::is_mmio(phys) {
            self.mmio_trace.record(self.cycles, kind, phys, value);
        }
    }

    /// Advance the SIO0 byte/ACK timers to the current bus cycle,
    /// forward any newly latched controller IRQ to `I_STAT`, and
    /// (re)schedule [`EventSlot::Sio`] for whatever deadline is
    /// next pending. With the scheduler firing the wake-up, the
    /// per-instruction poll is no longer needed — `Bus::tick`
    /// dropped its `service_sio0` call. Read paths still call
    /// this synchronously so a load that happens between the
    /// branch tests sees a deadline that's already due.
    fn service_sio0(&mut self) {
        self.sio0.tick(self.cycles);
        if self.sio0.take_pending_irq() {
            self.irq.raise(IrqSource::Controller);
        }
        self.reschedule_sio0_event();
    }

    /// (Re)plant the [`EventSlot::Sio`] entry on the scheduler so
    /// the next deadline (transfer / ack / ack-end) wakes us up
    /// without us polling every instruction. Cancels any prior
    /// pending Sio event when SIO0 has gone idle.
    fn reschedule_sio0_event(&mut self) {
        if let Some(deadline) = self.sio0.next_deadline() {
            let delta = deadline.saturating_sub(self.cycles);
            self.scheduler
                .schedule(crate::scheduler::EventSlot::Sio, self.cycles, delta);
        } else {
            self.scheduler.cancel(crate::scheduler::EventSlot::Sio);
        }
    }

    /// Cycle count at which the next VBlank is scheduled to fire.
    /// Exposed for diagnostics / the HUD. Reads through to the
    /// scheduler, which is the source of truth since Phase 5a.
    pub fn next_vblank_cycle(&self) -> u64 {
        self.scheduler
            .target(crate::scheduler::EventSlot::VBlank)
            .unwrap_or(u64::MAX)
    }

    /// Advance the cycle counter by `n` cycles and run any scheduled
    /// peripheral events that have come due. Called once per
    /// instruction from `Cpu::step` (charging BIAS before the opcode).
    ///
    /// CDROM is deliberately NOT processed here — only at
    /// `drain_scheduler_events_post_op`, which the CPU calls at
    /// branch-delay-slot boundaries. That matches Redux's
    /// `branchTest` timing (psxinterpreter.cc:1650) where CDROM
    /// `interrupt()` fires at end-of-delay-slot, not on every
    /// instruction. Ticking CDROM on every BIAS makes our ACK
    /// land one or two instructions earlier than Redux's — long
    /// enough for a CDROM-polling spin-wait (e.g. MGS's BIOS
    /// ReadTOC-Ack wait at step ~90M) to see a different register
    /// byte than Redux and exit the loop early.
    pub fn tick(&mut self, n: u32) {
        self.advance_cycles(n);
        self.drain_scheduler_events_without_cdr_dma();
        // SIO0 used to be polled here (every instruction).
        // It's now woken up by `EventSlot::Sio` from the scheduler
        // — see `drain_scheduler_events_inner`. Read paths still
        // call `service_sio0` synchronously so MMIO loads observe
        // any deadline that's already due.
    }

    /// Public entry point that the CPU calls between the opcode's
    /// `add_cycles` (memory-access charges) and the delay-slot IRQ
    /// check. Ensures any scheduler event whose target was crossed
    /// DURING the opcode (not just at the BIAS tick that starts the
    /// instruction) raises its IRQ bit in time for the same step's
    /// exception dispatch. Redux achieves the same effect via
    /// `branchTest` → `counters->update()`.
    pub fn drain_scheduler_events_post_op(&mut self) {
        // Advance timer state to `now` once per branch boundary so
        // any IRQ that would have fired between the last branchTest
        // and this one lands in `I_STAT` in time for the same
        // step's exception dispatch. Per-instruction `Bus::tick`
        // doesn't touch timers anymore; this is the only path that
        // matters for IRQ visibility. Mirrors Redux's
        // `Counters::update` call at the top of `branchTest`.
        self.service_timers();
        self.drain_scheduler_events();
        let cdrom_irq_pending =
            self.irq.stat() & self.irq.mask() & (1 << (IrqSource::Cdrom as u32)) != 0;
        if self
            .cdrom
            .tick_with_irq_pending(self.cycles, cdrom_irq_pending)
            && self.cdrom.should_wake_cpu()
        {
            self.irq.raise(IrqSource::Cdrom);
        }
        // SIO0 wake-up comes from the scheduler dispatch above
        // (`EventSlot::Sio` in `drain_scheduler_events_inner`),
        // not from a separate poll. The `take_due` walk is
        // strict-greater-than, so events that were due as of
        // `now` will fire next branch test; SIO0's parity
        // tolerance is well within that window.
    }

    /// Walk every scheduler slot whose deadline has passed and
    /// dispatch its handler. Mirrors Redux's `branchTest` interrupt
    /// loop (`core/r3000a.cc`), which uses a single 15-slot queue
    /// to drive DMA / CDROM / SPU / MDEC / SIO completions.
    ///
    /// DMA channel completions all funnel through the shared
    /// `Dma` IRQ line: each per-channel slot clears CHCR bit 24
    /// and records a master-edge if the channel's DICR bit is
    /// armed. One IRQ raise covers any number of simultaneous
    /// completions this tick.
    ///
    /// Slots we haven't migrated yet (CDROM, VBlank, SPU, SIO,
    /// MDEC) will never appear here because no subsystem schedules
    /// them — the legacy timers still own those. Each migration
    /// replaces a legacy timer with `scheduler.schedule(...)` and
    /// adds a `match` arm here.
    fn drain_scheduler_events_without_cdr_dma(&mut self) {
        self.drain_scheduler_events_inner(false);
    }

    fn drain_scheduler_events(&mut self) {
        self.drain_scheduler_events_inner(true);
    }

    fn drain_scheduler_events_inner(&mut self, include_cdr_dma: bool) {
        use crate::scheduler::EventSlot;
        let now = self.cycles;
        let mut dma_edge = false;
        // NOTE: `service_timers()` is intentionally NOT called here.
        // This function runs from the per-instruction `Bus::tick`
        // path (via `drain_scheduler_events_without_cdr_dma`); the
        // whole point of the lazy refactor is to avoid touching
        // timer state on every instruction. The branch-boundary
        // drain (`drain_scheduler_events_post_op`) advances timers
        // before doing anything else, which is enough to keep
        // timer IRQs firing at parity-relevant cycles.

        // Redux updates root counters with `cycle >= nextCounter`
        // before walking the strict interrupt-slot queue. VBlank is
        // our root-counter-style event, so handle it inclusively here
        // instead of via `take_due`'s generic `target < now` rule.
        while let Some(target) = self
            .scheduler
            .take_slot_due_inclusive(EventSlot::VBlank, now)
        {
            self.irq.raise(IrqSource::VBlank);
            self.gpu.toggle_vblank_field();
            self.timers.notify_vblank();
            self.scheduler
                .schedule(EventSlot::VBlank, target, self.vblank_period);
        }

        // SIO0 also fires inclusively: the legacy polling path used
        // `deadline <= now`, so we keep that semantic to avoid an
        // off-by-one parity drift on controller IRQ timing. Each
        // service may set a new deadline (transfer → ack → ack-end);
        // `service_sio0` reschedules accordingly via
        // `reschedule_sio0_event`, so the loop drains chained
        // transitions in one pass.
        while self
            .scheduler
            .take_slot_due_inclusive(EventSlot::Sio, now)
            .is_some()
        {
            self.service_sio0();
            // If `service_sio0` chained a follow-up deadline that's
            // also already due (e.g. transfer → ack within one
            // dispatch), `take_slot_due_inclusive` picks it up on
            // the next iteration. Otherwise the loop exits and the
            // future deadline waits for its scheduled wake-up.
        }

        let cdr_dma_mask = 1 << EventSlot::CdrDma.bit();
        while let Some((slot, target)) = if include_cdr_dma {
            self.scheduler.take_due(now)
        } else {
            self.scheduler.take_due_excluding(now, cdr_dma_mask)
        } {
            match slot {
                EventSlot::MdecInDma => {
                    if self.complete_dma_channel(0) {
                        dma_edge = true;
                    }
                }
                EventSlot::MdecOutDma => {
                    if self.mdec.complete_dma_out() && self.complete_dma_channel(0) {
                        dma_edge = true;
                    }
                    if self.complete_dma_channel(1) {
                        dma_edge = true;
                    }
                }
                EventSlot::GpuDma => {
                    if self.complete_dma_channel(2) {
                        dma_edge = true;
                    }
                }
                EventSlot::CdrDma => {
                    if self.complete_dma_channel(3) {
                        dma_edge = true;
                    }
                }
                EventSlot::SpuDma => {
                    if self.complete_dma_channel(4) {
                        dma_edge = true;
                    }
                }
                EventSlot::GpuOtcDma => {
                    if self.complete_dma_channel(6) {
                        dma_edge = true;
                    }
                }
                EventSlot::VBlank => {
                    self.irq.raise(IrqSource::VBlank);
                    // Toggle GPUSTAT bit 31 (interlace / field flag)
                    // — some BIOS and game code polls this instead
                    // of (or in addition to) the VBlank IRQ to detect
                    // frame boundaries. Matches Redux's
                    // `SoftGPU::vblank` which XORs the same bit.
                    self.gpu.toggle_vblank_field();
                    // Tell the timer bank — Timer 1 sync-mode-1
                    // resets its counter on this pulse.
                    self.timers.notify_vblank();
                    // Re-arm the next VBlank from the original
                    // target, not `now`. A 500K-cycle drain lag
                    // would otherwise accumulate drift every time.
                    self.scheduler
                        .schedule(EventSlot::VBlank, target, self.vblank_period);
                }
                EventSlot::SpuAsync => {
                    // Kept for forward compatibility; we pump the SPU
                    // from the frontend instead of the scheduler while
                    // the parity oracle runs with a dormant SPU
                    // thread. If anything schedules this slot it is
                    // a logic bug — log and drop.
                    debug_assert!(false, "SpuAsync fired but SPU pumps from frontend");
                }
                // Not-yet-migrated slots. A subsystem scheduling one
                // of these today would silently do nothing; they're
                // listed so `match` stays exhaustive as migrations
                // roll in.
                EventSlot::Sio
                | EventSlot::Sio1
                | EventSlot::Cdr
                | EventSlot::CdRead
                | EventSlot::CdrPlay
                | EventSlot::CdrDbuf
                | EventSlot::CdrLid => {}
            }
        }
        // CDROM DMA completion is observed by Redux at the exact
        // target boundary in retail boot paths (MGS license-sector
        // DMA lands here). Keep the generic scheduler strict for the
        // other interrupt slots, but let CDR DMA finish on equality.
        if include_cdr_dma
            && self
                .scheduler
                .take_slot_due_inclusive(EventSlot::CdrDma, now)
                .is_some()
            && self.complete_dma_channel(3)
        {
            dma_edge = true;
        }
        if dma_edge {
            self.irq.raise(IrqSource::Dma);
        }
    }

    /// Finalise a DMA channel's transfer: clear the start bit in
    /// CHCR and notify the DMA controller so it updates DICR and
    /// returns whether this channel's IRQ-enable bit caused the
    /// shared `IrqSource::Dma` line to transition high. Caller
    /// raises that IRQ once per tick if any channel was on the
    /// edge.
    fn complete_dma_channel(&mut self, ch: usize) -> bool {
        self.dma.channels[ch].channel_control &= !(1 << 24);
        self.dma.notify_channel_done(ch)
    }

    /// Advance the cycle counter without running peripheral schedulers.
    /// Used by load/store opcodes to charge the per-data-access cycle
    /// (Redux's `m_regs.cycle += 1` inside `read8/16/32` and
    /// `write8/16/32` in `psxmem.cc`). VBlank / DMA6 / CDROM schedulers
    /// still see the accumulated cycle count when `tick()` runs at end
    /// of instruction — matching Redux's `psxBranchTest`, which only
    /// runs after delay slots and observes the post-BIAS,
    /// post-data-access total. Timers, however, see every cycle so
    /// their counter values stay in lock-step with Redux's cycle-derived
    /// `count = (now - cycle_start) / rate` model.
    pub fn add_cycles(&mut self, n: u32) {
        self.advance_cycles(n);
    }

    /// Inner cycle-advancement helper shared by `tick` and `add_cycles`.
    /// Any cycle delta must flow through this function so the timer
    /// bank's accumulator matches Redux's lazy-read timer model.
    fn advance_cycles(&mut self, n: u32) {
        self.cycles = self.cycles.wrapping_add(n as u64);
        // Timers used to be ticked here every instruction (~25M
        // calls/sec, three accumulator-divides each). They're now
        // advanced lazily — `service_timers()` runs once per
        // scheduler drain and on demand from MMIO read / write
        // paths. The lazy advance reads `self.cycles` so it
        // observes the same effective time the per-tick path did.
        // Decay the GPU's pseudo-busy credit as time passes. At
        // 32 units per cycle a full-screen (~640×478) fill-rect
        // credit of 153k takes ~4.8k cycles to drain — about the
        // duration of a short VBlank window, matching real
        // hardware's "a few scanlines to finish the batch".
        self.gpu.decay_busy((n as u64) * 32);
    }

    /// Advance the timer bank to the current bus cycle and forward
    /// any newly-latched timer IRQs to `I_STAT`. Cheap when nothing
    /// changed (just a `last_advance` saturating-sub returning 0);
    /// the work happens only in proportion to the cycles elapsed
    /// since the last call. Read / write paths call this before
    /// observing timer state; the scheduler drain calls it once
    /// per branch-test boundary so IRQs fire on time.
    fn service_timers(&mut self) {
        let fired =
            self.timers
                .advance_to(self.cycles, self.hsync_cycles, self.gpu.dot_clock_divisor());
        if fired & 1 != 0 {
            self.irq.raise(IrqSource::Timer0);
        }
        if fired & 2 != 0 {
            self.irq.raise(IrqSource::Timer1);
        }
        if fired & 4 != 0 {
            self.irq.raise(IrqSource::Timer2);
        }
    }

    /// Cumulative cycles since reset.
    pub fn cycles(&self) -> u64 {
        self.cycles
    }

    /// Record `addr` as seen on an unmapped *read*, returning
    /// `true` the first time so the caller logs once.
    fn log_unmapped_read_once(&mut self, addr: u32) -> bool {
        self.unmapped_read_seen.insert(addr)
    }

    /// Record `addr` as seen on an unmapped *write*, returning
    /// `true` the first time so the caller logs once.
    fn log_unmapped_write_once(&mut self, addr: u32) -> bool {
        self.unmapped_write_seen.insert(addr)
    }

    /// Drive the VBlank + DMA + (eventually) CDROM / SPU event
    /// loops once. Provided as a public entry point so tests can
    /// advance peripheral state directly without stepping the CPU.
    /// Production callers hit it transitively via
    /// [`Bus::tick`] → [`Bus::drain_scheduler_events`].
    pub fn run_vblank_scheduler(&mut self) {
        self.drain_scheduler_events();
    }

    /// Pump the SPU forward by `n` samples. Called by the frontend
    /// once per displayed frame (or when the audio callback drains
    /// the output ring), producing stereo output for playback.
    ///
    /// We decouple SPU timing from the CPU's cycle counter to stay
    /// bit-exact with the Redux parity oracle: Redux's SPU runs in
    /// a background thread that doesn't tick during the trace, so
    /// ticking on every 768th cycle diverges. Callers supply a
    /// sample count that matches their display refresh cadence
    /// (e.g. 735 samples per NTSC frame at 44.1 kHz).
    ///
    /// Also forwards any CD audio samples the CDROM has decoded
    /// (CD-DA / XA ADPCM) into the SPU's CD input mix — one
    /// drain-and-feed per call keeps the latency bounded.
    pub fn run_spu_samples(&mut self, n: usize) {
        // Move any freshly-decoded CDROM audio into the SPU's CD
        // input queue so it participates in this frame's mix.
        let cd_samples = self.cdrom.drain_cd_audio();
        if !cd_samples.is_empty() {
            self.spu.feed_cd_audio(&cd_samples);
        }
        for _ in 0..n {
            self.spu.tick_sample(self.cycles);
            if self.spu.take_irq_pending() {
                self.irq.raise(IrqSource::Spu);
            }
        }
    }

    /// Borrow the interrupt controller — caller can `.raise()` sources
    /// or inspect state without going through MMIO.
    pub fn irq_mut(&mut self) -> &mut Irq {
        &mut self.irq
    }

    /// Diagnostic: per-channel count of CHCR writes with the start
    /// bit set since reset. Index 0..=6 corresponds to MDEC-in,
    /// MDEC-out, GPU, CD-ROM, SPU, PIO, OTC.
    pub fn dma_start_triggers(&self) -> [u64; 7] {
        self.dma.start_trigger_counts
    }

    /// True when some source is both pending in `I_STAT` and enabled
    /// in `I_MASK`. The CPU mirrors this into `COP0.CAUSE.IP[2]`.
    pub fn external_interrupt_pending(&mut self) -> bool {
        self.irq.pending_tick()
    }

    /// Borrow the IRQ controller immutably for diagnostics.
    pub fn irq(&self) -> &Irq {
        &self.irq
    }

    /// Borrow the timer bank immutably for diagnostics.
    pub fn timers(&self) -> &Timers {
        &self.timers
    }

    /// Copy a PSX-EXE payload into RAM at its declared load address.
    ///
    /// The caller is expected to also seed the CPU (see
    /// [`crate::Cpu::seed_from_exe`]) so execution begins at the
    /// EXE's entry point. `load_addr` must point inside the 2 MiB
    /// RAM window; addresses outside panic.
    ///
    /// Used by `PSOXIDE_EXE` side-loading in the frontend / smoke
    /// harness to bypass the BIOS entirely and run homebrew directly.
    pub fn load_exe_payload(&mut self, load_addr: u32, payload: &[u8]) {
        let base = load_addr & 0x001F_FFFF; // KSEG/KUSEG -> physical RAM
        assert!(
            (base as usize) + payload.len() <= self.ram.len(),
            "EXE payload overflows RAM: load_addr={load_addr:#010x} len={}",
            payload.len()
        );
        self.ram[base as usize..base as usize + payload.len()].copy_from_slice(payload);
    }

    /// Zero the optional BSS range declared by a PSX-EXE header.
    ///
    /// The BIOS clears this area before jumping to the executable;
    /// side-load and fast-boot paths need to do the same because they
    /// bypass the BIOS loader.
    pub fn clear_exe_bss(&mut self, bss_addr: u32, bss_size: u32) {
        if bss_size == 0 {
            return;
        }
        let base = bss_addr & 0x001F_FFFF; // KSEG/KUSEG -> physical RAM
        let size = bss_size as usize;
        assert!(
            (base as usize) + size <= self.ram.len(),
            "EXE BSS overflows RAM: bss_addr={bss_addr:#010x} len={size}",
        );
        self.ram[base as usize..base as usize + size].fill(0);
    }

    /// Zero an address range in main RAM after KSEG/KUSEG address
    /// normalization. The end address is exclusive.
    pub fn clear_ram_range(&mut self, start_addr: u32, end_addr: u32) {
        let start = (start_addr & 0x001F_FFFF) as usize;
        let end = (end_addr & 0x001F_FFFF) as usize;
        if end <= start {
            return;
        }
        assert!(
            end <= self.ram.len(),
            "RAM clear range overflows: start={start_addr:#010x} end={end_addr:#010x}",
        );
        self.ram[start..end].fill(0);
    }

    /// Run DMA on a single channel after its CHCR was just written
    /// with the start bit set. Mirrors Redux's per-channel
    /// `dmaExec<N>` dispatch in `psxhw.cc` — each CHCR write goes
    /// to exactly one channel's handler, NOT a sweep across every
    /// channel. That distinction matters: if another channel's
    /// transfer was still in-flight (start bit set, awaiting its
    /// scheduled completion), a sweep re-runs it and schedules a
    /// second target that overwrites the first.
    fn run_dma_channel(&mut self, ch: usize) {
        // Each channel: run the transfer now (so memory / GPU state is
        // up-to-date for any immediate follow-up reads), but defer the
        // CHCR start-bit clear and DMA IRQ raise to the channel's
        // scheduled completion cycle. Redux schedules one cycle per
        // word transferred (`scheduleGPUOTCDMAIRQ(size)`, etc.), which
        // keeps the BIOS's "poll CHCR until done" loop matching our
        // trace step-for-step. An immediate IRQ raise triggers the
        // handler ~1 hblank early and diverges the trace by dozens of
        // instructions.
        use crate::scheduler::EventSlot;
        // Run only the channel whose CHCR was just written.
        match ch {
            0 => {
                if let Some(mdec_words) = self.run_dma_mdec_in() {
                    if self.mdec.decode_dma0_waits_for_output() {
                        self.try_schedule_ready_mdec_out();
                    } else {
                        let target = self.cycles + mdec_words as u64;
                        self.log_dma_schedule("MdecIn", mdec_words as u64, target);
                        self.scheduler.schedule(
                            EventSlot::MdecInDma,
                            self.cycles,
                            mdec_words as u64,
                        );
                    }
                }
            }
            1 => {
                self.try_schedule_ready_mdec_out();
            }
            2 => {
                if let Some(gpu_cycles) = self.run_dma_gpu() {
                    let target = self.cycles + gpu_cycles as u64;
                    self.log_dma_schedule("GpuDma", gpu_cycles as u64, target);
                    self.scheduler
                        .schedule(EventSlot::GpuDma, self.cycles, gpu_cycles as u64);
                }
            }
            3 => {
                let ch = self.dma.channels[3];
                let fifo_len = self.cdrom.data_fifo_len();
                let armed = self.cdrom.data_transfer_armed();
                if let Some(cdrom_words) = self.run_dma_cdrom() {
                    let label = format!(
                        "CdrDma words={cdrom_words} fifo={fifo_len} armed={} madr=0x{:08x} bcr=0x{:08x} chcr=0x{:08x}",
                        armed as u8, ch.base, ch.block_control, ch.channel_control
                    );
                    if cdrom_words == 0 {
                        self.log_dma_schedule(&label, 0, self.cycles);
                        if self.complete_dma_channel(3) {
                            self.irq.raise(IrqSource::Dma);
                        }
                    } else {
                        let delay = match self.dma.channels[3].channel_control {
                            0x1140_0100 => (cdrom_words / 4).max(1) as u64,
                            _ => cdrom_words as u64,
                        };
                        let target = self.cycles + delay;
                        self.log_dma_schedule(&label, delay, target);
                        self.scheduler
                            .schedule(EventSlot::CdrDma, self.cycles, delay);
                    }
                }
            }
            4 => {
                if let Some(spu_words) = self.run_dma_spu() {
                    let target = self.cycles + spu_words as u64;
                    self.log_dma_schedule("SpuDma", spu_words as u64, target);
                    self.scheduler
                        .schedule(EventSlot::SpuDma, self.cycles, spu_words as u64);
                }
            }
            6 => {
                let otc_words = if self.dma.is_channel_enabled(6) {
                    self.dma.run_otc(&mut self.ram[..])
                } else {
                    0
                };
                if otc_words > 0 {
                    let target = self.cycles + otc_words as u64;
                    self.log_dma_schedule("GpuOtc", otc_words as u64, target);
                    self.scheduler
                        .schedule(EventSlot::GpuOtcDma, self.cycles, otc_words as u64);
                }
            }
            _ => {
                // Channel 5 (PIO) + invalid indices — skip silently.
                // Matches Redux's `#if 0` guard that disables PIO DMA.
            }
        }
    }

    /// Optional per-DMA-schedule log. Off by default; the
    /// `probe_dma_schedules` example enables it via the setter to
    /// capture every DMA completion's `(cycle_now, delta, target)`
    /// for cycle-parity diagnosis. Stored on the bus so the probe
    /// can drain it after a run without poking CPU-execution paths.
    pub fn set_dma_log_enabled(&mut self, enabled: bool) {
        self.dma_log_enabled = enabled;
        if enabled {
            self.dma_log.clear();
        }
    }

    /// Drain collected DMA schedule events.
    pub fn drain_dma_log(&mut self) -> Vec<(String, u64, u64, u64)> {
        std::mem::take(&mut self.dma_log)
    }

    fn log_dma_schedule(&mut self, kind: &str, delta: u64, target: u64) {
        if self.dma_log_enabled {
            self.dma_log
                .push((kind.to_string(), self.cycles, delta, target));
        }
    }

    fn try_schedule_ready_mdec_out(&mut self) {
        use crate::scheduler::EventSlot;

        if self.scheduler.is_pending(EventSlot::MdecOutDma) || !self.mdec.can_dma_out() {
            return;
        }
        if let Some(mdec_words) = self.run_dma_mdec_out() {
            // Redux's MDEC model schedules output DMA by byte count
            // multiplied by MDEC_BIAS=2.0, i.e. 8 cycles per 32-bit word.
            let delay = mdec_words as u64 * 8;
            let target = self.cycles + delay;
            self.log_dma_schedule("MdecOut", delay, target);
            self.scheduler
                .schedule(EventSlot::MdecOutDma, self.cycles, delay);
        }
    }

    /// Execute DMA channel 0 → MDEC input. Ships command + RLE data
    /// from main RAM to the MDEC's input queue. Sync mode 1 (block)
    /// is the only mode PS1 software uses for this channel.
    fn run_dma_mdec_in(&mut self) -> Option<u32> {
        if !self.dma.is_channel_enabled(0) {
            return None;
        }
        let ch = &self.dma.channels[0];
        if (ch.channel_control >> 24) & 1 == 0 {
            return None;
        }
        let total_words = mdec_dma_word_count(ch.block_control, ch.channel_control);
        let step: u32 = if (ch.channel_control >> 1) & 1 != 0 {
            0xFFFF_FFFCu32
        } else {
            4
        };
        let mut addr = ch.base & 0x001F_FFFC;
        let mut words: Vec<u32> = Vec::with_capacity(total_words as usize);
        for _ in 0..total_words {
            words.push(read_ram_u32(&self.ram[..], addr));
            addr = addr.wrapping_add(step);
        }
        self.mdec.dma_write_in(&words);
        Some(total_words)
    }

    /// Execute DMA channel 1 → main RAM from MDEC output. Pulls
    /// decoded pixel words from the MDEC's output queue and writes
    /// them to main RAM at `MADR`.
    fn run_dma_mdec_out(&mut self) -> Option<u32> {
        if !self.dma.is_channel_enabled(1) {
            return None;
        }
        if !self.mdec.can_dma_out() {
            return None;
        }
        let ch = &self.dma.channels[1];
        if (ch.channel_control >> 24) & 1 == 0 {
            return None;
        }
        let total_words = mdec_dma_word_count(ch.block_control, ch.channel_control);
        let step: u32 = if (ch.channel_control >> 1) & 1 != 0 {
            0xFFFF_FFFCu32
        } else {
            4
        };
        let mut addr = ch.base & 0x001F_FFFC;
        let mut words = vec![0u32; total_words as usize];
        self.mdec.dma_read_out(&mut words);
        for word in words {
            let offset = (addr & 0x001F_FFFF) as usize;
            if offset + 4 <= self.ram.len() {
                self.ram[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            addr = addr.wrapping_add(step);
        }
        Some(total_words)
    }

    /// Execute DMA channel 4 ↔ SPU. Sync mode 1 (block) is the only
    /// mode games use; mode 0 (manual) falls through as a single-block
    /// transfer with BCR as the total word count (matches Redux). The
    /// SPU's transfer pointer tracks the current RAM-side address; we
    /// copy halfword-pairs in the direction the channel selects.
    ///
    /// - Direction bit 0 = 1: main RAM → SPU RAM (normal — upload sample data).
    /// - Direction bit 0 = 0: SPU RAM → main RAM (rare — live capture).
    ///
    /// CHCR start/busy bits are NOT cleared here; the completion
    /// handler in `drain_scheduler_events` does that at the scheduled
    /// cycle (one tick per word transferred, matching Redux's
    /// `scheduleSPUDMAIRQ(size)`).
    fn run_dma_spu(&mut self) -> Option<u32> {
        if !self.dma.is_channel_enabled(4) {
            return None;
        }
        let ch = &self.dma.channels[4];
        if (ch.channel_control >> 24) & 1 == 0 {
            return None;
        }
        // SPUCNT must be in a DMA transfer mode (write or read) for
        // the transfer to land; otherwise the channel is armed but
        // the SPU doesn't accept words. Games always program SPUCNT
        // before kicking the channel, so this is a belt-and-braces
        // check.
        if !self.spu.dma_transfer_enabled() {
            // Still return Some so the completion IRQ fires — the
            // CHCR start bit must clear or the BIOS's DMA-wait loop
            // hangs forever.
            return Some(0);
        }
        let sync_mode = (ch.channel_control >> 9) & 0x3;
        let bcr = ch.block_control;
        let total_words: u32 = match sync_mode {
            0 => bcr & 0xFFFF,
            1 => {
                let block_size = bcr & 0xFFFF;
                let block_count = (bcr >> 16).max(1) & 0xFFFF;
                block_size * block_count
            }
            _ => 0, // Linked list + reserved — not used for SPU.
        };
        let to_spu = ch.channel_control & 1 != 0;
        let step: u32 = if (ch.channel_control >> 1) & 1 != 0 {
            0xFFFF_FFFCu32
        } else {
            4
        };
        let mut addr = ch.base & 0x001F_FFFC;
        if to_spu {
            // Buffer the RAM words so we can ship them to the SPU's
            // `dma_write` which takes a slice of halfwords. Each 32-bit
            // word becomes two halfwords, low then high.
            let mut words: Vec<u16> = Vec::with_capacity((total_words * 2) as usize);
            for _ in 0..total_words {
                let word = read_ram_u32(&self.ram[..], addr);
                words.push(word as u16);
                words.push((word >> 16) as u16);
                addr = addr.wrapping_add(step);
            }
            self.spu.dma_write(&words);
        } else {
            let mut words = vec![0u16; (total_words * 2) as usize];
            self.spu.dma_read(&mut words);
            for i in 0..total_words {
                let lo = words[(i as usize) * 2] as u32;
                let hi = words[(i as usize) * 2 + 1] as u32;
                let word = lo | (hi << 16);
                let offset = (addr & 0x001F_FFFF) as usize;
                if offset + 4 <= self.ram.len() {
                    self.ram[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
                }
                addr = addr.wrapping_add(step);
            }
        }
        Some(total_words)
    }

    /// Execute DMA channel 3 → CPU. Block mode (sync=1) is the only
    /// mode used for CD-ROM reads: pull `BS × BA` words from the data
    /// FIFO and write them to RAM at `MADR` with +4 step. Returns
    /// `Some(word_count)` when a transfer was kicked (so the caller
    /// can schedule the completion IRQ), `None` when the channel
    /// wasn't armed. CHCR start/busy bits are NOT cleared here — the
    /// per-channel scheduler does that at the completion cycle.
    fn run_dma_cdrom(&mut self) -> Option<u32> {
        if !self.dma.is_channel_enabled(3) {
            return None;
        }
        let ch = self.dma.channels[3];
        if (ch.channel_control >> 24) & 1 == 0 {
            return None;
        }
        // Redux rejects DMA3 kicks only until a sector is ready in
        // the transfer buffer (`m_read == 0`). It does not require
        // the request-register bit that gates MMIO data reads; the
        // BIOS kicks DMA before that latch is armed in Tekken 3's
        // CDROM handler and expects CHCR bit 24 to remain busy for
        // the scheduled DMA window.
        if self.cdrom.data_fifo_len() == 0 {
            return Some(0);
        }
        let sync_mode = (ch.channel_control >> 9) & 0x3;
        // PSX BIOS + most games use sync mode 1 (block request) for
        // CDROM reads, but some firmware paths use sync mode 0
        // (manual / immediate). They differ only in how BCR is
        // interpreted:
        //
        //   mode 0 (manual): BCR is the total number of words to
        //                    transfer. BA is ignored.
        //   mode 1 (block):  BCR is (BA << 16) | BS — transfer BS
        //                    words per request, BA times.
        //
        // Both result in the same byte flow from the CDROM data
        // FIFO to RAM; computing `total_words` from the right BCR
        // encoding is what matters. Earlier we short-circuited
        // sync_mode!=1 to Some(0), which silently dropped every
        // BIOS disc read: the FIFO still filled (we saw LBA 16 in
        // it from cdrom_drive_test) but its bytes never landed in
        // RAM, and the BIOS's PVD parse fell back to reading
        // LBA 0 on empty input.
        let bcr = ch.block_control;
        let requested_words = match sync_mode {
            0 => bcr & 0xFFFF,
            1 => {
                let block_size = bcr & 0xFFFF;
                let block_count = (bcr >> 16) & 0xFFFF;
                block_size * block_count.max(1)
            }
            _ => {
                // Linked-list (2) + reserved (3) — not used for
                // CDROM. Drop the trigger silently.
                return Some(0);
            }
        };
        // Redux falls back to the active sector size when BCR asks
        // for zero words (for example Ape Escape programs `0001/0000`
        // and expects a full 2048-byte transfer). Our FIFO already
        // holds the exact transfer payload, so derive the word count
        // from its live length.
        let total_words = if requested_words == 0 {
            self.cdrom.data_fifo_words()
        } else {
            requested_words
        };
        let mut addr = ch.base & 0x001F_FFFC;
        let step: u32 = if (ch.channel_control >> 1) & 1 != 0 {
            0xFFFF_FFFCu32
        } else {
            4
        };

        for _ in 0..total_words {
            let b0 = self.cdrom.pop_dma_data_byte() as u32;
            let b1 = self.cdrom.pop_dma_data_byte() as u32;
            let b2 = self.cdrom.pop_dma_data_byte() as u32;
            let b3 = self.cdrom.pop_dma_data_byte() as u32;
            let word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            let offset = (addr & 0x001F_FFFF) as usize;
            if offset + 4 <= self.ram.len() {
                self.ram[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            }
            addr = addr.wrapping_add(step);
        }
        Some(total_words)
    }

    /// Execute DMA channel 2 → GPU (GP0). Supports the two sync modes
    /// the BIOS + games actually use for this channel:
    ///
    /// - **Mode 1 (block)**: Ship `BS × BA` words starting at
    ///   `MADR` straight into GP0, with the `MADR` step direction
    ///   given by CHCR bit 1 (+4 or -4). PS1 always uses +4 direction
    ///   for CPU→GPU.
    /// - **Mode 2 (linked list)**: Walk a chain of packets in RAM.
    ///   Each node header is `[NN AAAAAA]` — top byte = word count
    ///   (following 32-bit words to ship to GP0), low 24 bits = next
    ///   node address. Terminator is `AAAAAA == 0xFFFFFF`.
    ///
    /// Returns `Some(completion_cycles)` when a transfer was kicked
    /// (caller uses it to schedule the GpuDma event), `None` when
    /// the channel wasn't armed. CHCR start/busy bits are NOT cleared
    /// here — the scheduler does that at the completion cycle.
    ///
    /// `completion_cycles` is the Redux-accurate event delay, which
    /// depends on sync mode and transfer direction:
    ///
    /// - **Block, mem2vram** (RAM→GPU, bit 0 = 1): `7 * block_count`,
    ///   per Redux `core/gpu.cc` L551: `scheduleGPUDMAIRQ((7 * size)
    ///   / bs)` = `7 * block_count`. That's the active path in
    ///   modern Redux (the `#if 0` branch above it uses `size`).
    /// - **Block, vram2mem** (GPU→RAM, bit 0 = 0): `total_words`,
    ///   per Redux L523: `scheduleGPUDMAIRQ(size)`.
    /// - **Linked list**: `total_words`, per Redux L568:
    ///   `scheduleGPUDMAIRQ(size)` where size is the
    ///   `gpuDmaChainSize` traversed count.
    fn run_dma_gpu(&mut self) -> Option<u32> {
        if !self.dma.is_channel_enabled(2) {
            return None;
        }
        let ch = &self.dma.channels[2];
        if (ch.channel_control >> 24) & 1 == 0 {
            return None;
        }
        let sync_mode = (ch.channel_control >> 9) & 0x3;
        let direction_to_device = ch.channel_control & 1 != 0;
        let completion = match sync_mode {
            1 => self.dma_gpu_block(direction_to_device),
            2 => self.dma_gpu_linked_list(),
            _ => 0, // Manual (0) + prohibited (3) — nothing standard uses
                    // them for the GPU channel. Drop the trigger silently.
        };
        // Start bit stays set until the scheduled completion event
        // fires — Redux's `gpuInterrupt` is where `clearDMABusy<2>()`
        // is called. BIOS polling of CHCR bit 24 during the transfer
        // window must read 1 until the IRQ fires.
        Some(completion)
    }

    fn dma_gpu_block(&mut self, to_device: bool) -> u32 {
        let ch = self.dma.channels[2];
        let mut addr = ch.base & 0x001F_FFFC;
        let bcr = ch.block_control;
        let block_size = bcr & 0xFFFF;
        let block_count = (bcr >> 16).max(1) & 0xFFFF;
        let total_words = block_size * block_count;
        let step = if (ch.channel_control >> 1) & 1 != 0 {
            // Decrement mode — rarely used for GPU but handle for safety.
            0xFFFF_FFFCu32
        } else {
            4
        };
        for _ in 0..total_words {
            let word = read_ram_u32(&self.ram[..], addr);
            self.gpu.gp0_push(word);
            addr = addr.wrapping_add(step);
        }
        // Redux's formulas: mem2vram uses `7 * block_count` (the
        // "X-Files video interlace. Experimental delay depending
        // of BS" active path in `core/gpu.cc`); vram2mem uses the
        // raw word count.
        if to_device {
            7 * block_count
        } else {
            total_words
        }
    }

    fn dma_gpu_linked_list(&mut self) -> u32 {
        let mut addr = self.dma.channels[2].base & 0x001F_FFFC;
        // Completion delay is the traversed command-list word count.
        // Redux's helper seeds its local counter at 1, but the
        // scheduled interrupt reaches `branchTest` one cycle earlier
        // than our strict-slot scheduler when we include that seed.
        // Billing only the headers + payload words keeps the BIOS DMA
        // IRQ fold aligned for disc boots such as MGS.
        let mut total_words: u32 = 0;
        // Bound the walk so a malformed list can't spin forever.
        for _ in 0..0x100_0000 {
            let header = read_ram_u32(&self.ram[..], addr);
            let word_count = (header >> 24) & 0xFF;
            for i in 0..word_count {
                let word_addr = addr.wrapping_add(4 + i * 4);
                let word = read_ram_u32(&self.ram[..], word_addr);
                self.gpu.gp0_push(word);
            }
            // Redux charges `(header >> 24) + 1` per node (see
            // `gpuDmaChainSize:474`). The `+1` covers the header
            // fetch; payload accounts for the rest.
            total_words = total_words.saturating_add(word_count + 1);
            // End-of-chain: hardware uses *any* pointer with bit 23
            // set (0x800000) as the terminator, not just the common
            // 0x00FF_FFFF sentinel. Matches Redux's
            // `while (!(addr & 0x800000))` at gpu.cc:483.
            if (header & 0x800000) != 0 {
                return total_words;
            }
            addr = header & 0x00FF_FFFF;
        }
        total_words
    }

    /// Non-panicking byte read. Returns `None` for addresses outside
    /// any currently-mapped region. Diagnostic UIs (memory viewer,
    /// disassembler) use this to browse arbitrary ranges without
    /// crashing the emulator on unmapped addresses.
    ///
    /// Byte-granular reads of the GPU / timer / DMA / IRQ MMIO don't
    /// try to decompose the typed 32-bit registers — they return the
    /// echo-buffer byte, which is fine for a diagnostic dump.
    /// Byte write that silently drops addresses outside mapped RAM /
    /// scratchpad. Used by HLE BIOS helpers (like `memset`) where
    /// a buggy guest program shouldn't panic the host.
    pub fn write8_safe(&mut self, virt: u32, value: u8) -> bool {
        let phys = to_physical(virt);
        if phys < memory::ram::MIRROR_END {
            self.ram[(phys as usize) % memory::ram::SIZE] = value;
            true
        } else if (memory::scratchpad::BASE
            ..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            self.scratchpad[(phys - memory::scratchpad::BASE) as usize] = value;
            true
        } else {
            false
        }
    }

    /// Side-effect-free byte read — used by diagnostics (trace
    /// printer, parity oracle) that must not perturb peripheral
    /// state. Returns `None` for addresses that aren't backed by
    /// plain memory (CD-ROM FIFO, timers, etc.); the caller is
    /// expected to read those through [`Bus::read8`] if needed.
    pub fn try_read8(&self, virt: u32) -> Option<u8> {
        let phys = to_physical(virt);
        if phys < memory::ram::MIRROR_END {
            return Some(self.ram[(phys as usize) % memory::ram::SIZE]);
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            return Some(self.scratchpad[(phys - memory::scratchpad::BASE) as usize]);
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return Some(self.bios[(phys - memory::bios::BASE) as usize]);
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            return Some(self.io[(phys - memory::io::BASE) as usize]);
        }
        if (memory::expansion1::BASE..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return Some(0xFF);
        }
        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return Some(0xFF);
        }
        None
    }

    /// Read a 32-bit little-endian word from a virtual address.
    ///
    /// Panics on any address that does not resolve to a currently-mapped
    /// region. This is intentional — unmapped reads during development
    /// should surface immediately, not return silent zeros.
    ///
    /// `&mut self` because some peripherals (notably CD-ROM) mutate on
    /// read — popping response FIFOs, advancing data-transfer state.
    pub fn read8(&mut self, virt: u32) -> u8 {
        let phys = to_physical(virt);
        let value = self.read8_impl(virt, phys);
        self.trace_mmio(MmioKind::R8, phys, value as u32);
        value
    }

    fn read8_impl(&mut self, virt: u32, phys: u32) -> u8 {
        if phys < memory::ram::MIRROR_END {
            return self.ram[(phys as usize) % memory::ram::SIZE];
        }
        if CdRom::contains(phys) {
            return self.cdrom.read8(phys);
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            return self.scratchpad[(phys - memory::scratchpad::BASE) as usize];
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return self.bios[(phys - memory::bios::BASE) as usize];
        }
        if (memory::expansion1::BASE..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return 0xFF;
        }
        if Dma::contains(phys) {
            return self.dma.read8(phys);
        }
        if Sio0::contains(phys) {
            self.service_sio0();
            return self.sio0.read8(phys).unwrap_or(0);
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            return self.io[(phys - memory::io::BASE) as usize];
        }
        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return 0xFF;
        }
        // Unmapped read on real hardware returns the last bus
        // value (essentially random from software's POV). Many
        // games have wild pointers here and there; panicking
        // would halt perfectly-good emulation. Return 0xFF so
        // software sees "no peripheral." Log once at non-trivial
        // addresses so a real bug doesn't hide in the noise.
        if self.log_unmapped_read_once(virt) {
            eprintln!("[bus] unmapped read8 @ virt={virt:#010x} phys={phys:#010x}");
        }
        0xFF
    }

    /// Read a 16-bit little-endian half-word from a virtual address.
    /// Unmapped regions behave identically to [`Bus::read8`] (see the
    /// region-by-region notes there).
    ///
    /// `&mut self` for the same reason as `read8` — peripheral-side
    /// effects.
    pub fn read16(&mut self, virt: u32) -> u16 {
        let phys = to_physical(virt);
        let value = self.read16_impl(virt, phys);
        self.trace_mmio(MmioKind::R16, phys, value as u32);
        value
    }

    fn read16_impl(&mut self, virt: u32, phys: u32) -> u16 {
        if phys < memory::ram::MIRROR_END {
            let off = (phys as usize) % memory::ram::SIZE;
            return u16::from_le_bytes([self.ram[off], self.ram[off + 1]]);
        }
        if CdRom::contains(phys) {
            return self.cdrom.read8(phys) as u16;
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            let off = (phys - memory::scratchpad::BASE) as usize;
            return u16::from_le_bytes([self.scratchpad[off], self.scratchpad[off + 1]]);
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            let off = (phys - memory::bios::BASE) as usize;
            return u16::from_le_bytes([self.bios[off], self.bios[off + 1]]);
        }
        if (memory::expansion1::BASE..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return 0xFFFF;
        }
        // Same rationale as in `write16_impl`: BIOS reads `I_STAT` /
        // `I_MASK` via `lhu` and would otherwise see the stale echo
        // buffer instead of the live interrupt-controller state.
        if phys == IRQ_STAT_ADDR {
            return self.irq.stat() as u16;
        }
        if phys == IRQ_MASK_ADDR {
            return self.irq.mask() as u16;
        }
        // Timer registers are 16-bit on hardware; the BIOS's
        // counter-polling loop uses `lhu`. Without this dispatch the
        // counter reads zero from the io[] echo buffer and the loop
        // never sees the tick advance.
        if Timers::contains(phys) {
            self.service_timers();
            return self.timers.read32(phys) as u16;
        }
        if Dma::contains(phys) {
            return self.dma.read16(phys);
        }
        if Spu::contains(phys) {
            return self.spu.read16_at(phys, self.cycles);
        }
        if Sio0::contains(phys) {
            self.service_sio0();
            return self.sio0.read16(phys).unwrap_or(0);
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let off = (phys - memory::io::BASE) as usize;
            return u16::from_le_bytes([self.io[off], self.io[off + 1]]);
        }
        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return 0xFFFF;
        }
        if self.log_unmapped_read_once(virt) {
            eprintln!("[bus] unmapped read16 @ virt={virt:#010x} phys={phys:#010x}");
        }
        0xFFFF
    }

    /// Read a 32-bit little-endian word from a virtual address. This is
    /// the instruction-fetch path.
    ///
    /// `&mut self` because CD-ROM byte reads (composited into a u32 for
    /// the rare case software word-accesses that range) mutate.
    pub fn read32(&mut self, virt: u32) -> u32 {
        let phys = to_physical(virt);
        let value = self.read32_impl(virt, phys);
        self.trace_mmio(MmioKind::R32, phys, value);
        value
    }

    /// Side-effect-free peek at a 32-bit instruction word. Returns
    /// `None` when `virt` isn't in RAM or BIOS — those are the only
    /// places PS1 code ever executes from, so a `None` here is
    /// cheap to treat as "can't be a GTE cofun" at the call site.
    ///
    /// Used by [`crate::Cpu::should_take_interrupt`] — see the
    /// "interrupts vs GTE" hardware bug workaround.
    pub fn peek_instruction(&self, virt: u32) -> Option<u32> {
        let phys = to_physical(virt);
        if phys < memory::ram::MIRROR_END {
            let offset = (phys as usize) % memory::ram::SIZE;
            Some(read_u32_le(&self.ram[offset..]))
        } else if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32)
            .contains(&phys)
        {
            let offset = (phys - memory::bios::BASE) as usize;
            Some(read_u32_le(&self.bios[offset..]))
        } else {
            None
        }
    }

    fn read32_impl(&mut self, virt: u32, phys: u32) -> u32 {
        if phys < memory::ram::MIRROR_END {
            let offset = (phys as usize) % memory::ram::SIZE;
            return read_u32_le(&self.ram[offset..]);
        }

        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            let offset = (phys - memory::scratchpad::BASE) as usize;
            return read_u32_le(&self.scratchpad[offset..]);
        }

        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            let offset = (phys - memory::bios::BASE) as usize;
            return read_u32_le(&self.bios[offset..]);
        }

        if (memory::expansion1::BASE..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return 0xFFFF_FFFF;
        }

        if phys == IRQ_STAT_ADDR {
            return self.irq.stat();
        }
        if phys == IRQ_MASK_ADDR {
            return self.irq.mask();
        }
        if Timers::contains(phys) {
            self.service_timers();
            return self.timers.read32(phys);
        }
        if Dma::contains(phys) {
            return self.dma.read32(phys);
        }
        if let Some(v) = self.gpu.read32(phys) {
            return v;
        }
        if Spu::contains(phys) {
            return self.spu.read32_at(phys, self.cycles);
        }
        if Sio0::contains(phys) {
            self.service_sio0();
            return self.sio0.read32(phys).unwrap_or(0);
        }
        if CdRom::contains(phys) {
            // CD-ROM regs are 8-bit; word access composites them.
            let b0 = self.cdrom.read8(phys) as u32;
            let b1 = self.cdrom.read8(phys + 1) as u32;
            let b2 = self.cdrom.read8(phys + 2) as u32;
            let b3 = self.cdrom.read8(phys + 3) as u32;
            return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        if crate::mdec::Mdec::contains(phys) {
            return self.mdec.read32(phys);
        }

        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let offset = (phys - memory::io::BASE) as usize;
            return read_u32_le(&self.io[offset..]);
        }

        if self.log_unmapped_read_once(virt) {
            eprintln!("[bus] unmapped read32 @ virt={virt:#010x} phys={phys:#010x}");
        }
        0xFFFF_FFFF
    }

    /// Write a 32-bit little-endian word to a virtual address.
    ///
    /// - **RAM / scratchpad**: committed to the backing storage.
    /// - **BIOS ROM**: silently dropped (ROM is read-only).
    /// - **Cache-control register** `0xFFFE_0130`: silently dropped
    ///   until we model the I-cache.
    /// - **MMIO window** `0x1F80_1000..0x1F80_2000`: silently dropped
    ///   for now. Individual peripheral stubs will attach as we add
    ///   them; until then, BIOS's memory-controller init writes are
    ///   no-ops for architectural parity.
    pub fn write32(&mut self, virt: u32, value: u32) {
        if virt == memory::cache_control::ADDR {
            return;
        }

        let phys = to_physical(virt);
        self.trace_mmio(MmioKind::W32, phys, value);
        self.write32_impl(virt, phys, value);
    }

    fn write32_impl(&mut self, virt: u32, phys: u32, value: u32) {
        if phys == IRQ_STAT_ADDR {
            self.irq.write_stat_at(value, self.cycles);
            return;
        }
        if phys == IRQ_MASK_ADDR {
            self.irq.write_mask_at(value, self.cycles);
            return;
        }
        if Timers::contains(phys) {
            // Advance pre-write so an in-flight tick fires its IRQ
            // before the write resets / re-arms the counter, then
            // perform the write.
            self.service_timers();
            self.timers.write32(phys, value, self.cycles);
            return;
        }
        if Dma::contains(phys) {
            if self.dma.write32(phys, value) {
                self.irq.raise(IrqSource::Dma);
            }
            // Only a CHCR write with bit 24 set starts a transfer —
            // matches Redux's `dmaExec<N>` dispatcher in `psxhw.cc`,
            // which runs from the per-channel `case 0x1f80_1088/98/
            // a8/b8/c8/e8` arms. Crucially it runs ONLY channel N,
            // not a sweep across all channels.
            //
            // Earlier we called `maybe_run_dma()` (iterates every
            // channel) on every CHCR trigger. If another channel's
            // transfer was still in-flight (start bit still set,
            // awaiting its scheduled completion), the sweep re-ran
            // it, scheduling a second completion that overwrote the
            // first. Example caught by `probe_dma_schedules`:
            // writing channel 6's CHCR at cycle 46246689 re-ran
            // channel 2's in-flight DMA, pushing its IRQ target from
            // 46247457 to 46247720 — which is exactly the -2488-
            // cycle drift `probe_cycle_first_divergence` flagged at
            // step 19474544.
            let offset = phys.wrapping_sub(Dma::BASE);
            let field = offset & 0xF;
            let is_chcr_write = field == 0x8;
            let start_bit_set = value & (1 << 24) != 0;
            if is_chcr_write && start_bit_set {
                let channel = ((offset & 0x70) >> 4) as usize;
                self.run_dma_channel(channel);
            }
            return;
        }
        if self.gpu.write32(phys, value) {
            if phys == crate::gpu::GP1_ADDR && (value >> 24) == 0x08 {
                if value & (1 << 3) != 0 {
                    self.set_pal_mode();
                } else {
                    self.set_ntsc_mode();
                }
            }
            return;
        }
        if Spu::contains(phys) {
            self.spu.write32_at(phys, value, self.cycles);
            return;
        }
        if Sio0::contains(phys) {
            self.service_sio0();
            self.sio0.write32_at(phys, value, self.cycles);
            self.service_sio0();
            return;
        }
        if crate::mdec::Mdec::contains(phys) {
            self.mdec.write32(phys, value);
            return;
        }

        let bytes = value.to_le_bytes();

        if phys < memory::ram::MIRROR_END {
            let offset = (phys as usize) % memory::ram::SIZE;
            self.ram[offset..offset + 4].copy_from_slice(&bytes);
            return;
        }

        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            let offset = (phys - memory::scratchpad::BASE) as usize;
            self.scratchpad[offset..offset + 4].copy_from_slice(&bytes);
            return;
        }

        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let offset = (phys - memory::io::BASE) as usize;
            self.io[offset..offset + 4].copy_from_slice(&bytes);
            return;
        }

        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return;
        }

        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return;
        }

        if self.log_unmapped_write_once(virt) {
            eprintln!(
                "[bus] unmapped write32 @ virt={virt:#010x} phys={phys:#010x} value={value:#010x}"
            );
        }
    }

    /// Write a byte to a virtual address. Unmapped writes in MMIO /
    /// expansion / BIOS ranges are silently dropped (same rationale as
    /// [`Bus::write32`]).
    pub fn write8(&mut self, virt: u32, value: u8) {
        let phys = to_physical(virt);
        self.trace_mmio(MmioKind::W8, phys, value as u32);
        self.write8_impl(virt, phys, value);
    }

    fn write8_impl(&mut self, virt: u32, phys: u32, value: u8) {
        if phys < memory::ram::MIRROR_END {
            self.ram[(phys as usize) % memory::ram::SIZE] = value;
            return;
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            self.scratchpad[(phys - memory::scratchpad::BASE) as usize] = value;
            return;
        }
        // CDROM is byte-addressed (4 registers, each switching meaning by
        // index). Without this dispatch the BIOS's `sb` to 1F801800..1803
        // ends up in the io[] echo buffer and the CDROM module never sees
        // the index switch / param push / command write — so commands are
        // silently dropped, no IRQ ever fires, and the BIOS stalls in the
        // event-wait loop after the Sony intro.
        if CdRom::contains(phys) {
            // Thread `self.cycles` through so the CDROM scheduler
            // anchors response-IRQ deadlines on the exact cycle at
            // the cmd-port write, matching Redux's `AddIrqQueue`.
            if self.cdrom.write8_at(phys, value, self.cycles) {
                self.irq.raise(IrqSource::Cdrom);
            }
            return;
        }
        if Dma::contains(phys) {
            if self.dma.write8(phys, value) {
                self.irq.raise(IrqSource::Dma);
            }
            return;
        }
        if Sio0::contains(phys) {
            self.service_sio0();
            self.sio0.write8_at(phys, value, self.cycles);
            self.service_sio0();
            return;
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            self.io[(phys - memory::io::BASE) as usize] = value;
            return;
        }
        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return;
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return;
        }
        if self.log_unmapped_write_once(virt) {
            eprintln!(
                "[bus] unmapped write8 @ virt={virt:#010x} phys={phys:#010x} value={value:#04x}"
            );
        }
    }

    /// Write a 16-bit half-word to a virtual address. Same unmapped-region
    /// policy as [`Bus::write32`].
    pub fn write16(&mut self, virt: u32, value: u16) {
        let phys = to_physical(virt);
        self.trace_mmio(MmioKind::W16, phys, value as u32);
        self.write16_impl(virt, phys, value);
    }

    fn write16_impl(&mut self, virt: u32, phys: u32, value: u16) {
        let bytes = value.to_le_bytes();
        if phys < memory::ram::MIRROR_END {
            let off = (phys as usize) % memory::ram::SIZE;
            self.ram[off..off + 2].copy_from_slice(&bytes);
            return;
        }
        if (memory::scratchpad::BASE..memory::scratchpad::BASE + memory::scratchpad::SIZE as u32)
            .contains(&phys)
        {
            let off = (phys - memory::scratchpad::BASE) as usize;
            self.scratchpad[off..off + 2].copy_from_slice(&bytes);
            return;
        }
        // The BIOS uses `sh` (16-bit store) to write `I_MASK` and ack
        // `I_STAT`. Without this dispatch the value lands in the io[]
        // echo buffer and the IRQ controller never sees it — meaning
        // mask stays 0 and pending() always returns false, so no IRQ
        // exception is ever taken.
        if phys == IRQ_STAT_ADDR {
            self.irq.write_stat_at(value as u32, self.cycles);
            return;
        }
        if phys == IRQ_MASK_ADDR {
            self.irq.write_mask_at(value as u32, self.cycles);
            return;
        }
        // Timer registers are 16-bit on hardware.
        if Timers::contains(phys) {
            self.service_timers();
            self.timers.write32(phys, value as u32, self.cycles);
            return;
        }
        if Dma::contains(phys) {
            if self.dma.write16(phys, value) {
                self.irq.raise(IrqSource::Dma);
            }
            return;
        }
        if Spu::contains(phys) {
            self.spu.write16_at(phys, value, self.cycles);
            return;
        }
        if Sio0::contains(phys) {
            self.service_sio0();
            self.sio0.write16_at(phys, value, self.cycles);
            self.service_sio0();
            return;
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let off = (phys - memory::io::BASE) as usize;
            self.io[off..off + 2].copy_from_slice(&bytes);
            return;
        }
        if (memory::expansion2::BASE..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return;
        }
        if (memory::bios::BASE..memory::bios::BASE + memory::bios::SIZE as u32).contains(&phys) {
            return;
        }
        if self.log_unmapped_write_once(virt) {
            eprintln!(
                "[bus] unmapped write16 @ virt={virt:#010x} phys={phys:#010x} value={value:#06x}"
            );
        }
    }
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Word read from a RAM slice at a physical RAM offset (already masked
/// to the 2 MiB range). Used by the DMA-GPU paths to pull command
/// words without going through the full bus dispatch.
fn read_ram_u32(ram: &[u8], phys: u32) -> u32 {
    let offset = (phys & 0x001F_FFFF) as usize;
    if offset + 4 <= ram.len() {
        u32::from_le_bytes([
            ram[offset],
            ram[offset + 1],
            ram[offset + 2],
            ram[offset + 3],
        ])
    } else {
        0
    }
}

fn mdec_dma_word_count(block_control: u32, channel_control: u32) -> u32 {
    match (channel_control >> 9) & 0x3 {
        0 => block_control & 0xFFFF,
        1 => {
            let block_size = block_control & 0xFFFF;
            let block_count = (block_control >> 16).max(1) & 0xFFFF;
            block_size * block_count
        }
        _ => 0,
    }
}

fn zeroed_box<const N: usize>() -> Box<[u8; N]> {
    // Allocates a zero-initialised slice and converts it. The try_into
    // cannot fail because the source slice has exactly N elements.
    vec![0u8; N]
        .into_boxed_slice()
        .try_into()
        .expect("vec length matches const N")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::EventSlot;
    use crate::IrqSource;

    fn synthetic_bios() -> Vec<u8> {
        // 512 KiB. First word is 0xDEADBEEF little-endian, then zeros.
        let mut bios = vec![0u8; memory::bios::SIZE];
        bios[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        bios
    }

    #[test]
    fn rejects_wrong_sized_bios() {
        assert!(matches!(
            Bus::new(vec![0u8; 1024]),
            Err(BusError::BiosSize { .. })
        ));
    }

    #[test]
    fn default_video_region_is_ntsc() {
        let bus = Bus::new(synthetic_bios()).unwrap();
        assert_eq!(bus.hsync_cycles(), HSYNC_CYCLES_NTSC);
        assert_eq!(bus.vblank_period(), VBLANK_PERIOD_CYCLES_NTSC);
    }

    #[test]
    fn set_pal_mode_switches_timings_and_reschedules_vblank() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.set_pal_mode();
        assert_eq!(bus.hsync_cycles(), HSYNC_CYCLES_PAL);
        assert_eq!(bus.vblank_period(), VBLANK_PERIOD_CYCLES_PAL);
        // VBlank slot should be armed for the PAL first-VBlank cycle.
        let target = bus
            .scheduler
            .target(crate::scheduler::EventSlot::VBlank)
            .expect("VBlank must be rescheduled on PAL switch");
        assert_eq!(target, FIRST_VBLANK_CYCLE_PAL);
    }

    #[test]
    fn gp1_display_mode_switches_video_region() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();

        bus.write32(crate::gpu::GP1_ADDR, 0x0800_0008);
        assert_eq!(bus.hsync_cycles(), HSYNC_CYCLES_PAL);
        assert_eq!(bus.vblank_period(), VBLANK_PERIOD_CYCLES_PAL);
        assert_eq!(
            bus.scheduler
                .target(crate::scheduler::EventSlot::VBlank)
                .unwrap(),
            FIRST_VBLANK_CYCLE_PAL,
        );

        bus.write32(crate::gpu::GP1_ADDR, 0x0800_0000);
        assert_eq!(bus.hsync_cycles(), HSYNC_CYCLES_NTSC);
        assert_eq!(bus.vblank_period(), VBLANK_PERIOD_CYCLES_NTSC);
        assert_eq!(
            bus.scheduler
                .target(crate::scheduler::EventSlot::VBlank)
                .unwrap(),
            FIRST_VBLANK_CYCLE_NTSC,
        );
    }

    #[test]
    fn video_region_switch_preserves_scanline_phase() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        let phase = 500;
        bus.cycles = FIRST_VBLANK_CYCLE_NTSC + 100 * HSYNC_CYCLES_NTSC + phase;
        bus.scheduler.cancel(crate::scheduler::EventSlot::VBlank);
        bus.scheduler.schedule(
            crate::scheduler::EventSlot::VBlank,
            0,
            FIRST_VBLANK_CYCLE_NTSC + VBLANK_PERIOD_CYCLES_NTSC,
        );

        bus.write32(crate::gpu::GP1_ADDR, 0x0800_0008);

        let current_scanline = (VBLANK_START_SCANLINE_NTSC + 100) % HSYNC_TOTAL_NTSC;
        let remaining = VBLANK_START_SCANLINE_PAL - current_scanline;
        let expected = bus.cycles + remaining * HSYNC_CYCLES_NTSC - phase + 4;
        assert_eq!(
            bus.scheduler
                .target(crate::scheduler::EventSlot::VBlank)
                .unwrap(),
            expected,
        );
    }

    #[test]
    fn reads_first_bios_word_via_kseg1_reset_vector() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        assert_eq!(bus.read32(memory::bios::RESET_VECTOR), 0xDEAD_BEEF);
    }

    #[test]
    fn reads_first_bios_word_via_kseg0_and_kuseg() {
        // BIOS physical base mapped into KSEG0 (cached) and KUSEG.
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        assert_eq!(bus.read32(0x9FC0_0000), 0xDEAD_BEEF); // KSEG0
        assert_eq!(bus.read32(0x1FC0_0000), 0xDEAD_BEEF); // KUSEG physical alias
    }

    #[test]
    fn ram_starts_zeroed() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        assert_eq!(bus.read32(0x0000_0000), 0);
        assert_eq!(bus.read32(0x8000_0000), 0); // KSEG0 RAM
    }

    #[test]
    fn ram_mirrors_wrap_within_8mib() {
        // Hardware mirrors the 2 MiB RAM four times up to 0x0080_0000.
        // A write to offset 0 should be visible at +2 MiB, +4 MiB, +6 MiB.
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.ram[0..4].copy_from_slice(&0x1122_3344u32.to_le_bytes());
        assert_eq!(bus.read32(0x0000_0000), 0x1122_3344);
        assert_eq!(bus.read32(0x0020_0000), 0x1122_3344);
        assert_eq!(bus.read32(0x0040_0000), 0x1122_3344);
        assert_eq!(bus.read32(0x0060_0000), 0x1122_3344);
    }

    #[test]
    fn vblank_scheduler_fires_at_first_threshold() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        // Just below the first VBlank — no fire yet.
        bus.tick(FIRST_VBLANK_CYCLE as u32 - 1);
        bus.run_vblank_scheduler();
        assert_eq!(bus.irq.stat() & 1, 0);

        // Cross the threshold — one VBlank fires.
        bus.tick(2);
        bus.run_vblank_scheduler();
        assert_eq!(bus.irq.stat() & 1, 1);
        assert_eq!(
            bus.next_vblank_cycle(),
            FIRST_VBLANK_CYCLE + VBLANK_PERIOD_CYCLES
        );
    }

    #[test]
    fn vblank_scheduler_fires_on_exact_threshold() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.tick(FIRST_VBLANK_CYCLE as u32);
        assert_eq!(bus.irq.stat() & 1, 1);
        assert_eq!(
            bus.next_vblank_cycle(),
            FIRST_VBLANK_CYCLE + VBLANK_PERIOD_CYCLES
        );
    }

    #[test]
    fn vblank_scheduler_catches_up_after_long_tick() {
        // Tick far past the first VBlank in one go — the scheduler
        // must fire every VBlank that would have elapsed, not just one.
        // Ack each time so we can count.
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.tick((FIRST_VBLANK_CYCLE + 3 * VBLANK_PERIOD_CYCLES + 1) as u32);
        bus.run_vblank_scheduler();
        // irq.stat bit 0 is either 0 or 1 — can't count from there.
        // Instead, check next_vblank_cycle advanced by 4 periods.
        let expected = FIRST_VBLANK_CYCLE + 4 * VBLANK_PERIOD_CYCLES;
        assert_eq!(bus.next_vblank_cycle(), expected);
        // VBlank bit should be set (at least one fire happened).
        assert_eq!(bus.irq.stat() & 1, 1);
    }

    #[test]
    fn vblank_source_index_is_0() {
        // Sanity: IrqSource::VBlank is bit 0, matching Redux's setIrq(0x01).
        assert_eq!(IrqSource::VBlank as u32, 0);
    }

    #[test]
    fn cdrom_dma_requires_ready_sector() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.cdrom.debug_seed_data_fifo(&[], false, false);
        bus.dma.dpcr = 1 << (3 * 4 + 3);
        bus.dma.channels[3].base = 0;
        bus.dma.channels[3].block_control = 1;
        bus.dma.channels[3].channel_control = 0x1100_0000;

        bus.run_dma_channel(3);
        assert_eq!(read_ram_u32(&bus.ram[..], 0), 0);
        assert_eq!(bus.dma.channels[3].channel_control & (1 << 24), 0);
        assert_eq!(bus.scheduler.target(EventSlot::CdrDma), None);
    }

    #[test]
    fn cdrom_dma_drains_ready_sector_without_request_latch() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.cdrom.debug_seed_data_fifo(&[1, 2, 3, 4], true, false);
        bus.dma.dpcr = 1 << (3 * 4 + 3);
        bus.dma.channels[3].base = 0;
        bus.dma.channels[3].block_control = 1;
        bus.dma.channels[3].channel_control = 0x1100_0000;

        bus.run_dma_channel(3);

        assert_eq!(read_ram_u32(&bus.ram[..], 0), 0x0403_0201);
        assert_ne!(bus.dma.channels[3].channel_control & (1 << 24), 0);
        assert_eq!(bus.scheduler.target(EventSlot::CdrDma), Some(1));

        bus.tick(1);
        assert_ne!(bus.dma.channels[3].channel_control & (1 << 24), 0);
        bus.drain_scheduler_events_post_op();
        assert_eq!(bus.dma.channels[3].channel_control & (1 << 24), 0);
    }

    #[test]
    fn dma_does_not_lose_byte_writes_to_dicr() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();

        bus.write8(0x1F80_10F6, 0x80);
        assert_eq!(bus.read8(0x1F80_10F6) & 0x08, 0);
        assert!(!bus.dma.notify_channel_done(3));

        bus.write8(0x1F80_10F6, 0x88);
        assert_ne!(bus.read8(0x1F80_10F6) & 0x08, 0);
        assert!(bus.dma.notify_channel_done(3));
    }

    #[test]
    fn cdrom_dma_zero_bcr_falls_back_to_buffered_sector_size() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.cdrom
            .debug_seed_data_fifo(&[1, 2, 3, 4, 5, 6, 7, 8], true, true);
        bus.dma.dpcr = 1 << (3 * 4 + 3);
        bus.dma.channels[3].base = 0;
        bus.dma.channels[3].block_control = 0;
        bus.dma.channels[3].channel_control = 0x1100_0000;

        assert_eq!(bus.run_dma_cdrom(), Some(2));
        assert_eq!(read_ram_u32(&bus.ram[..], 0), 0x0403_0201);
        assert_eq!(read_ram_u32(&bus.ram[..], 4), 0x0807_0605);
        assert_eq!(bus.cdrom.data_fifo_len(), 0);
        assert!(!bus.cdrom.data_transfer_armed());
    }

    #[test]
    fn cdrom_burst_dma_uses_redux_quarter_rate_completion_delay() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        bus.cycles = 100;
        bus.cdrom
            .debug_seed_data_fifo(&[1, 2, 3, 4, 5, 6, 7, 8], true, true);
        bus.dma.dpcr = 1 << (3 * 4 + 3);
        bus.dma.channels[3].base = 0;
        bus.dma.channels[3].block_control = 2;
        bus.dma.channels[3].channel_control = 0x1140_0100;

        bus.run_dma_channel(3);

        assert_eq!(bus.scheduler.target(EventSlot::CdrDma), Some(101));
    }

    #[test]
    fn mdec_decode_dma0_completes_with_final_dma1() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        seed_one_macroblock_decode(&mut bus);
        enable_mdec_dma(&mut bus);

        bus.dma.channels[0].base = 0x100;
        bus.dma.channels[0].block_control = 6;
        bus.dma.channels[0].channel_control = 0x0100_0201;
        bus.run_dma_channel(0);

        assert_eq!(bus.scheduler.target(EventSlot::MdecInDma), None);
        assert_ne!(bus.dma.channels[0].channel_control & (1 << 24), 0);

        bus.dma.channels[1].base = 0x200;
        bus.dma.channels[1].block_control = 192;
        bus.dma.channels[1].channel_control = 0x0100_0200;
        bus.run_dma_channel(1);

        assert_eq!(bus.scheduler.target(EventSlot::MdecOutDma), Some(192 * 8));
        assert_ne!(bus.dma.channels[0].channel_control & (1 << 24), 0);
        assert_ne!(bus.dma.channels[1].channel_control & (1 << 24), 0);

        bus.tick(192 * 8 + 1);
        assert_eq!(bus.dma.channels[0].channel_control & (1 << 24), 0);
        assert_eq!(bus.dma.channels[1].channel_control & (1 << 24), 0);
    }

    #[test]
    fn pending_mdec_dma1_fires_after_dma0_produces_output() {
        let mut bus = Bus::new(synthetic_bios()).unwrap();
        seed_one_macroblock_decode(&mut bus);
        enable_mdec_dma(&mut bus);

        bus.dma.channels[1].base = 0x200;
        bus.dma.channels[1].block_control = 192;
        bus.dma.channels[1].channel_control = 0x0100_0200;
        bus.run_dma_channel(1);

        assert_eq!(bus.scheduler.target(EventSlot::MdecOutDma), None);
        assert_ne!(bus.dma.channels[1].channel_control & (1 << 24), 0);

        bus.dma.channels[0].base = 0x100;
        bus.dma.channels[0].block_control = 6;
        bus.dma.channels[0].channel_control = 0x0100_0201;
        bus.run_dma_channel(0);

        assert_eq!(bus.scheduler.target(EventSlot::MdecInDma), None);
        assert_eq!(bus.scheduler.target(EventSlot::MdecOutDma), Some(192 * 8));
        assert_ne!(read_ram_u32(&bus.ram[..], 0x200), 0);
    }

    fn enable_mdec_dma(bus: &mut Bus) {
        bus.dma.dpcr = (1 << (0 * 4 + 3)) | (1 << (1 * 4 + 3));
    }

    fn seed_one_macroblock_decode(bus: &mut Bus) {
        bus.mdec.write32(crate::mdec::MDEC_CMD_DATA, 0x4000_0020);
        bus.mdec.dma_write_in(&[0x01_01_01_01; 32]);
        bus.mdec.write32(crate::mdec::MDEC_CMD_DATA, 0x3000_0006);
        for i in 0..6 {
            let offset = 0x100 + i * 4;
            bus.ram[offset..offset + 4].copy_from_slice(&0xFE00_0010u32.to_le_bytes());
        }
    }
}
