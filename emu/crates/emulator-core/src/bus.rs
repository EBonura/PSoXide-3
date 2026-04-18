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
    timers: Timers,
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
    /// Addresses we've already logged as unmapped reads. Keeps
    /// log noise bounded when a buggy game pokes a bad pointer
    /// in a tight loop.
    unmapped_read_seen: std::collections::BTreeSet<u32>,
    /// Same, writes. Separate so read + write to the same bad
    /// address both log at least once.
    unmapped_write_seen: std::collections::BTreeSet<u32>,
}

// --- Phase 4 scheduler constants (NTSC) ---
//
// Match Redux's `psxcounters.cc` math exactly so VBlank fires at the
// same cycle — and therefore at the same instruction — on both sides.
//
//   HSync period   = psxClockSpeed / (FrameRate × HSyncTotal)
//                  = 33_868_800 / (60 × 263) = 2146 cycles
//   VBlank period  = HSyncTotal × HSync     = 263 × 2146 = 564_398 cycles
//   First VBlank   = VBlankStart × HSync    = 243 × 2146 = 521_478 cycles
//
// `VBLANK_PERIOD_CYCLES` is kept for Phase 4b even though it's unused
// in 4a — silenced with `allow(dead_code)`.

const HSYNC_CYCLES_NTSC: u64 = 2146;
#[allow(dead_code)]
const HSYNC_TOTAL_NTSC: u64 = 263;
const VBLANK_START_SCANLINE: u64 = 243;
const FIRST_VBLANK_CYCLE: u64 = HSYNC_CYCLES_NTSC * VBLANK_START_SCANLINE;
#[allow(dead_code)]
const VBLANK_PERIOD_CYCLES: u64 = HSYNC_CYCLES_NTSC * HSYNC_TOTAL_NTSC;

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
            unmapped_read_seen: std::collections::BTreeSet::new(),
            unmapped_write_seen: std::collections::BTreeSet::new(),
        })
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
        let device = crate::pad::PortDevice::empty().with_pad(crate::pad::DigitalPad::new());
        self.sio0.attach_port1(device);
    }

    /// Plug a memory card into port 1 with the given backing
    /// contents (128 KiB buffer, typically loaded from a
    /// `.mcd` file). Pass an empty `Vec` to start with a fresh
    /// card. Keeps any pad already attached — real hardware
    /// multiplexes pad + memcard on the same port.
    pub fn attach_memcard_port1(&mut self, initial_bytes: Vec<u8>) {
        let bytes = if initial_bytes.len() == crate::pad::MEMCARD_SIZE {
            initial_bytes
        } else {
            vec![0u8; crate::pad::MEMCARD_SIZE]
        };
        // Preserve any existing pad.
        let pad = std::mem::replace(self.sio0.port1_mut(), crate::pad::PortDevice::empty())
            .into_pad();
        let mut device =
            crate::pad::PortDevice::empty().with_memcard(crate::pad::MemoryCard::from_bytes(bytes));
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
    pub fn tick(&mut self, n: u32) {
        self.advance_cycles(n);
        self.drain_scheduler_events();
        if self.cdrom.tick(self.cycles) {
            self.irq.raise(IrqSource::Cdrom);
        }
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
    fn drain_scheduler_events(&mut self) {
        use crate::scheduler::EventSlot;
        let now = self.cycles;
        let mut dma_edge = false;
        while let Some((slot, target)) = self.scheduler.take_due(now) {
            match slot {
                EventSlot::MdecInDma => {
                    if self.complete_dma_channel(0) {
                        dma_edge = true;
                    }
                }
                EventSlot::MdecOutDma => {
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
                    // Re-arm the next VBlank from the original
                    // target, not `now`. A 500K-cycle drain lag
                    // would otherwise accumulate drift every time.
                    self.scheduler
                        .schedule(EventSlot::VBlank, target, VBLANK_PERIOD_CYCLES);
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
        let fired = self
            .timers
            .tick(n as u64, HSYNC_CYCLES_NTSC, self.gpu.dot_clock_divisor());
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
    pub fn run_spu_samples(&mut self, n: usize) {
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

    /// Run any DMA channels whose start bit was just set.
    /// - Channel 6 (OTC) fills the ordering table in RAM.
    /// - Channel 2 (GPU) ships command words from RAM to the GPU's GP0 port.
    /// - Other channels need their consumer subsystems online first.
    ///
    /// After any transfer completes, `IrqSource::Dma` is raised — the
    /// BIOS's DMA-wait event handlers need this to see completion.
    /// DICR per-channel / master enable filtering isn't modeled yet;
    /// the interrupt controller's `I_MASK` is the only gate.
    fn maybe_run_dma(&mut self) {
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
        let otc_words = self.dma.run_otc(&mut self.ram[..]);
        if otc_words > 0 {
            self.scheduler
                .schedule(EventSlot::GpuOtcDma, self.cycles, otc_words as u64);
        }
        if let Some(gpu_cycles) = self.run_dma_gpu() {
            self.scheduler
                .schedule(EventSlot::GpuDma, self.cycles, gpu_cycles as u64);
        }
        if let Some(cdrom_words) = self.run_dma_cdrom() {
            self.scheduler
                .schedule(EventSlot::CdrDma, self.cycles, cdrom_words as u64);
        }
        if let Some(spu_words) = self.run_dma_spu() {
            self.scheduler
                .schedule(EventSlot::SpuDma, self.cycles, spu_words as u64);
        }
        if let Some(mdec_words) = self.run_dma_mdec_in() {
            self.scheduler
                .schedule(EventSlot::MdecInDma, self.cycles, mdec_words as u64);
        }
        if let Some(mdec_words) = self.run_dma_mdec_out() {
            self.scheduler
                .schedule(EventSlot::MdecOutDma, self.cycles, mdec_words as u64);
        }
    }

    /// Execute DMA channel 0 → MDEC input. Ships command + RLE data
    /// from main RAM to the MDEC's input queue. Sync mode 1 (block)
    /// is the only mode PS1 software uses for this channel.
    fn run_dma_mdec_in(&mut self) -> Option<u32> {
        let ch = &self.dma.channels[0];
        if (ch.channel_control >> 24) & 1 == 0 {
            return None;
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
            _ => 0,
        };
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
        let ch = &self.dma.channels[1];
        if (ch.channel_control >> 24) & 1 == 0 {
            return None;
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
            _ => 0,
        };
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
        let ch = self.dma.channels[3];
        if (ch.channel_control >> 24) & 1 == 0 {
            return None;
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
        let total_words = match sync_mode {
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

        let mut addr = ch.base & 0x001F_FFFC;
        let step: u32 = if (ch.channel_control >> 1) & 1 != 0 {
            0xFFFF_FFFCu32
        } else {
            4
        };

        for _ in 0..total_words {
            let b0 = self.cdrom.pop_data_byte() as u32;
            let b1 = self.cdrom.pop_data_byte() as u32;
            let b2 = self.cdrom.pop_data_byte() as u32;
            let b3 = self.cdrom.pop_data_byte() as u32;
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
            // Header + payload both contribute to Redux's per-node
            // cycle cost accounting.
            total_words = total_words.saturating_add(1 + word_count);
            let next = header & 0x00FF_FFFF;
            if next == 0x00FF_FFFF {
                return total_words;
            }
            addr = next;
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
        if Sio0::contains(phys) {
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
            return self.timers.read32(phys) as u16;
        }
        if Spu::contains(phys) {
            return self.spu.read16_at(phys, self.cycles);
        }
        if Sio0::contains(phys) {
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
            self.timers.write32(phys, value, self.cycles);
            return;
        }
        if Dma::contains(phys) {
            self.dma.write32(phys, value);
            // A CHCR write can request a transfer; run whatever we
            // can self-contain right now (OTC only, so far).
            self.maybe_run_dma();
            return;
        }
        if self.gpu.write32(phys, value) {
            return;
        }
        if Spu::contains(phys) {
            self.spu.write32_at(phys, value, self.cycles);
            return;
        }
        if Sio0::contains(phys) {
            self.sio0.write32(phys, value);
            if self.sio0.take_pending_irq() {
                self.irq.raise(IrqSource::Controller);
            }
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
            self.cdrom.write8(phys, value);
            return;
        }
        if Sio0::contains(phys) {
            self.sio0.write8(phys, value);
            if self.sio0.take_pending_irq() {
                self.irq.raise(IrqSource::Controller);
            }
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
            self.timers.write32(phys, value as u32, self.cycles);
            return;
        }
        if Spu::contains(phys) {
            self.spu.write16_at(phys, value, self.cycles);
            return;
        }
        if Sio0::contains(phys) {
            self.sio0.write16(phys, value);
            if self.sio0.take_pending_irq() {
                self.irq.raise(IrqSource::Controller);
            }
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
}
