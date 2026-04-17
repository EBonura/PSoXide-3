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
    /// SPU — phase 3a scope: just `SPUCNT` + `SPUSTAT`. Everything
    /// else SPU-related still round-trips through the echo buffer.
    spu: Spu,
    /// CD-ROM controller — byte-granular MMIO at 0x1F80_1800..=0x1803.
    /// Exposed public so diagnostics can inspect FIFO / command state.
    pub cdrom: CdRom,
    /// Cumulative CPU cycles retired since reset. Fed by `Cpu::step`
    /// via [`Bus::tick`]. Peripherals read this to schedule events
    /// (VBlank, timer ticks, DMA completion). Phase 4a just counts;
    /// Phase 4b starts firing IRQs off it.
    cycles: u64,
    /// Absolute cycle count at which the *next* VBlank should fire.
    /// Matches PCSX-Redux's counter-3 math: first VBlank at scanline
    /// 243 of 263 (NTSC) at `HSync * 243 = 2146 * 243 = 521_478`
    /// cycles, then every `HSync * 263 = 564_398` cycles thereafter.
    /// Phase 4a tracks this but doesn't fire — Phase 4b hangs the
    /// VBlank IRQ off reaching this threshold.
    next_vblank_cycle: u64,
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
            cdrom: CdRom::new(),
            cycles: 0,
            next_vblank_cycle: FIRST_VBLANK_CYCLE,
        })
    }

    /// Cycle count at which the next VBlank is scheduled to fire.
    /// Exposed for diagnostics / the HUD.
    pub fn next_vblank_cycle(&self) -> u64 {
        self.next_vblank_cycle
    }

    /// Advance the cycle counter by `n` cycles and run any scheduled
    /// peripheral events that have come due. Called once per
    /// instruction from `Cpu::step`.
    pub fn tick(&mut self, n: u32) {
        self.cycles = self.cycles.wrapping_add(n as u64);
        self.run_vblank_scheduler();
        let fired = self.timers.tick(n as u64, HSYNC_CYCLES_NTSC);
        if fired & 1 != 0 {
            self.irq.raise(IrqSource::Timer0);
        }
        if fired & 2 != 0 {
            self.irq.raise(IrqSource::Timer1);
        }
        if fired & 4 != 0 {
            self.irq.raise(IrqSource::Timer2);
        }
        if self.cdrom.tick(self.cycles) {
            self.irq.raise(IrqSource::Cdrom);
        }
    }

    /// Cumulative cycles since reset.
    pub fn cycles(&self) -> u64 {
        self.cycles
    }

    /// Advance the VBlank schedule and raise `IrqSource::VBlank` for
    /// every period that's elapsed. Not called yet — Phase 4b turns
    /// this on by invoking it from `tick`. Exposed + public so the
    /// frontend can turn it on experimentally, and so unit tests can
    /// exercise it in isolation.
    pub fn run_vblank_scheduler(&mut self) {
        while self.cycles >= self.next_vblank_cycle {
            self.irq.raise(IrqSource::VBlank);
            self.next_vblank_cycle =
                self.next_vblank_cycle.wrapping_add(VBLANK_PERIOD_CYCLES);
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
    pub fn external_interrupt_pending(&self) -> bool {
        self.irq.pending()
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
        let otc_ran = self.dma.run_otc(&mut self.ram[..]);
        let gpu_ran = self.run_dma_gpu();
        if otc_ran || gpu_ran {
            self.irq.raise(IrqSource::Dma);
        }
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
    /// Both variants clear `CHCR.start` + `busy` bits on completion
    /// so BIOS polling sees "done".
    fn run_dma_gpu(&mut self) -> bool {
        let ch = &self.dma.channels[2];
        if (ch.channel_control >> 24) & 1 == 0 {
            return false;
        }
        let sync_mode = (ch.channel_control >> 9) & 0x3;
        match sync_mode {
            1 => self.dma_gpu_block(),
            2 => self.dma_gpu_linked_list(),
            _ => {
                // Manual (0) + prohibited (3) — nothing standard uses
                // them for the GPU channel. Drop the trigger silently.
            }
        }
        let ch = &mut self.dma.channels[2];
        ch.channel_control &= !((1 << 24) | (1 << 28));
        true
    }

    fn dma_gpu_block(&mut self) {
        let ch = self.dma.channels[2];
        let mut addr = ch.base & 0x001F_FFFC;
        let bcr = ch.block_control;
        let block_size = bcr & 0xFFFF;
        let block_count = (bcr >> 16) & 0xFFFF;
        let total_words = block_size * block_count.max(1);
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
    }

    fn dma_gpu_linked_list(&mut self) {
        let mut addr = self.dma.channels[2].base & 0x001F_FFFC;
        // Bound the walk so a malformed list can't spin forever.
        for _ in 0..0x100_0000 {
            let header = read_ram_u32(&self.ram[..], addr);
            let word_count = (header >> 24) & 0xFF;
            for i in 0..word_count {
                let word_addr = addr.wrapping_add(4 + i * 4);
                let word = read_ram_u32(&self.ram[..], word_addr);
                self.gpu.gp0_push(word);
            }
            let next = header & 0x00FF_FFFF;
            if next == 0x00FF_FFFF {
                return;
            }
            addr = next;
        }
    }

    /// Non-panicking byte read. Returns `None` for addresses outside
    /// any currently-mapped region. Diagnostic UIs (memory viewer,
    /// disassembler) use this to browse arbitrary ranges without
    /// crashing the emulator on unmapped addresses.
    ///
    /// Byte-granular reads of the GPU / timer / DMA / IRQ MMIO don't
    /// try to decompose the typed 32-bit registers — they return the
    /// echo-buffer byte, which is fine for a diagnostic dump.
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
        if (memory::expansion1::BASE
            ..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return Some(0xFF);
        }
        if (memory::expansion2::BASE
            ..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
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
        if (memory::expansion1::BASE
            ..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return 0xFF;
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            return self.io[(phys - memory::io::BASE) as usize];
        }
        if (memory::expansion2::BASE
            ..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return 0xFF;
        }
        panic!("bus: unmapped read8 @ virt={virt:#010x} phys={phys:#010x}");
    }

    /// Read a 16-bit little-endian half-word from a virtual address.
    /// Unmapped regions behave identically to [`Bus::read8`] (see the
    /// region-by-region notes there).
    ///
    /// `&mut self` for the same reason as `read8` — peripheral-side
    /// effects.
    pub fn read16(&mut self, virt: u32) -> u16 {
        let phys = to_physical(virt);
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
        if (memory::expansion1::BASE
            ..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
            .contains(&phys)
        {
            return 0xFFFF;
        }
        if Spu::contains(phys) {
            return self.spu.read16(phys);
        }
        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let off = (phys - memory::io::BASE) as usize;
            return u16::from_le_bytes([self.io[off], self.io[off + 1]]);
        }
        if (memory::expansion2::BASE
            ..memory::expansion2::BASE + memory::expansion2::SIZE as u32)
            .contains(&phys)
        {
            return 0xFFFF;
        }
        panic!("bus: unmapped read16 @ virt={virt:#010x} phys={phys:#010x}");
    }

    /// Read a 32-bit little-endian word from a virtual address. This is
    /// the instruction-fetch path.
    ///
    /// `&mut self` because CD-ROM byte reads (composited into a u32 for
    /// the rare case software word-accesses that range) mutate.
    pub fn read32(&mut self, virt: u32) -> u32 {
        let phys = to_physical(virt);

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

        if (memory::expansion1::BASE
            ..memory::expansion1::BASE + memory::expansion1::SIZE as u32)
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
            return self.spu.read32(phys);
        }
        if CdRom::contains(phys) {
            // CD-ROM regs are 8-bit; word access composites them.
            let b0 = self.cdrom.read8(phys) as u32;
            let b1 = self.cdrom.read8(phys + 1) as u32;
            let b2 = self.cdrom.read8(phys + 2) as u32;
            let b3 = self.cdrom.read8(phys + 3) as u32;
            return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }

        if (memory::io::BASE..memory::io::BASE + memory::io::SIZE as u32).contains(&phys) {
            let offset = (phys - memory::io::BASE) as usize;
            return read_u32_le(&self.io[offset..]);
        }

        panic!("bus: unmapped read32 @ virt={virt:#010x} phys={phys:#010x}");
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

        if phys == IRQ_STAT_ADDR {
            self.irq.write_stat(value);
            return;
        }
        if phys == IRQ_MASK_ADDR {
            self.irq.write_mask(value);
            return;
        }
        if Timers::contains(phys) {
            self.timers.write32(phys, value);
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
            self.spu.write32(phys, value);
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

        panic!(
            "bus: unmapped write32 @ virt={virt:#010x} phys={phys:#010x} value={value:#010x}"
        );
    }

    /// Write a byte to a virtual address. Unmapped writes in MMIO /
    /// expansion / BIOS ranges are silently dropped (same rationale as
    /// [`Bus::write32`]).
    pub fn write8(&mut self, virt: u32, value: u8) {
        let phys = to_physical(virt);
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
        panic!("bus: unmapped write8 @ virt={virt:#010x} phys={phys:#010x} value={value:#04x}");
    }

    /// Write a 16-bit half-word to a virtual address. Same unmapped-region
    /// policy as [`Bus::write32`].
    pub fn write16(&mut self, virt: u32, value: u16) {
        let phys = to_physical(virt);
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
        if Spu::contains(phys) {
            self.spu.write16(phys, value);
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
        panic!("bus: unmapped write16 @ virt={virt:#010x} phys={phys:#010x} value={value:#06x}");
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
        u32::from_le_bytes([ram[offset], ram[offset + 1], ram[offset + 2], ram[offset + 3]])
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
        assert_eq!(bus.next_vblank_cycle(), FIRST_VBLANK_CYCLE + VBLANK_PERIOD_CYCLES);
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
