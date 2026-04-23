//! Run a retail disc to step N and dump the SPU + DMA4 state. Useful
//! when a game blocks on a BIOS SPU event and we need to know whether
//! the sound hardware ever armed the relevant IRQ/transfer path.

use emulator_core::spu::{IRQ_ADDR, SPUCNT, SPUSTAT, TRANSFER_ADDR};
use emulator_core::{spu::SAMPLE_CYCLES, Bus, Cpu};

const DMA4_MADR: u32 = 0x1F80_10C0;
const DMA4_BCR: u32 = 0x1F80_10C4;
const DMA4_CHCR: u32 = 0x1F80_10C8;

fn main() {
    let step: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_spu_state_at <step> <disc.bin>");
    let disc_path = std::env::args()
        .nth(2)
        .expect("usage: probe_spu_state_at <step> <disc.bin>");
    let pump_spu = std::env::var_os("PSOXIDE_PUMP_SPU").is_some();

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc_bytes = std::fs::read(disc_path).expect("disc readable");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom
        .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    let mut cpu = Cpu::new();
    let mut audio_cycle_accum = 0u64;

    for _ in 0..step {
        let cycles_before = bus.cycles();
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                let isr_before = bus.cycles();
                cpu.step(&mut bus).expect("isr step");
                if pump_spu {
                    pump_spu_by_cycles(&mut bus, &mut audio_cycle_accum, isr_before);
                }
            }
        }
        if pump_spu {
            pump_spu_by_cycles(&mut bus, &mut audio_cycle_accum, cycles_before);
        }
    }

    println!("step            : {step}");
    println!("cycles          : {}", bus.cycles());
    println!("pc              : 0x{:08x}", cpu.pc());
    println!(
        "istat / imask   : 0x{:03x} / 0x{:03x}",
        bus.irq().stat(),
        bus.irq().mask()
    );
    println!("spu_irq_raises  : {}", bus.irq().raise_counts()[9]);
    println!("samples_prod    : {}", bus.spu.samples_produced());
    println!("spucnt          : 0x{:04x}", bus.spu.read16(SPUCNT));
    println!("spustat         : 0x{:04x}", bus.spu.read16(SPUSTAT));
    println!("irq_addr        : 0x{:04x}", bus.spu.read16(IRQ_ADDR));
    println!("transfer_addr   : 0x{:04x}", bus.spu.read16(TRANSFER_ADDR));
    println!("dma4.madr       : 0x{:08x}", bus.read32(DMA4_MADR));
    println!("dma4.bcr        : 0x{:08x}", bus.read32(DMA4_BCR));
    println!("dma4.chcr       : 0x{:08x}", bus.read32(DMA4_CHCR));
}

fn pump_spu_by_cycles(bus: &mut Bus, audio_cycle_accum: &mut u64, cycles_before: u64) {
    *audio_cycle_accum =
        audio_cycle_accum.saturating_add(bus.cycles().saturating_sub(cycles_before));
    let sample_count = (*audio_cycle_accum / SAMPLE_CYCLES) as usize;
    *audio_cycle_accum %= SAMPLE_CYCLES;
    if sample_count > 0 {
        bus.run_spu_samples(sample_count);
        let _ = bus.spu.drain_audio();
    }
}
