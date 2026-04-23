//! Run BIOS-only boot until the first non-silent SPU sample and dump
//! the recent SPU MMIO writes. Build with `--features trace-mmio` to
//! get the write trace.

use emulator_core::{
    spu::{KON_HI, KON_LO, SAMPLE_CYCLES},
    Bus, Cpu,
};

const BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

fn main() {
    let bios = std::fs::read(BIOS).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();
    let mut audio_cycle_accum = 0u64;
    let mut frames = 0u64;
    let mut prev_kon = read_kon(&bus);

    for user_step in 1..=100_000_000u64 {
        let was_in_isr = cpu.in_isr();
        if step_and_pump(
            user_step,
            &mut cpu,
            &mut bus,
            &mut audio_cycle_accum,
            &mut frames,
            &mut prev_kon,
        ) {
            return;
        }
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                if step_and_pump(
                    user_step,
                    &mut cpu,
                    &mut bus,
                    &mut audio_cycle_accum,
                    &mut frames,
                    &mut prev_kon,
                ) {
                    return;
                }
            }
        }
    }

    println!("No non-silent BIOS sample reached.");
}

fn step_and_pump(
    user_step: u64,
    cpu: &mut Cpu,
    bus: &mut Bus,
    audio_cycle_accum: &mut u64,
    frames: &mut u64,
    prev_kon: &mut u32,
) -> bool {
    let before = bus.cycles();
    cpu.step(bus).expect("cpu step");
    let after = bus.cycles();

    let kon = read_kon(bus);
    if *prev_kon != kon {
        println!(
            "KON change: step={user_step} cycle={after} old=0x{old:08x} new=0x{kon:08x}",
            old = *prev_kon
        );
        *prev_kon = kon;
    }

    *audio_cycle_accum = audio_cycle_accum.saturating_add(after - before);
    let sample_count = (*audio_cycle_accum / SAMPLE_CYCLES) as usize;
    *audio_cycle_accum %= SAMPLE_CYCLES;
    if sample_count == 0 {
        return false;
    }

    bus.run_spu_samples(sample_count);
    let samples = bus.spu.drain_audio();
    for (idx, &(l, r)) in samples.iter().enumerate() {
        if l != 0 || r != 0 {
            let frame = *frames + idx as u64;
            println!(
                "first nonzero: step={user_step} cycle={after} frame={frame} sample=({l},{r}) spucnt=0x{:04x}",
                bus.spu.spucnt()
            );
            dump_spu_trace(bus, after);
            return true;
        }
    }
    *frames += samples.len() as u64;
    false
}

fn read_kon(bus: &Bus) -> u32 {
    let lo = bus.spu.read16(KON_LO) as u32;
    let hi = bus.spu.read16(KON_HI) as u32;
    lo | (hi << 16)
}

#[cfg(feature = "trace-mmio")]
fn dump_spu_trace(bus: &Bus, cycle: u64) {
    use emulator_core::mmio_trace::MmioKind;

    let mut entries: Vec<_> = bus
        .mmio_trace
        .iter_chronological()
        .filter(|e| {
            (0x1F80_1C00..0x1F80_1E00).contains(&e.addr)
                && matches!(e.kind, MmioKind::W16 | MmioKind::W32)
        })
        .collect();
    let focus_start = cycle.saturating_sub(4_000_000);
    entries.retain(|e| e.cycle >= focus_start);
    let skip = entries.len().saturating_sub(160);
    println!("recent SPU writes ({} shown):", entries.len() - skip);
    for e in &entries[skip..] {
        println!(
            "  cyc={:>10} {} {:08x} {:08x}",
            e.cycle,
            e.kind.tag(),
            e.addr,
            e.value
        );
    }
}

#[cfg(not(feature = "trace-mmio"))]
fn dump_spu_trace(_bus: &Bus, _cycle: u64) {
    println!("SPU write trace unavailable; rebuild with --features trace-mmio");
}
