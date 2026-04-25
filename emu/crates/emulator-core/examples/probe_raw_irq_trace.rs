//! Dump the raw CPU instructions hidden inside one folded user step.
//! `probe_fine_divergence` folds IRQ handlers into the user-side
//! record to match Redux's oracle protocol; this diagnostic shows the
//! local ISR path when a folded record has a cycle-only mismatch.

#[path = "support/disc.rs"]
mod disc_support;

use std::collections::HashMap;
use std::path::PathBuf;

use emulator_core::{Bus, Cpu};
#[cfg(feature = "trace-mmio")]
use emulator_core::{MmioKind, Sio0};

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

fn main() {
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_raw_irq_trace <completed_user_steps> <disc.cue|disc.bin>");
    let disc_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .expect("usage: probe_raw_irq_trace <completed_user_steps> <disc.cue|disc.bin>");

    let bios = std::fs::read(DEFAULT_BIOS).expect("BIOS");
    let disc = disc_support::load_disc_path(&disc_path).expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    let mut cpu = Cpu::new();

    for _ in 0..start {
        step_user(&mut cpu, &mut bus);
    }
    bus.set_dma_log_enabled(std::env::var("PSOXIDE_RAW_DMA_LOG").is_ok());

    println!(
        "at user_step={start} pc=0x{:08x} cycles={} istat=0x{:03x} imask=0x{:03x} dicr=0x{:08x} cdflag=0x{:02x} lastcmd=0x{:02x}",
        cpu.pc(),
        bus.cycles(),
        bus.irq().stat(),
        bus.irq().mask(),
        bus.read32(0x1f80_10f4),
        bus.cdrom.irq_flag(),
        bus.cdrom.last_command()
    );

    let before_cycles = bus.cycles();
    let rec = cpu.step(&mut bus).expect("next user step");
    println!(
        "user raw pc=0x{:08x} instr=0x{:08x} tick={} next_pc=0x{:08x} in_irq={} istat=0x{:03x} cdflag=0x{:02x} lastcmd=0x{:02x}",
        rec.pc,
        rec.instr,
        rec.tick,
        cpu.pc(),
        cpu.in_irq_handler() as u8,
        bus.irq().stat(),
        bus.cdrom.irq_flag(),
        bus.cdrom.last_command()
    );

    let max_raw = std::env::var("PSOXIDE_RAW_IRQ_MAX")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(2_000);
    let print_range = std::env::var("PSOXIDE_RAW_IRQ_RANGE")
        .ok()
        .and_then(|s| parse_range(&s));
    let summarize = std::env::var("PSOXIDE_RAW_SUMMARY").is_ok();
    let mut pc_counts = HashMap::<u32, u64>::new();
    let mut raw = 0u64;
    while cpu.in_irq_handler() {
        let pc = cpu.pc();
        if summarize {
            *pc_counts.entry(pc).or_insert(0) += 1;
        }
        let instr = bus.peek_instruction(pc).unwrap_or(0xdead_beef);
        let cycles_before = bus.cycles();
        let rec = cpu.step(&mut bus).expect("isr step");
        raw += 1;
        if print_range.map_or(true, |(lo, hi)| (lo..=hi).contains(&pc)) {
            println!(
                "isr {raw:>5} pc=0x{pc:08x} instr=0x{instr:08x} before={cycles_before} after={} rec_pc=0x{:08x} istat=0x{:03x} imask=0x{:03x} dicr=0x{:08x} cdflag=0x{:02x} v0=0x{:08x} a0=0x{:08x} t0=0x{:08x} t1=0x{:08x} ra=0x{:08x}",
                bus.cycles(),
                rec.pc,
                bus.irq().stat(),
                bus.irq().mask(),
                bus.read32(0x1f80_10f4),
                bus.cdrom.irq_flag(),
                cpu.gpr(2),
                cpu.gpr(4),
                cpu.gpr(8),
                cpu.gpr(9),
                cpu.gpr(31),
            );
        }
        for (slot, scheduled_at, delay, target) in bus.drain_dma_log() {
            println!(
                "dma raw={raw:>5} slot={slot} scheduled_at={scheduled_at} delay={delay} target={target}"
            );
        }
        if raw > max_raw {
            println!("stopping after {max_raw} raw ISR instructions");
            break;
        }
    }

    println!(
        "done raw_isr={raw} pc=0x{:08x} cycles={} delta={} istat=0x{:03x} imask=0x{:03x}",
        cpu.pc(),
        bus.cycles(),
        bus.cycles().saturating_sub(before_cycles),
        bus.irq().stat(),
        bus.irq().mask()
    );
    if summarize {
        let mut top = pc_counts.into_iter().collect::<Vec<_>>();
        top.sort_by_key(|&(pc, count)| (std::cmp::Reverse(count), pc));
        println!("top_pcs:");
        for (pc, count) in top.into_iter().take(32) {
            let instr = bus.peek_instruction(pc).unwrap_or(0);
            println!("  pc=0x{pc:08x} instr=0x{instr:08x} count={count}");
        }
    }
    dump_sio_mmio(&bus);
}

#[cfg(feature = "trace-mmio")]
fn dump_sio_mmio(bus: &Bus) {
    let entries = bus
        .mmio_trace
        .iter_chronological()
        .filter(|e| {
            Sio0::contains(e.addr)
                && matches!(
                    (e.addr - Sio0::BASE, e.kind),
                    (0x0, MmioKind::R8 | MmioKind::R16 | MmioKind::R32)
                        | (0x0, MmioKind::W8 | MmioKind::W16 | MmioKind::W32)
                        | (0x4, MmioKind::R16 | MmioKind::R32)
                        | (0x8, MmioKind::W16 | MmioKind::W32)
                        | (0xA, MmioKind::W16 | MmioKind::W32)
                        | (0xE, MmioKind::W16 | MmioKind::W32)
                )
        })
        .collect::<Vec<_>>();
    let skip = entries.len().saturating_sub(160);
    println!(
        "sio_mmio_tail count={} showing={}",
        entries.len(),
        entries.len() - skip
    );
    for e in &entries[skip..] {
        println!(
            "  sio cyc={:>12} {} addr=0x{:08x} value=0x{:08x}",
            e.cycle,
            e.kind.tag(),
            e.addr,
            e.value
        );
    }
}

#[cfg(not(feature = "trace-mmio"))]
fn dump_sio_mmio(_bus: &Bus) {}

fn parse_range(text: &str) -> Option<(u32, u32)> {
    let (lo, hi) = text.split_once('-')?;
    Some((parse_u32(lo)?, parse_u32(hi)?))
}

fn parse_u32(text: &str) -> Option<u32> {
    let s = text.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else {
        u32::from_str_radix(s, 16)
            .or_else(|_| s.parse::<u32>())
            .ok()
    }
}

fn step_user(cpu: &mut Cpu, bus: &mut Bus) {
    let was_in_isr = cpu.in_isr();
    cpu.step(bus).expect("step");
    if !was_in_isr && cpu.in_irq_handler() {
        while cpu.in_irq_handler() {
            cpu.step(bus).expect("isr step");
        }
    }
}
