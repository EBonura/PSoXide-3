//! Trace Resident Evil 2's movie-stream state machine around the
//! `0x80032fcc` frame dequeue/decode function. This is intentionally
//! narrow: it logs the ring entry pointer, entry sequence word, return
//! value, and movie flag bytes that decide whether the game continues
//! or falls into the trap loop at `0x80032884`.

#[path = "support/disc.rs"]
mod disc_support;

use std::path::PathBuf;

use emulator_core::{Bus, Cpu};

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

fn main() {
    let steps: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(340_000_000);
    let disc_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .expect("usage: probe_re2_movie_trace <steps> <disc.cue|disc.bin>");

    let bios = std::fs::read(DEFAULT_BIOS).expect("BIOS");
    let disc = disc_support::load_disc_path(&disc_path).expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    bus.set_dma_log_enabled(true);
    let mut cpu = Cpu::new();

    let mut calls = 0u64;
    let mut last_istat = bus.irq().stat();
    let mut last_dicr = bus.read32(0x1f80_10f4);
    let mut last_flags = bus.read32(0x800d_7684);
    for step in 0..steps {
        let pc = cpu.pc();
        match pc {
            0x8003_2edc => {
                calls = calls.saturating_add(1);
                let sp = cpu.gpr(29);
                let data_ptr = bus.read32(sp.wrapping_add(0x20));
                let desc_ptr = bus.read32(sp.wrapping_add(0x24));
                let desc8 = bus.read32(desc_ptr.wrapping_add(8));
                let desc12 = bus.read32(desc_ptr.wrapping_add(12));
                let state = bus.read32(0x800d_7680);
                let flags = bus.read32(0x800d_7684);
                println!(
                    "init={calls:>3} step={step:>10} cyc={:>12} data=0x{data_ptr:08x} desc=0x{desc_ptr:08x} d8=0x{desc8:08x} d12=0x{desc12:08x} state=0x{state:08x} flags=0x{flags:08x}",
                    bus.cycles()
                );
            }
            0x8003_2fa8 => {
                let sp = cpu.gpr(29);
                let data_ptr = bus.read32(sp.wrapping_add(0x20));
                let desc_ptr = bus.read32(sp.wrapping_add(0x24));
                let flags = bus.read32(0x800d_7684);
                let current = bus.read32(0x800d_763c);
                println!(
                    "iret ={calls:>3} step={step:>10} cyc={:>12} data=0x{data_ptr:08x} desc=0x{desc_ptr:08x} current=0x{current:08x} flags=0x{flags:08x}",
                    bus.cycles()
                );
            }
            0x8003_3034 => {
                calls = calls.saturating_add(1);
                let sp = cpu.gpr(29);
                let data_ptr = bus.read32(sp.wrapping_add(0x10));
                let entry_ptr = bus.read32(sp.wrapping_add(0x14));
                let seq = bus.read32(entry_ptr.wrapping_add(8));
                let seq_prev = bus.read32(entry_ptr.wrapping_add(4));
                let state = bus.read32(0x800d_7680);
                let flags = bus.read32(0x800d_7684);
                let frame = bus.read16(0x800d_7688);
                println!(
                    "call={calls:>3} step={step:>10} cyc={:>12} data=0x{data_ptr:08x} entry=0x{entry_ptr:08x} e4=0x{seq_prev:08x} e8=0x{seq:08x} state=0x{state:08x} flags=0x{flags:08x} frame={frame}",
                    bus.cycles()
                );
            }
            0x8003_30f4 => {
                let sp = cpu.gpr(29);
                let data_ptr = bus.read32(sp.wrapping_add(0x10));
                let entry_ptr = bus.read32(sp.wrapping_add(0x14));
                let seq = bus.read32(entry_ptr.wrapping_add(8));
                let flags = bus.read32(0x800d_7684);
                println!(
                    "ret  ={calls:>3} step={step:>10} cyc={:>12} s0={} data=0x{data_ptr:08x} entry=0x{entry_ptr:08x} e8=0x{seq:08x} flags=0x{flags:08x}",
                    bus.cycles(),
                    cpu.gpr(16)
                );
            }
            0x8003_2860 => {
                let flags = bus.read32(0x800d_7684);
                println!(
                    "post ={calls:>3} step={step:>10} cyc={:>12} v0={} flags=0x{flags:08x}",
                    bus.cycles(),
                    cpu.gpr(2)
                );
            }
            0x8003_2884 => {
                let flags = bus.read32(0x800d_7684);
                println!(
                    "trap step={step:>10} cyc={:>12} flags=0x{flags:08x} movie=0x{:08x}",
                    bus.cycles(),
                    bus.read32(0x800d_7680)
                );
                break;
            }
            _ => {}
        }

        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        log_runtime(
            &mut bus,
            step,
            "run ",
            &mut last_istat,
            &mut last_dicr,
            &mut last_flags,
        );
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
                log_runtime(
                    &mut bus,
                    step,
                    "isr ",
                    &mut last_istat,
                    &mut last_dicr,
                    &mut last_flags,
                );
            }
        }
    }
}

fn log_runtime(
    bus: &mut Bus,
    step: u64,
    tag: &str,
    last_istat: &mut u32,
    last_dicr: &mut u32,
    last_flags: &mut u32,
) {
    for (kind, at, delay, target) in bus.drain_dma_log() {
        if at >= 700_000_000 || target >= 700_000_000 {
            println!(
                "dma  step={step:>10} cyc={at:>12} kind={kind:<8} delay={delay:<8} target={target}"
            );
        }
    }
    if bus.cycles() < 700_000_000 {
        return;
    }

    let flags = bus.read32(0x800d_7684);
    if flags != *last_flags {
        println!(
            "flag step={step:>10} cyc={:>12} flags=0x{flags:08x}",
            bus.cycles()
        );
        *last_flags = flags;
    }

    let istat = bus.irq().stat();
    let dicr = bus.read32(0x1f80_10f4);
    let dma_istat_changed = (istat ^ *last_istat) & 0x008 != 0;
    let mdec_or_cd_flags_changed = (dicr ^ *last_dicr) & 0x8b00_0000 != 0;
    if dma_istat_changed || mdec_or_cd_flags_changed {
        println!(
            "irq  step={step:>10} cyc={:>12} tag={tag} istat=0x{istat:03x} dicr=0x{dicr:08x}",
            bus.cycles()
        );
    }
    *last_istat = istat;
    *last_dicr = dicr;
}
