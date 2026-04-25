//! Stop at the first exception of a requested kind and print the
//! recent instruction window. Default is BREAK (ExcCode 9), useful for
//! retail loaders that jump into the BIOS fatal-error path.

#[path = "support/disc.rs"]
mod disc_support;

use std::collections::VecDeque;
use std::path::Path;

use emulator_core::{Bus, Cpu};

#[cfg(feature = "trace-mmio")]
use emulator_core::MmioKind;

#[derive(Clone)]
struct Recent {
    step: u64,
    pc: u32,
    instr: u32,
    tick: u64,
    gprs: [u32; 32],
    cycles: u64,
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let limit = args
        .first()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(500_000_000);
    let disc_path = args
        .get(1)
        .expect("usage: probe_first_exception <limit_steps> <disc.cue|disc.bin> [exc_code]");
    let exc_code = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(9);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = disc_support::load_disc_path(Path::new(disc_path)).expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    bus.attach_digital_pad_port1();
    let mut cpu = Cpu::new();
    let mut recent = VecDeque::with_capacity(48);
    let mut before = cpu.exception_counts()[exc_code];

    for step in 1..=limit {
        let cycles_before = bus.cycles();
        let rec = cpu.step(&mut bus).expect("step");
        if recent.len() == 48 {
            recent.pop_front();
        }
        recent.push_back(Recent {
            step,
            pc: rec.pc,
            instr: rec.instr,
            tick: rec.tick,
            gprs: rec.gprs,
            cycles: cycles_before,
        });

        let after = cpu.exception_counts()[exc_code];
        if after != before {
            println!("first exc_code={exc_code} at user_step={step}");
            println!(
                "cycles={} pc_after=0x{:08x} sr=0x{:08x} cause=0x{:08x} epc=0x{:08x} badv=0x{:08x}",
                bus.cycles(),
                cpu.pc(),
                cpu.cop0()[12],
                cpu.cop0()[13],
                cpu.cop0()[14],
                cpu.cop0()[8],
            );
            println!(
                "istat=0x{:03x} imask=0x{:03x} cd_irq(flag=0x{:02x},mask=0x{:02x}) read_lba={} fifo={} ready={} armed={} pending={}",
                bus.irq().stat(),
                bus.irq().mask(),
                bus.cdrom.irq_flag(),
                bus.cdrom.irq_mask_value(),
                bus.cdrom.debug_read_lba(),
                bus.cdrom.data_fifo_len(),
                bus.cdrom.data_fifo_ready(),
                bus.cdrom.data_transfer_armed(),
                bus.cdrom.pending_queue_len(),
            );
            if let Some((header, subheader)) = bus.cdrom.debug_last_sector() {
                println!("last_sector header={header:02x?} subheader={subheader:02x?}");
            }
            println!(
                "dma: dicr=0x{:08x} ch3_madr=0x{:08x} ch3_bcr=0x{:08x} ch3_chcr=0x{:08x}",
                bus.read32(0x1f80_10f4),
                bus.read32(0x1f80_10b0),
                bus.read32(0x1f80_10b4),
                bus.read32(0x1f80_10b8),
            );
            dump_mmio_tail(&bus);
            if let Ok(spec) = std::env::var("PSOXIDE_DUMP_MEM") {
                for (base, len) in parse_ranges(&spec) {
                    println!();
                    println!("mem 0x{base:08x}:0x{len:x}");
                    dump_mem(&bus, base, len);
                }
            }
            println!();
            println!("recent:");
            for entry in recent {
                println!(
                    "  step={:>10} cyc={:>12} tick={:>12} pc=0x{:08x} instr=0x{:08x} ra=0x{:08x} sp=0x{:08x} a0=0x{:08x} a1=0x{:08x}",
                    entry.step,
                    entry.cycles,
                    entry.tick,
                    entry.pc,
                    entry.instr,
                    entry.gprs[31],
                    entry.gprs[29],
                    entry.gprs[4],
                    entry.gprs[5],
                );
            }
            return;
        }
        before = after;
    }

    println!("no exc_code={exc_code} within {limit} steps");
}

#[cfg(feature = "trace-mmio")]
fn dump_mmio_tail(bus: &Bus) {
    let entries = bus
        .mmio_trace
        .iter_chronological()
        .filter(|entry| {
            (0x1f80_1800..=0x1f80_1803).contains(&entry.addr)
                || (0x1f80_10b0..=0x1f80_10b8).contains(&entry.addr)
                || matches!(entry.addr, 0x1f80_1070 | 0x1f80_1074)
        })
        .collect::<Vec<_>>();
    let start = entries.len().saturating_sub(160);
    println!(
        "mmio_tail selected={} showing={}",
        entries.len(),
        entries.len() - start
    );
    for entry in entries.into_iter().skip(start) {
        println!(
            "  cyc={:>12} {} addr=0x{:08x} value=0x{:08x}",
            entry.cycle,
            match entry.kind {
                MmioKind::R8 => "r8 ",
                MmioKind::R16 => "r16",
                MmioKind::R32 => "r32",
                MmioKind::W8 => "w8 ",
                MmioKind::W16 => "w16",
                MmioKind::W32 => "w32",
            },
            entry.addr,
            entry.value,
        );
    }
}

#[cfg(not(feature = "trace-mmio"))]
fn dump_mmio_tail(_bus: &Bus) {}

fn parse_ranges(spec: &str) -> Vec<(u32, u32)> {
    spec.split(',')
        .filter_map(|part| {
            let (base, len) = part.split_once(':')?;
            Some((parse_u32(base)?, parse_u32(len)?))
        })
        .collect()
}

fn parse_u32(text: &str) -> Option<u32> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else {
        text.parse().ok()
    }
}

fn dump_mem(bus: &Bus, base: u32, len: u32) {
    let end = base.saturating_add(len);
    let mut addr = base & !0xF;
    while addr < end {
        print!("  0x{addr:08x}:");
        for off in (0..16).step_by(4) {
            let cur = addr.wrapping_add(off);
            if cur >= base && cur < end {
                print!(" {:08x}", peek_u32(bus, cur));
            } else {
                print!("         ");
            }
        }
        println!();
        addr = addr.wrapping_add(16);
    }
}

fn peek_u32(bus: &Bus, addr: u32) -> u32 {
    let b0 = bus.try_read8(addr).unwrap_or(0);
    let b1 = bus.try_read8(addr.wrapping_add(1)).unwrap_or(0);
    let b2 = bus.try_read8(addr.wrapping_add(2)).unwrap_or(0);
    let b3 = bus.try_read8(addr.wrapping_add(3)).unwrap_or(0);
    u32::from_le_bytes([b0, b1, b2, b3])
}
