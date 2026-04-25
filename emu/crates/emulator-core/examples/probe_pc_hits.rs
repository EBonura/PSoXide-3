//! Log visits to selected PCs while booting a disc.

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{Bus, Cpu};
use std::path::Path;

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let limit = args
        .first()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(100_000_000);
    let pcs = args
        .get(1)
        .map(|s| parse_addrs(s))
        .filter(|v| !v.is_empty())
        .expect("usage: probe_pc_hits <steps> <pc[,pc...]> <disc.cue|disc.bin>");
    let disc_path = args
        .get(2)
        .expect("usage: probe_pc_hits <steps> <pc[,pc...]> <disc.cue|disc.bin>");

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = disc_support::load_disc_path(Path::new(disc_path)).expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    bus.attach_digital_pad_port1();
    let mut cpu = Cpu::new();

    let mut hits = 0usize;
    let hit_limit = std::env::var("PSOXIDE_PC_HIT_LIMIT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(200);

    for step in 1..=limit {
        let pc = cpu.pc();
        if pcs.contains(&pc) {
            hits += 1;
            println!(
                "hit={hits:>4} step={step:>10} cyc={:>12} pc=0x{pc:08x} instr=0x{:08x} \
                 ra=0x{:08x} sp=0x{:08x} a0=0x{:08x} a1=0x{:08x} a2=0x{:08x} a3=0x{:08x} \
                 v0=0x{:08x} v1=0x{:08x}",
                bus.cycles(),
                bus.peek_instruction(pc).unwrap_or(0),
                cpu.gpr(31),
                cpu.gpr(29),
                cpu.gpr(4),
                cpu.gpr(5),
                cpu.gpr(6),
                cpu.gpr(7),
                cpu.gpr(2),
                cpu.gpr(3),
            );
            if hits >= hit_limit {
                println!("stopping after {hits} hits");
                return;
            }
        }
        cpu.step(&mut bus).expect("step");
    }

    println!("done. hits={hits}");
}

fn parse_addrs(spec: &str) -> Vec<u32> {
    spec.split(',')
        .filter_map(|part| parse_u32(part.trim()))
        .collect()
}

fn parse_u32(text: &str) -> Option<u32> {
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else {
        text.parse().ok()
    }
}
