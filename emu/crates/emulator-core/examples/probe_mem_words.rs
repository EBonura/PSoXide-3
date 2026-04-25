//! Trace changes to a comma-separated list of RAM words while booting a disc.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_mem_words -- \
//!   414000000 0x801fe830,0x801fe834,0x801fe848 "/path/to/game.cue"
//! ```

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
    let addrs = args
        .get(1)
        .map(|s| parse_addrs(s))
        .filter(|v| !v.is_empty())
        .expect("usage: probe_mem_words <up_to_step> <addr[,addr...]> <disc.cue|disc.bin>");
    let disc_path = args
        .get(2)
        .expect("usage: probe_mem_words <up_to_step> <addr[,addr...]> <disc.cue|disc.bin>");

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = disc_support::load_disc_path(Path::new(disc_path)).expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    bus.attach_digital_pad_port1();
    let mut cpu = Cpu::new();

    let mut last = addrs
        .iter()
        .map(|&addr| peek_u32(&bus, addr & !3))
        .collect::<Vec<_>>();
    let mut hits = 0usize;
    let hit_limit = std::env::var("PSOXIDE_MEM_WORD_HIT_LIMIT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(400);

    for step in 1..=limit {
        let pc = cpu.pc();
        let ra = cpu.gpr(31);
        let sp = cpu.gpr(29);
        let a = [cpu.gpr(4), cpu.gpr(5), cpu.gpr(6), cpu.gpr(7)];
        cpu.step(&mut bus).expect("step");
        for (idx, &addr) in addrs.iter().enumerate() {
            let word_addr = addr & !3;
            let value = peek_u32(&bus, word_addr);
            if value != last[idx] {
                println!(
                    "step={step:>10} cyc={:>12} pc=0x{pc:08x} ra=0x{ra:08x} sp=0x{sp:08x} \
                     a0=0x{:08x} a1=0x{:08x} a2=0x{:08x} a3=0x{:08x} addr=0x{word_addr:08x} \
                     0x{:08x}->0x{value:08x}",
                    bus.cycles(),
                    a[0],
                    a[1],
                    a[2],
                    a[3],
                    last[idx],
                );
                last[idx] = value;
                hits += 1;
                if hits >= hit_limit {
                    println!("stopping after {hits} hits");
                    return;
                }
            }
        }
    }

    println!("done. {hits} changes");
    for (addr, value) in addrs.iter().zip(last.iter()) {
        println!("  0x{:08x}=0x{value:08x}", addr & !3);
    }
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

fn peek_u32(bus: &Bus, addr: u32) -> u32 {
    let b0 = bus.try_read8(addr).unwrap_or(0);
    let b1 = bus.try_read8(addr.wrapping_add(1)).unwrap_or(0);
    let b2 = bus.try_read8(addr.wrapping_add(2)).unwrap_or(0);
    let b3 = bus.try_read8(addr.wrapping_add(3)).unwrap_or(0);
    u32::from_le_bytes([b0, b1, b2, b3])
}
