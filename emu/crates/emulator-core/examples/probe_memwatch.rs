//! Run to step N with disc, report value at a RAM word. Also dumps
//! a range so you can see if it's a flag array or single cell.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_memwatch -- 500000000 0x800A64E0 4 "/path/to/game.bin"
//! cargo run --release -p emulator-core --example probe_memwatch -- --fastboot 500000000 0x800A64E0 4 "/path/to/game.cue"
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{
    fast_boot_disc_with_hle, warm_bios_for_disc_fast_boot, Bus, Cpu, DISC_FAST_BOOT_WARMUP_STEPS,
};
use std::path::Path;

fn main() {
    let mut fastboot = false;
    let mut args = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--fastboot" {
            fastboot = true;
        } else {
            args.push(arg);
        }
    }
    let n: u64 = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000_000);
    let addr: u32 = args
        .get(1)
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .expect("need addr hex");
    let count: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let disc_path = args.get(3).cloned();

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();
    if let Some(ref p) = disc_path {
        let disc = disc_support::load_disc_path(Path::new(p)).expect("disc");
        if fastboot {
            warm_bios_for_disc_fast_boot(&mut bus, &mut cpu, DISC_FAST_BOOT_WARMUP_STEPS)
                .expect("BIOS warmup");
            let info =
                fast_boot_disc_with_hle(&mut bus, &mut cpu, &disc, false).expect("fast boot");
            println!(
                "fastboot={} entry=0x{:08x} payload={}B",
                info.boot_path, info.initial_pc, info.payload_len
            );
        }
        bus.cdrom.insert_disc(Some(disc));
        bus.attach_digital_pad_port1();
    }
    for _ in 0..n {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    println!(
        "=== RAM @ 0x{addr:08x} after step {n} (cycles={}) ===",
        bus.cycles()
    );
    for i in 0..count {
        let a = addr + i * 4;
        let w = bus.peek_instruction(a).unwrap_or_else(|| bus.read32(a));
        println!("  0x{a:08x}: 0x{w:08x}");
    }
}
