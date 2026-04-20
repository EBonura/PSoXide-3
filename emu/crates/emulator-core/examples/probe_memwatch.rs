//! Run to step N with disc, report value at a RAM word. Also dumps
//! a range so you can see if it's a flag array or single cell.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_memwatch -- 500000000 0x800A64E0 4 "/path/to/game.bin"
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(500_000_000);
    let addr: u32 = args
        .get(1)
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .expect("need addr hex");
    let count: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let disc_path = args.get(3).cloned();

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();
    for _ in 0..n {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    println!("=== RAM @ 0x{addr:08x} after step {n} (cycles={}) ===", bus.cycles());
    for i in 0..count {
        let a = addr + i * 4;
        let w = bus.peek_instruction(a).unwrap_or(0xFFFFFFFF);
        println!("  0x{a:08x}: 0x{w:08x}");
    }
}
