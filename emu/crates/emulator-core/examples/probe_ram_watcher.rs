//! Run to just before step N, then step instruction-by-instruction
//! (no ISR folding) for a window, logging any change to a watched
//! RAM word. Tells us which PC (= which code) wrote the byte.

use emulator_core::{Bus, Cpu};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let stop_before: u64 = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(90_146_543);
    let window: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5_000);
    let addr: u32 = args
        .get(2)
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x80059E10);
    let disc_path = args.get(3).cloned();

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();

    // Fast-forward to stop_before, WITH folding (same as fine probe).
    for _ in 0..stop_before {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }
    let initial = bus.peek_instruction(addr).unwrap_or(0);
    println!(
        "=== at step {stop_before}: pc=0x{:08x}, cycles={}, RAM[0x{addr:08x}]=0x{initial:08x} ===",
        cpu.pc(),
        bus.cycles()
    );
    println!();
    println!("stepping next {window} instructions (NO folding), watching RAM[0x{addr:08x}]...");
    let mut prev = initial;
    let mut in_isr_start: Option<u32> = None;
    for i in 0..window {
        let pc_before = cpu.pc();
        let was_in_isr = cpu.in_isr();
        if !was_in_isr && cpu.in_irq_handler() && in_isr_start.is_none() {
            in_isr_start = Some(pc_before);
        }
        cpu.step(&mut bus).expect("step");
        let now = bus.peek_instruction(addr).unwrap_or(0);
        if now != prev {
            let ctx = if cpu.in_irq_handler() { "ISR" } else { "USER" };
            println!(
                "  step +{i:>5}  pc=0x{pc_before:08x} [{ctx}]  RAM[0x{addr:08x}]: 0x{prev:08x} -> 0x{now:08x}"
            );
            prev = now;
        }
    }
    println!();
    println!("final: RAM[0x{addr:08x}] = 0x{prev:08x}");
}
