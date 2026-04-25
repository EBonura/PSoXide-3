//! Step our CPU to a target step and then print a few-step window
//! showing pc/SR/CAUSE around a suspected IRQ-at-delay-slot bug.
//!
//! ```bash
//! cargo run -p emulator-core --example trace_window --release -- 19258530 10
//! ```

use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

fn main() {
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(19_258_530);
    let window: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    // Run up to `start` without tracing.
    for _ in 0..start {
        cpu.step(&mut bus).expect("step");
    }

    // Then step `window` more times, dumping state.
    println!("step  pc         sr         cause      epc        istat  imask  in_irq",);
    for i in 0..window {
        let pc_before = cpu.pc();
        let sr_before = cpu.cop0()[12];
        let cause_before = cpu.cop0()[13];
        let epc_before = cpu.cop0()[14];
        let istat = bus.irq().stat();
        let imask = bus.irq().mask();
        let in_irq = cpu.in_irq_handler();
        println!(
            "{:>4}  0x{:08x}  0x{:08x}  0x{:08x}  0x{:08x}  0x{:03x}    0x{:03x}  {}",
            start + i,
            pc_before,
            sr_before,
            cause_before,
            epc_before,
            istat,
            imask,
            in_irq as u8,
        );
        if let Err(e) = cpu.step(&mut bus) {
            println!("  step error: {e:?}");
            break;
        }
    }
    // One more line showing final state.
    println!(
        "{:>4}  0x{:08x}  0x{:08x}  0x{:08x}  0x{:08x}  0x{:03x}    0x{:03x}  {}",
        start + window,
        cpu.pc(),
        cpu.cop0()[12],
        cpu.cop0()[13],
        cpu.cop0()[14],
        bus.irq().stat(),
        bus.irq().mask(),
        cpu.in_irq_handler() as u8,
    );
}
