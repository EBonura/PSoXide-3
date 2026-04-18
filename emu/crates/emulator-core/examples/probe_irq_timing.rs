//! Show our emulator's instruction flow + IRQ state in a narrow
//! window around step 19474543 — where the 20M parity test
//! diverges on `$r26`. Prints every step's PC, instr, cycles,
//! and IRQ-controller state so we can line up against Redux's
//! cached trace and see who fires an IRQ differently.

use emulator_core::{Bus, Cpu};

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("BIOS size");
    let mut cpu = Cpu::new();

    let start: usize = 19_474_530;
    let end: usize = 19_474_560;

    for retired in 0..=end {
        if retired >= start {
            let stat = bus.irq().stat();
            let mask = bus.irq().mask();
            println!(
                "step={retired:>9}  cyc={:>10}  pc=0x{:08x}  in_isr={}  istat=0x{:03x}  imask=0x{:03x}",
                bus.cycles(),
                cpu.pc(),
                cpu.in_isr(),
                stat,
                mask,
            );
        }
        if retired == end {
            break;
        }
        cpu.step(&mut bus).expect("step");
    }
}
