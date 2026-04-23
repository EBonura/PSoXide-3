//! Count IRQ raises per source while running a disc for N steps.
//! Used to spot a runaway IRQ (one source firing ~10× more than
//! the others is a classic "ISR re-entrance" signature).
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_irq_counts -- 500000000 "/path/to/game.bin"
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000_000);
    let disc_path = std::env::args().nth(2);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc readable");
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

    let counts = bus.irq().raise_counts();
    let names = [
        "VBlank",
        "GPU",
        "CDROM",
        "DMA",
        "Timer0",
        "Timer1",
        "Timer2",
        "Controller",
        "SIO",
        "SPU",
        "Lightpen",
    ];
    println!("=== IRQ raises at step {n} (cycles={}) ===", bus.cycles());
    println!("{:<12} {:>14}", "source", "raises");
    println!("{}", "-".repeat(30));
    for (name, count) in names.iter().zip(counts.iter()) {
        println!("{name:<12} {count:>14}");
    }
    let total: u64 = counts.iter().sum();
    println!("{}", "-".repeat(30));
    println!("{:<12} {:>14}", "total", total);
    println!();
    println!(
        "cycles/step ratio: {:.2} (normal: ~2.3)",
        bus.cycles() as f64 / n as f64
    );
}
