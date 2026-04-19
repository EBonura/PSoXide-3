//! Capture every DMA-completion schedule our emulator makes during
//! the first N steps of BIOS boot. Lets us compare what delta and
//! target each DMA gets vs what Redux would compute.
//!
//! The specific question this answers: at the first cycle-accounting
//! divergence (step 19474544 in the no-disc cached trace), WHICH DMA
//! channel completed late? With what `total_words` / delta? What was
//! our `self.cycles` at the time of the schedule call?
//!
//! ```bash
//! cargo run -p emulator-core --example probe_dma_schedules --release
//! ```

use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

fn main() {
    let steps: u64 = std::env::var("PSOXIDE_PROBE_STEPS_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(19_480_000);
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");

    let mut bus = Bus::new(bios).expect("bus");
    bus.set_dma_log_enabled(true);
    let mut cpu = Cpu::new();

    for _ in 0..steps {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }

    let log = bus.drain_dma_log();
    println!(
        "Ran {steps} steps, captured {} DMA schedule events.",
        log.len(),
    );
    println!();
    println!("{:>6}  {:<9}  {:>10}  {:>10}  {:>12}",
        "#", "kind", "cycle_now", "delta", "target");
    for (i, (kind, cycle, delta, target)) in log.iter().enumerate() {
        println!(
            "{:>6}  {kind:<9}  {cycle:>10}  {delta:>10}  {target:>12}",
            i + 1,
        );
    }
    println!();
    // Final cumulative info.
    println!("Final bus cycles: {}", bus.cycles());
}
