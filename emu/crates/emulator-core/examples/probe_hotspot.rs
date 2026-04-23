//! Find the busiest PC and what its instruction stream looks like.
//! Used to diagnose hangs: if one PC accounts for 80% of the last
//! million steps, the game is spinning there, and we need to know
//! what it's waiting for.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_hotspot -- 500000000 "/path/to/game.bin"
//! ```

use std::collections::HashMap;

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
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();

    // Run to within 1M of target.
    let warmup = n.saturating_sub(1_000_000);
    for _ in 0..warmup {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }
    println!(
        "warmup done @ step {warmup}, pc=0x{:08x}, cycles={}",
        cpu.pc(),
        bus.cycles()
    );

    // Sample last 1M.
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for _ in 0..1_000_000 {
        let was_in_isr = cpu.in_isr();
        let pc = cpu.pc();
        *counts.entry(pc).or_insert(0) += 1;
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

    println!();
    println!("=== top 20 PCs over last 1M steps ===");
    println!("{:>10}  {:<12}", "count", "pc");
    let top = sorted.len().min(20);
    for (pc, c) in &sorted[..top] {
        println!("{c:>10}  0x{pc:08x}");
    }
    println!();
    println!("unique PCs: {}", sorted.len());
    let top5_sum: u64 = sorted.iter().take(5).map(|(_, c)| c).sum();
    println!(
        "top-5 concentration: {:.1}%",
        100.0 * top5_sum as f64 / 1_000_000.0
    );
}
