//! Print OUR DMA schedule + completion events intermixed with a
//! Redux tick trace, to see exactly which cycle each side fires
//! the DMA IRQ. If Redux's fire cycle lands on an earlier target
//! than ours, we've got a scheduler-ordering bug; if Redux fires
//! at the same cycle but we've queued a later event that masks
//! it, we've got a queue-semantics bug.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_dma_timeline --release
//! ```

use emulator_core::{Bus, Cpu};
use parity_oracle::cache;
use std::path::PathBuf;

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");

    let dir = cache::default_dir();
    let trace = cache::load_prefix(&dir, &bios, 50_000_000).expect("trace");

    let mut bus = Bus::new(bios).expect("bus");
    bus.set_dma_log_enabled(true);
    let mut cpu = Cpu::new();

    // Run to 19_480_000 — past the first cycle divergence at 19474544.
    for _ in 0..19_480_000u64 {
        if cpu.step(&mut bus).is_err() { break; }
    }

    let log = bus.drain_dma_log();

    // Also identify Redux's big-tick-delta steps (ISR folds) in a
    // neighbouring window.
    let mut redux_folds = Vec::new();
    let mut prev_tick = 0u64;
    for (i, r) in trace.iter().enumerate() {
        let delta = r.tick - prev_tick;
        if i > 0 && delta > 200 {
            redux_folds.push((i, prev_tick, r.tick, delta));
        }
        prev_tick = r.tick;
    }

    println!("=== Our DMA schedules ===");
    println!("{:<9} {:>12} {:>8} {:>12}", "kind", "cycle_now", "delta", "target");
    for (kind, cycle, delta, target) in &log {
        println!("{kind:<9} {cycle:>12} {delta:>8} {target:>12}");
    }

    println!();
    println!("=== Redux trace folds (step N with tick delta > 200) ===");
    println!("(each is a step where Redux's cycle jumped — likely an ISR fold)");
    println!("{:<12} {:>12} {:>12} {:>8}", "step", "prev_tick", "new_tick", "delta");
    for (step, prev, new, delta) in redux_folds.iter().take(15) {
        println!("{step:<12} {prev:>12} {new:>12} {delta:>8}");
    }

    // Explicit: find Redux's ISR fold target cycles and compare to
    // our schedule targets.
    println!();
    println!("=== Matchup ===");
    let mut matched = 0;
    for (_kind, _cycle, _delta, target) in &log {
        if let Some((step, _, new_tick, _)) = redux_folds.iter().find(|(_, _, new_tick, _)| {
            // Redux's ISR fires when its tick jumps; the PRE-fold tick is the
            // cycle at which the ISR "started" and is ~1 instruction past
            // the target. Match if the Redux pre-fold tick is within 10
            // cycles of our target.
            (*new_tick as i64 - *target as i64).abs() < 2500
                // post-ISR tick is within ~2500 of our target
        }) {
            println!(
                "our_target={target:>12}  redux step={step:<10}  redux_tick={new_tick:>12}  delta={}",
                *new_tick as i64 - *target as i64,
            );
            matched += 1;
        }
    }
    println!("Matched {matched} of {} schedule events to Redux ISR folds.", log.len());
}
