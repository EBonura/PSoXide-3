//! Walk the cached Redux trace and print per-step tick deltas.
//! Useful to see whether cycle "jumps" are real or an artifact of
//! our reader's interpretation of the cache format.
//!
//! ```bash
//! cd emu/crates/emulator-core
//! cargo run --example probe_trace_ticks --release
//! ```

use parity_oracle::cache;
use std::path::PathBuf;

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");

    let dir = cache::default_dir();
    let trace = cache::load_prefix(&dir, &bios, 50_000_000).expect("No cached trace long enough");
    eprintln!("Loaded {} records", trace.len());

    // Print the steps around 19_472_416 where we saw the jump.
    let window = 19_474_540usize..19_474_560;
    println!("{:<12} {:>12} {:>8}  {:>10}", "step", "tick", "delta", "pc");
    let mut prev_tick = 0u64;
    for i in window {
        let r = &trace[i];
        let delta = r.tick as i64 - prev_tick as i64;
        println!(
            "{:<12} {:>12} {:>+8}  0x{:08x}  instr=0x{:08x}",
            i, r.tick, delta, r.pc, r.instr,
        );
        prev_tick = r.tick;
    }
    println!();

    // Also scan the whole 50M trace for large per-step deltas (>20 cycles)
    // to see how often Redux "jumps".
    let mut jumps: Vec<(usize, u64, u64)> = Vec::new();
    let mut prev_tick = 0u64;
    for (i, r) in trace.iter().enumerate() {
        let delta = r.tick - prev_tick;
        if i > 0 && delta > 20 {
            jumps.push((i, prev_tick, r.tick));
        }
        prev_tick = r.tick;
    }
    println!(
        "Found {} per-step deltas > 20 cycles out of {}",
        jumps.len(),
        trace.len()
    );
    for &(i, prev, curr) in jumps.iter().take(20) {
        println!("  step {i}: {prev} -> {curr} (+{})", curr - prev);
    }
    if jumps.len() > 20 {
        println!("  ... {} more", jumps.len() - 20);
    }

    // Per-instruction average for the whole trace.
    let total_cycles = trace.last().unwrap().tick;
    let steps = trace.len() as u64;
    println!();
    println!(
        "Average cycles/step over {steps} steps: {:.4}",
        total_cycles as f64 / steps as f64,
    );
}
