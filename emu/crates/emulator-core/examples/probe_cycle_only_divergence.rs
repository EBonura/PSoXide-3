//! Find the first step where our tick diverges from Redux's *while
//! pc/instr/gprs still match* — a pure cycle-accounting bug (our CPU
//! ran the same code but charged a different number of cycles).
//! Those are the most actionable signals; a divergence accompanied
//! by register changes is downstream of an earlier cycle bug.
//!
//! Uses the same ISR folding as `probe_cycle_first_divergence` so
//! Redux's ISR-fold doesn't appear as a false divergence.

use emulator_core::{Bus, Cpu};
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

    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    let mut first_cycle_only: Option<(usize, i64)> = None;
    let mut first_full_divergence: Option<usize> = None;
    let mut cycle_only_count = 0usize;
    let mut cumulative_cycle_delta: i64 = 0;

    // Scan. Print first 5 cycle-only divergences in detail; headline
    // everything else.
    for (i, expected) in trace.iter().enumerate().take(50_000_000) {
        let was_in_isr = cpu.in_isr();
        let mut our_rec = cpu.step_traced(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                let r = cpu.step_traced(&mut bus).expect("step");
                our_rec.tick = r.tick;
                our_rec.gprs = r.gprs;
            }
        }

        let tick_diff = our_rec.tick != expected.tick;
        let pc_diff = our_rec.pc != expected.pc;
        let instr_diff = our_rec.instr != expected.instr;
        let gprs_diff = our_rec.gprs != expected.gprs;
        let non_tick_diff = pc_diff || instr_diff || gprs_diff;

        if tick_diff && !non_tick_diff {
            cycle_only_count += 1;
            let delta = our_rec.tick as i64 - expected.tick as i64;
            cumulative_cycle_delta = delta;
            if cycle_only_count <= 5 {
                println!();
                println!(
                    "=== CYCLE-ONLY divergence #{} at step {i} ===",
                    cycle_only_count
                );
                println!(
                    "  tick:  ours={:>12}  redux={:>12}  delta={:+}",
                    our_rec.tick, expected.tick, delta,
                );
                println!(
                    "  pc=0x{:08x}  instr=0x{:08x}  (op=0x{:02x})",
                    our_rec.pc,
                    our_rec.instr,
                    (our_rec.instr >> 26) & 0x3F,
                );
                // Also print last 5 records to see preceding instructions.
                if i >= 5 {
                    println!("  preceding 5 instructions:");
                    for k in (i - 5)..i {
                        let r = &trace[k];
                        println!(
                            "    step {k:>9}  tick={:>12}  pc=0x{:08x}  instr=0x{:08x}  op=0x{:02x}",
                            r.tick,
                            r.pc,
                            r.instr,
                            (r.instr >> 26) & 0x3F,
                        );
                    }
                }
            }
            if first_cycle_only.is_none() {
                first_cycle_only = Some((i, delta));
            }
        }

        if non_tick_diff && first_full_divergence.is_none() {
            first_full_divergence = Some(i);
            println!();
            println!("=== FULL divergence (pc/instr/gprs) at step {i} ===");
            println!(
                "  tick:  ours={:>12}  redux={:>12}  delta={:+}",
                our_rec.tick,
                expected.tick,
                our_rec.tick as i64 - expected.tick as i64,
            );
            println!(
                "  pc:    ours=0x{:08x}  redux=0x{:08x}",
                our_rec.pc, expected.pc
            );
            println!(
                "  By this point cumulative cycle drift was {:+} cycles",
                cumulative_cycle_delta
            );
            println!("  {cycle_only_count} cycle-only divergences happened before this.");
            break;
        }
    }

    if first_full_divergence.is_none() {
        println!();
        println!(
            "Walked all {} records without full divergence.",
            trace.len()
        );
    }
    println!();
    println!(
        "Total cycle-only divergences found: {} (before first full divergence).",
        cycle_only_count
    );
    if let Some((step, delta)) = first_cycle_only {
        println!(
            "First cycle-only divergence at step {step}, delta {:+}. Inspect above for context.",
            delta,
        );
    }
}
