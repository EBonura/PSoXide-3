//! Use the cached Redux trace to find the first step where our
//! cycle count diverges from Redux's. Each `InstructionRecord`
//! carries the Redux-side `bus.cycles()` for that step; we run our
//! emulator in lock-step and compare after every instruction.
//!
//! Printing the first mismatch and the nearby context (the last 5
//! matching instructions, the current state of registers, the
//! instruction that caused the divergence) tells us exactly which
//! opcode is under- or over-charging cycles.
//!
//! Requires a cached trace at `target/parity-cache/`. Generate one
//! by running the existing parity tests once; they populate the
//! cache transparently.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_cycle_first_divergence --release
//! ```

use emulator_core::{Bus, Cpu};
use parity_oracle::cache;
use psx_iso::Disc;
use std::path::PathBuf;

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");

    // If PSOXIDE_DISC is set, mount that disc (and use the non-cached
    // trace). For now cache only exists for no-disc; disc-side trace
    // capture is a separate add.
    let disc_path = std::env::var("PSOXIDE_DISC").ok();
    if disc_path.is_some() {
        eprintln!(
            "[cycle_first_divergence] Note: cached traces are no-disc only; disc mode requires fresh capture."
        );
    }

    // Default: walk the longest cache on disk. Caller can cap the
    // walk via arg 1 (useful for localizing a known-early bug
    // without reading the whole 14 GiB file).
    let max_steps: Option<usize> = std::env::args().nth(1).and_then(|s| s.parse().ok());
    let dir = cache::default_dir();
    eprintln!("[cache] loading trace from {}", dir.display());
    let mut trace =
        cache::load_longest(&dir, &bios).expect("No cached trace — run `generate_trace` first");
    if let Some(cap) = max_steps {
        if trace.len() > cap {
            trace.truncate(cap);
            eprintln!("[cache] truncated walk to {cap} records");
        }
    }
    eprintln!("[cache] loaded {} records", trace.len());

    let mut bus = Bus::new(bios).expect("bus");
    if let Some(p) = disc_path {
        let disc_bytes = std::fs::read(&p).expect("disc readable");
        bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();

    // Walk step-by-step, comparing bus.cycles() + pc + gprs to the
    // cached Redux record at the same step.
    let mut last_match_step: Option<usize> = None;
    let mut reported = 0usize;
    // First 10 mismatches are printed in full; after that, just
    // headline counts so we don't spam.
    const FULL_REPORT_LIMIT: usize = 10;

    for (i, expected) in trace.iter().enumerate() {
        // Capture our state BEFORE step -- Redux's record is "the state
        // after step i retired", so we check AFTER we step too.
        //
        // Redux's oracle folds IRQ-handler instructions into the
        // trace record of the instruction that triggered the IRQ
        // (see `debug.cc`: process() returns early when entering ISR
        // with cause=0, so stepIn's breakpoint fires only after RFE).
        // Our `cpu.step` retires one instruction at a time -- user OR
        // ISR -- so we have to perform the same folding in the
        // walker, otherwise Redux's "step 19472417" (= NOP + entire
        // VBlank ISR) appears to us as just "NOP".
        let was_in_isr = cpu.in_isr();
        let mut our_rec = cpu.step_traced(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                let r = cpu.step_traced(&mut bus).expect("step");
                our_rec.tick = r.tick;
                our_rec.gprs = r.gprs;
            }
        }

        if our_rec.tick == expected.tick
            && our_rec.pc == expected.pc
            && our_rec.instr == expected.instr
            && our_rec.gprs == expected.gprs
        {
            last_match_step = Some(i);
            continue;
        }

        if reported < FULL_REPORT_LIMIT {
            println!();
            println!("=== Mismatch at step {i} ===");
            if our_rec.tick != expected.tick {
                println!(
                    "  tick:  ours={:>12}  redux={:>12}  delta={:+}",
                    our_rec.tick,
                    expected.tick,
                    our_rec.tick as i64 - expected.tick as i64,
                );
            }
            if our_rec.pc != expected.pc {
                println!(
                    "  pc:    ours=0x{:08x}  redux=0x{:08x}",
                    our_rec.pc, expected.pc
                );
            }
            if our_rec.instr != expected.instr {
                println!(
                    "  instr: ours=0x{:08x}  redux=0x{:08x}",
                    our_rec.instr, expected.instr,
                );
            }
            if our_rec.gprs != expected.gprs {
                let mut differ = 0;
                for r in 0..32 {
                    if our_rec.gprs[r] != expected.gprs[r] {
                        println!(
                            "  $r{r:>2}:   ours=0x{:08x}  redux=0x{:08x}",
                            our_rec.gprs[r], expected.gprs[r],
                        );
                        differ += 1;
                        if differ >= 4 {
                            println!("  ... (more register diffs suppressed)");
                            break;
                        }
                    }
                }
            }
            if let Some(last) = last_match_step {
                if last + 1 == i {
                    // First-ever divergence: dump the last matching record
                    // so the diff is self-contained.
                    println!("  Last matching record (step {last}):");
                    let prev = &trace[last];
                    println!(
                        "    tick={}  pc=0x{:08x}  instr=0x{:08x}",
                        prev.tick, prev.pc, prev.instr,
                    );
                }
            }
        }
        reported += 1;

        // Break early on first cycle-only mismatch since that tells us
        // everything we need to know (instruction was the same but
        // cycles differed -- scheduler / memory-region cost issue).
        if our_rec.tick != expected.tick
            && our_rec.pc == expected.pc
            && our_rec.instr == expected.instr
            && our_rec.gprs == expected.gprs
        {
            // This is exactly the case we want to find -- a cycle-only
            // divergence. Print opcode category + surrounding context
            // then stop.
            println!();
            println!("=== FIRST CYCLE-ONLY DIVERGENCE at step {i} ===");
            let op = (our_rec.instr >> 26) & 0x3F;
            let func = our_rec.instr & 0x3F;
            println!(
                "  instr 0x{:08x}  op=0x{:02x}  func=0x{:02x}  pc=0x{:08x}",
                our_rec.instr, op, func, our_rec.pc,
            );
            println!(
                "  cycle delta: {:+} (ours {} vs redux {})",
                our_rec.tick as i64 - expected.tick as i64,
                our_rec.tick,
                expected.tick,
            );
            return;
        }
        if reported >= 100 {
            println!();
            println!("Stopping after 100 mismatches at step {i}.");
            return;
        }
    }

    if reported == 0 {
        println!("Walked {} records with no mismatch.", trace.len());
    } else {
        println!(
            "Finished walk. {} mismatches found; first at step after {:?}.",
            reported, last_match_step,
        );
    }
}
