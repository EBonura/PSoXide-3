//! Zoom in on a known-small divergence window by fast-forwarding
//! both emulators silently to a start step, then capturing per-step
//! records for a small window. Much faster than a full trace cache —
//! a 10 K-step window costs ~0.3 s of tracing vs 14 GiB of cache.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_fine_divergence -- 90140000 10000 "/path/to/game.bin"
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const TRACE_STEP_TIMEOUT: Duration = Duration::from_secs(30);

fn main() {
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_fine_divergence <start_step> <window> [disc]");
    let window: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let disc_path = std::env::args().nth(3);

    let bios_path = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");

    // --- Ours ---
    eprintln!("[ours] fast-forwarding {start} steps...");
    let t0 = Instant::now();
    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();
    for _ in 0..start {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }
    eprintln!(
        "[ours] at start step, pc=0x{:08x}, cycles={}, elapsed={:.1}s",
        cpu.pc(),
        bus.cycles(),
        t0.elapsed().as_secs_f64()
    );

    // Capture per-step records for window. Match Redux folding
    // semantics: preserve the USER-side pc+instr across the ISR
    // fold, only advance tick+gprs. Getting this wrong (overwriting
    // instr with an ISR-body instruction) produces phantom
    // divergences at the fold step.
    let mut our_records: Vec<psx_trace::InstructionRecord> = Vec::with_capacity(window as usize);
    for _ in 0..window {
        let was_in_isr = cpu.in_isr();
        let mut rec = cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                let r = cpu.step(&mut bus).expect("isr step");
                rec.tick = r.tick;
                rec.gprs = r.gprs;
            }
        }
        our_records.push(rec);
    }
    eprintln!("[ours] captured {window} records");

    // --- Redux ---
    eprintln!("[redux] launching, fast-forwarding {start} steps...");
    let t0 = Instant::now();
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios_path, lua).expect("Redux resolves");
    if let Some(ref p) = disc_path {
        config = config.with_disc(PathBuf::from(p));
    }
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");
    let ff_timeout = Duration::from_secs((start / 200_000).max(60));
    redux.run(start, ff_timeout).expect("fast-forward");
    eprintln!(
        "[redux] ff done in {:.1}s, tracing {window} steps...",
        t0.elapsed().as_secs_f64()
    );
    let trace = redux
        .step(window, TRACE_STEP_TIMEOUT)
        .expect("step");
    eprintln!("[redux] trace done, {} records", trace.len());
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();

    // --- Compare: full record match (tick+pc+instr+gprs) ---
    println!();
    println!("=== fine divergence scan from step {start} (+{window}) ===");
    let mut first: Option<usize> = None;
    for i in 0..window as usize {
        let o = &our_records[i];
        let r = &trace[i];
        if o.tick == r.tick && o.pc == r.pc && o.instr == r.instr && o.gprs == r.gprs {
            continue;
        }
        first = Some(i);
        break;
    }
    match first {
        None => println!("All {window} records match — divergence past window"),
        Some(idx) => {
            let absolute = start + idx as u64 + 1;
            println!();
            println!("First divergence at window idx {idx} = absolute step {absolute}");
            println!();
            let lo = idx.saturating_sub(6);
            let hi = (idx + 1).min(window as usize - 1);
            println!("--- last 6 matching + divergence ---");
            for j in lo..=hi {
                let o = &our_records[j];
                let r = &trace[j];
                let marker = if j == idx { ">>>" } else { "   " };
                println!(
                    "{marker} idx={j:>5} step={:>10} pc=0x{:08x} instr=0x{:08x} tick={:>10}",
                    start + j as u64 + 1,
                    o.pc, o.instr, o.tick,
                );
                if o.gprs != r.gprs {
                    for reg in 0..32 {
                        if o.gprs[reg] != r.gprs[reg] {
                            println!(
                                "         $r{reg:>2}: ours=0x{:08x}  redux=0x{:08x}",
                                o.gprs[reg], r.gprs[reg]
                            );
                        }
                    }
                }
                if o.pc != r.pc || o.instr != r.instr || o.tick != r.tick {
                    println!(
                        "         redux pc=0x{:08x} instr=0x{:08x} tick={:>10}",
                        r.pc, r.instr, r.tick,
                    );
                }
            }
        }
    }
}
