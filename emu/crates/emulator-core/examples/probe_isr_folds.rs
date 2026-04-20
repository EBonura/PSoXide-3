//! Walk the first N ISR folds in a window on both sides of the
//! oracle. An "ISR fold" is a step whose tick delta is larger than
//! a normal instruction (> 20 cycles), indicating the step ran an
//! IRQ handler to completion. Side-by-side listing of both
//! emulators' fold cycles lets us see which IRQ fires at a different
//! time even when the surrounding user code is in lockstep.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const TRACE_STEP_TIMEOUT: Duration = Duration::from_secs(30);
const FOLD_THRESHOLD: u64 = 20; // cycles; normal is 2-3

fn main() {
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(80_000_000);
    let window: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let disc_path = std::env::args().nth(3);

    let bios_path = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");

    // --- Ours ---
    eprintln!("[ours] fast-forwarding {start} steps, then tracing {window}...");
    let t0 = Instant::now();
    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.enable_irq_log(200);
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
    let mut prev_tick = bus.cycles();
    let mut our_folds: Vec<(u64, u64, u64)> = Vec::new(); // (step, cycle, delta)
    for i in 0..window {
        let was_in_isr = cpu.in_isr();
        let rec = cpu.step(&mut bus).expect("step");
        let mut tick = rec.tick;
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                let r = cpu.step(&mut bus).expect("isr step");
                tick = r.tick;
            }
        }
        let delta = tick - prev_tick;
        if delta > FOLD_THRESHOLD {
            our_folds.push((start + i as u64 + 1, tick, delta));
        }
        prev_tick = tick;
    }
    let our_log = bus.cdrom.cdrom_irq_log.clone();
    eprintln!(
        "[ours] done in {:.1}s: {} folds, {} CDROM IRQs in log",
        t0.elapsed().as_secs_f64(),
        our_folds.len(),
        our_log.len()
    );

    // --- Redux ---
    eprintln!("[redux] launching...");
    let t0 = Instant::now();
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios_path, lua).expect("Redux resolves");
    if let Some(ref p) = disc_path {
        config = config.with_disc(PathBuf::from(p));
    }
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");
    let ff_timeout = Duration::from_secs((start / 200_000).max(60));
    redux.run(start, ff_timeout).expect("ff");
    eprintln!(
        "[redux] ff done in {:.1}s, tracing {window} steps...",
        t0.elapsed().as_secs_f64()
    );
    let trace = redux
        .step(window, TRACE_STEP_TIMEOUT)
        .expect("step trace");
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();

    let mut redux_folds: Vec<(u64, u64, u64)> = Vec::new();
    let mut prev = if trace.is_empty() { 0 } else { trace[0].tick };
    // We want Redux's tick BEFORE the first trace step — that's
    // "our_tick_at_start" which is known from ours since both
    // emulators are in lockstep at that point.
    // Skip the first; can't compute delta without prior.
    for (i, rec) in trace.iter().enumerate().skip(1) {
        let delta = rec.tick - prev;
        if delta > FOLD_THRESHOLD {
            redux_folds.push((start + i as u64 + 1, rec.tick, delta));
        }
        prev = rec.tick;
    }
    eprintln!("[redux] {} folds detected", redux_folds.len());

    // --- Dump all folds on each side ---
    println!();
    println!("=== All ISR folds (ours) ===");
    println!("{:>4}  {:>12} {:>12}  {:>6}", "idx", "step", "cyc", "Δtick");
    for (i, &(s, c, d)) in our_folds.iter().enumerate() {
        println!("{i:>4}  {s:>12} {c:>12}  {d:>6}");
    }
    println!();
    println!("=== All ISR folds (redux) ===");
    println!("{:>4}  {:>12} {:>12}  {:>6}", "idx", "step", "cyc", "Δtick");
    for (i, &(s, c, d)) in redux_folds.iter().enumerate() {
        println!("{i:>4}  {s:>12} {c:>12}  {d:>6}");
    }
    println!();
    println!("=== Our CDROM IRQ log ===");
    let names = ["None", "DataReady", "Complete", "Acknowledge", "DataEnd", "Error"];
    for (i, &(cyc, ty)) in our_log.iter().enumerate() {
        let name = names.get(ty as usize).copied().unwrap_or("?");
        println!("  #{i:>3}  cyc={cyc:>12}  {name}");
    }
}
