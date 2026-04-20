//! Walk the first N CDROM IRQ raises side-by-side between ours and
//! Redux, pinpoint the first one that fires at a different cycle.
//! Output shows both emulators' logs aligned by index with a Δcycle
//! column — the first row with |Δ| > 0 is the culprit IRQ.
//!
//! Expected usage:
//! ```bash
//! cargo run --release -p emulator-core --example probe_cdrom_irq_timing -- 100000000 50 "/path/to/game.bin"
//! ```
//!
//! Runs silently on both sides; ours takes ~2 s for 100 M steps,
//! Redux ~3 min.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const TIMEOUT: Duration = Duration::from_secs(900);

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);
    let max_log: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let disc_path = std::env::args().nth(3);

    let bios_path = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");

    // --- Ours ---
    eprintln!("[ours] running {n} steps with irq_log enabled (max {max_log})...");
    let t0 = Instant::now();
    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.enable_irq_log(max_log as usize);
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
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
    let our_log = bus.cdrom.cdrom_irq_log.clone();
    eprintln!(
        "[ours] captured {} CDROM IRQs in {:.1}s",
        our_log.len(),
        t0.elapsed().as_secs_f64()
    );

    // --- Redux ---
    eprintln!("[redux] launching, then running {n} steps silently...");
    let t0 = Instant::now();
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios_path, lua).expect("Redux resolves");
    if let Some(ref p) = disc_path {
        config = config.with_disc(PathBuf::from(p));
    }
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");
    let redux_log = redux
        .log_cdrom_irqs(n, max_log, TIMEOUT)
        .expect("log_cdrom_irqs");
    eprintln!(
        "[redux] captured {} CDROM IRQs in {:.1}s",
        redux_log.len(),
        t0.elapsed().as_secs_f64()
    );
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();

    // --- Compare ---
    println!();
    println!("=== CDROM IRQ timing comparison (first {max_log}) ===");
    println!(
        "{:>4}  {:<9}  {:>13}  {:>13}  {:>10}",
        "idx", "type", "ours_tick", "redux_tick", "Δtick"
    );
    println!("{}", "-".repeat(60));
    let irq_names = [
        "NONE", "DataReady", "Complete", "Acknowledge", "DataEnd", "Error",
    ];
    let pairs = our_log.len().min(redux_log.len());
    let mut first_diff: Option<usize> = None;
    for i in 0..pairs {
        let (ot, oty) = our_log[i];
        let (_rs, rt, rty) = redux_log[i];
        let delta = ot as i64 - rt as i64;
        let oname = irq_names
            .get(oty as usize)
            .copied()
            .unwrap_or("?");
        let marker = if delta != 0 && first_diff.is_none() {
            first_diff = Some(i);
            " <<<"
        } else {
            ""
        };
        let flags = if oty != rty {
            format!(" (type diff: redux={})", irq_names.get(rty as usize).copied().unwrap_or("?"))
        } else {
            String::new()
        };
        println!(
            "{i:>4}  {oname:<9}  {ot:>13}  {rt:>13}  {delta:>+10}{marker}{flags}"
        );
    }
    if our_log.len() != redux_log.len() {
        println!();
        println!(
            "count diff: ours={}, redux={}",
            our_log.len(),
            redux_log.len()
        );
    }
    match first_diff {
        None => println!("\nNo timing divergence in the first {pairs} IRQs."),
        Some(i) => {
            let (ot, oty) = our_log[i];
            let (_rs, rt, rty) = redux_log[i];
            println!();
            println!(
                "First timing divergence: IRQ #{i} (type {}, ours={}, redux={})",
                irq_names.get(oty as usize).copied().unwrap_or("?"),
                ot,
                rt,
            );
            let _ = rty;
        }
    }
}
