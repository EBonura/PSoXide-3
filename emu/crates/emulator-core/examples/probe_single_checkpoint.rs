//! Run N silent steps in both emulators once, compare final cycles.
//! For picking a reference point on a new game without a trace cache.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_single_checkpoint -- 100000000 "/path/to/game.bin"
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);
    let disc_path = std::env::args().nth(2);

    let bios_path = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");

    // --- Ours ---
    eprintln!("[ours] running {n} steps...");
    let t0 = Instant::now();
    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
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
    let our_cycles = bus.cycles();
    eprintln!(
        "[ours] done in {:.1}s, cycles = {our_cycles}",
        t0.elapsed().as_secs_f64()
    );

    // --- Redux silent run ---
    eprintln!("[redux] launching...");
    let t0 = Instant::now();
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios_path, lua).expect("Redux resolves");
    if let Some(ref p) = disc_path {
        config = config.with_disc(PathBuf::from(p));
    }
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux
        .handshake(Duration::from_secs(30))
        .expect("handshake");
    eprintln!("[redux] running {n} steps silently...");
    // Generous timeout: ~4 min per 100M steps at Redux's silent rate.
    let run_timeout = Duration::from_secs((n / 200_000).max(60));
    let redux_tick = redux.run(n, run_timeout).expect("run");
    eprintln!(
        "[redux] done in {:.1}s, tick = {redux_tick}",
        t0.elapsed().as_secs_f64()
    );
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(5));
    let _ = redux.terminate();

    // --- Compare ---
    let delta = our_cycles as i64 - redux_tick as i64;
    let pct = 100.0 * delta as f64 / (redux_tick as f64);
    println!();
    println!("=== {n}-step parity ({}) ===",
        disc_path.as_deref().unwrap_or("no-disc"));
    println!("our cycles : {our_cycles:>14}");
    println!("redux tick : {redux_tick:>14}");
    println!("delta      : {delta:>+14}");
    println!("pct_off    : {pct:>+14.3}%");
}
