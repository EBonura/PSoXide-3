//! Dump cached Redux trace records around a specific step. Used to
//! understand what Redux was doing a few instructions before a
//! parity divergence surfaces — registers, PC, instructions.
//!
//! ```bash
//! cargo run --example trace_inspect --release -- 19258368 20
//! ```
//! First argument: centre step. Second: radius (± steps).

use parity_oracle::cache;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let step: usize = args
        .first()
        .and_then(|s| s.parse().ok())
        .expect("usage: trace_inspect <step> [radius]");
    let radius: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);

    let bios_path = env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let bios = fs::read(&bios_path).expect("BIOS readable");
    let dir = cache::default_dir();
    let records = cache::load_prefix(&dir, &bios, step + radius + 1)
        .expect("no cached trace at or past that step");

    let lo = step.saturating_sub(radius);
    let hi = (step + radius).min(records.len() - 1);

    let full = env::var("FULL_GPR").is_ok();
    for (i, r) in records[lo..=hi].iter().enumerate() {
        let n = lo + i;
        let marker = if n == step { ">>>" } else { "   " };
        println!(
            "{marker} step={n:>10}  cyc={:>10}  pc=0x{:08x}  instr=0x{:08x}",
            r.tick, r.pc, r.instr,
        );
        if full || n == step {
            for row in 0..8 {
                let mut line = String::from("        ");
                for col in 0..4 {
                    let reg = row * 4 + col;
                    line.push_str(&format!("$r{reg:02}={:08x}  ", r.gprs[reg]));
                }
                println!("{line}");
            }
        }
    }
}
