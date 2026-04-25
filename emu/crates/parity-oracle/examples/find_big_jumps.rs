//! Scan the cached Redux trace for anomalously-large cycle jumps
//! between consecutive steps. A normal step is 2-3 cycles; jumps
//! larger than ~100 cycles suggest Redux's oracle collapsed a
//! sequence of instructions (e.g. an IRQ-handler body) into a
//! single record.

use parity_oracle::cache;
use std::env;
use std::fs;

fn main() {
    let bios_path = env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let bios = fs::read(&bios_path).expect("BIOS readable");
    let dir = cache::default_dir();
    let records = cache::load_prefix(&dir, &bios, 20_000_000).expect("no cached trace past 20M");

    let threshold: u64 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let max_print: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let mut count = 0usize;
    let mut printed = 0usize;
    for i in 1..records.len() {
        let delta = records[i].tick.saturating_sub(records[i - 1].tick);
        if delta >= threshold {
            count += 1;
            if printed < max_print {
                println!(
                    "step={:>9}  cyc={:>10}  delta={:>7}  pc=0x{:08x}  instr=0x{:08x}  prev_pc=0x{:08x}",
                    i, records[i].tick, delta, records[i].pc, records[i].instr, records[i - 1].pc,
                );
                printed += 1;
            }
        }
    }
    println!();
    println!(
        "Total steps with cycle delta >= {threshold}: {count} out of {}",
        records.len()
    );
}
