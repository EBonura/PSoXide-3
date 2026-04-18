//! Scan the cached Redux trace for entries to the general exception
//! handler (PC = 0x80000080) to verify whether Redux's oracle records
//! handler-body instructions or skips them — which determines whether
//! our parity harness needs to aggregate them.

use parity_oracle::cache;
use std::env;
use std::fs;

fn main() {
    let bios_path = env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let bios = fs::read(&bios_path).expect("BIOS readable");
    let dir = cache::default_dir();
    let records =
        cache::load_prefix(&dir, &bios, 20_000_000).expect("no cached trace past 20M");

    // Count IRQ entries and print the first few with a few context steps.
    let mut count = 0usize;
    let mut handler_starts: Vec<usize> = Vec::new();
    for (i, r) in records.iter().enumerate() {
        if r.pc == 0x8000_0080 {
            count += 1;
            if handler_starts.len() < 10 {
                handler_starts.push(i);
            }
        }
    }
    println!("Total 0x80000080 entries in trace: {count}");
    println!();
    for &start in &handler_starts {
        println!("--- IRQ entry at step {start} ---");
        let lo = start.saturating_sub(2);
        let hi = (start + 10).min(records.len() - 1);
        for i in lo..=hi {
            let r = &records[i];
            let mark = if i == start { ">>>" } else { "   " };
            println!(
                "{mark} step={i:>9}  cyc={:>10}  pc=0x{:08x}  instr=0x{:08x}",
                r.tick, r.pc, r.instr
            );
        }
        println!();
    }
}
