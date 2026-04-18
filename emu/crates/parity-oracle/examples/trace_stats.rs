//! Scan the cached Redux trace for high-level statistics: how many
//! times PC entered the exception vector (0x80000080), how many were
//! syscalls (next handler reads CAUSE shortly after), and roughly
//! how many were IRQs (the rest). Used to settle the question of
//! whether Redux ever took IRQs in regions where our PCs match.

use parity_oracle::cache;
use std::env;
use std::fs;

fn main() {
    let max: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let bios_path = env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let bios = fs::read(&bios_path).expect("BIOS readable");
    let dir = cache::default_dir();
    let want = if max == usize::MAX { 50_000_000 } else { max };
    let records = cache::load_prefix(&dir, &bios, want).expect("any cached trace");
    let limit = max.min(records.len());

    let mut vector_hits: Vec<usize> = Vec::new();
    for (i, r) in records[..limit].iter().enumerate() {
        if r.pc == 0x8000_0080 || r.pc == 0xBFC0_0180 {
            vector_hits.push(i);
        }
    }

    println!("scanned {limit} records");
    println!("exception-vector entries: {}", vector_hits.len());
    println!("first 20:");
    for &step in vector_hits.iter().take(20) {
        let r = &records[step];
        println!("  step={step:>10} pc=0x{:08x} cyc={}", r.pc, r.tick);
    }
    if vector_hits.len() > 20 {
        println!("last 5:");
        for &step in vector_hits.iter().rev().take(5).rev() {
            let r = &records[step];
            println!("  step={step:>10} pc=0x{:08x} cyc={}", r.pc, r.tick);
        }
    }
}
