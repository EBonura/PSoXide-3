//! Run our CPU with the same aggregation the parity test uses, then
//! dump our records at specific step indices alongside Redux's cached
//! records. Used to narrow where a specific GPR diverges.
//!
//! ```bash
//! cargo run -p emulator-core --example parity_diff_at --release -- 19472843 5
//! ```

use emulator_core::{Bus, Cpu};
use parity_oracle::cache;
use psx_trace::InstructionRecord;
use std::env;
use std::fs;
use std::path::PathBuf;

fn our_trace_to(bios: Vec<u8>, n: usize) -> Vec<InstructionRecord> {
    let mut bus = Bus::new(bios).expect("BIOS");
    let mut cpu = Cpu::new();
    let mut records = Vec::with_capacity(n);
    while records.len() < n {
        let was_in_isr = cpu.in_isr();
        let mut rec = cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                let r = cpu.step(&mut bus).expect("step");
                rec.tick = r.tick;
                rec.gprs = r.gprs;
            }
        }
        records.push(rec);
    }
    records
}

fn main() {
    let step: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: parity_diff_at <step> [radius]");
    let radius: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let bios_path = env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = fs::read(&bios_path).expect("BIOS readable");

    // Redux: load cached trace (prefix-matched).
    let dir = cache::default_dir();
    let theirs =
        cache::load_prefix(&dir, &bios, step + radius + 1).expect("no cached trace long enough");

    // Ours: run the parity loop up to the same index.
    let ours = our_trace_to(bios, step + radius + 1);

    let lo = step.saturating_sub(radius);
    let hi = (step + radius).min(ours.len() - 1).min(theirs.len() - 1);

    for i in lo..=hi {
        let u = &ours[i];
        let t = &theirs[i];
        let match_pc = u.pc == t.pc;
        let match_instr = u.instr == t.instr;
        let match_tick = u.tick == t.tick;
        let match_gprs = u.gprs == t.gprs;
        let mark = if i == step { ">>>" } else { "   " };
        println!(
            "{mark} step={i:>8}  pc={}  instr={}  tick={}  gprs={}",
            if match_pc { "== " } else { "!= " },
            if match_instr { "== " } else { "!= " },
            if match_tick { "== " } else { "!= " },
            if match_gprs { "== " } else { "!= " },
        );
        if !match_pc {
            println!("      pc:    ours=0x{:08x}  theirs=0x{:08x}", u.pc, t.pc);
        }
        if !match_tick {
            println!("      tick:  ours={:>10}  theirs={:>10}  delta={}", u.tick, t.tick, u.tick as i64 - t.tick as i64);
        }
        if !match_gprs {
            for r in 0..32 {
                if u.gprs[r] != t.gprs[r] {
                    println!(
                        "      $r{r:>2}:   ours=0x{:08x}  theirs=0x{:08x}",
                        u.gprs[r], t.gprs[r],
                    );
                }
            }
        }
    }
}
