//! Trace every change to a given RAM word, step-by-step (no ISR
//! folding) so the actual writing instruction's PC is captured —
//! including writes from inside an IRQ handler.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_mem_writes -- 0x800ED294 89184518
//! ```

use emulator_core::{Bus, Cpu};

fn parse_hex(s: &str) -> u32 {
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(h, 16).expect("bad hex")
    } else {
        s.parse::<u32>().expect("bad decimal")
    }
}

fn main() {
    let target_addr: u32 = std::env::args()
        .nth(1)
        .as_deref()
        .map(parse_hex)
        .expect("usage: probe_mem_writes <addr> <up_to_step>");
    let up_to_step: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    let target_word = target_addr & !3;

    let mut last = bus.peek_instruction(target_word).unwrap_or(0);
    let mut hits = 0usize;

    // Pure single-stepping without ISR folding, so we can attribute
    // each write to the exact PC (user or ISR). `up_to_step` is
    // interpreted as user-side-equivalent budget — we cap total steps
    // at `up_to_step * 2` to allow for in-ISR time.
    let max_raw_steps = up_to_step.saturating_mul(2);
    for raw_step in 1..=max_raw_steps {
        let pre_pc = cpu.pc();
        cpu.step(&mut bus).expect("step");
        let w = bus.peek_instruction(target_word).unwrap_or(0);
        if w != last {
            hits += 1;
            let b = w.to_le_bytes();
            let o = last.to_le_bytes();
            println!(
                "raw_step={raw_step:>11}  cyc={:>12}  pc=0x{pre_pc:08x}  \
                 in_isr={}  word 0x{w:08x} ({:02x} {:02x} {:02x} {:02x}) \
                 was 0x{last:08x} ({:02x} {:02x} {:02x} {:02x})",
                bus.cycles(),
                cpu.in_irq_handler(),
                b[0], b[1], b[2], b[3],
                o[0], o[1], o[2], o[3],
            );
            last = w;
            if hits >= 200 {
                println!("... stopping after 200 hits");
                return;
            }
        }
    }
    println!("done. {hits} writes to {target_word:#010x}");
}
