//! Dump timer-bank state at a target step so we can diagnose
//! Timer 1 / Timer 2 drift against Redux's trace. Cycle count is
//! printed alongside so the counter value can be cross-checked
//! against Redux's cycle-derived `count = (now - cycleStart) / rate`.
//!
//! ```bash
//! cargo run -p emulator-core --example timer_at_step --release -- 19472446
//! ```

use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(19_472_446);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    for _ in 0..target {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    println!("=== Timer-bank state at step {target} ===");
    println!("cycles: {}", bus.cycles());
    println!();
    for i in 0..3 {
        let t = &bus.timers().timers[i];
        let counter = t.counter & 0xFFFF;
        let mode = t.mode & 0xFFFF;
        let target_val = t.target & 0xFFFF;
        let source = (mode >> 8) & 0x3;
        let source_name = match (i, source) {
            (0, 0) | (0, 2) => "system",
            (0, 1) | (0, 3) => "dotclk",
            (1, 0) | (1, 2) => "system",
            (1, 1) | (1, 3) => "hblank",
            (2, 0) | (2, 1) => "system",
            (2, 2) | (2, 3) => "sys/8",
            _ => "?",
        };
        println!(
            "Timer {i}: counter={counter:>5} (0x{counter:04x})  \
             mode=0x{mode:08x}  target={target_val:>5} (0x{target_val:04x})  \
             source={source} ({source_name})",
        );
        println!(
            "  bit0 sync_en={}  bit3 reset_at_target={}  bit4 irq_on_target={}  \
             bit5 irq_on_wrap={}  bit6 repeat={}  bit10 irq_active_low={}  \
             bit11 reached_target={}  bit12 reached_wrap={}",
            (mode >> 0) & 1,
            (mode >> 3) & 1,
            (mode >> 4) & 1,
            (mode >> 5) & 1,
            (mode >> 6) & 1,
            (mode >> 10) & 1,
            (mode >> 11) & 1,
            (mode >> 12) & 1,
        );
        println!(
            "  last_reset_cycle={}  mode_writes={}  cycles_since_reset={}",
            t.last_reset_cycle,
            t.mode_write_count,
            bus.cycles().saturating_sub(t.last_reset_cycle),
        );
    }
}
