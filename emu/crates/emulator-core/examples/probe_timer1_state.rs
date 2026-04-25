//! Dump one root-counter state at a given step count. Used to
//! understand why a timer counter read differs from Redux:
//! is the mode different? Was a recent reset missed? Is accum in a
//! weird place?
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_timer1_state -- 79389318 1
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{Bus, Cpu};
use std::path::Path;

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(79_389_318);
    let timer_idx: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let disc_path = std::env::args().nth(3);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc = disc_support::load_disc_path(Path::new(p)).expect("disc");
        bus.cdrom.insert_disc(Some(disc));
    }
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

    let timer = &bus.timers.timers[timer_idx];
    let rate = match timer_idx {
        0 => bus.gpu.dot_clock_divisor(),
        1 => bus.hsync_cycles(),
        _ => 1,
    };
    println!("=== Timer {timer_idx} state at step {target} ===");
    println!("bus.cycles        = {}", bus.cycles());
    println!(
        "timer.counter     = {:#x} ({})",
        timer.counter, timer.counter
    );
    println!("timer.mode        = {:#x}", timer.mode);
    println!("timer.target      = {:#x} ({})", timer.target, timer.target);
    println!("timer.last_reset  = {}", timer.last_reset_cycle);
    println!("timer.mode_writes = {}", timer.mode_write_count);
    println!("rate              = {rate}");
    println!();
    // Re-derive what a "lazy" read would return.
    let since_reset = bus.cycles() - timer.last_reset_cycle;
    let lazy_counter = since_reset / rate;
    println!(
        "lazy counter      = (cycles - last_reset) / rate = ({} - {}) / {} = {}",
        bus.cycles(),
        timer.last_reset_cycle,
        rate,
        lazy_counter,
    );
    println!("(mod 0xFFFF)      = {}", lazy_counter & 0xFFFF);
}
