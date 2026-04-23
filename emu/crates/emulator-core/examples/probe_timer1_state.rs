//! Dump Timer 1 state at a given step count. Used to understand why
//! our Timer 1 counter reads 1 less than Redux's at step 79,389,318:
//! is the mode different? Was a recent reset missed? Is accum in a
//! weird place?
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_timer1_state -- 79389318
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(79_389_318);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
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

    let t1 = &bus.timers.timers[1];
    let hsync_period = bus.hsync_cycles();
    println!("=== Timer 1 state at step {target} ===");
    println!("bus.cycles        = {}", bus.cycles());
    println!("timer.counter     = {:#x} ({})", t1.counter, t1.counter);
    println!("timer.mode        = {:#x}", t1.mode);
    println!("timer.target      = {:#x} ({})", t1.target, t1.target);
    println!("timer.last_reset  = {}", t1.last_reset_cycle);
    println!("timer.mode_writes = {}", t1.mode_write_count);
    println!("hsync_period      = {hsync_period}");
    println!();
    // Re-derive what a "lazy" read would return.
    let since_reset = bus.cycles() - t1.last_reset_cycle;
    let lazy_counter = since_reset / hsync_period;
    println!(
        "lazy counter      = (cycles - last_reset) / rate = ({} - {}) / {} = {}",
        bus.cycles(),
        t1.last_reset_cycle,
        hsync_period,
        lazy_counter,
    );
    println!("(mod 0xFFFF)      = {}", lazy_counter & 0xFFFF);
}
