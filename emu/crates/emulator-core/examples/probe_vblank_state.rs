//! Print VBlank scheduler state around the 129th VBlank firing,
//! which is where we now diverge from Redux (step 31139235).
//! The question: at cycle 72764419 (step 31139234), what's our
//! pending VBlank target? And when does the scheduler actually
//! fire it? If the target is 72764422 as the formula predicts,
//! and our cycle at end of step 31139235 also equals 72764422,
//! the IRQ should raise at that step.

use emulator_core::scheduler::EventSlot;
use emulator_core::{Bus, Cpu};
use psx_iso::Disc;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = std::fs::read(
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin",
    )
    .expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    let mut cpu = Cpu::new();

    // Run to just before step 31139234 -- using ISR-folded step
    // indexing to match probe_cycle_first_divergence. When we enter
    // an ISR, we keep calling `cpu.step` until we're out, all
    // counted as a single folded step.
    let target = 31_139_233u64;
    let mut steps = 0u64;
    while steps < target {
        let was_in_isr = cpu.in_isr();
        if cpu.step(&mut bus).is_err() {
            break;
        }
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                if cpu.step(&mut bus).is_err() {
                    break;
                }
            }
        }
        steps += 1;
    }

    let vb_target = bus.scheduler.target(EventSlot::VBlank);
    println!(
        "Pre-step {}: our cycles={}, VBlank target={:?}",
        target,
        bus.cycles(),
        vb_target,
    );

    // Step one at a time and log cycle + VBlank-pending changes.
    for i in 0..15 {
        let step_n = target + 1 + i;
        let pre_cycles = bus.cycles();
        let pre_vbt = bus.scheduler.target(EventSlot::VBlank);
        let pre_istat = bus.irq().stat();
        let pre_pc = cpu.pc();
        // Step with fold -- if ISR fires during this step, consume
        // the whole handler as one folded step.
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        let mut folded_isr_len = 0;
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("step");
                folded_isr_len += 1;
            }
        }
        let post_cycles = bus.cycles();
        let post_vbt = bus.scheduler.target(EventSlot::VBlank);
        let post_istat = bus.irq().stat();
        println!(
            "step={:<9} pre_pc=0x{:08x} post_pc=0x{:08x} pre_cyc={pre_cycles:>12} post_cyc={post_cycles:>12} (+{})  vbt={pre_vbt:?} → {post_vbt:?}  istat=0x{pre_istat:08x}→0x{post_istat:08x}  isr_len={folded_isr_len}",
            step_n,
            pre_pc,
            cpu.pc(),
            post_cycles - pre_cycles,
        );
    }
}
