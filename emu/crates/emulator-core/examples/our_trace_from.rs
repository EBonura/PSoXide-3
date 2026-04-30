//! Dump our emulator's trace starting from a given step for N steps.
//! Helps cross-reference Redux's trace when we diverge -- especially
//! to see which path we took after a missed IRQ.

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{Bus, Cpu};
use std::path::Path;

fn main() {
    let from_step: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: our_trace_from <from_step> [count] [disc]");
    let count: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let disc_path = std::env::args().nth(3);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc = disc_support::load_disc_path(Path::new(p)).expect("disc");
        bus.cdrom.insert_disc(Some(disc));
    }
    let mut cpu = Cpu::new();

    // Fast-forward to from_step - 1 with ISR folding.
    for _ in 0..(from_step - 1) {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    // Emit step-by-step records for `count` steps with folding.
    for i in 0..count {
        let step_n = from_step + i;
        let pre_pc = cpu.pc();
        let pre_instr = bus.peek_instruction(pre_pc).unwrap_or(0);
        let pre_cyc = bus.cycles();

        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        let mut isr_cycles = 0u64;
        let mut isr_count = 0;
        if !was_in_isr && cpu.in_irq_handler() {
            let isr_start_cyc = bus.cycles();
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
                isr_count += 1;
            }
            isr_cycles = bus.cycles() - isr_start_cyc;
        }

        println!(
            "step={step_n:>10}  pre_cyc={pre_cyc:>12}  post_cyc={:>12} (+{}{})  pc=0x{pre_pc:08x}  instr=0x{pre_instr:08x}  v0=0x{:08x} v1=0x{:08x} a0=0x{:08x} s0=0x{:08x} ra=0x{:08x}",
            bus.cycles(),
            bus.cycles() - pre_cyc,
            if isr_count > 0 {
                format!("  ISR={isr_count} steps, {isr_cycles} cycles")
            } else { "".into() },
            cpu.gpr(2),
            cpu.gpr(3),
            cpu.gpr(4),
            cpu.gpr(16),
            cpu.gpr(31),
        );
    }
}
