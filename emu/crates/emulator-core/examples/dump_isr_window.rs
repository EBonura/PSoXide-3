//! Dump emitted instructions from an ISR that just ran, by
//! single-stepping around a known user step and printing each
//! instruction that executes inside `in_irq_handler() == true`.
//!
//! Usage: `cargo run --release -p emulator-core --example dump_isr_window -- <user_step_at_isr_entry> <post_fold_cycle>`
//!
//! Run ours up to the user step that triggers the ISR, then
//! log every in-ISR instruction we retire until we exit the ISR,
//! with PC + instr + key GPR state so we can cross-reference against
//! the BIOS ISR code.

use emulator_core::{Bus, Cpu};

fn main() {
    let trigger_step: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: dump_isr_window <trigger_user_step>");

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    // Run folded steps up to the trigger step (one user step retires
    // per iteration, ISR body counted in the fold).
    let mut user_step = 0u64;
    while user_step < trigger_step - 1 {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
        user_step += 1;
    }

    // Now one more user-side step that's expected to trigger the ISR.
    // Single-step through it + the ISR body, logging everything.
    let was_in_isr = cpu.in_isr();
    let trigger_pc = cpu.pc();
    let trigger_instr = bus.peek_instruction(trigger_pc).unwrap_or(0);
    println!(
        "[trigger user step {trigger_step}]  pc=0x{trigger_pc:08x} instr=0x{trigger_instr:08x}  cyc_pre={}",
        bus.cycles(),
    );
    cpu.step(&mut bus).expect("step");
    println!(
        "  after user step: pc=0x{:08x} cyc={}  in_irq={}  in_isr={}",
        cpu.pc(),
        bus.cycles(),
        cpu.in_irq_handler(),
        cpu.in_isr(),
    );

    if !was_in_isr && cpu.in_irq_handler() {
        let mut isr_n = 0;
        while cpu.in_irq_handler() {
            let pc = cpu.pc();
            let instr = bus.peek_instruction(pc).unwrap_or(0);
            let gpr_a0 = cpu.gpr(4);
            let gpr_t0 = cpu.gpr(8);
            let gpr_t2 = cpu.gpr(10);
            cpu.step(&mut bus).expect("isr step");
            println!(
                "  isr[{isr_n:>4}]  pc=0x{pc:08x} instr=0x{instr:08x}  cyc={}  a0=0x{gpr_a0:08x} t0=0x{gpr_t0:08x} t2=0x{gpr_t2:08x}",
                bus.cycles(),
            );
            isr_n += 1;
            if isr_n > 2000 {
                println!("  ... truncating after 2000 isr steps");
                break;
            }
        }
        println!("isr length: {isr_n} instructions");
    }
}
