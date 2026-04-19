//! Dump CDROM controller state at a given step.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_cdrom_state -- 89198894
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(89_198_894);

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

    println!("=== CDROM state at step {target} (cyc={}) ===", bus.cycles());
    println!("commands dispatched: {}", bus.cdrom.commands_dispatched());
    println!("last command byte:   0x{:02x}", bus.cdrom.last_command());
    println!("irq_flag:            0x{:x}", bus.cdrom.irq_flag());
    println!("irq_mask:            0x{:x}", bus.cdrom.irq_mask_raw());
    println!();
    println!("command histogram (non-zero):");
    let h = bus.cdrom.command_histogram();
    for (i, n) in h.iter().enumerate() {
        if *n > 0 {
            println!("  0x{:02x}: {n:>6}", i);
        }
    }
    println!();
    println!("IRQ STAT = 0x{:08x}", bus.irq().stat());
    println!("IRQ MASK = 0x{:08x}", bus.irq().mask());
    println!();
    println!("scheduler (pending events):");
    use emulator_core::scheduler::EventSlot;
    for slot in [
        EventSlot::Sio, EventSlot::Sio1, EventSlot::Cdr, EventSlot::CdRead,
        EventSlot::GpuDma, EventSlot::MdecOutDma, EventSlot::SpuDma,
        EventSlot::MdecInDma, EventSlot::GpuOtcDma, EventSlot::CdrDma,
        EventSlot::CdrPlay, EventSlot::CdrDbuf, EventSlot::CdrLid,
        EventSlot::SpuAsync, EventSlot::VBlank,
    ] {
        if let Some(target) = bus.scheduler.target(slot) {
            println!("  {slot:?}: target = {target}  (in {} cycles)",
                (target as i64) - (bus.cycles() as i64));
        }
    }
}
