//! Fast-forward a retail disc to a step, then scan forward until the
//! next folded hardware IRQ on our side. Prints the pre-step I_STAT /
//! I_MASK source bits so we can identify which subsystem eventually
//! fires.

use emulator_core::{Bus, Cpu};

fn main() {
    let start_step: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_disc_next_irq <start_step> <max_steps> <disc.bin>");
    let max_steps: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let disc_path = std::env::args()
        .nth(3)
        .expect("usage: probe_disc_next_irq <start_step> <max_steps> <disc.bin>");

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc_bytes = std::fs::read(&disc_path).expect("disc readable");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom
        .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    let mut cpu = Cpu::new();

    for _ in 0..start_step {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    let irq_names = [
        "VBlank",
        "Gpu",
        "Cdrom",
        "Dma",
        "Timer0",
        "Timer1",
        "Timer2",
        "Controller",
        "Sio",
        "Spu",
        "Lightpen",
    ];

    for i in 1..=max_steps {
        let step_n = start_step + i;
        let pre_pc = cpu.pc();
        let pre_cycles = bus.cycles();
        let pre_istat = bus.irq().stat();
        let pre_imask = bus.irq().mask();

        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            let mut isr_steps = 0u64;
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
                isr_steps += 1;
            }
            let pending = pre_istat & pre_imask;
            let mut which = Vec::new();
            for bit in 0..irq_names.len() {
                if pending & (1 << bit) != 0 {
                    which.push(irq_names[bit]);
                }
            }
            println!("next_irq_step    : {step_n}");
            println!("pre_pc           : 0x{pre_pc:08x}");
            println!("cycles           : {pre_cycles} -> {}", bus.cycles());
            println!("istat/imask      : 0x{pre_istat:03x} / 0x{pre_imask:03x}");
            println!("sources          : {}", which.join(", "));
            println!("isr_steps        : {isr_steps}");
            return;
        }
    }

    println!("No folded IRQ in the next {max_steps} steps after {start_step}.");
}
