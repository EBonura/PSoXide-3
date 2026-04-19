//! Trace IRQ state changes around a given step to find exactly when
//! each IRQ source is raised vs dispatched.

use emulator_core::{Bus, Cpu};

fn main() {
    let from_step: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_irq_timeline <from_step> [count]");
    let count: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    for _ in 0..(from_step - 1) {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    let mut last_istat = bus.irq().stat();
    let mut last_cflag = bus.cdrom.irq_flag();

    for i in 0..count {
        let step_n = from_step + i;
        let pre_cyc = bus.cycles();
        let pre_pc = cpu.pc();

        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        let mut isr_info = String::new();
        if !was_in_isr && cpu.in_irq_handler() {
            let isr_start_cyc = bus.cycles();
            let mut isr_count = 0;
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
                isr_count += 1;
            }
            isr_info = format!(" ISR={}steps/{}cyc", isr_count, bus.cycles() - isr_start_cyc);
        }

        let istat = bus.irq().stat();
        let cflag = bus.cdrom.irq_flag();

        let istat_delta = if istat != last_istat {
            format!(" ISTAT:{last_istat:#010x}->{istat:#010x}")
        } else { "".into() };
        let cflag_delta = if cflag != last_cflag {
            format!(" CDROM_FLAG:{last_cflag:#x}->{cflag:#x}")
        } else { "".into() };

        println!(
            "step={step_n:>10}  cyc={pre_cyc}->{}  pc=0x{pre_pc:08x}{isr_info}{istat_delta}{cflag_delta}",
            bus.cycles(),
        );
        last_istat = istat;
        last_cflag = cflag;
    }
}
