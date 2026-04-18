//! Produce aggregated parity records up to a target index, then
//! dump state. Matches `our_trace` in tests/parity.rs exactly —
//! outer iteration = 1 record; inside each, any ISR body is
//! drained synchronously and its final gprs replace the record's.

use emulator_core::{Bus, Cpu};

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("BIOS size");
    let mut cpu = Cpu::new();

    let n: usize = 19_474_560;
    let dump_from: usize = 19_474_540;

    let mut records_len = 0;
    while records_len <= n {
        let was_in_isr = cpu.in_isr();
        let rec = cpu.step(&mut bus).expect("step");
        let mut final_pc = cpu.pc();
        let mut final_cyc = bus.cycles();
        let mut isr_steps = 0usize;
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("step inside ISR");
                isr_steps += 1;
                if isr_steps > 100_000 {
                    eprintln!("WARN: stuck inside ISR for >100k steps, breaking");
                    break;
                }
            }
            final_pc = cpu.pc();
            final_cyc = bus.cycles();
        }
        if records_len >= dump_from {
            let gprs = cpu.gprs();
            let stat = bus.irq().stat();
            let mask = bus.irq().mask();
            println!(
                "rec={records_len:>9}  rec.pc=0x{:08x}  rec.instr=0x{:08x}  isr={isr_steps}  final_pc=0x{:08x}  final_cyc={final_cyc}  r26=0x{:08x}  istat=0x{:03x}  imask=0x{:03x}",
                rec.pc, rec.instr, final_pc, gprs[26], stat, mask
            );
        }
        records_len += 1;
    }
}
