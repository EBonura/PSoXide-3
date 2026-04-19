//! Log every CDROM command byte the BIOS issues up to a step count,
//! with step + cycle. Tells us the exact command sequence so we can
//! spot missing commands against Redux.

use emulator_core::{Bus, Cpu};

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(89_198_894);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    let mut last_hist = *bus.cdrom.command_histogram();
    let mut last_cmds = bus.cdrom.commands_dispatched();

    for step in 1..=target {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }

        let now_cmds = bus.cdrom.commands_dispatched();
        if now_cmds != last_cmds {
            // Something changed in the histogram.
            let now_hist = *bus.cdrom.command_histogram();
            for (i, (old, new)) in last_hist.iter().zip(now_hist.iter()).enumerate() {
                if new > old {
                    let delta = new - old;
                    for _ in 0..delta {
                        println!(
                            "step={step:>10}  cyc={:>12}  cmd=0x{i:02x}",
                            bus.cycles(),
                        );
                    }
                }
            }
            last_hist = now_hist;
            last_cmds = now_cmds;
        }
    }
    println!();
    println!("Total: {} commands dispatched by step {target}", last_cmds);
}
