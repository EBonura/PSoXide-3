//! Sample the CPU PC at a few checkpoints to see whether we're
//! in BIOS ROM, in RAM user code, or spinning somewhere. For
//! debugging games that hang past the BIOS splash (MGS).
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_pc_progression -- "/path/to/game.bin"
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let disc_path = std::env::args().nth(1);
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();

    let checkpoints = [
        100_000_000u64,
        200_000_000,
        250_000_000,
        300_000_000,
        350_000_000,
        400_000_000,
        450_000_000,
        500_000_000,
    ];
    let mut cur = 0u64;
    // Sample the last 1000 PCs near each checkpoint to see if we're
    // stuck in a tight loop.
    let mut recent_pcs: Vec<u32> = Vec::with_capacity(1000);
    let mut last_sector_events = 0u64;
    for target in checkpoints {
        while cur < target {
            let was_in_isr = cpu.in_isr();
            cpu.step(&mut bus).expect("step");
            if !was_in_isr && cpu.in_irq_handler() {
                while cpu.in_irq_handler() {
                    cpu.step(&mut bus).expect("isr step");
                }
            }
            cur += 1;
            if target - cur < 1000 {
                recent_pcs.push(cpu.pc());
            }
        }
        recent_pcs.sort();
        recent_pcs.dedup();
        let pc = cpu.pc();
        let region = match pc {
            p if p >= 0xBFC0_0000 => "BIOS-ROM",
            p if (0x8000_0000..0x8020_0000).contains(&p) => "RAM-user",
            p if (0xA000_0000..0xA020_0000).contains(&p) => "RAM-user(A)",
            _ => "???",
        };
        let sec_delta = bus.cdrom.sector_events_scheduled - last_sector_events;
        last_sector_events = bus.cdrom.sector_events_scheduled;
        println!(
            "step={target:>12}  cyc={:>12}  pc=0x{pc:08x} [{region}]  unique_pcs_in_last_1000={}  sec_events+{}",
            bus.cycles(),
            recent_pcs.len(),
            sec_delta,
        );
        recent_pcs.clear();
    }
}
