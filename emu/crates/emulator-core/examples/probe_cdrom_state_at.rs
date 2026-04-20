//! Run our CDROM to step N, dump its state: current LBA, mode,
//! pending events. Used to cross-check against Redux when sector
//! data differs between them.

use emulator_core::{Bus, Cpu};

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(90_146_543);
    let disc_path = std::env::args().nth(2);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();
    for _ in 0..n {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    println!("=== CDROM state at step {n} (cycles={}) ===", bus.cycles());
    println!("irq_flag         : {}", bus.cdrom.irq_flag());
    println!("data_fifo_pops   : {}", bus.cdrom.data_fifo_pops());
    println!("sector_events    : {}", bus.cdrom.sector_events_scheduled);
    let cmd_hist = bus.cdrom.command_histogram();
    println!();
    println!("command histogram:");
    for i in 0..32 {
        let c = cmd_hist[i];
        if c > 0 {
            println!("  0x{i:02X}: {c}");
        }
    }
}
