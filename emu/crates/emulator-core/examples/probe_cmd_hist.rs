//! Dump CDROM command histogram + sector-events-scheduled
//! delta-per-checkpoint, to see what kind of traffic a game is
//! producing in the loader/intro stages.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_cmd_hist -- "/path/to/game.bin" 500000000
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let disc_path = std::env::args().nth(1);
    let n: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000_000);

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

    println!("=== CDROM command histogram @ step {n} ===");
    let hist = bus.cdrom.command_histogram();
    let names: [(u8, &str); 32] = [
        (0x01, "GetStat"),
        (0x02, "SetLoc"),
        (0x03, "Play"),
        (0x04, "Forward"),
        (0x05, "Backward"),
        (0x06, "ReadN"),
        (0x07, "MotorOn"),
        (0x08, "Stop"),
        (0x09, "Pause"),
        (0x0A, "Init"),
        (0x0B, "Mute"),
        (0x0C, "Demute"),
        (0x0D, "SetFilter"),
        (0x0E, "SetMode"),
        (0x0F, "GetParam"),
        (0x10, "GetLocL"),
        (0x11, "GetLocP"),
        (0x12, "SetSession"),
        (0x13, "GetTN"),
        (0x14, "GetTD"),
        (0x15, "SeekL"),
        (0x16, "SeekP"),
        (0x19, "Test"),
        (0x1A, "GetID"),
        (0x1B, "ReadS"),
        (0x1C, "Reset"),
        (0x1D, "GetQ"),
        (0x1E, "ReadTOC"),
        (0x1F, "VideoCD"),
        (0x00, ""),
        (0x17, ""),
        (0x18, ""),
    ];
    let mut total: u64 = 0;
    for (cmd, name) in names.iter() {
        let c = hist[*cmd as usize] as u64;
        if c > 0 {
            println!("  0x{cmd:02X} {name:<10} {c:>8}");
            total += c;
        }
    }
    println!("  {:>25}", "---");
    println!("  total:                {total:>8}");
    println!();
    println!(
        "sector_events_scheduled = {}",
        bus.cdrom.sector_events_scheduled
    );
    println!("data_fifo_pops          = {}", bus.cdrom.data_fifo_pops());
    let counts = bus.cdrom.irq_type_counts;
    let types = [
        "None",
        "DataReady",
        "Complete",
        "Acknowledge",
        "DataEnd",
        "Error",
    ];
    println!();
    println!("=== CDROM IrqType raises ===");
    for (name, count) in types.iter().zip(counts.iter()) {
        println!("  {name:<12} {count:>14}");
    }
}
