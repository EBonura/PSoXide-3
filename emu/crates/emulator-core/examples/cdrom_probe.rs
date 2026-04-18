//! Stream every CDROM command issued by the BIOS during a disc boot,
//! along with the response bytes returned and the cycle at which each
//! event happens. Used to debug why the BIOS shell doesn't advance to
//! the boot-EXE path even with a valid disc mounted — by comparing our
//! command trace to what's expected (nocash spec / Redux behaviour).
//!
//! ```bash
//! PSOXIDE_DISC=/path/to/Crash.bin \
//! cargo run -p emulator-core --example cdrom_probe --release -- 200000000
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::path::PathBuf;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000_000);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let disc_path = std::env::var("PSOXIDE_DISC").expect("set PSOXIDE_DISC");

    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let disc_bytes = std::fs::read(&disc_path).expect("disc readable");

    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
    let mut cpu = Cpu::new();

    // Snapshot command count each step; emit a line whenever it grows.
    let mut last_cmd_count: u64 = 0;
    for _ in 0..n {
        cpu.step(&mut bus).expect("step");
        let c = bus.cdrom.commands_dispatched();
        if c != last_cmd_count {
            let hist = bus.cdrom.command_histogram();
            let latest_op = hist
                .iter()
                .enumerate()
                .rev()
                .find(|(_, &c)| c > 0)
                .map(|(i, _)| i as u8)
                .unwrap_or(0);
            // Can't directly tell which was the latest command without more
            // instrumentation; print the new total and delta-counts across
            // the histogram vs the previous snapshot.
            println!(
                "step={:>10}  cyc={:>12}  pc=0x{:08x}  cmds={} (latest op: 0x{:02x})  irq_flag=0x{:02x}  irq_mask=0x{:02x}",
                cpu.tick(),
                bus.cycles(),
                cpu.pc(),
                c,
                latest_op,
                bus.cdrom.irq_flag(),
                bus.cdrom.irq_mask_raw(),
            );
            last_cmd_count = c;
        }
    }

    println!();
    println!("=== Final command histogram ===");
    let hist = bus.cdrom.command_histogram();
    for (op, &c) in hist.iter().enumerate() {
        if c > 0 {
            let name = match op as u8 {
                0x01 => "GetStat",
                0x02 => "SetLoc",
                0x06 => "ReadN",
                0x09 => "Pause",
                0x0A => "Init",
                0x0B => "Mute",
                0x0C => "Demute",
                0x0D => "SetFilter",
                0x0E => "SetMode",
                0x13 => "GetTN",
                0x14 => "GetTD",
                0x15 => "SeekL",
                0x16 => "SeekP",
                0x19 => "Test",
                0x1A => "GetID",
                0x1B => "ReadS",
                0x1E => "ReadTOC",
                _ => "?",
            };
            println!("  0x{op:02X} {name:<10} {c}");
        }
    }
}
