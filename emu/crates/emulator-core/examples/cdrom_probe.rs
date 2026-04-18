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
            let op = bus.cdrom.last_command();
            let name = cmd_name(op);
            println!(
                "step={:>10}  cyc={:>12}  pc=0x{:08x}  cmds={:>3}  cmd=0x{:02X} {:<8}  irq_flag=0x{:02x}",
                cpu.tick(),
                bus.cycles(),
                cpu.pc(),
                c,
                op,
                name,
                bus.cdrom.irq_flag(),
            );
            last_cmd_count = c;
        }
    }

    println!();
    println!("=== Final command histogram ===");
    let hist = bus.cdrom.command_histogram();
    for (op, &c) in hist.iter().enumerate() {
        if c > 0 {
            println!("  0x{op:02X} {:<10} {c}", cmd_name(op as u8));
        }
    }
}

fn cmd_name(op: u8) -> &'static str {
    match op {
        0x00 => "Sync",
        0x01 => "GetStat",
        0x02 => "SetLoc",
        0x03 => "Play",
        0x04 => "Forward",
        0x05 => "Backward",
        0x06 => "ReadN",
        0x07 => "MotorOn",
        0x08 => "Stop",
        0x09 => "Pause",
        0x0A => "Init",
        0x0B => "Mute",
        0x0C => "Demute",
        0x0D => "SetFilter",
        0x0E => "SetMode",
        0x0F => "GetParam",
        0x10 => "GetLocL",
        0x11 => "GetLocP",
        0x12 => "SetSession",
        0x13 => "GetTN",
        0x14 => "GetTD",
        0x15 => "SeekL",
        0x16 => "SeekP",
        0x17 => "SetClock",
        0x18 => "GetClock",
        0x19 => "Test",
        0x1A => "GetID",
        0x1B => "ReadS",
        0x1C => "Reset",
        0x1D => "GetQ",
        0x1E => "ReadTOC",
        0x1F => "VideoCD",
        _ => "?",
    }
}
