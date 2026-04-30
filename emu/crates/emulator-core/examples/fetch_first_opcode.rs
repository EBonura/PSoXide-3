//! Smoke-test tool: load a real BIOS image, seat the CPU at the reset
//! vector, and print the first instruction word.
//!
//! ```bash
//! cargo run -p emulator-core --example fetch_first_opcode -- \
//!     "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"
//! ```
//!
//! Expected first opcode of stock BIOS images: `0x3C08_1F80`
//! (`lui $t0, 0x1F80` -- load the I/O base into `$t0`).

use std::env;
use std::fs;
use std::process::ExitCode;

use emulator_core::{Bus, Cpu};

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: fetch_first_opcode <path-to-bios.bin>");
            return ExitCode::from(2);
        }
    };

    let bios = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read BIOS at {path}: {e}");
            return ExitCode::from(1);
        }
    };

    let mut bus = match Bus::new(bios) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bus setup failed: {e}");
            return ExitCode::from(1);
        }
    };

    let cpu = Cpu::new();
    let opcode = cpu.fetch(&mut bus);

    println!("BIOS        : {path}");
    println!("reset vector: {:#010x}", cpu.pc());
    println!("first opcode: {opcode:#010x}");

    ExitCode::SUCCESS
}
