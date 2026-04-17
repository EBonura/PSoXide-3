//! `psx-exe-info` — validate a PSX-EXE and print its header.
//!
//! Our linker script emits a binary that already has the PSX-EXE header
//! baked in at offset 0, so "packing" is a no-op today. This tool just
//! parses the file with `psx_iso::Exe` and pretty-prints the important
//! fields so the user can sanity-check a freshly built homebrew before
//! feeding it to the emulator.

use psx_iso::Exe;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: psx-exe-info <path-to-psx-exe>");
        return ExitCode::from(2);
    };

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {path}: {e}");
            return ExitCode::from(1);
        }
    };

    let exe = match Exe::parse(&bytes) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("parse {path}: {e:?}");
            return ExitCode::from(1);
        }
    };

    println!("file              {path}");
    println!("size              {} bytes (header + {} payload)", bytes.len(), exe.payload.len());
    println!("initial_pc        0x{:08x}", exe.initial_pc);
    println!("initial_gp        0x{:08x}", exe.initial_gp);
    println!("load_addr         0x{:08x}", exe.load_addr);
    println!("payload_bytes     {}", exe.payload.len());
    println!("bss_addr          0x{:08x}", exe.bss_addr);
    println!("bss_size          0x{:08x} ({} bytes)", exe.bss_size, exe.bss_size);
    match exe.initial_sp() {
        Some(sp) => println!("initial_sp        0x{:08x}", sp),
        None => println!("initial_sp        (keep default)"),
    }
    ExitCode::SUCCESS
}
