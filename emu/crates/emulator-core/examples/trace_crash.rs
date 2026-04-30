//! Run the emulator with a disc until CPU step errors, then dump
//! the last N instructions that retired + key register state.
//! Helps trace "where did the game jump into unmapped memory" --
//! the jump target is usually somewhere near the last few retired
//! instructions, either via a GTE MFC2 to a pointer-sized register
//! or a DMA completion that corrupted RAM.

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::collections::VecDeque;
use std::path::PathBuf;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000_000);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    if let Ok(disc_path) = std::env::var("PSOXIDE_DISC") {
        bus.cdrom
            .insert_disc(Some(Disc::from_bin(std::fs::read(&disc_path).unwrap())));
    }
    let mut cpu = Cpu::new();

    // Keep a rolling window of the last 32 retired records so we
    // can print them post-crash. Running the full history would
    // blow up memory for a 180M-step run.
    let mut recent: VecDeque<(u32, u32, [u32; 32])> = VecDeque::with_capacity(32);

    for i in 0..n {
        let pc = cpu.pc();
        let instr = bus.read32(pc);
        // Snapshot gprs BEFORE step so we can see the JR source.
        let gprs_before = *cpu.gprs();
        match cpu.step(&mut bus) {
            Ok(_) => {
                if recent.len() >= 32 {
                    recent.pop_front();
                }
                recent.push_back((pc, instr, gprs_before));
            }
            Err(e) => {
                println!("[trace] crashed at step {i}: {e:?}");
                println!("[trace] pc at crash: 0x{:08x}", cpu.pc());
                println!();
                println!("=== last {} retired instructions ===", recent.len());
                for (pc, instr, gprs) in recent.iter().rev() {
                    let mnemonic = describe_instr(*instr);
                    println!("  pc=0x{pc:08x} instr=0x{instr:08x}  {mnemonic}");
                    if is_register_based_jump(*instr) {
                        let rs = ((*instr >> 21) & 0x1F) as usize;
                        println!("    → jump source: $r{rs}=0x{:08x} at issue time", gprs[rs]);
                    }
                }
                println!();
                println!("=== register state at crash ===");
                let gpr_names = [
                    "$0 ", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3", "$t0", "$t1", "$t2",
                    "$t3", "$t4", "$t5", "$t6", "$t7", "$s0", "$s1", "$s2", "$s3", "$s4", "$s5",
                    "$s6", "$s7", "$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra",
                ];
                for row in 0..4 {
                    let mut line = String::new();
                    for col in 0..8 {
                        let i = row * 8 + col;
                        line.push_str(&format!(" {}={:08x}", gpr_names[i], cpu.gprs()[i]));
                    }
                    println!("{line}");
                }
                return;
            }
        }
    }
    println!("[trace] completed {n} steps without a crash");
}

fn is_register_based_jump(instr: u32) -> bool {
    let primary = (instr >> 26) & 0x3F;
    if primary != 0 {
        return false;
    }
    let secondary = instr & 0x3F;
    matches!(secondary, 0x08 | 0x09) // JR, JALR
}

fn describe_instr(instr: u32) -> &'static str {
    let primary = (instr >> 26) & 0x3F;
    match primary {
        0x00 => {
            let secondary = instr & 0x3F;
            match secondary {
                0x08 => "JR",
                0x09 => "JALR",
                0x0C => "SYSCALL",
                _ => "SPECIAL",
            }
        }
        0x02 => "J",
        0x03 => "JAL",
        0x04 => "BEQ",
        0x05 => "BNE",
        0x0F => "LUI",
        0x23 => "LW",
        0x2B => "SW",
        _ => "?",
    }
}
