//! Trace the last 32 instructions before Crash's wild-pointer
//! jump to 0x09070026. The jump target is garbage -- we want to
//! see the branch / jump instruction that computed it, and its
//! input register values.

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::collections::VecDeque;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = std::fs::read(
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin",
    )
    .expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    let mut cpu = Cpu::new();

    // Ring of the most recent 64 (pc, instr, gprs) samples so we
    // can print the context around a jump off a cliff.
    let mut window: VecDeque<(u32, u32, [u32; 32])> = VecDeque::with_capacity(64);

    let target_pc: u32 = 0x0907_0026;
    let mut cycles_at_last_pump = 0u64;

    for retired in 0..250_000_000u64 {
        let pc_before = cpu.pc();
        // Stop as soon as we're about to execute at the bad PC.
        if pc_before == target_pc {
            eprintln!("Caught PC entering {target_pc:#010x} at step {retired}");
            break;
        }
        let instr = bus.peek_instruction(pc_before).unwrap_or(0);
        if window.len() == 64 {
            window.pop_front();
        }
        window.push_back((pc_before, instr, *cpu.gprs()));
        if cpu.step(&mut bus).is_err() {
            eprintln!("CPU err at step {retired}, pc={:#010x}", cpu.pc());
            break;
        }
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }
    }

    eprintln!("=== last {} instructions ===", window.len());
    for (pc, instr, gprs) in &window {
        eprintln!(
            "  pc=0x{pc:08x}  instr=0x{instr:08x}  ra=0x{:08x} t0=0x{:08x} t1=0x{:08x} v0=0x{:08x} a0=0x{:08x}",
            gprs[31], gprs[8], gprs[9], gprs[2], gprs[4]
        );
    }
    eprintln!("Final cpu.pc = 0x{:08x}", cpu.pc());
}
