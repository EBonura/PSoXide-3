//! Trace the first few BIOS `B(0x08) OpenEvent` calls and report the
//! returned handle values. Useful when the BIOS appears to reach the
//! CDROM event bootstrap path but the handle slots in RAM remain zero.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_openevent_returns -- 20000000 "/path/to/game.bin"
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{Bus, Cpu};
use std::path::Path;

#[derive(Copy, Clone)]
struct PendingCall {
    step: u64,
    ra: u32,
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
}

fn main() {
    let max_steps: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    let disc_path = std::env::args().nth(2);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref path) = disc_path {
        let disc = disc_support::load_disc_path(Path::new(path)).expect("disc");
        bus.cdrom.insert_disc(Some(disc));
    }
    let mut cpu = Cpu::new();

    let mut pending: Option<PendingCall> = None;
    let mut seen = 0u32;

    for step in 0..max_steps {
        let pc = cpu.pc();
        if pending.is_none() && pc == 0x0000_00b0 && cpu.gpr(9) as u8 == 0x08 {
            pending = Some(PendingCall {
                step,
                ra: cpu.gpr(31),
                a0: cpu.gpr(4),
                a1: cpu.gpr(5),
                a2: cpu.gpr(6),
                a3: cpu.gpr(7),
            });
        }

        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");

        if let Some(call) = pending {
            if cpu.pc() == call.ra {
                println!(
                    "OpenEvent call@step={} args=({:#010x}, {:#010x}, {:#010x}, {:#010x}) -> v0={:#010x} v1={:#010x} pc={:#010x} cycles={}",
                    call.step,
                    call.a0,
                    call.a1,
                    call.a2,
                    call.a3,
                    cpu.gpr(2),
                    cpu.gpr(3),
                    cpu.pc(),
                    bus.cycles()
                );
                pending = None;
                seen += 1;
                if seen >= 8 {
                    break;
                }
            }
        }

        if pending.is_none() && !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }
}
