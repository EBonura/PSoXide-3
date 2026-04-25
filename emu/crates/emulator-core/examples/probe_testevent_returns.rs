//! Trace BIOS `B(0x0B) TestEvent` calls and summarize which event
//! handles return signaled vs. unsignaled. Useful when the BIOS wedges
//! in an event-poll loop after the PlayStation logo.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_testevent_returns -- 120000000 "/path/to/game.cue"
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use std::collections::BTreeMap;

use emulator_core::{Bus, Cpu};

#[derive(Copy, Clone)]
struct PendingCall {
    step: u64,
    ra: u32,
    handle: u32,
}

#[derive(Default, Clone, Copy)]
struct HandleStats {
    total: u64,
    true_count: u64,
    false_count: u64,
    first_true_step: Option<u64>,
}

fn main() {
    let max_steps: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(120_000_000);
    let disc_path = std::env::args().nth(2);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref path) = disc_path {
        let disc = disc_support::load_disc_path(std::path::Path::new(path)).expect("disc");
        bus.cdrom.insert_disc(Some(disc));
    }
    let mut cpu = Cpu::new();

    let mut pending: Option<PendingCall> = None;
    let mut stats: BTreeMap<u32, HandleStats> = BTreeMap::new();
    let mut printed = 0u32;
    let max_prints = 64u32;

    for step in 0..max_steps {
        let pc = cpu.pc();
        if pending.is_none() && pc == 0x0000_00b0 && cpu.gpr(9) as u8 == 0x0b {
            pending = Some(PendingCall {
                step,
                ra: cpu.gpr(31),
                handle: cpu.gpr(4),
            });
        }

        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");

        if let Some(call) = pending {
            if cpu.pc() == call.ra {
                let ret = cpu.gpr(2);
                let entry = stats.entry(call.handle).or_default();
                entry.total = entry.total.saturating_add(1);
                if ret != 0 {
                    entry.true_count = entry.true_count.saturating_add(1);
                    entry.first_true_step.get_or_insert(call.step);
                } else {
                    entry.false_count = entry.false_count.saturating_add(1);
                }

                let should_print = printed < max_prints
                    || ret != 0
                    || entry.total.is_power_of_two()
                    || call.handle == 0;
                if should_print {
                    println!(
                        "TestEvent call@step={} handle={:#010x} -> v0={} pc={:#010x} cycles={}",
                        call.step,
                        call.handle,
                        ret,
                        cpu.pc(),
                        bus.cycles()
                    );
                    printed = printed.saturating_add(1);
                }
                pending = None;
            }
        }

        if pending.is_none() && !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    println!("=== TestEvent summary @ step {} ===", cpu.tick());
    println!("cycles: {}", bus.cycles());
    println!("final pc: 0x{:08x}", cpu.pc());
    for (handle, entry) in stats {
        println!(
            "  handle={:#010x} total={} true={} false={} first_true_step={}",
            handle,
            entry.total,
            entry.true_count,
            entry.false_count,
            entry
                .first_true_step
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-".into())
        );
    }
}
