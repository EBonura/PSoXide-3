//! Parity-divergence probe: step our CPU to the 20M-step parity
//! test's first mismatch (record index 19_474_544) and dump our
//! state alongside Redux's cached-trace values. Used to narrow down
//! why the two emulators disagree on `$r26` at that point.
//!
//! Finding (2026-04-18):
//! - Both trace records agree on PC + instruction (`0x80050294`,
//!   NOP delay slot of `jal 0x80051300`).
//! - Our `$r26` = 0x80050dfc; Redux's = 0x80051300.
//! - Redux's cycles at that record = 46249947; ours = 46247459.
//! - The 2490-cycle gap is an ISR body Redux runs that we don't.
//! - Our `istat=0x000` — no IRQ pending. Whatever DMA-completion
//!   event would have raised `IrqSource::Dma` didn't flip our
//!   DICR master edge (channel-enable bit not armed, or ordering
//!   drift in the scheduler's per-channel target cycle).
//!
//! Next steps (future session):
//! - Instrument `Bus::drain_scheduler_events` to log every slot
//!   fire + its DICR enable bit during a full parity run.
//! - Cross-check with Redux's `dmaInterrupt<n>` log at matching
//!   cycles.
//!
//! Usage:
//! ```bash
//! cargo run --release --example probe_19474544 -p emulator-core
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("BIOS size");
    let mut cpu = Cpu::new();

    // Match the parity test's trace aggregation — see tests/parity.rs
    // `our_trace` for the shape.
    let target_rec: usize = 19_474_544;

    let mut records_len = 0;
    while records_len < target_rec {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("step inside ISR");
            }
        }
        records_len += 1;
    }

    // State at the start of record N — about to execute that step.
    let gprs = cpu.gprs();
    println!("=== our state at record {} ===", target_rec);
    println!("PC         = 0x{:08x}", cpu.pc());
    println!("bus cycles = {}", bus.cycles());
    println!("in_isr     = {}", cpu.in_isr());
    println!("GPRs:");
    for r in (0..32).step_by(4) {
        println!(
            "  $r{:02}={:08x}  $r{:02}={:08x}  $r{:02}={:08x}  $r{:02}={:08x}",
            r,
            gprs[r],
            r + 1,
            gprs[r + 1],
            r + 2,
            gprs[r + 2],
            r + 3,
            gprs[r + 3],
        );
    }
    println!("IRQ:  stat=0x{:08x}  mask=0x{:08x}", bus.irq().stat(), bus.irq().mask());
    println!("Raise counts {:?}", bus.irq().raise_counts());
    println!(
        "Scheduler pending=0b{:b}  lowest_target={}",
        bus.scheduler.pending_bitmap(),
        bus.scheduler.lowest_target()
    );
}
