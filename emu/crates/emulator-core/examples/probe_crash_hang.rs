//! Boot Crash Bandicoot end-to-end with SPU pumping and report
//! where the CPU spends its time. If execution hangs on a tight
//! loop (the "stops after PS logo" symptom), this shows the hot
//! PC range — usually the game is polling an MMIO register we're
//! returning the wrong value for.
//!
//! Usage:
//! ```bash
//! cargo run --release --example probe_crash_hang -p emulator-core
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::collections::HashMap;

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into()
    });
    let disc_path = std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin".into()
    });

    let bios = std::fs::read(&bios_path).expect("BIOS");
    let disc = std::fs::read(&disc_path).expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    let mut cpu = Cpu::new();

    // Track PC hits in 64-byte buckets over the last N instructions.
    // A sustained hot bucket = a spin loop.
    let bucket_size = 64u32;
    let mut pc_buckets: HashMap<u32, u64> = HashMap::new();

    let total_steps = 1_500_000_000u64;
    let mut retired = 0u64;
    let mut cycles_at_last_pump = 0u64;
    let mut cycles_at_last_report = 0u64;
    let mut last_retired = 0u64;
    let mut err: Option<String> = None;

    while retired < total_steps {
        match cpu.step(&mut bus) {
            Ok(_) => {}
            Err(e) => {
                err = Some(format!("{e:?}"));
                break;
            }
        }
        let bucket = cpu.pc() / bucket_size * bucket_size;
        *pc_buckets.entry(bucket).or_insert(0) += 1;
        retired += 1;

        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }

        // Every ~50M instructions, report the current hot bucket
        // + the delta.
        if retired - last_retired > 50_000_000 {
            let delta = retired - last_retired;
            let cyc_delta = bus.cycles() - cycles_at_last_report;
            last_retired = retired;
            cycles_at_last_report = bus.cycles();
            let mut top: Vec<_> = pc_buckets.iter().collect();
            top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
            let top5: Vec<_> = top.iter().take(5).collect();
            eprintln!(
                "step={retired:>12}  cyc={:>13}  delta_instr={delta}  delta_cyc={cyc_delta}  top5:",
                bus.cycles()
            );
            for (bucket, count) in top5 {
                eprintln!(
                    "    pc~0x{:08x}  hits={}  ({:.1}%)",
                    bucket,
                    count,
                    **count as f64 * 100.0 / delta as f64
                );
            }
            pc_buckets.clear(); // rolling window
        }
    }

    eprintln!();
    eprintln!("=== final state ===");
    eprintln!("retired instructions: {retired}");
    eprintln!("cycles: {}", bus.cycles());
    eprintln!("cpu.pc = 0x{:08x}", cpu.pc());
    if let Some(e) = err {
        eprintln!("last error: {e}");
    }
    eprintln!(
        "irq raise counts: {:?}",
        bus.irq().raise_counts()
    );
    eprintln!(
        "cdrom commands dispatched: {}, last cmd: 0x{:02x}",
        bus.cdrom.commands_dispatched(),
        bus.cdrom.last_command()
    );
    eprintln!("spu samples produced: {}", bus.spu.samples_produced());
}
