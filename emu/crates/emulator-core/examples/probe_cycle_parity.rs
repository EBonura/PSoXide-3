//! Quantify our CPU cycle-count per step vs Redux's. At step N,
//! ours reports `bus.cycles()` and Redux reports its tick counter —
//! if the two diverge at matching step counts, render parity
//! across frames is impossible (we're at different moments in the
//! same game).
//!
//! Redux's ticks at well-known checkpoints (captured from prior
//! `display_parity_at` runs):
//!   - 100M steps no-disc: tick = 233653764
//!   - 500M steps no-disc: tick = 1104781373  (our hash was diff → maybe cycle diff too)
//!   - 100M steps Crash:   tick = 235279073
//!   - 300M steps Crash:   tick = 705660075
//!   - 900M steps Crash:   tick = 2177711744
//!
//! This probe runs ours-only at the same checkpoints and reports
//! our `bus.cycles()`. The delta vs Redux's tick is the per-step
//! divergence we need to close for frame-level parity.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_cycle_parity --release
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;

fn main() {
    // Each scenario: (name, disc_path or None, (step, expected_redux_tick) list)
    let scenarios: &[(&str, Option<&str>, &[(u64, u64)])] = &[
        (
            "bios",
            None,
            &[
                (100_000_000, 233_653_764),
                (500_000_000, 1_104_781_373),
            ],
        ),
        (
            "crash",
            Some("/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin"),
            &[
                (100_000_000, 235_279_073),
                (300_000_000, 705_660_075),
                (900_000_000, 2_177_711_744),
            ],
        ),
    ];

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    println!("{:<8} {:>12} {:>14} {:>14} {:>10} {:>10}",
        "scenario", "steps", "our_cycles", "redux_tick", "delta", "pct_off");
    println!("{}", "-".repeat(76));
    for (name, disc_path, checkpoints) in scenarios {
        let mut bus = Bus::new(bios.clone()).expect("bus");
        if let Some(p) = disc_path {
            let disc_bytes = std::fs::read(p).expect("disc");
            bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
        }
        let mut cpu = Cpu::new();
        let mut cycles_at_last_pump = 0u64;
        let mut cursor = 0u64;

        for (target_step, redux_tick) in *checkpoints {
            while cursor < *target_step {
                if cpu.step(&mut bus).is_err() {
                    eprintln!("[{name}] CPU errored at step {cursor}");
                    return;
                }
                cursor += 1;
                if bus.cycles() - cycles_at_last_pump > 560_000 {
                    cycles_at_last_pump = bus.cycles();
                    bus.run_spu_samples(735);
                    let _ = bus.spu.drain_audio();
                }
            }
            let our = bus.cycles();
            let delta = our as i64 - *redux_tick as i64;
            let pct = 100.0 * delta as f64 / (*redux_tick as f64);
            println!(
                "{:<8} {:>12} {:>14} {:>14} {:>+10} {:>+9.3}%",
                name, target_step, our, redux_tick, delta, pct,
            );
        }
    }
}
