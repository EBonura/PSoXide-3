//! Quantify our CPU cycle-count per step vs Redux's. At step N,
//! ours reports `bus.cycles()` and Redux reports its tick counter --
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

type Scenario<'a> = (&'a str, Option<&'a str>, &'a [(u64, u64)]);

fn main() {
    // Each scenario: (name, disc_path or None, (step, expected_redux_tick) list)
    //
    // Redux tick numbers captured via `display_parity_at` runs (the
    // Redux side prints `reached tick=N`). Checkpoints chosen to
    // bisect where the drift becomes visible -- 1M, 10M, 30M at the
    // front find early divergence; 100M/500M/1B show long-run trend.
    //
    // To add a new row: run `display_parity_at <steps> [disc]` and
    // copy the "reached tick=..." line.
    let scenarios: &[Scenario<'_>] = &[
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
                // Print just this one -- we'll run with different step
                // counts via `cargo run` + arg override if needed.
                (300_000_000, 705_660_075),
                (900_000_000, 2_177_711_744),
            ],
        ),
    ];

    // If env var `PSOXIDE_PROBE_STEPS` is set (comma-separated list),
    // override the crash schedule with those step counts and reuse
    // the last-known Redux tick for each as a placeholder (caller can
    // run `display_parity_at <steps> <disc>` to get the exact tick).
    if let Ok(list) = std::env::var("PSOXIDE_PROBE_STEPS") {
        let steps: Vec<u64> = list
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        println!("(override) running crash at {steps:?}");
        let disc =
            "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin";
        let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
        let mut bus = Bus::new(bios).expect("bus");
        let disc_bytes = std::fs::read(disc).expect("disc");
        bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
        let mut cpu = Cpu::new();
        let mut cycles_at_last_pump = 0u64;
        let mut cursor = 0u64;
        for target in steps {
            while cursor < target {
                if step_user_step(&mut cpu, &mut bus, &mut cycles_at_last_pump).is_err() {
                    break;
                }
                cursor += 1;
            }
            println!(
                "crash step={target:>12}  our_cycles={:>14}  cpi={:.4}",
                bus.cycles(),
                bus.cycles() as f64 / target as f64,
            );
        }
        return;
    }

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    println!(
        "{:<8} {:>12} {:>14} {:>14} {:>10} {:>10}",
        "scenario", "steps", "our_cycles", "redux_tick", "delta", "pct_off"
    );
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
                if step_user_step(&mut cpu, &mut bus, &mut cycles_at_last_pump).is_err() {
                    eprintln!("[{name}] CPU errored at step {cursor}");
                    return;
                }
                cursor += 1;
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

fn step_user_step(
    cpu: &mut Cpu,
    bus: &mut Bus,
    cycles_at_last_pump: &mut u64,
) -> Result<(), emulator_core::ExecutionError> {
    let was_in_isr = cpu.in_isr();
    step_one_and_pump(cpu, bus, cycles_at_last_pump)?;
    if !was_in_isr && cpu.in_irq_handler() {
        while cpu.in_irq_handler() {
            step_one_and_pump(cpu, bus, cycles_at_last_pump)?;
        }
    }
    Ok(())
}

fn step_one_and_pump(
    cpu: &mut Cpu,
    bus: &mut Bus,
    cycles_at_last_pump: &mut u64,
) -> Result<(), emulator_core::ExecutionError> {
    cpu.step(bus)?;
    if bus.cycles() - *cycles_at_last_pump > 560_000 {
        *cycles_at_last_pump = bus.cycles();
        bus.run_spu_samples(735);
        let _ = bus.spu.drain_audio();
    }
    Ok(())
}
