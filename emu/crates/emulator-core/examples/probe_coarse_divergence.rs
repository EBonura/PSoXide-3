//! Fast divergence hunt. Instead of capturing a full per-instruction
//! trace (14 GiB for 100 M steps, ~40 min of Redux runtime), sample
//! `{step, tick, pc}` every `INTERVAL` user-side steps and compare.
//! Finds the first divergence to within an `INTERVAL`-step window in
//! ~30 s — a 70–80× speedup over the full trace cache.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_coarse_divergence -- 100000000 10000
//! ```
//!
//! Once the window is identified, `probe_cycle_first_divergence`
//! (full-trace mode) can zero in on the exact instruction — but only
//! needs to trace `INTERVAL` records (~250 KiB, ~0.3 s), not the
//! full run.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// Per-checkpoint timeout. Redux silent-run rate is ~0.5 M steps/s,
/// so a 100 M-step interval can take ~3.5 min. Give it 10 min to
/// cover slow-boot intervals (game intros do lots of FMV decoding).
const CHECKPOINT_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);
    let interval: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let disc_path = std::env::args().nth(3);
    let held_buttons = std::env::var("PSOXIDE_PAD1")
        .ok()
        .and_then(|s| parse_u16_mask(&s))
        .unwrap_or(0);
    let pad_pulses = std::env::var("PSOXIDE_PAD1_PULSES")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            parse_pad_pulses(&s).unwrap_or_else(|| {
                panic!(
                    "PSOXIDE_PAD1_PULSES must be comma-separated \
                     <mask>@<start_vblank>+<frames> entries"
                )
            })
        })
        .unwrap_or_default();
    let wants_pad = held_buttons != 0 || !pad_pulses.is_empty();

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));

    // --- Ours ---
    eprintln!("[ours] running {n} steps, emitting checkpoint every {interval} steps...");
    let t0 = Instant::now();
    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    if wants_pad {
        bus.attach_digital_pad_port1();
    }
    let mut cpu = Cpu::new();
    let mut current_pad_mask = None;
    if wants_pad {
        sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
    }
    let mut our_checks: Vec<(u64, u64, u32)> = Vec::with_capacity((n / interval) as usize);
    for step in 1..=n {
        // ISR fold — match Redux's user-side step counting.
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if wants_pad {
            sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
        }
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
                if wants_pad {
                    sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
                }
            }
        }
        if step % interval == 0 {
            our_checks.push((step, bus.cycles(), cpu.pc()));
        }
    }
    let ours_elapsed = t0.elapsed();
    eprintln!(
        "[ours] {} checkpoints in {:.1} s ({:.0} steps/s)",
        our_checks.len(),
        ours_elapsed.as_secs_f64(),
        n as f64 / ours_elapsed.as_secs_f64(),
    );

    // --- Redux ---
    eprintln!("[redux] launching, then running {n} steps silently with same interval...");
    let t0 = Instant::now();
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios_path, lua).expect("Redux resolves");
    if let Some(ref p) = disc_path {
        config = config.with_disc(PathBuf::from(p));
    }
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");
    let mut redux_checks: Vec<(u64, u64, u32)> = Vec::with_capacity((n / interval) as usize);
    eprintln!(
        "[redux] CHECKPOINT_TIMEOUT = {}s",
        CHECKPOINT_TIMEOUT.as_secs()
    );
    if wants_pad {
        let pulse_tuples = pad_pulses
            .iter()
            .map(|pulse| (pulse.mask, pulse.start_vblank, pulse.frames))
            .collect::<Vec<_>>();
        redux
            .run_checkpoint_pad(
                n,
                interval,
                1,
                held_buttons,
                &pulse_tuples,
                CHECKPOINT_TIMEOUT,
                |step, tick, pc| {
                    redux_checks.push((step, tick, pc));
                    Ok(())
                },
            )
            .expect("run_checkpoint_pad");
    } else {
        redux
            .run_checkpoint(n, interval, CHECKPOINT_TIMEOUT, |step, tick, pc| {
                redux_checks.push((step, tick, pc));
                Ok(())
            })
            .expect("run_checkpoint");
    }
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();
    let redux_elapsed = t0.elapsed();
    eprintln!(
        "[redux] {} checkpoints in {:.1} s ({:.0} steps/s)",
        redux_checks.len(),
        redux_elapsed.as_secs_f64(),
        n as f64 / redux_elapsed.as_secs_f64(),
    );

    // --- Compare ---
    println!();
    println!("=== coarse divergence scan ===");
    println!("n={n} interval={interval}");
    let cmp = our_checks.len().min(redux_checks.len());
    let mut first: Option<usize> = None;
    for i in 0..cmp {
        let (os, otick, opc) = our_checks[i];
        let (rs, rtick, rpc) = redux_checks[i];
        if os != rs {
            println!("step counter mismatch at idx {i}: ours={os} redux={rs} — probe broken");
            return;
        }
        if otick != rtick || opc != rpc {
            first = Some(i);
            break;
        }
    }
    match first {
        None => {
            println!(
                "All {cmp} checkpoints match. Max tick = {} (step {})",
                our_checks.last().map(|c| c.1).unwrap_or(0),
                our_checks.last().map(|c| c.0).unwrap_or(0),
            );
        }
        Some(idx) => {
            let (step, otick, opc) = our_checks[idx];
            let (_, rtick, rpc) = redux_checks[idx];
            let prev_step = if idx == 0 { 0 } else { our_checks[idx - 1].0 };
            println!("First divergence at checkpoint idx {idx} (step {step}):");
            println!("  ours:  tick={otick:>12}  pc=0x{opc:08x}");
            println!(
                "  redux: tick={rtick:>12}  pc=0x{rpc:08x}  (delta tick={:+})",
                otick as i64 - rtick as i64
            );
            println!(
                "  → bug is in user-step window ({prev_step}, {step}] ({} steps wide)",
                step - prev_step,
            );
            println!();
            println!(
                "Next: zoom in with `probe_cycle_first_divergence` run from step {prev_step}."
            );
        }
    }
}

fn parse_u16_mask(text: &str) -> Option<u16> {
    let s = text.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u16>().ok()
    }
}

fn parse_pad_pulses(text: &str) -> Option<Vec<PadPulse>> {
    let mut pulses = Vec::new();
    for entry in text.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        pulses.push(parse_pad_pulse(entry)?);
    }
    Some(pulses)
}

fn parse_pad_pulse(text: &str) -> Option<PadPulse> {
    let (mask_text, rest) = text.split_once('@')?;
    let mask = parse_u16_mask(mask_text)?;
    let (start_text, frames_text) = match rest.split_once('+') {
        Some((start, frames)) => (start.trim(), frames.trim()),
        None => (rest.trim(), "1"),
    };
    let start_vblank = start_text.parse().ok()?;
    let frames = frames_text.parse().ok()?;
    if frames == 0 {
        return None;
    }
    Some(PadPulse {
        mask,
        start_vblank,
        frames,
    })
}

fn effective_pad_mask(base_mask: u16, pulses: &[PadPulse], current_vblank: u64) -> u16 {
    let mut mask = base_mask;
    for pulse in pulses {
        let end_vblank = pulse.start_vblank.saturating_add(pulse.frames);
        if current_vblank >= pulse.start_vblank && current_vblank < end_vblank {
            mask |= pulse.mask;
        }
    }
    mask
}

fn sync_pad_mask(
    bus: &mut Bus,
    base_mask: u16,
    pulses: &[PadPulse],
    current_mask: &mut Option<u16>,
) {
    let next_mask = effective_pad_mask(base_mask, pulses, bus.irq().raise_counts()[0]);
    if current_mask.is_some_and(|mask| mask == next_mask) {
        return;
    }
    bus.set_port1_buttons(emulator_core::ButtonState::from_bits(next_mask));
    *current_mask = Some(next_mask);
}
