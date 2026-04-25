//! Zoom in on a known-small divergence window by fast-forwarding
//! both emulators silently to a start step, then capturing per-step
//! records for a small window. Much faster than a full trace cache —
//! a 10 K-step window costs ~0.3 s of tracing vs 14 GiB of cache.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_fine_divergence -- 90140000 10000 "/path/to/game.bin"
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const TRACE_STEP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

fn main() {
    let start: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_fine_divergence <start_step> <window> [disc]");
    let window: u32 = std::env::args()
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

    let bios_path = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");

    // --- Ours ---
    eprintln!("[ours] fast-forwarding {start} steps...");
    let t0 = Instant::now();
    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc = disc_support::load_disc_path(std::path::Path::new(p)).expect("disc");
        bus.cdrom.insert_disc(Some(disc));
    }
    if wants_pad {
        bus.attach_digital_pad_port1();
    }
    let mut cpu = Cpu::new();
    let mut current_pad_mask = None;
    if wants_pad {
        sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
    }
    for _ in 0..start {
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
    }
    eprintln!(
        "[ours] at start step, pc=0x{:08x}, cycles={}, elapsed={:.1}s",
        cpu.pc(),
        bus.cycles(),
        t0.elapsed().as_secs_f64()
    );

    // Capture per-step records for window. Match Redux folding
    // semantics: preserve the USER-side pc+instr across the ISR
    // fold, only advance tick+gprs. Getting this wrong (overwriting
    // instr with an ISR-body instruction) produces phantom
    // divergences at the fold step.
    let mut our_records: Vec<psx_trace::InstructionRecord> = Vec::with_capacity(window as usize);
    for _ in 0..window {
        let was_in_isr = cpu.in_isr();
        let mut rec = cpu.step_traced(&mut bus).expect("step");
        if wants_pad {
            sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
        }
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                let r = cpu.step_traced(&mut bus).expect("isr step");
                if wants_pad {
                    sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
                }
                rec.tick = r.tick;
                rec.gprs = r.gprs;
            }
        }
        our_records.push(rec);
    }
    eprintln!("[ours] captured {window} records");

    // --- Redux ---
    eprintln!("[redux] launching, fast-forwarding {start} steps...");
    let t0 = Instant::now();
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios_path, lua).expect("Redux resolves");
    if let Some(ref p) = disc_path {
        config = config.with_disc(PathBuf::from(p));
    }
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");
    let ff_timeout = Duration::from_secs((start / 200_000).max(60));
    if wants_pad && start > 0 {
        let pulse_tuples = pad_pulses
            .iter()
            .map(|pulse| (pulse.mask, pulse.start_vblank, pulse.frames))
            .collect::<Vec<_>>();
        redux
            .run_checkpoint_pad(
                start,
                start.max(1),
                1,
                held_buttons,
                &pulse_tuples,
                ff_timeout,
                |_step, _tick, _pc| Ok(()),
            )
            .expect("fast-forward with pad");
    } else if start > 0 {
        redux.run(start, ff_timeout).expect("fast-forward");
    }
    eprintln!(
        "[redux] ff done in {:.1}s, tracing {window} steps...",
        t0.elapsed().as_secs_f64()
    );
    let trace = redux.step(window, TRACE_STEP_TIMEOUT).expect("step");
    eprintln!("[redux] trace done, {} records", trace.len());
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();

    // --- Compare: full record match (tick+pc+instr+gprs) ---
    println!();
    println!("=== fine divergence scan from step {start} (+{window}) ===");
    let mut first: Option<usize> = None;
    for i in 0..window as usize {
        let o = &our_records[i];
        let r = &trace[i];
        if o.tick == r.tick && o.pc == r.pc && o.instr == r.instr && o.gprs == r.gprs {
            continue;
        }
        first = Some(i);
        break;
    }
    match first {
        None => println!("All {window} records match — divergence past window"),
        Some(idx) => {
            let absolute = start + idx as u64 + 1;
            println!();
            println!("First divergence at window idx {idx} = absolute step {absolute}");
            println!();
            let lo = idx.saturating_sub(6);
            let hi = (idx + 1).min(window as usize - 1);
            println!("--- last 6 matching + divergence ---");
            for j in lo..=hi {
                let o = &our_records[j];
                let r = &trace[j];
                let marker = if j == idx { ">>>" } else { "   " };
                println!(
                    "{marker} idx={j:>5} step={:>10} pc=0x{:08x} instr=0x{:08x} tick={:>10}",
                    start + j as u64 + 1,
                    o.pc,
                    o.instr,
                    o.tick,
                );
                if o.gprs != r.gprs {
                    for reg in 0..32 {
                        if o.gprs[reg] != r.gprs[reg] {
                            println!(
                                "         $r{reg:>2}: ours=0x{:08x}  redux=0x{:08x}",
                                o.gprs[reg], r.gprs[reg]
                            );
                        }
                    }
                }
                if o.pc != r.pc || o.instr != r.instr || o.tick != r.tick {
                    println!(
                        "         redux pc=0x{:08x} instr=0x{:08x} tick={:>10}",
                        r.pc, r.instr, r.tick,
                    );
                }
            }
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
