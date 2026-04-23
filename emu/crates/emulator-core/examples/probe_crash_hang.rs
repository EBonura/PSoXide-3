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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

fn main() {
    let total_steps: u64 = std::env::var("PSOXIDE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000_000);
    let report_interval: u64 = std::env::var("PSOXIDE_REPORT_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000_000);
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let disc_path = std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin".into()
    });
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

    let bios = std::fs::read(&bios_path).expect("BIOS");
    let disc = std::fs::read(&disc_path).expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    bus.attach_digital_pad_port1();
    let mut cpu = Cpu::new();

    // Track PC hits in 64-byte buckets over the last N instructions.
    // A sustained hot bucket = a spin loop.
    let bucket_size = 64u32;
    let mut pc_buckets: HashMap<u32, u64> = HashMap::new();

    let mut retired = 0u64;
    let mut cycles_at_last_pump = 0u64;
    let mut cycles_at_last_report = 0u64;
    let mut last_retired = 0u64;
    let mut err: Option<String> = None;
    let mut current_pad_mask = None;
    sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);

    while retired < total_steps {
        match cpu.step(&mut bus) {
            Ok(_) => {}
            Err(e) => {
                err = Some(format!("{e:?}"));
                break;
            }
        }
        sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
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
        if retired - last_retired > report_interval {
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
    eprintln!("pad mask: 0x{held_buttons:04x}");
    eprintln!("pad pulses: {}", format_pad_pulses(&pad_pulses));
    eprintln!("retired instructions: {retired}");
    eprintln!("cycles: {}", bus.cycles());
    eprintln!("cpu.pc = 0x{:08x}", cpu.pc());
    if let Some(e) = err {
        eprintln!("last error: {e}");
    }
    eprintln!("irq raise counts: {:?}", bus.irq().raise_counts());
    eprintln!(
        "cdrom commands dispatched: {}, last cmd: 0x{:02x}",
        bus.cdrom.commands_dispatched(),
        bus.cdrom.last_command()
    );
    eprintln!("spu samples produced: {}", bus.spu.samples_produced());
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

fn format_pad_pulses(pulses: &[PadPulse]) -> String {
    if pulses.is_empty() {
        return "(none)".into();
    }
    pulses
        .iter()
        .map(|pulse| {
            format!(
                "0x{:04x}@{}+{}",
                pulse.mask, pulse.start_vblank, pulse.frames
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
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
