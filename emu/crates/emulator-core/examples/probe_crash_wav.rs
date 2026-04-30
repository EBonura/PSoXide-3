//! Dump Crash's SPU output as a WAV file so we can listen to (or
//! spectrum-analyse) the "garbled audio" the user reports. Runs
//! long enough to hit the game's first audio output, drains every
//! sample, and writes a standard-format 44.1 kHz 16-bit stereo
//! WAV -- playable directly in any media player, and trivially
//! diff-able against Redux's audio output for parity work.
//!
//! ```bash
//! PSOXIDE_AUDIO_SECONDS=20 cargo run -p emulator-core --example probe_crash_wav --release
//! # → /tmp/crash_audio.wav
//! ```

use emulator_core::{spu::SAMPLE_CYCLES, Bus, ButtonState, Cpu};
use psx_iso::Disc;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

#[derive(Default)]
struct QueueStats {
    pumps: u64,
    cd_enabled_pumps: u64,
    cd_volume_nonzero_pumps: u64,
    cdrom_nonempty_pumps: u64,
    spu_cd_nonempty_before_pumps: u64,
    max_cd_vol_l: u16,
    max_cd_vol_r: u16,
    max_cdrom_queue: usize,
    max_spu_cd_queue_before: usize,
    max_spu_cd_queue_after: usize,
}

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .unwrap_or_else(|_| "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN".into());
    let disc_path = std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin".into()
    });
    let out_path = std::env::var("PSOXIDE_WAV").unwrap_or_else(|_| "/tmp/crash_audio.wav".into());
    let seconds = std::env::var("PSOXIDE_AUDIO_SECONDS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(10.0)
        .max(0.1);
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

    // Run until we have the requested amount of audio. Pump from the
    // actual emulated-cycle delta, matching the frontend path.
    let target_samples = (seconds * 44_100.0).round() as usize;
    let mut collected: Vec<(i16, i16)> = Vec::with_capacity(target_samples as usize);
    let mut audio_cycle_accum = 0u64;
    let mut steps = 0u64;
    let mut last_report_second = 0usize;
    let mut current_pad_mask = None;
    let mut queue_stats = QueueStats::default();
    sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);

    while collected.len() < target_samples && steps < 4_000_000_000 {
        let cycles_before = bus.cycles();
        if cpu.step(&mut bus).is_err() {
            break;
        }
        sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
        steps += 1;
        let cycles_after = bus.cycles();
        audio_cycle_accum = audio_cycle_accum.saturating_add(cycles_after - cycles_before);
        let sample_count = (audio_cycle_accum / SAMPLE_CYCLES) as usize;
        audio_cycle_accum %= SAMPLE_CYCLES;
        if sample_count > 0 {
            queue_stats.pumps += 1;
            let spucnt = bus.spu.spucnt();
            let cd_vol_l = bus.spu.read16(0x1F80_1DB0);
            let cd_vol_r = bus.spu.read16(0x1F80_1DB2);
            let cdrom_q = bus.cdrom.cd_audio_queue_len();
            let spu_cd_q_before = bus.spu.cd_audio_queue_len();
            if spucnt & 1 != 0 {
                queue_stats.cd_enabled_pumps += 1;
            }
            if cd_vol_l != 0 || cd_vol_r != 0 {
                queue_stats.cd_volume_nonzero_pumps += 1;
            }
            queue_stats.max_cd_vol_l = queue_stats.max_cd_vol_l.max(cd_vol_l);
            queue_stats.max_cd_vol_r = queue_stats.max_cd_vol_r.max(cd_vol_r);
            queue_stats.max_cdrom_queue = queue_stats.max_cdrom_queue.max(cdrom_q);
            queue_stats.max_spu_cd_queue_before =
                queue_stats.max_spu_cd_queue_before.max(spu_cd_q_before);
            if cdrom_q != 0 {
                queue_stats.cdrom_nonempty_pumps += 1;
            }
            if spu_cd_q_before != 0 {
                queue_stats.spu_cd_nonempty_before_pumps += 1;
            }

            bus.run_spu_samples(sample_count);
            queue_stats.max_spu_cd_queue_after = queue_stats
                .max_spu_cd_queue_after
                .max(bus.spu.cd_audio_queue_len());
            let samples = bus.spu.drain_audio();
            collected.extend_from_slice(&samples);
            if collected.len() > target_samples {
                collected.truncate(target_samples);
            }
            let report_second = collected.len() / 44_100;
            if report_second > last_report_second {
                last_report_second = report_second;
                eprintln!(
                    "collected {:>7} samples  ({:.1}s of audio)  steps={}  cpu_cyc={}  spucnt=0x{:04x} cdvol={:04x}/{:04x} cdq={}/{}",
                    collected.len(),
                    collected.len() as f32 / 44_100.0,
                    steps,
                    bus.cycles(),
                    bus.spu.spucnt(),
                    bus.spu.read16(0x1F80_1DB0),
                    bus.spu.read16(0x1F80_1DB2),
                    bus.cdrom.cd_audio_queue_len(),
                    bus.spu.cd_audio_queue_len(),
                );
            }
        }
    }

    // Write a standard little-endian PCM WAV.
    let mut f = BufWriter::new(File::create(&out_path).expect("create wav"));
    let sample_rate = 44_100u32;
    let channels = 2u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = (collected.len() * 4) as u32;
    let file_size = 36 + data_size;

    f.write_all(b"RIFF").unwrap();
    f.write_all(&file_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap(); // fmt chunk size
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    f.write_all(&channels.to_le_bytes()).unwrap();
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&byte_rate.to_le_bytes()).unwrap();
    f.write_all(&block_align.to_le_bytes()).unwrap();
    f.write_all(&bits_per_sample.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for (l, r) in &collected {
        f.write_all(&l.to_le_bytes()).unwrap();
        f.write_all(&r.to_le_bytes()).unwrap();
    }
    drop(f);

    // Summary stats so we can tell "garbled" from "silent" from "clipping".
    let mut peak_l: i32 = 0;
    let mut peak_r: i32 = 0;
    let mut nonzero = 0usize;
    let mut clipping = 0usize;
    for (l, r) in &collected {
        peak_l = peak_l.max(l.unsigned_abs() as i32);
        peak_r = peak_r.max(r.unsigned_abs() as i32);
        if *l != 0 || *r != 0 {
            nonzero += 1;
        }
        if *l == i16::MAX || *l == i16::MIN || *r == i16::MAX || *r == i16::MIN {
            clipping += 1;
        }
    }
    println!();
    println!("=== Crash audio capture ===");
    println!("Path:     {out_path}");
    println!("Duration: {:.2} s", collected.len() as f32 / 44_100.0);
    println!("Samples:  {}", collected.len());
    println!(
        "Nonzero:  {nonzero} ({:.1}%)",
        100.0 * nonzero as f32 / collected.len() as f32
    );
    println!(
        "Peak:     L={peak_l}  R={peak_r} (max would be {})",
        i16::MAX
    );
    println!("Clipping: {clipping} samples at the i16 limit");
    println!("CPU steps run: {steps}");
    println!("VBlanks:  {}", bus.irq().raise_counts()[0]);
    println!("SPUCNT:   0x{:04x}", bus.spu.spucnt());
    println!(
        "CD_VOL:   L=0x{:04x} R=0x{:04x}",
        bus.spu.read16(0x1F80_1DB0),
        bus.spu.read16(0x1F80_1DB2)
    );
    println!(
        "CD queues: pumps={} cd_enabled={} cdvol_nonzero={} cdrom_nonempty={} spu_nonempty_before={} max_cdvol={:04x}/{:04x} max_cdrom={} max_spu_before={} max_spu_after={}",
        queue_stats.pumps,
        queue_stats.cd_enabled_pumps,
        queue_stats.cd_volume_nonzero_pumps,
        queue_stats.cdrom_nonempty_pumps,
        queue_stats.spu_cd_nonempty_before_pumps,
        queue_stats.max_cd_vol_l,
        queue_stats.max_cd_vol_r,
        queue_stats.max_cdrom_queue,
        queue_stats.max_spu_cd_queue_before,
        queue_stats.max_spu_cd_queue_after
    );
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
    bus.set_port1_buttons(ButtonState::from_bits(next_mask));
    *current_mask = Some(next_mask);
}
