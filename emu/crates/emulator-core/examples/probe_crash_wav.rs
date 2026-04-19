//! Dump Crash's SPU output as a WAV file so we can listen to (or
//! spectrum-analyse) the "garbled audio" the user reports. Runs
//! long enough to hit the game's first audio output, drains every
//! sample, and writes a standard-format 44.1 kHz 16-bit stereo
//! WAV — playable directly in any media player, and trivially
//! diff-able against Redux's audio output for parity work.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_crash_wav --release
//! # → /tmp/crash_audio.wav
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = std::fs::read(
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin",
    )
    .expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    let mut cpu = Cpu::new();

    // Run until we have ~10 seconds of audio. At 44.1 kHz stereo
    // that's ~441k samples. Collect every sample drained per frame.
    let target_samples = 441_000u64;
    let mut collected: Vec<(i16, i16)> = Vec::with_capacity(target_samples as usize);
    let mut cycles_at_last_pump = 0u64;
    let mut steps = 0u64;
    while collected.len() < target_samples as usize && steps < 2_000_000_000 {
        if cpu.step(&mut bus).is_err() {
            break;
        }
        steps += 1;
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let samples = bus.spu.drain_audio();
            collected.extend_from_slice(&samples);
            if collected.len() % (44_100 * 2) < 735 {
                eprintln!(
                    "collected {:>7} samples  ({:.1}s of audio)  steps={}  cpu_cyc={}",
                    collected.len(),
                    collected.len() as f32 / 44_100.0,
                    steps,
                    bus.cycles(),
                );
            }
        }
    }

    // Write a standard little-endian PCM WAV.
    let path = "/tmp/crash_audio.wav";
    let mut f = BufWriter::new(File::create(path).expect("create wav"));
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
    f.write_all(&16u32.to_le_bytes()).unwrap();       // fmt chunk size
    f.write_all(&1u16.to_le_bytes()).unwrap();        // PCM
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
    println!("Path:     {path}");
    println!("Duration: {:.2} s", collected.len() as f32 / 44_100.0);
    println!("Samples:  {}", collected.len());
    println!("Nonzero:  {nonzero} ({:.1}%)", 100.0 * nonzero as f32 / collected.len() as f32);
    println!("Peak:     L={peak_l}  R={peak_r} (max would be {})", i16::MAX);
    println!("Clipping: {clipping} samples at the i16 limit");
    println!("CPU steps run: {steps}");
}
