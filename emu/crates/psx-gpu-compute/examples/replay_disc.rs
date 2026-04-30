//! Headless validation of Phase C -- boot a disc on the CPU
//! rasterizer, run for some cycles, drain the cmd_log onto the
//! compute backend, then compare CPU VRAM vs GPU VRAM.
//!
//! This mirrors what the frontend does at runtime when
//! `--gpu-compute` is passed, minus the window. Useful for:
//!   - Verifying replay parity on a real game without driving
//!     the GUI.
//!   - CI: catch divergence regressions when the parity-tracking
//!     CPU rasterizer changes.
//!
//! ```bash
//! PSOXIDE_DISC="/path/to/Tekken3.cue" \
//!   cargo run --release -p psx-gpu-compute \
//!     --example replay_disc -- --steps 50000000
//! ```
//!
//! Output: per-checkpoint hashes, divergence summary, top
//! `unhandled` GP0 opcodes. Exits non-zero if final VRAM hashes
//! differ.
//!
//! Requires `emulator-core` to expose disc support; uses
//! `psx_iso::Disc::from_bin` for raw `.bin` files and
//! `psoxide_settings::library::load_disc_from_cue` for `.cue` images.

use std::path::PathBuf;

use emulator_core::{Bus, Cpu};
use psx_gpu_compute::ComputeBackend;
use psx_iso::Disc;

fn parse_args() -> (PathBuf, PathBuf, u64, u64) {
    let mut bios = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");
    let mut disc = PathBuf::from(std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue".into()
    }));
    let mut steps: u64 = 50_000_000;
    let mut chunk: u64 = 5_000_000;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bios" => bios = PathBuf::from(it.next().unwrap()),
            "--disc" => disc = PathBuf::from(it.next().unwrap()),
            "--steps" => steps = it.next().unwrap().parse().unwrap(),
            "--chunk" => chunk = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg: {other}"),
        }
    }
    (bios, disc, steps, chunk)
}

fn load_disc(p: &std::path::Path) -> Disc {
    if p.extension().and_then(|e| e.to_str()) == Some("cue") {
        psoxide_settings::library::load_disc_from_cue(p).expect("load cue")
    } else {
        Disc::from_bin(std::fs::read(p).expect("read bin"))
    }
}

fn vram_hash(words: &[u16]) -> u64 {
    // FNV-1a-64 over the BGR15 cells.
    let mut h: u64 = 0xCBF29CE484222325;
    for &w in words {
        h ^= w as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

fn main() {
    let (bios, disc_path, steps, chunk) = parse_args();
    eprintln!("[replay] bios={}", bios.display());
    eprintln!("[replay] disc={}", disc_path.display());
    eprintln!("[replay] steps={steps} chunk={chunk}");

    let bios_bytes = std::fs::read(&bios).expect("bios readable");
    let mut bus = Bus::new(bios_bytes).expect("bus");
    let mut cpu = Cpu::new();
    let disc = load_disc(&disc_path);
    bus.cdrom.insert_disc(Some(disc));

    // Required for replay: CPU rasterizer must populate `cmd_log`.
    bus.gpu.enable_pixel_tracer();

    let mut backend = ComputeBackend::new_headless();

    let mut cursor: u64 = 0;
    let mut chunk_idx: u32 = 0;
    let mut mismatches: u32 = 0;
    let started = std::time::Instant::now();

    while cursor < steps {
        let chunk_size = chunk.min(steps - cursor);
        for _ in 0..chunk_size {
            if cpu.step(&mut bus).is_err() {
                break;
            }
        }
        cursor += chunk_size;
        chunk_idx += 1;

        // Sync VRAM from CPU → compute, then replay this chunk's
        // packets. Drain so cmd_log doesn't grow unbounded.
        backend.sync_vram_from_cpu(bus.gpu.vram.words());
        let log = std::mem::take(&mut bus.gpu.cmd_log);
        for entry in &log {
            backend.replay_packet(entry);
        }
        if let Some(owner) = bus.gpu.pixel_owner.as_mut() {
            owner.fill(u32::MAX);
        }

        let cpu_hash = vram_hash(bus.gpu.vram.words());
        let gpu_words = backend.download_vram();
        let gpu_hash = vram_hash(&gpu_words);
        let elapsed = started.elapsed().as_secs_f64();
        let mips = cursor as f64 / elapsed / 1.0e6;
        let agree = cpu_hash == gpu_hash;
        if !agree {
            mismatches += 1;
        }
        eprintln!(
            "[{chunk_idx:>3}] step {cursor:>11}  cpu=0x{cpu_hash:016x} \
             gpu=0x{gpu_hash:016x}  {}  packets={:<5}  ({:.1} MIPS)",
            if agree { "OK" } else { "DIVERGE" },
            log.len(),
            mips,
        );
    }

    eprintln!();
    eprintln!("=== Replay summary ===");
    eprintln!("  total chunks: {chunk_idx}, mismatches: {mismatches}",);
    if !backend.unhandled.is_empty() {
        eprintln!("  unhandled GP0 opcodes (replay no-op):");
        for (op, count) in &backend.unhandled {
            eprintln!("    0x{op:02X}: {count}");
        }
    } else {
        eprintln!("  no unhandled GP0 opcodes encountered");
    }

    if mismatches > 0 {
        std::process::exit(1);
    }
}
