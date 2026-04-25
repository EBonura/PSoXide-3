//! Frontend-shaped benchmark. Mirrors what the GUI does per-frame
//! and times each phase separately, so we can see where the time
//! actually goes during real gameplay (instead of the unrealistic
//! "replay all packets back-to-back, no per-frame sync" of
//! `bench_rasterizer`).
//!
//! Per simulated frame:
//!   1. Run the CPU emulator forward by `--cycles-per-frame` cycles
//!      (~558k for ~30 fps Tekken 3 boot, the ratio the frontend
//!      uses).
//!   2. `sync_vram_from_cpu` — 1 MiB CPU-to-GPU bounce.
//!   3. Drain `cmd_log` and `replay_packet` each entry through the
//!      compute backend.
//!   4. `download_vram` — 1 MiB GPU-to-CPU bounce + device poll.
//!
//! Outputs per-phase wall-clock totals over the whole run, plus
//! per-frame averages, so we can see whether `sync` / `replay` /
//! `download` is the bottleneck.
//!
//! ```bash
//! PSOXIDE_DISC="/path/Tekken3.cue" \
//!   cargo run --release -p psx-gpu-compute \
//!     --example bench_frontend_loop -- --frames 600
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use psx_gpu_compute::ComputeBackend;
use psx_iso::Disc;

fn parse_args() -> (PathBuf, PathBuf, u64, u64, bool) {
    let mut bios = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");
    let mut disc = PathBuf::from(std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue".into()
    }));
    // ~558_000 cycles per frame matches the frontend's emulator
    // tick rate (PSX master / 60Hz). We expose it because games
    // run at 30 fps for some scenes.
    let mut cycles_per_frame: u64 = 558_000;
    let mut frames: u64 = 600;
    let mut disable_cpu_raster = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bios" => bios = PathBuf::from(it.next().unwrap()),
            "--disc" => disc = PathBuf::from(it.next().unwrap()),
            "--cycles-per-frame" => cycles_per_frame = it.next().unwrap().parse().unwrap(),
            "--frames" => frames = it.next().unwrap().parse().unwrap(),
            "--disable-cpu-raster" => disable_cpu_raster = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    (bios, disc, cycles_per_frame, frames, disable_cpu_raster)
}

fn load_disc(p: &std::path::Path) -> Disc {
    if p.extension().and_then(|e| e.to_str()) == Some("cue") {
        psoxide_settings::library::load_disc_from_cue(p).expect("load cue")
    } else {
        Disc::from_bin(std::fs::read(p).expect("read bin"))
    }
}

fn main() {
    let (bios, disc_path, cycles_per_frame, frames, disable_cpu_raster) = parse_args();
    eprintln!("[bench] bios={}", bios.display());
    eprintln!("[bench] disc={}", disc_path.display());
    eprintln!(
        "[bench] cycles/frame={cycles_per_frame} frames={frames} disable_cpu_raster={disable_cpu_raster}"
    );

    let bios_bytes = std::fs::read(&bios).expect("bios readable");
    let mut bus = Bus::new(bios_bytes).expect("bus");
    let mut cpu = Cpu::new();
    let disc = load_disc(&disc_path);
    bus.cdrom.insert_disc(Some(disc));
    bus.gpu.enable_pixel_tracer();
    // `--disable-cpu-raster` was wired to a runtime flag that has
    // since been reverted; leaving the CLI flag in for back-compat
    // but it's now a no-op.
    let _ = disable_cpu_raster;

    let mut backend = ComputeBackend::new_headless();

    // Warm up: run a bit before measuring so the BIOS isn't in its
    // "sleep waiting for CD" state.
    eprintln!("[bench] warming up 50M cycles...");
    for _ in 0..50_000_000u64 {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }
    let _ = std::mem::take(&mut bus.gpu.cmd_log);
    if let Some(owner) = bus.gpu.pixel_owner.as_mut() {
        owner.fill(u32::MAX);
    }

    eprintln!("[bench] measuring {frames} frames...");
    let mut t_emu = Duration::ZERO;
    let mut t_sync = Duration::ZERO;
    let mut t_replay = Duration::ZERO;
    let mut t_download = Duration::ZERO;
    let mut total_packets = 0u64;
    let started = Instant::now();

    for _ in 0..frames {
        // Phase 1: CPU emulation.
        let t = Instant::now();
        for _ in 0..cycles_per_frame {
            if cpu.step(&mut bus).is_err() {
                break;
            }
        }
        t_emu += t.elapsed();

        // Phase 2: sync VRAM from CPU.
        let t = Instant::now();
        backend.sync_vram_from_cpu(bus.gpu.vram.words());
        t_sync += t.elapsed();

        // Phase 3: drain + replay cmd_log.
        let log = std::mem::take(&mut bus.gpu.cmd_log);
        total_packets += log.len() as u64;
        let t = Instant::now();
        for entry in &log {
            backend.replay_packet(entry);
        }
        t_replay += t.elapsed();
        if let Some(owner) = bus.gpu.pixel_owner.as_mut() {
            owner.fill(u32::MAX);
        }

        // Phase 4: download VRAM (incl. device poll).
        let t = Instant::now();
        let _gpu_words = backend.download_vram();
        t_download += t.elapsed();
    }

    let total = started.elapsed();
    let f = frames as f64;
    let pf = |d: Duration| d.as_secs_f64() * 1000.0 / f;
    let pct = |d: Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64();

    eprintln!();
    eprintln!("=== Frontend-loop timings ===");
    eprintln!(
        "  total wall-clock: {:.1} ms ({:.1} fps if every frame is one redraw)",
        total.as_secs_f64() * 1000.0,
        f / total.as_secs_f64(),
    );
    eprintln!(
        "  emulation:        {:.2} ms/frame ({:.1}%)",
        pf(t_emu),
        pct(t_emu)
    );
    eprintln!(
        "  sync_vram:        {:.2} ms/frame ({:.1}%)",
        pf(t_sync),
        pct(t_sync)
    );
    eprintln!(
        "  replay_cmd_log:   {:.2} ms/frame ({:.1}%) [{:.1} pkts/frame avg]",
        pf(t_replay),
        pct(t_replay),
        total_packets as f64 / f,
    );
    eprintln!(
        "  download_vram:    {:.2} ms/frame ({:.1}%)",
        pf(t_download),
        pct(t_download)
    );

    let budget_60fps = 1000.0 / 60.0;
    let budget_30fps = 1000.0 / 30.0;
    let total_per_frame_ms = total.as_secs_f64() * 1000.0 / f;
    eprintln!();
    if total_per_frame_ms <= budget_60fps {
        eprintln!("  HEADROOM @ 60fps: {:.2} ms left", budget_60fps - total_per_frame_ms);
    } else if total_per_frame_ms <= budget_30fps {
        eprintln!(
            "  60fps OVER by {:.2} ms; @30fps headroom: {:.2} ms",
            total_per_frame_ms - budget_60fps,
            budget_30fps - total_per_frame_ms
        );
    } else {
        eprintln!(
            "  30fps OVER by {:.2} ms — won't sustain 30 fps",
            total_per_frame_ms - budget_30fps
        );
    }
}
