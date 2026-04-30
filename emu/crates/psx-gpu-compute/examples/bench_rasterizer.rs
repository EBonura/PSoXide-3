//! Wall-clock comparison of the CPU rasterizer vs the GPU compute
//! replay path, both fed the same captured cmd_log from a real game.
//!
//! The benchmark isolates rasterization throughput: we run the CPU
//! emulator just long enough to capture a representative cmd_log
//! (state setters + draws + bulk writes), then replay that log
//! through both paths against a fresh VRAM and time only the
//! replay loop. CPU-to-VRAM uploads (0xA0..=0xBF) are skipped on
//! both sides because their pixel data isn't in cmd_log -- the GPU
//! compute backend already short-circuits them, and we mirror that
//! on the CPU side so the comparison stays apples-to-apples.
//!
//! The GPU side's timing includes the final `download_vram` so we
//! account for the GPU→CPU bounce that the frontend pays each
//! frame; that's the realistic "drop-in replacement" cost.
//!
//! ```bash
//! PSOXIDE_DISC="/path/Tekken3.cue" \
//!   cargo run --release -p psx-gpu-compute \
//!     --example bench_rasterizer -- --steps 350000000
//! ```
//!
//! Output: ms total, ms per packet, packets/sec for each path,
//! and the GPU/CPU ratio.

use std::path::PathBuf;
use std::time::Instant;

use emulator_core::gpu::GpuCmdLogEntry;
use emulator_core::{Bus, Cpu, Gpu};
use psx_gpu_compute::ComputeBackend;
use psx_iso::Disc;

fn parse_args() -> (PathBuf, PathBuf, u64) {
    let mut bios = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");
    let mut disc = PathBuf::from(std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue".into()
    }));
    let mut steps: u64 = 350_000_000;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bios" => bios = PathBuf::from(it.next().unwrap()),
            "--disc" => disc = PathBuf::from(it.next().unwrap()),
            "--steps" => steps = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg: {other}"),
        }
    }
    (bios, disc, steps)
}

fn load_disc(p: &std::path::Path) -> Disc {
    if p.extension().and_then(|e| e.to_str()) == Some("cue") {
        psoxide_settings::library::load_disc_from_cue(p).expect("load cue")
    } else {
        Disc::from_bin(std::fs::read(p).expect("read bin"))
    }
}

fn capture_cmd_log(
    bios: &std::path::Path,
    disc_path: &std::path::Path,
    steps: u64,
) -> Vec<GpuCmdLogEntry> {
    let bios_bytes = std::fs::read(bios).expect("bios readable");
    let mut bus = Bus::new(bios_bytes).expect("bus");
    let mut cpu = Cpu::new();
    let disc = load_disc(disc_path);
    bus.cdrom.insert_disc(Some(disc));
    bus.gpu.enable_pixel_tracer();

    eprintln!("[bench] capturing cmd_log over {steps} cycles...");
    let mut accumulated: Vec<GpuCmdLogEntry> = Vec::new();
    let chunk = 5_000_000u64;
    let mut cursor = 0u64;
    while cursor < steps {
        let n = chunk.min(steps - cursor);
        for _ in 0..n {
            if cpu.step(&mut bus).is_err() {
                break;
            }
        }
        cursor += n;
        // Drain so cmd_log doesn't grow without bound between
        // its `pixel_owner` resets -- we fold each chunk into our
        // accumulator with re-numbered indices later if needed.
        let log = std::mem::take(&mut bus.gpu.cmd_log);
        accumulated.extend(log);
        if let Some(owner) = bus.gpu.pixel_owner.as_mut() {
            owner.fill(u32::MAX);
        }
    }
    eprintln!("[bench] captured {} packets", accumulated.len());
    let mut hist: std::collections::BTreeMap<&'static str, u64> = std::collections::BTreeMap::new();
    for e in &accumulated {
        let k = match e.opcode {
            // `0x02` is GP0 fill-rectangle. Match it before the
            // 0x00..=0x1F state/nop range so it doesn't get
            // swallowed silently.
            0x02 => "fill",
            0x00..=0x1F => "state/nop",
            0x20..=0x23 => "mono-tri",
            0x24..=0x27 => "tex-tri",
            0x28..=0x2B => "mono-quad",
            0x2C..=0x2F => "tex-quad",
            0x30..=0x33 => "shaded-tri",
            0x34..=0x37 => "shaded-tex-tri",
            0x38..=0x3B => "shaded-quad",
            0x3C..=0x3F => "shaded-tex-quad",
            0x40..=0x5F => "lines",
            0x60..=0x7F => "rect",
            0x80..=0x9F => "vram-copy",
            0xA0..=0xBF => "cpu-upload",
            0xC0..=0xDF => "vram-readback",
            0xE0..=0xE6 => "tpage/draw-area",
            _ => "other",
        };
        *hist.entry(k).or_insert(0) += 1;
    }
    eprintln!("[bench] packet histogram:");
    for (k, n) in &hist {
        eprintln!("  {k:>16}: {n:>6}");
    }
    accumulated
}

/// Replay `log` onto a fresh CPU `Gpu` instance, timed. CPU-to-VRAM
/// uploads (0xA0..=0xBF) are skipped -- their pixel data lives
/// outside the cmd_log so re-playing the 3-word header would just
/// stick the GPU in a half-completed transfer state. Mirrors what
/// the GPU compute path does for parity, so the comparison is fair.
fn run_cpu(log: &[GpuCmdLogEntry]) -> (std::time::Duration, Vec<u16>, usize) {
    let mut gpu = Gpu::new();
    let mut replayed = 0usize;
    let start = Instant::now();
    for entry in log {
        if (0xA0..=0xBF).contains(&entry.opcode) {
            continue;
        }
        for &w in &entry.fifo {
            gpu.gp0_push(w);
        }
        replayed += 1;
    }
    let elapsed = start.elapsed();
    (elapsed, gpu.vram.words().to_vec(), replayed)
}

/// Replay `log` onto a fresh `ComputeBackend`. Timing includes the
/// final `download_vram` so the comparison accounts for the
/// GPU→CPU readback that the frontend pays once per frame to
/// blit into the egui texture.
fn run_gpu(log: &[GpuCmdLogEntry]) -> (std::time::Duration, Vec<u16>, usize) {
    let mut backend = ComputeBackend::new_headless();
    let mut replayed = 0usize;
    let start = Instant::now();
    for entry in log {
        backend.replay_packet(entry);
        replayed += 1;
    }
    let vram = backend.download_vram();
    let elapsed = start.elapsed();
    (elapsed, vram, replayed)
}

fn main() {
    let (bios, disc_path, steps) = parse_args();
    eprintln!("[bench] bios={}", bios.display());
    eprintln!("[bench] disc={}", disc_path.display());
    eprintln!("[bench] steps={steps}");
    let log = capture_cmd_log(&bios, &disc_path, steps);
    if log.is_empty() {
        eprintln!("[bench] no packets captured — nothing to compare");
        return;
    }

    eprintln!();
    eprintln!("[bench] === CPU rasterizer ===");
    let (cpu_t, _cpu_vram, cpu_n) = run_cpu(&log);
    let cpu_ms = cpu_t.as_secs_f64() * 1000.0;
    let cpu_us_per = (cpu_t.as_secs_f64() * 1_000_000.0) / cpu_n as f64;
    let cpu_pps = cpu_n as f64 / cpu_t.as_secs_f64();
    eprintln!(
        "  {} packets in {:.2} ms ({:.2} us/pkt, {:.2}M packets/s)",
        cpu_n,
        cpu_ms,
        cpu_us_per,
        cpu_pps / 1e6,
    );

    eprintln!();
    eprintln!("[bench] === GPU compute (incl. download_vram) ===");
    let (gpu_t, _gpu_vram, gpu_n) = run_gpu(&log);
    let gpu_ms = gpu_t.as_secs_f64() * 1000.0;
    let gpu_us_per = (gpu_t.as_secs_f64() * 1_000_000.0) / gpu_n as f64;
    let gpu_pps = gpu_n as f64 / gpu_t.as_secs_f64();
    eprintln!(
        "  {} packets in {:.2} ms ({:.2} us/pkt, {:.2}M packets/s)",
        gpu_n,
        gpu_ms,
        gpu_us_per,
        gpu_pps / 1e6,
    );

    eprintln!();
    eprintln!("[bench] === Ratio ===");
    let ratio = gpu_t.as_secs_f64() / cpu_t.as_secs_f64();
    if ratio < 1.0 {
        eprintln!(
            "  GPU compute is {:.2}x FASTER than CPU rasterizer",
            1.0 / ratio
        );
    } else {
        eprintln!("  GPU compute is {:.2}x SLOWER than CPU rasterizer", ratio);
    }
}
