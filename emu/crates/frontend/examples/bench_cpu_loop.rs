//! CPU-path frontend-shaped benchmark.
//!
//! Mirrors what the GUI does each redraw on the *CPU rasterizer*
//! path (no `--gpu-compute`), and times the phases independently
//! so we can see where the per-frame budget actually goes.
//!
//! Per simulated frame:
//!   1. Run the CPU emulator forward by `--cycles-per-frame`
//!      cycles. This is the "everything inside emulator-core":
//!      MIPS interpreter, bus, DMA, CDROM, SPU, GPU CPU rasterizer.
//!   2. Convert the full 1024×512 VRAM to RGBA8 — what the
//!      frontend does for the VRAM viewer panel.
//!   3. Convert the active display rect to RGBA8 — what the
//!      frontend does for the central framebuffer panel.
//!
//! The bench skips the wgpu texture-upload + present steps because
//! those are GUI-side and dominated by the swapchain, not the
//! emulator. If frame time is still over budget after subtracting
//! these phases, the wgpu side is worth a separate probe.
//!
//! ```bash
//! PSOXIDE_DISC="/path/Tekken3.cue" \
//!   cargo run --release -p frontend --example bench_cpu_loop -- \
//!     --frames 600
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu, VRAM_HEIGHT, VRAM_WIDTH};
use psx_iso::Disc;

struct Args {
    bios: PathBuf,
    disc: PathBuf,
    cycles_per_frame: u64,
    frames: u64,
    warmup: u64,
    /// Skip the VRAM → RGBA conversion phase. Lets us see the real
    /// frame cost when the user doesn't have the VRAM viewer panel
    /// open (the 1024×512 conversion is its only cost).
    skip_vram_rgba: bool,
    /// Skip enabling the pixel tracer. Captures cmd_log for free
    /// when on (just a Vec push per packet); off as a sanity check
    /// that the tracer itself isn't the cost.
    skip_tracer: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        bios: PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"),
        disc: PathBuf::from(std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
            "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue".into()
        })),
        cycles_per_frame: 558_000,
        frames: 600,
        warmup: 50_000_000,
        skip_vram_rgba: false,
        skip_tracer: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bios" => a.bios = PathBuf::from(it.next().unwrap()),
            "--disc" => a.disc = PathBuf::from(it.next().unwrap()),
            "--cycles-per-frame" => a.cycles_per_frame = it.next().unwrap().parse().unwrap(),
            "--frames" => a.frames = it.next().unwrap().parse().unwrap(),
            "--warmup" => a.warmup = it.next().unwrap().parse().unwrap(),
            "--skip-vram-rgba" => a.skip_vram_rgba = true,
            "--skip-tracer" => a.skip_tracer = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    a
}

fn load_disc(p: &std::path::Path) -> Disc {
    if p.extension().and_then(|e| e.to_str()) == Some("cue") {
        psoxide_settings::library::load_disc_from_cue(p).expect("load cue")
    } else {
        Disc::from_bin(std::fs::read(p).expect("read bin"))
    }
}

fn main() {
    let args = parse_args();
    eprintln!("[bench] bios={}", args.bios.display());
    eprintln!("[bench] disc={}", args.disc.display());
    eprintln!(
        "[bench] cycles/frame={} frames={} warmup_cycles={} skip_vram_rgba={} skip_tracer={}",
        args.cycles_per_frame, args.frames, args.warmup, args.skip_vram_rgba, args.skip_tracer,
    );

    let bios_bytes = std::fs::read(&args.bios).expect("bios readable");
    let mut bus = Bus::new(bios_bytes).expect("bus");
    let mut cpu = Cpu::new();
    let disc = load_disc(&args.disc);
    bus.cdrom.insert_disc(Some(disc));
    if !args.skip_tracer {
        bus.gpu.enable_pixel_tracer();
    }

    eprintln!("[bench] warming up {} cycles...", args.warmup);
    for _ in 0..args.warmup {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }
    let _ = std::mem::take(&mut bus.gpu.cmd_log);
    if let Some(owner) = bus.gpu.pixel_owner.as_mut() {
        owner.fill(u32::MAX);
    }

    eprintln!("[bench] measuring {} frames...", args.frames);
    let mut t_emu = Duration::ZERO;
    let mut t_vram_rgba = Duration::ZERO;
    let mut t_display_rgba = Duration::ZERO;
    let mut total_packets = 0u64;
    let mut total_display_pixels = 0u64;
    let started = Instant::now();

    for _ in 0..args.frames {
        // Phase 1: CPU emulation. Includes the CPU rasterizer that
        // produces VRAM pixels — the dominant cost when GP0 is hot.
        let t = Instant::now();
        for _ in 0..args.cycles_per_frame {
            if cpu.step(&mut bus).is_err() {
                break;
            }
        }
        t_emu += t.elapsed();

        // Count this frame's draws so we can attribute back. The
        // tracer's owner buffer needs resetting too — its
        // `current_cmd_index` would otherwise drift past u32::MAX.
        let log = std::mem::take(&mut bus.gpu.cmd_log);
        total_packets += log.len() as u64;
        if let Some(owner) = bus.gpu.pixel_owner.as_mut() {
            owner.fill(u32::MAX);
        }

        // Phase 2: VRAM → RGBA8 (1024×512 = 524288 px). This is
        // what `Graphics::prepare_vram` does pre-upload.
        if !args.skip_vram_rgba {
            let t = Instant::now();
            let vram_rgba =
                bus.gpu
                    .vram
                    .to_rgba8(0, 0, VRAM_WIDTH as u16, VRAM_HEIGHT as u16);
            t_vram_rgba += t.elapsed();
            std::hint::black_box(&vram_rgba);
        }

        // Phase 3: display area → RGBA8. Smaller (typically
        // 320×240 to 640×480) but pays a 24bpp-mode branch and a
        // BGR→RGB swap per pixel.
        let t = Instant::now();
        let (display_rgba, dw, dh) = bus.gpu.display_rgba8();
        t_display_rgba += t.elapsed();
        total_display_pixels += dw as u64 * dh as u64;
        std::hint::black_box(&display_rgba);
    }

    let total = started.elapsed();
    let f = args.frames as f64;
    let pf = |d: Duration| d.as_secs_f64() * 1000.0 / f;
    let pct = |d: Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64();

    eprintln!();
    eprintln!("=== CPU-path frontend-loop timings ===");
    eprintln!(
        "  total wall-clock: {:.1} ms ({:.1} fps if every frame is one redraw)",
        total.as_secs_f64() * 1000.0,
        f / total.as_secs_f64(),
    );
    eprintln!(
        "  emulation:        {:.2} ms/frame ({:.1}%) [{:.1} pkts/frame avg, includes CPU rasterizer]",
        pf(t_emu),
        pct(t_emu),
        total_packets as f64 / f,
    );
    eprintln!(
        "  vram→rgba8:       {:.2} ms/frame ({:.1}%)  [524288 pixels each frame]",
        pf(t_vram_rgba),
        pct(t_vram_rgba)
    );
    eprintln!(
        "  display→rgba8:    {:.2} ms/frame ({:.1}%)  [{:.0} pixels avg]",
        pf(t_display_rgba),
        pct(t_display_rgba),
        total_display_pixels as f64 / f,
    );

    let total_per_frame_ms = total.as_secs_f64() * 1000.0 / f;
    let budget_60fps = 1000.0 / 60.0;
    let budget_30fps = 1000.0 / 30.0;
    eprintln!();
    if total_per_frame_ms <= budget_60fps {
        eprintln!(
            "  HEADROOM @ 60fps: {:.2} ms left",
            budget_60fps - total_per_frame_ms
        );
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
