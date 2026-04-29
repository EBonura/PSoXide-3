//! Visual parity harness — compare CPU rasterizer vs HW renderer
//! pixel-for-pixel on the active display rect after running an EXE
//! for N CPU steps.
//!
//! Designed as the regression-test substrate the HW renderer phases
//! were missing: every architectural change re-runs this on the seven
//! `engine/examples/showcase-*` builds and prints mismatch %, so a
//! shader bug stops being "did this look right in one screenshot"
//! and starts being "did parity drop on fixture N".
//!
//! Usage:
//!   parity <BIOS> <EXE> <STEPS> [OUT_DIR] [TOLERANCE]
//!
//! Outputs in OUT_DIR (default: /tmp/psx-parity):
//!   cpu.ppm     — CPU rasterizer's display rect, RGBA8 → P6 PPM
//!   hw.ppm      — HW renderer's output, same shape, same encoding
//!   diff.ppm    — greyscale: black = pixel matches within TOLERANCE,
//!                 white = mismatch (intensity = max channel delta)
//!   report.txt  — pixel counts, mismatch %, opcode histogram
//!
//! Exit code: 0 if mismatch <= 1.0% of pixels, else 1.
//!
//! Tolerance defaults to 8 (per channel LSBs). Picked to absorb the
//! sRGB-vs-5-bit-replication encoding gap between the two backends
//! so structural mismatches (wrong geometry, missing texture, wrong
//! tint) dominate the signal. Tighten once the HW pipeline matches
//! the CPU's BGR15 quantisation exactly.

use std::path::PathBuf;

use emulator_core::{Bus, Cpu};
use psx_gpu_render::HwRenderer;
use psx_iso::Exe;

const DEFAULT_TOLERANCE: u8 = 8;
const PASS_THRESHOLD_PCT: f64 = 1.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <BIOS> <EXE> <STEPS> [OUT_DIR] [TOLERANCE]",
            args.first().map(String::as_str).unwrap_or("parity")
        );
        std::process::exit(2);
    }
    let bios = PathBuf::from(&args[1]);
    let exe_path = PathBuf::from(&args[2]);
    let steps: u64 = args[3].parse().expect("STEPS must be u64");
    let out_dir = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/psx-parity"));
    let tolerance: u8 = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TOLERANCE);

    std::fs::create_dir_all(&out_dir).expect("create out dir");

    // ---- Boot + run ---------------------------------------------------------
    let bios_bytes = std::fs::read(&bios).expect("read BIOS");
    let mut bus = Bus::new(bios_bytes).expect("BIOS rejected");
    bus.gpu.enable_cmd_log();
    bus.enable_hle_bios();
    bus.attach_digital_pad_port1();

    let exe_bytes = std::fs::read(&exe_path).expect("read EXE");
    let exe = Exe::parse(&exe_bytes).expect("parse EXE");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.clear_exe_bss(exe.bss_addr, exe.bss_size);

    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    for _ in 0..steps {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }

    // ---- Capture CPU reference ---------------------------------------------
    let display = bus.gpu.display_area();
    let (cpu_rgba, cpu_w, cpu_h) = bus.gpu.display_rgba8();

    // ---- Capture HW output --------------------------------------------------
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no compatible wgpu adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("parity-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("request device");

    let mut hw = HwRenderer::new_headless(device, queue);
    let vram_words: Vec<u16> = bus.gpu.vram.words().to_vec();
    hw.render_frame(&bus.gpu, &bus.gpu.cmd_log, &vram_words);
    // Extract just the display sub-rect from the VRAM-shaped target
    // — that's what the user sees on screen, and what the CPU
    // `display_rgba8()` returns. At Native scale (S=1) the sub-rect
    // pixel dims equal the display rect dims, so they directly
    // pixel-compare.
    let s = hw.internal_scale();
    let (hw_w, hw_h, hw_rgba) = hw.read_subrect_rgba8(
        display.x as u32 * s,
        display.y as u32 * s,
        display.width as u32 * s,
        display.height as u32 * s,
    );

    // ---- Validate shape ----------------------------------------------------
    if (cpu_w, cpu_h) != (hw_w, hw_h) {
        eprintln!(
            "[parity] SHAPE MISMATCH: cpu={cpu_w}x{cpu_h} hw={hw_w}x{hw_h} — \
             can't pixel-compare. Likely the HW renderer is sized to a \
             different rect than the CPU display."
        );
        std::process::exit(2);
    }

    // ---- Compare -----------------------------------------------------------
    let total_px = (cpu_w * cpu_h) as usize;
    let mut diff = Vec::with_capacity(cpu_rgba.len());
    let mut mismatched = 0usize;
    let mut total_delta_r: u64 = 0;
    let mut total_delta_g: u64 = 0;
    let mut total_delta_b: u64 = 0;
    let mut max_delta: u8 = 0;

    for i in 0..total_px {
        let off = i * 4;
        let dr = (cpu_rgba[off] as i16 - hw_rgba[off] as i16).unsigned_abs() as u8;
        let dg = (cpu_rgba[off + 1] as i16 - hw_rgba[off + 1] as i16).unsigned_abs() as u8;
        let db = (cpu_rgba[off + 2] as i16 - hw_rgba[off + 2] as i16).unsigned_abs() as u8;
        let max_ch = dr.max(dg).max(db);
        if max_ch > tolerance {
            mismatched += 1;
        }
        if max_ch > max_delta {
            max_delta = max_ch;
        }
        total_delta_r += dr as u64;
        total_delta_g += dg as u64;
        total_delta_b += db as u64;
        // Diff image: amplify so any non-zero delta is visible.
        let amp = (max_ch as u16 * 4).min(255) as u8;
        diff.extend_from_slice(&[amp, amp, amp, 0xFF]);
    }

    let mismatch_pct = (mismatched as f64) / (total_px as f64) * 100.0;
    let mean_dr = total_delta_r as f64 / total_px as f64;
    let mean_dg = total_delta_g as f64 / total_px as f64;
    let mean_db = total_delta_b as f64 / total_px as f64;

    // ---- Opcode histogram (helps explain mismatch causes) ------------------
    let mut hist: std::collections::BTreeMap<u8, u32> = std::collections::BTreeMap::new();
    for entry in &bus.gpu.cmd_log {
        *hist.entry(entry.opcode).or_insert(0) += 1;
    }

    // ---- Write outputs ------------------------------------------------------
    write_ppm(&out_dir.join("cpu.ppm"), &cpu_rgba, cpu_w, cpu_h);
    write_ppm(&out_dir.join("hw.ppm"), &hw_rgba, hw_w, hw_h);
    write_ppm(&out_dir.join("diff.ppm"), &diff, cpu_w, cpu_h);

    let mut report = String::new();
    report.push_str(&format!("fixture: {}\n", exe_path.display()));
    report.push_str(&format!("steps:   {}\n", steps));
    report.push_str(&format!(
        "display: x={} y={} w={} h={} bpp24={}\n",
        display.x, display.y, display.width, display.height, display.bpp24
    ));
    report.push_str(&format!("size:    {cpu_w}x{cpu_h} ({total_px} px)\n"));
    report.push_str(&format!("tolerance: {tolerance} LSB/channel\n"));
    report.push_str(&format!(
        "mismatched: {mismatched}/{total_px} ({mismatch_pct:.3}%)\n"
    ));
    report.push_str(&format!(
        "mean delta: r={mean_dr:.2} g={mean_dg:.2} b={mean_db:.2}\n"
    ));
    report.push_str(&format!("max delta: {max_delta}\n"));
    report.push_str(&format!("cmd_log entries: {}\n", bus.gpu.cmd_log.len()));
    report.push_str("opcode histogram:\n");
    for (op, n) in &hist {
        report.push_str(&format!("  0x{op:02X}: {n}\n"));
    }
    std::fs::write(out_dir.join("report.txt"), &report).expect("write report");

    // ---- Stdout summary -----------------------------------------------------
    let verdict = if mismatch_pct <= PASS_THRESHOLD_PCT {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "{verdict} {:>6.2}% mismatch  max_delta={max_delta:>3}  mean=(r={:.1} g={:.1} b={:.1})  {}",
        mismatch_pct,
        mean_dr,
        mean_dg,
        mean_db,
        exe_path.file_name().and_then(|s| s.to_str()).unwrap_or("?")
    );

    if mismatch_pct > PASS_THRESHOLD_PCT {
        std::process::exit(1);
    }
}

fn write_ppm(path: &std::path::Path, rgba: &[u8], w: u32, h: u32) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("create ppm");
    writeln!(f, "P6\n{w} {h}\n255").unwrap();
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for chunk in rgba.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    f.write_all(&rgb).unwrap();
}
