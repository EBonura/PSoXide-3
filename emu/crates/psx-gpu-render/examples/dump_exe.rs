//! Headless harness: load a PSX-EXE, step the emulator, render the
//! cumulative `cmd_log` through the HW pipeline, dump the result to
//! a PPM. Lets the renderer be inspected without launching the GUI
//! (useful when screen-capture permissions block the agent and when
//! the editor crates aren't building).
//!
//! Usage:
//!   cargo run --release -p psx-gpu-render --example dump_exe -- \
//!       <BIOS> <EXE> <OUT.ppm> [STEPS]

use std::path::PathBuf;

use emulator_core::{Bus, Cpu};
use psx_gpu_render::HwRenderer;
use psx_iso::Exe;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <BIOS> <EXE> <OUT.ppm> [STEPS]",
            args.first().map(String::as_str).unwrap_or("dump_exe")
        );
        std::process::exit(2);
    }
    let bios = PathBuf::from(&args[1]);
    let exe_path = PathBuf::from(&args[2]);
    let out = PathBuf::from(&args[3]);
    let steps: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);

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

    eprintln!(
        "[dump_exe] BIOS={} EXE={} entry=0x{:08x} payload={}B steps={}",
        bios.display(),
        exe_path.display(),
        exe.initial_pc,
        exe.payload.len(),
        steps
    );

    for i in 0..steps {
        if let Err(e) = cpu.step(&mut bus) {
            eprintln!("[dump_exe] step {i} failed: {e:?}");
            break;
        }
    }

    eprintln!(
        "[dump_exe] tick={} cycles={} pc=0x{:08x} cmd_log_entries={}",
        cpu.tick(),
        bus.cycles(),
        cpu.pc(),
        bus.gpu.cmd_log.len(),
    );

    // Histogram of GP0 opcodes to verify what the demo emits.
    let mut hist: std::collections::BTreeMap<u8, u32> = std::collections::BTreeMap::new();
    for entry in &bus.gpu.cmd_log {
        *hist.entry(entry.opcode).or_insert(0) += 1;
    }
    eprintln!("[dump_exe] opcode histogram:");
    for (op, n) in &hist {
        eprintln!("  0x{op:02X}: {n}");
    }

    let display = bus.gpu.display_area();
    eprintln!(
        "[dump_exe] display_area: x={} y={} w={} h={} bpp24={}",
        display.x, display.y, display.width, display.height, display.bpp24
    );

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
            label: Some("dump-exe-device"),
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
    // Display sub-rect of the VRAM-shaped target (= what the
    // central panel paints in the live frontend).
    let s = hw.internal_scale();
    let (w, h, rgba) = hw.read_subrect_rgba8(
        display.x as u32 * s,
        display.y as u32 * s,
        display.width as u32 * s,
        display.height as u32 * s,
    );
    eprintln!("[dump_exe] rendered display sub-rect {w}x{h}, {} bytes", rgba.len());

    use std::io::Write;
    let mut f = std::fs::File::create(&out).expect("create output");
    writeln!(f, "P6\n{w} {h}\n255").unwrap();
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for chunk in rgba.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    f.write_all(&rgb).unwrap();
    eprintln!("[dump_exe] wrote {}", out.display());
}
