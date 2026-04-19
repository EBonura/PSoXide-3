//! Side-load hello-gte and dump N frames to PPM so we can SEE what
//! the cube actually looks like on screen. Hash-based milestone
//! tests don't catch "visible garbage that happens to be
//! deterministic" — only a human looking at a rendered frame does.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_hello_gte_frames --release
//! ```
//!
//! Writes `/tmp/hello-gte-fXXX.ppm` for a handful of sampled frames.

use emulator_core::{Bus, Cpu};
use psx_iso::Exe;
use std::io::Write;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("bios");
    let exe_bytes = std::fs::read(
        "/Users/ebonura/Desktop/repos/PSoXide/build/examples/mipsel-sony-psx/release/hello-gte.exe",
    )
    .expect("hello-gte.exe");
    let exe = Exe::parse(&exe_bytes).expect("parse");

    let mut bus = Bus::new(bios).expect("bus");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.enable_hle_bios();
    bus.attach_digital_pad_port1();
    bus.gpu.enable_pixel_tracer();
    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    // Sample frames: 2, 5, 20, 40, 80. First captures the startup
    // state; the later ones span one full cube rotation at the
    // new YAW_STEP of 4 per frame (256/4 = 64 frames/rev).
    let sample_targets: &[u64] = &[2, 5, 20, 40, 80];

    let mut cycles_at_last_pump = 0u64;
    for &target in sample_targets {
        while bus.irq().raise_counts()[0] < target {
            if cpu.step(&mut bus).is_err() {
                break;
            }
            if bus.cycles() - cycles_at_last_pump > 560_000 {
                cycles_at_last_pump = bus.cycles();
                bus.run_spu_samples(735);
                let _ = bus.spu.drain_audio();
            }
        }
        dump_ppm(&bus, target);
        if target == 2 {
            dump_gp0_lines(&bus, target);
        }
    }
}

/// Print every GP0 0x40 (mono line) packet from `cmd_log`. Lets us
/// see the actual screen-space coordinates the example hands the
/// GPU — if they're all at (0, 0) or off-screen, the GTE
/// projection is producing garbage.
fn dump_gp0_lines(bus: &Bus, frame: u64) {
    eprintln!();
    eprintln!("=== GP0 0x40 (mono line) packets at frame {frame} ===");
    let mut count = 0;
    for e in &bus.gpu.cmd_log {
        if e.opcode == 0x40 && e.fifo.len() >= 3 {
            let v0 = e.fifo[1];
            let v1 = e.fifo[2];
            let (x0, y0) = (v0 as i16, (v0 >> 16) as i16);
            let (x1, y1) = (v1 as i16, (v1 >> 16) as i16);
            eprintln!(
                "  line {count:>2}: ({x0:>5}, {y0:>5}) -> ({x1:>5}, {y1:>5})  color=0x{:06x}",
                e.fifo[0] & 0xFFFFFF,
            );
            count += 1;
        }
    }
    eprintln!("=== total mono lines = {count} ===");
    // Also count other opcodes to see if the example reached the
    // render path at all.
    let mut histogram = std::collections::BTreeMap::new();
    for e in &bus.gpu.cmd_log {
        *histogram.entry(e.opcode).or_insert(0u32) += 1;
    }
    eprintln!("=== GP0 opcode histogram (all packets) ===");
    for (op, n) in histogram {
        eprintln!("  0x{op:02x}: {n}");
    }
}

fn dump_ppm(bus: &Bus, frame: u64) {
    let da = bus.gpu.display_area();
    let path = format!("/tmp/hello-gte-f{:03}.ppm", frame);
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "P6\n{} {}\n255", da.width, da.height).unwrap();
    let mut buf = Vec::with_capacity((da.width as usize) * (da.height as usize) * 3);
    for dy in 0..da.height {
        for dx in 0..da.width {
            let pix = bus.gpu.vram.get_pixel(da.x + dx, da.y + dy);
            let r5 = (pix & 0x1F) as u8;
            let g5 = ((pix >> 5) & 0x1F) as u8;
            let b5 = ((pix >> 10) & 0x1F) as u8;
            buf.push((r5 << 3) | (r5 >> 2));
            buf.push((g5 << 3) | (g5 >> 2));
            buf.push((b5 << 3) | (b5 >> 2));
        }
    }
    f.write_all(&buf).unwrap();
    eprintln!("wrote {path}  ({}×{})", da.width, da.height);
}
