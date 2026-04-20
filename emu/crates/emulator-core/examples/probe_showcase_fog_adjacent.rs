//! Dump two consecutive showcase-fog frames across the ring-wrap
//! boundary so we can see exactly what changes frame-to-frame.
//!
//! Ring wrap period = RING_SPACING / SCROLL_SPEED = 0x500 / 0x20 =
//! 40 frames. So wraps happen at frames 40, 80, 120, ... We dump
//! frames around frame 40 (both sides) plus two deep mid-run frames
//! to confirm non-wrap frames are continuous.
use emulator_core::{Bus, Cpu};
use psx_iso::Exe;
use std::io::Write;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("bios");
    let exe_bytes = std::fs::read(
        "/Users/ebonura/Desktop/repos/PSoXide/build/examples/mipsel-sony-psx/release/showcase-fog.exe",
    )
    .expect("showcase-fog.exe");
    let exe = Exe::parse(&exe_bytes).expect("parse");

    let mut bus = Bus::new(bios).expect("bus");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.enable_hle_bios();
    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    // Around the first wrap (frame 40) + a non-wrap neighbourhood.
    let sample_targets: &[u64] = &[30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42];
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
    }
}

fn dump_ppm(bus: &Bus, frame: u64) {
    let da = bus.gpu.display_area();
    let path = format!("/tmp/showcase-fog-adj-f{:03}.ppm", frame);
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
    eprintln!("wrote {path}");
}
