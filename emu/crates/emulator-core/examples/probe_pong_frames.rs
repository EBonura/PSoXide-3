//! Side-load pong and dump several frames, letting the ball bounce
//! around under the AI-vs-idle-player default. Used to eyeball that
//! the ball physics + AI tracking + SFX triggering all work.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_pong_frames --release
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Exe;
use std::io::Write;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("bios");
    let exe_bytes = std::fs::read(
        "/Users/ebonura/Desktop/repos/PSoXide/build/examples/mipsel-sony-psx/release/pong.exe",
    )
    .expect("pong.exe");
    let exe = Exe::parse(&exe_bytes).expect("parse");

    let mut bus = Bus::new(bios).expect("bus");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.enable_hle_bios();
    bus.attach_digital_pad_port1();
    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    // Sample across a rally: early, mid-rally, bounce region.
    let sample_targets: &[u64] = &[2, 30, 60, 120, 240];

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
    let path = format!("/tmp/pong-f{:03}.ppm", frame);
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
