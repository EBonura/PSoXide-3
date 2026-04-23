//! Side-load hello-input and dump a frame to PPM so we can SEE the
//! new font-rendered label row. Hash-based milestone goldens don't
//! catch "text rendering is visibly broken" — only eyeballing a
//! rendered frame does.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_hello_input_frames --release
//! ```
//!
//! Writes `/tmp/hello-input-fXXX.ppm` for a handful of sampled
//! frames. No buttons held in this probe (the pad controller defaults
//! to all-up), so what should render is:
//!   - Blue background from the baseline `b=32` starting color
//!   - "HELD:" header at (4, 4) in light grey
//!   - Tiny white triangle in the centre
//!   - Bottom-right hex = `0xFFFF` (no buttons held = raw mask stays
//!     at 0xFFFF per PSX's active-low pad protocol). If the hex is
//!     `0x0000` instead the `ButtonState` inversion is off.

use emulator_core::{Bus, Cpu};
use psx_iso::Exe;
use std::io::Write;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("bios");
    let exe_bytes = std::fs::read(
        "/Users/ebonura/Desktop/repos/psoxide/build/examples/mipsel-sony-psx/release/hello-input.exe",
    )
    .expect("hello-input.exe");
    let exe = Exe::parse(&exe_bytes).expect("parse");

    let mut bus = Bus::new(bios).expect("bus");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.enable_hle_bios();
    bus.attach_digital_pad_port1();
    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    let sample_targets: &[u64] = &[2, 4, 8, 16];

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
    let path = format!("/tmp/hello-input-f{:03}.ppm", frame);
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
