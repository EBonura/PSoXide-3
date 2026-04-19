//! Capture our Crash framebuffer at a schedule of instruction counts,
//! emit a PPM + a basic diff-from-black summary. This is a
//! *unilateral* probe — no Redux needed — so we can iterate on
//! render bugs fast. Use the parity harness once we know where to
//! look.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_crash_frames --release
//! # → /tmp/crash_frame_{N}.ppm for each checkpoint
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::fs;
use std::io::Write;

fn main() {
    let bios = fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = fs::read(
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin",
    )
    .expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    let mut cpu = Cpu::new();

    // The "tent + Crash portrait + black bar" frame the user showed
    // is part of the intro-logo sequence (Naughty Dog / Universal /
    // SCE). It lives somewhere between the SCEA-licensed splash and
    // the CRASH BANDICOOT title screen. Scan broadly.
    // Finer scan covering the whole Crash intro window so we can
    // locate the broken-black-bar frame the user screenshotted.
    // Find when the "NAUGHTY DOG" logo-text appears between step
    // 600M (bar with no text) and step 700M (past the animation).
    let checkpoints = [
        610_000_000u64,
        620_000_000,
        630_000_000,
        640_000_000,
        650_000_000,
        660_000_000,
        670_000_000,
        680_000_000,
        690_000_000,
    ];
    let mut cycles_at_last_pump = 0u64;
    let mut cursor = 0u64;

    for &cp in &checkpoints {
        let start = std::time::Instant::now();
        while cursor < cp {
            if cpu.step(&mut bus).is_err() {
                eprintln!("[crash_frames] CPU errored at step {cursor}");
                return;
            }
            cursor += 1;
            if bus.cycles() - cycles_at_last_pump > 560_000 {
                cycles_at_last_pump = bus.cycles();
                bus.run_spu_samples(735);
                let _ = bus.spu.drain_audio();
            }
        }
        let elapsed = start.elapsed();
        let da = bus.gpu.display_area();
        let w = da.width as usize;
        let h = da.height as usize;

        // Write PPM from the display area.
        let path = format!("/tmp/crash_frame_{cp}.ppm");
        let mut f = fs::File::create(&path).expect("create ppm");
        writeln!(f, "P6\n{w} {h}\n255").unwrap();
        let mut buf = Vec::with_capacity(w * h * 3);
        let mut nonzero = 0usize;
        for dy in 0..h as u16 {
            for dx in 0..w as u16 {
                let pix = bus.gpu.vram.get_pixel(da.x + dx, da.y + dy);
                if pix != 0 {
                    nonzero += 1;
                }
                let r5 = (pix & 0x1F) as u8;
                let g5 = ((pix >> 5) & 0x1F) as u8;
                let b5 = ((pix >> 10) & 0x1F) as u8;
                buf.push((r5 << 3) | (r5 >> 2));
                buf.push((g5 << 3) | (g5 >> 2));
                buf.push((b5 << 3) | (b5 >> 2));
            }
        }
        f.write_all(&buf).unwrap();

        println!(
            "step={cp:>10}  elapsed={:>5.1}s  display=({},{}) {}×{}  non-black={}/{} ({:.1}%)  pc=0x{:08x}  → {}",
            elapsed.as_secs_f32(),
            da.x, da.y, w, h,
            nonzero, w * h,
            100.0 * nonzero as f32 / (w * h) as f32,
            cpu.pc(),
            path,
        );
    }
}
