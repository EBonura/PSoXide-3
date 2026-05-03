//! Capture our Crash framebuffer at a schedule of instruction counts,
//! emit a PPM + a basic diff-from-black summary. This is a
//! *unilateral* probe -- no Redux needed -- so we can iterate on
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

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let bios = frame_probe::read_bios();
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
            if !frame_probe::step_cpu_and_pump_spu(&mut cpu, &mut bus, &mut cycles_at_last_pump) {
                eprintln!("[crash_frames] CPU errored at step {cursor}");
                return;
            }
            cursor += 1;
        }
        let elapsed = start.elapsed();
        let dump = frame_probe::dump_display_ppm(&bus, "crash-frame", cp).expect("dump frame");
        let nonzero = frame_probe::display_nonzero_pixels(&bus);
        let pixels = (dump.width as usize) * (dump.height as usize);

        println!(
            "step={cp:>10}  elapsed={:>5.1}s  display={}×{} hash=0x{:016x}  non-black={}/{} ({:.1}%)  pc=0x{:08x}  -> {}",
            elapsed.as_secs_f32(),
            dump.width,
            dump.height,
            dump.display_hash,
            nonzero,
            pixels,
            100.0 * nonzero as f32 / pixels as f32,
            cpu.pc(),
            dump.path.display(),
        );
    }
}
