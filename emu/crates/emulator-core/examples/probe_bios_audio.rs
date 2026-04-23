//! Audit SPU output during BIOS boot (no disc). The BIOS plays the
//! Sony-logo chime *only when a licensed disc boots successfully*,
//! so a pure no-disc boot producing silence is EXPECTED. The user
//! reports "no sound during BIOS intro" — we want to know whether
//! the chime is genuinely missing in the disc-boot path or if it's
//! producing samples we're failing to surface.
//!
//! Run twice (once per mode) — compare peak amplitudes:
//!
//! ```bash
//! cargo run -p emulator-core --example probe_bios_audio --release
//! PSOXIDE_DISC=/path/to/crash.bin cargo run -p emulator-core --example probe_bios_audio --release
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let disc_tag = if let Ok(disc_path) = std::env::var("PSOXIDE_DISC") {
        let bytes = std::fs::read(&disc_path).expect("disc");
        bus.cdrom.insert_disc(Some(Disc::from_bin(bytes)));
        format!("disc={disc_path}")
    } else {
        "no-disc".to_string()
    };
    let mut cpu = Cpu::new();

    println!("=== BIOS audio probe ({disc_tag}) ===");
    let mut cycles_at_last_pump = 0u64;
    let mut window_peak_l: i32 = 0;
    let mut window_peak_r: i32 = 0;
    let mut last_report_samples = 0u64;
    let mut total_samples = 0u64;
    let mut peak_ever_l: i32 = 0;
    let mut peak_ever_r: i32 = 0;

    // Run 500M instructions — well past the Sony logo (100M) and
    // into the shell (500M) or disc boot (500M+).
    for _ in 0..500_000_000u64 {
        if cpu.step(&mut bus).is_err() {
            break;
        }
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let samples = bus.spu.drain_audio();
            for (l, r) in &samples {
                let al = l.unsigned_abs() as i32;
                let ar = r.unsigned_abs() as i32;
                window_peak_l = window_peak_l.max(al);
                window_peak_r = window_peak_r.max(ar);
                peak_ever_l = peak_ever_l.max(al);
                peak_ever_r = peak_ever_r.max(ar);
                total_samples += 1;
            }
            if total_samples - last_report_samples > 44_100 {
                eprintln!(
                    "t≈{:.1}s  window_peak=(L={:5}, R={:5})  cumulative_peak=(L={:5}, R={:5})  cpu_cyc={:>10}",
                    total_samples as f32 / 44_100.0,
                    window_peak_l, window_peak_r,
                    peak_ever_l, peak_ever_r,
                    bus.cycles(),
                );
                last_report_samples = total_samples;
                window_peak_l = 0;
                window_peak_r = 0;
            }
        }
    }

    // Summary.
    let spucnt = bus.spu.spucnt();
    let main_l = bus.spu.read16(0x1F80_1D80);
    let main_r = bus.spu.read16(0x1F80_1D82);
    println!();
    println!("=== summary ===");
    println!("total samples  = {total_samples}");
    println!("peak_ever      = (L={peak_ever_l}, R={peak_ever_r})");
    println!("SPUCNT         = 0x{spucnt:04x}  (bit 15 = enable)");
    println!("MAIN_VOL_L     = 0x{main_l:04x}");
    println!("MAIN_VOL_R     = 0x{main_r:04x}");
    let kon_lo = bus.spu.read16(0x1F80_1D88);
    let kon_hi = bus.spu.read16(0x1F80_1D8A);
    println!(
        "KON            = 0x{:08x}",
        ((kon_hi as u32) << 16) | kon_lo as u32
    );
}
