//! Measure whether the SPU is producing non-silent audio when
//! running a game. If peak amplitude stays at 0 across the whole
//! run, the SPU isn't being driven (voices never keyed on) — or
//! we have a mixer bug. If peak is high but you still hear nothing,
//! the problem is downstream (cpal ring / host device).

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = std::fs::read(
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin",
    )
    .expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    let mut cpu = Cpu::new();

    let mut cycles_at_last_pump = 0u64;
    let mut window_peak_l: i32 = 0;
    let mut window_peak_r: i32 = 0;
    let mut window_samples: u64 = 0;
    let mut last_report_samples = 0u64;

    for _ in 0..1_500_000_000u64 {
        if cpu.step(&mut bus).is_err() {
            break;
        }
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let samples = bus.spu.drain_audio();
            for (l, r) in &samples {
                window_peak_l = window_peak_l.max(l.unsigned_abs() as i32);
                window_peak_r = window_peak_r.max(r.unsigned_abs() as i32);
                window_samples += 1;
            }
            // Report every ~1 second of simulated audio.
            if window_samples - last_report_samples > 44_100 {
                eprintln!(
                    "[audio] spu_samples={}  window_peak=(L={:5}, R={:5})  cpu_cyc={:>10}",
                    bus.spu.samples_produced(),
                    window_peak_l,
                    window_peak_r,
                    bus.cycles(),
                );
                last_report_samples = window_samples;
                window_peak_l = 0;
                window_peak_r = 0;
            }
        }
    }

    // Inspect some SPU state at the end.
    eprintln!();
    eprintln!("=== final SPU state ===");
    eprintln!("samples produced = {}", bus.spu.samples_produced());
    // Pull out main/voice volumes via MMIO reads.
    let main_l = bus.spu.read16(0x1F80_1D80);
    let main_r = bus.spu.read16(0x1F80_1D82);
    let spucnt = bus.spu.spucnt();
    eprintln!("SPUCNT  = 0x{spucnt:04x}  (bit 15 = enable)");
    eprintln!("MAIN_VOL_L = 0x{main_l:04x}");
    eprintln!("MAIN_VOL_R = 0x{main_r:04x}");
    // Scan voice bank for any voice with non-zero ADSR_CURRENT (which
    // we pin to 1 for Redux parity — so non-1 means the game has
    // written there, or a voice is active in our model).
    let kon_raw_lo = bus.spu.read16(0x1F80_1D88);
    let kon_raw_hi = bus.spu.read16(0x1F80_1D8A);
    let koff_raw_lo = bus.spu.read16(0x1F80_1D8C);
    let koff_raw_hi = bus.spu.read16(0x1F80_1D8E);
    eprintln!(
        "KON  = 0x{:08x}",
        ((kon_raw_hi as u32) << 16) | kon_raw_lo as u32
    );
    eprintln!(
        "KOFF = 0x{:08x}",
        ((koff_raw_hi as u32) << 16) | koff_raw_lo as u32
    );
}
