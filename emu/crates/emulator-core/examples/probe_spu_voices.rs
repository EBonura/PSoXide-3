//! Snapshot SPU voice state at periodic checkpoints during Crash
//! boot. "Peak=841 over 10 seconds" means voices are either not
//! keyed on, at near-zero envelope, or at near-zero per-voice
//! volume. This probe dumps enough state to tell which.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_spu_voices --release
//! ```

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

    let checkpoints = [
        30_000_000u64,
        60_000_000,
        100_000_000,
        150_000_000,
        200_000_000,
        250_000_000,
    ];
    let mut cycles_at_last_pump = 0u64;
    let mut cursor = 0u64;
    for &cp in &checkpoints {
        while cursor < cp {
            if cpu.step(&mut bus).is_err() {
                break;
            }
            cursor += 1;
            if bus.cycles() - cycles_at_last_pump > 560_000 {
                cycles_at_last_pump = bus.cycles();
                bus.run_spu_samples(735);
                let _ = bus.spu.drain_audio();
            }
        }
        println!("=== step {cp} (cpu_cyc={}) ===", bus.cycles());
        println!("SPUCNT        = 0x{:04x}", bus.spu.spucnt());
        println!("MAIN_VOL_L/R  = 0x{:04x} / 0x{:04x}",
            bus.spu.read16(0x1F80_1D80), bus.spu.read16(0x1F80_1D82));
        println!("CD_VOL_L/R    = 0x{:04x} / 0x{:04x}",
            bus.spu.read16(0x1F80_1DB0), bus.spu.read16(0x1F80_1DB2));
        // Per-voice snapshot. Each voice occupies 16 bytes starting
        // at 0x1F80_1C00. Offsets: +0 vol_l, +2 vol_r, +4 pitch, +6
        // start_addr, +8 adsr_low, +10 adsr_high, +12 current_envelope,
        // +14 repeat_addr.
        println!();
        println!(
            " v  vol_l  vol_r  pitch  start adsr_l adsr_h  env  repeat"
        );
        for v in 0..24 {
            let base = 0x1F80_1C00 + v * 16;
            let vol_l = bus.spu.read16(base);
            let vol_r = bus.spu.read16(base + 2);
            let pitch = bus.spu.read16(base + 4);
            let start = bus.spu.read16(base + 6);
            let adsr_l = bus.spu.read16(base + 8);
            let adsr_h = bus.spu.read16(base + 10);
            let env = bus.spu.read16(base + 12);
            let repeat = bus.spu.read16(base + 14);
            // Only print voices with non-trivial state to keep output manageable.
            if vol_l != 0 || vol_r != 0 || pitch != 0 || env != 0 || start != 0 {
                println!(
                    "{v:2}  {vol_l:04x}   {vol_r:04x}   {pitch:04x}   {start:04x}  {adsr_l:04x}   {adsr_h:04x}   {env:04x}  {repeat:04x}"
                );
            }
        }
        println!();
    }
}
