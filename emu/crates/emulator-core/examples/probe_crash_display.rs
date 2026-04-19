//! Dump GPU display-mode state for Crash Bandicoot at several
//! instruction counts, so we can see what resolution / display-area
//! Crash programs and when it changes modes. The user reported the
//! game renders "stretched horizontally" — the first question is
//! whether that's our fault (mis-decoding GP1 0x08) or just an
//! unusual-but-correct mode that our framebuffer painter handles
//! without pixel-aspect correction.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_crash_display --release
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

    // Checkpoints aligned with the parity suite so we can correlate.
    let checkpoints = [
        100_000_000u64,
        200_000_000,
        300_000_000,
        400_000_000,
        500_000_000,
        600_000_000,
        700_000_000,
    ];

    let mut cycles_at_last_pump = 0u64;
    let mut cursor = 0u64;
    for &cp in &checkpoints {
        while cursor < cp {
            if cpu.step(&mut bus).is_err() {
                eprintln!("[crash_display] CPU errored at step {cursor}");
                break;
            }
            cursor += 1;
            if bus.cycles() - cycles_at_last_pump > 560_000 {
                cycles_at_last_pump = bus.cycles();
                bus.run_spu_samples(735);
                let _ = bus.spu.drain_audio();
            }
        }
        let da = bus.gpu.display_area();
        println!(
            "step={cp:>10}  display=({x}, {y})  size={w}×{h}  bpp={bpp}  pc=0x{pc:08x}",
            x = da.x,
            y = da.y,
            w = da.width,
            h = da.height,
            bpp = if da.bpp24 { 24 } else { 15 },
            pc = cpu.pc(),
        );
    }
}
