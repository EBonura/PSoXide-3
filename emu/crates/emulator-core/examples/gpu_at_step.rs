//! Dump GPU + DMA channel state at a target step. Used to reason
//! about the GPUSTAT-bit-25 divergence where Redux reports bit 25=0
//! while we compute bit 25=1.
//!
//! ```bash
//! cargo run -p emulator-core --example gpu_at_step --release -- 19474030
//! ```

use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(19_474_030);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    for _ in 0..target {
        cpu.step(&mut bus).expect("step");
    }

    let gpustat = bus.read32(0x1F80_1814);
    println!("=== State at step {target} ===");
    println!("cycles    : {}", bus.cycles());
    println!("PC        : 0x{:08x}", cpu.pc());
    println!("GPUSTAT   : 0x{gpustat:08x}");
    println!();
    println!("DMA channels:");
    for ch in 0..7 {
        let base = 0x1F80_1080u32 + (ch as u32) * 0x10;
        let madr = bus.read32(base);
        let bcr = bus.read32(base + 4);
        let chcr = bus.read32(base + 8);
        let chcr_busy = (chcr >> 24) & 1;
        let chcr_start = (chcr >> 28) & 1;
        println!(
            "  ch{ch}: MADR=0x{madr:08x}  BCR=0x{bcr:08x}  \
             CHCR=0x{chcr:08x}  start={chcr_start}  enable={chcr_busy}",
        );
    }
    println!();
    println!("DPCR      : 0x{:08x}", bus.read32(0x1F80_10F0));
    println!("DICR      : 0x{:08x}", bus.read32(0x1F80_10F4));
}
