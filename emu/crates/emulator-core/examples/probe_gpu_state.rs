//! Dump our GPU display-area + status state at step N for a game,
//! to compare against Redux's in-game expectations.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_gpu_state -- 500000000 "/path/to/game.bin"
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000_000);
    let disc_path = std::env::args().nth(2);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();
    for _ in 0..n {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    let (h, w, hh, len) = bus.gpu.display_hash();
    let da = bus.gpu.display_area();
    println!("=== GPU state at step {n} (cycles={}) ===", bus.cycles());
    println!("display_hash : 0x{h:016x}  ({w}×{hh}, {len} bytes)");
    println!("display_area(): x={} y={} width={} height={} 24bpp={}",
        da.x, da.y, da.width, da.height, da.bpp24);
}
