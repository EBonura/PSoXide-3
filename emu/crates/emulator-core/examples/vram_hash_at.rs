//! Emit an FNV-1a-64 hash of VRAM after running the BIOS for a given
//! instruction count. Used to capture golden hashes for milestone
//! regression tests.
//!
//! ```bash
//! cargo run -p emulator-core --example vram_hash_at --release -- 100000000
//! ```

use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: vram_hash_at <instruction_count>");

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    for _ in 0..n {
        cpu.step(&mut bus).expect("step");
    }

    // Hash each 16-bit word as two LE bytes.
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &w in bus.gpu.vram.words() {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01B3);
        }
    }
    let nz = bus.gpu.vram.words().iter().filter(|&&w| w != 0).count();
    println!("steps={n}");
    println!("cycles={}", bus.cycles());
    println!("vram_fnv1a_64=0x{h:016x}");
    println!("vram_nonzero_words={nz}");
}

