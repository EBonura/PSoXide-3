//! Emit an FNV-1a-64 hash of VRAM after running the BIOS for a given
//! instruction count. Used to capture golden hashes for milestone
//! regression tests.
//!
//! ```bash
//! cargo run -p emulator-core --example vram_hash_at --release -- 100000000
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
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
    // Optional disc image. Setting PSOXIDE_DISC mounts a BIN image
    // before stepping so milestone-D-plus goldens capture the
    // licensed-disc boot path instead of the shell.
    if let Ok(disc_path) = std::env::var("PSOXIDE_DISC") {
        let disc_bytes = std::fs::read(&disc_path).expect("disc readable");
        bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();

    let mut stopped_at: Option<u64> = None;
    for i in 0..n {
        if cpu.step(&mut bus).is_err() {
            stopped_at = Some(i);
            break;
        }
    }
    if let Some(at) = stopped_at {
        eprintln!("[vram_hash_at] stopped at step {at}/{n}");
    }

    // Full 1 MiB VRAM hash (self-regression only — determinism check).
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &w in bus.gpu.vram.words() {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01B3);
        }
    }
    let nz = bus.gpu.vram.words().iter().filter(|&&w| w != 0).count();
    // Visible-display-area hash — this is what's comparable against
    // Redux's `PCSX.GPU.takeScreenShot()` via the oracle's
    // `vram_hash` command, and therefore the one that verifies we
    // render the same pixels as Redux (not just that we're
    // self-consistent run-to-run).
    let (disp_hash, disp_w, disp_h, disp_len) = bus.gpu.display_hash();
    println!("steps={n}");
    println!("cycles={}", bus.cycles());
    println!("vram_fnv1a_64=0x{h:016x}");
    println!("vram_nonzero_words={nz}");
    println!("display_width={disp_w}");
    println!("display_height={disp_h}");
    println!("display_byte_len={disp_len}");
    println!("display_fnv1a_64=0x{disp_hash:016x}");
}
