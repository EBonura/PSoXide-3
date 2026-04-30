//! For a specific post-bar glyph command (idx 840377), walk
//! backwards through the cmd_log to find the MOST RECENT
//! 0xE1 (draw_mode), 0xE2 (tex_window), 0xE5 (draw_offset),
//! 0xE6 (mask_bits) that was in effect. Tells us the exact GPU
//! state at the moment the glyph draw fires -- so we can
//! reproduce the failing draw in isolation.

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
    bus.gpu.enable_pixel_tracer();
    let mut cpu = Cpu::new();

    for _ in 0..600_000_000u64 {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }

    let target: u32 = std::env::var("PSOXIDE_TARGET_IDX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(840377);

    println!("=== State before GPU cmd index {target} ===");
    // Walk backwards, collecting first occurrence of each 0xE?.
    let mut seen_e1 = false;
    let mut seen_e2 = false;
    let mut seen_e3 = false;
    let mut seen_e4 = false;
    let mut seen_e5 = false;
    let mut seen_e6 = false;
    for back in (0..=target as i64).rev() {
        let e = &bus.gpu.cmd_log[back as usize];
        match e.opcode {
            0xE1 if !seen_e1 => {
                println!("  E1 draw_mode    = 0x{:08x}", e.fifo[0]);
                seen_e1 = true;
            }
            0xE2 if !seen_e2 => {
                println!("  E2 tex_window   = 0x{:08x}", e.fifo[0]);
                seen_e2 = true;
            }
            0xE3 if !seen_e3 => {
                println!("  E3 draw_area_tl = 0x{:08x}", e.fifo[0]);
                seen_e3 = true;
            }
            0xE4 if !seen_e4 => {
                println!("  E4 draw_area_br = 0x{:08x}", e.fifo[0]);
                seen_e4 = true;
            }
            0xE5 if !seen_e5 => {
                println!("  E5 draw_offset  = 0x{:08x}", e.fifo[0]);
                seen_e5 = true;
            }
            0xE6 if !seen_e6 => {
                println!("  E6 mask_bits    = 0x{:08x}", e.fifo[0]);
                seen_e6 = true;
            }
            _ => {}
        }
        if seen_e1 && seen_e2 && seen_e3 && seen_e4 && seen_e5 && seen_e6 {
            break;
        }
    }

    // Decode E1 draw_mode.
    println!();
    let e = &bus.gpu.cmd_log[target as usize];
    println!("--- target cmd ---");
    println!("idx={}  opcode=0x{:02x}", e.index, e.opcode);
    println!("fifo ({} words):", e.fifo.len());
    for (i, w) in e.fifo.iter().enumerate() {
        println!("  [{i}] = 0x{w:08x}");
    }
}
