//! Dump VRAM at the font texture + CLUT locations referenced by
//! Crash's NAUGHTY-DOG banner glyphs:
//!   tpage (256, 256) 8bpp
//!   CLUT (512, 304)
//!
//! If the CLUT is all zeros, every texel resolves to transparent
//! and the text renders invisibly — that's what probe_banner_order
//! turned up as the likely cause of the blank bar.
//!
//! Output: pixel values at those locations, plus non-zero counts.

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;

fn main() {
    let steps: u64 = std::env::var("PSOXIDE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600_000_000);
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = std::fs::read(
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin",
    )
    .expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    let mut cpu = Cpu::new();

    let mut cycles_at_last_pump = 0u64;
    for _ in 0..steps {
        if cpu.step(&mut bus).is_err() { break; }
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }
    }

    let vram = &bus.gpu.vram;

    // --- CLUT at (512, 304): 256 entries (8bpp), 1 row.
    println!("=== CLUT at VRAM (512, 304), row 304, x=512..768 (256 entries, 8bpp) ===");
    let mut non_zero = 0;
    let mut first_nz: Option<(u16, u16)> = None;
    for i in 0..256u16 {
        let p = vram.get_pixel(512 + i, 304);
        if p != 0 {
            non_zero += 1;
            if first_nz.is_none() {
                first_nz = Some((512 + i, 304));
            }
        }
        if i < 32 {
            print!(" {p:04x}");
            if (i + 1) % 8 == 0 { println!(); }
        }
    }
    println!();
    println!("CLUT non-zero entries: {non_zero} / 256");
    if let Some((x, y)) = first_nz {
        println!("First non-zero at VRAM ({x}, {y})");
    }

    // --- Font texture at tpage (256, 256), 8bpp. 256 wide × 256 tall.
    // For 8bpp, each VRAM word holds 2 texels. So a 256-wide texture
    // uses 128 VRAM columns. Row 256 starting at col 256.
    println!();
    println!("=== Font texture at VRAM (256..384, 256..512) — 8bpp sample ===");
    let mut tex_non_zero = 0;
    let tex_total: u32 = 128 * 256;  // words
    for y in 0..256u16 {
        for x in 0..128u16 {
            let p = vram.get_pixel(256 + x, 256 + y);
            if p != 0 {
                tex_non_zero += 1;
            }
        }
    }
    println!("Font-texture non-zero words: {tex_non_zero} / {tex_total}");

    // --- Dump a 16×16 region around a known "N" glyph
    // UV for the first post-bar glyph I saw: uv=(192,112).
    // 8bpp: tpx = 256 + 192/2 = 352. tpy = 256 + 112 = 368.
    println!();
    println!("=== Glyph around UV(192,112) → VRAM (352, 368) ===");
    for dy in 0..16 {
        for dx in 0..8 {  // 16 texels = 8 VRAM words
            let p = vram.get_pixel(352 + dx, 368 + dy);
            print!(" {p:04x}");
        }
        println!();
    }
}
