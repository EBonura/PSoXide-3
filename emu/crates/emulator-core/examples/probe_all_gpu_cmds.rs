//! Walk the full GPU command log of our Crash run to step 600M
//! and print a histogram of ALL opcodes issued. The question
//! we need answered: does our CPU ever issue the textured
//! primitives (textured_rect / textured_tri) that would draw the
//! NAUGHTY DOG logo text? If zero such commands over the whole
//! boot, the CPU is never executing the logo-draw routine at all
//! — a game-state / IRQ-timing divergence.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_all_gpu_cmds --release
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::collections::BTreeMap;

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

    let steps: u64 = std::env::var("PSOXIDE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600_000_000);

    eprintln!("Running {steps} CPU steps with GPU-command logging...");
    let mut cycles_at_last_pump = 0u64;
    for _ in 0..steps {
        if cpu.step(&mut bus).is_err() {
            break;
        }
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }
    }

    let mut hist: BTreeMap<u8, usize> = BTreeMap::new();
    for e in &bus.gpu.cmd_log {
        *hist.entry(e.opcode).or_insert(0) += 1;
    }

    println!("Total GPU commands: {}", bus.gpu.cmd_log.len());
    println!();
    println!("{:>5}  {:>8}  {}", "op", "count", "name");
    println!("{}", "-".repeat(60));
    for (op, count) in &hist {
        println!("0x{op:02x}  {count:>8}  {}", opcode_name(*op));
    }

    // Specifically call out texture-related commands — those are
    // the ones that'd draw the NAUGHTY DOG logo text.
    let textured = [
        0x24u8, 0x25, 0x26, 0x27, 0x2c, 0x2d, 0x2e, 0x2f, 0x34, 0x35, 0x36, 0x37, 0x3c, 0x3d, 0x3e,
        0x3f, 0x64, 0x65, 0x66, 0x67, 0x6c, 0x6d, 0x6e, 0x6f, 0x74, 0x75, 0x76, 0x77, 0x7c, 0x7d,
        0x7e, 0x7f,
    ];
    let tex_count: usize = textured.iter().map(|op| *hist.get(op).unwrap_or(&0)).sum();
    println!();
    println!("Total TEXTURED primitives: {tex_count}");
    if tex_count == 0 {
        println!("→ Our game NEVER issued a textured primitive. The CPU never");
        println!("  reached the logo-draw routine. This is a CPU timing / game");
        println!("  state bug, not a renderer bug.");
    }
}

fn opcode_name(op: u8) -> &'static str {
    match op {
        0x02 => "fill_rect",
        0x20..=0x23 => "mono_tri",
        0x28..=0x2B => "mono_quad",
        0x24..=0x27 => "TEXTURED_tri",
        0x2C..=0x2F => "TEXTURED_quad",
        0x30..=0x33 => "shaded_tri",
        0x38..=0x3B => "shaded_quad",
        0x34..=0x37 => "TEXTURED_shaded_tri",
        0x3C..=0x3F => "TEXTURED_shaded_quad",
        0x40..=0x43 => "mono_line",
        0x48..=0x4B => "mono_polyline",
        0x50..=0x53 => "shaded_line",
        0x58..=0x5B => "shaded_polyline",
        0x60..=0x63 => "mono_rect_var",
        0x64..=0x67 => "TEXTURED_rect_var",
        0x68..=0x6B => "mono_rect_1x1",
        0x6C..=0x6F => "TEXTURED_rect_1x1",
        0x70..=0x73 => "mono_rect_8x8",
        0x74..=0x77 => "TEXTURED_rect_8x8",
        0x78..=0x7B => "mono_rect_16x16",
        0x7C..=0x7F => "TEXTURED_rect_16x16",
        0x80..=0x9F => "vram_to_vram_copy",
        0xA0 => "vram_upload",
        0xC0 => "vram_download",
        0xE1 => "set_draw_mode",
        0xE2 => "set_tex_window",
        0xE3 => "set_draw_area_tl",
        0xE4 => "set_draw_area_br",
        0xE5 => "set_draw_offset",
        0xE6 => "set_mask_bits",
        _ => "unknown",
    }
}
