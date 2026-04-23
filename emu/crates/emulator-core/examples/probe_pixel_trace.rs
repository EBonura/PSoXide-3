//! Run Crash to a checkpoint with the GPU pixel-tracer armed; then
//! look up which GPU command last drew a specific (x, y) on the
//! display. Prints the command's opcode, FIFO words, and a decoded
//! interpretation (vertices, tint, UV, CLUT, texpage, mode).
//!
//! The goal is to turn "pixel (210, 26) at step 900M differs from
//! Redux by a few shading units" into "command 52731 was a textured-
//! Gouraud triangle with CLUT at X,Y and tint RGB X,Y,Z" — so the
//! divergence can be traced to a specific input rather than guessed
//! at from the resulting pixel color.
//!
//! Usage:
//! ```bash
//! cargo run -p emulator-core --example probe_pixel_trace --release -- <steps> <x> <y>
//! # e.g.:
//! cargo run -p emulator-core --example probe_pixel_trace --release -- 900000000 210 26
//! ```
//!
//! The coordinates are in DISPLAY space (not VRAM) — they're shifted
//! into VRAM via the current `display_area()` before lookup.

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let steps: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_pixel_trace <steps> <x> <y>");
    let x: u16 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_pixel_trace <steps> <x> <y>");
    let y: u16 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_pixel_trace <steps> <x> <y>");

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc = std::fs::read(
        "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin",
    )
    .expect("disc");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    bus.gpu.enable_pixel_tracer();
    let mut cpu = Cpu::new();

    let mut cycles_at_last_pump = 0u64;
    eprintln!("[pixel_trace] running {steps} CPU steps with tracer armed...");
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

    let da = bus.gpu.display_area();
    let vram_x = da.x + x;
    let vram_y = da.y + y;
    let pixel_value = bus.gpu.vram.get_pixel(vram_x, vram_y);

    println!();
    println!("=== Pixel trace @ step {steps} ===");
    println!(
        "display origin = ({}, {}) size = {}×{}",
        da.x, da.y, da.width, da.height,
    );
    println!("target display pixel = ({x}, {y}) → VRAM ({vram_x}, {vram_y})",);
    println!("current pixel value  = 0x{pixel_value:04x}");
    println!("cmd_log size         = {}", bus.gpu.cmd_log.len());
    println!();

    match bus.gpu.pixel_owner_at(vram_x, vram_y) {
        None => {
            println!("No command owns this pixel — either never written, or the");
            println!("tracer was off when it was last written. If the pixel value");
            println!("is non-zero, that's a bug.");
        }
        Some(entry) => {
            println!("--- Last command to write this pixel ---");
            println!("index   = {}", entry.index);
            println!(
                "opcode  = 0x{:02x} ({})",
                entry.opcode,
                opcode_name(entry.opcode)
            );
            println!("fifo ({} words):", entry.fifo.len());
            for (i, w) in entry.fifo.iter().enumerate() {
                println!("  [{i}] = 0x{w:08x}");
            }
            println!();
            decode_packet(entry.opcode, &entry.fifo);
        }
    }

    // Also dump the 5 commands before and after — useful when the
    // divergence is actually a neighbour's bleed / overwrite.
    let owner_idx = bus
        .gpu
        .pixel_owner_at(vram_x, vram_y)
        .map(|e| e.index as i64);
    if let Some(idx) = owner_idx {
        println!();
        println!("--- Nearby commands (opcode + first FIFO word) ---");
        // Widened from 5 to 40 so we can see whether text / logo
        // primitives follow the bar background — if they do, we're
        // drawing them but getting the blend wrong; if they don't,
        // the game code never issued them.
        let lo = (idx - 5).max(0);
        let hi = (idx + 40).min(bus.gpu.cmd_log.len() as i64 - 1);
        for i in lo..=hi {
            let e = &bus.gpu.cmd_log[i as usize];
            let mark = if i == idx { ">>>" } else { "   " };
            println!(
                "{mark} idx={:>7}  op=0x{:02x}  word[0]=0x{:08x}  ({})",
                e.index,
                e.opcode,
                e.fifo[0],
                opcode_name(e.opcode),
            );
        }
        // Also scan AHEAD for any primitives that should draw ON TOP
        // of this pixel's triangle — they'd cover our bar with the
        // "NAUGHTY DOG" logo / Crash portrait. If none exist, the
        // game code never emitted them (CPU-side bug). If they
        // exist but we got nothing visible, the renderer is
        // rejecting them (GPU-side bug).
        println!();
        println!("--- Scan forward 400 commands for texture/line primitives ---");
        let scan_hi = (idx + 400).min(bus.gpu.cmd_log.len() as i64 - 1);
        let mut non_shaded_tri_count = 0usize;
        let mut histogram: std::collections::BTreeMap<u8, usize> = Default::default();
        for i in (idx + 1)..=scan_hi {
            let e = &bus.gpu.cmd_log[i as usize];
            *histogram.entry(e.opcode).or_insert(0) += 1;
            if !matches!(e.opcode, 0x30..=0x33 | 0x38..=0x3B) {
                // Non-shaded-tri/quad — log the first 10 of these.
                if non_shaded_tri_count < 10 {
                    println!(
                        "    idx={:>7}  op=0x{:02x}  word[0]=0x{:08x}  ({})",
                        e.index,
                        e.opcode,
                        e.fifo[0],
                        opcode_name(e.opcode),
                    );
                }
                non_shaded_tri_count += 1;
            }
        }
        println!("  Total non-shaded-tri commands in next 400: {non_shaded_tri_count}",);
        println!("  Opcode histogram:");
        for (op, count) in histogram.iter() {
            println!("    op=0x{op:02x} ({:>20}) : {count}", opcode_name(*op));
        }

        // Walk backwards from the owning draw to find the most-recent
        // state packet of each type. This reconstructs the exact GPU
        // state in effect when the draw happened — essential context
        // for reproducing the draw in isolation.
        println!();
        println!("--- GPU state context at owning draw ---");
        let mut seen_e1 = false;
        let mut seen_e2 = false;
        let mut seen_e3 = false;
        let mut seen_e4 = false;
        let mut seen_e5 = false;
        let mut seen_e6 = false;
        let lookback_start = idx;
        for back in (0..=lookback_start).rev() {
            let e = &bus.gpu.cmd_log[back as usize];
            let mut emit = |label: &str| println!("  {label:<22} = 0x{:08x}", e.fifo[0]);
            match e.opcode {
                0xE1 if !seen_e1 => {
                    emit("E1 draw_mode");
                    seen_e1 = true;
                }
                0xE2 if !seen_e2 => {
                    emit("E2 tex_window");
                    seen_e2 = true;
                }
                0xE3 if !seen_e3 => {
                    emit("E3 draw_area_tl");
                    seen_e3 = true;
                }
                0xE4 if !seen_e4 => {
                    emit("E4 draw_area_br");
                    seen_e4 = true;
                }
                0xE5 if !seen_e5 => {
                    emit("E5 draw_offset");
                    seen_e5 = true;
                }
                0xE6 if !seen_e6 => {
                    emit("E6 mask_bits");
                    seen_e6 = true;
                }
                _ => {}
            }
            if seen_e1 && seen_e2 && seen_e3 && seen_e4 && seen_e5 && seen_e6 {
                break;
            }
        }

        // Decode draw_offset specifically since it shifts the vertices
        // we printed earlier from relative to screen.
        let owner = &bus.gpu.cmd_log[idx as usize];
        println!();
        println!("--- Effective screen-space positions ---");
        let e5 = (0..=idx).rev().find_map(|i| {
            let e = &bus.gpu.cmd_log[i as usize];
            (e.opcode == 0xE5).then_some(e.fifo[0])
        });
        if let Some(e5) = e5 {
            let ox = sign_extend_11((e5 & 0x7FF) as i32);
            let oy = sign_extend_11(((e5 >> 11) & 0x7FF) as i32);
            println!("  draw_offset   = ({ox}, {oy})");
            // Try to decode screen vertices for a triangle-shaped packet.
            if matches!(owner.opcode, 0x24..=0x27 | 0x30..=0x33 | 0x34..=0x37) {
                let (v0, v1, v2) = triangle_vertices(owner.opcode, &owner.fifo);
                println!("  screen v0 = ({}, {})", v0.0 + ox, v0.1 + oy,);
                println!("  screen v1 = ({}, {})", v1.0 + ox, v1.1 + oy,);
                println!("  screen v2 = ({}, {})", v2.0 + ox, v2.1 + oy,);
            }
        }
    }
}

fn triangle_vertices(op: u8, fifo: &[u32]) -> ((i32, i32), (i32, i32), (i32, i32)) {
    // Picks the three vertex words out of each triangle-ish packet.
    match op {
        0x24..=0x27 => (decode_xy(fifo[1]), decode_xy(fifo[3]), decode_xy(fifo[5])),
        0x30..=0x33 => (decode_xy(fifo[1]), decode_xy(fifo[3]), decode_xy(fifo[5])),
        0x34..=0x37 => (decode_xy(fifo[1]), decode_xy(fifo[4]), decode_xy(fifo[7])),
        _ => ((0, 0), (0, 0), (0, 0)),
    }
}

fn opcode_name(op: u8) -> &'static str {
    match op {
        0x02 => "fill_rect",
        0x20..=0x23 => "mono_tri",
        0x28..=0x2B => "mono_quad",
        0x24..=0x27 => "textured_tri",
        0x2C..=0x2F => "textured_quad",
        0x30..=0x33 => "shaded_tri",
        0x38..=0x3B => "shaded_quad",
        0x34..=0x37 => "textured_shaded_tri",
        0x3C..=0x3F => "textured_shaded_quad",
        0x40..=0x43 => "mono_line",
        0x48..=0x4B => "mono_polyline",
        0x50..=0x53 => "shaded_line",
        0x58..=0x5B => "shaded_polyline",
        0x60..=0x63 => "mono_rect_variable",
        0x64..=0x67 => "textured_rect_variable",
        0x68..=0x6B => "mono_rect_1x1",
        0x6C..=0x6F => "textured_rect_1x1",
        0x70..=0x73 => "mono_rect_8x8",
        0x74..=0x77 => "textured_rect_8x8",
        0x78..=0x7B => "mono_rect_16x16",
        0x7C..=0x7F => "textured_rect_16x16",
        0x80..=0x9F => "vram_to_vram_copy",
        0xA0 => "begin_vram_upload",
        0xC0 => "begin_vram_download",
        0xE1 => "set_draw_mode",
        0xE2 => "set_tex_window",
        0xE3 => "set_draw_area_tl",
        0xE4 => "set_draw_area_br",
        0xE5 => "set_draw_offset",
        0xE6 => "set_mask_bits",
        _ => "unknown",
    }
}

fn decode_packet(op: u8, fifo: &[u32]) {
    // Decode the first-word tint and the primitive category so the
    // author can spot suspicious fields fast.
    let cmd = fifo.first().copied().unwrap_or(0);
    let cmd_tint_r = (cmd >> 0) & 0xFF;
    let cmd_tint_g = (cmd >> 8) & 0xFF;
    let cmd_tint_b = (cmd >> 16) & 0xFF;
    let raw_tex = cmd & (1 << 24) != 0; // bit 24 of cmd0
    let semi_tr = cmd & (1 << 25) != 0; // bit 25
    println!("decoded:");
    println!(
        "  cmd0 tint       = ({cmd_tint_r}, {cmd_tint_g}, {cmd_tint_b})  raw_tex={raw_tex}  semi_trans={semi_tr}",
    );

    match op {
        0x24..=0x27 => {
            // Textured tri: [cmd+tint, v0, clut+uv0, v1, tpage+uv1, v2, uv2]
            if fifo.len() >= 7 {
                let v0 = decode_xy(fifo[1]);
                let uv0 = fifo[2];
                let v1 = decode_xy(fifo[3]);
                let uv1 = fifo[4];
                let v2 = decode_xy(fifo[5]);
                let uv2 = fifo[6];
                let clut = (uv0 >> 16) & 0xFFFF;
                let tpage = (uv1 >> 16) & 0xFFFF;
                println!("  v0={:?}  uv0=({}, {})", v0, uv0 & 0xFF, (uv0 >> 8) & 0xFF);
                println!("  v1={:?}  uv1=({}, {})", v1, uv1 & 0xFF, (uv1 >> 8) & 0xFF);
                println!("  v2={:?}  uv2=({}, {})", v2, uv2 & 0xFF, (uv2 >> 8) & 0xFF);
                println!("  clut={:>16}", fmt_clut(clut));
                println!("  tpage={:>15}", fmt_tpage(tpage));
            }
        }
        0x30..=0x33 => {
            // Shaded tri: [c0+cmd, v0, c1, v1, c2, v2]
            if fifo.len() >= 6 {
                println!(
                    "  c0=(r={}, g={}, b={})  v0={:?}",
                    cmd_tint_r,
                    cmd_tint_g,
                    cmd_tint_b,
                    decode_xy(fifo[1])
                );
                let c1 = fifo[2] & 0x00FF_FFFF;
                let c2 = fifo[4] & 0x00FF_FFFF;
                println!(
                    "  c1=(r={}, g={}, b={})  v1={:?}",
                    c1 & 0xFF,
                    (c1 >> 8) & 0xFF,
                    (c1 >> 16) & 0xFF,
                    decode_xy(fifo[3])
                );
                println!(
                    "  c2=(r={}, g={}, b={})  v2={:?}",
                    c2 & 0xFF,
                    (c2 >> 8) & 0xFF,
                    (c2 >> 16) & 0xFF,
                    decode_xy(fifo[5])
                );
            }
        }
        0x34..=0x37 => {
            // Textured + shaded tri: [c0+cmd, v0, clut+uv0, c1, v1, tpage+uv1, c2, v2, uv2]
            if fifo.len() >= 9 {
                let uv0 = fifo[2];
                let c1 = fifo[3] & 0x00FF_FFFF;
                let uv1 = fifo[5];
                let c2 = fifo[6] & 0x00FF_FFFF;
                let uv2 = fifo[8];
                let clut = (uv0 >> 16) & 0xFFFF;
                let tpage = (uv1 >> 16) & 0xFFFF;
                println!(
                    "  c0=(r={}, g={}, b={})  v0={:?}  uv0=({}, {})",
                    cmd_tint_r,
                    cmd_tint_g,
                    cmd_tint_b,
                    decode_xy(fifo[1]),
                    uv0 & 0xFF,
                    (uv0 >> 8) & 0xFF
                );
                println!(
                    "  c1=(r={}, g={}, b={})  v1={:?}  uv1=({}, {})",
                    c1 & 0xFF,
                    (c1 >> 8) & 0xFF,
                    (c1 >> 16) & 0xFF,
                    decode_xy(fifo[4]),
                    uv1 & 0xFF,
                    (uv1 >> 8) & 0xFF
                );
                println!(
                    "  c2=(r={}, g={}, b={})  v2={:?}  uv2=({}, {})",
                    c2 & 0xFF,
                    (c2 >> 8) & 0xFF,
                    (c2 >> 16) & 0xFF,
                    decode_xy(fifo[7]),
                    uv2 & 0xFF,
                    (uv2 >> 8) & 0xFF
                );
                println!("  clut={:>16}", fmt_clut(clut));
                println!("  tpage={:>15}", fmt_tpage(tpage));
            }
        }
        0x38..=0x3B => {
            // Shaded quad: 8 words — same pattern as shaded tri, 4 vtx.
            if fifo.len() >= 8 {
                for i in 0..4 {
                    let (c, v) = if i == 0 {
                        (cmd & 0x00FF_FFFF, fifo[1])
                    } else {
                        let ci = fifo[i * 2] & 0x00FF_FFFF;
                        let vi = fifo[i * 2 + 1];
                        (ci, vi)
                    };
                    println!(
                        "  c{i}=(r={}, g={}, b={})  v{i}={:?}",
                        c & 0xFF,
                        (c >> 8) & 0xFF,
                        (c >> 16) & 0xFF,
                        decode_xy(v)
                    );
                }
            }
        }
        _ => {}
    }
}

fn decode_xy(w: u32) -> (i32, i32) {
    let x = sign_extend_11((w & 0x7FF) as i32);
    let y = sign_extend_11(((w >> 16) & 0x7FF) as i32);
    (x, y)
}

fn sign_extend_11(v: i32) -> i32 {
    if v & 0x400 != 0 {
        v | !0x7FF
    } else {
        v & 0x7FF
    }
}

fn fmt_clut(clut: u32) -> String {
    let cx = (clut & 0x3F) * 16;
    let cy = (clut >> 6) & 0x1FF;
    format!("({cx}, {cy})")
}

fn fmt_tpage(tpage: u32) -> String {
    let x = (tpage & 0x0F) * 64;
    let y = if (tpage >> 4) & 1 != 0 { 256 } else { 0 };
    let depth = match (tpage >> 7) & 3 {
        0 => "4bpp",
        1 => "8bpp",
        2 | _ => "15bpp",
    };
    let blend = match (tpage >> 5) & 3 {
        0 => "avg",
        1 => "add",
        2 => "sub",
        3 | _ => "quarter",
    };
    format!("({x}, {y}) {depth} blend={blend}")
}
