//! Boot Tekken 3 to the VS pre-fight screen, enable the pixel-owner
//! tracer ONLY for the frame that paints the VS screen, and dump the
//! GP0 commands that drew the LEFT (working) vs RIGHT (broken) portrait.
//!
//! Strategy:
//! 1. Run for `--enable-at` cycles with input pulses to get past the
//!    intro / menu and reach the VS screen.
//! 2. Enable `pixel_owner` tracer.
//! 3. Run another `--trace-cycles` cycles so the next frame draws.
//! 4. Sample pixels at fixed coordinates inside both portraits and
//!    print the GP0 packet (opcode + FIFO) that drew each.
//!
//! ```bash
//! PSOXIDE_DISC="/path/to/Tekken 3 (USA).cue" \
//! cargo run -p emulator-core --example probe_tekken_vs_pixel_owner \
//!   --release -- --enable-at 340000000 --trace-cycles 8000000
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::gpu::GpuCmdLogEntry;
use emulator_core::{
    fast_boot_disc_with_hle, warm_bios_for_disc_fast_boot, Bus, Cpu, DISC_FAST_BOOT_WARMUP_STEPS,
};
use std::io::Write;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

fn parse_u16_mask(s: &str) -> Option<u16> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        s.parse().ok()
    }
}

fn parse_pad_pulses(text: &str) -> Vec<PadPulse> {
    let mut out = Vec::new();
    for entry in text.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (mask_text, rest) = entry.split_once('@').unwrap();
        let mask = parse_u16_mask(mask_text).unwrap();
        let (start_text, frames_text) = match rest.split_once('+') {
            Some((s, f)) => (s.trim(), f.trim()),
            None => (rest.trim(), "1"),
        };
        out.push(PadPulse {
            mask,
            start_vblank: start_text.parse().unwrap(),
            frames: frames_text.parse().unwrap(),
        });
    }
    out
}

fn effective_mask(base: u16, pulses: &[PadPulse], current_vblank: u64) -> u16 {
    let mut mask = base;
    for p in pulses {
        if current_vblank >= p.start_vblank && current_vblank < p.start_vblank + p.frames {
            mask |= p.mask;
        }
    }
    mask
}

fn main() {
    let mut enable_at: u64 = 340_000_000;
    let mut trace_cycles: u64 = 8_000_000;
    let mut held_mask: u16 = 0;
    let mut pulses: Vec<PadPulse> = parse_pad_pulses(
        "0x0008@100+30,0x0008@500+30,0x0008@850+30,0x4000@950+10,0x4000@1100+10,0x4000@1300+10,0x4000@1500+10",
    );
    let mut out_dir = PathBuf::from("/tmp/tekken_owner");
    let mut left_x: u16 = 70;
    let mut left_y: u16 = 110;
    let mut right_x: u16 = 280;
    let mut right_y: u16 = 280;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--enable-at" => enable_at = it.next().unwrap().parse().unwrap(),
            "--trace-cycles" => trace_cycles = it.next().unwrap().parse().unwrap(),
            "--pad-mask" => held_mask = parse_u16_mask(&it.next().unwrap()).unwrap(),
            "--pad-pulses" => pulses = parse_pad_pulses(&it.next().unwrap()),
            "--out-dir" => out_dir = PathBuf::from(it.next().unwrap()),
            "--left-xy" => {
                let s = it.next().unwrap();
                let (x, y) = s.split_once(',').unwrap();
                left_x = x.parse().unwrap();
                left_y = y.parse().unwrap();
            }
            "--right-xy" => {
                let s = it.next().unwrap();
                let (x, y) = s.split_once(',').unwrap();
                right_x = x.parse().unwrap();
                right_y = y.parse().unwrap();
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let disc_path = std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue".into()
    });

    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();
    let disc =
        disc_support::load_disc_path(std::path::Path::new(&disc_path)).expect("disc readable");

    warm_bios_for_disc_fast_boot(&mut bus, &mut cpu, DISC_FAST_BOOT_WARMUP_STEPS).unwrap();
    let info = fast_boot_disc_with_hle(&mut bus, &mut cpu, &disc, false).expect("fast boot");
    eprintln!(
        "[probe] fastboot {} entry=0x{:08x}",
        info.boot_path, info.initial_pc
    );
    bus.cdrom.insert_disc(Some(disc));
    bus.attach_digital_pad_port1();

    let mut current_mask = u16::MAX;
    let mut tracer_enabled = false;
    let total = enable_at + trace_cycles;
    for i in 0..total {
        if i & 0x1FFF == 0 {
            let vblank = bus.irq().raise_counts()[0];
            let mask = effective_mask(held_mask, &pulses, vblank);
            if mask != current_mask {
                bus.set_port1_buttons(emulator_core::ButtonState::from_bits(mask));
                current_mask = mask;
            }
        }

        if !tracer_enabled && i >= enable_at {
            eprintln!(
                "[probe] enabling pixel-owner tracer at step {i}, vblank {}",
                bus.irq().raise_counts()[0]
            );
            bus.gpu.enable_pixel_tracer();
            tracer_enabled = true;
        }

        if cpu.step(&mut bus).is_err() {
            eprintln!("[probe] CPU error at step {i}");
            break;
        }
    }

    eprintln!("[probe] done, recorded {} commands", bus.gpu.cmd_log.len());

    // Save the screenshot at the end of the trace window.
    let path = out_dir.join("final.ppm");
    let (rgba, w, h) = bus.gpu.display_rgba8();
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "P6\n{w} {h}\n255").unwrap();
        for px in rgba.chunks_exact(4) {
            f.write_all(&px[..3]).unwrap();
        }
    }
    eprintln!("[probe] wrote {}", path.display());

    // Translate (display-space x,y) → (vram x,y) using the configured
    // display area. tracer stamps VRAM coords.
    let da = bus.gpu.display_area();
    let to_vram =
        |dx: u16, dy: u16| -> (u16, u16) { (da.x.wrapping_add(dx), da.y.wrapping_add(dy)) };

    eprintln!(
        "[probe] display area: x={} y={} w={} h={}",
        da.x, da.y, da.width, da.height
    );

    // Probe pixels around each portrait centre -- print the unique
    // commands found.
    let probe = |label: &str, cx: u16, cy: u16| {
        eprintln!("\n=== {label} (display-coords {cx},{cy}) ===");
        let mut seen: Vec<u32> = Vec::new();
        // Sample a 5x5 grid around the centre.
        for dy in &[-20i32, -10, 0, 10, 20] {
            for dx in &[-20i32, -10, 0, 10, 20] {
                let px = (cx as i32 + dx).max(0) as u16;
                let py = (cy as i32 + dy).max(0) as u16;
                let (vx, vy) = to_vram(px, py);
                if let Some(entry) = bus.gpu.pixel_owner_at(vx, vy) {
                    if !seen.contains(&entry.index) {
                        seen.push(entry.index);
                        print_entry(entry);
                    }
                } else {
                    eprintln!(
                        "  ({px},{py}) -> vram({vx},{vy}) : no owner (untouched / pre-tracer)"
                    );
                }
            }
        }
        eprintln!("  unique commands: {}", seen.len());
    };

    probe("LEFT portrait", left_x, left_y);
    probe("RIGHT portrait (broken)", right_x, right_y);

    // For each TexQuad we found, dump:
    //  - first 32 texture bytes at the tpage origin (to see if the
    //    texture data is even there)
    //  - first 32 CLUT entries (to see if any non-zero entries exist)
    let dump_at = |label: &str, vx: u16, vy: u16, count: u16| {
        let mut bytes = Vec::new();
        for i in 0..count {
            let pix = bus.gpu.vram.get_pixel(vx + i, vy);
            bytes.push(format!("{:04x}", pix));
        }
        eprintln!("  {label} VRAM({vx},{vy}): {}", bytes.join(" "));
    };

    eprintln!("\n=== VRAM dumps for the RIGHT-portrait textured quads ===");
    eprintln!("    King's TexQuads point to tpage_x=832 tpage_y=256 (8bpp), CLUT 0x7dd8 / 0x7dd4 / 0x7dd0");
    // The upload was 63 wide x 252 tall starting at (833,256). Sample
    // rows across the entire vertical span to see WHICH rows got
    // data and which didn't.
    eprintln!("    King upload range: x=[833..896) y=[256..508)");
    for y in (256..508).step_by(20) {
        let mut nonzero = 0;
        for col in 0..63u16 {
            if bus.gpu.vram.get_pixel(833 + col, y) != 0 {
                nonzero += 1;
            }
        }
        eprintln!("    row {y}: {nonzero}/63 nonzero pixels");
    }
    dump_at("CLUT 0x7dd0 (320,503)", 320, 503, 16);
    dump_at("CLUT 0x7dd4 (320,503+0)", 320, 503, 16);
    dump_at("CLUT 0x7dd8 (384,503)", 384, 503, 16);

    eprintln!("\n=== ALL CPU->VRAM uploads with X >= 700 (likely portraits) ===");
    for entry in &bus.gpu.cmd_log {
        if !(0xA0..=0xBF).contains(&entry.opcode) {
            continue;
        }
        if entry.fifo.len() < 3 {
            continue;
        }
        let xy = entry.fifo[1];
        let wh = entry.fifo[2];
        let x = (xy & 0x3FF) as u16;
        let y = ((xy >> 16) & 0x1FF) as u16;
        let raw_w = (wh & 0x3FF) as u16;
        let raw_h = ((wh >> 16) & 0x1FF) as u16;
        let w = if raw_w == 0 { 1024 } else { raw_w };
        let h = if raw_h == 0 { 512 } else { raw_h };
        if x < 700 {
            continue;
        }
        eprintln!(
            "  cmd #{:>5} CPU->VRAM dst=({x},{y}) {w}x{h}  fifo[0]=0x{:08x} xy=0x{:08x} wh=0x{:08x}",
            entry.index, entry.fifo[0], xy, wh
        );
    }

    eprintln!("\n=== VRAM dumps for the LEFT-portrait textured quad (working) ===");
    eprintln!("    Xiaoyu's TexQuad points to tpage_x=768 tpage_y=256 (8bpp), CLUT 0x7d94");
    dump_at("Xiaoyu texture (768,256)", 768, 256, 16);
    dump_at("Xiaoyu texture (768,310)", 768, 310, 16);
    let clut_x = (0x7d94u32 & 0x3F) * 16;
    let clut_y = (0x7d94u32 >> 6) & 0x1FF;
    dump_at(
        &format!("Xiaoyu CLUT 0x7d94 → ({clut_x},{clut_y})"),
        clut_x as u16,
        clut_y as u16,
        16,
    );

    // Hunt: all TexQuad / TexTri / ShadedTexQuad commands whose vertex
    // coords land inside the broken portrait rect. Pixel-owner only
    // tracks the LAST writer, so a textured polygon that produced
    // ALL-TRANSPARENT texels (sample_texture returned None for every
    // pixel) won't show up in `probe`. List every textured primitive
    // by inspecting raw vertex coords.
    eprintln!(
        "\n=== Textured primitives whose bbox intersects right portrait area ({}+- ) ===",
        right_x
    );
    let bbox_x0 = right_x.saturating_sub(60) as i32;
    let bbox_x1 = (right_x as i32) + 60;
    let bbox_y0 = right_y.saturating_sub(60) as i32;
    let bbox_y1 = (right_y as i32) + 60;
    for entry in &bus.gpu.cmd_log {
        let is_textured = matches!(
            entry.opcode,
            0x24..=0x27 | 0x2C..=0x2F | 0x34..=0x37 | 0x3C..=0x3F | 0x64..=0x67
                | 0x6C..=0x6F | 0x74..=0x77 | 0x7C..=0x7F
        );
        if !is_textured {
            continue;
        }
        // Find a vertex inside the bbox. Vertex word locations vary by
        // opcode but the X is always low 11 bits, Y next 11 bits of a
        // word; cheaper to scan ALL words and pick "looks like vertex"
        // (low/high half are signed-11). False positives are fine --
        // we just want a coarse hit.
        let mut hit = false;
        for &w in &entry.fifo {
            let x = ((w & 0x7FF) as i32) << 21 >> 21;
            let y = (((w >> 16) & 0x7FF) as i32) << 21 >> 21;
            if x >= bbox_x0 && x <= bbox_x1 && y >= bbox_y0 && y <= bbox_y1 {
                hit = true;
                break;
            }
        }
        if hit {
            print_entry(entry);
        }
    }
}

#[allow(clippy::collapsible_match)]
fn print_entry(entry: &GpuCmdLogEntry) {
    let opcode = entry.opcode;
    let opname = match opcode {
        0x00 => "NOP",
        0x01 => "ClearCache",
        0x02 => "QuickFill",
        0x20..=0x23 => "MonoTri",
        0x24..=0x27 => "TexTri",
        0x28..=0x2B => "MonoQuad",
        0x2C..=0x2F => "TexQuad",
        0x30..=0x33 => "ShadedTri",
        0x34..=0x37 => "ShadedTexTri",
        0x38..=0x3B => "ShadedQuad",
        0x3C..=0x3F => "ShadedTexQuad",
        0x40..=0x47 => "Line",
        0x48..=0x4F => "PolyLine",
        0x50..=0x57 => "ShadedLine",
        0x58..=0x5F => "ShadedPolyLine",
        0x60..=0x63 => "MonoRect",
        0x64..=0x67 => "TexRect",
        0x68..=0x6F => "MonoRectFixed",
        0x70..=0x77 => "TexRectFixed",
        0x78..=0x7B => "MonoRect16",
        0x7C..=0x7F => "TexRect16",
        0x80..=0x9F => "VRAMtoVRAM",
        0xA0..=0xBF => "CPUtoVRAM",
        0xC0..=0xDF => "VRAMtoCPU",
        0xE1 => "DrawMode",
        0xE2 => "TexWindow",
        0xE3 => "DrawAreaTL",
        0xE4 => "DrawAreaBR",
        0xE5 => "DrawOffset",
        0xE6 => "MaskBit",
        _ => "?",
    };
    let words: Vec<String> = entry.fifo.iter().map(|w| format!("{:08x}", w)).collect();
    eprintln!(
        "  cmd #{:>5} op=0x{:02x} ({:<14}) fifo=[{}]",
        entry.index,
        opcode,
        opname,
        words.join(" ")
    );

    // Decode tpage/CLUT for textured polys.
    let decode_uv = |w: u32| (w & 0xFF, (w >> 8) & 0xFF);
    match opcode {
        0x24..=0x27 => {
            // TexTri: cmd, v0, uv0+clut, v1, uv1+tpage, v2, uv2
            if entry.fifo.len() >= 7 {
                let clut = (entry.fifo[2] >> 16) & 0xFFFF;
                let tpage = (entry.fifo[4] >> 16) & 0xFFFF;
                let (u0, v0) = decode_uv(entry.fifo[2]);
                let (u1, v1) = decode_uv(entry.fifo[4]);
                let (u2, v2) = decode_uv(entry.fifo[6]);
                eprintln!(
                    "    clut=0x{clut:04x} tpage=0x{tpage:04x} \
                     uvs=({u0},{v0}) ({u1},{v1}) ({u2},{v2})"
                );
                describe_tpage(tpage as u16);
            }
        }
        0x2C..=0x2F => {
            // TexQuad: cmd, v0, uv0+clut, v1, uv1+tpage, v2, uv2, v3, uv3
            if entry.fifo.len() >= 9 {
                let clut = (entry.fifo[2] >> 16) & 0xFFFF;
                let tpage = (entry.fifo[4] >> 16) & 0xFFFF;
                let (u0, v0) = decode_uv(entry.fifo[2]);
                let (u1, v1) = decode_uv(entry.fifo[4]);
                let (u2, v2) = decode_uv(entry.fifo[6]);
                let (u3, v3) = decode_uv(entry.fifo[8]);
                eprintln!(
                    "    clut=0x{clut:04x} tpage=0x{tpage:04x} \
                     uvs=({u0},{v0}) ({u1},{v1}) ({u2},{v2}) ({u3},{v3})"
                );
                describe_tpage(tpage as u16);
            }
        }
        0x34..=0x37 => {
            // ShadedTexTri: cmd+c0, v0, uv0+clut, c1, v1, uv1+tpage, c2, v2, uv2
            if entry.fifo.len() >= 9 {
                let clut = (entry.fifo[2] >> 16) & 0xFFFF;
                let tpage = (entry.fifo[5] >> 16) & 0xFFFF;
                eprintln!("    clut=0x{clut:04x} tpage=0x{tpage:04x}");
                describe_tpage(tpage as u16);
            }
        }
        0x3C..=0x3F => {
            // ShadedTexQuad: cmd+c0, v0, uv0+clut, c1, v1, uv1+tpage, c2, v2, uv2, c3, v3, uv3
            if entry.fifo.len() >= 12 {
                let clut = (entry.fifo[2] >> 16) & 0xFFFF;
                let tpage = (entry.fifo[5] >> 16) & 0xFFFF;
                eprintln!("    clut=0x{clut:04x} tpage=0x{tpage:04x}");
                describe_tpage(tpage as u16);
            }
        }
        0x60..=0x7F => {
            if entry.fifo.len() >= 3 {
                let clut = (entry.fifo[2] >> 16) & 0xFFFF;
                eprintln!("    clut=0x{clut:04x}");
            }
        }
        _ => {}
    }
}

fn describe_tpage(tpage: u16) {
    let tx = (tpage & 0x0F) * 64;
    let ty = if (tpage >> 4) & 1 != 0 { 256 } else { 0 };
    let bpp = (tpage >> 7) & 0x3;
    let blend = (tpage >> 5) & 0x3;
    let bpp_str = match bpp {
        0 => "4bpp",
        1 => "8bpp",
        2 => "15bpp",
        _ => "RES",
    };
    let tex_dis = (tpage >> 11) & 1;
    eprintln!("      tpage_x={tx} tpage_y={ty} {bpp_str} blend={blend} tex_disable={tex_dis}");
}
