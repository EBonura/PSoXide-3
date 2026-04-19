//! Find the FIRST command that writes pixel (80, 120) — should be
//! a textured primitive drawing an 'N' glyph, before the bar
//! overwrites it. If the bar comes BEFORE the text, they come
//! in the wrong order. If text comes before bar and gets
//! overwritten, the game assumes an ordering table (semi-trans
//! blend with the bar keeping the text visible) that we're
//! rendering differently.

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
        if cpu.step(&mut bus).is_err() { break; }
    }

    // Pixel (80, 120) — where Redux draws the 'N' in NAUGHTY.
    let target_vram_x = 80u16;
    let target_vram_y = 120u16;

    // Walk cmd_log and print commands whose bounding box covers the
    // target pixel. We approximate "covers" from vertex positions for
    // triangles/quads and from x/y/w/h for rects.
    println!("Commands that may touch pixel ({target_vram_x},{target_vram_y}):");
    println!("(look for textured_rect or textured_tri/quad)");
    println!();

    // Focus on commands in the last ~2% of the log — these draw
    // the current frame. Earlier commands draw prior frames.
    let start = bus.gpu.cmd_log.len().saturating_sub(bus.gpu.cmd_log.len() / 50);
    println!("Scanning last-frame commands (index >= {start})...");
    let mut hits = 0;
    for e in bus.gpu.cmd_log.iter().skip(start) {
        if touches_pixel(e.opcode, &e.fifo, target_vram_x as i32, target_vram_y as i32) {
            hits += 1;
            if hits <= 50 {
                println!(
                    "idx={:>7}  op=0x{:02x}  word[0]=0x{:08x}  ({})",
                    e.index, e.opcode, e.fifo[0], opcode_name(e.opcode),
                );
            }
        }
    }
    println!();
    println!("Total candidate commands in last frame: {hits}");

    // Also: find ALL textured_rect_16x16 in the last frame and show
    // their (x, y) positions, so we can see which part of the
    // framebuffer they're targeting.
    println!();
    println!("All TEXTURED_rect_16x16 in last frame (showing their position):");
    let mut count = 0;
    for e in bus.gpu.cmd_log.iter().skip(start) {
        if e.opcode == 0x7C && e.fifo.len() >= 3 {
            let pos = e.fifo[1];
            let x = sign_extend_11((pos & 0x7FF) as i32);
            let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32);
            count += 1;
            if count <= 30 {
                println!(
                    "  idx={:>7}  raw pos=({x:>4},{y:>4})  uv_clut=0x{:08x}",
                    e.index, e.fifo[2],
                );
            }
        }
    }
    println!("  Total textured_rect_16x16 in last frame: {count}");

    // Now look for text-sized glyphs in the BANNER REGION.
    // Banner y-range in screen coords is ~105..140, and we need to
    // know the draw_offset active when each glyph runs to resolve
    // screen Y. For a quick pass: assume the common draw_offset we
    // saw (256, 120) and flag glyphs with raw Y that maps inside
    // the banner.
    println!();
    println!("TEXTURED_rect_16x16 likely targeting the banner area (screen y=105..140 assuming oy=120):");
    // Candidate raw Y range: y_raw where 105 <= (y_raw + 120) <= 140
    // → y_raw in -15..20. Also (y_raw + 120 + 16) > 105 → y_raw > -31.
    let mut banner_hits = 0;
    for e in bus.gpu.cmd_log.iter().skip(start) {
        if e.opcode == 0x7C && e.fifo.len() >= 2 {
            let pos = e.fifo[1];
            let x_raw = sign_extend_11((pos & 0x7FF) as i32);
            let y_raw = sign_extend_11(((pos >> 16) & 0x7FF) as i32);
            if (-15..=20).contains(&y_raw) {
                banner_hits += 1;
                if banner_hits <= 25 {
                    let uv_clut = e.fifo[2];
                    let u = uv_clut & 0xFF;
                    let v = (uv_clut >> 8) & 0xFF;
                    let clut = (uv_clut >> 16) & 0xFFFF;
                    println!(
                        "  idx={:>7}  raw=({x_raw:>4},{y_raw:>4})  uv=({u:>3},{v:>3})  clut=0x{clut:04x}  cmd0=0x{:08x}",
                        e.index, e.fifo[0],
                    );
                }
            }
        }
    }
    println!("  Total banner-area glyphs: {banner_hits}");

    // Which of those banner-area glyphs come AFTER the bar is
    // drawn? The bar's first shaded_tri inside the banner is at
    // index ~839668. Anything past that with the bar y-range is
    // drawn "on top" of the bar — if those exist in our emulator,
    // the bar can't be overwriting them, so something else is
    // keeping them invisible (maybe CLUT transparency bug we missed).
    let bar_start_idx = 839668u32;
    println!();
    println!("Banner-area glyphs drawn AFTER the bar (idx > {bar_start_idx}):");
    let mut post_bar_count = 0;
    for e in bus.gpu.cmd_log.iter() {
        if e.opcode == 0x7C && e.fifo.len() >= 2 && e.index > bar_start_idx {
            let pos = e.fifo[1];
            let y_raw = sign_extend_11(((pos >> 16) & 0x7FF) as i32);
            if (-15..=20).contains(&y_raw) {
                post_bar_count += 1;
                if post_bar_count <= 25 {
                    let x_raw = sign_extend_11((pos & 0x7FF) as i32);
                    let uv_clut = e.fifo[2];
                    let u = uv_clut & 0xFF;
                    let v = (uv_clut >> 8) & 0xFF;
                    let clut = (uv_clut >> 16) & 0xFFFF;
                    println!(
                        "  idx={:>7}  raw=({x_raw:>4},{y_raw:>4})  uv=({u:>3},{v:>3})  clut=0x{clut:04x}",
                        e.index,
                    );
                }
            }
        }
    }
    println!("  Total banner-area glyphs post-bar: {post_bar_count}");
}

fn touches_pixel(op: u8, fifo: &[u32], target_x: i32, target_y: i32) -> bool {
    // Texture-rect primitives carry xy+size.
    let is_tex_rect = matches!(op, 0x64..=0x67 | 0x6C..=0x6F | 0x74..=0x77 | 0x7C..=0x7F);
    let is_mono_rect = matches!(op, 0x60..=0x63 | 0x68..=0x6B | 0x70..=0x73 | 0x78..=0x7B);
    let fixed_size = match op {
        0x6C..=0x6F | 0x68..=0x6B => Some(1),
        0x74..=0x77 | 0x70..=0x73 => Some(8),
        0x7C..=0x7F | 0x78..=0x7B => Some(16),
        _ => None,
    };
    if is_tex_rect || is_mono_rect {
        if fifo.len() < 2 { return false; }
        // word 1: xy (11-bit each)
        let pos = fifo[1];
        let x = sign_extend_11((pos & 0x7FF) as i32);
        let y = sign_extend_11(((pos >> 16) & 0x7FF) as i32);
        // Apply draw_offset — approximate with scan. For our purposes
        // we need the DISPLAY-SPACE draw; the logs capture 0xE5 which
        // we'd have to back-walk for exact. For a rough hit test the
        // unadjusted XY usually lands in the right neighborhood, but
        // this is approximate.
        let (w, h) = if let Some(s) = fixed_size {
            (s, s)
        } else {
            // Variable-size: w/h word lives AFTER uv/clut (if tex) or
            // is the next word for mono.
            let wh_idx = if is_tex_rect { 3 } else { 2 };
            if fifo.len() <= wh_idx { return false; }
            let wh = fifo[wh_idx];
            ((wh & 0xFFFF) as i32, ((wh >> 16) & 0xFFFF) as i32)
        };
        // Add offset constant from tracer-captured 0xE5 if present —
        // we assume 256,120 from probe output.
        let ox = 256; // draw offset observed at banner time
        let oy = 120;
        let sx = x + ox;
        let sy = y + oy;
        return target_x >= sx && target_x < sx + w && target_y >= sy && target_y < sy + h;
    }
    // Triangle / quad — bounding-box test on vertex positions.
    if matches!(op, 0x20..=0x3F) {
        // Extract vertex offsets depending on opcode shape.
        let vtx_indices: &[usize] = match op {
            0x20..=0x23 => &[1, 2, 3],            // mono_tri
            0x24..=0x27 => &[1, 3, 5],            // textured_tri
            0x28..=0x2B => &[1, 2, 3, 4],         // mono_quad
            0x2C..=0x2F => &[1, 3, 5, 7],         // textured_quad
            0x30..=0x33 => &[1, 3, 5],            // shaded_tri
            0x34..=0x37 => &[1, 4, 7],            // textured_shaded_tri
            0x38..=0x3B => &[1, 3, 5, 7],         // shaded_quad
            0x3C..=0x3F => &[1, 4, 7, 10],        // textured_shaded_quad
            _ => return false,
        };
        let mut min_x = i32::MAX; let mut max_x = i32::MIN;
        let mut min_y = i32::MAX; let mut max_y = i32::MIN;
        for &v in vtx_indices {
            if v >= fifo.len() { return false; }
            let w = fifo[v];
            let x = sign_extend_11((w & 0x7FF) as i32);
            let y = sign_extend_11(((w >> 16) & 0x7FF) as i32);
            min_x = min_x.min(x); max_x = max_x.max(x);
            min_y = min_y.min(y); max_y = max_y.max(y);
        }
        let ox = 256;
        let oy = 120;
        return target_x >= min_x + ox && target_x <= max_x + ox
            && target_y >= min_y + oy && target_y <= max_y + oy;
    }
    false
}

fn sign_extend_11(v: i32) -> i32 {
    if v & 0x400 != 0 { v | !0x7FF } else { v & 0x7FF }
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
        0x64..=0x67 => "TEXTURED_rect_var",
        0x6C..=0x6F => "TEXTURED_rect_1x1",
        0x74..=0x77 => "TEXTURED_rect_8x8",
        0x7C..=0x7F => "TEXTURED_rect_16x16",
        _ => "other",
    }
}
