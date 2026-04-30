//! Localize where our display diverges from Redux -- pixel heatmap +
//! side-by-side PPMs so we can eyeball the TM-glyph artifact.
//!
//! Reads `/tmp/redux_display_<n>.bin` + `/tmp/ours_display_<n>.bin`
//! (both produced by `display_parity_at`) and emits:
//! - `/tmp/diff_<n>_ours.ppm`   -- our display, 5bpc→8bpc expand
//! - `/tmp/diff_<n>_redux.ppm`  -- their display
//! - `/tmp/diff_<n>_mask.ppm`   -- black-on-red mask of differing pixels
//! - stdout -- a 40×30 ASCII heatmap of diff density and the bounding
//!   box of the densest cluster.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_tm_diff --release -- 100000000
//! ```

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let n: u64 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);

    // `display_parity_at` writes the raw 15bpp BGR pixel bytes for
    // each of the 640×478 display pixels (width×height×2 bytes).
    let redux = fs::read(format!("/tmp/redux_display_{n}.bin")).expect("redux bin");
    let ours = fs::read(format!("/tmp/ours_display_{n}.bin")).expect("ours bin");
    assert_eq!(redux.len(), ours.len(), "dim mismatch");

    // Meta file carries width / height -- parse once.
    let meta = fs::read_to_string(format!("/tmp/redux_display_{n}.bin.txt"))
        .unwrap_or_else(|_| "w=640 h=478".into());
    let (w, h) = parse_wh(&meta);
    assert_eq!(redux.len(), (w * h * 2) as usize, "byte count != w*h*2");

    let w = w as usize;
    let h = h as usize;

    // --- Write both frames as PPM + the diff mask. ---
    write_ppm(
        &PathBuf::from(format!("/tmp/diff_{n}_redux.ppm")),
        w,
        h,
        |x, y| bgr15_to_rgb24(&redux, x, y, w),
    );
    write_ppm(
        &PathBuf::from(format!("/tmp/diff_{n}_ours.ppm")),
        w,
        h,
        |x, y| bgr15_to_rgb24(&ours, x, y, w),
    );
    write_ppm(
        &PathBuf::from(format!("/tmp/diff_{n}_mask.ppm")),
        w,
        h,
        |x, y| {
            let off = (y * w + x) * 2;
            let r_px = u16::from_le_bytes([redux[off], redux[off + 1]]);
            let o_px = u16::from_le_bytes([ours[off], ours[off + 1]]);
            if r_px != o_px {
                (0xFF, 0x00, 0x00)
            } else {
                let o24 = bgr15_to_rgb24(&ours, x, y, w);
                // Dim correct pixels so diffs stand out.
                let dim = |c: u8| (c as u16 * 3 / 10) as u8;
                (dim(o24.0), dim(o24.1), dim(o24.2))
            }
        },
    );

    // --- Spatial heatmap of differing pixels: 40×30 cells.
    let hw = 40usize;
    let hh = 30usize;
    let mut cells = vec![0u32; hw * hh];
    let mut diffs_total = 0u32;
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * 2;
            let r_px = u16::from_le_bytes([redux[off], redux[off + 1]]);
            let o_px = u16::from_le_bytes([ours[off], ours[off + 1]]);
            if r_px != o_px {
                diffs_total += 1;
                let cx = x * hw / w;
                let cy = y * hh / h;
                cells[cy * hw + cx] += 1;
            }
        }
    }
    let max_cell = *cells.iter().max().unwrap_or(&0);
    println!(
        "diffs total = {diffs_total} / {} ({:.2}%)",
        w * h,
        100.0 * diffs_total as f64 / (w * h) as f64
    );
    println!("heatmap ({hw}×{hh} cells, '.'=0, '#'=hot):");
    let steps = [' ', '.', ':', 'o', 'O', '#', '@'];
    for cy in 0..hh {
        let mut row = String::with_capacity(hw);
        for cx in 0..hw {
            let v = cells[cy * hw + cx];
            let idx = if max_cell == 0 {
                0
            } else {
                (v * (steps.len() as u32 - 1) / max_cell.max(1)) as usize
            };
            row.push(steps[idx.min(steps.len() - 1)]);
        }
        println!("{row}");
    }

    // --- Bounding box of the hottest cluster (cells above 50% max).
    let threshold = max_cell / 2;
    let (mut min_cx, mut min_cy, mut max_cx, mut max_cy) = (hw, hh, 0usize, 0usize);
    for cy in 0..hh {
        for cx in 0..hw {
            if cells[cy * hw + cx] >= threshold.max(1) {
                min_cx = min_cx.min(cx);
                min_cy = min_cy.min(cy);
                max_cx = max_cx.max(cx);
                max_cy = max_cy.max(cy);
            }
        }
    }
    if min_cx <= max_cx {
        let px0 = min_cx * w / hw;
        let px1 = (max_cx + 1) * w / hw;
        let py0 = min_cy * h / hh;
        let py1 = (max_cy + 1) * h / hh;
        println!("hottest cluster (≥50% peak) spans display pixels x={px0}..{px1}  y={py0}..{py1}");
    }

    // --- Dump a small RGB swatch of mismatches at the hottest cell
    //     so we can eyeball what colour values we're producing vs
    //     expected. First 16 mismatches, pixel coords + RGB each side.
    println!();
    println!("first 16 differing pixels (coord → ours vs redux, 5bpc BGR):");
    let mut shown = 0;
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * 2;
            let r_px = u16::from_le_bytes([redux[off], redux[off + 1]]);
            let o_px = u16::from_le_bytes([ours[off], ours[off + 1]]);
            if r_px != o_px {
                let (or, og, ob) = (o_px & 0x1F, (o_px >> 5) & 0x1F, (o_px >> 10) & 0x1F);
                let (rr, rg, rb) = (r_px & 0x1F, (r_px >> 5) & 0x1F, (r_px >> 10) & 0x1F);
                println!(
                    "  ({x:>3},{y:>3})  ours=0x{o_px:04x} ({or:02},{og:02},{ob:02})  redux=0x{r_px:04x} ({rr:02},{rg:02},{rb:02})"
                );
                shown += 1;
                if shown == 16 {
                    return;
                }
            }
        }
    }
}

fn parse_wh(meta: &str) -> (u32, u32) {
    let mut w = 640u32;
    let mut h = 478u32;
    for tok in meta.split_whitespace() {
        if let Some(v) = tok.strip_prefix("w=") {
            w = v.parse().unwrap_or(w);
        } else if let Some(v) = tok.strip_prefix("h=") {
            h = v.parse().unwrap_or(h);
        }
    }
    (w, h)
}

fn bgr15_to_rgb24(buf: &[u8], x: usize, y: usize, w: usize) -> (u8, u8, u8) {
    let off = (y * w + x) * 2;
    let pix = u16::from_le_bytes([buf[off], buf[off + 1]]);
    let r5 = (pix & 0x1F) as u8;
    let g5 = ((pix >> 5) & 0x1F) as u8;
    let b5 = ((pix >> 10) & 0x1F) as u8;
    (
        (r5 << 3) | (r5 >> 2),
        (g5 << 3) | (g5 >> 2),
        (b5 << 3) | (b5 >> 2),
    )
}

fn write_ppm(path: &PathBuf, w: usize, h: usize, pixel: impl Fn(usize, usize) -> (u8, u8, u8)) {
    let mut f = fs::File::create(path).expect("create ppm");
    writeln!(f, "P6\n{w} {h}\n255").unwrap();
    let mut buf = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = pixel(x, y);
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }
    f.write_all(&buf).unwrap();
}
