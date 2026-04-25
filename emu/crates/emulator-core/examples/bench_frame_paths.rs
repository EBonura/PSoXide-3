//! Benchmark the per-frame CPU paths the frontend runs every render:
//! 1. `Vram::to_rgba8` for the 1024x512 VRAM debug texture
//! 2. `Gpu::display_rgba8` for the visible-display texture
//! 3. The packed-buffer copy that `prepare_display` runs in gfx.rs
//! 4. The actual GPU rasterizer hot paths (rasterize_textured_triangle etc.)
//!
//! Run with:
//!   cargo run -p emulator-core --example bench_frame_paths --release
//!
//! Reports ns/iter and frames-per-second equivalent for each path.

use emulator_core::{Bus, Cpu, Vram, VRAM_HEIGHT, VRAM_WIDTH};
use std::path::PathBuf;
use std::time::Instant;

#[path = "support/disc.rs"]
mod disc_support;

fn main() {
    eprintln!("== Per-frame CPU benchmark ==\n");

    bench_vram_to_rgba8_zeroed();
    bench_vram_to_rgba8_filled();
    bench_display_rgba8_with_real_disc_state();
    bench_packed_copy_only();
    bench_full_prepare_pair();
    bench_rasterizer_textured_quad();
}

fn bench(label: &str, iters: u32, mut body: impl FnMut()) {
    // Warm.
    for _ in 0..(iters / 10).max(1) {
        body();
    }
    let start = Instant::now();
    for _ in 0..iters {
        body();
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() / iters as u128;
    let mb_per_s = if ns_per > 0 {
        // Most paths produce ~2 MiB; this is approximate.
        let approx_bytes = (VRAM_WIDTH * VRAM_HEIGHT * 4) as f64;
        approx_bytes / (ns_per as f64 / 1.0e9) / 1.0e6
    } else {
        f64::NAN
    };
    let fps_budget_pct = (ns_per as f64) / (16_666_666.0) * 100.0;
    eprintln!(
        "  {label:<54}  {:>8} ns/iter  ({:>5.1}% of 60Hz frame, ~{:>6.0} MB/s)",
        ns_per, fps_budget_pct, mb_per_s
    );
}

fn bench_vram_to_rgba8_zeroed() {
    eprintln!("[1] Vram::to_rgba8 — full 1024x512 (zeroed VRAM)");
    let vram = Vram::new();
    bench("zeroed VRAM, fresh allocation each call", 200, || {
        let buf = vram.to_rgba8(0, 0, VRAM_WIDTH as u16, VRAM_HEIGHT as u16);
        std::hint::black_box(buf);
    });
    eprintln!();
}

fn bench_vram_to_rgba8_filled() {
    eprintln!("[2] Vram::to_rgba8 — full 1024x512 (random VRAM, no allocator-friendly zeros)");
    let mut vram = Vram::new();
    // Fill with non-zero data so the inner loop branch isn't biased.
    let mut s = 0xACE1u16;
    for y in 0..VRAM_HEIGHT as u16 {
        for x in 0..VRAM_WIDTH as u16 {
            // Cheap LFSR-ish random.
            s = s.wrapping_mul(0x4F1B).wrapping_add(0x7C5D);
            vram.set_pixel(x, y, s);
        }
    }
    bench("random VRAM, fresh allocation each call", 200, || {
        let buf = vram.to_rgba8(0, 0, VRAM_WIDTH as u16, VRAM_HEIGHT as u16);
        std::hint::black_box(buf);
    });
    eprintln!();
}

fn bench_display_rgba8_with_real_disc_state() {
    eprintln!("[3] Gpu::display_rgba8 — visible-area readback");
    // Boot a disc enough to have a 320x240 display configured + content.
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let disc_path = std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue".into()
    });
    let bios = match std::fs::read(&bios_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  (skipped — BIOS not readable at {}: {e})", bios_path.display());
            return;
        }
    };
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();
    if let Ok(disc) = disc_support::load_disc_path(std::path::Path::new(&disc_path)) {
        bus.cdrom.insert_disc(Some(disc));
    }
    // Run far enough to populate VRAM with real game content.
    for _ in 0..400_000_000 {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }
    let (rgba, w, h) = bus.gpu.display_rgba8();
    eprintln!(
        "  display config: {w}x{h} = {} bytes",
        rgba.len()
    );
    bench(
        &format!("display_rgba8 ({w}x{h}, fresh alloc each call)"),
        500,
        || {
            let r = bus.gpu.display_rgba8();
            std::hint::black_box(r);
        },
    );
    eprintln!();
}

fn bench_packed_copy_only() {
    eprintln!("[4] gfx.rs prepare_display — pack {w}x{h} into 1024x512 RGBA buffer", w = 320, h = 240);
    let src = vec![0xCDu8; 320 * 240 * 4];
    let width = 320u32;
    let height = 240u32;
    bench(
        "alloc 2MB zeroed + copy 320x240 rows in",
        500,
        || {
            let mut packed = vec![0u8; VRAM_WIDTH * VRAM_HEIGHT * 4];
            let src_stride = width as usize * 4;
            let dst_stride = VRAM_WIDTH * 4;
            for row in 0..height as usize {
                let src_off = row * src_stride;
                let dst_off = row * dst_stride;
                packed[dst_off..dst_off + src_stride].copy_from_slice(&src[src_off..src_off + src_stride]);
            }
            std::hint::black_box(packed);
        },
    );
    eprintln!();
}

fn bench_full_prepare_pair() {
    eprintln!("[5] Total per-frame CPU work for the display path (prepare_vram + prepare_display)");
    let mut vram = Vram::new();
    let mut s = 0x1234u16;
    for y in 0..VRAM_HEIGHT as u16 {
        for x in 0..VRAM_WIDTH as u16 {
            s = s.wrapping_mul(0x4F1B).wrapping_add(0x7C5D);
            vram.set_pixel(x, y, s);
        }
    }
    bench("vram.to_rgba8 (full) + simulated 320x240 packed copy", 200, || {
        // prepare_vram path
        let v = vram.to_rgba8(0, 0, VRAM_WIDTH as u16, VRAM_HEIGHT as u16);
        // prepare_display path (simulated 320x240)
        let mut packed = vec![0u8; VRAM_WIDTH * VRAM_HEIGHT * 4];
        let src_stride = 320 * 4;
        let dst_stride = VRAM_WIDTH * 4;
        for row in 0..240 {
            packed[row * dst_stride..row * dst_stride + src_stride]
                .copy_from_slice(&v[row * src_stride..row * src_stride + src_stride]);
        }
        std::hint::black_box((v, packed));
    });
    eprintln!();
}

fn bench_rasterizer_textured_quad() {
    eprintln!("[6] Emulator-core rasterizer — emit 1000 textured quads");
    use emulator_core::Gpu;
    let mut gpu = Gpu::new();
    // Set up a tpage + draw area first.
    gpu.gp0_push(0xE100_0000); // E1 draw mode (4bpp tpage 0)
    gpu.gp0_push(0xE300_0000); // E3 draw area TL
    gpu.gp0_push(0xE400_BFFFu32 | (479 << 10) | 639); // E4 draw area BR — clamp to 640x480
    // Seed VRAM with random texels so sampling does work.
    for y in 0..256u16 {
        for x in 0..64u16 {
            gpu.vram.set_pixel(x, y, ((x as u16) ^ (y as u16)).wrapping_mul(0xACE1));
        }
    }
    bench(
        "1000 textured quads, 64x64 each, sampled 4bpp",
        50,
        || {
            for i in 0..1000 {
                let x0 = (i & 7) * 64;
                let y0 = ((i >> 3) & 7) * 64;
                let x1 = x0 + 63;
                let y1 = y0 + 63;
                gpu.gp0_push(0x2C808080); // 0x2C textured quad, raw flag set
                gpu.gp0_push((y0 as u32) << 16 | x0 as u32); // v0
                gpu.gp0_push(0x0001_0000 | 0x0000); // uv0 + clut(=0x0001)
                gpu.gp0_push((y0 as u32) << 16 | x1 as u32); // v1
                gpu.gp0_push(0x0008_0000 | 0x003F); // uv1 + tpage
                gpu.gp0_push((y1 as u32) << 16 | x0 as u32); // v2
                gpu.gp0_push(0x0000_0000 | 0x3F00); // uv2
                gpu.gp0_push((y1 as u32) << 16 | x1 as u32); // v3
                gpu.gp0_push(0x0000_0000 | 0x3F3F); // uv3
            }
        },
    );
    eprintln!();
}
