//! Per-packet divergence bisector.
//!
//! Runs the CPU rasterizer for N steps, captures `cmd_log`, then
//! replays one packet at a time onto the compute backend. After
//! each replay, snapshots the VRAM region affected by the packet
//! and compares to the CPU's VRAM. Reports the FIRST packet that
//! causes a divergence, including its opcode + raw FIFO words +
//! a small per-pixel diff sample.
//!
//! This isolates which primitive type the replay path has wrong,
//! and gives us a reproducible test case to add to the parity
//! suite.
//!
//! ```bash
//! PSOXIDE_DISC=/path/Tekken3.cue \
//!   cargo run --release -p psx-gpu-compute --example replay_bisect \
//!     -- --steps 150000000 --window 50000000
//! ```
//!
//! The probe runs CPU `--steps` cycles total; the LAST `--window`
//! cycles are the ones whose packets get bisected. Earlier cycles
//! just bring the bus + VRAM into a known state. This keeps
//! per-packet bisection tractable on long traces.

use std::path::PathBuf;

use emulator_core::gpu::GpuCmdLogEntry;
use emulator_core::{Bus, Cpu};
use psx_gpu_compute::ComputeBackend;
use psx_iso::Disc;

fn parse_args() -> (PathBuf, PathBuf, u64, u64) {
    let mut bios = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");
    let mut disc = PathBuf::from(std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue".into()
    }));
    let mut steps: u64 = 150_000_000;
    let mut window: u64 = 50_000_000;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bios" => bios = PathBuf::from(it.next().unwrap()),
            "--disc" => disc = PathBuf::from(it.next().unwrap()),
            "--steps" => steps = it.next().unwrap().parse().unwrap(),
            "--window" => window = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg: {other}"),
        }
    }
    (bios, disc, steps, window)
}

fn load_disc(p: &std::path::Path) -> Disc {
    if p.extension().and_then(|e| e.to_str()) == Some("cue") {
        psoxide_settings::library::load_disc_from_cue(p).expect("load cue")
    } else {
        Disc::from_bin(std::fs::read(p).expect("read bin"))
    }
}

fn opcode_name(op: u8) -> &'static str {
    match op {
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
        0x68..=0x6B => "MonoRect1x1",
        0x6C..=0x6F => "TexRect1x1",
        0x70..=0x73 => "MonoRect8x8",
        0x74..=0x77 => "TexRect8x8",
        0x78..=0x7B => "MonoRect16x16",
        0x7C..=0x7F => "TexRect16x16",
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
    }
}

fn main() {
    let (bios, disc_path, steps, window) = parse_args();
    eprintln!("[bisect] bios={}", bios.display());
    eprintln!("[bisect] disc={}", disc_path.display());
    eprintln!("[bisect] steps={steps}, bisect-window={window}");

    let bios_bytes = std::fs::read(&bios).expect("bios readable");
    let mut bus = Bus::new(bios_bytes).expect("bus");
    let mut cpu = Cpu::new();
    let disc = load_disc(&disc_path);
    bus.cdrom.insert_disc(Some(disc));
    bus.gpu.enable_pixel_tracer();

    let mut backend = ComputeBackend::new_headless();

    // ---------- Phase 1: warm up. Run all but the last `window`
    //  cycles WITHOUT bisection — we just want CPU + GPU VRAM in
    //  agreement at the start of the bisect window.
    let warmup = steps.saturating_sub(window);
    if warmup > 0 {
        eprintln!("[bisect] phase 1: warming up CPU for {warmup} cycles...");
        let chunk = 5_000_000u64;
        let mut cursor = 0u64;
        while cursor < warmup {
            let n = chunk.min(warmup - cursor);
            for _ in 0..n {
                if cpu.step(&mut bus).is_err() {
                    break;
                }
            }
            cursor += n;
            // Sync + replay each chunk so we don't accumulate a
            // huge cmd_log; same pattern as replay_disc.
            backend.sync_vram_from_cpu(bus.gpu.vram.words());
            let log = std::mem::take(&mut bus.gpu.cmd_log);
            for entry in &log {
                backend.replay_packet(entry);
            }
            if let Some(owner) = bus.gpu.pixel_owner.as_mut() {
                owner.fill(u32::MAX);
            }
        }
        // Sanity: VRAMs agree before bisection?
        let cpu_h = fnv1a_64(bytemuck::cast_slice(bus.gpu.vram.words()));
        let gpu_words = backend.download_vram();
        let gpu_h = fnv1a_64(bytemuck::cast_slice(&gpu_words));
        eprintln!(
            "[bisect] post-warmup: cpu=0x{cpu_h:016x} gpu=0x{gpu_h:016x} {}",
            if cpu_h == gpu_h { "OK" } else { "ALREADY DIVERGED — narrow --steps" },
        );
        if cpu_h != gpu_h {
            std::process::exit(1);
        }
    }

    // ---------- Phase 2: bisect. Run the bisect window in tiny
    //  steps, capturing exactly which packet first causes divergence.
    eprintln!("[bisect] phase 2: per-packet bisection over {window} cycles...");
    // Re-sync VRAM so the start state is identical.
    backend.sync_vram_from_cpu(bus.gpu.vram.words());

    // Run the CPU forward `window` cycles all at once, then walk
    // its cmd_log packet-by-packet on the compute backend.
    for _ in 0..window {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }
    let log = std::mem::take(&mut bus.gpu.cmd_log);
    eprintln!("[bisect] {} packets to bisect", log.len());

    // For per-packet bisection, we need the CPU's VRAM AFTER each
    // packet. We don't have that directly — `bus.gpu.vram` is the
    // FINAL state. To get post-i state, we'd need to re-run the CPU
    // packet-by-packet. Instead, use a simpler heuristic: the FIRST
    // GPU divergence appears against the FINAL CPU VRAM, and we can
    // narrow which packet caused it by snapshotting GPU VRAM before
    // each packet, comparing to GPU VRAM before the next, and
    // checking whether the affected pixels match the CPU's final.
    //
    // Even simpler: after replaying packet i, hash GPU VRAM. Compare
    // to a precomputed reference hash that we'd get if we knew the
    // CPU's incremental VRAM. We don't.
    //
    // Pragmatic approach: the CPU rasterizer's pixel_owner buffer
    // already tags each pixel with the cmd index that drew it. So
    // after replaying packet i, every pixel where pixel_owner[p].i
    // <= entry.index has been touched by SOME prior CPU packet.
    // Compare those to the GPU's VRAM at the same pixel. Once they
    // diverge for a pixel whose owner == entry.index, packet i is
    // the culprit.
    //
    // But pixel_owner only has the LAST cmd to touch each pixel,
    // not a per-packet history. So this is approximate. Still, in
    // practice the first divergence usually coincides with a packet
    // that owns the divergent pixels.
    let cpu_words: Vec<u16> = bus.gpu.vram.words().to_vec();
    let owner = bus.gpu.pixel_owner.as_ref().expect("tracer enabled");

    // The CPU's `pixel_owner` only stamps writes that go through
    // `plot_pixel` — i.e., primitive rasterization. Bulk-write
    // packets (FillRect 0x02, VRAM-to-VRAM 0x80..=0x9F,
    // CPU-to-VRAM 0xA0..=0xBF) bypass `plot_pixel` and write VRAM
    // directly, so they leave `owner` stale at the LAST plot-pixel
    // packet to touch each pixel they overwrite.
    //
    // Build an augmented owner that overlays bulk-write footprints
    // on top of the CPU's plot_pixel-only owner. This way, when a
    // FillRect later overwrites a pixel that an earlier MonoRect
    // drew, the bisector attributes the pixel to the FillRect (the
    // actual last writer), not the MonoRect.
    let mut owner_aug: Vec<u32> = owner.clone();
    for entry in &log {
        if let Some((x, y, w, h)) = bulk_writer_footprint(entry) {
            stamp_rect(&mut owner_aug, x, y, w, h, entry.index);
        }
    }

    let mut first_diverge: Option<u32> = None;
    for entry in &log {
        backend.replay_packet(entry);
        // Find any pixel that BOTH this packet owns (per augmented
        // owner) AND disagrees between CPU and GPU. That packet is
        // the offender.
        let gpu_words = backend.download_vram();
        let mut bad: Vec<(usize, u16, u16)> = Vec::new();
        for (i, (&cpu_w, &gpu_w)) in cpu_words.iter().zip(gpu_words.iter()).enumerate() {
            if cpu_w != gpu_w && owner_aug[i] == entry.index {
                bad.push((i, cpu_w, gpu_w));
                if bad.len() >= 8 {
                    break;
                }
            }
        }
        if !bad.is_empty() {
            first_diverge = Some(entry.index);
            eprintln!();
            eprintln!("=== First divergent packet ===");
            eprintln!(
                "  cmd #{} op=0x{:02X} ({})",
                entry.index,
                entry.opcode,
                opcode_name(entry.opcode),
            );
            eprintln!(
                "  fifo: [{}]",
                entry
                    .fifo
                    .iter()
                    .map(|w| format!("0x{w:08X}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            eprintln!("  diverging pixels (cpu vs gpu, max 8):");
            for (idx, c, g) in bad {
                let x = idx % 1024;
                let y = idx / 1024;
                eprintln!(
                    "    vram({x:>4},{y:>3}) cpu=0x{c:04x} gpu=0x{g:04x}",
                );
            }
            break;
        }
    }

    eprintln!();
    if first_diverge.is_some() {
        std::process::exit(1);
    } else {
        eprintln!("=== No per-packet divergence found ===");
        eprintln!("(any final-VRAM divergence is from packets whose owner-tag was overwritten by a later packet — try a smaller window)");
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

/// Return the destination-pixel footprint of bulk-write packets that
/// bypass `plot_pixel` (FillRect, VRAM-to-VRAM copy, CPU-to-VRAM
/// upload). Returns `None` for ordinary primitives, since those
/// already get accurate `pixel_owner` stamps from the CPU rasterizer.
fn bulk_writer_footprint(entry: &GpuCmdLogEntry) -> Option<(u16, u16, u16, u16)> {
    match entry.opcode {
        // GP0 0x02 — FillRect. word1 = pos (x 16-aligned, y 9-bit),
        // word2 = size (w rounded up to 16, h 9-bit).
        0x02 if entry.fifo.len() >= 3 => {
            let pos = entry.fifo[1];
            let size = entry.fifo[2];
            let x = (pos & 0x3F0) as u16;
            let y = ((pos >> 16) & 0x1FF) as u16;
            let w = (((size & 0x3FF) + 0x0F) & !0x0F) as u16;
            let h = ((size >> 16) & 0x1FF) as u16;
            Some((x, y, w, h))
        }
        // GP0 0x80..=0x9F — VRAM-to-VRAM copy. word2 = dst,
        // word3 = wh (0 → 1024/512).
        op if (0x80..=0x9F).contains(&op) && entry.fifo.len() >= 4 => {
            let dst = entry.fifo[2];
            let wh = entry.fifo[3];
            let dx = (dst & 0xFFFF) as u16;
            let dy = ((dst >> 16) & 0xFFFF) as u16;
            let raw_w = (wh & 0xFFFF) as u16;
            let raw_h = ((wh >> 16) & 0xFFFF) as u16;
            let w = if raw_w == 0 { 1024 } else { raw_w };
            let h = if raw_h == 0 { 512 } else { raw_h };
            Some((dx, dy, w, h))
        }
        // GP0 0xA0..=0xBF — CPU-to-VRAM upload. word1 = dst,
        // word2 = wh (0 → 1024/512).
        op if (0xA0..=0xBF).contains(&op) && entry.fifo.len() >= 3 => {
            let dst = entry.fifo[1];
            let wh = entry.fifo[2];
            let dx = (dst & 0xFFFF) as u16;
            let dy = ((dst >> 16) & 0xFFFF) as u16;
            let raw_w = (wh & 0xFFFF) as u16;
            let raw_h = ((wh >> 16) & 0xFFFF) as u16;
            let w = if raw_w == 0 { 1024 } else { raw_w };
            let h = if raw_h == 0 { 512 } else { raw_h };
            Some((dx, dy, w, h))
        }
        _ => None,
    }
}

fn stamp_rect(owner: &mut [u32], x: u16, y: u16, w: u16, h: u16, index: u32) {
    const VRAM_W: u16 = 1024;
    const VRAM_H: u16 = 512;
    if w == 0 || h == 0 {
        return;
    }
    for row in 0..h {
        let py = (y as u32 + row as u32) % VRAM_H as u32;
        for col in 0..w {
            let px = (x as u32 + col as u32) % VRAM_W as u32;
            owner[py as usize * VRAM_W as usize + px as usize] = index;
        }
    }
}
