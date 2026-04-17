//! Quick diagnostic: run BIOS for N instructions (default 5M; pass a
//! larger number as argv[1]) and report GPU/IRQ state. Useful for
//! verifying Phase 4b/5b didn't silently break anything and whether
//! the BIOS has reached the screen-fill phase yet.

use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000_000);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("read BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    // Sample the last ~1% of the run for a PC histogram — tells us
    // whether BIOS is in a tight loop or executing broadly.
    let sample_start = n.saturating_sub(n / 100);
    let mut pc_hits: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();

    let mut stopped_at: Option<(u64, emulator_core::ExecutionError)> = None;
    for i in 0..n {
        if i >= sample_start {
            *pc_hits.entry(cpu.pc()).or_insert(0) += 1;
        }
        if let Err(e) = cpu.step(&mut bus) {
            stopped_at = Some((i, e));
            break;
        }
    }

    println!("=== smoke_draw after {n} attempted steps ===");
    println!("cpu.tick         = {}", cpu.tick());
    println!("bus.cycles       = {}", bus.cycles());
    println!("cpu.pc           = 0x{:08x}", cpu.pc());
    println!("next_vblank_cyc  = {}", bus.next_vblank_cycle());
    println!("gp0_write_count  = {}", bus.gpu.gp0_write_count());
    println!("cdrom commands   = {}", bus.cdrom.commands_dispatched());
    let hist = bus.cdrom.command_histogram();
    let nonzero: Vec<(u8, u32)> = hist
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if c > 0 { Some((i as u8, c)) } else { None })
        .collect();
    if !nonzero.is_empty() {
        println!(
            "cdrom cmd histogram (op: count):  {:?}",
            nonzero
                .iter()
                .map(|(op, c)| format!("0x{op:02X}:{c}"))
                .collect::<Vec<_>>()
        );
    }
    println!("cdrom irq_flag   = 0x{:02X}", bus.cdrom.irq_flag());
    // Need a safe peek at IRQ controller state — reading through the bus
    // would need &mut. Read raw IRQ register via the MMIO path.
    let i_stat = {
        let b0 = bus.try_read8(0xBF80_1070).unwrap_or(0) as u32;
        let b1 = bus.try_read8(0xBF80_1071).unwrap_or(0) as u32;
        let b2 = bus.try_read8(0xBF80_1072).unwrap_or(0) as u32;
        let b3 = bus.try_read8(0xBF80_1073).unwrap_or(0) as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    };
    let i_mask = {
        let b0 = bus.try_read8(0xBF80_1074).unwrap_or(0) as u32;
        let b1 = bus.try_read8(0xBF80_1075).unwrap_or(0) as u32;
        let b2 = bus.try_read8(0xBF80_1076).unwrap_or(0) as u32;
        let b3 = bus.try_read8(0xBF80_1077).unwrap_or(0) as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    };
    println!("I_STAT           = 0x{i_stat:08x}");
    println!("I_MASK           = 0x{i_mask:08x}");

    let da = bus.gpu.display_area();
    println!(
        "display area     = {}×{} at ({},{}){}",
        da.width,
        da.height,
        da.x,
        da.y,
        if da.bpp24 { " · 24bpp" } else { "" }
    );

    let gp0_hist = bus.gpu.gp0_opcode_histogram();
    let nonzero_gp0: Vec<(usize, u32)> = gp0_hist
        .iter()
        .enumerate()
        .filter_map(|(op, &c)| if c > 0 { Some((op, c)) } else { None })
        .collect();
    if !nonzero_gp0.is_empty() {
        println!("GP0 opcode histogram (op: count):");
        for (op, c) in &nonzero_gp0 {
            let name = match *op as u8 {
                0x00 => "nop",
                0x01 => "clear-cache",
                0x02 => "fill-rect",
                0x20..=0x23 => "mono-tri",
                0x24..=0x27 => "textured-tri",
                0x28..=0x2B => "mono-quad",
                0x2C..=0x2F => "textured-quad",
                0x30..=0x33 => "shaded-tri",
                0x38..=0x3B => "shaded-quad",
                0x60..=0x7F => "rect",
                0x80..=0x9F => "vram-blit",
                0xA0..=0xBF => "cpu->vram",
                0xC0..=0xDF => "vram->cpu",
                0xE1 => "draw-mode",
                0xE2 => "tex-window",
                0xE3 => "draw-area-tl",
                0xE4 => "draw-area-br",
                0xE5 => "draw-offset",
                0xE6 => "mask-bit",
                _ => "?",
            };
            println!("  0x{op:02X} {name:<14} {c}");
        }
    }

    // Top 10 PCs in the sampled window.
    println!("\n=== top PCs in last 1% of run ===");
    let mut top: Vec<(u32, u32)> = pc_hits.iter().map(|(k, v)| (*k, *v)).collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    for (pc, count) in top.iter().take(10) {
        println!("  {pc:08x}: {count:>8} hits");
    }
    println!("  (unique PCs in sample: {})", pc_hits.len());
    println!(
        "DMA start triggers per channel (0=MDECin 1=MDECout 2=GPU 3=CDROM 4=SPU 5=PIO 6=OTC):"
    );
    // Access via a public getter or expose fields? Use bus.dma... but that's private.
    // Expose via accessor on Bus.
    println!("  {:?}", bus.dma_start_triggers());
    println!("SR               = 0x{:08x}", cpu.cop0()[12]);
    println!("CAUSE            = 0x{:08x}", cpu.cop0()[13]);
    println!("EPC              = 0x{:08x}", cpu.cop0()[14]);
    let exc_counts = cpu.exception_counts();
    let exc_names: [&str; 32] = [
        "Int", "Mod", "TLBL", "TLBS", "AdEL", "AdES", "IBE", "DBE", "Syscall",
        "Break", "RI", "CpU", "Ov", "Tr", "-", "-", "-", "-", "-", "-", "-", "-",
        "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
    ];
    let mut any = false;
    for (i, &n) in exc_counts.iter().enumerate() {
        if n > 0 {
            if !any {
                print!("exceptions       =");
                any = true;
            }
            print!(" {}:{n}", exc_names[i]);
        }
    }
    if any {
        println!();
    } else {
        println!("exceptions       = (none)");
    }
    println!(
        "irq_line_high    = {} steps",
        cpu.irq_line_high_steps()
    );
    println!(
        "should_take_irq  = {} steps",
        cpu.should_take_interrupt_steps()
    );
    let irq = bus.irq();
    let raise_names = [
        "VBlank", "Gpu", "Cdrom", "Dma", "Timer0", "Timer1", "Timer2",
        "Controller", "Sio", "Spu", "Lightpen",
    ];
    let raised = irq.raise_counts();
    print!("irq raises       =");
    for (i, &n) in raised.iter().enumerate() {
        if n > 0 {
            print!(" {}:{n}", raise_names[i]);
        }
    }
    println!();
    println!("peak I_STAT      = 0x{:08x}", irq.peak_stat());
    println!("pending_true     = {} calls", irq.pending_true_calls());
    println!(
        "irq writes       = mask:{} stat:{}",
        irq.mask_write_count(),
        irq.stat_write_count()
    );
    println!("irq.stat() raw   = 0x{:08x}", irq.stat());
    println!("irq.mask() raw   = 0x{:08x}", irq.mask());
    println!(
        "mask write log   = {:?}",
        irq.mask_write_log()
            .iter()
            .map(|v| format!("0x{v:08x}"))
            .collect::<Vec<_>>()
    );
    println!(
        "  SR.IE={} IM.HW={} CAUSE.IP2={} BEV={}",
        cpu.cop0()[12] & 1,
        (cpu.cop0()[12] >> 10) & 1,
        (cpu.cop0()[13] >> 10) & 1,
        (cpu.cop0()[12] >> 22) & 1,
    );
    if let Some((i, e)) = stopped_at {
        println!("stopped at step  = {i}  ({e:?})");
    }

    // Disassemble ±4 instructions around the current PC so we can tell
    // at a glance whether we're in a wait loop and what it's waiting on.
    let gpr_names = [
        "$0 ", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3",
        "$t0", "$t1", "$t2", "$t3", "$t4", "$t5", "$t6", "$t7",
        "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
        "$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra",
    ];
    println!("\n=== GPRs ===");
    for row in 0..4 {
        let mut line = String::new();
        for col in 0..8 {
            let i = row * 8 + col;
            line.push_str(&format!(" {}={:08x}", gpr_names[i], cpu.gprs()[i]));
        }
        println!("{line}");
    }

    // Probes for the BIOS event-counter wait loop at PC ~0x80059dxx.
    // The loop waits for *(0x80079D9C) to reach $a1; both should be
    // visible here so we can tell whether the counter is being
    // incremented by IRQs.
    if let Some(counter) = read32(&bus, 0x80079D9C) {
        println!(
            "wait counter     = *(0x80079D9C) = 0x{counter:08x} ({counter})  threshold $a1 = 0x{:08x} ({})",
            cpu.gprs()[5],
            cpu.gprs()[5]
        );
    }

    println!("\n=== instructions around PC ===");
    let pc = cpu.pc();
    for i in -4i32..=4 {
        let addr = pc.wrapping_add((i * 4) as u32);
        let word = read32(&bus, addr).unwrap_or(0);
        let arrow = if i == 0 { " →" } else { "  " };
        println!("{arrow} {addr:08x}: {word:08x}");
    }

    // Dump every distinct hot region. Group PCs whose gap < 0x100 into
    // one region so the outer/caller + inner/callee of a tight
    // call loop both show up.
    if !top.is_empty() {
        let mut hot_pcs: Vec<u32> = top.iter().take(30).map(|(pc, _)| *pc).collect();
        hot_pcs.sort_unstable();
        let mut regions: Vec<(u32, u32)> = Vec::new();
        for pc in hot_pcs {
            match regions.last_mut() {
                Some((_lo, hi)) if pc.wrapping_sub(*hi) < 0x100 => *hi = pc,
                _ => regions.push((pc, pc)),
            }
        }
        for (lo, hi) in regions {
            let start = lo.wrapping_sub(0x10);
            let end = hi.wrapping_add(0x30);
            println!("\n=== hot region {start:08x}..={end:08x} ===");
            let mut a = start;
            while a <= end {
                let word = read32(&bus, a).unwrap_or(0);
                let hit = pc_hits.get(&a).copied().unwrap_or(0);
                let mark = if hit > 0 { "•" } else { " " };
                println!("  {mark} {a:08x}: {word:08x}  ({hit} hits)");
                a = a.wrapping_add(4);
            }
        }
    }

    // MMIO trace tail — only populated when built with
    //   cargo run --example smoke_draw --features emulator-core/trace-mmio
    // Otherwise `len()` is 0 and this block is silent.
    #[cfg(feature = "trace-mmio")]
    {
        let n_trace = bus.mmio_trace.len();
        if n_trace > 0 {
            println!("\n=== MMIO trace tail (last 60 of {n_trace} recorded) ===");
            let entries: Vec<_> = bus.mmio_trace.iter_chronological().collect();
            let skip = entries.len().saturating_sub(60);
            for e in &entries[skip..] {
                println!(
                    "  cyc={:>12}  {}  {:08x}  {:08x}",
                    e.cycle,
                    e.kind.tag(),
                    e.addr,
                    e.value
                );
            }
            // Hot-address histogram — which MMIO ports is the BIOS
            // hammering? Spinning on one is a tell for a wait loop.
            let mut counts: std::collections::BTreeMap<u32, u32> =
                std::collections::BTreeMap::new();
            for e in &entries {
                *counts.entry(e.addr).or_insert(0) += 1;
            }
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            println!("\n=== MMIO trace top-10 hot addresses ===");
            for (addr, hits) in sorted.iter().take(10) {
                println!("  {addr:08x}: {hits} accesses");
            }
        }
    }

    // Peek kernel RAM addresses via env var:
    //   PSOXIDE_PEEK=0x80079CB4,0x80079D9C
    // Each address is read as a 32-bit word; if it looks like a RAM
    // pointer, the next 48 bytes it points to are also dumped so we
    // can inspect descriptor tables the BIOS is polling.
    if let Ok(list) = std::env::var("PSOXIDE_PEEK") {
        println!("\n=== RAM peek (PSOXIDE_PEEK, via real read path) ===");
        let addrs: Vec<u32> = list
            .split(',')
            .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        // Must use `bus.read32` (not `try_read8`) so MMIO dispatch
        // runs through live peripheral state, not the echo buffer.
        for a in addrs {
            let w = bus.read32(a);
            println!("  *{a:08x} = {w:08x}");
        }
    }

    let vram = &bus.gpu.vram;
    let nz = vram.words().iter().filter(|&&w| w != 0).count();
    println!("\nVRAM non-zero    = {nz} / {} words", 1024 * 512);

    // Dump VRAM if the caller asked via the env var. Keeps the default
    // smoke_draw invocation fast / output-free, but lets us actually
    // see what the BIOS is drawing when we want to.
    if let Ok(path) = std::env::var("PSOXIDE_VRAM_DUMP") {
        match dump_vram_ppm(vram, &path) {
            Ok(()) => println!("VRAM dumped to   = {path}"),
            Err(e) => eprintln!("VRAM dump failed = {e}"),
        }
    }

    if nz > 0 {
        let colors: std::collections::BTreeSet<u16> =
            vram.words().iter().copied().filter(|&w| w != 0).collect();
        println!("unique colors    = {}", colors.len());
        println!(
            "first 10 colors  = {:?}",
            colors
                .iter()
                .take(10)
                .map(|c| format!("0x{c:04X}"))
                .collect::<Vec<_>>()
        );

        // Pixel density heatmap: which 64×64 blocks of VRAM have
        // non-zero content? Gives a rough shape of what's been drawn.
        println!("\n=== VRAM density map (1 char = 64×64 block, # = >10% filled) ===");
        println!("(X runs 0..1024 left→right; Y runs 0..512 top→bottom)");
        for by in 0..8u16 {
            let mut line = String::new();
            for bx in 0..16u16 {
                let mut block_nz = 0u32;
                for dy in 0..64u16 {
                    for dx in 0..64u16 {
                        if vram.get_pixel(bx * 64 + dx, by * 64 + dy) != 0 {
                            block_nz += 1;
                        }
                    }
                }
                // 4096 pixels per block; threshold at 10% / 50% / 90%.
                let c = if block_nz > 64 * 64 * 9 / 10 {
                    '#'
                } else if block_nz > 64 * 64 / 2 {
                    '@'
                } else if block_nz > 64 * 64 / 10 {
                    '.'
                } else if block_nz > 0 {
                    ','
                } else {
                    ' '
                };
                line.push(c);
            }
            println!("{line}");
        }
    }
}

fn read32(bus: &Bus, addr: u32) -> Option<u32> {
    let b0 = bus.try_read8(addr)? as u32;
    let b1 = bus.try_read8(addr.wrapping_add(1))? as u32;
    let b2 = bus.try_read8(addr.wrapping_add(2))? as u32;
    let b3 = bus.try_read8(addr.wrapping_add(3))? as u32;
    Some(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
}

/// Dump full 1024×512 VRAM to a binary PPM at `path`. PPM is header +
/// raw RGB — every image viewer (Preview, feh, `open`, etc.) opens it
/// with no extra deps. 15bpp → 8bpp uses the `(v << 3) | (v >> 2)`
/// expansion to reach full 0..=255 range.
#[allow(dead_code)]
fn dump_vram_ppm(vram: &emulator_core::Vram, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let w = emulator_core::VRAM_WIDTH;
    let h = emulator_core::VRAM_HEIGHT;
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6\n{w} {h}\n255")?;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for &pix in vram.words() {
        let r5 = (pix & 0x1F) as u8;
        let g5 = ((pix >> 5) & 0x1F) as u8;
        let b5 = ((pix >> 10) & 0x1F) as u8;
        rgb.push((r5 << 3) | (r5 >> 2));
        rgb.push((g5 << 3) | (g5 >> 2));
        rgb.push((b5 << 3) | (b5 >> 2));
    }
    file.write_all(&rgb)?;
    Ok(())
}
