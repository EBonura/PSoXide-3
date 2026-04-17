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
    let mut pc_hits: std::collections::BTreeMap<u32, u32> =
        std::collections::BTreeMap::new();

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
    println!("\n=== instructions around PC ===");
    let pc = cpu.pc();
    for i in -4i32..=4 {
        let addr = pc.wrapping_add((i * 4) as u32);
        let word = read32(&bus, addr).unwrap_or(0);
        let arrow = if i == 0 { " →" } else { "  " };
        println!("{arrow} {addr:08x}: {word:08x}");
    }

    let vram = &bus.gpu.vram;
    let nz = vram.words().iter().filter(|&&w| w != 0).count();
    println!("\nVRAM non-zero    = {nz} / {} words", 1024 * 512);

    if nz > 0 {
        let colors: std::collections::BTreeSet<u16> =
            vram.words().iter().copied().filter(|&w| w != 0).collect();
        println!("unique colors    = {}", colors.len());
        println!(
            "first 10 colors  = {:?}",
            colors.iter().take(10).map(|c| format!("0x{c:04X}")).collect::<Vec<_>>()
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
