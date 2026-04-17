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

    let mut stopped_at: Option<(u64, emulator_core::ExecutionError)> = None;
    for i in 0..n {
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
        'outer: for y in 0..512u16 {
            for x in 0..1024u16 {
                let p = vram.get_pixel(x, y);
                if p != 0 {
                    println!("first nz px      = ({x},{y}) = 0x{p:04x}");
                    break 'outer;
                }
            }
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
