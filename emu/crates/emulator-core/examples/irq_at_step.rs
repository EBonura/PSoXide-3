//! Run our CPU to a target step and dump IRQ + COP0 CAUSE state. Used
//! to chase the spurious-IRQ divergence at step 19,258,368: Redux's
//! CAUSE has only the syscall ExcCode, ours additionally has IP[2]
//! set, meaning we have something pending in I_STAT&I_MASK that Redux
//! doesn't.
//!
//! ```bash
//! cargo run -p emulator-core --example irq_at_step --release -- 19258368
//! ```

use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

const SOURCE_NAMES: [&str; 11] = [
    "VBlank", "Gpu", "Cdrom", "Dma", "Timer0", "Timer1", "Timer2", "Controller", "Sio", "Spu",
    "Lightpen",
];

const EXC_NAMES: [&str; 32] = [
    "Int", "Mod", "TLBL", "TLBS", "AdEL", "AdES", "IBE", "DBE", "Sys", "Bp", "RI", "CpU", "Ovf",
    "Tr", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27",
    "28", "29", "30",
];

fn main() {
    let target: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(19_258_368);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    for _ in 0..target {
        cpu.step(&mut bus).expect("step");
    }

    let cop0 = cpu.cop0();
    let cause = cop0[13];
    let sr = cop0[12];
    let epc = cop0[14];
    let irq = bus.irq();

    println!("=== State at step {target} ===");
    println!("cycles      : {}", bus.cycles());
    println!("PC          : 0x{:08x}", cpu.pc());
    println!("SR          : 0x{:08x}  (IEc={}, IM=0x{:02x}, BEV={})",
        sr, sr & 1, (sr >> 8) & 0xFF, (sr >> 22) & 1);
    println!("CAUSE       : 0x{:08x}  (ExcCode={} {}, IP=0x{:02x}, BD={})",
        cause,
        (cause >> 2) & 0x1F,
        EXC_NAMES.get(((cause >> 2) & 0x1F) as usize).unwrap_or(&"?"),
        (cause >> 8) & 0xFF,
        (cause >> 31) & 1);
    println!("EPC         : 0x{:08x}", epc);
    println!();
    println!("I_STAT      : 0x{:08x}", irq.stat());
    println!("I_MASK      : 0x{:08x}", irq.mask());
    println!("I_STAT&MASK : 0x{:08x}", irq.stat() & irq.mask());
    println!("peak I_STAT : 0x{:08x}", irq.peak_stat());
    println!();

    let raise_counts = irq.raise_counts();
    println!("Per-source raise() counts:");
    for (i, count) in raise_counts.iter().enumerate() {
        if *count > 0 {
            let pending = (irq.stat() >> i) & 1;
            let masked = (irq.mask() >> i) & 1;
            println!(
                "  [{}] {:<10} raised={:>10}  pending={}  enabled={}",
                i, SOURCE_NAMES[i], count, pending, masked,
            );
        }
    }
    println!();

    println!("Exception counts (taken):");
    for (i, count) in cpu.exception_counts().iter().enumerate() {
        if *count > 0 {
            println!("  [{:>2}] {:<6} {}", i, EXC_NAMES.get(i).unwrap_or(&"?"), count);
        }
    }
    println!();

    println!("Diagnostics:");
    println!("  irq_line_high_steps      = {}", cpu.irq_line_high_steps());
    println!("  should_take_interrupt_steps = {}", cpu.should_take_interrupt_steps());
    println!("  pending_true_calls       = {}", irq.pending_true_calls());
    println!("  mask_write_count         = {}", irq.mask_write_count());
    println!("  stat_write_count         = {}", irq.stat_write_count());
    println!("  mask_write_log (first 16):");
    for (i, m) in irq.mask_write_log().iter().enumerate() {
        println!("    {i:>2}: 0x{m:08x}");
    }
    println!("  mask_write_events (cycle, value):");
    for (i, (c, v)) in irq.mask_write_events().iter().enumerate() {
        println!("    {i:>2}: cyc={c:>10}  value=0x{v:08x}");
    }
    println!("  stat_write_events (cycle, value):");
    for (i, (c, v)) in irq.stat_write_events().iter().enumerate() {
        println!("    {i:>2}: cyc={c:>10}  value=0x{v:08x}");
    }
}
