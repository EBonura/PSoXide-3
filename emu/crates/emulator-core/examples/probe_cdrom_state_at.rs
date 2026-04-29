//! Run our CDROM to step N, dump its state: current LBA, mode,
//! pending events. Used to cross-check against Redux when sector
//! data differs between them.

use emulator_core::{Bus, Cpu};

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(90_146_543);
    let disc_path = std::env::args().nth(2);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.enable_irq_log(200);
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();
    for _ in 0..n {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    println!("=== CDROM state at step {n} (cycles={}) ===", bus.cycles());
    println!("irq_flag         : {}", bus.cdrom.irq_flag());
    println!("index            : {}", bus.cdrom.index_value());
    println!("irq_mask         : 0x{:02x}", bus.cdrom.irq_mask_value());
    println!("data_fifo_pops   : {}", bus.cdrom.data_fifo_pops());
    println!("sector_events    : {}", bus.cdrom.sector_events_scheduled);
    println!("pending_queue_len: {}", bus.cdrom.pending_queue_len());
    if let Some((deadline, irq)) = bus.cdrom.next_pending_event() {
        println!(
            "next_pending     : {:?} at {deadline} (in {})",
            irq,
            deadline as i64 - bus.cycles() as i64
        );
    }
    println!("irq_stat         : 0x{:03x}", bus.irq().stat());
    println!("irq_mask         : 0x{:03x}", bus.irq().mask());

    let irq_names = [
        "VBlank",
        "Gpu",
        "Cdrom",
        "Dma",
        "Timer0",
        "Timer1",
        "Timer2",
        "Controller",
        "Sio",
        "Spu",
        "Lightpen",
    ];
    println!();
    println!("irq raises:");
    for (name, count) in irq_names.iter().zip(bus.irq().raise_counts()) {
        if count > 0 {
            println!("  {name:<10} {count}");
        }
    }

    println!();
    println!("scheduler:");
    use emulator_core::scheduler::EventSlot;
    for slot in [
        EventSlot::Sio,
        EventSlot::Sio1,
        EventSlot::Cdr,
        EventSlot::CdRead,
        EventSlot::GpuDma,
        EventSlot::MdecOutDma,
        EventSlot::SpuDma,
        EventSlot::MdecInDma,
        EventSlot::GpuOtcDma,
        EventSlot::CdrDma,
        EventSlot::CdrPlay,
        EventSlot::CdrDbuf,
        EventSlot::CdrLid,
        EventSlot::SpuAsync,
        EventSlot::VBlank,
    ] {
        if let Some(target) = bus.scheduler.target(slot) {
            println!(
                "  {slot:?}: target={target} in {}",
                target as i64 - bus.cycles() as i64
            );
        }
    }

    println!();
    println!(
        "=== CDROM IRQ log ({} entries) ===",
        bus.cdrom.cdrom_irq_log.len()
    );
    let names = [
        "None",
        "DataReady",
        "Complete",
        "Acknowledge",
        "DataEnd",
        "Error",
    ];
    for (i, &(cyc, ty)) in bus.cdrom.cdrom_irq_log.iter().enumerate() {
        let name = names.get(ty as usize).copied().unwrap_or("?");
        println!("  #{i:>3} cyc={cyc:>12} type={ty} ({name})");
    }
    let cmd_hist = bus.cdrom.command_histogram();
    println!();
    println!("command histogram:");
    for (i, c) in cmd_hist.iter().enumerate().take(32) {
        if *c > 0 {
            println!("  0x{i:02X}: {c}");
        }
    }
}
