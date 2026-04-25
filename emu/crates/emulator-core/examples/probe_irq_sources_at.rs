//! Run a retail disc to a folded-step count and dump the exact state of
//! every IRQ source we currently model: I_STAT/I_MASK, scheduler-backed
//! DMA/VBlank targets, CDROM pending event, and SIO timing. Used to
//! identify which source could plausibly fire a hidden Redux IRQ at a
//! given step.

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::scheduler::EventSlot;
use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

fn main() {
    let step: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_irq_sources_at <step> <disc.bin>");
    let disc_path = std::env::args()
        .nth(2)
        .expect("usage: probe_irq_sources_at <step> <disc.bin>");

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let disc = disc_support::load_disc_path(&PathBuf::from(disc_path)).expect("disc readable");
    bus.cdrom.insert_disc(Some(disc));
    let mut cpu = Cpu::new();

    for _ in 0..step {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    println!("step            : {step}");
    println!("cycles          : {}", bus.cycles());
    println!("pc              : 0x{:08x}", cpu.pc());
    println!(
        "istat / imask   : 0x{:03x} / 0x{:03x}",
        bus.irq().stat(),
        bus.irq().mask()
    );
    println!(
        "next_vblank     : {}",
        delta_target(bus.next_vblank_cycle(), bus.cycles())
    );

    for (name, slot) in [
        ("mdec_in", EventSlot::MdecInDma),
        ("mdec_out", EventSlot::MdecOutDma),
        ("gpu_dma", EventSlot::GpuDma),
        ("cdr_dma", EventSlot::CdrDma),
        ("spu_dma", EventSlot::SpuDma),
        ("gpu_otc", EventSlot::GpuOtcDma),
    ] {
        let target = bus.scheduler.target(slot);
        match target {
            Some(t) => println!("{name:16}: {}", delta_target(t, bus.cycles())),
            None => println!("{name:16}: none"),
        }
    }

    match bus.cdrom.next_pending_event() {
        Some((deadline, irq)) => {
            println!(
                "cdrom_pending   : {:?} {}",
                irq,
                delta_target(deadline, bus.cycles())
            );
        }
        None => println!("cdrom_pending   : none"),
    }

    let sio = bus.sio0();
    println!("sio_stat        : 0x{:08x}", sio.debug_stat());
    println!("sio_ctrl        : 0x{:04x}", sio.debug_ctrl());
    println!("sio_pending_irq : {}", sio.debug_pending_irq());
    println!("sio_irq_latched : {}", sio.debug_irq_latched());
    println!(
        "sio_transfer    : {}",
        opt_delta_target(sio.debug_transfer_deadline(), bus.cycles())
    );
    println!(
        "sio_ack_start   : {}",
        opt_delta_target(sio.debug_ack_deadline(), bus.cycles())
    );
    println!(
        "sio_ack_end     : {}",
        opt_delta_target(sio.debug_ack_end_deadline(), bus.cycles())
    );
}

fn delta_target(target: u64, now: u64) -> String {
    format!("{target} (in {})", target as i64 - now as i64)
}

fn opt_delta_target(target: Option<u64>, now: u64) -> String {
    match target {
        Some(t) => delta_target(t, now),
        None => "none".to_string(),
    }
}
