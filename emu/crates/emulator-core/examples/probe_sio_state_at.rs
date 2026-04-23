//! Run a disc to step N and dump the SIO0/controller timing state.
//! Useful when commercial games poll BIOS events that Redux wakes via
//! controller ACK IRQs but our side misses.

use emulator_core::{Bus, Cpu};

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);
    let disc_path = std::env::args().nth(2);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc readable");
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

    let sio = bus.sio0();
    println!("=== SIO0 state at step {n} (cycles={}) ===", bus.cycles());
    println!("pc               : 0x{:08x}", cpu.pc());
    println!(
        "istat/imask      : 0x{:03x} / 0x{:03x}",
        bus.irq().stat(),
        bus.irq().mask()
    );
    println!("stat             : 0x{:08x}", sio.debug_stat());
    println!("ctrl             : 0x{:04x}", sio.debug_ctrl());
    println!("pending_irq      : {}", sio.debug_pending_irq());
    println!("irq_latched      : {}", sio.debug_irq_latched());
    println!("transfer_busy    : {}", sio.debug_transfer_busy());
    println!("awaiting_ack     : {}", sio.debug_awaiting_ack());
    println!("rx               : {:?}", sio.debug_rx());
    println!("queued_tx        : {:?}", sio.debug_queued_tx());
    println!("transfer_deadline: {:?}", sio.debug_transfer_deadline());
    println!("ack_deadline     : {:?}", sio.debug_ack_deadline());
    println!("ack_end_deadline : {:?}", sio.debug_ack_end_deadline());
}
