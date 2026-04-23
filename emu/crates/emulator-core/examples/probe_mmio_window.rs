use emulator_core::{Bus, Cpu};

#[cfg(feature = "trace-mmio")]
use emulator_core::MmioKind;

fn main() {
    let from_step: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: probe_mmio_window <from_step> [count] [disc]");
    let count: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let disc_path = std::env::args().nth(3);

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc = std::fs::read(p).expect("disc");
        bus.cdrom.insert_disc(Some(psx_iso::Disc::from_bin(disc)));
    }
    let mut cpu = Cpu::new();

    for _ in 0..from_step.saturating_sub(1) {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    for i in 0..count {
        let step_n = from_step + i;
        let pc = cpu.pc();
        let cyc = bus.cycles();
        cpu.step(&mut bus).expect("raw step");
        println!(
            "step={step_n:>10} cyc={cyc:>12}->{:>12} pc=0x{pc:08x} in_irq={}",
            bus.cycles(),
            cpu.in_irq_handler()
        );
    }

    #[cfg(feature = "trace-mmio")]
    {
        println!();
        println!("=== MMIO trace tail ===");
        for entry in bus.mmio_trace.iter_chronological() {
            if !is_interest(entry.addr) {
                continue;
            }
            println!(
                "cyc={:>12}  {:<3}  addr=0x{:08x}  value=0x{:08x}",
                entry.cycle,
                match entry.kind {
                    MmioKind::R8 => "r8",
                    MmioKind::R16 => "r16",
                    MmioKind::R32 => "r32",
                    MmioKind::W8 => "w8",
                    MmioKind::W16 => "w16",
                    MmioKind::W32 => "w32",
                },
                entry.addr,
                entry.value,
            );
        }
    }

    #[cfg(not(feature = "trace-mmio"))]
    println!("MMIO trace unavailable; rebuild with --features emulator-core/trace-mmio");
}

#[cfg(feature = "trace-mmio")]
fn is_interest(addr: u32) -> bool {
    (0x1f80_1800..=0x1f80_1803).contains(&addr) || (0x1f80_1070..=0x1f80_1074).contains(&addr)
}
