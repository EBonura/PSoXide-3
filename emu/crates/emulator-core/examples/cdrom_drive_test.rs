//! Drive the CDROM directly (no BIOS): issue GetStat→Init→SetLoc
//! (LBA 16)→SeekL→ReadN, tick the bus until DataReady fires, drain
//! the data FIFO via the MMIO port 0x1F801802, and verify the
//! first 64 bytes match the PVD.
//!
//! If this fails, the CDROM internals are broken. If it passes,
//! the bug has to be in how the BIOS interacts with CDROM (IRQ
//! ordering, request-register gating, or DMA setup).

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::path::PathBuf;

fn main() {
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    let disc_path = std::env::var("PSOXIDE_DISC").expect("set PSOXIDE_DISC");
    bus.cdrom
        .insert_disc(Some(Disc::from_bin(std::fs::read(&disc_path).unwrap())));
    let _cpu = Cpu::new();

    // Issue commands through the bus by emulating what a BIOS
    // would write at the MMIO port. CDROM:
    //   0x1F801800 -- index / status
    //   0x1F801801 -- command (idx=0) / response FIFO (read)
    //   0x1F801802 -- parameter push (idx=0) / data FIFO (read)
    //   0x1F801803 -- interrupt enable / ack
    //
    // To issue a command: set index=0 (write 0 to port 0x1800),
    // push parameters to 0x1802, then write the opcode to 0x1801.

    let issue = |bus: &mut Bus, opcode: u8, params: &[u8]| {
        bus.write8(0x1F801800, 0); // select index 0
        for &p in params {
            bus.write8(0x1F801802, p); // param FIFO push
        }
        bus.write8(0x1F801801, opcode); // command
    };
    // Tick forward, acking every CDROM IRQ as soon as it latches.
    // The BIOS does this in software; without it, later responses
    // get re-enqueued endlessly because irq_flag never clears.
    //
    // IMPORTANT: the IRQ flag register is only visible at index
    // 1 or 3. Our command-issue path leaves the index at 0 (to
    // address the command + param ports), so we need to switch
    // back to index 1 before touching irq-related registers.
    let tick_and_ack = |bus: &mut Bus, cycles: u64| {
        let target = bus.cycles() + cycles;
        while bus.cycles() < target {
            bus.tick(128);
            bus.write8(0x1F801800, 1); // idx=1 for IRQ flag access
            let irq = bus.read8(0x1F801803);
            if irq & 0x1F != 0 {
                // Drain response FIFO -- the first non-zero-bits
                // status byte tells us there are response bytes.
                while bus.read8(0x1F801800) & 0x20 != 0 {
                    let _ = bus.read8(0x1F801801);
                }
                // Ack (write 1-to-clear at idx=1 port 0x1803).
                bus.write8(0x1F801803, irq & 0x1F);
            }
            bus.write8(0x1F801800, 0); // restore idx=0
        }
    };

    // Enable all CDROM IRQ types so nothing gets masked.
    bus.write8(0x1F801800, 1); // idx=1
    bus.write8(0x1F801802, 0x1F); // IRQ mask = 0x1F (all INT types)
                                  // Ack any pending.
    bus.write8(0x1F801803, 0x1F);

    // 0x0a = Init: spin motor, reset mode.
    issue(&mut bus, 0x0a, &[]);
    tick_and_ack(&mut bus, 2_000_000);

    // 0x02 SetLoc (02, 16, 00) -- LBA 16 in BCD.
    issue(&mut bus, 0x02, &[0x00, 0x02, 0x16]); // FAKE: we're writing BCD minute, second, frame
                                                //                                              ^
                                                // PSX convention: params are (minute, second, frame) in BCD.
                                                // For LBA 16 → 00:02:16 BCD.
    tick_and_ack(&mut bus, 200_000);

    // 0x15 SeekL
    issue(&mut bus, 0x15, &[]);
    tick_and_ack(&mut bus, 800_000);

    // 0x0e SetMode with mode byte 0 (normal, 2048-byte sectors).
    issue(&mut bus, 0x0e, &[0x00]);
    tick_and_ack(&mut bus, 200_000);

    // 0x06 ReadN -- start streaming sectors from LBA 16.
    issue(&mut bus, 0x06, &[]);
    // Wait just long enough for the FIRST sector to land
    // (SECTOR_READ_CYCLES = 225_000 + some slack) without letting
    // the streamer overwrite it with LBA 17's contents.
    tick_and_ack(&mut bus, 300_000);
    // 0x09 Pause -- halt the stream so the FIFO holds LBA 16.
    issue(&mut bus, 0x09, &[]);
    tick_and_ack(&mut bus, 700_000);

    println!(
        "data_fifo_len after ReadN+tick: {}",
        bus.cdrom.data_fifo_len()
    );
    // Drain the FIFO via MMIO reads.
    let mut drained = Vec::with_capacity(2048);
    for _ in 0..2048 {
        drained.push(bus.read8(0x1F801802));
    }
    println!("drained {} bytes", drained.len());
    println!("data_fifo_pops counter: {}", bus.cdrom.data_fifo_pops());
    println!();
    println!("first 256 bytes read from port 0x1F801802:");
    for chunk in drained[..256].chunks(16) {
        let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
        let ascii: String = chunk
            .iter()
            .map(|&b| {
                if (0x20..0x7f).contains(&b) {
                    b as char
                } else {
                    '.'
                }
            })
            .collect();
        println!("  {}  {}", hex.join(" "), ascii);
    }

    // Expected: "01 43 44 30 30 31 01 00 50 4c 41 59 53 54 41 54" (PVD: CD001 PLAYSTAT…)
    if drained[0..6] == [0x01, b'C', b'D', b'0', b'0', b'1'] {
        println!("\n[ok] PVD bytes delivered correctly via MMIO read at offset 0.");
    } else if drained[1..7] == [0x01, b'C', b'D', b'0', b'0', b'1'] {
        println!(
            "\n[SHIFTED BY 1] PVD magic found at offset 1, not 0. byte[0] = 0x{:02x}",
            drained[0]
        );
    } else {
        println!("\n[FAIL] drained bytes do not start with PVD magic 01 'CD001'.");
    }
}
