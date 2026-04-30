//! Audit DMA3 (CDROM → RAM) behaviour. After the BIOS has
//! dispatched its LBA-16 ReadN we search RAM for the "CD001" PVD
//! magic -- if the full PVD landed in RAM, the bug is downstream
//! (e.g. BIOS parse logic we don't mimic). If the magic is
//! missing or the root-directory-LBA field is wrong, DMA3 itself
//! is corrupting data.
//!
//! ```bash
//! PSOXIDE_DISC=path/to/game.bin \
//!   cargo run -p emulator-core --example dma3_audit --release
//! ```

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
    let mut cpu = Cpu::new();

    // Run until the 3rd ReadN finishes (PVD @ LBA 16). From the
    // earlier probe we know that's at step ~140M. Run past it so
    // DMA3 has had time to complete.
    let target_step = 145_000_000u64;
    for i in 0..target_step {
        if cpu.step(&mut bus).is_err() {
            eprintln!("[audit] step {i} errored");
            break;
        }
    }
    println!("[audit] ran to step {}", cpu.tick());

    // Scan the entire 2 MiB RAM for "CD001" (ASCII).
    let mut found: Vec<u32> = Vec::new();
    for base in 0x80000000u32..0x80200000u32 {
        // Cheap first-byte check via `try_read8`, then full string.
        if bus.try_read8(base) != Some(b'C') {
            continue;
        }
        if bus.try_read8(base + 1) != Some(b'D') {
            continue;
        }
        if bus.try_read8(base + 2) != Some(b'0') {
            continue;
        }
        if bus.try_read8(base + 3) != Some(b'0') {
            continue;
        }
        if bus.try_read8(base + 4) != Some(b'1') {
            continue;
        }
        found.push(base);
        if found.len() > 4 {
            break;
        }
    }

    println!("[audit] found {} 'CD001' sites in RAM:", found.len());
    for &addr in &found {
        println!();
        println!("=== RAM @ 0x{addr:08x} ===");
        // The PVD's type byte sits 1 byte before "CD001" -- so the
        // PVD *block* starts at addr - 1.
        let pvd_base = addr.wrapping_sub(1);
        let ty = bus.try_read8(pvd_base).unwrap_or(0xff);
        println!("  PVD type byte: 0x{ty:02x} (expect 0x01)");
        // system_identifier at +8
        let sys_id: String = (0..32)
            .map(|i| bus.try_read8(pvd_base + 8 + i).unwrap_or(0))
            .map(|b| {
                if (0x20..0x7f).contains(&b) {
                    b as char
                } else {
                    '?'
                }
            })
            .collect();
        println!("  system_id:     '{}'", sys_id.trim_end());
        // volume_identifier at +40
        let vol_id: String = (0..32)
            .map(|i| bus.try_read8(pvd_base + 40 + i).unwrap_or(0))
            .map(|b| {
                if (0x20..0x7f).contains(&b) {
                    b as char
                } else {
                    '?'
                }
            })
            .collect();
        println!("  volume_id:     '{}'", vol_id.trim_end());
        // Root-dir record at +156.
        let rec_len = bus.try_read8(pvd_base + 156).unwrap_or(0);
        let lba_le_0 = bus.try_read8(pvd_base + 158).unwrap_or(0) as u32;
        let lba_le_1 = bus.try_read8(pvd_base + 159).unwrap_or(0) as u32;
        let lba_le_2 = bus.try_read8(pvd_base + 160).unwrap_or(0) as u32;
        let lba_le_3 = bus.try_read8(pvd_base + 161).unwrap_or(0) as u32;
        let lba_le = lba_le_0 | (lba_le_1 << 8) | (lba_le_2 << 16) | (lba_le_3 << 24);
        println!("  rec_len:       {rec_len}");
        println!("  root-dir LBA:  {lba_le}  (expect 22 for Crash)");
        // Also raw hex of the first 32 bytes.
        println!("  first 32 bytes:");
        for row in 0..2 {
            let bytes: Vec<u8> = (0..16)
                .map(|i| bus.try_read8(pvd_base + (row * 16) + i).unwrap_or(0))
                .collect();
            let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
            println!("    {}", hex.join(" "));
        }
    }

    if found.is_empty() {
        println!("[audit] NO 'CD001' magic in RAM — PVD didn't land.");
        println!("[audit] Diagnostic state:");
        let chcr = bus.read32(0x1F801088);
        let bcr = bus.read32(0x1F801084);
        let madr = bus.read32(0x1F801080);
        println!("  DMA3  MADR=0x{madr:08x}  BCR=0x{bcr:08x}  CHCR=0x{chcr:08x}",);
        println!(
            "  CDROM data_fifo_pops = {}  (each pop = 1 MMIO byte-read \
             from 0x1F801802; a full sector is 2048)",
            bus.cdrom.data_fifo_pops()
        );
        println!(
            "  CDROM commands_dispatched = {}",
            bus.cdrom.commands_dispatched()
        );
        println!("  DMA per-channel CHCR-start triggers:");
        for (ch, &c) in bus.dma_start_triggers().iter().enumerate() {
            if c > 0 {
                let name = match ch {
                    0 => "MDECin",
                    1 => "MDECout",
                    2 => "GPU",
                    3 => "CDROM",
                    4 => "SPU",
                    5 => "PIO",
                    6 => "OTC",
                    _ => "?",
                };
                println!("    ch{ch} {name}: {c}");
            }
        }
    }
}
