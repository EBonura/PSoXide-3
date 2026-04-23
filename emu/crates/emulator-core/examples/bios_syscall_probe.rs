//! Log the frequency of each A/B/C BIOS function call while the
//! emulator is running. Used to figure out what the BIOS is
//! spinning on after the PlayStation splash — if one A-function
//! is called millions of times and the disc-read functions never
//! fire, the BIOS is waiting on a syscall-visible state (a
//! counter, a flag) that we're not updating correctly.
//!
//! ```bash
//! PSOXIDE_DISC=path/to/game.bin \
//!   cargo run -p emulator-core --example bios_syscall_probe --release -- 2000000000
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::path::PathBuf;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000_000);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    if let Ok(disc_path) = std::env::var("PSOXIDE_DISC") {
        let disc_bytes = std::fs::read(&disc_path).expect("disc readable");
        bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
        eprintln!("[probe] mounted {disc_path}");
    }
    let mut cpu = Cpu::new();

    // Histograms: [table][function_index] → call count.
    //   table 0 = A-functions @ 0xa0
    //   table 1 = B-functions @ 0xb0
    //   table 2 = C-functions @ 0xc0
    let mut hist: [[u64; 256]; 3] = [[0; 256]; 3];
    // Keep a short recent-call ring so if the BIOS wedges we can
    // print the last few calls to pinpoint the loop.
    let mut recent: std::collections::VecDeque<(u64, u8, u8, u32)> =
        std::collections::VecDeque::with_capacity(64);
    // Capture every `putchar`-ish byte the BIOS emits.
    // A(0x3C), B(0x3C), and B(0x3D) are all stdout-character
    // calls (different kernel versions reach the TTY differently).
    // If the BIOS printed an error before wedging, this is the
    // fastest way to see it.
    let mut putchar_log: String = String::new();
    // Track CDROM SetLoc MSF writes so we can correlate each
    // ReadN with the target LBA. `(step, m, s, f, lba)`.
    let mut cdrom_setloc_log: Vec<(u64, u8, u8, u8, u32)> = Vec::new();
    let mut last_cdrom_cmd_count: u64 = 0;

    for i in 0..n {
        // Before each step, sample pc — dispatch happens when we
        // execute at exactly 0xa0 / 0xb0 / 0xc0 (the J to the
        // table dispatcher). `t1` carries the function number.
        let pc = cpu.pc();
        let table = match pc {
            0xa0 => Some(0u8),
            0xb0 => Some(1u8),
            0xc0 => Some(2u8),
            _ => None,
        };
        if let Some(t) = table {
            let t1 = cpu.gprs()[9] as u8;
            hist[t as usize][t1 as usize] = hist[t as usize][t1 as usize].saturating_add(1);
            if recent.len() >= 64 {
                recent.pop_front();
            }
            recent.push_back((i, t, t1, cpu.gprs()[31]));

            // Putchar capture: argument is in $a0.
            let a0 = cpu.gprs()[4];
            match (t, t1) {
                (0, 0x3D) | (1, 0x3D) => {
                    // Single-char putchar.
                    if a0 < 128 {
                        putchar_log.push(a0 as u8 as char);
                    } else {
                        putchar_log.push_str(&format!("\\x{:02x}", a0 & 0xFF));
                    }
                }
                (1, 0x3F) => {
                    // std_out_puts: $a0 is a C-string pointer.
                    let mut addr = a0;
                    for _ in 0..512 {
                        let ch = bus.try_read8(addr).unwrap_or(0);
                        if ch == 0 {
                            break;
                        }
                        if ch < 128 {
                            putchar_log.push(ch as char);
                        } else {
                            putchar_log.push_str(&format!("\\x{:02x}", ch));
                        }
                        addr = addr.wrapping_add(1);
                    }
                    putchar_log.push('\n');
                }
                _ => {}
            }
        }
        if let Err(e) = cpu.step(&mut bus) {
            eprintln!("[probe] step {i} error: {e:?}");
            break;
        }

        // After each step, check whether a new CDROM command
        // got dispatched. If so, log its last-seen setloc so we
        // know which LBA the BIOS is targeting.
        let c = bus.cdrom.commands_dispatched();
        if c != last_cdrom_cmd_count {
            last_cdrom_cmd_count = c;
            let op = bus.cdrom.last_command();
            if op == 0x02 || op == 0x15 || op == 0x06 {
                // SetLoc / SeekL / ReadN — snapshot current MSF.
                let (m, s, f) = bus.cdrom.debug_setloc_msf();
                let lba = psx_iso::msf_to_lba(m, s, f);
                let tag = match op {
                    0x02 => "SetLoc",
                    0x15 => "SeekL",
                    0x06 => "ReadN",
                    _ => "?",
                };
                cdrom_setloc_log.push((i, m, s, f, lba));
                if tag == "ReadN" {
                    // Don't blow up the log for one run.
                    if cdrom_setloc_log.len() < 100 {
                        eprintln!(
                            "[probe] {tag} step={i} MSF={:02x}:{:02x}:{:02x} → LBA {}",
                            m, s, f, lba
                        );
                    }
                }
            }
        }
    }

    let labels = ["A", "B", "C"];
    println!("=== BIOS syscall histogram @ step {} ===", cpu.tick());
    println!("cycles: {}", bus.cycles());
    println!("final pc: 0x{:08x}", cpu.pc());
    println!();
    for (t, table) in hist.iter().enumerate() {
        let mut pairs: Vec<(u8, u64)> = table
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| (i as u8, c))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        let total: u64 = pairs.iter().map(|(_, c)| c).sum();
        println!("{}-functions ({total} calls):", labels[t]);
        for (op, c) in pairs.iter().take(20) {
            let name = function_name(t as u8, *op);
            println!(
                "  {:>3}%  {:>9}× {}(0x{:02X}) {}",
                100 * c / total.max(1),
                c,
                labels[t],
                op,
                name
            );
        }
        println!();
    }

    if !putchar_log.is_empty() {
        println!("=== BIOS stdout ({} bytes) ===", putchar_log.len());
        // Print each line so newlines / error messages are readable.
        for line in putchar_log.lines() {
            println!("  | {line}");
        }
        println!();
    }

    println!("=== Last 32 syscalls ===");
    for (step, table, fn_no, ra) in recent.iter().rev().take(32) {
        let name = function_name(*table, *fn_no);
        println!(
            "  step {step:>10}  {}(0x{:02X}) {} ra=0x{:08x}",
            labels[*table as usize], fn_no, name, ra
        );
    }
}

/// Canonical names for the BIOS A/B/C function numbers. Pulled
/// from nocash PSX-SPX — not exhaustive, just the common ones so
/// the histogram reads more usefully than raw hex.
fn function_name(table: u8, fn_no: u8) -> &'static str {
    match (table, fn_no) {
        (0, 0x00) => "FileOpen",
        (0, 0x01) => "FileSeek",
        (0, 0x02) => "FileRead",
        (0, 0x03) => "FileWrite",
        (0, 0x04) => "FileClose",
        (0, 0x05) => "FileIoctl",
        (0, 0x06) => "exit",
        (0, 0x13) => "SaveState",
        (0, 0x17) => "strcmp",
        (0, 0x18) => "strncmp",
        (0, 0x19) => "strcpy",
        (0, 0x1A) => "strncpy",
        (0, 0x1B) => "strlen",
        (0, 0x25) => "toupper",
        (0, 0x28) => "bzero",
        (0, 0x2A) => "memcpy",
        (0, 0x2B) => "memset",
        (0, 0x2C) => "memmove",
        (0, 0x2F) => "rand",
        (0, 0x33) => "malloc",
        (0, 0x34) => "free",
        (0, 0x39) => "InitHeap",
        (0, 0x3C) => "std_in_getchar",
        (0, 0x3D) => "std_out_putchar",
        (0, 0x40) => "SystemErrorUnresolvedException",
        (0, 0x44) => "FlushCache",
        (0, 0x47) => "GPU_cw",
        (0, 0x49) => "GPU_cwb",
        (0, 0x54) => "CdInit",
        (0, 0x70) => "_bu_init",
        (0, 0x78) => "CdReadSector",
        (0, 0x96) => "AddCDROMDevice",
        (0, 0xa1) => "SystemError",
        (0, 0xa2) => "EnqueueCdIntr",
        (0, 0xa3) => "DequeueCdIntr",
        (0, 0xa4) => "CdGetLbn",
        (0, 0xa5) => "CdReadSector",
        (0, 0xa6) => "CdGetStatus",
        (0, 0xa7) => "bu_callback_okay",
        (0, 0xa8) => "bu_callback_err_write",
        (0, 0xa9) => "bu_callback_err_busy",
        (0, 0xaa) => "bu_callback_err_eject",
        (0, 0xab) => "_card_info",
        (0, 0xac) => "_card_async_load_directory",
        (0, 0xae) => "_card_status",
        (0, 0xaf) => "_card_wait",

        (1, 0x00) => "alloc_kernel_memory",
        (1, 0x07) => "DeliverEvent",
        (1, 0x08) => "OpenEvent",
        (1, 0x09) => "CloseEvent",
        (1, 0x0A) => "WaitEvent",
        (1, 0x0B) => "TestEvent",
        (1, 0x0C) => "EnableEvent",
        (1, 0x0D) => "DisableEvent",
        (1, 0x0E) => "OpenTh",
        (1, 0x0F) => "CloseTh",
        (1, 0x10) => "ChangeTh",
        (1, 0x12) => "InitPad",
        (1, 0x13) => "StartPad",
        (1, 0x14) => "StopPad",
        (1, 0x17) => "ReturnFromException",
        (1, 0x18) => "SetDefaultExitFromException",
        (1, 0x19) => "SetCustomExitFromException",
        (1, 0x20) => "UnDeliverEvent",
        (1, 0x32) => "FileOpen",
        (1, 0x3C) => "std_in_getchar",
        (1, 0x3D) => "std_out_putchar",
        (1, 0x3F) => "std_out_puts",
        (1, 0x4A) => "InitCard",
        (1, 0x4B) => "StartCard",
        (1, 0x4C) => "StopCard",
        (1, 0x56) => "GetC0Table",
        (1, 0x57) => "GetB0Table",
        (1, 0x5B) => "ChangeClearPad",

        (2, 0x00) => "EnqueueTimerAndVblankIrqs",
        (2, 0x01) => "EnqueueSyscallHandler",
        (2, 0x02) => "SysEnqIntRP",
        (2, 0x03) => "SysDeqIntRP",
        (2, 0x07) => "InstallExceptionHandlers",
        (2, 0x08) => "SysInitMemory",
        (2, 0x09) => "SysInitKernelVariables",
        (2, 0x0A) => "ChangeClearRCnt",
        (2, 0x0C) => "InitDefInt",
        (2, 0x12) => "InstallDevices",
        (2, 0x1A) => "FlushStdInOutPut",

        _ => "?",
    }
}
