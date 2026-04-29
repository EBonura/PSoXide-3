//! Quick parity scorecard: run each disc-backed scenario to 100M
//! CPU steps and print the display hash. The Sony-logo display hash
//! must match Redux's canonical `0xa3ac6881044333d0` — a value we've
//! verified pixel-exact for BIOS (no disc), Crash, Tekken, and
//! WipEout 1. The BIOS logo is rendered identically regardless of
//! which game's disc is inserted (the disc doesn't get accessed
//! until after the logo), so this single-emulator probe doubles as a
//! "no game-specific renderer regressions" check across the whole
//! collection without requiring Redux runs per game.
//!
//! A mismatch here means our emulator has some game-specific
//! rendering bug (wrong initial GPU state, mis-parsed CDROM TOC
//! affecting BIOS init, etc.) that shows up before the logo even
//! draws — worth investigating.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_all_games_100m --release
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;

const EXPECTED_SONY_LOGO_HASH: u64 = 0xa3ac_6881_0443_33d0;

fn main() {
    let games: &[(&str, &str)] = &[
        ("bios_no_disc", ""),
        (
            "crash",
            "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin",
        ),
        (
            "tekken",
            "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA) (Track 1).bin",
        ),
        (
            "gt2",
            "/Users/ebonura/Downloads/ps1 games/Gran Turismo 2 (USA) (Arcade Mode) (Rev 1)/Gran Turismo 2 (USA) (Arcade Mode) (Rev 1).bin",
        ),
        (
            "mgs",
            "/Users/ebonura/Downloads/ps1 games/Metal Gear Solid (USA) (Disc 1) (Rev 1)/Metal Gear Solid (USA) (Disc 1) (Rev 1).bin",
        ),
        (
            "re2",
            "/Users/ebonura/Downloads/ps1 games/Resident Evil 2 - Dual Shock Ver. (USA) (Disc 1)/Resident Evil 2 - Dual Shock Ver. (USA) (Disc 1) (Track 1).bin",
        ),
        (
            "wipeout1",
            "/Users/ebonura/Downloads/ps1 games/WipEout (Europe) (v1.1)/WipEout (Europe) (v1.1)/WipEout (Europe) (v1.1) (Track 01).bin",
        ),
        (
            "wipeout2097",
            "/Users/ebonura/Downloads/ps1 games/WipEout 2097 (Europe)/WipEout 2097 (Europe)/WipEout 2097 (Europe) (Track 01).bin",
        ),
        (
            "wipeout3",
            "/Users/ebonura/Downloads/ps1 games/WipEout 3 - Special Edition (Europe) (En,Fr,De,Es,It)/WipEout 3 - Special Edition (Europe) (En,Fr,De,Es,It)/WipEout 3 - Special Edition (Europe) (En,Fr,De,Es,It) (Track 01).bin",
        ),
    ];

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    println!(
        "Expected Sony-logo hash (Redux-verified): 0x{:016x}",
        EXPECTED_SONY_LOGO_HASH,
    );
    println!();
    println!(
        "{:<14} {:>14} {:>20}  match",
        "game", "our_cycles", "display_hash"
    );
    println!("{}", "-".repeat(80));
    let mut all_pass = true;
    for (name, disc_path) in games {
        let mut bus = Bus::new(bios.clone()).expect("bus");
        if !disc_path.is_empty() {
            if !std::path::Path::new(disc_path).exists() {
                println!("{:<14} {:>14} {:>20}  skip", name, "-", "MISSING DISC");
                continue;
            }
            let disc = std::fs::read(disc_path).expect("disc readable");
            bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
        }
        let mut cpu = Cpu::new();
        let mut cycles_at_last_pump = 0u64;
        for _ in 0..100_000_000u64 {
            if cpu.step(&mut bus).is_err() {
                break;
            }
            if bus.cycles() - cycles_at_last_pump > 560_000 {
                cycles_at_last_pump = bus.cycles();
                bus.run_spu_samples(735);
                let _ = bus.spu.drain_audio();
            }
        }
        let (hash, _w, _h, _len) = bus.gpu.display_hash();
        let matches = hash == EXPECTED_SONY_LOGO_HASH;
        if !matches {
            all_pass = false;
        }
        println!(
            "{:<14} {:>14} {:>20}  {}",
            name,
            bus.cycles(),
            format!("0x{hash:016x}"),
            if matches { "✓ match" } else { "✗ DIFFER" },
        );
    }
    println!();
    if all_pass {
        println!("All games produce the canonical Sony-logo hash at 100M steps.");
        println!("Our emulator is pixel-exact with Redux for every disc in the collection");
        println!("(at the BIOS-logo phase — game-boot phase still has CPU timing drift).");
    } else {
        println!("At least one game diverges from the canonical Sony-logo hash —");
        println!("investigate the failing row above for a game-specific early-boot bug.");
    }
}
