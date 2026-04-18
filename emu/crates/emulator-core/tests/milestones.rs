//! Canary milestone regression tests.
//!
//! Each test runs the real BIOS for a fixed instruction count and
//! asserts that VRAM hashes to a known-good value. The hashes were
//! captured from a passing run and frozen as goldens — any code
//! change that alters what the BIOS draws at the same step count
//! breaks the test.
//!
//! These tests are `#[ignore]` by default because they take several
//! seconds each (500M instructions ≈ 11s in release). Run via:
//!
//! ```bash
//! cargo test -p emulator-core --release --test milestones -- --ignored
//! ```
//!
//! The hashes are FNV-1a-64 over the raw little-endian bytes of VRAM
//! (1024×512 × 2 bytes = 1 MiB). FNV-1a isn't cryptographic but
//! collisions by accident are vanishingly unlikely for our purpose
//! — any real regression will flip many VRAM bytes and produce a
//! completely different hash.
//!
//! Milestone ladder reference:
//! - A = BIOS boots to Sony logo (SCPH1001, no disc)
//! - B = BIOS boots to shell (MAIN MENU / MEMORY CARD / CD PLAYER)
//! - C = Homebrew SDK triangle renders (see `sdk/examples/hello-tri`)
//! - D = BIOS disc-check passes (Crash Bandicoot) — next rung
//!
//! When a new milestone passes for the first time, capture its hash
//! with:
//! ```bash
//! cargo run -p emulator-core --example vram_hash_at --release -- <steps>
//! ```
//! and paste the value here.

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::path::PathBuf;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const CRASH_DISC: &str = "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin";

fn bios_path() -> PathBuf {
    std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

/// Run the BIOS for `steps` instructions and return the FNV-1a-64
/// hash of the resulting VRAM. Panics on any CPU step error — the
/// BIOS should never encounter an undecoded opcode in a passing
/// canary, and the milestone goldens are captured assuming clean
/// completion.
fn run_and_hash_vram(steps: u64) -> u64 {
    run_and_hash_vram_with_disc(steps, None)
}

fn run_and_hash_vram_with_disc(steps: u64, disc_path: Option<&str>) -> u64 {
    let bios = std::fs::read(bios_path()).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(path) = disc_path {
        let disc_bytes = std::fs::read(path).expect("disc readable");
        bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();
    for _ in 0..steps {
        cpu.step(&mut bus).expect("CPU step failed");
    }
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &w in bus.gpu.vram.words() {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01B3);
        }
    }
    h
}

#[test]
#[ignore = "long-running (~2s); run via `cargo test --test milestones -- --ignored`"]
fn milestone_a_bios_to_sony_logo() {
    // After 100M instructions the BIOS has rendered the iconic
    // "SONY / COMPUTER ENTERTAINMENT" diamond logo onto VRAM,
    // with the double-buffer swap laid out correctly. Captured
    // 2026-04-18, updated after the CDROM response-ordering fix
    // (second response was firing before first, shifting the
    // boot-time command sequence slightly; the displayed logo is
    // identical, only off-screen CLUT regions differ).
    let hash = run_and_hash_vram(100_000_000);
    assert_eq!(
        hash, 0x97d2_2145_75f2_99d4,
        "Milestone A regression: VRAM hash changed. \
         Inspect with `cargo run --example vram_hash_at --release -- 100000000` \
         and `smoke_draw` to see what diverged."
    );
}

#[test]
#[ignore = "long-running (~11s); run via `cargo test --test milestones -- --ignored`"]
fn milestone_b_bios_to_shell() {
    // After 500M instructions the BIOS has transitioned from the
    // boot logo to the MAIN MENU shell screen with MEMORY CARD and
    // CD PLAYER options on a radial-gradient blue background —
    // the "no disc" path of the BIOS. Captured 2026-04-18.
    let hash = run_and_hash_vram(500_000_000);
    assert_eq!(
        hash, 0x0f00_2542_a50c_0dd0,
        "Milestone B regression: VRAM hash changed. \
         Inspect with `cargo run --example vram_hash_at --release -- 500000000` \
         and `smoke_draw` to see what diverged."
    );
}

#[test]
#[ignore = "requires Crash Bandicoot USA disc + long-running (~11s)"]
fn milestone_d_bios_accepts_licensed_disc() {
    // After 600M instructions with Crash Bandicoot USA mounted, the
    // BIOS has detected the licensed disc (via GetID returning
    // "SCEA" + disc-type 0x20), issued the cold-boot disc-read
    // sequence (SetLoc/SeekL/SetMode/ReadN/Pause x4) from ROM code,
    // cleared the boot-logo VRAM, and rendered the "SONY /
    // PlayStation\u2122" licensed-disc splash on a black background. This
    // is the screen a real PSX shows *between* the Sony Computer
    // Entertainment logo and the game's own intro. Captured
    // 2026-04-18 after the CDROM response-ordering + Pause-stops-read
    // fixes.
    //
    // The BIOS then idles at PC=0x000000a4 (A-function syscall
    // dispatcher) waiting for work that doesn't fire yet — the next
    // rung past D requires SYSTEM.CNF parsing and boot-EXE load to
    // advance further.
    if !std::path::Path::new(CRASH_DISC).exists() {
        eprintln!("skip milestone_d: Crash disc not found at {CRASH_DISC}");
        return;
    }
    let hash = run_and_hash_vram_with_disc(600_000_000, Some(CRASH_DISC));
    assert_eq!(
        hash, 0x31b1_c1ff_5e22_3daf,
        "Milestone D regression: VRAM hash changed. \
         Inspect with `PSOXIDE_DISC=/path/to/Crash.bin \
         cargo run --example boot_disc --release -- 600000000` \
         and view /tmp/crash_600m.ppm."
    );
}
