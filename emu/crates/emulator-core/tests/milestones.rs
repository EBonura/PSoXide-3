//! Canary milestone regression tests.
//!
//! Each test runs the real BIOS for a fixed instruction count and
//! asserts that the result matches a known-good state. We check
//! three hashes per milestone to give us both correctness and
//! determinism coverage:
//!
//! 1. **Full-VRAM hash** (self-regression) — FNV-1a-64 over the
//!    1 MiB VRAM buffer. Catches any change in VRAM bytes, even
//!    off-screen CLUT/texture regions the user never sees. This
//!    pins our emulator to a bit-exact behavior run-to-run so any
//!    code change that alters rendering fails the test. **Does not
//!    validate against Redux** — if we're wrong in some consistent
//!    way, this test will still pass.
//!
//! 2. **Display-area hash** (self-regression) — FNV-1a-64 over the
//!    visible-display rectangle only (`GPU::display_hash`). Same
//!    caveat: self-consistent, not Redux-correct.
//!
//! 3. **Redux-verified display hash** (correctness) — where
//!    captured, the expected display-area hash that Redux produces
//!    at the same step count. If `redux_display_hash` is `Some(h)`
//!    and our `display_hash` doesn't equal it, the test fails and
//!    we know pixels diverged from reference. If `None`, we haven't
//!    captured Redux parity yet for that milestone.
//!
//! Capture a Redux-verified golden with:
//! ```bash
//! cargo run -p emulator-core --example display_parity_at --release -- <steps>
//! ```
//! which launches Redux, steps N instructions silently, reads its
//! screenshot, and diffs byte-for-byte against ours.
//!
//! Tests are `#[ignore]` by default because they take ~13s total
//! in release (the full-VRAM runs are fast; the Redux-parity step
//! is only triggered by `display_parity_at`, not by the unit test).
//! Run via:
//! ```bash
//! cargo test -p emulator-core --release --test milestones -- --ignored
//! ```
//!
//! Milestone ladder reference:
//! - A = BIOS boots to Sony logo (SCPH1001, no disc)
//! - B = BIOS boots to shell (MAIN MENU / MEMORY CARD / CD PLAYER)
//! - C = Homebrew SDK triangle renders (see `sdk/examples/hello-tri`)
//! - D = BIOS disc-check passes (Crash Bandicoot) — licensed-disc
//!       splash rendered; game boot-EXE load is still pending.

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::path::PathBuf;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const CRASH_DISC: &str = "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin";
const TEKKEN_DISC: &str =
    "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA) (Track 1).bin";

fn bios_path() -> PathBuf {
    std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

/// Aggregate state captured at a milestone step. `vram_hash` is the
/// full 1 MiB hash (self-regression); `display_hash` is the
/// visible-display-rect hash (comparable byte-for-byte to Redux's
/// `PCSX.GPU.takeScreenShot`).
struct MilestoneState {
    vram_hash: u64,
    display_hash: u64,
    display_width: u32,
    display_height: u32,
}

fn run_milestone(steps: u64, disc_path: Option<&str>) -> MilestoneState {
    let bios = std::fs::read(bios_path()).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(path) = disc_path {
        let disc_bytes = std::fs::read(path).expect("disc readable");
        bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();
    // Tolerate CPU step errors so later milestones (game-code
    // paths) can still hash whatever state landed before we hit
    // an incomplete-emulation edge. Earlier milestones (A, B) are
    // pure BIOS and never trip this; D+ will until GTE / SPU /
    // MDEC land.
    for _ in 0..steps {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }

    // Full-VRAM FNV-1a-64.
    let mut vh = 0xCBF2_9CE4_8422_2325u64;
    for &w in bus.gpu.vram.words() {
        for b in w.to_le_bytes() {
            vh ^= b as u64;
            vh = vh.wrapping_mul(0x0100_0000_01B3);
        }
    }
    // Display-area FNV-1a-64 (matches Redux's takeScreenShot path).
    let (dh, dw, dhi, _dlen) = bus.gpu.display_hash();
    MilestoneState {
        vram_hash: vh,
        display_hash: dh,
        display_width: dw,
        display_height: dhi,
    }
}

/// Assert our state matches the frozen goldens. `redux_display_hash`
/// carries the Redux-verified correctness check if we've captured
/// one — when `Some`, a mismatch means we diverge from Redux at
/// the pixel level (a real bug); when `None`, we haven't captured
/// Redux parity yet.
fn assert_milestone(
    name: &str,
    state: &MilestoneState,
    expected_vram_hash: u64,
    expected_display_hash: u64,
    expected_display_size: (u32, u32),
    redux_display_hash: Option<u64>,
) {
    assert_eq!(
        (state.display_width, state.display_height),
        expected_display_size,
        "{name}: display dimensions changed — width/height from V-range or mode-bit differ",
    );
    assert_eq!(
        state.display_hash, expected_display_hash,
        "{name}: display-area hash changed. \
         Inspect with `vram_hash_at` and `smoke_draw`.",
    );
    assert_eq!(
        state.vram_hash, expected_vram_hash,
        "{name}: full-VRAM hash changed (off-screen VRAM differs). \
         Display may still look right — check display_hash first.",
    );
    if let Some(expected) = redux_display_hash {
        assert_eq!(
            state.display_hash, expected,
            "{name}: display hash doesn't match Redux's at the same step count \
             — we're rendering the wrong pixels. \
             Compare with `cargo run --example display_parity_at --release -- <steps>`.",
        );
    }
}

#[test]
#[ignore = "long-running (~2s)"]
fn milestone_a_bios_to_sony_logo() {
    // After 100M instructions the BIOS has rendered the iconic
    // "SONY / COMPUTER ENTERTAINMENT" diamond logo onto VRAM
    // (640×478 visible display area in NTSC 480-interlaced mode).
    //
    // Current Redux parity: ~3.21% of display-area bytes differ
    // from Redux at this step — concentrated in the diamond's
    // gradient shading and the "TM" text region, where our
    // rasterizer / semi-transparency / dither paths diverge. The
    // `redux_display_hash` field is left `None` until we fix those;
    // when pixel parity hits 0% differing bytes, replace with the
    // captured Redux hash to lock in correctness.
    let state = run_milestone(100_000_000, None);
    assert_milestone(
        "Milestone A",
        &state,
        0x97d2_2145_75f2_99d4, // full VRAM (self)
        0x2035_49d0_b4b8_5eb6, // display area (self)
        (640, 478),
        None, // Redux-parity hash pending renderer fixes (~3.21% pixel diff)
    );
}

#[test]
#[ignore = "long-running (~11s)"]
fn milestone_b_bios_to_shell() {
    // After 500M instructions the BIOS has transitioned from the
    // boot logo to the MAIN MENU shell screen (MEMORY CARD / CD
    // PLAYER, radial blue gradient).
    let state = run_milestone(500_000_000, None);
    assert_milestone(
        "Milestone B",
        &state,
        0x0f00_2542_a50c_0dd0, // full VRAM (self)
        0x7410_746e_003a_8d85, // display area (self)
        (640, 478),
        None, // Redux parity capture pending (~17 min oracle run)
    );
}

#[test]
#[ignore = "requires Crash Bandicoot USA disc + long-running (~11s)"]
fn milestone_d_bios_accepts_licensed_disc() {
    // After 600M instructions with Crash Bandicoot USA mounted, the
    // BIOS has detected the licensed disc, issued the cold-boot
    // disc-read sequence from ROM, cleared the boot-logo VRAM, and
    // rendered the "SONY / PlayStation™" licensed-disc splash.
    //
    // Redux-parity on this is pending — the oracle doesn't pass
    // `-iso` to Redux yet, so we can't run Redux with Crash mounted
    // through the oracle path.
    if !std::path::Path::new(CRASH_DISC).exists() {
        eprintln!("skip milestone_d: Crash disc not found at {CRASH_DISC}");
        return;
    }
    let state = run_milestone(600_000_000, Some(CRASH_DISC));
    // Updated 2026-04-18: CDROM DMA3 sync-mode-0 fix landed. The
    // BIOS now *actually* boots past the PlayStation splash,
    // parses SYSTEM.CNF, loads the boot EXE, and starts running
    // game code. Game itself crashes at step 180,600,054 on a
    // wild-pointer fetch into unmapped memory (`pc=0x09070026`)
    // — downstream from our emulator's known gaps (GTE
    // completion, semi-transparency, etc.). Current golden is
    // the VRAM state at crash time; it should keep shifting as
    // the renderer accuracy improves.
    assert_milestone(
        "Milestone D",
        &state,
        0xa967_7706_9d62_2325, // full VRAM after game-code crash
        0xcd3d_81a7_020d_2325, // display area (self)
        (512, 240),            // game switched to a low-res mode
        None, // Redux disc-parity pending oracle `-iso` support
    );
}

#[test]
#[ignore = "requires Tekken 3 USA disc"]
fn milestone_d_tekken_licensed_screen() {
    // Secondary D-level canary: Tekken 3 boots all the way through
    // the BIOS handoff and into its own boot-EXE, which renders
    // the 3D red "PlayStation / Licensed by Sony Computer
    // Entertainment America / SCEA™" screen. Unlike Crash (which
    // crashes on a wild pointer at step 180M, likely downstream
    // of GTE gaps), Tekken holds stably on the license screen —
    // probably waiting for SPU / MDEC that we don't implement yet.
    //
    // Having two D-level goldens from different games doubles the
    // regression surface: a CDROM change that regresses one but
    // not the other still gets caught, and the Tekken path proves
    // the full BIOS→EXE→render chain works end-to-end for a
    // different SCUS disc. Captured 2026-04-18 right after the
    // DMA3 sync-mode-0 fix landed.
    if !std::path::Path::new(TEKKEN_DISC).exists() {
        eprintln!("skip milestone_d_tekken: disc not found at {TEKKEN_DISC}");
        return;
    }
    let state = run_milestone(800_000_000, Some(TEKKEN_DISC));
    assert_milestone(
        "Milestone D (Tekken license screen)",
        &state,
        0xb18b_d59c_f3ef_3062,
        0xe751_905e_59a6_3e02,
        (640, 478),
        None,
    );
}
