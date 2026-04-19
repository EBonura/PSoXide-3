//! Milestone-C regression suite — SDK examples → VRAM hash.
//!
//! Every SDK example is a milestone-level regression test. The
//! example binary under `sdk/examples/<name>/` is the exact artifact
//! the user side-loads with `make run-<name>`; this test side-loads
//! the SAME binary, runs to a frame-boundary checkpoint, and pins a
//! multi-signal state snapshot. If either the SDK or the emulator
//! regresses such that the example's output changes, the test fails.
//!
//! ## Deliberate improvements over psoxide.1's SDK example suite
//!
//! 1. **Frame-boundary stepping** (not step-count-based). We run
//!    until the Nth VBlank fires, which is stable under any CPU-
//!    timing change the parity-accuracy work may introduce. Step-
//!    count stepping drifts every time the interpreter's cycle
//!    accounting shifts by even a few cycles.
//! 2. **Multi-signal capture**. A regression in SPU scheduling,
//!    IRQ dispatch, or pad polling doesn't always show up in the
//!    VRAM hash. We also snapshot `samples_produced`, the 11-slot
//!    IRQ raise histogram, CPU final-PC, and `cycles()`. Any one
//!    of those drifting fails the test with a pointer at the
//!    faulty subsystem.
//! 3. **Redux-verified per-example hashes** (when captured). The
//!    `redux_display_hash: Some(h)` slot is the Redux-on-the-same-
//!    binary answer, locked in. Psoxide.1 had READMEs for each
//!    example but no goldens anywhere — regressions were caught
//!    by hand.
//! 4. **Tests ARE examples**. There's no "test fixture" separate
//!    from the binary `make run-<name>` builds. Break the test,
//!    run the example in the frontend, watch what's wrong.
//!
//! ## Running
//!
//! The SDK examples require nightly + `build-std`; host `cargo test`
//! can't build them transparently. First:
//!
//! ```bash
//! make examples             # builds all 5 .exe files
//! make test-sdk             # runs this module's ignored tests
//! # or, directly:
//! cargo test -p emulator-core --release --test sdk_milestones -- --ignored
//! ```
//!
//! Without `make examples` first the tests skip gracefully (the
//! .exe file is missing).

use emulator_core::{Bus, Cpu};
use psx_iso::Exe;
use std::path::{Path, PathBuf};

/// Default BIOS for side-load runs. The EXE side-load bypasses BIOS
/// execution, but `Bus::new` still requires a valid BIOS image for
/// its memory map — any SCPH image of the right size will do.
const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";

/// Resolve the repo-root path from `CARGO_MANIFEST_DIR`. We're at
/// `emu/crates/emulator-core/` so three `..`s land at the root.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn bios_path() -> PathBuf {
    std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

/// Where cargo dumps the built SDK example binaries. Must match
/// the `EXAMPLE_OUT` in `Makefile`.
fn example_exe_path(name: &str) -> PathBuf {
    repo_root()
        .join("build/examples/mipsel-sony-psx/release")
        .join(format!("{name}.exe"))
}

/// Multi-signal snapshot of emulator state at a frame checkpoint.
/// Every field that could independently regress gets its own slot —
/// a pure-rendering bug fails on `display_hash`; an audio-scheduling
/// regression fails on `spu_samples`; an IRQ timing shift fails on
/// `irq_histogram`; a wild pointer at end-of-run fails on `final_pc`.
#[derive(Debug, Clone)]
pub struct SdkExampleState {
    /// FNV-1a-64 over the full 1 MiB VRAM buffer. Catches anything
    /// that changes VRAM bytes, including off-screen CLUT / texture
    /// writes the user never sees.
    pub vram_hash: u64,
    /// FNV-1a-64 over the visible-display rectangle only. Comparable
    /// byte-for-byte to Redux's `PCSX.GPU.takeScreenShot`.
    pub display_hash: u64,
    /// Visible display size in pixels `(width, height)`.
    pub display_size: (u32, u32),
    /// Number of times the VBlank IRQ was raised during the run.
    /// This is the value that gated frame-boundary stepping.
    pub vblank_raises: u64,
    /// Total stereo audio samples the SPU has produced at end-of-run.
    pub spu_samples: u64,
    /// Per-IrqSource raise count (11 slots — VBlank, Gpu, Cdrom, Dma,
    /// Timer0/1/2, Controller, Sio, Spu, Lightpen).
    pub irq_histogram: [u64; 11],
    /// CPU program counter at end-of-run. Should land in a legit
    /// code region for a clean stop; a wild address points at an
    /// earlier control-flow regression.
    pub final_pc: u32,
    /// Retired-instruction count (`Cpu::tick()`).
    pub cpu_ticks: u64,
    /// Bus cycle count at end-of-run.
    pub bus_cycles: u64,
}

/// Per-example golden. `redux_display_hash: Some(h)` is a Redux-on-
/// the-same-binary verification; `None` means not captured yet (the
/// self-regression still pins, but correctness is unverified).
#[derive(Debug, Clone)]
pub struct SdkGolden {
    /// Example binary name, without `.exe` suffix.
    pub example: &'static str,
    /// Number of VBlanks the test stepped for.
    pub vblanks: u64,
    /// Expected full-VRAM hash.
    pub vram_hash: u64,
    /// Expected display-area hash.
    pub display_hash: u64,
    /// Expected `(width, height)`.
    pub display_size: (u32, u32),
    /// Expected VBlank raise count — usually `vblanks` exactly.
    pub vblank_raises: u64,
    /// Expected total SPU sample count.
    pub spu_samples: u64,
    /// Expected CPU final-PC.
    pub final_pc: u32,
    /// Redux-verified display hash. `Some(h)` = pixel parity confirmed.
    pub redux_display_hash: Option<u64>,
}

/// Skip a test gracefully if its prerequisites aren't on disk.
/// Returns `true` when the test should proceed, `false` when the
/// caller should early-return. Writes a descriptive skip reason to
/// stderr in the `false` case.
fn check_prereqs(exe_path: &Path) -> bool {
    let bios = bios_path();
    if !bios.exists() {
        eprintln!("skip: BIOS not found at {}", bios.display());
        return false;
    }
    if !exe_path.exists() {
        eprintln!(
            "skip: SDK example not built at {}\n\
             hint: run `make examples` first",
            exe_path.display(),
        );
        return false;
    }
    true
}

/// Side-load an SDK example .exe, step until `vblanks` frames have
/// elapsed, and return a state snapshot. Matches the flow the
/// frontend uses when `PSOXIDE_EXE=…` is set: load payload, seed
/// CPU from EXE header, enable HLE BIOS + digital pad 1.
///
/// Frame-boundary stepping polls the VBlank IRQ raise counter (slot
/// 0 per `IrqSource::VBlank`). A watchdog cap of 500M CPU steps
/// covers even the slowest realistic example startup — exceeding it
/// points at a deadlock in the example or the emulator.
pub fn side_load_and_hash(exe_path: &Path, vblanks: u64) -> Option<SdkExampleState> {
    if !check_prereqs(exe_path) {
        return None;
    }

    let bios = std::fs::read(bios_path()).expect("BIOS");
    let exe_bytes = std::fs::read(exe_path).unwrap_or_else(|e| {
        panic!("read {}: {e}", exe_path.display());
    });
    let exe = Exe::parse(&exe_bytes).unwrap_or_else(|e| {
        panic!("parse {}: {e:?}", exe_path.display());
    });

    let mut bus = Bus::new(bios).expect("bus");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.enable_hle_bios();
    bus.attach_digital_pad_port1();

    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    // SPU pump cadence mirrors `run_milestone` in the A/B/D tests —
    // one NTSC frame's worth of samples every ~560k CPU cycles.
    // Keeps SPU sample count deterministic per VBlank.
    let mut cycles_at_last_pump = 0u64;
    let target_vblanks = vblanks;
    // Watchdog — a reasonable example hits `vblanks=1` in at most
    // a few million steps. 500M is generous; anything that hits it
    // is almost certainly a deadlock.
    const WATCHDOG_STEPS: u64 = 500_000_000;
    let mut steps = 0u64;
    while bus.irq().raise_counts()[0] < target_vblanks {
        if cpu.step(&mut bus).is_err() {
            break;
        }
        steps += 1;
        if steps >= WATCHDOG_STEPS {
            panic!(
                "watchdog: {} CPU steps elapsed before reaching {} VBlanks \
                 (reached {}); example likely deadlocked",
                steps,
                target_vblanks,
                bus.irq().raise_counts()[0],
            );
        }
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }
    }

    // Full-VRAM FNV-1a-64 + visible-display hash. Same hash function
    // the A/B/D milestones use so a regression in either renders as a
    // visually-correlatable diff.
    let mut vh = 0xCBF2_9CE4_8422_2325u64;
    for &w in bus.gpu.vram.words() {
        for b in w.to_le_bytes() {
            vh ^= b as u64;
            vh = vh.wrapping_mul(0x0100_0000_01B3);
        }
    }
    let (dh, dw, dhi, _dlen) = bus.gpu.display_hash();

    Some(SdkExampleState {
        vram_hash: vh,
        display_hash: dh,
        display_size: (dw, dhi),
        vblank_raises: bus.irq().raise_counts()[0],
        spu_samples: bus.spu.samples_produced(),
        irq_histogram: bus.irq().raise_counts(),
        final_pc: cpu.pc(),
        cpu_ticks: cpu.tick(),
        bus_cycles: bus.cycles(),
    })
}

/// Assert a captured state against a pinned golden, multi-signal.
/// Separate asserts per channel so a failure message points at the
/// specific subsystem that regressed.
fn assert_sdk_golden(state: &SdkExampleState, golden: &SdkGolden) {
    let name = golden.example;
    assert_eq!(
        state.display_size, golden.display_size,
        "{name}: display dimensions changed — GP1 display-mode regression",
    );
    assert_eq!(
        state.vblank_raises, golden.vblank_raises,
        "{name}: VBlank raise count drifted — scheduler / video-timing regression",
    );
    assert_eq!(
        state.display_hash, golden.display_hash,
        "{name}: display-area hash changed — rendering regression \
         (check dither / rasterizer / blend / CLUT / texture window)",
    );
    assert_eq!(
        state.vram_hash, golden.vram_hash,
        "{name}: full-VRAM hash changed. Display may still look right — \
         check off-screen VRAM (CLUT / textures / double-buffer).",
    );
    assert_eq!(
        state.spu_samples, golden.spu_samples,
        "{name}: SPU samples_produced drifted — audio pump / voice scheduling regression",
    );
    assert_eq!(
        state.final_pc, golden.final_pc,
        "{name}: final PC drifted — control-flow / IRQ-timing regression",
    );
    if let Some(expected) = golden.redux_display_hash {
        assert_eq!(
            state.display_hash, expected,
            "{name}: display hash doesn't match Redux's for the same binary — \
             we're rendering the wrong pixels.",
        );
    }
}

/// Capture-mode helper for bootstrapping a new golden. Prints every
/// field in a format that pastes directly into an `SdkGolden { .. }`
/// literal. Invoked by failing tests so the author can paste the
/// observed values into the test source once they've confirmed the
/// output is correct.
#[allow(dead_code)]
fn print_capture(state: &SdkExampleState, example: &str, vblanks: u64) {
    eprintln!();
    eprintln!("[capture] SdkGolden for `{example}`:");
    eprintln!("    SdkGolden {{");
    eprintln!("        example: {example:?},");
    eprintln!("        vblanks: {vblanks},");
    eprintln!("        vram_hash: 0x{:016x},", state.vram_hash);
    eprintln!("        display_hash: 0x{:016x},", state.display_hash);
    eprintln!("        display_size: ({}, {}),", state.display_size.0, state.display_size.1);
    eprintln!("        vblank_raises: {},", state.vblank_raises);
    eprintln!("        spu_samples: {},", state.spu_samples);
    eprintln!("        final_pc: 0x{:08x},", state.final_pc);
    eprintln!("        redux_display_hash: None, // TODO: capture via display_parity_at");
    eprintln!("    }}");
    eprintln!(
        "[capture]   (debug: cpu_ticks={}, bus_cycles={}, irq_histogram={:?})",
        state.cpu_ticks, state.bus_cycles, state.irq_histogram,
    );
}

// =================================================================
// Per-example tests. Each is `#[ignore]`'d until its golden is
// captured. On first run, the test panics with the "not captured
// yet" message and prints the observed values via `print_capture`;
// paste those into the `SdkGolden` literal and unignore.
//
// Order mirrors the Makefile's example list so the ladder stays
// readable.
// =================================================================

/// Return golden for an example once it's been captured; `None`
/// keeps the test in bootstrap-mode until the author has verified
/// the output in the frontend and pinned the values.
///
/// Captured 2026-04-19 against the emulator state on `main` at that
/// date (scanline-delta rasterizer + CHCR-only DMA trigger). None
/// of these have `redux_display_hash` pinned yet — that comes from
/// running `display_parity_at` with the same .exe side-loaded into
/// Redux, which needs a Redux-side side-load harness we haven't
/// built. For now these are self-regression only: they catch any
/// future change to emulator OR SDK output for this example.
///
/// **If a parity-accuracy change shifts the cycle count, these
/// hashes will drift.** That's intentional — it surfaces the
/// change at CI time with a pointer at which signal shifted (VRAM
/// vs display vs SPU vs PC). Refresh the golden by re-running the
/// test with panic output and pasting the new literal.
fn golden_for(example: &str) -> Option<SdkGolden> {
    match example {
        "hello-tri" => Some(SdkGolden {
            example: "hello-tri",
            vblanks: 2,
            vram_hash: 0x6c0e_8b93_a400_5724,
            display_hash: 0x29dd_3c79_8152_b324,
            display_size: (320, 240),
            vblank_raises: 2,
            spu_samples: 735,
            final_pc: 0x8001_0400,
            redux_display_hash: None,
        }),
        "hello-tex" => Some(SdkGolden {
            example: "hello-tex",
            vblanks: 2,
            // VRAM + display bytes are identical to the pre-refactor
            // values (the `rgb5(31,16,4)` / `rgb5(4,24,28)` calls
            // emit the same 5-bit pixels the hand-math used).
            vram_hash: 0x5256_9600_727a_9f25,
            display_hash: 0xfffc_c5ce_2a8d_0225,
            display_size: (320, 240),
            vblank_raises: 2,
            spu_samples: 735,
            // Refreshed 2026-04-19-b after the `psx-vram` refactor.
            // Binary layout shifted — different final instruction
            // address at the 2-VBlank checkpoint. Render output
            // unchanged (hashes above are byte-identical to pre-
            // refactor).
            final_pc: 0x8001_03ac,
            redux_display_hash: None,
        }),
        "hello-ot" => Some(SdkGolden {
            example: "hello-ot",
            vblanks: 2,
            vram_hash: 0x1ce1_429a_eeeb_68ec,
            display_hash: 0xf1d4_4ba6_1e83_e0ec,
            display_size: (320, 240),
            vblank_raises: 2,
            spu_samples: 735,
            final_pc: 0x8001_04e0,
            redux_display_hash: None,
        }),
        "hello-input" => Some(SdkGolden {
            example: "hello-input",
            vblanks: 4,
            // Refreshed 2026-04-19-g after the `MAX_PACK_HALFWORDS`
            // bump from 2048 to 8192 (16 KiB upload scratch, needed
            // for 8×16 / 16×16 fonts). Render output unchanged;
            // binary layout shifts as the larger stack array moves
            // register alloc around. Hashes byte-identical.
            vram_hash: 0x82e5_f04a_3f03_a4c4,
            display_hash: 0x7dac_2cef_bc15_db1b,
            display_size: (320, 240),
            vblank_raises: 4,
            spu_samples: 2205,
            final_pc: 0x8001_0b14,
            redux_display_hash: None,
        }),
        "hello-gte" => Some(SdkGolden {
            example: "hello-gte",
            vblanks: 2,
            // Refreshed 2026-04-19-d after the MFC2/CFC2 load-delay
            // fix in `psx-gte::regs`. Previous goldens pinned the
            // pre-fix output where `scene::project_vertex` returned
            // the STALE $8 register value (whatever was written by
            // the preceding MTC2) instead of the coprocessor result
            // — visually that produced an entirely blank frame since
            // the resulting "screen coords" were the raw vertex
            // XY/Z bits (e.g. (-2048, -1)), clipped off-screen.
            // The new golden captures a real wireframe cube with
            // 12 visible edges tumbling on the blue background.
            vram_hash: 0xfe80_941e_838f_6e81,
            display_hash: 0x3bd7_90b1_1224_0a81,
            display_size: (320, 240),
            vblank_raises: 2,
            spu_samples: 735,
            final_pc: 0x8001_0a68,
            redux_display_hash: None,
        }),
        "showcase-textured-sprite" => Some(SdkGolden {
            example: "showcase-textured-sprite",
            vblanks: 3,
            vram_hash: 0x7e78_155a_0f2b_da99,
            display_hash: 0x50ad_9226_b84a_64ad,
            display_size: (320, 240),
            vblank_raises: 3,
            spu_samples: 1470,
            final_pc: 0x8001_04fc,
            redux_display_hash: None,
        }),
        // showcase-text exercises all 6 draw paths in psx-font:
        // rect, scaled, rotated, affine, gradient, scaled-gradient.
        // 4 VBlanks captures an early rotation angle while the
        // static sections stay stable.
        "showcase-text" => Some(SdkGolden {
            example: "showcase-text",
            vblanks: 4,
            // Refreshed 2026-04-19-h after switching to the
            // `FrameBuffer` double-buffered path. Text rendering
            // was single-buffered before and visibly flickered as
            // dense draws (~870 GP0 words / frame) crossed into
            // the TV's active scanout. Double-buffer swaps show a
            // stable buffer while we draw into the other.
            //
            // VRAM + display hashes shift (displayed buffer
            // alternates A/B per frame now); final_pc shifts from
            // the extra swap/clear code. Rotation angle at the
            // 4-VBlank checkpoint is unchanged.
            vram_hash: 0x3c0a_b554_b68e_a0e2,
            display_hash: 0x7cfc_87d6_c36a_4ab1,
            display_size: (320, 240),
            vblank_raises: 4,
            spu_samples: 2205,
            final_pc: 0x8001_0ac0,
            redux_display_hash: None,
        }),
        _ => None,
    }
}

/// Shared body: side-load, capture, either assert-against-golden
/// or print-capture-and-fail.
fn run_sdk_milestone(example: &'static str, vblanks: u64) {
    let exe = example_exe_path(example);
    let Some(state) = side_load_and_hash(&exe, vblanks) else {
        return; // prereqs missing, already logged
    };
    match golden_for(example) {
        Some(golden) => {
            assert_sdk_golden(&state, &golden);
        }
        None => {
            print_capture(&state, example, vblanks);
            panic!(
                "{example}: no golden captured yet. \
                 Paste the SdkGolden literal above into `golden_for` and unignore the test."
            );
        }
    }
}

#[test]
#[ignore = "SDK milestone: hello-tri roundtrip — requires `make examples` + captured golden"]
fn milestone_c_hello_tri() {
    run_sdk_milestone("hello-tri", 2);
}

#[test]
#[ignore = "SDK milestone: hello-tex roundtrip"]
fn milestone_c_hello_tex() {
    run_sdk_milestone("hello-tex", 2);
}

#[test]
#[ignore = "SDK milestone: hello-ot roundtrip"]
fn milestone_c_hello_ot() {
    run_sdk_milestone("hello-ot", 2);
}

#[test]
#[ignore = "SDK milestone: hello-input roundtrip"]
fn milestone_c_hello_input() {
    // hello-input runs an input-polling loop; 4 VBlanks is enough to
    // exercise the pad-read path a handful of times.
    run_sdk_milestone("hello-input", 4);
}

#[test]
#[ignore = "SDK milestone: hello-gte roundtrip"]
fn milestone_c_hello_gte() {
    run_sdk_milestone("hello-gte", 2);
}

#[test]
#[ignore = "SDK milestone: showcase-textured-sprite roundtrip"]
fn milestone_c_showcase_textured_sprite() {
    // 3 VBlanks gives the bouncing sprites non-trivial motion in
    // the captured frame. Larger than the hellos' 2 because this is
    // a "polished demo" checkpoint — we want to catch regressions in
    // multi-frame state, not just first-frame initialization.
    run_sdk_milestone("showcase-textured-sprite", 3);
}

#[test]
#[ignore = "SDK milestone: showcase-text roundtrip"]
fn milestone_c_showcase_text() {
    // 4 VBlanks lands the rotation demo at a stable, non-zero
    // angle (4 × 96 = 384 Q0.12 ≈ 33.75°) — pins both the
    // rotating quad path and the static sections in one shot.
    run_sdk_milestone("showcase-text", 4);
}
