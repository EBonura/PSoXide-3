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
    if state.display_size != golden.display_size
        || state.vblank_raises != golden.vblank_raises
        || state.display_hash != golden.display_hash
        || state.vram_hash != golden.vram_hash
        || state.spu_samples != golden.spu_samples
        || state.final_pc != golden.final_pc
    {
        print_capture(state, golden.example, golden.vblanks);
    }
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
    eprintln!(
        "        display_size: ({}, {}),",
        state.display_size.0, state.display_size.1
    );
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
        // All examples refreshed 2026-04-19-j for the "every
        // example is double-buffered" pass. Display hashes shift
        // because the displayed buffer alternates A/B per frame
        // now — the capture grabs whichever is currently on
        // display. Render output visually identical to the
        // single-buffered goldens.
        "hello-tri" => Some(SdkGolden {
            example: "hello-tri",
            vblanks: 2,
            vram_hash: 0xe2b4_fca3_d005_dbc0,
            display_hash: 0x29dd_3c79_8152_b324,
            display_size: (320, 240),
            vblank_raises: 2,
            spu_samples: 735,
            final_pc: 0x8001_0480,
            redux_display_hash: None,
        }),
        // Two 4bpp CLUT textures cooked by `psxed tex` from 512×512
        // pre-cropped source JPGs — a brick wall + a cobblestone
        // floor, centre-square-cropped by the cooker then
        // Lanczos3-resampled to 64×64, sharing one tpage with
        // distinct CLUTs. Sprites drift via `psx_math::sincos` on
        // Lissajous curves rather than modulo-sawtooth motion.
        "hello-tex" => Some(SdkGolden {
            example: "hello-tex",
            vblanks: 2,
            vram_hash: 0x7d8c_c1a8_3fd6_ee47,
            display_hash: 0x6b0c_8a05_afca_2bae,
            display_size: (320, 240),
            vblank_raises: 2,
            spu_samples: 735,
            final_pc: 0x8001_130c,
            redux_display_hash: None,
        }),
        "hello-ot" => Some(SdkGolden {
            example: "hello-ot",
            vblanks: 2,
            vram_hash: 0x759a_f51d_8acc_34c1,
            display_hash: 0xd1dd_e059_00c0_5be3,
            display_size: (320, 240),
            vblank_raises: 2,
            spu_samples: 735,
            final_pc: 0x8001_05e4,
            redux_display_hash: None,
        }),
        "hello-input" => Some(SdkGolden {
            example: "hello-input",
            vblanks: 4,
            vram_hash: 0xcb9c_cb9f_9940_820a,
            display_hash: 0x7dac_2cef_bc15_db1b,
            display_size: (320, 240),
            vblank_raises: 4,
            spu_samples: 2205,
            final_pc: 0x8001_0bb4,
            redux_display_hash: None,
        }),
        "hello-gte" => Some(SdkGolden {
            example: "hello-gte",
            vblanks: 2,
            vram_hash: 0x13bd_5425_2a05_0149,
            display_hash: 0x579e_bb71_f21c_fd8d,
            display_size: (320, 240),
            vblank_raises: 2,
            spu_samples: 735,
            final_pc: 0x8001_0acc,
            redux_display_hash: None,
        }),
        "showcase-textured-sprite" => Some(SdkGolden {
            example: "showcase-textured-sprite",
            vblanks: 3,
            vram_hash: 0x4258_6aa5_5493_4f74,
            display_hash: 0xc949_202f_e476_f5ce,
            display_size: (320, 240),
            vblank_raises: 3,
            spu_samples: 1470,
            // Interactive material viewer: a compact room with a
            // single upright material pane. Face buttons swap the
            // texture sample and blend mode while the HUD names the
            // active material. World geometry is now authored as quads but
            // submitted through `WorldRenderPass` as independently
            // culled/sorted textured triangles: 3x3 floor, four
            // backface-culled walls, and a double-sided material card.
            // Floor/card surfaces use near-plane clipping; walls use
            // the stricter all-corners-projected path to avoid giant
            // behind-camera slabs during the orbit.
            // The camera no longer auto-orbits: D-pad rotates/dollies
            // the view while face buttons keep material selection.
            // Pitch is derived from the dolly radius so the view keeps
            // looking at the centre of the material card.
            // The world textured path now splits projected triangles
            // that would exceed the PS1 hardware extent limits, so the
            // floor stays stable at close and pulled-back camera ranges.
            // Visual hashes unchanged after moving the camera/world
            // transform math into `psx-engine::WorldCamera`, then after
            // adding material-derived world render layers; only code
            // layout shifted the frame-boundary PC.
            final_pc: 0x8001_2cf4,
            redux_display_hash: None,
        }),
        // showcase-text exercises all 6 draw paths in psx-font:
        // rect, scaled, rotated, affine, gradient, scaled-gradient.
        // 4 VBlanks captures an early rotation angle while the
        // static sections stay stable.
        "showcase-text" => Some(SdkGolden {
            example: "showcase-text",
            vblanks: 4,
            vram_hash: 0xb545_ba8f_6fc9_5f7b,
            display_hash: 0x7f11_de8d_527d_f7c3,
            display_size: (320, 240),
            vblank_raises: 4,
            spu_samples: 2205,
            // 0xb40 -> 0xe70 after Phase 3e: `App::run` + `Scene`
            // plumbing adds ~816 bytes of text-section code, so
            // `main`'s final `lw ra / jr ra` sits further on. VRAM
            // + display bytes-identical — port is pure plumbing.
            // Refreshed after engine render-helper growth changed
            // frame-boundary timing for this animation capture.
            final_pc: 0x8001_0e9c,
            redux_display_hash: None,
        }),
        // hello-audio: SPU init + 4 voices configured + ADPCM
        // uploaded. With no pad input in the test harness, no key-on
        // fires — the checkpoint therefore pins "SPU set up, no
        // voices playing yet" which still verifies the upload /
        // register-write path end-to-end. The `irq_histogram`
        // shows SPU IRQs firing regularly (index 7), confirming
        // the SPU pipeline runs.
        "hello-audio" => Some(SdkGolden {
            example: "hello-audio",
            vblanks: 3,
            vram_hash: 0x8b19_03a7_4844_8012,
            display_hash: 0xf33b_e6bf_7624_17cd,
            display_size: (320, 240),
            vblank_raises: 3,
            spu_samples: 1470,
            final_pc: 0x8001_0d0c,
            redux_display_hash: None,
        }),
        // First mini-game. At the 8-VBlank checkpoint the ball
        // has bounced off the right paddle and is coming back
        // left; AI paddle is tracking. Nothing scored yet.
        // First mini-game. At the 8-VBlank checkpoint the ball
        // has bounced off the right paddle and is coming back
        // left; AI paddle is tracking. Nothing scored yet.
        //
        // Ported to the engine framework (commit after 964f022).
        // VRAM + display hashes match the pre-engine build exactly —
        // proof that `App::run` + `Scene` perfectly replace the
        // hand-rolled main loop. Only `final_pc` shifted, because
        // the engine's code went into the text section.
        "game-pong" => Some(SdkGolden {
            example: "game-pong",
            vblanks: 8,
            vram_hash: 0x9a79_14eb_0915_3cfd,
            display_hash: 0x7da9_7577_f30c_22fe,
            display_size: (320, 240),
            vblank_raises: 8,
            spu_samples: 5145,
            final_pc: 0x8001_17c8,
            redux_display_hash: None,
        }),
        // Second mini-game. 60 VBlanks captures one serve-arc +
        // brick-break region with effects active (gradient BG,
        // ball trail, particles, potentially screen shake).
        // Exercises ~50-primitive OT path during effect bursts.
        // Second mini-game. 60 VBlanks captures one serve-arc +
        // brick-break region with effects active (gradient BG,
        // ball trail, particles via psx-fx ParticlePool, screen
        // shake via psx-fx ShakeState). Refreshed after the
        // psx-fx extraction — VRAM + display hashes are byte-
        // identical to the previous golden (same pixels), only
        // `final_pc` drifted from LTO code re-layout.
        // Ported to the engine framework. VRAM + display hashes
        // match the pre-engine build exactly — same pixels, same
        // physics, same AI. Only `final_pc` shifted because the
        // engine's code went into the text section.
        "game-breakout" => Some(SdkGolden {
            example: "game-breakout",
            vblanks: 60,
            vram_hash: 0x287c_bcfb_0959_e891,
            display_hash: 0x96c6_0f48_bdd3_5cd3,
            display_size: (320, 240),
            vblank_raises: 60,
            spu_samples: 44100,
            final_pc: 0x8001_3968,
            redux_display_hash: None,
        }),
        // Third mini-game. Space Invaders: 5×10 alien grid, ship
        // at bottom, bullet + bomb pools, wave progression. At
        // the 120 VBlank checkpoint the aliens have marched a
        // couple of steps + dropped their first bombs; no pad
        // input = no player shots but the enemy AI is firing.
        // Third mini-game. Space Invaders: 5×10 alien grid, ship
        // at bottom, bullet + bomb pools, wave progression. At
        // the 120 VBlank checkpoint the aliens have marched a
        // couple of steps + dropped their first bombs; no pad
        // input = no player shots but the enemy AI is firing.
        // Refreshed after psx-fx extraction — pixels unchanged,
        // only `final_pc` drifted from LTO re-layout.
        // Ported to the engine framework. Unlike breakout, both
        // VRAM and display hashes shifted — invaders uses a
        // per-frame counter in `maybe_drop_enemy_bomb` (`frame %
        // 40`) and the ship-flash strobe (`frame & 2`). The old
        // build incremented its own `g.frame` at the *start* of
        // update; the engine increments `ctx.frame` at the *end*
        // of the loop. That one-frame offset shifts bomb-drop
        // timing by one frame, which compounds to different alien
        // + bomb positions after 120 VBlanks. Same game — just a
        // time-shifted snapshot.
        "game-invaders" => Some(SdkGolden {
            example: "game-invaders",
            vblanks: 120,
            vram_hash: 0xfc55_f23a_1005_b52d,
            display_hash: 0x34ca_5c24_5e5a_4b19,
            display_size: (320, 240),
            vblank_raises: 120,
            spu_samples: 88200,
            final_pc: 0x8001_5964,
            redux_display_hash: None,
        }),
        // Flagship 3D showcase. Starfield + Suzanne (Blender
        // monkey) + Utah teapot (Martin Newell), each decimated
        // to PSX poly budget and driven by GTE projection into an
        // OT with back-face culling. ~200 rendered triangles per
        // frame (~400 pre-cull). 60 VBlanks pins a non-trivial
        // tumble angle.
        // Companion to `showcase-3d`: 4 moving coloured point
        // lights illuminating 6 scaled cubes via CPU-side per-
        // vertex shading (point lights aren't native to the
        // GTE). Validates the full normals→lighting→TriGouraud
        // pipeline + multi-light colour blending.
        // Phase 3e rebake: port to `psx-engine`. Phase 3f-followup:
        // cube render loop now uses `ActorTransform::load_gte()` —
        // byte-identical VRAM + display (the new helper produces
        // the same scaled-rotation matrix the old inline `scale_mat`
        // did), only `final_pc` shifted as code layout changed.
        // Phase 3f SIO rebake: the stricter controller shifter model
        // now keeps the visual/audio outputs byte-identical while the
        // per-frame pad poll consumes a different amount of CPU time,
        // so only the captured `final_pc` moved.
        "showcase-lights" => Some(SdkGolden {
            example: "showcase-lights",
            vblanks: 60,
            vram_hash: 0x393c_1b7b_a3ff_b8bd,
            display_hash: 0xcbb2_2705_0c4f_7575,
            display_size: (320, 240),
            vblank_raises: 60,
            spu_samples: 44100,
            // Engine `GouraudRenderPass` now owns culling, depth
            // policy, command sorting, and OT insertion for the
            // CPU-lit cube layer.
            final_pc: 0x8001_147c,
            redux_display_hash: None,
        }),

        // Now lit via the GTE's NCCS pipeline — 3 directional
        // camera-space lights rotated per-object into local frames,
        // per-vertex `project_lit` computes both screen coords
        // (RTPS) and lit colour (NCCS → RGB2). Triangles emit as
        // `TriGouraud` with per-vertex colours. Camera now orbits
        // around the scene centre from a slightly raised angle, so
        // object positions and rotations include a shared view matrix.
        "showcase-3d" => Some(SdkGolden {
            example: "showcase-3d",
            vblanks: 60,
            vram_hash: 0xbf7e_537c_28b4_ca87,
            display_hash: 0x9eb4_a149_9747_33f9,
            display_size: (320, 240),
            vblank_raises: 60,
            spu_samples: 44100,
            // Suzanne and teapot now share an engine depth band using
            // per-face projected SZ instead of fixed object slots.
            // The engine `GouraudRenderPass` sorts all opaque mesh
            // triangles before OT insertion, so same-slot order is
            // deterministic across both meshes instead of source-order.
            // The showcase uses 128 OT slots to make depth-bucket
            // artifacts easier to inspect under real GTE load.
            final_pc: 0x8001_3b7c,
            redux_display_hash: None,
        }),
        // PS1-commercial textured-Gouraud pipeline: per-vertex RTPS
        // + NCDS feeds depth-cue-blended colours into textured-Gouraud
        // triangles (GP0 0x34), followed by NCLIP + AVSZ3 for cull
        // and OT placement. Brick walls + cobblestone floor are
        // sampled from two cooked PSXT textures in a shared tpage.
        // Scrolling 16-ring corridor: each ring gap maps the full
        // 64x64 tile once, avoiding hardware UV repeat while reducing
        // the long-poly affine stretch of the old single-quad version.
        // Refreshed after restoring segmented geometry and slow
        // forward motion.
        // The display hash now covers 120 textured-Gouraud tris
        // instead of the temporary 8-tri single-quad corridor.
        "showcase-fog" => Some(SdkGolden {
            example: "showcase-fog",
            vblanks: 60,
            vram_hash: 0x6f2c_e3f9_b389_3e6d,
            display_hash: 0x1e65_dee2_c74a_167f,
            display_size: (320, 240),
            vblank_raises: 60,
            spu_samples: 44100,
            // Pixels unchanged; render loop now uses `OtFrame`,
            // `PrimitiveArena`, and `DepthBand` for the GTE OTZ map.
            final_pc: 0x8001_130c,
            redux_display_hash: None,
        }),
        "showcase-particles" => Some(SdkGolden {
            example: "showcase-particles",
            vblanks: 60,
            vram_hash: 0x8320_acb5_19ff_3047,
            display_hash: 0xe183_25de_aac2_0540,
            display_size: (320, 240),
            vblank_raises: 60,
            spu_samples: 44100,
            // Standalone `psx-fx::ParticlePool` demo: fixed pool,
            // OT-backed RectFlat arena, auto emitter, and HUD.
            final_pc: 0x8001_0f94,
            redux_display_hash: None,
        }),
        // First engine-domain example. Exercises `App::run`'s
        // main loop, the `Scene` trait, `Ctx` frame counter, and
        // `Angle`'s `per_frames` / `sin_q12_arg` conversions.
        "hello-engine" => Some(SdkGolden {
            example: "hello-engine",
            vblanks: 4,
            vram_hash: 0xed9a_1715_859f_f925,
            display_hash: 0xcf68_7a3c_953c_2d25,
            display_size: (320, 240),
            vblank_raises: 4,
            spu_samples: 2205,
            final_pc: 0x8001_07d8,
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
#[ignore = "SDK milestone: hello-audio roundtrip"]
fn milestone_c_hello_audio() {
    // 3 VBlanks lets the SPU run long enough for the init + upload
    // sequence to fully complete and for a handful of audio-pump
    // ticks to advance the `samples_produced` counter.
    run_sdk_milestone("hello-audio", 3);
}

#[test]
#[ignore = "SDK milestone: pong roundtrip"]
fn milestone_c_game_pong() {
    // 8 VBlanks exercises the game loop through its first paddle
    // bounce. The ball starts at centre moving right at (2, 1);
    // after 8 frames it's hit the right paddle area, triggered
    // the paddle-hit SFX, and the AI has begun tracking.
    run_sdk_milestone("game-pong", 8);
}

#[test]
#[ignore = "SDK milestone: breakout roundtrip"]
fn milestone_c_game_breakout() {
    // 60 VBlanks covers one serve arc + first brick break. Serve
    // auto-launches at frame 30 (no pad in harness), ball climbs
    // and hits a blue brick around frame 85-90 in the probe —
    // we leave 60 frames here so the test captures the ball
    // mid-flight on the way up with all 40 bricks still in place.
    // Still exercises the 44-primitive OT path every frame.
    run_sdk_milestone("game-breakout", 60);
}

#[test]
#[ignore = "SDK milestone: invaders roundtrip"]
fn milestone_c_game_invaders() {
    // 120 VBlanks captures the grid mid-march with enemy bombs
    // in flight. Exercises the 50+-primitive OT path + the
    // march / bullet / particle state machines.
    run_sdk_milestone("game-invaders", 120);
}

#[test]
#[ignore = "SDK milestone: showcase-lights roundtrip"]
fn milestone_c_showcase_lights() {
    // 60 VBlanks puts the 4 orbiting lights at a mid-cycle
    // position — cubes visibly tinted by nearby lights, 72
    // Gouraud triangles, 4 light markers on screen.
    run_sdk_milestone("showcase-lights", 60);
}

#[test]
#[ignore = "SDK milestone: showcase-3d roundtrip"]
fn milestone_c_showcase_3d() {
    // 60 VBlanks locks a frame where all three meshes have
    // rotated to an interesting angle + the starfield is fully
    // flowing. Covers the complete 3D pipeline: GTE projection
    // → back-face cull → OT depth-slot insert → DMA submit.
    run_sdk_milestone("showcase-3d", 60);
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

#[test]
#[ignore = "SDK milestone: showcase-fog roundtrip"]
fn milestone_c_showcase_fog() {
    // 60 VBlanks captures the segmented corridor after ~1 s of slow
    // forward motion. Covers per-vertex RTPS + NCDS, then NCLIP +
    // AVSZ3, feeding textured-Gouraud tris end-to-end.
    run_sdk_milestone("showcase-fog", 60);
}

#[test]
#[ignore = "SDK milestone: showcase-particles roundtrip"]
fn milestone_c_showcase_particles() {
    // 60 VBlanks captures the fixed particle pool with the automatic
    // emitter actively feeding the OT-backed RectFlat arena.
    run_sdk_milestone("showcase-particles", 60);
}

#[test]
#[ignore = "SDK milestone: hello-engine — smallest Scene/App round-trip"]
fn milestone_c_hello_engine() {
    // 4 VBlanks lands the drifting quad at a non-zero sine offset
    // from centre. Pins the engine's main-loop cadence
    // (poll/update/clear/render/sync/vsync/swap) plus the Angle →
    // sin_q12 conversion path.
    run_sdk_milestone("hello-engine", 4);
}
