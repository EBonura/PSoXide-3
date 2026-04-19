//! Automated multi-checkpoint display parity against Redux.
//!
//! The intent is to **catch visual-regression bugs automatically**
//! rather than relying on self-consistency hashes. Each checkpoint is
//! a (step_count, name, threshold%) triple. At every checkpoint:
//!
//! 1. Drive Redux `step_delta` more instructions (cumulative) and
//!    capture its visible-display pixels via `screenshot_save`.
//! 2. Run ours from scratch to the same step count (we don't have a
//!    "resume from checkpoint" hook yet; starting fresh is ~2s/100M
//!    release and keeps the per-checkpoint run deterministic).
//! 3. Byte-compare the two framebuffers; `% diverging` must stay
//!    below the checkpoint's threshold.
//! 4. On failure, emit three PPMs to `target/parity-report/`:
//!      - `<name>_ours.ppm`  — our framebuffer
//!      - `<name>_redux.ppm` — redux framebuffer
//!      - `<name>_mask.ppm`  — red pixels where we diverge
//!
//! The threshold is NOT zero — PSX rasterizer edge ties and dither
//! patterns produce a few hundred pixel diffs that are not bugs per
//! se, just implementation-choice differences. The threshold gives
//! us a ceiling to catch *regressions* (things getting WORSE).
//!
//! Running:
//!
//! ```bash
//! cargo test -p parity-oracle --test display_parity --release -- --ignored --nocapture
//! ```
//!
//! Individual tests (each is `#[ignore]` because Redux takes minutes):
//! - `parity_bios_boot` — 50M/100M/200M/500M, no disc
//! - `parity_crash_boot` — 100M/200M/400M/600M, Crash Bandicoot disc
//! - `parity_tekken_boot` — 100M/300M/500M/800M, Tekken 3 disc

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};
use psx_iso::Disc;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const CRASH_DISC: &str = "/Users/ebonura/Downloads/ps1 games/Crash Bandicoot (USA)/Crash Bandicoot (USA).bin";
const TEKKEN_DISC: &str =
    "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA) (Track 1).bin";

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);

/// One entry in a parity schedule. The suite runs each in order,
/// cumulatively stepping both emulators forward to `steps` before
/// capturing.
#[derive(Clone, Copy, Debug)]
struct Checkpoint {
    /// Instruction count from cold boot.
    steps: u64,
    /// Human-readable label; shows up in the report + PPM filenames.
    name: &'static str,
    /// Max % of display bytes that may differ before this checkpoint
    /// fails. Tuned per-checkpoint because different scenes have
    /// different rasterizer/dither surfaces. `100.0` effectively
    /// disables the ceiling — use it for new checkpoints where we're
    /// still learning the baseline.
    max_diverge_pct: f64,
}

/// Outcome of a single checkpoint compare.
struct CheckpointResult {
    checkpoint: Checkpoint,
    our_hash: u64,
    redux_hash: u64,
    our_size: (u32, u32),
    redux_size: (u32, u32),
    diverge_pct: f64,
    first_diff: Option<(u32, u32)>,
    diff_bytes: usize,
    /// Number of bytes actually compared — usually `w*h*2`, smaller
    /// when sizes disagree and we compare overlap only. Retained for
    /// the summary file even though the panic message doesn't need it.
    #[allow(dead_code)]
    compared_bytes: usize,
}

fn bios_path() -> PathBuf {
    std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

fn report_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target")
        .join("parity-report")
}

/// Drive a single Redux process through the given schedule, compare
/// to ours at each stop, and dump the report. Panics if any
/// checkpoint exceeds its threshold — exactly what we want out of
/// a regression test.
fn run_parity_suite(
    suite: &str,
    disc_path: Option<&str>,
    checkpoints: &[Checkpoint],
) {
    assert!(!checkpoints.is_empty());
    if let Some(p) = disc_path {
        if !Path::new(p).exists() {
            eprintln!("[parity] skip {suite}: disc not found at {p}");
            return;
        }
    }

    // Both emulators need the BIOS.
    let bios_file = bios_path();
    if !bios_file.exists() {
        eprintln!("[parity] skip {suite}: BIOS not found at {}", bios_file.display());
        return;
    }

    let report = report_dir().join(suite);
    fs::create_dir_all(&report).expect("create report dir");

    // --- Launch Redux once, step-run through every checkpoint.
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut cfg = OracleConfig::new(bios_file.clone(), lua).expect("Redux binary resolves");
    if let Some(p) = disc_path {
        cfg = cfg.with_disc(PathBuf::from(p));
    }
    let mut redux = ReduxProcess::launch(&cfg).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    // --- Ours lives on the same timeline; we step it forward as we
    //     walk the schedule rather than restarting from scratch per
    //     checkpoint. At ~500k steps/s release this means a 3-point
    //     schedule finishes in ~2s ours-side (dwarfed by Redux).
    let bios = fs::read(&bios_file).expect("BIOS readable");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(p) = disc_path {
        let disc_bytes = fs::read(p).expect("disc readable");
        bus.cdrom.insert_disc(Some(Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();

    let mut cursor_steps: u64 = 0;
    // SPU pump cadence — mirrors the frontend's per-frame cadence so
    // the SPU's time-dependent state (voice envelopes, ADSR, reverb)
    // is exercised. Without this the SPU never runs and divergences
    // that only manifest under active audio generation get masked.
    let mut cycles_at_last_pump: u64 = 0;

    let mut results: Vec<CheckpointResult> = Vec::with_capacity(checkpoints.len());
    for cp in checkpoints {
        assert!(cp.steps > cursor_steps, "checkpoints must be strictly increasing");
        let delta = cp.steps - cursor_steps;

        // --- Redux: silent run forward, then screenshot_save.
        let run_timeout = Duration::from_secs((delta / 200_000).max(60));
        let r_tick = redux.run(delta, run_timeout).expect("redux run");
        let redux_bin = report.join(format!("{}_redux.bin", cp.name));
        redux
            .screenshot_save(&redux_bin, Duration::from_secs(60))
            .expect("screenshot save");
        let redux_bytes = fs::read(&redux_bin).expect("read redux bin");
        let meta = fs::read_to_string(format!("{}.txt", redux_bin.display()))
            .unwrap_or_default();
        let (rw, rh) = parse_wh(&meta);

        // --- Ours: step forward `delta` more instructions, pumping
        //     SPU at the frontend's cadence so audio-adjacent
        //     regressions aren't masked.
        let stopped_at = step_ours(&mut cpu, &mut bus, delta, &mut cycles_at_last_pump);
        if let Some(s) = stopped_at {
            panic!(
                "{}:{}: ours CPU errored at sub-step {s}/{delta} (pc=0x{:08x})",
                suite, cp.name, cpu.pc(),
            );
        }
        let (our_hash, our_w, our_h, our_len) = bus.gpu.display_hash();
        let our_bytes = read_display_bytes(&bus, our_w, our_h);
        assert_eq!(our_bytes.len(), our_len);

        // --- Compare. Mismatched dimensions are a separate class
        //     of failure — we still dump PPMs so the author can see
        //     what went wrong.
        let (diff_bytes, first_diff, compared) = if (rw, rh) == (our_w, our_h) {
            byte_compare(&our_bytes, &redux_bytes, rw)
        } else {
            // Compare the overlapping corner only; flag as "dimension
            // mismatch" after.
            let w = rw.min(our_w);
            let h = rh.min(our_h);
            let mut diffs = 0usize;
            let mut first: Option<(u32, u32)> = None;
            for y in 0..h {
                let r_row = (y as usize) * (rw as usize) * 2;
                let o_row = (y as usize) * (our_w as usize) * 2;
                let row_bytes = (w * 2) as usize;
                for x in 0..row_bytes {
                    let r = redux_bytes.get(r_row + x).copied();
                    let o = our_bytes.get(o_row + x).copied();
                    if r.is_some() && o.is_some() && r != o {
                        diffs += 1;
                        if first.is_none() {
                            first = Some((y, (x / 2) as u32));
                        }
                    }
                }
            }
            (diffs, first, (w * h * 2) as usize)
        };
        let diverge_pct = 100.0 * diff_bytes as f64 / compared.max(1) as f64;

        eprintln!(
            "[parity/{}/{}] steps={} r_tick={}  redux={}×{} ours={}×{}  diff={}/{} ({:.2}%)  threshold={:.1}%",
            suite, cp.name, cp.steps, r_tick,
            rw, rh, our_w, our_h,
            diff_bytes, compared,
            diverge_pct, cp.max_diverge_pct,
        );

        // Save our side (redux is already saved via screenshot_save).
        // Diff mask lives next to them so the author can open all three
        // together to localise regressions without rerunning the suite.
        let our_bin = report.join(format!("{}_ours.bin", cp.name));
        fs::write(&our_bin, &our_bytes).expect("write ours bin");
        write_ppm(&report.join(format!("{}_ours.ppm", cp.name)),
                  our_w as usize, our_h as usize,
                  |x, y| bgr15_to_rgb(&our_bytes, x, y, our_w as usize));
        if (rw, rh) == (our_w, our_h) {
            write_ppm(&report.join(format!("{}_redux.ppm", cp.name)),
                      rw as usize, rh as usize,
                      |x, y| bgr15_to_rgb(&redux_bytes, x, y, rw as usize));
            write_ppm(&report.join(format!("{}_mask.ppm", cp.name)),
                      rw as usize, rh as usize,
                      |x, y| {
                          let off = (y * rw as usize + x) * 2;
                          let r = u16::from_le_bytes([redux_bytes[off], redux_bytes[off + 1]]);
                          let o = u16::from_le_bytes([our_bytes[off], our_bytes[off + 1]]);
                          if r != o {
                              (0xFF, 0x00, 0x00)
                          } else {
                              let rgb = bgr15_to_rgb(&our_bytes, x, y, rw as usize);
                              let dim = |c: u8| (c as u16 * 3 / 10) as u8;
                              (dim(rgb.0), dim(rgb.1), dim(rgb.2))
                          }
                      });
        }

        results.push(CheckpointResult {
            checkpoint: *cp,
            our_hash,
            redux_hash: fnv1a_64(&redux_bytes),
            our_size: (our_w, our_h),
            redux_size: (rw, rh),
            diverge_pct,
            first_diff,
            diff_bytes,
            compared_bytes: compared,
        });

        cursor_steps = cp.steps;
    }

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(5));
    let _ = redux.terminate();

    // --- Write a summary file and enforce thresholds.
    let summary = report.join("SUMMARY.txt");
    let mut f = fs::File::create(&summary).expect("create summary");
    writeln!(f, "Parity suite: {}", suite).unwrap();
    writeln!(f, "Disc: {}", disc_path.unwrap_or("<none>")).unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "{:<20} {:>10} {:>10} {:>8} {:>7}  size_ok  first_diff",
        "checkpoint", "steps", "diff_bytes", "pct", "thresh").unwrap();
    for r in &results {
        let size_ok = if r.our_size == r.redux_size { "yes" } else { "NO " };
        let fd = r.first_diff.map(|(y, x)| format!("({x},{y})")).unwrap_or("-".into());
        writeln!(f, "{:<20} {:>10} {:>10} {:>7.2}% {:>6.1}%    {}     {}",
            r.checkpoint.name, r.checkpoint.steps,
            r.diff_bytes, r.diverge_pct, r.checkpoint.max_diverge_pct,
            size_ok, fd,
        ).unwrap();
    }
    writeln!(f, "").unwrap();
    for r in &results {
        writeln!(f, "{}:  our_hash=0x{:016x}  redux_hash=0x{:016x}",
            r.checkpoint.name, r.our_hash, r.redux_hash).unwrap();
    }
    eprintln!("[parity/{}] report: {}", suite, report.display());

    // Enforce thresholds — collect ALL failures before panicking so
    // the author gets the full picture, not just the first bad one.
    let mut failures: Vec<String> = Vec::new();
    for r in &results {
        if r.our_size != r.redux_size {
            failures.push(format!(
                "{}/{}: display size mismatch — ours={}×{} redux={}×{}",
                suite, r.checkpoint.name,
                r.our_size.0, r.our_size.1, r.redux_size.0, r.redux_size.1,
            ));
        }
        if r.diverge_pct > r.checkpoint.max_diverge_pct {
            failures.push(format!(
                "{}/{}: {:.2}% diverging bytes exceeds threshold {:.1}% \
                 (see {}/{}_mask.ppm)",
                suite, r.checkpoint.name,
                r.diverge_pct, r.checkpoint.max_diverge_pct,
                report.display(), r.checkpoint.name,
            ));
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} parity failure(s):\n  {}",
            failures.len(),
            failures.join("\n  "),
        );
    }
}

/// Step our CPU forward `delta` instructions, pumping SPU every
/// ~560k cycles to mirror the frontend's per-frame cadence. Returns
/// the sub-step the CPU errored at, or `None` on clean finish.
fn step_ours(
    cpu: &mut Cpu,
    bus: &mut Bus,
    delta: u64,
    cycles_at_last_pump: &mut u64,
) -> Option<u64> {
    for i in 0..delta {
        if cpu.step(bus).is_err() {
            return Some(i);
        }
        if bus.cycles() - *cycles_at_last_pump > 560_000 {
            *cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }
    }
    None
}

/// Compare two equally-sized framebuffers. Returns
/// `(differing bytes, first pixel where they differ, total bytes)`.
fn byte_compare(a: &[u8], b: &[u8], width: u32) -> (usize, Option<(u32, u32)>, usize) {
    assert_eq!(a.len(), b.len(), "size mismatch inside byte_compare");
    let mut diffs = 0usize;
    let mut first: Option<(u32, u32)> = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            diffs += 1;
            if first.is_none() {
                let px = i / 2;
                let py = px as u32 / width;
                let pxx = px as u32 % width;
                first = Some((py, pxx));
            }
        }
    }
    (diffs, first, a.len())
}

/// Extract the visible display area as raw 15bpp little-endian
/// bytes in scanline order — matches what Redux's `screenshot_save`
/// emits.
fn read_display_bytes(bus: &Bus, w: u32, h: u32) -> Vec<u8> {
    let da = bus.gpu.display_area();
    let mut out = Vec::with_capacity((w * h * 2) as usize);
    for dy in 0..h as u16 {
        for dx in 0..w as u16 {
            let pixel = bus.gpu.vram.get_pixel(da.x + dx, da.y + dy);
            out.extend_from_slice(&pixel.to_le_bytes());
        }
    }
    out
}

fn parse_wh(meta: &str) -> (u32, u32) {
    let mut w = 0u32;
    let mut h = 0u32;
    for tok in meta.split_whitespace() {
        if let Some(v) = tok.strip_prefix("w=") {
            w = v.parse().unwrap_or(0);
        } else if let Some(v) = tok.strip_prefix("h=") {
            h = v.parse().unwrap_or(0);
        }
    }
    (w, h)
}

fn bgr15_to_rgb(buf: &[u8], x: usize, y: usize, w: usize) -> (u8, u8, u8) {
    let off = (y * w + x) * 2;
    let pix = u16::from_le_bytes([buf[off], buf[off + 1]]);
    let r5 = (pix & 0x1F) as u8;
    let g5 = ((pix >> 5) & 0x1F) as u8;
    let b5 = ((pix >> 10) & 0x1F) as u8;
    ((r5 << 3) | (r5 >> 2), (g5 << 3) | (g5 >> 2), (b5 << 3) | (b5 >> 2))
}

fn write_ppm(
    path: &Path,
    w: usize,
    h: usize,
    pixel: impl Fn(usize, usize) -> (u8, u8, u8),
) {
    let mut f = fs::File::create(path).expect("create ppm");
    writeln!(f, "P6\n{w} {h}\n255").unwrap();
    let mut buf = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = pixel(x, y);
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }
    f.write_all(&buf).unwrap();
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}

// =========================================================================
// Actual test schedules
// =========================================================================

/// BIOS-only boot — no disc inserted. Progresses through the Sony
/// logo gradient (100M), holds stably on the logo (200M), transitions
/// to the shell (500M). Thresholds are calibrated against the
/// current state of the renderer: the Sony diamond picks up ~3.5%
/// divergence from dither/rasterizer-edge differences that are not
/// yet fixed. Keep thresholds tight enough that regressions show up.
#[test]
#[ignore = "requires PCSX-Redux + BIOS; ~2min in release"]
fn parity_bios_boot() {
    let schedule = [
        Checkpoint { steps:  50_000_000, name: "early_boot", max_diverge_pct: 0.50 },
        Checkpoint { steps: 100_000_000, name: "sony_logo",  max_diverge_pct: 4.00 },
        Checkpoint { steps: 200_000_000, name: "logo_hold",  max_diverge_pct: 5.00 },
        Checkpoint { steps: 500_000_000, name: "shell",      max_diverge_pct: 8.00 },
    ];
    run_parity_suite("bios", None, &schedule);
}

/// Crash Bandicoot — stresses the full BIOS → disc-handoff → game-
/// boot path that the user reports is *visually broken*. If Crash's
/// intro diverges from Redux at these checkpoints, we want the
/// regression to show up here rather than in hand-eyed QA.
///
/// Tight thresholds are unrealistic on the game-boot checkpoints —
/// Crash's splash + title screen exercise GPU paths (MDEC, more
/// complex transparency, viewport changes) that we don't track pixel-
/// exactly yet. The thresholds are set to *catch regressions*
/// rather than demand parity: pick a baseline, then fail when things
/// get significantly worse.
#[test]
#[ignore = "requires PCSX-Redux + Crash disc; ~3min in release"]
fn parity_crash_boot() {
    let schedule = [
        Checkpoint { steps: 100_000_000, name: "sony_logo",    max_diverge_pct:  4.00 },
        Checkpoint { steps: 200_000_000, name: "disc_handoff", max_diverge_pct: 10.00 },
        Checkpoint { steps: 400_000_000, name: "game_boot",    max_diverge_pct: 30.00 },
        Checkpoint { steps: 600_000_000, name: "intro_early",  max_diverge_pct: 50.00 },
    ];
    run_parity_suite("crash", Some(CRASH_DISC), &schedule);
}

/// Tekken 3 — secondary disc canary. Holds stably on the red-
/// PlayStation "Licensed by SCEA" screen rather than progressing
/// into gameplay, so later checkpoints stay visually identical
/// (good signal that the renderer is deterministic).
#[test]
#[ignore = "requires PCSX-Redux + Tekken 3 disc; ~4min in release"]
fn parity_tekken_boot() {
    let schedule = [
        Checkpoint { steps: 100_000_000, name: "sony_logo",    max_diverge_pct:  4.00 },
        Checkpoint { steps: 300_000_000, name: "disc_handoff", max_diverge_pct: 10.00 },
        Checkpoint { steps: 500_000_000, name: "licensed",     max_diverge_pct: 40.00 },
        Checkpoint { steps: 800_000_000, name: "licensed_hold", max_diverge_pct: 40.00 },
    ];
    run_parity_suite("tekken", Some(TEKKEN_DISC), &schedule);
}
