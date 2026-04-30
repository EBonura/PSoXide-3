//! Redux-anchored parity tests.
//!
//! Each test boots both the emulator-core CPU and a headless PCSX-Redux
//! from the same BIOS, single-steps the same number of instructions,
//! and asserts bit-identical [`InstructionRecord`]s.
//!
//! Marked `#[ignore]` so `cargo test` stays fast; run via
//! `make parity`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use emulator_core::{Bus, Cpu};
use parity_oracle::{cache, OracleConfig, ReduxProcess};
use psx_trace::InstructionRecord;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const STEP_TIMEOUT: Duration = Duration::from_secs(5);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

fn bios_path() -> PathBuf {
    env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

/// Step our CPU for `n` trace records, short-circuiting with the
/// last good records and the first execution error if we hit an
/// opcode the emulator doesn't yet decode.
///
/// IRQ-handler bodies are aggregated into the pre-IRQ record only
/// for *clean* IRQ entries -- i.e. when we weren't already in any
/// exception handler before the step. This mirrors Redux's
/// `debug.cc:235` early-return which triggers iff
/// `!m_wasInISR && m_inISR && cause == 0`. Syscalls (cause=8) are
/// recorded per-instruction, and IRQs taken from inside a syscall
/// handler or immediately after an RFE (`was_in_isr` true) are
/// also per-instruction -- Redux's trace shows them as distinct
/// 2-3-cycle steps rather than the 2000+-cycle collapsed jumps.
fn our_trace(
    bios: Vec<u8>,
    n: usize,
) -> (
    Vec<InstructionRecord>,
    Option<emulator_core::ExecutionError>,
) {
    let mut bus = Bus::new(bios).expect("BIOS size");
    let mut cpu = Cpu::new();
    let mut records = Vec::with_capacity(n);
    while records.len() < n {
        // `was_in_isr` = were we in ANY handler (IRQ or syscall) at
        // the start of this step? Matches Redux's `m_wasInISR`
        // captured by `startStepping()` before the interpreter runs.
        let was_in_isr = cpu.in_isr();
        let mut rec = match cpu.step_traced(&mut bus) {
            Ok(r) => r,
            Err(e) => return (records, Some(e)),
        };
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                match cpu.step_traced(&mut bus) {
                    Ok(r) => {
                        // Fold post-IRQ state forward as if the
                        // handler ran atomically. COP2 must come
                        // along too -- a GTE op tucked inside an
                        // IRQ handler is rare on real games but
                        // the BIOS does poke COP0/MTC2-style ops
                        // during init, and dropping the snapshot
                        // would mask divergences in those slots.
                        rec.tick = r.tick;
                        rec.gprs = r.gprs;
                        rec.cop2_data = r.cop2_data;
                        rec.cop2_ctl = r.cop2_ctl;
                    }
                    Err(e) => {
                        records.push(rec);
                        return (records, Some(e));
                    }
                }
            }
        }
        records.push(rec);
    }
    (records, None)
}

fn redux_trace(n: u32) -> Vec<InstructionRecord> {
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios_path(), lua).expect("Redux binary resolves");
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");
    let records = redux.step(n, STEP_TIMEOUT).expect("step n");
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(STEP_TIMEOUT);
    let _ = redux.terminate();
    records
}

/// Redux trace with on-disk cache. First run pays the full Redux
/// cost (several minutes for 10M+ steps); subsequent runs with the
/// same BIOS and step count load from disk in a fraction of a
/// second. See [`parity_oracle::cache`] for the file format.
///
/// `PSOXIDE_PARITY_NO_CACHE=1` forces a fresh Redux run even when
/// a cache exists -- useful when tweaking the oracle Lua script.
fn redux_trace_cached(n: usize) -> Vec<InstructionRecord> {
    let bios_bytes = fs::read(bios_path()).expect("BIOS readable");
    let dir = cache::default_dir();
    let path = cache::path_for(&dir, &bios_bytes, n);

    if std::env::var("PSOXIDE_PARITY_NO_CACHE").is_err() {
        // Prefer the exact-step cache; fall back to any longer
        // cache with the same BIOS hash. This means a single 50M
        // run satisfies every parity rung below 50M.
        if let Some(records) = cache::load(&path) {
            eprintln!(
                "[parity-cache] hit {} ({} records)",
                path.display(),
                records.len()
            );
            return records;
        }
        if let Some(records) = cache::load_prefix(&dir, &bios_bytes, n) {
            return records;
        }
    }

    eprintln!("[parity-cache] miss {} — invoking Redux", path.display());
    let records = redux_trace(n as u32);
    if let Err(e) = cache::save(&path, &records) {
        eprintln!("[parity-cache] save failed: {e}");
    } else {
        eprintln!(
            "[parity-cache] saved {} ({} records)",
            path.display(),
            records.len()
        );
    }
    records
}

/// Software-visible names for the 32 COP2 (GTE) data registers,
/// in `MFC2` index order. Used only for divergence diagnostics.
const COP2_DATA_NAMES: [&str; 32] = [
    "VXY0", "VZ0", "VXY1", "VZ1", "VXY2", "VZ2", "RGBC", "OTZ", "IR0", "IR1", "IR2", "IR3", "SXY0",
    "SXY1", "SXY2", "SXYP", "SZ0", "SZ1", "SZ2", "SZ3", "RGB0", "RGB1", "RGB2", "RES1", "MAC0",
    "MAC1", "MAC2", "MAC3", "IRGB", "ORGB", "LZCS", "LZCR",
];

/// Software-visible names for the 32 COP2 (GTE) control registers,
/// in `CFC2` index order. Used only for divergence diagnostics.
const COP2_CTL_NAMES: [&str; 32] = [
    "R11R12", "R13R21", "R22R23", "R31R32", "R33", "TRX", "TRY", "TRZ", "L11L12", "L13L21",
    "L22L23", "L31L32", "L33", "RBK", "GBK", "BBK", "LR1LR2", "LR3LG1", "LG2LG3", "LB1LB2", "LB3",
    "RFC", "GFC", "BFC", "OFX", "OFY", "H", "DQA", "DQB", "ZSF3", "ZSF4", "FLAG",
];

/// Compare trace by trace. Return the index of the first mismatch,
/// if any.
///
/// Ignores `tick` -- see notes in the top-of-file comment of the
/// `InstructionRecord` and the session README about why.
fn first_divergence(
    ours: &[InstructionRecord],
    theirs: &[InstructionRecord],
) -> Option<(usize, String)> {
    let pairs = ours.len().min(theirs.len());
    for i in 0..pairs {
        let (us, them) = (&ours[i], &theirs[i]);
        if us.pc == them.pc
            && us.instr == them.instr
            && us.gprs == them.gprs
            && us.cop2_data == them.cop2_data
            && us.cop2_ctl == them.cop2_ctl
        {
            continue;
        }

        let mut lines = Vec::new();
        lines.push(format!(
            "  at pc=0x{:08x} instr=0x{:08x}",
            them.pc, them.instr
        ));
        if i > 0 {
            let prev = &theirs[i - 1];
            lines.push(format!(
                "  prev instr: pc=0x{:08x} instr=0x{:08x}",
                prev.pc, prev.instr
            ));
            // Dump the full prior GPR state so we can compute effective
            // addresses and cross-check memory reads.
            lines.push("  prev gprs (theirs):".to_string());
            for r in (0..32).step_by(4) {
                lines.push(format!(
                    "    $r{:02}={:08x}  $r{:02}={:08x}  $r{:02}={:08x}  $r{:02}={:08x}",
                    r,
                    prev.gprs[r],
                    r + 1,
                    prev.gprs[r + 1],
                    r + 2,
                    prev.gprs[r + 2],
                    r + 3,
                    prev.gprs[r + 3],
                ));
            }
        }
        if us.pc != them.pc {
            lines.push(format!(
                "  pc:    ours=0x{:08x}  theirs=0x{:08x}",
                us.pc, them.pc
            ));
        }
        if us.instr != them.instr {
            lines.push(format!(
                "  instr: ours=0x{:08x}  theirs=0x{:08x}",
                us.instr, them.instr
            ));
        }
        for r in 0..32 {
            if us.gprs[r] != them.gprs[r] {
                lines.push(format!(
                    "  $r{:<2}:  ours=0x{:08x}  theirs=0x{:08x}",
                    r, us.gprs[r], them.gprs[r]
                ));
            }
        }
        for (r, name) in COP2_DATA_NAMES.iter().enumerate() {
            if us.cop2_data[r] != them.cop2_data[r] {
                lines.push(format!(
                    "  cop2d[{r:<2}] {name:<5}: ours=0x{:08x}  theirs=0x{:08x}",
                    us.cop2_data[r],
                    them.cop2_data[r],
                    name = name,
                ));
            }
        }
        for (r, name) in COP2_CTL_NAMES.iter().enumerate() {
            if us.cop2_ctl[r] != them.cop2_ctl[r] {
                lines.push(format!(
                    "  cop2c[{r:<2}] {name:<7}: ours=0x{:08x}  theirs=0x{:08x}",
                    us.cop2_ctl[r],
                    them.cop2_ctl[r],
                    name = name,
                ));
            }
        }
        return Some((i, lines.join("\n")));
    }

    if ours.len() != theirs.len() {
        let next = theirs.get(ours.len()).expect("theirs longer");
        return Some((
            ours.len(),
            format!(
                "  our trace ended early; redux next: pc=0x{:08x} instr=0x{:08x}",
                next.pc, next.instr
            ),
        ));
    }

    None
}

/// Run both emulators for `n` steps and assert full parity step-by-step.
/// On divergence, prints which step diverged and why, plus the opcode
/// that needs implementation (if we failed to decode).
fn assert_parity_for_steps(n: usize) {
    let bios_bytes = fs::read(bios_path()).expect("BIOS readable");
    let (ours, err) = our_trace(bios_bytes, n);
    let theirs = redux_trace_cached(n);

    let report_prefix = || -> String {
        let mut s = format!("parity ran for {}/{} steps\n", ours.len(), n);
        if let Some(e) = &err {
            s.push_str(&format!("our emulator stopped with: {e}\n"));
            if let Some(next) = theirs.get(ours.len()) {
                s.push_str(&format!(
                    "redux's next instruction: pc=0x{:08x} instr=0x{:08x}\n",
                    next.pc, next.instr
                ));
            }
        }
        s
    };

    if let Some((idx, diff)) = first_divergence(&ours, &theirs) {
        panic!("{}\ndivergence at step {}:\n{}", report_prefix(), idx, diff);
    }

    if err.is_some() {
        panic!("{}", report_prefix());
    }

    eprintln!(
        "parity OK for {} steps; final PC = 0x{:08x}",
        n,
        ours.last().map(|r| r.pc).unwrap_or(0)
    );
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_step_matches_redux() {
    assert_parity_for_steps(1);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_twenty_steps_match_redux() {
    assert_parity_for_steps(20);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_hundred_steps_match_redux() {
    assert_parity_for_steps(100);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_five_hundred_steps_match_redux() {
    assert_parity_for_steps(500);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_two_thousand_steps_match_redux() {
    assert_parity_for_steps(2000);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_five_thousand_steps_match_redux() {
    assert_parity_for_steps(5000);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_fifty_thousand_steps_match_redux() {
    assert_parity_for_steps(50_000);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_two_hundred_thousand_steps_match_redux() {
    assert_parity_for_steps(200_000);
}

#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_million_steps_match_redux() {
    assert_parity_for_steps(1_000_000);
}

/// Highest currently-passing count. Crosses the former SPUSTAT
/// blocker (~2.735M) and runs comfortably into the post-Sony-intro
/// region where the BIOS first pokes the CDROM (~99M lands the next
/// stall, beyond this milestone).
#[test]
#[ignore = "requires PCSX-Redux binary; run via `make parity`"]
fn first_ten_million_steps_match_redux() {
    assert_parity_for_steps(10_000_000);
}

// Binary-search rungs between 10M and 50M. Once a 50M (or longer)
// cache exists, these all resolve in ~1s instead of re-invoking
// Redux -- handy for localising a divergence without rebuilding
// the whole trace.

#[test]
#[ignore = "requires Redux cache; run via `make parity`"]
fn first_twenty_million_steps_match_redux() {
    assert_parity_for_steps(20_000_000);
}

#[test]
#[ignore = "requires Redux cache; run via `make parity`"]
fn first_thirty_million_steps_match_redux() {
    assert_parity_for_steps(30_000_000);
}

#[test]
#[ignore = "requires Redux cache; run via `make parity`"]
fn first_forty_million_steps_match_redux() {
    assert_parity_for_steps(40_000_000);
}

/// Probe past the next expected blocker -- the BIOS reaches state 25
/// of the CDROM-init state machine around step 99M. Use this to find
/// what diverges next, then absorb into the ladder.
#[test]
#[ignore = "probe — move to named milestone once the next finding is in"]
fn probe_next_divergence_at_cdrom_init() {
    assert_parity_for_steps(50_000_000);
}
