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
use parity_oracle::{OracleConfig, ReduxProcess};
use psx_trace::InstructionRecord;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const STEP_TIMEOUT: Duration = Duration::from_secs(5);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

fn bios_path() -> PathBuf {
    env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

/// Step our CPU for `n` instructions, short-circuiting with the last
/// good records and the first execution error if we hit an opcode the
/// emulator doesn't yet decode. The caller can then compare the partial
/// trace against Redux's full trace to see exactly which opcode we
/// need to implement next.
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
    for _ in 0..n {
        match cpu.step(&mut bus) {
            Ok(rec) => records.push(rec),
            Err(e) => return (records, Some(e)),
        }
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

/// Compare trace by trace. Return the index of the first mismatch,
/// if any.
///
/// Ignores `tick` — see notes in the top-of-file comment of the
/// `InstructionRecord` and the session README about why.
fn first_divergence(
    ours: &[InstructionRecord],
    theirs: &[InstructionRecord],
) -> Option<(usize, String)> {
    let pairs = ours.len().min(theirs.len());
    for i in 0..pairs {
        let (us, them) = (&ours[i], &theirs[i]);
        if us.pc == them.pc && us.instr == them.instr && us.gprs == them.gprs {
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
    let theirs = redux_trace(n as u32);

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

/// Probe past the next expected blocker — the BIOS reaches state 25
/// of the CDROM-init state machine around step 99M. Use this to find
/// what diverges next, then absorb into the ladder.
#[test]
#[ignore = "probe — move to named milestone once the next finding is in"]
fn probe_next_divergence_at_cdrom_init() {
    assert_parity_for_steps(50_000_000);
}
