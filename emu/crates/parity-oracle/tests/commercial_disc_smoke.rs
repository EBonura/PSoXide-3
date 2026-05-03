//! Bounded commercial-disc display parity smoke test.
//!
//! This test intentionally does not name or vendor any copyrighted
//! disc image. Point it at a local BIN or CUE with `PSOXIDE_DISC`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use emulator_core::{Bus, Cpu, ExecutionError};
use parity_oracle::{OracleConfig, ReduxProcess};
use psx_iso::Disc;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const DEFAULT_STEPS: u64 = 1_000_000;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);
const HASH_TIMEOUT: Duration = Duration::from_secs(60);

#[test]
#[ignore = "requires patched PCSX-Redux, BIOS, and a local PSOXIDE_DISC BIN/CUE"]
fn commercial_disc_display_hash_matches_redux_at_checkpoint() {
    let Some(disc_path) = disc_path() else {
        eprintln!("[commercial-disc-smoke] skip: PSOXIDE_DISC is not set");
        return;
    };
    if !disc_path.exists() {
        eprintln!(
            "[commercial-disc-smoke] skip: disc not found at {}",
            disc_path.display()
        );
        return;
    }

    let bios_path = bios_path();
    if !bios_path.exists() {
        eprintln!(
            "[commercial-disc-smoke] skip: BIOS not found at {}",
            bios_path.display()
        );
        return;
    }

    let steps = step_count();
    eprintln!("[commercial-disc-smoke] BIOS : {}", bios_path.display());
    eprintln!("[commercial-disc-smoke] disc : {}", disc_path.display());
    eprintln!("[commercial-disc-smoke] steps: {steps}");

    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = match OracleConfig::new(bios_path.clone(), lua) {
        Ok(config) => config.with_disc(disc_path.clone()),
        Err(err) => {
            eprintln!("[commercial-disc-smoke] skip: Redux binary not available: {err}");
            return;
        }
    };

    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    let bios = fs::read(&bios_path).expect("BIOS readable");
    let disc = load_disc_path(&disc_path).expect("disc readable");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    let mut cpu = Cpu::new();

    let run_timeout = Duration::from_secs((steps / 200_000).max(60));
    redux.run(steps, run_timeout).expect("Redux run");
    if let Some((sub_step, err)) = step_ours_user_steps(&mut cpu, &mut bus, steps) {
        cleanup(redux);
        panic!("local emulator stopped at user step {sub_step}/{steps}: {err:?}");
    }

    let their = redux
        .display_hash(HASH_TIMEOUT)
        .expect("Redux display hash");
    let (our_hash, our_w, our_h, our_len) = bus.gpu.display_hash();
    eprintln!(
        "[commercial-disc-smoke] ours=0x{our_hash:016x} ({our_w}x{our_h} len={our_len}) \
         redux=0x{:016x} ({}x{} len={})",
        their.hash, their.width, their.height, their.byte_len,
    );

    cleanup(redux);

    assert_eq!(
        (our_w, our_h),
        (their.width, their.height),
        "display dimensions diverged after {steps} commercial-disc steps"
    );
    assert_eq!(
        our_hash, their.hash,
        "display hash diverged after {steps} commercial-disc steps"
    );
}

fn disc_path() -> Option<PathBuf> {
    env::var("PSOXIDE_DISC").ok().map(PathBuf::from)
}

fn bios_path() -> PathBuf {
    env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

fn step_count() -> u64 {
    env::var("PSOXIDE_ORACLE_DISC_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_STEPS)
}

fn load_disc_path(path: &Path) -> Result<Disc, String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("cue"))
    {
        psoxide_settings::library::load_disc_from_cue(path)
    } else {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Disc::from_bin(bytes))
    }
}

fn step_ours_user_steps(cpu: &mut Cpu, bus: &mut Bus, steps: u64) -> Option<(u64, ExecutionError)> {
    for i in 0..steps {
        let was_in_isr = cpu.in_isr();
        if let Err(e) = cpu.step(bus) {
            return Some((i, e));
        }
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                if let Err(e) = cpu.step(bus) {
                    return Some((i, e));
                }
            }
        }
    }
    None
}

fn cleanup(mut redux: ReduxProcess) {
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(5));
    let _ = redux.terminate();
}
