//! Side-loaded PSX-EXE display parity against Redux.
//!
//! This is the Redux-facing counterpart to the fast frame canaries in
//! `emulator-core/examples`: both emulators first warm the real BIOS,
//! then the same EXE payload is copied into RAM and both sides run to
//! VBlank checkpoints. That keeps this test out of PSoXide's HLE BIOS
//! side-load path, which is useful for canaries but not comparable to
//! Redux's real-BIOS execution model.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use emulator_core::{
    warm_bios_for_disc_fast_boot, Bus, Cpu, ExecutionError, DISC_FAST_BOOT_WARMUP_STEPS,
};
use parity_oracle::{OracleConfig, ReduxProcess};
use psx_iso::Exe;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const DEFAULT_EXE_NAME: &str = "hello-tri";
const EXAMPLE_OUT: &str = "build/examples/mipsel-sony-psx/release";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);
const LOAD_TIMEOUT: Duration = Duration::from_secs(15);
const HASH_TIMEOUT: Duration = Duration::from_secs(60);
const VBLANK_TIMEOUT: Duration = Duration::from_secs(60);
const IRQ_STAT_ADDR: u32 = 0x1F80_1070;
const SPU_PUMP_CYCLES: u64 = 560_000;
const SPU_SAMPLES_PER_PUMP: usize = 735;
const DEFAULT_CHECKPOINT_FRAMES: u64 = 3;
const DEFAULT_POST_VBLANK_SETTLE_STEPS: u64 = 0;

struct LocalExeRun {
    bus: Bus,
    cpu: Cpu,
    cycles_at_last_spu_pump: u64,
}

#[test]
#[ignore = "requires patched PCSX-Redux, BIOS, and a built SDK EXE; run via `make oracle-side-load`"]
fn side_loaded_hello_tri_vblank_display_parity() {
    let bios = bios_path();
    if !bios.exists() {
        eprintln!(
            "[side-load-parity] skip: BIOS not found at {}",
            bios.display()
        );
        return;
    }

    let exe = exe_path();
    if !exe.exists() {
        eprintln!(
            "[side-load-parity] skip: EXE not found at {}",
            exe.display()
        );
        return;
    }

    let warmup_steps = warmup_steps();
    let settle_steps = post_vblank_settle_steps();
    eprintln!("[side-load-parity] BIOS : {}", bios.display());
    eprintln!("[side-load-parity] EXE  : {}", exe.display());
    eprintln!("[side-load-parity] warm : {warmup_steps} steps");
    eprintln!("[side-load-parity] settle: {settle_steps} post-vblank steps");

    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(bios.clone(), lua).expect("Redux binary resolves");
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    let run_timeout = Duration::from_secs((warmup_steps / 200_000).max(60));
    redux.run(warmup_steps, run_timeout).expect("Redux warmup");
    let redux_info = redux.load_exe(&exe, LOAD_TIMEOUT).expect("Redux load_exe");
    eprintln!("[side-load-parity] Redux load: {redux_info:?}");

    let mut ours = load_local_exe_after_bios_warmup(&bios, &exe, warmup_steps);

    for frame in 1_u64..=checkpoint_frames() {
        let max_steps = max_vblank_steps();
        let redux_run = redux
            .run_vblanks(1, max_steps, VBLANK_TIMEOUT)
            .expect("Redux VBlank run");
        let our_steps = ours.run_vblanks(1, max_steps).expect("local VBlank run");
        if settle_steps > 0 {
            redux.run(settle_steps, VBLANK_TIMEOUT).expect("Redux settle");
            ours.run_steps(settle_steps).expect("local settle");
        }

        let their = redux
            .display_hash(HASH_TIMEOUT)
            .expect("Redux display hash");
        let (our_hash, our_w, our_h, our_len) = ours.bus.gpu.display_hash();
        let our_area = ours.bus.gpu.display_area();
        eprintln!(
            "[side-load-parity] frame={frame} redux_steps={} ours_steps={} settle={} ours=0x{our_hash:016x} ({our_w}x{our_h} @{},{} len={our_len}) redux=0x{:016x} ({}x{} len={})",
            redux_run.steps,
            our_steps,
            settle_steps,
            our_area.x,
            our_area.y,
            their.hash,
            their.width,
            their.height,
            their.byte_len,
        );

        assert_eq!(
            (our_w, our_h),
            (their.width, their.height),
            "display dimensions diverged at frame {frame}"
        );
        assert_eq!(
            our_hash, their.hash,
            "display hash diverged at frame {frame}"
        );
    }

    eprintln!(
        "[side-load-parity] local display starts: {:?}",
        ours.bus.gpu.display_start_history().collect::<Vec<_>>()
    );

    let _ = redux.terminate();
}

impl LocalExeRun {
    fn run_vblanks(&mut self, frames: u64, max_steps: u64) -> Result<u64, ExecutionError> {
        self.bus.write32(IRQ_STAT_ADDR, !1);
        let target = self.bus.irq().raise_counts()[0].saturating_add(frames);
        for step in 1..=max_steps {
            self.cpu.step(&mut self.bus)?;
            self.pump_spu_if_needed();
            if self.bus.irq().raise_counts()[0] >= target {
                return Ok(step);
            }
        }
        panic!(
            "local VBlank cap hit: target={} current={} max_steps={} pc=0x{:08x}",
            target,
            self.bus.irq().raise_counts()[0],
            max_steps,
            self.cpu.pc()
        );
    }

    fn run_steps(&mut self, steps: u64) -> Result<(), ExecutionError> {
        for _ in 0..steps {
            self.cpu.step(&mut self.bus)?;
            self.pump_spu_if_needed();
        }
        Ok(())
    }

    fn pump_spu_if_needed(&mut self) {
        if self.bus.cycles() - self.cycles_at_last_spu_pump > SPU_PUMP_CYCLES {
            self.cycles_at_last_spu_pump = self.bus.cycles();
            self.bus.run_spu_samples(SPU_SAMPLES_PER_PUMP);
            let _ = self.bus.spu.drain_audio();
        }
    }
}

fn load_local_exe_after_bios_warmup(
    bios_path: &Path,
    exe_path: &Path,
    warmup_steps: u64,
) -> LocalExeRun {
    let bios = fs::read(bios_path).expect("BIOS readable");
    let exe_bytes = fs::read(exe_path).expect("EXE readable");
    let exe = Exe::parse(&exe_bytes).expect("parse PSX-EXE");

    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();
    warm_bios_for_disc_fast_boot(&mut bus, &mut cpu, warmup_steps).expect("local BIOS warmup");

    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.clear_exe_bss(exe.bss_addr, exe.bss_size);
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    LocalExeRun {
        bus,
        cpu,
        cycles_at_last_spu_pump: 0,
    }
}

fn bios_path() -> PathBuf {
    env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
}

fn exe_path() -> PathBuf {
    if let Ok(path) = env::var("PSOXIDE_ORACLE_EXE") {
        return PathBuf::from(path);
    }
    repo_root()
        .join(EXAMPLE_OUT)
        .join(format!("{DEFAULT_EXE_NAME}.exe"))
}

fn warmup_steps() -> u64 {
    env::var("PSOXIDE_ORACLE_EXE_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DISC_FAST_BOOT_WARMUP_STEPS)
}

fn max_vblank_steps() -> u64 {
    env::var("PSOXIDE_ORACLE_EXE_MAX_STEPS_PER_FRAME")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000_000)
}

fn checkpoint_frames() -> u64 {
    env::var("PSOXIDE_ORACLE_EXE_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CHECKPOINT_FRAMES)
}

fn post_vblank_settle_steps() -> u64 {
    env::var("PSOXIDE_ORACLE_EXE_SETTLE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_POST_VBLANK_SETTLE_STEPS)
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}
