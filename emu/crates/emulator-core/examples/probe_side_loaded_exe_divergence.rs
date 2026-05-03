//! Find the first exact CPU/GTE state divergence for a side-loaded PSX-EXE.
//!
//! Both emulators warm the real BIOS, side-load the same PSX-EXE, then
//! compare Redux-style user steps. This is the exact-state counterpart
//! to the side-loaded display oracle tests.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_side_loaded_exe_divergence --release -- hello-gte
//! cargo run -p emulator-core --example probe_side_loaded_exe_divergence --release -- hello-gte 250000 5000
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use emulator_core::{
    warm_bios_for_disc_fast_boot, Bus, Cpu, ExecutionError, DISC_FAST_BOOT_WARMUP_STEPS,
};
use parity_oracle::{OracleConfig, ReduxProcess};
use psx_iso::Exe;
use psx_trace::InstructionRecord;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const EXAMPLE_OUT: &str = "build/examples/mipsel-sony-psx/release";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const LOAD_TIMEOUT: Duration = Duration::from_secs(15);
const STEP_TIMEOUT: Duration = Duration::from_secs(30);

struct Config {
    bios: PathBuf,
    exe: PathBuf,
    start: u64,
    window: u32,
    warmup_steps: u64,
    ignore_cop2: bool,
    ignore_cop2_control: bool,
}

fn main() {
    let cfg = parse_args();
    if !cfg.bios.is_file() {
        eprintln!("BIOS not found: {}", cfg.bios.display());
        std::process::exit(2);
    }
    if !cfg.exe.is_file() {
        eprintln!("EXE not found: {}", cfg.exe.display());
        std::process::exit(2);
    }

    println!(
        "side-loaded exact divergence: exe={} warmup={} start={} window={}",
        cfg.exe.display(),
        cfg.warmup_steps,
        cfg.start,
        cfg.window,
    );
    if cfg.ignore_cop2 {
        println!("comparison: ignoring all COP2 snapshot differences");
    } else if cfg.ignore_cop2_control {
        println!("comparison: ignoring COP2 control-register snapshot differences");
    }

    let mut ours = load_local_exe_after_bios_warmup(&cfg);
    let mut redux = load_redux_exe_after_bios_warmup(&cfg);

    if cfg.start > 0 {
        fast_forward_local(&mut ours.1, &mut ours.0, cfg.start).expect("local fast-forward");
        let timeout = Duration::from_secs((cfg.start / 200_000).max(60));
        redux.run(cfg.start, timeout).expect("Redux fast-forward");
    }

    let t0 = Instant::now();
    let ours = trace_local(&mut ours.1, &mut ours.0, cfg.window).expect("local trace");
    eprintln!(
        "[ours] captured {} records in {:.1}s",
        ours.len(),
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let theirs = redux.step(cfg.window, STEP_TIMEOUT).expect("Redux trace");
    eprintln!(
        "[redux] captured {} records in {:.1}s",
        theirs.len(),
        t0.elapsed().as_secs_f64()
    );

    report_first_diff(
        cfg.start,
        &ours,
        &theirs,
        cfg.ignore_cop2,
        cfg.ignore_cop2_control,
    );

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();
}

fn parse_args() -> Config {
    let mut args = std::env::args().skip(1);
    let exe_arg = args.next().unwrap_or_else(|| "hello-gte".to_string());
    let start = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let window = args.next().and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let bios = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS));
    let warmup_steps = std::env::var("PSOXIDE_ORACLE_EXE_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DISC_FAST_BOOT_WARMUP_STEPS);
    let ignore_cop2_control = std::env::var("PSOXIDE_IGNORE_COP2C")
        .ok()
        .is_some_and(|value| value != "0");
    let ignore_cop2 = std::env::var("PSOXIDE_IGNORE_COP2")
        .ok()
        .is_some_and(|value| value != "0");
    Config {
        bios,
        exe: resolve_exe_arg(&exe_arg),
        start,
        window,
        warmup_steps,
        ignore_cop2,
        ignore_cop2_control,
    }
}

fn resolve_exe_arg(arg: &str) -> PathBuf {
    let path = PathBuf::from(arg);
    if path.is_file() || arg.ends_with(".exe") {
        return path;
    }
    repo_root().join(EXAMPLE_OUT).join(format!("{arg}.exe"))
}

fn load_local_exe_after_bios_warmup(cfg: &Config) -> (Bus, Cpu) {
    let bios = std::fs::read(&cfg.bios).expect("BIOS readable");
    let exe_bytes = std::fs::read(&cfg.exe).expect("EXE readable");
    let exe = Exe::parse(&exe_bytes).expect("parse PSX-EXE");

    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();
    warm_bios_for_disc_fast_boot(&mut bus, &mut cpu, cfg.warmup_steps).expect("local BIOS warmup");

    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.clear_exe_bss(exe.bss_addr, exe.bss_size);
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());
    (bus, cpu)
}

fn load_redux_exe_after_bios_warmup(cfg: &Config) -> ReduxProcess {
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let oracle_cfg = OracleConfig::new(cfg.bios.clone(), lua).expect("Redux binary resolves");
    let mut redux = ReduxProcess::launch(&oracle_cfg).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");
    let timeout = Duration::from_secs((cfg.warmup_steps / 200_000).max(60));
    redux.run(cfg.warmup_steps, timeout).expect("Redux warmup");
    let info = redux
        .load_exe(&cfg.exe, LOAD_TIMEOUT)
        .expect("Redux load_exe");
    eprintln!("[redux] load: {info:?}");
    redux
}

fn fast_forward_local(cpu: &mut Cpu, bus: &mut Bus, steps: u64) -> Result<(), ExecutionError> {
    for _ in 0..steps {
        step_user(cpu, bus)?;
    }
    Ok(())
}

fn trace_local(
    cpu: &mut Cpu,
    bus: &mut Bus,
    window: u32,
) -> Result<Vec<InstructionRecord>, ExecutionError> {
    let mut out = Vec::with_capacity(window as usize);
    for _ in 0..window {
        out.push(step_user(cpu, bus)?);
    }
    Ok(out)
}

fn step_user(cpu: &mut Cpu, bus: &mut Bus) -> Result<InstructionRecord, ExecutionError> {
    let was_in_isr = cpu.in_isr();
    let mut rec = cpu.step_traced(bus)?;
    if !was_in_isr && cpu.in_irq_handler() {
        while cpu.in_irq_handler() {
            let r = cpu.step_traced(bus)?;
            rec.tick = r.tick;
            rec.gprs = r.gprs;
        }
    }
    let (cop2_data, cop2_ctl) = snapshot_cop2(cpu);
    rec.cop2_data = cop2_data;
    rec.cop2_ctl = cop2_ctl;
    Ok(rec)
}

fn snapshot_cop2(cpu: &Cpu) -> ([u32; 32], [u32; 32]) {
    let mut data = [0u32; 32];
    let mut ctl = [0u32; 32];
    for i in 0..32u8 {
        data[i as usize] = cpu.cop2().read_data(i);
        ctl[i as usize] = cpu.cop2().read_control(i);
    }
    (data, ctl)
}

fn report_first_diff(
    start: u64,
    ours: &[InstructionRecord],
    theirs: &[InstructionRecord],
    ignore_cop2: bool,
    ignore_cop2_control: bool,
) {
    let count = ours.len().min(theirs.len());
    for i in 0..count {
        let a = &ours[i];
        let b = &theirs[i];
        if records_match(a, b, ignore_cop2, ignore_cop2_control) {
            continue;
        }
        let step = start + i as u64 + 1;
        println!();
        println!("First exact divergence at step {step} (window idx {i})");
        println!(
            "  ours : tick={} pc=0x{:08x} instr=0x{:08x}",
            a.tick, a.pc, a.instr
        );
        println!(
            "  redux: tick={} pc=0x{:08x} instr=0x{:08x}",
            b.tick, b.pc, b.instr
        );
        print_record_diffs(a, b, ignore_cop2, ignore_cop2_control);
        return;
    }
    if ours.len() == theirs.len() {
        println!("All {} records match.", ours.len());
    } else {
        println!(
            "Trace length mismatch after {count} records: ours={} redux={}",
            ours.len(),
            theirs.len()
        );
    }
}

fn records_match(
    a: &InstructionRecord,
    b: &InstructionRecord,
    ignore_cop2: bool,
    ignore_cop2_control: bool,
) -> bool {
    a.tick == b.tick
        && a.pc == b.pc
        && a.instr == b.instr
        && a.gprs == b.gprs
        && (ignore_cop2 || a.cop2_data == b.cop2_data)
        && (ignore_cop2 || ignore_cop2_control || a.cop2_ctl == b.cop2_ctl)
}

fn print_record_diffs(
    a: &InstructionRecord,
    b: &InstructionRecord,
    ignore_cop2: bool,
    ignore_cop2_control: bool,
) {
    for r in 0..32 {
        if a.gprs[r] != b.gprs[r] {
            println!(
                "  gpr[{r:02}]: ours=0x{:08x} redux=0x{:08x}",
                a.gprs[r], b.gprs[r]
            );
        }
    }
    if !ignore_cop2 {
        for r in 0..32 {
            if a.cop2_data[r] != b.cop2_data[r] {
                println!(
                    "  cop2d[{r:02}]: ours=0x{:08x} redux=0x{:08x}",
                    a.cop2_data[r], b.cop2_data[r]
                );
            }
        }
    }
    if !ignore_cop2 && !ignore_cop2_control {
        for r in 0..32 {
            if a.cop2_ctl[r] != b.cop2_ctl[r] {
                println!(
                    "  cop2c[{r:02}]: ours=0x{:08x} redux=0x{:08x}",
                    a.cop2_ctl[r], b.cop2_ctl[r]
                );
            }
        }
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}
