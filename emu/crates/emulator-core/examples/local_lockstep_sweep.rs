//! Local-library lockstep sweep against PCSX-Redux.
//!
//! This is the umbrella parity harness for the user's actual game
//! collection. It discovers CUE files under the local PS1 library,
//! runs each game in Redux and PSoXide for the same number of
//! Redux-style user steps, and checks:
//!
//! - CPU clock/PC/state checkpoints (`GPR + COP2` hash)
//! - exact instruction records for the first divergent checkpoint window
//! - visible-display byte parity at the final checkpoint
//!
//! Example:
//!
//! ```bash
//! cargo run -p emulator-core --example local_lockstep_sweep --release -- \
//!   --steps 100000000 --interval 10000
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu, ExecutionError};
use parity_oracle::{OracleConfig, ReduxProcess, StateCheckpoint};
use psx_trace::InstructionRecord;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const DEFAULT_GAMES_ROOT: &str = "/Users/ebonura/Downloads/ps1 games";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const STEP_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
struct Config {
    bios: PathBuf,
    root: PathBuf,
    discs: Vec<PathBuf>,
    steps: u64,
    interval: u64,
    limit: Option<usize>,
    visual: bool,
    exact_window: u64,
    report_dir: PathBuf,
}

#[derive(Debug)]
struct GameResult {
    name: String,
    disc: PathBuf,
    cpu_ok: bool,
    visual_ok: bool,
    first_checkpoint_mismatch: Option<Mismatch>,
    exact_mismatch: Option<String>,
    visual_diff: Option<VisualDiff>,
}

#[derive(Debug, Clone, Copy)]
struct Mismatch {
    previous_step: u64,
    checkpoint: StateCheckpoint,
    ours: StateCheckpoint,
    redux: StateCheckpoint,
}

#[derive(Debug, Clone)]
struct VisualDiff {
    ours_size: (u32, u32),
    redux_size: (u32, u32),
    diff_bytes: usize,
    compared_bytes: usize,
    first_diff: Option<(u32, u32)>,
}

fn main() {
    let cfg = parse_args();
    if !cfg.bios.is_file() {
        eprintln!("BIOS not found: {}", cfg.bios.display());
        std::process::exit(2);
    }
    fs::create_dir_all(&cfg.report_dir).expect("create report dir");

    let games = discover_games(&cfg);
    if games.is_empty() {
        eprintln!("No discs found. Pass --disc PATH or --root PATH.");
        std::process::exit(2);
    }

    println!(
        "local lockstep sweep: games={} steps={} interval={} visual={} report={}",
        games.len(),
        cfg.steps,
        cfg.interval,
        cfg.visual,
        cfg.report_dir.display(),
    );

    let mut results = Vec::with_capacity(games.len());
    for (idx, disc) in games.iter().enumerate() {
        let name = game_name(disc);
        println!();
        println!(
            "[{}/{}] {}",
            idx + 1,
            games.len(),
            disc.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("<unknown>")
        );
        match run_game(&cfg, &name, disc) {
            Ok(result) => {
                print_game_result(&result);
                results.push(result);
            }
            Err(e) => {
                eprintln!("  ERROR: {e}");
                results.push(GameResult {
                    name,
                    disc: disc.clone(),
                    cpu_ok: false,
                    visual_ok: false,
                    first_checkpoint_mismatch: None,
                    exact_mismatch: Some(e),
                    visual_diff: None,
                });
            }
        }
    }

    write_index_summary(&cfg, &results);
    let failed = results.iter().filter(|r| !r.cpu_ok || !r.visual_ok).count();
    println!();
    if failed == 0 {
        println!("LOCKSTEP OK: every swept game matched Redux at the configured checkpoints.");
    } else {
        println!("LOCKSTEP FAIL: {failed}/{} games diverged.", results.len());
        std::process::exit(1);
    }
}

fn parse_args() -> Config {
    let mut cfg = Config {
        bios: std::env::var("PSOXIDE_BIOS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS)),
        root: std::env::var("PSOXIDE_GAMES_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_GAMES_ROOT)),
        discs: Vec::new(),
        steps: 100_000_000,
        interval: 10_000,
        limit: None,
        visual: true,
        exact_window: 10_000,
        report_dir: default_report_dir(),
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bios" => cfg.bios = take_path(&mut args, "--bios"),
            "--root" => cfg.root = take_path(&mut args, "--root"),
            "--disc" => cfg.discs.push(take_path(&mut args, "--disc")),
            "--steps" => cfg.steps = take_u64(&mut args, "--steps"),
            "--interval" => cfg.interval = take_u64(&mut args, "--interval"),
            "--limit" => cfg.limit = Some(take_usize(&mut args, "--limit")),
            "--exact-window" => cfg.exact_window = take_u64(&mut args, "--exact-window"),
            "--report-dir" => cfg.report_dir = take_path(&mut args, "--report-dir"),
            "--no-visual" => cfg.visual = false,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}; pass --help"),
        }
    }
    assert!(cfg.steps > 0, "--steps must be > 0");
    assert!(cfg.interval > 0, "--interval must be > 0");
    if !cfg.report_dir.is_absolute() {
        cfg.report_dir = std::env::current_dir()
            .expect("current dir")
            .join(&cfg.report_dir);
    }
    cfg
}

fn take_path(args: &mut impl Iterator<Item = String>, flag: &str) -> PathBuf {
    args.next()
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn take_u64(args: &mut impl Iterator<Item = String>, flag: &str) -> u64 {
    args.next()
        .unwrap_or_else(|| panic!("{flag} requires a value"))
        .parse()
        .unwrap_or_else(|_| panic!("{flag} requires an integer"))
}

fn take_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> usize {
    args.next()
        .unwrap_or_else(|| panic!("{flag} requires a value"))
        .parse()
        .unwrap_or_else(|_| panic!("{flag} requires an integer"))
}

fn print_help() {
    println!(
        "\
local_lockstep_sweep

Options:
  --bios PATH          BIOS image (default: PSOXIDE_BIOS or {DEFAULT_BIOS})
  --root PATH          game-library root (default: PSOXIDE_GAMES_ROOT or {DEFAULT_GAMES_ROOT})
  --disc PATH          add a specific CUE/BIN/ISO; can be repeated
  --steps N            Redux-style user steps per game (default: 100000000)
  --interval N         CPU state checkpoint interval (default: 10000)
  --exact-window N     full-trace window after first coarse mismatch (default: 10000)
  --limit N            run only the first N discovered games
  --report-dir PATH    report output dir
  --no-visual          skip final screenshot byte comparison
"
    );
}

fn default_report_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target")
        .join("local-lockstep")
}

fn discover_games(cfg: &Config) -> Vec<PathBuf> {
    let mut games = if cfg.discs.is_empty() {
        disc_support::discover_cue_files(&cfg.root)
            .unwrap_or_else(|e| panic!("discover {}: {e}", cfg.root.display()))
    } else {
        cfg.discs.clone()
    };
    games.sort();
    games.dedup();
    if let Some(limit) = cfg.limit {
        games.truncate(limit);
    }
    games
}

fn run_game(cfg: &Config, name: &str, disc: &Path) -> Result<GameResult, String> {
    let game_dir = cfg.report_dir.join(name);
    fs::create_dir_all(&game_dir).map_err(|e| format!("create {}: {e}", game_dir.display()))?;

    let t0 = Instant::now();
    let bios = fs::read(&cfg.bios).map_err(|e| format!("read BIOS: {e}"))?;
    let mut bus = Bus::new(bios).map_err(|e| format!("bus init: {e}"))?;
    let image = disc_support::load_disc_path(disc)?;
    bus.cdrom.insert_disc(Some(image));
    let mut cpu = Cpu::new();
    let ours = run_ours_checkpoints(&mut cpu, &mut bus, cfg.steps, cfg.interval)?;
    eprintln!(
        "  ours: {} checkpoints in {:.1}s",
        ours.checkpoints.len(),
        t0.elapsed().as_secs_f64()
    );

    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut oracle_cfg =
        OracleConfig::new(cfg.bios.clone(), lua).map_err(|e| format!("oracle config: {e}"))?;
    oracle_cfg = oracle_cfg.with_disc(disc.to_path_buf());
    let mut redux = ReduxProcess::launch(&oracle_cfg).map_err(|e| format!("launch Redux: {e}"))?;
    redux
        .handshake(HANDSHAKE_TIMEOUT)
        .map_err(|e| format!("Redux handshake: {e}"))?;

    let timeout = Duration::from_secs((cfg.steps / 200_000).max(60));
    let mut redux_checks = Vec::with_capacity(ours.checkpoints.len());
    let redux_final = redux
        .run_state_checkpoint(cfg.steps, cfg.interval, timeout, |cp| {
            redux_checks.push(cp);
            Ok(())
        })
        .map_err(|e| format!("Redux state checkpoints: {e}"))?;
    eprintln!("  redux: {} checkpoints", redux_checks.len());

    let first_checkpoint_mismatch = compare_checkpoints(
        cfg.interval,
        &ours.checkpoints,
        ours.final_checkpoint,
        &redux_checks,
        redux_final,
    );

    let exact_mismatch = if let Some(mismatch) = first_checkpoint_mismatch {
        let window = (mismatch.checkpoint.step - mismatch.previous_step).min(cfg.exact_window);
        pinpoint_exact_divergence(cfg, disc, mismatch.previous_step, window)
            .map_err(|e| format!("exact divergence probe: {e}"))?
    } else {
        None
    };

    let visual_diff = if cfg.visual {
        Some(compare_visual(&mut redux, &bus, &game_dir, "final")?)
    } else {
        None
    };

    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();

    let cpu_ok = first_checkpoint_mismatch.is_none() && exact_mismatch.is_none();
    let visual_ok = visual_diff
        .as_ref()
        .map(|d| d.ours_size == d.redux_size && d.diff_bytes == 0)
        .unwrap_or(true);

    let result = GameResult {
        name: name.to_string(),
        disc: disc.to_path_buf(),
        cpu_ok,
        visual_ok,
        first_checkpoint_mismatch,
        exact_mismatch,
        visual_diff,
    };
    write_game_summary(&game_dir, &result);
    Ok(result)
}

#[derive(Debug)]
struct OurRun {
    checkpoints: Vec<StateCheckpoint>,
    final_checkpoint: StateCheckpoint,
}

fn run_ours_checkpoints(
    cpu: &mut Cpu,
    bus: &mut Bus,
    steps: u64,
    interval: u64,
) -> Result<OurRun, String> {
    let mut checkpoints = Vec::with_capacity((steps / interval) as usize);
    for step in 1..=steps {
        step_user(cpu, bus).map_err(|e| format!("our CPU at step {step}: {e}"))?;
        if step % interval == 0 {
            checkpoints.push(state_checkpoint(step, cpu, bus));
        }
    }
    Ok(OurRun {
        checkpoints,
        final_checkpoint: state_checkpoint(steps, cpu, bus),
    })
}

fn step_user(cpu: &mut Cpu, bus: &mut Bus) -> Result<InstructionRecord, ExecutionError> {
    let was_in_isr = cpu.in_isr();
    let mut rec = cpu.step(bus)?;
    if !was_in_isr && cpu.in_irq_handler() {
        while cpu.in_irq_handler() {
            let r = cpu.step(bus)?;
            rec.tick = r.tick;
            rec.gprs = r.gprs;
            rec.cop2_data = snapshot_cop2(cpu).0;
            rec.cop2_ctl = snapshot_cop2(cpu).1;
        }
    }
    let (cop2_data, cop2_ctl) = snapshot_cop2(cpu);
    rec.cop2_data = cop2_data;
    rec.cop2_ctl = cop2_ctl;
    Ok(rec)
}

fn state_checkpoint(step: u64, cpu: &Cpu, bus: &Bus) -> StateCheckpoint {
    StateCheckpoint {
        step,
        tick: bus.cycles(),
        pc: cpu.pc(),
        state_hash: cpu_state_hash(cpu),
    }
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

fn cpu_state_hash(cpu: &Cpu) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &v in cpu.gprs() {
        h = fnv_update_u32(h, v);
    }
    let (cop2_data, cop2_ctl) = snapshot_cop2(cpu);
    for &v in &cop2_data {
        h = fnv_update_u32(h, v);
    }
    for &v in &cop2_ctl {
        h = fnv_update_u32(h, v);
    }
    h
}

fn fnv_update_u32(mut h: u64, value: u32) -> u64 {
    for byte in value.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}

fn compare_checkpoints(
    interval: u64,
    ours: &[StateCheckpoint],
    ours_final: StateCheckpoint,
    redux: &[StateCheckpoint],
    redux_final: StateCheckpoint,
) -> Option<Mismatch> {
    let count = ours.len().min(redux.len());
    for i in 0..count {
        if !same_state(ours[i], redux[i]) {
            return Some(Mismatch {
                previous_step: if i == 0 { 0 } else { ours[i - 1].step },
                checkpoint: redux[i],
                ours: ours[i],
                redux: redux[i],
            });
        }
    }
    if ours.len() != redux.len() {
        let step = ((count as u64) + 1) * interval;
        return Some(Mismatch {
            previous_step: count as u64 * interval,
            checkpoint: StateCheckpoint {
                step,
                tick: 0,
                pc: 0,
                state_hash: 0,
            },
            ours: ours.get(count).copied().unwrap_or(ours_final),
            redux: redux.get(count).copied().unwrap_or(redux_final),
        });
    }
    if !same_state(ours_final, redux_final) {
        return Some(Mismatch {
            previous_step: ours.last().map(|c| c.step).unwrap_or(0),
            checkpoint: redux_final,
            ours: ours_final,
            redux: redux_final,
        });
    }
    None
}

fn same_state(a: StateCheckpoint, b: StateCheckpoint) -> bool {
    a.step == b.step && a.tick == b.tick && a.pc == b.pc && a.state_hash == b.state_hash
}

fn pinpoint_exact_divergence(
    cfg: &Config,
    disc: &Path,
    start: u64,
    window: u64,
) -> Result<Option<String>, String> {
    if window == 0 {
        return Ok(None);
    }

    let bios = fs::read(&cfg.bios).map_err(|e| format!("read BIOS: {e}"))?;
    let mut bus = Bus::new(bios).map_err(|e| format!("bus init: {e}"))?;
    let image = disc_support::load_disc_path(disc)?;
    bus.cdrom.insert_disc(Some(image));
    let mut cpu = Cpu::new();
    for step in 1..=start {
        step_user(&mut cpu, &mut bus).map_err(|e| format!("skip ours step {step}: {e}"))?;
    }
    let mut ours = Vec::with_capacity(window as usize);
    for offset in 0..window {
        let rec = step_user(&mut cpu, &mut bus)
            .map_err(|e| format!("record ours step {}: {e}", start + offset + 1))?;
        ours.push(rec);
    }

    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut oracle_cfg =
        OracleConfig::new(cfg.bios.clone(), lua).map_err(|e| format!("oracle config: {e}"))?;
    oracle_cfg = oracle_cfg.with_disc(disc.to_path_buf());
    let mut redux = ReduxProcess::launch(&oracle_cfg).map_err(|e| format!("launch Redux: {e}"))?;
    redux
        .handshake(HANDSHAKE_TIMEOUT)
        .map_err(|e| format!("Redux handshake: {e}"))?;
    if start > 0 {
        let timeout = Duration::from_secs((start / 200_000).max(60));
        redux
            .run(start, timeout)
            .map_err(|e| format!("Redux skip to {start}: {e}"))?;
    }
    let theirs = redux
        .step(window as u32, STEP_TIMEOUT)
        .map_err(|e| format!("Redux exact window: {e}"))?;
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(2));
    let _ = redux.terminate();

    Ok(first_record_diff(start, &ours, &theirs))
}

fn first_record_diff(
    start: u64,
    ours: &[InstructionRecord],
    theirs: &[InstructionRecord],
) -> Option<String> {
    let count = ours.len().min(theirs.len());
    for i in 0..count {
        let a = &ours[i];
        let b = &theirs[i];
        if a.tick == b.tick
            && a.pc == b.pc
            && a.instr == b.instr
            && a.gprs == b.gprs
            && a.cop2_data == b.cop2_data
            && a.cop2_ctl == b.cop2_ctl
        {
            continue;
        }
        let step = start + i as u64 + 1;
        let mut out = format!(
            "exact divergence at step {step}: ours tick={} pc=0x{:08x} instr=0x{:08x}; redux tick={} pc=0x{:08x} instr=0x{:08x}",
            a.tick, a.pc, a.instr, b.tick, b.pc, b.instr,
        );
        for r in 0..32 {
            if a.gprs[r] != b.gprs[r] {
                out.push_str(&format!(
                    "\n  gpr[{r}]: ours=0x{:08x} redux=0x{:08x}",
                    a.gprs[r], b.gprs[r]
                ));
            }
        }
        for r in 0..32 {
            if a.cop2_data[r] != b.cop2_data[r] {
                out.push_str(&format!(
                    "\n  cop2d[{r}]: ours=0x{:08x} redux=0x{:08x}",
                    a.cop2_data[r], b.cop2_data[r]
                ));
            }
        }
        for r in 0..32 {
            if a.cop2_ctl[r] != b.cop2_ctl[r] {
                out.push_str(&format!(
                    "\n  cop2c[{r}]: ours=0x{:08x} redux=0x{:08x}",
                    a.cop2_ctl[r], b.cop2_ctl[r]
                ));
            }
        }
        return Some(out);
    }
    if ours.len() != theirs.len() {
        return Some(format!(
            "exact trace length mismatch after step {start}: ours={} redux={}",
            ours.len(),
            theirs.len()
        ));
    }
    None
}

fn compare_visual(
    redux: &mut ReduxProcess,
    bus: &Bus,
    game_dir: &Path,
    label: &str,
) -> Result<VisualDiff, String> {
    let redux_bin = game_dir.join(format!("{label}_redux.bin"));
    redux
        .screenshot_save(&redux_bin, Duration::from_secs(60))
        .map_err(|e| format!("Redux screenshot: {e}"))?;
    let redux_bytes = fs::read(&redux_bin).map_err(|e| format!("read Redux screenshot: {e}"))?;
    let meta = fs::read_to_string(format!("{}.txt", redux_bin.display())).unwrap_or_default();
    let (rw, rh, _redux_bpp, _redux_len) = parse_screenshot_meta(&meta);

    let (_hash, ow, oh, len) = bus.gpu.display_hash();
    let our_bytes = read_display_bytes(bus, ow, oh);
    assert_eq!(our_bytes.len(), len);
    let our_bin = game_dir.join(format!("{label}_ours.bin"));
    fs::write(&our_bin, &our_bytes).map_err(|e| format!("write ours screenshot: {e}"))?;

    let our_stride = bytes_per_pixel(&our_bytes, ow, oh);
    let redux_stride = bytes_per_pixel(&redux_bytes, rw, rh);
    let (diff_bytes, first_diff, compared_bytes) = compare_framebuffers(
        &our_bytes,
        ow,
        oh,
        our_stride,
        &redux_bytes,
        rw,
        rh,
        redux_stride,
    );
    if ow == rw && oh == rh && our_stride == redux_stride {
        write_mask_ppm(
            &game_dir.join(format!("{label}_mask.ppm")),
            ow as usize,
            oh as usize,
            our_stride,
            &our_bytes,
            &redux_bytes,
        )?;
    }

    Ok(VisualDiff {
        ours_size: (ow, oh),
        redux_size: (rw, rh),
        diff_bytes,
        compared_bytes,
        first_diff,
    })
}

fn compare_framebuffers(
    ours: &[u8],
    ow: u32,
    oh: u32,
    ours_stride: usize,
    redux: &[u8],
    rw: u32,
    rh: u32,
    redux_stride: usize,
) -> (usize, Option<(u32, u32)>, usize) {
    let w = ow.min(rw);
    let h = oh.min(rh);
    let stride = ours_stride.min(redux_stride).max(1);
    let mut diffs = 0usize;
    let mut first = None;
    for y in 0..h {
        let o_row = y as usize * ow as usize * ours_stride;
        let r_row = y as usize * rw as usize * redux_stride;
        for x in 0..w as usize {
            let o_px = o_row + x * ours_stride;
            let r_px = r_row + x * redux_stride;
            for b in 0..stride {
                let o = ours.get(o_px + b).copied();
                let r = redux.get(r_px + b).copied();
                if o != r {
                    diffs += 1;
                    if first.is_none() {
                        first = Some((x as u32, y));
                    }
                }
            }
        }
    }
    let compared = w as usize * h as usize * stride;
    let max_len = ours.len().max(redux.len());
    if max_len > compared {
        diffs += max_len - compared;
    }
    (diffs, first, compared)
}

fn read_display_bytes(bus: &Bus, w: u32, h: u32) -> Vec<u8> {
    let da = bus.gpu.display_area();
    if da.bpp24 {
        let mut out = Vec::with_capacity((w * h * 3) as usize);
        for dy in 0..h as u16 {
            for dx in 0..w as u16 {
                let x = da.x.wrapping_add(dx);
                out.push(display_24bpp_byte(bus, x, da.y.wrapping_add(dy), 0));
                out.push(display_24bpp_byte(bus, x, da.y.wrapping_add(dy), 1));
                out.push(display_24bpp_byte(bus, x, da.y.wrapping_add(dy), 2));
            }
        }
        out
    } else {
        let mut out = Vec::with_capacity((w * h * 2) as usize);
        for dy in 0..h as u16 {
            for dx in 0..w as u16 {
                let pixel = bus.gpu.vram.get_pixel(da.x + dx, da.y + dy);
                out.extend_from_slice(&pixel.to_le_bytes());
            }
        }
        out
    }
}

fn display_24bpp_byte(bus: &Bus, x: u16, y: u16, channel_offset: u32) -> u8 {
    let byte_x = x as u32 * 3 + channel_offset;
    let word_x = (byte_x / 2) as u16;
    let word = bus.gpu.vram.get_pixel(word_x, y);
    if byte_x & 1 == 0 {
        (word & 0x00ff) as u8
    } else {
        (word >> 8) as u8
    }
}

fn parse_screenshot_meta(meta: &str) -> (u32, u32, u32, usize) {
    let mut w = 0u32;
    let mut h = 0u32;
    let mut bpp = 0u32;
    let mut len = 0usize;
    for tok in meta.split_whitespace() {
        if let Some(v) = tok.strip_prefix("w=") {
            w = v.parse().unwrap_or(0);
        } else if let Some(v) = tok.strip_prefix("h=") {
            h = v.parse().unwrap_or(0);
        } else if let Some(v) = tok.strip_prefix("bpp=") {
            bpp = v.parse().unwrap_or(0);
        } else if let Some(v) = tok.strip_prefix("len=") {
            len = v.parse().unwrap_or(0);
        }
    }
    (w, h, bpp, len)
}

fn bytes_per_pixel(bytes: &[u8], w: u32, h: u32) -> usize {
    let pixels = w as usize * h as usize;
    if pixels == 0 {
        1
    } else {
        (bytes.len() / pixels).max(1)
    }
}

fn write_mask_ppm(
    path: &Path,
    w: usize,
    h: usize,
    bytes_per_pixel: usize,
    ours: &[u8],
    redux: &[u8],
) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    writeln!(file, "P6\n{w} {h}\n255").map_err(|e| e.to_string())?;
    let mut buf = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * bytes_per_pixel;
            let a = &ours[off..off + bytes_per_pixel];
            let b = &redux[off..off + bytes_per_pixel];
            if a != b {
                buf.extend_from_slice(&[0xFF, 0, 0]);
            } else if bytes_per_pixel == 2 {
                let pixel = u16::from_le_bytes([a[0], a[1]]);
                let (r, g, bl) = bgr15_to_rgb(pixel);
                buf.extend_from_slice(&[r / 3, g / 3, bl / 3]);
            } else if bytes_per_pixel == 3 {
                buf.extend_from_slice(&[a[0] / 3, a[1] / 3, a[2] / 3]);
            } else {
                let v = a[0] / 3;
                buf.extend_from_slice(&[v, v, v]);
            }
        }
    }
    file.write_all(&buf).map_err(|e| e.to_string())
}

fn bgr15_to_rgb(pixel: u16) -> (u8, u8, u8) {
    let r5 = (pixel & 0x1F) as u8;
    let g5 = ((pixel >> 5) & 0x1F) as u8;
    let b5 = ((pixel >> 10) & 0x1F) as u8;
    (
        (r5 << 3) | (r5 >> 2),
        (g5 << 3) | (g5 >> 2),
        (b5 << 3) | (b5 >> 2),
    )
}

fn game_name(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("game");
    stem.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn print_game_result(result: &GameResult) {
    println!(
        "  cpu={} visual={}",
        if result.cpu_ok { "OK" } else { "FAIL" },
        if result.visual_ok { "OK" } else { "FAIL" },
    );
    if let Some(m) = result.first_checkpoint_mismatch {
        println!(
            "  first coarse mismatch: window=({}, {}] ours tick={} pc=0x{:08x} state={:016x}; redux tick={} pc=0x{:08x} state={:016x}",
            m.previous_step,
            m.checkpoint.step,
            m.ours.tick,
            m.ours.pc,
            m.ours.state_hash,
            m.redux.tick,
            m.redux.pc,
            m.redux.state_hash,
        );
    }
    if let Some(diff) = &result.exact_mismatch {
        for line in diff.lines().take(8) {
            println!("  {line}");
        }
    }
    if let Some(v) = &result.visual_diff {
        println!(
            "  visual: ours={}x{} redux={}x{} diff={}/{} first={}",
            v.ours_size.0,
            v.ours_size.1,
            v.redux_size.0,
            v.redux_size.1,
            v.diff_bytes,
            v.compared_bytes,
            v.first_diff
                .map(|(x, y)| format!("({x},{y})"))
                .unwrap_or_else(|| "-".to_string())
        );
    }
}

fn write_game_summary(dir: &Path, result: &GameResult) {
    let path = dir.join("SUMMARY.txt");
    let mut file = fs::File::create(&path).expect("create game summary");
    writeln!(file, "game: {}", result.name).unwrap();
    writeln!(file, "disc: {}", result.disc.display()).unwrap();
    writeln!(file, "cpu_ok: {}", result.cpu_ok).unwrap();
    writeln!(file, "visual_ok: {}", result.visual_ok).unwrap();
    if let Some(m) = result.first_checkpoint_mismatch {
        writeln!(
            file,
            "first_checkpoint_mismatch: window=({}, {}] ours={{tick:{} pc:0x{:08x} state:{:016x}}} redux={{tick:{} pc:0x{:08x} state:{:016x}}}",
            m.previous_step,
            m.checkpoint.step,
            m.ours.tick,
            m.ours.pc,
            m.ours.state_hash,
            m.redux.tick,
            m.redux.pc,
            m.redux.state_hash,
        )
        .unwrap();
    }
    if let Some(diff) = &result.exact_mismatch {
        writeln!(file, "\n{diff}").unwrap();
    }
    if let Some(v) = &result.visual_diff {
        writeln!(
            file,
            "visual: ours={}x{} redux={}x{} diff={}/{} first={}",
            v.ours_size.0,
            v.ours_size.1,
            v.redux_size.0,
            v.redux_size.1,
            v.diff_bytes,
            v.compared_bytes,
            v.first_diff
                .map(|(x, y)| format!("({x},{y})"))
                .unwrap_or_else(|| "-".to_string())
        )
        .unwrap();
    }
}

fn write_index_summary(cfg: &Config, results: &[GameResult]) {
    let path = cfg.report_dir.join("SUMMARY.txt");
    let mut file = fs::File::create(&path).expect("create index summary");
    writeln!(file, "steps: {}", cfg.steps).unwrap();
    writeln!(file, "interval: {}", cfg.interval).unwrap();
    writeln!(file, "").unwrap();
    writeln!(file, "{:<44} {:<6} {:<6} disc", "game", "cpu", "visual").unwrap();
    for r in results {
        writeln!(
            file,
            "{:<44} {:<6} {:<6} {}",
            r.name,
            if r.cpu_ok { "ok" } else { "FAIL" },
            if r.visual_ok { "ok" } else { "FAIL" },
            r.disc.display(),
        )
        .unwrap();
    }
    println!("summary: {}", path.display());
}
