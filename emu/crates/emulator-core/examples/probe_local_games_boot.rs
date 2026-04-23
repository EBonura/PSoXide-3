//! Headless local-library boot sweep.
//!
//! This is the first-line compatibility tool for the local game set:
//! it discovers CUE sheets, mounts them through the frontend's CUE
//! loader, runs the BIOS/game for a fixed instruction budget, and
//! prints the shared signals we need to decide which emulator bug to
//! fix next.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_local_games_boot -- 300000000
//! ```

#[path = "support/disc.rs"]
mod disc_support;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use emulator_core::{Bus, Cpu};

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const DEFAULT_GAMES_DIR: &str = "/Users/ebonura/Downloads/ps1 games";
const DEFAULT_STEPS: u64 = 300_000_000;
const SAMPLE_STEPS: u64 = 10_000_000;
const SPU_PUMP_CYCLES: u64 = 560_000;
const SPU_FRAME_SAMPLES: usize = 735;
const SONY_LOGO_HASH: u64 = 0xa3ac_6881_0443_33d0;

#[derive(Default)]
struct BootResult {
    cpu_tick: u64,
    cycles: u64,
    pc: u32,
    display_hash: u64,
    display_size: (u32, u32),
    vram_nonzero_words: usize,
    cdrom_cmds: Vec<(u8, u32)>,
    cdrom_irq_counts: [u64; 6],
    data_fifo_pops: u64,
    data_fifo_len: usize,
    pending_cdrom_events: usize,
    cdrom_irq_flag: u8,
    cdrom_irq_mask: u8,
    cdrom_next_event: String,
    error: Option<String>,
    recent_pcs: VecDeque<u32>,
}

fn main() {
    let steps = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_STEPS);
    let games_root = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_GAMES_DIR));
    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS));

    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let cues = disc_support::discover_cue_files(&games_root).expect("discover CUE files");
    let total_games = cues.len();
    println!(
        "boot sweep: {} games, {} steps each, bios={}",
        total_games,
        steps,
        bios_path.display()
    );
    println!(
        "{:<42} {:>10} {:>12} {:>12} {:>10} {:>8} {:>8}  {}",
        "game", "signal", "pc", "hash", "cdcmds", "drdy", "pops", "tail pcs"
    );
    println!("{}", "-".repeat(128));

    let mut passed_logo = 0usize;
    let mut errored = 0usize;
    for cue in cues {
        let name = game_name(&cue);
        let result = run_one(&bios, &cue, steps);
        if result.error.is_some() {
            errored += 1;
        }
        let signal = boot_signal(&result);
        if signal == "passed-logo" {
            passed_logo += 1;
        }
        println!(
            "{:<42} {:>10}  0x{:08x}  0x{:016x} {:>10} {:>8} {:>8}  {}",
            truncate(&name, 42),
            signal,
            result.pc,
            result.display_hash,
            summarize_cmds(&result.cdrom_cmds),
            result.cdrom_irq_counts[1],
            result.data_fifo_pops,
            summarize_pcs(&result.recent_pcs),
        );
        println!(
            "  cycles={} tick={} display={}x{} vram_nz={} cd_irq(flag=0x{:02x},mask=0x{:02x}) fifo={} pending={} next={}{}",
            result.cycles,
            result.cpu_tick,
            result.display_size.0,
            result.display_size.1,
            result.vram_nonzero_words,
            result.cdrom_irq_flag,
            result.cdrom_irq_mask,
            result.data_fifo_len,
            result.pending_cdrom_events,
            result.cdrom_next_event,
            result
                .error
                .as_ref()
                .map(|e| format!(" error={e}"))
                .unwrap_or_default(),
        );
    }

    println!("{}", "-".repeat(128));
    println!(
        "summary: {passed_logo} passed-logo signal, {errored} errored, {total_games} total"
    );
}

fn run_one(bios: &[u8], cue: &Path, steps: u64) -> BootResult {
    let mut bus = match Bus::new(bios.to_vec()) {
        Ok(bus) => bus,
        Err(e) => {
            return BootResult {
                error: Some(format!("bus init: {e:?}")),
                ..BootResult::default()
            };
        }
    };
    match disc_support::load_disc_path(cue) {
        Ok(disc) => bus.cdrom.insert_disc(Some(disc)),
        Err(e) => {
            return BootResult {
                error: Some(format!("disc load: {e}")),
                ..BootResult::default()
            };
        }
    }

    let mut cpu = Cpu::new();
    let mut cycles_at_last_pump = 0u64;
    let mut recent_pcs = VecDeque::with_capacity(12);
    let mut error = None;

    for step in 0..steps {
        if step % SAMPLE_STEPS == 0 {
            push_recent_pc(&mut recent_pcs, cpu.pc());
        }
        if let Err(e) = cpu.step(&mut bus) {
            error = Some(format!("{e:?}"));
            break;
        }
        if bus.cycles().saturating_sub(cycles_at_last_pump) > SPU_PUMP_CYCLES {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(SPU_FRAME_SAMPLES);
            let _ = bus.spu.drain_audio();
        }
    }
    push_recent_pc(&mut recent_pcs, cpu.pc());

    let (display_hash, display_width, display_height, _) = bus.gpu.display_hash();
    let vram_nonzero_words = bus.gpu.vram.words().iter().filter(|&&w| w != 0).count();
    let cdrom_cmds = bus
        .cdrom
        .command_histogram()
        .iter()
        .enumerate()
        .filter_map(|(op, &count)| (count > 0).then_some((op as u8, count)))
        .collect();
    let cdrom_next_event = bus
        .cdrom
        .next_pending_event()
        .map(|(deadline, irq)| format!("{irq:?}@{deadline}"))
        .unwrap_or_else(|| "-".to_string());

    BootResult {
        cpu_tick: cpu.tick(),
        cycles: bus.cycles(),
        pc: cpu.pc(),
        display_hash,
        display_size: (display_width, display_height),
        vram_nonzero_words,
        cdrom_cmds,
        cdrom_irq_counts: bus.cdrom.irq_type_counts,
        data_fifo_pops: bus.cdrom.data_fifo_pops(),
        data_fifo_len: bus.cdrom.data_fifo_len(),
        pending_cdrom_events: bus.cdrom.pending_queue_len(),
        cdrom_irq_flag: bus.cdrom.irq_flag(),
        cdrom_irq_mask: bus.cdrom.irq_mask_value(),
        cdrom_next_event,
        error,
        recent_pcs,
    }
}

fn push_recent_pc(recent_pcs: &mut VecDeque<u32>, pc: u32) {
    if recent_pcs.len() == 12 {
        recent_pcs.pop_front();
    }
    recent_pcs.push_back(pc);
}

fn boot_signal(result: &BootResult) -> &'static str {
    if result.error.is_some() {
        "error"
    } else if result.display_hash != SONY_LOGO_HASH {
        "passed-logo"
    } else if result.cdrom_cmds.iter().any(|(op, _)| matches!(*op, 0x06 | 0x1b)) {
        "reading"
    } else {
        "sony-logo"
    }
}

fn summarize_cmds(cmds: &[(u8, u32)]) -> String {
    let total: u32 = cmds.iter().map(|(_, count)| *count).sum();
    let read_count: u32 = cmds
        .iter()
        .filter(|(op, _)| matches!(*op, 0x06 | 0x1b))
        .map(|(_, count)| *count)
        .sum();
    if read_count == 0 {
        total.to_string()
    } else {
        format!("{total}/{read_count}r")
    }
}

fn summarize_pcs(pcs: &VecDeque<u32>) -> String {
    pcs.iter()
        .map(|pc| format!("{pc:08x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn game_name(cue: &Path) -> String {
    if let Some(name) = cue.file_stem().and_then(|name| name.to_str()) {
        name.to_string()
    } else {
        cue.to_string_lossy().into_owned()
    }
}

fn truncate(s: &str, width: usize) -> String {
    if s.len() <= width {
        s.to_string()
    } else {
        let keep = width.saturating_sub(3);
        format!("{}...", &s[..keep])
    }
}
