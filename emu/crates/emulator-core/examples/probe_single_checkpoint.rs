//! Run N silent steps in both emulators once, compare final cycles.
//! For picking a reference point on a new game without a trace cache.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_single_checkpoint -- 100000000 "/path/to/game.bin"
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);
    let disc_path = std::env::args().nth(2);
    let held_buttons = std::env::var("PSOXIDE_PAD1")
        .ok()
        .and_then(|s| parse_u16_mask(&s))
        .unwrap_or(0);
    let pad_pulses = std::env::var("PSOXIDE_PAD1_PULSES")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            parse_pad_pulses(&s).unwrap_or_else(|| {
                panic!(
                    "PSOXIDE_PAD1_PULSES must be comma-separated \
                     <mask>@<start_vblank>+<frames> entries"
                )
            })
        })
        .unwrap_or_default();
    let wants_pad = held_buttons != 0 || !pad_pulses.is_empty();
    let peek_addrs = std::env::var("PSOXIDE_PEEK32")
        .ok()
        .map(|s| parse_addr_list(&s))
        .unwrap_or_default();

    let bios_path = PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN");

    // --- Ours ---
    eprintln!("[ours] running {n} steps...");
    let t0 = Instant::now();
    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    if wants_pad {
        bus.attach_digital_pad_port1();
    }
    let mut cpu = Cpu::new();
    let mut current_pad_mask = None;
    if wants_pad {
        sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
    }
    for _ in 0..n {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if wants_pad {
            sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
        }
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
                if wants_pad {
                    sync_pad_mask(&mut bus, held_buttons, &pad_pulses, &mut current_pad_mask);
                }
            }
        }
    }
    let our_cycles = bus.cycles();
    eprintln!(
        "[ours] done in {:.1}s, cycles = {our_cycles}",
        t0.elapsed().as_secs_f64()
    );

    // --- Redux silent run ---
    eprintln!("[redux] launching...");
    let t0 = Instant::now();
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let mut config = OracleConfig::new(bios_path, lua).expect("Redux resolves");
    if let Some(ref p) = disc_path {
        config = config.with_disc(PathBuf::from(p));
    }
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(Duration::from_secs(30)).expect("handshake");
    eprintln!("[redux] running {n} steps silently...");
    // Generous timeout: ~4 min per 100M steps at Redux's silent rate.
    let run_timeout = Duration::from_secs((n / 200_000).max(60));
    let redux_tick = if wants_pad {
        let pulse_tuples = pad_pulses
            .iter()
            .map(|pulse| (pulse.mask, pulse.start_vblank, pulse.frames))
            .collect::<Vec<_>>();
        redux
            .run_checkpoint_pad(
                n,
                n.max(1),
                1,
                held_buttons,
                &pulse_tuples,
                run_timeout,
                |_step, _tick, _pc| Ok(()),
            )
            .expect("run_checkpoint_pad")
    } else {
        redux.run(n, run_timeout).expect("run")
    };
    eprintln!(
        "[redux] done in {:.1}s, tick = {redux_tick}",
        t0.elapsed().as_secs_f64()
    );
    let redux_peeks = peek_addrs
        .iter()
        .map(|&addr| {
            (
                addr,
                redux
                    .peek32(addr, Duration::from_secs(5))
                    .unwrap_or(0xFFFF_FFFF),
            )
        })
        .collect::<Vec<_>>();
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(5));
    let _ = redux.terminate();

    // --- Compare ---
    let delta = our_cycles as i64 - redux_tick as i64;
    let pct = 100.0 * delta as f64 / (redux_tick as f64);
    println!();
    println!(
        "=== {n}-step parity ({}) ===",
        disc_path.as_deref().unwrap_or("no-disc")
    );
    println!("our cycles : {our_cycles:>14}");
    println!("redux tick : {redux_tick:>14}");
    println!("delta      : {delta:>+14}");
    println!("pct_off    : {pct:>+14.3}%");
    if !peek_addrs.is_empty() {
        println!();
        println!("peek32:");
        for (idx, &addr) in peek_addrs.iter().enumerate() {
            let ours = bus.read32(addr);
            let redux = redux_peeks[idx].1;
            println!("  0x{addr:08x}  ours=0x{ours:08x}  redux=0x{redux:08x}");
        }
    }
}

fn parse_u16_mask(text: &str) -> Option<u16> {
    let s = text.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u16>().ok()
    }
}

fn parse_pad_pulses(text: &str) -> Option<Vec<PadPulse>> {
    let mut pulses = Vec::new();
    for entry in text.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        pulses.push(parse_pad_pulse(entry)?);
    }
    Some(pulses)
}

fn parse_pad_pulse(text: &str) -> Option<PadPulse> {
    let (mask_text, rest) = text.split_once('@')?;
    let mask = parse_u16_mask(mask_text)?;
    let (start_text, frames_text) = match rest.split_once('+') {
        Some((start, frames)) => (start.trim(), frames.trim()),
        None => (rest.trim(), "1"),
    };
    let start_vblank = start_text.parse().ok()?;
    let frames = frames_text.parse().ok()?;
    if frames == 0 {
        return None;
    }
    Some(PadPulse {
        mask,
        start_vblank,
        frames,
    })
}

fn effective_pad_mask(base_mask: u16, pulses: &[PadPulse], current_vblank: u64) -> u16 {
    let mut mask = base_mask;
    for pulse in pulses {
        let end_vblank = pulse.start_vblank.saturating_add(pulse.frames);
        if current_vblank >= pulse.start_vblank && current_vblank < end_vblank {
            mask |= pulse.mask;
        }
    }
    mask
}

fn sync_pad_mask(
    bus: &mut Bus,
    base_mask: u16,
    pulses: &[PadPulse],
    current_mask: &mut Option<u16>,
) {
    let next_mask = effective_pad_mask(base_mask, pulses, bus.irq().raise_counts()[0]);
    if current_mask.is_some_and(|mask| mask == next_mask) {
        return;
    }
    bus.set_port1_buttons(emulator_core::ButtonState::from_bits(next_mask));
    *current_mask = Some(next_mask);
}

fn parse_addr_list(text: &str) -> Vec<u32> {
    text.split(',')
        .filter_map(|entry| {
            let s = entry.trim();
            if s.is_empty() {
                return None;
            }
            u32::from_str_radix(s.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()
        })
        .collect()
}
