//! Boot a retail disc with a pad attached, run for a while, then dump
//! the recent SIO0 controller transactions.
//!
//! Built for the exact "SDK demos see input, commercial games don't"
//! investigation: it tells us whether a game is raw-polling with
//! `0x42`, doing the full DualShock config dance (`0x43`/`0x44`/`0x45`
//! / `0x46` / `0x47` / `0x4C` / `0x4D`), or never getting a sane pad
//! response at all.
//!
//! Supports two input modes:
//! - `PSOXIDE_PAD1=0x0008` keeps a mask held for the whole run.
//! - `PSOXIDE_PAD1_PULSES='0x0008@1200+4,0x4000@1250+1'` presses one
//!   or more masks for a fixed number of VBlanks starting at the given
//!   VBlank count. Format per entry: `<mask>@<start_vblank>+<frames>`.
//!
//! Best used with MMIO tracing enabled:
//!
//! ```bash
//! PSOXIDE_DISC="/path/to/game.bin" \
//! cargo run -p emulator-core --example probe_disc_pad_trace \
//!   --features emulator-core/trace-mmio --release -- 120000000
//! ```

use emulator_core::{Bus, Cpu};
use psx_iso::Disc;
use std::path::PathBuf;

#[cfg(feature = "trace-mmio")]
use emulator_core::mmio_trace::MmioEntry;
#[cfg(feature = "trace-mmio")]
use emulator_core::{MmioKind, Sio0};
#[cfg(feature = "trace-mmio")]
use std::collections::BTreeMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PadMaskChange {
    vblank: u64,
    cycles: u64,
    mask: u16,
}

fn main() {
    let steps: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(120_000_000);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let disc_path = std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA) (Track 1).bin".into()
    });
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

    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let disc = std::fs::read(&disc_path).expect("disc readable");

    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(Disc::from_bin(disc)));
    bus.attach_digital_pad_port1();

    let mut cpu = Cpu::new();
    let mut cycles_at_last_pump = 0u64;
    let mut current_pad_mask = None;
    let mut pad_mask_changes = Vec::new();
    sync_pad_mask(
        &mut bus,
        held_buttons,
        &pad_pulses,
        &mut current_pad_mask,
        &mut pad_mask_changes,
    );
    for _ in 0..steps {
        if cpu.step(&mut bus).is_err() {
            break;
        }
        sync_pad_mask(
            &mut bus,
            held_buttons,
            &pad_pulses,
            &mut current_pad_mask,
            &mut pad_mask_changes,
        );
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }
    }

    let (display_hash, w, h, _) = bus.gpu.display_hash();
    println!("disc: {disc_path}");
    println!("steps:      {steps}");
    println!("pad mask:   0x{held_buttons:04x}");
    println!("pad pulses: {}", format_pad_pulses(&pad_pulses));
    println!("cpu.tick:   {}", cpu.tick());
    println!("bus.cycles: {}", bus.cycles());
    println!("final pc:   0x{:08x}", cpu.pc());
    println!("vblank:     {}", bus.irq().raise_counts()[0]);
    println!("display:    {w}x{h}  hash=0x{display_hash:016x}");
    dump_pad_mask_changes(&pad_mask_changes);
    dump_pad_histogram(&bus);

    #[cfg(feature = "trace-mmio")]
    dump_trace(&bus);
    #[cfg(not(feature = "trace-mmio"))]
    println!("\nMMIO tracing is disabled. Re-run with --features emulator-core/trace-mmio.");
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

fn format_pad_pulses(pulses: &[PadPulse]) -> String {
    if pulses.is_empty() {
        return "(none)".into();
    }
    pulses
        .iter()
        .map(|pulse| {
            format!(
                "0x{:04x}@{}+{}",
                pulse.mask, pulse.start_vblank, pulse.frames
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
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
    changes: &mut Vec<PadMaskChange>,
) {
    let vblank = bus.irq().raise_counts()[0];
    let next_mask = effective_pad_mask(base_mask, pulses, vblank);
    if current_mask.is_some_and(|mask| mask == next_mask) {
        return;
    }

    bus.set_port1_buttons(emulator_core::ButtonState::from_bits(next_mask));
    if current_mask.is_some() || next_mask != 0 {
        changes.push(PadMaskChange {
            vblank,
            cycles: bus.cycles(),
            mask: next_mask,
        });
    }
    *current_mask = Some(next_mask);
}

fn dump_pad_mask_changes(changes: &[PadMaskChange]) {
    println!("\n=== Pad-1 mask changes ===");
    if changes.is_empty() {
        println!("  (none)");
        return;
    }
    for change in changes {
        println!(
            "  vblank={}  cycles={}  mask=0x{:04x}",
            change.vblank, change.cycles, change.mask
        );
    }
}

fn dump_pad_histogram(bus: &Bus) {
    println!("\n=== Port-1 pad command histogram ===");
    match bus.port1_pad_command_histogram() {
        Some(hist) => {
            let mut any = false;
            for (cmd, &count) in hist.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                any = true;
                println!("  cmd 0x{cmd:02x}: {count}");
            }
            if !any {
                println!("  (no controller commands observed)");
            }
        }
        None => println!("  (no pad attached)"),
    }

    println!("\n=== Port-1 first-byte histogram ===");
    let mut any = false;
    for (byte, &count) in bus.port1_first_byte_histogram().iter().enumerate() {
        if count == 0 {
            continue;
        }
        any = true;
        println!("  first 0x{byte:02x}: {count}");
    }
    if !any {
        println!("  (no transactions observed)");
    }

    let recent = bus.port1_pad_recent_commands();
    println!("\n=== Recent port-1 pad commands ===");
    if recent.is_empty() {
        println!("  (none)");
    } else {
        let cmds = recent
            .iter()
            .map(|cmd| format!("0x{cmd:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {cmds}");
    }

    let recent_first = bus.port1_recent_first_bytes();
    println!("\n=== Recent port-1 first bytes ===");
    if recent_first.is_empty() {
        println!("  (none)");
    } else {
        let bytes = recent_first
            .iter()
            .map(|b| format!("0x{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {bytes}");
    }

    let polls = bus.port1_recent_polls();
    println!("\n=== Recent port-1 0x42 polls ===");
    if polls.is_empty() {
        println!("  (none)");
    } else {
        for poll in polls {
            let tx = poll
                .tx
                .iter()
                .take(poll.len as usize)
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            let rx = poll
                .rx
                .iter()
                .take(poll.len as usize)
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "  {}  tx=[{tx}]  rx=[{rx}]",
                if poll.complete {
                    "complete"
                } else {
                    "partial "
                }
            );
        }
    }
}

#[cfg(feature = "trace-mmio")]
fn dump_trace(bus: &Bus) {
    let entries: Vec<_> = bus
        .mmio_trace
        .iter_chronological()
        .filter(|e| Sio0::contains(e.addr))
        .collect();
    if entries.is_empty() {
        println!("\nNo SIO0 MMIO accesses recorded.");
        return;
    }

    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|e| match (e.addr - Sio0::BASE, e.kind) {
            (0x0, MmioKind::R8 | MmioKind::R16 | MmioKind::R32) => true,
            (0x0, MmioKind::W8 | MmioKind::W16 | MmioKind::W32) => true,
            (0xA, MmioKind::W16 | MmioKind::W32) => true,
            (0x8, MmioKind::W16 | MmioKind::W32) => true,
            (0xE, MmioKind::W16 | MmioKind::W32) => true,
            _ => false,
        })
        .collect();
    if filtered.is_empty() {
        println!("\nNo relevant SIO0 accesses captured.");
        return;
    }

    let txns = decode_transactions(&filtered);
    println!("\n=== SIO0 command histogram ===");
    let mut cmd_hist: BTreeMap<u8, u32> = BTreeMap::new();
    let mut id_hist: BTreeMap<u8, u32> = BTreeMap::new();
    for txn in &txns {
        if let Some(&cmd) = txn.tx_bytes.get(1) {
            *cmd_hist.entry(cmd).or_insert(0) += 1;
        }
        if let Some(&id) = txn.rx_bytes.first() {
            *id_hist.entry(id).or_insert(0) += 1;
        }
    }
    for (cmd, count) in &cmd_hist {
        println!("  cmd 0x{cmd:02x}: {count}");
    }

    println!("\n=== SIO0 ID histogram ===");
    for (id, count) in &id_hist {
        println!("  id  0x{id:02x}: {count}");
    }

    println!("\n=== Last 24 controller transactions ===");
    let skip = txns.len().saturating_sub(24);
    for txn in &txns[skip..] {
        println!(
            "  cyc={:>12}  ctrl={:#06x}  {}",
            txn.start_cycle,
            txn.ctrl,
            format_transaction(txn),
        );
    }

    println!("\n=== Last 48 relevant SIO0 accesses ===");
    let skip = filtered.len().saturating_sub(48);
    for e in &filtered[skip..] {
        println!(
            "  cyc={:>12}  {}  {:08x}  {:08x}",
            e.cycle,
            e.kind.tag(),
            e.addr,
            e.value
        );
    }
}

#[cfg(feature = "trace-mmio")]
#[derive(Default)]
struct Transaction {
    start_cycle: u64,
    ctrl: u16,
    tx_bytes: Vec<u8>,
    rx_bytes: Vec<u8>,
}

#[cfg(feature = "trace-mmio")]
fn decode_transactions(entries: &[&MmioEntry]) -> Vec<Transaction> {
    let mut txns = Vec::new();
    let mut current = Transaction::default();
    let mut joyn_selected = false;

    for e in entries {
        match (e.addr - Sio0::BASE, e.kind) {
            (0xA, MmioKind::W16 | MmioKind::W32) => {
                let ctrl = e.value as u16;
                let new_joyn = ctrl & (1 << 1) != 0;
                if joyn_selected && !new_joyn && !current.tx_bytes.is_empty() {
                    txns.push(std::mem::take(&mut current));
                }
                if new_joyn && !joyn_selected {
                    current = Transaction {
                        start_cycle: e.cycle,
                        ctrl,
                        ..Transaction::default()
                    };
                } else if new_joyn {
                    current.ctrl = ctrl;
                }
                joyn_selected = new_joyn;
            }
            (0x0, MmioKind::W8 | MmioKind::W16 | MmioKind::W32) => {
                if current.tx_bytes.is_empty() {
                    current.start_cycle = e.cycle;
                }
                current.tx_bytes.push(e.value as u8);
            }
            (0x0, MmioKind::R8 | MmioKind::R16 | MmioKind::R32) => {
                current.rx_bytes.push(e.value as u8);
            }
            _ => {}
        }
    }

    if !current.tx_bytes.is_empty() {
        txns.push(current);
    }
    txns
}

#[cfg(feature = "trace-mmio")]
fn format_transaction(txn: &Transaction) -> String {
    let mut out = String::new();
    let len = txn.tx_bytes.len().max(txn.rx_bytes.len());
    for i in 0..len {
        if i > 0 {
            out.push(' ');
        }
        let tx = txn
            .tx_bytes
            .get(i)
            .map(|b| format!(">{b:02x}"))
            .unwrap_or_else(|| ">--".into());
        let rx = txn
            .rx_bytes
            .get(i)
            .map(|b| format!("<{b:02x}"))
            .unwrap_or_else(|| "<--".into());
        out.push_str(&format!("{tx}/{rx}"));
    }
    out
}
