//! Boot Tekken 3 to character select with input injection and dump
//! the visible display at intervals. Runs without Redux so it's fast
//! enough to iterate on rendering bugs that only manifest deep into
//! gameplay (the user's reported "garbled portraits" bug).
//!
//! The probe doesn't try to read game memory to detect the screen --
//! it just runs for `--steps` user-instructions and dumps PPMs every
//! `--dump-every` cycles so a human can scrub through them.
//!
//! ```bash
//! PSOXIDE_DISC="/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue" \
//! cargo run -p emulator-core --example probe_tekken_charsel --release -- \
//!   --steps 3000000000 --dump-every 100000000 --out-dir /tmp/tekken
//! ```
//!
//! Pad pulses use the same shape as `disc_vram_parity`:
//! `<mask>@<vblank>+<frames>`, e.g. `0x0008@200+5,0x4000@500+1` to
//! press START at vblank 200 (held 5 frames) and CROSS at vblank 500.

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{
    fast_boot_disc_with_hle, warm_bios_for_disc_fast_boot, Bus, Cpu, DISC_FAST_BOOT_WARMUP_STEPS,
};
use std::io::Write;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

fn parse_u16_mask(s: &str) -> Option<u16> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        s.parse().ok()
    }
}

fn parse_pad_pulses(text: &str) -> Vec<PadPulse> {
    let mut out = Vec::new();
    for entry in text.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (mask_text, rest) = entry.split_once('@').expect("pulse needs @");
        let mask = parse_u16_mask(mask_text).expect("bad mask");
        let (start_text, frames_text) = match rest.split_once('+') {
            Some((s, f)) => (s.trim(), f.trim()),
            None => (rest.trim(), "1"),
        };
        out.push(PadPulse {
            mask,
            start_vblank: start_text.parse().expect("bad vblank"),
            frames: frames_text.parse().expect("bad frames"),
        });
    }
    out
}

fn effective_mask(base: u16, pulses: &[PadPulse], current_vblank: u64) -> u16 {
    let mut mask = base;
    for p in pulses {
        if current_vblank >= p.start_vblank && current_vblank < p.start_vblank + p.frames {
            mask |= p.mask;
        }
    }
    mask
}

fn main() {
    let mut steps: u64 = 3_000_000_000;
    let mut dump_every: u64 = 100_000_000;
    let mut out_dir = PathBuf::from("/tmp/tekken_charsel");
    let mut held_mask: u16 = 0;
    let mut pulses: Vec<PadPulse> = Vec::new();
    let mut fastboot = true;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--steps" => steps = it.next().unwrap().parse().unwrap(),
            "--dump-every" => dump_every = it.next().unwrap().parse().unwrap(),
            "--out-dir" => out_dir = PathBuf::from(it.next().unwrap()),
            "--pad-mask" => held_mask = parse_u16_mask(&it.next().unwrap()).unwrap(),
            "--pad-pulses" => pulses = parse_pad_pulses(&it.next().unwrap()),
            "--no-fastboot" => fastboot = false,
            other => panic!("unknown argument: {other}"),
        }
    }

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let disc_path = std::env::var("PSOXIDE_DISC").unwrap_or_else(|_| {
        "/Users/ebonura/Downloads/ps1 games/Tekken 3 (USA)/Tekken 3 (USA).cue".into()
    });

    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let bios = std::fs::read(&bios_path).expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();
    let disc =
        disc_support::load_disc_path(std::path::Path::new(&disc_path)).expect("disc readable");

    if fastboot {
        warm_bios_for_disc_fast_boot(&mut bus, &mut cpu, DISC_FAST_BOOT_WARMUP_STEPS).unwrap();
        let info = fast_boot_disc_with_hle(&mut bus, &mut cpu, &disc, false).expect("fast boot");
        eprintln!(
            "[probe] fastboot {} entry=0x{:08x}",
            info.boot_path, info.initial_pc
        );
    }
    bus.cdrom.insert_disc(Some(disc));
    bus.attach_digital_pad_port1();

    let started = std::time::Instant::now();
    let mut next_dump = dump_every;
    let mut current_mask = u16::MAX;
    let mut last_log_step = 0u64;
    let mut last_log_time = started;
    for i in 0..steps {
        // Refresh pad mask once per VBlank-ish (cheaper than every step).
        if i & 0x1FFF == 0 {
            let vblank = bus.irq().raise_counts()[0];
            let mask = effective_mask(held_mask, &pulses, vblank);
            if mask != current_mask {
                bus.set_port1_buttons(emulator_core::ButtonState::from_bits(mask));
                current_mask = mask;
            }
        }

        if cpu.step(&mut bus).is_err() {
            eprintln!("[probe] CPU error at step {i}");
            break;
        }

        if i >= next_dump {
            let path = out_dir.join(format!("step_{:012}.ppm", i));
            dump_visible(&bus, &path).expect("dump");
            let vram_path = out_dir.join(format!("vram_{:012}.ppm", i));
            dump_full_vram(&bus, &vram_path).expect("vram");
            let now = std::time::Instant::now();
            let mips =
                (i - last_log_step) as f64 / now.duration_since(last_log_time).as_secs_f64() / 1e6;
            eprintln!(
                "[probe] step {i:>12}  vblank={:>5}  pad=0x{current_mask:04x}  dumped {} ({:.1} MIPS)",
                bus.irq().raise_counts()[0],
                path.display(),
                mips,
            );
            last_log_step = i;
            last_log_time = now;
            next_dump += dump_every;
        }
    }
    let elapsed = started.elapsed();
    eprintln!(
        "[probe] done in {:.1}s ({:.1} MIPS avg)",
        elapsed.as_secs_f64(),
        steps as f64 / elapsed.as_secs_f64() / 1e6,
    );
}

fn dump_visible(bus: &Bus, path: &std::path::Path) -> std::io::Result<()> {
    let (rgba, w, h) = bus.gpu.display_rgba8();
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "P6\n{w} {h}\n255")?;
    for px in rgba.chunks_exact(4) {
        f.write_all(&px[..3])?;
    }
    Ok(())
}

fn dump_full_vram(bus: &Bus, path: &std::path::Path) -> std::io::Result<()> {
    let w = emulator_core::VRAM_WIDTH;
    let h = emulator_core::VRAM_HEIGHT;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "P6\n{w} {h}\n255")?;
    for &pix in bus.gpu.vram.words() {
        let r5 = (pix & 0x1F) as u8;
        let g5 = ((pix >> 5) & 0x1F) as u8;
        let b5 = ((pix >> 10) & 0x1F) as u8;
        f.write_all(&[
            (r5 << 3) | (r5 >> 2),
            (g5 << 3) | (g5 >> 2),
            (b5 << 3) | (b5 >> 2),
        ])?;
    }
    Ok(())
}
