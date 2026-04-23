//! Boot the BIOS with a disc inserted, run for N instructions, and
//! dump VRAM + high-level state. Used to work on Milestone D (BIOS
//! disc-check passes) and onwards.
//!
//! ```bash
//! PSOXIDE_DISC="/path/to/Crash.cue" \
//! PSOXIDE_VRAM_DUMP=/tmp/crash.ppm \
//! cargo run -p emulator-core --example boot_disc --release -- 500000000
//! ```
//!
//! Unlike `smoke_draw`, this mounts a disc image before the first
//! CPU step so the BIOS's CDROM GetID path reaches the
//! "licensed-disc + seek to SYSTEM.CNF" branch instead of the
//! "please insert disc" branch.

#[path = "support/disc.rs"]
mod disc_support;

use emulator_core::{Bus, Cpu};
use std::path::PathBuf;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000_000);

    let bios_path = std::env::var("PSOXIDE_BIOS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN"));
    let disc_path = std::env::var("PSOXIDE_DISC")
        .expect("Set PSOXIDE_DISC to the path of a CUE or BIN disc image");

    let bios = std::fs::read(&bios_path).expect("BIOS readable");
    let disc = disc_support::load_disc_path(std::path::Path::new(&disc_path)).expect("disc readable");
    eprintln!(
        "[boot_disc] bios={}  disc={}",
        bios_path.display(),
        disc_path,
    );

    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    let mut cpu = Cpu::new();

    let mut stopped_at: Option<(u64, emulator_core::ExecutionError)> = None;
    for i in 0..n {
        if let Err(e) = cpu.step(&mut bus) {
            stopped_at = Some((i, e));
            break;
        }
    }

    println!("=== boot_disc after {n} attempted steps ===");
    if let Some((i, e)) = stopped_at {
        println!("stopped at step  = {i}  ({e:?})");
    }
    println!("cpu.tick         = {}", cpu.tick());
    println!("bus.cycles       = {}", bus.cycles());
    println!("cpu.pc           = 0x{:08x}", cpu.pc());
    println!("cdrom commands   = {}", bus.cdrom.commands_dispatched());
    let hist = bus.cdrom.command_histogram();
    let nonzero: Vec<(u8, u32)> = hist
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if c > 0 { Some((i as u8, c)) } else { None })
        .collect();
    if !nonzero.is_empty() {
        println!(
            "cdrom cmd histogram: {:?}",
            nonzero
                .iter()
                .map(|(op, c)| format!("0x{op:02X}:{c}"))
                .collect::<Vec<_>>(),
        );
    }
    let exc = cpu.exception_counts();
    let exc_names: [&str; 32] = [
        "Int", "Mod", "TLBL", "TLBS", "AdEL", "AdES", "IBE", "DBE", "Syscall", "Break", "RI",
        "CpU", "Ov", "Tr", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
        "-", "-", "-", "-",
    ];
    print!("exceptions       =");
    let mut any = false;
    for (i, &n) in exc.iter().enumerate() {
        if n > 0 {
            print!(" {}:{n}", exc_names[i]);
            any = true;
        }
    }
    if !any {
        print!(" (none)");
    }
    println!();

    let irq = bus.irq();
    let raise_names = [
        "VBlank",
        "Gpu",
        "Cdrom",
        "Dma",
        "Timer0",
        "Timer1",
        "Timer2",
        "Controller",
        "Sio",
        "Spu",
        "Lightpen",
    ];
    print!("irq raises       =");
    for (i, &n) in irq.raise_counts().iter().enumerate() {
        if n > 0 {
            print!(" {}:{n}", raise_names[i]);
        }
    }
    println!();

    let vram = &bus.gpu.vram;
    let nz = vram.words().iter().filter(|&&w| w != 0).count();
    println!("VRAM non-zero    = {nz} / {} words", 1024 * 512);

    if let Ok(path) = std::env::var("PSOXIDE_VRAM_DUMP") {
        use std::io::Write;
        let w = emulator_core::VRAM_WIDTH;
        let h = emulator_core::VRAM_HEIGHT;
        let mut file = std::fs::File::create(&path).expect("create vram dump");
        writeln!(file, "P6\n{w} {h}\n255").unwrap();
        let mut rgb = Vec::with_capacity(w * h * 3);
        for &pix in vram.words() {
            let r5 = (pix & 0x1F) as u8;
            let g5 = ((pix >> 5) & 0x1F) as u8;
            let b5 = ((pix >> 10) & 0x1F) as u8;
            rgb.push((r5 << 3) | (r5 >> 2));
            rgb.push((g5 << 3) | (g5 >> 2));
            rgb.push((b5 << 3) | (b5 >> 2));
        }
        file.write_all(&rgb).unwrap();
        println!("VRAM dumped to   = {path}");
    }
}
