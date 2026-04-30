//! Headless CLI -- exercises every stateful path the GUI exposes
//! without opening a window.
//!
//! Existed for three reasons:
//!
//! 1. **Verification substrate.** Every feature added to the GUI
//!    should land first here as a subcommand -- then the UI is a
//!    thin layer over a tested CLI. "Does the game library scan
//!    find my games?" becomes a deterministic test instead of a
//!    click-test.
//! 2. **Regression scripts.** `frontend launch <game> --steps 100M
//!    --dump-hash` is a one-liner you can wrap in a shell test to
//!    pin BIOS / SDK behaviour without rebuilding the GUI.
//! 3. **CI.** No display server → `cargo test` on Linux boxes
//!    without Xvfb. The existing milestone tests already run
//!    headless; this extends the same principle to the
//!    user-facing features.
//!
//! When the frontend binary is run with a subcommand argument
//! (`scan`, `list`, `launch`, `info`), this module handles it and
//! returns -- `main()` never spins up winit/wgpu. Without a
//! subcommand, the GUI runs as normal.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use emulator_core::{
    fast_boot_disc_with_hle, spu::SAMPLE_CYCLES, warm_bios_for_disc_fast_boot, Bus, Cpu,
    DISC_FAST_BOOT_WARMUP_STEPS,
};
use psoxide_settings::{
    library::{GameKind, LibraryEntry},
    ConfigPaths, Library, Settings,
};
use psx_iso::{Disc, Exe};

use crate::app::resolve_bios_path;

/// Top-level argument parser. Passed to `clap::Parser::parse()`
/// from `main.rs`.
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to an alternate config directory (defaults to the
    /// platform config dir). Useful for portable installs and
    /// keeping tests from touching your real settings.
    #[arg(long, global = true)]
    pub config_dir: Option<PathBuf>,

    /// Launch the GUI in a regular floating window instead of the
    /// default borderless-fullscreen mode. Useful when developing
    /// with the editor side-by-side with a terminal or docs. Only
    /// meaningful when no headless subcommand is given --
    /// subcommands always run windowless.
    #[arg(long)]
    pub windowed: bool,

    /// Run the experimental compute-shader rasterizer in parallel
    /// with the CPU rasterizer (Phase C). Per-frame the frontend
    /// drains the CPU's `cmd_log` and replays each GP0 packet
    /// through the GPU compute path. Off by default -- opt-in until
    /// parity is confirmed in a wide enough test set. Press F12 in
    /// the GUI to toggle at runtime once the bus is wired up.
    #[arg(long)]
    pub gpu_compute: bool,

    /// Headless subcommand. Omit to launch the GUI.
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Every headless operation the frontend exposes. Add new variants
/// as UI features are built so each one has a scriptable
/// equivalent.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Print resolved config paths + effective settings values.
    Info,
    /// Walk the configured (or given) library root and refresh the
    /// on-disk game cache.
    Scan(ScanArgs),
    /// Print the cached library contents (one line per game).
    List,
    /// Run the emulator headlessly on a specific game or EXE and
    /// emit final state info.
    Launch(LaunchArgs),
}

/// Arguments for `scan`.
#[derive(Debug, Args)]
pub struct ScanArgs {
    /// Library root to scan. Overrides `settings.paths.game_library`
    /// if set; otherwise uses the configured value.
    #[arg(long)]
    pub root: Option<PathBuf>,
}

/// Arguments for `launch`.
#[derive(Debug, Args)]
pub struct LaunchArgs {
    /// Path to a `.bin` (disc image) or `.exe` (homebrew) to run.
    /// Either this or `--game-id` must be provided.
    #[arg(long)]
    pub path: Option<PathBuf>,
    /// Alternative to `--path`: pick a game from the cached library
    /// by its stable ID (16-hex-char fingerprint).
    #[arg(long)]
    pub game_id: Option<String>,
    /// Number of CPU instructions to retire before stopping.
    #[arg(long, default_value_t = 100_000_000)]
    pub steps: u64,
    /// Force the real BIOS disc boot path instead of direct
    /// SYSTEM.CNF fast boot.
    #[arg(long)]
    pub bios_boot: bool,
    /// Print an FNV-1a-64 VRAM hash at the end. Same algorithm the
    /// milestone regression tests use, so a CLI run + a unit test
    /// should produce identical numbers.
    #[arg(long)]
    pub dump_hash: bool,
    /// Optional path to dump the final VRAM as a raw PPM image.
    /// Lets you eyeball the boot state without firing up the GUI.
    #[arg(long)]
    pub dump_vram: Option<PathBuf>,
    /// Optional path to dump the HW renderer's output as a PPM. Spins
    /// up a headless wgpu device, replays the cumulative `cmd_log`
    /// through the same pipeline the live GUI uses, and writes the
    /// result. Use this to regression-test the HW pipeline without
    /// a window or screen-capture permission.
    #[arg(long)]
    pub dump_hw: Option<PathBuf>,
}

/// Entry point. Dispatches on `cli.command`; returns `Ok(())` on
/// success, `Err` with a user-visible message on failure. `main()`
/// prints the error and exits non-zero.
pub fn run(cli: Cli) -> Result<(), String> {
    let paths = resolve_paths(cli.config_dir.as_deref())?;
    match cli.command.expect("CLI dispatch called without a command") {
        Command::Info => cmd_info(&paths),
        Command::Scan(args) => cmd_scan(&paths, args),
        Command::List => cmd_list(&paths),
        Command::Launch(args) => cmd_launch(&paths, args),
    }
}

/// Dedicated resolver because the `--config-dir` override + the
/// platform-default path need consistent "one place to ask" logic
/// both here and in the GUI.
fn resolve_paths(override_dir: Option<&std::path::Path>) -> Result<ConfigPaths, String> {
    match override_dir {
        Some(p) => {
            let paths = ConfigPaths::rooted(p);
            paths.ensure_dir(paths.root()).map_err(|e| e.to_string())?;
            Ok(paths)
        }
        None => ConfigPaths::platform_default().map_err(|e| e.to_string()),
    }
}

fn cmd_info(paths: &ConfigPaths) -> Result<(), String> {
    let settings_path = paths.settings_file();
    let library_path = paths.library_file();
    let settings = Settings::load(&settings_path).unwrap_or_default();

    println!("# PSoXide headless");
    println!();
    println!("Paths:");
    println!("  config dir       : {}", paths.root().display());
    println!("  settings.ron     : {}", settings_path.display());
    println!("  library.ron      : {}", library_path.display());
    println!();
    println!("Settings:");
    println!("  version          : {}", settings.version);
    println!("  paths.bios       : {}", fmt_empty(&settings.paths.bios));
    println!(
        "  paths.library    : {}",
        fmt_empty(&settings.paths.game_library)
    );
    println!("  video.int.scale  : {}", settings.video.integer_scale);
    println!(
        "  emu.hle-bios-exe : {}",
        settings.emulator.hle_bios_for_side_load
    );
    println!(
        "  input.port1.cross: {}",
        settings.input.port1.cross.label()
    );
    Ok(())
}

fn cmd_scan(paths: &ConfigPaths, args: ScanArgs) -> Result<(), String> {
    let mut settings = Settings::load(&paths.settings_file()).unwrap_or_default();
    let explicit_root = args.root.clone();
    let root = args.root.map(Ok).unwrap_or_else(|| {
        if settings.paths.game_library.is_empty() {
            Err(
                "No library root. Pass --root <dir> or set paths.game_library in settings.ron."
                    .to_string(),
            )
        } else {
            Ok(PathBuf::from(&settings.paths.game_library))
        }
    })?;
    if !root.exists() {
        return Err(format!("library root does not exist: {}", root.display()));
    }

    let mut lib = Library::load_or_empty(&paths.library_file());
    let before = lib.entries.len();
    let changed = lib.scan(&root).map_err(|e| e.to_string())?;
    lib.save(&paths.library_file()).map_err(|e| e.to_string())?;
    println!(
        "scanned {} → {} entries ({} parsed / re-parsed, {} reused)",
        root.display(),
        lib.entries.len(),
        changed,
        lib.entries.len().saturating_sub(changed),
    );
    if before != lib.entries.len() {
        println!("(cache size changed: {} → {})", before, lib.entries.len());
    }

    // Persist the root into settings.ron whenever `--root` was passed
    // explicitly. A fresh config dir that never had settings.ron
    // written stays empty otherwise -- the GUI would find the library
    // but wouldn't know where to rescan from, so the next GUI-triggered
    // rescan would fail. Writing here keeps the "scan once on the CLI,
    // then use the GUI" path frictionless.
    if let Some(new_root) = explicit_root {
        let new_str = new_root.to_string_lossy().into_owned();
        if settings.paths.game_library != new_str {
            settings.paths.game_library = new_str;
            if let Err(e) = settings.save(&paths.settings_file()) {
                eprintln!("warning: could not save settings.ron: {e}");
            } else {
                println!(
                    "settings.paths.game_library updated -> {}",
                    new_root.display()
                );
            }
        }
    }
    Ok(())
}

fn cmd_list(paths: &ConfigPaths) -> Result<(), String> {
    let lib = Library::load_or_empty(&paths.library_file());
    if lib.entries.is_empty() {
        println!("(library is empty — run `scan` first)");
        return Ok(());
    }
    // Sort alphabetically by title for stable output.
    let mut sorted = lib.entries.clone();
    sorted.sort_by_key(|a| a.title.to_lowercase());
    for e in &sorted {
        println!(
            "{:<16}  {:<10}  {:<7}  {:>8} MiB  {}",
            e.id,
            kind_label(e.kind),
            region_label(e),
            e.size / (1024 * 1024),
            e.title,
        );
    }
    println!();
    println!("{} entries", sorted.len());
    Ok(())
}

fn cmd_launch(paths: &ConfigPaths, args: LaunchArgs) -> Result<(), String> {
    let settings = Settings::load(&paths.settings_file()).unwrap_or_default();

    // Resolve `path`: direct flag or lookup by game-id.
    let game_path = match (args.path, args.game_id) {
        (Some(p), _) => p,
        (None, Some(id)) => {
            let lib = Library::load_or_empty(&paths.library_file());
            lib.entries
                .iter()
                .find(|e| e.id == id)
                .map(|e| e.path.clone())
                .ok_or_else(|| format!("no game with id={id} in library.ron"))?
        }
        (None, None) => {
            return Err("Provide --path or --game-id".to_string());
        }
    };

    // BIOS: settings override if set, else PSOXIDE_BIOS.
    let bios_path = resolve_bios_path(&settings)?;

    let bios =
        std::fs::read(&bios_path).map_err(|e| format!("BIOS {}: {e}", bios_path.display()))?;
    let mut bus = Bus::new(bios).map_err(|e| format!("BIOS rejected: {e}"))?;
    let mut cpu = Cpu::new();

    // `--dump-hw` needs the GP0 packet log armed; it stays off by
    // default to avoid the per-instruction push cost.
    if args.dump_hw.is_some() {
        bus.gpu.enable_cmd_log();
    }

    // Dispatch on extension: .bin = disc, .exe = homebrew side-load.
    let ext = game_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "exe" => {
            let bytes = std::fs::read(&game_path).map_err(|e| e.to_string())?;
            let exe = Exe::parse(&bytes).map_err(|e| format!("parse EXE: {e:?}"))?;
            bus.load_exe_payload(exe.load_addr, &exe.payload);
            bus.clear_exe_bss(exe.bss_addr, exe.bss_size);
            cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());
            // Match the GUI launch path: side-loaded EXEs need the
            // HLE syscall tables even for users with old settings.ron
            // files where the former preference is still false.
            bus.enable_hle_bios();
            bus.attach_digital_pad_port1();
            eprintln!(
                "[cli] side-loaded {} — entry=0x{:08x} payload={}B",
                game_path.display(),
                exe.initial_pc,
                exe.payload.len()
            );
        }
        "bin" | "iso" => {
            let bytes = std::fs::read(&game_path).map_err(|e| e.to_string())?;
            let disc = Disc::from_bin(bytes);
            maybe_fast_boot_disc(
                &mut bus,
                &mut cpu,
                &disc,
                &game_path,
                settings.emulator.fast_boot_disc && !args.bios_boot,
            );
            bus.cdrom.insert_disc(Some(disc));
            eprintln!("[cli] mounted disc {}", game_path.display());
        }
        "cue" => {
            let disc = psoxide_settings::library::load_disc_from_cue(&game_path)?;
            maybe_fast_boot_disc(
                &mut bus,
                &mut cpu,
                &disc,
                &game_path,
                settings.emulator.fast_boot_disc && !args.bios_boot,
            );
            bus.cdrom.insert_disc(Some(disc));
            eprintln!("[cli] mounted cue-backed disc {}", game_path.display());
        }
        other => {
            return Err(format!("unsupported file extension: .{other}"));
        }
    }

    // Step the CPU. Report early on opcode errors -- they're usually
    // "we hit an unimplemented instruction" and worth surfacing.
    let mut stopped_at: Option<u64> = None;
    let mut audio_cycle_accum = 0u64;
    for i in 0..args.steps {
        let cycles_before = bus.cycles();
        if let Err(e) = cpu.step(&mut bus) {
            eprintln!("[cli] step {i} failed: {e:?}");
            stopped_at = Some(i);
            break;
        }
        audio_cycle_accum =
            audio_cycle_accum.saturating_add(bus.cycles().saturating_sub(cycles_before));
        let sample_count = (audio_cycle_accum / SAMPLE_CYCLES) as usize;
        audio_cycle_accum %= SAMPLE_CYCLES;
        if sample_count > 0 {
            bus.run_spu_samples(sample_count);
            let _ = bus.spu.drain_audio();
        }
    }

    println!(
        "tick={}  cycles={}  pc=0x{:08x}{}",
        cpu.tick(),
        bus.cycles(),
        cpu.pc(),
        match stopped_at {
            Some(i) => format!("  stopped-at={i}"),
            None => String::new(),
        }
    );

    if args.dump_hash {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for &w in bus.gpu.vram.words() {
            for b in w.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01B3);
            }
        }
        let (dh, dw, dhi, _) = bus.gpu.display_hash();
        println!("vram_fnv1a_64=0x{h:016x}");
        println!("display_fnv1a_64=0x{dh:016x}  w={dw}  h={dhi}");
    }

    if let Some(path) = args.dump_vram {
        dump_vram_ppm(&bus, &path)?;
        eprintln!("[cli] VRAM → {}", path.display());
    }

    if let Some(path) = args.dump_hw {
        let used_24bpp_fallback = dump_hw_ppm(&bus, &path)?;
        if used_24bpp_fallback {
            eprintln!(
                "[cli] HW renderer → {} (24bpp display fallback)",
                path.display()
            );
        } else {
            eprintln!(
                "[cli] HW renderer → {} ({} cmd_log entries replayed)",
                path.display(),
                bus.gpu.cmd_log.len()
            );
        }
    }

    Ok(())
}

fn maybe_fast_boot_disc(
    bus: &mut Bus,
    cpu: &mut Cpu,
    disc: &Disc,
    path: &std::path::Path,
    enabled: bool,
) {
    if !enabled {
        return;
    }
    if let Err(e) = warm_bios_for_disc_fast_boot(bus, cpu, DISC_FAST_BOOT_WARMUP_STEPS) {
        eprintln!(
            "[cli] BIOS warmup failed for {} ({e:?}); leaving BIOS boot fallback in place",
            path.display()
        );
        return;
    }
    match fast_boot_disc_with_hle(bus, cpu, disc, false) {
        Ok(info) => eprintln!(
            "[cli] warm-fast-booted {} via {} entry=0x{:08x} load=0x{:08x} payload={}B",
            path.display(),
            info.boot_path,
            info.initial_pc,
            info.load_addr,
            info.payload_len
        ),
        Err(e) => eprintln!(
            "[cli] fast boot unavailable for {} ({e:?}); falling back to BIOS boot",
            path.display()
        ),
    }
}

fn fmt_empty(s: &str) -> String {
    if s.is_empty() {
        "(unset)".into()
    } else {
        s.to_string()
    }
}

fn kind_label(k: GameKind) -> &'static str {
    match k {
        GameKind::DiscBin => "disc-bin",
        GameKind::DiscIso => "disc-iso",
        GameKind::DiscCue => "disc-cue",
        GameKind::Exe => "homebrew",
        GameKind::Unknown => "unknown",
    }
}

fn region_label(e: &LibraryEntry) -> &'static str {
    use psoxide_settings::library::Region;
    match e.region {
        Region::NtscU => "NTSC-U",
        Region::Pal => "PAL",
        Region::NtscJ => "NTSC-J",
        Region::Unknown => "unknown",
    }
}

fn dump_hw_ppm(bus: &Bus, path: &std::path::Path) -> Result<bool, String> {
    let display = bus.gpu.display_area();
    if display.bpp24 {
        let (rgba, w, h) = bus.gpu.display_rgba8();
        write_rgb_ppm_from_rgba(path, w, h, &rgba)?;
        return Ok(true);
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no compatible wgpu adapter".to_string())?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("psoxide-hw-dump-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| format!("request device: {e:?}"))?;

    let mut hw = psx_gpu_render::HwRenderer::new_headless(device, queue);
    let vram_words: Vec<u16> = bus.gpu.vram.words().to_vec();
    hw.render_frame(&bus.gpu, &bus.gpu.cmd_log, &vram_words);

    let s = hw.internal_scale();
    let (w, h, rgba) = hw.read_subrect_rgba8(
        display.x as u32 * s,
        display.y as u32 * s,
        display.width as u32 * s,
        display.height as u32 * s,
    );
    write_rgb_ppm_from_rgba(path, w, h, &rgba)?;
    Ok(false)
}

fn write_rgb_ppm_from_rgba(
    path: &std::path::Path,
    w: u32,
    h: u32,
    rgba: &[u8],
) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    writeln!(f, "P6\n{w} {h}\n255").map_err(|e| e.to_string())?;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for chunk in rgba.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    f.write_all(&rgb).map_err(|e| e.to_string())?;
    Ok(())
}

fn dump_vram_ppm(bus: &Bus, path: &std::path::Path) -> Result<(), String> {
    use std::io::Write;
    let w = emulator_core::VRAM_WIDTH;
    let h = emulator_core::VRAM_HEIGHT;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    writeln!(f, "P6\n{w} {h}\n255").map_err(|e| e.to_string())?;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for &pix in bus.gpu.vram.words() {
        let r5 = (pix & 0x1F) as u8;
        let g5 = ((pix >> 5) & 0x1F) as u8;
        let b5 = ((pix >> 10) & 0x1F) as u8;
        rgb.push((r5 << 3) | (r5 >> 2));
        rgb.push((g5 << 3) | (g5 >> 2));
        rgb.push((b5 << 3) | (b5 >> 2));
    }
    f.write_all(&rgb).map_err(|e| e.to_string())?;
    Ok(())
}
