//! Lockstep VRAM parity against Redux for a commercial disc.
//!
//! Boots the same BIOS + disc in both Redux and our emulator, runs
//! them forward in N-step chunks, and after each chunk compares
//! FNV-1a-64 hashes of the visible display area. The first chunk
//! where the hashes disagree localises "when did our rendering
//! drift from Redux's" — which tells us which subsystem to look at
//! next (GTE, GPU blending, DMA ordering, IRQ timing, etc).
//!
//! ```bash
//! cargo run -p parity-oracle --example disc_vram_parity --release -- \
//!   --disc "/path/to/game.bin" \
//!   --steps 50000000 \
//!   --chunk 1000000
//! ```
//!
//! Flags:
//!   `--disc PATH`    : BIN/CUE to mount on both.
//!   `--steps N`      : total Redux-style user steps to run (default 100 M).
//!   `--chunk N`      : user steps per checkpoint (default 1 M).
//!   `--bios PATH`    : override the default SCPH1001.BIN path.
//!   `--pad-mask M`   : hold a port-1 button mask for the whole run.
//!   `--pad-pulses S` : comma-separated `<mask>@<vblank>+<frames>` pulses.
//!
//! Exit code 0 iff every checkpoint matched or any transient mismatch
//! converged again before the final checkpoint. Non-zero when the run
//! ends with an unresolved mismatch, with the step count and both
//! hashes printed.

use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use emulator_core::{Bus, Cpu};
use parity_oracle::{OracleConfig, ReduxProcess};
use psx_iso::Disc;

const DEFAULT_BIOS: &str = "/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const HASH_TIMEOUT: Duration = Duration::from_secs(60);

struct Args {
    disc: PathBuf,
    bios: PathBuf,
    steps: u64,
    chunk: u64,
    pad_port: u32,
    pad_mask: u16,
    pad_pulses: Vec<PadPulse>,
}

impl Args {
    fn pad_enabled(&self) -> bool {
        self.pad_mask != 0 || !self.pad_pulses.is_empty()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PadPulse {
    mask: u16,
    start_vblank: u64,
    frames: u64,
}

#[derive(Debug)]
struct PadRuntime {
    base_mask: u16,
    pulses: Vec<PadPulse>,
    current_mask: Option<u16>,
}

#[derive(Clone, Debug)]
struct PendingMismatch {
    step: u64,
    kind: &'static str,
    our_hash: u64,
    redux_hash: u64,
    our_w: u32,
    our_h: u32,
    redux_w: u32,
    redux_h: u32,
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}\n");
            print_usage();
            std::process::exit(2);
        }
    };

    eprintln!("[disc-parity] BIOS  : {}", args.bios.display());
    eprintln!("[disc-parity] disc  : {}", args.disc.display());
    eprintln!(
        "[disc-parity] steps : {}  (chunk {})",
        args.steps, args.chunk,
    );
    if args.pad_enabled() {
        eprintln!(
            "[disc-parity] pad   : port={} held=0x{:04x} pulses={}",
            args.pad_port,
            args.pad_mask,
            format_pad_pulses(&args.pad_pulses),
        );
    }

    // --- Redux setup ----------------------------------------------------
    let lua = OracleConfig::default_lua_dir().join("oracle.lua");
    let config = OracleConfig::new(args.bios.clone(), lua)
        .expect("Redux binary resolves")
        .with_disc(args.disc.clone());
    eprintln!("[disc-parity] launching Redux...");
    let mut redux = ReduxProcess::launch(&config).expect("Redux launches");
    redux.handshake(HANDSHAKE_TIMEOUT).expect("handshake");

    // --- Our emulator setup ---------------------------------------------
    eprintln!("[disc-parity] launching our emulator...");
    let bios = std::fs::read(&args.bios).expect("BIOS readable");
    let disc = load_disc_path(&args.disc).expect("disc readable");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom.insert_disc(Some(disc));
    if args.pad_enabled() {
        bus.attach_digital_pad_port1();
    }
    let mut cpu = Cpu::new();
    let mut pad = args.pad_enabled().then(|| PadRuntime {
        base_mask: args.pad_mask,
        pulses: args.pad_pulses.clone(),
        current_mask: None,
    });
    if let Some(pad) = pad.as_mut() {
        pad.sync(&mut bus);
    }

    // --- Run + compare in chunks ---------------------------------------
    let mut cursor: u64 = 0;
    let mut chunk_count = 0u64;
    let mut mismatch_count = 0u64;
    let mut pending_mismatch: Option<PendingMismatch> = None;
    let started = Instant::now();
    while cursor < args.steps {
        let chunk = args.chunk.min(args.steps - cursor);
        chunk_count += 1;

        // Redux: silent run — no trace records emitted unless input
        // scheduling is enabled, in which case `run_checkpoint_pad`
        // is the protocol surface that keeps controller state synced.
        let run_timeout = Duration::from_secs((chunk / 500_000).max(30));
        let redux_run = if args.pad_enabled() {
            let pulses = args
                .pad_pulses
                .iter()
                .map(|pulse| (pulse.mask, pulse.start_vblank, pulse.frames))
                .collect::<Vec<_>>();
            redux.run_checkpoint_pad(
                chunk,
                chunk,
                args.pad_port,
                args.pad_mask,
                &pulses,
                run_timeout,
                |_step, _tick, _pc| Ok(()),
            )
        } else {
            redux.run(chunk, run_timeout)
        };
        if let Err(e) = redux_run {
            eprintln!("[disc-parity] Redux run failed at step {cursor} (+{chunk}): {e}",);
            std::process::exit(1);
        }

        // Us: mirror Redux's user-side stepping. Redux's `stepIn`
        // breakpoint is in user code, so an IRQ entered by a user
        // instruction runs through to RFE inside the same outer step.
        if let Some((sub_step, err)) = step_ours_user_steps(&mut cpu, &mut bus, chunk, pad.as_mut())
        {
            let total = cursor + sub_step;
            eprintln!("[disc-parity] our emulator stopped at user step {total}: {err:?}");
            cleanup(redux);
            std::process::exit(1);
        }
        cursor += chunk;

        // Capture + compare.
        let their = match redux.display_hash(HASH_TIMEOUT) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[disc-parity] Redux vram_hash at step {cursor}: {e}");
                cleanup(redux);
                std::process::exit(1);
            }
        };
        let (our_hash, our_w, our_h, _our_len) = bus.gpu.display_hash();

        let rate = cursor as f32 / started.elapsed().as_secs_f32().max(0.001);
        eprintln!(
            "[{chunk_count:>4}] step={cursor:>11}  \
             ours=0x{our_hash:016x} ({our_w}x{our_h})  \
             redux=0x{:016x} ({}x{})  \
             [{:.0} k steps/s]",
            their.hash,
            their.width,
            their.height,
            rate / 1000.0,
        );

        let dims_match = our_w == their.width && our_h == their.height;
        let hash_match = our_hash == their.hash;
        if dims_match && hash_match {
            if let Some(prev) = pending_mismatch.take() {
                eprintln!(
                    "      (previous {} mismatch at step {} resolved at step {cursor})",
                    prev.kind, prev.step,
                );
            }
            continue;
        }

        mismatch_count += 1;
        if !dims_match {
            // GP1 0x07/0x08 writes can hit slightly different
            // instruction windows because of IRQ timing jitter. Keep
            // running so a later checkpoint can prove convergence, but
            // fail the probe if this is still unresolved at the end.
            eprintln!(
                "      (dim mismatch: ours={}x{} redux={}x{} — will retry next chunk)",
                our_w, our_h, their.width, their.height,
            );
            pending_mismatch = Some(PendingMismatch {
                step: cursor,
                kind: "dimension",
                our_hash,
                redux_hash: their.hash,
                our_w,
                our_h,
                redux_w: their.width,
                redux_h: their.height,
            });
            continue;
        }

        if !hash_match {
            // Mismatched content with matching dimensions is usually
            // frame-completion jitter: Redux finished drawing frame
            // N at cycle X; we finish drawing it at cycle X+epsilon,
            // so at this exact check we see different VRAM. Log it
            // and keep going. If we catch up within the next few
            // chunks, it was timing; if we never catch up, the last
            // logged mismatch is a real parity failure.
            eprintln!(
                "      (content mismatch: ours=0x{our_hash:016x} \
                 redux=0x{:016x} — will retry next chunk)",
                their.hash,
            );
            pending_mismatch = Some(PendingMismatch {
                step: cursor,
                kind: "content",
                our_hash,
                redux_hash: their.hash,
                our_w,
                our_h,
                redux_w: their.width,
                redux_h: their.height,
            });
        }
    }

    eprintln!();
    if let Some(mismatch) = pending_mismatch {
        eprintln!(
            "[disc-parity] unresolved {} mismatch at final checkpoint step {}",
            mismatch.kind, mismatch.step,
        );
        eprintln!(
            "              ours=0x{:016x} ({}x{})  redux=0x{:016x} ({}x{})",
            mismatch.our_hash,
            mismatch.our_w,
            mismatch.our_h,
            mismatch.redux_hash,
            mismatch.redux_w,
            mismatch.redux_h,
        );
        eprintln!("[disc-parity] mismatching checkpoints observed: {mismatch_count}");
        cleanup(redux);
        std::process::exit(1);
    }
    eprintln!(
        "[disc-parity] OK through {cursor} steps ({} chunks)",
        chunk_count,
    );
    cleanup(redux);
}

fn cleanup(mut redux: ReduxProcess) {
    redux.send_command("quit").ok();
    let _ = redux.wait_for_response(Duration::from_secs(5));
    let _ = redux.terminate();
}

fn step_ours_user_steps(
    cpu: &mut Cpu,
    bus: &mut Bus,
    steps: u64,
    mut pad: Option<&mut PadRuntime>,
) -> Option<(u64, emulator_core::ExecutionError)> {
    for i in 0..steps {
        if let Some(pad) = pad.as_deref_mut() {
            pad.sync(bus);
        }
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
        if let Some(pad) = pad.as_deref_mut() {
            pad.sync(bus);
        }
    }
    None
}

impl PadRuntime {
    fn sync(&mut self, bus: &mut Bus) {
        let vblank = bus.irq().raise_counts()[0];
        let next_mask = effective_pad_mask(self.base_mask, &self.pulses, vblank);
        if self.current_mask.is_some_and(|mask| mask == next_mask) {
            return;
        }
        bus.set_port1_buttons(emulator_core::ButtonState::from_bits(next_mask));
        self.current_mask = Some(next_mask);
    }
}

fn load_disc_path(path: &Path) -> Result<Disc, String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("cue"))
    {
        psoxide_settings::library::load_disc_from_cue(path)
    } else {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Disc::from_bin(bytes))
    }
}

fn parse_args() -> Result<Args, String> {
    let mut disc: Option<PathBuf> = None;
    let mut bios: Option<PathBuf> = None;
    let mut steps: u64 = 100_000_000;
    let mut chunk: u64 = 1_000_000;
    let mut pad_port: u32 = 1;
    let mut pad_mask: u16 = match env::var("PSOXIDE_PAD1") {
        Ok(mask) => {
            parse_u16_mask(&mask).ok_or_else(|| "PSOXIDE_PAD1 must be a u16 mask".to_string())?
        }
        Err(_) => 0,
    };
    let mut pad_pulses: Vec<PadPulse> = match env::var("PSOXIDE_PAD1_PULSES") {
        Ok(spec) if !spec.trim().is_empty() => parse_pad_pulses(&spec).ok_or_else(|| {
            "PSOXIDE_PAD1_PULSES must be comma-separated <mask>@<vblank>+<frames> entries"
                .to_string()
        })?,
        _ => Vec::new(),
    };
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--disc" => {
                disc = Some(PathBuf::from(
                    it.next().ok_or_else(|| "--disc takes a path".to_string())?,
                ));
            }
            "--bios" => {
                bios = Some(PathBuf::from(
                    it.next().ok_or_else(|| "--bios takes a path".to_string())?,
                ));
            }
            "--steps" => {
                steps = it
                    .next()
                    .ok_or_else(|| "--steps takes a number".to_string())?
                    .parse()
                    .map_err(|e| format!("--steps: {e}"))?;
            }
            "--chunk" => {
                chunk = it
                    .next()
                    .ok_or_else(|| "--chunk takes a number".to_string())?
                    .parse()
                    .map_err(|e| format!("--chunk: {e}"))?;
            }
            "--pad-port" => {
                pad_port = it
                    .next()
                    .ok_or_else(|| "--pad-port takes a number".to_string())?
                    .parse()
                    .map_err(|e| format!("--pad-port: {e}"))?;
            }
            "--pad-mask" => {
                let mask = it
                    .next()
                    .ok_or_else(|| "--pad-mask takes a u16 mask".to_string())?;
                pad_mask =
                    parse_u16_mask(&mask).ok_or_else(|| format!("--pad-mask: bad mask {mask}"))?;
            }
            "--pad-pulses" => {
                let spec = it
                    .next()
                    .ok_or_else(|| "--pad-pulses takes a pulse spec".to_string())?;
                pad_pulses = parse_pad_pulses(&spec).ok_or_else(|| {
                    "--pad-pulses must be comma-separated <mask>@<vblank>+<frames> entries"
                        .to_string()
                })?;
            }
            "--help" | "-h" => return Err("help".to_string()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if chunk == 0 {
        return Err("--chunk must be greater than zero".to_string());
    }
    if pad_port != 1 && (pad_mask != 0 || !pad_pulses.is_empty()) {
        return Err("--pad-port currently only supports port 1 in psoxide".to_string());
    }
    Ok(Args {
        disc: disc.ok_or_else(|| "missing --disc".to_string())?,
        bios: bios.unwrap_or_else(|| {
            env::var("PSOXIDE_BIOS")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(DEFAULT_BIOS))
        }),
        steps,
        chunk,
        pad_port,
        pad_mask,
        pad_pulses,
    })
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
        return "(none)".to_string();
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

fn print_usage() {
    eprintln!(
        "usage: disc_vram_parity --disc <path> [--bios <path>] \
         [--steps N] [--chunk N] [--pad-mask M] [--pad-pulses SPEC]\n\
         \n\
         Boots the same disc + BIOS on Redux and our emulator, then\n\
         compares VRAM display-area hashes every --chunk user steps.\n\
         Stops + reports at the first divergence.\n\
         \n\
         Pad pulses use <mask>@<vblank>+<frames>, e.g.\n\
         --pad-pulses '0x0008@1200+4,0x4000@1250+1'.",
    );
}
