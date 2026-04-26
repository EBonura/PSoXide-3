//! Dump a frame from any SDK example to verify it renders.
//! Takes the example name as arg 1 and an optional comma-separated
//! vblank list as arg 2.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_any_example --release -- hello-tri
//! cargo run -p emulator-core --example probe_any_example --release -- showcase-model 8,9,10
//! ```

use emulator_core::{gpu::GpuCmdLogEntry, Bus, ButtonState, Cpu};
use psx_iso::Exe;
use std::io::Write;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args
        .next()
        .expect("usage: probe_any_example <name> [vblank[,vblank...]]");
    let targets = args
        .next()
        .map(|text| parse_targets(&text))
        .unwrap_or_else(|| vec![8]);
    let pad_mask = std::env::var("PSOXIDE_PAD1")
        .ok()
        .and_then(|text| parse_u16_mask(&text))
        .unwrap_or(0);
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("bios");
    let exe_path = format!(
        "/Users/ebonura/Desktop/repos/psoxide/build/examples/mipsel-sony-psx/release/{name}.exe"
    );
    let exe_bytes = std::fs::read(&exe_path).expect("exe");
    let exe = Exe::parse(&exe_bytes).expect("parse");

    let mut bus = Bus::new(bios).expect("bus");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.enable_hle_bios();
    bus.attach_digital_pad_port1();
    bus.set_port1_buttons(ButtonState::from_bits(pad_mask));
    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());
    bus.gpu.enable_cmd_log();

    let mut cycles_at_last_pump = 0u64;
    for target in targets {
        let vblank_before = bus.irq().raise_counts()[0];
        let cycles_before = bus.cycles();
        let tick_before = cpu.tick();
        let gte_before = cpu.cop2().profile_snapshot();
        let host_start = Instant::now();
        while bus.irq().raise_counts()[0] < target {
            if cpu.step(&mut bus).is_err() {
                break;
            }
            if bus.cycles() - cycles_at_last_pump > 560_000 {
                cycles_at_last_pump = bus.cycles();
                bus.run_spu_samples(735);
                let _ = bus.spu.drain_audio();
            }
        }
        let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
        let cycles = bus.cycles().saturating_sub(cycles_before);
        let frames = bus.irq().raise_counts()[0].saturating_sub(vblank_before);
        let budget = bus.vblank_period().saturating_mul(frames.max(1));
        let psx_pct = if budget > 0 {
            cycles as f64 * 100.0 / budget as f64
        } else {
            0.0
        };
        let gte_after = cpu.cop2().profile_snapshot();
        let log = std::mem::take(&mut bus.gpu.cmd_log);
        let (cmds, words, draw_cmds, image_cmds) = gpu_log_counters(&log);
        eprintln!(
            "[probe] target={target} frames={frames} host={host_ms:.2}ms psx={psx_pct:.1}% \
             cycles={cycles} budget={budget} instr={} gte={} gtecy={} cmds={cmds} draw={draw_cmds} \
             image={image_cmds} words={words} pc=0x{:08x}",
            cpu.tick().saturating_sub(tick_before),
            gte_after.ops.saturating_sub(gte_before.ops),
            gte_after
                .estimated_cycles
                .saturating_sub(gte_before.estimated_cycles),
            cpu.pc(),
        );
        dump_ppm(&bus, &name, target);
    }
}

fn gpu_log_counters(log: &[GpuCmdLogEntry]) -> (usize, usize, usize, usize) {
    let mut words = 0usize;
    let mut draw_cmds = 0usize;
    let mut image_cmds = 0usize;
    for entry in log {
        words = words.saturating_add(entry.fifo.len());
        match entry.opcode {
            0x20..=0x7F => draw_cmds += 1,
            0x80..=0xBF => image_cmds += 1,
            _ => {}
        }
    }
    (log.len(), words, draw_cmds, image_cmds)
}

fn dump_ppm(bus: &Bus, name: &str, target: u64) {
    let da = bus.gpu.display_area();
    let path = format!("/tmp/{name}-f{:03}.ppm", target);
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "P6\n{} {}\n255", da.width, da.height).unwrap();
    let mut buf = Vec::with_capacity((da.width as usize) * (da.height as usize) * 3);
    for dy in 0..da.height {
        for dx in 0..da.width {
            let pix = bus.gpu.vram.get_pixel(da.x + dx, da.y + dy);
            let r5 = (pix & 0x1F) as u8;
            let g5 = ((pix >> 5) & 0x1F) as u8;
            let b5 = ((pix >> 10) & 0x1F) as u8;
            buf.push((r5 << 3) | (r5 >> 2));
            buf.push((g5 << 3) | (g5 >> 2));
            buf.push((b5 << 3) | (b5 >> 2));
        }
    }
    f.write_all(&buf).unwrap();
    eprintln!("wrote {path}");
}

fn parse_targets(text: &str) -> Vec<u64> {
    let mut out: Vec<u64> = text
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    if out.is_empty() {
        out.push(8);
    }
    out
}

fn parse_u16_mask(text: &str) -> Option<u16> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        text.parse().ok()
    }
}
