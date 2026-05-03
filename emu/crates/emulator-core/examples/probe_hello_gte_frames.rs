//! Side-load hello-gte and dump N frames to PPM so we can SEE what
//! the cube actually looks like on screen. Hash-based milestone
//! tests don't catch "visible garbage that happens to be
//! deterministic" -- only a human looking at a rendered frame does.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_hello_gte_frames --release
//! ```
//!
//! Writes `/tmp/hello-gte-fXXX.ppm` for a handful of sampled frames.

use emulator_core::Bus;

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("hello-gte", true);
    probe.bus.gpu.enable_pixel_tracer();

    // Sample frames: 2, 5, 20, 40, 80. First captures the startup
    // state; the later ones span one full cube rotation at the
    // new YAW_STEP of 4 per frame (256/4 = 64 frames/rev).
    let sample_targets: &[u64] = &[2, 5, 20, 40, 80];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("hello-gte", target)
            .expect("dump frame")
            .log(target);
        if target == 2 {
            dump_gp0_lines(&probe.bus, target);
        }
    }
}

/// Print every GP0 0x40 (mono line) packet from `cmd_log`. Lets us
/// see the actual screen-space coordinates the example hands the
/// GPU -- if they're all at (0, 0) or off-screen, the GTE
/// projection is producing garbage.
fn dump_gp0_lines(bus: &Bus, frame: u64) {
    eprintln!();
    eprintln!("=== GP0 0x40 (mono line) packets at frame {frame} ===");
    let mut count = 0;
    for e in &bus.gpu.cmd_log {
        if e.opcode == 0x40 && e.fifo.len() >= 3 {
            let v0 = e.fifo[1];
            let v1 = e.fifo[2];
            let (x0, y0) = (v0 as i16, (v0 >> 16) as i16);
            let (x1, y1) = (v1 as i16, (v1 >> 16) as i16);
            eprintln!(
                "  line {count:>2}: ({x0:>5}, {y0:>5}) -> ({x1:>5}, {y1:>5})  color=0x{:06x}",
                e.fifo[0] & 0xFFFFFF,
            );
            count += 1;
        }
    }
    eprintln!("=== total mono lines = {count} ===");
    // Also count other opcodes to see if the example reached the
    // render path at all.
    let mut histogram = std::collections::BTreeMap::new();
    for e in &bus.gpu.cmd_log {
        *histogram.entry(e.opcode).or_insert(0u32) += 1;
    }
    eprintln!("=== GP0 opcode histogram (all packets) ===");
    for (op, n) in histogram {
        eprintln!("  0x{op:02x}: {n}");
    }
}
