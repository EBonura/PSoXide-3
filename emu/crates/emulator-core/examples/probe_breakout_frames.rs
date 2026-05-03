//! Side-load breakout and dump several frames across the auto-
//! serve → ball-in-play arc. Validates the ball physics, brick
//! collision, and HUD updates all work without human pad input.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_breakout_frames --release
//! ```

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("breakout", true);

    // 30 = auto-launch just fires; 45 = ball halfway up; 60 =
    // hitting the bottom brick row; 90 = first brick break; 180 =
    // a few more bricks cleared.
    let sample_targets: &[u64] = &[4, 30, 60, 85, 88, 92, 96, 180, 300];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("breakout", target)
            .expect("dump frame")
            .log(target);
    }
}
