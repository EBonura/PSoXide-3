//! Side-load pong and dump several frames, letting the ball bounce
//! around under the AI-vs-idle-player default. Used to eyeball that
//! the ball physics + AI tracking + SFX triggering all work.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_pong_frames --release
//! ```

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("pong", true);

    // Sample across a rally: early, mid-rally, bounce region.
    let sample_targets: &[u64] = &[2, 30, 60, 120, 240];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("pong", target)
            .expect("dump frame")
            .log(target);
    }
}
