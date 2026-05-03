//! Side-load showcase-text and dump several frames to PPM.
//!
//! Renders frames where the rotation demo is at visually-
//! different angles (frames 3, 10, 25, 42) -- 42 is approximately
//! one full rotation at the showcase's 96 Q0.12 units/frame, so
//! frame 42 should look nearly identical to frame 0 for the
//! rotation component while the rest of the frame stays stable.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_showcase_text_frames --release
//! ```

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("showcase-text", false);

    let sample_targets: &[u64] = &[2, 4, 10, 25, 42];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("showcase-text", target)
            .expect("dump frame")
            .log(target);
    }
}
