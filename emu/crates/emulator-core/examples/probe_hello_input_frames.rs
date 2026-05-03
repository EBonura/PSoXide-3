//! Side-load hello-input and dump a frame to PPM so we can SEE the
//! new font-rendered label row. Hash-based milestone goldens don't
//! catch "text rendering is visibly broken" -- only eyeballing a
//! rendered frame does.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_hello_input_frames --release
//! ```
//!
//! Writes `/tmp/hello-input-fXXX.ppm` for a handful of sampled
//! frames. No buttons held in this probe (the pad controller defaults
//! to all-up), so what should render is:
//!   - Blue background from the baseline `b=32` starting color
//!   - "HELD:" header at (4, 4) in light grey
//!   - Tiny white triangle in the centre
//!   - Bottom-right hex = `0xFFFF` (no buttons held = raw mask stays
//!     at 0xFFFF per PSX's active-low pad protocol). If the hex is
//!     `0x0000` instead the `ButtonState` inversion is off.

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("hello-input", true);

    let sample_targets: &[u64] = &[2, 4, 8, 16];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("hello-input", target)
            .expect("dump frame")
            .log(target);
    }
}
