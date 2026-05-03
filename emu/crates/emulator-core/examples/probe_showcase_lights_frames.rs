//! Dump showcase-lights frames across the light-orbit cycle.
#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("showcase-lights", false);

    let sample_targets: &[u64] = &[8, 30, 60, 120, 180, 240];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("showcase-lights", target)
            .expect("dump frame")
            .log(target);
    }
}
