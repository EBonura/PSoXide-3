//! Dump showcase-textured-sprite frames for visual inspection.
#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("showcase-textured-sprite", true);

    let sample_targets: &[u64] = &[3, 60, 180, 360];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("showcase-textured-sprite", target)
            .expect("dump frame")
            .log(target);
    }
}
