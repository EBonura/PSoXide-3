//! Dump invaders gameplay frames to validate wave progression /
//! bomb drops / alien march + auto-start.
#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("invaders", true);

    let sample_targets: &[u64] = &[4, 30, 60, 120, 240, 500];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("invaders", target)
            .expect("dump frame")
            .log(target);
    }
}
