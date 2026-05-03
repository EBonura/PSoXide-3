//! Dump showcase-fog frames so we can inspect the segmented corridor
//! moving slowly forward through the far fog.
#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("showcase-fog", false);

    // Sample across multiple scroll phases so texture density, fog,
    // and the one-segment wrap stay easy to inspect.
    let sample_targets: &[u64] = &[16, 128, 256, 384, 512, 640, 768, 900];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("showcase-fog", target)
            .expect("dump frame")
            .log(target);
    }
}
