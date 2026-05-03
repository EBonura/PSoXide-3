//! Dump neighbouring showcase-fog frames around scroll-wrap points so
//! we can see exactly what changes frame-to-frame.

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("showcase-fog", false);

    // RING_SPACING / SCROLL_SPEED is about 162 frames. Dump groups
    // around two wraps to check that the one-tile phase reset is clean.
    let sample_targets: &[u64] = &[
        157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 319, 320, 321, 322, 323, 324, 325, 326,
        327, 328,
    ];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("showcase-fog-adj", target)
            .expect("dump frame")
            .log(target);
    }
}
