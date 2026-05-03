//! Dump hello-ot frames across a full sine cycle so we can see
//! the three triangles drift smoothly without snap-back.
#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("hello-ot", false);

    // ~4 s of frames spaced across the red triangle's full cycle
    // (128 frames = one red revolution at 32 Q0.12/frame).
    let sample_targets: &[u64] = &[4, 32, 64, 96, 128, 192, 256];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("hello-ot", target)
            .expect("dump frame")
            .log(target);
    }
}
