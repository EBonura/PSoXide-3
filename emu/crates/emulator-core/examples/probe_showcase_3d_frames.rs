//! Dump showcase-3d frames spaced across the rotation cycle so we
//! can eyeball the meshes tumbling + starfield flowing.
#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("showcase-3d", false);

    let sample_targets: &[u64] = &[8, 30, 60, 120, 180];
    for &target in sample_targets {
        probe.run_until_vblank(target);
        probe
            .dump_display_ppm("showcase-3d", target)
            .expect("dump frame")
            .log(target);
    }
}
