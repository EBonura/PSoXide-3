//! Simulate the frontend's `launch_entry` flow for an SDK example
//! EXE: load payload, seed CPU from EXE header, enable HLE BIOS,
//! attach a digital pad. Then run a few frames and hash VRAM.
//!
//! This replicates the EXACT sequence the Menu uses when the user
//! clicks an Examples row, to verify the library-launch path
//! produces the same render the env-var / probe paths do.
//! Before the `hle_bios_for_side_load` default fix, this would
//! come back with an entirely-blank display hash; after the fix
//! it matches the canonical milestone-C captures.
//!
//! ```bash
//! cargo run -p emulator-core --example probe_menu_launch_sim --release -- hello-tri
//! cargo run -p emulator-core --example probe_menu_launch_sim --release -- hello-gte
//! ```

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "hello-tri".into());

    // Mirror `AppState::launch_entry` line-for-line:
    let mut probe = frame_probe::SideLoadedExe::example(&name, true);

    // Step two frames -- same checkpoint the milestone tests use.
    if !probe.run_until_vblank(2) {
        eprintln!("CPU errored at pc=0x{:08x}", probe.cpu.pc());
    }
    let (display_hash, w, h, _) = probe.bus.gpu.display_hash();
    println!(
        "{name}: display={w}×{h}  hash=0x{:016x}  final_pc=0x{:08x}",
        display_hash,
        probe.cpu.pc(),
    );

    // Also count how many primitive writes hit GPU -- a blank frame
    // has no geometry commands, a working one has lots. Cheap smoke
    // check independent of hash values.
    let hist = probe.bus.gpu.gp0_opcode_histogram();
    let draw_ops: u32 = (0x20..=0x7F).map(|op| hist[op as usize]).sum();
    let fill_ops = hist[0x02];
    let line_ops: u32 = (0x40..=0x5F).map(|op| hist[op as usize]).sum();
    println!("  GP0 draw ops: primitives={draw_ops} (fills={fill_ops}, lines={line_ops})",);
    if draw_ops == 0 {
        println!("  ✗ ZERO draw primitives — EXE didn't render anything.");
    } else {
        println!("  ✓ {draw_ops} draw primitives — EXE is rendering.");
    }
}
