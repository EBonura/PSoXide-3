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

use emulator_core::{Bus, Cpu};
use psx_iso::Exe;
use std::path::PathBuf;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "hello-tri".into());
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("bios");
    let exe_path = PathBuf::from("/Users/ebonura/Desktop/repos/psoxide/build/examples/mipsel-sony-psx/release")
        .join(format!("{name}.exe"));
    let exe_bytes = std::fs::read(&exe_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", exe_path.display()));
    let exe = Exe::parse(&exe_bytes).expect("parse");

    // Mirror `AppState::launch_entry` line-for-line:
    let mut bus = Bus::new(bios).expect("bus");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.enable_hle_bios(); // now unconditional in launch_entry
    bus.attach_digital_pad_port1();
    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    // Step two frames — same checkpoint the milestone tests use.
    let mut cycles_at_last_pump = 0u64;
    while bus.irq().raise_counts()[0] < 2 {
        if cpu.step(&mut bus).is_err() {
            eprintln!("CPU errored at pc=0x{:08x}", cpu.pc());
            break;
        }
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }
    }
    let (display_hash, w, h, _) = bus.gpu.display_hash();
    println!(
        "{name}: display={w}×{h}  hash=0x{:016x}  final_pc=0x{:08x}",
        display_hash, cpu.pc(),
    );

    // Also count how many primitive writes hit GPU — a blank frame
    // has no geometry commands, a working one has lots. Cheap smoke
    // check independent of hash values.
    let hist = bus.gpu.gp0_opcode_histogram();
    let draw_ops: u32 = (0x20..=0x7F).map(|op| hist[op as usize]).sum();
    let fill_ops = hist[0x02];
    let line_ops: u32 = (0x40..=0x5F).map(|op| hist[op as usize]).sum();
    println!(
        "  GP0 draw ops: primitives={draw_ops} (fills={fill_ops}, lines={line_ops})",
    );
    if draw_ops == 0 {
        println!("  ✗ ZERO draw primitives — EXE didn't render anything.");
    } else {
        println!("  ✓ {draw_ops} draw primitives — EXE is rendering.");
    }
}
