//! Disc fast boot helpers.
//!
//! This path mirrors the BIOS loader's `SYSTEM.CNF -> PSX-EXE` work
//! without relying on the BIOS license-screen handoff. The disc stays
//! mounted in the CD-ROM controller; only the initial executable load
//! is short-circuited.

use psx_iso::{load_boot_exe_from_disc, BootError, Disc};

use crate::{Bus, Cpu, ExecutionError};

/// Number of BIOS instructions to run before warm disc fast boot.
///
/// By this point SCPH1001 has installed the syscall tables, exception
/// vectors, and interrupt mask state that retail games expect, but it
/// has not spent time on the disc license path.
pub const DISC_FAST_BOOT_WARMUP_STEPS: u64 = 10_000_000;

/// Summary of a successful disc fast boot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscFastBootInfo {
    /// Normalized `SYSTEM.CNF` boot path.
    pub boot_path: String,
    /// Executable entry point.
    pub initial_pc: u32,
    /// Executable load address.
    pub load_addr: u32,
    /// Bytes copied into RAM.
    pub payload_len: usize,
    /// Stack pointer applied to the CPU, if one was provided.
    pub stack_pointer: Option<u32>,
}

/// Load a disc's boot EXE into RAM and seed the CPU at its entry point.
///
/// Callers should mount the same [`Disc`] in the CD-ROM controller
/// after this returns, so the running game can continue issuing normal
/// CD commands.
pub fn fast_boot_disc(
    bus: &mut Bus,
    cpu: &mut Cpu,
    disc: &Disc,
) -> Result<DiscFastBootInfo, BootError> {
    fast_boot_disc_with_hle(bus, cpu, disc, true)
}

/// Variant of [`fast_boot_disc`] that lets callers choose whether to
/// enable HLE BIOS dispatch after loading the EXE.
///
/// Set `enable_hle_bios` to `false` when the real BIOS has already run
/// far enough to install its RAM syscall and exception handlers.
pub fn fast_boot_disc_with_hle(
    bus: &mut Bus,
    cpu: &mut Cpu,
    disc: &Disc,
    enable_hle_bios: bool,
) -> Result<DiscFastBootInfo, BootError> {
    let boot = load_boot_exe_from_disc(disc)?;
    let payload_len = boot.exe.payload.len();
    let stack_pointer = boot.stack_pointer.or_else(|| boot.exe.initial_sp());

    if let Some(sp) = stack_pointer {
        bus.clear_ram_range(0x8001_0000, sp);
    }
    bus.load_exe_payload(boot.exe.load_addr, &boot.exe.payload);
    bus.clear_exe_bss(boot.exe.bss_addr, boot.exe.bss_size);
    cpu.seed_from_exe_with_args(boot.exe.initial_pc, boot.exe.initial_gp, stack_pointer, 1, 0);
    if enable_hle_bios {
        bus.enable_hle_bios();
    }

    Ok(DiscFastBootInfo {
        boot_path: boot.boot_path,
        initial_pc: boot.exe.initial_pc,
        load_addr: boot.exe.load_addr,
        payload_len,
        stack_pointer,
    })
}

/// Run the real BIOS long enough to install its RAM kernel state.
pub fn warm_bios_for_disc_fast_boot(
    bus: &mut Bus,
    cpu: &mut Cpu,
    steps: u64,
) -> Result<(), ExecutionError> {
    for _ in 0..steps {
        cpu.step(bus)?;
    }
    Ok(())
}
