//! GPU MMIO: `GP0`, `GP1`, `GPUREAD`, `GPUSTAT`.
//!
//! Thin wrappers over [`crate::read32`] / [`crate::write32`] that use
//! the register addresses from `psx-hw`. Each helper commits exactly
//! one MMIO access; higher-level SDK code composes them into commands.

use psx_hw::gpu::{GpuStat, GP0, GP1, GPUREAD, GPUSTAT};

/// Push a command or data word to `GP0`. Named `write_gp0` (not just
/// `gp0`) so it doesn't collide with the `psx_hw::gpu::gp0` module
/// that holds the command-word constructors -- that way callers can
/// write `write_gp0(gp0::draw_mode(...))` and each half of the name
/// is unambiguous.
#[inline(always)]
pub fn write_gp0(word: u32) {
    unsafe { crate::write32(GP0, word) }
}

/// Push a command to `GP1`.
#[inline(always)]
pub fn write_gp1(word: u32) {
    unsafe { crate::write32(GP1, word) }
}

/// Read the GPU status register.
#[inline(always)]
pub fn gpustat() -> GpuStat {
    GpuStat::from_bits_retain(unsafe { crate::read32(GPUSTAT) })
}

/// Read the VRAM-to-CPU / GP1(10h) return latch.
#[inline(always)]
pub fn gpuread() -> u32 {
    unsafe { crate::read32(GPUREAD) }
}

/// Spin until the GPU is ready to accept a new command word.
///
/// Polls `GPUSTAT.READY_CMD`. On real hardware this bit is briefly
/// cleared while the GPU is busy ingesting a multi-word packet; our
/// emulator forces it on, so the loop is essentially a single read.
#[inline]
pub fn wait_cmd_ready() {
    while !gpustat().contains(GpuStat::READY_CMD) {}
}

/// Spin until the GPU can start a DMA block transfer.
#[inline]
pub fn wait_dma_ready() {
    while !gpustat().contains(GpuStat::READY_DMA_RECV) {}
}
