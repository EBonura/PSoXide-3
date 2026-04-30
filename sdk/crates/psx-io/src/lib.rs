//! Volatile MMIO primitives.
//!
//! The lowest layer of the SDK. Wraps `core::ptr::read_volatile` /
//! `write_volatile` in zero-overhead helpers that each peripheral
//! module builds on. Keeping these centralised means we can audit
//! every MMIO touchpoint in one file.
//!
//! All functions here are `#[inline(always)]` and `unsafe` -- they
//! accept arbitrary addresses and widths. Higher-level SDK crates
//! present safe APIs on top of these.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod cdrom;
pub mod dma;
pub mod gpu;
pub mod gte;
pub mod irq;
pub mod sio;
pub mod spu;
pub mod timers;

/// Read a 32-bit word from `addr`. Caller must guarantee `addr` is
/// valid MMIO for a 32-bit read.
///
/// # Safety
/// Targets unchecked memory-mapped I/O. The caller is responsible for
/// the address mapping and any side effects.
#[inline(always)]
pub unsafe fn read32(addr: u32) -> u32 {
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}

/// Read a 16-bit half-word from `addr`.
///
/// # Safety
/// See [`read32`].
#[inline(always)]
pub unsafe fn read16(addr: u32) -> u16 {
    unsafe { core::ptr::read_volatile(addr as *const u16) }
}

/// Read an 8-bit byte from `addr`.
///
/// # Safety
/// See [`read32`].
#[inline(always)]
pub unsafe fn read8(addr: u32) -> u8 {
    unsafe { core::ptr::read_volatile(addr as *const u8) }
}

/// Write a 32-bit word to `addr`.
///
/// # Safety
/// See [`read32`].
#[inline(always)]
pub unsafe fn write32(addr: u32, value: u32) {
    unsafe { core::ptr::write_volatile(addr as *mut u32, value) }
}

/// Write a 16-bit half-word to `addr`.
///
/// # Safety
/// See [`read32`].
#[inline(always)]
pub unsafe fn write16(addr: u32, value: u16) {
    unsafe { core::ptr::write_volatile(addr as *mut u16, value) }
}

/// Write an 8-bit byte to `addr`.
///
/// # Safety
/// See [`read32`].
#[inline(always)]
pub unsafe fn write8(addr: u32, value: u8) {
    unsafe { core::ptr::write_volatile(addr as *mut u8, value) }
}
