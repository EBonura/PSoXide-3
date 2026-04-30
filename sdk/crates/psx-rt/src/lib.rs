//! PS1 bare-metal runtime.
//!
//! Provides the `_start` entry point that the PSX-EXE loader jumps to,
//! a panic handler, BIOS syscall trampolines, and optional heap
//! initialisation. Homebrew crates depend on this crate to get a
//! working `main()` environment without thinking about linker
//! symbols or cache-flushing.
//!
//! # Entry sequence
//!
//! 1. The loader (our emulator or the real BIOS) copies the EXE
//!    payload to `LOAD_ADDR` and jumps to [`_start`].
//! 2. [`_start`] zeroes the `.bss` section using the linker-defined
//!    `__bss_start` / `__bss_end` symbols.
//! 3. With the `alloc` feature, the bump allocator is seeded from
//!    `__heap_start..__heap_end`.
//! 4. `main()` is called. When it returns, we halt.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
// Required for MIPS `global_asm!` blocks (BIOS trampolines). Suppress
// the "feature is declared but not used" warning that shows on the
// host target where only the panic handler + heap are live.
#![cfg_attr(target_arch = "mips", feature(asm_experimental_arch))]

#[cfg(target_arch = "mips")]
pub mod bios;
#[cfg(target_arch = "mips")]
pub mod interrupts;
#[cfg(target_arch = "mips")]
pub mod tty;

#[cfg(feature = "alloc")]
pub mod heap;

// Symbols emitted by `psoxide.ld`.
extern "C" {
    static mut __bss_start: u8;
    static mut __bss_end: u8;
    #[cfg(feature = "alloc")]
    static __heap_start: u8;
    #[cfg(feature = "alloc")]
    static __heap_end: u8;
}

#[cfg(target_arch = "mips")]
extern "Rust" {
    fn main();
}

/// Entry point the PSX-EXE loader jumps to.
///
/// # Safety
/// Called exactly once at boot by the loader. The caller must
/// have set up a valid stack pointer before branching here -- the
/// PSX-EXE header's `initial_sp_base` + `initial_sp_offset` fields
/// guarantee this.
#[cfg(target_arch = "mips")]
#[no_mangle]
#[link_section = ".text._start"]
pub unsafe extern "C" fn _start() -> ! {
    // Zero BSS.
    let bss_start = &raw mut __bss_start as *mut u8;
    let bss_end = &raw const __bss_end as *const u8;
    let bss_len = bss_end as usize - bss_start as usize;
    if bss_len > 0 {
        unsafe { core::ptr::write_bytes(bss_start, 0, bss_len) };
    }

    #[cfg(feature = "alloc")]
    {
        let heap_start = &raw const __heap_start as *const u8 as usize;
        let heap_end = &raw const __heap_end as *const u8 as usize;
        unsafe { heap::init(heap_start, heap_end - heap_start) };
    }

    unsafe { main() };
    halt();
}

/// Infinite loop with no useful side effects. Used after `main()`
/// returns or from panic / reset paths.
#[inline(never)]
pub fn halt() -> ! {
    loop {
        core::hint::spin_loop();
    }
}

/// Panic handler. Tries to write the message to TTY so PCSX-Redux
/// (and our emulator's future console hook) shows it, then halts.
///
/// Only registered when targeting PS1 hardware. Host builds of the
/// SDK use `std`'s panic handler -- this avoids the lang-item conflict
/// that would otherwise fire when something on host transitively
/// depends on both `psx-rt` and `std`.
#[cfg(target_arch = "mips")]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    tty::print("PANIC: ");
    if let Some(msg) = info.message().as_str() {
        tty::print(msg);
    }
    tty::print("\n");
    if let Some(loc) = info.location() {
        tty::print("  at ");
        tty::print(loc.file());
        tty::print("\n");
    }
    halt()
}
