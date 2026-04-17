//! PS1 BIOS syscall trampolines.
//!
//! Each BIOS call is a 3-instruction assembly stub: load the table
//! address (A/B/C → 0xA0/0xB0/0xC0) into `$t0`, jump to it, put the
//! function index into `$t1` in the branch-delay slot. The BIOS
//! dispatcher reads `$t1` and dispatches to the right handler,
//! then returns like any other function.
//!
//! We use a declarative macro so the trampoline bodies can't drift
//! from the signatures.

/// Build a BIOS-call trampoline + its `extern "C"` binding.
macro_rules! bios_calls {
    ($(
        $(#[doc = $doc:literal])*
        $table:ident ( $func:literal ) fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)?;
    )*) => {
        core::arch::global_asm!(
            ".set noreorder",
            $(
                concat!(
                    ".section .text.bios.", stringify!($name), "\n",
                    ".globl __bios_", stringify!($name), "\n",
                    "__bios_", stringify!($name), ":\n",
                    "  la $8, ", bios_calls!(@table $table), "\n",
                    "  jr $8\n",
                    "  li $9, ", stringify!($func), "\n",
                ),
            )*
        );
        extern "C" {
            $(
                $(#[doc = $doc])*
                #[link_name = concat!("__bios_", stringify!($name))]
                pub fn $name($($arg: $ty),*) $(-> $ret)?;
            )*
        }
    };

    (@table A) => { "0xA0" };
    (@table B) => { "0xB0" };
    (@table C) => { "0xC0" };
}

bios_calls! {
    // ---- A-table ----

    /// A(3Ch) `putchar` — write one byte to the TTY.
    A(0x3C) fn bios_putchar(ch: u32);

    /// A(3Eh) `puts` — write a null-terminated string to the TTY.
    A(0x3E) fn bios_puts(s: *const u8);

    /// A(44h) `FlushCache` — invalidate the instruction cache.
    A(0x44) fn bios_flush_cache();

    // ---- B-table ----

    /// B(08h) `OpenEvent` — register an event handler.
    B(0x08) fn bios_open_event(class: u32, spec: u16, mode: u16, func: u32) -> u32;

    /// B(0Ah) `WaitEvent` — block until event fires.
    B(0x0A) fn bios_wait_event(event: u32) -> u32;

    /// B(0Bh) `TestEvent` — non-blocking: returns 1 if fired, 0 else.
    B(0x0B) fn bios_test_event(event: u32) -> u32;

    /// B(0Ch) `EnableEvent`.
    B(0x0C) fn bios_enable_event(event: u32);

    /// B(3Dh) `std_out_putchar` — write one byte to stdout device.
    B(0x3D) fn bios_std_out_putchar(ch: u32);
}

/// Flush the instruction cache. Must be called after writing to the
/// `.text` region (self-modifying code, dynamic loading, etc.).
#[inline(always)]
pub fn flush_cache() {
    unsafe { bios_flush_cache() }
}

/// Write one byte to TTY via BIOS `putchar`.
#[inline(always)]
pub fn putchar(ch: u8) {
    unsafe { bios_putchar(ch as u32) }
}

/// Write a null-terminated string pointer to TTY via BIOS `puts`.
///
/// # Safety
/// `s` must point to a NUL-terminated byte sequence in readable memory.
#[inline(always)]
pub unsafe fn puts(s: *const u8) {
    unsafe { bios_puts(s) }
}
