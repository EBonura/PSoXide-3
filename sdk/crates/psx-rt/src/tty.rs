//! TTY output for debug printing.
//!
//! The PS1 BIOS has a `putchar` handler that forwards bytes to the
//! Expansion 2 TTY port — PCSX-Redux reads that port and prints to
//! stdout. Our emulator does the same. So calling [`print`] /
//! [`println`] from homebrew lands in the host terminal, which is
//! the canonical PS1 debug channel.

use crate::bios;

/// Write a string to TTY. Each byte is emitted via the BIOS
/// `putchar` trampoline.
pub fn print(s: &str) {
    for b in s.bytes() {
        bios::putchar(b);
    }
}

/// [`print`] + a trailing `\n`.
pub fn println(s: &str) {
    print(s);
    bios::putchar(b'\n');
}

/// Write a single u32 as 8 hex digits, no prefix, uppercase.
pub fn print_hex_u32(v: u32) {
    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    for shift in (0..8).rev() {
        let nib = ((v >> (shift * 4)) & 0xF) as usize;
        bios::putchar(DIGITS[nib]);
    }
}
