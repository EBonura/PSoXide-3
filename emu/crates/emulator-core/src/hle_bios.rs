//! High-level emulation of the PS1 BIOS syscall tables.
//!
//! The BIOS publishes three entry points — at physical addresses
//! `0xA0`, `0xB0`, and `0xC0` — that dispatch to a table of service
//! functions. Each caller does:
//!
//! ```text
//!     la $t0, 0xA0       # or 0xB0 / 0xC0
//!     jr $t0
//!     li $t1, <func>     # in the branch delay slot
//! ```
//!
//! and the BIOS dispatcher reads `$t1`, calls the right handler, and
//! returns to `$ra`.
//!
//! When we side-load a PSX-EXE we bypass the BIOS boot sequence, so
//! the dispatcher stubs at those RAM addresses aren't populated. This
//! module fills the gap by intercepting the instruction fetch when
//! `PC` hits one of the three entry addresses, running the requested
//! service in host Rust, and "returning" by setting `PC = $ra`.
//!
//! Scope for the first pass: TTY output, `FlushCache`, and the event
//! system in "always-ready" mode so homebrew that polls `TestEvent`
//! doesn't spin forever. Games that use richer BIOS facilities
//! (file I/O, memory cards, controllers) can land their handlers
//! here incrementally as we exercise them.

use crate::Bus;
use psx_hw::memory::to_physical;

/// One of the three BIOS dispatcher tables.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Table {
    /// Entry point at physical `0xA0`.
    A,
    /// Entry point at physical `0xB0`.
    B,
    /// Entry point at physical `0xC0`.
    C,
}

impl Table {
    fn from_phys(phys: u32) -> Option<Self> {
        match phys {
            0xA0 => Some(Table::A),
            0xB0 => Some(Table::B),
            0xC0 => Some(Table::C),
            _ => None,
        }
    }
}

/// Result of one HLE dispatch: `$v0` return value and the updated PC.
#[derive(Copy, Clone, Debug)]
pub struct Hle {
    /// Value to write into `$r2 ($v0)`. `0` if the syscall doesn't
    /// return a meaningful value.
    pub v0: u32,
    /// Value to set PC to after the call. Normally `$ra`, so the CPU
    /// resumes right after the caller's `jalr` (or, in the BIOS-stub
    /// pattern, right after the `jr $t0 ; li $t1, N` pair).
    pub next_pc: u32,
}

/// Look at `cpu_pc`; if it matches a BIOS table entry, run the
/// service that `$t1 ($r9)` selects and return the post-call state.
/// Otherwise return `None` and let the CPU fetch normally.
///
/// `args` is the four argument registers `$a0..$a3` (`$r4..$r7`),
/// `t1_func_num` is `$r9` (the function selector set by the caller's
/// delay-slot load), and `ra` is `$r31`.
pub fn dispatch(cpu_pc: u32, bus: &mut Bus, args: [u32; 4], t1_func_num: u32, ra: u32) -> Option<Hle> {
    let phys = to_physical(cpu_pc);
    let table = Table::from_phys(phys)?;
    let func = (t1_func_num & 0xFF) as u8;
    let v0 = run(table, func, bus, args);
    Some(Hle { v0, next_pc: ra })
}

fn run(table: Table, func: u8, bus: &mut Bus, args: [u32; 4]) -> u32 {
    bus.hle_bios_log_call(table, func);
    match (table, func) {
        // --- A-table ---

        // A(3Ch) putchar
        (Table::A, 0x3C) => {
            write_byte_to_stdout(args[0] as u8);
            0
        }

        // A(3Eh) puts(*s): write a NUL-terminated string.
        (Table::A, 0x3E) => {
            write_cstring_to_stdout(bus, args[0]);
            0
        }

        // A(3Fh) printf — partial. Print the format string verbatim
        // so messages are at least readable; `%` formatting is
        // intentionally unimplemented until a real use-case lands.
        (Table::A, 0x3F) => {
            write_cstring_to_stdout(bus, args[0]);
            0
        }

        // A(44h) FlushCache — no I-cache model yet, so no-op.
        (Table::A, 0x44) => 0,

        // --- B-table ---

        // B(08h) OpenEvent: return a synthetic handle. We accept
        // everything; the handle encodes table + slot for debug.
        (Table::B, 0x08) => 0xF400_0000 | (args[0] & 0xFFFF),

        // B(09h) CloseEvent — accept.
        (Table::B, 0x09) => 1,

        // B(0Ah) WaitEvent — say the event fired.
        (Table::B, 0x0A) => 1,

        // B(0Bh) TestEvent — say it fired. Homebrew using BIOS event
        // polling for VSync will see a positive result every call and
        // advance its frame loop; full event state machine lands here
        // when a commercial game actually needs edge-sensitive events.
        (Table::B, 0x0B) => 1,

        // B(0Ch) EnableEvent / B(0Dh) DisableEvent — accept.
        (Table::B, 0x0C) | (Table::B, 0x0D) => 1,

        // B(18h) SetDefaultExceptionHandler — accept.
        (Table::B, 0x18) => 0,

        // B(3Dh) std_out_putchar — same as A(3Ch).
        (Table::B, 0x3D) => {
            write_byte_to_stdout(args[0] as u8);
            0
        }

        // --- C-table is mostly specialty (timer / interrupt setup)
        // and isn't needed for hello-tri. Fall-through logs a warning.
        _ => 0,
    }
}

fn write_byte_to_stdout(byte: u8) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(&[byte]);
    let _ = out.flush();
}

fn write_cstring_to_stdout(bus: &mut Bus, addr: u32) {
    let mut p = addr;
    // Bound at 4 KiB per call so a bogus pointer can't hang us.
    for _ in 0..4096 {
        let b = bus.try_read8(p).unwrap_or(0);
        if b == 0 {
            break;
        }
        write_byte_to_stdout(b);
        p = p.wrapping_add(1);
    }
}
