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

        // A(0x2A) malloc / A(0x33) memset / similar memory helpers.
        // Most commercial games roll their own allocator; these
        // fallthroughs prevent a jump-to-zero on a stray call.
        (Table::A, 0x2A) => 0,
        (Table::A, 0x33) => {
            // memset(dest, val, n) — write `n` bytes of `val` to `dest`.
            let (dest, val, n) = (args[0], args[1] as u8, args[2]);
            for i in 0..n.min(0x20_0000) {
                let _ = bus.write8_safe(dest.wrapping_add(i), val);
            }
            dest
        }

        // A(0x3C) putchar / A(0x3D) getchar.
        (Table::A, 0x3C) => {
            write_byte_to_stdout(args[0] as u8);
            0
        }
        // A(0x3D) getchar — no stdin source yet; return -1 (EOF).
        (Table::A, 0x3D) => u32::MAX,

        // A(0x3E) puts(*s) / A(0x3F) printf.
        (Table::A, 0x3E) => {
            write_cstring_to_stdout(bus, args[0]);
            0
        }
        (Table::A, 0x3F) => {
            // Partial printf: emit the format string verbatim.
            // Real %-format support lands when a game actually
            // relies on it — most debug output is static text.
            write_cstring_to_stdout(bus, args[0]);
            0
        }

        // A(0x44) FlushCache — no I-cache model yet, so no-op.
        (Table::A, 0x44) => 0,

        // A(0x70) _bu_init (memcard filesystem init) — accept.
        (Table::A, 0x70) => 0,

        // A(0x96) AddCDROMDevice / A(0x97) AddMemCardDevice — games
        // call these during init to register filesystem drivers.
        // We don't model the device table; accept so the game moves on.
        (Table::A, 0x96) | (Table::A, 0x97) => 0,

        // A(0x9F) EnterCriticalSection / A(0xA0) ExitCriticalSection.
        // On hardware these manipulate SR.IE. HLE BIOS can't safely
        // forge IE-manipulation, but games use them as bracket
        // scopes — as long as pairs balance and both return plausibly,
        // the game proceeds. EnterCriticalSection returns 1.
        (Table::A, 0x9F) => 1,
        (Table::A, 0xA0) => 0,

        // --- B-table ---

        // B(0x00) SysMalloc — not a real malloc; many games replace
        // the kernel heap with their own and never call this.
        (Table::B, 0x00) => 0,

        // B(0x07) DeliverEvent — accept; our event system is always-
        // ready so there's nothing to deliver.
        (Table::B, 0x07) => 0,

        // B(0x08) OpenEvent: return a synthetic handle. We accept
        // everything; the handle encodes table + slot for debug.
        (Table::B, 0x08) => 0xF400_0000 | (args[0] & 0xFFFF),

        // B(0x09) CloseEvent, B(0x0A) WaitEvent, B(0x0B) TestEvent,
        // B(0x0C) EnableEvent, B(0x0D) DisableEvent — always-ready.
        (Table::B, 0x09)
        | (Table::B, 0x0A)
        | (Table::B, 0x0B)
        | (Table::B, 0x0C)
        | (Table::B, 0x0D) => 1,

        // B(0x12) InitPad(buf1, siz1, buf2, siz2): tell the kernel
        // where to stash pad state. Since we poll the hardware
        // directly via psx-pad there's nothing for us to do.
        (Table::B, 0x12) => 1,

        // B(0x13) StartPad, B(0x14) StopPad — accept.
        (Table::B, 0x13) | (Table::B, 0x14) => 1,

        // B(0x17) ReturnFromException — tricky: real impl restores
        // SR from K0 and jumps to EPC. For HLE we punt: games that
        // depend on this usually also install their own exception
        // vectors, in which case our intercept never fires for them.
        (Table::B, 0x17) => 0,

        // B(0x18) SetDefaultExceptionHandler — accept.
        (Table::B, 0x18) => 0,

        // B(0x3D) std_out_putchar — same as A(0x3C).
        (Table::B, 0x3D) => {
            write_byte_to_stdout(args[0] as u8);
            0
        }

        // B(0x4A) InitCard, B(0x4B) StartCard, B(0x4C) StopCard.
        (Table::B, 0x4A) | (Table::B, 0x4B) | (Table::B, 0x4C) => 1,

        // --- C-table (kernel interrupt handlers) ---

        // C(0x00) EnqueueTimerAndVblankIrqs / C(0x01) EnqueueSyscallHandler.
        // Install canned handlers. We never actually invoke them —
        // but accepting the registration lets games proceed.
        (Table::C, 0x00) | (Table::C, 0x01) | (Table::C, 0x02) | (Table::C, 0x03) => 0,

        // C(0x0A) ChangeClearRCnt — affects how the kernel's
        // root-counter handler clears flags. No-op.
        (Table::C, 0x0A) => args[1],

        // Everything else: zero. Games that trip a real missing
        // syscall will show up in the HLE call histogram and we
        // can fill them in one at a time.
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
