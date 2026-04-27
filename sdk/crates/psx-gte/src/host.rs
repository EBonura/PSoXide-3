//! Host-side GTE shim.
//!
//! On real PS1 hardware the GTE is a coprocessor — a singleton owned
//! by the CPU. The SDK's macros and op functions encode that as
//! parameter-less inline-asm calls. Host builds preserve the same
//! shape by routing those calls through a per-thread
//! [`psx_gte_core::Gte`] kept in a [`thread_local!`] cell.
//!
//! Why thread-local rather than an explicit `&mut Gte` argument:
//! - the PS1 GTE *is* a process-wide singleton, so the implicit-state
//!   API matches the hardware semantics.
//! - keeps `psx-engine`, `psx-gpu`, every showcase demo, and the
//!   editor's render loop calling the same parameter-less helpers
//!   that they already use on MIPS.
//! - sequential renders on one thread reset the GTE before each
//!   submission, so multi-viewport scenarios still cooperate.
//!
//! Tests or callers that need explicit state can reach the Gte
//! directly via [`with`].

use core::cell::RefCell;
use psx_gte_core::Gte;

std::thread_local! {
    static GTE: RefCell<Gte> = RefCell::new(Gte::new());
}

/// Read a GTE data register through the per-thread Gte.
#[inline]
pub fn read_data(idx: u8) -> u32 {
    GTE.with(|g| g.borrow().read_data(idx))
}

/// Write a GTE data register through the per-thread Gte.
#[inline]
pub fn write_data(idx: u8, value: u32) {
    GTE.with(|g| g.borrow_mut().write_data(idx, value));
}

/// Read a GTE control register through the per-thread Gte.
#[inline]
pub fn read_control(idx: u8) -> u32 {
    GTE.with(|g| g.borrow().read_control(idx))
}

/// Write a GTE control register through the per-thread Gte.
#[inline]
pub fn write_control(idx: u8, value: u32) {
    GTE.with(|g| g.borrow_mut().write_control(idx, value));
}

/// Execute a packed COP2 cofun command word (RTPS / RTPT / NCDS / …).
/// The lower 25 bits encode the opcode + sf/lm/mx/vx/cv selectors;
/// upper bits (the `0x12 << 26 | 1 << 25` COP2 prefix on hardware) are
/// ignored by the simulator.
#[inline]
pub fn execute(instr: u32) {
    GTE.with(|g| g.borrow_mut().execute(instr));
}

/// Reset this thread's GTE to a freshly-zeroed state. Convenient at
/// the top of a render frame where you don't want stale FIFO contents
/// from the previous draw bleeding into the current one.
pub fn reset() {
    GTE.with(|g| *g.borrow_mut() = Gte::new());
}

/// Borrow the per-thread Gte for advanced inspection or batch state
/// changes. The closure runs with `&mut Gte` and the borrow is
/// released on return.
pub fn with<R>(f: impl FnOnce(&mut Gte) -> R) -> R {
    GTE.with(|g| f(&mut g.borrow_mut()))
}
