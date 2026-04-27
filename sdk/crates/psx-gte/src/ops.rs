//! GTE function-op wrappers.
//!
//! Each op corresponds to one COP2 cofun instruction. The constants
//! below are the exact 32-bit encodings PSX-SPX documents — the same
//! bits the MIPS CPU would dispatch on hardware and the same bits the
//! emulator's `Gte::execute` decodes. On host we hand them straight
//! to the per-thread software GTE in [`crate::host`]; on MIPS we emit
//! them as `.word` instructions so LLVM's integrated assembler can't
//! reject the COP2 mnemonics it doesn't recognise.
//!
//! Encoding (per PSX-SPX, "GTE Coprocessor Opcodes"):
//!
//! ```text
//!   bits 31..26 = 0x12 (010010)      — COP2
//!   bit  25     = 1                   — `cop2 cofun` prefix
//!   bit  19     = sf                  — fraction-shift
//!   bits 18..17 = mx                  — MVMVA matrix select
//!   bits 16..15 = vx                  — MVMVA vector select
//!   bits 14..13 = cv                  — MVMVA translation select
//!   bit  10     = lm                  — IR lower-bound mode
//!   bits 5..0   = opcode              — function id
//! ```
//!
//! The `0x4A000000` base (`0x12<<26 | 1<<25`) is common to every op.

#[cfg(target_arch = "mips")]
use core::arch::asm;

/// Emit one cofun instruction. On MIPS this is a `.word` with the
/// encoding baked into the immediate; on host it executes against the
/// thread-local software Gte.
macro_rules! cofun {
    ($instr:expr) => {{
        #[cfg(target_arch = "mips")]
        unsafe {
            asm!(
                ".word {instr}",
                instr = const $instr,
                options(nostack, preserves_flags),
            )
        }
        #[cfg(not(target_arch = "mips"))]
        {
            $crate::host::execute($instr);
        }
    }};
}

/// RTPS — perspective transform of V0. `sf=1, lm=0`.
///
/// # Safety
/// Assumes V0, RT, TR, OFX/OFY, H, DQA/DQB are loaded.
#[inline(always)]
pub unsafe fn rtps() {
    cofun!(0x4A08_0001)
}

/// RTPT — RTPS applied to V0, V1, V2 in sequence. `sf=1, lm=0`.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn rtpt() {
    cofun!(0x4A08_0030)
}

/// NCLIP — Z component of `(SXY1-SXY0) × (SXY2-SXY0)` into MAC0.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn nclip() {
    cofun!(0x4A00_0006)
}

/// OP — outer product of IR with the rotation matrix diagonal. `sf=1`.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn op_sf1() {
    cofun!(0x4A08_000C)
}

/// AVSZ3 — average SZ1..SZ3 weighted by ZSF3, store in OTZ + MAC0.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn avsz3() {
    cofun!(0x4A00_002D)
}

/// AVSZ4 — average SZ0..SZ3 weighted by ZSF4.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn avsz4() {
    cofun!(0x4A00_002E)
}

/// SQR — squares the current IR vector into MAC1/2/3, `sf=1`.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn sqr() {
    cofun!(0x4A08_0028)
}

/// NCDS — normal-colour depth-cue single vertex. `sf=1, lm=0`.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn ncds() {
    cofun!(0x4A08_0013)
}

/// NCCS — normal-colour single vertex (no depth cue).
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn nccs() {
    cofun!(0x4A08_001B)
}

/// NCS — normal-colour single vertex (lighting + RGBC modulate).
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn ncs() {
    cofun!(0x4A08_001E)
}

/// NCDT — NCDS for V0, V1, V2.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn ncdt() {
    cofun!(0x4A08_0016)
}

/// NCT — NCS for V0, V1, V2.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn nct() {
    cofun!(0x4A08_0020)
}

/// NCCT — NCCS for V0, V1, V2.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn ncct() {
    cofun!(0x4A08_003F)
}

/// DPCS — depth-cue single (RGBC ↔ FC by IR0).
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn dpcs() {
    cofun!(0x4A08_0010)
}

/// DPCT — DPCS run three times against the RGB FIFO.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn dpct() {
    cofun!(0x4A08_002A)
}

/// INTPL — interpolate IR toward FC by IR0; FIFO push.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn intpl() {
    cofun!(0x4A08_0011)
}

/// DCPL — depth-cue colour light: `RGBC*IR ↔ FC` by IR0.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn dcpl() {
    cofun!(0x4A08_0029)
}

/// CC — colour-colour blend.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn cc() {
    cofun!(0x4A08_001C)
}

/// CDP — colour depth queue.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn cdp() {
    cofun!(0x4A08_0014)
}

/// GPF — `MAC = IR * IR0`, FIFO-push, `sf=1`.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn gpf() {
    cofun!(0x4A08_003D)
}

/// GPL — `MAC = MAC + IR * IR0`, FIFO-push, `sf=1`.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn gpl() {
    cofun!(0x4A08_003E)
}

/// MVMVA(`mx=RT, vx=V0, cv=TR, sf=1, lm=0`) — view-space transform
/// without the perspective divide.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn mvmva_rt_v0_tr_sf1() {
    cofun!(0x4A08_0012)
}
