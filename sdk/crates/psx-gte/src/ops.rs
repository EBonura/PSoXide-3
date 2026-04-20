//! GTE function-op wrappers.
//!
//! Each function emits a single hand-encoded 32-bit COP2 instruction
//! via `.word`. We skip the assembler's mnemonic support because
//! Rust's LLVM-based integrated assembler doesn't always recognise the
//! full COP2 opcode set — `.word` avoids that rabbit hole entirely.
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

use core::arch::asm;

/// RTPS — perspective transform of V0. `sf=1, lm=0` is the variant
/// games use 99% of the time; it scales the 44-bit MAC result down by
/// 4096 after the matrix multiply.
///
/// Inputs: V0 (data regs 0–1), RT (control 0–4), TR (control 5–7),
/// H (control 26), OFX/OFY (24/25), DQA/DQB (27/28).
///
/// Outputs: IR1/2/3 (9–11), MAC1/2/3 (25–27), SZ3 (19), SXY2 (14).
///
/// # Safety
/// Assumes all input registers are loaded. Reads caller-invisible COP2
/// state.
#[inline(always)]
pub unsafe fn rtps() {
    unsafe { asm!(".word 0x4A080001", options(nostack, preserves_flags)) }
}

/// RTPT — RTPS applied to V0, V1, V2 in sequence. `sf=1, lm=0`.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn rtpt() {
    unsafe { asm!(".word 0x4A080030", options(nostack, preserves_flags)) }
}

/// NCLIP — Z component of the cross-product `(SXY1-SXY0) × (SXY2-SXY0)`,
/// written to MAC0. Used for backface culling before a triangle is
/// queued: positive = front-facing, negative = back-facing.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn nclip() {
    unsafe { asm!(".word 0x4A000006", options(nostack, preserves_flags)) }
}

/// OP — outer product of the IR vector with the rotation matrix
/// diagonal. `sf=1, lm=0`.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn op_sf1() {
    unsafe { asm!(".word 0x4A08000C", options(nostack, preserves_flags)) }
}

/// AVSZ3 — average of SZ1, SZ2, SZ3 weighted by ZSF3 (control 29).
/// Writes OTZ (data reg 7) and MAC0 (24).
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn avsz3() {
    unsafe { asm!(".word 0x4A00002D", options(nostack, preserves_flags)) }
}

/// AVSZ4 — average of SZ0..SZ3 weighted by ZSF4 (control 30).
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn avsz4() {
    unsafe { asm!(".word 0x4A00002E", options(nostack, preserves_flags)) }
}

/// SQR — squares the current IR vector into MAC1/2/3 with sf=1.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn sqr() {
    unsafe { asm!(".word 0x4A080028", options(nostack, preserves_flags)) }
}

/// NCDS — normal-colour depth-cue single vertex. Lighting + colour +
/// FC depth interpolation on V0. `sf=1, lm=0`.
///
/// Inputs: V0, RGBC, LLM, LCM, BK, FC, IR0.
/// Outputs: IR1/2/3, MAC1/2/3, RGB FIFO slot 2.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn ncds() {
    unsafe { asm!(".word 0x4A080013", options(nostack, preserves_flags)) }
}

/// NCCS — normal-colour single vertex (no depth cue).
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn nccs() {
    unsafe { asm!(".word 0x4A08001B", options(nostack, preserves_flags)) }
}

/// NCS — normal-colour single vertex (lighting + RGBC modulate, no FC).
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn ncs() {
    unsafe { asm!(".word 0x4A08001E", options(nostack, preserves_flags)) }
}

/// NCDT — NCDS for V0, V1, V2 in one op. Produces three entries in
/// the RGB FIFO. `sf=1, lm=0`.
///
/// This is the classic commercial-game fogged-triangle primitive:
/// three vertices projected (via a preceding `RTPT`) and three lit
/// + depth-cue-blended colours emitted in the FIFO for rasterisation
/// as a Gouraud triangle with per-vertex fog attenuation.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn ncdt() {
    unsafe { asm!(".word 0x4A080016", options(nostack, preserves_flags)) }
}

/// NCT — NCS for V0, V1, V2. Lit colour, no depth cue.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn nct() {
    unsafe { asm!(".word 0x4A080020", options(nostack, preserves_flags)) }
}

/// NCCT — NCCS for V0, V1, V2. Lit colour multiplied by RGBC, emits
/// three FIFO colours.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn ncct() {
    unsafe { asm!(".word 0x4A08003F", options(nostack, preserves_flags)) }
}

/// DPCS — depth-cue single. Interpolates RGBC toward FC using IR0,
/// pushing the result into the RGB FIFO. No lighting, no normal —
/// pure fog on the current base colour.
///
/// Inputs: RGBC (data 6), FC (control 21..=23), IR0 (data 8).
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn dpcs() {
    unsafe { asm!(".word 0x4A080010", options(nostack, preserves_flags)) }
}

/// DPCT — DPCS run three times against the RGB FIFO slots. Depth-cue
/// the three most-recent colours in place.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn dpct() {
    unsafe { asm!(".word 0x4A08002A", options(nostack, preserves_flags)) }
}

/// INTPL — interpolate the current IR vector toward FC by IR0 and
/// push the result as an RGB FIFO entry.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn intpl() {
    unsafe { asm!(".word 0x4A080011", options(nostack, preserves_flags)) }
}

/// DCPL — depth-cue colour light. Interpolates `RGBC * IR` toward
/// FC using IR0, FIFO-push.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn dcpl() {
    unsafe { asm!(".word 0x4A080029", options(nostack, preserves_flags)) }
}

/// CC — colour-colour. Blend RGBC against IR using the LCM; no FC
/// interpolation. Useful for applying a light colour to an already-
/// lit base.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn cc() {
    unsafe { asm!(".word 0x4A08001C", options(nostack, preserves_flags)) }
}

/// CDP — colour depth queue. Like `CC` but with FC interpolation
/// layered on top. Used for coloured fogging of pre-lit surfaces.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn cdp() {
    unsafe { asm!(".word 0x4A080014", options(nostack, preserves_flags)) }
}

/// GPF — general-purpose interpolation `MAC = IR * IR0` (sf=1).
/// Pushes the result into the RGB FIFO. Used for flat-shaded prim
/// colour computation.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn gpf() {
    unsafe { asm!(".word 0x4A08003D", options(nostack, preserves_flags)) }
}

/// GPL — GPF plus an accumulator base: `MAC = MAC + IR * IR0` (sf=1).
/// Used for interpolating successive colour stages.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn gpl() {
    unsafe { asm!(".word 0x4A08003E", options(nostack, preserves_flags)) }
}

/// MVMVA with the common "rotate V0 + TR" setup: `mx=0` (RT),
/// `vx=0` (V0), `cv=0` (TR), `sf=1, lm=0`. This is RTPS's geometry
/// half without the perspective divide — useful when you want the
/// view-space position but not the screen-space projection.
///
/// # Safety
/// See [`rtps`].
#[inline(always)]
pub unsafe fn mvmva_rt_v0_tr_sf1() {
    unsafe { asm!(".word 0x4A080012", options(nostack, preserves_flags)) }
}
