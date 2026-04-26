//! Minimal interrupt support used by engine-level clocks.
//!
//! The first consumer is a monotonic VBlank counter. We install a
//! tiny exception-vector trampoline that handles VBlank IRQs itself,
//! increments a volatile counter, acknowledges the IRQ, then returns
//! with `rfe`. The handler deliberately uses only the MIPS kernel
//! registers `$k0/$k1`, so it does not need a stack frame.

#[cfg(target_arch = "mips")]
use psx_io::irq;

#[cfg(target_arch = "mips")]
core::arch::global_asm!(
    r#"
    .set noreorder
    .section .text.psx_rt_exception
    .globl __psx_rt_exception_handler
__psx_rt_exception_handler:
    lui   $26, 0x1f80
    lw    $27, 0x1070($26)
    lw    $26, 0x1074($26)
    nop
    and   $27, $27, $26
    andi  $27, $27, 0x0001
    beqz  $27, 1f
    nop

    lui   $26, %hi(__psx_rt_vblank_count)
    lw    $27, %lo(__psx_rt_vblank_count)($26)
    nop
    addiu $27, $27, 1
    sw    $27, %lo(__psx_rt_vblank_count)($26)

    lui   $26, 0x1f80
    addiu $27, $zero, -2
    sw    $27, 0x1070($26)

1:
    mfc0  $26, $14
    nop
    jr    $26
    .word 0x42000010
    .set reorder
    "#
);

/// Monotonic VBlank IRQ count.
#[no_mangle]
pub static mut __psx_rt_vblank_count: u32 = 0;

#[cfg(target_arch = "mips")]
extern "C" {
    fn __psx_rt_exception_handler();
}

/// Install and enable the VBlank counter interrupt path.
///
/// This writes a branch into the MIPS general exception vector,
/// enables the VBlank source in `I_MASK`, and sets the COP0 interrupt
/// enable bits used by the R3000A. The operation is idempotent for
/// the current runtime: reinstalling simply resets the software
/// counter and refreshes the vector.
#[cfg(target_arch = "mips")]
pub fn install_vblank_counter() {
    const EXCEPTION_VECTOR: *mut u32 = 0x8000_0080 as *mut u32;
    const J_OPCODE: u32 = 0x0800_0000;

    unsafe {
        let handler = __psx_rt_exception_handler as *const () as usize as u32;
        core::ptr::write_volatile(EXCEPTION_VECTOR, J_OPCODE | ((handler >> 2) & 0x03ff_ffff));
        core::ptr::write_volatile(EXCEPTION_VECTOR.add(1), 0);
        crate::bios::flush_cache();

        core::ptr::write_volatile(&raw mut __psx_rt_vblank_count, 0);
        irq::ack(1 << irq::source::VBLANK);
        irq::set_mask(irq::mask() | (1 << irq::source::VBLANK));
        enable_cpu_interrupts();
    }
}

/// Install and enable the VBlank counter interrupt path.
#[cfg(not(target_arch = "mips"))]
pub fn install_vblank_counter() {}

/// Current monotonic VBlank count.
#[inline]
pub fn vblank_count() -> u32 {
    unsafe { core::ptr::read_volatile(&raw const __psx_rt_vblank_count) }
}

#[cfg(target_arch = "mips")]
unsafe fn enable_cpu_interrupts() {
    let mut sr: u32;
    unsafe { core::arch::asm!("mfc0 $8, $12", lateout("$8") sr) };
    sr |= 0x0401;
    unsafe { core::arch::asm!("mtc0 $8, $12", in("$8") sr) };
}
