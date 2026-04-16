//! Root counters: three 16-bit timers with programmable clock sources.
//!
//! Each counter has three registers at `TIMER_N_BASE + {0, 4, 8}`:
//! current value, mode, and target. The clock source for each counter is
//! mode-dependent:
//!
//! | Counter | Sources                                              |
//! |---------|------------------------------------------------------|
//! | 0       | System clock / dot clock                             |
//! | 1       | System clock / horizontal retrace (hblank)           |
//! | 2       | System clock / (system clock / 8)                    |
//!
//! To be populated: mode-register bitfield (sync enable, sync mode,
//! reset on target, IRQ on target/overflow, IRQ repeat, IRQ pulse mode,
//! clock source, target reached flag, overflow reached flag).
//!
//! Reference: nocash PSX-SPX "Timers" section.

/// Counter 0 register base.
pub const TIMER0_BASE: u32 = 0x1F80_1100;

/// Counter 1 register base.
pub const TIMER1_BASE: u32 = 0x1F80_1110;

/// Counter 2 register base.
pub const TIMER2_BASE: u32 = 0x1F80_1120;

/// Offset from a counter's base to its current-value register.
pub const OFFSET_VALUE: u32 = 0x0;
/// Offset from a counter's base to its mode register.
pub const OFFSET_MODE: u32 = 0x4;
/// Offset from a counter's base to its target register.
pub const OFFSET_TARGET: u32 = 0x8;
