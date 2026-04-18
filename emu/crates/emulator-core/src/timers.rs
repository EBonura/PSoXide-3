//! Root counters (Timer 0 / 1 / 2).
//!
//! Three 16-bit counters at `0x1F80_1100`, `0x1F80_1110`, `0x1F80_1120`,
//! each with three registers:
//! - counter value (offset 0)
//! - mode / control (offset 4)
//! - target value (offset 8)
//!
//! **Phase 2e scope:** typed register backing + MMIO dispatch. No
//! ticking, no interrupts. Software that writes a counter and reads
//! it back sees its own value (same as the MMIO echo buffer used to
//! provide), and peripherals that read timer state get zeros. Once
//! the emulator gains a cycle model, the counters start advancing
//! and firing IRQs via [`crate::irq::IrqSource::Timer0`]…`Timer2`.

/// One of the three root counters. Fields are 16 bits on hardware but
/// held as `u32` for uniform bus access — upper bits read as 0.
#[derive(Default, Clone, Copy)]
pub struct Timer {
    /// Current counter value (bits 0..=15 meaningful).
    pub counter: u32,
    /// Mode / control register. See nocash PSX-SPX "Timers" section for
    /// the bit layout; we store the written value verbatim for now.
    pub mode: u32,
    /// Target value the counter compares against.
    pub target: u32,
    /// Sub-tick accumulator in fractional clock-source units. When a
    /// timer's clock source ticks slower than the system clock (HBlank
    /// on T1, /8 on T2), the residual cycles accumulate here until
    /// they cross the source period.
    accum: u64,
    /// Bus cycle at which the counter was last reset (mode write).
    /// Diagnostic — lets us compare against Redux's `cycleStart`.
    pub last_reset_cycle: u64,
    /// Number of mode writes since reset. Diagnostic.
    pub mode_write_count: u64,
}

// Mode-register bit layout (nocash PSX-SPX, section "Timers").
/// bit 3: reset counter when reaching target (0 = reset at 0xFFFF).
const MODE_RESET_AT_TARGET: u32 = 1 << 3;
/// bit 4: raise IRQ when target reached.
const MODE_IRQ_ON_TARGET: u32 = 1 << 4;
/// bit 5: raise IRQ when counter wraps at 0xFFFF.
const MODE_IRQ_ON_WRAP: u32 = 1 << 5;
/// bit 6: IRQ repeat mode (0 = one-shot until mode-write).
const MODE_IRQ_REPEAT: u32 = 1 << 6;
/// bit 10: IRQ status — active-low flag; cleared on fire.
const MODE_IRQ_ACTIVE_LOW: u32 = 1 << 10;
/// bit 11: "reached target" sticky flag.
const MODE_REACHED_TARGET: u32 = 1 << 11;
/// bit 12: "reached 0xFFFF" sticky flag.
const MODE_REACHED_WRAP: u32 = 1 << 12;

/// The full three-timer bank.
#[derive(Default)]
pub struct Timers {
    /// Per-counter state. Index 0 / 1 / 2 corresponds to Timer 0 / 1 / 2.
    pub timers: [Timer; 3],
}

impl Timers {
    /// Base address of the timer bank.
    pub const BASE: u32 = 0x1F80_1100;
    /// Size of the bank (3 timers × 16 bytes each; last 4 bytes unused).
    pub const SIZE: u32 = 0x30;
    /// Stride between consecutive timers.
    pub const STRIDE: u32 = 0x10;

    /// All counters / modes / targets zero-initialised.
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` when `phys` falls inside `0x1F80_1100..0x1F80_1130`.
    pub fn contains(phys: u32) -> bool {
        (Self::BASE..Self::BASE + Self::SIZE).contains(&phys)
    }

    /// Read a 32-bit word. `phys` must be inside `BASE..BASE+SIZE`.
    pub fn read32(&self, phys: u32) -> u32 {
        let (idx, off) = decode(phys);
        let t = &self.timers[idx];
        match off {
            0x0 => t.counter,
            0x4 => t.mode,
            0x8 => t.target,
            _ => 0,
        }
    }

    /// Write a 32-bit word. `phys` must be inside `BASE..BASE+SIZE`.
    /// `now` is the current bus cycle; used only for diagnostics
    /// (records when mode writes reset the counter).
    pub fn write32(&mut self, phys: u32, value: u32, now: u64) {
        let (idx, off) = decode(phys);
        let t = &mut self.timers[idx];
        let v16 = value & 0xFFFF;
        match off {
            0x0 => t.counter = v16,
            0x4 => {
                // Mode write resets counter + re-arms IRQ (bit 10 set).
                t.mode = (value & !MODE_IRQ_ACTIVE_LOW) | MODE_IRQ_ACTIVE_LOW;
                t.counter = 0;
                t.accum = 0;
                t.last_reset_cycle = now;
                t.mode_write_count = t.mode_write_count.saturating_add(1);
            }
            0x8 => t.target = v16,
            _ => {}
        }
    }

    /// Advance all three timers by `cycles` system-clock ticks. Each
    /// timer converts that to its own clock source first (HBlank for
    /// T1, /8 for T2, system clock elsewhere — dot-clock for T0 would
    /// need GPU scan-out to land first; treat as system clock for now).
    ///
    /// Returns a 3-bit mask of timers that fired an IRQ this tick.
    /// The caller (Bus) uses it to call `Irq::raise(IrqSource::Timer0/1/2)`.
    pub fn tick(&mut self, cycles: u64, hsync_period: u64) -> u8 {
        let mut fired: u8 = 0;
        for i in 0..3 {
            if self.advance_timer(i, cycles, hsync_period) {
                fired |= 1 << i;
            }
        }
        fired
    }

    fn advance_timer(&mut self, idx: usize, cycles: u64, hsync_period: u64) -> bool {
        let t = &mut self.timers[idx];
        let source = (t.mode >> 8) & 0x3;

        // Convert `cycles` system clocks into source clocks.
        let ticks = match (idx, source) {
            // Timer 0: dot clock (1) uses GPU scan-out — not modelled,
            // fall back to system clock.
            // Timer 1 source 1 = HBlank: one tick per HSync period.
            (1, 1) | (1, 3) => {
                t.accum += cycles;
                let n = t.accum / hsync_period;
                t.accum %= hsync_period;
                n
            }
            // Timer 2 source 2/3 = system clock / 8.
            (2, 2) | (2, 3) => {
                t.accum += cycles;
                let n = t.accum / 8;
                t.accum %= 8;
                n
            }
            // Everything else is 1:1 system clock.
            _ => cycles,
        };

        if ticks == 0 {
            return false;
        }

        // Pre-tick position.
        let old = t.counter;
        let target = t.target & 0xFFFF;
        let mut new_val = (old as u64) + ticks;

        let mut fired = false;
        let mut reached_target = false;
        let mut reached_wrap = false;

        // Target-reset mode: when counter crosses the target, reset to 0.
        if t.mode & MODE_RESET_AT_TARGET != 0 && target != 0 {
            if new_val > target as u64 {
                reached_target = true;
                new_val %= (target as u64) + 1;
            } else if new_val == target as u64 {
                reached_target = true;
            }
        }

        if new_val > 0xFFFF {
            reached_wrap = true;
            new_val &= 0xFFFF;
        }

        // Detect target pass for non-reset-mode too.
        if t.mode & MODE_RESET_AT_TARGET == 0
            && (old as u64) < (target as u64)
            && new_val >= (target as u64)
        {
            reached_target = true;
        }

        if reached_target {
            t.mode |= MODE_REACHED_TARGET;
            if t.mode & MODE_IRQ_ON_TARGET != 0 && t.mode & MODE_IRQ_ACTIVE_LOW != 0 {
                // Fire: clear bit 10 (active-low).
                t.mode &= !MODE_IRQ_ACTIVE_LOW;
                fired = true;
                if t.mode & MODE_IRQ_REPEAT != 0 {
                    // Re-arm for repeat mode.
                    t.mode |= MODE_IRQ_ACTIVE_LOW;
                }
            }
        }
        if reached_wrap {
            t.mode |= MODE_REACHED_WRAP;
            if t.mode & MODE_IRQ_ON_WRAP != 0 && t.mode & MODE_IRQ_ACTIVE_LOW != 0 {
                t.mode &= !MODE_IRQ_ACTIVE_LOW;
                fired = true;
                if t.mode & MODE_IRQ_REPEAT != 0 {
                    t.mode |= MODE_IRQ_ACTIVE_LOW;
                }
            }
        }

        t.counter = new_val as u32;
        fired
    }
}

fn decode(phys: u32) -> (usize, u32) {
    let rel = phys - Timers::BASE;
    let idx = (rel / Timers::STRIDE) as usize;
    let off = rel % Timers::STRIDE;
    (idx, off)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_each_timer_and_field() {
        assert_eq!(decode(0x1F80_1100), (0, 0x0));
        assert_eq!(decode(0x1F80_1104), (0, 0x4));
        assert_eq!(decode(0x1F80_1108), (0, 0x8));
        assert_eq!(decode(0x1F80_1110), (1, 0x0));
        assert_eq!(decode(0x1F80_1124), (2, 0x4));
    }

    #[test]
    fn write_then_read_roundtrips() {
        let mut t = Timers::new();
        t.write32(0x1F80_1100, 0x1234, 0);
        assert_eq!(t.read32(0x1F80_1100), 0x1234);

        t.write32(0x1F80_1108, 0xABCD, 0);
        assert_eq!(t.read32(0x1F80_1108), 0xABCD);
    }

    #[test]
    fn mode_write_resets_counter() {
        let mut t = Timers::new();
        t.write32(0x1F80_1100, 0xFF, 0);
        t.write32(0x1F80_1104, 0x0001, 100); // mode write at cycle 100
        assert_eq!(t.read32(0x1F80_1100), 0);
        assert_eq!(t.timers[0].last_reset_cycle, 100);
    }

    #[test]
    fn upper_bits_masked_to_16() {
        let mut t = Timers::new();
        t.write32(0x1F80_1100, 0x1234_5678, 0);
        assert_eq!(t.read32(0x1F80_1100), 0x5678);
    }

    #[test]
    fn contains_covers_full_bank() {
        assert!(Timers::contains(0x1F80_1100));
        assert!(Timers::contains(0x1F80_1128));
        assert!(!Timers::contains(0x1F80_1130));
        assert!(!Timers::contains(0x1F80_10FF));
    }
}
