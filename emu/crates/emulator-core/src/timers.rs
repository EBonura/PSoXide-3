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
/// bit 0: sync enable. When set, the timer obeys the sync-mode
/// bits below; when clear, the timer is "free-run" (pure clock).
const MODE_SYNC_ENABLE: u32 = 1 << 0;
/// bits 1..2: sync mode. Meaning depends on the timer index:
/// - Timer 0 (HBlank-synced): 0=pause in HBlank, 1=reset at HBlank,
///   2=reset+pause, 3=pause until HBlank then free-run.
/// - Timer 1 (VBlank-synced): same four modes.
/// - Timer 2: modes 0/3 free-run, modes 1/2 stop counter.
#[allow(dead_code)]
const MODE_SYNC_MODE_MASK: u32 = 0x6;
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
    /// Bus cycle at which the timers were last advanced. Used by
    /// `advance_to` so the bus can drive timer state lazily —
    /// once per branch-test scheduler drain and on demand from
    /// MMIO read paths — instead of paying the per-instruction
    /// 3-counter accumulator/divider cost. Mirrors PCSX-Redux's
    /// `Counters::set` / `update` model where each counter holds
    /// `cycleStart` and the live count is `(cycle - cycleStart) /
    /// rate` evaluated lazily.
    last_advance_cycle: u64,
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
    pub fn read32(&mut self, phys: u32) -> u32 {
        let (idx, off) = decode(phys);
        match off {
            0x0 => self.timers[idx].counter,
            0x4 => {
                let mode = self.timers[idx].mode;
                // Hardware/Redux side effect: reading the mode register
                // clears the sticky "reached target" / "reached 0xFFFF"
                // bits (11 and 12). Leaving them latched forever makes
                // timer-polling code think a wrap/target event is still
                // pending long after software consumed it.
                self.timers[idx].mode &= !(MODE_REACHED_TARGET | MODE_REACHED_WRAP);
                mode
            }
            0x8 => self.timers[idx].target,
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
    /// timer converts that to its own clock source first:
    /// - Timer 0 source 1 / 3 → dot-clock (pixels per `dot_clock_divisor`
    ///   system cycles — `GPU::dot_clock_divisor` gives the current
    ///   value based on H-resolution).
    /// - Timer 1 source 1 / 3 → HBlank (`hsync_period` cycles per tick).
    /// - Timer 2 source 2 / 3 → system / 8.
    /// - Everything else → system clock.
    ///
    /// Returns a 3-bit mask of timers that fired an IRQ this tick.
    /// The caller (Bus) uses it to call `Irq::raise(IrqSource::Timer0/1/2)`.
    pub fn tick(&mut self, cycles: u64, hsync_period: u64, dot_clock_divisor: u64) -> u8 {
        let mut fired: u8 = 0;
        for i in 0..3 {
            if self.advance_timer(i, cycles, hsync_period, dot_clock_divisor) {
                fired |= 1 << i;
            }
        }
        self.last_advance_cycle = self.last_advance_cycle.saturating_add(cycles);
        fired
    }

    /// Advance the timer bank to the absolute cycle `now`, using the
    /// time elapsed since the last advance. Equivalent to calling
    /// `tick(delta, ...)` where `delta = now - last_advance_cycle`.
    /// Returns the same 3-bit "fired" bitmap.
    ///
    /// This is the lazy entry point: bus calls it once per
    /// scheduler drain (instead of every `Bus::tick`), and read /
    /// write paths that observe timer state call it first so the
    /// values they see match the cycle they observe at.
    pub fn advance_to(&mut self, now: u64, hsync_period: u64, dot_clock_divisor: u64) -> u8 {
        let delta = now.saturating_sub(self.last_advance_cycle);
        if delta == 0 {
            return 0;
        }
        let mut fired: u8 = 0;
        for i in 0..3 {
            if self.advance_timer(i, delta, hsync_period, dot_clock_divisor) {
                fired |= 1 << i;
            }
        }
        self.last_advance_cycle = now;
        fired
    }

    /// Sync the lazy clock to `now` without advancing any state.
    /// Used by the bus when it discards skipped cycles (e.g.
    /// post-warmup resets in tests) — calling this prevents
    /// `advance_to` from later seeing a huge backlog and trying
    /// to fast-forward through millions of cycles in one go.
    #[allow(dead_code)]
    pub fn sync_clock_to(&mut self, now: u64) {
        self.last_advance_cycle = now;
    }

    /// Is this timer currently paused per its sync-mode bits?
    ///
    /// Redux does not model Timer 0/1 sync pauses; it only changes the
    /// selected clock rate and handles Timer 1 sync-mode-1's VBlank
    /// reset. Match that behavior for lockstep. Timer 2 modes 1/2 are
    /// honest pauses.
    fn is_timer_paused(&self, idx: usize) -> bool {
        let t = &self.timers[idx];
        if t.mode & MODE_SYNC_ENABLE == 0 {
            return false;
        }
        let sync = (t.mode >> 1) & 3;
        match (idx, sync) {
            // Timer 2 sync-mode-1 / 2: stop counter.
            (2, 1) | (2, 2) => true,
            _ => false,
        }
    }

    /// Pulse a VBlank event to the timer bank. Called by the bus
    /// when `EventSlot::VBlank` fires. Timer 1 sync-mode-1 ("reset at
    /// VBlank") resets its counter to 0.
    pub fn notify_vblank(&mut self) {
        // Sync-mode-1 for Timer 1: reset on VBlank.
        let t1 = &mut self.timers[1];
        if t1.mode & MODE_SYNC_ENABLE != 0 && ((t1.mode >> 1) & 3) == 1 {
            t1.counter = 0;
            t1.accum = 0;
        }
    }

    fn advance_timer(
        &mut self,
        idx: usize,
        cycles: u64,
        hsync_period: u64,
        dot_clock_divisor: u64,
    ) -> bool {
        // Sync-mode gating — decide whether this timer is
        // currently paused.
        let paused = self.is_timer_paused(idx);
        if paused {
            return false;
        }

        let t = &mut self.timers[idx];
        let source = (t.mode >> 8) & 0x3;

        // Convert `cycles` system clocks into source clocks.
        let ticks = match (idx, source) {
            // Timer 0 source 1/3 = dot clock.
            (0, 1) | (0, 3) => {
                t.accum += cycles;
                let divisor = dot_clock_divisor.max(1);
                let n = t.accum / divisor;
                t.accum %= divisor;
                n
            }
            // Timer 1 source 1/3 = HBlank: one tick per HSync period.
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
        //
        // Redux (`psxcounters.cc:reset`) wraps at exactly `target*rate`
        // cycles per lap — the counter visits 0..=target-1 and a reset
        // fires the instant it would reach `target`, snapping it back
        // to 0. We were wrapping at `(target+1)*rate` (visiting
        // 0..=target and wrapping on the tick after), losing one count
        // per lap. That surfaced at parity step 79,389,318 as Timer 1
        // reading one less than Redux a lap after reset.
        if t.mode & MODE_RESET_AT_TARGET != 0 && target != 0 {
            if new_val >= target as u64 {
                reached_target = true;
                new_val %= target as u64;
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

    #[test]
    fn timer0_dot_clock_source_ticks_at_divisor_rate() {
        let mut t = Timers::new();
        // Set Timer 0 mode with clock source = 1 (dot clock).
        t.write32(0x1F80_1104, 1 << 8, 0);
        // At 320-pixel resolution, divisor = 8 (system clocks per dot).
        // Advance by 80 cycles → 10 dot-clock ticks.
        let fired = t.tick(80, 2146, 8);
        assert_eq!(fired, 0);
        assert_eq!(t.read32(0x1F80_1100) & 0xFFFF, 10);
        // Another 40 cycles → 5 more dot-clock ticks.
        t.tick(40, 2146, 8);
        assert_eq!(t.read32(0x1F80_1100) & 0xFFFF, 15);
    }

    #[test]
    fn timer0_system_clock_source_unaffected_by_divisor() {
        let mut t = Timers::new();
        t.write32(0x1F80_1104, 0, 0); // source 0 = system clock
                                      // Divisor changes shouldn't matter at system-clock source.
        t.tick(100, 2146, 8);
        assert_eq!(t.read32(0x1F80_1100) & 0xFFFF, 100);
    }

    #[test]
    fn timer1_sync_mode_3_free_runs_like_redux() {
        let mut t = Timers::new();
        // Redux ignores the Timer 1 sync-mode-3 pause and only uses
        // the selected clock source/rate.
        t.write32(0x1F80_1114, MODE_SYNC_ENABLE | (3 << 1), 0);
        t.tick(500, 2146, 8);
        assert_eq!(t.read32(0x1F80_1110) & 0xFFFF, 500);
        t.notify_vblank();
        t.tick(500, 2146, 8);
        assert_eq!(t.read32(0x1F80_1110) & 0xFFFF, 1000);
    }

    #[test]
    fn timer1_sync_mode_1_resets_on_vblank() {
        let mut t = Timers::new();
        // Sync enable + sync mode 1 (reset at VBlank).
        t.write32(0x1F80_1114, MODE_SYNC_ENABLE | (1 << 1), 0);
        t.tick(200, 2146, 8);
        assert_eq!(t.read32(0x1F80_1110) & 0xFFFF, 200);
        t.notify_vblank();
        assert_eq!(t.read32(0x1F80_1110) & 0xFFFF, 0);
    }

    #[test]
    fn timer2_sync_mode_1_stops_counter() {
        let mut t = Timers::new();
        // Sync enable + sync mode 1 on Timer 2 = stop counter.
        t.write32(0x1F80_1124, MODE_SYNC_ENABLE | (1 << 1), 0);
        t.tick(100, 2146, 8);
        assert_eq!(t.read32(0x1F80_1120) & 0xFFFF, 0);
    }

    #[test]
    fn timer2_sync_mode_0_free_runs() {
        let mut t = Timers::new();
        // Sync enable + sync mode 0 on Timer 2 = free-run.
        t.write32(0x1F80_1124, MODE_SYNC_ENABLE, 0);
        t.tick(100, 2146, 8);
        assert_eq!(t.read32(0x1F80_1120) & 0xFFFF, 100);
    }

    #[test]
    fn reading_mode_clears_reached_flags() {
        let mut t = Timers::new();
        t.timers[1].mode = MODE_REACHED_TARGET | MODE_REACHED_WRAP | MODE_IRQ_ACTIVE_LOW;
        assert_eq!(
            t.read32(0x1F80_1114),
            MODE_REACHED_TARGET | MODE_REACHED_WRAP | MODE_IRQ_ACTIVE_LOW
        );
        assert_eq!(t.timers[1].mode, MODE_IRQ_ACTIVE_LOW);
    }
}
