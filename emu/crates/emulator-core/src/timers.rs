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
}

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
    pub fn write32(&mut self, phys: u32, value: u32) {
        let (idx, off) = decode(phys);
        let t = &mut self.timers[idx];
        let v16 = value & 0xFFFF;
        match off {
            0x0 => t.counter = v16,
            0x4 => {
                t.mode = value;
                // A real timer resets the counter to 0 on every mode
                // write; we'll model that once ticking is wired up.
                t.counter = 0;
            }
            0x8 => t.target = v16,
            _ => {}
        }
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
        t.write32(0x1F80_1100, 0x1234);
        assert_eq!(t.read32(0x1F80_1100), 0x1234);

        t.write32(0x1F80_1108, 0xABCD);
        assert_eq!(t.read32(0x1F80_1108), 0xABCD);
    }

    #[test]
    fn mode_write_resets_counter() {
        let mut t = Timers::new();
        t.write32(0x1F80_1100, 0xFF);
        t.write32(0x1F80_1104, 0x0001); // mode write — should reset counter
        assert_eq!(t.read32(0x1F80_1100), 0);
    }

    #[test]
    fn upper_bits_masked_to_16() {
        let mut t = Timers::new();
        t.write32(0x1F80_1100, 0x1234_5678);
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
