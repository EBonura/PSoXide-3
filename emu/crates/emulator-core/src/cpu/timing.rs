/// One extra cycle bias applied to each instruction retirement.
///
/// Keeping this equal to Redux's `BIAS` is what makes the VBlank
/// scheduler line up on the same instruction as Redux and preserves
/// parity once it turns on.
const BIAS: u32 = 2;

pub(super) fn cycle_cost(_instr: u32) -> u32 {
    BIAS
}
