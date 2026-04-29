/// Target address for a conditional branch. The 16-bit immediate is
/// sign-extended and shifted left by 2, then added to the delay slot's
/// PC (`pc + 4`).
pub(super) fn branch_target(pc: u32, instr: u32) -> u32 {
    let offset = (instr as i16) as i32;
    let delay_slot_pc = pc.wrapping_add(4);
    delay_slot_pc.wrapping_add((offset << 2) as u32)
}
