//! MIPS R3000A CPU.
//!
//! Instruction set coverage grows one opcode at a time, each added
//! alongside a parity assertion against PCSX-Redux. The decoder itself
//! is intentionally a flat match on the primary opcode field — we'll
//! refactor to a jump table only if a profiler says to.

use psx_hw::memory;
use psx_trace::InstructionRecord;
use thiserror::Error;

use crate::bus::Bus;

/// Errors raised during instruction execution.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    /// Decoder encountered a primary opcode we haven't implemented yet.
    #[error("unimplemented primary opcode {opcode:#04x} at pc={pc:#010x} (instr={instr:#010x})")]
    Unimplemented {
        /// Primary opcode field (bits 31..=26).
        opcode: u8,
        /// PC at which the offending instruction was fetched.
        pc: u32,
        /// Raw 32-bit instruction word.
        instr: u32,
    },

    /// Decoder encountered a SPECIAL funct value we haven't implemented yet.
    #[error("unimplemented SPECIAL funct {funct:#04x} at pc={pc:#010x} (instr={instr:#010x})")]
    UnimplementedSpecial {
        /// Function field (bits 5..=0) for primary opcode 0.
        funct: u8,
        /// PC at which the offending instruction was fetched.
        pc: u32,
        /// Raw 32-bit instruction word.
        instr: u32,
    },
}

/// MIPS R3000A CPU state.
pub struct Cpu {
    pc: u32,
    gprs: [u32; 32],
    /// COP0 (System Control) registers: SR, Cause, EPC, BadVaddr, etc.
    /// Most of these are untouched by early BIOS init; the important
    /// one early on is `SR` (index 12) which gates interrupts and
    /// cache isolation.
    cop0: [u32; 32],
    /// Monotonically increasing step counter. We use "instructions
    /// retired" as our tick metric; cycle-accurate timing lands when
    /// the scheduler does.
    tick: u64,
    /// Branch-delay slot machinery. A branch instruction sets this to
    /// `Some(target)`; the *next* instruction executes as the delay
    /// slot, and *after* it retires PC jumps to `target` instead of
    /// the usual `pc + 4`.
    pending_pc: Option<u32>,
    /// One-slot load-delay machinery. `LW` (and friends) stage their
    /// result here instead of committing immediately; the commit
    /// happens at the end of the *next* instruction's `step`, so the
    /// delay-slot instruction observes the old register value — which
    /// is what the R3000A hardware does.
    pending_load: Option<(u8, u32)>,
}

impl Cpu {
    /// Construct a CPU in its reset state.
    ///
    /// PC is seated at the KSEG1 BIOS reset vector (`0xBFC0_0000`) so
    /// the first fetch goes through the uncached path, matching
    /// hardware behaviour at power-on.
    pub fn new() -> Self {
        Self {
            pc: memory::bios::RESET_VECTOR,
            gprs: [0; 32],
            cop0: [0; 32],
            tick: 0,
            pending_pc: None,
            pending_load: None,
        }
    }

    /// Current program counter.
    #[inline]
    pub fn pc(&self) -> u32 {
        self.pc
    }

    /// Read a general-purpose register. `$0` always reads as zero.
    #[inline]
    pub fn gpr(&self, index: u8) -> u32 {
        self.gprs[(index & 31) as usize]
    }

    /// Write a general-purpose register, enforcing the MIPS invariant
    /// that `$0` is hardwired to zero.
    #[inline]
    fn set_gpr(&mut self, index: u8, value: u32) {
        let i = (index & 31) as usize;
        if i != 0 {
            self.gprs[i] = value;
        }
    }

    /// Fetch the 32-bit instruction word at the current PC without
    /// advancing it. Exposed for diagnostic tools; `step` uses it
    /// internally too.
    pub fn fetch(&self, bus: &Bus) -> u32 {
        bus.read32(self.pc)
    }

    /// Execute one instruction and return a trace record of the state
    /// after retirement.
    pub fn step(&mut self, bus: &mut Bus) -> Result<InstructionRecord, ExecutionError> {
        let pc_before = self.pc;
        let instr = bus.read32(pc_before);

        // If the *previous* instruction was a branch, the current
        // instruction is its delay slot — after retiring, PC goes to
        // the branch target instead of the usual `pc + 4`.
        let branch_after_this = self.pending_pc.take();

        // The load delay queued by the *previous* instruction is held
        // and applied after this instruction executes. The delay slot
        // itself sees the pre-load value; the commit happens after.
        // Any new load this instruction issues will be picked up on
        // the *next* call to `step`.
        let load_to_commit = self.pending_load.take();

        self.execute(instr, pc_before, bus)?;

        if let Some((reg, value)) = load_to_commit {
            self.set_gpr(reg, value);
        }

        self.pc = match branch_after_this {
            Some(target) => target,
            None => self.pc.wrapping_add(4),
        };
        self.tick += 1;

        Ok(InstructionRecord {
            tick: self.tick,
            pc: pc_before,
            instr,
            gprs: self.gprs,
        })
    }

    /// Decode and execute a single instruction. Does not advance PC
    /// (the caller is responsible, to keep branch-delay handling
    /// localised to the branch opcodes that will add it later).
    fn execute(&mut self, instr: u32, pc: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let opcode = ((instr >> 26) & 0x3F) as u8;
        match opcode {
            0x00 => self.dispatch_special(instr, pc),
            0x02 => self.op_j(instr, pc),
            0x04 => self.op_beq(instr, pc),
            0x05 => self.op_bne(instr, pc),
            0x08 => self.op_addi(instr),
            0x09 => self.op_addiu(instr),
            0x0D => self.op_ori(instr),
            0x0F => self.op_lui(instr),
            0x10 => self.dispatch_cop0(instr, pc),
            0x23 => self.op_lw(instr, bus),
            0x2B => self.op_sw(instr, bus),
            _ => Err(ExecutionError::Unimplemented { opcode, pc, instr }),
        }
    }

    /// Dispatch table for COP0 instructions (primary opcode `0x10`).
    /// The sub-operation lives in bits 25..=21.
    fn dispatch_cop0(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let cop_op = ((instr >> 21) & 0x1F) as u8;
        match cop_op {
            0x04 => self.op_mtc0(instr),
            _ => Err(ExecutionError::Unimplemented {
                opcode: 0x10,
                pc,
                instr,
            }),
        }
    }

    /// `MTC0 rt, rd` — move from CPU GPR `rt` to COP0 register `rd`.
    fn op_mtc0(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as usize;
        self.cop0[rd] = self.gpr(rt);
        Ok(())
    }

    /// Dispatch table for primary-opcode `SPECIAL` (0x00), selected by
    /// the 6-bit function field in bits 5..=0.
    fn dispatch_special(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let funct = (instr & 0x3F) as u8;
        match funct {
            0x00 => self.op_sll(instr),
            0x25 => self.op_or(instr),
            0x2A => self.op_slt(instr),
            0x2B => self.op_sltu(instr),
            _ => Err(ExecutionError::UnimplementedSpecial { funct, pc, instr }),
        }
    }

    /// `LUI rt, imm16` — load upper immediate: `rt = imm16 << 16`.
    fn op_lui(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let imm = instr & 0xFFFF;
        self.set_gpr(rt, imm << 16);
        Ok(())
    }

    /// `ADDI rt, rs, imm16` — add sign-extended immediate.
    ///
    /// Like `ADDIU` but raises an overflow exception on signed
    /// overflow. Overflow handling is not yet implemented; for the
    /// hand-written BIOS sequences we've encountered so far the
    /// arithmetic doesn't overflow, so for now we treat this as ADDIU.
    /// TODO: raise `Overflow` exception when `rs + imm` overflows i32.
    fn op_addi(&mut self, instr: u32) -> Result<(), ExecutionError> {
        self.op_addiu(instr)
    }

    /// `ADDIU rt, rs, imm16` — add sign-extended immediate, no overflow trap:
    /// `rt = rs + sign_extend(imm16)`.
    ///
    /// Despite the "U" in the name, both operands are interpreted with
    /// the same bit pattern; the difference from `ADDI` is only that
    /// arithmetic overflow does not raise an exception.
    fn op_addiu(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let imm = (instr as i16) as i32 as u32;
        self.set_gpr(rt, self.gpr(rs).wrapping_add(imm));
        Ok(())
    }

    /// `ORI rt, rs, imm16` — bitwise OR with zero-extended immediate:
    /// `rt = rs | imm16`.
    fn op_ori(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let imm = instr & 0xFFFF;
        self.set_gpr(rt, self.gpr(rs) | imm);
        Ok(())
    }

    /// `OR rd, rs, rt` — bitwise OR of two registers: `rd = rs | rt`.
    fn op_or(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, self.gpr(rs) | self.gpr(rt));
        Ok(())
    }

    /// `SLT rd, rs, rt` — set-less-than, signed: `rd = (rs < rt) ? 1 : 0`.
    fn op_slt(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let lhs = self.gpr(rs) as i32;
        let rhs = self.gpr(rt) as i32;
        self.set_gpr(rd, (lhs < rhs) as u32);
        Ok(())
    }

    /// `SLTU rd, rs, rt` — set-less-than, unsigned.
    fn op_sltu(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, (self.gpr(rs) < self.gpr(rt)) as u32);
        Ok(())
    }

    /// `SLL rd, rt, sa` — shift left logical by `sa` bits.
    /// When `rd = rt = sa = 0`, the whole encoding is `0x0000_0000`,
    /// which is the canonical `NOP`.
    fn op_sll(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let sa = (instr >> 6) & 0x1F;
        self.set_gpr(rd, self.gpr(rt) << sa);
        Ok(())
    }

    /// `J target` — unconditional jump. The 26-bit `target` is left-shifted
    /// by 2 and merged with the top 4 bits of the delay-slot's PC
    /// (`pc + 4`) to form the absolute destination.
    ///
    /// The delay slot (the instruction immediately after the jump)
    /// executes before PC actually lands at the target — this happens
    /// via [`Cpu::step`]'s `pending_pc` handling.
    fn op_j(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let target_field = instr & 0x03FF_FFFF;
        let delay_slot_pc = pc.wrapping_add(4);
        let target = (delay_slot_pc & 0xF000_0000) | (target_field << 2);
        self.pending_pc = Some(target);
        Ok(())
    }

    /// `BEQ rs, rt, offset` — branch (delay-slotted) if `rs == rt`.
    /// Target = `(pc + 4) + (sign_extend(offset) << 2)`.
    fn op_beq(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        if self.gpr(rs) == self.gpr(rt) {
            self.pending_pc = Some(branch_target(pc, instr));
        }
        Ok(())
    }

    /// `BNE rs, rt, offset` — branch (delay-slotted) if `rs != rt`.
    fn op_bne(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        if self.gpr(rs) != self.gpr(rt) {
            self.pending_pc = Some(branch_target(pc, instr));
        }
        Ok(())
    }

    /// `LW rt, offset(rs)` — load word: `rt = mem[rs + sign_ext(offset)]`.
    ///
    /// The R3000A has a one-slot load delay: the loaded value lands in
    /// `rt` at the end of the *next* instruction, not this one. We
    /// stage the load in `pending_load`; [`Cpu::step`] commits it
    /// after the following instruction executes.
    fn op_lw(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let value = bus.read32(addr);
        // Loads to $zero are no-ops; never queue a commit to it.
        if rt != 0 {
            self.pending_load = Some((rt, value));
        }
        Ok(())
    }

    /// `SW rt, offset(rs)` — store word: `mem[rs + sign_ext(offset)] = rt`.
    ///
    /// If COP0 Status Register bit 16 (IsC — isolate cache) is set, the
    /// store is redirected away from main memory. The BIOS uses this
    /// during init to scrub the data cache without touching RAM. We
    /// model the side effect by simply dropping the write — matching
    /// PS1 hardware, which has no separate D-cache for us to populate.
    fn op_sw(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        if self.cache_isolated() {
            return Ok(());
        }
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        bus.write32(addr, self.gpr(rt));
        Ok(())
    }

    /// COP0 SR bit 16 (IsC). When set, D-cache is isolated from memory
    /// and loads/stores don't reach RAM.
    #[inline]
    fn cache_isolated(&self) -> bool {
        self.cop0[12] & (1 << 16) != 0
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Target address for a conditional branch. The 16-bit immediate is
/// sign-extended and shifted left by 2, then added to the delay slot's
/// PC (`pc + 4`).
fn branch_target(pc: u32, instr: u32) -> u32 {
    let offset = (instr as i16) as i32;
    let delay_slot_pc = pc.wrapping_add(4);
    delay_slot_pc.wrapping_add((offset << 2) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_bios_with_first_word(word: u32) -> Vec<u8> {
        let mut bios = vec![0u8; memory::bios::SIZE];
        bios[0..4].copy_from_slice(&word.to_le_bytes());
        bios
    }

    #[test]
    fn reset_state_points_at_bios_reset_vector() {
        let cpu = Cpu::new();
        assert_eq!(cpu.pc(), 0xBFC0_0000);
    }

    #[test]
    fn reset_state_has_zeroed_registers() {
        let cpu = Cpu::new();
        for i in 0..32 {
            assert_eq!(cpu.gpr(i), 0);
        }
    }

    #[test]
    fn fetch_returns_first_bios_word() {
        // Real stock BIOSes (SCPH1001 / 5500 / 5501 / 5502) all begin with
        // `lui $t0, 0x0013` = 0x3C08_0013 as part of cache-control init.
        let bus = Bus::new(synthetic_bios_with_first_word(0x3C08_0013)).unwrap();
        let cpu = Cpu::new();
        assert_eq!(cpu.fetch(&bus), 0x3C08_0013);
    }

    #[test]
    fn step_executes_lui_and_advances_pc() {
        // lui $t0, 0x0013 → $t0 = 0x0013_0000, PC += 4
        let mut bus = Bus::new(synthetic_bios_with_first_word(0x3C08_0013)).unwrap();
        let mut cpu = Cpu::new();
        let record = cpu.step(&mut bus).expect("lui decodes");

        assert_eq!(record.pc, 0xBFC0_0000);
        assert_eq!(record.instr, 0x3C08_0013);
        assert_eq!(record.gprs[8], 0x0013_0000); // $t0
        assert_eq!(record.tick, 1);
        assert_eq!(cpu.pc(), 0xBFC0_0004);
    }

    #[test]
    fn lui_to_r0_is_silently_discarded() {
        // lui $0, 0xDEAD — writing to $0 must leave it at zero.
        // opcode=0x0F, rt=0, imm=0xDEAD → 0x3C00_DEAD
        let mut bus = Bus::new(synthetic_bios_with_first_word(0x3C00_DEAD)).unwrap();
        let mut cpu = Cpu::new();
        let record = cpu.step(&mut bus).expect("lui to r0 decodes");
        assert_eq!(record.gprs[0], 0);
    }

    #[test]
    fn unimplemented_opcode_returns_structured_error() {
        // opcode 0x01 = REGIMM (BLTZ/BGEZ family); not implemented yet.
        // Encoding: 0x04000000 = (0x01 << 26) | 0.
        let mut bus = Bus::new(synthetic_bios_with_first_word(0x0400_0000)).unwrap();
        let mut cpu = Cpu::new();
        let err = cpu.step(&mut bus).unwrap_err();
        assert!(matches!(
            err,
            ExecutionError::Unimplemented { opcode: 0x01, .. }
        ));
    }

    #[test]
    fn nop_advances_pc_without_side_effects() {
        // SLL $0, $0, 0 with all fields zero is the canonical NOP.
        let mut bus = Bus::new(synthetic_bios_with_first_word(0x0000_0000)).unwrap();
        let mut cpu = Cpu::new();
        let record = cpu.step(&mut bus).expect("nop decodes");
        assert_eq!(cpu.pc(), 0xBFC0_0004);
        assert!(record.gprs.iter().all(|&v| v == 0));
    }

    #[test]
    fn ori_zero_extends_immediate() {
        // ori $t0, $t1, 0xABCD; opcode=0x0D, rs=9, rt=8, imm=0xABCD
        // Encoding: (0x0D << 26) | (9 << 21) | (8 << 16) | 0xABCD = 0x352 8ABCD
        let mut bus = Bus::new(synthetic_bios_with_first_word(0x3528_ABCD)).unwrap();
        let mut cpu = Cpu::new();
        cpu.gprs[9] = 0xFFFF_0000; // $t1 = 0xFFFF0000
        let record = cpu.step(&mut bus).expect("ori decodes");
        assert_eq!(record.gprs[8], 0xFFFF_ABCD);
    }
}
