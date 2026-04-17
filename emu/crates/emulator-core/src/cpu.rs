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
    /// Load staged at the start of `step` and held during `execute`.
    /// Separate from `pending_load` so `set_gpr` can look at it and
    /// squash a same-register writeback when a non-load in the delay
    /// slot writes the same target — R3000 load-delay semantics.
    committing_load: Option<(u8, u32)>,
    hi: u32,
    lo: u32,
    /// When a SYSCALL/BREAK/exception fires, the post-retire PC goes
    /// here instead of `pc + 4` or a pending branch target. The value
    /// is the exception vector (0x8000_0080 or 0xBFC0_0180 depending on
    /// the BEV bit in SR).
    pending_exception_pc: Option<u32>,
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
            committing_load: None,
            hi: 0,
            lo: 0,
            pending_exception_pc: None,
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

    /// All 32 general-purpose registers. Useful for UI snapshots.
    #[inline]
    pub fn gprs(&self) -> &[u32; 32] {
        &self.gprs
    }

    /// COP0 register file — System Control coprocessor state (SR, Cause,
    /// EPC, BadVAddr, …).
    #[inline]
    pub fn cop0(&self) -> &[u32; 32] {
        &self.cop0
    }

    /// HI half of the multiply/divide result register.
    #[inline]
    pub fn hi(&self) -> u32 {
        self.hi
    }

    /// LO half of the multiply/divide result register.
    #[inline]
    pub fn lo(&self) -> u32 {
        self.lo
    }

    /// Retired-instruction counter since reset.
    #[inline]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Write a general-purpose register, enforcing the MIPS invariant
    /// that `$0` is hardwired to zero.
    ///
    /// Also implements R3000 load-delay squashing: if a load is about
    /// to commit into the same register this instruction is writing,
    /// the load's writeback is cancelled. The hardware only has one
    /// writeback port per GPR, and if the delay slot non-load-wise
    /// writes to the load's target, its value wins — the load is lost.
    #[inline]
    fn set_gpr(&mut self, index: u8, value: u32) {
        let i = (index & 31) as usize;
        if i != 0 {
            self.gprs[i] = value;
        }
        if let Some((reg, _)) = &self.committing_load {
            if *reg == index {
                self.committing_load = None;
            }
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
        // Mirror the external interrupt line into COP0.CAUSE.IP[2]
        // each step. Software reads CAUSE directly to poll for
        // interrupts even when globally disabled.
        self.sync_external_interrupt(bus);

        let pc_before = self.pc;
        let instr = bus.read32(pc_before);

        // If an external interrupt is pending AND enabled AND globally
        // allowed, take the exception instead of running this instruction.
        // We still record the pre-empted PC+instr so the trace reads as
        // "we were about to run X when IRQ n fired" — matching how
        // typical MIPS emulators surface this in per-instruction hooks.
        if self.should_take_interrupt() {
            let in_delay_slot = self.pending_pc.is_some();
            self.pending_pc = None;
            self.enter_exception(ExceptionCode::Interrupt, pc_before, in_delay_slot);
            self.pc = self
                .pending_exception_pc
                .take()
                .expect("enter_exception staged a vector");
            self.tick += 1;
            return Ok(InstructionRecord {
                tick: self.tick,
                pc: pc_before,
                instr,
                gprs: self.gprs,
            });
        }

        // If the *previous* instruction was a branch, the current
        // instruction is its delay slot — after retiring, PC goes to
        // the branch target instead of the usual `pc + 4`.
        let branch_after_this = self.pending_pc.take();
        let in_delay_slot = branch_after_this.is_some();

        // The load delay queued by the *previous* instruction is held
        // in `committing_load` for the duration of `execute`. The
        // delay slot itself sees the pre-load value; any `set_gpr`
        // in execute that targets the load's register will squash
        // the commit (R3000 writeback-port collision — the
        // non-load write wins). Any new load this instruction issues
        // goes into `pending_load` and fires on the *next* call.
        self.committing_load = self.pending_load.take();

        self.execute(instr, pc_before, in_delay_slot, bus)?;

        if let Some((reg, value)) = self.committing_load.take() {
            let i = (reg & 31) as usize;
            if i != 0 {
                self.gprs[i] = value;
            }
        }

        // Exception takes priority: it cancels any pending branch and
        // redirects PC to the exception vector.
        self.pc = if let Some(exc_pc) = self.pending_exception_pc.take() {
            exc_pc
        } else {
            match branch_after_this {
                Some(target) => target,
                None => self.pc.wrapping_add(4),
            }
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
    ///
    /// `in_delay_slot` is `true` when the current instruction sits in
    /// the delay slot of a taken branch — the SYSCALL/BREAK handlers
    /// need it to set the BD bit in `CAUSE` correctly.
    fn execute(
        &mut self,
        instr: u32,
        pc: u32,
        in_delay_slot: bool,
        bus: &mut Bus,
    ) -> Result<(), ExecutionError> {
        let opcode = ((instr >> 26) & 0x3F) as u8;
        match opcode {
            0x00 => self.dispatch_special(instr, pc, in_delay_slot),
            0x01 => self.dispatch_regimm(instr, pc),
            0x02 => self.op_j(instr, pc),
            0x03 => self.op_jal(instr, pc),
            0x04 => self.op_beq(instr, pc),
            0x05 => self.op_bne(instr, pc),
            0x06 => self.op_blez(instr, pc),
            0x07 => self.op_bgtz(instr, pc),
            0x08 => self.op_addi(instr),
            0x09 => self.op_addiu(instr),
            0x0A => self.op_slti(instr),
            0x0B => self.op_sltiu(instr),
            0x0C => self.op_andi(instr),
            0x0D => self.op_ori(instr),
            0x0E => self.op_xori(instr),
            0x0F => self.op_lui(instr),
            0x10 => self.dispatch_cop0(instr, pc),
            0x20 => self.op_lb(instr, bus),
            0x21 => self.op_lh(instr, bus),
            0x23 => self.op_lw(instr, bus),
            0x24 => self.op_lbu(instr, bus),
            0x25 => self.op_lhu(instr, bus),
            0x28 => self.op_sb(instr, bus),
            0x29 => self.op_sh(instr, bus),
            0x2B => self.op_sw(instr, bus),
            _ => Err(ExecutionError::Unimplemented { opcode, pc, instr }),
        }
    }

    /// Dispatch table for COP0 instructions (primary opcode `0x10`).
    /// The sub-operation lives in bits 25..=21.
    fn dispatch_cop0(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let cop_op = ((instr >> 21) & 0x1F) as u8;
        match cop_op {
            0x00 => self.op_mfc0(instr),
            0x04 => self.op_mtc0(instr),
            0x10 => self.op_rfe(),
            _ => Err(ExecutionError::Unimplemented {
                opcode: 0x10,
                pc,
                instr,
            }),
        }
    }

    fn dispatch_regimm(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        match rt {
            0x00 => self.op_bltz(instr, pc),
            0x01 => self.op_bgez(instr, pc),
            0x10 => self.op_bltzal(instr, pc),
            0x11 => self.op_bgezal(instr, pc),
            _ => Err(ExecutionError::Unimplemented { opcode: 0x01, pc, instr }),
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
    fn dispatch_special(
        &mut self,
        instr: u32,
        pc: u32,
        in_delay_slot: bool,
    ) -> Result<(), ExecutionError> {
        let funct = (instr & 0x3F) as u8;
        match funct {
            0x00 => self.op_sll(instr),
            0x02 => self.op_srl(instr),
            0x03 => self.op_sra(instr),
            0x04 => self.op_sllv(instr),
            0x06 => self.op_srlv(instr),
            0x07 => self.op_srav(instr),
            0x08 => self.op_jr(instr, pc),
            0x09 => self.op_jalr(instr, pc),
            0x0C => self.op_syscall(pc, in_delay_slot),
            0x0D => self.op_break(pc, in_delay_slot),
            0x10 => self.op_mfhi(instr),
            0x11 => self.op_mthi(instr),
            0x12 => self.op_mflo(instr),
            0x13 => self.op_mtlo(instr),
            0x18 => self.op_mult(instr),
            0x19 => self.op_multu(instr),
            0x1A => self.op_div(instr),
            0x1B => self.op_divu(instr),
            0x20 => self.op_add(instr),
            0x21 => self.op_addu(instr),
            0x22 => self.op_sub(instr),
            0x23 => self.op_subu(instr),
            0x24 => self.op_and(instr),
            0x25 => self.op_or(instr),
            0x26 => self.op_xor(instr),
            0x27 => self.op_nor(instr),
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

    /// `ADDU rd, rs, rt` — add unsigned (no overflow trap): `rd = rs + rt`.
    fn op_addu(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, self.gpr(rs).wrapping_add(self.gpr(rt)));
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

    fn op_mfc0(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as usize;
        if rt != 0 {
            self.pending_load = Some((rt, self.cop0[rd]));
        }
        Ok(())
    }

    fn op_rfe(&mut self) -> Result<(), ExecutionError> {
        // Restore previous interrupt enable/mode bits in SR.
        let sr = self.cop0[12];
        let restored = (sr & !0b1111) | ((sr >> 2) & 0b1111);
        self.cop0[12] = restored;
        Ok(())
    }

    fn op_jal(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let target_field = instr & 0x03FF_FFFF;
        let delay_slot_pc = pc.wrapping_add(4);
        let target = (delay_slot_pc & 0xF000_0000) | (target_field << 2);
        self.set_gpr(31, delay_slot_pc.wrapping_add(4));
        self.pending_pc = Some(target);
        Ok(())
    }

    fn op_jr(&mut self, instr: u32, _pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        self.pending_pc = Some(self.gpr(rs));
        Ok(())
    }

    fn op_jalr(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let target = self.gpr(rs);
        self.set_gpr(rd, pc.wrapping_add(8));
        self.pending_pc = Some(target);
        Ok(())
    }

    fn op_blez(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        if (self.gpr(rs) as i32) <= 0 {
            self.pending_pc = Some(branch_target(pc, instr));
        }
        Ok(())
    }

    fn op_bgtz(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        if (self.gpr(rs) as i32) > 0 {
            self.pending_pc = Some(branch_target(pc, instr));
        }
        Ok(())
    }

    fn op_bltz(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        if (self.gpr(rs) as i32) < 0 {
            self.pending_pc = Some(branch_target(pc, instr));
        }
        Ok(())
    }

    fn op_bgez(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        if (self.gpr(rs) as i32) >= 0 {
            self.pending_pc = Some(branch_target(pc, instr));
        }
        Ok(())
    }

    fn op_bltzal(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        self.set_gpr(31, pc.wrapping_add(8));
        if (self.gpr(rs) as i32) < 0 {
            self.pending_pc = Some(branch_target(pc, instr));
        }
        Ok(())
    }

    fn op_bgezal(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        self.set_gpr(31, pc.wrapping_add(8));
        if (self.gpr(rs) as i32) >= 0 {
            self.pending_pc = Some(branch_target(pc, instr));
        }
        Ok(())
    }

    fn op_slti(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let imm = (instr as i16) as i32;
        self.set_gpr(rt, ((self.gpr(rs) as i32) < imm) as u32);
        Ok(())
    }

    fn op_sltiu(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let imm = (instr as i16) as i32 as u32;
        self.set_gpr(rt, (self.gpr(rs) < imm) as u32);
        Ok(())
    }

    fn op_andi(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let imm = instr & 0xFFFF;
        self.set_gpr(rt, self.gpr(rs) & imm);
        Ok(())
    }

    fn op_xori(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let imm = instr & 0xFFFF;
        self.set_gpr(rt, self.gpr(rs) ^ imm);
        Ok(())
    }

    fn op_lb(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let value = bus.read8(addr) as i8 as i32 as u32;
        if rt != 0 {
            self.pending_load = Some((rt, value));
        }
        Ok(())
    }

    fn op_lbu(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let value = bus.read8(addr) as u32;
        if rt != 0 {
            self.pending_load = Some((rt, value));
        }
        Ok(())
    }

    fn op_lh(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let value = bus.read16(addr) as i16 as i32 as u32;
        if rt != 0 {
            self.pending_load = Some((rt, value));
        }
        Ok(())
    }

    fn op_lhu(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let value = bus.read16(addr) as u32;
        if rt != 0 {
            self.pending_load = Some((rt, value));
        }
        Ok(())
    }

    fn op_sb(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        if self.cache_isolated() {
            return Ok(());
        }
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        bus.write8(addr, self.gpr(rt) as u8);
        Ok(())
    }

    fn op_sh(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        if self.cache_isolated() {
            return Ok(());
        }
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        bus.write16(addr, self.gpr(rt) as u16);
        Ok(())
    }

    fn op_srl(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let sa = (instr >> 6) & 0x1F;
        self.set_gpr(rd, self.gpr(rt) >> sa);
        Ok(())
    }

    fn op_sra(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let sa = (instr >> 6) & 0x1F;
        self.set_gpr(rd, ((self.gpr(rt) as i32) >> sa) as u32);
        Ok(())
    }

    fn op_sllv(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, self.gpr(rt) << (self.gpr(rs) & 0x1F));
        Ok(())
    }

    fn op_srlv(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, self.gpr(rt) >> (self.gpr(rs) & 0x1F));
        Ok(())
    }

    fn op_srav(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, ((self.gpr(rt) as i32) >> (self.gpr(rs) & 0x1F)) as u32);
        Ok(())
    }

    fn op_add(&mut self, instr: u32) -> Result<(), ExecutionError> {
        // TODO: trap on signed overflow. BIOS code doesn't overflow here.
        self.op_addu(instr)
    }

    fn op_sub(&mut self, instr: u32) -> Result<(), ExecutionError> {
        // TODO: trap on signed overflow.
        self.op_subu(instr)
    }

    fn op_subu(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, self.gpr(rs).wrapping_sub(self.gpr(rt)));
        Ok(())
    }

    fn op_and(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, self.gpr(rs) & self.gpr(rt));
        Ok(())
    }

    fn op_xor(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, self.gpr(rs) ^ self.gpr(rt));
        Ok(())
    }

    fn op_nor(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.set_gpr(rd, !(self.gpr(rs) | self.gpr(rt)));
        Ok(())
    }

    fn op_mfhi(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rd = ((instr >> 11) & 0x1F) as u8;
        let hi = self.hi;
        self.set_gpr(rd, hi);
        Ok(())
    }

    fn op_mthi(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        self.hi = self.gpr(rs);
        Ok(())
    }

    fn op_mflo(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rd = ((instr >> 11) & 0x1F) as u8;
        let lo = self.lo;
        self.set_gpr(rd, lo);
        Ok(())
    }

    fn op_mtlo(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        self.lo = self.gpr(rs);
        Ok(())
    }

    fn op_mult(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let result = (self.gpr(rs) as i32 as i64) * (self.gpr(rt) as i32 as i64);
        self.hi = (result >> 32) as u32;
        self.lo = result as u32;
        Ok(())
    }

    fn op_multu(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let result = (self.gpr(rs) as u64) * (self.gpr(rt) as u64);
        self.hi = (result >> 32) as u32;
        self.lo = result as u32;
        Ok(())
    }

    fn op_div(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let n = self.gpr(rs) as i32;
        let d = self.gpr(rt) as i32;
        if d == 0 {
            self.hi = n as u32;
            self.lo = if n < 0 { 1 } else { u32::MAX };
        } else if n == i32::MIN && d == -1 {
            self.hi = 0;
            self.lo = i32::MIN as u32;
        } else {
            self.hi = (n % d) as u32;
            self.lo = (n / d) as u32;
        }
        Ok(())
    }

    fn op_divu(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let n = self.gpr(rs);
        let d = self.gpr(rt);
        if d == 0 {
            self.hi = n;
            self.lo = u32::MAX;
        } else {
            self.hi = n % d;
            self.lo = n / d;
        }
        Ok(())
    }

    /// COP0 SR bit 16 (IsC). When set, D-cache is isolated from memory
    /// and loads/stores don't reach RAM.
    #[inline]
    fn cache_isolated(&self) -> bool {
        self.cop0[12] & (1 << 16) != 0
    }

    /// Mirror the external IRQ line into `CAUSE.IP[2]`. The CPU reads
    /// this bit to decide whether to take an interrupt exception; the
    /// BIOS also reads it directly to poll.
    fn sync_external_interrupt(&mut self, bus: &Bus) {
        if bus.external_interrupt_pending() {
            self.cop0[13] |= 1 << 10;
        } else {
            self.cop0[13] &= !(1 << 10);
        }
    }

    /// `true` when the CPU should take an interrupt exception right
    /// now: globally enabled (`SR.IEc`), mask-enabled for the pending
    /// source (`SR.IM` over bits 8..=15 of `CAUSE.IP`).
    fn should_take_interrupt(&self) -> bool {
        let sr = self.cop0[12];
        let cause = self.cop0[13];
        let global_enable = sr & 1 != 0;
        let im = (sr >> 8) & 0xFF;
        let ip = (cause >> 8) & 0xFF;
        global_enable && (im & ip) != 0
    }

    /// `SYSCALL` — raise a syscall exception (CAUSE.ExcCode = 8). The
    /// BIOS uses this for every kernel-mode thunk: A/B/C-table calls,
    /// memcpy, printf, event handling, etc.
    fn op_syscall(&mut self, pc: u32, in_delay_slot: bool) -> Result<(), ExecutionError> {
        self.enter_exception(ExceptionCode::Syscall, pc, in_delay_slot);
        Ok(())
    }

    /// `BREAK` — raise a breakpoint exception (CAUSE.ExcCode = 9). Not
    /// hit during normal BIOS boot but cheap to add alongside SYSCALL
    /// since they share the exception-entry plumbing.
    fn op_break(&mut self, pc: u32, in_delay_slot: bool) -> Result<(), ExecutionError> {
        self.enter_exception(ExceptionCode::Break, pc, in_delay_slot);
        Ok(())
    }

    /// Shared exception-entry sequence. Mutates COP0 registers and
    /// stages the exception-vector PC for [`Cpu::step`] to apply.
    ///
    /// - **CAUSE**: write `code` into `ExcCode` (bits 6..=2). Set `BD`
    ///   (bit 31) iff the faulting instruction was in a branch delay
    ///   slot; in that case `EPC` is backed up to the branch PC so
    ///   `RFE` can re-execute the branch.
    /// - **SR**: push the 3-level KU/IE stack — bits `SR[5:0]` become
    ///   `(SR[3:0] << 2)`, with the new current pair (bits 1..0)
    ///   entering kernel-mode / interrupts-disabled.
    /// - **Vector**: `0xBFC0_0180` when `SR.BEV` (bit 22) is set (the
    ///   post-reset default the BIOS boots in), else `0x8000_0080`.
    fn enter_exception(&mut self, code: ExceptionCode, pc: u32, in_delay_slot: bool) {
        let code_bits = (code as u32) & 0x1F;

        let mut cause = self.cop0[13];
        cause &= !((0x1F << 2) | (1 << 31));
        cause |= code_bits << 2;
        if in_delay_slot {
            cause |= 1 << 31;
        }
        self.cop0[13] = cause;

        self.cop0[14] = if in_delay_slot {
            pc.wrapping_sub(4)
        } else {
            pc
        };

        let sr = self.cop0[12];
        self.cop0[12] = (sr & !0x3F) | ((sr & 0x0F) << 2);

        let vector = if sr & (1 << 22) != 0 {
            0xBFC0_0180
        } else {
            0x8000_0080
        };
        self.pending_exception_pc = Some(vector);
    }
}

/// MIPS R3000 exception codes (CAUSE.ExcCode). Only the ones we
/// actively raise are listed; the rest arrive as they're implemented.
#[repr(u8)]
#[derive(Copy, Clone)]
enum ExceptionCode {
    /// External interrupt — asserted by the IRQ controller. See
    /// [`Cpu::should_take_interrupt`] for the gating logic.
    Interrupt = 0,
    Syscall = 8,
    Break = 9,
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
        // opcode 0x22 = LWL (Load Word Left); genuine R3000 opcode we
        // haven't decoded yet. Encoding: 0x88000000 = (0x22 << 26) | 0.
        let mut bus = Bus::new(synthetic_bios_with_first_word(0x8800_0000)).unwrap();
        let mut cpu = Cpu::new();
        let err = cpu.step(&mut bus).unwrap_err();
        assert!(matches!(
            err,
            ExecutionError::Unimplemented { opcode: 0x22, .. }
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

    #[test]
    fn load_delay_squashed_by_same_reg_addiu() {
        // Regression test for the 12.7M-step parity divergence:
        //   lw    $t1, 0($a0)      # stages load delay
        //   addiu $t1, $zero, 1    # delay slot writes $t1 non-load-wise
        //   nop                    # reveals committed $t1
        // R3000 semantics: addiu's write squashes the load's writeback,
        // so after the three instructions $t1 = 1, not the loaded word.
        let mut bios = vec![0u8; memory::bios::SIZE];
        // lw $t1, 0($a0): opcode=0x23, rs=4 ($a0), rt=9 ($t1), offset=0
        bios[0..4].copy_from_slice(&0x8C89_0000u32.to_le_bytes());
        // addiu $t1, $zero, 1: opcode=0x09, rs=0, rt=9, imm=1
        bios[4..8].copy_from_slice(&0x2409_0001u32.to_le_bytes());
        // nop
        bios[8..12].copy_from_slice(&0u32.to_le_bytes());

        let mut bus = Bus::new(bios).unwrap();
        let mut cpu = Cpu::new();
        // $a0 = 0xBFC0_0000 (some RAM-ish address — LW reads whatever's
        // there, which for this test is the BIOS itself / zeroes).
        // Actual loaded value doesn't matter; what matters is that
        // ADDIU's write survives.
        cpu.gprs[4] = 0xBFC0_0000;

        cpu.step(&mut bus).expect("lw");
        cpu.step(&mut bus).expect("addiu in delay slot");
        let record = cpu.step(&mut bus).expect("nop reveals state");

        assert_eq!(record.gprs[9], 1, "addiu must survive LW's delay");
    }
}
