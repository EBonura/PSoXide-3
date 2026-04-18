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
use crate::gte::Gte;

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
    /// Per-ExcCode (0..=31) count of exception entries. Diagnostic only.
    exception_counts: [u64; 32],
    /// Count of `step()` calls where `bus.external_interrupt_pending()`
    /// was true — answers "did the IRQ line ever go high from the
    /// CPU's point of view?". Diagnostic.
    irq_line_high_steps: u64,
    /// Count of `step()` calls where `should_take_interrupt()` was
    /// true — answers "did we reach the threshold that enters an IRQ
    /// exception?".
    should_take_interrupt_steps: u64,
    /// COP2 — Geometry Transformation Engine. Holds 32 data + 32
    /// control registers and dispatches the GTE function set.
    cop2: Gte,
    /// Depth of nested exception handlers. Incremented on every
    /// exception entry (IRQ, syscall, break) and decremented on
    /// every `RFE`. `in_isr()` returns `true` iff this is > 0.
    /// Counted as depth (not boolean) so that nested RFEs don't
    /// spuriously clear the flag while we're still inside the
    /// outer handler — critical for the parity harness's
    /// aggregation of clean IRQ entries, which must continue
    /// across nested syscalls/IRQs inside the handler body.
    isr_depth: u32,
    /// Latched on the *clean* (depth 0 → 1) entry: `true` iff the
    /// outermost exception was an `Interrupt` (cause=0). Stays set
    /// until `isr_depth` returns to 0 via the final RFE. Mirrors
    /// the `!m_wasInISR && cause == 0` condition in Redux's
    /// `debug.cc:235` early-return that hides the IRQ-handler body
    /// from the recorded trace. Syscall-entered spans clear this to
    /// false and stay that way until the outermost RFE.
    clean_irq_entry: bool,
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
            exception_counts: [0; 32],
            irq_line_high_steps: 0,
            should_take_interrupt_steps: 0,
            cop2: Gte::new(),
            isr_depth: 0,
            clean_irq_entry: false,
        }
    }

    /// `true` while the CPU is inside any exception handler (at any
    /// nesting depth). Mirrors Redux's `m_inISR`. Diagnostic.
    #[inline]
    pub fn in_isr(&self) -> bool {
        self.isr_depth > 0
    }

    /// `true` iff we're still inside the span of a *clean* IRQ
    /// entry — i.e. the outermost handler on the exception stack
    /// was entered via `Interrupt` (cause=0) from user mode. Stays
    /// set across nested exceptions until the outermost RFE. The
    /// parity harness uses this to decide whether to aggregate
    /// handler-body steps into the pre-IRQ record, matching
    /// Redux's `debug.cc:235` early-return which hides the trace
    /// body of clean IRQ entries.
    #[inline]
    pub fn in_irq_handler(&self) -> bool {
        self.clean_irq_entry
    }

    /// COP2 (GTE) state. Diagnostics / UI surfaces only — opcode
    /// dispatch goes through the inherent methods directly.
    #[inline]
    pub fn cop2(&self) -> &Gte {
        &self.cop2
    }

    /// Cumulative exception counts, keyed by CAUSE.ExcCode. Diagnostic.
    #[inline]
    pub fn exception_counts(&self) -> &[u64; 32] {
        &self.exception_counts
    }

    /// How many `step()` calls observed the IRQ line high.
    #[inline]
    pub fn irq_line_high_steps(&self) -> u64 {
        self.irq_line_high_steps
    }

    /// How many `step()` calls would have entered an IRQ exception.
    #[inline]
    pub fn should_take_interrupt_steps(&self) -> u64 {
        self.should_take_interrupt_steps
    }

    /// Seed CPU state from a PSX-EXE header so the next `step()`
    /// enters the homebrew at its entry point. Used for
    /// `PSOXIDE_EXE` side-loading that bypasses the BIOS.
    ///
    /// The corresponding payload must already have been copied into
    /// RAM by [`crate::Bus::load_exe_payload`].
    pub fn seed_from_exe(&mut self, initial_pc: u32, initial_gp: u32, initial_sp: Option<u32>) {
        self.pc = initial_pc;
        self.pending_pc = None;
        self.pending_load = None;
        self.committing_load = None;
        self.pending_exception_pc = None;
        self.set_gpr(28, initial_gp);
        if let Some(sp) = initial_sp {
            self.set_gpr(29, sp);
            // Frame pointer tracks SP on bare-metal boot so backtraces
            // look sane before main() sets up its own frame.
            self.set_gpr(30, sp);
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
    /// internally too. `&mut Bus` because some peripheral reads
    /// (CD-ROM, future DMA) mutate.
    pub fn fetch(&self, bus: &mut Bus) -> u32 {
        bus.read32(self.pc)
    }

    /// Execute one instruction and return a trace record of the state
    /// after retirement.
    pub fn step(&mut self, bus: &mut Bus) -> Result<InstructionRecord, ExecutionError> {
        // Diagnostic only — track how many steps the IRQ pin was high.
        // We deliberately do NOT mirror the pin into `cop0[13].IP[2]`:
        // PCSX-Redux's CAUSE register is only written at exception
        // entry, never live-updated, so software `mfc0 v0, $13` reads
        // see only what the last exception stored. Mirroring the live
        // pin would surface as IP[2]=1 in syscall handlers' CAUSE
        // reads (e.g. step 19258368) and break parity.
        if bus.external_interrupt_pending() {
            self.irq_line_high_steps = self.irq_line_high_steps.saturating_add(1);
        }

        // HLE BIOS: when enabled (only for side-loaded EXEs, never
        // for parity), intercept jumps into the BIOS dispatcher
        // addresses 0xA0 / 0xB0 / 0xC0. The caller's trampoline has
        // already loaded `$t1` with the function number and parked
        // the return address in `$ra`.
        if bus.hle_bios_enabled {
            let args = [self.gpr(4), self.gpr(5), self.gpr(6), self.gpr(7)];
            let t1 = self.gpr(9);
            let ra = self.gpr(31);
            if let Some(out) = crate::hle_bios::dispatch(self.pc, bus, args, t1, ra) {
                self.set_gpr(2, out.v0);
                self.pc = out.next_pc;
                self.pending_pc = None;
                self.pending_load = None;
                self.committing_load = None;
                bus.tick(2);
                self.tick += 1;
                return Ok(InstructionRecord {
                    // Trace records report bus cycles (same unit Redux's
                    // `m_regs.cycle` uses). `self.tick` keeps counting
                    // retired instructions for diagnostics.
                    tick: bus.cycles(),
                    pc: self.pc,
                    instr: 0,
                    gprs: self.gprs,
                });
            }
        }

        let pc_before = self.pc;
        let instr = bus.read32(pc_before);

        // BIAS charged BEFORE the opcode runs — matches Redux's
        // `m_regs.cycle += BIAS` at psxinterpreter.cc:1631, which is
        // *ahead* of the opcode dispatch. Any MMIO reads the opcode
        // issues (Timer counters, GPUSTAT, CDROM status) therefore
        // observe the POST-BIAS cycle. Placing the tick after the
        // opcode would have them observe the pre-BIAS cycle, drifting
        // Timer 1's counter behind Redux's by ~2 cycles per memory
        // access — which showed as a 34-count offset at step
        // 19,472,447's Timer 1 read.
        bus.tick(cycle_cost(instr));

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

        // Hardware-IRQ check, end-of-step. Mirrors Redux's `branchTest`,
        // which only runs when the just-retired instruction was a delay
        // slot (`if (m_inDelaySlot) ... branchTest()`). Doing it after
        // the instruction (rather than before the next) means the trace
        // record still shows the regular instruction at this step, and
        // the interrupt-vector PC shows up at the *next* step — exactly
        // how Redux's trace reads.
        if in_delay_slot && self.should_take_interrupt(bus) {
            self.should_take_interrupt_steps =
                self.should_take_interrupt_steps.saturating_add(1);
            // Redux passes `bd=0` to `exception(0x400, 0)`: the IRQ
            // is taken cleanly between instructions, not in a delay
            // slot of its own.
            self.enter_exception(ExceptionCode::Interrupt, self.pc, false);
            self.pc = self
                .pending_exception_pc
                .take()
                .expect("enter_exception staged a vector");
        }

        self.tick += 1;
        Ok(InstructionRecord {
            // Report bus cycles, not retired-instruction count — the
            // psx-trace docs call this field a "cycle count", Redux's
            // oracle populates it from `m_regs.cycle`, and our IRQ
            // timing is already driven off `bus.cycles()`. Before this
            // change our tick was step-index-based and silently
            // diverged from Redux's by a factor of ~2.3.
            tick: bus.cycles(),
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
            0x12 => self.dispatch_cop2(instr, pc),
            0x20 => self.op_lb(instr, bus),
            0x21 => self.op_lh(instr, bus),
            0x22 => self.op_lwl(instr, bus),
            0x23 => self.op_lw(instr, bus),
            0x24 => self.op_lbu(instr, bus),
            0x25 => self.op_lhu(instr, bus),
            0x26 => self.op_lwr(instr, bus),
            0x28 => self.op_sb(instr, bus),
            0x29 => self.op_sh(instr, bus),
            0x2A => self.op_swl(instr, bus),
            0x2B => self.op_sw(instr, bus),
            0x2E => self.op_swr(instr, bus),
            0x32 => self.op_lwc2(instr, bus),
            0x3A => self.op_swc2(instr, bus),
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

    /// Dispatch table for COP2 (GTE) instructions (primary opcode
    /// `0x12`). Bit 25 selects: when clear, the upper 5 bits of bits
    /// 25..=21 pick MFC2/CFC2/MTC2/CTC2; when set, the bottom 25 bits
    /// encode a GTE function (RTPS, NCLIP, MVMVA, …).
    fn dispatch_cop2(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        if instr & (1 << 25) != 0 {
            self.cop2.execute(instr);
            return Ok(());
        }
        let cop_op = ((instr >> 21) & 0x1F) as u8;
        match cop_op {
            0x00 => self.op_mfc2(instr),
            0x02 => self.op_cfc2(instr),
            0x04 => self.op_mtc2(instr),
            0x06 => self.op_ctc2(instr),
            _ => Err(ExecutionError::Unimplemented {
                opcode: 0x12,
                pc,
                instr,
            }),
        }
    }

    /// `MFC2 rt, rd` — move from COP2 data register `rd` into GPR
    /// `rt`. Like LW, this respects the one-slot load delay so the
    /// next instruction sees the *old* register value.
    fn op_mfc2(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let value = self.cop2.read_data(rd);
        if rt != 0 {
            self.pending_load = Some((rt, value));
        }
        Ok(())
    }

    /// `CFC2 rt, rd` — same as MFC2 but reads a control register.
    fn op_cfc2(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let value = self.cop2.read_control(rd);
        if rt != 0 {
            self.pending_load = Some((rt, value));
        }
        Ok(())
    }

    /// `MTC2 rt, rd` — move from GPR `rt` to COP2 data register `rd`.
    /// Coprocessor writes commit immediately (no delay slot).
    fn op_mtc2(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.cop2.write_data(rd, self.gpr(rt));
        Ok(())
    }

    /// `CTC2 rt, rd` — same as MTC2 but writes a control register.
    fn op_ctc2(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        self.cop2.write_control(rd, self.gpr(rt));
        Ok(())
    }

    /// `LWC2 rt, offset(rs)` — load 32-bit word from memory into COP2
    /// data register `rt`. No GPR is touched, so no load-delay slot.
    fn op_lwc2(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let value = bus.read32(addr);
        bus.add_cycles(1);
        self.cop2.write_data(rt, value);
        Ok(())
    }

    /// `SWC2 rt, offset(rs)` — store COP2 data register `rt` to memory.
    fn op_swc2(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        // Redux's psxSWC2 unconditionally calls write32, which always
        // charges +1 cycle in psxmem.cc. Even cache-isolated stores
        // pay the bus cycle, so charge before the bypass.
        bus.add_cycles(1);
        if self.cache_isolated() {
            return Ok(());
        }
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        bus.write32(addr, self.cop2.read_data(rt));
        Ok(())
    }

    fn dispatch_regimm(&mut self, instr: u32, pc: u32) -> Result<(), ExecutionError> {
        let rt = ((instr >> 16) & 0x1F) as u8;
        match rt {
            0x00 => self.op_bltz(instr, pc),
            0x01 => self.op_bgez(instr, pc),
            0x10 => self.op_bltzal(instr, pc),
            0x11 => self.op_bgezal(instr, pc),
            _ => Err(ExecutionError::Unimplemented {
                opcode: 0x01,
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

    /// `ADDI rt, rs, imm16` — add sign-extended immediate, signed.
    ///
    /// Differs from `ADDIU` in one place: on signed overflow, the
    /// destination register is left unchanged and a 12 (Overflow)
    /// exception fires. Games occasionally rely on the trap for
    /// range-check idioms — treating `ADDI` as `ADDIU` means those
    /// games silently run past what should have been a clamped value.
    fn op_addi(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let imm = (instr as i16) as i32;
        let a = self.gpr(rs) as i32;
        match a.checked_add(imm) {
            Some(sum) => {
                self.set_gpr(rt, sum as u32);
                Ok(())
            }
            None => {
                // Signed overflow — destination unchanged, raise
                // CAUSE.ExcCode = 12 (Overflow). `in_delay_slot` is
                // inferred from the pending branch already staged
                // when Cpu::step dispatched us.
                let in_delay_slot = self.pending_pc.is_some();
                self.enter_exception(ExceptionCode::Overflow, self.pc, in_delay_slot);
                Ok(())
            }
        }
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
        bus.add_cycles(1);
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
        bus.add_cycles(1);
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
        // Mirrors Redux's `psxRFE` (psxinterpreter.cc:779): restore
        // the previous KU/IE pair by shifting SR[5:0] right by two.
        //
        // Exception-depth bookkeeping: decrement the handler-depth
        // counter. When it reaches zero we've exited the outermost
        // handler and can clear the `clean_irq_entry` latch — the
        // parity harness's aggregation then stops silently
        // absorbing steps and starts recording user-code
        // instructions again.
        let sr = self.cop0[12];
        let restored = (sr & !0b1111) | ((sr >> 2) & 0b1111);
        self.cop0[12] = restored;
        self.isr_depth = self.isr_depth.saturating_sub(1);
        if self.isr_depth == 0 {
            self.clean_irq_entry = false;
        }
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
        bus.add_cycles(1);
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
        bus.add_cycles(1);
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
        bus.add_cycles(1);
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
        bus.add_cycles(1);
        if rt != 0 {
            self.pending_load = Some((rt, value));
        }
        Ok(())
    }

    fn op_sb(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        bus.add_cycles(1);
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
        bus.add_cycles(1);
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

    /// `LWL rt, offset(rs)` — Load Word Left. Together with `LWR`,
    /// loads a 32-bit word that may be unaligned. LWL reads the word
    /// containing `addr` and merges its high-order bytes into `rt`
    /// from the top down.
    ///
    /// Canonical memcpy use: `LWL rt, 3(rs)` + `LWR rt, 0(rs)`
    /// loads 4 bytes from `rs..rs+4` regardless of alignment.
    ///
    /// Also: LWL/LWR preserve bytes in the destination they don't
    /// overwrite, so the delay-slot value of `rt` matters — the
    /// previous instruction's committing load gets **merged**, not
    /// squashed (the opposite of non-load writes). Redux's model
    /// peeks at the staged `rt` via the pending-load slot.
    fn op_lwl(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let aligned = addr & !3;
        let word = bus.read32(aligned);
        bus.add_cycles(1);
        // If there's a pending-commit load for the same register, see
        // its staged value instead of the current register file —
        // that's what matches hardware's LWL-LWR-merge convention.
        let current = self.staged_gpr(rt);
        let shift = (addr & 3) * 8;
        let merged = (current & !(0xFFFF_FFFFu32 << shift)) | (word << shift);
        if rt != 0 {
            self.pending_load = Some((rt, merged));
        }
        Ok(())
    }

    /// `LWR rt, offset(rs)` — Load Word Right. Mirror of LWL; merges
    /// the low-order bytes of the word containing `addr` into `rt`.
    fn op_lwr(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let aligned = addr & !3;
        let word = bus.read32(aligned);
        bus.add_cycles(1);
        let current = self.staged_gpr(rt);
        let shift = (addr & 3) * 8;
        let mask = 0xFFFF_FFFFu32 >> (24 - shift);
        let merged = (current & !(0xFFFF_FFFFu32 >> shift)) | (word >> shift);
        let _ = mask; // the mask form above is equivalent to the >> chain used
        if rt != 0 {
            self.pending_load = Some((rt, merged));
        }
        Ok(())
    }

    /// `SWL rt, offset(rs)` — Store Word Left. Mirror of LWL on the
    /// store side. Writes `rt`'s high bytes into the word containing
    /// `addr`, preserving the lower bytes of the memory word.
    fn op_swl(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        // Redux's psxSWL always calls read32 + write32 unconditionally;
        // both charge a cycle each, even under cache isolation.
        bus.add_cycles(2);
        if self.cache_isolated() {
            return Ok(());
        }
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let aligned = addr & !3;
        let mem = bus.read32(aligned);
        let reg = self.gpr(rt);
        let shift = (addr & 3) * 8;
        let merged = (mem & !(0xFFFF_FFFFu32 >> (24 - shift))) | (reg >> (24 - shift));
        bus.write32(aligned, merged);
        Ok(())
    }

    /// `SWR rt, offset(rs)` — Store Word Right. Mirror of LWR on the
    /// store side.
    fn op_swr(&mut self, instr: u32, bus: &mut Bus) -> Result<(), ExecutionError> {
        bus.add_cycles(2);
        if self.cache_isolated() {
            return Ok(());
        }
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let offset = (instr as i16) as i32 as u32;
        let addr = self.gpr(rs).wrapping_add(offset);
        let aligned = addr & !3;
        let mem = bus.read32(aligned);
        let reg = self.gpr(rt);
        let shift = (addr & 3) * 8;
        let merged = (mem & !(0xFFFF_FFFFu32 << shift)) | (reg << shift);
        bus.write32(aligned, merged);
        Ok(())
    }

    /// Return the register value that would be seen by an LWL/LWR
    /// merge: prefer the staged (`committing_load`) value if one is
    /// pending for this register, else the current register file.
    /// Matches R3000 hardware behaviour where LWL and LWR merge with
    /// a load delay they share with each other.
    fn staged_gpr(&self, index: u8) -> u32 {
        if let Some((reg, value)) = &self.committing_load {
            if *reg == index {
                return *value;
            }
        }
        self.gpr(index)
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

    /// `ADD rd, rs, rt` — signed add. Raises Overflow (code 12) on
    /// signed overflow; destination unchanged. `ADDU` is the wrap-
    /// silently variant.
    fn op_add(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let a = self.gpr(rs) as i32;
        let b = self.gpr(rt) as i32;
        match a.checked_add(b) {
            Some(sum) => {
                self.set_gpr(rd, sum as u32);
                Ok(())
            }
            None => {
                let in_delay_slot = self.pending_pc.is_some();
                self.enter_exception(ExceptionCode::Overflow, self.pc, in_delay_slot);
                Ok(())
            }
        }
    }

    /// `SUB rd, rs, rt` — signed subtract. Raises Overflow (code 12)
    /// on signed overflow; destination unchanged. `SUBU` wraps.
    fn op_sub(&mut self, instr: u32) -> Result<(), ExecutionError> {
        let rs = ((instr >> 21) & 0x1F) as u8;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let a = self.gpr(rs) as i32;
        let b = self.gpr(rt) as i32;
        match a.checked_sub(b) {
            Some(diff) => {
                self.set_gpr(rd, diff as u32);
                Ok(())
            }
            None => {
                let in_delay_slot = self.pending_pc.is_some();
                self.enter_exception(ExceptionCode::Overflow, self.pc, in_delay_slot);
                Ok(())
            }
        }
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

    /// `true` when the CPU should take an interrupt exception right
    /// now. Mirrors PCSX-Redux's `branchTest`:
    ///   `(I_STAT & I_MASK) && ((SR & 0x401) == 0x401)`
    /// — i.e. some hardware source is both pending and enabled,
    /// SR.IM[2] (hardware-IRQ mask, bit 10) is on, and SR.IEc (global
    /// interrupt enable, bit 0) is on. Software interrupts on IP[0..1]
    /// would also raise via this path on real hardware; the BIOS
    /// doesn't use them, so we don't model that.
    fn should_take_interrupt(&self, bus: &mut Bus) -> bool {
        let sr = self.cop0[12];
        bus.external_interrupt_pending() && (sr & 0x401) == 0x401
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
    /// - **CAUSE**: *overwrite* with `(ExcCode << 2) | (BD bit) | (IP[2]
    ///   for Interrupt)`. Mirrors PCSX-Redux's `m_regs.CP0.n.Cause = code`
    ///   in `R3000Acpu::exception` — Redux blows the whole register
    ///   away on every exception, including IP bits, so software-side
    ///   `mfc0 v0, $13` reads only ever see what the most recent
    ///   exception parked there. We have to mirror this exactly: if we
    ///   preserved IP[2] (the natural real-hardware behaviour) BIOS
    ///   syscall handlers would observe `CAUSE = 0x420` while Redux
    ///   sees `0x20`, breaking GPR parity.
    /// - **SR**: push the 3-level KU/IE stack — bits `SR[5:0]` become
    ///   `(SR[3:0] << 2)`, with the new current pair (bits 1..0)
    ///   entering kernel-mode / interrupts-disabled.
    /// - **Vector**: `0xBFC0_0180` when `SR.BEV` (bit 22) is set (the
    ///   post-reset default the BIOS boots in), else `0x8000_0080`.
    fn enter_exception(&mut self, code: ExceptionCode, pc: u32, in_delay_slot: bool) {
        let code_bits = (code as u32) & 0x1F;
        self.exception_counts[code_bits as usize] =
            self.exception_counts[code_bits as usize].saturating_add(1);

        let mut cause = code_bits << 2;
        if matches!(code, ExceptionCode::Interrupt) {
            cause |= 1 << 10;
        }
        // Latch `clean_irq_entry` only on the outermost entry (depth
        // 0 → 1) and only if that entry was an IRQ. Nested
        // exceptions inside an IRQ handler don't flip the latch —
        // the parity harness keeps aggregating through them until
        // the outermost RFE brings the depth back to zero.
        // Matches Redux's `m_wasInISR` which is snapshotted at
        // `startStepping()` and governs the `debug.cc:235`
        // early-return for the whole stepIn span.
        if self.isr_depth == 0 {
            self.clean_irq_entry = matches!(code, ExceptionCode::Interrupt);
        }
        self.isr_depth = self.isr_depth.saturating_add(1);
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

/// Cycle cost per instruction — matches PCSX-Redux's simple-interpreter
/// `BIAS = 2` (every instruction adds 2 to its cycle counter before
/// any opcode-specific accounting). Some opcodes on real hardware cost
/// more (MULT ≈ 7–13, DIV ≈ 36, memory stalls by region) and Redux
/// models a handful of those in its accurate mode; when our parity
/// probes reveal a divergence where the extra cycles matter, specific
/// opcodes pick up their costs here.
///
/// Keeping this equal to Redux's `BIAS` is what makes Phase 4b's
/// VBlank scheduler line up on the same instruction as Redux — and
/// thus preserves parity once it turns on.
const BIAS: u32 = 2;

fn cycle_cost(_instr: u32) -> u32 {
    BIAS
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
    /// Integer arithmetic overflow — raised by `ADD`, `ADDI`, and
    /// `SUB` when the signed result doesn't fit in 32 bits. `ADDU`,
    /// `ADDIU`, and `SUBU` are the silently-wrapping variants.
    Overflow = 12,
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
        let mut bus = Bus::new(synthetic_bios_with_first_word(0x3C08_0013)).unwrap();
        let cpu = Cpu::new();
        assert_eq!(cpu.fetch(&mut bus), 0x3C08_0013);
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
        // tick = bus.cycles() = BIAS=2 for the single retired lui.
        // (Lui is not a load/store so no +1 extra.)
        assert_eq!(record.tick, 2);
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
        // opcode 0x11 = COP1 (FPU); the PS1 has no FPU and the BIOS
        // never issues this, so we leave it unimplemented as a
        // sentinel. Encoding: 0x44000000 = (0x11 << 26) | 0.
        let mut bus = Bus::new(synthetic_bios_with_first_word(0x4400_0000)).unwrap();
        let mut cpu = Cpu::new();
        let err = cpu.step(&mut bus).unwrap_err();
        assert!(matches!(
            err,
            ExecutionError::Unimplemented { opcode: 0x11, .. }
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

    #[test]
    fn addi_traps_on_signed_overflow() {
        // $t0 = 0x7FFFFFFF (i32::MAX). ADDI $t1, $t0, 1 overflows.
        // Post-step: $t1 should be unchanged (not 0x80000000), and
        // CAUSE.ExcCode should read 12 (Overflow).
        let mut bios = vec![0u8; memory::bios::SIZE];
        // lui $t0, 0x7FFF
        bios[0..4].copy_from_slice(&0x3C08_7FFFu32.to_le_bytes());
        // ori $t0, $t0, 0xFFFF → $t0 = 0x7FFFFFFF
        bios[4..8].copy_from_slice(&0x3508_FFFFu32.to_le_bytes());
        // addi $t1, $t0, 1 → overflow
        bios[8..12].copy_from_slice(&0x2109_0001u32.to_le_bytes());

        let mut bus = Bus::new(bios).unwrap();
        let mut cpu = Cpu::new();
        cpu.step(&mut bus).expect("lui");
        cpu.step(&mut bus).expect("ori");
        let exc_count_before = cpu.exception_counts()[12];
        cpu.step(&mut bus).expect("addi does not bubble an Err");
        assert_eq!(cpu.gprs[9], 0, "t1 must remain unchanged on overflow");
        assert_eq!(
            cpu.exception_counts()[12],
            exc_count_before + 1,
            "Overflow (12) exception must have fired"
        );
        let cause = cpu.cop0[13];
        assert_eq!((cause >> 2) & 0x1F, 12, "CAUSE.ExcCode = 12 after trap");
    }

    #[test]
    fn addi_negative_overflow_traps() {
        // $t0 = 0x80000000 (i32::MIN). ADDI $t1, $t0, -1 overflows.
        let mut bios = vec![0u8; memory::bios::SIZE];
        // lui $t0, 0x8000
        bios[0..4].copy_from_slice(&0x3C08_8000u32.to_le_bytes());
        // addi $t1, $t0, -1 (imm = 0xFFFF)
        bios[4..8].copy_from_slice(&0x2109_FFFFu32.to_le_bytes());

        let mut bus = Bus::new(bios).unwrap();
        let mut cpu = Cpu::new();
        cpu.step(&mut bus).expect("lui");
        cpu.step(&mut bus).expect("addi");
        assert_eq!(cpu.gprs[9], 0, "t1 unchanged on negative overflow");
        assert_eq!(cpu.exception_counts()[12], 1);
    }

    #[test]
    fn addi_no_overflow_writes_destination() {
        // Edge: exactly at the boundary (i32::MAX - 1) + 1 = i32::MAX.
        // No overflow; $t1 should receive the result.
        let mut bios = vec![0u8; memory::bios::SIZE];
        // lui $t0, 0x7FFF
        bios[0..4].copy_from_slice(&0x3C08_7FFFu32.to_le_bytes());
        // ori $t0, $t0, 0xFFFE → 0x7FFFFFFE
        bios[4..8].copy_from_slice(&0x3508_FFFEu32.to_le_bytes());
        // addi $t1, $t0, 1 → 0x7FFFFFFF (no overflow)
        bios[8..12].copy_from_slice(&0x2109_0001u32.to_le_bytes());

        let mut bus = Bus::new(bios).unwrap();
        let mut cpu = Cpu::new();
        cpu.step(&mut bus).expect("lui");
        cpu.step(&mut bus).expect("ori");
        cpu.step(&mut bus).expect("addi");
        assert_eq!(cpu.gprs[9], 0x7FFF_FFFF);
        assert_eq!(cpu.exception_counts()[12], 0);
    }

    #[test]
    fn add_traps_on_signed_overflow() {
        // $t0 = 0x7FFFFFFF, $t1 = 1. ADD $t2, $t0, $t1 overflows.
        let mut bios = vec![0u8; memory::bios::SIZE];
        // lui $t0, 0x7FFF
        bios[0..4].copy_from_slice(&0x3C08_7FFFu32.to_le_bytes());
        // ori $t0, $t0, 0xFFFF
        bios[4..8].copy_from_slice(&0x3508_FFFFu32.to_le_bytes());
        // ori $t1, $zero, 1
        bios[8..12].copy_from_slice(&0x3409_0001u32.to_le_bytes());
        // add $t2, $t0, $t1 — special=0, rs=8, rt=9, rd=10, funct=0x20
        let add = (8u32 << 21) | (9u32 << 16) | (10u32 << 11) | 0x20u32;
        bios[12..16].copy_from_slice(&add.to_le_bytes());

        let mut bus = Bus::new(bios).unwrap();
        let mut cpu = Cpu::new();
        cpu.step(&mut bus).expect("lui");
        cpu.step(&mut bus).expect("ori t0");
        cpu.step(&mut bus).expect("ori t1");
        cpu.step(&mut bus).expect("add");
        assert_eq!(cpu.gprs[10], 0, "t2 unchanged on overflow");
        assert_eq!(cpu.exception_counts()[12], 1);
    }

    #[test]
    fn sub_traps_on_signed_overflow() {
        // $t0 = 0x80000000 (i32::MIN), $t1 = 1. SUB $t2, $t0, $t1 =
        // i32::MIN - 1, which overflows.
        let mut bios = vec![0u8; memory::bios::SIZE];
        // lui $t0, 0x8000
        bios[0..4].copy_from_slice(&0x3C08_8000u32.to_le_bytes());
        // ori $t1, $zero, 1
        bios[4..8].copy_from_slice(&0x3409_0001u32.to_le_bytes());
        // sub $t2, $t0, $t1 — funct=0x22
        let sub = (8u32 << 21) | (9u32 << 16) | (10u32 << 11) | 0x22u32;
        bios[8..12].copy_from_slice(&sub.to_le_bytes());

        let mut bus = Bus::new(bios).unwrap();
        let mut cpu = Cpu::new();
        cpu.step(&mut bus).expect("lui");
        cpu.step(&mut bus).expect("ori t1");
        cpu.step(&mut bus).expect("sub");
        assert_eq!(cpu.gprs[10], 0, "t2 unchanged on overflow");
        assert_eq!(cpu.exception_counts()[12], 1);
    }

    #[test]
    fn sub_no_overflow_writes_destination() {
        // 10 - 3 = 7 — ordinary subtract, no trap.
        let mut bios = vec![0u8; memory::bios::SIZE];
        // ori $t0, $zero, 10
        bios[0..4].copy_from_slice(&0x3408_000Au32.to_le_bytes());
        // ori $t1, $zero, 3
        bios[4..8].copy_from_slice(&0x3409_0003u32.to_le_bytes());
        // sub $t2, $t0, $t1
        let sub = (8u32 << 21) | (9u32 << 16) | (10u32 << 11) | 0x22u32;
        bios[8..12].copy_from_slice(&sub.to_le_bytes());

        let mut bus = Bus::new(bios).unwrap();
        let mut cpu = Cpu::new();
        cpu.step(&mut bus).expect("ori t0");
        cpu.step(&mut bus).expect("ori t1");
        cpu.step(&mut bus).expect("sub");
        assert_eq!(cpu.gprs[10], 7);
        assert_eq!(cpu.exception_counts()[12], 0);
    }
}
