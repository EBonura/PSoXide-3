//! Log every COP2 (GTE) instruction hello-gte retires during its
//! first-frame setup + render. The hypothesis we're testing: is the
//! `ctc2!` / `mtc2!` macro in `psx-gte::regs` actually placing the
//! value in `$t0` ($8) before the instruction executes? If the asm
//! block's `in("$8")` doesn't route through to the GTE (for whatever
//! reason -- LLVM optimisation, macro bug, register-allocation
//! quirk), the emulator will see CTC2 $8, rd where $8 still has
//! whatever was there from the previous instruction.
//!
//! For each COP2 write we log:
//! - PC
//! - raw instruction
//! - decoded op (MFC2 / CFC2 / MTC2 / CTC2 / function-id)
//! - source register ($t), its value, destination register index
//!
//! A quick eyeball check: CTC2 writes to control reg 0 (RT00/RT01)
//! should carry an `(i16, i16)` pair where the low half is
//! `0x1000` (= 1.0 in 1.3.12) for most frames' rotation matrix
//! top-left element.

use emulator_core::Cpu;

#[path = "support/frame_probe.rs"]
mod frame_probe;

fn main() {
    let mut probe = frame_probe::SideLoadedExe::example("hello-gte", true);

    // Step manually; before each instruction, peek the instruction
    // word. If it's COP2 (primary opcode 0x12 = top 6 bits = 0b010010),
    // decode + log before stepping so we see inputs.
    let max_steps = 5_000_000u64;
    let mut cop2_count = 0u64;
    let mut stop_at_nth = 200u64; // Log this many then stop; enough for first render loop.
    let mut cycles_at_last_pump = 0u64;

    for _ in 0..max_steps {
        let pc = probe.cpu.pc();
        let instr = probe.bus.peek_instruction(pc).unwrap_or(0);
        let primary = (instr >> 26) & 0x3F;
        if primary == 0x12 {
            cop2_count += 1;
            if cop2_count <= stop_at_nth {
                log_cop2(pc, instr, &probe.cpu);
            }
            if cop2_count == stop_at_nth {
                eprintln!("--- reached {stop_at_nth} COP2 ops, stopping trace ---");
                stop_at_nth = u64::MAX; // silence further logs
            }
        }
        if !frame_probe::step_cpu_and_pump_spu(
            &mut probe.cpu,
            &mut probe.bus,
            &mut cycles_at_last_pump,
        ) {
            break;
        }
        if probe.bus.irq().raise_counts()[0] >= 2 {
            break;
        }
    }

    eprintln!();
    eprintln!("=== total COP2 ops seen in first 2 frames: {cop2_count} ===");
}

fn log_cop2(pc: u32, instr: u32, cpu: &Cpu) {
    let func_bit = instr & (1 << 25);
    if func_bit != 0 {
        // COP2 function-op (RTPS, RTPT, NCLIP, …). Bits 5..0 = op id.
        let op = instr & 0x3F;
        let sf = (instr >> 19) & 1;
        eprintln!("pc=0x{pc:08x} instr=0x{instr:08x}  COP2 COFUN op=0x{op:02x} sf={sf}",);
    } else {
        let cop_op = (instr >> 21) & 0x1F;
        let rt = ((instr >> 16) & 0x1F) as u8;
        let rd = ((instr >> 11) & 0x1F) as u8;
        let rt_val = cpu.gpr(rt);
        let op_name = match cop_op {
            0 => "MFC2",
            2 => "CFC2",
            4 => "MTC2",
            6 => "CTC2",
            _ => "???",
        };
        eprintln!(
            "pc=0x{pc:08x} instr=0x{instr:08x}  {op_name:4} $r{rt:<2} rd={rd:<2}  \
             src=${rt}=0x{rt_val:08x}",
        );
    }
}
