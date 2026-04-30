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

use emulator_core::{Bus, Cpu};
use psx_iso::Exe;

fn main() {
    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("bios");
    let exe_bytes = std::fs::read(
        "/Users/ebonura/Desktop/repos/psoxide/build/examples/mipsel-sony-psx/release/hello-gte.exe",
    )
    .expect("hello-gte");
    let exe = Exe::parse(&exe_bytes).expect("parse");
    let mut bus = Bus::new(bios).expect("bus");
    bus.load_exe_payload(exe.load_addr, &exe.payload);
    bus.enable_hle_bios();
    bus.attach_digital_pad_port1();
    let mut cpu = Cpu::new();
    cpu.seed_from_exe(exe.initial_pc, exe.initial_gp, exe.initial_sp());

    // Step manually; before each instruction, peek the instruction
    // word. If it's COP2 (primary opcode 0x12 = top 6 bits = 0b010010),
    // decode + log before stepping so we see inputs.
    let max_steps = 5_000_000u64;
    let mut cop2_count = 0u64;
    let mut stop_at_nth = 200u64; // Log this many then stop; enough for first render loop.
    let mut cycles_at_last_pump = 0u64;

    for _ in 0..max_steps {
        let pc = cpu.pc();
        let instr = bus.peek_instruction(pc).unwrap_or(0);
        let primary = (instr >> 26) & 0x3F;
        if primary == 0x12 {
            cop2_count += 1;
            if cop2_count <= stop_at_nth {
                log_cop2(pc, instr, &cpu);
            }
            if cop2_count == stop_at_nth {
                eprintln!("--- reached {stop_at_nth} COP2 ops, stopping trace ---");
                stop_at_nth = u64::MAX; // silence further logs
            }
        }
        if cpu.step(&mut bus).is_err() {
            break;
        }
        if bus.cycles() - cycles_at_last_pump > 560_000 {
            cycles_at_last_pump = bus.cycles();
            bus.run_spu_samples(735);
            let _ = bus.spu.drain_audio();
        }
        if bus.irq().raise_counts()[0] >= 2 {
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
