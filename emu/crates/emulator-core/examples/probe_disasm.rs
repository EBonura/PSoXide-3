//! Run to step N with disc, then dump RAM words around a given
//! PC. Used to read the instruction stream at a hang site.
//!
//! ```bash
//! cargo run --release -p emulator-core --example probe_disasm -- 500000000 0x8008e528 20 "/path/to/game.bin"
//! ```

use emulator_core::{Bus, Cpu};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: u64 = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000_000);
    let pc_start: u32 = args
        .get(1)
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .expect("need start PC hex");
    let count: u32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let disc_path = args.get(3).cloned();

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    if let Some(ref p) = disc_path {
        let disc_bytes = std::fs::read(p).expect("disc");
        bus.cdrom
            .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
    }
    let mut cpu = Cpu::new();
    for _ in 0..n {
        let was_in_isr = cpu.in_isr();
        cpu.step(&mut bus).expect("step");
        if !was_in_isr && cpu.in_irq_handler() {
            while cpu.in_irq_handler() {
                cpu.step(&mut bus).expect("isr step");
            }
        }
    }

    println!("=== RAM disasm @ 0x{pc_start:08x} ({count} words), after step {n} ===");
    for i in 0..count {
        let addr = pc_start + i * 4;
        let word = bus.peek_instruction(addr).unwrap_or(0xFFFFFFFF);
        println!("  0x{addr:08x}:  0x{word:08x}  {}", decode_mips(word));
    }
}

/// Minimal MIPS-I disassembler for common opcodes. Enough to
/// recognize branches, loads, stores, syscalls, and spin-wait
/// patterns. Uses numeric regs ($rN) rather than ABI names.
fn decode_mips(instr: u32) -> String {
    let op = (instr >> 26) & 0x3F;
    let rs = (instr >> 21) & 0x1F;
    let rt = (instr >> 16) & 0x1F;
    let rd = (instr >> 11) & 0x1F;
    let shamt = (instr >> 6) & 0x1F;
    let func = instr & 0x3F;
    let imm = (instr & 0xFFFF) as i16 as i32;
    let u_imm = instr & 0xFFFF;
    let target = instr & 0x03FF_FFFF;
    match op {
        0x00 => match func {
            0x00 => if instr == 0 { "nop".into() } else { format!("sll $r{rd},$r{rt},{shamt}") },
            0x02 => format!("srl $r{rd},$r{rt},{shamt}"),
            0x03 => format!("sra $r{rd},$r{rt},{shamt}"),
            0x08 => format!("jr $r{rs}"),
            0x09 => format!("jalr $r{rs}"),
            0x0C => format!("syscall 0x{:x}", (instr >> 6) & 0xFFFFF),
            0x0D => format!("break"),
            0x21 => format!("addu $r{rd},$r{rs},$r{rt}"),
            0x23 => format!("subu $r{rd},$r{rs},$r{rt}"),
            0x24 => format!("and $r{rd},$r{rs},$r{rt}"),
            0x25 => format!("or $r{rd},$r{rs},$r{rt}"),
            _ => format!("spec func=0x{func:02x}"),
        },
        0x02 => format!("j 0x{:08x}", (target << 2) | (instr & 0xF000_0000)),
        0x03 => format!("jal 0x{:08x}", (target << 2) | (instr & 0xF000_0000)),
        0x04 => format!("beq $r{rs},$r{rt},{imm:+}"),
        0x05 => format!("bne $r{rs},$r{rt},{imm:+}"),
        0x06 => format!("blez $r{rs},{imm:+}"),
        0x07 => format!("bgtz $r{rs},{imm:+}"),
        0x08 => format!("addi $r{rt},$r{rs},{imm}"),
        0x09 => format!("addiu $r{rt},$r{rs},{imm}"),
        0x0C => format!("andi $r{rt},$r{rs},0x{u_imm:x}"),
        0x0D => format!("ori $r{rt},$r{rs},0x{u_imm:x}"),
        0x0F => format!("lui $r{rt},0x{u_imm:x}"),
        0x10 => format!("cop0 0x{:07x}", instr & 0x3FF_FFFF),
        0x20 => format!("lb $r{rt},{imm}($r{rs})"),
        0x21 => format!("lh $r{rt},{imm}($r{rs})"),
        0x23 => format!("lw $r{rt},{imm}($r{rs})"),
        0x24 => format!("lbu $r{rt},{imm}($r{rs})"),
        0x25 => format!("lhu $r{rt},{imm}($r{rs})"),
        0x28 => format!("sb $r{rt},{imm}($r{rs})"),
        0x29 => format!("sh $r{rt},{imm}($r{rs})"),
        0x2B => format!("sw $r{rt},{imm}($r{rs})"),
        _ => format!("op=0x{op:02x}"),
    }
}
