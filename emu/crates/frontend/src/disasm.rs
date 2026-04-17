//! MIPS R3000 disassembler.
//!
//! Takes a `(pc, instr)` pair and returns a printable mnemonic string.
//! Coverage is the opcodes we actually care about in the debug UI —
//! everything the BIOS emits in the first few million instructions,
//! plus a handful of common ones we'll hit soon. Unknowns render as
//! `<??? op=0xNN>` so they stand out in the exec history.
//!
//! This is a UI convenience; it's intentionally separate from the
//! CPU's dispatch so changes here can't regress execution semantics.

/// Canonical MIPS GPR names, indexed 0..=31.
const GPR: [&str; 32] = [
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
    "t7", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp",
    "ra",
];

/// Disassemble one instruction word.
pub fn disasm(pc: u32, instr: u32) -> String {
    if instr == 0 {
        return "nop".to_string();
    }

    let op = (instr >> 26) & 0x3F;
    let rs = ((instr >> 21) & 0x1F) as usize;
    let rt = ((instr >> 16) & 0x1F) as usize;
    let rd = ((instr >> 11) & 0x1F) as usize;
    let sa = (instr >> 6) & 0x1F;
    let funct = instr & 0x3F;
    let imm = instr & 0xFFFF;
    let simm = (instr as i16) as i32;

    match op {
        0x00 => special(rs, rt, rd, sa, funct),
        0x01 => regimm(rs, rt, pc, simm),
        0x02 => format!("j {}", jump_target(pc, instr)),
        0x03 => format!("jal {}", jump_target(pc, instr)),
        0x04 => format!("beq {}, {}, {}", GPR[rs], GPR[rt], branch_target(pc, simm)),
        0x05 => format!("bne {}, {}, {}", GPR[rs], GPR[rt], branch_target(pc, simm)),
        0x06 => format!("blez {}, {}", GPR[rs], branch_target(pc, simm)),
        0x07 => format!("bgtz {}, {}", GPR[rs], branch_target(pc, simm)),
        0x08 => format!("addi {}, {}, {}", GPR[rt], GPR[rs], simm),
        0x09 => format!("addiu {}, {}, {}", GPR[rt], GPR[rs], simm),
        0x0A => format!("slti {}, {}, {}", GPR[rt], GPR[rs], simm),
        0x0B => format!("sltiu {}, {}, {}", GPR[rt], GPR[rs], simm),
        0x0C => format!("andi {}, {}, 0x{:04X}", GPR[rt], GPR[rs], imm),
        0x0D => format!("ori {}, {}, 0x{:04X}", GPR[rt], GPR[rs], imm),
        0x0E => format!("xori {}, {}, 0x{:04X}", GPR[rt], GPR[rs], imm),
        0x0F => format!("lui {}, 0x{:04X}", GPR[rt], imm),
        0x10 => cop0(rs, rt, rd),
        0x12 => format!("cop2 0x{instr:08X}"), // GTE — detail later
        0x20 => format!("lb {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x21 => format!("lh {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x22 => format!("lwl {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x23 => format!("lw {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x24 => format!("lbu {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x25 => format!("lhu {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x26 => format!("lwr {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x28 => format!("sb {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x29 => format!("sh {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x2A => format!("swl {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x2B => format!("sw {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x2E => format!("swr {}, {}({})", GPR[rt], simm, GPR[rs]),
        0x32 => format!("lwc2 {}, {}({})", rt, simm, GPR[rs]),
        0x3A => format!("swc2 {}, {}({})", rt, simm, GPR[rs]),
        _ => format!("<??? op=0x{op:02X} word=0x{instr:08X}>"),
    }
}

fn special(rs: usize, rt: usize, rd: usize, sa: u32, funct: u32) -> String {
    match funct {
        0x00 => {
            if rd == 0 && rt == 0 && sa == 0 {
                "nop".to_string()
            } else {
                format!("sll {}, {}, {}", GPR[rd], GPR[rt], sa)
            }
        }
        0x02 => format!("srl {}, {}, {}", GPR[rd], GPR[rt], sa),
        0x03 => format!("sra {}, {}, {}", GPR[rd], GPR[rt], sa),
        0x04 => format!("sllv {}, {}, {}", GPR[rd], GPR[rt], GPR[rs]),
        0x06 => format!("srlv {}, {}, {}", GPR[rd], GPR[rt], GPR[rs]),
        0x07 => format!("srav {}, {}, {}", GPR[rd], GPR[rt], GPR[rs]),
        0x08 => format!("jr {}", GPR[rs]),
        0x09 => {
            if rd == 31 {
                format!("jalr {}", GPR[rs])
            } else {
                format!("jalr {}, {}", GPR[rd], GPR[rs])
            }
        }
        0x0C => "syscall".to_string(),
        0x0D => "break".to_string(),
        0x10 => format!("mfhi {}", GPR[rd]),
        0x11 => format!("mthi {}", GPR[rs]),
        0x12 => format!("mflo {}", GPR[rd]),
        0x13 => format!("mtlo {}", GPR[rs]),
        0x18 => format!("mult {}, {}", GPR[rs], GPR[rt]),
        0x19 => format!("multu {}, {}", GPR[rs], GPR[rt]),
        0x1A => format!("div {}, {}", GPR[rs], GPR[rt]),
        0x1B => format!("divu {}, {}", GPR[rs], GPR[rt]),
        0x20 => format!("add {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x21 => format!("addu {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x22 => format!("sub {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x23 => format!("subu {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x24 => format!("and {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x25 => format!("or {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x26 => format!("xor {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x27 => format!("nor {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x2A => format!("slt {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        0x2B => format!("sltu {}, {}, {}", GPR[rd], GPR[rs], GPR[rt]),
        _ => format!("<??? special funct=0x{funct:02X}>"),
    }
}

fn regimm(rs: usize, rt: usize, pc: u32, simm: i32) -> String {
    match rt {
        0x00 => format!("bltz {}, {}", GPR[rs], branch_target(pc, simm)),
        0x01 => format!("bgez {}, {}", GPR[rs], branch_target(pc, simm)),
        0x10 => format!("bltzal {}, {}", GPR[rs], branch_target(pc, simm)),
        0x11 => format!("bgezal {}, {}", GPR[rs], branch_target(pc, simm)),
        _ => format!("<??? regimm rt=0x{rt:02X}>"),
    }
}

fn cop0(rs: usize, rt: usize, rd: usize) -> String {
    match rs {
        0x00 => format!("mfc0 {}, cop0[{rd}]", GPR[rt]),
        0x04 => format!("mtc0 {}, cop0[{rd}]", GPR[rt]),
        0x10 => "rfe".to_string(),
        _ => format!("<??? cop0 rs=0x{rs:02X}>"),
    }
}

/// Target for conditional branches: pc + 4 + (sign_extend(imm16) << 2).
fn branch_target(pc: u32, simm: i32) -> String {
    let target = pc.wrapping_add(4).wrapping_add((simm << 2) as u32);
    format!("0x{target:08X}")
}

/// Target for J / JAL: (pc + 4)[31..=28] | (imm26 << 2).
fn jump_target(pc: u32, instr: u32) -> String {
    let target_field = instr & 0x03FF_FFFF;
    let delay_slot_pc = pc.wrapping_add(4);
    let target = (delay_slot_pc & 0xF000_0000) | (target_field << 2);
    format!("0x{target:08X}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_word_is_nop() {
        assert_eq!(disasm(0, 0), "nop");
    }

    #[test]
    fn lui_formats_immediate_as_hex() {
        // lui $t0, 0x0013 — the first real BIOS instruction
        assert_eq!(disasm(0xBFC0_0000, 0x3C08_0013), "lui t0, 0x0013");
    }

    #[test]
    fn ori_uses_gpr_names() {
        // ori $t0, $t1, 0xABCD
        let instr = (0x0D << 26) | (9 << 21) | (8 << 16) | 0xABCD;
        assert_eq!(disasm(0, instr), "ori t0, t1, 0xABCD");
    }

    #[test]
    fn sw_formats_offset_base_syntax() {
        // sw $t0, -4($sp) — very common BIOS pattern
        let instr = (0x2B << 26) | (29 << 21) | (8 << 16) | ((-4i32 as u32) & 0xFFFF);
        assert_eq!(disasm(0, instr), "sw t0, -4(sp)");
    }

    #[test]
    fn jr_ra_is_common_return() {
        let instr = (31 << 21) | 0x08;
        assert_eq!(disasm(0, instr), "jr ra");
    }

    #[test]
    fn syscall_decodes() {
        assert_eq!(disasm(0, 0x0000_000C), "syscall");
    }

    #[test]
    fn branch_target_uses_delay_slot() {
        // bne $t0, $zero, +8 (target = pc + 4 + 8 = pc + 12)
        let instr: u32 = (0x05 << 26) | (8 << 21) | 0x0002;
        assert_eq!(disasm(0xBFC0_0100, instr), "bne t0, zero, 0xBFC0010C");
    }

    #[test]
    fn jump_target_uses_upper_pc_bits() {
        // j 0xBFC00100 (from pc in BFC region)
        let target_field = (0xBFC0_0100 >> 2) & 0x03FF_FFFF;
        let instr = (0x02 << 26) | target_field;
        assert_eq!(disasm(0xBFC0_0000, instr), "j 0xBFC00100");
    }

    #[test]
    fn unknown_primary_opcode_is_flagged() {
        // opcode 0x14 is undefined on R3000
        let instr = 0x14 << 26;
        assert!(disasm(0, instr).starts_with("<??? op=0x14"));
    }
}
