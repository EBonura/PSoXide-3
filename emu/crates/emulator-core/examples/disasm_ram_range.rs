//! Run the emulator long enough to load all BIOS kernel code into
//! RAM, then print instructions in a given PC range so we can
//! cross-reference an ISR's disassembly.

use emulator_core::{Bus, Cpu};

fn main() {
    let lo: u32 = std::env::args()
        .nth(1)
        .map(|s| parse_hex(&s))
        .unwrap_or(0x80059700);
    let hi: u32 = std::env::args()
        .nth(2)
        .map(|s| parse_hex(&s))
        .unwrap_or(0x80059800);
    let boot_steps: u64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000_000); // plenty to fold BIOS into RAM

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let mut bus = Bus::new(bios).expect("bus");
    let mut cpu = Cpu::new();

    for _ in 0..boot_steps {
        if cpu.step(&mut bus).is_err() {
            break;
        }
    }

    println!("=== 0x{lo:08x}..=0x{hi:08x} ===");
    let mut pc = lo & !3;
    while pc <= hi {
        let instr = bus.peek_instruction(pc).unwrap_or(0xDEAD_BEEF);
        println!("0x{pc:08x}: 0x{instr:08x}  {}", disasm(instr));
        pc = pc.wrapping_add(4);
    }
}

fn parse_hex(s: &str) -> u32 {
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(h, 16).unwrap_or(0)
    } else {
        s.parse().unwrap_or(0)
    }
}

fn disasm(instr: u32) -> String {
    // A bare-bones disassembler covering the common BIOS ops.
    let op = (instr >> 26) & 0x3F;
    let rs = (instr >> 21) & 0x1F;
    let rt = (instr >> 16) & 0x1F;
    let rd = (instr >> 11) & 0x1F;
    let shamt = (instr >> 6) & 0x1F;
    let funct = instr & 0x3F;
    let imm = (instr & 0xFFFF) as i16 as i32;
    let uimm = instr & 0xFFFF;
    let target = instr & 0x03FF_FFFF;
    match op {
        0x00 => match funct {
            0x00 if instr == 0 => "nop".into(),
            0x00 => format!("sll ${rd}, ${rt}, {shamt}"),
            0x02 => format!("srl ${rd}, ${rt}, {shamt}"),
            0x03 => format!("sra ${rd}, ${rt}, {shamt}"),
            0x08 => format!("jr ${rs}"),
            0x09 => format!("jalr ${rd}, ${rs}"),
            0x0C => "syscall".into(),
            0x10 => format!("mfhi ${rd}"),
            0x12 => format!("mflo ${rd}"),
            0x20 => format!("add ${rd}, ${rs}, ${rt}"),
            0x21 => format!("addu ${rd}, ${rs}, ${rt}"),
            0x22 => format!("sub ${rd}, ${rs}, ${rt}"),
            0x23 => format!("subu ${rd}, ${rs}, ${rt}"),
            0x24 => format!("and ${rd}, ${rs}, ${rt}"),
            0x25 => format!("or ${rd}, ${rs}, ${rt}"),
            0x26 => format!("xor ${rd}, ${rs}, ${rt}"),
            0x27 => format!("nor ${rd}, ${rs}, ${rt}"),
            0x2A => format!("slt ${rd}, ${rs}, ${rt}"),
            0x2B => format!("sltu ${rd}, ${rs}, ${rt}"),
            _ => format!("r-type funct=0x{funct:02x}"),
        },
        0x02 => format!("j 0x{:08x}", (target << 2)),
        0x03 => format!("jal 0x{:08x}", (target << 2)),
        0x04 => format!("beq ${rs}, ${rt}, {imm:+}"),
        0x05 => format!("bne ${rs}, ${rt}, {imm:+}"),
        0x06 => format!("blez ${rs}, {imm:+}"),
        0x07 => format!("bgtz ${rs}, {imm:+}"),
        0x08 => format!("addi ${rt}, ${rs}, {imm}"),
        0x09 => format!("addiu ${rt}, ${rs}, {imm}"),
        0x0A => format!("slti ${rt}, ${rs}, {imm}"),
        0x0C => format!("andi ${rt}, ${rs}, 0x{uimm:04x}"),
        0x0D => format!("ori ${rt}, ${rs}, 0x{uimm:04x}"),
        0x0F => format!("lui ${rt}, 0x{uimm:04x}"),
        0x10 => format!("cop0 0x{instr:08x}"),
        0x20 => format!("lb ${rt}, {imm}(${rs})"),
        0x21 => format!("lh ${rt}, {imm}(${rs})"),
        0x23 => format!("lw ${rt}, {imm}(${rs})"),
        0x24 => format!("lbu ${rt}, {imm}(${rs})"),
        0x25 => format!("lhu ${rt}, {imm}(${rs})"),
        0x28 => format!("sb ${rt}, {imm}(${rs})"),
        0x29 => format!("sh ${rt}, {imm}(${rs})"),
        0x2B => format!("sw ${rt}, {imm}(${rs})"),
        _ => format!("op=0x{op:02x}"),
    }
}
