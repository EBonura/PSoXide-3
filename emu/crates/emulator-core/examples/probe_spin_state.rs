//! Run a disc to a target user-step count and dump the local CPU
//! state around the final PC. Useful when a commercial title parks in
//! a tight polling loop and we need the register values, not just the
//! PC.

use emulator_core::{Bus, Cpu};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: u64 = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(260_000_000);
    let disc_path = args
        .get(1)
        .expect("usage: probe_spin_state <steps> <disc.bin>");

    let bios = std::fs::read("/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN").expect("BIOS");
    let disc_bytes = std::fs::read(disc_path).expect("disc readable");
    let mut bus = Bus::new(bios).expect("bus");
    bus.cdrom
        .insert_disc(Some(psx_iso::Disc::from_bin(disc_bytes)));
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

    let pc = cpu.pc();
    println!("steps={n} cycles={} pc=0x{pc:08x}", bus.cycles());
    println!(
        "sr=0x{:08x} cause=0x{:08x} epc=0x{:08x} istat=0x{:03x} imask=0x{:03x}",
        cpu.cop0()[12],
        cpu.cop0()[13],
        cpu.cop0()[14],
        bus.irq().stat(),
        bus.irq().mask(),
    );
    dump_gprs(&cpu);

    let v0 = cpu.gpr(2);
    let a1 = cpu.gpr(5);
    let watch = v0.wrapping_add(4);
    let watch_value = bus
        .peek_instruction(watch)
        .unwrap_or_else(|| bus.read32(watch));
    println!();
    println!("spin-watch guess: $v0+4=0x{watch:08x} value=0x{watch_value:08x} $a1=0x{a1:08x}",);

    let lo = pc.wrapping_sub(0x40) & !3;
    let hi = pc.wrapping_add(0x40) & !3;
    println!();
    println!("=== disasm 0x{lo:08x}..=0x{hi:08x} ===");
    dump_disasm(&bus, lo, hi, pc);

    let ra = cpu.gpr(31);
    let ra_lo = ra.wrapping_sub(0x40) & !3;
    let ra_hi = ra.wrapping_add(0x20) & !3;
    println!();
    println!("=== caller disasm around $ra=0x{ra:08x} ===");
    dump_disasm(&bus, ra_lo, ra_hi, ra);

    if std::env::var_os("PSOXIDE_ADVANCE_ONE").is_some() {
        println!();
        println!("=== advancing one raw CPU step ===");
        let before_pc = cpu.pc();
        let before_instr = bus.peek_instruction(before_pc).unwrap_or(0xDEAD_BEEF);
        let before_cycles = bus.cycles();
        let before_irq_stat = bus.irq().stat();
        let before_irq_handler = cpu.in_irq_handler();
        let rec = cpu.step(&mut bus).expect("advance one step");
        println!(
            "before pc=0x{before_pc:08x} instr=0x{before_instr:08x} cycles={before_cycles} irq_stat=0x{before_irq_stat:03x} in_irq={}",
            before_irq_handler as u8
        );
        println!(
            "record pc=0x{:08x} instr=0x{:08x} tick={} clean_irq_after={} cpu.pc=0x{:08x} irq_stat=0x{:03x}",
            rec.pc,
            rec.instr,
            rec.tick,
            cpu.in_irq_handler() as u8,
            cpu.pc(),
            bus.irq().stat(),
        );
        println!(
            "exception counts: Int={} Syscall={} Break={}",
            cpu.exception_counts()[0],
            cpu.exception_counts()[8],
            cpu.exception_counts()[9],
        );
    }
}

fn dump_gprs(cpu: &Cpu) {
    let names = [
        "$0 ", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3", "$t0", "$t1", "$t2", "$t3", "$t4",
        "$t5", "$t6", "$t7", "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7", "$t8", "$t9",
        "$k0", "$k1", "$gp", "$sp", "$fp", "$ra",
    ];
    for row in 0..4 {
        let mut line = String::new();
        for col in 0..8 {
            let i = row * 8 + col;
            line.push_str(&format!(" {}={:08x}", names[i], cpu.gprs()[i]));
        }
        println!("{line}");
    }
}

fn disasm(instr: u32, pc: u32) -> String {
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
        0x02 => format!("j 0x{:08x}", (pc & 0xF000_0000) | (target << 2)),
        0x03 => format!("jal 0x{:08x}", (pc & 0xF000_0000) | (target << 2)),
        0x04 => format!("beq ${rs}, ${rt}, 0x{:08x}", branch_target(pc, imm)),
        0x05 => format!("bne ${rs}, ${rt}, 0x{:08x}", branch_target(pc, imm)),
        0x06 => format!("blez ${rs}, 0x{:08x}", branch_target(pc, imm)),
        0x07 => format!("bgtz ${rs}, 0x{:08x}", branch_target(pc, imm)),
        0x08 => format!("addi ${rt}, ${rs}, {imm}"),
        0x09 => format!("addiu ${rt}, ${rs}, {imm}"),
        0x0A => format!("slti ${rt}, ${rs}, {imm}"),
        0x0B => format!("sltiu ${rt}, ${rs}, {imm}"),
        0x0C => format!("andi ${rt}, ${rs}, 0x{uimm:04x}"),
        0x0D => format!("ori ${rt}, ${rs}, 0x{uimm:04x}"),
        0x0E => format!("xori ${rt}, ${rs}, 0x{uimm:04x}"),
        0x0F => format!("lui ${rt}, 0x{uimm:04x}"),
        0x10 => format!("cop0 0x{instr:08x}"),
        0x12 => format!("cop2 0x{instr:08x}"),
        0x20 => format!("lb ${rt}, {imm}(${rs})"),
        0x21 => format!("lh ${rt}, {imm}(${rs})"),
        0x22 => format!("lwl ${rt}, {imm}(${rs})"),
        0x23 => format!("lw ${rt}, {imm}(${rs})"),
        0x24 => format!("lbu ${rt}, {imm}(${rs})"),
        0x25 => format!("lhu ${rt}, {imm}(${rs})"),
        0x26 => format!("lwr ${rt}, {imm}(${rs})"),
        0x28 => format!("sb ${rt}, {imm}(${rs})"),
        0x29 => format!("sh ${rt}, {imm}(${rs})"),
        0x2A => format!("swl ${rt}, {imm}(${rs})"),
        0x2B => format!("sw ${rt}, {imm}(${rs})"),
        0x2E => format!("swr ${rt}, {imm}(${rs})"),
        0x32 => format!("lwc2 ${rt}, {imm}(${rs})"),
        0x3A => format!("swc2 ${rt}, {imm}(${rs})"),
        _ => format!("op=0x{op:02x}"),
    }
}

fn branch_target(pc: u32, imm: i32) -> u32 {
    pc.wrapping_add(4).wrapping_add(((imm as i64) << 2) as u32)
}

fn dump_disasm(bus: &Bus, lo: u32, hi: u32, marker_addr: u32) {
    let mut addr = lo;
    while addr <= hi {
        let instr = bus.peek_instruction(addr).unwrap_or(0xDEAD_BEEF);
        let marker = if addr == marker_addr { "=>" } else { "  " };
        println!(
            "{marker} 0x{addr:08x}: 0x{instr:08x}  {}",
            disasm(instr, addr)
        );
        addr = addr.wrapping_add(4);
    }
}
