//! CPU / COP0 register viewer.
//!
//! Left-docked side panel showing:
//! - GPRs in a 2-column grid (named + hex value)
//! - PC / HI / LO
//! - COP0: SR, Cause, EPC, BadVAddr (the registers we actually touch)
//! - Retired instruction count
//! - Execution history (newest last)
//!
//! Layout + grouping mirrors PSoXide-2's `debug_pane_contents`, using
//! the themed `section` helper so each group reads as a framed block.

use std::collections::{BTreeSet, VecDeque};

use emulator_core::Cpu;
use psx_trace::InstructionRecord;

use crate::theme;

/// Canonical MIPS GPR names, indexed 0..=31.
const GPR_NAMES: [&str; 32] = [
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
    "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra",
];

/// COP0 register indices we display. These are the ones the emulator
/// actually reads or writes today; others stay zeroed until needed.
const COP0_LABELS: &[(usize, &str)] = &[
    (12, "SR"),
    (13, "Cause"),
    (14, "EPC"),
    (8, "BadVAddr"),
];

/// R3000 exception codes, by their numeric value in `CAUSE.ExcCode`.
const EXC_CODES: &[(u32, &str)] = &[
    (0, "Int"),
    (4, "AdEL"),
    (5, "AdES"),
    (6, "IBE"),
    (7, "DBE"),
    (8, "Syscall"),
    (9, "Bp"),
    (10, "RI"),
    (11, "CpU"),
    (12, "Ov"),
];

/// Paint the register viewer as a left-docked resizable side panel.
pub fn draw(
    ctx: &egui::Context,
    cpu: &Cpu,
    history: &VecDeque<InstructionRecord>,
    breakpoints: &mut BTreeSet<u32>,
) {
    egui::SidePanel::left("registers")
        .resizable(true)
        .default_width(340.0)
        .min_width(260.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                theme::section(ui, "GPR", |ui| draw_gprs(ui, cpu));
                theme::section(ui, "PC / HI / LO", |ui| draw_pc_hi_lo(ui, cpu));
                theme::section(ui, "COP0", |ui| draw_cop0(ui, cpu));
                theme::section(ui, "Retired", |ui| {
                    ui.monospace(format!("tick = {}", cpu.tick()));
                });
                theme::section(ui, "Breakpoints", |ui| draw_breakpoints(ui, breakpoints));
                theme::section(ui, "Execution history", |ui| draw_history(ui, history));
            });
        });
}

fn draw_breakpoints(ui: &mut egui::Ui, breakpoints: &mut BTreeSet<u32>) {
    if breakpoints.is_empty() {
        ui.monospace("(none — set from the memory panel)");
        return;
    }
    // Collect first so we can mutate the set while iterating.
    let addrs: Vec<u32> = breakpoints.iter().copied().collect();
    for addr in addrs {
        ui.horizontal(|ui| {
            ui.monospace(format!("{addr:08X}"));
            if ui.small_button("×").on_hover_text("Remove").clicked() {
                breakpoints.remove(&addr);
            }
        });
    }
}

fn draw_history(ui: &mut egui::Ui, history: &VecDeque<InstructionRecord>) {
    if history.is_empty() {
        ui.monospace("(empty — step or run the CPU)");
        return;
    }
    // Newest at the bottom: log-style reading order. Each row is
    // "PC: mnemonic" via the in-tree MIPS disassembler.
    for record in history {
        let mnem = crate::disasm::disasm(record.pc, record.instr);
        ui.monospace(format!("{:08X}  {mnem}", record.pc));
    }
}

fn draw_gprs(ui: &mut egui::Ui, cpu: &Cpu) {
    let gprs = cpu.gprs();
    egui::Grid::new("gprs")
        .num_columns(2)
        .spacing(egui::vec2(12.0, 2.0))
        .show(ui, |ui| {
            // Lay out as two columns: registers 0..16 on left, 16..32 on right.
            for i in 0..16 {
                reg_cell(ui, GPR_NAMES[i], gprs[i]);
                reg_cell(ui, GPR_NAMES[i + 16], gprs[i + 16]);
                ui.end_row();
            }
        });
}

fn draw_pc_hi_lo(ui: &mut egui::Ui, cpu: &Cpu) {
    egui::Grid::new("pc_hi_lo")
        .num_columns(2)
        .spacing(egui::vec2(12.0, 2.0))
        .show(ui, |ui| {
            reg_cell(ui, "PC", cpu.pc());
            reg_cell(ui, "HI", cpu.hi());
            ui.end_row();
            reg_cell(ui, "LO", cpu.lo());
            ui.label(""); // keep 2 cols
            ui.end_row();
        });
}

fn draw_cop0(ui: &mut egui::Ui, cpu: &Cpu) {
    let cop0 = cpu.cop0();
    egui::Grid::new("cop0")
        .num_columns(2)
        .spacing(egui::vec2(12.0, 2.0))
        .show(ui, |ui| {
            for (idx, label) in COP0_LABELS {
                reg_cell(ui, label, cop0[*idx]);
                ui.label(""); // single-column layout for COP0, keep grid aligned
                ui.end_row();
            }
        });

    // Bit-level breakdowns for the two registers whose hex values are
    // opaque at a glance. BadVAddr / EPC are raw addresses; they don't
    // benefit from the same treatment.
    ui.add_space(4.0);
    ui.small(format!("SR: {}", format_sr_bits(cop0[12])));
    ui.small(format!("Cause: {}", format_cause_bits(cop0[13])));
}

fn format_sr_bits(sr: u32) -> String {
    // Bits we actually care about at a glance. Flags show by name only
    // when set; the KU/IE stack shows as three comma-joined pairs.
    let mut flags: Vec<&str> = Vec::new();
    for (bit, name) in [
        (16, "IsC"), (17, "SwC"), (22, "BEV"),
        (28, "CU0"), (30, "CU2"),
    ] {
        if sr & (1 << bit) != 0 {
            flags.push(name);
        }
    }
    let stack = format!(
        "c={ku_c}/{ie_c} p={ku_p}/{ie_p} o={ku_o}/{ie_o}",
        ie_c = sr & 1,
        ku_c = (sr >> 1) & 1,
        ie_p = (sr >> 2) & 1,
        ku_p = (sr >> 3) & 1,
        ie_o = (sr >> 4) & 1,
        ku_o = (sr >> 5) & 1,
    );
    let im = (sr >> 8) & 0xFF;
    let flags_str = if flags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", flags.join(" "))
    };
    format!("{stack}  IM=0x{im:02X}{flags_str}")
}

fn format_cause_bits(cause: u32) -> String {
    let exc_code = (cause >> 2) & 0x1F;
    let ip = (cause >> 8) & 0xFF;
    let bd = if cause & (1 << 31) != 0 { " BD" } else { "" };
    let exc_name = EXC_CODES
        .iter()
        .find_map(|(c, name)| if *c == exc_code { Some(*name) } else { None })
        .unwrap_or("?");
    format!("ExcCode={exc_code} ({exc_name})  IP=0x{ip:02X}{bd}")
}

fn reg_cell(ui: &mut egui::Ui, name: &str, value: u32) {
    ui.monospace(format!("{name:>4}={value:08X}"));
}
