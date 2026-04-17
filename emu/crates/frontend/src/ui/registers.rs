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

use std::collections::VecDeque;

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

/// Paint the register viewer as a left-docked resizable side panel.
pub fn draw(ctx: &egui::Context, cpu: &Cpu, history: &VecDeque<InstructionRecord>) {
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
                theme::section(ui, "Execution history", |ui| draw_history(ui, history));
            });
        });
}

fn draw_history(ui: &mut egui::Ui, history: &VecDeque<InstructionRecord>) {
    if history.is_empty() {
        ui.monospace("(empty — step or run the CPU)");
        return;
    }
    // Newest at the bottom: log-style reading order.
    for record in history {
        ui.monospace(format!("{:08X}: {:08X}", record.pc, record.instr));
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
}

fn reg_cell(ui: &mut egui::Ui, name: &str, value: u32) {
    ui.monospace(format!("{name:>4}={value:08X}"));
}
