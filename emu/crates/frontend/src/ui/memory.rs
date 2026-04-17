//! Memory viewer panel — hex+ASCII dump of a 1 KiB window anchored
//! at a user-selectable address.
//!
//! Quick-jump buttons land the window at the canonical entry points
//! for each region (RAM, scratchpad, MMIO, BIOS) and at the current
//! PC. Unmapped rows render as `--` so the viewer doesn't panic when
//! the user scrolls past the end of a region.

use std::collections::BTreeSet;

use emulator_core::{Bus, Cpu};

use crate::disasm;
use crate::theme;

const BYTES_PER_ROW: usize = 16;
const ROWS: usize = 64;
const WINDOW_SIZE: u32 = (BYTES_PER_ROW * ROWS) as u32;
/// In disasm mode: one row per instruction (4 bytes).
const DISASM_ROWS: usize = 64;

/// Which format the memory panel is displaying.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ViewMode {
    /// Hex+ASCII dump at 16 bytes/row.
    Hex,
    /// Mnemonic-per-row disassembly at 4 bytes/row. Each row also
    /// echoes the raw instruction word on the right so you can still
    /// cross-check opcodes by eye.
    Disasm,
}

/// Mutable view-state the panel owns.
pub struct MemoryView {
    pub addr: u32,
    pub mode: ViewMode,
    /// Address input as a string — kept separately so partial typing
    /// ("0x8001_") doesn't immediately clobber the numeric anchor.
    addr_input: String,
}

impl Default for MemoryView {
    fn default() -> Self {
        Self {
            addr: 0x8000_0000,
            mode: ViewMode::Hex,
            addr_input: "80000000".into(),
        }
    }
}

impl MemoryView {
    /// Move the viewer to `addr` and sync the text field. Alignment
    /// depends on the current mode — 16-byte rows in hex, 4-byte rows
    /// in disasm.
    pub fn jump_to(&mut self, addr: u32) {
        let mask = match self.mode {
            ViewMode::Hex => !0x0F,
            ViewMode::Disasm => !0x03,
        };
        self.addr = addr & mask;
        self.addr_input = format!("{:08X}", self.addr);
    }
}

pub fn draw(
    ctx: &egui::Context,
    view: &mut MemoryView,
    bus: Option<&Bus>,
    cpu: &Cpu,
    breakpoints: &mut BTreeSet<u32>,
) {
    egui::SidePanel::right("memory")
        .resizable(true)
        .default_width(420.0)
        .min_width(340.0)
        .show(ctx, |ui| {
            theme::viz_frame(ui, "Memory", |ui| {
                draw_header(ui, view, cpu, breakpoints);
                ui.separator();
                match view.mode {
                    ViewMode::Hex => draw_hex_dump(ui, view, bus, breakpoints, cpu.pc()),
                    ViewMode::Disasm => draw_disasm(ui, view, bus, breakpoints, cpu.pc()),
                }
            });
        });
}

fn draw_header(
    ui: &mut egui::Ui,
    view: &mut MemoryView,
    cpu: &Cpu,
    breakpoints: &mut BTreeSet<u32>,
) {
    ui.horizontal(|ui| {
        ui.label("addr");
        let resp = ui.add(
            egui::TextEdit::singleline(&mut view.addr_input)
                .desired_width(80.0)
                .font(egui::TextStyle::Monospace),
        );
        if resp.lost_focus() && resp.ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
            apply_addr_input(view);
        }
    });

    ui.horizontal_wrapped(|ui| {
        if ui.button("RAM").clicked() {
            view.jump_to(0x8000_0000);
        }
        if ui.button("Scratchpad").clicked() {
            view.jump_to(0x1F80_0000);
        }
        if ui.button("MMIO").clicked() {
            view.jump_to(0x1F80_1000);
        }
        if ui.button("BIOS").clicked() {
            view.jump_to(0xBFC0_0000);
        }
        if ui.button("PC").clicked() {
            view.jump_to(cpu.pc());
        }
    });

    ui.horizontal(|ui| {
        let step = match view.mode {
            ViewMode::Hex => 256,
            ViewMode::Disasm => 64, // 16 instructions
        };
        if ui.button(format!("◀ -{step}")).clicked() {
            view.addr = view.addr.wrapping_sub(step);
            view.addr_input = format!("{:08X}", view.addr);
        }
        if ui.button(format!("+{step} ▶")).clicked() {
            view.addr = view.addr.wrapping_add(step);
            view.addr_input = format!("{:08X}", view.addr);
        }
        let bp_label = if breakpoints.contains(&view.addr) {
            "Clear BP"
        } else {
            "Set BP"
        };
        if ui.button(bp_label).clicked() && !breakpoints.remove(&view.addr) {
            breakpoints.insert(view.addr);
        }
    });

    // Mode toggle — separate row so it doesn't fight for space with
    // the nav buttons.
    ui.horizontal(|ui| {
        ui.radio_value(&mut view.mode, ViewMode::Hex, "Hex");
        ui.radio_value(&mut view.mode, ViewMode::Disasm, "Disasm");
    });
}

fn apply_addr_input(view: &mut MemoryView) {
    let s = view.addr_input.trim_start_matches("0x").trim();
    if let Ok(a) = u32::from_str_radix(s, 16) {
        view.addr = a & !0x0F;
        view.addr_input = format!("{:08X}", view.addr);
    }
}

fn draw_hex_dump(
    ui: &mut egui::Ui,
    view: &MemoryView,
    bus: Option<&Bus>,
    breakpoints: &BTreeSet<u32>,
    pc: u32,
) {
    let Some(bus) = bus else {
        ui.monospace("(no BIOS loaded — Bus unavailable)");
        return;
    };

    egui::ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for row in 0..ROWS {
                let row_addr = view.addr.wrapping_add(row as u32 * BYTES_PER_ROW as u32);
                let has_bp = row_has_breakpoint(row_addr, breakpoints);
                let has_pc = row_contains(row_addr, pc);
                let text = format_row(bus, row_addr, has_bp, has_pc);

                let color = match (has_pc, has_bp) {
                    // PC wins over BP — the arrow marker is the one we
                    // most want to eyeball.
                    (true, _) => Some(egui::Color32::from_rgb(80, 200, 120)),
                    (false, true) => Some(theme::ACCENT),
                    (false, false) => None,
                };
                match color {
                    Some(c) => ui.monospace(egui::RichText::new(text).color(c)),
                    None => ui.monospace(text),
                };
                if row_addr.wrapping_add(BYTES_PER_ROW as u32) < row_addr {
                    break;
                }
            }
            let _ = WINDOW_SIZE;
        });
}

fn row_has_breakpoint(base: u32, breakpoints: &BTreeSet<u32>) -> bool {
    for i in 0..BYTES_PER_ROW as u32 {
        if breakpoints.contains(&base.wrapping_add(i)) {
            return true;
        }
    }
    false
}

fn row_contains(base: u32, addr: u32) -> bool {
    addr.wrapping_sub(base) < BYTES_PER_ROW as u32
}

fn draw_disasm(
    ui: &mut egui::Ui,
    view: &MemoryView,
    bus: Option<&Bus>,
    breakpoints: &BTreeSet<u32>,
    pc: u32,
) {
    let Some(bus) = bus else {
        ui.monospace("(no BIOS loaded — Bus unavailable)");
        return;
    };

    egui::ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for row in 0..DISASM_ROWS {
                let addr = view.addr.wrapping_add(row as u32 * 4);
                let instr = read_instr(bus, addr);
                let has_bp = breakpoints.contains(&addr);
                let has_pc = addr == pc;

                let marker = match (has_pc, has_bp) {
                    (true, _) => '▸',
                    (false, true) => '●',
                    (false, false) => ' ',
                };
                let text = match instr {
                    Some(w) => format!("{marker} {addr:08X}  {w:08X}  {}", disasm::disasm(addr, w)),
                    None => format!("{marker} {addr:08X}  --------  (unmapped)"),
                };

                let color = match (has_pc, has_bp) {
                    (true, _) => Some(egui::Color32::from_rgb(80, 200, 120)),
                    (false, true) => Some(theme::ACCENT),
                    (false, false) => None,
                };
                match color {
                    Some(c) => ui.monospace(egui::RichText::new(text).color(c)),
                    None => ui.monospace(text),
                };
            }
        });
}

fn read_instr(bus: &Bus, addr: u32) -> Option<u32> {
    let b0 = bus.try_read8(addr)?;
    let b1 = bus.try_read8(addr.wrapping_add(1))?;
    let b2 = bus.try_read8(addr.wrapping_add(2))?;
    let b3 = bus.try_read8(addr.wrapping_add(3))?;
    Some(u32::from_le_bytes([b0, b1, b2, b3]))
}

fn format_row(bus: &Bus, base: u32, has_bp: bool, has_pc: bool) -> String {
    // Markers: `▸` for PC row, `●` for breakpoint row. PC wins since
    // knowing where execution is now is more urgent than which
    // addresses we've decided to stop on.
    let marker = match (has_pc, has_bp) {
        (true, _) => '▸',
        (false, true) => '●',
        (false, false) => ' ',
    };
    let mut out = format!("{marker} {base:08X}  ");
    let mut ascii = String::with_capacity(BYTES_PER_ROW);

    for i in 0..BYTES_PER_ROW {
        let addr = base.wrapping_add(i as u32);
        match bus.try_read8(addr) {
            Some(b) => {
                out.push_str(&format!("{b:02X} "));
                ascii.push(printable_ascii(b));
            }
            None => {
                out.push_str("-- ");
                ascii.push('.');
            }
        }
        if i == 7 {
            out.push(' ');
        }
    }
    out.push(' ');
    out.push_str(&ascii);
    out
}

fn printable_ascii(b: u8) -> char {
    if (0x20..0x7F).contains(&b) {
        b as char
    } else {
        '.'
    }
}
