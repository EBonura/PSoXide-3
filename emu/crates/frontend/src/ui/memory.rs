//! Memory viewer panel — hex+ASCII dump of a 1 KiB window anchored
//! at a user-selectable address.
//!
//! Quick-jump buttons land the window at the canonical entry points
//! for each region (RAM, scratchpad, MMIO, BIOS) and at the current
//! PC. Unmapped rows render as `--` so the viewer doesn't panic when
//! the user scrolls past the end of a region.

use emulator_core::{Bus, Cpu};

use crate::theme;

const BYTES_PER_ROW: usize = 16;
const ROWS: usize = 64;
const WINDOW_SIZE: u32 = (BYTES_PER_ROW * ROWS) as u32;

/// Mutable view-state the panel owns.
pub struct MemoryView {
    pub addr: u32,
    /// Address input as a string — kept separately so partial typing
    /// ("0x8001_") doesn't immediately clobber the numeric anchor.
    addr_input: String,
}

impl Default for MemoryView {
    fn default() -> Self {
        Self {
            addr: 0x8000_0000,
            addr_input: "80000000".into(),
        }
    }
}

impl MemoryView {
    /// Move the viewer to `addr` and sync the text field.
    pub fn jump_to(&mut self, addr: u32) {
        self.addr = addr & !0x0F; // align to 16-byte row
        self.addr_input = format!("{:08X}", self.addr);
    }
}

pub fn draw(ctx: &egui::Context, view: &mut MemoryView, bus: Option<&Bus>, cpu: &Cpu) {
    egui::SidePanel::right("memory")
        .resizable(true)
        .default_width(360.0)
        .min_width(300.0)
        .show(ctx, |ui| {
            theme::viz_frame(ui, "Memory", |ui| {
                draw_header(ui, view, cpu);
                ui.separator();
                draw_dump(ui, view, bus);
            });
        });
}

fn draw_header(ui: &mut egui::Ui, view: &mut MemoryView, cpu: &Cpu) {
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
        if ui.button("◀ -256").clicked() {
            view.addr = view.addr.wrapping_sub(256);
            view.addr_input = format!("{:08X}", view.addr);
        }
        if ui.button("+256 ▶").clicked() {
            view.addr = view.addr.wrapping_add(256);
            view.addr_input = format!("{:08X}", view.addr);
        }
    });
}

fn apply_addr_input(view: &mut MemoryView) {
    let s = view.addr_input.trim_start_matches("0x").trim();
    if let Ok(a) = u32::from_str_radix(s, 16) {
        view.addr = a & !0x0F;
        view.addr_input = format!("{:08X}", view.addr);
    }
}

fn draw_dump(ui: &mut egui::Ui, view: &MemoryView, bus: Option<&Bus>) {
    let Some(bus) = bus else {
        ui.monospace("(no BIOS loaded — Bus unavailable)");
        return;
    };

    egui::ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for row in 0..ROWS {
                let row_addr = view.addr.wrapping_add(row as u32 * BYTES_PER_ROW as u32);
                ui.monospace(format_row(bus, row_addr));
                if row_addr.wrapping_add(BYTES_PER_ROW as u32) < row_addr {
                    break; // overflow — stop instead of wrapping around
                }
            }
            // Reference the constant so the dead_code lint doesn't fire
            // on the window-size declaration while the panel's scroll
            // area grows beyond it.
            let _ = WINDOW_SIZE;
        });
}

fn format_row(bus: &Bus, base: u32) -> String {
    let mut out = format!("{base:08X}  ");
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
            out.push(' '); // extra gap at half-row
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
