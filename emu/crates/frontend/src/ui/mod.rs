//! UI panel orchestration.
//!
//! Individual panels live in submodules; `draw_layout` composes them in
//! the order that makes the visual layering work: docked panels first
//! (bottom/side get clipped to remaining space), then the central area,
//! then free-floating overlays (Menu, HUD) on top.

pub mod hud;
pub mod registers;
pub mod vram;
pub mod menu;

use crate::app::AppState;

/// Paint every panel for this frame, in layering order.
pub fn draw_layout(
    ctx: &egui::Context,
    state: &mut AppState,
    vram_tex: egui::TextureId,
    dt: f32,
) {
    state.hud.push(dt);

    if state.panels.registers {
        registers::draw(ctx, &state.cpu);
    }
    if state.panels.vram {
        vram::draw(ctx, vram_tex);
    }

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("PSoXide");
        ui.label("Central area hosts the framebuffer once GPU lands.");
        ui.label(format!(
            "Menu: {} — press Esc to toggle.",
            if state.menu.open { "open" } else { "closed" }
        ));
    });

    state.menu.draw(ctx, dt);

    if state.panels.hud {
        hud::draw(ctx, &state.hud, &state.cpu, state.running);
    }
}

pub fn apply_menu_action(state: &mut AppState, action: menu::MenuAction) -> MenuOutcome {
    use menu::MenuAction::*;
    match action {
        ToggleRun => {
            state.running = !state.running;
            state.menu.sync_run_label(state.running);
            // Auto-close the overlay so Run is observable immediately.
            if state.running {
                state.menu.open = false;
            }
            MenuOutcome::None
        }
        StepOne => {
            if let Some(bus) = state.bus.as_mut() {
                let _ = state.cpu.step(bus);
            }
            MenuOutcome::None
        }
        Reset => {
            state.cpu = emulator_core::Cpu::new();
            state.running = false;
            state.menu.sync_run_label(false);
            MenuOutcome::None
        }
        FillVramTestPattern => {
            fill_vram_test_pattern(&mut state.vram);
            MenuOutcome::None
        }
        ToggleRegisters => {
            state.panels.registers = !state.panels.registers;
            MenuOutcome::None
        }
        ToggleVram => {
            state.panels.vram = !state.panels.vram;
            MenuOutcome::None
        }
        ToggleHud => {
            state.panels.hud = !state.panels.hud;
            MenuOutcome::None
        }
        Quit => MenuOutcome::Quit,
    }
}

/// What the shell needs to do after an Menu action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuOutcome {
    None,
    Quit,
}

fn fill_vram_test_pattern(vram: &mut emulator_core::Vram) {
    use emulator_core::{VRAM_HEIGHT, VRAM_WIDTH};
    // Gradient: R = x/32, G = y/16, B = (x^y) & 0x1F. Makes the viewer
    // instantly recognisable as "working" — distinct from the noise a
    // buggy upload would produce.
    for y in 0..VRAM_HEIGHT {
        for x in 0..VRAM_WIDTH {
            let r = ((x * 31 / VRAM_WIDTH) & 0x1F) as u16;
            let g = ((y * 31 / VRAM_HEIGHT) & 0x1F) as u16;
            let b = ((x ^ y) as u16) & 0x1F;
            let color = r | (g << 5) | (b << 10);
            vram.set_pixel(x as u16, y as u16, color);
        }
    }
}
