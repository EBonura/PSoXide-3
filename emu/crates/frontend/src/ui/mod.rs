//! UI panel orchestration.
//!
//! Individual panels live in submodules; `draw_layout` composes them in
//! the order that makes the visual layering work: docked panels first
//! (bottom/side get clipped to remaining space), then the central area,
//! then free-floating overlays (Menu, HUD) on top.

pub mod framebuffer;
pub mod hud;
pub mod memory;
pub mod registers;
pub mod toolbar;
pub mod vram;
pub mod menu;

use crate::app::AppState;

/// Paint every panel for this frame, in layering order.
pub fn draw_layout(
    ctx: &egui::Context,
    state: &mut AppState,
    vram_tex: egui::TextureId,
    display_tex: egui::TextureId,
    framebuffer_source: framebuffer::FramebufferSource,
    dt: f32,
) {
    state.hud.update(dt, state.cpu.tick());
    state.tick_status(dt);

    if state.workspace.is_editor() {
        state.editor.draw(ctx);
        state.menu.draw(ctx, dt);
        draw_status_toast(ctx, state);
        return;
    }

    // Top-bar controls go first so the central panel (framebuffer)
    // clips to what's left under them. Docked side panels come next.
    toolbar::draw(ctx, state);

    if state.panels.registers {
        registers::draw(
            ctx,
            &state.cpu,
            &state.exec_history,
            &mut state.breakpoints,
            &mut state.gpr_snapshot,
        );
    }
    if state.panels.memory {
        memory::draw(
            ctx,
            &mut state.memory_view,
            state.bus.as_ref(),
            &state.cpu,
            &mut state.breakpoints,
        );
    }
    if state.panels.vram {
        vram::draw(ctx, vram_tex);
    }

    egui::CentralPanel::default().show(ctx, |ui| {
        framebuffer::draw(
            ui,
            display_tex,
            framebuffer_source,
            state.bus.as_ref(),
            &mut state.framebuffer_present_size_px,
        );
    });

    state.menu.draw(ctx, dt);
    draw_status_toast(ctx, state);
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
                if let Ok(record) = state.cpu.step_traced(bus) {
                    crate::app::push_history(&mut state.exec_history, record);
                }
            }
            MenuOutcome::None
        }
        Reset => {
            state.cpu = emulator_core::Cpu::new();
            state.running = false;
            state.exec_history.clear();
            state.gpr_snapshot = None;
            state.menu.sync_run_label(false);
            if let Some(bus) = state.bus.as_mut() {
                bus.gpu.vram.clear();
            }
            MenuOutcome::None
        }
        ToggleFastBoot => {
            state.toggle_fast_boot_disc();
            MenuOutcome::None
        }
        FillVramTestPattern => {
            if let Some(bus) = state.bus.as_mut() {
                fill_vram_test_pattern(&mut bus.gpu.vram);
            }
            MenuOutcome::None
        }
        LaunchGame(id) => {
            // Game-launch rebuilds Bus + Cpu from scratch. Close
            // the Menu on success so the user sees the freshly-
            // booted BIOS / EXE, exactly like a real PSX shell.
            match state.launch_by_id(&id) {
                Ok(()) => {
                    state.menu.open = false;
                }
                Err(e) => {
                    eprintln!("[frontend] launch failed: {e}");
                    state.status_message_set(format!("Launch failed: {e}"));
                }
            }
            MenuOutcome::None
        }
        RescanLibrary => {
            match state.rescan_library() {
                Ok(n) => {
                    state.status_message_set(format!(
                        "Scan complete: {} entries ({n} new/changed)",
                        state.library.entries.len()
                    ));
                }
                Err(e) => {
                    eprintln!("[frontend] rescan failed: {e}");
                    state.status_message_set(format!("Rescan failed: {e}"));
                }
            }
            MenuOutcome::None
        }
        ToggleEditorWorkspace => {
            state.toggle_editor_workspace();
            state.menu.open = false;
            MenuOutcome::None
        }
        ToggleRegisters => {
            state.panels.registers = !state.panels.registers;
            MenuOutcome::None
        }
        ToggleMemory => {
            state.panels.memory = !state.panels.memory;
            MenuOutcome::None
        }
        ToggleVram => {
            state.panels.vram = !state.panels.vram;
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

fn draw_status_toast(ctx: &egui::Context, state: &AppState) {
    let Some((msg, ttl)) = state.status_message.as_ref() else {
        return;
    };
    let alpha = (*ttl / 0.35).clamp(0.0, 1.0);
    let bg = egui::Color32::from_rgba_premultiplied(16, 18, 22, (230.0 * alpha) as u8);
    let stroke = egui::Stroke::new(
        1.0,
        egui::Color32::from_rgba_premultiplied(0, 191, 230, (180.0 * alpha) as u8),
    );
    let text = egui::Color32::from_rgba_premultiplied(235, 238, 242, (255.0 * alpha) as u8);

    egui::Area::new("status-toast".into())
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-18.0, 48.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(bg)
                .stroke(stroke)
                .corner_radius(egui::CornerRadius::same(4))
                .inner_margin(egui::Margin::symmetric(12, 8))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(msg).color(text).size(13.0));
                });
        });
}
