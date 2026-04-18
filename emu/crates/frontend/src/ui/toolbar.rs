//! Top-of-window toolbar.
//!
//! Three icon buttons on the right: play/pause (context-sensitive),
//! reset, advance-one-frame. The rest of the bar stays empty for now —
//! this is where a game-title or status pill would sit as the UI
//! grows. Everything is clickable; pressing a button fires the same
//! state transition as the corresponding Menu menu item, so keyboard
//! and mouse users stay in sync.
//!
//! Keeps the CentralPanel free of widgets — before this, the only
//! top-of-viewport control was a "Run speed" slider that most users
//! never touched. Removing it reclaims vertical space for the
//! framebuffer.
//!
//! Layout shape:
//!
//! ```text
//!  ┌────────────────────────────────────────[▶][⟲][⇥]──┐
//!  │                                                    │
//!  │                 (framebuffer)                      │
//!  │                                                    │
//!  └────────────────────────────────────────────────────┘
//! ```

use egui::{Button, Context, Layout, TopBottomPanel};

use crate::app::{self, AppState};
use crate::icons;

/// Icon font size in the toolbar. Matches the HUD's text size so the
/// bar feels like a sibling of the other system chrome, not a
/// separate visual register.
const ICON_SIZE: f32 = 18.0;

/// Paint the top toolbar. Called once per frame before the central
/// panel so the framebuffer clips underneath it.
pub fn draw(ctx: &Context, state: &mut AppState) {
    TopBottomPanel::top("toolbar")
        .resizable(false)
        .show(ctx, |ui| {
            ui.add_space(2.0);
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                // Advance one frame — runs `run_steps_per_frame`
                // instructions unconditionally, regardless of the
                // current run/pause state. Useful for nudging a
                // paused game forward when investigating a hang.
                let step_btn = Button::new(icons::text(icons::SKIP_FORWARD, ICON_SIZE))
                    .min_size(egui::vec2(32.0, 28.0));
                if ui
                    .add(step_btn)
                    .on_hover_text("Advance one frame")
                    .clicked()
                {
                    // Pause first so the shell's per-frame `run`
                    // doesn't also step; otherwise clicking while
                    // running advances two frames instead of one.
                    state.running = false;
                    state.menu.sync_run_label(false);
                    app::step_one_frame(state);
                }

                // Reset — rebuild the CPU, clear VRAM, leave the Bus
                // (disc still inserted). Mirrors `MenuAction::Reset`.
                let reset_btn = Button::new(icons::text(icons::ROTATE_CCW, ICON_SIZE))
                    .min_size(egui::vec2(32.0, 28.0));
                if ui
                    .add(reset_btn)
                    .on_hover_text("Reset CPU")
                    .clicked()
                {
                    state.cpu = emulator_core::Cpu::new();
                    state.running = false;
                    state.exec_history.clear();
                    state.gpr_snapshot = None;
                    state.menu.sync_run_label(false);
                    if let Some(bus) = state.bus.as_mut() {
                        bus.gpu.vram.clear();
                    }
                }

                // Play / Pause — icon flips with `running`. Tooltip
                // also flips so the button's intent is never
                // ambiguous.
                let (icon, tooltip) = if state.running {
                    (icons::PAUSE, "Pause")
                } else {
                    (icons::PLAY, "Play")
                };
                let play_btn = Button::new(icons::text(icon, ICON_SIZE))
                    .min_size(egui::vec2(32.0, 28.0));
                if ui.add(play_btn).on_hover_text(tooltip).clicked() {
                    state.running = !state.running;
                    state.menu.sync_run_label(state.running);
                }
            });
            ui.add_space(2.0);
        });
}
