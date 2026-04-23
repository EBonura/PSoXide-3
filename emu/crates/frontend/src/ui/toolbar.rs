//! Top-of-window toolbar.
//!
//! Two halves on a single strip:
//!
//! - **Left**: live emulator status. Colored dot + RUNNING/PAUSED
//!   label, followed by FPS / MIPS / frame-time — the same values
//!   the bottom HUD used to show, now consolidated so the
//!   framebuffer isn't bracketed by two metric bars.
//! - **Right**: icon buttons. Play/pause (context-sensitive),
//!   reset, advance-one-frame. Clicking fires the same state
//!   transition as the equivalent Menu menu item.
//!
//! Layout shape:
//!
//! ```text
//!  ┌────────────────────────────────────────────────────────────────┐
//!  │ ● RUNNING    60.0 FPS   58 MIPS   16.7 ms            ▶  ⟲  ⇥ │
//!  └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The left half lays out left-to-right; we then open a
//! `right_to_left` sub-layout so the buttons cluster on the right
//! without needing a manual spacer. Both halves live inside a
//! single `horizontal_centered` so the toolbar's height is bounded
//! (a bare `right_to_left` at the panel root claimed unbounded
//! height and inflated to cover the whole window — the previous
//! iteration's regression).

use egui::{Align, Button, Color32, Context, Label, Layout, RichText, TopBottomPanel};

use crate::app::{self, AppState, ScaleMode};
use crate::icons;

/// Icon font size in the button row.
const ICON_SIZE: f32 = 16.0;
/// Minimum clickable size for each icon button. egui will grow the
/// button to fit its content but won't shrink below this.
const BUTTON_SIZE: egui::Vec2 = egui::vec2(30.0, 26.0);
/// Exact height of the toolbar panel.
const BAR_HEIGHT: f32 = 34.0;

/// Text size for the left-hand metrics cluster. Slightly smaller than
/// the default UI font so the three values fit without wrapping on a
/// narrow window.
const METRIC_TEXT_SIZE: f32 = 12.0;

/// Running / paused status colours. Green when the CPU is stepping,
/// dim-text when paused. Kept out of `theme.rs` for now — it's the
/// only place these get used.
const STATUS_RUNNING: Color32 = Color32::from_rgb(80, 200, 120);
const STATUS_PAUSED: Color32 = Color32::from_rgb(153, 153, 166);
const METRIC_TEXT: Color32 = Color32::from_rgb(204, 204, 217);
const METRIC_LABEL: Color32 = Color32::from_rgb(102, 102, 115);

/// Paint the top toolbar. Called once per frame before the central
/// panel so the framebuffer clips underneath it.
pub fn draw(ctx: &Context, state: &mut AppState) {
    TopBottomPanel::top("toolbar")
        .resizable(false)
        .exact_height(BAR_HEIGHT)
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                draw_metrics(ui, state);
                // Push buttons to the right of the remaining space.
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    draw_buttons(ui, state);
                    ui.add_space(12.0);
                    draw_audio_controls(ui, state);
                    ui.add_space(12.0);
                    draw_boot_toggle(ui, state);
                    ui.add_space(12.0);
                    draw_debug_toggles(ui, state);
                });
            });
        });
}

/// Compact volume control: mute button plus gain slider.
fn draw_audio_controls(ui: &mut egui::Ui, state: &mut AppState) {
    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
        let effective = state.effective_audio_volume();
        let icon = if effective <= 0.0 {
            icons::VOLUME_X
        } else if effective < 0.6 {
            icons::VOLUME_1
        } else {
            icons::VOLUME_2
        };
        let btn = toggle_button(icon, !state.audio_muted && state.audio_volume > 0.0);
        if ui.add(btn).on_hover_text("Mute / unmute audio").clicked() {
            state.audio_muted = !state.audio_muted;
            state.status_message_set(if state.audio_muted {
                "Audio muted"
            } else {
                "Audio unmuted"
            });
        }
        let before = state.audio_volume;
        let slider = egui::Slider::new(&mut state.audio_volume, 0.0..=1.5)
            .show_value(false)
            .clamping(egui::SliderClamping::Always);
        let response = ui.add_sized([86.0, 18.0], slider).on_hover_text("Volume");
        if response.changed() {
            state.audio_volume = state.audio_volume.clamp(0.0, 1.5);
            if state.audio_volume > 0.0 && before <= 0.0 {
                state.audio_muted = false;
            }
        }
    });
}

/// Disc boot-mode toggle. Active means warm fast boot is used on the
/// next disc launch; inactive means the full BIOS logo path.
fn draw_boot_toggle(ui: &mut egui::Ui, state: &mut AppState) {
    let enabled = state.settings.emulator.fast_boot_disc;
    let tooltip = if enabled {
        "Disc fast boot enabled - skips BIOS logo"
    } else {
        "Disc fast boot disabled - full BIOS logo boot"
    };
    let btn = toggle_button(icons::DISC, enabled);
    if ui.add(btn).on_hover_text(tooltip).clicked() {
        state.toggle_fast_boot_disc();
    }
}

/// Debug-panel toggle cluster. Each button reflects the current on/off
/// state of its panel — active buttons tint with the "running" green;
/// inactive stay dim. Clicking flips the panel's visibility.
fn draw_debug_toggles(ui: &mut egui::Ui, state: &mut AppState) {
    // Added right-to-left; reading left-to-right on screen the icons
    // will appear in the opposite order (wireframe on the left).
    debug_toggle(
        ui,
        icons::LAYERS,
        "Toggle VRAM viewer",
        &mut state.panels.vram,
    );
    debug_toggle(
        ui,
        icons::TERMINAL,
        "Toggle memory viewer",
        &mut state.panels.memory,
    );
    debug_toggle(
        ui,
        icons::CPU,
        "Toggle CPU registers panel",
        &mut state.panels.registers,
    );
    // Wireframe mode lives on the GPU, not a frontend panel — we
    // dereference through Bus to flip it.
    let wf_active = state
        .bus
        .as_ref()
        .map(|b| b.gpu.wireframe_enabled)
        .unwrap_or(false);
    let wf_btn = toggle_button(icons::GRID, wf_active);
    if ui
        .add(wf_btn)
        .on_hover_text("Wireframe mode — edges only")
        .clicked()
    {
        if let Some(bus) = state.bus.as_mut() {
            bus.gpu.wireframe_enabled = !bus.gpu.wireframe_enabled;
        }
    }
    // Resolution / scale-mode switch: full-window presentation ↔
    // exact native 1× pixels. Active means native inspection mode.
    let native_active = state.scale_mode == ScaleMode::Native;
    let scale_icon = if native_active {
        icons::MINIMIZE
    } else {
        icons::MAXIMIZE
    };
    let scale_btn = toggle_button(scale_icon, native_active);
    if ui
        .add(scale_btn)
        .on_hover_text("Toggle full-window scale vs. native 1x pixels")
        .clicked()
    {
        state.scale_mode = match state.scale_mode {
            ScaleMode::Window => ScaleMode::Native,
            ScaleMode::Native => ScaleMode::Window,
        };
    }
}

/// One flag-backed toggle button. Tooltip + tint reflect the flag;
/// clicking mutates it in place.
fn debug_toggle(ui: &mut egui::Ui, icon: char, tooltip: &str, flag: &mut bool) {
    let btn = toggle_button(icon, *flag);
    if ui.add(btn).on_hover_text(tooltip).clicked() {
        *flag = !*flag;
    }
}

/// Build a Button at the shared icon size, tinted to indicate active
/// vs. inactive state. Keeps the toggle cluster visually coherent.
fn toggle_button(icon: char, active: bool) -> Button<'static> {
    let color = if active { STATUS_RUNNING } else { METRIC_LABEL };
    let label = RichText::new(icon.to_string())
        .font(icons::font(ICON_SIZE))
        .color(color);
    Button::new(label).min_size(BUTTON_SIZE)
}

/// Left-hand cluster: status pill + FPS / MIPS / dt metrics.
fn draw_metrics(ui: &mut egui::Ui, state: &AppState) {
    ui.add_space(8.0);

    let (dot_color, status_label) = if state.running {
        (STATUS_RUNNING, "RUNNING")
    } else {
        (STATUS_PAUSED, "PAUSED")
    };
    ui.add(Label::new(
        RichText::new("●").color(dot_color).size(METRIC_TEXT_SIZE),
    ));
    ui.add(Label::new(
        RichText::new(status_label)
            .color(METRIC_TEXT)
            .size(METRIC_TEXT_SIZE),
    ));

    ui.add_space(16.0);

    let fps = state.hud.fps();
    let ms = state.hud.average_dt() * 1000.0;
    let mips = state.hud.ips() / 1_000_000.0;
    let audio = state.hud.audio_queue_len();

    metric(ui, "FPS", format!("{fps:4.1}"));
    ui.add_space(12.0);
    metric(ui, "MIPS", format!("{mips:4.1}"));
    ui.add_space(12.0);
    metric(ui, "dt", format!("{ms:4.1} ms"));
    ui.add_space(12.0);
    metric(ui, "AUDIO", format!("{audio}"));
}

/// One "LABEL value" pair, formatted so the label is dim and the
/// value uses the primary text colour. Looks uniform across the row.
fn metric(ui: &mut egui::Ui, label: &str, value: String) {
    ui.add(Label::new(
        RichText::new(label)
            .color(METRIC_LABEL)
            .size(METRIC_TEXT_SIZE),
    ));
    ui.add_space(4.0);
    ui.add(Label::new(
        RichText::new(value)
            .color(METRIC_TEXT)
            .monospace()
            .size(METRIC_TEXT_SIZE),
    ));
}

/// Right-hand cluster of icon buttons. Added in visual-right-to-left
/// order because the surrounding layout is `right_to_left`, so the
/// first button added sits at the rightmost edge.
fn draw_buttons(ui: &mut egui::Ui, state: &mut AppState) {
    ui.add_space(4.0);

    // Advance one emulated frame. Pauses first so clicking while
    // running doesn't advance two frames (one from the click, one
    // from the shell's run loop).
    let step_btn = Button::new(icons::text(icons::SKIP_FORWARD, ICON_SIZE)).min_size(BUTTON_SIZE);
    if ui
        .add(step_btn)
        .on_hover_text("Advance one frame")
        .clicked()
    {
        state.running = false;
        state.menu.sync_run_label(false);
        app::step_one_frame(state);
    }

    // Reset — rebuild the CPU, clear VRAM, keep the Bus (disc stays
    // inserted).
    let reset_btn = Button::new(icons::text(icons::ROTATE_CCW, ICON_SIZE)).min_size(BUTTON_SIZE);
    if ui.add(reset_btn).on_hover_text("Reset CPU").clicked() {
        state.cpu = emulator_core::Cpu::new();
        state.running = false;
        state.exec_history.clear();
        state.gpr_snapshot = None;
        state.menu.sync_run_label(false);
        if let Some(bus) = state.bus.as_mut() {
            bus.gpu.vram.clear();
        }
    }

    // Play / Pause — icon + tooltip flip with state.
    let (icon, tooltip) = if state.running {
        (icons::PAUSE, "Pause")
    } else {
        (icons::PLAY, "Play")
    };
    let play_btn = Button::new(icons::text(icon, ICON_SIZE)).min_size(BUTTON_SIZE);
    if ui.add(play_btn).on_hover_text(tooltip).clicked() {
        state.running = !state.running;
        state.menu.sync_run_label(state.running);
    }
}
