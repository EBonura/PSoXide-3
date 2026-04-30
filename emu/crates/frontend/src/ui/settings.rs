//! First-run and emulator path settings.

use std::path::{Path, PathBuf};

use crate::app::AppState;
use crate::theme;
use psoxide_settings::Settings;

/// Editable state for the emulator settings page.
#[derive(Debug, Clone)]
pub struct SettingsPanelState {
    /// Whether the settings page is currently visible.
    pub open: bool,
    /// Draft BIOS image path.
    pub bios_path: String,
    /// Draft game-library directory.
    pub game_library_path: String,
}

impl SettingsPanelState {
    /// Build draft state from persisted settings.
    pub fn from_settings(settings: &Settings, open: bool) -> Self {
        Self {
            open,
            bios_path: settings.paths.bios.clone(),
            game_library_path: settings.paths.game_library.clone(),
        }
    }

    /// Reset draft fields from persisted settings.
    pub fn sync_from_settings(&mut self, settings: &Settings) {
        self.bios_path = settings.paths.bios.clone();
        self.game_library_path = settings.paths.game_library.clone();
    }
}

/// Draw the settings page when open.
pub fn draw(ctx: &egui::Context, state: &mut AppState) {
    if !state.settings_panel.open {
        return;
    }

    let settings_file = state.paths.settings_file();
    let config_root = state.paths.root().to_path_buf();
    let env_bios = std::env::var("PSOXIDE_BIOS").ok();
    let mut open = state.settings_panel.open;
    let mut close_requested = false;
    let mut save_requested = false;
    let mut rescan_requested = false;
    let mut use_env_bios = false;

    egui::Window::new("Settings")
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .default_width(680.0)
        .collapsible(false)
        .resizable(false)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.heading("Paths");
            ui.add_space(4.0);

            if state.bios_path_missing() {
                warning_row(ui, "please chose a bios path");
            }
            if state.games_path_missing() {
                warning_row(ui, "please chose a games path");
            }

            path_field(
                ui,
                "BIOS path",
                &mut state.settings_panel.bios_path,
                "Path to SCPH1001.BIN or another PSX BIOS image",
                PathExpectation::File,
            );
            if let Some(env_bios) = env_bios.as_deref() {
                ui.horizontal(|ui| {
                    ui.add_space(110.0);
                    if ui.button("Use PSOXIDE_BIOS").clicked() {
                        use_env_bios = true;
                    }
                    ui.label(
                        egui::RichText::new(env_bios)
                            .size(12.0)
                            .color(theme::TEXT_DIM),
                    );
                });
            }

            ui.add_space(8.0);
            path_field(
                ui,
                "Games path",
                &mut state.settings_panel.game_library_path,
                "Folder containing .cue, .bin, .iso, and .exe files",
                PathExpectation::Dir,
            );

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    save_requested = true;
                }
                if ui.button("Save and refresh library").clicked() {
                    save_requested = true;
                    rescan_requested = true;
                }
                if ui.button("Reset fields").clicked() {
                    state.settings_panel.sync_from_settings(&state.settings);
                }
                if ui.button("Close").clicked() {
                    close_requested = true;
                }
            });

            ui.add_space(10.0);
            ui.separator();
            ui.small(format!("settings.ron: {}", settings_file.display()));
            ui.small(format!("config: {}", config_root.display()));
        });

    if use_env_bios {
        if let Some(env_bios) = env_bios {
            state.settings_panel.bios_path = env_bios;
        }
    }
    state.settings_panel.open = open && !close_requested;
    if save_requested {
        state.commit_settings_panel(rescan_requested);
    }
}

#[derive(Debug, Clone, Copy)]
enum PathExpectation {
    File,
    Dir,
}

fn path_field(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    hint: &str,
    expectation: PathExpectation,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).strong());
        ui.add_sized(
            [500.0, 24.0],
            egui::TextEdit::singleline(value).hint_text(hint),
        );
    });

    let status = path_status(value, expectation);
    ui.horizontal(|ui| {
        ui.add_space(110.0);
        ui.label(
            egui::RichText::new(status.text)
                .size(12.0)
                .color(status.color),
        );
    });
}

struct PathStatus {
    text: String,
    color: egui::Color32,
}

fn path_status(value: &str, expectation: PathExpectation) -> PathStatus {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return PathStatus {
            text: "Not configured".into(),
            color: egui::Color32::from_rgb(240, 96, 96),
        };
    }
    let path = PathBuf::from(trimmed);
    let ok = match expectation {
        PathExpectation::File => path.is_file(),
        PathExpectation::Dir => path.is_dir(),
    };
    if ok {
        PathStatus {
            text: "Found".into(),
            color: egui::Color32::from_rgb(120, 220, 150),
        }
    } else {
        let want = match expectation {
            PathExpectation::File => "file",
            PathExpectation::Dir => "folder",
        };
        PathStatus {
            text: format!("No {want} found at {}", display_path(&path)),
            color: egui::Color32::from_rgb(240, 170, 80),
        }
    }
}

fn display_path(path: &Path) -> String {
    path.to_str()
        .map(str::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn warning_row(ui: &mut egui::Ui, text: &str) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(126, 24, 34))
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).color(egui::Color32::WHITE));
        });
}
