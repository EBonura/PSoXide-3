//! Menu overlay — launcher menu ported from psoxide-1.
//!
//! Horizontal animated category icons with a vertical item list beneath
//! the active category. Drawn via `egui::Painter` on a middle layer so
//! it overlays the framebuffer/central area but sits below the HUD.
//!
//! Navigation: arrows + Enter + Escape (gamepad will land when the
//! input subsystem does). Escape also toggles the overlay open/closed.
//!
//! Phase 1e ships three categories — Game / Debug / System. Games,
//! Examples, Save States, Video, Input gain their sections as the
//! corresponding subsystems come online.

use egui::{Align2, FontId, Pos2, Rect, Vec2};

use crate::icons;
use crate::theme;

const CATEGORY_SPACING: f32 = 100.0;
const ICON_SIZE_ACTIVE: f32 = 32.0;
const ICON_SIZE_INACTIVE: f32 = 20.0;
const ITEM_HEIGHT: f32 = 40.0;
const ITEM_WIDTH: f32 = 400.0;
const ITEM_GAP: f32 = 2.0;
const ANIM_SPEED: f32 = 10.0;

/// A menu action the Menu emits when the user confirms an item. The
/// app layer interprets these — Menu stays stateless about the
/// emulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    /// Toggle between continuous-run and paused.
    ToggleRun,
    /// Advance the CPU by one retired instruction.
    StepOne,
    /// Reseat the CPU at its reset vector.
    Reset,
    /// Paint a test pattern into VRAM (dev aid until the GPU renders).
    FillVramTestPattern,
    /// Toggle visibility of the register side panel.
    ToggleRegisters,
    /// Toggle visibility of the memory viewer panel.
    ToggleMemory,
    /// Toggle visibility of the VRAM bottom panel.
    ToggleVram,
    /// Toggle visibility of the HUD overlay.
    ToggleHud,
    /// Quit the application.
    Quit,
}

/// Per-frame input snapshot the shell assembles from keyboard events.
#[derive(Default, Debug, Clone, Copy)]
pub struct MenuInput {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub confirm: bool,
    pub back: bool,
    pub toggle_open: bool,
}

struct MenuItem {
    /// Mutable so the Run/Pause label can swap in place without
    /// rebuilding the whole category tree on every toggle.
    label: &'static str,
    action: MenuAction,
    value: Option<&'static str>,
}

struct Category {
    name: &'static str,
    icon: char,
    items: Vec<MenuItem>,
}

pub struct MenuState {
    pub open: bool,
    category_index: usize,
    item_index: usize,
    anim_x: f32,
    categories: Vec<Category>,
}

impl Default for MenuState {
    fn default() -> Self {
        Self::new()
    }
}

impl MenuState {
    pub fn new() -> Self {
        Self::with_running(false)
    }

    pub fn with_running(running: bool) -> Self {
        let run_label = if running { "Pause" } else { "Run" };
        let categories = vec![
            Category {
                name: "Game",
                icon: icons::PLAY,
                items: vec![
                    MenuItem {
                        label: run_label,
                        action: MenuAction::ToggleRun,
                        value: None,
                    },
                    MenuItem {
                        label: "Step one instruction",
                        action: MenuAction::StepOne,
                        value: None,
                    },
                    MenuItem {
                        label: "Reset CPU",
                        action: MenuAction::Reset,
                        value: None,
                    },
                ],
            },
            Category {
                name: "Debug",
                icon: icons::BUG,
                items: vec![
                    MenuItem {
                        label: "Toggle registers panel",
                        action: MenuAction::ToggleRegisters,
                        value: None,
                    },
                    MenuItem {
                        label: "Toggle memory panel",
                        action: MenuAction::ToggleMemory,
                        value: None,
                    },
                    MenuItem {
                        label: "Toggle VRAM panel",
                        action: MenuAction::ToggleVram,
                        value: None,
                    },
                    MenuItem {
                        label: "Toggle HUD",
                        action: MenuAction::ToggleHud,
                        value: None,
                    },
                    MenuItem {
                        label: "Fill VRAM test pattern",
                        action: MenuAction::FillVramTestPattern,
                        value: None,
                    },
                ],
            },
            Category {
                name: "System",
                icon: icons::CPU,
                items: vec![MenuItem {
                    label: "Quit",
                    action: MenuAction::Quit,
                    value: Some("Esc ×2"),
                }],
            },
        ];

        Self {
            open: true,
            category_index: 0,
            item_index: 0,
            anim_x: 0.0,
            categories,
        }
    }

    /// Rebuild categories with a fresh "Run"/"Pause" label. Called when
    /// `AppState.running` flips.
    pub fn sync_run_label(&mut self, running: bool) {
        let run_label = if running { "Pause" } else { "Run" };
        if let Some(game) = self.categories.first_mut() {
            if let Some(item) = game.items.first_mut() {
                item.label = run_label;
            }
        }
    }

    /// Feed one frame of input. Returns `Some(action)` when a confirm
    /// selects an item.
    pub fn update(&mut self, input: &MenuInput) -> Option<MenuAction> {
        if input.toggle_open {
            self.open = !self.open;
        }
        if !self.open {
            return None;
        }

        let num_cats = self.categories.len();
        if input.left && self.category_index > 0 {
            self.category_index -= 1;
            self.item_index = 0;
        }
        if input.right && self.category_index + 1 < num_cats {
            self.category_index += 1;
            self.item_index = 0;
        }

        let num_items = self.categories[self.category_index].items.len();
        if input.up && self.item_index > 0 {
            self.item_index -= 1;
        }
        if input.down && self.item_index + 1 < num_items {
            self.item_index += 1;
        }

        if input.confirm && num_items > 0 {
            return Some(self.categories[self.category_index].items[self.item_index].action);
        }

        if input.back {
            self.open = false;
        }
        None
    }

    /// Draw the Menu overlay on a middle-layer painter. `dt` drives the
    /// slide animation.
    pub fn draw(&mut self, ctx: &egui::Context, dt: f32) {
        if !self.open {
            return;
        }

        let screen = ctx.screen_rect();
        let sw = screen.width();
        let sh = screen.height();

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Middle,
            egui::Id::new("menu"),
        ));

        painter.rect_filled(screen, 0.0, theme::MENU_BACKDROP);

        // Animate horizontal slide.
        let target_x = self.category_index as f32;
        self.anim_x += (target_x - self.anim_x) * ANIM_SPEED * dt;
        if (self.anim_x - target_x).abs() < 0.001 {
            self.anim_x = target_x;
        }

        let center_x = sw / 2.0;
        let center_y = sh * 0.38;

        // Category row.
        for (i, cat) in self.categories.iter().enumerate() {
            let offset = i as f32 - self.anim_x;
            let x = center_x + offset * CATEGORY_SPACING;
            let is_active = i == self.category_index;
            let size = if is_active {
                ICON_SIZE_ACTIVE
            } else {
                ICON_SIZE_INACTIVE
            };
            let color = if is_active {
                theme::MENU_ACCENT
            } else {
                theme::MENU_TEXT_DIM
            };

            if x < -50.0 || x > sw + 50.0 {
                continue;
            }
            painter.text(
                Pos2::new(x, center_y),
                Align2::CENTER_CENTER,
                cat.icon.to_string(),
                icons::font(size),
                color,
            );
            if is_active {
                painter.text(
                    Pos2::new(x, center_y + size / 2.0 + 20.0),
                    Align2::CENTER_TOP,
                    cat.name,
                    FontId::proportional(16.0),
                    theme::MENU_TEXT_BRIGHT,
                );
            }
        }

        // Item list.
        let cat = &self.categories[self.category_index];
        let items_start_y = center_y + ICON_SIZE_ACTIVE + 44.0;
        let items_x = center_x - ITEM_WIDTH / 2.0;
        let label_font = FontId::proportional(15.0);
        let value_font = FontId::proportional(13.0);

        for (i, item) in cat.items.iter().enumerate() {
            let y = items_start_y + i as f32 * (ITEM_HEIGHT + ITEM_GAP);
            let is_selected = i == self.item_index;

            let bg = if is_selected {
                theme::MENU_ITEM_SEL
            } else {
                theme::MENU_ITEM_BG
            };
            let rect =
                Rect::from_min_size(Pos2::new(items_x, y), Vec2::new(ITEM_WIDTH, ITEM_HEIGHT));
            painter.rect_filled(rect, 0.0, bg);

            if is_selected {
                painter.rect_filled(
                    Rect::from_min_size(Pos2::new(items_x, y), Vec2::new(3.0, ITEM_HEIGHT)),
                    0.0,
                    theme::MENU_ACCENT,
                );
            }

            let label_color = if is_selected {
                theme::MENU_TEXT_BRIGHT
            } else {
                theme::MENU_TEXT_DIM
            };
            painter.text(
                Pos2::new(items_x + 14.0, y + ITEM_HEIGHT / 2.0),
                Align2::LEFT_CENTER,
                item.label,
                label_font.clone(),
                label_color,
            );

            if let Some(val) = item.value {
                let val_color = if is_selected {
                    theme::MENU_TEXT_VALUE
                } else {
                    theme::MENU_TEXT_DIM
                };
                painter.text(
                    Pos2::new(items_x + ITEM_WIDTH - 12.0, y + ITEM_HEIGHT / 2.0),
                    Align2::RIGHT_CENTER,
                    val,
                    value_font.clone(),
                    val_color,
                );
            }
        }

        // Bottom hint bar.
        painter.text(
            Pos2::new(sw / 2.0, sh - 30.0),
            Align2::CENTER_TOP,
            "Enter: Select   Esc: Close   Arrows: Navigate",
            FontId::proportional(12.0),
            theme::MENU_HINT,
        );
    }
}
