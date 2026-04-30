//! Menu overlay -- launcher menu ported from psoxide-1.
//!
//! Horizontal animated category icons with a vertical item list beneath
//! the active category. Drawn via `egui::Painter` on a middle layer so
//! it overlays the framebuffer/central area but sits below the HUD.
//!
//! Navigation: arrows + Enter + Escape (gamepad will land when the
//! input subsystem does). Escape also toggles the overlay open/closed.
//!
//! Phase 1e ships three categories -- Game / Debug / System. Games,
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
/// app layer interprets these -- Menu stays stateless about the
/// emulator.
///
/// Note: dropped `Copy` in favour of `Clone` to carry the
/// game-ID payload on `LaunchGame`. The dispatch cost is one
/// `String::clone` per selection -- negligible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    /// Toggle between continuous-run and paused.
    ToggleRun,
    /// Advance the CPU by one retired instruction.
    StepOne,
    /// Reseat the CPU at its reset vector.
    Reset,
    /// Toggle warm SYSTEM.CNF disc fast boot. When disabled, discs
    /// boot through the full BIOS logo path.
    ToggleFastBoot,
    /// Paint a test pattern into VRAM (dev aid until the GPU renders).
    FillVramTestPattern,
    /// Launch a game by its stable library ID. The app layer
    /// looks the entry up in `AppState::library` and rebuilds
    /// the emulator around it.
    LaunchGame(String),
    /// Re-walk the configured library root and refresh
    /// `library.ron`. Surfaced as a "Refresh library" item in
    /// the Games / Examples categories so users can trigger a
    /// rescan without leaving the Menu.
    RescanLibrary,
    /// Enter or leave the host-side editor workspace.
    ToggleEditorWorkspace,
    /// Pick and persist the BIOS image path.
    ChooseBiosPath,
    /// Pick and persist the games library root.
    ChooseGamesPath,
    /// Toggle visibility of the register side panel.
    ToggleRegisters,
    /// Toggle visibility of the memory viewer panel.
    ToggleMemory,
    /// Toggle visibility of the VRAM bottom panel.
    ToggleVram,
    /// Toggle visibility of the frame-profiler window.
    ToggleProfiler,
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

/// One row inside a category. Labels + values are `String` so we
/// can populate them from library entries at runtime (titles,
/// region tags, sizes). Static strings like "Run" / "Pause" also
/// fit the same shape at a small allocation cost -- the whole
/// category tree is rebuilt at most a few times a session.
struct MenuItem {
    label: String,
    action: MenuAction,
    /// Optional right-aligned subtitle -- used for region tags
    /// ("NTSC-U"), file sizes, and keyboard shortcut hints.
    value: Option<String>,
}

/// One Menu column. `icon_name` is a short tag used by tests /
/// diagnostics so we can identify a category without comparing
/// Unicode codepoints.
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
    /// Per-frame animated scroll position for the item list, in
    /// "rows of (ITEM_HEIGHT + ITEM_GAP)". A value of `N` means
    /// item `N` is drawn at the top of the visible strip.
    /// Eased toward the integer target computed from `item_index`
    /// each frame by the same `ANIM_SPEED` knob that drives the
    /// category slide, so navigating a long list produces a smooth
    /// scroll rather than a snap.
    scroll_y: f32,
    categories: Vec<Category>,
}

impl Default for MenuState {
    fn default() -> Self {
        Self::new()
    }
}

/// An entry passed into the Menu from the library layer -- minimal
/// subset of [`psoxide_settings::LibraryEntry`] the Menu needs to
/// render an item (title + id for dispatch + region/size as the
/// right-aligned value). Kept separate so the Menu module stays
/// decoupled from the settings crate's types (and from the GUI
/// from the tests' perspective).
#[derive(Debug, Clone)]
pub struct LibraryItem {
    /// Stable game ID (16-hex-char fingerprint). Payload of
    /// [`MenuAction::LaunchGame`] when the user confirms.
    pub id: String,
    /// Main label -- typically the PVD volume identifier or the
    /// file stem.
    pub title: String,
    /// Right-aligned subtitle, e.g. "NTSC-U · 602 MiB".
    pub subtitle: String,
}

impl MenuState {
    pub fn new() -> Self {
        Self::with_running(false)
    }

    pub fn with_running(running: bool) -> Self {
        // Boot categories with the library sections empty -- they
        // get filled by `set_library` once AppState loads the
        // cached entries. A fresh install sees placeholder "No
        // games found -- run Refresh library" rows.
        let categories = vec![
            build_games_category(&[]),
            build_examples_category(&[]),
            build_settings_category(),
            build_create_category(false),
            build_system_category(running),
            build_debug_category(),
            Category {
                name: "Quit",
                icon: icons::POWER,
                items: vec![MenuItem {
                    label: "Quit PSoXide".to_string(),
                    action: MenuAction::Quit,
                    value: Some("Esc ×2".to_string()),
                }],
            },
        ];

        Self {
            open: true,
            category_index: 0,
            item_index: 0,
            anim_x: 0.0,
            scroll_y: 0.0,
            categories,
        }
    }

    /// Rebuild the Games + Examples categories from a library
    /// snapshot. Call after load, after a rescan, and whenever the
    /// library changes. Existing selection is preserved when
    /// possible (same category + in-range item) and clamped to the
    /// new bounds otherwise.
    pub fn set_library(&mut self, games: &[LibraryItem], examples: &[LibraryItem]) {
        // Snapshot the current selection's category NAME so we can
        // re-resolve after rebuilding (indices may change).
        let current_cat_name = self
            .categories
            .get(self.category_index)
            .map(|c| c.name)
            .unwrap_or("");

        if let Some(games_cat) = self.categories.first_mut() {
            *games_cat = build_games_category(games);
        }
        if let Some(examples_cat) = self.categories.get_mut(1) {
            *examples_cat = build_examples_category(examples);
        }

        // Try to preserve the user's category if it still exists.
        if let Some(idx) = self
            .categories
            .iter()
            .position(|c| c.name == current_cat_name)
        {
            self.category_index = idx;
        } else {
            self.category_index = 0;
        }
        // Clamp item index to the new category bounds.
        let item_count = self.categories[self.category_index].items.len();
        if self.item_index >= item_count {
            self.item_index = item_count.saturating_sub(1);
        }
    }

    /// Rebuild categories with a fresh "Run"/"Pause" label. Called
    /// when `AppState.running` flips.
    pub fn sync_run_label(&mut self, running: bool) {
        if let Some(system) = self.categories.iter_mut().find(|c| c.name == "System") {
            if let Some(item) = system.items.first_mut() {
                item.label = if running {
                    "Pause".into()
                } else {
                    "Run".into()
                };
            }
        }
    }

    /// Update the System category's disc fast-boot value after the
    /// persisted setting changes.
    pub fn sync_fast_boot_label(&mut self, enabled: bool) {
        if let Some(system) = self.categories.iter_mut().find(|c| c.name == "System") {
            if let Some(item) = system
                .items
                .iter_mut()
                .find(|item| item.action == MenuAction::ToggleFastBoot)
            {
                item.value = Some(if enabled { "On" } else { "Off" }.into());
            }
        }
    }

    /// Update the Create category label for the current workspace.
    pub fn sync_editor_label(&mut self, editor_open: bool) {
        if let Some(create) = self.categories.iter_mut().find(|c| c.name == "Create") {
            if let Some(item) = create
                .items
                .iter_mut()
                .find(|item| item.action == MenuAction::ToggleEditorWorkspace)
            {
                item.label = if editor_open {
                    "Close editor workspace".into()
                } else {
                    "Open editor workspace".into()
                };
                item.value = Some(if editor_open { "Active" } else { "Studio" }.into());
            }
        }
    }

    /// Update the Settings category path summaries.
    pub fn sync_settings_paths(&mut self, bios: impl Into<String>, games: impl Into<String>) {
        let bios = bios.into();
        let games = games.into();
        if let Some(settings) = self.categories.iter_mut().find(|c| c.name == "Settings") {
            for item in &mut settings.items {
                match item.action {
                    MenuAction::ChooseBiosPath => item.value = Some(bios.clone()),
                    MenuAction::ChooseGamesPath => item.value = Some(games.clone()),
                    _ => {}
                }
            }
        }
    }

    /// Move selection to the category named `name`, if it exists.
    pub fn select_category(&mut self, name: &str) {
        if let Some(idx) = self.categories.iter().position(|c| c.name == name) {
            self.category_index = idx;
            self.item_index = 0;
            self.scroll_y = 0.0;
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
            // Snap the scroll so the new category's list shows from
            // the top -- matches the Menu convention. The target for
            // next frame will be 0.0 regardless; this avoids an
            // awkward animation from mid-list in the previous
            // category to top of the new one.
            self.scroll_y = 0.0;
        }
        if input.right && self.category_index + 1 < num_cats {
            self.category_index += 1;
            self.item_index = 0;
            self.scroll_y = 0.0;
        }

        let num_items = self.categories[self.category_index].items.len();
        if input.up && self.item_index > 0 {
            self.item_index -= 1;
        }
        if input.down && self.item_index + 1 < num_items {
            self.item_index += 1;
        }

        if input.confirm && num_items > 0 {
            return Some(
                self.categories[self.category_index].items[self.item_index]
                    .action
                    .clone(),
            );
        }

        if input.back {
            self.open = false;
        }
        None
    }

    /// Public reader for the currently-selected item's action --
    /// tests use it to assert the menu is populated correctly
    /// without driving input events.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn selected_action(&self) -> Option<&MenuAction> {
        self.categories
            .get(self.category_index)
            .and_then(|c| c.items.get(self.item_index))
            .map(|i| &i.action)
    }

    /// Current category name -- also exposed for test assertions.
    #[cfg(test)]
    pub fn current_category(&self) -> Option<&'static str> {
        self.categories.get(self.category_index).map(|c| c.name)
    }

    /// Draw the Menu overlay on a middle-layer painter. `dt` drives the
    /// slide animation.
    pub fn draw(&mut self, ctx: &egui::Context, dt: f32, warning: Option<&str>) {
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
        if let Some(warning) = warning {
            let banner_h = 34.0;
            let rect = Rect::from_min_size(screen.min, Vec2::new(sw, banner_h));
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(126, 24, 34));
            painter.text(
                Pos2::new(sw / 2.0, banner_h / 2.0),
                Align2::CENTER_CENTER,
                warning,
                FontId::proportional(15.0),
                egui::Color32::WHITE,
            );
        }

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
        let row_stride = ITEM_HEIGHT + ITEM_GAP;

        // How many full rows fit between `items_start_y` and the
        // bottom edge of the screen (with a small bottom margin so
        // the list doesn't butt against the edge).
        //
        // `max(1)` so a degenerate window height (tiny resize during
        // launch) still produces at least one visible row and avoids
        // a divide-by-zero in the visible-count math below.
        let bottom_margin = 16.0;
        let available_h = (sh - items_start_y - bottom_margin).max(row_stride);
        let visible_rows = (available_h / row_stride).floor().max(1.0) as usize;

        // Compute a TARGET scroll position that keeps the selected
        // item visible with a lead-in margin: once you hit row
        // `edge_margin` from the top or bottom, further navigation
        // scrolls the whole list instead of just moving the cursor.
        // Matches the console Menu behaviour.
        //
        // For very short lists (num_items ≤ visible_rows) the target
        // is 0 -- nothing to scroll.
        let num_items = cat.items.len();
        let edge_margin: usize = if visible_rows >= 5 { 2 } else { 1 };
        let target_scroll = if num_items <= visible_rows {
            0.0_f32
        } else {
            let max_scroll = (num_items - visible_rows) as f32;
            let sel = self.item_index as f32;
            let top_lead = edge_margin as f32;
            let bottom_lead = (visible_rows - 1 - edge_margin) as f32;
            // Ideal scroll keeps the selected row between
            // [scroll + top_lead, scroll + bottom_lead] inclusive.
            let t = if sel < self.scroll_y + top_lead {
                sel - top_lead
            } else if sel > self.scroll_y + bottom_lead {
                sel - bottom_lead
            } else {
                self.scroll_y
            };
            t.clamp(0.0, max_scroll)
        };

        // Ease `scroll_y` toward the target using the same
        // `ANIM_SPEED * dt` blend that drives the horizontal
        // category slide -- so navigation feels uniform between
        // axes. Snap when we're within a pixel of the target.
        self.scroll_y += (target_scroll - self.scroll_y) * ANIM_SPEED * dt;
        if (self.scroll_y - target_scroll).abs() * row_stride < 0.5 {
            self.scroll_y = target_scroll;
        }

        for (i, item) in cat.items.iter().enumerate() {
            let y = items_start_y + (i as f32 - self.scroll_y) * row_stride;
            let row_bottom = y + ITEM_HEIGHT;
            // Cull items entirely above the list region or below the
            // bottom margin. One row of overhang on each side so the
            // scroll animation doesn't "pop" items in/out at the
            // moment they fully arrive.
            if row_bottom < items_start_y - row_stride || y > sh - bottom_margin {
                continue;
            }
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
                item.label.clone(),
                label_font.clone(),
                label_color,
            );

            if let Some(val) = item.value.as_deref() {
                let val_color = if is_selected {
                    theme::MENU_TEXT_VALUE
                } else {
                    theme::MENU_TEXT_DIM
                };
                painter.text(
                    Pos2::new(items_x + ITEM_WIDTH - 12.0, y + ITEM_HEIGHT / 2.0),
                    Align2::RIGHT_CENTER,
                    val.to_string(),
                    value_font.clone(),
                    val_color,
                );
            }
        }

        // Scroll indicators: small triangles at the top/bottom edges
        // of the item strip when there's content outside the visible
        // window. Gives the user an affordance that "there's more
        // here" without waiting for them to hit the edge.
        let indicator_color = theme::MENU_TEXT_DIM;
        let has_above = self.scroll_y > 0.1;
        let has_below = (self.scroll_y + visible_rows as f32) < num_items as f32 - 0.1;
        if has_above {
            painter.text(
                Pos2::new(center_x, items_start_y - 6.0),
                Align2::CENTER_BOTTOM,
                "▲",
                FontId::proportional(10.0),
                indicator_color,
            );
        }
        if has_below {
            painter.text(
                Pos2::new(center_x, sh - bottom_margin + 4.0),
                Align2::CENTER_TOP,
                "▼",
                FontId::proportional(10.0),
                indicator_color,
            );
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

/// Emulator settings entry point.
fn build_settings_category() -> Category {
    Category {
        name: "Settings",
        icon: icons::HARD_DRIVE,
        items: vec![
            MenuItem {
                label: "Choose BIOS path".into(),
                action: MenuAction::ChooseBiosPath,
                value: Some("Missing".into()),
            },
            MenuItem {
                label: "Choose games path".into(),
                action: MenuAction::ChooseGamesPath,
                value: Some("Missing".into()),
            },
        ],
    }
}

/// Construct the Games category from a library snapshot. Empty
/// libraries get a helpful placeholder item so the user
/// understands the category isn't broken, just unpopulated.
fn build_games_category(games: &[LibraryItem]) -> Category {
    let mut items = Vec::with_capacity(games.len() + 1);
    if games.is_empty() {
        items.push(MenuItem {
            label: "No games found yet".into(),
            action: MenuAction::RescanLibrary,
            value: Some("Refresh".into()),
        });
    } else {
        for g in games {
            items.push(MenuItem {
                label: g.title.clone(),
                action: MenuAction::LaunchGame(g.id.clone()),
                value: if g.subtitle.is_empty() {
                    None
                } else {
                    Some(g.subtitle.clone())
                },
            });
        }
        // Always offer a rescan at the end of the Games list --
        // matches menu UX where "Refresh" sits below the
        // scrollable section.
        items.push(MenuItem {
            label: "Refresh library".into(),
            action: MenuAction::RescanLibrary,
            value: Some("↻".into()),
        });
    }
    Category {
        name: "Games",
        icon: icons::DISC,
        items,
    }
}

/// Construct the Examples category. Homebrew EXEs are shown here
/// so they don't compete with commercial games; the two lists
/// have different conventions for naming + running.
fn build_examples_category(examples: &[LibraryItem]) -> Category {
    let mut items = Vec::with_capacity(examples.len() + 1);
    if examples.is_empty() {
        items.push(MenuItem {
            label: "No homebrew EXEs found".into(),
            action: MenuAction::RescanLibrary,
            value: Some("Refresh".into()),
        });
    } else {
        for e in examples {
            items.push(MenuItem {
                label: e.title.clone(),
                action: MenuAction::LaunchGame(e.id.clone()),
                value: if e.subtitle.is_empty() {
                    None
                } else {
                    Some(e.subtitle.clone())
                },
            });
        }
        items.push(MenuItem {
            label: "Refresh library".into(),
            action: MenuAction::RescanLibrary,
            value: Some("↻".into()),
        });
    }
    Category {
        name: "Examples",
        icon: icons::FOLDER,
        items,
    }
}

/// Host-side creation tools.
fn build_create_category(editor_open: bool) -> Category {
    Category {
        name: "Create",
        icon: icons::FOLDER,
        items: vec![MenuItem {
            label: if editor_open {
                "Close editor workspace".into()
            } else {
                "Open editor workspace".into()
            },
            action: MenuAction::ToggleEditorWorkspace,
            value: Some(if editor_open { "Active" } else { "Studio" }.into()),
        }],
    }
}

/// The System category holds emulator-wide actions: run/pause,
/// step, reset. Renamed from "Game" -- on the PSX-style Menu, the
/// Game column holds games, System holds controls. Matches
/// menu convention.
fn build_system_category(running: bool) -> Category {
    let run_label = if running { "Pause" } else { "Run" };
    Category {
        name: "System",
        icon: icons::CPU,
        items: vec![
            MenuItem {
                label: run_label.into(),
                action: MenuAction::ToggleRun,
                value: Some("Space".into()),
            },
            MenuItem {
                label: "Step one instruction".into(),
                action: MenuAction::StepOne,
                value: None,
            },
            MenuItem {
                label: "Reset emulator".into(),
                action: MenuAction::Reset,
                value: None,
            },
            MenuItem {
                label: "Fast boot discs".into(),
                action: MenuAction::ToggleFastBoot,
                value: Some("On".into()),
            },
        ],
    }
}

/// Debug utilities -- panel toggles + VRAM test pattern.
fn build_debug_category() -> Category {
    Category {
        name: "Debug",
        icon: icons::BUG,
        items: vec![
            MenuItem {
                label: "Toggle registers panel".into(),
                action: MenuAction::ToggleRegisters,
                value: None,
            },
            MenuItem {
                label: "Toggle memory panel".into(),
                action: MenuAction::ToggleMemory,
                value: None,
            },
            MenuItem {
                label: "Toggle VRAM panel".into(),
                action: MenuAction::ToggleVram,
                value: None,
            },
            MenuItem {
                label: "Toggle profiler".into(),
                action: MenuAction::ToggleProfiler,
                value: None,
            },
            MenuItem {
                label: "Fill VRAM test pattern".into(),
                action: MenuAction::FillVramTestPattern,
                value: None,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_item(id: &str, title: &str, sub: &str) -> LibraryItem {
        LibraryItem {
            id: id.into(),
            title: title.into(),
            subtitle: sub.into(),
        }
    }

    #[test]
    fn fresh_state_has_expected_categories() {
        let s = MenuState::new();
        assert_eq!(s.categories.len(), 7);
        assert_eq!(s.categories[0].name, "Games");
        assert_eq!(s.categories[1].name, "Examples");
        assert_eq!(s.categories[2].name, "Settings");
        assert_eq!(s.categories[3].name, "Create");
        assert_eq!(s.categories[4].name, "System");
        assert_eq!(s.categories[5].name, "Debug");
        assert_eq!(s.categories[6].name, "Quit");
    }

    #[test]
    fn empty_library_shows_placeholder_that_triggers_rescan() {
        let s = MenuState::new();
        let first = s.categories[0].items.first().unwrap();
        assert_eq!(first.action, MenuAction::RescanLibrary);
    }

    #[test]
    fn set_library_populates_games_and_examples() {
        let mut s = MenuState::new();
        s.set_library(
            &[dummy_item("g1", "Crash", "NTSC-U · 600 MiB")],
            &[dummy_item("e1", "hello-tri", "EXE")],
        );
        assert_eq!(s.categories[0].items[0].label, "Crash");
        assert_eq!(
            s.categories[0].items[0].action,
            MenuAction::LaunchGame("g1".to_string())
        );
        // Refresh row is appended after the actual entries.
        assert_eq!(
            s.categories[0].items.last().unwrap().action,
            MenuAction::RescanLibrary
        );
        assert_eq!(s.categories[1].items[0].label, "hello-tri");
    }

    #[test]
    fn set_library_preserves_category_across_rebuild() {
        let mut s = MenuState::new();
        // Move to "System" category before rebuilding.
        s.category_index = 4;
        s.set_library(&[], &[]);
        assert_eq!(s.current_category(), Some("System"));
    }

    #[test]
    fn sync_run_label_flips_system_run_item() {
        let mut s = MenuState::new();
        assert_eq!(s.categories[4].items[0].label, "Run");
        s.sync_run_label(true);
        assert_eq!(s.categories[4].items[0].label, "Pause");
        s.sync_run_label(false);
        assert_eq!(s.categories[4].items[0].label, "Run");
    }

    #[test]
    fn sync_fast_boot_label_flips_system_value() {
        let mut s = MenuState::new();
        let fast_boot = s.categories[4]
            .items
            .iter()
            .find(|item| item.action == MenuAction::ToggleFastBoot)
            .unwrap();
        assert_eq!(fast_boot.value.as_deref(), Some("On"));

        s.sync_fast_boot_label(false);
        let fast_boot = s.categories[4]
            .items
            .iter()
            .find(|item| item.action == MenuAction::ToggleFastBoot)
            .unwrap();
        assert_eq!(fast_boot.value.as_deref(), Some("Off"));

        s.sync_fast_boot_label(true);
        let fast_boot = s.categories[4]
            .items
            .iter()
            .find(|item| item.action == MenuAction::ToggleFastBoot)
            .unwrap();
        assert_eq!(fast_boot.value.as_deref(), Some("On"));
    }

    #[test]
    fn navigation_stays_in_bounds() {
        let mut s = MenuState::new();
        // Populate some games so there's something to navigate.
        s.set_library(
            &[
                dummy_item("a", "A", ""),
                dummy_item("b", "B", ""),
                dummy_item("c", "C", ""),
            ],
            &[],
        );
        let right = MenuInput {
            right: true,
            ..Default::default()
        };
        s.update(&right); // Examples
        s.update(&right); // Settings
        s.update(&right); // Create
        s.update(&right); // System
        s.update(&right); // Debug
        s.update(&right); // Quit
        s.update(&right); // past end -- should clamp
        assert_eq!(s.current_category(), Some("Quit"));
        let left = MenuInput {
            left: true,
            ..Default::default()
        };
        for _ in 0..10 {
            s.update(&left);
        }
        assert_eq!(s.current_category(), Some("Games"));
    }

    #[test]
    fn select_category_moves_to_settings() {
        let mut s = MenuState::new();
        s.select_category("Settings");
        assert_eq!(s.current_category(), Some("Settings"));
        assert_eq!(s.selected_action(), Some(&MenuAction::ChooseBiosPath));
    }

    #[test]
    fn sync_settings_paths_updates_menu_values() {
        let mut s = MenuState::new();
        s.sync_settings_paths("SCPH1001.BIN", "ps1 games");
        assert_eq!(
            s.categories[2].items[0].value.as_deref(),
            Some("SCPH1001.BIN")
        );
        assert_eq!(s.categories[2].items[1].value.as_deref(), Some("ps1 games"));
    }
}
