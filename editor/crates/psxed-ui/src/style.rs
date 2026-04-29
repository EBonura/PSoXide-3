//! Shared editor visual styling.

use egui::{Color32, CornerRadius, Frame, Margin, RichText, Stroke, Vec2};

use crate::icons;

pub(crate) const DEFAULT_VIEWPORT_ZOOM: f32 = 96.0;
pub(crate) const MIN_VIEWPORT_ZOOM: f32 = 24.0;
pub(crate) const MAX_VIEWPORT_ZOOM: f32 = 220.0;

pub(crate) const STUDIO_BG: Color32 = Color32::from_rgb(12, 16, 21);
pub(crate) const STUDIO_TOP_BAR: Color32 = Color32::from_rgb(18, 22, 28);
pub(crate) const STUDIO_PANEL: Color32 = Color32::from_rgb(17, 22, 28);
pub(crate) const STUDIO_PANEL_DARK: Color32 = Color32::from_rgb(13, 17, 22);
pub(crate) const STUDIO_INPUT: Color32 = Color32::from_rgb(11, 15, 20);
pub(crate) const STUDIO_VIEWPORT: Color32 = Color32::from_rgb(12, 24, 31);
pub(crate) const STUDIO_BORDER: Color32 = Color32::from_rgb(41, 51, 63);
pub(crate) const STUDIO_BORDER_DARK: Color32 = Color32::from_rgb(28, 36, 45);
pub(crate) const STUDIO_TEXT: Color32 = Color32::from_rgb(220, 229, 238);
pub(crate) const STUDIO_TEXT_WEAK: Color32 = Color32::from_rgb(142, 154, 168);
pub(crate) const STUDIO_ACCENT: Color32 = Color32::from_rgb(45, 177, 207);
pub(crate) const STUDIO_ACCENT_DIM: Color32 = Color32::from_rgb(17, 82, 101);
pub(crate) const STUDIO_GOLD: Color32 = Color32::from_rgb(238, 197, 119);
pub(crate) const STUDIO_ROOM_FLOOR: Color32 = Color32::from_rgb(119, 132, 143);
pub(crate) const STUDIO_ROOM_WALL: Color32 = Color32::from_rgb(126, 73, 43);

pub(crate) fn apply_studio_visuals(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut(|style| {
        style.spacing.item_spacing = Vec2::new(6.0, 4.0);
        style.spacing.button_padding = Vec2::new(8.0, 3.0);
        style.spacing.interact_size = Vec2::new(30.0, 22.0);
        style.spacing.window_margin = Margin::same(6);
        style.spacing.menu_margin = Margin::symmetric(8, 5);
        style.spacing.indent = 16.0;
        style.visuals = studio_visuals();
    });
}

pub(crate) fn studio_visuals() -> egui::Visuals {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(STUDIO_TEXT);
    visuals.panel_fill = STUDIO_PANEL_DARK;
    visuals.window_fill = STUDIO_PANEL;
    visuals.window_stroke = Stroke::new(1.0, STUDIO_BORDER);
    visuals.window_corner_radius = CornerRadius::same(3);
    visuals.menu_corner_radius = CornerRadius::same(3);
    visuals.faint_bg_color = Color32::from_rgb(20, 27, 35);
    visuals.extreme_bg_color = STUDIO_INPUT;
    visuals.code_bg_color = Color32::from_rgb(16, 21, 27);
    visuals.hyperlink_color = STUDIO_ACCENT;
    visuals.selection.bg_fill = STUDIO_ACCENT_DIM;
    visuals.selection.stroke = Stroke::new(1.0, STUDIO_ACCENT);
    visuals.button_frame = true;
    visuals.collapsing_header_frame = false;
    visuals.indent_has_left_vline = true;
    visuals.slider_trailing_fill = true;

    visuals.widgets.noninteractive.bg_fill = STUDIO_PANEL;
    visuals.widgets.noninteractive.weak_bg_fill = STUDIO_PANEL_DARK;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, STUDIO_BORDER);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, STUDIO_TEXT);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(3);

    visuals.widgets.inactive.bg_fill = Color32::from_rgb(20, 28, 36);
    visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(18, 26, 34);
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, STUDIO_BORDER);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, STUDIO_TEXT);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(3);

    visuals.widgets.hovered.bg_fill = Color32::from_rgb(28, 46, 57);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(23, 38, 48);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(65, 95, 112));
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(3);

    visuals.widgets.active.bg_fill = STUDIO_ACCENT_DIM;
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(18, 92, 113);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, STUDIO_ACCENT);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.active.corner_radius = CornerRadius::same(3);

    visuals.widgets.open.bg_fill = Color32::from_rgb(20, 30, 38);
    visuals.widgets.open.weak_bg_fill = Color32::from_rgb(20, 30, 38);
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, STUDIO_BORDER);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, STUDIO_TEXT);
    visuals.widgets.open.corner_radius = CornerRadius::same(3);
    visuals
}

pub(crate) fn top_bar_frame() -> Frame {
    Frame::new()
        .fill(STUDIO_TOP_BAR)
        .stroke(Stroke::new(1.0, STUDIO_BORDER_DARK))
        .inner_margin(Margin::symmetric(10, 4))
}

pub(crate) fn dock_frame() -> Frame {
    Frame::new()
        .fill(STUDIO_PANEL_DARK)
        .stroke(Stroke::new(1.0, STUDIO_BORDER))
        .inner_margin(Margin::symmetric(6, 6))
}

pub(crate) fn section_frame() -> Frame {
    Frame::new()
        .fill(STUDIO_PANEL)
        .stroke(Stroke::new(1.0, STUDIO_BORDER))
        .corner_radius(CornerRadius::same(3))
        .inner_margin(Margin::symmetric(8, 7))
}

pub(crate) fn viewport_frame() -> Frame {
    Frame::new()
        .fill(STUDIO_BG)
        .stroke(Stroke::new(1.0, STUDIO_BORDER))
        .inner_margin(Margin::same(4))
}

pub(crate) fn panel_heading(ui: &mut egui::Ui, icon: char, label: &str) {
    ui.horizontal(|ui| {
        ui.label(icons::text(icon, 15.0).color(STUDIO_ACCENT));
        ui.label(RichText::new(label).strong().color(STUDIO_TEXT));
    });
}
