//! UI theme -- charcoal background, teal accent, VT323 monospace, Lucide icons.
//!
//! Ported from PSoXide-2's theme, itself derived from the Bonnie-32 editor.
//! Same palette and typography across both debug panels and the Menu overlay
//! so the two reads as one UI, not two.

// Palette + helpers are a pool the UI layer draws from; individual items
// come online one panel at a time.
#![allow(dead_code)]

use egui::{
    epaint::CornerRadius,
    style::{TextStyle, WidgetVisuals},
    Color32, FontFamily, FontId, Stroke, Visuals,
};

// --- Base palette ---

/// Main backdrop.
pub const BG: Color32 = Color32::from_rgb(20, 20, 24);
/// Panel fill (side panels, bottom panels).
pub const PANEL_BG: Color32 = Color32::from_rgb(26, 26, 30);
/// Flush header strip at the top of viz panes.
pub const HEADER_BG: Color32 = Color32::from_rgb(18, 18, 22);
/// Inner content area within a framed viz pane.
pub const CONTENT_BG: Color32 = Color32::from_rgb(22, 22, 26);

pub const TEXT: Color32 = Color32::from_rgb(204, 204, 217);
pub const TEXT_DIM: Color32 = Color32::from_rgb(102, 102, 115);

pub const WIDGET_BG: Color32 = Color32::from_rgb(38, 38, 46);
pub const WIDGET_HOVER: Color32 = Color32::from_rgb(52, 52, 62);

pub const ACCENT: Color32 = Color32::from_rgb(0, 180, 180);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(0, 210, 210);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0, 120, 120);

pub const SEPARATOR: Color32 = Color32::from_rgb(45, 45, 52);
pub const SECTION_BG: Color32 = Color32::from_rgb(30, 30, 35);

// --- Menu-specific palette ---

/// Menu backdrop overlay (dims the game below).
pub const MENU_BACKDROP: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 128);
/// Menu accent (selected category icon + selection bar).
pub const MENU_ACCENT: Color32 = Color32::from_rgb(0, 191, 230);
pub const MENU_TEXT_BRIGHT: Color32 = Color32::from_rgb(230, 230, 230);
pub const MENU_TEXT_DIM: Color32 = Color32::from_rgb(128, 128, 140);
pub const MENU_TEXT_VALUE: Color32 = Color32::from_rgb(179, 179, 191);
pub const MENU_HINT: Color32 = Color32::from_rgb(102, 102, 115);
pub const MENU_ITEM_BG: Color32 = Color32::from_rgba_premultiplied(25, 25, 30, 234);
pub const MENU_ITEM_SEL: Color32 = Color32::from_rgba_premultiplied(0, 38, 51, 242);

// --- Font sizes ---

pub const FONT_SIZE_UI: f32 = 13.0;
pub const FONT_SIZE_SMALL: f32 = 11.0;
pub const FONT_SIZE_HEADING: f32 = 15.0;
pub const FONT_SIZE_MONO: f32 = 16.0;

/// Apply theme + embedded fonts to the egui context. Call once at startup.
pub fn apply(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    const LUCIDE_TTF: &[u8] = include_bytes!("../assets/fonts/lucide.ttf");
    fonts.font_data.insert(
        "lucide".to_owned(),
        egui::FontData::from_static(LUCIDE_TTF).into(),
    );
    fonts
        .families
        .entry(FontFamily::Name("lucide".into()))
        .or_default()
        .push("lucide".to_owned());
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("lucide".to_owned());
    }

    const VT323_TTF: &[u8] = include_bytes!("../assets/fonts/VT323-Regular.ttf");
    fonts.font_data.insert(
        "VT323".to_owned(),
        egui::FontData::from_static(VT323_TTF).into(),
    );
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .insert(0, "VT323".to_owned());

    ctx.set_fonts(fonts);

    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (
            TextStyle::Small,
            FontId::new(FONT_SIZE_SMALL, FontFamily::Proportional),
        ),
        (
            TextStyle::Body,
            FontId::new(FONT_SIZE_UI, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(FONT_SIZE_MONO, FontFamily::Monospace),
        ),
        (
            TextStyle::Button,
            FontId::new(FONT_SIZE_UI, FontFamily::Proportional),
        ),
        (
            TextStyle::Heading,
            FontId::new(FONT_SIZE_HEADING, FontFamily::Proportional),
        ),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(8.0, 5.0);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    style.spacing.indent = 14.0;
    ctx.set_style(style);
    ctx.set_visuals(make_visuals());
}

fn make_visuals() -> Visuals {
    let mut v = Visuals::dark();
    v.override_text_color = Some(TEXT);
    v.window_fill = PANEL_BG;
    v.panel_fill = PANEL_BG;
    v.faint_bg_color = CONTENT_BG;
    v.extreme_bg_color = BG;
    v.window_corner_radius = CornerRadius::same(4);
    v.window_stroke = Stroke::new(1.0, SEPARATOR);
    v.popup_shadow = egui::Shadow::NONE;

    let base = WidgetVisuals {
        bg_fill: WIDGET_BG,
        weak_bg_fill: Color32::from_rgb(32, 32, 38),
        bg_stroke: Stroke::new(1.0, Color32::from_rgb(52, 52, 62)),
        corner_radius: CornerRadius::same(3),
        fg_stroke: Stroke::new(1.0, TEXT),
        expansion: 0.0,
    };
    v.widgets.noninteractive = WidgetVisuals {
        bg_fill: PANEL_BG,
        weak_bg_fill: PANEL_BG,
        bg_stroke: Stroke::new(1.0, SEPARATOR),
        fg_stroke: Stroke::new(1.0, TEXT_DIM),
        corner_radius: CornerRadius::same(3),
        expansion: 0.0,
    };
    v.widgets.inactive = base;
    let mut hov = base;
    hov.bg_fill = WIDGET_HOVER;
    hov.bg_stroke = Stroke::new(1.0, ACCENT_DIM);
    v.widgets.hovered = hov;
    let mut act = base;
    act.bg_fill = ACCENT_DIM;
    act.fg_stroke = Stroke::new(1.5, Color32::WHITE);
    v.widgets.active = act;
    let mut open = base;
    open.bg_fill = Color32::from_rgb(0, 70, 70);
    open.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    v.widgets.open = open;

    v.selection.bg_fill = Color32::from_rgba_premultiplied(0, 180, 180, 60);
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = Color32::from_rgb(220, 180, 60);
    v.error_fg_color = Color32::from_rgb(220, 80, 80);
    v
}

/// Framed section with an accent-colored title, for left-panel groups.
pub fn section<R>(
    ui: &mut egui::Ui,
    title: &str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let resp = egui::Frame::NONE
        .fill(SECTION_BG)
        .corner_radius(CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(10, 8))
        .outer_margin(egui::Margin {
            left: 0,
            right: 0,
            top: 0,
            bottom: 8,
        })
        .show(ui, |ui| {
            ui.label(egui::RichText::new(title).color(ACCENT).size(FONT_SIZE_UI));
            ui.add_space(4.0);
            add_contents(ui)
        });
    resp.inner
}

/// Compact flush header strip for a viz pane.
pub fn pane_header(ui: &mut egui::Ui, title: &str) {
    egui::Frame::NONE
        .fill(HEADER_BG)
        .inner_margin(egui::Margin::symmetric(8, 4))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(title)
                    .color(ACCENT)
                    .size(FONT_SIZE_SMALL),
            );
        });
}

/// Framed container for a viz pane. Dark content bg, subtle separator, flush header.
pub fn viz_frame<R>(
    ui: &mut egui::Ui,
    title: &str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let resp = egui::Frame::NONE
        .fill(CONTENT_BG)
        .stroke(Stroke::new(1.0, SEPARATOR))
        .corner_radius(CornerRadius::same(3))
        .inner_margin(egui::Margin::ZERO)
        .outer_margin(egui::Margin::same(4))
        .show(ui, |ui| {
            if !title.is_empty() {
                pane_header(ui, title);
            }
            let inner_resp = egui::Frame::NONE
                .fill(BG)
                .inner_margin(egui::Margin::same(4))
                .show(ui, add_contents);
            inner_resp.inner
        });
    resp.inner
}
