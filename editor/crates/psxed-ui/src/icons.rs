//! Lucide icon helpers for the embedded editor UI.
//!
//! The frontend registers `lucide.ttf` in the egui context at startup. This
//! module keeps the editor crate independent while sharing the same icon
//! codepoint style used by Bonnie-32 and the emulator shell.

#![allow(dead_code)]

use egui::{FontFamily, FontId, RichText};

pub const AUDIO_LINES: char = '\u{e2b2}';
pub const BLEND: char = '\u{e59c}';
pub const BOX: char = '\u{e061}';
pub const BRICK_WALL: char = '\u{e581}';
pub const CHEVRON_DOWN: char = '\u{e06d}';
pub const CIRCLE_DOT: char = '\u{e345}';
pub const COPY: char = '\u{e08b}';
pub const EYE: char = '\u{e0ba}';
pub const FILE: char = '\u{e0c5}';
pub const FILE_PLUS: char = '\u{e0c9}';
pub const FOCUS: char = '\u{e29e}';
pub const FOLDER: char = '\u{e0d8}';
pub const GRID: char = '\u{e0e9}';
pub const HOUSE: char = '\u{e0f5}';
pub const LAYERS: char = '\u{e529}';
pub const MAP_PIN: char = '\u{e111}';
pub const MOVE: char = '\u{e121}';
pub const PALETTE: char = '\u{e12f}';
pub const PLAY: char = '\u{e13c}';
pub const PLUS: char = '\u{e13d}';
pub const POINTER: char = '\u{e1e8}';
pub const ROTATE_3D: char = '\u{e2ea}';
pub const ROTATE_CCW: char = '\u{e144}';
pub const SAVE: char = '\u{e14d}';
pub const SCALE_3D: char = '\u{e2eb}';
pub const SCAN: char = '\u{e257}';
pub const SUN: char = '\u{e178}';
pub const TERMINAL: char = '\u{e183}';
pub const TRASH: char = '\u{e18d}';
pub const WAYPOINT: char = '\u{e546}';

pub fn font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("lucide".into()))
}

pub fn text(ch: char, size: f32) -> RichText {
    RichText::new(ch.to_string()).font(font(size))
}

pub fn label(ch: char, label: &str) -> String {
    format!("{ch}  {label}")
}
