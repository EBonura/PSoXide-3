//! Lucide icon codepoints (subset used by PSoXide UI).
//!
//! Keep this file small — add codepoints only when a panel actually uses
//! them. The full Lucide set is in `assets/fonts/lucide.ttf`.

#![allow(dead_code)]

use egui::{FontFamily, FontId, RichText};

pub const PLAY: char = '\u{e13c}';
pub const PAUSE: char = '\u{e12e}';
pub const SQUARE: char = '\u{e167}';
pub const SKIP_FORWARD: char = '\u{e160}';

pub const BUG: char = '\u{e1d0}';
pub const TERMINAL: char = '\u{e183}';
pub const HASH: char = '\u{e0eb}';

pub const MONITOR: char = '\u{e11d}';
pub const CPU: char = '\u{e1cf}';
pub const LAYERS: char = '\u{e529}';
pub const HARD_DRIVE: char = '\u{e0ed}';
pub const GAMEPAD_2: char = '\u{e0df}';
pub const GRID: char = '\u{e0e9}';
pub const SAVE: char = '\u{e14d}';

pub const EYE: char = '\u{e0ba}';
pub const EYE_OFF: char = '\u{e0bb}';

/// Lucide FontId at a given size.
pub fn font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("lucide".into()))
}

/// Icon as RichText at a given size.
pub fn text(ch: char, size: f32) -> RichText {
    RichText::new(ch.to_string()).font(font(size))
}
