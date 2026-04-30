//! Lucide icon codepoints (subset used by PSoXide UI).
//!
//! Keep this file small -- add codepoints only when a panel actually uses
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
pub const MAXIMIZE: char = '\u{e112}';
pub const MINIMIZE: char = '\u{e11a}';
pub const CPU: char = '\u{e1cf}';
pub const LAYERS: char = '\u{e529}';
pub const HARD_DRIVE: char = '\u{e0ed}';
pub const GAMEPAD_2: char = '\u{e0df}';
pub const GRID: char = '\u{e0e9}';
pub const SAVE: char = '\u{e14d}';
pub const VOLUME_1: char = '\u{e1aa}';
pub const VOLUME_2: char = '\u{e1ab}';
pub const VOLUME_X: char = '\u{e1ac}';

pub const EYE: char = '\u{e0ba}';
pub const EYE_OFF: char = '\u{e0bb}';

/// Power-off / quit icon -- used for the rightmost Menu category
/// so "close the app" has its own place instead of hiding inside
/// Debug.
pub const POWER: char = '\u{e13e}';
/// Folder icon -- Examples category badge (SDK homebrew feels
/// more like "scroll through a folder" than "spin a disc").
pub const FOLDER: char = '\u{e0d8}';
/// Refresh/rotate icon -- rescan-library action.
pub const ROTATE_CCW: char = '\u{e144}';
/// Disc icon -- Games category badge.
pub const DISC: char = '\u{e528}';

/// Lucide FontId at a given size.
pub fn font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("lucide".into()))
}

/// Icon as RichText at a given size.
pub fn text(ch: char, size: f32) -> RichText {
    RichText::new(ch.to_string()).font(font(size))
}
