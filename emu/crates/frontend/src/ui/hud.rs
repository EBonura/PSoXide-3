//! HUD bar overlay — ported from psoxide-1.
//!
//! Thin strip at the bottom of the screen showing frame-rate and
//! frame-time at a glance. Rendered via `egui::Painter` on the
//! Foreground layer so it sits above the Menu and panels.
//!
//! Two-line format (right-aligned):
//! - top line: FPS (rolling average)
//! - bottom line: frame time, CPU tick counter
//!
//! The original HUD has many more columns (video scale, filter, pad,
//! draw-log timings, upload-VRAM timings, …); those will accrete as
//! the corresponding subsystems land. The palette and position are
//! faithful to the psoxide-1 look.

use egui::{Align2, Color32, FontId, Pos2, Rect, Vec2};

use emulator_core::Cpu;

const HUD_HEIGHT: f32 = 34.0;
const BG: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 102);
const TEXT: Color32 = Color32::from_rgb(153, 153, 166);

/// Rolling-window frame-rate tracker. Keeps the last ~1 s of frame
/// times and reports the average.
pub struct HudState {
    samples: std::collections::VecDeque<f32>,
}

impl Default for HudState {
    fn default() -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(120),
        }
    }
}

impl HudState {
    /// Record the elapsed seconds for one frame.
    pub fn push(&mut self, dt: f32) {
        if self.samples.len() >= 120 {
            self.samples.pop_front();
        }
        self.samples.push_back(dt);
    }

    fn average_dt(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().sum();
        sum / self.samples.len() as f32
    }

    fn fps(&self) -> f32 {
        let avg = self.average_dt();
        if avg > 0.0 {
            1.0 / avg
        } else {
            0.0
        }
    }
}

/// Paint the HUD bar.
pub fn draw(ctx: &egui::Context, hud: &HudState, cpu: &Cpu) {
    let screen = ctx.screen_rect();
    let rect = Rect::from_min_size(
        Pos2::new(screen.left(), screen.bottom() - HUD_HEIGHT),
        Vec2::new(screen.width(), HUD_HEIGHT),
    );

    let painter =
        ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("hud")));

    painter.rect_filled(rect, 0.0, BG);

    let right = screen.right() - 12.0;
    let mid_y = rect.top() + HUD_HEIGHT * 0.5;
    let line1_y = mid_y - 8.0;
    let line2_y = mid_y + 8.0;

    let fps = hud.fps();
    let dt_ms = hud.average_dt() * 1000.0;

    painter.text(
        Pos2::new(right, line1_y),
        Align2::RIGHT_CENTER,
        format!("FPS {fps:5.1}"),
        FontId::proportional(12.0),
        TEXT,
    );
    painter.text(
        Pos2::new(right, line2_y),
        Align2::RIGHT_CENTER,
        format!("dt {dt_ms:5.2} ms   tick {}", cpu.tick()),
        FontId::proportional(11.0),
        TEXT,
    );
}
