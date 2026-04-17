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

use emulator_core::{Bus, Cpu};

const HUD_HEIGHT: f32 = 34.0;
const BG: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 102);
const TEXT: Color32 = Color32::from_rgb(153, 153, 166);

/// Rolling-window tracker for frame time + CPU instruction rate.
/// Keeps ~1 s worth of samples (at 60 Hz that's 120 frames).
pub struct HudState {
    dt_samples: std::collections::VecDeque<f32>,
    tick_samples: std::collections::VecDeque<u64>,
    prev_tick: u64,
}

impl Default for HudState {
    fn default() -> Self {
        Self {
            dt_samples: std::collections::VecDeque::with_capacity(120),
            tick_samples: std::collections::VecDeque::with_capacity(120),
            prev_tick: 0,
        }
    }
}

impl HudState {
    /// Record this frame's wall-time dt and the CPU's current retired-
    /// instruction tick. The per-frame tick delta is computed here so
    /// callers don't need to remember the previous tick.
    pub fn update(&mut self, dt: f32, current_tick: u64) {
        if self.dt_samples.len() >= 120 {
            self.dt_samples.pop_front();
        }
        self.dt_samples.push_back(dt);

        let delta = current_tick.saturating_sub(self.prev_tick);
        self.prev_tick = current_tick;
        if self.tick_samples.len() >= 120 {
            self.tick_samples.pop_front();
        }
        self.tick_samples.push_back(delta);
    }

    fn average_dt(&self) -> f32 {
        if self.dt_samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.dt_samples.iter().sum();
        sum / self.dt_samples.len() as f32
    }

    fn fps(&self) -> f32 {
        let avg = self.average_dt();
        if avg > 0.0 {
            1.0 / avg
        } else {
            0.0
        }
    }

    /// CPU instructions per second, averaged over the rolling window.
    /// Returns 0 when paused or when the window has no data yet.
    fn ips(&self) -> f32 {
        let total_ticks: u64 = self.tick_samples.iter().sum();
        let total_dt: f32 = self.dt_samples.iter().sum();
        if total_dt > 0.0 {
            total_ticks as f32 / total_dt
        } else {
            0.0
        }
    }
}

/// Paint the HUD bar.
pub fn draw(ctx: &egui::Context, hud: &HudState, cpu: &Cpu, bus: Option<&Bus>, running: bool) {
    let screen = ctx.screen_rect();
    let rect = Rect::from_min_size(
        Pos2::new(screen.left(), screen.bottom() - HUD_HEIGHT),
        Vec2::new(screen.width(), HUD_HEIGHT),
    );

    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("hud"),
    ));

    painter.rect_filled(rect, 0.0, BG);

    // Left-aligned status indicator; right-aligned perf metrics.
    let left = screen.left() + 12.0;
    let right = screen.right() - 12.0;
    let mid_y = rect.top() + HUD_HEIGHT * 0.5;
    let line1_y = mid_y - 8.0;
    let line2_y = mid_y + 8.0;

    let fps = hud.fps();
    let dt_ms = hud.average_dt() * 1000.0;
    let mips = hud.ips() / 1_000_000.0;
    let (status_text, status_color) = if running {
        ("● RUNNING", Color32::from_rgb(80, 200, 120))
    } else {
        ("❚❚ PAUSED", TEXT)
    };

    painter.text(
        Pos2::new(left, mid_y),
        Align2::LEFT_CENTER,
        status_text,
        FontId::proportional(12.0),
        status_color,
    );

    painter.text(
        Pos2::new(right, line1_y),
        Align2::RIGHT_CENTER,
        format!("FPS {fps:5.1}   {mips:5.1} Mips"),
        FontId::proportional(12.0),
        TEXT,
    );
    // Second line: frame timing + instruction tick + (when a Bus is
    // present) the bus cycle counter. In phase 4a cycles == tick; when
    // per-opcode cycle costs land they'll start to diverge, and this
    // line is where that shows up.
    let tail = match bus {
        Some(b) => format!(
            "dt {dt_ms:5.2} ms   tick {}   cyc {}",
            cpu.tick(),
            b.cycles()
        ),
        None => format!("dt {dt_ms:5.2} ms   tick {}", cpu.tick()),
    };
    painter.text(
        Pos2::new(right, line2_y),
        Align2::RIGHT_CENTER,
        tail,
        FontId::proportional(11.0),
        TEXT,
    );
}
