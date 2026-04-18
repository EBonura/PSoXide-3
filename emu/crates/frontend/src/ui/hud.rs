//! HUD metric tracker.
//!
//! Keeps rolling windows for frame time and CPU instruction rate so
//! the toolbar can render a stable-looking FPS / MIPS readout without
//! jittering every frame. The bar itself is rendered by
//! [`crate::ui::toolbar`]; this module is the pure data pipe.
//!
//! Earlier this module also owned a bottom-of-screen overlay. That
//! overlay duplicated what the toolbar now shows, so it was removed
//! when the controls + metrics merged into the top strip.

/// Rolling-window tracker for frame time + CPU instruction rate.
/// Keeps ~1 s worth of samples (at 60 Hz that's 120 frames).
pub struct HudState {
    dt_samples: std::collections::VecDeque<f32>,
    tick_samples: std::collections::VecDeque<u64>,
    prev_tick: u64,
    /// Most recent audio-ring-buffer depth, in stereo samples.
    /// Updated by [`HudState::set_audio_queue_len`] from the
    /// shell each frame right after the SPU pump drain. Shown in
    /// the toolbar so you can see audio backlog — low values
    /// (< 100) mean we're starving the audio thread; high values
    /// (> 10k) mean the emu is outrunning real-time playback.
    audio_queue_len: usize,
}

impl Default for HudState {
    fn default() -> Self {
        Self {
            dt_samples: std::collections::VecDeque::with_capacity(120),
            tick_samples: std::collections::VecDeque::with_capacity(120),
            prev_tick: 0,
            audio_queue_len: 0,
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

    /// Average frame time over the rolling window, in seconds.
    pub fn average_dt(&self) -> f32 {
        if self.dt_samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.dt_samples.iter().sum();
        sum / self.dt_samples.len() as f32
    }

    /// Frames per second, averaged over the rolling window.
    pub fn fps(&self) -> f32 {
        let avg = self.average_dt();
        if avg > 0.0 {
            1.0 / avg
        } else {
            0.0
        }
    }

    /// CPU instructions per second, averaged over the rolling window.
    /// Returns 0 when paused or when the window has no data yet.
    pub fn ips(&self) -> f32 {
        let total_ticks: u64 = self.tick_samples.iter().sum();
        let total_dt: f32 = self.dt_samples.iter().sum();
        if total_dt > 0.0 {
            total_ticks as f32 / total_dt
        } else {
            0.0
        }
    }

    /// Snapshot the current audio ring depth. Called by the shell
    /// after it pushes freshly-drained SPU samples into cpal's
    /// queue; surfaced in the toolbar as an "AUDIO" metric.
    pub fn set_audio_queue_len(&mut self, len: usize) {
        self.audio_queue_len = len;
    }

    /// Current cached audio backlog (stereo samples).
    pub fn audio_queue_len(&self) -> usize {
        self.audio_queue_len
    }
}
