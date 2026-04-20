//! Screen-shake counter.
//!
//! The cheapest possible "the world reacted" feedback. A single
//! frame counter; `tick()` decrements it and returns a small
//! `(dx, dy)` offset the caller adds to their vertex positions.
//!
//! Triangle-wave pattern — sign flips each frame so it reads as a
//! judder rather than a smooth drift. Magnitude decays linearly
//! with the counter, so the last frames are subtle.
//!
//! # Example
//!
//! ```ignore
//! let mut shake = ShakeState::new();
//! // On impact:
//! shake.trigger(6);
//! // Every frame, folded into rendering:
//! let (dx, dy) = shake.tick();
//! draw_game_world_at_offset(dx, dy);
//! ```

/// Frame-countdown that emits a judder offset.
#[derive(Copy, Clone, Debug, Default)]
pub struct ShakeState {
    /// Frames of shake remaining. `0` = idle.
    pub frames: u8,
}

impl ShakeState {
    /// Idle instance.
    pub const fn new() -> Self {
        Self { frames: 0 }
    }

    /// Start (or retrigger) a shake. If a bigger shake is already
    /// active, it stays — smaller triggers don't interrupt.
    pub fn trigger(&mut self, frames: u8) {
        self.frames = self.frames.max(frames);
    }

    /// Advance one frame. Returns `(dx, dy)` in pixels. Safe to
    /// call every frame even when idle — returns `(0, 0)` then.
    pub fn tick(&mut self) -> (i16, i16) {
        if self.frames == 0 {
            return (0, 0);
        }
        let s = self.frames as i16;
        // Sign flips with parity so the shake rattles instead of
        // drifting. Magnitudes decrease linearly with `frames`.
        let sign = if s & 1 == 0 { 1 } else { -1 };
        let offset = ((s / 2) * sign, ((s + 1) / 2) * -sign);
        self.frames -= 1;
        offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_is_zero() {
        let mut s = ShakeState::new();
        assert_eq!(s.tick(), (0, 0));
        assert_eq!(s.tick(), (0, 0));
    }

    #[test]
    fn trigger_then_decay_to_zero() {
        let mut s = ShakeState::new();
        s.trigger(4);
        for _ in 0..4 {
            let _ = s.tick();
        }
        assert_eq!(s.tick(), (0, 0));
        assert_eq!(s.frames, 0);
    }

    #[test]
    fn trigger_doesnt_shrink() {
        let mut s = ShakeState::new();
        s.trigger(10);
        s.trigger(3); // smaller trigger shouldn't reset
        assert_eq!(s.frames, 10);
    }
}
