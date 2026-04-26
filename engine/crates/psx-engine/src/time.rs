//! Engine timekeeping.
//!
//! `ctx.frame` counts rendered app iterations for legacy/simple uses.
//! `EngineTime` is the PS1-time view: it advances from elapsed VBlanks,
//! so simulation and animation can stay tied to display time even when
//! a heavy render path drops below one rendered frame per VBlank.

/// Per-frame engine timing snapshot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EngineTime {
    rendered_frame: u32,
    elapsed_vblanks: u32,
    delta_vblanks: u16,
    video_hz: u16,
}

impl EngineTime {
    /// Initial timing snapshot before the first app tick.
    pub const fn start(video_hz: u16) -> Self {
        Self {
            rendered_frame: 0,
            elapsed_vblanks: 0,
            delta_vblanks: 1,
            video_hz,
        }
    }

    /// Rendered app frame index, matching `ctx.frame`.
    #[inline]
    pub const fn rendered_frame(self) -> u32 {
        self.rendered_frame
    }

    /// Total VBlank ticks observed since the engine clock started.
    #[inline]
    pub const fn elapsed_vblanks(self) -> u32 {
        self.elapsed_vblanks
    }

    /// VBlank ticks elapsed since the previous rendered app frame.
    ///
    /// A scene running at full NTSC cadence usually sees `1`. A heavy
    /// frame that misses a refresh sees `2` or more, which lets game
    /// code advance animation/simulation by display time instead of by
    /// rendered-frame count.
    #[inline]
    pub const fn delta_vblanks(self) -> u16 {
        self.delta_vblanks
    }

    /// Display cadence used for time conversion (`60` NTSC, `50` PAL).
    #[inline]
    pub const fn video_hz(self) -> u16 {
        self.video_hz
    }

    /// Delta time as Q12 seconds.
    #[inline]
    pub fn delta_seconds_q12(self) -> u32 {
        ((self.delta_vblanks as u32) << 12) / self.video_hz.max(1) as u32
    }

    /// Elapsed display time as Q12 seconds.
    #[inline]
    pub fn elapsed_seconds_q12(self) -> u32 {
        self.elapsed_vblanks.saturating_mul(1 << 12) / self.video_hz.max(1) as u32
    }
}

pub(crate) struct EngineClock {
    origin_vblank: u32,
    last_frame_vblank: u32,
    last_present_vblank: u32,
    video_hz: u16,
    first_frame: bool,
}

impl EngineClock {
    pub(crate) fn new(video_hz: u16) -> Self {
        platform::init();
        let now = platform::vblank_count();
        Self {
            origin_vblank: now,
            last_frame_vblank: now,
            last_present_vblank: now,
            video_hz,
            first_frame: true,
        }
    }

    pub(crate) fn begin_frame(&mut self, rendered_frame: u32) -> EngineTime {
        let now = platform::vblank_count();
        let elapsed = now.wrapping_sub(self.origin_vblank);
        let raw_delta = now.wrapping_sub(self.last_frame_vblank);
        self.last_frame_vblank = now;

        let delta = if self.first_frame {
            self.first_frame = false;
            1
        } else {
            raw_delta.clamp(1, u16::MAX as u32) as u16
        };

        EngineTime {
            rendered_frame,
            elapsed_vblanks: elapsed,
            delta_vblanks: delta,
            video_hz: self.video_hz,
        }
    }

    pub(crate) fn wait_present(&mut self) {
        self.last_present_vblank = platform::wait_present_vblank(self.last_present_vblank);
    }
}

#[cfg(target_arch = "mips")]
mod platform {
    use psx_gpu as gpu;

    pub(super) fn init() {
        gpu::configure_vsync_timer();
        psx_rt::interrupts::install_vblank_counter();
    }

    pub(super) fn vblank_count() -> u32 {
        psx_rt::interrupts::vblank_count()
    }

    pub(super) fn wait_present_vblank(last_present: u32) -> u32 {
        loop {
            let now = vblank_count();
            if now != last_present {
                return now;
            }
        }
    }
}

#[cfg(not(target_arch = "mips"))]
mod platform {
    pub(super) fn init() {}

    pub(super) fn vblank_count() -> u32 {
        0
    }

    pub(super) fn wait_present_vblank(last_present: u32) -> u32 {
        last_present.wrapping_add(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_time_uses_one_vblank_delta_for_legacy_cadence() {
        let t = EngineTime::start(60);
        assert_eq!(t.delta_vblanks(), 1);
        assert_eq!(t.delta_seconds_q12(), 68);
    }

    #[test]
    fn elapsed_seconds_uses_video_cadence() {
        let t = EngineTime {
            rendered_frame: 8,
            elapsed_vblanks: 30,
            delta_vblanks: 2,
            video_hz: 60,
        };
        assert_eq!(t.rendered_frame(), 8);
        assert_eq!(t.elapsed_seconds_q12(), 2048);
        assert_eq!(t.delta_seconds_q12(), 136);
    }
}
