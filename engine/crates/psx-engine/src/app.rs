//! App runner — the fixed-shape main loop that every game inherits
//! instead of rewriting.
//!
//! # Shape of the loop
//!
//! ```text
//!   gpu::init + draw_area + draw_offset
//!   FrameBuffer::new
//!   scene.init(&mut ctx)
//!   loop:
//!     ctx.pad_prev ← ctx.pad           (one-frame input history)
//!     ctx.pad      ← poll_port1()
//!     ctx.time     ← elapsed display-time snapshot
//!     scene.update(&mut ctx)
//!     ctx.fb.clear(config.clear_color)
//!     scene.render(&mut ctx)
//!     display-clock wait + draw_sync + fb.swap
//!     ctx.frame += 1
//! ```
//!
//! This mirrors every `sdk/examples/game-*/src/main.rs` file's
//! inner loop by eye — the engine just factors the shared cadence
//! out. If a scene wants a different cadence (custom clear, no
//! vsync, manual OT submission, …), the door's still open: the
//! scene's `update` / `render` methods can do whatever they want
//! with the ctx before the engine ticks over to the next frame.
//!
//! # No `!` impl on the scene
//!
//! [`App::run`] returns `!` because the main loop never terminates
//! on PSX (no OS to return to). The scene's methods return `()` —
//! they just tick and go. A scene that wants "exit" behaviour can
//! idle its own state machine in place.

use psx_gpu::framebuf::FrameBuffer;
use psx_gpu::{self as gpu, Resolution, VideoMode};
use psx_pad::{poll_port1, PadState};

use crate::scene::{Ctx, Scene};
use crate::time::{EngineClock, EngineTime};

/// Configuration passed to [`App::run`]. Sensible defaults via
/// [`Config::default`] so simple games can just write
/// `App::run(Config::default(), &mut game)`.
#[derive(Copy, Clone, Debug)]
pub struct Config {
    /// Visible framebuffer width in pixels.
    pub screen_w: u16,
    /// Visible framebuffer height in pixels.
    pub screen_h: u16,
    /// Video mode (NTSC / PAL). PAL games running in NTSC (or the
    /// reverse) show vertical compression / overscan — match the
    /// region you're testing on.
    pub video_mode: VideoMode,
    /// GP1 display resolution. Must match `screen_w × screen_h`.
    pub resolution: Resolution,
    /// RGB triple used to clear `ctx.fb` before each
    /// [`Scene::render`] call. Scenes that want a more elaborate
    /// background (textured backdrop, gouraud gradient, etc.) can
    /// set this to black and draw their own full-screen quad.
    pub clear_color: (u8, u8, u8),
}

impl Config {
    /// Display cadence in whole frames per second.
    #[inline]
    pub const fn video_hz(self) -> u16 {
        match self.video_mode {
            VideoMode::Ntsc => 60,
            VideoMode::Pal => 50,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            screen_w: 320,
            screen_h: 240,
            video_mode: VideoMode::Ntsc,
            resolution: Resolution::R320X240,
            clear_color: (0, 0, 0),
        }
    }
}

/// Engine entry point. Namespaced as a type (rather than a free
/// function) so future engine-level state (config getters, exit
/// handlers, debug introspection) has a natural home.
pub struct App;

impl App {
    /// Run `scene` under `config`. Never returns.
    ///
    /// Calls [`Scene::init`] once, then loops forever:
    /// poll-pad → update → clear → render → display-clock wait →
    /// draw-sync → swap.
    ///
    /// Typical call site in `main`:
    ///
    /// ```ignore
    /// #[no_mangle]
    /// fn main() -> ! {
    ///     let mut game = MyGame::new();
    ///     App::run(Config::default(), &mut game);
    /// }
    /// ```
    pub fn run<S: Scene>(config: Config, scene: &mut S) -> ! {
        gpu::init(config.video_mode, config.resolution);
        let mut clock = EngineClock::new(config.video_hz());
        let fb = FrameBuffer::new(config.screen_w, config.screen_h);
        gpu::set_draw_area(
            0,
            0,
            config.screen_w.saturating_sub(1),
            config.screen_h.saturating_sub(1),
        );
        gpu::set_draw_offset(0, 0);

        let mut ctx = Ctx {
            frame: 0,
            time: EngineTime::start(config.video_hz()),
            pad: PadState::NONE,
            pad_prev: PadState::NONE,
            fb,
        };

        scene.init(&mut ctx);

        loop {
            ctx.time = clock.begin_frame(ctx.frame);
            ctx.pad_prev = ctx.pad;
            ctx.pad = poll_port1();

            scene.update(&mut ctx);

            ctx.fb.clear(
                config.clear_color.0,
                config.clear_color.1,
                config.clear_color.2,
            );

            scene.render(&mut ctx);

            clock.wait_present();
            gpu::draw_sync();
            ctx.fb.swap();
            ctx.frame = ctx.frame.wrapping_add(1);
        }
    }
}
