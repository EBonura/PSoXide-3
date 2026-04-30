//! Scene trait + per-frame context.
//!
//! A `Scene` is whatever your game wants to run. The engine calls
//! [`Scene::init`] once at boot, then [`Scene::update`] + [`Scene::render`]
//! in a loop, passing a [`Ctx`] that carries the live-per-frame
//! things the scene needs: the current pad state (with edge-detection
//! helpers), the render-frame counter, engine time, and a
//! [`FrameBuffer`] ready to draw into.
//!
//! The split into `update` + `render` is cosmetic -- both get the
//! same `Ctx`. Keeping them separate reads better and makes it easy
//! to add determinism/replay hooks later (record `Ctx.pad` during
//! update, replay without re-rendering, etc).

use psx_gpu::framebuf::FrameBuffer;
use psx_pad::{button, PadState};

use crate::time::EngineTime;

/// Per-frame context passed to [`Scene::update`] and
/// [`Scene::render`]. The engine owns and updates this between
/// frames; the scene reads from it and draws through it.
pub struct Ctx {
    /// Monotonic visible-frame counter. Both `update` and `render`
    /// see the same value on a given iteration; the engine
    /// advances it once per end-of-loop. Wraps at `u32::MAX`
    /// (≈828 days at 60 fps). Engine APIs that want the type-
    /// safe variant take [`crate::Frames`]; `ctx.frame` stays a
    /// raw `u32` so the common `frame % N` / `frame & N` cases
    /// compose without unwrap ceremony.
    pub frame: u32,
    /// PS1 display-time snapshot for this app iteration. Unlike
    /// `frame`, this advances by elapsed VBlanks, so heavy render
    /// paths can keep animation/simulation time independent of
    /// rendered FPS.
    pub time: EngineTime,
    /// Port-1 pad state this frame.
    pub pad: PadState,
    /// Port-1 pad state last frame -- used by [`Ctx::just_pressed`]
    /// to distinguish "newly pressed this frame" from "held across
    /// multiple frames".
    pub pad_prev: PadState,
    /// Frame buffer the scene draws into. The engine clears it
    /// before [`Scene::render`] runs, and swaps it after.
    pub fb: FrameBuffer,
}

impl Ctx {
    /// `true` if `button` is pressed right now (held).
    #[inline]
    pub fn is_held(&self, button: u16) -> bool {
        self.pad.buttons.is_held(button)
    }

    /// `true` if `button` transitioned from released to pressed
    /// *this frame*. Exactly the edge-detect pattern every game
    /// reinvents; factored here so menus / fire buttons / etc don't
    /// have to track `pad_prev` themselves.
    #[inline]
    pub fn just_pressed(&self, button: u16) -> bool {
        self.pad.buttons.is_held(button) && !self.pad_prev.buttons.is_held(button)
    }

    /// `true` if `button` transitioned from pressed to released
    /// this frame.
    #[inline]
    pub fn just_released(&self, button: u16) -> bool {
        !self.pad.buttons.is_held(button) && self.pad_prev.buttons.is_held(button)
    }

    /// Convenience: any D-pad direction currently held.
    #[inline]
    pub fn dpad_any_held(&self) -> bool {
        self.is_held(button::UP)
            || self.is_held(button::DOWN)
            || self.is_held(button::LEFT)
            || self.is_held(button::RIGHT)
    }
}

/// Implement this trait on your game type and hand an instance to
/// [`App::run`][crate::app::App::run].
///
/// All methods take `&mut self` so the scene can keep its own state
/// inline (no globals needed). All methods take `&mut Ctx` so the
/// scene can read pad state and draw into the framebuffer.
pub trait Scene {
    /// Called once, before the main loop starts. Use for asset
    /// uploads (font atlas, textures), SPU sample loads, state
    /// initialisation. Default is a no-op.
    #[allow(unused_variables)]
    fn init(&mut self, ctx: &mut Ctx) {}

    /// Advance game state by one frame. Called before [`render`].
    /// Read pad input from `ctx`, write your internal state.
    ///
    /// [`render`]: Scene::render
    fn update(&mut self, ctx: &mut Ctx);

    /// Draw the current frame. Called after [`update`] and after
    /// `ctx.fb` has been cleared. The engine submits the final
    /// swap after this returns.
    ///
    /// [`update`]: Scene::update
    fn render(&mut self, ctx: &mut Ctx);
}
