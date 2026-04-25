//! `psx-engine` — PSoXide engine layer.
//!
//! The SDK exposes the PS1's hardware surface (GPU, SPU, GTE, pad
//! VRAM layout, primitives). The engine sits one level up and
//! provides the things a *game* actually wants:
//!
//! - a [`Scene`] trait and an [`App::run`] entry point so games
//!   don't each reinvent the main loop;
//! - a [`Ctx`] carrying per-frame state (pad, frame counter,
//!   framebuffer) to the scene;
//! - a canonical [`Angle`] unit so we stop hitting the recurring
//!   "256-per-revolution vs 4096-per-revolution" angle-mismatch bug
//!   that cost an afternoon on showcase-fog's light orbit;
//! - render helpers for ordering-table frames and fixed primitive
//!   arenas, so games can build PS1 painter's-algorithm command
//!   streams without rewriting OT ceremony in every scene.
//!
//! The engine is `no_std`, has no allocator dependency, and compiles
//! only for `target_arch = "mips"` (host stubs mirror the SDK's
//! pattern so `cargo check` still works on the host). Nothing here
//! touches disc / asset streaming — that's the game's or the
//! content-pipeline's concern.
//!
//! # Minimal usage
//!
//! ```ignore
//! #![no_std]
//! #![no_main]
//! extern crate psx_rt;
//!
//! use psx_engine::{App, Config, Ctx, Scene};
//!
//! struct Game;
//!
//! impl Scene for Game {
//!     fn update(&mut self, _ctx: &mut Ctx) {}
//!     fn render(&mut self, _ctx: &mut Ctx) {}
//! }
//!
//! #[no_mangle]
//! fn main() -> ! {
//!     App::run(Config::default(), &mut Game);
//! }
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod angle;
pub mod app;
pub mod frames;
pub mod render;
pub mod render3d;
pub mod scene;
pub mod sfx;
pub mod transform;

pub use angle::Angle;
pub use app::{App, Config};
pub use frames::{Frames, Ticks};
pub use render::{DepthBand, DepthRange, DepthSlot, GpuPacket, OtFrame, PrimitiveArena};
pub use render3d::{
    DepthPolicy, GouraudMeshOptions, GouraudRenderPass, GouraudTriCommand, MeshRenderStats,
};
pub use scene::{Ctx, Scene};
pub use transform::{ActorTransform, Vec3World};

/// Button-mask constants (UP, DOWN, CROSS, START, …) re-exported
/// from `psx_pad::button` so games using `Ctx::just_pressed` /
/// `is_held` don't need a direct `psx-pad` dep just for the button
/// names.
pub use psx_pad::button;
