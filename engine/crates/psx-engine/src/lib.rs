//! `psx-engine` -- PSoXide engine layer.
//!
//! The SDK exposes the PS1's hardware surface (GPU, SPU, GTE, pad
//! VRAM layout, primitives). The engine sits one level up and
//! provides the things a *game* actually wants:
//!
//! - a [`Scene`] trait and an [`App::run`] entry point so games
//!   don't each reinvent the main loop;
//! - a [`Ctx`] carrying per-frame state (pad, frame counter,
//!   engine time, framebuffer) to the scene;
//! - a canonical [`Angle`] unit so we stop hitting the recurring
//!   "256-per-revolution vs 4096-per-revolution" angle-mismatch bug
//!   that cost an afternoon on showcase-fog's light orbit;
//! - typed coordinate-space wrappers like [`RoomPoint`] so gameplay
//!   room-local positions do not quietly mix with raw renderer
//!   submission vertices;
//! - typed fixed-point scalars like [`Q12`] and [`Q8`] so movement,
//!   light intensity, and falloff code can name its unit scale;
//! - render helpers for ordering-table frames and fixed primitive
//!   arenas, so games can build PS1 painter's-algorithm command
//!   streams without rewriting OT ceremony in every scene.
//!
//! The engine is `no_std`, has no allocator dependency, and compiles
//! only for `target_arch = "mips"` (host stubs mirror the SDK's
//! pattern so `cargo check` still works on the host). Nothing here
//! touches disc / asset streaming -- that's the game's or the
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
pub mod character_motor;
pub mod fixed;
pub mod frames;
pub mod lighting;
pub mod movement;
pub mod render;
pub mod render3d;
pub mod scene;
pub mod sfx;
pub mod telemetry;
pub mod third_person_camera;
pub mod time;
pub mod transform;
pub mod world;
pub mod world_render;

pub use angle::Angle;
pub use app::{App, Config};
pub use character_motor::{
    CharacterMotorAction, CharacterMotorAnim, CharacterMotorConfig, CharacterMotorFrame,
    CharacterMotorInput, CharacterMotorState,
};
pub use fixed::{Q12, Q8};
pub use frames::{Frames, Ticks};
pub use lighting::{
    accumulate_point_lights, accumulate_point_lights_rgb, modulate_material_tint, modulate_tint,
    shade_material_tint_with_lights, shade_tint_with_lights, LightingRgb, MaterialTint,
    PointLightSample, Rgb8, LIGHTING_MAX, LIGHTING_NEUTRAL,
};
pub use movement::{
    camera_relative_move, camera_relative_move_axes, camera_relative_move_q12, CameraRelativeMove,
    InputAxis, InputAxisProfile, InputVector,
};
pub use render::{
    CameraDepth, DepthBand, DepthRange, DepthSlot, GpuPacket, OtDepth, OtFrame, PrimitiveArena,
};
pub use render3d::{
    compute_joint_view_transform, compute_joint_world_transform,
    project_model_vertex_with_joint_transforms, CullMode, DepthPolicy, GouraudMeshOptions,
    GouraudRenderPass, GouraudTriCommand, JointViewTransform, JointWorldTransform,
    LocalToWorldScale, MeshRenderStats, ProjectedTexturedVertex, ProjectedVertex,
    TexturedModelRenderStats, TexturedViewVertex, ViewVertex, WorldCamera, WorldProjection,
    WorldRenderLayer, WorldRenderPass, WorldRenderStats, WorldSurfaceOptions, WorldTriCommand,
};
// Re-export the GTE math types callers need to construct
// arguments for `submit_textured_model` (instance rotation,
// joint transforms) without pulling in `psx-gte` directly.
pub use psx_gte::math::Mat3I16;
pub use scene::{Ctx, Scene};
pub use third_person_camera::{
    ThirdPersonCameraConfig, ThirdPersonCameraFrame, ThirdPersonCameraInput,
    ThirdPersonCameraState, ThirdPersonCameraTarget,
};
pub use time::EngineTime;
pub use transform::{ActorTransform, RoomPoint, Vec3World, WorldVertex};
pub use world::{
    GridCoord, GridDirection, GridFloorSample, GridHorizontalFace, GridRoom, GridSector, GridSplit,
    GridVerticalFace, GridWalls, GridWorld, RoomCollision, RoomRender, RuntimeRoom,
    SectorCollision, SectorRender, WallCollision, WallRender, WorldMaterialId, GRID_SECTOR_SIZE,
};
pub use world_render::{
    draw_room, draw_room_lit, draw_room_lit_grid_visible, draw_room_vertex_lit,
    draw_room_vertex_lit_grid_visible, GridVisibility, GridVisibilityStats, NoWorldSurfaceLighting,
    SurfaceSidedness, WorldRenderMaterial, WorldSurfaceKind, WorldSurfaceLighting,
    WorldSurfaceSample,
};

/// Button-mask constants (UP, DOWN, CROSS, START, …) re-exported
/// from `psx_pad::button` so games using `Ctx::just_pressed` /
/// `is_held` don't need a direct `psx-pad` dep just for the button
/// names.
pub use psx_pad::button;
/// Pad-state types re-exported for scenes that need analog stick data.
pub use psx_pad::{AnalogSticks, PadMode, PadState};
