//! GPU-side VRAM for the compute-shader rasterizer.
//!
//! Phase A — infrastructure only. This crate owns nothing about
//! primitive rasterization yet. It exposes a [`VramGpu`] handle that
//! wraps a 1024×512 `R16Uint` storage texture (the PS1's BGR15
//! framebuffer) plus rect upload/download helpers and round-trip
//! parity tests against `emulator_core::Vram`.
//!
//! The CPU rasterizer in `emulator-core` is the parity oracle. The
//! compute path is opt-in until it matches pixel-for-pixel.

pub mod primitive;
pub mod rasterizer;
pub mod vram;

pub use primitive::{
    BlendMode, DrawArea, Fill, MonoRect, MonoTri, PrimFlags, ShadedTexTri, ShadedTri, TexRect,
    TexTri, Tpage,
};
pub use rasterizer::Rasterizer;
pub use vram::{VramGpu, VramGpuError, VRAM_FORMAT, VRAM_HEIGHT, VRAM_WIDTH};
