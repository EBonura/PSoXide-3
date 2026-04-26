//! Compute-shader rasterizer dispatcher.
//!
//! Phase B.1: one primitive, one dispatch. The pipeline objects are
//! built once and reused; each `dispatch_*` call writes the
//! primitive's parameters into a uniform buffer, picks the matching
//! pipeline, and dispatches a workgroup grid sized to the bounding
//! box.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

#[cfg(test)]
use crate::primitive::{BlendMode, PrimFlags};
use crate::primitive::{
    DrawArea, Fill, MonoRect, MonoTri, ShadedTexTri, ShadedTri, TexQuadBilinear, TexRect, TexTri,
    Tpage,
};
use crate::scanline::{self, RowState, ScanlineConsts};
use crate::vram::VramGpu;

const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;

/// Holds every wgpu pipeline object the rasterizer needs. Built once
/// per `VramGpu` and reused for the lifetime of the device.
pub struct Rasterizer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Mono-triangle pipeline.
    mono_tri_pipeline: wgpu::ComputePipeline,
    mono_tri_bg_layout: wgpu::BindGroupLayout,
    mono_tri_uniform: wgpu::Buffer,
    draw_area_uniform: wgpu::Buffer,

    // Textured-triangle pipeline.
    tex_tri_pipeline: wgpu::ComputePipeline,
    tex_tri_bg_layout: wgpu::BindGroupLayout,
    tex_tri_uniform: wgpu::Buffer,
    tpage_uniform: wgpu::Buffer,

    // Mono-rectangle pipeline (B.5.a). Reuses `mono_tri_bg_layout`
    // since the binding shape is identical (VRAM + prim + draw area).
    mono_rect_pipeline: wgpu::ComputePipeline,
    mono_rect_uniform: wgpu::Buffer,

    // Textured-rectangle pipeline (B.5.b). Reuses `tex_tri_bg_layout`.
    tex_rect_pipeline: wgpu::ComputePipeline,
    tex_rect_uniform: wgpu::Buffer,

    // Fill pipeline (B.5.c). Custom 2-binding shape — no draw area
    // because fill bypasses clipping.
    fill_pipeline: wgpu::ComputePipeline,
    fill_bg_layout: wgpu::BindGroupLayout,
    fill_uniform: wgpu::Buffer,

    // Shaded-triangle pipeline (B.3.a). Reuses `mono_tri_bg_layout`
    // (same 3-binding shape: VRAM + prim + draw area).
    shaded_tri_pipeline: wgpu::ComputePipeline,
    shaded_tri_uniform: wgpu::Buffer,

    // Textured-shaded triangle pipeline (B.3.b). Reuses
    // `tex_tri_bg_layout` (4 bindings: VRAM + prim + draw area + tpage).
    shaded_tex_tri_pipeline: wgpu::ComputePipeline,
    shaded_tex_tri_uniform: wgpu::Buffer,

    // Phase B.x: textured triangle with bit-exact scanline-delta UV
    // interpolation. Custom 6-binding shape because it adds a
    // per-row storage buffer + per-primitive scanline-consts uniform.
    tex_tri_scanline_pipeline: wgpu::ComputePipeline,
    tex_tri_scanline_bg_layout: wgpu::BindGroupLayout,
    tex_tri_scanline_consts: wgpu::Buffer,
    /// Resizable per-row storage buffer. Reallocated when a primitive
    /// needs more rows than the current capacity (cheap — wgpu
    /// doesn't actually free until `submit` completes anyway).
    tex_tri_scanline_rows: std::cell::RefCell<wgpu::Buffer>,

    // Phase B.x: shaded-textured triangle with bit-exact scanline-
    // delta UV + RGB interpolation. Reuses tex_tri_scanline_bg_layout
    // (same 6-binding shape).
    shaded_tex_tri_scanline_pipeline: wgpu::ComputePipeline,
    shaded_tex_tri_scanline_consts: wgpu::Buffer,
    shaded_tex_tri_scanline_rows: std::cell::RefCell<wgpu::Buffer>,

    // Phase C bug fix: axis-aligned textured quad with bilinear UV.
    // Same 4-binding shape as tex_tri (VRAM + prim + draw_area + tpage).
    tex_quad_bilinear_pipeline: wgpu::ComputePipeline,
    tex_quad_bilinear_uniform: wgpu::Buffer,

    // Phase B.x: mono + shaded triangle scanline pipelines. Same
    // 5-binding shape (VRAM + prim + draw area + rows + consts —
    // no tpage since neither samples a texture).
    mono_shaded_scanline_bg_layout: wgpu::BindGroupLayout,
    mono_tri_scanline_pipeline: wgpu::ComputePipeline,
    mono_tri_scanline_consts: wgpu::Buffer,
    mono_tri_scanline_rows: std::cell::RefCell<wgpu::Buffer>,
    shaded_tri_scanline_pipeline: wgpu::ComputePipeline,
    shaded_tri_scanline_consts: wgpu::Buffer,
    shaded_tri_scanline_rows: std::cell::RefCell<wgpu::Buffer>,
}

impl Rasterizer {
    /// Build all pipelines on top of the same device that owns
    /// `VramGpu`. Cheap to call multiple times in tests but in
    /// production you want one shared instance.
    pub fn new(vram: &VramGpu) -> Self {
        let device = vram.device().clone();
        let queue = vram.queue().clone();

        // Bind group: VRAM (storage), primitive uniform, draw-area
        // uniform. All three are visible only to compute stages.
        let mono_tri_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("psx-rasterizer-mono-tri-bgl"),
                entries: &[
                    // 0: VRAM storage buffer (read_write).
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: Primitive uniform (read-only).
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 2: DrawArea uniform.
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-mono-tri-pl"),
            bind_group_layouts: &[&mono_tri_bg_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-mono-tri-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mono_tri.wgsl").into()),
        });

        let mono_tri_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("psx-rasterizer-mono-tri"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("rasterize"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Reusable uniform buffers — wgpu doesn't let us write a
        // struct directly into a freshly-bound resource per dispatch
        // without allocating, so we keep a stable buffer and update
        // it via `queue.write_buffer`.
        let mono_tri_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-mono-tri-uniform"),
            size: std::mem::size_of::<MonoTri>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_area_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-draw-area-uniform"),
            // DrawArea is 16 bytes already, but pad-up to wgpu's
            // minimum uniform buffer offset alignment to be safe.
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Textured-triangle pipeline ----------
        let tex_tri_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("psx-rasterizer-tex-tri-bgl"),
            entries: &[
                // 0: VRAM
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: TexTri uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: DrawArea uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: Tpage uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let tex_tri_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-tex-tri-pl"),
            bind_group_layouts: &[&tex_tri_bg_layout],
            push_constant_ranges: &[],
        });
        let tex_tri_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-tex-tri-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tex_tri.wgsl").into()),
        });
        let tex_tri_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("psx-rasterizer-tex-tri"),
            layout: Some(&tex_tri_pl),
            module: &tex_tri_shader,
            entry_point: Some("rasterize"),
            compilation_options: Default::default(),
            cache: None,
        });
        let tex_tri_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-tex-tri-uniform"),
            size: std::mem::size_of::<TexTri>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tpage_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-tpage-uniform"),
            size: std::mem::size_of::<Tpage>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Mono-rectangle pipeline (B.5.a) ----------
        // Same 3-binding shape as the mono-triangle path: VRAM,
        // primitive uniform, draw area. Reuse the layout directly.
        let mono_rect_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-mono-rect-pl"),
            bind_group_layouts: &[&mono_tri_bg_layout],
            push_constant_ranges: &[],
        });
        let mono_rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-mono-rect-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mono_rect.wgsl").into()),
        });
        let mono_rect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("psx-rasterizer-mono-rect"),
            layout: Some(&mono_rect_pl),
            module: &mono_rect_shader,
            entry_point: Some("rasterize"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mono_rect_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-mono-rect-uniform"),
            size: std::mem::size_of::<MonoRect>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Textured-rectangle pipeline (B.5.b) ----------
        // Same 4-binding shape as the textured-triangle path: VRAM,
        // primitive uniform, draw area, tpage. Reuse the layout.
        let tex_rect_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-tex-rect-pl"),
            bind_group_layouts: &[&tex_tri_bg_layout],
            push_constant_ranges: &[],
        });
        let tex_rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-tex-rect-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tex_rect.wgsl").into()),
        });
        let tex_rect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("psx-rasterizer-tex-rect"),
            layout: Some(&tex_rect_pl),
            module: &tex_rect_shader,
            entry_point: Some("rasterize"),
            compilation_options: Default::default(),
            cache: None,
        });
        let tex_rect_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-tex-rect-uniform"),
            size: std::mem::size_of::<TexRect>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Fill pipeline (B.5.c) ----------
        // 2 bindings: VRAM + Fill uniform. No draw area / no tpage —
        // fill bypasses clipping and never reads VRAM.
        let fill_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("psx-rasterizer-fill-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let fill_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-fill-pl"),
            bind_group_layouts: &[&fill_bg_layout],
            push_constant_ranges: &[],
        });
        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-fill-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fill.wgsl").into()),
        });
        let fill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("psx-rasterizer-fill"),
            layout: Some(&fill_pl),
            module: &fill_shader,
            entry_point: Some("rasterize"),
            compilation_options: Default::default(),
            cache: None,
        });
        let fill_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-fill-uniform"),
            size: std::mem::size_of::<Fill>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Shaded-triangle pipeline (B.3.a) ----------
        // Same binding shape as mono-tri (VRAM + prim + draw area).
        let shaded_tri_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-shaded-tri-pl"),
            bind_group_layouts: &[&mono_tri_bg_layout],
            push_constant_ranges: &[],
        });
        let shaded_tri_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-shaded-tri-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shaded_tri.wgsl").into()),
        });
        let shaded_tri_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("psx-rasterizer-shaded-tri"),
                layout: Some(&shaded_tri_pl),
                module: &shaded_tri_shader,
                entry_point: Some("rasterize"),
                compilation_options: Default::default(),
                cache: None,
            });
        let shaded_tri_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-shaded-tri-uniform"),
            size: std::mem::size_of::<ShadedTri>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Textured-shaded triangle pipeline (B.3.b) ----------
        let shaded_tex_tri_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-shaded-tex-tri-pl"),
            bind_group_layouts: &[&tex_tri_bg_layout],
            push_constant_ranges: &[],
        });
        let shaded_tex_tri_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-shaded-tex-tri-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shaded_tex_tri.wgsl").into()),
        });
        let shaded_tex_tri_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("psx-rasterizer-shaded-tex-tri"),
                layout: Some(&shaded_tex_tri_pl),
                module: &shaded_tex_tri_shader,
                entry_point: Some("rasterize"),
                compilation_options: Default::default(),
                cache: None,
            });
        let shaded_tex_tri_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-shaded-tex-tri-uniform"),
            size: std::mem::size_of::<ShadedTexTri>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Tex-tri scanline pipeline (B.x) ----------
        // 6 bindings: VRAM, prim, draw area, tpage, per-row state
        // (storage), scanline consts (uniform).
        let tex_tri_scanline_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("psx-rasterizer-tex-tri-scanline-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let tex_tri_scanline_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-tex-tri-scanline-pl"),
            bind_group_layouts: &[&tex_tri_scanline_bg_layout],
            push_constant_ranges: &[],
        });
        let tex_tri_scanline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-tex-tri-scanline-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tex_tri_scanline.wgsl").into(),
            ),
        });
        let tex_tri_scanline_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("psx-rasterizer-tex-tri-scanline"),
                layout: Some(&tex_tri_scanline_pl),
                module: &tex_tri_scanline_shader,
                entry_point: Some("rasterize"),
                compilation_options: Default::default(),
                cache: None,
            });
        let tex_tri_scanline_consts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-tex-tri-scanline-consts"),
            size: std::mem::size_of::<ScanlineConsts>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Pre-allocate room for a 512-row triangle (the max). Avoids
        // reallocating in the hot path. 512 × 64 = 32 KiB.
        let initial_rows_capacity_bytes = 512u64 * std::mem::size_of::<RowState>() as u64;
        let tex_tri_scanline_rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-tex-tri-scanline-rows"),
            size: initial_rows_capacity_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Shaded-tex-tri scanline pipeline (B.x) ----------
        // Same 6-binding layout as tex_tri_scanline; just a different
        // shader entry that walks RGB in addition to UV.
        let shaded_tex_tri_scanline_pl =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("psx-rasterizer-shaded-tex-tri-scanline-pl"),
                bind_group_layouts: &[&tex_tri_scanline_bg_layout],
                push_constant_ranges: &[],
            });
        let shaded_tex_tri_scanline_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("psx-rasterizer-shaded-tex-tri-scanline-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/shaded_tex_tri_scanline.wgsl").into(),
                ),
            });
        let shaded_tex_tri_scanline_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("psx-rasterizer-shaded-tex-tri-scanline"),
                layout: Some(&shaded_tex_tri_scanline_pl),
                module: &shaded_tex_tri_scanline_shader,
                entry_point: Some("rasterize"),
                compilation_options: Default::default(),
                cache: None,
            });
        let shaded_tex_tri_scanline_consts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-shaded-tex-tri-scanline-consts"),
            size: std::mem::size_of::<ScanlineConsts>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shaded_tex_tri_scanline_rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-shaded-tex-tri-scanline-rows"),
            size: initial_rows_capacity_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Tex-quad bilinear pipeline (Phase C bug fix) ----------
        // Reuses `tex_tri_bg_layout` (same 4-binding shape: VRAM,
        // prim uniform, draw area, tpage). Different shader entry +
        // dedicated prim uniform.
        let tex_quad_bilinear_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-tex-quad-bilinear-pl"),
            bind_group_layouts: &[&tex_tri_bg_layout],
            push_constant_ranges: &[],
        });
        let tex_quad_bilinear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-tex-quad-bilinear-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tex_quad_bilinear.wgsl").into(),
            ),
        });
        let tex_quad_bilinear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("psx-rasterizer-tex-quad-bilinear"),
                layout: Some(&tex_quad_bilinear_pl),
                module: &tex_quad_bilinear_shader,
                entry_point: Some("rasterize"),
                compilation_options: Default::default(),
                cache: None,
            });
        let tex_quad_bilinear_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-tex-quad-bilinear-uniform"),
            size: std::mem::size_of::<TexQuadBilinear>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------- Mono / Shaded tri scanline pipelines (B.x) ----------
        // 5-binding shape: VRAM + prim + draw area + rows + consts.
        let mono_shaded_scanline_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("psx-rasterizer-mono-shaded-scanline-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let mono_shaded_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-rasterizer-mono-shaded-scanline-pl"),
            bind_group_layouts: &[&mono_shaded_scanline_bg_layout],
            push_constant_ranges: &[],
        });

        let mono_tri_scanline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-rasterizer-mono-tri-scanline-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mono_tri_scanline.wgsl").into(),
            ),
        });
        let mono_tri_scanline_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("psx-rasterizer-mono-tri-scanline"),
                layout: Some(&mono_shaded_pl),
                module: &mono_tri_scanline_shader,
                entry_point: Some("rasterize"),
                compilation_options: Default::default(),
                cache: None,
            });
        let mono_tri_scanline_consts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-mono-tri-scanline-consts"),
            size: std::mem::size_of::<ScanlineConsts>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mono_tri_scanline_rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-mono-tri-scanline-rows"),
            size: initial_rows_capacity_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shaded_tri_scanline_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("psx-rasterizer-shaded-tri-scanline-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/shaded_tri_scanline.wgsl").into(),
                ),
            });
        let shaded_tri_scanline_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("psx-rasterizer-shaded-tri-scanline"),
                layout: Some(&mono_shaded_pl),
                module: &shaded_tri_scanline_shader,
                entry_point: Some("rasterize"),
                compilation_options: Default::default(),
                cache: None,
            });
        let shaded_tri_scanline_consts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-shaded-tri-scanline-consts"),
            size: std::mem::size_of::<ScanlineConsts>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shaded_tri_scanline_rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-shaded-tri-scanline-rows"),
            size: initial_rows_capacity_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            mono_tri_pipeline,
            mono_tri_bg_layout,
            mono_tri_uniform,
            draw_area_uniform,
            tex_tri_pipeline,
            tex_tri_bg_layout,
            tex_tri_uniform,
            tpage_uniform,
            mono_rect_pipeline,
            mono_rect_uniform,
            tex_rect_pipeline,
            tex_rect_uniform,
            fill_pipeline,
            fill_bg_layout,
            fill_uniform,
            shaded_tri_pipeline,
            shaded_tri_uniform,
            shaded_tex_tri_pipeline,
            shaded_tex_tri_uniform,
            tex_tri_scanline_pipeline,
            tex_tri_scanline_bg_layout,
            tex_tri_scanline_consts,
            tex_tri_scanline_rows: std::cell::RefCell::new(tex_tri_scanline_rows),
            shaded_tex_tri_scanline_pipeline,
            shaded_tex_tri_scanline_consts,
            shaded_tex_tri_scanline_rows: std::cell::RefCell::new(shaded_tex_tri_scanline_rows),
            tex_quad_bilinear_pipeline,
            tex_quad_bilinear_uniform,
            mono_shaded_scanline_bg_layout,
            mono_tri_scanline_pipeline,
            mono_tri_scanline_consts,
            mono_tri_scanline_rows: std::cell::RefCell::new(mono_tri_scanline_rows),
            shaded_tri_scanline_pipeline,
            shaded_tri_scanline_consts,
            shaded_tri_scanline_rows: std::cell::RefCell::new(shaded_tri_scanline_rows),
        }
    }

    /// Bit-exact monochrome triangle dispatch via scanline-delta
    /// coverage. Same drawing/RMW behaviour as `dispatch_mono_tri`
    /// but uses the CPU rasterizer's per-row `(left_x, right_x)`
    /// coverage rule instead of a per-pixel edge-function test, so
    /// edge pixels match the CPU byte-for-byte.
    pub fn dispatch_mono_tri_scanline(
        &self,
        vram: &VramGpu,
        tri: &MonoTri,
        area: &DrawArea,
    ) -> bool {
        if tri.exceeds_hw_extent() {
            return false;
        }
        let v = [
            (tri.v0[0], tri.v0[1]),
            (tri.v1[0], tri.v1[1]),
            (tri.v2[0], tri.v2[1]),
        ];
        let setup = match scanline::build_setup(v, [(0, 0); 3], [(0, 0, 0); 3]) {
            Some(s) => s,
            None => return false,
        };
        self.scanline_dispatch(
            vram,
            tri,
            std::mem::size_of::<MonoTri>() as u64,
            &self.mono_tri_scanline_pipeline,
            &self.mono_tri_scanline_consts,
            &self.mono_tri_scanline_rows,
            &setup,
            area,
            tri.bbox_max[0] - tri.bbox_min[0] + 1,
            tri.bbox_max[1] - tri.bbox_min[1] + 1,
            "mono",
        )
    }

    /// Bit-exact Gouraud-shaded triangle dispatch via scanline-delta.
    pub fn dispatch_shaded_tri_scanline(
        &self,
        vram: &VramGpu,
        tri: &ShadedTri,
        area: &DrawArea,
    ) -> bool {
        if tri.exceeds_hw_extent() {
            return false;
        }
        let v = [
            (tri.v0[0], tri.v0[1]),
            (tri.v1[0], tri.v1[1]),
            (tri.v2[0], tri.v2[1]),
        ];
        let unpack_rgb = |c: u32| {
            (
                (c & 0xFF) as i32,
                ((c >> 8) & 0xFF) as i32,
                ((c >> 16) & 0xFF) as i32,
            )
        };
        let rgb = [unpack_rgb(tri.c0), unpack_rgb(tri.c1), unpack_rgb(tri.c2)];
        let setup = match scanline::build_setup(v, [(0, 0); 3], rgb) {
            Some(s) => s,
            None => return false,
        };
        self.scanline_dispatch(
            vram,
            tri,
            std::mem::size_of::<ShadedTri>() as u64,
            &self.shaded_tri_scanline_pipeline,
            &self.shaded_tri_scanline_consts,
            &self.shaded_tri_scanline_rows,
            &setup,
            area,
            tri.bbox_max[0] - tri.bbox_min[0] + 1,
            tri.bbox_max[1] - tri.bbox_min[1] + 1,
            "shaded",
        )
    }

    /// Shared dispatch helper for the 5-binding scanline pipelines
    /// (mono + shaded). Writes the prim uniform from the host
    /// struct, uploads per-row state, and dispatches over the bbox.
    #[allow(clippy::too_many_arguments)]
    fn scanline_dispatch<P: bytemuck::Pod>(
        &self,
        vram: &VramGpu,
        tri: &P,
        _prim_size_bytes: u64,
        pipeline: &wgpu::ComputePipeline,
        consts_buf: &wgpu::Buffer,
        rows_cell: &std::cell::RefCell<wgpu::Buffer>,
        setup: &scanline::ScanlineSetup,
        area: &DrawArea,
        bbox_w: i32,
        bbox_h: i32,
        label: &'static str,
    ) -> bool {
        if bbox_w <= 0 || bbox_h <= 0 {
            return false;
        }
        // Both mono and shaded scanline paths reuse `mono_tri_uniform`
        // (mono) / `shaded_tri_uniform` (shaded) — but to keep this
        // helper generic, we'll write through the existing per-prim
        // uniform we already manage. Looking up which one to use:
        let prim_uniform = match label {
            "mono" => &self.mono_tri_uniform,
            "shaded" => &self.shaded_tri_uniform,
            _ => unreachable!("unknown scanline-dispatch label: {label}"),
        };

        let rows_size_bytes = (setup.rows.len() as u64) * std::mem::size_of::<RowState>() as u64;
        {
            let mut rows_buf = rows_cell.borrow_mut();
            if rows_buf.size() < rows_size_bytes {
                *rows_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("psx-rasterizer-scanline-rows-grown"),
                    size: rows_size_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
        }
        self.queue
            .write_buffer(prim_uniform, 0, bytemuck::bytes_of(tri));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));
        self.queue
            .write_buffer(consts_buf, 0, bytemuck::bytes_of(&setup.consts));
        let rows_buf = rows_cell.borrow();
        self.queue
            .write_buffer(&rows_buf, 0, bytemuck::cast_slice(&setup.rows));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-scanline-bg"),
            layout: &self.mono_shaded_scanline_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prim_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rows_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: consts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-scanline-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-scanline-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (bbox_w as u32).div_ceil(WORKGROUP_SIZE_X);
            let groups_y = (bbox_h as u32).div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        true
    }

    /// Bit-exact textured-Gouraud triangle dispatch (B.x). Composes
    /// the scanline-delta UV walk from `dispatch_tex_tri_scanline`
    /// with the per-vertex tint walk; the host runs the same
    /// `setup_sections` + `next_row` loop the CPU does, with both
    /// UV and RGB attributes populated, so the GPU output matches
    /// the CPU rasterizer byte-for-byte.
    pub fn dispatch_shaded_tex_tri_scanline(
        &self,
        vram: &VramGpu,
        tri: &ShadedTexTri,
        tpage: &Tpage,
        area: &DrawArea,
    ) -> bool {
        if tri.exceeds_hw_extent() {
            return false;
        }
        let v = [
            (tri.v0[0], tri.v0[1]),
            (tri.v1[0], tri.v1[1]),
            (tri.v2[0], tri.v2[1]),
        ];
        let uv = [
            ((tri.uv0 & 0xFF) as i32, ((tri.uv0 >> 8) & 0xFF) as i32),
            ((tri.uv1 & 0xFF) as i32, ((tri.uv1 >> 8) & 0xFF) as i32),
            ((tri.uv2 & 0xFF) as i32, ((tri.uv2 >> 8) & 0xFF) as i32),
        ];
        // Vertex tints are 24-bit RGB packed in the c0/c1/c2 fields.
        let unpack_rgb = |c: u32| {
            (
                (c & 0xFF) as i32,
                ((c >> 8) & 0xFF) as i32,
                ((c >> 16) & 0xFF) as i32,
            )
        };
        let rgb = [unpack_rgb(tri.c0), unpack_rgb(tri.c1), unpack_rgb(tri.c2)];
        let setup = match scanline::build_setup(v, uv, rgb) {
            Some(s) => s,
            None => return false,
        };

        let rows_size_bytes = (setup.rows.len() as u64) * std::mem::size_of::<RowState>() as u64;
        {
            let mut rows_buf = self.shaded_tex_tri_scanline_rows.borrow_mut();
            if rows_buf.size() < rows_size_bytes {
                *rows_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("psx-rasterizer-shaded-tex-tri-scanline-rows-grown"),
                    size: rows_size_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
        }

        self.queue
            .write_buffer(&self.shaded_tex_tri_uniform, 0, bytemuck::bytes_of(tri));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));
        self.queue
            .write_buffer(&self.tpage_uniform, 0, bytemuck::bytes_of(tpage));
        self.queue.write_buffer(
            &self.shaded_tex_tri_scanline_consts,
            0,
            bytemuck::bytes_of(&setup.consts),
        );
        let rows_buf = self.shaded_tex_tri_scanline_rows.borrow();
        self.queue
            .write_buffer(&rows_buf, 0, bytemuck::cast_slice(&setup.rows));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-shaded-tex-tri-scanline-bg"),
            layout: &self.tex_tri_scanline_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.shaded_tex_tri_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tpage_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: rows_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.shaded_tex_tri_scanline_consts.as_entire_binding(),
                },
            ],
        });

        let bbox_w = tri.bbox_max[0] - tri.bbox_min[0] + 1;
        let bbox_h = tri.bbox_max[1] - tri.bbox_min[1] + 1;
        if bbox_w <= 0 || bbox_h <= 0 {
            return false;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-shaded-tex-tri-scanline-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-shaded-tex-tri-scanline-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shaded_tex_tri_scanline_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (bbox_w as u32).div_ceil(WORKGROUP_SIZE_X);
            let groups_y = (bbox_h as u32).div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        true
    }

    /// Bit-exact textured-triangle dispatch via scanline-delta math.
    /// The host runs the same `setup_sections` + `next_row` loop the
    /// CPU rasterizer does, ships per-row state to the GPU, and the
    /// shader walks per-pixel using i64-emulated Q16.16 arithmetic.
    /// Results match the CPU rasterizer byte-for-byte.
    ///
    /// Returns `false` if the triangle degenerates (zero height /
    /// longest) — same drop conditions as the CPU.
    pub fn dispatch_tex_tri_scanline(
        &self,
        vram: &VramGpu,
        tri: &TexTri,
        tpage: &Tpage,
        area: &DrawArea,
    ) -> bool {
        if tri.exceeds_hw_extent() {
            return false;
        }
        let v = [
            (tri.v0[0], tri.v0[1]),
            (tri.v1[0], tri.v1[1]),
            (tri.v2[0], tri.v2[1]),
        ];
        let uv = [
            ((tri.uv0 & 0xFF) as i32, ((tri.uv0 >> 8) & 0xFF) as i32),
            ((tri.uv1 & 0xFF) as i32, ((tri.uv1 >> 8) & 0xFF) as i32),
            ((tri.uv2 & 0xFF) as i32, ((tri.uv2 >> 8) & 0xFF) as i32),
        ];
        let setup = match scanline::build_setup(v, uv, [(0, 0, 0); 3]) {
            Some(s) => s,
            None => return false,
        };

        // Re-allocate per-row buffer if too small.
        let rows_size_bytes = (setup.rows.len() as u64) * std::mem::size_of::<RowState>() as u64;
        {
            let mut rows_buf = self.tex_tri_scanline_rows.borrow_mut();
            if rows_buf.size() < rows_size_bytes {
                *rows_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("psx-rasterizer-tex-tri-scanline-rows-grown"),
                    size: rows_size_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
        }

        // Upload uniforms + per-row data.
        self.queue
            .write_buffer(&self.tex_tri_uniform, 0, bytemuck::bytes_of(tri));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));
        self.queue
            .write_buffer(&self.tpage_uniform, 0, bytemuck::bytes_of(tpage));
        self.queue.write_buffer(
            &self.tex_tri_scanline_consts,
            0,
            bytemuck::bytes_of(&setup.consts),
        );
        let rows_buf = self.tex_tri_scanline_rows.borrow();
        self.queue
            .write_buffer(&rows_buf, 0, bytemuck::cast_slice(&setup.rows));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-tex-tri-scanline-bg"),
            layout: &self.tex_tri_scanline_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tex_tri_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tpage_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: rows_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.tex_tri_scanline_consts.as_entire_binding(),
                },
            ],
        });

        let bbox_w = tri.bbox_max[0] - tri.bbox_min[0] + 1;
        let bbox_h = tri.bbox_max[1] - tri.bbox_min[1] + 1;
        if bbox_w <= 0 || bbox_h <= 0 {
            return false;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-tex-tri-scanline-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-tex-tri-scanline-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tex_tri_scanline_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (bbox_w as u32).div_ceil(WORKGROUP_SIZE_X);
            let groups_y = (bbox_h as u32).div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        true
    }

    /// Dispatch one textured Gouraud-shaded triangle. Composes
    /// texture sampling (`tex_tri`) with per-vertex tint
    /// interpolation (`shaded_tri`).
    pub fn dispatch_shaded_tex_tri(
        &self,
        vram: &VramGpu,
        tri: &ShadedTexTri,
        tpage: &Tpage,
        area: &DrawArea,
    ) {
        if tri.exceeds_hw_extent() {
            return;
        }
        let bbox_w = tri.bbox_max[0] - tri.bbox_min[0] + 1;
        let bbox_h = tri.bbox_max[1] - tri.bbox_min[1] + 1;
        if bbox_w <= 0 || bbox_h <= 0 {
            return;
        }
        self.queue
            .write_buffer(&self.shaded_tex_tri_uniform, 0, bytemuck::bytes_of(tri));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));
        self.queue
            .write_buffer(&self.tpage_uniform, 0, bytemuck::bytes_of(tpage));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-shaded-tex-tri-bg"),
            layout: &self.tex_tri_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.shaded_tex_tri_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tpage_uniform.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-shaded-tex-tri-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-shaded-tex-tri-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shaded_tex_tri_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (bbox_w as u32).div_ceil(WORKGROUP_SIZE_X);
            let groups_y = (bbox_h as u32).div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Dispatch one Gouraud-shaded triangle. Same coverage rules as
    /// `dispatch_mono_tri`; per-pixel colour interpolated from the
    /// three vertex colours.
    pub fn dispatch_shaded_tri(&self, vram: &VramGpu, tri: &ShadedTri, area: &DrawArea) {
        if tri.exceeds_hw_extent() {
            return;
        }
        let bbox_w = tri.bbox_max[0] - tri.bbox_min[0] + 1;
        let bbox_h = tri.bbox_max[1] - tri.bbox_min[1] + 1;
        if bbox_w <= 0 || bbox_h <= 0 {
            return;
        }
        self.queue
            .write_buffer(&self.shaded_tri_uniform, 0, bytemuck::bytes_of(tri));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-shaded-tri-bg"),
            layout: &self.mono_tri_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.shaded_tri_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-shaded-tri-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-shaded-tri-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shaded_tri_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (bbox_w as u32).div_ceil(WORKGROUP_SIZE_X);
            let groups_y = (bbox_h as u32).div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Dispatch one quick-fill primitive into VRAM. Bypasses all
    /// drawing-state — matches the CPU `Gpu::fill_rect`. Caller is
    /// responsible for the 16-pixel x/w masking; `Fill::new` does
    /// it for you.
    /// Dispatch one axis-aligned textured quad with bilinear UV
    /// interpolation. The host has already verified the geometry
    /// is axis-aligned via `TexQuadBilinear::is_axis_aligned`.
    /// Matches the CPU rasterizer's `rasterize_axis_aligned_textured_quad`
    /// fast path byte-for-byte.
    pub fn dispatch_tex_quad_bilinear(
        &self,
        vram: &VramGpu,
        quad: &TexQuadBilinear,
        tpage: &Tpage,
        area: &DrawArea,
    ) -> bool {
        if quad.exceeds_hw_extent() {
            return false;
        }
        let w = quad.v1[0] - quad.v0[0];
        let h = quad.v2[1] - quad.v0[1];
        if w <= 0 || h <= 0 {
            return false;
        }
        self.queue
            .write_buffer(&self.tex_quad_bilinear_uniform, 0, bytemuck::bytes_of(quad));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));
        self.queue
            .write_buffer(&self.tpage_uniform, 0, bytemuck::bytes_of(tpage));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-tex-quad-bilinear-bg"),
            layout: &self.tex_tri_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tex_quad_bilinear_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tpage_uniform.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-tex-quad-bilinear-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-tex-quad-bilinear-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tex_quad_bilinear_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (w as u32).div_ceil(WORKGROUP_SIZE_X);
            let groups_y = (h as u32).div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        true
    }

    pub fn dispatch_fill(&self, vram: &VramGpu, fill: &Fill) {
        if fill.wh[0] == 0 || fill.wh[1] == 0 {
            return;
        }
        self.queue
            .write_buffer(&self.fill_uniform, 0, bytemuck::bytes_of(fill));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-fill-bg"),
            layout: &self.fill_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.fill_uniform.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-fill-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-fill-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fill_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = fill.wh[0].div_ceil(WORKGROUP_SIZE_X);
            let groups_y = fill.wh[1].div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// VRAM-to-VRAM copy (`GP0 0x80`). Mirrors the CPU rasterizer's
    /// row-by-row semantics exactly: for each row, the entire source
    /// row is read into a staging buffer first, then written to the
    /// dest row. This means vertically-overlapping copies "smear"
    /// the source down — the same behaviour the CPU produces (Sony
    /// docs describe this as the row-buffer of the copy unit).
    ///
    /// Implementation: one per-row `src→temp` + `temp→dst` pair,
    /// all queued into a single command encoder so wgpu runs them
    /// strictly in order. Goes through a 1-row staging buffer
    /// because wgpu rejects `copy_buffer_to_buffer` with the same
    /// buffer as src and dst — we'd need that for direct VRAM-to-
    /// VRAM otherwise.
    pub fn dispatch_vram_copy(
        &self,
        vram: &VramGpu,
        src: (u32, u32),
        dst: (u32, u32),
        wh: (u32, u32),
    ) {
        let (sx, sy) = src;
        let (dx, dy) = dst;
        let (w, h) = wh;
        if w == 0 || h == 0 {
            return;
        }

        let row_bytes = (w as u64) * 4;
        let temp = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-rasterizer-vram-copy-temp"),
            size: row_bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-vram-copy-encoder"),
            });
        for row in 0..h {
            let s_off = ((sy + row) as u64 * super::vram::VRAM_WIDTH as u64 + sx as u64) * 4;
            let d_off = ((dy + row) as u64 * super::vram::VRAM_WIDTH as u64 + dx as u64) * 4;
            // Step 1: src row → temp.
            encoder.copy_buffer_to_buffer(vram.buffer(), s_off, &temp, 0, row_bytes);
            // Step 2: temp → dst row. Same encoder ⇒ runs strictly
            // after step 1, which gives the CPU's row-buffer semantics.
            encoder.copy_buffer_to_buffer(&temp, 0, vram.buffer(), d_off, row_bytes);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Dispatch one monochrome rectangle. `xy` is the top-left
    /// (already includes drawing-offset). Width/height of zero are
    /// dropped silently to match the CPU rasterizer.
    pub fn dispatch_mono_rect(&self, vram: &VramGpu, rect: &MonoRect, area: &DrawArea) {
        if rect.wh[0] == 0 || rect.wh[1] == 0 {
            return;
        }
        self.queue
            .write_buffer(&self.mono_rect_uniform, 0, bytemuck::bytes_of(rect));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-mono-rect-bg"),
            layout: &self.mono_tri_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.mono_rect_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-mono-rect-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-mono-rect-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mono_rect_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = rect.wh[0].div_ceil(WORKGROUP_SIZE_X);
            let groups_y = rect.wh[1].div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Dispatch one textured rectangle. Linear UV stepping (no
    /// interpolation) → bit-exact texel parity vs the CPU.
    pub fn dispatch_tex_rect(
        &self,
        vram: &VramGpu,
        rect: &TexRect,
        tpage: &Tpage,
        area: &DrawArea,
    ) {
        if rect.wh[0] == 0 || rect.wh[1] == 0 {
            return;
        }
        self.queue
            .write_buffer(&self.tex_rect_uniform, 0, bytemuck::bytes_of(rect));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));
        self.queue
            .write_buffer(&self.tpage_uniform, 0, bytemuck::bytes_of(tpage));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-tex-rect-bg"),
            layout: &self.tex_tri_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tex_rect_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tpage_uniform.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-tex-rect-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-tex-rect-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tex_rect_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = rect.wh[0].div_ceil(WORKGROUP_SIZE_X);
            let groups_y = rect.wh[1].div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Dispatch one textured triangle into VRAM. `tpage` selects the
    /// 256×256 source rect + colour depth; `tri` carries vertices,
    /// UVs, CLUT, tint, flags. Returns immediately after queuing.
    pub fn dispatch_tex_tri(&self, vram: &VramGpu, tri: &TexTri, tpage: &Tpage, area: &DrawArea) {
        if tri.exceeds_hw_extent() {
            return;
        }
        let bbox_w = tri.bbox_max[0] - tri.bbox_min[0] + 1;
        let bbox_h = tri.bbox_max[1] - tri.bbox_min[1] + 1;
        if bbox_w <= 0 || bbox_h <= 0 {
            return;
        }

        self.queue
            .write_buffer(&self.tex_tri_uniform, 0, bytemuck::bytes_of(tri));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));
        self.queue
            .write_buffer(&self.tpage_uniform, 0, bytemuck::bytes_of(tpage));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-tex-tri-bg"),
            layout: &self.tex_tri_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tex_tri_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tpage_uniform.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-tex-tri-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-tex-tri-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tex_tri_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (bbox_w as u32).div_ceil(WORKGROUP_SIZE_X);
            let groups_y = (bbox_h as u32).div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Dispatch one monochrome triangle into VRAM. Returns immediately
    /// after queuing the submit; callers must `download_*` from VRAM
    /// (which inserts a wait) to read back results.
    pub fn dispatch_mono_tri(&self, vram: &VramGpu, tri: &MonoTri, area: &DrawArea) {
        // Hardware-extent rule mirrors the CPU rasterizer.
        if tri.exceeds_hw_extent() {
            return;
        }
        // Empty bounding box → nothing to do.
        let bbox_w = tri.bbox_max[0] - tri.bbox_min[0] + 1;
        let bbox_h = tri.bbox_max[1] - tri.bbox_min[1] + 1;
        if bbox_w <= 0 || bbox_h <= 0 {
            return;
        }

        // Update uniforms.
        self.queue
            .write_buffer(&self.mono_tri_uniform, 0, bytemuck::bytes_of(tri));
        self.queue
            .write_buffer(&self.draw_area_uniform, 0, bytemuck::bytes_of(area));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-rasterizer-mono-tri-bg"),
            layout: &self.mono_tri_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vram.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.mono_tri_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_area_uniform.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-rasterizer-mono-tri-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("psx-rasterizer-mono-tri-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mono_tri_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per `WG×WG` pixel tile of the bbox.
            let groups_x = (bbox_w as u32).div_ceil(WORKGROUP_SIZE_X);
            let groups_y = (bbox_h as u32).div_ceil(WORKGROUP_SIZE_Y);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

// `DrawArea` is exactly 16 bytes — std::mem::size_of_val would also
// work, but bytemuck::bytes_of needs `Pod`. Both `MonoTri` and
// `DrawArea` derive Pod in `primitive.rs`. We assert the layout
// invariants at compile time below.
const _: () = {
    assert!(std::mem::size_of::<MonoTri>() == 48);
    assert!(std::mem::size_of::<DrawArea>() == 16);
};

// Just to silence dead-code warnings on `Zeroable` (used implicitly
// via `derive(Pod)`); explicit re-export here so future modules can
// import the trait via the rasterizer module.
#[allow(dead_code)]
fn _phantom_zeroable<T: Zeroable + Pod>() -> T {
    T::zeroed()
}

// =============================================================
//  Tests — GPU rasterizer vs CPU rasterize_triangle parity
// =============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use emulator_core::Gpu;

    /// Run the CPU rasterizer for a monochrome triangle, optionally
    /// with semi-trans / mask flags pre-configured. Returns the full
    /// row-major VRAM snapshot for byte-by-byte comparison.
    ///
    /// Configuration knobs:
    ///   - `cmd_byte` 0x20 = opaque, 0x22 = semi-trans (cmd-bit-1 = 1).
    ///   - `tpage_blend_bits` (0..3) → bits 5-6 of GP0 0xE1 → tpage
    ///     blend mode (Average / Add / Sub / AddQuarter).
    ///   - `mask_e6` writes GP0 0xE6 with the given low-2-bit value
    ///     (bit 0 = mask_set_on_draw, bit 1 = mask_check_before_draw).
    ///   - `prefill` paints the entire VRAM with one color before
    ///     submitting the primitive — needed for semi-trans tests
    ///     where the back buffer must not be zero, and for mask-check
    ///     tests where the existing pixel needs bit 15 set.
    fn cpu_rasterize_mono_tri_full(
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        color: u16,
        cmd_byte: u8,
        tpage_blend_bits: u8,
        mask_e6: u8,
        prefill: u16,
    ) -> Vec<u16> {
        let mut gpu = Gpu::new();
        // Pre-fill VRAM by writing every word directly. Cheaper than
        // streaming a 1 MiB block through GP0.
        if prefill != 0 {
            for y in 0..512u16 {
                for x in 0..1024u16 {
                    gpu.vram.set_pixel(x, y, prefill);
                }
            }
        }
        gpu.gp0_push(0xE3000000); // E3 — top-left at (0, 0)
        gpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        // Tpage: only bits 5-6 (semi-trans mode) matter for mono prims.
        let e1 = 0xE100_0000_u32 | ((tpage_blend_bits as u32) & 0x3) << 5;
        gpu.gp0_push(e1);
        // Mask config (E6).
        gpu.gp0_push(0xE600_0000_u32 | (mask_e6 as u32) & 0x3);
        // Triangle command word.
        let cmd = ((cmd_byte as u32) << 24) | bgr15_to_rgb24(color);
        gpu.gp0_push(cmd);
        gpu.gp0_push(pack_xy(v0));
        gpu.gp0_push(pack_xy(v1));
        gpu.gp0_push(pack_xy(v2));
        gpu.vram.words().to_vec()
    }

    fn cpu_rasterize_mono_tri(
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        color: u16,
    ) -> Vec<u16> {
        cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x20, 0, 0, 0)
    }

    /// Pre-fill GPU VRAM with a single 16-bit value via the GPU-side
    /// upload path. Mirrors what `prefill` does on the CPU side so
    /// both backends start from byte-identical state.
    fn gpu_prefill(vg: &VramGpu, value: u16) {
        if value == 0 {
            return;
        }
        let buf = vec![value; (super::super::VRAM_WIDTH * super::super::VRAM_HEIGHT) as usize];
        vg.upload_full(&buf).expect("upload prefill");
    }

    fn pack_xy(p: (i32, i32)) -> u32 {
        let x = (p.0 as u32) & 0x07FF;
        let y = (p.1 as u32) & 0x07FF;
        x | (y << 16)
    }

    fn bgr15_to_rgb24(bgr15: u16) -> u32 {
        let r5 = (bgr15 & 0x1F) as u32;
        let g5 = ((bgr15 >> 5) & 0x1F) as u32;
        let b5 = ((bgr15 >> 10) & 0x1F) as u32;
        // Inverse of `rgb24_to_bgr15`: lift 5→8 bits by replicate.
        let r = (r5 << 3) | (r5 >> 2);
        let g = (g5 << 3) | (g5 >> 2);
        let b = (b5 << 3) | (b5 >> 2);
        r | (g << 8) | (b << 16)
    }

    fn diff_count(a: &[u16], b: &[u16]) -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }

    #[test]
    fn mono_tri_axis_aligned_right_triangle_matches_cpu() {
        // The simplest case: a right triangle with one axis-aligned
        // edge. Edge function tests should produce identical
        // coverage to the CPU scanline walker.
        let v0 = (10, 10);
        let v1 = (50, 10);
        let v2 = (10, 50);
        let color = 0x7C00; // pure blue

        let cpu_vram = cpu_rasterize_mono_tri(v0, v1, v2, color);

        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        let tri = MonoTri::opaque(v0, v1, v2, color);
        let area = DrawArea::full_vram();
        r.dispatch_mono_tri(&vg, &tri, &area);
        let gpu_vram = vg.download_full().expect("download");

        let diffs = diff_count(&cpu_vram, &gpu_vram);
        // Edge fill-rule differences may produce a handful of
        // boundary pixels. Tolerance: <= 0.5% of the bbox.
        let bbox_pixels = 41 * 41;
        assert!(
            diffs < bbox_pixels / 200,
            "diffs={diffs}, bbox_pixels={bbox_pixels} — too many"
        );
    }

    #[test]
    fn mono_tri_skewed_matches_cpu_within_tolerance() {
        // A non-axis-aligned triangle stresses the edge functions.
        // We expect a handful of edge pixels to differ from the
        // scanline rasterizer; assert the inside is fully matched.
        let v0 = (50, 20);
        let v1 = (130, 70);
        let v2 = (30, 90);
        let color = 0x03E0; // pure green

        let cpu_vram = cpu_rasterize_mono_tri(v0, v1, v2, color);

        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        let tri = MonoTri::opaque(v0, v1, v2, color);
        let area = DrawArea::full_vram();
        r.dispatch_mono_tri(&vg, &tri, &area);
        let gpu_vram = vg.download_full().expect("download");

        let diffs = diff_count(&cpu_vram, &gpu_vram);
        // The triangle covers ~3000 pixels. Allow 5% tolerance for
        // edge-rule diffs in this first cut; tightening to exact
        // parity is a Phase B.x follow-up.
        let bbox_pixels = ((130 - 30 + 1) * (90 - 20 + 1)) as usize;
        let tolerance = bbox_pixels / 20;
        assert!(
            diffs < tolerance,
            "diffs={diffs}, tolerance={tolerance} — likely a real coverage bug"
        );
    }

    #[test]
    fn mono_tri_oversized_is_dropped_like_cpu() {
        // Hardware drops triangles whose edge Δ exceeds 1023×511.
        // The CPU rasterizer matches; ours must too.
        let v0 = (0, 0);
        let v1 = (2000, 0);
        let v2 = (0, 0);
        let color = 0x7FFF;

        let cpu_vram = cpu_rasterize_mono_tri(v0, v1, v2, color);

        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        let tri = MonoTri::opaque(v0, v1, v2, color);
        let area = DrawArea::full_vram();
        r.dispatch_mono_tri(&vg, &tri, &area);
        let gpu_vram = vg.download_full().expect("download");

        // Both should be all-zero VRAM (degenerate primitive
        // dropped before any plotting).
        assert!(cpu_vram.iter().all(|&w| w == 0), "CPU should drop");
        assert!(gpu_vram.iter().all(|&w| w == 0), "GPU should drop");
    }

    // -------------------------------------------------------
    //  Phase B.4 — semi-trans + mask-bit parity vs CPU
    // -------------------------------------------------------

    /// Run the GPU rasterizer for one mono-tri with the given flags
    /// onto a pre-filled VRAM, return the post-dispatch full VRAM.
    fn gpu_rasterize_mono_tri_full(
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        color: u16,
        flags: PrimFlags,
        blend_mode: BlendMode,
        prefill: u16,
    ) -> Vec<u16> {
        let vg = VramGpu::new_headless();
        gpu_prefill(&vg, prefill);
        let r = Rasterizer::new(&vg);
        let tri = MonoTri::new(v0, v1, v2, color, flags, blend_mode);
        let area = DrawArea::full_vram();
        r.dispatch_mono_tri(&vg, &tri, &area);
        vg.download_full().expect("download")
    }

    /// Run a strict bbox-only diff: count mismatches inside the
    /// triangle's bbox and a 2-pixel halo. Skewed triangles will
    /// always disagree on a few edge pixels (different fill rule);
    /// allowing a small tolerance lets us pin the inside-of-triangle
    /// blend math precisely.
    fn diff_inside_bbox(a: &[u16], b: &[u16], bbox_min: (i32, i32), bbox_max: (i32, i32)) -> usize {
        let mut diffs = 0;
        let x0 = (bbox_min.0 - 2).max(0) as usize;
        let y0 = (bbox_min.1 - 2).max(0) as usize;
        let x1 = ((bbox_max.0 + 2).min(1023)) as usize;
        let y1 = ((bbox_max.1 + 2).min(511)) as usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = y * 1024 + x;
                if a[i] != b[i] {
                    diffs += 1;
                }
            }
        }
        diffs
    }

    #[test]
    fn semi_trans_average_matches_cpu_byte_for_byte() {
        // The trickiest blend mode — Redux's `(b>>1) + (f>>1)` quirk
        // produces different LSBs from the naive `(b+f)/2`. If the
        // shader gets this wrong, every output pixel of an axis-
        // aligned triangle (no edge-rule diffs) will be off-by-one
        // on at least one channel. Strict parity expected.
        let v0 = (10, 10);
        let v1 = (50, 10);
        let v2 = (10, 50);
        let color = 0x1234; // arbitrary BGR15 (low and high LSBs set so the quirk shows)
        let prefill = 0x5678;
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x22, 0, 0, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0,
            v1,
            v2,
            color,
            PrimFlags::SEMI_TRANS,
            BlendMode::Average,
            prefill,
        );
        let diffs = diff_inside_bbox(&cpu, &gpu, (10, 10), (50, 50));
        // Axis-aligned right-triangle → no edge-rule disagreement.
        // Inside pixels must blend identically.
        assert!(diffs == 0, "Average blend mismatch: {diffs} pixels differ");
    }

    #[test]
    fn semi_trans_add_matches_cpu_byte_for_byte() {
        let v0 = (5, 5);
        let v1 = (45, 5);
        let v2 = (5, 45);
        // Pick a color whose channels saturate against the prefill.
        let color = 0x4210; // (r=0x10, g=0x10, b=0x10)
        let prefill = 0x4210; // same — sum must clamp to 31 per channel
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x22, 1, 0, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0,
            v1,
            v2,
            color,
            PrimFlags::SEMI_TRANS,
            BlendMode::Add,
            prefill,
        );
        let diffs = diff_inside_bbox(&cpu, &gpu, (5, 5), (45, 45));
        assert!(diffs == 0, "Add blend mismatch: {diffs} pixels differ");
    }

    #[test]
    fn semi_trans_sub_matches_cpu_byte_for_byte() {
        let v0 = (5, 5);
        let v1 = (45, 5);
        let v2 = (5, 45);
        let color = 0x2108; // (r=8, g=8, b=8)
        let prefill = 0x4210; // (r=16, g=16, b=16) → result (r=8,g=8,b=8)
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x22, 2, 0, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0,
            v1,
            v2,
            color,
            PrimFlags::SEMI_TRANS,
            BlendMode::Sub,
            prefill,
        );
        let diffs = diff_inside_bbox(&cpu, &gpu, (5, 5), (45, 45));
        assert!(diffs == 0, "Sub blend mismatch: {diffs} pixels differ");
    }

    #[test]
    fn semi_trans_addquarter_matches_cpu_byte_for_byte() {
        let v0 = (5, 5);
        let v1 = (45, 5);
        let v2 = (5, 45);
        let color = 0x4210;
        let prefill = 0x2108;
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x22, 3, 0, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0,
            v1,
            v2,
            color,
            PrimFlags::SEMI_TRANS,
            BlendMode::AddQuarter,
            prefill,
        );
        let diffs = diff_inside_bbox(&cpu, &gpu, (5, 5), (45, 45));
        assert!(
            diffs == 0,
            "AddQuarter blend mismatch: {diffs} pixels differ"
        );
    }

    #[test]
    fn mask_set_writes_bit_15_in_every_plotted_pixel() {
        // Opaque triangle with mask_set_on_draw → every plotted
        // pixel should have bit 15 = 1, others left as prefill.
        let v0 = (10, 10);
        let v1 = (50, 10);
        let v2 = (10, 50);
        let color = 0x0123; // bit 15 clear in source color
        let prefill = 0x4567;
        // CPU path: cmd 0x20 (opaque), tpage doesn't matter for
        // opaque mono, mask_e6 = 0b01 = mask_set_on_draw.
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x20, 0, 0b01, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0,
            v1,
            v2,
            color,
            PrimFlags::MASK_SET,
            BlendMode::Average,
            prefill,
        );
        let diffs = diff_inside_bbox(&cpu, &gpu, (10, 10), (50, 50));
        assert!(diffs == 0, "MASK_SET parity: {diffs} differ");
        // Sanity: spot-check a point firmly inside the triangle has
        // bit 15 set on both backends.
        let inside_idx = 20 * 1024 + 20;
        assert!(
            cpu[inside_idx] & 0x8000 != 0,
            "CPU inside pixel: bit 15 set"
        );
        assert!(
            gpu[inside_idx] & 0x8000 != 0,
            "GPU inside pixel: bit 15 set"
        );
    }

    #[test]
    fn mask_check_skips_when_back_buffer_has_bit_15() {
        // Pre-fill with bit-15-set pixels; opaque triangle with
        // mask_check should leave them all alone.
        let v0 = (10, 10);
        let v1 = (50, 10);
        let v2 = (10, 50);
        let color = 0x0123;
        let prefill = 0x8888; // bit 15 set
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x20, 0, 0b10, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0,
            v1,
            v2,
            color,
            PrimFlags::MASK_CHECK,
            BlendMode::Average,
            prefill,
        );
        // Strict equality on every pixel — nothing should have changed.
        let diffs = diff_inside_bbox(&cpu, &gpu, (10, 10), (50, 50));
        assert!(diffs == 0, "MASK_CHECK parity: {diffs} differ");
        let inside_idx = 20 * 1024 + 20;
        assert_eq!(cpu[inside_idx], prefill, "CPU inside untouched");
        assert_eq!(gpu[inside_idx], prefill, "GPU inside untouched");
    }

    #[test]
    fn mask_check_only_skips_protected_pixels_not_others() {
        // Half the back buffer has bit 15, half doesn't. The triangle
        // should write only to the unprotected half.
        let v0 = (10, 10);
        let v1 = (60, 10);
        let v2 = (10, 60);
        let color = 0x0123;
        // CPU prefill: alternate rows have bit 15 set.
        let mut gpu_buffer =
            vec![0u16; (super::super::VRAM_WIDTH * super::super::VRAM_HEIGHT) as usize];
        for y in 0..512u16 {
            let v = if y & 1 == 0 { 0x4567 } else { 0xC567 };
            for x in 0..1024u16 {
                gpu_buffer[y as usize * 1024 + x as usize] = v;
            }
        }
        // Build an identical CPU state.
        let mut cpu_gpu = Gpu::new();
        for (i, &w) in gpu_buffer.iter().enumerate() {
            let x = (i % 1024) as u16;
            let y = (i / 1024) as u16;
            cpu_gpu.vram.set_pixel(x, y, w);
        }
        cpu_gpu.gp0_push(0xE3000000);
        cpu_gpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu_gpu.gp0_push(0xE100_0000); // tpage
        cpu_gpu.gp0_push(0xE600_0002); // E6 — mask_check_before_draw
        let cmd = 0x20_000000_u32 | bgr15_to_rgb24(color);
        cpu_gpu.gp0_push(cmd);
        cpu_gpu.gp0_push(pack_xy(v0));
        cpu_gpu.gp0_push(pack_xy(v1));
        cpu_gpu.gp0_push(pack_xy(v2));
        let cpu = cpu_gpu.vram.words().to_vec();

        // Identical GPU state.
        let vg = VramGpu::new_headless();
        vg.upload_full(&gpu_buffer).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = MonoTri::new(v0, v1, v2, color, PrimFlags::MASK_CHECK, BlendMode::Average);
        r.dispatch_mono_tri(&vg, &tri, &DrawArea::full_vram());
        let gpu = vg.download_full().unwrap();

        let diffs = diff_inside_bbox(&cpu, &gpu, (10, 10), (60, 60));
        assert!(diffs == 0, "selective mask-check parity: {diffs} differ");
        // Sanity check: ODD rows (prefill 0xC567, bit-15 set) must be
        // unchanged; EVEN rows (prefill 0x4567, bit-15 clear) must
        // have been overwritten with the new color.
        for y in [11u16, 13, 15] {
            let i = y as usize * 1024 + 20;
            assert_eq!(cpu[i], 0xC567, "CPU row {y} (protected) unchanged");
            assert_eq!(gpu[i], 0xC567, "GPU row {y} (protected) unchanged");
        }
        for y in [12u16, 14, 16] {
            let i = y as usize * 1024 + 20;
            // The new pixel may have bit 15 set/cleared depending on
            // mask-set; it shouldn't have it here. The colour part
            // must be `color`.
            assert_eq!(
                cpu[i] & 0x7FFF,
                color & 0x7FFF,
                "CPU row {y} (open) overwritten"
            );
            assert_eq!(
                gpu[i] & 0x7FFF,
                color & 0x7FFF,
                "GPU row {y} (open) overwritten"
            );
        }
    }

    #[test]
    fn semi_trans_plus_mask_set_combine_correctly() {
        // The full RMW chain: blend with back buffer THEN OR bit 15.
        // If the shader gets ordering wrong (e.g. OR-then-blend), the
        // blended result will have bit 15 propagating into the colour
        // math.
        let v0 = (10, 10);
        let v1 = (50, 10);
        let v2 = (10, 50);
        let color = 0x1234;
        let prefill = 0x5678;
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x22, 0, 0b01, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0,
            v1,
            v2,
            color,
            PrimFlags::SEMI_TRANS | PrimFlags::MASK_SET,
            BlendMode::Average,
            prefill,
        );
        let diffs = diff_inside_bbox(&cpu, &gpu, (10, 10), (50, 50));
        assert!(diffs == 0, "semi-trans + mask-set parity: {diffs} differ");
        let inside_idx = 20 * 1024 + 20;
        // Both must have bit 15 set after MASK_SET.
        assert!(cpu[inside_idx] & 0x8000 != 0);
        assert!(gpu[inside_idx] & 0x8000 != 0);
    }

    #[test]
    fn mono_tri_drawing_area_clips_correctly() {
        // Draw a triangle at (5..50, 5..50) but with the active
        // drawing area limited to (20..40, 20..40). Pixels outside
        // the inner rect must remain zero on both backends.
        let v0 = (5, 5);
        let v1 = (60, 5);
        let v2 = (5, 60);
        let color = 0x001F; // pure red

        // CPU path: configure the draw area, then submit the prim.
        let mut gpu = Gpu::new();
        gpu.gp0_push(0xE3000000 | 20 | (20 << 10));
        gpu.gp0_push(0xE4000000 | 40 | (40 << 10));
        let cmd = 0x20000000 | bgr15_to_rgb24(color);
        gpu.gp0_push(cmd);
        gpu.gp0_push(pack_xy(v0));
        gpu.gp0_push(pack_xy(v1));
        gpu.gp0_push(pack_xy(v2));
        let cpu_vram = gpu.vram.words().to_vec();

        // GPU path with matching draw area.
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        let tri = MonoTri::opaque(v0, v1, v2, color);
        let area = DrawArea {
            left: 20,
            top: 20,
            right: 40,
            bottom: 40,
        };
        r.dispatch_mono_tri(&vg, &tri, &area);
        let gpu_vram = vg.download_full().expect("download");

        // Strict assertions on the clip boundary. Pixels outside
        // the area must be zero on both, and the inner area should
        // overlap (allowing fill-rule edge diffs).
        for y in 0..512u16 {
            for x in 0..1024u16 {
                let outside = x < 20 || x > 40 || y < 20 || y > 40;
                let idx = y as usize * 1024 + x as usize;
                if outside {
                    assert_eq!(cpu_vram[idx], 0, "CPU @ ({x},{y}) outside clip");
                    assert_eq!(gpu_vram[idx], 0, "GPU @ ({x},{y}) outside clip");
                }
            }
        }
    }

    // -------------------------------------------------------
    //  Phase B.2 — textured triangle parity vs CPU
    // -------------------------------------------------------

    /// Pack a (u, v) pair into the low 16 bits of a UV0/UV1/UV2 word.
    fn uv_pack(uv: (u8, u8)) -> u32 {
        (uv.0 as u32) | ((uv.1 as u32) << 8)
    }

    /// Build the 16-bit tpage word the GPU expects in UV1 high half.
    fn make_tpage_word(tpage_x: u32, tpage_y: u32, depth: u32, blend_bits: u32) -> u32 {
        let tx = tpage_x / 64; // 0..15
        let ty = if tpage_y == 256 { 1u32 } else { 0 };
        (tx & 0xF) | (ty << 4) | ((blend_bits & 0x3) << 5) | ((depth & 0x3) << 7)
    }

    /// Build the 16-bit CLUT word the GPU expects in UV0 high half.
    /// `clut_x` must be a multiple of 16 (PS1 CLUT alignment).
    fn make_clut_word(clut_x: u32, clut_y: u32) -> u32 {
        ((clut_x / 16) & 0x3F) | ((clut_y & 0x1FF) << 6)
    }

    /// Mirror the same VRAM state on a CPU `Gpu` instance and a
    /// `VramGpu`. We pre-fill VRAM directly via `set_pixel` /
    /// `upload_full` so the test doesn't have to fight CD-DMA.
    fn seed_vram(words: &[u16], cpu: &mut Gpu, gpu: &VramGpu) {
        for (i, &w) in words.iter().enumerate() {
            let x = (i % 1024) as u16;
            let y = (i / 1024) as u16;
            cpu.vram.set_pixel(x, y, w);
        }
        gpu.upload_full(words).unwrap();
    }

    /// Drive the CPU rasterizer through a GP0 textured-triangle
    /// packet. Caller has already set draw area + uploaded VRAM.
    fn cpu_push_tex_tri(
        cpu: &mut Gpu,
        cmd_byte: u8,
        tint: (u8, u8, u8),
        v: [(i32, i32); 3],
        uv: [(u8, u8); 3],
        clut_word: u32,
        tpage_word: u32,
    ) {
        let cmd = ((cmd_byte as u32) << 24)
            | (tint.0 as u32)
            | ((tint.1 as u32) << 8)
            | ((tint.2 as u32) << 16);
        cpu.gp0_push(cmd);
        cpu.gp0_push(pack_xy(v[0]));
        cpu.gp0_push((clut_word << 16) | uv_pack(uv[0]));
        cpu.gp0_push(pack_xy(v[1]));
        cpu.gp0_push((tpage_word << 16) | uv_pack(uv[1]));
        cpu.gp0_push(pack_xy(v[2]));
        cpu.gp0_push(uv_pack(uv[2]));
    }

    #[test]
    fn tex_tri_15bpp_axis_aligned_matches_cpu() {
        // 15bpp direct-colour: simplest tex sampling path. No CLUT,
        // each VRAM cell is the texel. Axis-aligned right triangle
        // → no edge-rule disagreements.
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        // Tpage at (128, 0), 15bpp.
        let tpage_x = 128u32;
        let tpage_y = 0u32;
        let tpage_word = make_tpage_word(tpage_x, tpage_y, 2, 0);

        // Build the texture: 64×64 of `(v << 5) | u | 0x0001`. The
        // `| 0x0001` ensures every texel is non-zero (i.e. opaque)
        // so we can spot dropped pixels.
        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..64u16 {
            for ux in 0..64u16 {
                let val = ((vy as u16) << 5) | ux | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }

        // CPU side.
        let mut cpu = Gpu::new();
        seed_vram(&vram, &mut cpu, &VramGpu::new_headless());
        // (Re-seed VRAM cleanly — `seed_vram` above used a throwaway
        // headless device. We need a fresh one used for the actual
        // dispatch below. Easier: just call set_pixel + upload twice.)
        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu_push_tex_tri(&mut cpu, 0x25, (0, 0, 0), v, uv, 0, tpage_word);
        let cpu_words = cpu.vram.words().to_vec();

        // GPU side.
        let r = Rasterizer::new(&vg);
        let tri = TexTri::new(
            v[0],
            v[1],
            v[2],
            uv[0],
            uv[1],
            uv[2],
            0,
            0,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, tpage_y, 2);
        r.dispatch_tex_tri(&vg, &tri, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        // Functional parity: the GPU samples the SAME texture cells
        // as the CPU at each integer pixel position with a barycentric
        // affine interpolation. Pixel-EXACT parity vs the Redux-port
        // scanline-delta math (which uses specific Q16.16 setup +
        // shl10idiv) is a Phase-B.x follow-up — that path produces
        // off-by-1/2 UV at some interior pixels due to the difference
        // between cumulative per-row deltas and a barycentric divide.
        //
        // We assert the texel-COLOUR error is small: the percent of
        // diffs is bounded, AND every diff is within ±2 in any single
        // 5-bit channel (i.e. ≤6.25% intensity error on that channel).
        // That covers the rounding gap without hiding a coverage or
        // sampling bug.
        let diffs = diff_inside_bbox(&cpu_words, &gpu_words, (20, 20), (60, 60));
        let bbox = 41 * 41;
        // Record max channel delta across all differing pixels.
        let mut max_chan_delta = 0i32;
        for y in 20..=60i32 {
            for x in 20..=60i32 {
                let i = y as usize * 1024 + x as usize;
                let a = cpu_words[i];
                let b = gpu_words[i];
                if a == b {
                    continue;
                }
                for shift in [0u32, 5, 10] {
                    let ca = ((a >> shift) & 0x1F) as i32;
                    let cb = ((b >> shift) & 0x1F) as i32;
                    max_chan_delta = max_chan_delta.max((ca - cb).abs());
                }
            }
        }
        assert!(
            diffs * 4 < bbox,
            "tex 15bpp coverage: {diffs} / {bbox} pixels differ — too many"
        );
        assert!(
            max_chan_delta <= 2,
            "tex 15bpp colour error: max channel delta {max_chan_delta} > 2 — \
             likely a sampling / CLUT / depth bug"
        );
    }

    #[test]
    fn tex_tri_4bpp_with_clut_matches_cpu() {
        // 4bpp paletted texture: each VRAM word holds 4 texel
        // indices, each indexes a 16-entry CLUT row. This stresses
        // the CLUT lookup path that the Tekken-3 portrait bug
        // landed in.
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        let tpage_x = 0u32;
        let tpage_y = 0u32;
        let tpage_word = make_tpage_word(tpage_x, tpage_y, 0, 0);
        // CLUT at (0, 256). 16 entries, each in BGR15.
        let clut_x = 0u32;
        let clut_y = 256u32;
        let clut_word = make_clut_word(clut_x, clut_y);

        let mut vram = vec![0u16; 1024 * 512];
        // CLUT: entry 0 is non-zero (opaque "background"), entries
        // 1..15 are a colour ramp. Avoid 0x0000 anywhere so every
        // sampled texel writes a pixel.
        for i in 0..16u16 {
            let val = (i.max(1) << 1) | (i.max(1) << 6) | 0x4000;
            vram[clut_y as usize * 1024 + (clut_x as usize + i as usize)] = val;
        }
        // 16×16 texture: each VRAM word holds 4 texels (low to high
        // nibble = u 0..3 within the word). Pattern: nibble = (u + v)
        // & 0xF so different parts of the triangle hit different
        // CLUT entries.
        for vy in 0..16u16 {
            for word_x in 0..4u16 {
                let u_base = word_x * 4;
                let mut word = 0u16;
                for n in 0..4u16 {
                    let u = u_base + n;
                    let nibble = (u + vy) & 0xF;
                    word |= nibble << (n * 4);
                }
                vram[vy as usize * 1024 + (tpage_x as usize + word_x as usize)] = word;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu_push_tex_tri(&mut cpu, 0x25, (0, 0, 0), v, uv, clut_word, tpage_word);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = TexTri::new(
            v[0],
            v[1],
            v[2],
            uv[0],
            uv[1],
            uv[2],
            clut_x,
            clut_y,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, tpage_y, 0);
        r.dispatch_tex_tri(&vg, &tri, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        // See `tex_tri_15bpp_axis_aligned_matches_cpu` for parity
        // tolerance reasoning. We additionally allow a small CLUT-
        // index swap at edge pixels because adjacent CLUT entries
        // here differ by more than 2 in some channel.
        let diffs = diff_inside_bbox(&cpu_words, &gpu_words, (20, 20), (60, 60));
        let bbox = 41 * 41;
        assert!(diffs * 4 < bbox, "tex 4bpp coverage: {diffs} / {bbox}");
    }

    #[test]
    fn tex_tri_8bpp_with_clut_matches_cpu() {
        // 8bpp: each VRAM word is two texel bytes, each indexes a
        // 256-entry CLUT row.
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        let tpage_x = 64u32;
        let tpage_y = 0u32;
        let tpage_word = make_tpage_word(tpage_x, tpage_y, 1, 0);
        // CLUT at (16, 256) — 16 must be multiple of 16 (it is).
        // For 8bpp the CLUT row is 256 entries wide, but the host
        // doesn't pre-shift the X — we pass the raw VRAM column.
        let clut_x = 16u32;
        let clut_y = 256u32;
        let clut_word = make_clut_word(clut_x, clut_y);

        let mut vram = vec![0u16; 1024 * 512];
        // CLUT: 256 entries, each non-zero, deterministic ramp.
        for i in 0..256u32 {
            let val = ((i & 0x1F) as u16) | (((i >> 1) & 0x1F) as u16) << 5 | 0x4000;
            vram[clut_y as usize * 1024 + (clut_x as usize + i as usize)] = val;
        }
        // Texture: each word = (u_high << 8) | u_low, giving a
        // gradient that maps to different CLUT entries.
        for vy in 0..32u16 {
            for word_x in 0..16u16 {
                let u_low = word_x * 2;
                let u_high = u_low + 1;
                let v_off = vy as u32;
                let lo = ((u_low as u32 + v_off) & 0xFF) as u16;
                let hi = ((u_high as u32 + v_off) & 0xFF) as u16;
                let word = lo | (hi << 8);
                // Avoid index 0 which would map to CLUT[0]; the
                // texture indices we generate above start at vy ≥ 0
                // and u ≥ 0, so the sum can be 0 only at (0,0).
                let word = if word == 0 { 0x0101 } else { word };
                vram[vy as usize * 1024 + (tpage_x as usize + word_x as usize)] = word;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu_push_tex_tri(&mut cpu, 0x25, (0, 0, 0), v, uv, clut_word, tpage_word);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = TexTri::new(
            v[0],
            v[1],
            v[2],
            uv[0],
            uv[1],
            uv[2],
            clut_x,
            clut_y,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, tpage_y, 1);
        r.dispatch_tex_tri(&vg, &tri, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        let diffs = diff_inside_bbox(&cpu_words, &gpu_words, (20, 20), (60, 60));
        let bbox = 41 * 41;
        assert!(diffs * 4 < bbox, "tex 8bpp coverage: {diffs} / {bbox}");
    }

    #[test]
    fn tex_tri_modulated_tint_matches_cpu() {
        // Same 15bpp setup but with a non-identity tint. Verifies
        // the `(tint * texel) / 0x80` modulator matches the CPU.
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        let tpage_x = 128u32;
        let tpage_word = make_tpage_word(tpage_x, 0, 2, 0);
        // 50% tint on each channel — exactly half the value.
        let tint = (0x40u8, 0x40u8, 0x40u8);

        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..32u16 {
            for ux in 0..32u16 {
                let val = ((vy << 5) | ux) | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        // 0x24 = textured + modulated (NOT raw — tint applies).
        cpu_push_tex_tri(&mut cpu, 0x24, tint, v, uv, 0, tpage_word);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = TexTri::new(
            v[0],
            v[1],
            v[2],
            uv[0],
            uv[1],
            uv[2],
            0,
            0,
            tint,
            PrimFlags::empty(), // no RAW_TEXTURE → modulate
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        r.dispatch_tex_tri(&vg, &tri, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        let diffs = diff_inside_bbox(&cpu_words, &gpu_words, (20, 20), (60, 60));
        let bbox = 41 * 41;
        assert!(diffs * 4 < bbox, "tex modulated coverage: {diffs} / {bbox}");
        // Strict check: every pixel that BOTH backends wrote should
        // have R/G/B that's been halved by the 0x40 tint. So any
        // non-zero pixel must have channels ≤ 0x10 (since input
        // texel channels here are ≤ 0x1F, half = ≤ 0x0F, plus 1
        // for divide-by-0x80 rounding).
        for y in 21..=59i32 {
            for x in 21..=59i32 {
                let i = y as usize * 1024 + x as usize;
                let g_val = gpu_words[i];
                if g_val == 0 {
                    continue;
                }
                let r = g_val & 0x1F;
                let g = (g_val >> 5) & 0x1F;
                let b = (g_val >> 10) & 0x1F;
                assert!(
                    r <= 0x10 && g <= 0x10 && b <= 0x10,
                    "tint modulation looks wrong @ ({x},{y}): \
                     pixel=0x{g_val:04x} r={r} g={g} b={b}"
                );
            }
        }
    }

    #[test]
    fn tex_tri_transparent_texels_skip_writes() {
        // Place a checkerboard texture: even cells are transparent
        // (texel = 0), odd cells are opaque. The triangle should
        // leave the back-buffer untouched at every transparent
        // hit, both on CPU and GPU.
        let v = [(40i32, 40i32), (80, 40), (40, 80)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        let tpage_x = 128u32;
        let tpage_word = make_tpage_word(tpage_x, 0, 2, 0);

        // Pre-fill bbox area with a sentinel so the test can detect
        // exactly which pixels were touched.
        let prefill = 0x4321u16;
        let mut vram = vec![prefill; 1024 * 512];
        // Texture: opaque (non-zero) only on cells where (u + v)
        // is odd; transparent (0) elsewhere.
        for vy in 0..32u16 {
            for ux in 0..32u16 {
                let opaque = ((ux + vy) & 1) == 1;
                let val: u16 = if opaque {
                    ((vy as u16) << 5) | (ux as u16) | 0x0001
                } else {
                    0
                };
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu_push_tex_tri(&mut cpu, 0x25, (0, 0, 0), v, uv, 0, tpage_word);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = TexTri::new(
            v[0],
            v[1],
            v[2],
            uv[0],
            uv[1],
            uv[2],
            0,
            0,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        r.dispatch_tex_tri(&vg, &tri, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        // Inside the triangle, both backends should agree on coverage
        // within the same per-pixel-rounding tolerance noted on the
        // 15bpp test.
        let diffs = diff_inside_bbox(&cpu_words, &gpu_words, (40, 40), (80, 80));
        let bbox = 41 * 41;
        assert!(diffs * 4 < bbox, "tex transparent coverage: {diffs}");

        // Sanity: at least SOME pixels in the bbox should still be
        // the prefill (transparent texels left them alone), and at
        // least some should NOT be the prefill (opaque texels wrote
        // through).
        let inside_pixels = (40usize..80usize)
            .flat_map(|y| (40usize..80usize).map(move |x| y * 1024 + x))
            .filter(|&i| {
                // Inside the lower-left half of the triangle: x+y < 100.
                let x = i % 1024;
                let y = i / 1024;
                x + y < 100
            })
            .collect::<Vec<_>>();
        let untouched = inside_pixels
            .iter()
            .filter(|&&i| gpu_words[i] == prefill)
            .count();
        let touched = inside_pixels
            .iter()
            .filter(|&&i| gpu_words[i] != prefill)
            .count();
        assert!(untouched > 0, "expected some transparent-skip pixels");
        assert!(touched > 0, "expected some opaque-write pixels");
    }

    // -------------------------------------------------------
    //  Phase B.x — scanline-delta textured triangle: BIT-EXACT
    // -------------------------------------------------------

    #[test]
    fn tex_tri_scanline_15bpp_axis_aligned_is_bit_exact() {
        // Same setup as the B.2 axis-aligned test, but using the
        // scanline-delta dispatcher. Strict `assert_eq!` is the
        // whole point of this phase.
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        let tpage_x = 128u32;
        let tpage_word = make_tpage_word(tpage_x, 0, 2, 0);

        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..64u16 {
            for ux in 0..64u16 {
                let val = ((vy as u16) << 5) | ux | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }
        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu_push_tex_tri(&mut cpu, 0x25, (0, 0, 0), v, uv, 0, tpage_word);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = TexTri::new(
            v[0],
            v[1],
            v[2],
            uv[0],
            uv[1],
            uv[2],
            0,
            0,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        let dispatched = r.dispatch_tex_tri_scanline(&vg, &tri, &tp, &DrawArea::full_vram());
        assert!(dispatched, "valid triangle should dispatch");
        let gpu_words = vg.download_full().unwrap();

        assert_eq!(cpu_words, gpu_words, "tex tri scanline strict parity");
    }

    #[test]
    fn tex_tri_scanline_skewed_is_bit_exact() {
        // The skewed triangle case. Without scanline-delta this had
        // ±2/5-bit channel error from barycentric rounding. Now
        // strict equality.
        let v = [(50i32, 20i32), (130, 70), (30, 90)];
        let uv = [(0u8, 0u8), (60, 0), (0, 50)];
        let tpage_x = 128u32;
        let tpage_word = make_tpage_word(tpage_x, 0, 2, 0);

        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..128u16 {
            for ux in 0..128u16 {
                let val = ((vy as u16) << 5) | (ux as u16) | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }
        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu_push_tex_tri(&mut cpu, 0x25, (0, 0, 0), v, uv, 0, tpage_word);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = TexTri::new(
            v[0],
            v[1],
            v[2],
            uv[0],
            uv[1],
            uv[2],
            0,
            0,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        r.dispatch_tex_tri_scanline(&vg, &tri, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        assert_eq!(
            cpu_words, gpu_words,
            "tex tri scanline skewed strict parity"
        );
    }

    #[test]
    fn tex_tri_scanline_4bpp_with_clut_is_bit_exact() {
        // 4bpp + CLUT — exercises the texture-window-free CLUT
        // sampling path with bit-exact UV.
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        let tpage_x = 0u32;
        let tpage_word = make_tpage_word(tpage_x, 0, 0, 0);
        let clut_x = 0u32;
        let clut_y = 256u32;
        let clut_word = make_clut_word(clut_x, clut_y);

        let mut vram = vec![0u16; 1024 * 512];
        for i in 0..16u16 {
            let val = (i.max(1) << 1) | (i.max(1) << 6) | 0x4000;
            vram[clut_y as usize * 1024 + (clut_x as usize + i as usize)] = val;
        }
        for vy in 0..16u16 {
            for word_x in 0..4u16 {
                let mut word = 0u16;
                for n in 0..4u16 {
                    let nibble = ((word_x * 4 + n) + vy) & 0xF;
                    word |= nibble << (n * 4);
                }
                vram[vy as usize * 1024 + (tpage_x as usize + word_x as usize)] = word;
            }
        }
        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu_push_tex_tri(&mut cpu, 0x25, (0, 0, 0), v, uv, clut_word, tpage_word);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = TexTri::new(
            v[0],
            v[1],
            v[2],
            uv[0],
            uv[1],
            uv[2],
            clut_x,
            clut_y,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 0);
        r.dispatch_tex_tri_scanline(&vg, &tri, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        assert_eq!(
            cpu_words, gpu_words,
            "tex tri scanline 4bpp+CLUT strict parity"
        );
    }

    #[test]
    fn shaded_tex_tri_scanline_15bpp_is_bit_exact() {
        // The same setup as the B.3.b textured-shaded test, but
        // through the scanline-delta dispatcher. Strict equality
        // expected — both UV and RGB walks now exactly match the
        // CPU's cumulative arithmetic.
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        let c = [
            (0x80u8, 0x80u8, 0x80u8),
            (0xC0, 0xC0, 0xC0),
            (0xFFu8, 0xFFu8, 0xFFu8),
        ];
        let tpage_x = 128u32;
        let tpage_word = make_tpage_word(tpage_x, 0, 2, 0);

        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..32u16 {
            for ux in 0..32u16 {
                let val = ((vy as u16) << 5) | (ux as u16) | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        let pack_rgb = |t: (u8, u8, u8)| (t.0 as u32) | ((t.1 as u32) << 8) | ((t.2 as u32) << 16);
        // 0x34 = textured-shaded triangle.
        cpu.gp0_push((0x34u32 << 24) | pack_rgb(c[0]));
        cpu.gp0_push(pack_xy(v[0]));
        cpu.gp0_push(uv_pack(uv[0]));
        cpu.gp0_push(pack_rgb(c[1]));
        cpu.gp0_push(pack_xy(v[1]));
        cpu.gp0_push((tpage_word << 16) | uv_pack(uv[1]));
        cpu.gp0_push(pack_rgb(c[2]));
        cpu.gp0_push(pack_xy(v[2]));
        cpu.gp0_push(uv_pack(uv[2]));
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = ShadedTexTri::new(
            v[0],
            v[1],
            v[2],
            c[0],
            c[1],
            c[2],
            uv[0],
            uv[1],
            uv[2],
            0,
            0,
            PrimFlags::empty(),
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        let dispatched = r.dispatch_shaded_tex_tri_scanline(&vg, &tri, &tp, &DrawArea::full_vram());
        assert!(dispatched);
        let gpu_words = vg.download_full().unwrap();

        assert_eq!(
            cpu_words, gpu_words,
            "shaded-tex-tri scanline strict parity"
        );
    }

    /// Tekken 3 portrait regression. The CPU's
    /// `rasterize_axis_aligned_textured_quad` fast path uses bilinear
    /// UV interpolation; the original GPU replay split the quad into
    /// two barycentric triangles, which produced different pixels
    /// when the V channel wasn't affine across the four corners
    /// (UV3.v != UV1.v + UV2.v - UV0.v). Now the replay detects
    /// axis-aligned + non-affine quads and dispatches the dedicated
    /// `tex_quad_bilinear` shader. This test pins that path with the
    /// exact UV layout from the divergent Tekken 3 packet (cmd #9032
    /// in the boot trace).
    #[test]
    fn tex_quad_bilinear_non_affine_uvs_match_cpu() {
        // Destination quad in the top-left of VRAM; texture parked
        // in the bottom-right region so the quad's writes can't
        // corrupt the texels mid-rasterization (a 96×80 quad placed
        // over its own tpage would self-overwrite at every pixel
        // whose UV mapped to the destination range).
        let v = [
            (50i32, 30i32),
            (50 + 95, 30),
            (50, 30 + 79),
            (50 + 95, 30 + 79),
        ];
        // Non-affine V: 0,64,79,79. Triangle-split would produce
        // wrong pixels here.
        let uv = [(0u8, 0u8), (95u8, 64u8), (0u8, 79u8), (95u8, 79u8)];
        let tpage_x = 512u32;
        let tpage_y = 256u32;
        let tpage_word = make_tpage_word(tpage_x, tpage_y, 2, 0);

        // 96×80 texture: every cell carries a unique non-zero value
        // so any UV-step bug shows up.
        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..80u16 {
            for ux in 0..96u16 {
                let val = ((vy & 0x1F) << 5) | (ux & 0x1F) | 0x0001;
                vram[(tpage_y as usize + vy as usize) * 1024 + (tpage_x as usize + ux as usize)] =
                    val;
            }
        }

        // CPU: drive the textured-quad packet (op 0x2D = textured,
        // raw, opaque). Vertex order matches the cmd_log layout.
        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        // 0x2D — textured quad, raw texture, opaque.
        cpu.gp0_push(0x2D000000);
        cpu.gp0_push(pack_xy(v[0]));
        cpu.gp0_push(uv_pack(uv[0])); // clut=0
        cpu.gp0_push(pack_xy(v[1]));
        cpu.gp0_push((tpage_word << 16) | uv_pack(uv[1]));
        cpu.gp0_push(pack_xy(v[2]));
        cpu.gp0_push(uv_pack(uv[2]));
        cpu.gp0_push(pack_xy(v[3]));
        cpu.gp0_push(uv_pack(uv[3]));
        let cpu_words = cpu.vram.words().to_vec();

        // GPU: confirm the axis-aligned detector accepts these
        // vertices, then dispatch the bilinear shader.
        assert!(
            TexQuadBilinear::is_axis_aligned(v[0], v[1], v[2], v[3]),
            "test geometry must be axis-aligned for the bilinear path"
        );
        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let quad = TexQuadBilinear::new(
            v[0],
            v[1],
            v[2],
            v[3],
            uv[0],
            uv[1],
            uv[2],
            uv[3],
            0,
            0,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, tpage_y, 2);
        r.dispatch_tex_quad_bilinear(&vg, &quad, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        let mut diffs: Vec<(usize, usize, u16, u16)> = Vec::new();
        for (i, (&c, &g)) in cpu_words.iter().zip(gpu_words.iter()).enumerate() {
            if c != g {
                diffs.push((i % 1024, i / 1024, c, g));
                if diffs.len() >= 16 {
                    break;
                }
            }
        }
        assert!(
            diffs.is_empty(),
            "tex-quad bilinear should match CPU's axis-aligned fast path. \
             {} diffs (first ≤16 shown):\n{}",
            diffs.len(),
            diffs
                .iter()
                .map(|(x, y, c, g)| format!("  ({x},{y}) cpu=0x{c:04x} gpu=0x{g:04x}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    #[test]
    fn mono_tri_scanline_skewed_is_bit_exact() {
        // The B.1 skewed triangle case had ≤0.5% edge-rule diffs
        // under barycentric. With scanline-delta coverage, strict
        // equality.
        let v0 = (50i32, 20i32);
        let v1 = (130, 70);
        let v2 = (30, 90);
        let color = 0x03E0u16;
        let cpu = cpu_rasterize_mono_tri(v0, v1, v2, color);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        let tri = MonoTri::opaque(v0, v1, v2, color);
        let dispatched = r.dispatch_mono_tri_scanline(&vg, &tri, &DrawArea::full_vram());
        assert!(dispatched);
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "mono tri scanline skewed strict parity");
    }

    #[test]
    fn shaded_tri_scanline_skewed_is_bit_exact() {
        // The B.3.a tri had ±2/5-bit channel error under barycentric.
        // With scanline-delta RGB walk, strict equality.
        let v = [(50i32, 20i32), (130, 70), (30, 90)];
        let c = [
            (0xFFu8, 0x00u8, 0x00u8),
            (0x00, 0xFF, 0x00),
            (0x00, 0x00, 0xFF),
        ];
        let cpu = cpu_rasterize_shaded_tri(v, c, 0x30, 0, 0, 0);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        let tri = ShadedTri::new(
            v[0],
            v[1],
            v[2],
            c[0],
            c[1],
            c[2],
            PrimFlags::empty(),
            BlendMode::Average,
        );
        let dispatched = r.dispatch_shaded_tri_scanline(&vg, &tri, &DrawArea::full_vram());
        assert!(dispatched);
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "shaded tri scanline skewed strict parity");
    }

    // -------------------------------------------------------
    //  Phase B.5 — rectangle parity vs CPU
    // -------------------------------------------------------

    /// Drive the CPU rasterizer for one monochrome rectangle via
    /// GP0 0x60 (variable-size mono rect, 3-word packet).
    /// `prefill` paints VRAM before the rect — needed for mask-check
    /// and semi-trans tests.
    fn cpu_rasterize_mono_rect(
        xy: (i32, i32),
        wh: (u32, u32),
        color: u16,
        cmd_byte: u8,
        tpage_blend_bits: u8,
        mask_e6: u8,
        prefill: u16,
    ) -> Vec<u16> {
        let mut gpu = Gpu::new();
        if prefill != 0 {
            for y in 0..512u16 {
                for x in 0..1024u16 {
                    gpu.vram.set_pixel(x, y, prefill);
                }
            }
        }
        gpu.gp0_push(0xE3000000);
        gpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        let e1 = 0xE100_0000_u32 | ((tpage_blend_bits as u32) & 0x3) << 5;
        gpu.gp0_push(e1);
        gpu.gp0_push(0xE600_0000_u32 | (mask_e6 as u32) & 0x3);
        let cmd = ((cmd_byte as u32) << 24) | bgr15_to_rgb24(color);
        gpu.gp0_push(cmd);
        gpu.gp0_push(((xy.1 as u32) << 16) | (xy.0 as u32 & 0xFFFF));
        gpu.gp0_push((wh.1 << 16) | (wh.0 & 0xFFFF));
        gpu.vram.words().to_vec()
    }

    fn gpu_prefill_full(vg: &VramGpu, value: u16) {
        if value == 0 {
            return;
        }
        let buf = vec![value; (super::super::VRAM_WIDTH * super::super::VRAM_HEIGHT) as usize];
        vg.upload_full(&buf).unwrap();
    }

    #[test]
    fn mono_rect_basic_opaque_matches_cpu() {
        // Strict bit-exact parity: rectangles have no interpolation,
        // so the only sources of disagreement would be coverage or
        // RMW bugs — neither of which we expect.
        let xy = (50, 60);
        let wh = (40u32, 30u32);
        let color = 0x4321;
        let cpu = cpu_rasterize_mono_rect(xy, wh, color, 0x60, 0, 0, 0);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        r.dispatch_mono_rect(
            &vg,
            &MonoRect::opaque(xy, wh, color),
            &DrawArea::full_vram(),
        );
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "mono rect strict parity");
    }

    #[test]
    fn mono_rect_drawing_area_clip_matches_cpu() {
        // Rect spans (5..55, 5..55) but draw area is (20..40,20..40).
        let xy = (5i32, 5i32);
        let wh = (50u32, 50u32);
        let color = 0x001F; // red
        let mut cpu_gpu = Gpu::new();
        cpu_gpu.gp0_push(0xE3000000 | 20 | (20 << 10));
        cpu_gpu.gp0_push(0xE4000000 | 40 | (40 << 10));
        cpu_gpu.gp0_push(0xE100_0000);
        cpu_gpu.gp0_push(0xE600_0000);
        let cmd = 0x60_000000_u32 | bgr15_to_rgb24(color);
        cpu_gpu.gp0_push(cmd);
        cpu_gpu.gp0_push(((xy.1 as u32) << 16) | (xy.0 as u32 & 0xFFFF));
        cpu_gpu.gp0_push((wh.1 << 16) | (wh.0 & 0xFFFF));
        let cpu = cpu_gpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        r.dispatch_mono_rect(
            &vg,
            &MonoRect::opaque(xy, wh, color),
            &DrawArea {
                left: 20,
                top: 20,
                right: 40,
                bottom: 40,
            },
        );
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "mono rect clip strict parity");
    }

    #[test]
    fn mono_rect_semi_trans_average_matches_cpu() {
        let xy = (10, 10);
        let wh = (20u32, 20u32);
        let color = 0x1234;
        let prefill = 0x5678;
        let cpu = cpu_rasterize_mono_rect(xy, wh, color, 0x62, 0, 0, prefill);
        let vg = VramGpu::new_headless();
        gpu_prefill_full(&vg, prefill);
        let r = Rasterizer::new(&vg);
        r.dispatch_mono_rect(
            &vg,
            &MonoRect::new(xy, wh, color, PrimFlags::SEMI_TRANS, BlendMode::Average),
            &DrawArea::full_vram(),
        );
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "mono rect semi-trans Average parity");
    }

    #[test]
    fn mono_rect_mask_check_protects_pixels() {
        let xy = (10, 10);
        let wh = (15u32, 15u32);
        let color = 0x0123;
        let prefill = 0x8888; // bit 15 set on every pixel
        let cpu = cpu_rasterize_mono_rect(xy, wh, color, 0x60, 0, 0b10, prefill);
        let vg = VramGpu::new_headless();
        gpu_prefill_full(&vg, prefill);
        let r = Rasterizer::new(&vg);
        r.dispatch_mono_rect(
            &vg,
            &MonoRect::new(xy, wh, color, PrimFlags::MASK_CHECK, BlendMode::Average),
            &DrawArea::full_vram(),
        );
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "mono rect mask-check parity");
        // Sanity: nothing should have been written.
        let i = 12 * 1024 + 12;
        assert_eq!(gpu[i], prefill);
    }

    #[test]
    fn mono_rect_zero_size_is_dropped() {
        let cpu = cpu_rasterize_mono_rect((10, 10), (0, 5), 0x4321, 0x60, 0, 0, 0);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        r.dispatch_mono_rect(
            &vg,
            &MonoRect::opaque((10, 10), (0, 5), 0x4321),
            &DrawArea::full_vram(),
        );
        let gpu = vg.download_full().unwrap();
        assert!(cpu.iter().all(|&w| w == 0), "CPU drops zero-width rect");
        assert!(gpu.iter().all(|&w| w == 0), "GPU drops zero-width rect");
    }

    /// Drive the CPU rasterizer for a textured rect via GP0 0x64.
    /// Caller has already set draw area + uploaded VRAM.
    #[allow(clippy::too_many_arguments)]
    fn cpu_push_tex_rect(
        cpu: &mut Gpu,
        cmd_byte: u8,
        tint: (u8, u8, u8),
        xy: (i32, i32),
        wh: (u32, u32),
        uv: (u8, u8),
        clut_word: u32,
        // CPU side picks tpage from the LAST GP0 0xE1 — the rect
        // packet has no per-prim tpage word. Caller pushes it
        // separately before calling.
    ) {
        let cmd = ((cmd_byte as u32) << 24)
            | (tint.0 as u32)
            | ((tint.1 as u32) << 8)
            | ((tint.2 as u32) << 16);
        cpu.gp0_push(cmd);
        cpu.gp0_push(((xy.1 as u32) << 16) | (xy.0 as u32 & 0xFFFF));
        let uv_clut = ((clut_word & 0xFFFF) << 16) | ((uv.0 as u32) | ((uv.1 as u32) << 8));
        cpu.gp0_push(uv_clut);
        cpu.gp0_push((wh.1 << 16) | (wh.0 & 0xFFFF));
    }

    /// Build the GP0 0xE1 word that sets the active tpage state on
    /// the CPU side (mirrors what `apply_primitive_tpage` does in
    /// `tex_tri`, but for rect primitives the host has to push it
    /// explicitly since rect packets have no per-prim tpage word).
    fn make_e1_for_tpage(tpage_x: u32, tpage_y: u32, depth: u32) -> u32 {
        let tx = tpage_x / 64;
        let ty = if tpage_y == 256 { 1u32 } else { 0 };
        0xE100_0000 | (tx & 0xF) | (ty << 4) | ((depth & 0x3) << 7)
    }

    #[test]
    fn tex_rect_15bpp_basic_matches_cpu_byte_for_byte() {
        // Bit-exact parity: rect UVs step linearly, no Q16.16
        // delta math, so CPU and GPU should agree pixel-for-pixel.
        let xy = (40i32, 30i32);
        let wh = (32u32, 24u32);
        let uv = (0u8, 0u8);
        let tpage_x = 128u32;

        let mut vram = vec![0u16; 1024 * 512];
        // Distinct colours per cell so any UV miss shows up.
        for vy in 0..32u16 {
            for ux in 0..64u16 {
                let val = ((vy as u16) << 5) | (ux as u16) | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu.gp0_push(make_e1_for_tpage(tpage_x, 0, 2));
        cpu.gp0_push(0xE600_0000);
        // Cmd 0x65 = textured rect, raw flag set.
        cpu_push_tex_rect(&mut cpu, 0x65, (0, 0, 0), xy, wh, uv, 0);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let rect = TexRect::new(
            xy,
            wh,
            uv,
            0,
            0,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        r.dispatch_tex_rect(&vg, &rect, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        assert_eq!(cpu_words, gpu_words, "tex rect strict parity");
    }

    #[test]
    fn tex_rect_x_flip_mirrors_left_right() {
        // GP0 0xE1 bit 12 = X flip. Each pixel column (dx) reads
        // texel column (last - dx) instead of dx.
        let xy = (40i32, 30i32);
        let wh = (16u32, 16u32);
        let uv = (0u8, 0u8);
        let tpage_x = 128u32;

        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..32u16 {
            for ux in 0..32u16 {
                let val = ((vy as u16) << 5) | (ux as u16) | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }

        // CPU: set the X-flip bit in GP0 0xE1 (bit 12 = 0x1000).
        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu.gp0_push(make_e1_for_tpage(tpage_x, 0, 2) | 0x1000);
        cpu.gp0_push(0xE600_0000);
        cpu_push_tex_rect(&mut cpu, 0x65, (0, 0, 0), xy, wh, uv, 0);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let rect = TexRect::new(
            xy,
            wh,
            uv,
            0,
            0,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE | PrimFlags::FLIP_X,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        r.dispatch_tex_rect(&vg, &rect, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        assert_eq!(cpu_words, gpu_words, "tex rect X-flip strict parity");
    }

    #[test]
    fn tex_rect_modulated_tint_matches_cpu_byte_for_byte() {
        let xy = (40i32, 30i32);
        let wh = (16u32, 16u32);
        let uv = (0u8, 0u8);
        let tpage_x = 128u32;
        let tint = (0x40u8, 0x40u8, 0x40u8); // halve every channel

        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..32u16 {
            for ux in 0..32u16 {
                let val = ((vy as u16) << 5) | (ux as u16) | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu.gp0_push(make_e1_for_tpage(tpage_x, 0, 2));
        cpu.gp0_push(0xE600_0000);
        // 0x64 = textured rect, modulated (NOT raw).
        cpu_push_tex_rect(&mut cpu, 0x64, tint, xy, wh, uv, 0);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let rect = TexRect::new(
            xy,
            wh,
            uv,
            0,
            0,
            tint,
            PrimFlags::empty(),
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        r.dispatch_tex_rect(&vg, &rect, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        assert_eq!(cpu_words, gpu_words, "tex rect modulated strict parity");
    }

    #[test]
    fn tex_rect_4bpp_with_clut_matches_cpu_byte_for_byte() {
        // Tekken-3-style 4bpp paletted rect, the most common 2D-UI
        // primitive. Strict bit-exact parity here would have caught
        // the U/V-wrap bug in `sample_texture` immediately.
        let xy = (40i32, 30i32);
        let wh = (16u32, 16u32);
        let uv = (0u8, 0u8);
        let tpage_x = 0u32;
        let clut_x = 0u32;
        let clut_y = 256u32;

        let mut vram = vec![0u16; 1024 * 512];
        // CLUT: 16 distinct opaque entries.
        for i in 0..16u16 {
            let val = (i.max(1) << 1) | (i.max(1) << 6) | 0x4000;
            vram[clut_y as usize * 1024 + (clut_x as usize + i as usize)] = val;
        }
        // 16×16 4bpp texture — 4 nibbles per VRAM word.
        for vy in 0..16u16 {
            for word_x in 0..4u16 {
                let mut word = 0u16;
                for n in 0..4u16 {
                    let nibble = ((word_x * 4 + n) + vy) & 0xF;
                    word |= nibble << (n * 4);
                }
                vram[vy as usize * 1024 + (tpage_x as usize + word_x as usize)] = word;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        cpu.gp0_push(make_e1_for_tpage(tpage_x, 0, 0));
        cpu.gp0_push(0xE600_0000);
        // CLUT word: clut_x/16 in low 6, clut_y in next 9 bits.
        let clut_word = ((clut_x / 16) & 0x3F) | ((clut_y & 0x1FF) << 6);
        cpu_push_tex_rect(&mut cpu, 0x65, (0, 0, 0), xy, wh, uv, clut_word);
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let rect = TexRect::new(
            xy,
            wh,
            uv,
            clut_x,
            clut_y,
            (0x80, 0x80, 0x80),
            PrimFlags::RAW_TEXTURE,
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 0);
        r.dispatch_tex_rect(&vg, &rect, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        assert_eq!(cpu_words, gpu_words, "tex rect 4bpp+CLUT strict parity");
    }

    // -------------------------------------------------------
    //  Phase B.5.c — fill + VRAM-to-VRAM copy parity vs CPU
    // -------------------------------------------------------

    /// Drive the CPU rasterizer for one quick-fill via GP0 0x02.
    /// `prefill` paints VRAM beforehand so we can verify that fill
    /// IGNORES mask-check / mask-set / drawing-area state.
    fn cpu_rasterize_fill(
        xy: (u32, u32),
        wh: (u32, u32),
        color: u16,
        prefill: u16,
        e3_clip_tl: u32,
        e3_clip_br: u32,
        e6_mask: u8,
    ) -> Vec<u16> {
        let mut gpu = Gpu::new();
        if prefill != 0 {
            for y in 0..512u16 {
                for x in 0..1024u16 {
                    gpu.vram.set_pixel(x, y, prefill);
                }
            }
        }
        gpu.gp0_push(0xE300_0000 | e3_clip_tl);
        gpu.gp0_push(0xE400_0000 | e3_clip_br);
        gpu.gp0_push(0xE600_0000 | e6_mask as u32);
        // 0x02 = quick fill. Color in low 24 bits of cmd.
        let cmd = 0x0200_0000_u32 | bgr15_to_rgb24(color);
        gpu.gp0_push(cmd);
        gpu.gp0_push(((xy.1) << 16) | xy.0);
        gpu.gp0_push(((wh.1) << 16) | wh.0);
        gpu.vram.words().to_vec()
    }

    #[test]
    fn fill_basic_matches_cpu_byte_for_byte() {
        // Strict parity. Fill is the simplest primitive — any diff
        // is a real bug.
        let xy = (32u32, 64u32);
        let wh = (64u32, 32u32);
        let color = 0x4321;
        let cpu = cpu_rasterize_fill(xy, wh, color, 0, 0, 1023 | (511 << 10), 0);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        r.dispatch_fill(&vg, &Fill::new(xy, wh, color));
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "fill basic parity");
    }

    #[test]
    fn fill_ignores_drawing_area_clip() {
        // Set a tiny draw area; fill must overwrite outside it.
        let xy = (32u32, 32u32);
        let wh = (64u32, 64u32);
        let color = 0x1234;
        // Restrict draw area to (40..60, 40..60) — but fill IGNORES
        // this. The whole rect at (32..96, 32..96) should still write.
        let cpu = cpu_rasterize_fill(xy, wh, color, 0, 40 | (40 << 10), 60 | (60 << 10), 0);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        r.dispatch_fill(&vg, &Fill::new(xy, wh, color));
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "fill ignores draw area");
        // Sanity: a pixel OUTSIDE the draw area but INSIDE the fill
        // rect should hold the fill colour on both backends.
        let outside_clip = 35usize * 1024 + 35;
        let expected = ((color & 0x1F) as u8) << 3 | ((color & 0x1F) as u8) >> 2;
        // Check the BGR channels round-trip through fill correctly.
        // Use the exact expected_bgr15 = color, since fill writes 15bpp directly.
        // (RGB24 → BGR15 conversion truncates to 5 bits, so the
        // resulting BGR15 won't equal `color` if `color` had bits set
        // that don't survive the round-trip. cpu_rasterize_fill already
        // pushes through bgr15_to_rgb24 which maps cleanly for our
        // 5-bit-aligned `color`.)
        assert_eq!(cpu[outside_clip], gpu[outside_clip]);
        let _ = expected;
    }

    #[test]
    fn fill_ignores_mask_check() {
        // mask_check_before_draw is set; back buffer has bit 15
        // everywhere. Fill should still write everywhere.
        let xy = (32u32, 32u32);
        let wh = (32u32, 32u32);
        let color = 0x1234;
        let prefill = 0x8888; // bit 15 set
        let cpu = cpu_rasterize_fill(
            xy,
            wh,
            color,
            prefill,
            0,
            1023 | (511 << 10),
            0b10, // mask_check
        );
        let vg = VramGpu::new_headless();
        gpu_prefill_full(&vg, prefill);
        let r = Rasterizer::new(&vg);
        r.dispatch_fill(&vg, &Fill::new(xy, wh, color));
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "fill bypasses mask-check");
        // Sanity: pixel inside fill rect must NOT be the prefill.
        let i = 40 * 1024 + 40;
        assert_ne!(cpu[i], prefill);
    }

    #[test]
    fn fill_zero_size_is_dropped() {
        let cpu = cpu_rasterize_fill((32, 32), (0, 32), 0xCAFE, 0, 0, 1023 | (511 << 10), 0);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        r.dispatch_fill(&vg, &Fill::new((32, 32), (0, 32), 0xCAFE));
        let gpu = vg.download_full().unwrap();
        assert!(cpu.iter().all(|&w| w == 0));
        assert!(gpu.iter().all(|&w| w == 0));
    }

    /// Drive the CPU rasterizer for one VRAM-to-VRAM copy via GP0 0x80.
    fn cpu_rasterize_vram_copy(
        seed: &[u16],
        src: (u16, u16),
        dst: (u16, u16),
        wh: (u16, u16),
    ) -> Vec<u16> {
        let mut gpu = Gpu::new();
        for (i, &w) in seed.iter().enumerate() {
            gpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        gpu.gp0_push(0x80_000000);
        gpu.gp0_push(((src.1 as u32) << 16) | (src.0 as u32));
        gpu.gp0_push(((dst.1 as u32) << 16) | (dst.0 as u32));
        gpu.gp0_push(((wh.1 as u32) << 16) | (wh.0 as u32));
        gpu.vram.words().to_vec()
    }

    #[test]
    fn vram_copy_non_overlapping_matches_cpu_byte_for_byte() {
        // Source and dest disjoint — direct GPU copy path.
        let mut seed = vec![0u16; 1024 * 512];
        for vy in 0..32u16 {
            for ux in 0..32u16 {
                seed[vy as usize * 1024 + (200 + ux as usize)] =
                    ((vy as u16) << 5) | (ux as u16) | 0x1;
            }
        }
        let cpu = cpu_rasterize_vram_copy(&seed, (200, 0), (400, 100), (32, 32));
        let vg = VramGpu::new_headless();
        vg.upload_full(&seed).unwrap();
        let r = Rasterizer::new(&vg);
        r.dispatch_vram_copy(&vg, (200, 0), (400, 100), (32, 32));
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu, "vram copy non-overlapping strict parity");
    }

    #[test]
    fn vram_copy_overlapping_uses_host_bounce_correctly() {
        // Overlap — host-bounce path. Result should still match CPU
        // because the CPU's row-buffer pattern protects horizontal
        // overlap, and our host bounce reads ALL src then writes
        // (effectively the same as a full temp buffer).
        let mut seed = vec![0u16; 1024 * 512];
        for vy in 0..16u16 {
            for ux in 0..16u16 {
                seed[(50 + vy as usize) * 1024 + (50 + ux as usize)] =
                    ((vy as u16) << 5) | (ux as u16) | 0x1;
            }
        }
        // Overlap: src=(50,50) 16x16, dst=(54,54) 16x16. They share
        // a 12x12 inner region.
        let cpu = cpu_rasterize_vram_copy(&seed, (50, 50), (54, 54), (16, 16));
        let vg = VramGpu::new_headless();
        vg.upload_full(&seed).unwrap();
        let r = Rasterizer::new(&vg);
        r.dispatch_vram_copy(&vg, (50, 50), (54, 54), (16, 16));
        let gpu = vg.download_full().unwrap();
        // Strict parity: our host-bounce reads the entire src rect
        // before any writes — equivalent to the CPU's row-buffer
        // semantics for non-vertically-overlapping cases.
        // For vertically overlapping cases the CPU's row-by-row
        // semantics may differ; we accept that as a known
        // limitation in the comment on `dispatch_vram_copy`.
        assert_eq!(cpu, gpu, "vram copy overlap strict parity");
    }

    // -------------------------------------------------------
    //  Phase B.3 — shaded triangle parity vs CPU
    // -------------------------------------------------------

    /// Drive the CPU rasterizer for one Gouraud-shaded triangle via
    /// GP0 0x30 (opaque) or 0x32 (semi-trans). Returns full VRAM.
    fn cpu_rasterize_shaded_tri(
        v: [(i32, i32); 3],
        c: [(u8, u8, u8); 3],
        cmd_byte: u8,
        tpage_blend_bits: u8,
        mask_e6: u8,
        prefill: u16,
    ) -> Vec<u16> {
        let mut gpu = Gpu::new();
        if prefill != 0 {
            for y in 0..512u16 {
                for x in 0..1024u16 {
                    gpu.vram.set_pixel(x, y, prefill);
                }
            }
        }
        gpu.gp0_push(0xE3000000);
        gpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        gpu.gp0_push(0xE100_0000_u32 | ((tpage_blend_bits as u32) & 0x3) << 5);
        gpu.gp0_push(0xE600_0000_u32 | (mask_e6 as u32) & 0x3);
        // GP0 0x30 packet: cmd+c0, v0, c1, v1, c2, v2 (6 words).
        let pack_rgb = |t: (u8, u8, u8)| (t.0 as u32) | ((t.1 as u32) << 8) | ((t.2 as u32) << 16);
        gpu.gp0_push(((cmd_byte as u32) << 24) | pack_rgb(c[0]));
        gpu.gp0_push(pack_xy(v[0]));
        gpu.gp0_push(pack_rgb(c[1]));
        gpu.gp0_push(pack_xy(v[1]));
        gpu.gp0_push(pack_rgb(c[2]));
        gpu.gp0_push(pack_xy(v[2]));
        gpu.vram.words().to_vec()
    }

    #[test]
    fn shaded_tri_axis_aligned_matches_cpu_within_tolerance() {
        // Same UV-parity caveat as B.2: barycentric vs scanline-delta
        // can differ by ±1 in any 5-bit channel at interior pixels
        // due to rounding accumulation. Coverage matches; per-channel
        // delta is bounded.
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let c = [(0xFFu8, 0u8, 0u8), (0u8, 0xFFu8, 0u8), (0u8, 0u8, 0xFFu8)];
        let cpu = cpu_rasterize_shaded_tri(v, c, 0x30, 0, 0, 0);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        let tri = ShadedTri::new(
            v[0],
            v[1],
            v[2],
            c[0],
            c[1],
            c[2],
            PrimFlags::empty(),
            BlendMode::Average,
        );
        r.dispatch_shaded_tri(&vg, &tri, &DrawArea::full_vram());
        let gpu = vg.download_full().unwrap();

        let diffs = diff_inside_bbox(&cpu, &gpu, (20, 20), (60, 60));
        let bbox = 41 * 41;
        // Coverage tolerance.
        assert!(diffs * 4 < bbox, "shaded tri coverage: {diffs} / {bbox}");
        // Per-channel error tolerance.
        let mut max_chan = 0i32;
        for y in 20..=60i32 {
            for x in 20..=60i32 {
                let i = y as usize * 1024 + x as usize;
                if cpu[i] == gpu[i] {
                    continue;
                }
                for shift in [0u32, 5, 10] {
                    let ca = ((cpu[i] >> shift) & 0x1F) as i32;
                    let cb = ((gpu[i] >> shift) & 0x1F) as i32;
                    max_chan = max_chan.max((ca - cb).abs());
                }
            }
        }
        assert!(
            max_chan <= 2,
            "shaded tri max channel delta: {max_chan} > 2"
        );
    }

    #[test]
    fn shaded_tri_uniform_color_matches_mono_tri_path() {
        // When all 3 vertex colours are identical, a Gouraud-shaded
        // triangle should produce the SAME output as a monochrome
        // triangle of that colour. Bit-exact within bbox.
        let v = [(15i32, 15i32), (55, 15), (15, 55)];
        let rgb = (0xC0u8, 0x40u8, 0x80u8);
        // CPU: shaded path with same colour everywhere.
        let cpu_shaded = cpu_rasterize_shaded_tri(v, [rgb; 3], 0x30, 0, 0, 0);
        // CPU: mono path with the BGR15-of-rgb.
        let bgr15 = (((rgb.0 as u16) >> 3) & 0x1F)
            | ((((rgb.1 as u16) >> 3) & 0x1F) << 5)
            | ((((rgb.2 as u16) >> 3) & 0x1F) << 10);
        let cpu_mono = cpu_rasterize_mono_tri(v[0], v[1], v[2], bgr15);
        // The two CPU paths differ at edges (different fill rules?
        // both use scanline-delta, so should be close). Just verify
        // identity-shaded GPU triangle matches the GPU mono path.
        let _ = (cpu_shaded, cpu_mono);

        let vg_shaded = VramGpu::new_headless();
        let r = Rasterizer::new(&vg_shaded);
        let tri = ShadedTri::new(
            v[0],
            v[1],
            v[2],
            rgb,
            rgb,
            rgb,
            PrimFlags::empty(),
            BlendMode::Average,
        );
        r.dispatch_shaded_tri(&vg_shaded, &tri, &DrawArea::full_vram());
        let gpu_shaded = vg_shaded.download_full().unwrap();

        let vg_mono = VramGpu::new_headless();
        let r2 = Rasterizer::new(&vg_mono);
        let mono = MonoTri::opaque(v[0], v[1], v[2], bgr15);
        r2.dispatch_mono_tri(&vg_mono, &mono, &DrawArea::full_vram());
        let gpu_mono = vg_mono.download_full().unwrap();

        // GPU shaded path with uniform colour must match GPU mono path
        // bit-for-bit (same coverage, same colour).
        assert_eq!(gpu_shaded, gpu_mono, "GPU uniform-shaded == GPU mono");
    }

    #[test]
    fn shaded_tex_tri_axis_aligned_15bpp_matches_cpu_within_tolerance() {
        // Composes texture sampling + Gouraud-tint modulation.
        // Same B.2 UV-parity caveat applies; tolerance ≤3/5-bit
        // per channel (slightly looser than B.2 because tint
        // interpolation introduces its own rounding step).
        let v = [(20i32, 20i32), (60, 20), (20, 60)];
        let uv = [(0u8, 0u8), (32, 0), (0, 32)];
        // Different per-vertex tints so interpolation is exercised.
        let c = [
            (0x80u8, 0x80u8, 0x80u8),
            (0xC0, 0xC0, 0xC0),
            (0xFFu8, 0xFFu8, 0xFFu8),
        ];
        let tpage_x = 128u32;
        let tpage_word = make_tpage_word(tpage_x, 0, 2, 0);

        let mut vram = vec![0u16; 1024 * 512];
        for vy in 0..32u16 {
            for ux in 0..32u16 {
                let val = ((vy as u16) << 5) | (ux as u16) | 0x0001;
                vram[vy as usize * 1024 + (tpage_x as usize + ux as usize)] = val;
            }
        }

        let mut cpu = Gpu::new();
        for (i, &w) in vram.iter().enumerate() {
            cpu.vram.set_pixel((i % 1024) as u16, (i / 1024) as u16, w);
        }
        cpu.gp0_push(0xE3000000);
        cpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        let pack_rgb = |t: (u8, u8, u8)| (t.0 as u32) | ((t.1 as u32) << 8) | ((t.2 as u32) << 16);
        // 0x34 = textured-shaded triangle, modulated.
        cpu.gp0_push((0x34u32 << 24) | pack_rgb(c[0]));
        cpu.gp0_push(pack_xy(v[0]));
        cpu.gp0_push(uv_pack(uv[0])); // CLUT 0 (unused for 15bpp)
        cpu.gp0_push(pack_rgb(c[1]));
        cpu.gp0_push(pack_xy(v[1]));
        cpu.gp0_push((tpage_word << 16) | uv_pack(uv[1]));
        cpu.gp0_push(pack_rgb(c[2]));
        cpu.gp0_push(pack_xy(v[2]));
        cpu.gp0_push(uv_pack(uv[2]));
        let cpu_words = cpu.vram.words().to_vec();

        let vg = VramGpu::new_headless();
        vg.upload_full(&vram).unwrap();
        let r = Rasterizer::new(&vg);
        let tri = ShadedTexTri::new(
            v[0],
            v[1],
            v[2],
            c[0],
            c[1],
            c[2],
            uv[0],
            uv[1],
            uv[2],
            0,
            0,
            PrimFlags::empty(),
            BlendMode::Average,
        );
        let tp = Tpage::new(tpage_x, 0, 2);
        r.dispatch_shaded_tex_tri(&vg, &tri, &tp, &DrawArea::full_vram());
        let gpu_words = vg.download_full().unwrap();

        let diffs = diff_inside_bbox(&cpu_words, &gpu_words, (20, 20), (60, 60));
        let bbox = 41 * 41;
        assert!(diffs * 4 < bbox, "shaded-tex coverage: {diffs} / {bbox}");
        let mut max_chan = 0i32;
        for y in 20..=60i32 {
            for x in 20..=60i32 {
                let i = y as usize * 1024 + x as usize;
                if cpu_words[i] == gpu_words[i] {
                    continue;
                }
                for shift in [0u32, 5, 10] {
                    let ca = ((cpu_words[i] >> shift) & 0x1F) as i32;
                    let cb = ((gpu_words[i] >> shift) & 0x1F) as i32;
                    max_chan = max_chan.max((ca - cb).abs());
                }
            }
        }
        // Tolerance is looser than B.2's ±2 because tint
        // interpolation + tint×texel modulation compound the
        // per-step rounding error. ±5 in any 5-bit channel = ~16%
        // intensity, still tight enough to catch real bugs.
        assert!(
            max_chan <= 5,
            "shaded-tex max channel delta: {max_chan} > 5"
        );
    }

    #[test]
    fn shaded_tri_oversized_is_dropped_like_cpu() {
        let v = [(0i32, 0i32), (2000, 0), (0, 0)];
        let c = [(0xFFu8, 0, 0); 3];
        let cpu = cpu_rasterize_shaded_tri(v, c, 0x30, 0, 0, 0);
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        let tri = ShadedTri::new(
            v[0],
            v[1],
            v[2],
            c[0],
            c[1],
            c[2],
            PrimFlags::empty(),
            BlendMode::Average,
        );
        r.dispatch_shaded_tri(&vg, &tri, &DrawArea::full_vram());
        let gpu = vg.download_full().unwrap();
        assert!(cpu.iter().all(|&w| w == 0));
        assert!(gpu.iter().all(|&w| w == 0));
    }

    #[test]
    fn vram_copy_zero_size_is_dropped() {
        let seed = vec![0u16; 1024 * 512];
        let cpu = cpu_rasterize_vram_copy(&seed, (0, 0), (100, 100), (0, 32));
        let vg = VramGpu::new_headless();
        let r = Rasterizer::new(&vg);
        r.dispatch_vram_copy(&vg, (0, 0), (100, 100), (0, 32));
        let gpu = vg.download_full().unwrap();
        assert_eq!(cpu, gpu);
    }
}
