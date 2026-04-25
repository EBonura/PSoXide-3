//! wgpu render pipeline + per-frame buffers for the HW renderer.

use wgpu::util::DeviceExt;

use crate::target::TARGET_FORMAT;

/// Per-vertex data. 16 bytes, `bytemuck::Pod` so we can blit a
/// `Vec<HwVertex>` into the GPU vertex buffer with `cast_slice`.
///
/// Phase 1 only writes `pos` and `color`; the rest carry zeros.
/// Their bindings exist now so the vertex format doesn't change
/// when textured / shaded variants land in Phase 2+.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HwVertex {
    /// PSX screen-space coords (post-`draw_offset`).
    pub pos: [i16; 2],
    /// Tint or per-vertex colour, 0..=255 per channel.
    pub color: [u8; 4],
    /// Texture UV in PSX texture-page-space, 0..=255 per axis.
    pub uv: [u16; 2],
    /// Bit-packed rendering flags (textured? shaded? tex depth?).
    pub flags: u32,
}

/// Globals UBO — uploaded once per frame after the render target
/// is sized.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub display_origin: [f32; 2],
    pub display_size:   [f32; 2],
    pub target_size:    [f32; 2],
    pub _pad:           [f32; 2],
}

impl Globals {
    pub fn zero() -> Self {
        Self {
            display_origin: [0.0, 0.0],
            display_size: [320.0, 240.0],
            target_size: [320.0, 240.0],
            _pad: [0.0, 0.0],
        }
    }
}

/// Initial vertex-buffer capacity. PSX games very rarely exceed a
/// few thousand primitives per frame; 16 K vertices ≈ 5 K mono
/// triangles is plenty of headroom for Phase 1. Grown on demand
/// (re-allocates the wgpu buffer; cheap because it happens once
/// then the high-water-mark sticks).
const INITIAL_VERTEX_CAPACITY: u64 = 16 * 1024;

pub struct HwPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    globals_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity_bytes: u64,
}

impl HwPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-hw-prim"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/prim.wgsl").into()),
        });

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-hw-globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("psx-hw-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-hw-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("psx-hw-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<HwVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // pos: i16x2 → Sint16x2 (2 × 2 bytes = 4)
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Sint16x2,
                    offset: 0,
                    shader_location: 0,
                },
                // color: u8x4 → Unorm8x4 (4 bytes), offset 4
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x4,
                    offset: 4,
                    shader_location: 1,
                },
                // uv: u16x2 → Uint16x2 (2 × 2 = 4 bytes), offset 8
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint16x2,
                    offset: 8,
                    shader_location: 2,
                },
                // flags: u32 → Uint32 (4 bytes), offset 12
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 12,
                    shader_location: 3,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("psx-hw-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // PSX has no winding constraint — quads / rects come
                // in with mixed windings depending on game UV layout.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TARGET_FORMAT,
                    // Phase 1: opaque only (REPLACE). Phase 4 adds
                    // PSX semi-transparency variants.
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        let vertex_capacity_bytes =
            INITIAL_VERTEX_CAPACITY * std::mem::size_of::<HwVertex>() as u64;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("psx-hw-vertices"),
            contents: &vec![0u8; vertex_capacity_bytes as usize],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            globals_buffer,
            vertex_buffer,
            vertex_capacity_bytes,
        }
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    pub fn globals_buffer(&self) -> &wgpu::Buffer {
        &self.globals_buffer
    }

    /// Upload the frame's vertex data. If the payload exceeds the
    /// current buffer capacity we silently truncate — the
    /// `Translator` itself caps emission so this shouldn't trigger
    /// in Phase 1 with PSX-typical primitive counts. Phase 7 will
    /// plumb a proper buffer-grow path through `&Device` if real
    /// frames get noisier.
    pub fn upload_vertices(&mut self, queue: &wgpu::Queue, bytes: &[u8]) {
        let cap = self.vertex_capacity_bytes as usize;
        let payload = if bytes.len() > cap {
            &bytes[..cap]
        } else {
            bytes
        };
        if !payload.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, payload);
        }
    }

    /// Vertex-buffer capacity in bytes (sticky high-water mark).
    /// Translator can use this to cap its emission so we never
    /// truncate. Phase 1 caps on the Translator side at 16 K
    /// vertices = 256 KiB, matches `INITIAL_VERTEX_CAPACITY`.
    pub fn vertex_capacity_bytes(&self) -> u64 {
        self.vertex_capacity_bytes
    }
}

// `bind_group_layout` exists on the pipeline for future bind-group
// rebuilds (when texture bindings land in Phase 2). Silence the
// dead-code lint until then.
#[allow(dead_code)]
fn _bgl_used(p: &HwPipeline) -> &wgpu::BindGroupLayout {
    &p.bind_group_layout
}
