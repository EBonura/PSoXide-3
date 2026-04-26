//! wgpu render pipeline + per-frame buffers for the HW renderer.

use wgpu::util::DeviceExt;

use crate::target::{TARGET_FORMAT, VRAM_HEIGHT, VRAM_WIDTH};

/// Per-vertex data. 16 bytes, `bytemuck::Pod` so we can blit a
/// `Vec<HwVertex>` into the GPU vertex buffer with `cast_slice`.
///
/// `flags` packs every per-primitive piece of state the fragment
/// shader needs — the format is identical across primitives, so a
/// single draw call handles mixed textured / mono / different
/// tpages / different CLUTs in one batch. See [`flags`] module for
/// the bit layout (host + WGSL must agree).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HwVertex {
    /// PSX screen-space coords (post-`draw_offset`).
    pub pos: [i16; 2],
    /// Tint or per-vertex colour, 0..=255 per channel. Modulates
    /// the sampled texel for textured primitives unless
    /// `flags::RAW_TEXTURE` is set.
    pub color: [u8; 4],
    /// Texture UV in PSX texture-page-space, 0..=255 per axis.
    /// `u16` for vertex-attribute alignment; PSX only emits 8-bit
    /// values today.
    pub uv: [u16; 2],
    /// Bit-packed per-primitive state. See [`flags`].
    pub flags: u32,
}

/// `HwVertex::flags` bit layout. Host + shader must agree —
/// any change here needs the matching constants in
/// `shaders/prim.wgsl` updated in lockstep.
///
/// Layout (low bit first):
/// ```text
/// bits  0..=3   tpage_x_units (0..15)         tpage_x = units * 64
/// bit       4   tpage_y_index (0 or 1)        tpage_y = index * 256
/// bits  5..=6   tex_depth (0=4bpp, 1=8bpp, 2=15bpp)
/// bits  7..=12  clut_x_units  (0..63)         clut_x  = units * 16
/// bits 13..=21  clut_y         (0..511)
/// bit      22   TEXTURED                      else flat color
/// bit      23   RAW_TEXTURE                   skip tint modulate
/// bit      24   SEMI_TRANS                    routed to a non-Opaque batch
/// ```
pub mod flags {
    pub const TEXTURED: u32 = 1 << 22;
    pub const RAW_TEXTURE: u32 = 1 << 23;
    pub const SEMI_TRANS: u32 = 1 << 24;

    /// Pack tpage origin (in pixels) into the flag bits.
    /// `tpage_x` must be a multiple of 64 (PSX alignment),
    /// `tpage_y` must be 0 or 256.
    pub fn pack_tpage(tpage_x: u32, tpage_y: u32, depth: u32) -> u32 {
        let tx = (tpage_x / 64) & 0xF;
        let ty = if tpage_y >= 256 { 1u32 } else { 0u32 };
        let d = depth & 0x3;
        tx | (ty << 4) | (d << 5)
    }

    /// Pack CLUT origin (in pixels) into the flag bits.
    /// `clut_x` must be a multiple of 16, `clut_y` must be 0..=511.
    pub fn pack_clut(clut_x: u32, clut_y: u32) -> u32 {
        let cx = (clut_x / 16) & 0x3F;
        let cy = clut_y & 0x1FF;
        (cx << 7) | (cy << 13)
    }
}

/// Initial vertex-buffer capacity. PSX games very rarely exceed a
/// few thousand primitives per frame; 16 K vertices ≈ 5 K mono
/// triangles is plenty of headroom for Phase 1. Grown on demand
/// (re-allocates the wgpu buffer; cheap because it happens once
/// then the high-water-mark sticks).
const INITIAL_VERTEX_CAPACITY: u64 = 16 * 1024;

/// Per-primitive PSX blend mode. Maps 1:1 to one of the 5 render
/// pipelines `HwPipeline` builds — the translator sorts vertices
/// into contiguous batches by `BlendKind`, the renderer issues
/// one draw per batch with the matching pipeline + blend constant.
///
/// Numeric discriminants are stable so they can index into the
/// pipeline / blend-constant arrays directly.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BlendKind {
    /// `cmd-bit-25` clear → standard opaque draw, `BlendState::REPLACE`.
    Opaque = 0,
    /// PSX semi-trans mode 0: `(B + F) / 2`. Implemented as
    /// `Constant * F + Constant * B` with blend_constant = 0.5.
    Average = 1,
    /// PSX semi-trans mode 1: `B + F`. `One * F + One * B`.
    Add = 2,
    /// PSX semi-trans mode 2: `B - F`. `One * B - One * F` (reverse subtract).
    Sub = 3,
    /// PSX semi-trans mode 3: `B + F/4`. `Constant * F + One * B` with
    /// blend_constant = 0.25.
    AddQuarter = 4,
}

impl BlendKind {
    pub const COUNT: usize = 5;

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Blend-constant value to bind before a batch's draw call.
/// Only Average / AddQuarter use it; the others ignore it.
fn blend_constant_for(kind: BlendKind) -> wgpu::Color {
    let v = match kind {
        BlendKind::Average => 0.5,
        BlendKind::AddQuarter => 0.25,
        _ => 0.0,
    };
    wgpu::Color {
        r: v,
        g: v,
        b: v,
        a: v,
    }
}

pub struct HwPipeline {
    /// One render pipeline per [`BlendKind`]. Same shader, same
    /// vertex layout — only `BlendState` differs. Indexed by
    /// `BlendKind::index()`.
    pipelines: [wgpu::RenderPipeline; BlendKind::COUNT],
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity_bytes: u64,
    /// VRAM texture sampled by the fragment shader for textured
    /// primitives. `R16Uint`, 1024×512, populated each frame from
    /// CPU `Vram` via `upload_vram`. `R16Uint` (vs the existing
    /// `Rgba8UnormSrgb` VRAM viewer texture) preserves PSX BGR15
    /// bit-exactly so the shader can decode 4/8/15 bpp and CLUT
    /// indices without going through sRGB / replication.
    vram_texture: wgpu::Texture,
    /// Held alive so the bind group's `TextureView` reference
    /// stays valid for the life of the pipeline. Read indirectly
    /// through the bind group; never queried directly.
    #[allow(dead_code)]
    vram_view: wgpu::TextureView,
}

impl HwPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("psx-hw-prim"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/prim.wgsl").into()),
        });

        // VRAM as a 1024×512 R16Uint texture. Filled each frame
        // by `upload_vram`. The fragment shader reads from it via
        // `textureLoad` (no filtering — every PSX texture access
        // is integer-addressed).
        let vram_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("psx-hw-vram-r16uint"),
            size: wgpu::Extent3d {
                width: VRAM_WIDTH,
                height: VRAM_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let vram_view = vram_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("psx-hw-bgl"),
            entries: &[
                // 0: VRAM texture (fragment shader). Vertex shader
                // doesn't need any uniforms — VRAM dims are
                // hardcoded constants in the WGSL.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psx-hw-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&vram_view),
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

        // Build five render-pipeline variants, one per BlendKind.
        // Same shader + same vertex layout; only the color
        // attachment's `BlendState` differs. Translator picks the
        // right pipeline per primitive at draw time.
        let make_pipeline = |label: &str, blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout.clone()],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // PSX has no winding constraint — quads / rects
                    // come in with mixed windings depending on game
                    // UV layout.
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
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            })
        };
        let pipelines = [
            make_pipeline("psx-hw-pipeline-opaque", wgpu::BlendState::REPLACE),
            make_pipeline("psx-hw-pipeline-average", blend_state_average()),
            make_pipeline("psx-hw-pipeline-add", blend_state_add()),
            make_pipeline("psx-hw-pipeline-sub", blend_state_sub()),
            make_pipeline("psx-hw-pipeline-add-quarter", blend_state_add_quarter()),
        ];

        let vertex_capacity_bytes =
            INITIAL_VERTEX_CAPACITY * std::mem::size_of::<HwVertex>() as u64;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("psx-hw-vertices"),
            contents: &vec![0u8; vertex_capacity_bytes as usize],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipelines,
            bind_group_layout,
            bind_group,
            vertex_buffer,
            vertex_capacity_bytes,
            vram_texture,
            vram_view,
        }
    }

    /// Mirror CPU VRAM into the GPU-side `R16Uint` texture used by
    /// the fragment shader. Cheap full-frame upload (1 MiB / frame)
    /// — same cost the existing `Graphics::prepare_vram` already
    /// pays for the VRAM viewer panel. Phase 7 may dirty-track
    /// regions to skip unchanged frames; for now full-upload keeps
    /// the renderer obviously correct.
    pub fn upload_vram(&self, queue: &wgpu::Queue, words: &[u16]) {
        if words.len() != (VRAM_WIDTH * VRAM_HEIGHT) as usize {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.vram_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(words),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(VRAM_WIDTH * 2),
                rows_per_image: Some(VRAM_HEIGHT),
            },
            wgpu::Extent3d {
                width: VRAM_WIDTH,
                height: VRAM_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn pipeline(&self, kind: BlendKind) -> &wgpu::RenderPipeline {
        &self.pipelines[kind.index()]
    }

    /// Blend constant the given kind expects bound before its
    /// draw call (`set_blend_constant`). Opaque / Add / Sub
    /// don't use it; we still bind a defined value for
    /// determinism.
    pub fn blend_constant(&self, kind: BlendKind) -> wgpu::Color {
        blend_constant_for(kind)
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
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

// ---- PSX semi-trans blend states ----
//
// All four reuse the same factors for color and alpha (alpha
// channel of the output is unused by egui's display path; we
// just keep it consistent). The blend constant is bound per draw
// call via `RenderPass::set_blend_constant`. PSX doesn't need
// per-component constants — same value on R/G/B/A is fine.

/// PSX mode 0 — `(B + F) / 2`. With blend_constant = 0.5,
/// `Constant * F + Constant * B = 0.5*F + 0.5*B`.
fn blend_state_average() -> wgpu::BlendState {
    let c = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Constant,
        dst_factor: wgpu::BlendFactor::Constant,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState { color: c, alpha: c }
}

/// PSX mode 1 — `B + F`. Plain additive: `One*F + One*B`.
fn blend_state_add() -> wgpu::BlendState {
    let c = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState { color: c, alpha: c }
}

/// PSX mode 2 — `B - F`. Reverse-subtract gives
/// `One*B - One*F = B - F`.
fn blend_state_sub() -> wgpu::BlendState {
    let c = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::ReverseSubtract,
    };
    wgpu::BlendState { color: c, alpha: c }
}

/// PSX mode 3 — `B + F/4`. `Constant=0.25`,
/// `Constant*F + One*B = 0.25*F + B`.
fn blend_state_add_quarter() -> wgpu::BlendState {
    let c = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Constant,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState { color: c, alpha: c }
}
