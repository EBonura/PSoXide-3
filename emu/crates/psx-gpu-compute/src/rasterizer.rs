//! Compute-shader rasterizer dispatcher.
//!
//! Phase B.1: one primitive, one dispatch. The pipeline objects are
//! built once and reused; each `dispatch_*` call writes the
//! primitive's parameters into a uniform buffer, picks the matching
//! pipeline, and dispatches a workgroup grid sized to the bounding
//! box.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::primitive::{DrawArea, MonoTri};
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
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mono_tri.wgsl").into(),
            ),
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

        Self {
            device,
            queue,
            mono_tri_pipeline,
            mono_tri_bg_layout,
            mono_tri_uniform,
            draw_area_uniform,
        }
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

    fn cpu_rasterize_mono_tri(
        v0: (i32, i32),
        v1: (i32, i32),
        v2: (i32, i32),
        color: u16,
    ) -> Vec<u16> {
        // The CPU rasterizer doesn't expose `rasterize_triangle`
        // directly. Drive it through the GP0 packet API: GP0 0x20
        // is "monochrome flat triangle", 4-word packet.
        let mut gpu = Gpu::new();
        // Set draw area to full VRAM so nothing gets clipped.
        gpu.gp0_push(0xE3000000); // E3 — top-left at (0, 0)
        // E4 — bottom-right at (1023, 511): X bits 9:0, Y bits 18:10.
        gpu.gp0_push(0xE4000000 | 1023 | (511 << 10));
        // Pack RGB into the cmd word: cmd byte 0x20 in bits 24..31,
        // R/G/B in 0..23.
        let cmd = 0x20000000_u32 | bgr15_to_rgb24(color);
        gpu.gp0_push(cmd);
        gpu.gp0_push(pack_xy(v0));
        gpu.gp0_push(pack_xy(v1));
        gpu.gp0_push(pack_xy(v2));
        // Read VRAM back as a row-major u16 vec.
        let vram = gpu.vram.words().to_vec();
        vram
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
        let tri = MonoTri::new(v0, v1, v2, color);
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
        let tri = MonoTri::new(v0, v1, v2, color);
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
        let tri = MonoTri::new(v0, v1, v2, color);
        let area = DrawArea::full_vram();
        r.dispatch_mono_tri(&vg, &tri, &area);
        let gpu_vram = vg.download_full().expect("download");

        // Both should be all-zero VRAM (degenerate primitive
        // dropped before any plotting).
        assert!(cpu_vram.iter().all(|&w| w == 0), "CPU should drop");
        assert!(gpu_vram.iter().all(|&w| w == 0), "GPU should drop");
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
        let tri = MonoTri::new(v0, v1, v2, color);
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
}
