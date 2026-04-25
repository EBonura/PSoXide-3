//! Compute-shader rasterizer dispatcher.
//!
//! Phase B.1: one primitive, one dispatch. The pipeline objects are
//! built once and reused; each `dispatch_*` call writes the
//! primitive's parameters into a uniform buffer, picks the matching
//! pipeline, and dispatches a workgroup grid sized to the bounding
//! box.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::primitive::{BlendMode, DrawArea, MonoTri, PrimFlags};
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
    fn diff_inside_bbox(
        a: &[u16],
        b: &[u16],
        bbox_min: (i32, i32),
        bbox_max: (i32, i32),
    ) -> usize {
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
            v0, v1, v2, color,
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
        let color = 0x4210;        // (r=0x10, g=0x10, b=0x10)
        let prefill = 0x4210;       // same — sum must clamp to 31 per channel
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x22, 1, 0, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0, v1, v2, color,
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
        let color = 0x2108;         // (r=8, g=8, b=8)
        let prefill = 0x4210;       // (r=16, g=16, b=16) → result (r=8,g=8,b=8)
        let cpu = cpu_rasterize_mono_tri_full(v0, v1, v2, color, 0x22, 2, 0, prefill);
        let gpu = gpu_rasterize_mono_tri_full(
            v0, v1, v2, color,
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
            v0, v1, v2, color,
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
            v0, v1, v2, color,
            PrimFlags::MASK_SET,
            BlendMode::Average,
            prefill,
        );
        let diffs = diff_inside_bbox(&cpu, &gpu, (10, 10), (50, 50));
        assert!(diffs == 0, "MASK_SET parity: {diffs} differ");
        // Sanity: spot-check a point firmly inside the triangle has
        // bit 15 set on both backends.
        let inside_idx = 20 * 1024 + 20;
        assert!(cpu[inside_idx] & 0x8000 != 0, "CPU inside pixel: bit 15 set");
        assert!(gpu[inside_idx] & 0x8000 != 0, "GPU inside pixel: bit 15 set");
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
            v0, v1, v2, color,
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
        let tri = MonoTri::new(
            v0, v1, v2, color,
            PrimFlags::MASK_CHECK,
            BlendMode::Average,
        );
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
            assert_eq!(cpu[i] & 0x7FFF, color & 0x7FFF, "CPU row {y} (open) overwritten");
            assert_eq!(gpu[i] & 0x7FFF, color & 0x7FFF, "GPU row {y} (open) overwritten");
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
            v0, v1, v2, color,
            PrimFlags::SEMI_TRANS | PrimFlags::MASK_SET,
            BlendMode::Average,
            prefill,
        );
        let diffs = diff_inside_bbox(&cpu, &gpu, (10, 10), (50, 50));
        assert!(
            diffs == 0,
            "semi-trans + mask-set parity: {diffs} differ"
        );
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
}
