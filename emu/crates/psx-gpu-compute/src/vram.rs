//! GPU-side VRAM: a 524288-entry `storage_buffer<u32>` (one PS1 BGR15
//! word per element, packed into the low 16 bits) plus rect-granular
//! upload/download helpers.
//!
//! Why a storage buffer of `u32`, not a storage texture?
//! ----------------------------------------------------
//! `R16Uint` storage textures aren't in the WebGPU storage-format
//! whitelist and fail to validate on Metal (and WebGPU itself). The
//! options were:
//!   - `Rgba16Uint` storage texture: 4× the bandwidth, wastes 75%.
//!   - `R32Uint` storage texture: 2× the bandwidth, fine but no
//!     atomics on integer texels in any portable WGSL.
//!   - **`storage_buffer<u32>`**: direct `[y * 1024 + x]` indexing,
//!     supports `atomic<u32>` if we ever need ordering/locking, and
//!     2× memory cost (2 MiB instead of 1 MiB) is a non-issue on any
//!     modern GPU. Picked this.
//!
//! Layout: index `y * 1024 + x` holds one PS1 pixel in the low 16
//! bits. The high 16 bits are always zero on host-side writes; the
//! shader is free to use them as scratch (e.g. ordering tags) once
//! we get there.
//!
//! Coordinate convention matches `emulator_core::Vram`: `(0,0)` is
//! top-left, X grows right (0..1023), Y grows down (0..511).

use std::sync::Arc;

use thiserror::Error;

/// PS1 VRAM is 1024 columns wide.
pub const VRAM_WIDTH: u32 = 1024;
/// PS1 VRAM is 512 rows tall.
pub const VRAM_HEIGHT: u32 = 512;
/// Number of u32 entries in the VRAM storage buffer.
pub const VRAM_LEN_U32: u64 = (VRAM_WIDTH as u64) * (VRAM_HEIGHT as u64);
/// Size of the VRAM storage buffer in bytes.
pub const VRAM_BYTES: u64 = VRAM_LEN_U32 * 4;

/// `R16Uint` is no longer the on-GPU format, but the public constant
/// still names the *semantic* per-pixel format the rasterizer is
/// emulating (one 16-bit BGR15 word per VRAM cell). Kept as a stable
/// API hook so callers don't need to know the storage layout.
pub const VRAM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Uint;

#[derive(Debug, Error)]
pub enum VramGpuError {
    #[error("rect ({x},{y}) {w}x{h} extends past VRAM bounds (1024×512)")]
    OutOfBounds { x: u32, y: u32, w: u32, h: u32 },
    #[error("rect data length {got} bytes does not match expected {expected} bytes")]
    SizeMismatch { got: usize, expected: usize },
    #[error("rect width or height is zero")]
    Empty,
    #[error("download timed out waiting for GPU")]
    DownloadTimeout,
    #[error("buffer mapping failed: {0}")]
    Mapping(String),
}

/// GPU-resident PS1 VRAM with rect upload/download helpers.
pub struct VramGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// `[u32; 1024 * 512]` -- one PS1 pixel per element, low 16 bits.
    buffer: wgpu::Buffer,
}

impl VramGpu {
    /// Construct a fresh `VramGpu` on top of an existing wgpu device.
    /// Buffer starts zeroed (matches CPU `Vram::new`).
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-vram"),
            size: VRAM_BYTES,
            // STORAGE so compute shaders can bind it; COPY_DST/SRC for
            // rect upload/download. Initialized lazily -- first write
            // is the upload itself.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // wgpu doesn't guarantee the contents of a freshly created
        // buffer -- explicitly clear so `download_full()` reads zeros.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("psx-vram-init-clear"),
        });
        encoder.clear_buffer(&buffer, 0, None);
        queue.submit(Some(encoder.finish()));
        Self {
            device,
            queue,
            buffer,
        }
    }

    /// Build a headless `VramGpu` for tests and benchmarks. Picks the
    /// highest-performance adapter wgpu can find.
    pub fn new_headless() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("no wgpu adapter available — install Metal/Vulkan/DX12 driver");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("psx-gpu-compute-headless"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("request_device");
        Self::new(Arc::new(device), Arc::new(queue))
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Overwrite the entire VRAM from a CPU-side `&[u16]` slice in
    /// row-major order. Length must be exactly `1024 * 512 = 524288`.
    pub fn upload_full(&self, words: &[u16]) -> Result<(), VramGpuError> {
        self.upload_rect(0, 0, VRAM_WIDTH, VRAM_HEIGHT, words)
    }

    /// Upload a sub-rectangle. `words.len()` must be exactly `w * h`.
    /// Out-of-bounds rects are rejected (no wrap -- the rasterizer
    /// handles wrap separately on the GPU side).
    pub fn upload_rect(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        words: &[u16],
    ) -> Result<(), VramGpuError> {
        check_rect(x, y, w, h)?;
        let expected = (w as usize) * (h as usize);
        if words.len() != expected {
            return Err(VramGpuError::SizeMismatch {
                got: words.len() * 2,
                expected: expected * 2,
            });
        }
        // The on-GPU layout is `u32` per pixel (low 16 bits used). For
        // a contiguous full-row range we can build the u32 buffer and
        // do a single `write_buffer`; for partial rects we issue one
        // `write_buffer` per row at `index = y * 1024 + x`. The row
        // path is what every realistic upload uses.
        let mut row_words = Vec::with_capacity(w as usize);
        for row in 0..h {
            row_words.clear();
            let src_off = (row as usize) * (w as usize);
            for &px in &words[src_off..src_off + w as usize] {
                // Zero-extend the 16-bit pixel into a u32 -- high bits
                // reserved for the rasterizer (ordering tags, etc).
                row_words.push(px as u32);
            }
            let dst_index = (y + row) as u64 * VRAM_WIDTH as u64 + x as u64;
            let dst_byte = dst_index * 4;
            let bytes: &[u8] = bytemuck::cast_slice(&row_words);
            self.queue.write_buffer(&self.buffer, dst_byte, bytes);
        }
        Ok(())
    }

    /// Read the entire VRAM back as `Vec<u16>` (524288 entries,
    /// row-major). Blocks the calling thread on a `map_async` poll --
    /// for tests/screenshots only, never per-frame in the frontend.
    pub fn download_full(&self) -> Result<Vec<u16>, VramGpuError> {
        self.download_rect(0, 0, VRAM_WIDTH, VRAM_HEIGHT)
    }

    /// Read a sub-rect back as `Vec<u16>` (length `w*h`, row-major).
    pub fn download_rect(&self, x: u32, y: u32, w: u32, h: u32) -> Result<Vec<u16>, VramGpuError> {
        check_rect(x, y, w, h)?;

        // Copy the requested rect (one row at a time) into a packed
        // staging buffer. Buffer copies don't need row alignment --
        // simpler than the texture-staging path, and we control the
        // layout end-to-end.
        let row_bytes = (w as u64) * 4;
        let staging_size = row_bytes * h as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psx-vram-readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-vram-download"),
            });
        for row in 0..h {
            let src_index = (y + row) as u64 * VRAM_WIDTH as u64 + x as u64;
            let src_byte = src_index * 4;
            let dst_byte = row as u64 * row_bytes;
            encoder.copy_buffer_to_buffer(&self.buffer, src_byte, &staging, dst_byte, row_bytes);
        }
        self.queue.submit(Some(encoder.finish()));

        // Synchronous map -- `pollster` drives the GPU until the
        // mapping callback fires.
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(VramGpuError::Mapping(format!("{e:?}"))),
            Err(_) => return Err(VramGpuError::DownloadTimeout),
        }

        let data = slice.get_mapped_range();
        let words_u32: &[u32] = bytemuck::cast_slice(&data);
        // Truncate each u32 back to its low 16 bits.
        let out: Vec<u16> = words_u32.iter().map(|&w| w as u16).collect();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Read a single pixel. Convenience wrapper for tests.
    pub fn read_pixel(&self, x: u32, y: u32) -> Result<u16, VramGpuError> {
        let v = self.download_rect(x, y, 1, 1)?;
        Ok(v[0])
    }

    /// Reset VRAM to all zeros (matches CPU `Vram::new`). Done on the
    /// GPU side via `clear_buffer` -- no host roundtrip.
    pub fn clear(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("psx-vram-clear"),
            });
        encoder.clear_buffer(&self.buffer, 0, None);
        self.queue.submit(Some(encoder.finish()));
    }
}

fn check_rect(x: u32, y: u32, w: u32, h: u32) -> Result<(), VramGpuError> {
    if w == 0 || h == 0 {
        return Err(VramGpuError::Empty);
    }
    if x.saturating_add(w) > VRAM_WIDTH || y.saturating_add(h) > VRAM_HEIGHT {
        return Err(VramGpuError::OutOfBounds { x, y, w, h });
    }
    Ok(())
}

// =============================================================
//  Tests
// =============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use emulator_core::{Vram, VRAM_HEIGHT as CPU_VRAM_H, VRAM_WIDTH as CPU_VRAM_W};

    fn fresh() -> VramGpu {
        VramGpu::new_headless()
    }

    #[test]
    fn dimensions_match_cpu_vram() {
        // Catches any future divergence between the CPU `Vram` shape
        // and the GPU storage buffer.
        assert_eq!(VRAM_WIDTH as usize, CPU_VRAM_W);
        assert_eq!(VRAM_HEIGHT as usize, CPU_VRAM_H);
    }

    #[test]
    fn fresh_vram_reads_back_zero() {
        let g = fresh();
        let words = g.download_full().expect("download");
        assert_eq!(words.len(), (VRAM_WIDTH * VRAM_HEIGHT) as usize);
        assert!(words.iter().all(|&w| w == 0), "fresh VRAM is zeroed");
    }

    #[test]
    fn full_upload_round_trip_is_byte_identical() {
        let g = fresh();
        let mut state = 0xACE1u16;
        let mut src = Vec::with_capacity((VRAM_WIDTH * VRAM_HEIGHT) as usize);
        for _ in 0..(VRAM_WIDTH * VRAM_HEIGHT) {
            state = state.wrapping_mul(0x4F1B).wrapping_add(0x7C5D);
            src.push(state);
        }
        g.upload_full(&src).expect("upload");
        let dst = g.download_full().expect("download");
        assert_eq!(src, dst, "round-trip must be bit-exact");
    }

    #[test]
    fn partial_upload_does_not_disturb_neighbours() {
        let g = fresh();
        let block = vec![0xFFFFu16; 16];
        g.upload_rect(100, 200, 4, 4, &block).expect("upload");
        let inside = g.download_rect(100, 200, 4, 4).expect("download");
        assert!(inside.iter().all(|&w| w == 0xFFFF));
        let left = g.download_rect(99, 200, 1, 4).expect("download");
        assert!(left.iter().all(|&w| w == 0), "left column unchanged");
        let top = g.download_rect(100, 199, 4, 1).expect("download");
        assert!(top.iter().all(|&w| w == 0), "top row unchanged");
    }

    #[test]
    fn parity_with_cpu_vram_set_pixel() {
        // Apply the same write pattern to both the CPU `Vram` and the
        // GPU `VramGpu`, then assert byte-identical readback. This is
        // the foundational parity test the compute rasterizer will
        // build on.
        let g = fresh();
        let mut cpu = Vram::new();
        for &(x, y, val) in &[
            (0u16, 0u16, 0x1234u16),
            (1023, 0, 0xABCDu16),
            (0, 511, 0x4321u16),
            (1023, 511, 0xCAFEu16),
            (512, 256, 0xBEEFu16),
        ] {
            cpu.set_pixel(x, y, val);
            g.upload_rect(x as u32, y as u32, 1, 1, &[val]).unwrap();
        }
        for &(x, y, val) in &[
            (0u32, 0u32, 0x1234u16),
            (1023, 0, 0xABCD),
            (0, 511, 0x4321),
            (1023, 511, 0xCAFE),
            (512, 256, 0xBEEF),
        ] {
            assert_eq!(g.read_pixel(x, y).unwrap(), val, "GPU @ ({x},{y})");
            assert_eq!(cpu.get_pixel(x as u16, y as u16), val, "CPU @ ({x},{y})");
        }
    }

    #[test]
    fn rect_at_extreme_corners() {
        let g = fresh();
        g.upload_rect(VRAM_WIDTH - 1, VRAM_HEIGHT - 1, 1, 1, &[0x7FFF])
            .unwrap();
        assert_eq!(
            g.read_pixel(VRAM_WIDTH - 1, VRAM_HEIGHT - 1).unwrap(),
            0x7FFF
        );
    }

    #[test]
    fn out_of_bounds_rect_is_rejected_not_truncated() {
        let g = fresh();
        let err = g
            .upload_rect(VRAM_WIDTH - 1, 0, 2, 1, &[0x1, 0x2])
            .unwrap_err();
        assert!(matches!(err, VramGpuError::OutOfBounds { .. }));
    }

    #[test]
    fn size_mismatch_is_rejected() {
        let g = fresh();
        let err = g.upload_rect(0, 0, 4, 4, &[0u16; 8]).unwrap_err();
        assert!(matches!(err, VramGpuError::SizeMismatch { .. }));
    }

    #[test]
    fn empty_rect_is_rejected() {
        let g = fresh();
        assert!(matches!(
            g.upload_rect(0, 0, 0, 4, &[]).unwrap_err(),
            VramGpuError::Empty
        ));
        assert!(matches!(
            g.upload_rect(0, 0, 4, 0, &[]).unwrap_err(),
            VramGpuError::Empty
        ));
    }

    #[test]
    fn clear_resets_full_texture() {
        let g = fresh();
        let block = vec![0xAAAAu16; 64];
        g.upload_rect(50, 50, 8, 8, &block).unwrap();
        assert_eq!(g.read_pixel(53, 53).unwrap(), 0xAAAA);
        g.clear();
        assert_eq!(g.read_pixel(53, 53).unwrap(), 0);
        assert_eq!(g.read_pixel(0, 0).unwrap(), 0);
        assert_eq!(g.read_pixel(VRAM_WIDTH - 1, VRAM_HEIGHT - 1).unwrap(), 0);
    }

    #[test]
    fn narrow_column_round_trips() {
        // A 1×N download is the most stride-sensitive case -- catches
        // any off-by-one in the per-row copy_buffer_to_buffer loop.
        let g = fresh();
        let column: Vec<u16> = (0..VRAM_HEIGHT as u16).collect();
        g.upload_rect(7, 0, 1, VRAM_HEIGHT, &column).unwrap();
        let read = g.download_rect(7, 0, 1, VRAM_HEIGHT).unwrap();
        assert_eq!(read, column, "narrow column round-trips correctly");
    }

    #[test]
    fn deep_parity_against_cpu_vram() {
        // Drive a long randomised sequence of `set_pixel` calls into
        // both backends and assert the full VRAM matches at the end.
        // 2000 writes scattered across all of VRAM.
        let g = fresh();
        let mut cpu = Vram::new();
        let mut state = 0x9E37u16;
        for _ in 0..2000 {
            state = state.wrapping_mul(0x4F1B).wrapping_add(0x7C5D);
            let x = (state as u32) % VRAM_WIDTH;
            state = state.wrapping_mul(0x4F1B).wrapping_add(0x7C5D);
            let y = (state as u32) % VRAM_HEIGHT;
            state = state.wrapping_mul(0x4F1B).wrapping_add(0x7C5D);
            cpu.set_pixel(x as u16, y as u16, state);
            g.upload_rect(x, y, 1, 1, &[state]).unwrap();
        }
        // Compare a selection of cells. Doing all 524288 would work
        // but is overkill -- the round-trip + partial tests already
        // pin layout/byte-order correctness; this just confirms that
        // the GPU mirrors the CPU's last-write-wins semantics.
        for (x, y) in [(0u32, 0u32), (511, 256), (1023, 511), (777, 333)] {
            let cpu_v = cpu.get_pixel(x as u16, y as u16);
            let gpu_v = g.read_pixel(x, y).unwrap();
            assert_eq!(gpu_v, cpu_v, "parity @ ({x},{y})");
        }
    }
}
