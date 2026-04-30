//! VRAM -- 1 MiB of 16bpp (5:5:5:1) video memory, arranged as 1024×512.
//!
//! Ported verbatim from PSoXide-2's GPU crate (since there is no GPU crate
//! here yet). When the GPU subsystem lands, `Vram` will become owned by
//! `Gpu` and this free-standing module goes away.

/// Native VRAM width in texels.
pub const VRAM_WIDTH: usize = 1024;
/// Native VRAM height in texels.
pub const VRAM_HEIGHT: usize = 512;

/// 1 MiB of 16bpp video memory.
pub struct Vram {
    data: Box<[u16; VRAM_WIDTH * VRAM_HEIGHT]>,
}

impl Vram {
    /// Zero-initialised VRAM. Real hardware comes up with noise; zero is
    /// deterministic and matches PSoXide-2's convention.
    pub fn new() -> Self {
        Self {
            data: vec![0u16; VRAM_WIDTH * VRAM_HEIGHT]
                .into_boxed_slice()
                .try_into()
                .expect("vec length matches Vram size"),
        }
    }

    /// Expose the raw 16bpp words. The GPU rasterizer and VRAM viewer both
    /// read this directly.
    pub fn words(&self) -> &[u16] {
        self.data.as_ref()
    }

    /// Zero all pixels.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Read a single pixel with wrap-around on both axes.
    #[inline]
    pub fn get_pixel(&self, x: u16, y: u16) -> u16 {
        let x = (x as usize) & (VRAM_WIDTH - 1);
        let y = (y as usize) & (VRAM_HEIGHT - 1);
        self.data[y * VRAM_WIDTH + x]
    }

    /// Write a single pixel with wrap-around on both axes.
    #[inline]
    pub fn set_pixel(&mut self, x: u16, y: u16, color: u16) {
        let x = (x as usize) & (VRAM_WIDTH - 1);
        let y = (y as usize) & (VRAM_HEIGHT - 1);
        self.data[y * VRAM_WIDTH + x] = color;
    }

    /// Decode a VRAM rectangle into an RGBA8 buffer for display.
    ///
    /// Full-range 5-to-8-bit expansion: `(v << 3) | (v >> 2)` maps 0→0 and
    /// 31→255. The naive `v << 3` only reaches 248, producing visibly
    /// dimmer whites -- a subtle bug PSoXide-2 learned the hard way.
    pub fn to_rgba8(&self, x_start: u16, y_start: u16, width: u16, height: u16) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for y in y_start..y_start + height {
            for x in x_start..x_start + width {
                let pixel = self.get_pixel(x, y);
                let r5 = pixel & 0x1F;
                let g5 = (pixel >> 5) & 0x1F;
                let b5 = (pixel >> 10) & 0x1F;
                rgba.push(((r5 << 3) | (r5 >> 2)) as u8);
                rgba.push(((g5 << 3) | (g5 >> 2)) as u8);
                rgba.push(((b5 << 3) | (b5 >> 2)) as u8);
                rgba.push(0xFF);
            }
        }
        rgba
    }
}

impl Default for Vram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_zeroed() {
        let vram = Vram::new();
        assert!(vram.words().iter().all(|&w| w == 0));
    }

    #[test]
    fn rgba_expansion_reaches_full_range() {
        let mut vram = Vram::new();
        vram.set_pixel(0, 0, 0x7FFF); // all 31s -- should be white
        let rgba = vram.to_rgba8(0, 0, 1, 1);
        assert_eq!(&rgba[..3], &[0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn pixel_coordinates_wrap() {
        let mut vram = Vram::new();
        vram.set_pixel(1024, 512, 0x1234); // wraps to (0, 0)
        assert_eq!(vram.get_pixel(0, 0), 0x1234);
    }
}
