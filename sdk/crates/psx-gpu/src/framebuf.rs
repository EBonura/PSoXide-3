//! Double-buffered framebuffer management.
//!
//! A `FrameBuffer` tracks two on-screen regions in VRAM — one being
//! displayed, one being drawn into. `swap()` flips them at a VBlank
//! boundary. Layout is vertical: buffer A at Y=0, buffer B at Y=height.
//! That fits 2×(640×240) side-by-side inside the 1024×512 VRAM on
//! standard NTSC resolutions and gives the engine a natural tear-free
//! presentation.

use psx_hw::gpu::{gp0, gp1};
use psx_io::gpu::{write_gp0, write_gp1};

/// Tracks display-start between two vertically stacked buffers.
pub struct FrameBuffer {
    /// Display width in pixels (set by [`FrameBuffer::new`]).
    pub width: u16,
    /// Display height in pixels.
    pub height: u16,
    /// Index (0 or 1) of the buffer currently drawn TO.
    pub drawing: u8,
}

impl FrameBuffer {
    /// Create a framebuffer for the given active display size. Buffer
    /// A lives at VRAM Y=0, buffer B at Y=`height`.
    pub const fn new(width: u16, height: u16) -> Self {
        Self { width, height, drawing: 0 }
    }

    /// Y-coordinate of buffer `idx` in VRAM.
    #[inline]
    pub const fn buffer_y(&self, idx: u8) -> u16 {
        if idx == 0 { 0 } else { self.height }
    }

    /// Push a display-start command for the buffer we're NOT currently
    /// drawing to — flipping the display at the next VBlank.
    pub fn swap(&mut self) {
        // Show the buffer we were drawing into; drain into the other.
        let show = self.drawing;
        self.drawing ^= 1;
        let show_y = self.buffer_y(show);
        write_gp1(gp1::display_start(0, show_y as u32));

        // Re-set the draw-area / draw-offset to match the new target buffer.
        let target_y = self.buffer_y(self.drawing);
        write_gp0(gp0::draw_area_top_left(0, target_y as u32));
        write_gp0(gp0::draw_area_bottom_right(
            (self.width - 1) as u32,
            (target_y + self.height - 1) as u32,
        ));
        write_gp0(gp0::draw_offset(0, target_y as i32));
    }

    /// Clear the back-buffer (the one currently being drawn to) to `(r, g, b)`.
    pub fn clear(&self, r: u8, g: u8, b: u8) {
        super::fill_rect(0, self.buffer_y(self.drawing), self.width, self.height, r, g, b);
    }
}
