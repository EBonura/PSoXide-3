/// Expected total word count for a GP0 command starting with opcode `op`.
///
/// For commands we don't decode yet (textured / shaded primitives,
/// VRAM-to-VRAM blits, VRAM-to-CPU transfers), returning `1` means
/// we consume just the first word and drop subsequent data on the
/// floor. That's usually harmless because nothing reads VRAM from
/// those paths yet; when we add them, the packet size here grows
/// to match.
pub(super) fn gp0_packet_size(op: u8) -> usize {
    match op {
        // NOP / clear cache / misc — single word.
        0x00 | 0x01 | 0x03..=0x1E => 1,
        // Quick fill — RGB + (X,Y) + (W,H) = 3 words.
        0x02 => 3,
        // Monochrome flat triangle: color + 3 vertices = 4 words.
        0x20..=0x23 => 4,
        // Shaded flat triangle: 3 × (color+vertex) = 6 words.
        0x30..=0x33 => 6,
        // Monochrome flat quad: color + 4 vertices = 5 words.
        0x28..=0x2B => 5,
        // Shaded flat quad: color + 4 vertices = 8 words.
        0x38..=0x3B => 8,
        // Textured primitives (flat + textured): 3-vert = 7, 4-vert = 9.
        0x24..=0x27 => 7,
        0x2C..=0x2F => 9,
        // Textured + shaded: 3-vert = 9, 4-vert = 12.
        0x34..=0x37 => 9,
        0x3C..=0x3F => 12,
        // Single lines: 3 words (color + 2 vertices) for monochrome,
        // 4 words for shaded.
        0x40..=0x43 => 3,
        0x50..=0x53 => 4,
        // Polyline starts: same initial shape, but after the first
        // endpoint the FIFO enters a streaming receive mode until
        // the terminator sentinel is seen.
        0x48..=0x4B => 3,
        0x58..=0x5B => 4,
        // Monochrome rectangles: variable (3), 1×1 (2), 8×8 (2), 16×16 (2).
        0x60..=0x63 => 3,
        0x64..=0x67 => 4, // variable + textured
        0x68..=0x6B => 2,
        0x6C..=0x6F => 3, // 1×1 textured
        0x70..=0x73 => 2,
        0x74..=0x77 => 3, // 8×8 textured
        0x78..=0x7B => 2,
        0x7C..=0x7F => 3, // 16×16 textured
        // VRAM-to-VRAM copy: opcode + src + dst + size = 4 words.
        0x80..=0x9F => 4,
        // CPU-to-VRAM transfer: opcode + xy + wh = 3 words, then
        // pixel data is consumed in a separate "upload mode".
        0xA0..=0xBF => 3,
        // VRAM-to-CPU: same 3-word header.
        0xC0..=0xDF => 3,
        // Draw-mode settings (E1..=E6) — single word each.
        0xE1..=0xE6 => 1,
        _ => 1,
    }
}
