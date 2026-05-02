//! Small VRAM upload helpers for editor-playtest.

use super::*;

/// Stamp the 0x8000 (semi-transparency-disable) bit on every
/// non-zero CLUT entry so opaque textures don't accidentally
/// trigger STP-bit blending.
pub(crate) fn upload_clut(rect: VramRect, bytes: &[u8]) {
    let mut marked = [0u8; 512];
    if bytes.len() > marked.len() || !bytes.len().is_multiple_of(2) {
        return;
    }

    let mut i = 0;
    while i < bytes.len() {
        let raw = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let stamped = if raw == 0 { 0 } else { raw | 0x8000 };
        let pair = stamped.to_le_bytes();
        marked[i] = pair[0];
        marked[i + 1] = pair[1];
        i += 2;
    }

    upload_bytes(rect, &marked[..bytes.len()]);
}

/// Upload a compact 4bpp room material tile.
///
/// The material's GP0(E2) texture window handles repetition, so the
/// runtime only uploads the source texels once.
pub(crate) fn upload_4bpp_tile(
    x: u16,
    y: u16,
    max_width_halfwords: u16,
    max_height: u16,
    texture: &Texture<'_>,
) -> bool {
    let src_hw = texture.halfwords_per_row() as usize;
    let src_h = texture.height() as usize;
    if max_width_halfwords == 0
        || max_height == 0
        || src_hw == 0
        || src_h == 0
        || src_hw > max_width_halfwords as usize
        || src_h > max_height as usize
    {
        return false;
    }

    let src_row_bytes = src_hw * 2;
    let src_bytes = texture.pixel_bytes();
    if src_bytes.len() != src_row_bytes.saturating_mul(src_h) {
        return false;
    }

    upload_bytes(VramRect::new(x, y, src_hw as u16, src_h as u16), src_bytes);
    true
}
