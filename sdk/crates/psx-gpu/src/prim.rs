//! Primitive packet types for DMA-based GPU submission.
//!
//! Each struct is `#[repr(C)]` with the tag word first — that's the
//! shape the DMA linked-list walker expects. The field names match
//! the on-wire GP0 word order so a reader can cross-reference
//! PSX-SPX without redundant decoding.
//!
//! Builders (`new` constructors) zero the tag; [`crate::ot::OrderingTable::add`]
//! fills it in during insertion with `(words_after_tag << 24) | next`.

use psx_hw::gpu::{gp0, pack_color, pack_texcoord, pack_vertex, pack_xy};

/// Flat-shaded triangle. 5 words (tag + 4 data).
#[repr(C, align(4))]
pub struct TriFlat {
    /// DMA / OT linkage word. Written by the OT at insert time.
    pub tag: u32,
    /// `0x20000000 | rgb24` header.
    pub color_cmd: u32,
    /// Vertex 0 packed via [`pack_vertex`].
    pub v0: u32,
    /// Vertex 1.
    pub v1: u32,
    /// Vertex 2.
    pub v2: u32,
}

impl TriFlat {
    /// Data-word count after the tag. Passed to `ot::add`.
    pub const WORDS: u8 = 4;

    /// Build a flat triangle ready for OT insertion.
    pub fn new(verts: [(i16, i16); 3], r: u8, g: u8, b: u8) -> Self {
        Self {
            tag: 0,
            color_cmd: gp0::polygon_opcode(false, false, false, false, false)
                | pack_color(r, g, b),
            v0: pack_vertex(verts[0].0, verts[0].1),
            v1: pack_vertex(verts[1].0, verts[1].1),
            v2: pack_vertex(verts[2].0, verts[2].1),
        }
    }
}

/// Gouraud-shaded triangle. 7 words (tag + 6 data).
#[repr(C, align(4))]
pub struct TriGouraud {
    /// OT linkage.
    pub tag: u32,
    /// Vertex 0: `opcode | color0`.
    pub color0_cmd: u32,
    /// Vertex 0 position.
    pub v0: u32,
    /// Vertex 1 color.
    pub color1: u32,
    /// Vertex 1 position.
    pub v1: u32,
    /// Vertex 2 color.
    pub color2: u32,
    /// Vertex 2 position.
    pub v2: u32,
}

impl TriGouraud {
    /// Data-word count after the tag.
    pub const WORDS: u8 = 6;

    /// Build a Gouraud-shaded triangle.
    pub fn new(verts: [(i16, i16); 3], colors: [(u8, u8, u8); 3]) -> Self {
        let (r0, g0, b0) = colors[0];
        let (r1, g1, b1) = colors[1];
        let (r2, g2, b2) = colors[2];
        Self {
            tag: 0,
            color0_cmd: gp0::polygon_opcode(true, false, false, false, false)
                | pack_color(r0, g0, b0),
            v0: pack_vertex(verts[0].0, verts[0].1),
            color1: pack_color(r1, g1, b1),
            v1: pack_vertex(verts[1].0, verts[1].1),
            color2: pack_color(r2, g2, b2),
            v2: pack_vertex(verts[2].0, verts[2].1),
        }
    }
}

/// Flat-shaded quad. 6 words (tag + 5 data).
#[repr(C, align(4))]
pub struct QuadFlat {
    /// OT linkage.
    pub tag: u32,
    /// `opcode | color`.
    pub color_cmd: u32,
    /// Vertex 0.
    pub v0: u32,
    /// Vertex 1.
    pub v1: u32,
    /// Vertex 2.
    pub v2: u32,
    /// Vertex 3.
    pub v3: u32,
}

impl QuadFlat {
    /// Data-word count.
    pub const WORDS: u8 = 5;

    /// Build a flat quad.
    pub fn new(verts: [(i16, i16); 4], r: u8, g: u8, b: u8) -> Self {
        Self {
            tag: 0,
            color_cmd: gp0::polygon_opcode(false, true, false, false, false)
                | pack_color(r, g, b),
            v0: pack_vertex(verts[0].0, verts[0].1),
            v1: pack_vertex(verts[1].0, verts[1].1),
            v2: pack_vertex(verts[2].0, verts[2].1),
            v3: pack_vertex(verts[3].0, verts[3].1),
        }
    }
}

/// Untextured variable-size rectangle. 4 words (tag + 3 data).
/// Ignores draw-area clip on some GPU revisions; prefer `QuadFlat`
/// when you need clipping.
#[repr(C, align(4))]
pub struct RectFlat {
    /// OT linkage.
    pub tag: u32,
    /// `0x60000000 | color` (monochrome rect opcode).
    pub color_cmd: u32,
    /// Top-left `xy`.
    pub xy: u32,
    /// Size `wh`.
    pub wh: u32,
}

impl RectFlat {
    /// Data-word count.
    pub const WORDS: u8 = 3;

    /// Build a rect.
    pub fn new(x: i16, y: i16, w: u16, h: u16, r: u8, g: u8, b: u8) -> Self {
        Self {
            tag: 0,
            color_cmd: 0x6000_0000 | pack_color(r, g, b),
            xy: pack_vertex(x, y),
            wh: pack_xy(w, h),
        }
    }
}

/// Textured sprite (variable size). 5 words (tag + 4 data).
#[repr(C, align(4))]
pub struct Sprite {
    /// OT linkage.
    pub tag: u32,
    /// `0x64000000 | color` header (blend color applied over texture).
    pub color_cmd: u32,
    /// Top-left `xy`.
    pub xy: u32,
    /// `uv | clut` (U/V in low half, CLUT handle in high half).
    pub uv_clut: u32,
    /// Size `wh`.
    pub wh: u32,
}

impl Sprite {
    /// Data-word count.
    pub const WORDS: u8 = 4;

    /// Build a textured sprite. `clut` is the CLUT register handle
    /// (`y << 6 | x >> 4`); `uv` is the 8-bit texcoord within the
    /// texture page.
    pub fn new(
        x: i16,
        y: i16,
        w: u16,
        h: u16,
        uv: (u8, u8),
        clut: u16,
        r: u8,
        g: u8,
        b: u8,
    ) -> Self {
        Self {
            tag: 0,
            color_cmd: 0x6400_0000 | pack_color(r, g, b),
            xy: pack_vertex(x, y),
            uv_clut: pack_texcoord(uv.0, uv.1, clut),
            wh: pack_xy(w, h),
        }
    }
}
