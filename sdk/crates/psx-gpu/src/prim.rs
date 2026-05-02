//! Primitive packet types for DMA-based GPU submission.
//!
//! Each struct is `#[repr(C)]` with the tag word first -- that's the
//! shape the DMA linked-list walker expects. The field names match
//! the on-wire GP0 word order so a reader can cross-reference
//! PSX-SPX without redundant decoding.
//!
//! Builders (`new` constructors) zero the tag; [`crate::ot::OrderingTable::add`]
//! fills it in during insertion with `(words_after_tag << 24) | next`.

use crate::material::TextureMaterial;
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
    pub const fn new(verts: [(i16, i16); 3], r: u8, g: u8, b: u8) -> Self {
        Self {
            tag: 0,
            color_cmd: gp0::polygon_opcode(false, false, false, false, false) | pack_color(r, g, b),
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
    pub const fn new(verts: [(i16, i16); 3], colors: [(u8, u8, u8); 3]) -> Self {
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
    pub const fn new(verts: [(i16, i16); 4], r: u8, g: u8, b: u8) -> Self {
        Self {
            tag: 0,
            color_cmd: gp0::polygon_opcode(false, true, false, false, false) | pack_color(r, g, b),
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
    pub const fn new(x: i16, y: i16, w: u16, h: u16, r: u8, g: u8, b: u8) -> Self {
        Self {
            tag: 0,
            color_cmd: 0x6000_0000 | pack_color(r, g, b),
            xy: pack_vertex(x, y),
            wh: pack_xy(w, h),
        }
    }
}

/// Gouraud-shaded quad. 9 words (tag + 8 data).
///
/// Same vertex order as [`QuadFlat`] (V0=TL, V1=TR, V2=BL, V3=BR
/// by convention, though the GPU actually draws (V0,V1,V2) then
/// (V1,V2,V3)). Each vertex carries its own RGB; the GPU
/// gouraud-interpolates across the primitive.
#[repr(C, align(4))]
pub struct QuadGouraud {
    /// OT linkage.
    pub tag: u32,
    /// Vertex 0 colour + polygon opcode.
    pub color0_cmd: u32,
    /// Vertex 0 position.
    pub v0: u32,
    /// Vertex 1 colour.
    pub color1: u32,
    /// Vertex 1 position.
    pub v1: u32,
    /// Vertex 2 colour.
    pub color2: u32,
    /// Vertex 2 position.
    pub v2: u32,
    /// Vertex 3 colour.
    pub color3: u32,
    /// Vertex 3 position.
    pub v3: u32,
}

impl QuadGouraud {
    /// Data-word count after the tag.
    pub const WORDS: u8 = 8;

    /// Build a Gouraud quad. `colors[i]` corresponds to `verts[i]`.
    pub const fn new(verts: [(i16, i16); 4], colors: [(u8, u8, u8); 4]) -> Self {
        let (r0, g0, b0) = colors[0];
        let (r1, g1, b1) = colors[1];
        let (r2, g2, b2) = colors[2];
        let (r3, g3, b3) = colors[3];
        Self {
            tag: 0,
            color0_cmd: gp0::polygon_opcode(true, true, false, false, false)
                | pack_color(r0, g0, b0),
            v0: pack_vertex(verts[0].0, verts[0].1),
            color1: pack_color(r1, g1, b1),
            v1: pack_vertex(verts[1].0, verts[1].1),
            color2: pack_color(r2, g2, b2),
            v2: pack_vertex(verts[2].0, verts[2].1),
            color3: pack_color(r3, g3, b3),
            v3: pack_vertex(verts[3].0, verts[3].1),
        }
    }
}

/// Monochrome single line. 4 words (tag + 3 data). GP0 0x40 -- the
/// real diagonal-capable line rasteriser (unlike `RectFlat`, which
/// the GPU snaps to 16-pixel X boundaries in its GP0 0x02 fill).
#[repr(C, align(4))]
pub struct LineMono {
    /// OT linkage.
    pub tag: u32,
    /// `0x40000000 | color` header.
    pub color_cmd: u32,
    /// First endpoint.
    pub v0: u32,
    /// Second endpoint.
    pub v1: u32,
}

impl LineMono {
    /// Data-word count.
    pub const WORDS: u8 = 3;

    /// Build a mono line.
    pub const fn new(x0: i16, y0: i16, x1: i16, y1: i16, r: u8, g: u8, b: u8) -> Self {
        Self {
            tag: 0,
            color_cmd: 0x4000_0000 | pack_color(r, g, b),
            v0: pack_vertex(x0, y0),
            v1: pack_vertex(x1, y1),
        }
    }
}

/// Textured triangle with a single flat tint. 9 words (tag + 8 data).
///
/// The first data word is GP0(E2) texture-window state, followed by
/// vertex + UV pairs; CLUT rides in vertex 0's UV high word, tpage in
/// vertex 1's UV high word (PSX-SPX convention for GP0 0x24). Emitting
/// E2 per triangle keeps windowed world materials from leaking state to
/// model triangles when the ordering table interleaves both.
#[repr(C, align(4))]
pub struct TriTextured {
    /// OT linkage.
    pub tag: u32,
    /// GP0(E2) texture-window command.
    pub tex_window: u32,
    /// `0x24000000 | tint` header.
    pub color_cmd: u32,
    /// Vertex 0 position.
    pub v0: u32,
    /// `(u0, v0, clut)` packed.
    pub uv0_clut: u32,
    /// Vertex 1 position.
    pub v1: u32,
    /// `(u1, v1, tpage)` packed.
    pub uv1_tpage: u32,
    /// Vertex 2 position.
    pub v2: u32,
    /// `(u2, v2, 0)` packed.
    pub uv2: u32,
}

impl TriTextured {
    /// Data-word count.
    pub const WORDS: u8 = 8;

    /// Build a textured triangle. `tint = (128, 128, 128)` leaves
    /// texels unmodulated.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        verts: [(i16, i16); 3],
        uvs: [(u8, u8); 3],
        clut: u16,
        tpage: u16,
        tint: (u8, u8, u8),
    ) -> Self {
        Self::with_material(verts, uvs, TextureMaterial::opaque(clut, tpage, tint))
    }

    /// Build a textured triangle using a [`TextureMaterial`].
    pub const fn with_material(
        verts: [(i16, i16); 3],
        uvs: [(u8, u8); 3],
        material: TextureMaterial,
    ) -> Self {
        Self {
            tag: 0,
            tex_window: material.texture_window_word(),
            color_cmd: material.flat_textured_polygon_header(false),
            v0: pack_vertex(verts[0].0, verts[0].1),
            uv0_clut: pack_texcoord(uvs[0].0, uvs[0].1, material.clut_word()),
            v1: pack_vertex(verts[1].0, verts[1].1),
            uv1_tpage: pack_texcoord(uvs[1].0, uvs[1].1, material.tpage_word()),
            v2: pack_vertex(verts[2].0, verts[2].1),
            uv2: pack_texcoord(uvs[2].0, uvs[2].1, 0),
        }
    }
}

/// Textured **Gouraud-shaded** triangle. 10 words (tag + 9 data).
///
/// Per-vertex tint: the GPU multiplies each texel by the
/// interpolated vertex colour, so GTE-lit-and-fogged per-vertex
/// colours from `NCDT` drive the final shade smoothly across the
/// triangle. Same CLUT + tpage embedding as [`TriTextured`]: CLUT
/// rides in v0's UV high word, tpage in v1's UV high word (PSX-SPX
/// convention for GP0 0x34).
#[repr(C, align(4))]
pub struct TriTexturedGouraud {
    /// OT linkage.
    pub tag: u32,
    /// `0x34000000 | color0` header -- v0's RGB is packed into the
    /// same word as the polygon opcode.
    pub color0_cmd: u32,
    /// Vertex 0 position.
    pub v0: u32,
    /// `(u0, v0, clut)` packed.
    pub uv0_clut: u32,
    /// Vertex 1 colour (RGB in low 24 bits; top byte ignored).
    pub color1: u32,
    /// Vertex 1 position.
    pub v1: u32,
    /// `(u1, v1, tpage)` packed.
    pub uv1_tpage: u32,
    /// Vertex 2 colour.
    pub color2: u32,
    /// Vertex 2 position.
    pub v2: u32,
    /// `(u2, v2, 0)` packed.
    pub uv2: u32,
}

impl TriTexturedGouraud {
    /// Data-word count.
    pub const WORDS: u8 = 9;

    /// Build a textured Gouraud triangle. Each vertex carries its
    /// own RGB (the NCDT-lit-and-fogged colour in the typical
    /// commercial-game path) which modulates the sampled texel.
    pub const fn new(
        verts: [(i16, i16); 3],
        uvs: [(u8, u8); 3],
        colors: [(u8, u8, u8); 3],
        clut: u16,
        tpage: u16,
    ) -> Self {
        Self::with_material(verts, uvs, colors, TextureMaterial::new(clut, tpage))
    }

    /// Build a textured Gouraud triangle using a [`TextureMaterial`].
    pub const fn with_material(
        verts: [(i16, i16); 3],
        uvs: [(u8, u8); 3],
        colors: [(u8, u8, u8); 3],
        material: TextureMaterial,
    ) -> Self {
        let (r0, g0, b0) = colors[0];
        let (r1, g1, b1) = colors[1];
        let (r2, g2, b2) = colors[2];
        Self {
            tag: 0,
            color0_cmd: material.textured_polygon_command(true, false) | pack_color(r0, g0, b0),
            v0: pack_vertex(verts[0].0, verts[0].1),
            uv0_clut: pack_texcoord(uvs[0].0, uvs[0].1, material.clut_word()),
            color1: pack_color(r1, g1, b1),
            v1: pack_vertex(verts[1].0, verts[1].1),
            uv1_tpage: pack_texcoord(uvs[1].0, uvs[1].1, material.tpage_word()),
            color2: pack_color(r2, g2, b2),
            v2: pack_vertex(verts[2].0, verts[2].1),
            uv2: pack_texcoord(uvs[2].0, uvs[2].1, 0),
        }
    }
}

/// Textured quad with a single flat tint. 10 words (tag + 9 data).
///
/// Same CLUT + tpage embedding as [`TriTextured`], extended by
/// one vertex. Vertex order: TL, TR, BL, BR.
#[repr(C, align(4))]
pub struct QuadTextured {
    /// OT linkage.
    pub tag: u32,
    /// `0x2C000000 | tint` header.
    pub color_cmd: u32,
    /// V0 position.
    pub v0: u32,
    /// `(u0, v0, clut)`.
    pub uv0_clut: u32,
    /// V1 position.
    pub v1: u32,
    /// `(u1, v1, tpage)`.
    pub uv1_tpage: u32,
    /// V2 position.
    pub v2: u32,
    /// `(u2, v2, 0)`.
    pub uv2: u32,
    /// V3 position.
    pub v3: u32,
    /// `(u3, v3, 0)`.
    pub uv3: u32,
}

impl QuadTextured {
    /// Data-word count.
    pub const WORDS: u8 = 9;

    /// Build a textured quad.
    pub const fn new(
        verts: [(i16, i16); 4],
        uvs: [(u8, u8); 4],
        clut: u16,
        tpage: u16,
        tint: (u8, u8, u8),
    ) -> Self {
        Self::with_material(verts, uvs, TextureMaterial::opaque(clut, tpage, tint))
    }

    /// Build a textured quad using a [`TextureMaterial`].
    pub const fn with_material(
        verts: [(i16, i16); 4],
        uvs: [(u8, u8); 4],
        material: TextureMaterial,
    ) -> Self {
        Self {
            tag: 0,
            color_cmd: material.flat_textured_polygon_header(true),
            v0: pack_vertex(verts[0].0, verts[0].1),
            uv0_clut: pack_texcoord(uvs[0].0, uvs[0].1, material.clut_word()),
            v1: pack_vertex(verts[1].0, verts[1].1),
            uv1_tpage: pack_texcoord(uvs[1].0, uvs[1].1, material.tpage_word()),
            v2: pack_vertex(verts[2].0, verts[2].1),
            uv2: pack_texcoord(uvs[2].0, uvs[2].1, 0),
            v3: pack_vertex(verts[3].0, verts[3].1),
            uv3: pack_texcoord(uvs[3].0, uvs[3].1, 0),
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
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
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
        let material = TextureMaterial::opaque(clut, 0, (r, g, b));
        Self::with_material(x, y, w, h, uv, material)
    }

    /// Build a textured sprite using a [`TextureMaterial`].
    ///
    /// Sprite packets do not carry a tpage word. The material's CLUT,
    /// tint, raw-texture bit, and semi-transparent command bit are
    /// encoded in the packet; the caller must set the matching draw
    /// mode before OT submission if the sprite samples a non-current
    /// tpage.
    pub const fn with_material(
        x: i16,
        y: i16,
        w: u16,
        h: u16,
        uv: (u8, u8),
        material: TextureMaterial,
    ) -> Self {
        Self {
            tag: 0,
            color_cmd: material.textured_rect_header(),
            xy: pack_vertex(x, y),
            uv_clut: pack_texcoord(uv.0, uv.1, material.clut_word()),
            wh: pack_xy(w, h),
        }
    }
}
