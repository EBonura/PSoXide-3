// PSX hardware-renderer shader.
//
// Single shader serving every primitive type by branching on
// `flags`. Phase 1 supported flat-color mono primitives. Phase 2
// adds textured tris + quads with 4 bpp / 8 bpp CLUT and 15 bpp
// direct sampling, tint modulation, and transparent-texel discard.
//
// PSX coordinate convention: vertex positions are post-`draw_offset`
// PSX pixel coords (signed, fits in i16). We map
// `(display_origin .. display_origin + display_size)` to NDC
// (-1..+1, Y-flipped). The viewport equals the high-res render
// target, so fractional scale "just works" — primitives rasterize
// at viewport resolution while vertex coords stay in PSX-space
// integers.
//
// `HwVertex::flags` bit layout (must mirror `pipeline::flags`):
//   bits  0..=3   tpage_x_units   (× 64 = pixel x)
//   bit       4   tpage_y_index   (× 256 = pixel y; 0 or 256)
//   bits  5..=6   tex_depth        (0=4bpp, 1=8bpp, 2=15bpp)
//   bits  7..=12  clut_x_units    (× 16 = pixel x)
//   bits 13..=21  clut_y          (pixel y, 0..=511)
//   bit      22   TEXTURED
//   bit      23   RAW_TEXTURE     (skip tint modulate)
//   bit      24   SEMI_TRANS      (Phase 4)

struct Globals {
    display_origin: vec2<f32>,
    display_size:   vec2<f32>,
    target_size:    vec2<f32>,
    _pad:           vec2<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var vram: texture_2d<u32>;

struct VertexIn {
    @location(0) pos:   vec2<i32>,
    @location(1) color: vec4<f32>,
    @location(2) uv:    vec2<u32>,
    @location(3) flags: u32,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0)       color:    vec4<f32>,
    @location(1) @interpolate(flat) uv: vec2<u32>,
    @location(2) @interpolate(flat) flags: u32,
}

const FLAG_TEXTURED:    u32 = 1u << 22u;
const FLAG_RAW_TEXTURE: u32 = 1u << 23u;
// FLAG_SEMI_TRANS reserved for Phase 4.

const VRAM_W: u32 = 1024u;
const VRAM_H: u32 =  512u;

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let pos_psx = vec2<f32>(f32(in.pos.x), f32(in.pos.y));
    let ndc_xy = ((pos_psx - g.display_origin) / g.display_size) * 2.0 - 1.0;
    var out: VertexOut;
    out.position = vec4<f32>(ndc_xy.x, -ndc_xy.y, 0.0, 1.0);
    out.color    = in.color;
    out.uv       = in.uv;
    out.flags    = in.flags;
    return out;
}

// PSX U/V are 8-bit per axis (so wrap on >255). The active
// tex-window is added in Phase 5; for now just mask to the page.
fn page_uv(uv: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(uv.x & 0xFFu, uv.y & 0xFFu);
}

fn tpage_origin(flags: u32) -> vec2<u32> {
    let tx = (flags & 0xFu) * 64u;
    let ty = ((flags >> 4u) & 1u) * 256u;
    return vec2<u32>(tx, ty);
}

fn clut_origin(flags: u32) -> vec2<u32> {
    let cx = ((flags >> 7u) & 0x3Fu) * 16u;
    let cy = (flags >> 13u) & 0x1FFu;
    return vec2<u32>(cx, cy);
}

fn tex_depth(flags: u32) -> u32 {
    return (flags >> 5u) & 0x3u;
}

fn vram_load(x: u32, y: u32) -> u32 {
    let xx = x & (VRAM_W - 1u);
    let yy = y & (VRAM_H - 1u);
    return textureLoad(vram, vec2<i32>(i32(xx), i32(yy)), 0).r;
}

// Convert a BGR15 word to linear-ish RGB (bit-replicated to 8-bit
// per channel, then divided by 255). Matches `Vram::to_rgba8` on
// the CPU side — same expansion the existing display path used.
fn bgr15_to_rgb(word: u32) -> vec3<f32> {
    let r5 = word & 0x1Fu;
    let g5 = (word >> 5u) & 0x1Fu;
    let b5 = (word >> 10u) & 0x1Fu;
    let r8 = (r5 << 3u) | (r5 >> 2u);
    let g8 = (g5 << 3u) | (g5 >> 2u);
    let b8 = (b5 << 3u) | (b5 >> 2u);
    return vec3<f32>(f32(r8), f32(g8), f32(b8)) / 255.0;
}

// Sample the active texture page at PSX UV. Returns the raw
// 16-bit BGR15 texel (not yet alpha-tested or expanded). Texel
// 0 is reserved for "transparent" — caller discards.
fn sample_texel(flags: u32, uv8: vec2<u32>) -> u32 {
    let tp = tpage_origin(flags);
    let depth = tex_depth(flags);
    if depth == 0u {
        // 4 bpp: 4 indices per VRAM word, picked by uv.x % 4.
        let word = vram_load(tp.x + (uv8.x >> 2u), tp.y + uv8.y);
        let nibble = (word >> ((uv8.x & 3u) * 4u)) & 0xFu;
        let cl = clut_origin(flags);
        return vram_load(cl.x + nibble, cl.y);
    } else if depth == 1u {
        // 8 bpp: 2 indices per VRAM word, picked by uv.x & 1.
        let word = vram_load(tp.x + (uv8.x >> 1u), tp.y + uv8.y);
        let byte = (word >> ((uv8.x & 1u) * 8u)) & 0xFFu;
        let cl = clut_origin(flags);
        return vram_load(cl.x + byte, cl.y);
    }
    // 15 bpp direct.
    return vram_load(tp.x + uv8.x, tp.y + uv8.y);
}

// Modulate a sampled BGR15 texel by the per-vertex tint. PSX
// formula: each channel = clamp(channel * tint * 2, 0..=255). We
// keep the colours in 0..1 throughout so the math is linear here.
// `RAW_TEXTURE` (raw-texture) skips this and uses the texel as-is.
fn modulate(texel_rgb: vec3<f32>, tint_rgba: vec4<f32>, raw: bool) -> vec3<f32> {
    if raw {
        return texel_rgb;
    }
    return clamp(texel_rgb * tint_rgba.rgb * 2.0, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let textured = (in.flags & FLAG_TEXTURED) != 0u;
    if !textured {
        return in.color;
    }
    let uv8 = page_uv(in.uv);
    let texel = sample_texel(in.flags, uv8);
    if texel == 0u {
        // PSX convention: a 0x0000 texel is fully transparent.
        // Discarding produces the right result for both opaque
        // primitives (no pixel written) and Phase 4 semi-trans
        // primitives (no blend either).
        discard;
    }
    let raw = (in.flags & FLAG_RAW_TEXTURE) != 0u;
    let rgb = modulate(bgr15_to_rgb(texel), in.color, raw);
    return vec4<f32>(rgb, 1.0);
}
