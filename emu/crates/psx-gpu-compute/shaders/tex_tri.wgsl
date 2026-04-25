// Textured flat-shaded triangle rasterizer (GP0 0x24..=0x27).
//
// Per pixel:
//   - inside the drawing-area clip rect?
//   - inside the triangle (edge-function test, top-left fill rule)?
//   - barycentric interpolate the three (U, V) per-vertex texcoords
//     to a (u, v) at this pixel
//   - sample VRAM at the tpage with the shader's port of
//     `emulator_core::Gpu::sample_texture` — texture window, 8-bit
//     wrap, then 4bpp/8bpp/15bpp decode + CLUT lookup
//   - if texel == 0 (transparent): skip
//   - if not RAW_TEXTURE: modulate by the per-prim tint
//   - if SEMI_TRANS && (texel & 0x8000): blend with back buffer
//   - mask-bit handling identical to mono_tri.wgsl
//
// Coverage and blending are byte-for-byte ports of mono_tri.wgsl —
// the new code is the texture sampler + tint modulation, which
// mirror `sample_texture` and `modulate_tint` exactly.

struct TexTri {
    v0: vec2<i32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    bbox_min: vec2<i32>,
    bbox_max: vec2<i32>,
    uv0: u32,    // u | (v << 8)
    uv1: u32,
    uv2: u32,
    tint: u32,   // r | (g << 8) | (b << 16)
    flags: u32,  // PrimFlags + (BlendMode << 8)
    clut_x: u32,
    clut_y: u32,
    // Individual u32 pads (NOT vec3<u32>) — vec3 would force 16-byte
    // alignment and bump the struct size to 96, but the host struct
    // is 80 bytes by design. Keep them as scalars.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Tpage {
    tpage_x: u32,
    tpage_y: u32,
    tex_depth: u32,
    _pad: u32,
    tex_window_mask_x: u32,
    tex_window_mask_y: u32,
    tex_window_off_x: u32,
    tex_window_off_y: u32,
}

struct DrawArea {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

@group(0) @binding(0) var<storage, read_write> vram: array<u32>;
@group(0) @binding(1) var<uniform> prim: TexTri;
@group(0) @binding(2) var<uniform> draw_area: DrawArea;
@group(0) @binding(3) var<uniform> tpage: Tpage;

const VRAM_WIDTH: i32 = 1024;
const VRAM_HEIGHT: i32 = 512;
const VRAM_WIDTH_U: u32 = 1024u;
const VRAM_HEIGHT_U: u32 = 512u;

const FLAG_SEMI_TRANS:  u32 = 1u << 0u;
const FLAG_MASK_CHECK:  u32 = 1u << 1u;
const FLAG_MASK_SET:    u32 = 1u << 2u;
const FLAG_RAW_TEXTURE: u32 = 1u << 3u;

const BLEND_AVERAGE:    u32 = 0u;
const BLEND_ADD:        u32 = 1u;
const BLEND_SUB:        u32 = 2u;
const BLEND_ADDQUARTER: u32 = 3u;

const DEPTH_4BPP:  u32 = 0u;
const DEPTH_8BPP:  u32 = 1u;
const DEPTH_15BPP: u32 = 2u;

fn edge(a: vec2<i32>, b: vec2<i32>, p: vec2<i32>) -> i32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn is_top_left(a: vec2<i32>, b: vec2<i32>) -> bool {
    let d = b - a;
    return (d.y == 0 && d.x > 0) || (d.y < 0);
}

// PSX semi-transparency blend — see mono_tri.wgsl for the three
// Redux quirks this mirrors. Identical math.
fn blend(bg_word: u32, fg_word: u32, mode: u32) -> u32 {
    let br = i32(bg_word & 0x1Fu);
    let bg = i32((bg_word >> 5u) & 0x1Fu);
    let bb = i32((bg_word >> 10u) & 0x1Fu);
    let fr = i32(fg_word & 0x1Fu);
    let fg = i32((fg_word >> 5u) & 0x1Fu);
    let fb = i32((fg_word >> 10u) & 0x1Fu);
    var r: i32; var g: i32; var b: i32;
    switch mode {
        case BLEND_AVERAGE: {
            r = (br >> 1u) + (fr >> 1u);
            g = (bg >> 1u) + (fg >> 1u);
            b = (bb >> 1u) + (fb >> 1u);
        }
        case BLEND_ADD: {
            r = min(br + fr, 31); g = min(bg + fg, 31); b = min(bb + fb, 31);
        }
        case BLEND_SUB: {
            r = max(br - fr, 0); g = max(bg - fg, 0); b = max(bb - fb, 0);
        }
        case BLEND_ADDQUARTER, default: {
            r = min(br + (fr >> 2u), 31);
            g = min(bg + (fg >> 2u), 31);
            b = min(bb + (fb >> 2u), 31);
        }
    }
    return u32(r) | (u32(g) << 5u) | (u32(b) << 10u) | (fg_word & 0x8000u);
}

// PS1 hardware-accurate texture fetch. Mirrors `Gpu::sample_texture`
// in emulator-core line-for-line. Returns 0 to mean "transparent" —
// the caller skips writing when the result is 0.
fn sample_texture(u_in: u32, v_in: u32) -> u32 {
    // 1. Mask to 8 bits — the PSX U/V counters are 8-bit, so a tpage
    //    wraps every 256 texels.
    let u8v = u_in & 0xFFu;
    let v8v = v_in & 0xFFu;
    // 2. Texture window: U' = (U & ~mask) | (off & mask), per-axis.
    //    Mask & off are already shifted ×8 on the host side.
    let mx = tpage.tex_window_mask_x;
    let my = tpage.tex_window_mask_y;
    let ox = tpage.tex_window_off_x;
    let oy = tpage.tex_window_off_y;
    let uw = (u8v & ~mx) | (ox & mx);
    let vw = (v8v & ~my) | (oy & my);
    // 3. tpage row: tpage_y + vw, with VRAM-height wrap.
    let tpy = (tpage.tpage_y + vw) & (VRAM_HEIGHT_U - 1u);
    var texel: u32 = 0u;
    switch tpage.tex_depth {
        case DEPTH_4BPP: {
            let tpx = (tpage.tpage_x + (uw >> 2u)) & (VRAM_WIDTH_U - 1u);
            let word = vram[tpy * VRAM_WIDTH_U + tpx];
            let idx = (word >> ((uw & 3u) * 4u)) & 0xFu;
            let cx = (prim.clut_x + idx) & (VRAM_WIDTH_U - 1u);
            let cy = prim.clut_y & (VRAM_HEIGHT_U - 1u);
            texel = vram[cy * VRAM_WIDTH_U + cx];
        }
        case DEPTH_8BPP: {
            let tpx = (tpage.tpage_x + (uw >> 1u)) & (VRAM_WIDTH_U - 1u);
            let word = vram[tpy * VRAM_WIDTH_U + tpx];
            let idx = (word >> ((uw & 1u) * 8u)) & 0xFFu;
            let cx = (prim.clut_x + idx) & (VRAM_WIDTH_U - 1u);
            let cy = prim.clut_y & (VRAM_HEIGHT_U - 1u);
            texel = vram[cy * VRAM_WIDTH_U + cx];
        }
        case DEPTH_15BPP, default: {
            let tpx = (tpage.tpage_x + uw) & (VRAM_WIDTH_U - 1u);
            texel = vram[tpy * VRAM_WIDTH_U + tpx];
        }
    }
    return texel;
}

// PSX tint modulation. CPU formula: result = (tint * texel) / 0x80,
// clamped to 0x1F per channel. tint = 0x80 is the identity — the
// raw-texture path bypasses this entirely.
fn modulate_tint(texel: u32, tint_packed: u32) -> u32 {
    let tr = texel & 0x1Fu;
    let tg = (texel >> 5u) & 0x1Fu;
    let tb = (texel >> 10u) & 0x1Fu;
    let cr = tint_packed & 0xFFu;
    let cg = (tint_packed >> 8u) & 0xFFu;
    let cb = (tint_packed >> 16u) & 0xFFu;
    let r = min((cr * tr) / 0x80u, 0x1Fu);
    let g = min((cg * tg) / 0x80u, 0x1Fu);
    let b = min((cb * tb) / 0x80u, 0x1Fu);
    // Bit 15 (mask / stp) preserved from the texel.
    return r | (g << 5u) | (b << 10u) | (texel & 0x8000u);
}

@compute @workgroup_size(8, 8)
fn rasterize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = prim.bbox_min.x + i32(gid.x);
    let py = prim.bbox_min.y + i32(gid.y);
    if px > prim.bbox_max.x || py > prim.bbox_max.y { return; }
    if px < draw_area.left || px > draw_area.right { return; }
    if py < draw_area.top || py > draw_area.bottom { return; }
    if px < 0 || px >= VRAM_WIDTH || py < 0 || py >= VRAM_HEIGHT { return; }

    let p = vec2<i32>(px, py);
    // Edge functions, oriented for both windings.
    let w0 = edge(prim.v1, prim.v2, p);
    let w1 = edge(prim.v2, prim.v0, p);
    let w2 = edge(prim.v0, prim.v1, p);
    let cw = (w0 < 0) || (w1 < 0) || (w2 < 0);
    let ccw = (w0 > 0) || (w1 > 0) || (w2 > 0);
    if cw && ccw { return; }
    if w0 == 0 && !is_top_left(prim.v1, prim.v2) { return; }
    if w1 == 0 && !is_top_left(prim.v2, prim.v0) { return; }
    if w2 == 0 && !is_top_left(prim.v0, prim.v1) { return; }

    // Barycentric UV. `area` is the signed 2× triangle area; sign
    // flips with winding, but it cancels because `w*` flip with it
    // too. Cast to f32 for the divide so the integer rounding mode
    // matches Q-format `>> 16` to within ±1 LSB on most pixels.
    //
    // (Pixel-exact parity vs the CPU scanline-delta walker is a
    // Phase-B.x follow-up — for now the parity tests use a small
    // tolerance on edge pixels.)
    let area = w0 + w1 + w2;
    let u0 = i32(prim.uv0 & 0xFFu);
    let v0v = i32((prim.uv0 >> 8u) & 0xFFu);
    let u1 = i32(prim.uv1 & 0xFFu);
    let v1v = i32((prim.uv1 >> 8u) & 0xFFu);
    let u2 = i32(prim.uv2 & 0xFFu);
    let v2v = i32((prim.uv2 >> 8u) & 0xFFu);
    let u_num = w0 * u0 + w1 * u1 + w2 * u2;
    let v_num = w0 * v0v + w1 * v1v + w2 * v2v;
    // Integer divide truncates toward zero; for parity we want
    // floor. Both the signed numerator and `area` may be negative
    // (CW winding), but their signs match → quotient is non-
    // negative, so truncation IS floor. Safe.
    let u = u32(u_num / area);
    let v = u32(v_num / area);

    let texel = sample_texture(u, v);
    if texel == 0u { return; }

    // Tint modulation.
    var fg: u32;
    if (prim.flags & FLAG_RAW_TEXTURE) != 0u {
        fg = texel;
    } else {
        fg = modulate_tint(texel, prim.tint);
    }

    let idx = u32(py * VRAM_WIDTH + px);

    // RMW: needs back-buffer read for mask-check or per-texel
    // semi-trans. PSX rule for textured prims: SEMI_TRANS only
    // applies when the texel's bit 15 is set. Otherwise the texel
    // draws opaque even on a SEMI_TRANS primitive.
    let semi_trans_active =
        ((prim.flags & FLAG_SEMI_TRANS) != 0u) && ((texel & 0x8000u) != 0u);
    let needs_read = ((prim.flags & FLAG_MASK_CHECK) != 0u) || semi_trans_active;
    var existing: u32 = 0u;
    if needs_read { existing = vram[idx]; }
    if (prim.flags & FLAG_MASK_CHECK) != 0u {
        if (existing & 0x8000u) != 0u { return; }
    }
    var pixel: u32;
    if semi_trans_active {
        let mode = (prim.flags >> 8u) & 0x3u;
        pixel = blend(existing, fg, mode);
    } else {
        pixel = fg;
    }
    if (prim.flags & FLAG_MASK_SET) != 0u {
        pixel = pixel | 0x8000u;
    }
    vram[idx] = pixel;
}
