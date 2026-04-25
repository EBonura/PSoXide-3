// Textured Gouraud-shaded triangle (GP0 0x34..=0x37).
//
// Composition of:
//   - tex_tri.wgsl  (texture sampling: tpage, CLUT, U/V wrap, window)
//   - shaded_tri.wgsl (per-vertex tint interpolation + dither)
//
// The PSX rule for textured-shaded: the per-vertex tint is
// interpolated barycentrically across the triangle, then modulates
// the sampled texel. With dither (`PrimFlags::DITHER`) the
// modulation runs in 8-bit space — see
// `emulator-core::modulate_tint_dithered`.

struct ShadedTexTri {
    v0: vec2<i32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    bbox_min: vec2<i32>,
    bbox_max: vec2<i32>,
    c0: u32,
    c1: u32,
    c2: u32,
    uv0: u32,
    uv1: u32,
    uv2: u32,
    flags: u32,
    clut_x: u32,
    clut_y: u32,
    _pad: u32,
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
@group(0) @binding(1) var<uniform> prim: ShadedTexTri;
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
const FLAG_DITHER:      u32 = 1u << 6u;

const BLEND_AVERAGE:    u32 = 0u;
const BLEND_ADD:        u32 = 1u;
const BLEND_SUB:        u32 = 2u;
const BLEND_ADDQUARTER: u32 = 3u;

const DEPTH_4BPP:  u32 = 0u;
const DEPTH_8BPP:  u32 = 1u;
const DEPTH_15BPP: u32 = 2u;

const DITHER_TABLE: array<u32, 16> = array<u32, 16>(
    7u, 0u, 6u, 1u, 2u, 5u, 3u, 4u,
    1u, 6u, 0u, 7u, 4u, 3u, 5u, 2u,
);

fn edge(a: vec2<i32>, b: vec2<i32>, p: vec2<i32>) -> i32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn is_top_left(a: vec2<i32>, b: vec2<i32>) -> bool {
    let d = b - a;
    return (d.y == 0 && d.x > 0) || (d.y < 0);
}

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

fn sample_texture(u_in: u32, v_in: u32) -> u32 {
    let u8v = u_in & 0xFFu;
    let v8v = v_in & 0xFFu;
    let mx = tpage.tex_window_mask_x;
    let my = tpage.tex_window_mask_y;
    let ox = tpage.tex_window_off_x;
    let oy = tpage.tex_window_off_y;
    let uw = (u8v & ~mx) | (ox & mx);
    let vw = (v8v & ~my) | (oy & my);
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

// Modulate (texel × tint / 0x80) per channel, NO dither — direct
// 5-bit output. Mirrors `emulator-core::modulate_tint`.
fn modulate_5bit(texel: u32, tr: u32, tg: u32, tb: u32) -> u32 {
    let txr = texel & 0x1Fu;
    let txg = (texel >> 5u) & 0x1Fu;
    let txb = (texel >> 10u) & 0x1Fu;
    let r = min((tr * txr) / 0x80u, 0x1Fu);
    let g = min((tg * txg) / 0x80u, 0x1Fu);
    let b = min((tb * txb) / 0x80u, 0x1Fu);
    return r | (g << 5u) | (b << 10u) | (texel & 0x8000u);
}

// Modulate WITH dither: scale 5-bit texel up to 8-bit, modulate in
// 8-bit space, then dither down to 5 bits per channel. Mirrors
// `emulator-core::modulate_tint_dithered`.
fn modulate_dithered(texel: u32, tr: u32, tg: u32, tb: u32, x: i32, y: i32) -> u32 {
    let txr = (texel & 0x1Fu) << 3u;
    let txg = ((texel >> 5u) & 0x1Fu) << 3u;
    let txb = ((texel >> 10u) & 0x1Fu) << 3u;
    let r = min((tr * txr) / 0x80u, 0xFFu);
    let g = min((tg * txg) / 0x80u, 0xFFu);
    let b = min((tb * txb) / 0x80u, 0xFFu);
    let coeff = DITHER_TABLE[u32(y & 3) * 4u + u32(x & 3)];
    var rc = r >> 3u;
    var gc = g >> 3u;
    var bc = b >> 3u;
    if rc < 0x1Fu && (r & 7u) > coeff { rc = rc + 1u; }
    if gc < 0x1Fu && (g & 7u) > coeff { gc = gc + 1u; }
    if bc < 0x1Fu && (b & 7u) > coeff { bc = bc + 1u; }
    return rc | (gc << 5u) | (bc << 10u) | (texel & 0x8000u);
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
    let w0 = edge(prim.v1, prim.v2, p);
    let w1 = edge(prim.v2, prim.v0, p);
    let w2 = edge(prim.v0, prim.v1, p);
    let cw = (w0 < 0) || (w1 < 0) || (w2 < 0);
    let ccw = (w0 > 0) || (w1 > 0) || (w2 > 0);
    if cw && ccw { return; }
    if w0 == 0 && !is_top_left(prim.v1, prim.v2) { return; }
    if w1 == 0 && !is_top_left(prim.v2, prim.v0) { return; }
    if w2 == 0 && !is_top_left(prim.v0, prim.v1) { return; }

    let area = w0 + w1 + w2;
    let u_num = w0 * i32(prim.uv0 & 0xFFu)
              + w1 * i32(prim.uv1 & 0xFFu)
              + w2 * i32(prim.uv2 & 0xFFu);
    let v_num = w0 * i32((prim.uv0 >> 8u) & 0xFFu)
              + w1 * i32((prim.uv1 >> 8u) & 0xFFu)
              + w2 * i32((prim.uv2 >> 8u) & 0xFFu);
    let u = u32(u_num / area);
    let v = u32(v_num / area);

    let texel = sample_texture(u, v);
    if texel == 0u { return; }

    var fg: u32;
    if (prim.flags & FLAG_RAW_TEXTURE) != 0u {
        fg = texel;
    } else {
        // Interpolated tint per channel.
        let r0 = i32(prim.c0 & 0xFFu);
        let g0 = i32((prim.c0 >> 8u) & 0xFFu);
        let b0 = i32((prim.c0 >> 16u) & 0xFFu);
        let r1 = i32(prim.c1 & 0xFFu);
        let g1 = i32((prim.c1 >> 8u) & 0xFFu);
        let b1 = i32((prim.c1 >> 16u) & 0xFFu);
        let r2 = i32(prim.c2 & 0xFFu);
        let g2 = i32((prim.c2 >> 8u) & 0xFFu);
        let b2 = i32((prim.c2 >> 16u) & 0xFFu);
        let tr = clamp((w0 * r0 + w1 * r1 + w2 * r2) / area, 0, 255);
        let tg = clamp((w0 * g0 + w1 * g1 + w2 * g2) / area, 0, 255);
        let tb = clamp((w0 * b0 + w1 * b1 + w2 * b2) / area, 0, 255);
        let dither = (prim.flags & FLAG_DITHER) != 0u;
        if dither {
            fg = modulate_dithered(texel, u32(tr), u32(tg), u32(tb), px, py);
        } else {
            fg = modulate_5bit(texel, u32(tr), u32(tg), u32(tb));
        }
    }

    let idx = u32(py * VRAM_WIDTH + px);
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
