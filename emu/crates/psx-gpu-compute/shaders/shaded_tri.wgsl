// Gouraud-shaded flat triangle (GP0 0x30..=0x33).
//
// Same coverage rules as `mono_tri.wgsl` (edge-function test, top-
// left fill rule). Per-pixel colour is barycentrically interpolated
// from the three vertex 24-bit RGB triples; if `DITHER` is set the
// 4×4 Bayer matrix is applied before truncating to 5 bits per
// channel. RMW (mask + semi-trans) identical to mono_tri.wgsl.
//
// Per-pixel UV-style parity tolerance vs the CPU scanline-delta
// math applies here too — the colour interpolation uses the same
// affine math but rounds slightly differently from the CPU's
// Q16.16 cumulative deltas. Inside-pixels match in colour to within
// ±1 in any 5-bit channel; tests bound that.

struct ShadedTri {
    v0: vec2<i32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    bbox_min: vec2<i32>,
    bbox_max: vec2<i32>,
    c0: u32,    // R | G<<8 | B<<16
    c1: u32,
    c2: u32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
}

struct DrawArea {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

@group(0) @binding(0) var<storage, read_write> vram: array<u32>;
@group(0) @binding(1) var<uniform> prim: ShadedTri;
@group(0) @binding(2) var<uniform> draw_area: DrawArea;

const VRAM_WIDTH: i32 = 1024;
const VRAM_HEIGHT: i32 = 512;

const FLAG_SEMI_TRANS: u32 = 1u << 0u;
const FLAG_MASK_CHECK: u32 = 1u << 1u;
const FLAG_MASK_SET:   u32 = 1u << 2u;
const FLAG_DITHER:     u32 = 1u << 6u;

const BLEND_AVERAGE:    u32 = 0u;
const BLEND_ADD:        u32 = 1u;
const BLEND_SUB:        u32 = 2u;
const BLEND_ADDQUARTER: u32 = 3u;

// 4×4 Bayer dither matrix. Index is `(y & 3) * 4 + (x & 3)`. Same
// table as `emulator-core::DITHER_COEFFS` so the per-pixel round-up
// rule matches Redux byte-for-byte.
const DITHER: array<u32, 16> = array<u32, 16>(
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

// Convert an 8-bit RGB triple to BGR15 with optional dither. Mirrors
// `dither_rgb` + `rgb24_to_bgr15` in `emulator-core` byte-for-byte.
fn pack_rgb_5bit(r8: u32, g8: u32, b8: u32, dither: bool, x: i32, y: i32) -> u32 {
    let r = clamp(i32(r8), 0, 255);
    let g = clamp(i32(g8), 0, 255);
    let b = clamp(i32(b8), 0, 255);
    var rc = u32(r) >> 3u;
    var gc = u32(g) >> 3u;
    var bc = u32(b) >> 3u;
    if dither {
        let coeff = DITHER[u32(y & 3) * 4u + u32(x & 3)];
        if rc < 0x1Fu && (u32(r) & 7u) > coeff { rc = rc + 1u; }
        if gc < 0x1Fu && (u32(g) & 7u) > coeff { gc = gc + 1u; }
        if bc < 0x1Fu && (u32(b) & 7u) > coeff { bc = bc + 1u; }
    }
    return rc | (gc << 5u) | (bc << 10u);
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

    // Barycentric interpolation of the 8-bit channels. The signed
    // `area` cancels because the `w*` flip with it (CW vs CCW).
    let area = w0 + w1 + w2;
    let r0 = i32(prim.c0 & 0xFFu);
    let g0 = i32((prim.c0 >> 8u) & 0xFFu);
    let b0 = i32((prim.c0 >> 16u) & 0xFFu);
    let r1 = i32(prim.c1 & 0xFFu);
    let g1 = i32((prim.c1 >> 8u) & 0xFFu);
    let b1 = i32((prim.c1 >> 16u) & 0xFFu);
    let r2 = i32(prim.c2 & 0xFFu);
    let g2 = i32((prim.c2 >> 8u) & 0xFFu);
    let b2 = i32((prim.c2 >> 16u) & 0xFFu);
    let r = (w0 * r0 + w1 * r1 + w2 * r2) / area;
    let g = (w0 * g0 + w1 * g1 + w2 * g2) / area;
    let b = (w0 * b0 + w1 * b1 + w2 * b2) / area;

    let dither = (prim.flags & FLAG_DITHER) != 0u;
    let fg = pack_rgb_5bit(u32(r), u32(g), u32(b), dither, px, py);

    let idx = u32(py * VRAM_WIDTH + px);
    let needs_read =
        ((prim.flags & FLAG_MASK_CHECK) != 0u) ||
        ((prim.flags & FLAG_SEMI_TRANS) != 0u);
    var existing: u32 = 0u;
    if needs_read { existing = vram[idx]; }
    if (prim.flags & FLAG_MASK_CHECK) != 0u {
        if (existing & 0x8000u) != 0u { return; }
    }
    var pixel: u32;
    if (prim.flags & FLAG_SEMI_TRANS) != 0u {
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
