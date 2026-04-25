// Textured Gouraud-shaded triangle — scanline-delta version (B.x).
//
// Composes the bit-exact UV walk from `tex_tri_scanline.wgsl` with
// the per-vertex tint interpolation walked the same way. The CPU
// rasterizer accumulates `c_r += delta_col_r` per pixel after the
// initial `left_r += delta_left_r` per row — we mirror it via the
// host-prepared `RowState` + `ScanlineConsts`.

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
    left: i32, top: i32, right: i32, bottom: i32,
}

struct RowState {
    left_x_hi: i32, left_x_lo: u32,
    right_x_hi: i32, right_x_lo: u32,
    left_u_hi: i32, left_u_lo: u32,
    left_v_hi: i32, left_v_lo: u32,
    left_r_hi: i32, left_r_lo: u32,
    left_g_hi: i32, left_g_lo: u32,
    left_b_hi: i32, left_b_lo: u32,
    _pad0: u32, _pad1: u32,
}

struct ScanlineConsts {
    y_min: i32, y_max: i32,
    _pad0: u32, _pad1: u32,
    delta_col_u_hi: i32, delta_col_u_lo: u32,
    delta_col_v_hi: i32, delta_col_v_lo: u32,
    delta_col_r_hi: i32, delta_col_r_lo: u32,
    delta_col_g_hi: i32, delta_col_g_lo: u32,
    delta_col_b_hi: i32, delta_col_b_lo: u32,
    _pad2: u32, _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> vram: array<u32>;
@group(0) @binding(1) var<uniform> prim: ShadedTexTri;
@group(0) @binding(2) var<uniform> draw_area: DrawArea;
@group(0) @binding(3) var<uniform> tpage: Tpage;
@group(0) @binding(4) var<storage, read> rows: array<RowState>;
@group(0) @binding(5) var<uniform> consts: ScanlineConsts;

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

// i64 emulation — same as tex_tri_scanline.wgsl. See that file for
// the explanation; in short, WGSL has no native i64 so we split
// every Q16.16 value into (hi: i32, lo: u32) and do explicit carry.
struct I64 { hi: i32, lo: u32 }
fn i64_pack(hi: i32, lo: u32) -> I64 { return I64(hi, lo); }

fn i64_add(a: I64, b: I64) -> I64 {
    let new_lo = a.lo + b.lo;
    let carry: i32 = select(0, 1, new_lo < a.lo);
    let new_hi = a.hi + b.hi + carry;
    return I64(new_hi, new_lo);
}

fn i64_mul_u32(col: u32, b: I64) -> I64 {
    let b_lo_l16 = b.lo & 0xFFFFu;
    let b_lo_h16 = b.lo >> 16u;
    let prod_l = col * b_lo_l16;
    let prod_h = col * b_lo_h16;
    let new_lo_a = prod_h << 16u;
    let new_lo_b = prod_l;
    let new_lo = new_lo_a + new_lo_b;
    let carry_lo: u32 = select(0u, 1u, new_lo < new_lo_a);
    let high_part_from_lo = (prod_h >> 16u) + carry_lo;
    let high_part_from_hi = i32(col) * b.hi;
    let new_hi = high_part_from_hi + i32(high_part_from_lo);
    return I64(new_hi, new_lo);
}

fn i64_arsh16(a: I64) -> I64 {
    let new_lo = (a.lo >> 16u) | (u32(a.hi) << 16u);
    let new_hi = a.hi >> 16u;
    return I64(new_hi, new_lo);
}

fn i64_to_i32(a: I64) -> i32 { return i32(a.lo); }

// Texture sampling, blend, modulation — same as tex_tri.

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

fn modulate_5bit(texel: u32, tr: u32, tg: u32, tb: u32) -> u32 {
    let txr = texel & 0x1Fu;
    let txg = (texel >> 5u) & 0x1Fu;
    let txb = (texel >> 10u) & 0x1Fu;
    let r = min((tr * txr) / 0x80u, 0x1Fu);
    let g = min((tg * txg) / 0x80u, 0x1Fu);
    let b = min((tb * txb) / 0x80u, 0x1Fu);
    return r | (g << 5u) | (b << 10u) | (texel & 0x8000u);
}

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
    if py < consts.y_min || py > consts.y_max { return; }
    if px < draw_area.left || px > draw_area.right { return; }
    if py < draw_area.top || py > draw_area.bottom { return; }
    if px < 0 || px >= VRAM_WIDTH || py < 0 || py >= VRAM_HEIGHT { return; }

    let row_idx = u32(py - consts.y_min);
    let row = rows[row_idx];

    // Coverage and per-pixel U/V (same as tex_tri_scanline).
    let left_x_q16 = i64_pack(row.left_x_hi, row.left_x_lo);
    let right_x_q16 = i64_pack(row.right_x_hi, row.right_x_lo);
    let xmin = i64_to_i32(i64_arsh16(left_x_q16));
    let xmax = i64_to_i32(i64_arsh16(right_x_q16)) - 1;
    if px < xmin || px > xmax { return; }
    let col = u32(px - xmin);

    let dcu = i64_pack(consts.delta_col_u_hi, consts.delta_col_u_lo);
    let dcv = i64_pack(consts.delta_col_v_hi, consts.delta_col_v_lo);
    let u_q16 = i64_add(
        i64_pack(row.left_u_hi, row.left_u_lo),
        i64_mul_u32(col, dcu),
    );
    let v_q16 = i64_add(
        i64_pack(row.left_v_hi, row.left_v_lo),
        i64_mul_u32(col, dcv),
    );
    let u = u32(i64_to_i32(i64_arsh16(u_q16)));
    let v = u32(i64_to_i32(i64_arsh16(v_q16)));

    let texel = sample_texture(u, v);
    if texel == 0u { return; }

    var fg: u32;
    if (prim.flags & FLAG_RAW_TEXTURE) != 0u {
        fg = texel;
    } else {
        // Per-pixel cumulative tint walk. Same i64 add as U/V but
        // for the R/G/B channels, then `>> 16` to recover the 8-bit
        // value the modulator expects.
        let dcr = i64_pack(consts.delta_col_r_hi, consts.delta_col_r_lo);
        let dcg = i64_pack(consts.delta_col_g_hi, consts.delta_col_g_lo);
        let dcb = i64_pack(consts.delta_col_b_hi, consts.delta_col_b_lo);
        let r_q16 = i64_add(
            i64_pack(row.left_r_hi, row.left_r_lo),
            i64_mul_u32(col, dcr),
        );
        let g_q16 = i64_add(
            i64_pack(row.left_g_hi, row.left_g_lo),
            i64_mul_u32(col, dcg),
        );
        let b_q16 = i64_add(
            i64_pack(row.left_b_hi, row.left_b_lo),
            i64_mul_u32(col, dcb),
        );
        let r = u32(clamp(i64_to_i32(i64_arsh16(r_q16)), 0, 255));
        let g = u32(clamp(i64_to_i32(i64_arsh16(g_q16)), 0, 255));
        let b = u32(clamp(i64_to_i32(i64_arsh16(b_q16)), 0, 255));
        let dither = (prim.flags & FLAG_DITHER) != 0u;
        if dither {
            fg = modulate_dithered(texel, r, g, b, px, py);
        } else {
            fg = modulate_5bit(texel, r, g, b);
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
