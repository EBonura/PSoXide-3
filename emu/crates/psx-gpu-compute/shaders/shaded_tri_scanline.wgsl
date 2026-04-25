// Gouraud-shaded triangle — scanline-delta version (B.x). RGB-only
// (no texture). Walks the same per-pixel `c_r += delta_col_r`
// accumulator the CPU uses, producing bit-exact output.

struct ShadedTri {
    v0: vec2<i32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    bbox_min: vec2<i32>,
    bbox_max: vec2<i32>,
    c0: u32, c1: u32, c2: u32,
    flags: u32,
    _pad0: u32, _pad1: u32,
}

struct DrawArea { left: i32, top: i32, right: i32, bottom: i32 }

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
@group(0) @binding(1) var<uniform> prim: ShadedTri;
@group(0) @binding(2) var<uniform> draw_area: DrawArea;
@group(0) @binding(3) var<storage, read> rows: array<RowState>;
@group(0) @binding(4) var<uniform> consts: ScanlineConsts;

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

const DITHER_TABLE: array<u32, 16> = array<u32, 16>(
    7u, 0u, 6u, 1u, 2u, 5u, 3u, 4u,
    1u, 6u, 0u, 7u, 4u, 3u, 5u, 2u,
);

struct I64 { hi: i32, lo: u32 }
fn i64_pack(hi: i32, lo: u32) -> I64 { return I64(hi, lo); }
fn i64_add(a: I64, b: I64) -> I64 {
    let new_lo = a.lo + b.lo;
    let carry: i32 = select(0, 1, new_lo < a.lo);
    return I64(a.hi + b.hi + carry, new_lo);
}
fn i64_mul_u32(col: u32, b: I64) -> I64 {
    let bL = b.lo & 0xFFFFu;
    let bH = b.lo >> 16u;
    let pL = col * bL;
    let pH = col * bH;
    let new_lo_a = pH << 16u;
    let new_lo = new_lo_a + pL;
    let carry: u32 = select(0u, 1u, new_lo < new_lo_a);
    let high_from_lo = (pH >> 16u) + carry;
    let high_from_hi = i32(col) * b.hi;
    return I64(high_from_hi + i32(high_from_lo), new_lo);
}
fn i64_arsh16(a: I64) -> I64 {
    return I64(a.hi >> 16u, (a.lo >> 16u) | (u32(a.hi) << 16u));
}
fn i64_to_i32(a: I64) -> i32 { return i32(a.lo); }

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

fn pack_rgb_5bit(r8: u32, g8: u32, b8: u32, dither: bool, x: i32, y: i32) -> u32 {
    let r = clamp(i32(r8), 0, 255);
    let g = clamp(i32(g8), 0, 255);
    let b = clamp(i32(b8), 0, 255);
    var rc = u32(r) >> 3u;
    var gc = u32(g) >> 3u;
    var bc = u32(b) >> 3u;
    if dither {
        let coeff = DITHER_TABLE[u32(y & 3) * 4u + u32(x & 3)];
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
    if py < consts.y_min || py > consts.y_max { return; }
    if px < draw_area.left || px > draw_area.right { return; }
    if py < draw_area.top || py > draw_area.bottom { return; }
    if px < 0 || px >= VRAM_WIDTH || py < 0 || py >= VRAM_HEIGHT { return; }

    let row_idx = u32(py - consts.y_min);
    let row = rows[row_idx];

    let xmin = i64_to_i32(i64_arsh16(i64_pack(row.left_x_hi, row.left_x_lo)));
    let xmax = i64_to_i32(i64_arsh16(i64_pack(row.right_x_hi, row.right_x_lo))) - 1;
    if px < xmin || px > xmax { return; }
    let col = u32(px - xmin);

    // Per-pixel cumulative RGB.
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
    let r = u32(i64_to_i32(i64_arsh16(r_q16)));
    let g = u32(i64_to_i32(i64_arsh16(g_q16)));
    let b = u32(i64_to_i32(i64_arsh16(b_q16)));

    let dither = (prim.flags & FLAG_DITHER) != 0u;
    let fg = pack_rgb_5bit(r, g, b, dither, px, py);

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
