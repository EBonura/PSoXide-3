// Textured triangle rasterizer — scanline-delta version (B.x).
//
// Reads per-row state from a storage buffer pre-computed on the host
// (mirroring `emulator-core::gpu::setup_sections` + `next_row` exactly)
// and walks per-pixel U / V using cumulative Q16.16 deltas. The math
// is bit-exact with the CPU rasterizer because both walk the same
// accumulator arithmetic.
//
// WGSL has no native i64. All Q16.16 values are split into
// `(hi: i32, lo: u32)` and the per-pixel addition / multiplication
// is done with explicit hi/lo emulation in `i64_*` helpers below.

struct TexTri {
    v0: vec2<i32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    bbox_min: vec2<i32>,
    bbox_max: vec2<i32>,
    uv0: u32,
    uv1: u32,
    uv2: u32,
    tint: u32,
    flags: u32,
    clut_x: u32,
    clut_y: u32,
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

struct RowState {
    left_x_hi: i32,
    left_x_lo: u32,
    right_x_hi: i32,
    right_x_lo: u32,
    left_u_hi: i32,
    left_u_lo: u32,
    left_v_hi: i32,
    left_v_lo: u32,
    left_r_hi: i32,
    left_r_lo: u32,
    left_g_hi: i32,
    left_g_lo: u32,
    left_b_hi: i32,
    left_b_lo: u32,
    _pad0: u32,
    _pad1: u32,
}

struct ScanlineConsts {
    y_min: i32,
    y_max: i32,
    _pad0: u32,
    _pad1: u32,
    delta_col_u_hi: i32,
    delta_col_u_lo: u32,
    delta_col_v_hi: i32,
    delta_col_v_lo: u32,
    delta_col_r_hi: i32,
    delta_col_r_lo: u32,
    delta_col_g_hi: i32,
    delta_col_g_lo: u32,
    delta_col_b_hi: i32,
    delta_col_b_lo: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> vram: array<u32>;
@group(0) @binding(1) var<uniform> prim: TexTri;
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

const BLEND_AVERAGE:    u32 = 0u;
const BLEND_ADD:        u32 = 1u;
const BLEND_SUB:        u32 = 2u;
const BLEND_ADDQUARTER: u32 = 3u;

const DEPTH_4BPP:  u32 = 0u;
const DEPTH_8BPP:  u32 = 1u;
const DEPTH_15BPP: u32 = 2u;

// =============================================================
//  i64 emulation: stored as (hi: i32, lo: u32). The "logical"
//  64-bit value is `(hi << 32) | lo`, treated as signed two's
//  complement. WGSL's `i32 >> N` is arithmetic and its `u32 + u32`
//  wraps mod 2^32 — both are what we need.
// =============================================================

struct I64 { hi: i32, lo: u32 }

fn i64_pack(hi: i32, lo: u32) -> I64 { return I64(hi, lo); }

// 64-bit add: a + b. Carry from the unsigned-low-half wrap.
fn i64_add(a: I64, b: I64) -> I64 {
    let new_lo = a.lo + b.lo;
    let carry: i32 = select(0, 1, new_lo < a.lo);
    let new_hi = a.hi + b.hi + carry;
    return I64(new_hi, new_lo);
}

// 64-bit unsigned-i32 multiply: `col * b` where `col` is a u32 in
// [0..1023] (always non-negative — `col = px - xmin`). The 64-bit
// product fits because `col` is bounded.
fn i64_mul_u32(col: u32, b: I64) -> I64 {
    // Split b.lo into 16-bit halves so each per-half multiply
    // fits in u32 (max `col * 0xFFFF = 1023 * 65535 ≈ 67M < 2^27`).
    let b_lo_l16 = b.lo & 0xFFFFu;
    let b_lo_h16 = b.lo >> 16u;
    let prod_l = col * b_lo_l16;     // contributes to low 32
    let prod_h = col * b_lo_h16;     // contributes to mid 32
    // Combine: result = (prod_h << 16) + prod_l, with carry into
    // the upper 32 bits.
    let new_lo_a = prod_h << 16u;
    let new_lo_b = prod_l;
    let new_lo = new_lo_a + new_lo_b;
    let carry_lo: u32 = select(0u, 1u, new_lo < new_lo_a);
    let high_part_from_lo = (prod_h >> 16u) + carry_lo;
    // `col * b.hi`: signed × unsigned. Treat col as i32 (it's small
    // — bbox width ≤ 1023 — so safe to cast).
    let high_part_from_hi = i32(col) * b.hi;
    let new_hi = high_part_from_hi + i32(high_part_from_lo);
    return I64(new_hi, new_lo);
}

// Arithmetic right-shift of an I64 by 16 — produces another I64,
// preserving sign. Used to convert Q16.16 → Q32.0 (still 64-bit
// in case the integer part doesn't fit in 32, though for PSX
// rasterizer outputs it always does).
fn i64_arsh16(a: I64) -> I64 {
    let new_lo = (a.lo >> 16u) | (u32(a.hi) << 16u);
    let new_hi = a.hi >> 16u; // arithmetic on i32
    return I64(new_hi, new_lo);
}

// Project to i32 from an I64 we know fits — for rasterizer outputs
// (xmin, u, v after >> 16) this is always true on real PSX content.
fn i64_to_i32(a: I64) -> i32 {
    return i32(a.lo);
}

// =============================================================
//  Sampler / blend / modulator (same as `tex_tri.wgsl`).
// =============================================================

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
    return r | (g << 5u) | (b << 10u) | (texel & 0x8000u);
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

    // Fetch this row's CPU-equivalent left-edge state.
    let row_idx = u32(py - consts.y_min);
    let row = rows[row_idx];

    // Per-row coverage (matches CPU: `xmin = left_x_q16 >> 16`,
    // `xmax = (right_x_q16 >> 16) - 1`). Both are shifts of the
    // hi-half (since left/right_x integer parts always fit in i32).
    let left_x_q16 = i64_pack(row.left_x_hi, row.left_x_lo);
    let right_x_q16 = i64_pack(row.right_x_hi, row.right_x_lo);
    let xmin_q32 = i64_arsh16(left_x_q16);
    let xmax_excl_q32 = i64_arsh16(right_x_q16);
    let xmin = i64_to_i32(xmin_q32);
    let xmax = i64_to_i32(xmax_excl_q32) - 1;
    if px < xmin || px > xmax { return; }

    // Per-pixel cumulative U / V via i64 add of (col * delta_col).
    let col = u32(px - xmin);
    let left_u_q16 = i64_pack(row.left_u_hi, row.left_u_lo);
    let left_v_q16 = i64_pack(row.left_v_hi, row.left_v_lo);
    let dcu = i64_pack(consts.delta_col_u_hi, consts.delta_col_u_lo);
    let dcv = i64_pack(consts.delta_col_v_hi, consts.delta_col_v_lo);
    let u_q16 = i64_add(left_u_q16, i64_mul_u32(col, dcu));
    let v_q16 = i64_add(left_v_q16, i64_mul_u32(col, dcv));
    let u = u32(i64_to_i32(i64_arsh16(u_q16)));
    let v = u32(i64_to_i32(i64_arsh16(v_q16)));

    let texel = sample_texture(u, v);
    if texel == 0u { return; }

    var fg: u32;
    if (prim.flags & FLAG_RAW_TEXTURE) != 0u {
        fg = texel;
    } else {
        fg = modulate_tint(texel, prim.tint);
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
