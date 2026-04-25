// Axis-aligned textured quad with bilinear UV interpolation.
//
// Mirrors `emulator-core::Gpu::rasterize_axis_aligned_textured_quad`
// byte-for-byte: per-row left/right UV walk in Q16.16, per-pixel
// linear interpolation between them. The CPU dispatches this fast
// path when a textured quad's vertices form an axis-aligned
// rectangle; for non-affine UV layouts the bilinear math here
// produces different pixels than triangle-split barycentric.
//
// Layout assumption (matches CPU): v0 top-left, v1 top-right,
// v2 bottom-left, v3 bottom-right. UVs in the same order.
//
// i64 emulation helpers identical to the scanline shaders — see
// `tex_tri_scanline.wgsl` for explanation.

struct TexQuadBilinear {
    v0: vec2<i32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    v3: vec2<i32>,
    uv0: u32, uv1: u32, uv2: u32, uv3: u32,
    clut_x: u32,
    clut_y: u32,
    tint: u32,
    flags: u32,
}

struct Tpage {
    tpage_x: u32, tpage_y: u32, tex_depth: u32, _pad: u32,
    tex_window_mask_x: u32, tex_window_mask_y: u32,
    tex_window_off_x: u32, tex_window_off_y: u32,
}

struct DrawArea { left: i32, top: i32, right: i32, bottom: i32 }

@group(0) @binding(0) var<storage, read_write> vram: array<u32>;
@group(0) @binding(1) var<uniform> prim: TexQuadBilinear;
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

struct I64 { hi: i32, lo: u32 }
fn i64_pack(hi: i32, lo: u32) -> I64 { return I64(hi, lo); }

fn i64_add(a: I64, b: I64) -> I64 {
    let new_lo = a.lo + b.lo;
    let carry: i32 = select(0, 1, new_lo < a.lo);
    return I64(a.hi + b.hi + carry, new_lo);
}

fn i64_sub(a: I64, b: I64) -> I64 {
    // a - b == a + (-b). Two's complement: -b = ~b + 1.
    let neg_lo = (~b.lo) + 1u;
    // If b.lo was 0, ~b.lo + 1 wraps to 0 (no carry); else there's a
    // borrow into the high half. Equivalently: borrow = 1 iff b.lo != 0.
    let neg_hi = (~b.hi) + select(0, 1, b.lo == 0u);
    return i64_add(a, I64(neg_hi, neg_lo));
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

// Q16.16 divide of an I64 by a positive i32 scalar. Used to compute
// per-pixel `delta_u = (right_u - pos_u) / width`. Implementation:
// long division. Width is at most 1023, so we don't need full
// 64-bit-by-32-bit precision — but we DO need it to handle negative
// numerators correctly.
fn i64_div_i32(a: I64, divisor: i32) -> I64 {
    // For PSX rasterizer outputs, `a` always fits in i32 by the
    // time it's being divided here (it's the difference of two
    // 5-bit-channel-times-Q16.16 values, whose magnitude is bounded
    // by 255 << 16 = 16M, well within i32). So we can safely
    // collapse to i32 division.
    let n_i32 = (i32(a.lo)) | (a.hi << 0); // hi already 0 or -1 typically
    // Defensive: if `a.hi` is non-zero AND inconsistent with the
    // sign of a.lo's MSB, the value didn't fit in i32 and we'd
    // overflow. In practice rasterizer values are within range.
    let q = n_i32 / divisor;
    if q < 0 { return I64(-1, u32(q)); }
    return I64(0, u32(q));
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
    // Quad bbox: v0 is top-left, v3 is bottom-right.
    let left = prim.v0.x;
    let right = prim.v1.x;
    let top = prim.v0.y;
    let bottom = prim.v2.y;
    let width = right - left;
    let height = bottom - top;
    if width <= 0 || height <= 0 { return; }

    let dx = i32(gid.x);
    let dy = i32(gid.y);
    if dx >= width || dy >= height { return; }
    let px = left + dx;
    let py = top + dy;
    if px < draw_area.left || px > draw_area.right { return; }
    if py < draw_area.top || py > draw_area.bottom { return; }
    if px < 0 || px >= VRAM_WIDTH || py < 0 || py >= VRAM_HEIGHT { return; }

    // CPU fast-path math (lines 1696..1731 of `gpu.rs`):
    //   left_u0  = t0.u << 16
    //   right_u0 = t1.u << 16
    //   delta_left_u  = ((t2.u << 16) - left_u0)  / height
    //   delta_right_u = ((t3.u << 16) - right_u0) / height
    //   pos_u   = left_u0 + row * delta_left_u
    //   right_u = right_u0 + row * delta_right_u
    //   delta_col_u = (right_u - pos_u) / width
    //   u = (pos_u + col * delta_col_u) >> 16
    let u0 = i32(prim.uv0 & 0xFFu);
    let v0 = i32((prim.uv0 >> 8u) & 0xFFu);
    let u1 = i32(prim.uv1 & 0xFFu);
    let v1 = i32((prim.uv1 >> 8u) & 0xFFu);
    let u2 = i32(prim.uv2 & 0xFFu);
    let v2 = i32((prim.uv2 >> 8u) & 0xFFu);
    let u3 = i32(prim.uv3 & 0xFFu);
    let v3 = i32((prim.uv3 >> 8u) & 0xFFu);

    // Q16.16 = original << 16. Pack as I64: hi=0 (since values are
    // small positive), lo = (value << 16) as u32. For negatives we
    // sign-extend.
    let left_u0 = i64_pack(u0 >> 16, u32(u0) << 16u);
    let left_v0 = i64_pack(v0 >> 16, u32(v0) << 16u);
    let right_u0 = i64_pack(u1 >> 16, u32(u1) << 16u);
    let right_v0 = i64_pack(v1 >> 16, u32(v1) << 16u);
    let bl_u = i64_pack(u2 >> 16, u32(u2) << 16u);
    let bl_v = i64_pack(v2 >> 16, u32(v2) << 16u);
    let br_u = i64_pack(u3 >> 16, u32(u3) << 16u);
    let br_v = i64_pack(v3 >> 16, u32(v3) << 16u);

    let delta_left_u = i64_div_i32(i64_sub(bl_u, left_u0), height);
    let delta_left_v = i64_div_i32(i64_sub(bl_v, left_v0), height);
    let delta_right_u = i64_div_i32(i64_sub(br_u, right_u0), height);
    let delta_right_v = i64_div_i32(i64_sub(br_v, right_v0), height);

    let row_u_left = i64_add(left_u0, i64_mul_u32(u32(dy), delta_left_u));
    let row_v_left = i64_add(left_v0, i64_mul_u32(u32(dy), delta_left_v));
    let row_u_right = i64_add(right_u0, i64_mul_u32(u32(dy), delta_right_u));
    let row_v_right = i64_add(right_v0, i64_mul_u32(u32(dy), delta_right_v));

    let delta_col_u = i64_div_i32(i64_sub(row_u_right, row_u_left), width);
    let delta_col_v = i64_div_i32(i64_sub(row_v_right, row_v_left), width);

    let u_q16 = i64_add(row_u_left, i64_mul_u32(u32(dx), delta_col_u));
    let v_q16 = i64_add(row_v_left, i64_mul_u32(u32(dx), delta_col_v));
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
