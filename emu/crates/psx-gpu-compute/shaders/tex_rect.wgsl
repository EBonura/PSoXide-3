// Textured rectangle rasterizer (GP0 0x64..=0x67 + fixed-size
// variants). Direct (U, V) blit from the active tpage with optional
// X/Y flip from GP0 0xE1 bits 12/13. No UV interpolation — the
// shader steps through texels linearly with `(base_u + dx,
// base_v + dy)`, so parity vs the CPU rasterizer is bit-exact.
//
// Sampling, modulation, and RMW logic mirrors `tex_tri.wgsl` line-
// for-line. See that file for the per-helper quirks (Redux blend
// math, raw-texture skip, per-texel semi-trans rule).

struct TexRect {
    xy: vec2<i32>,
    wh: vec2<u32>,
    uv: u32,         // u_base | (v_base << 8)
    clut_x: u32,
    clut_y: u32,
    tint: u32,
    flags: u32,      // PrimFlags + (BlendMode << 8) + FLIP_X/Y bits
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
@group(0) @binding(1) var<uniform> prim: TexRect;
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
const FLAG_FLIP_X:      u32 = 1u << 4u;
const FLAG_FLIP_Y:      u32 = 1u << 5u;

const BLEND_AVERAGE:    u32 = 0u;
const BLEND_ADD:        u32 = 1u;
const BLEND_SUB:        u32 = 2u;
const BLEND_ADDQUARTER: u32 = 3u;

const DEPTH_4BPP:  u32 = 0u;
const DEPTH_8BPP:  u32 = 1u;
const DEPTH_15BPP: u32 = 2u;

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
    if gid.x >= prim.wh.x || gid.y >= prim.wh.y { return; }
    let dx = gid.x;
    let dy = gid.y;
    let px = prim.xy.x + i32(dx);
    let py = prim.xy.y + i32(dy);
    if px < draw_area.left || px > draw_area.right { return; }
    if py < draw_area.top || py > draw_area.bottom { return; }
    if px < 0 || px >= VRAM_WIDTH || py < 0 || py >= VRAM_HEIGHT { return; }

    // Linear UV stepping. Texture-rect flip in GP0 0xE1 bits 12/13:
    // when set, the U or V offset COUNTS DOWN from `(w-1)` instead
    // of up from 0. Matches `paint_textured_rect` in emulator-core.
    let last_col = prim.wh.x - 1u;
    let last_row = prim.wh.y - 1u;
    let u_off = select(dx, last_col - dx, (prim.flags & FLAG_FLIP_X) != 0u);
    let v_off = select(dy, last_row - dy, (prim.flags & FLAG_FLIP_Y) != 0u);
    let u_base = prim.uv & 0xFFu;
    let v_base = (prim.uv >> 8u) & 0xFFu;
    let u = u_base + u_off;
    let v = v_base + v_off;

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
