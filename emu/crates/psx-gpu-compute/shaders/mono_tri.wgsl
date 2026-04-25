// Monochrome flat triangle rasterizer (GP0 0x20..=0x23).
//
// One dispatch covers the primitive's bounding box. Each thread tests
// ONE pixel:
//   - inside the drawing-area clip rect?
//   - inside the triangle (edge-function test, top-left fill rule)?
//   - if MASK_CHECK: skip if existing VRAM bit 15 is set.
//   - if SEMI_TRANS: read existing pixel, blend with `prim.color`,
//     write back; else write `prim.color` directly.
//   - if MASK_SET: OR bit 15 into the written pixel.
//
// The blend math mirrors `emulator-core::blend_pixel` exactly,
// including its three Redux quirks (see comments below).

struct MonoTri {
    v0: vec2<i32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    bbox_min: vec2<i32>,
    bbox_max: vec2<i32>,
    color: u32,
    /// PrimFlags::bits() | (BlendMode << 8). See primitive.rs.
    flags: u32,
}

struct DrawArea {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

@group(0) @binding(0) var<storage, read_write> vram: array<u32>;
@group(0) @binding(1) var<uniform> prim: MonoTri;
@group(0) @binding(2) var<uniform> draw_area: DrawArea;

const VRAM_WIDTH: i32 = 1024;
const VRAM_HEIGHT: i32 = 512;

const FLAG_SEMI_TRANS: u32 = 1u << 0u;
const FLAG_MASK_CHECK: u32 = 1u << 1u;
const FLAG_MASK_SET:   u32 = 1u << 2u;

const BLEND_AVERAGE:    u32 = 0u;
const BLEND_ADD:        u32 = 1u;
const BLEND_SUB:        u32 = 2u;
const BLEND_ADDQUARTER: u32 = 3u;

fn edge(a: vec2<i32>, b: vec2<i32>, p: vec2<i32>) -> i32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn is_top_left(a: vec2<i32>, b: vec2<i32>) -> bool {
    let d = b - a;
    let is_top = d.y == 0 && d.x > 0;
    let is_left = d.y < 0;
    return is_top || is_left;
}

// Blend two BGR15 pixels using the PSX semi-transparency rules.
// Mirrors `emulator-core::blend_pixel` byte-for-byte, including the
// three known Redux quirks:
//   1. Average pre-shifts BOTH operands before summing
//      (`(b>>1) + (f>>1)`, NOT `(b+f) >> 1`).
//   2. AddQuarter uses integer divide-by-4 on the 5-bit channel
//      (== `>> 2`).
//   3. The result's bit 15 comes from the FOREGROUND, not the back
//      buffer — even though we read the back buffer for the channel
//      math.
fn blend(bg_word: u32, fg_word: u32, mode: u32) -> u32 {
    let br = i32(bg_word & 0x1Fu);
    let bg = i32((bg_word >> 5u) & 0x1Fu);
    let bb = i32((bg_word >> 10u) & 0x1Fu);
    let fr = i32(fg_word & 0x1Fu);
    let fg = i32((fg_word >> 5u) & 0x1Fu);
    let fb = i32((fg_word >> 10u) & 0x1Fu);

    var r: i32;
    var g: i32;
    var b: i32;
    switch mode {
        case BLEND_AVERAGE: {
            // Quirk 1: pre-shift each operand independently.
            r = (br >> 1u) + (fr >> 1u);
            g = (bg >> 1u) + (fg >> 1u);
            b = (bb >> 1u) + (fb >> 1u);
        }
        case BLEND_ADD: {
            r = min(br + fr, 31);
            g = min(bg + fg, 31);
            b = min(bb + fb, 31);
        }
        case BLEND_SUB: {
            r = max(br - fr, 0);
            g = max(bg - fg, 0);
            b = max(bb - fb, 0);
        }
        case BLEND_ADDQUARTER, default: {
            // Quirk 2: `f / 4` on the 5-bit channel = `f >> 2`.
            r = min(br + (fr >> 2u), 31);
            g = min(bg + (fg >> 2u), 31);
            b = min(bb + (fb >> 2u), 31);
        }
    }
    // Quirk 3: bit 15 comes from the foreground word.
    return u32(r) | (u32(g) << 5u) | (u32(b) << 10u) | (fg_word & 0x8000u);
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

    let idx = u32(py * VRAM_WIDTH + px);

    // RMW path. Read the back buffer if either MASK_CHECK or
    // SEMI_TRANS is set; otherwise we can skip the load.
    let needs_read =
        ((prim.flags & FLAG_MASK_CHECK) != 0u) ||
        ((prim.flags & FLAG_SEMI_TRANS) != 0u);
    var existing: u32 = 0u;
    if needs_read {
        existing = vram[idx];
    }
    // Mask-check: skip pixels whose existing bit 15 is already set.
    if (prim.flags & FLAG_MASK_CHECK) != 0u {
        if (existing & 0x8000u) != 0u { return; }
    }
    // Compute the new pixel.
    var pixel: u32;
    if (prim.flags & FLAG_SEMI_TRANS) != 0u {
        let mode = (prim.flags >> 8u) & 0x3u;
        pixel = blend(existing, prim.color, mode);
    } else {
        pixel = prim.color;
    }
    // Mask-set: OR bit 15 into the written pixel.
    if (prim.flags & FLAG_MASK_SET) != 0u {
        pixel = pixel | 0x8000u;
    }
    vram[idx] = pixel;
}
