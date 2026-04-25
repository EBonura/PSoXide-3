// Monochrome flat triangle rasterizer (GP0 0x20..=0x23).
//
// One dispatch covers exactly the primitive's bounding box. Each
// thread tests ONE pixel:
//   - inside the drawing-area clip rect?
//   - inside the triangle (edge-function test, top-left fill rule)?
// If yes, write `prim.color` into VRAM at `[y * 1024 + x]`.
//
// The "top-left" fill rule: a pixel is inside the triangle when all
// three edge functions are >= 0, with edges that are top or
// upward-going-left treated as inclusive. This matches the CPU
// rasterizer's scanline-walk semantics (Redux's drawPoly3 family) at
// edges that cross integer pixel centres.
//
// Coordinate system:
//   - Pixel centres are at integer coordinates (PSX hardware
//     convention), not at +0.5.
//   - VRAM index = y * 1024 + x.

struct MonoTri {
    v0: vec2<i32>,
    v1: vec2<i32>,
    v2: vec2<i32>,
    bbox_min: vec2<i32>,
    bbox_max: vec2<i32>,
    color: u32,
    _pad: u32,
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

// Edge function: sign tells which side of edge a→b a point p is on.
// = (b - a) × (p - a). Positive → CCW side, zero → on edge.
fn edge(a: vec2<i32>, b: vec2<i32>, p: vec2<i32>) -> i32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Top-left rule: an edge is "inclusive" if it's a top edge (Δy = 0,
// Δx > 0) or a left edge (Δy < 0). This is the standard fill-rule
// nudge so two triangles sharing an edge don't double-cover or
// uncover the boundary pixels.
fn is_top_left(a: vec2<i32>, b: vec2<i32>) -> bool {
    let d = b - a;
    let is_top = d.y == 0 && d.x > 0;
    let is_left = d.y < 0;
    return is_top || is_left;
}

@compute @workgroup_size(8, 8)
fn rasterize(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Pixel coords = bbox_min + thread index. One dispatch covers
    // exactly the primitive's bounding rect; threads off the bottom-
    // right edge of the bbox return without writing.
    let px = prim.bbox_min.x + i32(gid.x);
    let py = prim.bbox_min.y + i32(gid.y);
    if px > prim.bbox_max.x || py > prim.bbox_max.y {
        return;
    }
    // Drawing-area clip: skip pixels outside the active rect. Both
    // bounds are inclusive on the CPU side, so we mirror that here.
    if px < draw_area.left || px > draw_area.right {
        return;
    }
    if py < draw_area.top || py > draw_area.bottom {
        return;
    }
    // VRAM bounds (defensive — bbox should already be inside).
    if px < 0 || px >= VRAM_WIDTH || py < 0 || py >= VRAM_HEIGHT {
        return;
    }

    let p = vec2<i32>(px, py);
    // Edge functions, oriented counter-clockwise. For a clockwise-
    // wound triangle these come out negative — we flip both winding
    // possibilities so the rasterizer accepts either.
    var w0 = edge(prim.v1, prim.v2, p);
    var w1 = edge(prim.v2, prim.v0, p);
    var w2 = edge(prim.v0, prim.v1, p);

    let cw = (w0 < 0) || (w1 < 0) || (w2 < 0);
    let ccw = (w0 > 0) || (w1 > 0) || (w2 > 0);
    if cw && ccw {
        // Mixed signs → outside the triangle on at least one edge.
        return;
    }

    // Top-left fill rule for boundary pixels (`w == 0`). If the edge
    // is top or left, accept; else reject.
    if w0 == 0 && !is_top_left(prim.v1, prim.v2) { return; }
    if w1 == 0 && !is_top_left(prim.v2, prim.v0) { return; }
    if w2 == 0 && !is_top_left(prim.v0, prim.v1) { return; }

    // Inside. Plot.
    let idx = u32(py * VRAM_WIDTH + px);
    vram[idx] = prim.color;
}
