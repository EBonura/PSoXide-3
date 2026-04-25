// Quick fill (GP0 0x02). Writes a constant 15bpp colour into a
// rectangle. No clip, no RMW, no semi-trans, no mask-bit — fill
// bypasses all of those by hardware design. The host has already
// applied the 16-pixel alignment to x / w.

struct Fill {
    xy: vec2<u32>,
    wh: vec2<u32>,
    color: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> vram: array<u32>;
@group(0) @binding(1) var<uniform> prim: Fill;

const VRAM_WIDTH: u32 = 1024u;
const VRAM_HEIGHT: u32 = 512u;

@compute @workgroup_size(8, 8)
fn rasterize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= prim.wh.x || gid.y >= prim.wh.y { return; }
    let px = prim.xy.x + gid.x;
    let py = prim.xy.y + gid.y;
    if px >= VRAM_WIDTH || py >= VRAM_HEIGHT { return; }
    vram[py * VRAM_WIDTH + px] = prim.color;
}
