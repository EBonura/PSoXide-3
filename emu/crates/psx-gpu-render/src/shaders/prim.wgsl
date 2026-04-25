// PSX hardware-renderer shader.
//
// Single shader serving every primitive type by branching on
// `flags`. Phase 1 only handles flat-color mono triangles — the
// `flags` plumbing is in place for Phase 2+ (textured / shaded /
// tex-shaded) so the vertex format and bind groups don't change
// when the next variants land.
//
// PSX coordinate convention:
//   - Vertex positions are post-`draw_offset` PSX pixel coords
//     (signed, fits in i16 because the 11-bit signed PSX coord +
//     draw_offset stays well inside ±2048).
//   - We map (display_origin .. display_origin + display_size) to
//     NDC (-1..+1, Y-flipped). The viewport is the high-res render
//     target — fractional scale comes "for free" because the
//     rasterizer rasterizes at viewport resolution while vertex
//     coords stay in PSX-space integers.

struct Globals {
    // Active display rect in VRAM pixel coords (Gpu::display_area()).
    display_origin: vec2<f32>,
    display_size:   vec2<f32>,
    // Output texture size in pixels (matches the render-pass
    // viewport). Carried for diagnostic / future use.
    target_size:    vec2<f32>,
    _pad:           vec2<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;

struct VertexIn {
    @location(0) pos:   vec2<i32>,
    @location(1) color: vec4<f32>,
    @location(2) uv:    vec2<u32>,
    @location(3) flags: u32,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0)       color:    vec4<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let pos_psx = vec2<f32>(f32(in.pos.x), f32(in.pos.y));
    // Map PSX-space → NDC. Y is flipped: PSX 0 is at the top, NDC +Y
    // is up; flipping the y-axis after the *2.0 - 1.0 puts the PSX
    // top-left at NDC top-left.
    let ndc_xy = ((pos_psx - g.display_origin) / g.display_size) * 2.0 - 1.0;
    var out: VertexOut;
    out.position = vec4<f32>(ndc_xy.x, -ndc_xy.y, 0.0, 1.0);
    out.color    = in.color;
    // Phase 1+ stub: silence unused-binding warnings until later
    // phases consume them. WGSL phony-assignment syntax (no `let`).
    _ = in.uv;
    _ = in.flags;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
