# Editor lighting

How `NodeKind::Light` flows from authored placement → cooked
manifest → editor 3D viewport + editor-playtest runtime.

## Scope of this pass

- **Dynamic point lights** -- radius, intensity, RGB tint.
- **No shadows.**
- **No baked lightmaps.**
- **No model lighting yet** -- runtime lights affect room
  surfaces only. The editor preview applies per-face lighting
  to room surfaces; placed model instances render at full
  brightness regardless of nearby lights.
- **Per-light enclosing room** -- lights belong to the room
  whose subtree they're authored under. Lights outside any
  Room are dropped at cook time.

## Authoring

Place a `NodeKind::Light` *under* a Room subtree -- lights
authored outside any Room are dropped at cook time, and the
editor preview filters them per-room so they only affect
their enclosing room. Each Light: Inspector
exposes:

- **Color** -- 8-bit RGB.
- **Intensity** -- `f32` multiplier, `0..4` range with a quick
  warning above 4.0 (the bright-sun preset hits 2.0).
- **Radius** -- `f32` in *sector units*. `1.0` = one sector, so
  in a 1024-engine-unit world a radius of 4.0 reaches 4096
  engine units. Cooker rejects `radius <= 0`.
- **Presets** -- `Torch` / `Room fill` / `Bright sun` buttons
  set sensible starting values for each light type.

The inspector surfaces inline warnings when:
- Radius is `0` or negative (cook will fail).
- Intensity is non-finite or negative (cook will fail).
- Intensity exceeds `4.0` (most surfaces will saturate).

## Wire format

`psx_level::PointLightRecord` (in
`engine/crates/psx-level/src/lib.rs`):

```rust
pub struct PointLightRecord {
    pub room: u16,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub radius: u16,        // engine units
    pub intensity_q8: u16,  // Q8.8: 256 = 1.0
    pub color: [u8; 3],
    pub flags: u16,
}
```

### Units

- **Editor `radius`**: sector units. `1.0` = one sector. The
  inspector slider maxes at `8.0` sectors.
- **Runtime `radius` (`PointLightRecord.radius`)**: engine
  (world) units. The cooker multiplies `editor_radius *
  sector_size` to convert; the runtime never has to know
  about sector_size.
- **Editor `intensity`**: `f32` (`0..4` typical, with a soft
  warning above 4).
- **Runtime `intensity_q8`**: Q8.8 (`256` = 1.0).

The cook-time conversion:

```text
radius_record    = clamp(editor_radius * sector_size, 1, u16::MAX)
intensity_q8     = clamp(editor_intensity * 256.0, 0, u16::MAX)
position         = node.transform.translation
                   converted into room-local engine units
                   via the same convention model + spawn use.
```

### Light defaults

All three light-creation paths use the same defaults:

- Place tool (`PlaceKind::LightMarker`): `radius = 4.0`, `intensity = 1.0`, `color = (255, 240, 200)`.
- Add Child → Light (scene tree): same.
- Inspector "Room fill" preset: same.

Historically the Add Child path defaulted to `radius = 4096.0`
(four thousand sectors!), which lit every room from across
the world. That's been fixed.

## Lighting convention

Both editor and runtime use the same **PSX-neutral light scale**:

```text
light_rgb in 0..=255 per channel:
  0    pitch black
  128  neutral -- material renders at its base brightness
  255  saturating overbright

final_rgb = clamp(base_rgb * light_rgb / 128, 0, 255)
```

A face's `light_rgb` is `ambient + sum(per-light contributions)`,
clamped to 255 per channel. So:

- `ambient = [0, 0, 0]` plus no lights → black surface.
- `ambient = [32, 32, 32]` plus no lights → dim surface (~25% of base).
- `ambient = [128, 128, 128]` plus no lights → exactly the base material.
- A bright light at a face centre saturates that face to 255.

`WorldGrid::ambient_color` ships at `[32, 32, 32]` for the
starter project so rooms read as dim-but-not-black before any
Light node is placed.

## Editor preview lighting

The editor 3D viewport applies **per-face** point-light
accumulation:

1. `walk_room` calls `collect_preview_lights(project, room_id, grid)`
   once per frame. Lights are filtered to the *enclosing* Room
   -- a Light authored under Room A never lights Room B's
   surfaces.
2. Each light's `color × intensity_q8` is pre-multiplied into
   `u32` channels.
3. For each face (floor, ceiling, wall) the renderer computes
   a face centre and runs every light through
   `light_face(base_shade, center, lights, ambient)`.
4. Falloff is linear in distance: `weight = (radius - d) / radius`.
5. Per-light contribution lands as
   `(weighted_color * weight_q8) >> 16`, accumulated into
   `light_rgb`.
6. The result modulates the material's base tint via
   `base * light_rgb / 128`.

Selected lights draw a bright yellow radius ring at floor
level; unselected lights draw a thin ring tinted by their
authored colour.

Each Light also carries a small selectable AABB at its
position so the user can click it in the 3D viewport and drag
it to move. The bound is intentionally small -- a wide-radius
light's ring would otherwise intercept every click in the
room. See
[`docs/editor-architecture.md`](editor-architecture.md) §
"Entity selection + 3D move".

## Runtime playtest lighting

`editor-playtest` uses the same per-face lighting convention as
the editor preview:

1. `psx_engine::draw_room_lit` exposes each emitted floor,
   ceiling, and wall surface centre through `WorldSurfaceSample`.
2. `RuntimeRoomLighting` filters `LIGHTS` to the active room and
   converts records to `psx_engine::PointLightSample`.
3. `psx_engine::shade_tint_with_lights` accumulates ambient plus
   linear point-light falloff at the surface centre, then modulates
   the material's base `tint_rgb`.

The shared `psx_engine::lighting` helper owns the neutral-128
math, so the editor viewport and embedded runtime stay in lockstep.

## Validation

Hard-fails:
- Light's `radius <= 0`.
- Light's `intensity` is non-finite or negative.

Warnings (soft):
- Lights outside any Room are dropped (existing scene-tree
  warning handles this).
- Inspector flags `intensity > 4.0`.

## Currently out of scope

| Feature | Status |
| ------- | ------ |
| Per-face runtime lighting | Implemented for room floors, ceilings, and walls. |
| Model lighting at runtime | Out of scope. Models render at full brightness; only room surfaces respond. |
| Spotlights / directional lights / area lights | Out of scope. |
| Shadows / shadow mapping | Out of scope. |
| Baked lightmaps | Out of scope. |
| Lightprobes / GI | Out of scope. |
| Animated / flickering lights | Out of scope. Lights are static at the data layer. |
| Per-vertex lighting subdivision | Out of scope. Lighting evaluates per-face centre. |

## See also

- `editor/crates/psxed-project/src/playtest.rs` -- cook side.
- `engine/crates/psx-level/src/lib.rs` -- schema.
- `emu/crates/frontend/src/editor_preview.rs` --
  `walk_light_gizmos`, `collect_preview_lights`, `light_face`.
- `engine/examples/editor-playtest/src/main.rs` --
  `RuntimeRoomLighting`.
- `engine/crates/psx-engine/src/lighting.rs` -- shared neutral-128 lighting math.
- `docs/editor-model-authoring.md` -- model authoring (lights
  do not yet apply to models).
