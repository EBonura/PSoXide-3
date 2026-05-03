# Editor lighting

How `NodeKind::PointLight` flows from authored placement to the
editor viewport, generated playtest manifest, and embedded runtime.

## Scope

- Static point lights with radius, intensity, and RGB tint.
- Room geometry uses static per-vertex lighting in embedded play.
- Player, model instances, and equipment sample point lights at
  their current origin.
- No shadows, lightmaps, directional lights, spotlights, or animated
  flicker yet.

Point lights belong to the Room subtree they are authored under.
Lights outside any Room are dropped at cook time; the editor preview
filters them per Room.

## Authoring

Place a `NodeKind::PointLight` under a Room subtree. The inspector
exposes:

- **Color**: 8-bit RGB.
- **Intensity**: `f32` multiplier. The common range is `0..4`;
  values above `4.0` usually saturate surfaces.
- **Radius**: `f32` in sector units. `1.0` means one sector, so a
  radius of `4.0` in a 1024-unit world reaches 4096 engine units.
- **Presets**: `Torch`, `Room fill`, and `Bright sun`.

The inspector warns when radius is `0` or negative, intensity is
non-finite or negative, or intensity is high enough to saturate most
surfaces. The cooker hard-fails invalid radius/intensity values.

## Runtime Records

Every authored point light cooks into `psx_level::PointLightRecord`:

```rust
pub struct PointLightRecord {
    pub room: RoomIndex,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub radius: u16,        // engine units
    pub intensity_q8: u16,  // Q8.8: 256 = 1.0
    pub color: [u8; 3],
    pub flags: u16,
}
```

Room geometry also cooks into static per-surface vertex lighting inside
the `.psxw` room blob. Version 3 keeps the compact v2 sector/wall
records and appends a direct-indexed `SurfaceLightRecord` table:

```rust
pub struct SurfaceLightRecord {
    pub vertex_rgb: [[u8; 3]; 4],
}
```

The table is indexed without a search:

- floor: `(sx * depth + sz) * 2`
- ceiling: `(sx * depth + sz) * 2 + 1`
- wall: `sector_count * 2 + global_wall_index`

`vertex_rgb` follows the same quad vertex order emitted by
`psx-engine`.

## Units

- Editor radius: sector units.
- Runtime radius: engine units, computed as `editor_radius * sector_size`.
- Editor intensity: `f32`.
- Runtime intensity: Q8.8 fixed point.
- Position: node transform converted to room-local engine units using
  the same convention as model instances and player spawn.

## Lighting Convention

Editor and runtime share the neutral-128 PSX tint convention:

```text
light_rgb in 0..=255 per channel:
  0    pitch black
  128  neutral; material renders at base brightness
  255  saturating overbright

final_rgb = clamp(base_rgb * light_rgb / 128, 0, 255)
```

The accumulator is ambient plus point-light contributions, clamped to
255 per channel. Falloff is linear in distance:

```text
weight = (radius - distance) / radius
```

The starter project uses dim room ambient so rooms are readable before
any authored light is added.

## Editor Preview

The editor 3D viewport keeps the authoring feedback live:

1. `collect_preview_lights(project, room_id, grid)` gathers
   `PointLight` nodes in the active Room.
2. Each light pre-multiplies `color * intensity_q8`.
3. Room faces are shaded from static point lights in the viewport.
4. Selected lights draw a bright radius ring and bulb icon.
5. Each light has a small clickable marker AABB so the radius ring does
   not steal every click in the room.

The preview path is intentionally host-side; it gives immediate spatial
feedback without requiring a runtime cook.

## Embedded Runtime

`editor-playtest` uses the generated manifest in two ways:

1. Room floors, ceilings, and walls render through textured Gouraud
   packets using per-vertex RGB embedded in the `.psxw` static-light
   table. Runtime fog is still applied per vertex.
2. Player, model instances, and attached equipment sample `LIGHTS` at
   their current origin and modulate their material tint dynamically.

This split keeps static room lighting cheap while preserving dynamic
lighting response for moving model-backed objects.

## Validation

Hard failures:

- Point light radius is `0` or negative.
- Point light intensity is non-finite or negative.

Soft warnings:

- Point lights outside any Room are dropped.
- High intensity values are likely to saturate.

## See Also

- `editor/crates/psxed-project/src/playtest.rs`: cook side.
- `engine/crates/psx-level/src/lib.rs`: manifest schema.
- `emu/crates/frontend/src/editor_preview.rs`: viewport lighting and gizmos.
- `engine/examples/editor-playtest/src/main.rs`: runtime lighting.
- `engine/crates/psx-engine/src/lighting.rs`: shared neutral-128 lighting math.
