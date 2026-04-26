# World Grid Architecture

The world grid has three layers. Keeping them separate is the main guardrail
that lets the editor stay pleasant while the PS1 runtime stays small.

## Layers

```text
Editor authoring model
  psxed-project::WorldGrid
  serde, Vec, ResourceId, mutable node payloads

Cooked host manifest
  psxed-project::CookedWorldGrid
  validated dimensions, resolved material slots, stripped empty sectors

Cooked binary asset
  .psxw
  AssetHeader + WorldHeader + SectorRecord[] + WallRecord[]

Runtime engine model
  psx_engine::GridWorld<'a>
  borrowed slices, no allocator, u16 material slots, ROM-backed data
```

The engine must not depend on editor nodes, RON, paths, or resource names. The
editor must not be forced to store borrowed runtime slices. The cooker is the
contract boundary.

## Contract

- Sector size is 1024 engine units. Non-1024 authored grids are rejected until
  the runtime supports per-room sector sizes.
- Horizontal face corners are stored as `[NW, NE, SE, SW]`.
- Vertical wall corners are stored as
  `[bottom-left, bottom-right, top-right, top-left]`.
- Sector storage is flat `[x * depth + z]`.
- Empty authored sectors are stripped to `None` in the cooked manifest.
- Floors and ceilings use `walkable` for collision policy.
- Walls use `solid` for collision policy.
- Every rendered floor, ceiling, and wall must resolve to a material slot.
- Editor `ResourceId` material references are mapped to runtime `u16` slots in
  first-use order while scanning sectors.
- Split and direction numeric ids live in `psxed-format::world`; the final
  `.psxw` writer uses those ids.

## Binary Shape

`.psxw` starts with the standard `AssetHeader` using magic `PSXW`. The payload
is:

```text
WorldHeader
SectorRecord[width * depth]
WallRecord[wall_count]
```

Sector records stay in flat `[x * depth + z]` order. Each sector points at a
contiguous run in the wall table through `first_wall` and `wall_count`.

## Naming

`WorldGrid` is the editor authoring payload. `GridWorld` is the runtime engine
view. That distinction is intentional: authoring owns mutation and resources;
runtime owns static query and render data.

## Next Step

The next layer should adapt parsed `.psxw` data into the engine's higher-level
world query/render APIs, starting with floor and wall collision helpers.
