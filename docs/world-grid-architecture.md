# World Grid Architecture

The world grid has four layers. Keeping them separate is the main guardrail
that lets the editor stay pleasant while the PS1 runtime stays small.

## Layers

```text
Editor authoring model
  psxed-project::WorldGrid
  serde, Vec, ResourceId, mutable node payloads, origin offset

Cooked host manifest
  psxed-project::CookedWorldGrid
  validated dimensions, resolved material slots, stripped empty sectors

Cooked binary asset
  .psxw
  AssetHeader + WorldHeader + SectorRecord[] + WallRecord[]

Runtime engine model
  psx_engine::RuntimeRoom<'a>            (PSX-resident wrapper over psx_asset::World)
    .render() -> RoomRender<'a, '_>      (renderer-only API surface)
    .collision() -> RoomCollision<'a, '_> (collision-only API surface)

  psx_engine::GridRoom<'a>               (authoring / test helper, NOT PSX-resident)
```

The engine must not depend on editor nodes, RON, paths, or resource names. The
editor must not be forced to store borrowed runtime slices. The cooker is the
contract boundary.

`GridRoom` / `GridWorld` (the older "engine model" type) are now flagged
**authoring / test helpers** in their doc comments. New runtime systems —
collision, rendering, AI floor sampling — should grow on top of `RuntimeRoom`,
which holds zero owned state and decodes sectors / walls by value on demand.

## Authoring vs Runtime

The editor's job is to make geometry pleasant to author. Authoring needs a
mutable, type-rich model (`Vec<Wall>`, optional materials, scene-level node
hierarchy, …). The runtime's job is to draw + simulate that geometry on a
33 MHz CPU with 2 MB of RAM. Runtime needs the inverse: zero-copy, byte-table
addressing, no allocator, and a record layout that the GTE / GP0 can fan out
without per-vertex chasing.

Two concrete consequences:

* **No `Vec`, no `String`, no `HashMap` reachable from `RuntimeRoom`**. The
  type is `Copy`. Adding any owned field would break the
  `_assert_copy::<RuntimeRoom>` const-assertion in `psx-engine::world`.

* **Authoring model is allowed to be wider than the runtime model.** The
  cooker is where authoring richness collapses into the runtime's compact
  shape. If the editor lets you assign a material that doesn't exist, that's
  caught at cook time, not at parse time.

## Render vs Collision Split

Same room data, two consumers, two read paths:

* `RoomRender` — heights and splits for vertex emission, materials for tpage /
  CLUT lookup, world-level lighting state.
* `RoomCollision` — heights and splits for floor / ceiling sampling, walkable /
  solid bits for stop-or-pass decisions.

Materials are render-only. Walkable / solid bits are collision-only. The two
views are zero-cost `Copy` newtypes over `RuntimeRoom`; they exist to enforce
discipline at the API level (a draw pass cannot accidentally branch on
walkability; a collision query cannot reach a tpage word).

The cooker today writes one record per sector / wall, with both render and
collision concerns interleaved. When the compact byte format lands, those two
concerns split into separate tables — the cooker pipeline is already organized
by render-emit / collision-emit phases so that future split is a small change.

## Tomb Raider parallels

Bonnie-32's room model deliberately echoes the original Tomb Raider runtime,
because the constraints are similar: 1 MB working set, sector-based level
geometry, integer math throughout. Two lessons we keep:

* **Sectors are a strong addressing mode**, not just a layout convenience. Once
  you commit to "1024 world units per sector, world-grid coords are sector
  coords times sector size," collision lookup becomes `O(1)` and floor / wall
  data becomes table-indexed instead of pointer-chased. Keep the world coords
  → sector coords division single-instruction; resist temptation to break the
  grid.

* **Heights stay room-local once compact format lands.** Today v1 ships
  absolute `i32` heights; the future compact format (see
  `docs/world-format-roadmap.md`) follows TR's `i16`-room-relative trick so a
  level can address ±32 KU vertically with 2 bytes per height set. The
  editor's `editor_preview::VIEW_ANCHOR` already does the same on the
  authoring side, keeping vertex data within i16 range from the camera target.

Two lessons we deliberately ignore for now:

* **No `tr_face4` / `tr_face3` lists** today. TR's render path iterates an
  explicit face list per room because sector geometry alone doesn't capture
  decoration (statues, columns, water surface). PSoXide currently emits all
  faces from sector data; once decorative geometry lands, a face table is the
  obvious shape — but the `RuntimeRoom` API will absorb it as
  `RoomRender::faces()` rather than another concern on `SectorRender`.

* **No item / entity table inside `.psxw`**. TR mixes statics into the room
  blob; we keep entities in scene nodes outside the cooked world. Cleaner
  authoring, slightly more bookkeeping at level-load.

The full TR-to-PSoXide name mapping (per future-format records) lives in
`docs/world-format-roadmap.md`.

The editor/runtime coordinate conversion is documented separately in
[`docs/editor-runtime-coordinates.md`](editor-runtime-coordinates.md). In
short: the editor preview treats North as `+Z`, while the runtime `.psxw`
format treats North as `-Z`; `psxed_project::world_cook` is the single
boundary that flips directions, corners, and split ids.

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
- **Diagonal walls (`NorthWestSouthEast` / `NorthEastSouthWest`) are rejected
  at cook time**. The data model has the slots so authoring + serialization
  works, but render / pick / collision aren't consistent for diagonals yet.
- **Wall ownership rule**: the physical wall between `(x, z)` and `(x+1, z)`
  is `East(x, z)` *and* `West(x+1, z)`. The editor's PaintWall stamps one
  side; if both opposing sides are authored, the cooker rejects the duplicate
  physical wall so render and collision never double-count the same face. Same
  for North / South.

## Room Budget

Authoring is bounded by the runtime's tolerance:

| Cap                | Value     |
| ------------------ | --------- |
| `MAX_ROOM_WIDTH`   | 32 sectors|
| `MAX_ROOM_DEPTH`   | 32 sectors|
| `MAX_WALL_STACK`   | 4 walls per edge |
| `MAX_ROOM_TRIANGLES` | 2048    |
| `MAX_ROOM_BYTES`   | 64 KiB    |

`WorldGrid::budget()` rolls up populated cells, faces, walls, triangles, plus
exact v1 wire size and a future-compact-format size estimate (see the
roadmap). The editor inspector shows the rollup and tints any over-cap metric
red. Cooker enforces the same caps so what the editor warns about also fails
at cook time.

## Binary Shape

`.psxw` starts with the standard `AssetHeader` using magic `PSXW`. The v1
payload is:

```text
WorldHeader        (20 B)
SectorRecord[]     (44 B each, width * depth records)
WallRecord[]       (24 B each, wall_count records)
```

Heights are `[i32; 4]` per face. Materials resolve via an external bank slot.
No sector logic. No portals.

A future compact format is sketched in `docs/world-format-roadmap.md`; it is
**not** in `psxed-format` as Rust types. The format crate carries only the
active wire records.

## i16 Vertex Safety

`SECTOR_SIZE * MAX_ROOM_WIDTH = 32 * 1024 = 32 768` — that's the i16 cliff.
A 32×32 room sits exactly on it; bigger rooms silently truncate.

Editor preview: vertices are emitted relative to a per-frame `VIEW_ANCHOR`
(the camera target). The GTE translation absorbs the offset via `cam_local`.
Debug builds assert the relative coord fits i16; release saturates loudly via
the same clamp. With sector_size 1024 this gives ±32 sectors of headroom from
the camera target — comfortably the editor cap.

Runtime: same trick, room-local i16 heights, room origin added during vertex
emission. Keeps i16 vertex space valid for any room up to the budget cap.
