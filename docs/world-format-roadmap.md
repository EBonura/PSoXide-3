# `.psxw` Format Roadmap

The active world wire format is **VERSION = 3**. It's the shape
`psxed-project` cooks and the main shape `psx_asset::World::from_bytes`
accepts; the parser also keeps v1/v2 compatibility for older blobs. The
compact format sketched below is **not in `psxed-format`** as Rust types
-- it lives here, in docs, until both producer and consumer support it
together.

## The rule

A world-format struct is allowed in `psxed-format` only when:

1. The cooker can emit it.
2. The runtime parser can parse it.
3. Malformed blobs are tested.
4. Version negotiation between producer and consumer is tested.

`psxed-format` is the byte-exact contract between editor and
runtime. Speculative records that meet none of those points
clutter that contract and trick callers into thinking a future
format already works. They go in this doc instead.

## Active format -- VERSION 3

```
AssetHeader               12 B
WorldHeader               20 B
SectorRecord[]            60 B each, width * depth records (dense)
WallRecord[]              32 B each, wall_count records
SurfaceLightRecord[]      12 B each, optional static-light table
```

Properties:

- Floor / ceiling / wall heights are `[i32; 4]` per face -- 16 B
  per height set.
- Floor / ceiling / wall UVs are four packed PS1 `[u, v]` byte pairs
  per face -- 8 B per face.
- If static vertex lighting is enabled, an appended direct-indexed
  light table stores two records per sector (floor, ceiling) plus one
  record per wall. The table is absent for unlit/non-baked rooms.
- A sector record exists for **every** cell, populated or not.
  The runtime parser rejects a blob when
  `sector_count != width * depth`.
- No embedded material table. Slot ids resolve via an external
  bank that the caller (game / playtest manifest) supplies.
- No sector logic stream, no portal records.
- Diagonal walls and cells outside the 32×32 / 2048-tri / 64
  KiB caps are rejected at cook time.

The `WorldGridBudget::psxw_bytes` figure is exact for the base
geometry payload. Static-light bake size is additive and should be
surfaced separately in editor budget UI.

## Future compact format

The goals -- driven by PSX RAM budget, not aesthetics:

- Room-local `[i16; 4]` heights -- 8 B per set instead of 16.
- 28 B sector record (down from 60).
- 12 B wall record (down from 32).
- Embedded material table or explicit material-bank reference,
  so a `.psxw` is self-resolving.
- FloorData-style sparse sector logic stream for triggers /
  slopes / portals (Tomb Raider's idea).
- Explicit portal records -- one for visibility (between rooms)
  and one for traversal (per sector: wall / pit / sky).

The reduction targets give roughly:

| Record       | active | future | savings |
| ------------ | ------ | ------ | ------- |
| WorldHeader  |  20 B  |  20 B  |   --     |
| Sector       |  60 B  |  28 B  | 53 %    |
| Wall         |  32 B  |  12 B  | 62 %    |
| SurfaceLight |  12 B  | TBD    | TBD     |

`WorldGridBudget::future_compact_estimated_bytes` shows the
compact estimate **as a planning aid**. No live `.psxw` is ever
this size today; nothing reads or writes the compact layout.

### Sketched record shapes

These are **pseudocode**, not Rust. They describe one possible
landing for the compact format and are intentionally not
public types:

```text
WorldHeaderCompact, target 20 bytes:
  width: u16
  depth: u16
  sector_size: i32
  material_count: u16          // embedded table count
  sector_count: u16
  wall_count: u16
  logic_stream_bytes: u16      // for the FloorData stream
  ambient_color: [u8; 3]
  flags: u8

MaterialRecordCompact, target 12 bytes:
  tpage_word: u16
  clut_word:  u16
  tint:       [u8; 3]
  blend_mode: u8
  flags:      u8                // RAW_TEXTURE, DOUBLE_SIDED, …
  _pad:       [u8; 3]           // align to 12

SectorRecordCompact, target 28 bytes:
  flags:            u16
  floor_material:   u16
  ceiling_material: u16
  first_wall:       u16
  logic_offset:     u16         // NO_LOGIC sentinel = 0xFFFF
  wall_count:       u8
  split_bits:       u8          // bit 0 floor NE-SW, bit 1 ceiling
  floor_heights:    [i16; 4]
  ceiling_heights:  [i16; 4]

WallRecordCompact, target 12 bytes:
  direction: u8
  flags:     u8
  material:  u16
  heights:   [i16; 4]

RoomPortalRecord (visibility):
  target_room: u16              // NO_ROOM sentinel = 0xFFFF
  _pad:        u16
  normal:      [i16; 3]
  vertices:    [[i16; 3]; 4]

SectorPortalRecord (traversal, target 8 bytes):
  wall_room: u16                // horizontal escape
  pit_room:  u16                // down through floor
  sky_room:  u16                // up through ceiling
  _pad:      u16
```

## Tomb Raider parallels

Bonnie-32 / PSoXide deliberately echoes TR's runtime model
because the constraints are similar. Mapping:

| TR concept                                  | PSoXide                                      |
| ------------------------------------------- | -------------------------------------------- |
| `tr_room`                                   | one `.psxw` blob, `RuntimeRoom<'a>`          |
| `tr_room_data` (vertex / face arrays)       | not yet emitted -- render path WIP            |
| `tr_room_sector` (4 B floor/ceiling heights)| compact sector record (target 28 B)          |
| FloorData stream (triggers / slopes)        | sparse sector logic stream (compact only)    |
| `tr_room_portal` (visibility)               | room portal record (compact only)            |
| `room_below` / `room_above` / wall portal   | sector traversal record (compact only)       |
| Object texture table                        | embedded material table (compact only)       |

## Migration plan (when it lands)

1. Author the cooker side of the compact emit. Bump VERSION to 2
   on the wire only when the cooker can actually produce it.
2. Author the runtime parser. Add tests that round-trip cooked
   blobs end-to-end.
3. Add public Rust types for the new records to `psxed-format`
   only after the round-trip test passes.
4. Keep the v1 parser path until at least one cooker still emits
   v1 -- version negotiation should be explicit, not implicit.
5. Retire v1 only after the editor stops emitting it.

Until step 3 lands, this document is the contract.
