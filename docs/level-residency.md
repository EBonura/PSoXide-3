# Level residency

How rooms, materials, and texture assets reach RAM and VRAM at
runtime -- the contract between the editor's playtest compiler
and the runtime example.

## Current pass: granular residency, embedded backing store

```text
on disk                 in-memory
  (later)              (now)
  +-------+           +--------+
  | pack0 |   ==>     | RAM    |   master asset table
  | pack1 |           | VRAM   |   per-room residency
  +-------+           +--------+
```

The runtime resolves room bytes through a **master asset
table** (`ASSETS`) keyed by `AssetId`. Per-room residency
records (`ROOM_RESIDENCY`) declare which assets must be in RAM
or VRAM before the room renders. The active backing store is
`include_bytes!`: every asset's payload sits next to the EXE's
text section, RAM-resident from the moment the program loads.

That's not the interesting part. The interesting part is that
the **runtime doesn't know it's embedded**. It walks the same
contract a future stream-pack loader will walk:

```text
current room  ── room.world_asset ──>  ASSETS[..]  ── asset.bytes
              ── ROOM_RESIDENCY[..]  ──>  required_ram / required_vram
              ── room.material_first / count ──>  MATERIALS[..]
                                          ── material.texture_asset ──> ASSETS[..]
```

When the backing store changes, the contract doesn't.

## Schema (psx-level crate)

`engine/crates/psx-level` is a tiny `no_std` crate that defines
the records the editor compiler writes and the runtime example
reads. Every record is `&'static`-borrowing -- nothing
allocates, everything pins to literal data.

| Record                  | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| `LevelAssetRecord`      | One asset (room world or texture). Carries the byte slice + RAM/VRAM cost. |
| `LevelRoomRecord`       | One room. Points at its world `AssetId` and a material slice. |
| `LevelMaterialRecord`   | Per-room material slot binding (cooked local slot → texture `AssetId`). |
| `RoomResidencyRecord`   | Per-room contract: required RAM + VRAM `AssetId` lists. Warm lists are reserved for future preload hints. |
| `PlayerSpawnRecord`     | Spawn pose in room-local engine units.         |
| `EntityRecord`          | Generic entity (Marker / StaticMesh).          |

`AssetKind` enumerates the asset classes the writer + reader
both understand right now. Today: `RoomWorld`, `Texture`. New
kinds (actor models, lights, audio banks) land here when the
writer + reader for them both ship in the same pass.

## Writer (psxed-project::playtest)

The editor's playtest compiler walks the project's scene tree
once and emits:

1. **Cooked rooms** -- `cook_world_grid` produces a
   `CookedWorldGrid` per Room node. Bytes get written to
   `generated/rooms/room_NNN.psxw`; a `LevelAssetRecord` of
   kind `RoomWorld` is added to `ASSETS`.
2. **Texture assets** -- for each cooked material slot, the
   writer resolves the material's texture, dedupes by
   `ResourceId`, copies the `.psxt` bytes into
   `generated/textures/texture_NNN.psxt`, and adds a
   `LevelAssetRecord` of kind `Texture`.
3. **Material records** -- `LevelMaterialRecord`s pinned to
   `(room, local_slot)`, ordered by slot. Each room's
   `material_first..material_count` slice covers exactly its
   cooked material count.
4. **Residency records** -- `ROOM_RESIDENCY` lists required RAM
   (the room's world asset) and required VRAM (the texture
   assets the room's materials reference, deduplicated).

`AssetId` order is deterministic across runs: rooms first (in
scene-tree order), then textures (in first-use order from the
material walk).

## Reader (engine/examples/editor-playtest)

The runtime example wires the contract end-to-end:

```rust
// 1. Pick a room.
let room = &ROOMS[0];

// 2. Run the residency contract. The manager tracks RAM/VRAM
//    membership; this call says which assets are missing
//    (need uploading) vs. already-resident.
let residency = ROOM_RESIDENCY.iter().find(|r| r.room == 0).unwrap();
let cs = RESIDENCY.ensure_room_resident(residency);

// 3. Resolve the world asset and parse it.
let asset = find_asset_of_kind(ASSETS, room.world_asset, AssetKind::RoomWorld)?;
let world = AssetWorld::from_bytes(asset.bytes)?;
let runtime_room = RuntimeRoom::from_world(world);

// 4. Build a material table from the room's material slice.
//    Each material's texture_asset is uploaded once (skipped
//    on subsequent rooms) and its CLUT/tpage word is cached.
let materials = build_room_materials(room);

// 5. Render.
draw_room(runtime_room.render(), &materials, &camera, options, ...);
```

`ResidencyManager<RAM_CAP, VRAM_CAP>` is a no-alloc fixed-size
tracker. RAM membership is logical (every asset is
`include_bytes!`-resident from program start), but VRAM
membership is meaningful -- each texture uploads once, and
subsequent room loads see them already-resident.

## Future backing stores

The schema doesn't change. What changes is who owns
`asset.bytes`:

```text
Today:    LevelAssetRecord { bytes: include_bytes!(...) }    pins to EXE text
Soon:     AssetStorage::StreamPack { pack, offset, size }    paged by CD reader
Later:    AssetStorage::Composite                            mixed embedded + paged
```

Stream packs are a **disk packaging optimisation**, not a
memory-residency unit. Multiple stream packs may share assets,
or one stream pack may carry assets that get evicted
independently. The residency manager already operates on
asset-granular RAM/VRAM membership; adding a CD-pager underneath
won't change `ROOM_RESIDENCY`'s semantics.

## Out of scope (deliberate)

- **CD streaming** -- no I/O, no async, no scheduler.
- **Runtime baking** -- farfield impostors are an offline-baked
  asset class. Render-to-texture from the runtime is not
  planned.
- **Portals / PVS** -- adjacency / visibility computation is the
  job of a future cooker pass; the residency lists this pass
  emits don't include warm preload hints yet.
- **Actors / lights / audio / scripts** -- separate asset
  kinds, separate writers, separate readers.

The schema enforces that policy: a record type only exists
when both the writer and reader for it ship in the same pass.
