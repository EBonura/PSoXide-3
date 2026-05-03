# `editor-playtest`

Vertical-slice target: render a level cooked from the editor on
real PSX hardware (or PCSX-Redux).

## What it does

The editor's Play action serializes the active scene
into this crate's `generated/` directory:

```
generated/
  level_manifest.rs              tracked placeholder manifest
  level_manifest.cooked.rs       ignored cooked manifest -- psx_level records + include_bytes!
  rooms/
    room_NNN.psxw                cooked room geometry (one per Room node)
  textures/
    texture_NNN.psxt             cooked texture blobs (one per unique texture)
```

`build.rs` selects `level_manifest.cooked.rs` when present, or
falls back to the placeholder manifest on a fresh clone.
`src/main.rs` `include!`s that selected manifest and consumes
the schema defined in
[`engine/crates/psx-level`](../../crates/psx-level/), which is
shared between the editor's playtest compiler and this runtime
example. See
[`docs/level-residency.md`](../../../docs/level-residency.md)
for the full contract.

The runtime walks the manifest's `ROOM_RESIDENCY` for the
current room, resolves the room's world asset through `ASSETS`,
parses it via [`psx_asset::World::from_bytes`], wraps it in a
[`psx_engine::RuntimeRoom`], and uploads the room's texture
assets through a tiny no-alloc [`psx_level::ResidencyManager`].
Materials are built from `MATERIALS` records -- the runtime no
longer hardcodes any starter texture binding; `local_slot →
texture_asset` is the source of truth.

The player starts at `generated::PLAYER_SPAWN`. Left stick
drives camera-relative movement, with D-pad as a fallback.
Circle held while moving runs. SELECT toggles a free-orbit
debug camera. The third-person camera and coarse room collision
consume the cooked grid so the play view matches the editor's
authored room.

## Embedded Play

The editor's **Play** button is one-click: it cooks the active
project, runs `make build-editor-playtest`, side-loads the built
EXE into the already-running frontend emulator, and paints the
live framebuffer into the editor's 3D viewport. No second window
or child frontend is launched. The toolbar swaps to `Stop` while
embedded play mode is active.

For headless / CI workflows, these Make targets remain:

```sh
# Cook the active editor project (or starter, with no args) into
# generated/. Destructive for ignored cooked outputs only.
make cook-playtest [PROJECT=path/to/project.ron]

# Build the EXE against generated/level_manifest.cooked.rs if it
# exists, otherwise the tracked placeholder.
# Doesn't recook.
make build-editor-playtest
```

## Placeholder vs cooked manifest

`generated/level_manifest.rs` is committed in a **placeholder**
state: imports the `psx_level` schema, declares empty asset /
room / material / model / light / character slices, and contains
no `include_bytes!` references. The EXE boots to a clear-coloured
screen until a cook lands real data in
`generated/level_manifest.cooked.rs`.

Cooked outputs live alongside the placeholder and are gitignored
so cooks don't churn diffs or dirty the tracked manifest:

```
generated/
  level_manifest.rs              tracked placeholder
  level_manifest.cooked.rs       gitignored cooked manifest
  rooms/
    room_NNN.psxw                gitignored
  textures/
    texture_NNN.psxt             gitignored
  models/
    model_NNN_*/                 gitignored
```

From a fresh clone:

1. `make build-editor-playtest` -- works from the placeholder.
2. Editor -> "Play" -- cooks your current scene into the ignored
   cooked manifest, builds, and runs inside the editor 3D viewport
   until you press Stop.

## Scope

This is a vertical slice. It uses `.psxw` v1, has *coarse*
sector-walkability collision, no portal traversal, no CD
streaming, no enemies, no AI, and no entity scripting.
Triggers / Portals / AudioSources surface as warnings during
cook and are skipped from the runtime manifest.

The runtime renders a player character at the spawn -- driven
by the Character resource the Player Spawn references. The
camera follows behind, and analog movement + Circle drive
idle / walk / run animations on the character's authored
clips. See [`docs/playable-character.md`](../../../docs/playable-character.md)
for the full Character → cook → runtime contract.

What it *does* render:

- **Rooms** via `psx_engine::draw_room`. Material slots come
  from generated `MATERIALS` records -- no hardcoded
  floor.psxt / brick-wall.psxt path remains.
- **Animated model instances** via the same
  `WorldRenderPass::submit_textured_model` path
  `showcase-model` uses. Each `MeshInstance` whose `mesh`
  references a `ResourceData::Model` produces a
  `LevelModelInstanceRecord`; the runtime parses the cooked
  `.psxmdl` + `.psxt` + `.psxanim`, uploads the atlas, and
  draws the textured animated model. See
  [`docs/editor-model-authoring.md`](../../../docs/editor-model-authoring.md).
- **Static lighting** -- every `PointLight` node cooks into a
  `PointLightRecord`; room surfaces also cook into
  `SurfaceLightRecord` per-vertex RGB, so embedded play can render
  textured Gouraud-lit rooms without re-accumulating every light per
  surface at runtime. Player, model instances, and equipment still
  sample point lights dynamically at their current origin. Linear
  falloff, no shadows. See
  [`docs/editor-lighting.md`](../../../docs/editor-lighting.md).
- **Entity markers** for legacy `MeshInstance` nodes that
  don't reference a Model (debug cubes, same as before).

## Files

- `src/main.rs` -- Scene impl, camera, input, draw call, residency wiring.
- `Cargo.toml` -- standalone PSX-target crate (its own `[workspace]`).
- `.cargo/config.toml` -- pins `target = "mipsel-sony-psx"`.
- `generated/level_manifest.rs` -- tracked placeholder; keep free of `include_bytes!`.
- `generated/level_manifest.cooked.rs` -- generated; do not edit; gitignored.
- `generated/rooms/*.psxw` -- generated; do not edit; gitignored.
- `generated/textures/*.psxt` -- generated; do not edit; gitignored.
- `generated/models/model_NNN_*/` -- per-model folders carrying
  cooked `.psxmdl` + `.psxt` + `.psxanim`; do not edit; gitignored.

## See also

- [`docs/level-residency.md`](../../../docs/level-residency.md) --
  the full contract: schema, writer, reader, future backing stores.
- [`engine/crates/psx-level`](../../crates/psx-level/) -- shared
  no_std schema crate.
- `editor/crates/psxed-project/src/playtest.rs` -- manifest types,
  validation, writer.
- `editor/crates/psxed-project/src/bin/cook_playtest.rs` -- CLI.
- `engine/crates/psx-engine/src/world_render.rs` -- `draw_room`.
- `engine/examples/showcase-room/` -- sister example that cooks
  the starter at build time and renders it directly (no editor
  generated/ step).
