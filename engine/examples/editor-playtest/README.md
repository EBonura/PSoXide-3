# `editor-playtest`

Vertical-slice target: render a level cooked from the editor on
real PSX hardware (or PCSX-Redux).

## What it does

The editor's "Cook & Play" action serializes the active scene
into this crate's `generated/` directory:

```
generated/
  level_manifest.rs              Rust source — psx_level records + include_bytes!
  rooms/
    room_NNN.psxw                cooked room geometry (one per Room node)
  textures/
    texture_NNN.psxt             cooked texture blobs (one per unique texture)
```

`src/main.rs` `include!`s the manifest and consumes the schema
defined in [`engine/crates/psx-level`](../../crates/psx-level/),
which is shared between the editor's playtest compiler and
this runtime example. See
[`docs/level-residency.md`](../../../docs/level-residency.md)
for the full contract.

The runtime walks the manifest's `ROOM_RESIDENCY` for the
current room, resolves the room's world asset through `ASSETS`,
parses it via [`psx_asset::World::from_bytes`], wraps it in a
[`psx_engine::RuntimeRoom`], and uploads the room's texture
assets through a tiny no-alloc [`psx_level::ResidencyManager`].
Materials are built from `MATERIALS` records — the runtime no
longer hardcodes any starter texture binding; `local_slot →
texture_asset` is the source of truth.

The player starts at `generated::PLAYER_SPAWN`. D-pad
LEFT/RIGHT yaws the player; UP/DOWN walks forward/back along
the player's facing. SELECT toggles a free-orbit debug camera
that pivots around the spawn point. No collision yet.

## Cook + run

Three Make targets, sharply scoped so the editor's cook output
isn't accidentally clobbered:

```sh
# Cook the active editor project (or starter, with no args) into
# generated/. Destructive — overwrites whatever was there.
make cook-playtest [PROJECT=path/to/project.ron]

# Build the EXE against whatever's in generated/ right now.
# Doesn't recook.
make build-editor-playtest

# Build + side-load the existing cooked output.
# Safe to run after the editor's "Cook & Play" — won't touch
# your scene.
make run-editor-playtest

# Convenience: cook starter + build + run.
make run-starter-playtest
```

The editor's "Cook & Play" button performs the cook half and
prints the exact `make run-editor-playtest` command in its
status line.

## Placeholder vs cooked manifest

`generated/level_manifest.rs` is committed in a **placeholder**
state: imports the `psx_level` schema, declares empty
`ASSETS` / `MATERIALS` / `ROOMS` / `ROOM_RESIDENCY`. The EXE
boots to a clear-coloured screen until a cook lands real data.

Cooked outputs live alongside the manifest and are gitignored
so cooks don't churn diffs:

```
generated/
  level_manifest.rs              tracked, placeholder state
  rooms/                         tracked dir
    room_NNN.psxw                gitignored
  textures/                      tracked dir
    texture_NNN.psxt             gitignored
```

From a fresh clone:

1. `make build-editor-playtest` — works, shows clear screen.
2. `make run-starter-playtest` — cooks the starter, runs.
3. Editor → "Cook & Play" → `make run-editor-playtest` — your
   custom scene, kept across runs until you cook again.

## Scope

This is a vertical slice. It uses `.psxw` v1, has no collision,
no portal traversal, no CD streaming, no enemies, no AI, no
lights, and no entity scripting. Lights / Triggers / Portals /
AudioSources surface as warnings during cook and are skipped
from the runtime manifest.

What it *does* render:

- **Rooms** via `psx_engine::draw_room`. Material slots come
  from generated `MATERIALS` records — no hardcoded
  floor.psxt / brick-wall.psxt path remains.
- **Animated model instances** via the same
  `WorldRenderPass::submit_textured_model` path
  `showcase-model` uses. Each `MeshInstance` whose `mesh`
  references a `ResourceData::Model` produces a
  `LevelModelInstanceRecord`; the runtime parses the cooked
  `.psxmdl` + `.psxt` + `.psxanim`, uploads the atlas, and
  draws the textured animated model. See
  [`docs/editor-model-authoring.md`](../../../docs/editor-model-authoring.md).
- **Room-level lighting** — every `Light` node cooks into a
  `PointLightRecord`; the runtime accumulates contributions
  at the room centre and modulates each room material's tint.
  Linear falloff, no shadows, models are not lit yet. The
  editor preview applies per-face lighting for spatial
  authoring feedback. See
  [`docs/editor-lighting.md`](../../../docs/editor-lighting.md).
- **Entity markers** for legacy `MeshInstance` nodes that
  don't reference a Model (debug cubes, same as before).

## Files

- `src/main.rs` — Scene impl, camera, input, draw call, residency wiring.
- `Cargo.toml` — standalone PSX-target crate (its own `[workspace]`).
- `.cargo/config.toml` — pins `target = "mipsel-sony-psx"`.
- `generated/level_manifest.rs` — generated; do not edit.
- `generated/rooms/*.psxw` — generated; do not edit; gitignored.
- `generated/textures/*.psxt` — generated; do not edit; gitignored.
- `generated/models/model_NNN_*/` — per-model folders carrying
  cooked `.psxmdl` + `.psxt` + `.psxanim`; do not edit; gitignored.

## See also

- [`docs/level-residency.md`](../../../docs/level-residency.md) —
  the full contract: schema, writer, reader, future backing stores.
- [`engine/crates/psx-level`](../../crates/psx-level/) — shared
  no_std schema crate.
- `editor/crates/psxed-project/src/playtest.rs` — manifest types,
  validation, writer.
- `editor/crates/psxed-project/src/bin/cook_playtest.rs` — CLI.
- `engine/crates/psx-engine/src/world_render.rs` — `draw_room`.
- `engine/examples/showcase-room/` — sister example that cooks
  the starter at build time and renders it directly (no editor
  generated/ step).
