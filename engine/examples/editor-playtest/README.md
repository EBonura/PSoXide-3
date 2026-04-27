# `editor-playtest`

Vertical-slice target: render a level cooked from the editor on
real PSX hardware (or PCSX-Redux).

## What it does

The editor's "Cook & Play" action serializes the active scene
into this crate's `generated/` directory:

```
generated/
  level_manifest.rs           Rust source — type defs + literal records
  rooms/
    room_000.psxw             cooked room geometry (one per Room node)
```

`src/main.rs` `include!`s the manifest, parses
`generated::ROOMS[0].world_bytes` through
[`psx_asset::World::from_bytes`], wraps it in a
[`psx_engine::RuntimeRoom`], and renders every populated sector
through the existing `psx_engine::draw_room` helper.

The player starts at `generated::PLAYER_SPAWN`. D-pad
LEFT/RIGHT yaws the player; UP/DOWN walks forward/back along
the player's facing. SELECT toggles a free-orbit debug camera
that pivots around the spawn point. No collision yet — you can
walk through walls.

## Cook + run

```sh
make run-editor-playtest
```

That target chains:

1. `make cook-playtest` — runs the `psxed-project::cook-playtest`
   CLI which calls `playtest::cook_to_dir(&project, &generated_dir)`.
2. `make editor-playtest` — builds this crate for `mipsel-sony-psx`.
3. Launches the desktop frontend with `PSOXIDE_EXE` pointing at
   the EXE.

The editor's "Cook & Play" button performs step 1 and prints the
exact `make` command for steps 2-3 in its status line. (The
editor doesn't spawn child processes from the cook path — running
the example is the user's call.)

## Placeholder vs cooked manifest

`generated/level_manifest.rs` is committed in a **placeholder**
state — type definitions only, no rooms. This lets a fresh clone
build the example even before the editor has run. Run
`make cook-playtest` once and the file gets overwritten with the
real cooked manifest.

The committed `rooms/` directory stays empty in placeholder
state; cooked blobs land on first cook.

## Scope

This is a vertical slice. It uses `.psxw` v1, has no collision,
no portal traversal, no asset streaming, and no entity scripting.
Lights / Triggers / Portals / AudioSources surface as warnings
during cook and are skipped from the runtime manifest.

The renderer is `psx_engine::draw_room` from the
`engine/examples/showcase-room` work — no per-frame allocation,
i16-safe vertices via `WorldVertex`, materials resolved from a
caller-provided slot table (slot 0 = floor, slot 1 = brick for
the starter project). v2 `MaterialRecordV2` will let the runtime
do its own material lookup; for now the example knows the
mapping out of band.

## Files

- `src/main.rs` — Scene impl, camera, input, draw call.
- `Cargo.toml` — standalone PSX-target crate (its own `[workspace]`).
- `.cargo/config.toml` — pins `target = "mipsel-sony-psx"`.
- `generated/level_manifest.rs` — generated; do not edit.
- `generated/rooms/*.psxw` — generated; do not edit.

## See also

- `editor/crates/psxed-project/src/playtest.rs` — manifest types,
  validation, writer.
- `editor/crates/psxed-project/src/bin/cook_playtest.rs` — CLI.
- `engine/crates/psx-engine/src/world_render.rs` — `draw_room`.
- `engine/examples/showcase-room/` — sister example that cooks
  the starter at build time and renders it directly (no editor
  generated/ step).
