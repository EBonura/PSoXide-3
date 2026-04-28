# Editor model authoring

How cooked PSX models flow from disk → editor → playtest runtime.
This pass covers data model, registration, inspection, placement,
and end-to-end playtest rendering. Lights / enemies / AI are
explicitly out of scope.

## What a Model resource is

`ResourceData::Model` (in `psxed-project`) is the editor's
representation of one cooked PSX model bundle:

```rust
pub struct ModelResource {
    pub model_path: String,                 // .psxmdl
    pub texture_path: Option<String>,       // .psxt atlas
    pub clips: Vec<ModelAnimationClip>,     // sorted .psxanim files
    pub default_clip: Option<u16>,
    pub preview_clip: Option<u16>,
    pub world_height: u16,
}

pub struct ModelAnimationClip {
    pub name: String,
    pub psxanim_path: String,
}
```

The bundle is **three classes of cooked artifact** that `psx_asset`
already knows how to parse:

| File | Format | Parsed by |
| ---- | ------ | --------- |
| `.psxmdl` | `PSMD` mesh — joints, parts, vertices, faces | `psx_asset::Model::from_bytes` |
| `.psxt` | `PSXT` 4 / 8 / 15 bpp texture — pixels + CLUT | `psx_asset::Texture::from_bytes` |
| `.psxanim` | `PSXA` skeletal animation — frames × joints | `psx_asset::Animation::from_bytes` |

A `Model` resource is *required* to point at a valid `.psxmdl`.
The atlas + clips are recommended but not strictly required (a
bundle with no atlas renders untextured at runtime; one with no
clips renders in bind pose).

## How to register a cooked model bundle

`psxed_project::model_import::register_cooked_model_bundle`
adopts an existing folder containing one `.psxmdl`, optionally
one `.psxt`, and any number of `.psxanim` clips:

```rust
let id = register_cooked_model_bundle(
    &mut project,
    Path::new("assets/models/obsidian_wraith"),
    "Obsidian Wraith",
    Some(project_root),
)?;
```

The registrar:

1. Walks the folder, classifying files by extension.
2. Errors loud on `NoModelFile`, `MultipleModelFiles`, or
   `MultipleTextureFiles` — bundles must be unambiguous.
3. Parses every blob through `psx_asset` and validates that
   each animation's joint count matches the model's joint
   count.
4. Sorts clips by file name for deterministic ordering.
5. Stores paths relative to `project_root` when possible so
   the project moves freely on disk.
6. Creates a single `ResourceData::Model` resource named
   `display_name`.

## How to import a `.glb`

`psxed_project::model_import::import_glb_model` runs the GLB
through `psxed_gltf::convert_rigid_model_path`, drops the
cooked outputs under
`project_root/assets/models/<safe_name>/`, then registers them
via the same path:

```rust
let id = import_glb_model(
    &mut project,
    Path::new("source/wraith.glb"),
    "Obsidian Wraith",
    project_root,
    psxed_gltf::RigidModelConfig::default(),
)?;
```

The importer refuses to merge into a directory that already
contains non-bundle content, so user data is never silently
clobbered.

## How to inspect a Model

The Model resource inspector shows:

- **Atlas thumbnail** — decoded via `decode_psxt_thumbnail`,
  which already handles 4bpp and 8bpp indexed textures.
- **Live `.psxmdl` stats** — joints, parts, vertices, faces,
  materials, atlas dimensions, AABB derived from the parsed
  vertex table.
- **Clip list** — per-clip frame count, sample rate, and
  joint count. Mismatched joint counts surface as inline
  warnings (the cooker also rejects them).
- **Default / preview clip pickers** — `default_clip` controls
  what runtime instances play when they don't carry an
  override; `preview_clip` is what the inspector previews.

## How to place a model instance

The Place tool's `ModelInstance` mode creates a
`NodeKind::MeshInstance` whose `mesh: Some(_)` references a
`ResourceData::Model`:

- If the resource browser has a Model resource selected, that
  resource is used.
- If exactly one Model resource exists project-wide, it's
  auto-picked.
- Otherwise the click refuses with an actionable status
  message — no generic-marker fallback.

Per-instance `animation_clip: Option<u16>` overrides the model
default — `None` inherits.

## How models flow into editor-playtest

The playtest cook (`psxed_project::playtest::build_package`)
walks the scene tree and:

1. Cooks each `Room` into a `.psxw` blob (existing path).
2. For each `MeshInstance` whose `mesh` references a Model
   resource, registers that resource into the package on first
   sight (deduplicated by `ResourceId`):
   - Loads + validates the `.psxmdl`, `.psxt` atlas, and every
     `.psxanim` clip.
   - Pushes assets to the master `assets` table tagged
     `ModelMesh` / `Texture` / `ModelAnimation`.
   - Builds a `PlaytestModel` + `PlaytestModelClip` slice.
3. Pushes a `PlaytestModelInstance` per placement, resolving
   per-instance clip overrides.

The writer (`write_package`) emits cooked bytes under
`generated/models/model_NNN_<safe>/` (mesh + atlas + per-clip
files) and renders a `level_manifest.rs` containing:

```rust
pub static MODELS: &[LevelModelRecord] = &[ ... ];
pub static MODEL_CLIPS: &[LevelModelClipRecord] = &[ ... ];
pub static MODEL_INSTANCES: &[LevelModelInstanceRecord] = &[ ... ];
```

Per-room residency lists track the model assets:

- **Required RAM** = room world + every model mesh + every
  animation clip used by an instance in this room.
- **Required VRAM** = every distinct texture asset (room
  materials + model atlases) used in this room.

## How the runtime renders model instances

`engine/examples/editor-playtest/src/main.rs` walks
`MODEL_INSTANCES`:

1. Looks up the `LevelModelRecord` and its `mesh_asset`,
   parses through `psx_asset::Model::from_bytes`.
2. Resolves the atlas through `find_asset_of_kind(ASSETS,
   model.texture_asset, AssetKind::Texture)`, uploads via
   `ensure_model_atlas_uploaded` to a dedicated 8bpp tpage at
   VRAM (384, 256) with CLUT rows starting at y=484 — disjoint
   from the room-material region so the two never collide.
3. Resolves the active clip (`inst.clip` or
   `model.default_clip`) and parses the `.psxanim`.
4. Calls `WorldRenderPass::submit_textured_model` (the same
   GTE-driven path `showcase-model` uses), feeding the
   pre-allocated `MODEL_VERTICES` + `JOINT_VIEW_TRANSFORMS`
   scratch buffers.

## Caps + budgets

| Cap | Value | Set in |
| --- | ----- | ------ |
| Resident RAM assets | 128 | `MAX_RESIDENT_RAM_ASSETS` |
| Resident VRAM assets | 32 | `MAX_RESIDENT_VRAM_ASSETS` |
| Model joints | 32 | `JOINT_CAP` |
| Model vertices per part | 1024 | `MODEL_VERTEX_CAP` |
| Active model instances per frame | 16 | `MAX_MODEL_INSTANCES` |

Over-cap conditions skip the offending instance / triangle
rather than crashing, and the cook stage reports them.

## Validation

The cook (`build_package`) hard-fails on:

- Missing model `.psxmdl` file or invalid mesh bytes.
- Missing atlas `.psxt` or invalid texture bytes.
- Missing clip `.psxanim` or invalid animation bytes.
- Animation joint count ≠ model joint count.
- `MeshInstance::animation_clip` index out of range for the
  bound model.
- A `MeshInstance` referencing a non-Model resource.

Warnings (non-fatal): missing texture (rendering will skip
this instance), zero clips (renders bind pose).

## Currently out of scope

| Feature | Status |
| ------- | ------ |
| Editor 3D viewport rendering of placed models | **Deferred** — placed models still appear via their `MeshInstance` marker in the editor preview today. Inspector previews atlas + stats; runtime renders the textured animated model. The 3D-viewport renderer port lands as a follow-up. |
| Lights | Out of scope. |
| Enemies / AI / actor archetypes | Out of scope. |
| Combat / player model | Out of scope. |
| Skeleton / rig editing | Out of scope. |
| Animation retargeting | Out of scope. |
| Runtime baking / farfield | Out of scope. |
| CD streaming | Out of scope. |
| Material editor for model materials | Out of scope (model materials are baked into the `.psxmdl`). |

## See also

- `editor/crates/psxed-project/src/model_import.rs` —
  registration + GLB import + parser/stats helpers.
- `editor/crates/psxed-project/src/playtest.rs` — playtest
  cook with model pipeline.
- `engine/examples/editor-playtest/src/main.rs` — runtime
  consumer; model rendering via `submit_textured_model`.
- `engine/examples/showcase-model/src/main.rs` — the
  reference implementation the runtime model path was ported
  from.
- `docs/level-residency.md` — broader contract for the asset
  table + per-room residency.
