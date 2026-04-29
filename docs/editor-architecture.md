# PSoXide Editor Architecture

The editor should live inside the PSoXide desktop application, but as a
separate host workspace from the emulator. The Menu remains the launcher shell:
Games and Examples boot PS1 software, while Create opens the editor.

This keeps one app identity and one preview loop, without forcing mouse-heavy
editing into the emulator overlay. Entering the editor pauses emulation, stops
routing keyboard input to the virtual pad, and gives egui full mouse/keyboard
ownership. The Menu can still be opened above the editor to return to games,
launch examples, or close the workspace.

## Workspaces

```text
PSoXide desktop app
  Menu shell
    Games       -> emulator workspace
    Examples    -> emulator workspace
    Create      -> editor workspace
    System      -> emulator controls
    Debug       -> emulator debug panels

  Emulator workspace
    framebuffer
    register / memory / VRAM panels
    gamepad and keyboard mapped to PS1 pad

  Editor workspace
    scene hierarchy
    inspector
    content browser
    editor viewport
    asset cooking / emulator preview
```

The editor code belongs under the existing `editor/` host workspace. That
workspace already owns the content pipeline (`psxed`, `.psxm`, `.psxt`), so the
natural growth path is:

- `psxed-project`: editor-side project, scene, node, and resource data.
- `psxed-ui`: reusable egui panels for the editor workspace.
- existing cooker crates: texture, mesh, and later scene/audio/font cooking.
- `emu/crates/frontend`: owns the window, Menu, workspace switch, and emulator
  preview integration.

## Data Model

The authoring model should be Godot-inspired, even though the runtime export is
PS1-specific.

```text
ProjectDocument
  Resources
    Texture
    Material
    Mesh
    Scene
    Script
    Audio

  Scene
    root: Node
      Node3D
      Room
      MeshInstance
      Light
      SpawnPoint
      Trigger
      AudioSource
```

Nodes are for authoring ergonomics. Export flattens them into runtime-friendly
data:

- world surfaces with material/culling/OT policy
- texture pages and CLUT placements
- static collision
- entity spawn records
- lights/fog/sky settings
- scripts and audio banks later

The runtime should not have to understand editor UI concepts. The editor gets a
friendly scene tree; the engine gets packed data it can submit efficiently.

## Bonnie-32 Lessons

Bonnie-32 v1 proved the all-in-one workflow: world editor, modeler, texture
tools, tracker, and game preview in one place. Bonnie-32 v2 pointed in the right
structural direction with docked panels, document tabs, a content browser,
hierarchy, inspector, and a project manifest.

For PSoXide we keep those ideas, but avoid directly embedding Godot. Integrating
Godot would move the hard problems into FFI and export sync. The better path is
to copy the useful editor grammar: scene tree, resources, inspector, document
tabs later, and an immediate emulator preview loop.

## First Vertical Slice

1. Add a Create category to the Menu.
2. Open an embedded editor workspace from Create.
3. Provide a persisted project with one scene and starter resources.
4. Show hierarchy, inspector, content browser, and viewport panels.
5. Let the user add nodes and edit names/transforms/material assignments.
6. Save/load the editor project as RON under the app config tree.
7. Add scene cooking after the material/world render path is stable.

## Entity selection + 3D move

Every entity-kind scene node (model instances, spawn points, lights,
triggers, portals, audio sources, legacy mesh markers) carries a
world-space AABB the user can click to select and drag to move.

### Selection priority

3D-viewport clicks resolve in this order:

1. **Entity bound** — `EditorWorkspace::pick_entity_bound` ray-tests
   the active room's collected `EntityBounds` and returns the nearest
   hit. A successful hit promotes that node to `selected_node`.
2. **Grid primitive** — `pick_face_with_hit` falls through to face /
   edge / vertex picking on the room geometry under the same ray.
3. **Empty space** — clears the selection.

This priority lets the user click directly on a light marker even
when it sits over a floor cell.

### Bound generation

`collect_entity_bounds(room_filter)` walks the active scene and emits
one `EntityBounds` per entity-kind node. Each bound carries:

- **`kind`** — selects the wireframe colour and whether a facing arrow
  is drawn (models / spawn points have a yaw arrow; lights / audio /
  portals don't).
- **`center` + `half_extents`** — world-space AABB. Sizes are
  per-kind heuristics (`entity_bound_kind_and_size` in psxed-ui):
  models use the parsed model bounds when available; spawns / lights /
  triggers / portals / audio fall back to fixed marker boxes sized to
  remain pickable without blocking grid clicks underneath.
- **`yaw_degrees`** — copy of the node's authored Y rotation, retained
  so the renderer can draw a facing arrow without re-walking the tree.

The picker filters by `room_filter` so a click in the active room
can't pick an entity from a neighbouring room. Bound rendering uses
the same filter for visual / picking consistency.

### Drag

`begin_node_drag` snapshots the entity's start translation and locks
the drag plane to its current world Y. `update_node_drag` re-casts
the cursor onto that plane each frame and writes
`start + (world_delta / sector_size)` back into the node's
translation, so 1 cursor-sector ≈ 1 editor-space sector. Undo is
lazy — the first non-zero delta pushes one snapshot, so a click that
doesn't move doesn't churn the undo stack.

### 2D parity

The 2D viewport selects by clicking a marker; both 2D and 3D paths
write to the same `selected_node` field and mutate
`transform.translation` directly, so a node moved in either viewport
is immediately reflected in the other.
