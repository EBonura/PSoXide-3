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
