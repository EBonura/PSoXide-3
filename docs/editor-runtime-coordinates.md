# Editor / Runtime Coordinate Conventions

The editor and the compact runtime room format intentionally use
different naming conventions for the Z axis. The cooker is the only
place where that conversion is allowed to happen.

## Editor Preview Space

- X grows east/right.
- Z grows north/forward in the editor preview.
- A sector's visual north edge is the `+Z` edge.
- Horizontal face corners are authored as:
  `[NW, NE, SE, SW] = [(x0,z1), (x1,z1), (x1,z0), (x0,z0)]`.
- Cardinal walls are authored from the preview point of view:
  `North = +Z`, `South = -Z`, `East = +X`, `West = -X`.

This matches how the 2D/3D editor view lets a designer reason about
rooms on screen.

## Runtime `.psxw` Space

- Runtime sector storage is array-rooted as `[x * depth + z]`.
- Runtime `North` means the lower-Z edge, matching the compact world
  record constants in `psxed-format::world`.
- Horizontal face corner records are rooted at array north:
  `[NW, NE, SE, SW] = [(x0,z0), (x1,z0), (x1,z1), (x0,z1)]`.
- Wall corner records keep the runtime bottom edge ordering used by
  `psx_engine::RuntimeRoom`.

The runtime uses this layout because it keeps sector addressing and
collision sampling table-driven and cheap.

## Cooker Boundary

`psxed_project::world_cook` converts editor-authored geometry to the
runtime convention:

- horizontal heights are Z-flipped;
- diagonal split ids are swapped to preserve the same physical
  diagonal after the flip;
- North/South wall directions are swapped;
- wall corner heights swap left/right pairs for the same physical
  face.

Tests in `world_cook` assert the authored preview shape survives the
conversion. New runtime consumers should read cooked `.psxw` data;
new editor tools should read the authoring model. Avoid duplicating
the conversion anywhere else.
