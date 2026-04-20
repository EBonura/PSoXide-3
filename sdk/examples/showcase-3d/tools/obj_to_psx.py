#!/usr/bin/env python3
"""Convert vendored OBJ meshes into PSX-friendly Rust const data.

PSX GTE wants:
- Vertices as i16 triples in 1.3.12 fixed-point (`0x1000` = 1.0).
- Triangle index lists referring to those vertices.
- Per-face colour tuples (we don't have lighting, so palette-
  cycling gives the GPU something to work with).

The Newell teapot + Suzanne at their published resolutions
(6320 and 500 triangles respectively) are well beyond the
PSX-per-frame budget. We run a *vertex-clustering* decimation:
quantise every vertex to a coarse grid, merge same-cell
vertices to their centroid, drop triangles that collapse to a
line. It's not quadric-error-metric quality, but for a demo
scene where silhouette matters more than microgeometry it
produces the right shape at ~10% of the original triangle
count — fast, deterministic, and self-contained (no external
libs).

Run from the example root:

    python3 tools/obj_to_psx.py

Writes `src/meshes.rs`.
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
OUT = ROOT / "src" / "meshes.rs"

# Each entry: (obj_file, mod_name, CONST_PREFIX, grid_size, palette_name, doc)
#
# `grid_size` = N means vertex quantisation to N × N × N cells;
# higher N retains more detail but more triangles. Tune per-mesh
# for a ~60-120 triangle target on PSX.
MESHES = [
    (
        "suzanne.obj",
        "suzanne",
        "SUZANNE",
        6,
        "WARM",
        "Suzanne — the Blender monkey. Vertex-cluster-decimated "
        "from the canonical 500-tri mesh to roughly 100 tris for "
        "PSX budget. Recognisable silhouette preserved.",
    ),
    (
        "teapot.obj",
        "teapot",
        "TEAPOT",
        6,
        "COOL",
        "Utah teapot (Martin Newell, 1975). Decimated from the "
        "6320-tri canonical OBJ down to PSX-manageable poly count. "
        "Silhouette + spout + handle all readable.",
    ),
]

# Two palettes — warm (oranges / reds) for Suzanne, cool
# (cyans / blues) for the teapot. Face index cycles through the
# palette so adjacent faces get different colours.
PALETTES = {
    "WARM": [
        (220, 140, 60),
        (240, 180, 70),
        (220, 100, 50),
        (240, 120, 40),
        (200, 90, 60),
        (220, 150, 80),
    ],
    "COOL": [
        (90, 180, 220),
        (60, 160, 220),
        (120, 200, 240),
        (80, 140, 200),
        (110, 190, 230),
        (70, 150, 210),
    ],
}


def parse_obj(path: Path):
    """Return (verts: list[(x,y,z)], faces: list[(i,j,k)]) from OBJ.

    Ignores normals, texcoords, groups, and materials. Triangulates
    faces via fan triangulation so the output is always tri-only.
    OBJ indices are 1-based; we convert to 0-based.
    """
    verts = []
    faces = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()[:4]
            verts.append((float(x), float(y), float(z)))
        elif line.startswith("f "):
            # OBJ face tokens can be "i", "i/ti", "i/ti/ni", "i//ni".
            toks = line.split()[1:]
            idx = [int(t.split("/")[0]) - 1 for t in toks]
            # Fan-triangulate.
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    return verts, faces


def normalise(verts):
    """Scale + centre verts into a unit-radius sphere around origin.

    PSX 1.3.12 fixed-point wants values in roughly [-1, +1], which
    we'll map to [-0x1000, +0x1000]. Centre on centroid, then
    divide by max-extent so the mesh fits ±1.
    """
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    cz = sum(v[2] for v in verts) / len(verts)
    shifted = [(v[0] - cx, v[1] - cy, v[2] - cz) for v in verts]
    # Maximum component across all axes; keeps aspect ratio intact.
    extent = max(max(abs(v[0]), abs(v[1]), abs(v[2])) for v in shifted)
    if extent == 0:
        return shifted
    return [(v[0] / extent, v[1] / extent, v[2] / extent) for v in shifted]


def cluster_decimate(verts, faces, grid):
    """Vertex-cluster decimation into `grid^3` cells.

    Steps:
      1. Compute axis-aligned bounds of input vertices.
      2. For each vertex, compute its cell index.
      3. Merge cells: each cell produces one output vertex =
         centroid of member vertices.
      4. Rewrite each triangle to use cell-level vertex indices.
         Drop triangles where two or more indices collapse to the
         same cell (degenerate).
      5. Deduplicate triangles with the same sorted vertex set
         (different windings of the same tri after collapse).
    """
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    eps = 1e-9

    def cell_of(v):
        cx = int((v[0] - xmin) / (xmax - xmin + eps) * grid)
        cy = int((v[1] - ymin) / (ymax - ymin + eps) * grid)
        cz = int((v[2] - zmin) / (zmax - zmin + eps) * grid)
        return (min(cx, grid - 1), min(cy, grid - 1), min(cz, grid - 1))

    # Group input verts by their cell.
    cell_to_vs = {}
    for v in verts:
        c = cell_of(v)
        cell_to_vs.setdefault(c, []).append(v)

    # Emit one output vert per occupied cell (centroid).
    cell_to_idx = {}
    out_verts = []
    for c, vs in cell_to_vs.items():
        cell_to_idx[c] = len(out_verts)
        mx = sum(v[0] for v in vs) / len(vs)
        my = sum(v[1] for v in vs) / len(vs)
        mz = sum(v[2] for v in vs) / len(vs)
        out_verts.append((mx, my, mz))

    # Remap faces; drop degenerates + dedupe.
    out_faces = []
    seen_keys = set()
    for (a, b, c) in faces:
        ca = cell_of(verts[a])
        cb = cell_of(verts[b])
        cc = cell_of(verts[c])
        if ca == cb or cb == cc or ca == cc:
            continue
        ia = cell_to_idx[ca]
        ib = cell_to_idx[cb]
        ic = cell_to_idx[cc]
        key = tuple(sorted((ia, ib, ic)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out_faces.append((ia, ib, ic))

    return out_verts, out_faces


def to_q3_12(v):
    """Convert a unit-range float to Q3.12 `i16`. `1.0` → `0x1000`,
    clamped to ±0x1000."""
    # Use 0x0E00 as the max magnitude so small animation scale-up
    # headroom is preserved without clipping. `0x1000` = 1.0.
    q = int(round(v * 0x0E00))
    return max(-0x1000, min(0x0FFF, q))


def emit_mesh_module(entry):
    obj_file, mod_name, const_prefix, grid, palette_name, doc = entry
    path = VENDOR / obj_file
    verts, faces = parse_obj(path)
    verts = normalise(verts)
    verts, faces = cluster_decimate(verts, faces, grid)

    # Quantise.
    q_verts = [(to_q3_12(v[0]), to_q3_12(v[1]), to_q3_12(v[2])) for v in verts]

    palette = PALETTES[palette_name]
    face_count = len(faces)
    vert_count = len(q_verts)

    # Each triangle must have its three indices fit in u8 (< 256).
    # Clustering usually gives 50-150 unique verts, well under.
    max_idx = max(max(f) for f in faces) if faces else 0
    assert max_idx < 256, f"{mod_name}: vert index {max_idx} >= 256"

    buf = []
    buf.append(f"// GENERATED by `tools/obj_to_psx.py` from `vendor/{obj_file}`.\n")
    buf.append(f"// Do NOT edit by hand — re-run the generator.\n\n")

    # Doc block.
    for line in doc.split("\n"):
        buf.append(f"/// {line.strip()}\n")
    buf.append(f"///\n")
    buf.append(
        f"/// Cluster-decimated at grid size {grid} → "
        f"{vert_count} verts, {face_count} triangles. Values are\n"
    )
    buf.append(f"/// Q3.12 fixed-point; `0x1000` = 1.0.\n")

    # Vertices.
    buf.append(f"pub const {const_prefix}_VERTS: [Vec3I16; {vert_count}] = [\n")
    for (x, y, z) in q_verts:
        buf.append(f"    Vec3I16::new({x:#06x}, {y:#06x}, {z:#06x}),\n")
    buf.append(f"];\n\n")

    # Triangle indices.
    buf.append(f"/// Triangle indices into [`{const_prefix}_VERTS`].\n")
    buf.append(
        f"pub const {const_prefix}_TRIS: [[u8; 3]; {face_count}] = [\n"
    )
    for (a, b, c) in faces:
        buf.append(f"    [{a}, {b}, {c}],\n")
    buf.append(f"];\n\n")

    # Per-face colour — cycle through the palette so adjacent
    # faces typically get different tints.
    buf.append(
        f"/// Per-triangle flat colours. Palette cycles so adjacent\n"
        f"/// faces read distinctly.\n"
    )
    buf.append(
        f"pub const {const_prefix}_FACE_COLORS: "
        f"[(u8, u8, u8); {face_count}] = [\n"
    )
    for i in range(face_count):
        r, g, b = palette[i % len(palette)]
        buf.append(f"    ({r}, {g}, {b}),\n")
    buf.append(f"];\n\n")

    return "".join(buf), vert_count, face_count


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "//! Built-in 3D meshes — Suzanne (Blender) + Utah teapot.\n",
        "//!\n",
        "//! Generated by `tools/obj_to_psx.py`. Re-run that script\n",
        "//! if the vendored OBJs or decimation parameters change.\n",
        "//!\n",
        "//! Provenance:\n",
        "//!   - Suzanne: Blender Foundation, public domain.\n",
        "//!   - Teapot:  Martin Newell, University of Utah, 1975.\n",
        "//!              Public domain through age + author's intent.\n",
        "\n",
        "use psx_gte::math::Vec3I16;\n\n",
    ]
    for entry in MESHES:
        mesh_src, vc, fc = emit_mesh_module(entry)
        parts.append(mesh_src)
        print(f"  {entry[1]}: {vc} verts, {fc} tris")
    OUT.write_text("".join(parts))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
