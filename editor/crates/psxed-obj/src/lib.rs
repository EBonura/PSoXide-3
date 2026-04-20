//! Wavefront OBJ → PSXM mesh converter.
//!
//! Parses `.obj` ASCII meshes, optionally decimates via vertex
//! clustering, normalises into Q3.12 fixed-point, and emits the
//! binary layout defined in [`psxed_format::mesh`].
//!
//! This is the Rust port of an earlier one-off Python script
//! (`sdk/examples/showcase-3d/tools/obj_to_psx.py`) lifted into
//! the content pipeline. Same algorithm, same determinism, same
//! output byte-for-byte — but now reusable across every asset we
//! build from source meshes, and usable from `build.rs` hooks so
//! asset generation happens as part of `cargo build`.
//!
//! # Example
//!
//! ```ignore
//! use psxed_obj::{convert, Config, Palette};
//!
//! let obj_bytes = std::fs::read("suzanne.obj").unwrap();
//! let cfg = Config {
//!     decimate_grid: Some(6),
//!     palette: Palette::Warm,
//!     include_face_colors: true,
//! };
//! let psxm = convert(&obj_bytes, &cfg).unwrap();
//! std::fs::write("suzanne.psxm", psxm).unwrap();
//! ```

use std::collections::{BTreeMap, HashSet};

/// Configuration for one OBJ → PSXM conversion.
#[derive(Debug, Clone)]
pub struct Config {
    /// If `Some(n)`, run vertex-cluster decimation into `n × n × n`
    /// cells. If `None`, preserve the input mesh unchanged — use
    /// this for meshes that were authored natively low-poly.
    pub decimate_grid: Option<u32>,
    /// Palette used to assign per-face flat colours.
    pub palette: Palette,
    /// Include per-face colour table in the output. Caller can
    /// skip it for meshes that will be tinted at runtime.
    pub include_face_colors: bool,
}

/// Built-in colour palettes that cycle through face indices.
#[derive(Debug, Clone)]
pub enum Palette {
    /// Oranges / reds — good for warm-toned objects.
    Warm,
    /// Cyans / blues — cool-toned objects.
    Cool,
    /// Greens — natural / foliage.
    Green,
    /// Explicit user-provided palette (repeats across face index).
    Custom(Vec<(u8, u8, u8)>),
}

impl Palette {
    /// Colour at face index `i` — wraps around the palette.
    pub fn at(&self, i: usize) -> (u8, u8, u8) {
        let table: &[(u8, u8, u8)] = match self {
            Palette::Warm => &WARM,
            Palette::Cool => &COOL,
            Palette::Green => &GREEN,
            Palette::Custom(v) => v.as_slice(),
        };
        if table.is_empty() {
            (200, 200, 200)
        } else {
            table[i % table.len()]
        }
    }
}

const WARM: [(u8, u8, u8); 6] = [
    (220, 140, 60),
    (240, 180, 70),
    (220, 100, 50),
    (240, 120, 40),
    (200, 90, 60),
    (220, 150, 80),
];

const COOL: [(u8, u8, u8); 6] = [
    (90, 180, 220),
    (60, 160, 220),
    (120, 200, 240),
    (80, 140, 200),
    (110, 190, 230),
    (70, 150, 210),
];

const GREEN: [(u8, u8, u8); 6] = [
    (80, 180, 80),
    (120, 200, 100),
    (60, 160, 80),
    (100, 220, 120),
    (70, 170, 60),
    (110, 190, 90),
];

/// Errors the converter can raise.
#[derive(Debug)]
pub enum Error {
    /// OBJ source text is not valid UTF-8.
    InvalidUtf8,
    /// A `v`/`f` line couldn't be parsed (malformed number, missing field, …).
    ParseLine { line_no: usize, reason: &'static str },
    /// Post-decimation mesh has more than 255 vertices — indices
    /// don't fit in `u8`. Pick a smaller grid.
    TooManyVerts { count: usize },
    /// Post-decimation mesh has more than 65535 triangles.
    TooManyFaces { count: usize },
    /// Mesh has zero triangles after parse/decimate.
    Empty,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidUtf8 => write!(f, "OBJ source is not valid UTF-8"),
            Error::ParseLine { line_no, reason } => {
                write!(f, "line {line_no}: {reason}")
            }
            Error::TooManyVerts { count } => write!(
                f,
                "{count} verts exceeds u8 index range; pick a smaller grid"
            ),
            Error::TooManyFaces { count } => {
                write!(f, "{count} faces exceeds u16 face-count field")
            }
            Error::Empty => write!(f, "mesh has no triangles"),
        }
    }
}

impl std::error::Error for Error {}

/// Parse + optionally decimate + encode an OBJ blob into PSXM
/// bytes. Top-level entry point used by the `psxed` CLI and by
/// `build.rs` hooks.
pub fn convert(obj_bytes: &[u8], cfg: &Config) -> Result<Vec<u8>, Error> {
    let src = std::str::from_utf8(obj_bytes).map_err(|_| Error::InvalidUtf8)?;
    let (verts, faces) = parse_obj(src)?;
    let verts = normalise(&verts);
    let (verts, faces) = if let Some(grid) = cfg.decimate_grid {
        cluster_decimate(&verts, &faces, grid)
    } else {
        (verts, faces)
    };
    encode_psxm(&verts, &faces, cfg)
}

// ----------------------------------------------------------------------
// OBJ parser
// ----------------------------------------------------------------------

/// Parse a Wavefront OBJ source into vertex + triangle lists.
///
/// Handles vertex lines (`v x y z`) and face lines (`f a b c …`)
/// with fan-triangulation for n-gons. `vt` / `vn` / `o` / `g` /
/// `usemtl` etc. are ignored — we only need geometry.
pub fn parse_obj(src: &str) -> Result<(Vec<[f32; 3]>, Vec<[usize; 3]>), Error> {
    let mut verts = Vec::new();
    let mut faces = Vec::new();
    for (line_no_zero, raw) in src.lines().enumerate() {
        let line_no = line_no_zero + 1;
        let line = raw.trim();
        if line.starts_with("v ") {
            let mut it = line.split_whitespace();
            it.next(); // "v"
            let x: f32 = it
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or(Error::ParseLine {
                    line_no,
                    reason: "expected x",
                })?;
            let y: f32 = it
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or(Error::ParseLine {
                    line_no,
                    reason: "expected y",
                })?;
            let z: f32 = it
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or(Error::ParseLine {
                    line_no,
                    reason: "expected z",
                })?;
            verts.push([x, y, z]);
        } else if line.starts_with("f ") {
            // OBJ tokens can be "i", "i/ti", "i/ti/ni", "i//ni".
            // Only the first component matters here.
            let idx: Vec<usize> = line
                .split_whitespace()
                .skip(1)
                .map(|t| {
                    let first = t.split('/').next().unwrap_or("");
                    first.parse::<isize>().map_err(|_| Error::ParseLine {
                        line_no,
                        reason: "face index not an int",
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|i| {
                    // OBJ is 1-indexed; negative indices count from end.
                    // We handled the common positive case; our inputs
                    // don't use relative indices.
                    (i - 1) as usize
                })
                .collect();
            if idx.len() < 3 {
                return Err(Error::ParseLine {
                    line_no,
                    reason: "face with fewer than 3 verts",
                });
            }
            // Fan-triangulate n-gons.
            for k in 1..idx.len() - 1 {
                faces.push([idx[0], idx[k], idx[k + 1]]);
            }
        }
        // Everything else is ignored silently — comments, `vt`, etc.
    }
    if faces.is_empty() {
        return Err(Error::Empty);
    }
    Ok((verts, faces))
}

// ----------------------------------------------------------------------
// Normalisation + decimation
// ----------------------------------------------------------------------

/// Centre verts on their centroid and scale by the maximum-axis
/// extent so the result lives in a `[-1, +1]` cube.
pub fn normalise(verts: &[[f32; 3]]) -> Vec<[f32; 3]> {
    if verts.is_empty() {
        return vec![];
    }
    let n = verts.len() as f32;
    let cx = verts.iter().map(|v| v[0]).sum::<f32>() / n;
    let cy = verts.iter().map(|v| v[1]).sum::<f32>() / n;
    let cz = verts.iter().map(|v| v[2]).sum::<f32>() / n;
    let shifted: Vec<[f32; 3]> = verts
        .iter()
        .map(|v| [v[0] - cx, v[1] - cy, v[2] - cz])
        .collect();
    let extent = shifted
        .iter()
        .flat_map(|v| [v[0].abs(), v[1].abs(), v[2].abs()])
        .fold(0.0_f32, f32::max);
    if extent == 0.0 {
        return shifted;
    }
    shifted
        .into_iter()
        .map(|v| [v[0] / extent, v[1] / extent, v[2] / extent])
        .collect()
}

/// Vertex-cluster decimation.
///
/// Quantises every input vertex to an `N × N × N` cell, merges
/// same-cell vertices to their centroid, rewrites face indices,
/// drops degenerate triangles (where ≥ 2 verts collapse to the
/// same cell), and deduplicates triangles that end up with the
/// same three output indices after collapse.
pub fn cluster_decimate(
    verts: &[[f32; 3]],
    faces: &[[usize; 3]],
    grid: u32,
) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
    if verts.is_empty() || grid == 0 {
        return (verts.to_vec(), faces.to_vec());
    }
    // Axis-aligned bounds.
    let mut xmin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    let mut zmin = f32::INFINITY;
    let mut zmax = f32::NEG_INFINITY;
    for v in verts {
        xmin = xmin.min(v[0]);
        xmax = xmax.max(v[0]);
        ymin = ymin.min(v[1]);
        ymax = ymax.max(v[1]);
        zmin = zmin.min(v[2]);
        zmax = zmax.max(v[2]);
    }
    let eps = 1e-9_f32;
    let grid_f = grid as f32;
    let cell_of = |v: &[f32; 3]| -> (u32, u32, u32) {
        let cx = ((v[0] - xmin) / (xmax - xmin + eps) * grid_f) as u32;
        let cy = ((v[1] - ymin) / (ymax - ymin + eps) * grid_f) as u32;
        let cz = ((v[2] - zmin) / (zmax - zmin + eps) * grid_f) as u32;
        (cx.min(grid - 1), cy.min(grid - 1), cz.min(grid - 1))
    };

    // Group input verts by cell. BTreeMap instead of HashMap so
    // iteration order is deterministic (sorted by cell key) —
    // otherwise every build gets different vertex indices, which
    // flows through to face-index palette cycling and makes the
    // rendered output non-reproducible.
    let mut cell_sum: BTreeMap<(u32, u32, u32), ([f32; 3], u32)> = BTreeMap::new();
    for v in verts {
        let c = cell_of(v);
        let entry = cell_sum.entry(c).or_insert(([0.0; 3], 0));
        entry.0[0] += v[0];
        entry.0[1] += v[1];
        entry.0[2] += v[2];
        entry.1 += 1;
    }

    // Emit one output vert per occupied cell (centroid of its members).
    let mut cell_to_idx: BTreeMap<(u32, u32, u32), usize> = BTreeMap::new();
    let mut out_verts: Vec<[f32; 3]> = Vec::with_capacity(cell_sum.len());
    for (cell, (sum, count)) in &cell_sum {
        let n = *count as f32;
        let c = [sum[0] / n, sum[1] / n, sum[2] / n];
        cell_to_idx.insert(*cell, out_verts.len());
        out_verts.push(c);
    }

    // Remap faces; drop degenerates + dedupe.
    let mut out_faces = Vec::new();
    let mut seen: HashSet<[usize; 3]> = HashSet::new();
    for f in faces {
        let ca = cell_of(&verts[f[0]]);
        let cb = cell_of(&verts[f[1]]);
        let cc = cell_of(&verts[f[2]]);
        if ca == cb || cb == cc || ca == cc {
            continue;
        }
        let ia = cell_to_idx[&ca];
        let ib = cell_to_idx[&cb];
        let ic = cell_to_idx[&cc];
        let mut key = [ia, ib, ic];
        key.sort_unstable();
        if seen.insert(key) {
            out_faces.push([ia, ib, ic]);
        }
    }
    (out_verts, out_faces)
}

// ----------------------------------------------------------------------
// PSXM encoder
// ----------------------------------------------------------------------

/// Quantise `[-1, +1]` float to Q3.12 `i16`. `1.0` maps to
/// `0x0E00` — leaves headroom below `0x1000` for animation
/// scale-up without overflow.
fn to_q3_12(v: f32) -> i16 {
    let q = (v * 0x0E00 as f32).round() as i32;
    q.clamp(-0x1000, 0x0FFF) as i16
}

/// Encode a vertex/face/colour set into `.psxm` bytes.
pub fn encode_psxm(
    verts: &[[f32; 3]],
    faces: &[[usize; 3]],
    cfg: &Config,
) -> Result<Vec<u8>, Error> {
    if faces.is_empty() {
        return Err(Error::Empty);
    }
    if verts.len() > 255 {
        return Err(Error::TooManyVerts { count: verts.len() });
    }
    if faces.len() > u16::MAX as usize {
        return Err(Error::TooManyFaces { count: faces.len() });
    }

    let vert_count = verts.len() as u16;
    let face_count = faces.len() as u16;

    // Compute payload size.
    let vert_bytes = (vert_count as usize) * 6;
    let index_bytes = (face_count as usize) * 3;
    let color_bytes = if cfg.include_face_colors {
        (face_count as usize) * 3
    } else {
        0
    };
    let payload_len = psxed_format::mesh::MeshHeader::SIZE + vert_bytes + index_bytes + color_bytes;

    let mut buf = Vec::with_capacity(psxed_format::AssetHeader::SIZE + payload_len);

    // AssetHeader.
    buf.extend_from_slice(&psxed_format::mesh::MAGIC);
    buf.extend_from_slice(&psxed_format::mesh::VERSION.to_le_bytes());
    let flags = if cfg.include_face_colors {
        psxed_format::mesh::flags::HAS_FACE_COLORS
    } else {
        0
    };
    buf.extend_from_slice(&flags.to_le_bytes());
    buf.extend_from_slice(&(payload_len as u32).to_le_bytes());

    // MeshHeader.
    buf.extend_from_slice(&vert_count.to_le_bytes());
    buf.extend_from_slice(&face_count.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // Vertex table.
    for v in verts {
        buf.extend_from_slice(&to_q3_12(v[0]).to_le_bytes());
        buf.extend_from_slice(&to_q3_12(v[1]).to_le_bytes());
        buf.extend_from_slice(&to_q3_12(v[2]).to_le_bytes());
    }

    // Index table.
    for f in faces {
        buf.push(f[0] as u8);
        buf.push(f[1] as u8);
        buf.push(f[2] as u8);
    }

    // Face colours.
    if cfg.include_face_colors {
        for i in 0..faces.len() {
            let (r, g, b) = cfg.palette.at(i);
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRI_OBJ: &str = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
f 1 2 3
";

    #[test]
    fn parses_simple_tri() {
        let (verts, faces) = parse_obj(TRI_OBJ).unwrap();
        assert_eq!(verts.len(), 3);
        assert_eq!(faces, vec![[0, 1, 2]]);
    }

    #[test]
    fn encodes_header_correctly() {
        let bytes = convert(
            TRI_OBJ.as_bytes(),
            &Config {
                decimate_grid: None,
                palette: Palette::Warm,
                include_face_colors: true,
            },
        )
        .unwrap();
        // Magic + version + flags + payload_len + mesh header + 3 verts + 1 face + 1 colour.
        assert_eq!(&bytes[..4], b"PSXM");
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 1);
        assert_eq!(
            u16::from_le_bytes([bytes[6], bytes[7]]),
            psxed_format::mesh::flags::HAS_FACE_COLORS
        );
    }

    #[test]
    fn decimate_preserves_single_tri() {
        let (verts, faces) = parse_obj(TRI_OBJ).unwrap();
        let verts = normalise(&verts);
        let (dv, df) = cluster_decimate(&verts, &faces, 8);
        assert_eq!(df.len(), 1);
        assert_eq!(dv.len(), 3);
    }

    #[test]
    fn quad_face_gets_fan_triangulated() {
        let quad_obj = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
";
        let (_, faces) = parse_obj(quad_obj).unwrap();
        assert_eq!(faces.len(), 2, "quad should fan-triangulate into 2 tris");
    }
}
