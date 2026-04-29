//! glTF/GLB -> PSXM mesh converter.
//!
//! This is a host-side importer only. The editor/content pipeline can
//! afford a modern glTF parser; the PS1 runtime should keep consuming
//! compact cooked `.psxm` blobs through `psx-asset`.
//!
//! Current scope is deliberately conservative:
//! - mesh primitives from the default scene, or all scenes if no default exists
//! - node transforms baked into vertex positions
//! - triangle, triangle-strip, and triangle-fan primitives
//! - material base colours baked into the PSXM face-colour table
//! - optional vertex-cluster decimation and computed normals
//!
//! Textures/UVs are parsed by glTF but not emitted yet because the
//! current PSXM runtime format has no UV/material table. That is the
//! next format bump once we are ready for textured imported models.

use std::collections::BTreeMap;
use std::path::Path;

use gltf::animation::{Interpolation, Property};
use gltf::image::Source;
use gltf::mesh::Mode;

pub use psxed_obj::Palette;

const MODEL_LOCAL_COORD_LIMIT: f32 = 30_000.0;
const DEFAULT_MODEL_WORLD_HEIGHT: u16 = 1024;

/// Conversion configuration for glTF/GLB imports.
#[derive(Debug, Clone)]
pub struct Config {
    /// If `Some(n)`, run vertex-cluster decimation into `n x n x n`
    /// cells. Keep this `None` for hand-authored low-poly meshes.
    pub decimate_grid: Option<u32>,
    /// Fallback palette used when material colours are disabled or
    /// unavailable.
    pub palette: Palette,
    /// Include a face-colour table in the cooked PSXM.
    pub include_face_colors: bool,
    /// Compute per-vertex normals for lit engine render passes.
    pub include_normals: bool,
    /// Use glTF material `baseColorFactor` as per-face colours.
    pub use_material_colors: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            decimate_grid: None,
            palette: Palette::Warm,
            include_face_colors: true,
            include_normals: true,
            use_material_colors: true,
        }
    }
}

/// Errors raised while importing a glTF/GLB mesh.
#[derive(Debug)]
pub enum Error {
    /// glTF parser/importer failure.
    Import(gltf::Error),
    /// A primitive had no POSITION attribute.
    MissingPositions { primitive_index: usize },
    /// A primitive mode cannot be represented as triangles.
    UnsupportedMode {
        /// glTF primitive mode.
        mode: Mode,
    },
    /// Primitive index points past its POSITION accessor.
    BadIndex {
        /// Invalid local index from the primitive.
        index: u32,
        /// Number of vertices in the primitive POSITION stream.
        vertex_count: usize,
    },
    /// No triangles survived import/decimation.
    Empty,
    /// Cooked PSXM encoding failed.
    Cook(psxed_obj::Error),
    /// Texture conversion failed.
    TextureCook(psxed_tex::Error),
    /// The GLB has no skinned mesh suitable for `.psxmdl` cooking.
    MissingSkinnedMesh,
    /// A skinned primitive is missing data needed by the native model cooker.
    MissingAttribute {
        /// glTF primitive index.
        primitive_index: usize,
        /// Attribute or table name.
        attribute: &'static str,
    },
    /// A count exceeds the cooked format's current index range.
    TooMany {
        /// Kind of item that exceeded the limit.
        kind: &'static str,
        /// Actual count.
        count: usize,
        /// Maximum supported count.
        max: usize,
    },
    /// A skin references data inconsistent with the mesh or animation.
    BadSkin(&'static str),
    /// Base-color texture source is not embedded in the GLB.
    UnsupportedImageSource,
    /// Animation channel input keyframes are missing.
    MissingAnimationInputs {
        /// glTF channel index.
        channel_index: usize,
    },
    /// Animation channel output values are missing.
    MissingAnimationOutputs {
        /// glTF channel index.
        channel_index: usize,
    },
    /// Animation channel interpolation is not supported by the cooker.
    UnsupportedAnimationInterpolation {
        /// glTF channel index.
        channel_index: usize,
        /// Interpolation mode.
        interpolation: Interpolation,
    },
    /// Animation channel property and output accessor type do not match.
    AnimationTypeMismatch {
        /// glTF channel index.
        channel_index: usize,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Import(error) => write!(f, "glTF import failed: {error}"),
            Self::MissingPositions { primitive_index } => {
                write!(f, "primitive #{primitive_index} is missing POSITION data")
            }
            Self::UnsupportedMode { mode } => {
                write!(f, "unsupported primitive mode: {mode:?}")
            }
            Self::BadIndex {
                index,
                vertex_count,
            } => write!(
                f,
                "primitive index {index} points past {vertex_count} POSITION vertices"
            ),
            Self::Empty => write!(f, "glTF scene contains no importable triangles"),
            Self::Cook(error) => write!(f, "PSXM encode failed: {error}"),
            Self::TextureCook(error) => write!(f, "PSXT encode failed: {error}"),
            Self::MissingSkinnedMesh => write!(f, "glTF scene contains no skinned mesh"),
            Self::MissingAttribute {
                primitive_index,
                attribute,
            } => write!(
                f,
                "skinned primitive #{primitive_index} is missing {attribute} data"
            ),
            Self::TooMany { kind, count, max } => {
                write!(f, "too many {kind}: {count} exceeds limit {max}")
            }
            Self::BadSkin(reason) => write!(f, "invalid skin: {reason}"),
            Self::UnsupportedImageSource => {
                write!(f, "base-color texture must be embedded in the GLB")
            }
            Self::MissingAnimationInputs { channel_index } => {
                write!(f, "animation channel #{channel_index} is missing input keys")
            }
            Self::MissingAnimationOutputs { channel_index } => {
                write!(f, "animation channel #{channel_index} is missing output values")
            }
            Self::UnsupportedAnimationInterpolation {
                channel_index,
                interpolation,
            } => write!(
                f,
                "animation channel #{channel_index} uses unsupported interpolation {interpolation:?}"
            ),
            Self::AnimationTypeMismatch { channel_index } => {
                write!(f, "animation channel #{channel_index} has mismatched output type")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<gltf::Error> for Error {
    fn from(value: gltf::Error) -> Self {
        Self::Import(value)
    }
}

impl From<psxed_obj::Error> for Error {
    fn from(value: psxed_obj::Error) -> Self {
        Self::Cook(value)
    }
}

impl From<psxed_tex::Error> for Error {
    fn from(value: psxed_tex::Error) -> Self {
        Self::TextureCook(value)
    }
}

#[derive(Default)]
struct CollectedMesh {
    verts: Vec<[f32; 3]>,
    faces: Vec<[usize; 3]>,
    face_colors: Vec<(u8, u8, u8)>,
}

/// Convert a `.gltf` or `.glb` file to cooked PSXM bytes.
pub fn convert_path(path: impl AsRef<Path>, cfg: &Config) -> Result<Vec<u8>, Error> {
    let (document, buffers, _images) = gltf::import(path)?;
    convert_document(&document, &buffers, cfg)
}

/// Convert an in-memory `.glb` or self-contained `.gltf` blob.
pub fn convert_slice(bytes: &[u8], cfg: &Config) -> Result<Vec<u8>, Error> {
    let (document, buffers, _images) = gltf::import_slice(bytes)?;
    convert_document(&document, &buffers, cfg)
}

fn convert_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    cfg: &Config,
) -> Result<Vec<u8>, Error> {
    let mut collected = CollectedMesh::default();
    let identity = identity_matrix();

    let visited_scene_mesh = if let Some(scene) = document.default_scene() {
        for node in scene.nodes() {
            visit_node(node, &identity, buffers, &mut collected)?;
        }
        !collected.faces.is_empty()
    } else {
        for scene in document.scenes() {
            for node in scene.nodes() {
                visit_node(node, &identity, buffers, &mut collected)?;
            }
        }
        !collected.faces.is_empty()
    };

    // Some authoring/export paths keep meshes unattached from scenes.
    // Import them at identity so users still get a useful asset.
    if !visited_scene_mesh {
        for mesh in document.meshes() {
            read_mesh(mesh, &identity, buffers, &mut collected)?;
        }
    }

    if collected.faces.is_empty() {
        return Err(Error::Empty);
    }

    let verts = psxed_obj::normalise(&collected.verts);
    let (verts, faces, face_colors) = if let Some(grid) = cfg.decimate_grid {
        psxed_obj::cluster_decimate_with_face_data(
            &verts,
            &collected.faces,
            &collected.face_colors,
            grid,
        )
    } else {
        (verts, collected.faces, collected.face_colors)
    };

    if faces.is_empty() {
        return Err(Error::Empty);
    }

    let normals_vec = if cfg.include_normals {
        Some(psxed_obj::compute_vertex_normals(&verts, &faces))
    } else {
        None
    };
    let normals = normals_vec.as_deref();

    let palette = if cfg.include_face_colors && cfg.use_material_colors {
        Palette::Custom(face_colors)
    } else {
        cfg.palette.clone()
    };
    let cook_cfg = psxed_obj::Config {
        decimate_grid: None,
        palette,
        include_face_colors: cfg.include_face_colors,
        include_normals: cfg.include_normals,
    };
    psxed_obj::encode_psxm(&verts, &faces, normals, &cook_cfg).map_err(Error::Cook)
}

fn visit_node(
    node: gltf::Node<'_>,
    parent: &[[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    out: &mut CollectedMesh,
) -> Result<(), Error> {
    let local = node.transform().matrix();
    let world = mul_matrix(parent, &local);
    if let Some(mesh) = node.mesh() {
        read_mesh(mesh, &world, buffers, out)?;
    }
    for child in node.children() {
        visit_node(child, &world, buffers, out)?;
    }
    Ok(())
}

fn read_mesh(
    mesh: gltf::Mesh<'_>,
    transform: &[[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    out: &mut CollectedMesh,
) -> Result<(), Error> {
    for primitive in mesh.primitives() {
        let primitive_index = primitive.index();
        let material_color = base_color_rgb(&primitive.material());
        let reader = primitive.reader(|buffer| Some(buffers[buffer.index()].0.as_slice()));
        let positions: Vec<[f32; 3]> = reader
            .read_positions()
            .ok_or(Error::MissingPositions { primitive_index })?
            .map(|p| transform_point(transform, p))
            .collect();
        let vertex_count = positions.len();
        let base = out.verts.len();
        out.verts.extend(positions);

        let local_indices: Vec<u32> = if let Some(indices) = reader.read_indices() {
            indices.into_u32().collect()
        } else {
            (0..vertex_count as u32).collect()
        };
        let local_faces = triangulate_indices(&local_indices, primitive.mode())?;
        for face in local_faces {
            let a = checked_index(face[0], vertex_count)? + base;
            let b = checked_index(face[1], vertex_count)? + base;
            let c = checked_index(face[2], vertex_count)? + base;
            out.faces.push([a, b, c]);
            out.face_colors.push(material_color);
        }
    }
    Ok(())
}

fn triangulate_indices(indices: &[u32], mode: Mode) -> Result<Vec<[u32; 3]>, Error> {
    let mut faces = Vec::new();
    match mode {
        Mode::Triangles => {
            for tri in indices.chunks_exact(3) {
                faces.push([tri[0], tri[1], tri[2]]);
            }
        }
        Mode::TriangleStrip => {
            for i in 0..indices.len().saturating_sub(2) {
                let tri = if i & 1 == 0 {
                    [indices[i], indices[i + 1], indices[i + 2]]
                } else {
                    [indices[i + 1], indices[i], indices[i + 2]]
                };
                if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                    faces.push(tri);
                }
            }
        }
        Mode::TriangleFan => {
            for i in 1..indices.len().saturating_sub(1) {
                let tri = [indices[0], indices[i], indices[i + 1]];
                if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                    faces.push(tri);
                }
            }
        }
        other => return Err(Error::UnsupportedMode { mode: other }),
    }
    Ok(faces)
}

fn checked_index(index: u32, vertex_count: usize) -> Result<usize, Error> {
    if index as usize >= vertex_count {
        Err(Error::BadIndex {
            index,
            vertex_count,
        })
    } else {
        Ok(index as usize)
    }
}

fn base_color_rgb(material: &gltf::Material<'_>) -> (u8, u8, u8) {
    let color = material.pbr_metallic_roughness().base_color_factor();
    (
        linear_to_u8(color[0]),
        linear_to_u8(color[1]),
        linear_to_u8(color[2]),
    )
}

fn linear_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// glTF matrices are column-major. Keep the same representation here:
// m[column][row].
fn mul_matrix(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}

fn transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

/// Conversion configuration for GLB -> native textured/skinned assets.
#[derive(Debug, Clone)]
pub struct RigidModelConfig {
    /// Target texture width in texels.
    pub texture_width: u16,
    /// Target texture height in texels.
    pub texture_height: u16,
    /// Target PSX texture colour depth.
    pub texture_depth: psxed_format::texture::Depth,
    /// Fixed animation sample rate in Hz.
    pub animation_fps: u16,
    /// Suggested world-space height for this model in engine units.
    ///
    /// The cooker still extracts maximum model-local precision
    /// automatically; this value only determines the recommended
    /// local-to-world scale stored in the `.psxmdl` header.
    pub world_height: u16,
}

impl Default for RigidModelConfig {
    fn default() -> Self {
        Self {
            texture_width: 128,
            texture_height: 128,
            texture_depth: psxed_format::texture::Depth::Bit8,
            animation_fps: 15,
            world_height: DEFAULT_MODEL_WORLD_HEIGHT,
        }
    }
}

/// Output package from the native model cooker.
#[derive(Debug, Clone)]
pub struct RigidModelPackage {
    /// Cooked `.psxmdl` bytes.
    pub model: Vec<u8>,
    /// Cooked `.psxanim` bytes per source animation clip.
    ///
    /// Empty when the source has no animations. One entry per glTF
    /// animation, in source order, each with a filesystem-safe name
    /// derived from `gltf::Animation::name`.
    pub clips: Vec<CookedClip>,
    /// Cooked `.psxt` base-colour texture, if present.
    pub texture: Option<Vec<u8>>,
    /// Counts and byte sizes useful for build logs and tests.
    pub report: RigidModelReport,
}

/// One cooked animation clip ready to write to disk.
#[derive(Debug, Clone)]
pub struct CookedClip {
    /// Original glTF clip name, if the source provided one.
    pub source_name: Option<String>,
    /// Filesystem-safe name suitable as a filename suffix.
    pub sanitized_name: String,
    /// Cooked `.psxanim` bytes for this clip.
    pub bytes: Vec<u8>,
    /// Number of sampled frames in the cooked clip.
    pub frames: usize,
}

/// Summary of a native model import.
#[derive(Debug, Clone)]
pub struct RigidModelReport {
    /// Number of source vertices before rigid part duplication.
    pub source_vertices: usize,
    /// Number of cooked vertices after per-joint duplication.
    pub cooked_vertices: usize,
    /// Number of cooked triangles.
    pub faces: usize,
    /// Number of rigid parts.
    pub parts: usize,
    /// Number of skin joints.
    pub joints: usize,
    /// Per-clip frame count, one entry per cooked clip.
    pub clip_frames: Vec<(String, usize)>,
    /// Cooked animated model height in model-local units.
    pub local_height: usize,
    /// Suggested model-local to world-space scale, Q12.
    pub local_to_world_q12: u16,
    /// Cooked model byte length.
    pub model_bytes: usize,
    /// Total cooked animation byte length across all clips.
    pub animation_bytes: usize,
    /// Cooked texture byte length, or zero when no texture exists.
    pub texture_bytes: usize,
}

/// Convert a `.glb` or `.gltf` file into native model/animation/texture blobs.
pub fn convert_rigid_model_path(
    path: impl AsRef<Path>,
    cfg: &RigidModelConfig,
) -> Result<RigidModelPackage, Error> {
    let (document, buffers, _images) = gltf::import(path)?;
    convert_rigid_model_document(&document, &buffers, cfg)
}

/// Convert an in-memory `.glb` into native model/animation/texture blobs.
pub fn convert_rigid_model_slice(
    bytes: &[u8],
    cfg: &RigidModelConfig,
) -> Result<RigidModelPackage, Error> {
    let (document, buffers, _images) = gltf::import_slice(bytes)?;
    convert_rigid_model_document(&document, &buffers, cfg)
}

fn convert_rigid_model_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    cfg: &RigidModelConfig,
) -> Result<RigidModelPackage, Error> {
    if cfg.animation_fps == 0 {
        return Err(Error::BadSkin("animation sample rate must be non-zero"));
    }

    let mesh_node = document
        .nodes()
        .find(|node| node.mesh().is_some() && node.skin().is_some())
        .ok_or(Error::MissingSkinnedMesh)?;
    let mesh = mesh_node.mesh().ok_or(Error::MissingSkinnedMesh)?;
    let skin = mesh_node.skin().ok_or(Error::MissingSkinnedMesh)?;
    let joints: Vec<usize> = skin.joints().map(|joint| joint.index()).collect();
    if joints.is_empty() {
        return Err(Error::BadSkin("skin has no joints"));
    }
    ensure_u16("joints", joints.len())?;

    let parents = build_parent_indices(document);
    let base_trs = collect_base_trs(document);
    let inverse_bind_matrices = read_inverse_bind_matrices(&skin, buffers, joints.len());
    if inverse_bind_matrices.len() != joints.len() {
        return Err(Error::BadSkin(
            "inverse bind matrix count does not match joint count",
        ));
    }

    let mut source = read_skinned_mesh(&mesh, buffers, joints.len())?;
    if source.faces.is_empty() {
        return Err(Error::Empty);
    }
    assign_face_joints(&mut source, joints.len());
    let precision_bounds = collect_precision_bounds(
        document,
        buffers,
        &source,
        &parents,
        &base_trs,
        &joints,
        &inverse_bind_matrices,
        cfg.animation_fps,
    )?;
    let bounds = ModelBounds::from_min_max(
        precision_bounds.min,
        precision_bounds.max,
        MODEL_LOCAL_COORD_LIMIT,
    )?;
    let local_height = bounds.encoded_axis_size(precision_bounds.min[1], precision_bounds.max[1]);
    let local_to_world_q12 = choose_local_to_world_q12(local_height, cfg.world_height);

    let texture = cook_base_color_texture(&mesh, buffers, cfg)?;
    let material_color = first_material_base_color(&mesh);
    let (model, cooked_vertices, parts) = cook_model_blob(
        &source,
        &bounds,
        &parents,
        &joints,
        material_color,
        cfg.texture_width,
        cfg.texture_height,
        local_to_world_q12,
    )?;
    let clips = cook_all_animations(
        document,
        buffers,
        &parents,
        &base_trs,
        &joints,
        &inverse_bind_matrices,
        &bounds,
        cfg.animation_fps,
    )?;

    let animation_bytes = clips.iter().map(|c| c.bytes.len()).sum();
    let clip_frames = clips
        .iter()
        .map(|c| (c.sanitized_name.clone(), c.frames))
        .collect();
    let report = RigidModelReport {
        source_vertices: source.vertices.len(),
        cooked_vertices,
        faces: source.faces.len(),
        parts,
        joints: joints.len(),
        clip_frames,
        local_height: local_height.max(0) as usize,
        local_to_world_q12,
        model_bytes: model.len(),
        animation_bytes,
        texture_bytes: texture.as_ref().map_or(0, Vec::len),
    };

    Ok(RigidModelPackage {
        model,
        clips,
        texture,
        report,
    })
}

#[derive(Clone, Copy, Debug)]
struct Trs {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

impl Trs {
    fn matrix(&self) -> [[f32; 4]; 4] {
        compose_trs(self.translation, self.rotation, self.scale)
    }
}

#[derive(Clone, Copy, Debug)]
struct SourceVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    joints: [u16; 4],
    weights: [f32; 4],
    /// Bone this vertex follows under rigid skinning. Picked as the
    /// joint with the highest weight in `weights`. Pre-computed so the
    /// face-grouping pass and the seam-duplication step both see the
    /// same per-vertex choice.
    dominant_joint: u16,
}

#[derive(Clone, Copy, Debug)]
struct SourceFace {
    indices: [usize; 3],
    joint: u16,
}

#[derive(Default)]
struct SkinnedSourceMesh {
    vertices: Vec<SourceVertex>,
    faces: Vec<SourceFace>,
}

#[derive(Clone, Copy)]
struct ModelBounds {
    center: [f32; 3],
    extent: f32,
}

impl ModelBounds {
    fn from_min_max(min: [f32; 3], max: [f32; 3], coord_limit: f32) -> Result<Self, Error> {
        if !min.iter().all(|v| v.is_finite()) || !max.iter().all(|v| v.is_finite()) {
            return Err(Error::Empty);
        }
        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let mut half_extent = 0.0f32;
        for axis in 0..3 {
            half_extent = half_extent.max((max[axis] - min[axis]).abs() * 0.5);
        }
        let coord_limit = coord_limit.clamp(1.0, i16::MAX as f32);
        let extent = half_extent * 4096.0 / coord_limit;
        Ok(Self {
            center,
            extent: extent.max(0.0001),
        })
    }

    fn normalize_point(&self, p: [f32; 3]) -> [f32; 3] {
        [
            (p[0] - self.center[0]) / self.extent,
            (p[1] - self.center[1]) / self.extent,
            (p[2] - self.center[2]) / self.extent,
        ]
    }

    fn encoded_axis_size(&self, min: f32, max: f32) -> i32 {
        let size = ((max - min).abs() / self.extent * 4096.0).round();
        if !size.is_finite() {
            return 0;
        }
        size.max(1.0).min(i32::MAX as f32) as i32
    }
}

#[derive(Clone, Copy, Debug)]
struct PrecisionBounds {
    min: [f32; 3],
    max: [f32; 3],
}

struct BoundsAccumulator {
    min: [f32; 3],
    max: [f32; 3],
    any: bool,
}

impl BoundsAccumulator {
    const fn new() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            any: false,
        }
    }

    fn include(&mut self, p: [f32; 3]) {
        if !p.iter().all(|v| v.is_finite()) {
            return;
        }
        self.any = true;
        for (axis, value) in p.iter().copied().enumerate() {
            self.min[axis] = self.min[axis].min(value);
            self.max[axis] = self.max[axis].max(value);
        }
    }

    fn finish(self) -> Result<PrecisionBounds, Error> {
        if !self.any {
            return Err(Error::Empty);
        }
        Ok(PrecisionBounds {
            min: self.min,
            max: self.max,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_precision_bounds(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    source: &SkinnedSourceMesh,
    parents: &[Option<usize>],
    base_trs: &[Trs],
    joints: &[usize],
    inverse_bind_matrices: &[[[f32; 4]; 4]],
    fps: u16,
) -> Result<PrecisionBounds, Error> {
    let mut bounds = BoundsAccumulator::new();
    include_pose_bounds(
        &mut bounds,
        base_trs,
        source,
        parents,
        joints,
        inverse_bind_matrices,
    );

    for animation in document.animations() {
        let channels = read_animation_channels(&animation, buffers)?;
        if channels.is_empty() {
            continue;
        }

        let Some((min_time, max_time)) = channel_time_range(&channels) else {
            continue;
        };

        let duration = max_time - min_time;
        let frame_count = (duration * fps as f32).round() as usize + 1;
        ensure_u16("animation frames", frame_count)?;
        for frame in 0..frame_count {
            let time = (min_time + frame as f32 / fps as f32).min(max_time);
            let mut frame_trs = base_trs.to_vec();
            for channel in &channels {
                channel.apply(time, &mut frame_trs);
            }
            include_pose_bounds(
                &mut bounds,
                &frame_trs,
                source,
                parents,
                joints,
                inverse_bind_matrices,
            );
        }
    }

    bounds.finish()
}

fn channel_time_range(channels: &[AnimationChannel]) -> Option<(f32, f32)> {
    let mut min_time = f32::INFINITY;
    let mut max_time = f32::NEG_INFINITY;
    for channel in channels {
        for time in &channel.times {
            min_time = min_time.min(*time);
            max_time = max_time.max(*time);
        }
    }
    if min_time.is_finite() && max_time.is_finite() && max_time >= min_time {
        Some((min_time, max_time))
    } else {
        None
    }
}

fn include_pose_bounds(
    bounds: &mut BoundsAccumulator,
    trs: &[Trs],
    source: &SkinnedSourceMesh,
    parents: &[Option<usize>],
    joints: &[usize],
    inverse_bind_matrices: &[[[f32; 4]; 4]],
) {
    let locals: Vec<[[f32; 4]; 4]> = trs.iter().map(|trs| trs.matrix()).collect();
    let globals = compute_global_matrices(parents, &locals);
    let skins: Vec<[[f32; 4]; 4]> = joints
        .iter()
        .copied()
        .enumerate()
        .map(|(joint_index, node_index)| {
            mul_matrix(&globals[node_index], &inverse_bind_matrices[joint_index])
        })
        .collect();

    for face in &source.faces {
        let skin = skins[face.joint as usize];
        for vertex_index in face.indices {
            bounds.include(transform_point(
                &skin,
                source.vertices[vertex_index].position,
            ));
        }
    }
}

fn choose_local_to_world_q12(local_height: i32, world_height: u16) -> u16 {
    if local_height <= 0 || world_height == 0 {
        return psxed_format::model::DEFAULT_LOCAL_TO_WORLD_Q12;
    }
    let local_height = local_height as u32;
    let numerator = world_height as u32 * 4096;
    ((numerator + local_height / 2) / local_height).clamp(1, u16::MAX as u32) as u16
}

fn read_skinned_mesh(
    mesh: &gltf::Mesh<'_>,
    buffers: &[gltf::buffer::Data],
    joint_count: usize,
) -> Result<SkinnedSourceMesh, Error> {
    let mut source = SkinnedSourceMesh::default();
    let mut normal_faces = Vec::new();
    for primitive in mesh.primitives() {
        let primitive_index = primitive.index();
        if primitive.mode() != Mode::Triangles
            && primitive.mode() != Mode::TriangleStrip
            && primitive.mode() != Mode::TriangleFan
        {
            return Err(Error::UnsupportedMode {
                mode: primitive.mode(),
            });
        }
        let reader = primitive.reader(|buffer| Some(buffers[buffer.index()].0.as_slice()));
        let positions: Vec<[f32; 3]> = reader
            .read_positions()
            .ok_or(Error::MissingAttribute {
                primitive_index,
                attribute: "POSITION",
            })?
            .collect();
        let normals: Option<Vec<[f32; 3]>> = reader
            .read_normals()
            .map(|iter| iter.map(normalize3).collect());
        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .ok_or(Error::MissingAttribute {
                primitive_index,
                attribute: "TEXCOORD_0",
            })?
            .into_f32()
            .collect();
        let joints: Vec<[u16; 4]> = reader
            .read_joints(0)
            .ok_or(Error::MissingAttribute {
                primitive_index,
                attribute: "JOINTS_0",
            })?
            .into_u16()
            .collect();
        let weights: Vec<[f32; 4]> = reader
            .read_weights(0)
            .ok_or(Error::MissingAttribute {
                primitive_index,
                attribute: "WEIGHTS_0",
            })?
            .into_f32()
            .collect();

        let vertex_count = positions.len();
        if normals
            .as_ref()
            .is_some_and(|normals| normals.len() != vertex_count)
            || uvs.len() != vertex_count
            || joints.len() != vertex_count
            || weights.len() != vertex_count
        {
            return Err(Error::BadSkin("primitive attribute counts differ"));
        }
        for (joint_indices, vertex_weights) in joints.iter().zip(&weights) {
            if joint_indices
                .iter()
                .zip(vertex_weights)
                .any(|(joint, weight)| *weight > 0.0 && *joint as usize >= joint_count)
            {
                return Err(Error::BadSkin(
                    "vertex joint index outside skin joint table",
                ));
            }
        }

        let base = source.vertices.len();
        source.vertices.extend((0..vertex_count).map(|index| {
            let cleaned_joints = joint_indices_or_zero(joints[index], weights[index]);
            SourceVertex {
                position: positions[index],
                normal: normals
                    .as_ref()
                    .map(|normals| normals[index])
                    .unwrap_or([0.0, 1.0, 0.0]),
                uv: uvs[index],
                joints: cleaned_joints,
                weights: weights[index],
                dominant_joint: dominant_vertex_joint(cleaned_joints, weights[index]),
            }
        }));

        let local_indices: Vec<u32> = if let Some(indices) = reader.read_indices() {
            indices.into_u32().collect()
        } else {
            (0..vertex_count as u32).collect()
        };
        for face in triangulate_indices(&local_indices, primitive.mode())? {
            let a = checked_index(face[0], vertex_count)? + base;
            let b = checked_index(face[1], vertex_count)? + base;
            let c = checked_index(face[2], vertex_count)? + base;
            normal_faces.push([a, b, c]);
            // glTF uses CCW front faces. The engine/GTE render
            // path culls by positive screen-space NCLIP after the
            // PS1-style Y projection flip, so cook imported models
            // into that convention once instead of making every
            // runtime submitter special-case glTF winding.
            source.faces.push(SourceFace {
                indices: [a, c, b],
                joint: 0,
            });
        }
    }
    // Normals stay in the source surface convention: only the cooked
    // packet index order is converted for GTE/NCLIP culling.
    rebuild_source_normals(&mut source, &normal_faces);
    Ok(source)
}

fn rebuild_source_normals(source: &mut SkinnedSourceMesh, normal_faces: &[[usize; 3]]) {
    let mut normals = vec![[0.0f32; 3]; source.vertices.len()];
    for face in normal_faces {
        let a = source.vertices[face[0]].position;
        let b = source.vertices[face[1]].position;
        let c = source.vertices[face[2]].position;
        let normal = cross3(sub3(b, a), sub3(c, a));
        if length_sq3(normal) <= 0.000001 {
            continue;
        }
        for index in *face {
            normals[index][0] += normal[0];
            normals[index][1] += normal[1];
            normals[index][2] += normal[2];
        }
    }

    for (vertex, normal) in source.vertices.iter_mut().zip(normals) {
        if length_sq3(normal) > 0.000001 {
            vertex.normal = normalize3(normal);
        } else {
            vertex.normal = normalize3(vertex.normal);
        }
    }
}

fn joint_indices_or_zero(joints: [u16; 4], weights: [f32; 4]) -> [u16; 4] {
    let mut out = joints;
    for i in 0..4 {
        if weights[i] <= 0.0 {
            out[i] = 0;
        }
    }
    out
}

fn dominant_vertex_joint(joints: [u16; 4], weights: [f32; 4]) -> u16 {
    let mut best = 0u16;
    let mut best_weight = f32::NEG_INFINITY;
    for influence in 0..4 {
        let weight = weights[influence];
        if weight > best_weight {
            best_weight = weight;
            best = joints[influence];
        }
    }
    best
}

/// Group faces by per-vertex bone choice. Each vertex has already
/// picked its dominant bone in `read_skinned_mesh`. A face is bound
/// to whichever bone owns the **majority** of its three corners.
///
/// On a 2-1 split the majority vertex pair wins, leaving the third
/// corner as the only "pulled" vertex bound to a foreign bone — much
/// less visible than the previous face-level binding, which could
/// pull all three corners together. On a 3-way disagreement we fall
/// back to summed-weight scoring so the choice still reflects the
/// face's overall bias.
fn assign_face_joints(source: &mut SkinnedSourceMesh, joint_count: usize) {
    let mut scores = vec![0.0f32; joint_count];
    for face in &mut source.faces {
        let bones = [
            source.vertices[face.indices[0]].dominant_joint,
            source.vertices[face.indices[1]].dominant_joint,
            source.vertices[face.indices[2]].dominant_joint,
        ];

        if bones[0] == bones[1] || bones[0] == bones[2] {
            face.joint = bones[0];
            continue;
        }
        if bones[1] == bones[2] {
            face.joint = bones[1];
            continue;
        }

        scores.fill(0.0);
        for vertex_index in face.indices {
            let vertex = source.vertices[vertex_index];
            for influence in 0..4 {
                let weight = vertex.weights[influence].max(0.0);
                if weight > 0.0 {
                    let joint = vertex.joints[influence] as usize;
                    if joint < scores.len() {
                        scores[joint] += weight;
                    }
                }
            }
        }
        let mut best_joint = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (joint, score) in scores.iter().copied().enumerate() {
            if score > best_score {
                best_joint = joint;
                best_score = score;
            }
        }
        face.joint = best_joint as u16;
    }
}

fn read_inverse_bind_matrices(
    skin: &gltf::Skin<'_>,
    buffers: &[gltf::buffer::Data],
    joint_count: usize,
) -> Vec<[[f32; 4]; 4]> {
    let reader = skin.reader(|buffer| Some(buffers[buffer.index()].0.as_slice()));
    reader
        .read_inverse_bind_matrices()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| vec![identity_matrix(); joint_count])
}

fn build_parent_indices(document: &gltf::Document) -> Vec<Option<usize>> {
    let mut parents = vec![None; document.nodes().count()];
    for node in document.nodes() {
        for child in node.children() {
            parents[child.index()] = Some(node.index());
        }
    }
    parents
}

fn collect_base_trs(document: &gltf::Document) -> Vec<Trs> {
    document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            Trs {
                translation,
                rotation,
                scale,
            }
        })
        .collect()
}

fn compute_global_matrices(
    parents: &[Option<usize>],
    locals: &[[[f32; 4]; 4]],
) -> Vec<[[f32; 4]; 4]> {
    let mut globals = vec![identity_matrix(); locals.len()];
    let mut done = vec![false; locals.len()];
    for index in 0..locals.len() {
        compute_global_matrix(index, parents, locals, &mut globals, &mut done);
    }
    globals
}

fn compute_global_matrix(
    index: usize,
    parents: &[Option<usize>],
    locals: &[[[f32; 4]; 4]],
    globals: &mut [[[f32; 4]; 4]],
    done: &mut [bool],
) -> [[f32; 4]; 4] {
    if done[index] {
        return globals[index];
    }
    let global = if let Some(parent) = parents[index] {
        let parent_global = compute_global_matrix(parent, parents, locals, globals, done);
        mul_matrix(&parent_global, &locals[index])
    } else {
        locals[index]
    };
    globals[index] = global;
    done[index] = true;
    global
}

#[allow(clippy::too_many_arguments)]
fn cook_model_blob(
    source: &SkinnedSourceMesh,
    bounds: &ModelBounds,
    parents: &[Option<usize>],
    joints: &[usize],
    material_color: [u8; 4],
    texture_width: u16,
    texture_height: u16,
    local_to_world_q12: u16,
) -> Result<(Vec<u8>, usize, usize), Error> {
    let mut node_to_joint = vec![None; parents.len()];
    for (joint_index, node_index) in joints.iter().copied().enumerate() {
        node_to_joint[node_index] = Some(joint_index as u16);
    }

    let mut grouped_faces: BTreeMap<u16, Vec<[usize; 3]>> = BTreeMap::new();
    for face in &source.faces {
        grouped_faces
            .entry(face.joint)
            .or_default()
            .push(face.indices);
    }

    let mut joint_records = Vec::new();
    for node_index in joints.iter().copied() {
        let parent = parents[node_index]
            .and_then(|parent| node_to_joint[parent])
            .unwrap_or(psxed_format::model::NO_JOINT);
        joint_records.push(parent);
    }

    let mut part_records = Vec::new();
    let mut cooked_vertices: Vec<[u8; psxed_format::model::VERTEX_RECORD_SIZE]> = Vec::new();
    let mut cooked_faces: Vec<[u16; 3]> = Vec::new();
    let mut blend_skin = false;

    for (joint_index, faces) in grouped_faces {
        let first_vertex = cooked_vertices.len();
        let first_face = cooked_faces.len();
        let mut local_map = BTreeMap::new();

        for face in faces {
            let mut out_face = [0u16; 3];
            for corner in 0..3 {
                let source_index = face[corner];
                let cooked_index = if let Some(index) = local_map.get(&source_index) {
                    *index
                } else {
                    let vertex = source.vertices[source_index];
                    let index = ensure_u16("vertices", cooked_vertices.len())?;
                    let record = encode_model_vertex(
                        vertex,
                        joint_index,
                        bounds,
                        texture_width,
                        texture_height,
                    );
                    if record[15] != 0 {
                        blend_skin = true;
                    }
                    cooked_vertices.push(record);
                    local_map.insert(source_index, index);
                    index
                };
                out_face[corner] = cooked_index;
            }
            cooked_faces.push(out_face);
        }

        let vertex_count = cooked_vertices.len() - first_vertex;
        let face_count = cooked_faces.len() - first_face;
        part_records.push((
            joint_index,
            ensure_u16("part first vertex", first_vertex)?,
            ensure_u16("part vertices", vertex_count)?,
            ensure_u16("part first face", first_face)?,
            ensure_u16("part faces", face_count)?,
            0u16,
        ));
    }

    ensure_u16("vertices", cooked_vertices.len())?;
    ensure_u16("faces", cooked_faces.len())?;
    ensure_u16("parts", part_records.len())?;

    let payload_len = psxed_format::model::ModelHeader::SIZE
        + joint_records.len() * psxed_format::model::JointRecord::SIZE
        + psxed_format::model::MaterialRecord::SIZE
        + part_records.len() * psxed_format::model::PartRecord::SIZE
        + cooked_vertices.len() * psxed_format::model::VERTEX_RECORD_SIZE
        + cooked_faces.len() * psxed_format::model::FACE_RECORD_SIZE;
    let mut out = Vec::with_capacity(psxed_format::AssetHeader::SIZE + payload_len);
    let mut model_flags = psxed_format::model::flags::HAS_NORMALS
        | psxed_format::model::flags::HAS_UVS
        | psxed_format::model::flags::RIGID_SKINNED;
    if blend_skin {
        model_flags |= psxed_format::model::flags::BLEND_SKIN;
    }
    append_asset_header(
        &mut out,
        psxed_format::model::MAGIC,
        psxed_format::model::VERSION,
        model_flags,
        payload_len,
    )?;
    append_u16(&mut out, ensure_u16("joints", joint_records.len())?);
    append_u16(&mut out, ensure_u16("parts", part_records.len())?);
    append_u16(&mut out, ensure_u16("vertices", cooked_vertices.len())?);
    append_u16(&mut out, ensure_u16("faces", cooked_faces.len())?);
    append_u16(&mut out, 1);
    append_u16(&mut out, texture_width);
    append_u16(&mut out, texture_height);
    append_u16(&mut out, local_to_world_q12);

    for parent in joint_records {
        append_u16(&mut out, parent);
        append_u16(&mut out, 0);
    }
    append_u16(&mut out, 0);
    append_u16(&mut out, 0);
    out.extend_from_slice(&material_color);

    for (joint, first_vertex, vertex_count, first_face, face_count, material_index) in &part_records
    {
        append_u16(&mut out, *joint);
        append_u16(&mut out, *first_vertex);
        append_u16(&mut out, *vertex_count);
        append_u16(&mut out, *first_face);
        append_u16(&mut out, *face_count);
        append_u16(&mut out, *material_index);
        append_u32(&mut out, 0);
    }
    for vertex in &cooked_vertices {
        out.extend_from_slice(vertex);
    }
    for face in &cooked_faces {
        append_u16(&mut out, face[0]);
        append_u16(&mut out, face[1]);
        append_u16(&mut out, face[2]);
    }

    Ok((out, cooked_vertices.len(), part_records.len()))
}

fn encode_model_vertex(
    vertex: SourceVertex,
    primary_joint: u16,
    bounds: &ModelBounds,
    texture_width: u16,
    texture_height: u16,
) -> [u8; psxed_format::model::VERTEX_RECORD_SIZE] {
    let position = bounds.normalize_point(vertex.position);
    let normal = normalize3(vertex.normal);
    let u = uv_to_u8(vertex.uv[0], texture_width);
    let v = uv_to_u8(vertex.uv[1], texture_height);
    let (joint1_byte, blend_byte) = blend_slot_for_vertex(vertex, primary_joint);
    let mut out = [0u8; psxed_format::model::VERTEX_RECORD_SIZE];
    write_i16(&mut out, 0, q12_i16(position[0]));
    write_i16(&mut out, 2, q12_i16(position[1]));
    write_i16(&mut out, 4, q12_i16(position[2]));
    write_i16(&mut out, 6, q12_i16(normal[0]));
    write_i16(&mut out, 8, q12_i16(normal[1]));
    write_i16(&mut out, 10, q12_i16(normal[2]));
    out[12] = u;
    out[13] = v;
    out[14] = joint1_byte;
    out[15] = blend_byte;
    out
}

/// Threshold below which a secondary bone's relative weight is dropped.
///
/// `weight1 / (weight0 + weight1)` smaller than this contributes a
/// blend so subtle the runtime CPU path costs more than the visual
/// difference — better to stay on the single-bone GTE fast path.
const BLEND_DROP_THRESHOLD: f32 = 0.04;

/// Pick the secondary bone + blend byte for a vertex, given the part
/// it ended up in.
///
/// `joint0` is implicit at runtime (it is the part's bone), so we only
/// store `joint1` and a relative weight in 0..=255. The secondary bone
/// is the strongest of:
///
/// * the vertex's own dominant bone, when the part's bone differs (a
///   "pulled" vertex on the wrong side of a seam — pulling it back
///   toward its natural bone removes the gap);
/// * otherwise the next-highest-weight bone after `primary_joint`.
fn blend_slot_for_vertex(vertex: SourceVertex, primary_joint: u16) -> (u8, u8) {
    let mut weight0 = 0.0f32;
    let mut weight1 = 0.0f32;
    let mut joint1: Option<u16> = None;

    if vertex.dominant_joint != primary_joint && vertex.weights[0] > 0.0 {
        joint1 = Some(vertex.dominant_joint);
    }

    for i in 0..4 {
        let w = vertex.weights[i].max(0.0);
        if w == 0.0 {
            continue;
        }
        let j = vertex.joints[i];
        if j == primary_joint {
            weight0 += w;
        } else if Some(j) == joint1 {
            weight1 += w;
        } else if joint1.is_none() {
            joint1 = Some(j);
            weight1 = w;
        } else if w > weight1 && Some(j) != joint1 {
            // Replace joint1 only if we have not committed it to the
            // pulled-vertex case above. The first branch already set
            // `joint1` to the natural bone in that case.
            if vertex.dominant_joint == primary_joint {
                joint1 = Some(j);
                weight1 = w;
            }
        }
    }

    let Some(j1) = joint1 else {
        return (psxed_format::model::NO_JOINT8, 0);
    };
    let total = weight0 + weight1;
    if total <= 0.0 {
        return (psxed_format::model::NO_JOINT8, 0);
    }
    let blend = weight1 / total;
    if blend < BLEND_DROP_THRESHOLD {
        return (psxed_format::model::NO_JOINT8, 0);
    }

    let blend_byte = (blend * 255.0).round().clamp(0.0, 255.0) as u8;
    if blend_byte == 0 {
        return (psxed_format::model::NO_JOINT8, 0);
    }
    let joint1_byte = if j1 < 255 {
        j1 as u8
    } else {
        psxed_format::model::NO_JOINT8
    };
    if joint1_byte == psxed_format::model::NO_JOINT8 {
        return (psxed_format::model::NO_JOINT8, 0);
    }
    (joint1_byte, blend_byte)
}

fn cook_base_color_texture(
    mesh: &gltf::Mesh<'_>,
    buffers: &[gltf::buffer::Data],
    cfg: &RigidModelConfig,
) -> Result<Option<Vec<u8>>, Error> {
    for primitive in mesh.primitives() {
        let Some(info) = primitive
            .material()
            .pbr_metallic_roughness()
            .base_color_texture()
        else {
            continue;
        };
        let source = info.texture().source();
        let Source::View { view, .. } = source.source() else {
            return Err(Error::UnsupportedImageSource);
        };
        let buffer = &buffers[view.buffer().index()].0;
        let start = view.offset();
        let end = start + view.length();
        let bytes = buffer
            .get(start..end)
            .ok_or(Error::UnsupportedImageSource)?;
        let tex_cfg = psxed_tex::Config {
            width: cfg.texture_width,
            height: cfg.texture_height,
            depth: cfg.texture_depth,
            crop: psxed_tex::CropMode::None,
            resampler: psxed_tex::Resampler::Lanczos3,
        };
        return psxed_tex::convert(bytes, &tex_cfg)
            .map(Some)
            .map_err(Error::TextureCook);
    }
    Ok(None)
}

fn first_material_base_color(mesh: &gltf::Mesh<'_>) -> [u8; 4] {
    if let Some(primitive) = mesh.primitives().next() {
        let color = primitive
            .material()
            .pbr_metallic_roughness()
            .base_color_factor();
        [
            linear_to_u8(color[0]),
            linear_to_u8(color[1]),
            linear_to_u8(color[2]),
            linear_to_u8(color[3]),
        ]
    } else {
        [255, 255, 255, 255]
    }
}

#[allow(clippy::too_many_arguments)]
fn cook_all_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    parents: &[Option<usize>],
    base_trs: &[Trs],
    joints: &[usize],
    inverse_bind_matrices: &[[[f32; 4]; 4]],
    bounds: &ModelBounds,
    fps: u16,
) -> Result<Vec<CookedClip>, Error> {
    let mut clips = Vec::new();
    for (index, animation) in document.animations().enumerate() {
        let channels = read_animation_channels(&animation, buffers)?;
        if channels.is_empty() {
            continue;
        }
        let Some((min_time, max_time)) = channel_time_range(&channels) else {
            continue;
        };
        let Some(bytes) = cook_animation_bytes(
            &channels,
            parents,
            base_trs,
            joints,
            inverse_bind_matrices,
            bounds,
            min_time,
            max_time,
            fps,
        )?
        else {
            continue;
        };
        let frames = animation_frame_count_from_bytes(&bytes);
        let raw_name = animation.name().map(|s| s.to_string());
        clips.push(CookedClip {
            source_name: raw_name.clone(),
            sanitized_name: sanitize_clip_name(raw_name.as_deref(), index),
            bytes,
            frames,
        });
    }
    Ok(clips)
}

#[allow(clippy::too_many_arguments)]
fn cook_animation_bytes(
    channels: &[AnimationChannel],
    parents: &[Option<usize>],
    base_trs: &[Trs],
    joints: &[usize],
    inverse_bind_matrices: &[[[f32; 4]; 4]],
    bounds: &ModelBounds,
    min_time: f32,
    max_time: f32,
    fps: u16,
) -> Result<Option<Vec<u8>>, Error> {
    let duration = max_time - min_time;
    let frame_count = (duration * fps as f32).round() as usize + 1;
    ensure_u16("animation frames", frame_count)?;

    let payload_len = psxed_format::animation::AnimationHeader::SIZE
        + frame_count * joints.len() * psxed_format::animation::POSE_RECORD_SIZE;
    let mut out = Vec::with_capacity(psxed_format::AssetHeader::SIZE + payload_len);
    append_asset_header(
        &mut out,
        psxed_format::animation::MAGIC,
        psxed_format::animation::VERSION,
        0,
        payload_len,
    )?;
    append_u16(&mut out, ensure_u16("joints", joints.len())?);
    append_u16(&mut out, ensure_u16("animation frames", frame_count)?);
    append_u16(&mut out, fps);
    append_u16(&mut out, 0);

    for frame in 0..frame_count {
        let time = (min_time + frame as f32 / fps as f32).min(max_time);
        let mut frame_trs = base_trs.to_vec();
        for channel in channels {
            channel.apply(time, &mut frame_trs);
        }
        let locals: Vec<[[f32; 4]; 4]> = frame_trs.iter().map(|trs| trs.matrix()).collect();
        let globals = compute_global_matrices(parents, &locals);
        for (joint_index, node_index) in joints.iter().copied().enumerate() {
            let skin = mul_matrix(&globals[node_index], &inverse_bind_matrices[joint_index]);
            append_pose_record(&mut out, &skin, bounds);
        }
    }

    Ok(Some(out))
}

fn sanitize_clip_name(source: Option<&str>, fallback_index: usize) -> String {
    let raw = source.unwrap_or("");
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        format!("clip{fallback_index}")
    } else {
        trimmed
    }
}

#[derive(Clone, Debug)]
struct AnimationChannel {
    node_index: usize,
    interpolation: Interpolation,
    times: Vec<f32>,
    values: ChannelValues,
}

#[derive(Clone, Debug)]
enum ChannelValues {
    Translation(Vec<[f32; 3]>),
    Rotation(Vec<[f32; 4]>),
    Scale(Vec<[f32; 3]>),
}

impl AnimationChannel {
    fn apply(&self, time: f32, trs: &mut [Trs]) {
        let Some(target) = trs.get_mut(self.node_index) else {
            return;
        };
        match &self.values {
            ChannelValues::Translation(values) => {
                target.translation = sample_vec3(&self.times, values, time, self.interpolation);
            }
            ChannelValues::Rotation(values) => {
                target.rotation = sample_quat(&self.times, values, time, self.interpolation);
            }
            ChannelValues::Scale(values) => {
                target.scale = sample_vec3(&self.times, values, time, self.interpolation);
            }
        }
    }
}

fn read_animation_channels(
    animation: &gltf::Animation<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<AnimationChannel>, Error> {
    let mut channels = Vec::new();
    for channel in animation.channels() {
        let channel_index = channel.index();
        let interpolation = channel.sampler().interpolation();
        if interpolation == Interpolation::CubicSpline {
            return Err(Error::UnsupportedAnimationInterpolation {
                channel_index,
                interpolation,
            });
        }
        let reader = channel.reader(|buffer| Some(buffers[buffer.index()].0.as_slice()));
        let times: Vec<f32> = reader
            .read_inputs()
            .ok_or(Error::MissingAnimationInputs { channel_index })?
            .collect();
        let outputs = reader
            .read_outputs()
            .ok_or(Error::MissingAnimationOutputs { channel_index })?;
        let values = match (channel.target().property(), outputs) {
            (Property::Translation, gltf::animation::util::ReadOutputs::Translations(values)) => {
                ChannelValues::Translation(values.collect())
            }
            (Property::Rotation, gltf::animation::util::ReadOutputs::Rotations(values)) => {
                ChannelValues::Rotation(values.into_f32().collect())
            }
            (Property::Scale, gltf::animation::util::ReadOutputs::Scales(values)) => {
                ChannelValues::Scale(values.collect())
            }
            (Property::MorphTargetWeights, _) => continue,
            _ => return Err(Error::AnimationTypeMismatch { channel_index }),
        };
        channels.push(AnimationChannel {
            node_index: channel.target().node().index(),
            interpolation,
            times,
            values,
        });
    }
    Ok(channels)
}

fn sample_vec3(
    times: &[f32],
    values: &[[f32; 3]],
    time: f32,
    interpolation: Interpolation,
) -> [f32; 3] {
    let (a, b, t) = sample_segment(times, time);
    if interpolation == Interpolation::Step || a == b {
        return values[a];
    }
    lerp3(values[a], values[b], t)
}

fn sample_quat(
    times: &[f32],
    values: &[[f32; 4]],
    time: f32,
    interpolation: Interpolation,
) -> [f32; 4] {
    let (a, b, t) = sample_segment(times, time);
    if interpolation == Interpolation::Step || a == b {
        return normalize4(values[a]);
    }
    nlerp_quat(values[a], values[b], t)
}

fn sample_segment(times: &[f32], time: f32) -> (usize, usize, f32) {
    if times.len() <= 1 || time <= times[0] {
        return (0, 0, 0.0);
    }
    let last = times.len() - 1;
    if time >= times[last] {
        return (last, last, 0.0);
    }
    for index in 0..last {
        let t0 = times[index];
        let t1 = times[index + 1];
        if time >= t0 && time <= t1 {
            let span = (t1 - t0).max(0.000001);
            return (index, index + 1, ((time - t0) / span).clamp(0.0, 1.0));
        }
    }
    (last, last, 0.0)
}

fn compose_trs(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> [[f32; 4]; 4] {
    let [x, y, z, w] = normalize4(rotation);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    let r00 = 1.0 - 2.0 * (yy + zz);
    let r01 = 2.0 * (xy - wz);
    let r02 = 2.0 * (xz + wy);
    let r10 = 2.0 * (xy + wz);
    let r11 = 1.0 - 2.0 * (xx + zz);
    let r12 = 2.0 * (yz - wx);
    let r20 = 2.0 * (xz - wy);
    let r21 = 2.0 * (yz + wx);
    let r22 = 1.0 - 2.0 * (xx + yy);

    [
        [r00 * scale[0], r10 * scale[0], r20 * scale[0], 0.0],
        [r01 * scale[1], r11 * scale[1], r21 * scale[1], 0.0],
        [r02 * scale[2], r12 * scale[2], r22 * scale[2], 0.0],
        [translation[0], translation[1], translation[2], 1.0],
    ]
}

fn append_pose_record(out: &mut Vec<u8>, skin_matrix: &[[f32; 4]; 4], bounds: &ModelBounds) {
    for column in skin_matrix.iter().take(3) {
        for value in column.iter().take(3) {
            append_i16(out, q12_i16(*value));
        }
    }
    let center_in_pose = transform_point(skin_matrix, bounds.center);
    let translation = [
        (center_in_pose[0] - bounds.center[0]) / bounds.extent,
        (center_in_pose[1] - bounds.center[1]) / bounds.extent,
        (center_in_pose[2] - bounds.center[2]) / bounds.extent,
    ];
    append_i32(out, q12_i32(translation[0]));
    append_i32(out, q12_i32(translation[1]));
    append_i32(out, q12_i32(translation[2]));
}

fn animation_frame_count_from_bytes(bytes: &[u8]) -> usize {
    if bytes.len()
        < psxed_format::AssetHeader::SIZE + psxed_format::animation::AnimationHeader::SIZE
    {
        return 0;
    }
    u16::from_le_bytes([bytes[14], bytes[15]]) as usize
}

fn append_asset_header(
    out: &mut Vec<u8>,
    magic: [u8; 4],
    version: u16,
    flags: u16,
    payload_len: usize,
) -> Result<(), Error> {
    if payload_len > u32::MAX as usize {
        return Err(Error::TooMany {
            kind: "payload bytes",
            count: payload_len,
            max: u32::MAX as usize,
        });
    }
    out.extend_from_slice(&magic);
    append_u16(out, version);
    append_u16(out, flags);
    append_u32(out, payload_len as u32);
    Ok(())
}

fn append_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_i16(out: &mut [u8], offset: usize, value: i16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn ensure_u16(kind: &'static str, count: usize) -> Result<u16, Error> {
    if count > u16::MAX as usize {
        Err(Error::TooMany {
            kind,
            count,
            max: u16::MAX as usize,
        })
    } else {
        Ok(count as u16)
    }
}

fn q12_i16(value: f32) -> i16 {
    (value * 4096.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn q12_i32(value: f32) -> i32 {
    (value * 4096.0)
        .round()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

fn uv_to_u8(value: f32, size: u16) -> u8 {
    let max_coord = size.saturating_sub(1).min(255) as f32;
    (value.clamp(0.0, 1.0) * max_coord)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length_sq3(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn nlerp_quat(mut a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if dot < 0.0 {
        a = [-a[0], -a[1], -a[2], -a[3]];
    }
    normalize4([
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ])
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 0.000001 {
        [0.0, 1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn normalize4(v: [f32; 4]) -> [f32; 4] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    if len <= 0.000001 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len, v[3] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_minimal_glb_triangle() {
        let glb = minimal_triangle_glb();
        let psxm = convert_slice(&glb, &Config::default()).unwrap();
        let mesh = psx_asset::Mesh::from_bytes(&psxm).unwrap();
        assert_eq!(mesh.vert_count(), 3);
        assert_eq!(mesh.face_count(), 1);
        assert!(mesh.has_face_colors());
        assert!(mesh.has_normals());
        assert_eq!(mesh.face_color(0), Some((64, 128, 255)));
    }

    #[test]
    fn triangle_strip_gets_triangulated() {
        let faces = triangulate_indices(&[0, 1, 2, 3], Mode::TriangleStrip).unwrap();
        assert_eq!(faces, vec![[0, 1, 2], [2, 1, 3]]);
    }

    #[test]
    fn model_precision_scale_targets_world_height() {
        let bounds = ModelBounds::from_min_max([0.0, 0.0, 0.0], [2.0, 4.0, 1.0], 30_000.0).unwrap();
        let local_height = bounds.encoded_axis_size(0.0, 4.0);
        assert_eq!(local_height, 60_000);
        assert_eq!(choose_local_to_world_q12(local_height, 1024), 70);
    }

    #[test]
    fn native_model_normals_use_source_winding_after_engine_face_flip() {
        let mut source = SkinnedSourceMesh {
            vertices: vec![
                test_source_vertex([0.0, 0.0, 0.0]),
                test_source_vertex([1.0, 0.0, 0.0]),
                test_source_vertex([0.0, 1.0, 0.0]),
            ],
            faces: vec![SourceFace {
                indices: [0, 2, 1],
                joint: 0,
            }],
        };

        rebuild_source_normals(&mut source, &[[0, 1, 2]]);

        assert_eq!(source.faces[0].indices, [0, 2, 1]);
        for vertex in &source.vertices {
            assert!((vertex.normal[0] - 0.0).abs() < 0.0001);
            assert!((vertex.normal[1] - 0.0).abs() < 0.0001);
            assert!((vertex.normal[2] - 1.0).abs() < 0.0001);
        }
    }

    fn minimal_triangle_glb() -> Vec<u8> {
        let mut bin = Vec::new();
        for f in [
            0.0f32, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ] {
            bin.extend_from_slice(&f.to_le_bytes());
        }
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }

        let json = format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0}}],
  "buffers": [{{"byteLength": {}}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
     "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]}},
    {{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}}
  ],
  "materials": [
    {{"pbrMetallicRoughness": {{"baseColorFactor": [0.25, 0.5, 1.0, 1.0]}}}}
  ],
  "meshes": [
    {{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1, "material": 0, "mode": 4}}]}}
  ]
}}"#,
            bin.len()
        );
        let json = padded(json.into_bytes(), b' ');
        let bin = padded(bin, 0);

        let total_len = 12 + 8 + json.len() + 8 + bin.len();
        let mut out = Vec::with_capacity(total_len);
        out.extend_from_slice(&0x4654_6C67u32.to_le_bytes()); // "glTF"
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total_len as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes()); // JSON
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E_4942u32.to_le_bytes()); // BIN
        out.extend_from_slice(&bin);
        out
    }

    fn padded(mut bytes: Vec<u8>, pad: u8) -> Vec<u8> {
        while !bytes.len().is_multiple_of(4) {
            bytes.push(pad);
        }
        bytes
    }

    fn test_source_vertex(position: [f32; 3]) -> SourceVertex {
        SourceVertex {
            position,
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
            dominant_joint: 0,
        }
    }
}
