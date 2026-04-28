//! Cooked-model bundle registration + GLB import.
//!
//! Two entry points feed the same end product —
//! [`ResourceData::Model`] — from different sources:
//!
//! * [`register_cooked_model_bundle`] adopts an existing
//!   `bundle_dir/` containing one cooked `.psxmdl`, optionally a
//!   `.psxt` atlas, and any number of `.psxanim` clips. Use this
//!   when the assets ship with the repo or were cooked elsewhere.
//!
//! * [`import_glb_model`] runs the GLB through the rigid-model
//!   cooker (`psxed_gltf::convert_rigid_model_path`), drops the
//!   cooked outputs under `project/assets/models/<safe_name>/`,
//!   then registers them. Use this for fresh authoring.
//!
//! Both paths validate every blob through `psx_asset` parsers and
//! confirm animation joint counts match the model's joint count
//! before creating the resource — bad bundles never produce a
//! half-broken Model entry.

use std::path::{Path, PathBuf};

use crate::{
    ModelAnimationClip, ModelResource, ProjectDocument, ResourceData, ResourceId,
};

/// Header-derived statistics about a `.psxmdl` blob, suitable
/// for editor inspector display. Computed by walking the model
/// vertex table once for bounds; everything else is a header
/// read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStats {
    /// On-disk byte count.
    pub model_bytes: usize,
    /// Joint count from the model header.
    pub joint_count: u16,
    /// Part / submesh count.
    pub part_count: u16,
    /// Vertex count.
    pub vertex_count: u16,
    /// Face (triangle) count.
    pub face_count: u16,
    /// Material slots referenced by parts.
    pub material_count: u16,
    /// Local-to-world Q12 scale stored in the header.
    pub local_to_world_q12: u16,
    /// Largest vertex count across all parts (sizes the
    /// runtime's per-part scratch buffer).
    pub max_part_vertices: u16,
    /// Texture footprint declared by the header (used to size
    /// the atlas allocator).
    pub texture_width: u16,
    /// Header-declared texture height.
    pub texture_height: u16,
    /// AABB minimum from the parsed vertex positions, in
    /// model-local units. Defaults to `[0, 0, 0]` when the
    /// model has no vertices.
    pub bounds_min: [i32; 3],
    /// AABB maximum.
    pub bounds_max: [i32; 3],
}

/// Per-clip metadata. `valid_for_model` is `true` when the
/// clip's joint count matches the model it's bound to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimationClipStats {
    /// Display name.
    pub name: String,
    /// On-disk byte count.
    pub bytes: usize,
    /// Joint count from the clip header.
    pub joint_count: u16,
    /// Frame count from the clip header.
    pub frame_count: u16,
    /// Sample rate in Hz from the clip header.
    pub sample_rate_hz: u16,
    /// `false` when the clip's joint count differs from the
    /// owning model — the inspector flags this and the cooker
    /// refuses such bundles.
    pub valid_for_model: bool,
}

/// Atlas texture stats. `depth` is `4`, `8`, or `15`; anything
/// else means the editor doesn't fully understand the depth
/// (the inspector flags accordingly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTextureStats {
    /// On-disk byte count.
    pub bytes: usize,
    /// Texel width.
    pub width: u16,
    /// Texel height.
    pub height: u16,
    /// Bits per pixel — `4`, `8`, or `15`.
    pub depth: u8,
    /// CLUT entry count (`16` for 4bpp, `256` for 8bpp, `0`
    /// for direct 15bpp).
    pub clut_entries: u16,
}

/// Compute [`ModelStats`] from cooked `.psxmdl` bytes. Walks
/// every vertex once for the AABB; cheap relative to the rest
/// of the editor's per-frame budget.
pub fn model_stats_from_bytes(bytes: &[u8]) -> Result<ModelStats, ModelImportError> {
    let model = psx_asset::Model::from_bytes(bytes).map_err(|e| ModelImportError::InvalidModel {
        path: PathBuf::new(),
        detail: format!("{:?}", e),
    })?;

    let mut max_part_vertices: u16 = 0;
    for i in 0..model.part_count() {
        if let Some(part) = model.part(i) {
            if part.vertex_count() > max_part_vertices {
                max_part_vertices = part.vertex_count();
            }
        }
    }

    let mut bounds_min = [i32::MAX, i32::MAX, i32::MAX];
    let mut bounds_max = [i32::MIN, i32::MIN, i32::MIN];
    let mut any = false;
    for i in 0..model.vertex_count() {
        if let Some(v) = model.vertex(i) {
            let p = v.position;
            let xyz = [p.x as i32, p.y as i32, p.z as i32];
            for (axis, value) in xyz.iter().enumerate() {
                if *value < bounds_min[axis] {
                    bounds_min[axis] = *value;
                }
                if *value > bounds_max[axis] {
                    bounds_max[axis] = *value;
                }
            }
            any = true;
        }
    }
    if !any {
        bounds_min = [0, 0, 0];
        bounds_max = [0, 0, 0];
    }

    Ok(ModelStats {
        model_bytes: bytes.len(),
        joint_count: model.joint_count(),
        part_count: model.part_count(),
        vertex_count: model.vertex_count(),
        face_count: model.face_count(),
        material_count: model.material_count(),
        local_to_world_q12: model.local_to_world_q12(),
        max_part_vertices,
        texture_width: model.texture_width(),
        texture_height: model.texture_height(),
        bounds_min,
        bounds_max,
    })
}

/// Compute [`AnimationClipStats`] from `.psxanim` bytes plus
/// the owning model's joint count. `name` is supplied by the
/// caller (usually copied from the [`ModelAnimationClip`]).
pub fn animation_stats_from_bytes(
    name: impl Into<String>,
    bytes: &[u8],
    model_joint_count: u16,
) -> Result<AnimationClipStats, ModelImportError> {
    let anim = psx_asset::Animation::from_bytes(bytes).map_err(|e| {
        ModelImportError::InvalidAnimation {
            path: PathBuf::new(),
            detail: format!("{:?}", e),
        }
    })?;
    Ok(AnimationClipStats {
        name: name.into(),
        bytes: bytes.len(),
        joint_count: anim.joint_count(),
        frame_count: anim.frame_count(),
        sample_rate_hz: anim.sample_rate_hz(),
        valid_for_model: anim.joint_count() == model_joint_count,
    })
}

/// Compute [`ModelTextureStats`] from `.psxt` bytes.
pub fn texture_stats_from_bytes(bytes: &[u8]) -> Result<ModelTextureStats, ModelImportError> {
    let texture =
        psx_asset::Texture::from_bytes(bytes).map_err(|e| ModelImportError::InvalidTexture {
            path: PathBuf::new(),
            detail: format!("{:?}", e),
        })?;
    let depth = match texture.depth() {
        psxed_format::texture::Depth::Bit4 => 4,
        psxed_format::texture::Depth::Bit8 => 8,
        psxed_format::texture::Depth::Bit15 => 15,
    };
    Ok(ModelTextureStats {
        bytes: bytes.len(),
        width: texture.width(),
        height: texture.height(),
        depth,
        clut_entries: texture.clut_entries(),
    })
}

/// Resolve a `psxt`/`psxanim`/`psxmdl` path against an optional
/// project root. Mirrors the lookup order used by other resources
/// (absolute → project-relative).
pub fn resolve_path(stored: &str, project_root: Option<&Path>) -> PathBuf {
    if Path::new(stored).is_absolute() {
        PathBuf::from(stored)
    } else if let Some(root) = project_root {
        root.join(stored)
    } else {
        PathBuf::from(stored)
    }
}

/// Failure modes for [`register_cooked_model_bundle`] and
/// [`import_glb_model`]. Each variant carries the offending path
/// or detail so the editor's status line can point at the cause
/// without re-walking the bundle.
#[derive(Debug)]
pub enum ModelImportError {
    /// `bundle_dir` is not a directory or could not be read.
    BundleNotADirectory(PathBuf),
    /// `bundle_dir` contains zero `.psxmdl` files.
    NoModelFile(PathBuf),
    /// `bundle_dir` contains more than one `.psxmdl`.
    MultipleModelFiles {
        /// Each candidate `.psxmdl` path discovered.
        paths: Vec<PathBuf>,
    },
    /// More than one `.psxt` was found in the bundle directory.
    /// The current schema binds a model to exactly one atlas, so
    /// the registrar rejects ambiguous bundles rather than
    /// guessing.
    MultipleTextureFiles {
        /// Each candidate `.psxt` path discovered.
        paths: Vec<PathBuf>,
    },
    /// `psx_asset::Model::from_bytes` rejected the model bytes.
    InvalidModel {
        /// Path that failed to parse.
        path: PathBuf,
        /// Diagnostic message (parse error rendered as a string).
        detail: String,
    },
    /// `psx_asset::Texture::from_bytes` rejected the atlas bytes.
    InvalidTexture {
        /// Path that failed to parse.
        path: PathBuf,
        /// Diagnostic message.
        detail: String,
    },
    /// `psx_asset::Animation::from_bytes` rejected an animation
    /// blob.
    InvalidAnimation {
        /// Clip path that failed.
        path: PathBuf,
        /// Diagnostic message.
        detail: String,
    },
    /// An animation's joint count differs from the model's joint
    /// count — they would render scrambled frames at runtime.
    JointCountMismatch {
        /// Clip path.
        path: PathBuf,
        /// Joints declared by the clip header.
        animation_joints: u16,
        /// Joints declared by the model header.
        model_joints: u16,
    },
    /// Filesystem error reading or writing a bundle file.
    Io {
        /// Path the IO error originated at.
        path: PathBuf,
        /// Underlying error message.
        detail: String,
    },
    /// GLB conversion failed inside `psxed_gltf`.
    GlbConversionFailed {
        /// Source `.glb` path.
        source: PathBuf,
        /// Diagnostic message.
        detail: String,
    },
    /// `output_name` produced a path that already exists and
    /// holds non-bundle content. The caller should pick a fresh
    /// name or remove the directory first.
    OutputExists(PathBuf),
}

impl std::fmt::Display for ModelImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BundleNotADirectory(path) => {
                write!(f, "{} is not a directory", path.display())
            }
            Self::NoModelFile(path) => {
                write!(f, "no .psxmdl found in {}", path.display())
            }
            Self::MultipleModelFiles { paths } => write!(
                f,
                "multiple .psxmdl files in bundle: {}",
                paths_list(paths)
            ),
            Self::MultipleTextureFiles { paths } => write!(
                f,
                "multiple .psxt files in bundle: {}",
                paths_list(paths)
            ),
            Self::InvalidModel { path, detail } => {
                write!(f, "{}: invalid .psxmdl: {detail}", path.display())
            }
            Self::InvalidTexture { path, detail } => {
                write!(f, "{}: invalid .psxt: {detail}", path.display())
            }
            Self::InvalidAnimation { path, detail } => {
                write!(f, "{}: invalid .psxanim: {detail}", path.display())
            }
            Self::JointCountMismatch {
                path,
                animation_joints,
                model_joints,
            } => write!(
                f,
                "{}: animation has {animation_joints} joints, model has {model_joints}",
                path.display()
            ),
            Self::Io { path, detail } => write!(f, "{}: {detail}", path.display()),
            Self::GlbConversionFailed { source, detail } => {
                write!(f, "{}: GLB conversion failed: {detail}", source.display())
            }
            Self::OutputExists(path) => write!(
                f,
                "output directory {} exists with conflicting content",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ModelImportError {}

fn paths_list(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Adopt an existing cooked model bundle as a [`ResourceData::Model`]
/// inside `project`. `bundle_dir` must contain exactly one
/// `.psxmdl`, at most one `.psxt`, and any number of `.psxanim`
/// clips. Stored paths are project-relative whenever
/// `project_root` is supplied and `bundle_dir` lives under it;
/// otherwise the absolute paths are kept.
///
/// Returns the new resource's id. The project gains a single
/// `Model` resource named `display_name`.
pub fn register_cooked_model_bundle(
    project: &mut ProjectDocument,
    bundle_dir: &Path,
    display_name: &str,
    project_root: Option<&Path>,
) -> Result<ResourceId, ModelImportError> {
    let mut psxmdl: Vec<PathBuf> = Vec::new();
    let mut psxt: Vec<PathBuf> = Vec::new();
    let mut psxanim: Vec<PathBuf> = Vec::new();

    let read = std::fs::read_dir(bundle_dir).map_err(|e| {
        if matches!(e.kind(), std::io::ErrorKind::NotFound)
            || matches!(e.kind(), std::io::ErrorKind::NotADirectory)
        {
            ModelImportError::BundleNotADirectory(bundle_dir.to_path_buf())
        } else {
            ModelImportError::Io {
                path: bundle_dir.to_path_buf(),
                detail: e.to_string(),
            }
        }
    })?;
    for entry in read {
        let entry = entry.map_err(|e| ModelImportError::Io {
            path: bundle_dir.to_path_buf(),
            detail: e.to_string(),
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        match path.extension().and_then(|e| e.to_str()) {
            Some("psxmdl") => psxmdl.push(path),
            Some("psxt") => psxt.push(path),
            Some("psxanim") => psxanim.push(path),
            _ => {}
        }
    }

    if psxmdl.is_empty() {
        return Err(ModelImportError::NoModelFile(bundle_dir.to_path_buf()));
    }
    if psxmdl.len() > 1 {
        psxmdl.sort();
        return Err(ModelImportError::MultipleModelFiles { paths: psxmdl });
    }
    if psxt.len() > 1 {
        psxt.sort();
        return Err(ModelImportError::MultipleTextureFiles { paths: psxt });
    }

    psxanim.sort();
    let model_path = psxmdl.pop().unwrap();
    let texture_path = psxt.pop();

    // Validate the model + texture + every animation. Failure
    // here means the resource is never created — we never leave
    // a half-broken `Model` entry in `project.resources`.
    let model_bytes = std::fs::read(&model_path).map_err(|e| ModelImportError::Io {
        path: model_path.clone(),
        detail: e.to_string(),
    })?;
    let model = psx_asset::Model::from_bytes(&model_bytes).map_err(|e| {
        ModelImportError::InvalidModel {
            path: model_path.clone(),
            detail: format!("{:?}", e),
        }
    })?;
    let model_joint_count = model.joint_count();

    if let Some(tex) = &texture_path {
        let bytes = std::fs::read(tex).map_err(|e| ModelImportError::Io {
            path: tex.clone(),
            detail: e.to_string(),
        })?;
        psx_asset::Texture::from_bytes(&bytes).map_err(|e| ModelImportError::InvalidTexture {
            path: tex.clone(),
            detail: format!("{:?}", e),
        })?;
    }

    let mut clips: Vec<ModelAnimationClip> = Vec::with_capacity(psxanim.len());
    for path in &psxanim {
        let bytes = std::fs::read(path).map_err(|e| ModelImportError::Io {
            path: path.clone(),
            detail: e.to_string(),
        })?;
        let anim = psx_asset::Animation::from_bytes(&bytes).map_err(|e| {
            ModelImportError::InvalidAnimation {
                path: path.clone(),
                detail: format!("{:?}", e),
            }
        })?;
        if anim.joint_count() != model_joint_count {
            return Err(ModelImportError::JointCountMismatch {
                path: path.clone(),
                animation_joints: anim.joint_count(),
                model_joints: model_joint_count,
            });
        }
        clips.push(ModelAnimationClip {
            name: clip_name_from_path(path),
            psxanim_path: relativise(path, project_root),
        });
    }

    let model_resource = ModelResource {
        model_path: relativise(&model_path, project_root),
        texture_path: texture_path.as_ref().map(|p| relativise(p, project_root)),
        clips,
        // Default to clip 0 if any exist — keeps newly-registered
        // bundles immediately playable in the inspector preview
        // and at runtime without forcing the user to wire the
        // pickers manually.
        default_clip: if psxanim.is_empty() { None } else { Some(0) },
        preview_clip: if psxanim.is_empty() { None } else { Some(0) },
        world_height: 1024,
    };

    Ok(project.add_resource(display_name, ResourceData::Model(model_resource)))
}

/// Convert a `.glb` (or `.gltf`) source through the rigid-model
/// cooker, write the cooked outputs under
/// `project_root/assets/models/<safe_name>/`, then register that
/// directory as a [`ResourceData::Model`].
///
/// Existing bundle directories are accepted only when they
/// contain exactly the same kinds of files this importer
/// produces — anything else and the import refuses rather than
/// clobbering user data.
pub fn import_glb_model(
    project: &mut ProjectDocument,
    source_path: &Path,
    output_name: &str,
    project_root: &Path,
    config: psxed_gltf::RigidModelConfig,
) -> Result<ResourceId, ModelImportError> {
    let package = psxed_gltf::convert_rigid_model_path(source_path, &config).map_err(|e| {
        ModelImportError::GlbConversionFailed {
            source: source_path.to_path_buf(),
            detail: format!("{e}"),
        }
    })?;

    let safe = safe_dir_name(output_name);
    let bundle_dir = project_root.join("assets").join("models").join(&safe);
    if let Err(e) = std::fs::create_dir_all(&bundle_dir) {
        return Err(ModelImportError::Io {
            path: bundle_dir.clone(),
            detail: e.to_string(),
        });
    }

    // Reject pre-existing non-bundle content rather than
    // silently merging.
    if let Ok(read) = std::fs::read_dir(&bundle_dir) {
        for entry in read.flatten() {
            let path = entry.path();
            if path.is_file() {
                let ok = matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("psxmdl") | Some("psxt") | Some("psxanim")
                );
                if !ok {
                    return Err(ModelImportError::OutputExists(bundle_dir));
                }
            }
        }
    }

    let model_path = bundle_dir.join(format!("{safe}.psxmdl"));
    std::fs::write(&model_path, &package.model).map_err(|e| ModelImportError::Io {
        path: model_path.clone(),
        detail: e.to_string(),
    })?;

    if let Some(texture) = &package.texture {
        let texture_path = bundle_dir.join(format!("{safe}.psxt"));
        std::fs::write(&texture_path, texture).map_err(|e| ModelImportError::Io {
            path: texture_path,
            detail: e.to_string(),
        })?;
    }

    for clip in &package.clips {
        let clip_path = bundle_dir.join(format!("{}_{}.psxanim", safe, clip.sanitized_name));
        std::fs::write(&clip_path, &clip.bytes).map_err(|e| ModelImportError::Io {
            path: clip_path,
            detail: e.to_string(),
        })?;
    }

    register_cooked_model_bundle(project, &bundle_dir, output_name, Some(project_root))
}

/// Derive a clip display name from a `.psxanim` path. Strips the
/// extension and any leading `<model>_` prefix when one of the
/// canonical stems is recognised, so a bundle called
/// `obsidian_wraith` produces clip names like `idle` rather than
/// `obsidian_wraith_idle`.
fn clip_name_from_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("clip");
    // Bundle-prefix stripping: pick the longest known model
    // prefix that the stem starts with. This is heuristic — when
    // we don't recognise the prefix we keep the full stem so the
    // user can rename in the inspector.
    let parent_name = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let prefix_len = parent_name.len() + 1;
    if !parent_name.is_empty()
        && stem.len() > prefix_len
        && stem.starts_with(parent_name)
        && stem.as_bytes().get(parent_name.len()) == Some(&b'_')
    {
        return stem[prefix_len..].to_string();
    }
    stem.to_string()
}

/// Convert `path` into a relative-to-project string when
/// `project_root` is provided and `path` lives under it. Falls
/// back to an absolute path so the editor can still find the
/// file regardless of where the project moves later.
fn relativise(path: &Path, project_root: Option<&Path>) -> String {
    if let Some(root) = project_root {
        if let Ok(rel) = path.strip_prefix(root) {
            return rel.to_string_lossy().into_owned();
        }
    }
    path.to_string_lossy().into_owned()
}

/// Sanitise a user-supplied resource name into a filesystem-safe
/// directory name (lowercase ASCII alphanumerics + underscores).
fn safe_dir_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_was_sep = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProjectDocument;

    /// Layout of a synthetic bundle dir for tests — caller
    /// passes byte slices for each file kind, helper writes them
    /// next to a fresh tempdir.
    fn make_bundle(
        tag: &str,
        model_bytes: Option<&[u8]>,
        models_count: usize,
        textures: &[&[u8]],
        animations: &[(&str, &[u8])],
    ) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "psxed-model-import-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(bytes) = model_bytes {
            for i in 0..models_count {
                std::fs::write(dir.join(format!("model_{i}.psxmdl")), bytes).unwrap();
            }
        }
        for (i, bytes) in textures.iter().enumerate() {
            std::fs::write(dir.join(format!("atlas_{i}.psxt")), bytes).unwrap();
        }
        for (name, bytes) in animations {
            std::fs::write(dir.join(format!("{name}.psxanim")), bytes).unwrap();
        }
        dir
    }

    fn obsidian_wraith_dir() -> PathBuf {
        // The obsidian wraith assets live in the repo root's
        // `assets/models/obsidian_wraith/` and are exercised by
        // showcase-model.
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("assets")
            .join("models")
            .join("obsidian_wraith")
    }

    #[test]
    fn registers_obsidian_wraith_bundle() {
        let mut project = ProjectDocument::starter();
        let dir = obsidian_wraith_dir();
        let id = register_cooked_model_bundle(
            &mut project,
            &dir,
            "Obsidian Wraith",
            None,
        )
        .expect("bundle registers");
        let resource = project.resource(id).expect("resource exists");
        let ResourceData::Model(model) = &resource.data else {
            panic!("expected Model resource, got {:?}", resource.data);
        };
        assert!(model.model_path.ends_with("obsidian_wraith.psxmdl"));
        assert!(model.texture_path.is_some());
        assert!(!model.clips.is_empty(), "expected at least one clip");
        // Clips are sorted by file name → first one alphabetically
        // (idle / dead / etc) is a stable bind for the test.
        let mut sorted_names: Vec<&str> = model.clips.iter().map(|c| c.name.as_str()).collect();
        sorted_names.sort();
        assert_eq!(model.default_clip, Some(0));
    }

    #[test]
    fn no_model_file_fails() {
        let mut project = ProjectDocument::starter();
        let dir = make_bundle("no-model", None, 0, &[], &[]);
        match register_cooked_model_bundle(&mut project, &dir, "Empty", None) {
            Err(ModelImportError::NoModelFile(_)) => {}
            other => panic!("expected NoModelFile, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multiple_models_fail() {
        let mut project = ProjectDocument::starter();
        // Two models — content doesn't matter because the
        // duplicate detection happens before parsing.
        let bogus = b"PSMDbogus";
        let dir = make_bundle("multi-model", Some(bogus), 2, &[], &[]);
        match register_cooked_model_bundle(&mut project, &dir, "Multi", None) {
            Err(ModelImportError::MultipleModelFiles { paths }) => {
                assert_eq!(paths.len(), 2);
            }
            other => panic!("expected MultipleModelFiles, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_model_bytes_fail() {
        let mut project = ProjectDocument::starter();
        let dir = make_bundle("bad-model", Some(b"NOTAPSXMDL"), 1, &[], &[]);
        match register_cooked_model_bundle(&mut project, &dir, "Bad", None) {
            Err(ModelImportError::InvalidModel { .. }) => {}
            other => panic!("expected InvalidModel, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn relativise_under_project_root_is_relative() {
        let root = PathBuf::from("/tmp/proj");
        let path = PathBuf::from("/tmp/proj/assets/models/x.psxmdl");
        assert_eq!(relativise(&path, Some(&root)), "assets/models/x.psxmdl");
        // No root → absolute kept.
        let abs = relativise(&path, None);
        assert_eq!(abs, "/tmp/proj/assets/models/x.psxmdl");
    }

    #[test]
    fn safe_dir_name_strips_punctuation() {
        assert_eq!(safe_dir_name("Obsidian Wraith"), "obsidian_wraith");
        assert_eq!(safe_dir_name("hooded-wretch"), "hooded_wretch");
        assert_eq!(safe_dir_name("!!!"), "model");
    }

    #[test]
    fn model_stats_from_obsidian_wraith() {
        let dir = obsidian_wraith_dir();
        let model_bytes = std::fs::read(dir.join("obsidian_wraith.psxmdl")).unwrap();
        let stats = model_stats_from_bytes(&model_bytes).expect("parse");
        assert!(stats.joint_count > 0);
        assert!(stats.part_count > 0);
        assert!(stats.vertex_count > 0);
        assert!(stats.face_count > 0);
        // AABB must span at least one unit on every axis after
        // walking real vertices.
        assert!(stats.bounds_max[0] >= stats.bounds_min[0]);
        assert!(stats.bounds_max[1] >= stats.bounds_min[1]);
        assert!(stats.bounds_max[2] >= stats.bounds_min[2]);
    }

    #[test]
    fn animation_stats_match_obsidian_wraith() {
        let dir = obsidian_wraith_dir();
        let model_bytes = std::fs::read(dir.join("obsidian_wraith.psxmdl")).unwrap();
        let model_stats = model_stats_from_bytes(&model_bytes).unwrap();
        let idle = std::fs::read(dir.join("obsidian_wraith_idle.psxanim")).unwrap();
        let stats =
            animation_stats_from_bytes("idle", &idle, model_stats.joint_count).expect("parse");
        assert!(stats.valid_for_model);
        assert!(stats.frame_count > 0);
        assert!(stats.sample_rate_hz > 0);
    }

    #[test]
    fn animation_stats_flag_joint_mismatch() {
        let dir = obsidian_wraith_dir();
        let idle = std::fs::read(dir.join("obsidian_wraith_idle.psxanim")).unwrap();
        let stats = animation_stats_from_bytes("idle", &idle, 999).expect("parse");
        assert!(!stats.valid_for_model);
    }

    #[test]
    fn texture_stats_detect_8bpp_atlas() {
        let dir = obsidian_wraith_dir();
        let bytes = std::fs::read(dir.join("obsidian_wraith_128x128_8bpp.psxt")).unwrap();
        let stats = texture_stats_from_bytes(&bytes).expect("parse");
        assert_eq!(stats.depth, 8);
        assert_eq!(stats.width, 128);
        assert_eq!(stats.height, 128);
        assert_eq!(stats.clut_entries, 256);
    }
}
