//! Persistent on-disk byte cache for the editor preview's
//! `.psxmdl` and `.psxanim` reads.
//!
//! Before this cache landed, the preview pass re-read every
//! placed model's mesh and animation off disk every frame, and
//! folded `(mesh_bytes, anim_bytes)` into a single cache row
//! keyed only by the Model `ResourceId`. That meant two
//! instances of the same model with different clip overrides
//! shared a single animation entry — whichever clip got there
//! first won, and the other instance silently played the wrong
//! pose. The player-spawn preview reuses the same cache, so its
//! idle clip would sometimes follow whatever a placed instance
//! had loaded.
//!
//! The fix splits the cache in two:
//!
//! * Mesh bytes keyed by the source `ResourceId`, signed by the
//!   model's authored `model_path` so editing the path forces a
//!   re-read on the next refresh.
//! * Animation bytes keyed by `(ResourceId, clip_idx)`, signed
//!   by the clip's authored `psxanim_path` so two instances of
//!   the same model with different clip overrides resolve to
//!   different entries.
//!
//! Cache lifetime matches the editor's other resource caches
//! (`EditorTextures`): persists across frames, walked by
//! [`EditorAssets::refresh`] once per frame to drop stale
//! entries and load newly-authored ones. Bytes are kept in a
//! `Box<[u8]>` so the slice borrow is stable across frames.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use psxed_project::{ProjectDocument, ResourceData, ResourceId};

/// Composite key for one cooked `.psxanim` blob — the owning
/// Model resource plus a clip index inside that model's clip
/// list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnimKey {
    pub model: ResourceId,
    pub clip: u16,
}

/// Cached `.psxmdl` bytes signed by the model's authored
/// `model_path`. Editing the path forces a re-read on the next
/// refresh.
#[derive(Debug, Clone)]
struct MeshEntry {
    bytes: Box<[u8]>,
    signature: String,
}

/// Cached `.psxanim` bytes signed by the clip's authored
/// `psxanim_path`. Two clips on the same model produce
/// separate entries because [`AnimKey`] embeds the clip index.
#[derive(Debug, Clone)]
struct AnimEntry {
    bytes: Box<[u8]>,
    signature: String,
}

/// Lazily-populated byte cache for the editor's 3D preview
/// pass. One per `Gfx` — mirrors [`crate::editor_textures::EditorTextures`]'s
/// resource-keyed shape so the editor preview never reads from
/// disk inside its render path.
#[derive(Debug, Default)]
pub struct EditorAssets {
    meshes: HashMap<ResourceId, MeshEntry>,
    animations: HashMap<AnimKey, AnimEntry>,
}

impl EditorAssets {
    pub fn new() -> Self {
        Self::default()
    }

    /// Walk every Model resource and ensure its mesh + every
    /// clip have entries. Removes entries for clips / models
    /// that no longer exist (signature mismatch or missing
    /// resource). Cheap when nothing changed because each
    /// entry's signature is compared against the project's
    /// current path.
    pub fn refresh(&mut self, project: &ProjectDocument, project_root: &Path) {
        let mut alive_meshes: Vec<ResourceId> = Vec::new();
        let mut alive_anims: Vec<AnimKey> = Vec::new();

        for resource in &project.resources {
            let ResourceData::Model(model) = &resource.data else {
                continue;
            };
            // Mesh.
            alive_meshes.push(resource.id);
            let mesh_signature = model.model_path.clone();
            let mesh_changed = self
                .meshes
                .get(&resource.id)
                .map(|e| e.signature != mesh_signature)
                .unwrap_or(true);
            if mesh_changed {
                let abs = resolve_path(&mesh_signature, project_root);
                match std::fs::read(&abs) {
                    Ok(bytes) => {
                        self.meshes.insert(
                            resource.id,
                            MeshEntry {
                                bytes: bytes.into_boxed_slice(),
                                signature: mesh_signature,
                            },
                        );
                    }
                    Err(_) => {
                        // Drop any stale cache row for a now-broken
                        // path so the preview can fall back gracefully.
                        self.meshes.remove(&resource.id);
                    }
                }
            }

            // Clips. Each clip is independently keyed +
            // signed; two clips on the same model resolve to
            // distinct entries even if they share a path.
            for (idx, clip) in model.clips.iter().enumerate() {
                let key = AnimKey {
                    model: resource.id,
                    clip: idx as u16,
                };
                alive_anims.push(key);
                let sig = clip.psxanim_path.clone();
                let changed = self
                    .animations
                    .get(&key)
                    .map(|e| e.signature != sig)
                    .unwrap_or(true);
                if !changed {
                    continue;
                }
                let abs = resolve_path(&sig, project_root);
                match std::fs::read(&abs) {
                    Ok(bytes) => {
                        self.animations.insert(
                            key,
                            AnimEntry {
                                bytes: bytes.into_boxed_slice(),
                                signature: sig,
                            },
                        );
                    }
                    Err(_) => {
                        self.animations.remove(&key);
                    }
                }
            }
        }

        // Drop entries that no longer correspond to a live
        // resource / clip. Keeps the cache from growing across
        // delete + re-add cycles.
        self.meshes.retain(|id, _| alive_meshes.contains(id));
        self.animations.retain(|k, _| alive_anims.contains(k));
    }

    /// Mesh bytes for `model`, or `None` when the cache hasn't
    /// learned about that resource yet (or it was authored with
    /// a broken path). Borrow lifetime matches `&self`, so
    /// callers can safely parse a `psx_asset::Model` from the
    /// returned slice and hold the parsed view for the rest of
    /// the frame.
    pub fn mesh_bytes(&self, model: ResourceId) -> Option<&[u8]> {
        self.meshes.get(&model).map(|e| &*e.bytes)
    }

    /// Animation bytes for one clip of one model. `None` for
    /// missing clip / missing file. Same borrow contract as
    /// [`Self::mesh_bytes`].
    pub fn clip_bytes(&self, model: ResourceId, clip: u16) -> Option<&[u8]> {
        self.animations
            .get(&AnimKey { model, clip })
            .map(|e| &*e.bytes)
    }
}

fn resolve_path(stored: &str, project_root: &Path) -> PathBuf {
    if Path::new(stored).is_absolute() {
        PathBuf::from(stored)
    } else {
        project_root.join(stored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psxed_project::{ModelAnimationClip, ModelResource};
    use std::fs;
    use std::time::SystemTime;

    /// Tiny scratch dir for the unit test. Each call seeds a
    /// uniquely-named directory so parallel test runs don't
    /// clobber each other.
    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "psoxide-editor-assets-{}-{}-{}",
            name,
            std::process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, rel: &str, bytes: &[u8]) -> String {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, bytes).unwrap();
        rel.to_string()
    }

    /// Build an empty project with a single Model resource whose
    /// mesh + two clips point at scratch-dir paths. Returns the
    /// project plus the new model's ResourceId so the test can
    /// look up the cache entry directly.
    fn project_with_dual_clip_model(
        dir: &Path,
        mesh_bytes: &[u8],
        clip_a: &[u8],
        clip_b: &[u8],
    ) -> (ProjectDocument, ResourceId) {
        let mesh_rel = write(dir, "mesh.psxmdl", mesh_bytes);
        let clip_a_rel = write(dir, "clip_a.psxanim", clip_a);
        let clip_b_rel = write(dir, "clip_b.psxanim", clip_b);
        // Empty project so the cache only sees the synthetic
        // model — the starter ships Obsidian Wraith with paths
        // relative to the real project root, which would never
        // resolve under our scratch dir.
        let mut project = ProjectDocument::new("Test");
        let id = project.add_resource(
            "Test Model",
            ResourceData::Model(ModelResource {
                model_path: mesh_rel,
                texture_path: None,
                clips: vec![
                    ModelAnimationClip {
                        name: "a".into(),
                        psxanim_path: clip_a_rel,
                    },
                    ModelAnimationClip {
                        name: "b".into(),
                        psxanim_path: clip_b_rel,
                    },
                ],
                default_clip: Some(0),
                preview_clip: Some(0),
                world_height: 1024,
            }),
        );
        (project, id)
    }

    #[test]
    fn separate_clips_on_same_model_resolve_to_distinct_entries() {
        // Regression: prior cache fused mesh + first-seen
        // animation into one row, so two instances of the same
        // model with different clip overrides played the same
        // pose. With clip-keyed entries each clip's bytes are
        // independent.
        let dir = scratch_dir("dual-clip");
        let mesh = b"MESHBYTES";
        let clip_a = b"AAAA-CLIP-A-AAAA";
        let clip_b = b"BBBB-CLIP-B-BBBB-DIFFERENT-LENGTH";
        let (project, model_id) = project_with_dual_clip_model(&dir, mesh, clip_a, clip_b);

        let mut assets = EditorAssets::new();
        assets.refresh(&project, &dir);
        let bytes_a = assets.clip_bytes(model_id, 0).expect("clip 0 cached");
        let bytes_b = assets.clip_bytes(model_id, 1).expect("clip 1 cached");
        assert_eq!(bytes_a, clip_a);
        assert_eq!(bytes_b, clip_b);
        assert_ne!(
            bytes_a, bytes_b,
            "clip 0 and clip 1 must not collapse to the same cache row"
        );
        let mesh_bytes = assets.mesh_bytes(model_id).expect("mesh cached");
        assert_eq!(mesh_bytes, mesh);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_drops_entries_when_resource_disappears() {
        let dir = scratch_dir("drop");
        let mesh = b"MESH";
        let clip = b"CLIP";
        let (project, model_id) = project_with_dual_clip_model(&dir, mesh, clip, clip);
        let mut assets = EditorAssets::new();
        assets.refresh(&project, &dir);
        assert!(assets.mesh_bytes(model_id).is_some());
        assert!(assets.clip_bytes(model_id, 1).is_some());

        // Drop the Model resource and re-refresh. Both the
        // mesh and the orphaned clip entry should be evicted.
        let mut project = project;
        project.resources.retain(|r| r.id != model_id);
        assets.refresh(&project, &dir);
        assert!(assets.mesh_bytes(model_id).is_none());
        assert!(assets.clip_bytes(model_id, 0).is_none());
        assert!(assets.clip_bytes(model_id, 1).is_none());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_reads_each_path_only_once_until_signature_changes() {
        // Indirect signal: rewrite the same path with new
        // bytes between refreshes. The cache should pick the
        // new bytes up only when the path *changes*, not on
        // every refresh — so editing the file in place keeps
        // the old bytes (matches the EditorTextures contract).
        let dir = scratch_dir("sig");
        let (project, model_id) = project_with_dual_clip_model(&dir, b"v1", b"v1", b"v1");
        let mut assets = EditorAssets::new();
        assets.refresh(&project, &dir);
        assert_eq!(assets.mesh_bytes(model_id).unwrap(), b"v1");

        // Rewrite the same on-disk path with new bytes.
        fs::write(dir.join("mesh.psxmdl"), b"v2").unwrap();
        assets.refresh(&project, &dir);
        // Path didn't change, so the cache is still v1 —
        // identical contract to EditorTextures.
        assert_eq!(assets.mesh_bytes(model_id).unwrap(), b"v1");

        let _ = fs::remove_dir_all(dir);
    }
}
