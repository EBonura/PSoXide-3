//! Persistent on-disk byte cache for the editor preview's
//! `.psxmdl` and `.psxanim` reads.
//!
//! Before this cache landed, the preview pass re-read every
//! placed model's mesh and animation off disk every frame, and
//! folded `(mesh_bytes, anim_bytes)` into a single cache row
//! keyed only by the Model `ResourceId`. That meant two
//! instances of the same model with different clip overrides
//! shared a single animation entry -- whichever clip got there
//! first won, and the other instance silently played the wrong
//! pose. The player-spawn preview reuses the same cache, so its
//! idle clip would sometimes follow whatever a placed instance
//! had loaded.
//!
//! The fix splits the cache in two:
//!
//! * Mesh bytes keyed by the source `ResourceId`, signed by the
//!   model's authored `model_path` *plus* the file's length and
//!   mtime so an in-place rewrite invalidates the cache row.
//! * Animation bytes keyed by `(ResourceId, clip_idx)`, signed
//!   the same way -- two instances of the same model with
//!   different clip overrides resolve to different entries.
//!
//! Failed reads are cached too. Without that, a project
//! authored against a non-existent path would re-attempt the
//! read every frame because the prior implementation simply
//! evicted broken rows, and the next refresh found nothing
//! cached and tried again. The failure row is signed against
//! whatever metadata was readable (or `None` for a missing
//! file) so the cache automatically retries when the file
//! reappears or its mtime changes.
//!
//! Cache lifetime matches the editor's other resource caches
//! (`EditorTextures`): persists across frames, walked by
//! [`EditorAssets::refresh`] once per frame to drop stale
//! entries and load newly-authored ones. Bytes are kept in a
//! `Box<[u8]>` so the slice borrow is stable across frames.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use psxed_project::{ProjectDocument, ResourceData, ResourceId};

/// Composite key for one cooked `.psxanim` blob -- the owning
/// Model resource plus a clip index inside that model's clip
/// list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnimKey {
    pub model: ResourceId,
    pub clip: u16,
}

/// File identity captured for cache invalidation. Equality on
/// all three fields means: the authored path string, the
/// resolved file's byte length, and the file's mtime. Two
/// signatures comparing equal across refreshes is the cache's
/// "nothing changed" predicate.
///
/// `metadata` is `None` when the file's metadata can't be read
/// (file missing, permissions). That's an explicit "no
/// metadata" signal -- it equals other `None`-metadata
/// signatures with the same path, so a persistently-broken
/// path stops re-reading. The moment the file appears, its
/// metadata becomes `Some` and the signature differs, which
/// triggers a re-read attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Signature {
    path: String,
    metadata: Option<FileMetadata>,
}

/// Just the bits of `std::fs::Metadata` the signature cares
/// about. Captured as a struct (rather than a tuple) so
/// equality is field-named at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileMetadata {
    len: u64,
    modified: SystemTime,
}

/// One cached entry -- either successfully loaded bytes, or a
/// negative cache row for a path the last refresh couldn't
/// read. Both variants carry their `Signature` so the next
/// refresh can decide whether to re-attempt.
#[derive(Debug)]
enum Slot {
    Loaded {
        bytes: Box<[u8]>,
        signature: Signature,
    },
    Failed {
        signature: Signature,
    },
}

impl Slot {
    fn signature(&self) -> &Signature {
        match self {
            Self::Loaded { signature, .. } | Self::Failed { signature } => signature,
        }
    }

    /// `Some(bytes)` for loaded rows; `None` for failed rows.
    /// The public `mesh_bytes` / `clip_bytes` accessors funnel
    /// through this so a cached failure looks the same to
    /// callers as "never loaded".
    fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Loaded { bytes, .. } => Some(&**bytes),
            Self::Failed { .. } => None,
        }
    }
}

/// Lazily-populated byte cache for the editor's 3D preview
/// pass. One per `Gfx` -- mirrors [`crate::editor_textures::EditorTextures`]'s
/// resource-keyed shape so the editor preview never reads from
/// disk inside its render path.
#[derive(Debug, Default)]
pub struct EditorAssets {
    meshes: HashMap<ResourceId, Slot>,
    animations: HashMap<AnimKey, Slot>,
}

impl EditorAssets {
    pub fn new() -> Self {
        Self::default()
    }

    /// Walk every Model resource and ensure its mesh + every
    /// clip have entries. Removes entries for clips / models
    /// that no longer exist (signature mismatch or missing
    /// resource). Cheap when nothing changed because each
    /// entry's signature is compared against the file's
    /// current metadata -- same authored path + same length +
    /// same mtime → skip.
    pub fn refresh(&mut self, project: &ProjectDocument, project_root: &Path) {
        let mut alive_meshes: Vec<ResourceId> = Vec::new();
        let mut alive_anims: Vec<AnimKey> = Vec::new();

        for resource in &project.resources {
            let ResourceData::Model(model) = &resource.data else {
                continue;
            };

            // Mesh.
            alive_meshes.push(resource.id);
            let abs = resolve_path(&model.model_path, project_root);
            let desired = Signature {
                path: model.model_path.clone(),
                metadata: read_file_metadata(&abs),
            };
            let needs_refresh = self
                .meshes
                .get(&resource.id)
                .map(|slot| slot.signature() != &desired)
                .unwrap_or(true);
            if needs_refresh {
                self.meshes.insert(resource.id, load_or_fail(&abs, desired));
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
                let abs = resolve_path(&clip.psxanim_path, project_root);
                let desired = Signature {
                    path: clip.psxanim_path.clone(),
                    metadata: read_file_metadata(&abs),
                };
                let needs_refresh = self
                    .animations
                    .get(&key)
                    .map(|slot| slot.signature() != &desired)
                    .unwrap_or(true);
                if needs_refresh {
                    self.animations.insert(key, load_or_fail(&abs, desired));
                }
            }
        }

        // Drop entries that no longer correspond to a live
        // resource / clip. Keeps the cache from growing across
        // delete + re-add cycles. Both Loaded and Failed rows
        // are dropped -- the failure cache only matters while
        // *some* resource still references the path.
        self.meshes.retain(|id, _| alive_meshes.contains(id));
        self.animations.retain(|k, _| alive_anims.contains(k));
    }

    /// Mesh bytes for `model`, or `None` when the cache hasn't
    /// learned about that resource yet, the resource isn't a
    /// Model, or the last refresh's read failed. Borrow
    /// lifetime matches `&self`, so callers can safely parse a
    /// `psx_asset::Model` from the returned slice and hold the
    /// parsed view for the rest of the frame.
    pub fn mesh_bytes(&self, model: ResourceId) -> Option<&[u8]> {
        self.meshes.get(&model).and_then(Slot::bytes)
    }

    /// Animation bytes for one clip of one model. `None` for
    /// missing clip / missing file / cached read failure. Same
    /// borrow contract as [`Self::mesh_bytes`].
    pub fn clip_bytes(&self, model: ResourceId, clip: u16) -> Option<&[u8]> {
        self.animations
            .get(&AnimKey { model, clip })
            .and_then(Slot::bytes)
    }
}

/// Attempt to read `abs` and return either a `Loaded` slot
/// (bytes alongside the resolved signature) or a `Failed` slot
/// (just the signature). Either way the desired signature is
/// stored verbatim so the next refresh can compare against it.
fn load_or_fail(abs: &Path, signature: Signature) -> Slot {
    match fs::read(abs) {
        Ok(bytes) => Slot::Loaded {
            bytes: bytes.into_boxed_slice(),
            signature,
        },
        Err(_) => Slot::Failed { signature },
    }
}

/// Capture the bits of `std::fs::Metadata` we care about, or
/// `None` if the file can't be stat'd. `modified()` can fail
/// on a handful of legacy filesystems; treating that as a
/// signature mismatch is fine -- the worst case is one extra
/// re-read per refresh on those targets.
fn read_file_metadata(abs: &Path) -> Option<FileMetadata> {
    let md = fs::metadata(abs).ok()?;
    let modified = md.modified().ok()?;
    Some(FileMetadata {
        len: md.len(),
        modified,
    })
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
    use std::thread::sleep;
    use std::time::Duration;

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
        // model -- the starter ships Obsidian Wraith with paths
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
                scale_q8: [psxed_project::MODEL_SCALE_ONE_Q8; 3],
            }),
        );
        (project, id)
    }

    /// Some filesystems quantise mtime to 1-second granularity
    /// (FAT, older HFS, some NFS). Sleep just past the next
    /// whole second so a follow-up rewrite always lands on a
    /// fresh mtime. macOS APFS / Linux ext4 don't need this,
    /// but the cost is one second of test wall time and it
    /// keeps CI portable.
    fn await_next_mtime_tick() {
        sleep(Duration::from_millis(1100));
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
    fn in_place_edit_updates_cached_bytes_on_next_refresh() {
        // The original signature was just the path string, so
        // editing the file in place was *not* picked up. The
        // richer (path, len, mtime) signature treats a rewrite
        // as a cache invalidation -- which is what authors
        // actually expect when they re-cook a model.
        let dir = scratch_dir("edit");
        let (project, model_id) = project_with_dual_clip_model(&dir, b"v1", b"v1", b"v1");
        let mut assets = EditorAssets::new();
        assets.refresh(&project, &dir);
        assert_eq!(assets.mesh_bytes(model_id).unwrap(), b"v1");

        // Same path, longer bytes -- len differs even on
        // filesystems with coarse mtime granularity.
        await_next_mtime_tick();
        fs::write(dir.join("mesh.psxmdl"), b"v2-different-length").unwrap();
        assets.refresh(&project, &dir);
        assert_eq!(assets.mesh_bytes(model_id).unwrap(), b"v2-different-length");

        // Same path, same length, fresh mtime. Catches the
        // "byte-length matches by coincidence" case the prior
        // path-only signature would also have missed.
        await_next_mtime_tick();
        fs::write(dir.join("mesh.psxmdl"), b"v3-different-length").unwrap();
        assets.refresh(&project, &dir);
        assert_eq!(assets.mesh_bytes(model_id).unwrap(), b"v3-different-length");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_file_caches_failure_and_does_not_retry_until_metadata_changes() {
        // Author a Model whose mesh path doesn't exist on
        // disk. The first refresh learns about it (read fails
        // → Failed slot stored). The second refresh sees the
        // same desired signature (still missing) and skips the
        // re-read entirely. The third -- after the file
        // appears -- picks the new bytes up.
        let dir = scratch_dir("fail-then-fix");
        // Stub the mesh path; don't actually write the file
        // until later.
        let mut project = ProjectDocument::new("Test");
        let model_id = project.add_resource(
            "Test Model",
            ResourceData::Model(ModelResource {
                model_path: "missing.psxmdl".into(),
                texture_path: None,
                clips: vec![],
                default_clip: None,
                preview_clip: None,
                world_height: 1024,
                scale_q8: [psxed_project::MODEL_SCALE_ONE_Q8; 3],
            }),
        );

        let mut assets = EditorAssets::new();
        assets.refresh(&project, &dir);
        assert!(
            assets.mesh_bytes(model_id).is_none(),
            "missing file must not produce bytes"
        );
        // The failure was cached -- confirm by inspecting the
        // internal slot directly. `mesh_bytes` returning None
        // alone could mean either "not cached" or "cached
        // failure"; we want the latter so we know the next
        // refresh skips the re-read.
        let signature_after_first = assets
            .meshes
            .get(&model_id)
            .map(|slot| slot.signature().clone())
            .expect("failure row was cached");
        assert!(matches!(
            assets.meshes.get(&model_id),
            Some(Slot::Failed { .. })
        ));

        // Second refresh on a still-missing file: signature
        // unchanged, no re-read attempt should mutate the
        // slot. The signature object compares equal across
        // refreshes -- that's the no-churn observation.
        assets.refresh(&project, &dir);
        let signature_after_second = assets
            .meshes
            .get(&model_id)
            .map(|slot| slot.signature().clone())
            .expect("failure row still cached");
        assert_eq!(signature_after_first, signature_after_second);

        // Now write the file. Metadata becomes Some(...), so
        // the desired signature differs from the cached
        // Failed signature → refresh re-reads → Loaded.
        fs::write(dir.join("missing.psxmdl"), b"loaded-now").unwrap();
        assets.refresh(&project, &dir);
        assert_eq!(
            assets.mesh_bytes(model_id).unwrap(),
            b"loaded-now",
            "appearing file must invalidate the failure cache"
        );

        let _ = fs::remove_dir_all(dir);
    }
}
