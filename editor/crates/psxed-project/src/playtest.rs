//! Playtest pipeline: scene-tree → cooked rooms + master asset
//! table + per-room residency lists, written as a Rust-source
//! manifest the `engine/examples/editor-playtest` example
//! `include!`s.
//!
//! # Why a Rust-source manifest?
//!
//! The runtime example is `no_std` and PSX-target only. It can't
//! deserialize RON / parse RAM-resident config without dragging in
//! crates the cooked path doesn't want. A generated Rust source
//! file with `include_bytes!` references is the lightest contract:
//! the runtime sees `static ASSETS: &[LevelAssetRecord]` /
//! `static ROOM_RESIDENCY: &[RoomResidencyRecord]` and the bytes
//! are baked into the EXE at build time.
//!
//! # Schema lives in `psx-level`
//!
//! The record types ([`psx_level::LevelAssetRecord`] and friends)
//! live in the shared `no_std` `psx-level` crate so the writer
//! here and the reader in the runtime example reference one
//! definition. Whenever a record's shape changes, both ends
//! pick up the change at compile time.
//!
//! # Backing store
//!
//! Today every asset is `include_bytes!`-baked. Tomorrow assets
//! may be paged in from a stream pack on CD; the schema doesn't
//! care. The residency manager already tracks RAM/VRAM membership
//! independently of where bytes live.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::world_cook::{cook_world_grid, CookedWorldMaterial, WorldGridCookError};
use crate::{
    ModelResource, NodeId, NodeKind, ProjectDocument, Resource, ResourceData, ResourceId,
    SceneNode, WorldGrid,
};

/// Generated subdirectory inside the playtest example that
/// receives `level_manifest.rs` + `rooms/` + `textures/`. Stable
/// so the example's `include!` paths don't move.
pub const GENERATED_DIRNAME: &str = "generated";

/// Filename of the generated Rust-source manifest the example
/// `include!`s.
pub const MANIFEST_FILENAME: &str = "level_manifest.rs";

/// Subdirectory inside `generated/` that holds cooked `.psxw`
/// blobs.
pub const ROOMS_DIRNAME: &str = "rooms";

/// Subdirectory inside `generated/` that holds copied `.psxt`
/// texture blobs.
pub const TEXTURES_DIRNAME: &str = "textures";

/// Subdirectory inside `generated/` that holds per-model
/// folders (`model_NNN/`) carrying mesh + atlas + animation
/// blobs. One subfolder per unique [`ResourceData::Model`]
/// referenced by any placed [`NodeKind::MeshInstance`].
pub const MODELS_DIRNAME: &str = "models";

/// Coarse asset class — mirrors [`psx_level::AssetKind`] but
/// stays host-side `String`/`Vec` friendly. Converted to the
/// runtime enum at write time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaytestAssetKind {
    /// Cooked `.psxw` room blob.
    RoomWorld,
    /// Cooked `.psxt` texture blob (room atlas or model atlas).
    Texture,
    /// Cooked `.psxmdl` mesh blob.
    ModelMesh,
    /// Cooked `.psxanim` skeletal animation clip.
    ModelAnimation,
}

/// One asset destined for the master table. Owns its bytes so
/// callers can write them out to the generated tree without
/// reaching back into the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestAsset {
    /// Asset class — drives extension + loader.
    pub kind: PlaytestAssetKind,
    /// Backing payload.
    pub bytes: Vec<u8>,
    /// Filename inside the kind's subdirectory (e.g.
    /// `room_000.psxw`). Stable across runs because asset order
    /// is deterministic.
    pub filename: String,
    /// Diagnostic label — display name of the source resource
    /// or room. Surfaces in cook reports and stays out of the
    /// runtime contract.
    pub source_label: String,
}

/// One room's residency-aware record. Carries indices into
/// [`PlaytestPackage::assets`] and [`PlaytestPackage::materials`]
/// so the writer can resolve `AssetId`s deterministically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestRoom {
    /// Display name lifted from the editor scene tree.
    pub name: String,
    /// Index into [`PlaytestPackage::assets`] of the room's
    /// `RoomWorld` asset.
    pub world_asset_index: usize,
    /// Editor-side `WorldGrid::origin[0]` (diagnostic only).
    pub origin_x: i32,
    /// Editor-side `WorldGrid::origin[1]`.
    pub origin_z: i32,
    /// Engine units per sector.
    pub sector_size: i32,
    /// First index into [`PlaytestPackage::materials`] for this
    /// room's slice.
    pub material_first: u16,
    /// Number of material records in the slice. Matches the
    /// cooked `.psxw`'s material count exactly.
    pub material_count: u16,
}

/// One material slot binding. Lifted from
/// [`CookedWorldMaterial`] and pinned to its owning room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestMaterial {
    /// Owning room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Cooked-world local material slot — matches the slot value
    /// stored in the `.psxw`.
    pub local_slot: u16,
    /// Index into [`PlaytestPackage::assets`] of the texture
    /// asset bound at this slot.
    pub texture_asset_index: usize,
    /// Per-material modulation tint.
    pub tint_rgb: [u8; 3],
}

/// One animation clip bound to a [`PlaytestModel`]. Carries
/// pre-resolved indices into the master asset table so the
/// writer can emit `LevelModelClipRecord`s without re-walking
/// the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestModelClip {
    /// Owning model index in [`PlaytestPackage::models`].
    pub model: u16,
    /// Display name surfaced in debug HUDs.
    pub name: String,
    /// Index into [`PlaytestPackage::assets`] of the cooked
    /// `.psxanim` blob.
    pub animation_asset_index: usize,
}

/// One cooked PSX model included in the playtest package. A
/// [`ResourceData::Model`] referenced by any placed instance is
/// promoted into one `PlaytestModel`; multiple instances share
/// the same record (deduplicated by source `ResourceId`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestModel {
    /// Display name lifted from the editor resource.
    pub name: String,
    /// Source resource id — used to deduplicate instances and
    /// to resolve per-instance clip overrides back to clip
    /// indices within this model's slice.
    pub source_resource: ResourceId,
    /// Index into [`PlaytestPackage::assets`] of the cooked
    /// `.psxmdl` blob.
    pub mesh_asset_index: usize,
    /// Index into [`PlaytestPackage::assets`] of the atlas
    /// `.psxt` blob. Always `Some` for placed models — the
    /// playtest cooker rejects instances of models without an
    /// atlas. Kept as `Option` so the schema can later carry
    /// untextured author-time bundles unchanged.
    pub texture_asset_index: Option<usize>,
    /// First index into [`PlaytestPackage::model_clips`] for
    /// this model's clip slice.
    pub clip_first: u16,
    /// Number of clips in this model's slice. Matches the
    /// editor resource's clip count exactly.
    pub clip_count: u16,
    /// Default clip index *within this model's slice*.
    /// Cooker validation guarantees this is `< clip_count`,
    /// so the runtime always has a clip to play.
    pub default_clip: u16,
    /// World-space height (engine units) — propagated from the
    /// editor resource.
    pub world_height: u16,
}

/// One placed model instance. Coordinates are room-local
/// engine units (the same space cooked rooms live in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestModelInstance {
    /// Owning room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Model index in [`PlaytestPackage::models`].
    pub model: u16,
    /// Per-instance clip override, or [`MODEL_CLIP_INHERIT`]
    /// to use the model's `default_clip`.
    pub clip: u16,
    /// Room-local X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Room-local Z.
    pub z: i32,
    /// Yaw, PSX angle units.
    pub yaw: i16,
    /// Reserved.
    pub flags: u16,
}

/// Sentinel for [`PlaytestModelInstance::clip`] meaning
/// "inherit model default" — same value as
/// [`psx_level::MODEL_CLIP_INHERIT`].
pub const MODEL_CLIP_INHERIT: u16 = 0xFFFF;

/// One placed point light, room-local engine units. Mirrors
/// [`psx_level::PointLightRecord`] one-for-one — intensity is
/// already quantised to Q8.8 so the cook output is a direct
/// copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestLight {
    /// Room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Room-local X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Room-local Z.
    pub z: i32,
    /// Cutoff distance in engine units. Cooker rejects `0`.
    pub radius: u16,
    /// Brightness multiplier in Q8.8 (`256` = 1.0). Derived
    /// from the editor's `f32` intensity at cook time.
    pub intensity_q8: u16,
    /// 8-bit RGB tint.
    pub color: [u8; 3],
}

/// Player spawn record. Coordinates are room-local engine units
/// (the same space the cooked `.psxw` lives in — array-rooted at
/// world `(0, 0)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestSpawn {
    /// Room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Room-local X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Room-local Z.
    pub z: i32,
    /// Yaw in PSX angle units.
    pub yaw: i16,
    /// Reserved flags. Bit 0 = "spawn enabled".
    pub flags: u16,
}

/// Coarse runtime kind for [`PlaytestEntity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaytestEntityKind {
    /// Visual marker (debug cube).
    Marker,
    /// Static mesh instance pinned by `resource_slot`.
    StaticMesh,
}

/// Cooked character record. Mirrors
/// [`psx_level::LevelCharacterRecord`] one-to-one — the writer
/// emits the static slice from this struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestCharacter {
    /// Source resource id (so the writer can dedupe + cross-link).
    pub source_resource: ResourceId,
    /// Index into [`PlaytestPackage::models`].
    pub model: u16,
    /// Idle clip index *within the model's clip slice*.
    pub idle_clip: u16,
    /// Walk clip index within the model's clip slice.
    pub walk_clip: u16,
    /// Optional run clip; `u16::MAX` (= `CHARACTER_CLIP_NONE`)
    /// means "no run clip authored".
    pub run_clip: u16,
    /// Optional turn clip; same sentinel as `run_clip`.
    pub turn_clip: u16,
    /// Capsule radius in engine units.
    pub radius: u16,
    /// Capsule height in engine units.
    pub height: u16,
    /// Walk speed (engine units / 60 Hz frame).
    pub walk_speed: i32,
    /// Run speed (engine units / 60 Hz frame).
    pub run_speed: i32,
    /// Turn speed (degrees / second).
    pub turn_speed_degrees_per_second: u16,
    /// Camera follow distance (engine units).
    pub camera_distance: i32,
    /// Camera vertical offset above the character origin.
    pub camera_height: i32,
    /// Vertical offset of the camera's look-at target.
    pub camera_target_height: i32,
}

/// Cooked player-controller record. Always paired with a
/// [`PlaytestSpawn`] in the same package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestPlayerController {
    /// Resolved spawn — same data the manifest's `PLAYER_SPAWN`
    /// carries.
    pub spawn: PlaytestSpawn,
    /// Character index in [`PlaytestPackage::characters`].
    pub character: u16,
}

/// Sentinel used in [`PlaytestCharacter::run_clip`] /
/// [`PlaytestCharacter::turn_clip`] when the role wasn't
/// authored. Mirrors [`psx_level::CHARACTER_CLIP_NONE`].
pub const CHARACTER_CLIP_NONE: u16 = u16::MAX;

/// One non-spawn entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestEntity {
    /// Owning room index.
    pub room: u16,
    /// Entity kind.
    pub kind: PlaytestEntityKind,
    /// Room-local X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Room-local Z.
    pub z: i32,
    /// Yaw, PSX angle units.
    pub yaw: i16,
    /// Resource slot (0 if unused).
    pub resource_slot: u16,
    /// Reserved flags.
    pub flags: u16,
}

/// Cooked playtest scene, ready to write to disk. Holds
/// everything the generated manifest needs to render: assets,
/// per-room metadata, per-room material slices, models, model
/// instances, and residency.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlaytestPackage {
    /// Master asset table — rooms first, then room textures,
    /// then per-model assets (mesh + atlas + clips), in
    /// deterministic order.
    pub assets: Vec<PlaytestAsset>,
    /// Cooked rooms with material-slice metadata.
    pub rooms: Vec<PlaytestRoom>,
    /// Material records ordered as `(room, local_slot)`.
    pub materials: Vec<PlaytestMaterial>,
    /// Cooked model bundles, deduplicated across instances.
    pub models: Vec<PlaytestModel>,
    /// Per-model clip records ordered as `(model, clip_index)`.
    pub model_clips: Vec<PlaytestModelClip>,
    /// Placed model instances, room-local coordinates.
    pub model_instances: Vec<PlaytestModelInstance>,
    /// Placed point lights, room-local coordinates.
    pub lights: Vec<PlaytestLight>,
    /// Single player spawn — required.
    pub spawn: Option<PlaytestSpawn>,
    /// Cooked Character resources used by player / future
    /// gameplay. Currently only the player spawn references
    /// these, but the slice ships in the manifest unconditionally
    /// so the runtime can table-drive any future controllers.
    pub characters: Vec<PlaytestCharacter>,
    /// Resolved player controller — `Some` when a player spawn
    /// was authored *and* a Character was assigned (or
    /// auto-picked). The runtime falls back to a debug camera
    /// when this is `None`.
    pub player_controller: Option<PlaytestPlayerController>,
    /// Optional entity markers (legacy, non-Model MeshInstance).
    pub entities: Vec<PlaytestEntity>,
}

impl PlaytestPackage {
    /// Number of `RoomWorld` entries in [`Self::assets`].
    pub fn room_asset_count(&self) -> usize {
        self.assets
            .iter()
            .filter(|a| a.kind == PlaytestAssetKind::RoomWorld)
            .count()
    }

    /// Number of `Texture` entries in [`Self::assets`].
    pub fn texture_asset_count(&self) -> usize {
        self.assets
            .iter()
            .filter(|a| a.kind == PlaytestAssetKind::Texture)
            .count()
    }

    /// Number of `ModelMesh` entries in [`Self::assets`].
    pub fn model_mesh_asset_count(&self) -> usize {
        self.assets
            .iter()
            .filter(|a| a.kind == PlaytestAssetKind::ModelMesh)
            .count()
    }

    /// Number of `ModelAnimation` entries in [`Self::assets`].
    pub fn model_animation_asset_count(&self) -> usize {
        self.assets
            .iter()
            .filter(|a| a.kind == PlaytestAssetKind::ModelAnimation)
            .count()
    }
}

/// Outcome of validating a project for playtest. Errors block
/// cooking; warnings are surfaced but not fatal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlaytestValidationReport {
    /// Hard errors. Cook & Play must refuse to launch when this
    /// list is non-empty.
    pub errors: Vec<String>,
    /// Soft warnings. Surface in the editor status line but
    /// don't block cooking.
    pub warnings: Vec<String>,
}

impl PlaytestValidationReport {
    /// `true` when there are zero hard errors.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    fn error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}

/// Build a playtest package from `project`. Validates the scene
/// tree, cooks every Room with non-empty geometry, resolves
/// material textures (loading bytes through `project_root` for
/// path-relative resources), and assigns the player spawn.
///
/// `project_root` anchors relative texture `psxt_path`s. Pass
/// the project's directory; absolute paths short-circuit through
/// the resolver unchanged.
///
/// On any validation error the returned package is `None`. Cooked
/// artifacts only land on disk via [`write_package`] / [`cook_to_dir`]
/// when validation passes — partial writes never happen.
pub fn build_package(
    project: &ProjectDocument,
    project_root: &Path,
) -> (Option<PlaytestPackage>, PlaytestValidationReport) {
    let mut report = PlaytestValidationReport::default();
    let scene = project.active_scene();

    // Pass 1: enumerate Room nodes. Index = runtime room id.
    let mut room_nodes: Vec<&SceneNode> = scene
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::Room { .. }))
        .collect();
    room_nodes.sort_by_key(|node| node.id.raw());

    if room_nodes.is_empty() {
        report.error("playtest needs at least one Room node — none found");
        return (None, report);
    }

    // Pass 2: cook each Room. We need the `CookedWorldGrid` for
    // material slot info; encode straight from it so we don't
    // pay for two cooks. Empty grids skip with a warning.
    let mut assets: Vec<PlaytestAsset> = Vec::new();
    let mut rooms: Vec<PlaytestRoom> = Vec::new();
    let mut materials: Vec<PlaytestMaterial> = Vec::new();
    // ResourceId → index into `assets` for texture deduplication.
    // First-use order is deterministic because we walk rooms +
    // material slots in deterministic order and assign the
    // texture's compact "texture index" via `texture_asset_for_resource.len()`
    // at first insertion (never removed). HashMap is fine — we
    // only use it for presence tests.
    let mut texture_asset_for_resource: std::collections::HashMap<ResourceId, usize> =
        std::collections::HashMap::new();
    let mut node_to_room_index: std::collections::HashMap<NodeId, u16> =
        std::collections::HashMap::new();

    for room_node in &room_nodes {
        let NodeKind::Room { grid } = &room_node.kind else {
            continue;
        };
        if grid.populated_sector_count() == 0 {
            report.warn(format!(
                "Room '{}' has no geometry — skipped",
                room_node.name
            ));
            continue;
        }
        let cooked = match cook_world_grid(project, grid) {
            Ok(c) => c,
            Err(e) => {
                report.error(cook_error_for_node(&room_node.name, e));
                return (None, report);
            }
        };
        let bytes = match cooked.to_psxw_bytes() {
            Ok(b) => b,
            Err(e) => {
                report.error(cook_error_for_node(&room_node.name, e));
                return (None, report);
            }
        };

        let room_index = u16::try_from(rooms.len()).unwrap_or(u16::MAX);
        node_to_room_index.insert(room_node.id, room_index);

        // Room asset goes into the master table first (ahead of
        // any material textures discovered while walking it).
        let world_asset_index = assets.len();
        assets.push(PlaytestAsset {
            kind: PlaytestAssetKind::RoomWorld,
            bytes,
            filename: format!("room_{:03}.psxw", room_index),
            source_label: room_node.name.clone(),
        });

        // Walk material slots in slot order. The cooker emits
        // CookedWorldMaterial per resolved slot id; we build
        // PlaytestMaterial mirrors keyed to (room, local_slot)
        // and register each unique texture asset on first use.
        let material_first = u16::try_from(materials.len()).unwrap_or(u16::MAX);
        let mut sorted_materials: Vec<&CookedWorldMaterial> = cooked.materials.iter().collect();
        sorted_materials.sort_by_key(|m| m.slot);

        for cooked_material in sorted_materials {
            let texture_id = match cooked_material.texture {
                Some(id) => id,
                None => {
                    report.error(format!(
                        "Room '{}' material slot {} has no texture (resource #{})",
                        room_node.name,
                        cooked_material.slot,
                        cooked_material.source.raw(),
                    ));
                    return (None, report);
                }
            };
            let texture_resource = match find_resource(project, texture_id) {
                Some(r) => r,
                None => {
                    report.error(format!(
                        "Room '{}' material slot {} references missing texture resource #{}",
                        room_node.name,
                        cooked_material.slot,
                        texture_id.raw(),
                    ));
                    return (None, report);
                }
            };
            let texture_asset_index =
                if let Some(&existing) = texture_asset_for_resource.get(&texture_id) {
                    existing
                } else {
                    let bytes = match load_texture_bytes(texture_resource, project_root) {
                        Ok(b) => b,
                        Err(msg) => {
                            report.error(format!(
                                "Room '{}' material slot {}: {}",
                                room_node.name, cooked_material.slot, msg,
                            ));
                            return (None, report);
                        }
                    };
                    // Room materials must be 4bpp (16-entry CLUT) —
                    // both the editor preview's material upload
                    // path and the runtime room material slots
                    // assume the 4bpp tpage layout. Loud failure
                    // here beats wrong-colour rendering at runtime.
                    if let Err(msg) = expect_room_material_depth(texture_resource, &bytes) {
                        report.error(format!(
                            "Room '{}' material slot {}: {}",
                            room_node.name, cooked_material.slot, msg,
                        ));
                        return (None, report);
                    }
                    let texture_index = texture_asset_for_resource.len();
                    let new_index = assets.len();
                    assets.push(PlaytestAsset {
                        kind: PlaytestAssetKind::Texture,
                        bytes,
                        filename: format!("texture_{:03}.psxt", texture_index),
                        source_label: texture_resource.name.clone(),
                    });
                    texture_asset_for_resource.insert(texture_id, new_index);
                    new_index
                };

            materials.push(PlaytestMaterial {
                room: room_index,
                local_slot: cooked_material.slot,
                texture_asset_index,
                tint_rgb: cooked_material.tint,
            });
        }
        let material_count =
            u16::try_from(materials.len() - material_first as usize).unwrap_or(u16::MAX);

        rooms.push(PlaytestRoom {
            name: room_node.name.clone(),
            world_asset_index,
            origin_x: grid.origin[0],
            origin_z: grid.origin[1],
            sector_size: grid.sector_size,
            material_first,
            material_count,
        });
    }

    if rooms.is_empty() {
        report.error("every Room is empty — cook needs at least one populated room");
        return (None, report);
    }

    // Pass 3: spawn + entities + model instances + lights.
    let mut player_spawns: Vec<(NodeId, &SceneNode, u16)> = Vec::new();
    let mut entities: Vec<PlaytestEntity> = Vec::new();
    let mut models: Vec<PlaytestModel> = Vec::new();
    let mut model_clips: Vec<PlaytestModelClip> = Vec::new();
    let mut model_instances: Vec<PlaytestModelInstance> = Vec::new();
    let mut lights: Vec<PlaytestLight> = Vec::new();
    // ResourceId → index into `models` for instance dedup.
    let mut model_for_resource: std::collections::HashMap<ResourceId, u16> =
        std::collections::HashMap::new();
    let mut warned_unsupported: HashSet<&'static str> = HashSet::new();

    for node in scene.nodes() {
        if node.id == scene.root || matches!(node.kind, NodeKind::Room { .. }) {
            continue;
        }
        let Some((room_node, room_index)) = enclosing_room(scene, node, &node_to_room_index) else {
            if !matches!(
                node.kind,
                NodeKind::Node | NodeKind::Node3D | NodeKind::World
            ) {
                report.warn(format!(
                    "{} '{}' has no enclosing Room — dropped",
                    node.kind.label(),
                    node.name
                ));
            }
            continue;
        };
        let NodeKind::Room { grid } = &room_node.kind else {
            continue;
        };
        let pos = node_room_local_position(node, grid);
        let yaw = yaw_from_degrees(node.transform.rotation_degrees[1]);

        match &node.kind {
            NodeKind::SpawnPoint { player: true, .. } => {
                player_spawns.push((node.id, node, room_index));
            }
            NodeKind::SpawnPoint { player: false, .. } => {
                entities.push(PlaytestEntity {
                    room: room_index,
                    kind: PlaytestEntityKind::Marker,
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                    yaw,
                    resource_slot: 0,
                    flags: 0,
                });
            }
            NodeKind::MeshInstance {
                mesh,
                animation_clip,
                ..
            } => {
                // Two cases:
                // (a) `mesh` is `Some(_)` and resolves to a
                //     `ResourceData::Model` → real model
                //     instance, register the model bundle on
                //     first sight and emit a model instance.
                // (b) `mesh` is `None` or points at a non-Model
                //     resource → falls through to a legacy
                //     entity marker so authored placements
                //     don't disappear silently.
                let model_id = mesh.and_then(|id| {
                    project
                        .resource(id)
                        .filter(|r| matches!(r.data, ResourceData::Model(_)))
                        .map(|_| id)
                });
                if let Some(model_resource_id) = model_id {
                    let model_index = match register_model_for_instance(
                        project,
                        project_root,
                        model_resource_id,
                        &mut assets,
                        &mut models,
                        &mut model_clips,
                        &mut model_for_resource,
                        &mut report,
                    ) {
                        Some(i) => i,
                        None => return (None, report),
                    };
                    let model = &models[model_index as usize];
                    let clip = match *animation_clip {
                        Some(idx) => {
                            if (idx as u16) >= model.clip_count {
                                report.error(format!(
                                    "MeshInstance '{}' clip override {idx} out of range (model has {})",
                                    node.name, model.clip_count
                                ));
                                return (None, report);
                            }
                            idx
                        }
                        None => MODEL_CLIP_INHERIT,
                    };
                    model_instances.push(PlaytestModelInstance {
                        room: room_index,
                        model: model_index,
                        clip,
                        x: pos[0],
                        y: pos[1],
                        z: pos[2],
                        yaw,
                        flags: 0,
                    });
                } else {
                    // Legacy / unbound MeshInstance → marker
                    // (matches the pre-Model-resource behaviour).
                    entities.push(PlaytestEntity {
                        room: room_index,
                        kind: PlaytestEntityKind::Marker,
                        x: pos[0],
                        y: pos[1],
                        z: pos[2],
                        yaw,
                        resource_slot: 0,
                        flags: 0,
                    });
                }
            }
            NodeKind::Light {
                color,
                intensity,
                radius,
            } => {
                // Reject obviously broken lights at cook time
                // — radius 0 contributes nothing, negative
                // intensity is meaningless. Clamp the rest into
                // the wire format's u16 ranges.
                if *radius <= 0.0 {
                    report.error(format!(
                        "Light '{}' has radius {} (must be > 0)",
                        node.name, radius
                    ));
                    return (None, report);
                }
                if !intensity.is_finite() || *intensity < 0.0 {
                    report.error(format!(
                        "Light '{}' has invalid intensity {}",
                        node.name, intensity
                    ));
                    return (None, report);
                }
                // Editor radius is in *sector units* — convert
                // to world units (engine units) at cook time so
                // the runtime record stays in one canonical
                // unit regardless of the room's `sector_size`.
                let radius_world =
                    (radius * grid.sector_size as f32).clamp(1.0, u16::MAX as f32) as u16;
                let intensity_q8 = (intensity * 256.0).clamp(0.0, u16::MAX as f32) as u16;
                lights.push(PlaytestLight {
                    room: room_index,
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                    radius: radius_world,
                    intensity_q8,
                    color: *color,
                });
            }
            NodeKind::Trigger { .. } => {
                if warned_unsupported.insert("Trigger") {
                    report.warn("Trigger volumes are skipped in this pass");
                }
            }
            NodeKind::AudioSource { .. } => {
                if warned_unsupported.insert("AudioSource") {
                    report.warn("AudioSource nodes are skipped in this pass");
                }
            }
            NodeKind::Portal { .. } => {
                if warned_unsupported.insert("Portal") {
                    report.warn("Portal nodes are skipped (no streaming yet)");
                }
            }
            NodeKind::Node | NodeKind::Node3D | NodeKind::World | NodeKind::Room { .. } => {}
        }
    }

    let spawn = match player_spawns.len() {
        0 => {
            report.error("playtest needs exactly one SpawnPoint with `player: true` — none found");
            None
        }
        1 => {
            let (_, node, room_index) = player_spawns[0];
            let NodeKind::Room { grid } = &room_nodes
                .iter()
                .find(|r| node_to_room_index.get(&r.id) == Some(&room_index))
                .expect("room index resolved above")
                .kind
            else {
                report.error("internal: player spawn's room kind shifted under us");
                return (None, report);
            };
            let pos = node_room_local_position(node, grid);
            Some(PlaytestSpawn {
                room: room_index,
                x: pos[0],
                y: pos[1],
                z: pos[2],
                yaw: yaw_from_degrees(node.transform.rotation_degrees[1]),
                flags: 1,
            })
        }
        n => {
            report.error(format!(
                "playtest needs exactly one player SpawnPoint, found {n}"
            ));
            None
        }
    };

    // Pass 4: resolve the player's Character, register its
    // model (deduped against MeshInstance-bound models above),
    // and emit a PlaytestCharacter + PlaytestPlayerController.
    //
    // Character resources unrelated to the player aren't cooked
    // in this pass — only the player slot consumes them. Once
    // enemies / NPCs surface, the same `register_model_for_instance`
    // dedupe path handles their backing models too.
    let mut characters: Vec<PlaytestCharacter> = Vec::new();
    let player_controller = match (spawn, &player_spawns[..]) {
        (Some(spawn_record), [(_, spawn_node, _)]) => {
            let NodeKind::SpawnPoint { character, .. } = &spawn_node.kind else {
                report.error("internal: player spawn node kind shifted under us");
                return (None, report);
            };
            // Resolution: explicit assignment > sole Character
            // resource > error. The "exactly one" rule keeps the
            // starter project zero-config while still flagging
            // ambiguity in projects with multiple Characters.
            let resolved = match character {
                Some(id) => Some(*id),
                None => {
                    let candidates: Vec<ResourceId> = project
                        .resources
                        .iter()
                        .filter_map(|r| match &r.data {
                            ResourceData::Character(_) => Some(r.id),
                            _ => None,
                        })
                        .collect();
                    match candidates.len() {
                        1 => {
                            report.warn(format!(
                                "Player Spawn '{}' had no Character — auto-picked the only one defined",
                                spawn_node.name,
                            ));
                            Some(candidates[0])
                        }
                        0 => {
                            report.error(format!(
                                "Player Spawn '{}' has no Character assigned and no Character resources exist",
                                spawn_node.name
                            ));
                            None
                        }
                        n => {
                            report.error(format!(
                                "Player Spawn '{}' has no Character assigned and {n} Characters are defined — pick one explicitly",
                                spawn_node.name
                            ));
                            None
                        }
                    }
                }
            };
            match resolved.and_then(|id| {
                cook_player_character(
                    project,
                    project_root,
                    spawn_node,
                    id,
                    &mut assets,
                    &mut models,
                    &mut model_clips,
                    &mut model_for_resource,
                    &mut characters,
                    &mut report,
                )
            }) {
                Some(character_index) => Some(PlaytestPlayerController {
                    spawn: spawn_record,
                    character: character_index,
                }),
                None => None,
            }
        }
        _ => None,
    };

    if !report.is_ok() {
        return (None, report);
    }

    (
        Some(PlaytestPackage {
            assets,
            rooms,
            materials,
            models,
            model_clips,
            model_instances,
            lights,
            spawn,
            characters,
            player_controller,
            entities,
        }),
        report,
    )
}

/// Cook one Character resource into a [`PlaytestCharacter`],
/// registering its backing model on first sight (deduped against
/// MeshInstance placements). Validates clip indices land inside
/// the resolved model's clip slice; the runtime trusts the
/// contract.
#[allow(clippy::too_many_arguments)]
fn cook_player_character(
    project: &ProjectDocument,
    project_root: &Path,
    spawn_node: &SceneNode,
    character_id: ResourceId,
    assets: &mut Vec<PlaytestAsset>,
    models: &mut Vec<PlaytestModel>,
    model_clips: &mut Vec<PlaytestModelClip>,
    model_for_resource: &mut std::collections::HashMap<ResourceId, u16>,
    characters: &mut Vec<PlaytestCharacter>,
    report: &mut PlaytestValidationReport,
) -> Option<u16> {
    let resource = match project.resource(character_id) {
        Some(r) => r,
        None => {
            report.error(format!(
                "Player Spawn '{}' references Character #{} which doesn't exist",
                spawn_node.name,
                character_id.raw()
            ));
            return None;
        }
    };
    let character = match &resource.data {
        ResourceData::Character(c) => c,
        _ => {
            report.error(format!(
                "Player Spawn '{}' references resource '{}' which is not a Character",
                spawn_node.name, resource.name
            ));
            return None;
        }
    };

    let model_resource_id = match character.model {
        Some(id) => id,
        None => {
            report.error(format!(
                "Character '{}' has no Model assigned — required for the player",
                resource.name
            ));
            return None;
        }
    };
    let model_index = register_model_for_instance(
        project,
        project_root,
        model_resource_id,
        assets,
        models,
        model_clips,
        model_for_resource,
        report,
    )?;
    let model = &models[model_index as usize];

    let clip_count = model.clip_count;
    let validate_required =
        |role: &str, slot: Option<u16>, report: &mut PlaytestValidationReport| -> Option<u16> {
            match slot {
                Some(idx) if idx < clip_count => Some(idx),
                Some(idx) => {
                    report.error(format!(
                    "Character '{}' {role} clip {idx} out of range ({clip_count} clips on model)",
                    resource.name
                ));
                    None
                }
                None => {
                    report.error(format!(
                        "Character '{}' has no {role} clip assigned",
                        resource.name
                    ));
                    None
                }
            }
        };
    let validate_optional =
        |role: &str, slot: Option<u16>, report: &mut PlaytestValidationReport| -> u16 {
            match slot {
                Some(idx) if idx < clip_count => idx,
                Some(idx) => {
                    report.error(format!(
                    "Character '{}' {role} clip {idx} out of range ({clip_count} clips on model)",
                    resource.name
                ));
                    CHARACTER_CLIP_NONE
                }
                None => CHARACTER_CLIP_NONE,
            }
        };
    let idle_clip = validate_required("idle", character.idle_clip, report)?;
    let walk_clip = validate_required("walk", character.walk_clip, report)?;
    let run_clip = validate_optional("run", character.run_clip, report);
    let turn_clip = validate_optional("turn", character.turn_clip, report);

    if character.radius == 0 {
        report.error(format!("Character '{}' radius must be > 0", resource.name));
        return None;
    }
    if character.height == 0 {
        report.error(format!("Character '{}' height must be > 0", resource.name));
        return None;
    }
    if character.walk_speed <= 0 {
        report.error(format!(
            "Character '{}' walk_speed must be > 0",
            resource.name
        ));
        return None;
    }
    if character.turn_speed_degrees_per_second == 0 {
        report.error(format!(
            "Character '{}' turn_speed must be > 0",
            resource.name
        ));
        return None;
    }
    if character.camera_distance <= 0 {
        report.error(format!(
            "Character '{}' camera_distance must be > 0",
            resource.name
        ));
        return None;
    }
    if character.camera_height < 0 || character.camera_target_height < 0 {
        report.error(format!(
            "Character '{}' camera offsets must be >= 0",
            resource.name
        ));
        return None;
    }

    if run_clip == CHARACTER_CLIP_NONE {
        report.warn(format!(
            "Character '{}' has no run clip — runtime will fall back to walk for run input",
            resource.name
        ));
    }
    if turn_clip == CHARACTER_CLIP_NONE {
        report.warn(format!("Character '{}' has no turn clip", resource.name));
    }

    let character_index = u16::try_from(characters.len()).unwrap_or(u16::MAX);
    characters.push(PlaytestCharacter {
        source_resource: character_id,
        model: model_index,
        idle_clip,
        walk_clip,
        run_clip,
        turn_clip,
        radius: character.radius,
        height: character.height,
        walk_speed: character.walk_speed,
        run_speed: character.run_speed,
        turn_speed_degrees_per_second: character.turn_speed_degrees_per_second,
        camera_distance: character.camera_distance,
        camera_height: character.camera_height,
        camera_target_height: character.camera_target_height,
    });
    Some(character_index)
}

/// Register a `ResourceData::Model` into the playtest package
/// on first sight; reuse the cached index otherwise. On
/// success, returns the model's index in `models`.
///
/// Failures (missing files, invalid blobs, joint-count
/// mismatches) push to `report.errors` and return `None`; the
/// caller turns that into a hard cook failure.
fn register_model_for_instance(
    project: &ProjectDocument,
    project_root: &Path,
    model_resource_id: ResourceId,
    assets: &mut Vec<PlaytestAsset>,
    models: &mut Vec<PlaytestModel>,
    model_clips: &mut Vec<PlaytestModelClip>,
    model_for_resource: &mut std::collections::HashMap<ResourceId, u16>,
    report: &mut PlaytestValidationReport,
) -> Option<u16> {
    if let Some(&existing) = model_for_resource.get(&model_resource_id) {
        return Some(existing);
    }
    let resource = project.resource(model_resource_id)?;
    let ResourceData::Model(model) = &resource.data else {
        report.error(format!(
            "MeshInstance references resource #{} which is not a Model",
            model_resource_id.raw()
        ));
        return None;
    };

    // Runtime contract: a placed model must carry an atlas
    // (the runtime renders textured) and at least one clip
    // (the runtime renders animated). Bind-pose / untextured
    // rendering would need engine-side work the current pass
    // doesn't ship — fail loud at cook so the editor surfaces
    // it rather than silently dropping the instance at runtime.
    if model.texture_path.is_none() {
        report.error(format!(
            "Model '{}' has no atlas; the runtime can't render untextured models in this pass",
            resource.name
        ));
        return None;
    }
    if model.clips.is_empty() {
        report.error(format!(
            "Model '{}' has no animation clips; the runtime requires at least one clip",
            resource.name
        ));
        return None;
    }

    let model_index = u16::try_from(models.len()).unwrap_or(u16::MAX);
    let safe = sanitise_model_dirname(&resource.name);
    let folder = format!("{MODELS_DIRNAME}/model_{:03}_{safe}", model_index);

    // Mesh asset.
    let mesh_path = resolve_path(&model.model_path, project_root);
    let mesh_bytes = match std::fs::read(&mesh_path) {
        Ok(b) => b,
        Err(e) => {
            report.error(format!(
                "Model '{}' mesh {}: {e}",
                resource.name,
                mesh_path.display()
            ));
            return None;
        }
    };
    let parsed_model = match psx_asset::Model::from_bytes(&mesh_bytes) {
        Ok(m) => m,
        Err(e) => {
            report.error(format!(
                "Model '{}' mesh parse failed: {e:?}",
                resource.name
            ));
            return None;
        }
    };
    let model_joint_count = parsed_model.joint_count();
    let mesh_asset_index = assets.len();
    assets.push(PlaytestAsset {
        kind: PlaytestAssetKind::ModelMesh,
        bytes: mesh_bytes,
        filename: format!("{folder}/mesh.psxmdl"),
        source_label: resource.name.clone(),
    });

    // Atlas asset (optional).
    let texture_asset_index = if let Some(tex_path) = &model.texture_path {
        let abs = resolve_path(tex_path, project_root);
        let bytes = match std::fs::read(&abs) {
            Ok(b) => b,
            Err(e) => {
                report.error(format!(
                    "Model '{}' atlas {}: {e}",
                    resource.name,
                    abs.display()
                ));
                return None;
            }
        };
        let parsed_atlas = match psx_asset::Texture::from_bytes(&bytes) {
            Ok(t) => t,
            Err(e) => {
                report.error(format!(
                    "Model '{}' atlas parse failed: {e:?}",
                    resource.name
                ));
                return None;
            }
        };
        // Model atlases must be 8bpp (256-entry CLUT) — the
        // runtime model atlas region uses an 8bpp tpage and a
        // 256-entry CLUT row per atlas. Other depths render with
        // wrong colours, so reject loud at cook time.
        if parsed_atlas.clut_entries() != 256 {
            report.error(format!(
                "Model '{}' atlas must be 8bpp (256-entry CLUT); found {} entries",
                resource.name,
                parsed_atlas.clut_entries(),
            ));
            return None;
        }
        let idx = assets.len();
        assets.push(PlaytestAsset {
            kind: PlaytestAssetKind::Texture,
            bytes,
            filename: format!("{folder}/atlas.psxt"),
            source_label: format!("{} atlas", resource.name),
        });
        Some(idx)
    } else {
        None
    };

    // Clip assets — one .psxanim per clip, validated for joint
    // parity. Clips ordered as authored in the resource.
    let clip_first = u16::try_from(model_clips.len()).unwrap_or(u16::MAX);
    for (i, clip) in model.clips.iter().enumerate() {
        let abs = resolve_path(&clip.psxanim_path, project_root);
        let bytes = match std::fs::read(&abs) {
            Ok(b) => b,
            Err(e) => {
                report.error(format!(
                    "Model '{}' clip '{}' {}: {e}",
                    resource.name,
                    clip.name,
                    abs.display()
                ));
                return None;
            }
        };
        let parsed_anim = match psx_asset::Animation::from_bytes(&bytes) {
            Ok(a) => a,
            Err(e) => {
                report.error(format!(
                    "Model '{}' clip '{}' parse failed: {e:?}",
                    resource.name, clip.name
                ));
                return None;
            }
        };
        if parsed_anim.joint_count() != model_joint_count {
            report.error(format!(
                "Model '{}' clip '{}': animation has {} joints, model has {}",
                resource.name,
                clip.name,
                parsed_anim.joint_count(),
                model_joint_count
            ));
            return None;
        }
        let asset_index = assets.len();
        let safe_clip = sanitise_model_dirname(&clip.name);
        assets.push(PlaytestAsset {
            kind: PlaytestAssetKind::ModelAnimation,
            bytes,
            filename: format!("{folder}/clip_{:02}_{safe_clip}.psxanim", i),
            source_label: format!("{} / {}", resource.name, clip.name),
        });
        model_clips.push(PlaytestModelClip {
            model: model_index,
            name: clip.name.clone(),
            animation_asset_index: asset_index,
        });
    }
    let clip_count = u16::try_from(model_clips.len() - clip_first as usize).unwrap_or(u16::MAX);

    // Resolve the model's default clip. Validation rules:
    //   - explicit `model.default_clip = Some(idx)` MUST be in
    //     range; out-of-range is a hard cook error so the user
    //     fixes the resource rather than a runtime instance
    //     silently pointing at clip 0.
    //   - `None` falls back to clip 0. Cooker has already
    //     refused empty-clip placed models, so `clip_count >= 1`.
    let default_clip = match model.default_clip {
        Some(idx) => {
            if idx >= clip_count {
                report.error(format!(
                    "Model '{}' default_clip {idx} is out of range ({} clips)",
                    resource.name, clip_count
                ));
                return None;
            }
            idx
        }
        None => 0,
    };

    models.push(PlaytestModel {
        name: resource.name.clone(),
        source_resource: model_resource_id,
        mesh_asset_index,
        texture_asset_index,
        clip_first,
        clip_count,
        default_clip,
        world_height: model.world_height,
    });
    model_for_resource.insert(model_resource_id, model_index);
    Some(model_index)
}

/// Resolve a path the same way the texture loader does so the
/// model writer / runtime stay in lockstep.
fn resolve_path(stored: &str, project_root: &Path) -> PathBuf {
    if Path::new(stored).is_absolute() {
        PathBuf::from(stored)
    } else {
        project_root.join(stored)
    }
}

/// Strip a free-form name down to a filesystem-safe stem.
fn sanitise_model_dirname(name: &str) -> String {
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

/// Suppress the legacy unused-`ModelResource` import once
/// build_package no longer references the type. Kept here so
/// downstream consumers can still reach it via a single module
/// path.
#[allow(dead_code)]
fn _model_resource_re_export(_: &ModelResource) {}

/// Write `package` and the generated manifest to `generated_dir`.
/// Creates `generated_dir/rooms/` and `generated_dir/textures/`,
/// strips any stale `.psxw` / `.psxt` files inside them, and
/// writes the manifest source last (so a half-failed write
/// doesn't leave a manifest pointing at missing files).
pub fn write_package(package: &PlaytestPackage, generated_dir: &Path) -> std::io::Result<()> {
    let rooms_dir = generated_dir.join(ROOMS_DIRNAME);
    let textures_dir = generated_dir.join(TEXTURES_DIRNAME);
    let models_dir = generated_dir.join(MODELS_DIRNAME);
    std::fs::create_dir_all(&rooms_dir)?;
    std::fs::create_dir_all(&textures_dir)?;
    std::fs::create_dir_all(&models_dir)?;
    purge_directory_files(&rooms_dir, "psxw")?;
    purge_directory_files(&textures_dir, "psxt")?;
    // Models live in per-model subfolders so the recursive
    // purge needs to traverse one level deeper than rooms /
    // textures.
    purge_models_dir(&models_dir)?;

    for asset in &package.assets {
        // ModelMesh / ModelAnimation / model-folder Texture
        // asset filenames already include their `models/...`
        // subpath; rooms + room-only textures stay flat in
        // their respective dirs.
        let target = match asset.kind {
            PlaytestAssetKind::RoomWorld => rooms_dir.join(&asset.filename),
            PlaytestAssetKind::Texture if asset.filename.contains('/') => {
                generated_dir.join(&asset.filename)
            }
            PlaytestAssetKind::Texture => textures_dir.join(&asset.filename),
            PlaytestAssetKind::ModelMesh | PlaytestAssetKind::ModelAnimation => {
                generated_dir.join(&asset.filename)
            }
        };
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, &asset.bytes)?;
    }

    let manifest = render_manifest_source(package);
    std::fs::write(generated_dir.join(MANIFEST_FILENAME), manifest)?;
    Ok(())
}

/// Render `package` as a Rust source string the runtime example
/// can `include!`. Imports types from `psx_level` rather than
/// re-defining them so the writer here and the reader there
/// stay in lockstep.
pub fn render_manifest_source(package: &PlaytestPackage) -> String {
    let mut out = String::new();
    out.push_str(MANIFEST_HEADER);

    // Emit one named static per asset so the include_bytes! call
    // sites are easy to grep for. Asset records reference these
    // statics so the slice is still constructible at compile time.
    for (i, asset) in package.assets.iter().enumerate() {
        let include_path = match asset.kind {
            PlaytestAssetKind::RoomWorld => format!("{ROOMS_DIRNAME}/{}", asset.filename),
            PlaytestAssetKind::Texture if asset.filename.contains('/') => asset.filename.clone(),
            PlaytestAssetKind::Texture => format!("{TEXTURES_DIRNAME}/{}", asset.filename),
            PlaytestAssetKind::ModelMesh | PlaytestAssetKind::ModelAnimation => {
                asset.filename.clone()
            }
        };
        let _ = writeln!(
            out,
            "/// {} — {}",
            asset_static_name(asset, i),
            asset.source_label,
        );
        let _ = writeln!(
            out,
            "pub static {}: &[u8] = include_bytes!(\"{include_path}\");",
            asset_static_name(asset, i),
        );
    }
    out.push('\n');

    out.push_str("/// Master asset table.\n");
    out.push_str("pub static ASSETS: &[LevelAssetRecord] = &[\n");
    for (i, asset) in package.assets.iter().enumerate() {
        let kind = match asset.kind {
            PlaytestAssetKind::RoomWorld => "AssetKind::RoomWorld",
            PlaytestAssetKind::Texture => "AssetKind::Texture",
            PlaytestAssetKind::ModelMesh => "AssetKind::ModelMesh",
            PlaytestAssetKind::ModelAnimation => "AssetKind::ModelAnimation",
        };
        let static_name = asset_static_name(asset, i);
        let vram_bytes = match asset.kind {
            PlaytestAssetKind::RoomWorld
            | PlaytestAssetKind::ModelMesh
            | PlaytestAssetKind::ModelAnimation => 0,
            // Heuristic: the upload cost is the texture's byte
            // length. Future loaders may compute a tighter
            // figure — for now this is the conservative bound
            // the residency budget has to honour.
            PlaytestAssetKind::Texture => asset.bytes.len(),
        };
        let _ = writeln!(
            out,
            "    LevelAssetRecord {{ id: AssetId({i}), kind: {kind}, bytes: {static_name}, ram_bytes: {static_name}.len() as u32, vram_bytes: {vram_bytes}, flags: 0 }},"
        );
    }
    out.push_str("];\n\n");

    out.push_str("/// Per-room material bindings — slot the `.psxw` stores → texture asset.\n");
    out.push_str("pub static MATERIALS: &[LevelMaterialRecord] = &[\n");
    for material in &package.materials {
        let _ = writeln!(
            out,
            "    LevelMaterialRecord {{ room: {}, local_slot: {}, texture_asset: AssetId({}), tint_rgb: [{}, {}, {}], flags: 0 }},",
            material.room,
            material.local_slot,
            material.texture_asset_index,
            material.tint_rgb[0],
            material.tint_rgb[1],
            material.tint_rgb[2],
        );
    }
    out.push_str("];\n\n");

    out.push_str("/// Rooms with material-slice metadata.\n");
    out.push_str("pub static ROOMS: &[LevelRoomRecord] = &[\n");
    for room in &package.rooms {
        let _ = writeln!(
            out,
            "    LevelRoomRecord {{ name: {:?}, world_asset: AssetId({}), origin_x: {}, origin_z: {}, sector_size: {}, material_first: {}, material_count: {}, flags: 0 }},",
            room.name,
            room.world_asset_index,
            room.origin_x,
            room.origin_z,
            room.sector_size,
            room.material_first,
            room.material_count,
        );
    }
    out.push_str("];\n\n");

    // Per-room residency: required RAM = the room's world
    // asset + every model mesh + every animation clip
    // referenced by an instance OR by the player character in
    // this room; required VRAM = every distinct texture asset
    // (room materials + model atlases) referenced by this room.
    for (i, room) in package.rooms.iter().enumerate() {
        let first = room.material_first as usize;
        let count = room.material_count as usize;
        let mut required_vram: Vec<usize> = Vec::with_capacity(count);
        for material in &package.materials[first..first + count] {
            if !required_vram.contains(&material.texture_asset_index) {
                required_vram.push(material.texture_asset_index);
            }
        }
        let mut required_ram: Vec<usize> = vec![room.world_asset_index];

        // Models the room references — placed MeshInstance
        // bindings *plus* the player controller's character
        // when its spawn lives in this room. The player's
        // model is residency-required even when it's never
        // placed as a regular MeshInstance; otherwise the
        // runtime would render the player from un-resident
        // bytes the moment the room loads.
        let i_u16 = i as u16;
        let mut seen_models: Vec<u16> = Vec::new();
        for inst in &package.model_instances {
            if inst.room != i_u16 || seen_models.contains(&inst.model) {
                continue;
            }
            seen_models.push(inst.model);
            include_model_in_residency(package, inst.model, &mut required_ram, &mut required_vram);
        }
        if let Some(pc) = package.player_controller {
            if pc.spawn.room == i_u16 {
                let model = package.characters[pc.character as usize].model;
                if !seen_models.contains(&model) {
                    seen_models.push(model);
                    include_model_in_residency(
                        package,
                        model,
                        &mut required_ram,
                        &mut required_vram,
                    );
                }
            }
        }

        let _ = writeln!(out, "/// Room {i} required RAM assets.");
        out.push_str(&format!(
            "pub static ROOM_{i}_REQUIRED_RAM: &[AssetId] = &["
        ));
        for (j, idx) in required_ram.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "AssetId({idx})");
        }
        out.push_str("];\n");
        let _ = writeln!(out, "/// Room {i} required VRAM assets.");
        out.push_str(&format!(
            "pub static ROOM_{i}_REQUIRED_VRAM: &[AssetId] = &["
        ));
        for (j, idx) in required_vram.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "AssetId({idx})");
        }
        out.push_str("];\n");
    }
    out.push('\n');

    out.push_str("/// Per-room residency contract.\n");
    out.push_str("pub static ROOM_RESIDENCY: &[RoomResidencyRecord] = &[\n");
    for (i, _room) in package.rooms.iter().enumerate() {
        let _ = writeln!(
            out,
            "    RoomResidencyRecord {{ room: {i}, required_ram: ROOM_{i}_REQUIRED_RAM, required_vram: ROOM_{i}_REQUIRED_VRAM, warm_ram: &[], warm_vram: &[] }},",
        );
    }
    out.push_str("];\n\n");

    let spawn = package.spawn.unwrap_or(PlaytestSpawn {
        room: 0,
        x: 0,
        y: 0,
        z: 0,
        yaw: 0,
        flags: 0,
    });
    let _ = writeln!(
        out,
        "/// Player spawn.\npub static PLAYER_SPAWN: PlayerSpawnRecord = PlayerSpawnRecord {{ room: {}, x: {}, y: {}, z: {}, yaw: {}, flags: {} }};",
        spawn.room, spawn.x, spawn.y, spawn.z, spawn.yaw, spawn.flags
    );
    out.push('\n');

    // MODELS / MODEL_CLIPS / MODEL_INSTANCES — emitted as
    // empty slices when there are no model instances, so the
    // runtime always has something to walk.
    out.push_str("/// Per-model clip records, ordered (model, clip).\n");
    out.push_str("pub static MODEL_CLIPS: &[LevelModelClipRecord] = &[\n");
    for clip in &package.model_clips {
        let _ = writeln!(
            out,
            "    LevelModelClipRecord {{ model: {}, name: {:?}, animation_asset: AssetId({}) }},",
            clip.model, clip.name, clip.animation_asset_index,
        );
    }
    out.push_str("];\n\n");

    out.push_str("/// Cooked models — instances reference these by index.\n");
    out.push_str("pub static MODELS: &[LevelModelRecord] = &[\n");
    for model in &package.models {
        let texture = match model.texture_asset_index {
            Some(idx) => format!("Some(AssetId({idx}))"),
            None => "None".to_string(),
        };
        let _ = writeln!(
            out,
            "    LevelModelRecord {{ name: {:?}, mesh_asset: AssetId({}), texture_asset: {texture}, clip_first: {}, clip_count: {}, default_clip: {}, world_height: {}, flags: 0 }},",
            model.name,
            model.mesh_asset_index,
            model.clip_first,
            model.clip_count,
            model.default_clip,
            model.world_height,
        );
    }
    out.push_str("];\n\n");

    out.push_str("/// Placed model instances, room-local coordinates.\n");
    out.push_str("pub static MODEL_INSTANCES: &[LevelModelInstanceRecord] = &[\n");
    for inst in &package.model_instances {
        let _ = writeln!(
            out,
            "    LevelModelInstanceRecord {{ room: {}, model: {}, clip: {}, x: {}, y: {}, z: {}, yaw: {}, flags: {} }},",
            inst.room, inst.model, inst.clip, inst.x, inst.y, inst.z, inst.yaw, inst.flags,
        );
    }
    out.push_str("];\n\n");

    out.push_str("/// Placed point lights, room-local coordinates.\n");
    out.push_str("pub static LIGHTS: &[PointLightRecord] = &[\n");
    for light in &package.lights {
        let _ = writeln!(
            out,
            "    PointLightRecord {{ room: {}, x: {}, y: {}, z: {}, radius: {}, intensity_q8: {}, color: [{}, {}, {}], flags: 0 }},",
            light.room,
            light.x,
            light.y,
            light.z,
            light.radius,
            light.intensity_q8,
            light.color[0],
            light.color[1],
            light.color[2],
        );
    }
    out.push_str("];\n\n");

    out.push_str("/// Cooked Character resources — gameplay metadata layered on top of MODELS.\n");
    out.push_str("pub static CHARACTERS: &[LevelCharacterRecord] = &[\n");
    for character in &package.characters {
        let clip_or_none = |slot: u16| -> String {
            if slot == CHARACTER_CLIP_NONE {
                "CHARACTER_CLIP_NONE".to_string()
            } else {
                format!("{slot}")
            }
        };
        let _ = writeln!(
            out,
            "    LevelCharacterRecord {{ model: {}, idle_clip: {}, walk_clip: {}, run_clip: {}, turn_clip: {}, radius: {}, height: {}, walk_speed: {}, run_speed: {}, turn_speed_degrees_per_second: {}, camera_distance: {}, camera_height: {}, camera_target_height: {}, flags: 0 }},",
            character.model,
            character.idle_clip,
            character.walk_clip,
            clip_or_none(character.run_clip),
            clip_or_none(character.turn_clip),
            character.radius,
            character.height,
            character.walk_speed,
            character.run_speed,
            character.turn_speed_degrees_per_second,
            character.camera_distance,
            character.camera_height,
            character.camera_target_height,
        );
    }
    out.push_str("];\n\n");

    match package.player_controller {
        Some(pc) => {
            let _ = writeln!(
                out,
                "/// Player controller — spawn + Character that drives the player.\npub static PLAYER_CONTROLLER: Option<PlayerControllerRecord> = Some(PlayerControllerRecord {{ spawn: PlayerSpawnRecord {{ room: {}, x: {}, y: {}, z: {}, yaw: {}, flags: {} }}, character: {}, flags: 0 }});",
                pc.spawn.room, pc.spawn.x, pc.spawn.y, pc.spawn.z, pc.spawn.yaw, pc.spawn.flags, pc.character,
            );
        }
        None => {
            out.push_str(
                "/// Player controller — `None` means no playable character was authored.\n\
                pub static PLAYER_CONTROLLER: Option<PlayerControllerRecord> = None;\n",
            );
        }
    }
    out.push('\n');

    out.push_str("/// Entity markers (legacy MeshInstance with no Model resource).\n");
    out.push_str("pub static ENTITIES: &[EntityRecord] = &[\n");
    for entity in &package.entities {
        let kind = match entity.kind {
            PlaytestEntityKind::Marker => "EntityKind::Marker",
            PlaytestEntityKind::StaticMesh => "EntityKind::StaticMesh",
        };
        let _ = writeln!(
            out,
            "    EntityRecord {{ room: {}, kind: {kind}, x: {}, y: {}, z: {}, yaw: {}, resource_slot: {}, flags: {} }},",
            entity.room, entity.x, entity.y, entity.z, entity.yaw, entity.resource_slot, entity.flags
        );
    }
    out.push_str("];\n");
    out
}

/// Default destination for the playtest example's generated
/// directory. Anchored at the editor crate's manifest dir so the
/// dev workflow finds it regardless of cwd.
pub fn default_generated_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("engine")
        .join("examples")
        .join("editor-playtest")
        .join(GENERATED_DIRNAME)
}

/// One-shot cook + write entry point: validate, package, drop
/// the result at `generated_dir`. Resolves relative texture
/// paths through `project_root`. Returns the validation report;
/// callers must check `report.is_ok()` before assuming the
/// files were written.
pub fn cook_to_dir(
    project: &ProjectDocument,
    project_root: &Path,
    generated_dir: &Path,
) -> std::io::Result<PlaytestValidationReport> {
    let (package, report) = build_package(project, project_root);
    if let Some(package) = package {
        write_package(&package, generated_dir)?;
    }
    Ok(report)
}

/// Header emitted at the top of every generated manifest. The
/// runtime example wraps the `include!` in a `mod generated`
/// with `#[allow(dead_code)]` on the wrapper, so we don't
/// repeat that here (would be an inner attribute on the wrong
/// item).
const MANIFEST_HEADER: &str = "\
// Generated by `psxed_project::playtest::write_package` —
// do not edit by hand. Regenerate with the editor's
// `Cook & Play` action or the `cook-playtest` CLI.

use psx_level::{
    AssetId,
    AssetKind,
    CHARACTER_CLIP_NONE,
    EntityKind,
    EntityRecord,
    LevelAssetRecord,
    LevelCharacterRecord,
    LevelMaterialRecord,
    LevelModelClipRecord,
    LevelModelInstanceRecord,
    LevelModelRecord,
    LevelRoomRecord,
    PlayerControllerRecord,
    PlayerSpawnRecord,
    PointLightRecord,
    RoomResidencyRecord,
};

";

/// Add `model_index`'s mesh + atlas + every clip to a room's
/// residency lists. Idempotent through the caller's seen-set
/// — also dedupes within `required_ram` / `required_vram` so
/// callers don't have to.
///
/// Pulled out so the per-room walk can register both placed
/// MeshInstance models and the player character's model
/// without duplicating bookkeeping. Without the player path,
/// a Character whose backing model isn't also placed as a
/// MeshInstance would be missing from residency entirely —
/// the runtime would then render the player from un-resident
/// bytes the moment the room loaded.
fn include_model_in_residency(
    package: &PlaytestPackage,
    model_index: u16,
    required_ram: &mut Vec<usize>,
    required_vram: &mut Vec<usize>,
) {
    let Some(model) = package.models.get(model_index as usize) else {
        return;
    };
    if !required_ram.contains(&model.mesh_asset_index) {
        required_ram.push(model.mesh_asset_index);
    }
    if let Some(atlas) = model.texture_asset_index {
        if !required_vram.contains(&atlas) {
            required_vram.push(atlas);
        }
    }
    let cf = model.clip_first as usize;
    let cc = model.clip_count as usize;
    if cf + cc > package.model_clips.len() {
        return;
    }
    for clip in &package.model_clips[cf..cf + cc] {
        if !required_ram.contains(&clip.animation_asset_index) {
            required_ram.push(clip.animation_asset_index);
        }
    }
}

/// Resolve the per-asset `static` name for the include_bytes
/// statement. Mirrors the filename so a reader can grep
/// `ROOM_000_BYTES` and immediately know it points at
/// `rooms/room_000.psxw`. The `_index` parameter is reserved
/// for future asset kinds with no filename component.
fn asset_static_name(asset: &PlaytestAsset, _index: usize) -> String {
    let stem = Path::new(&asset.filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&asset.filename);
    format!("{}_BYTES", stem.to_ascii_uppercase())
}

/// Find the resource record by id. Linear scan — resource
/// counts in this project are small (a few materials + a few
/// textures); a HashMap wrapper would buy nothing.
fn find_resource(project: &ProjectDocument, id: ResourceId) -> Option<&Resource> {
    project.resources.iter().find(|r| r.id == id)
}

/// Read the texture's `.psxt` bytes from disk. Resolves
/// `psxt_path` first as-is (absolute paths), then relative to
/// `project_root`. Returns a string error rather than `io::Error`
/// so callers can prepend room/material context.
/// Validate a room-material `.psxt` blob is 4bpp (16-entry
/// CLUT). Both the editor preview material upload path and the
/// runtime room material slots assume 4bpp; other depths
/// render with wrong colours.
fn expect_room_material_depth(resource: &Resource, bytes: &[u8]) -> Result<(), String> {
    let texture = psx_asset::Texture::from_bytes(bytes)
        .map_err(|e| format!("texture '{}' parse failed: {e:?}", resource.name))?;
    if texture.clut_entries() != 16 {
        return Err(format!(
            "texture '{}' must be 4bpp (16-entry CLUT) for room materials; found {} entries",
            resource.name,
            texture.clut_entries(),
        ));
    }
    Ok(())
}

fn load_texture_bytes(resource: &Resource, project_root: &Path) -> Result<Vec<u8>, String> {
    let ResourceData::Texture { psxt_path } = &resource.data else {
        return Err(format!(
            "resource '{}' (#{}) is not a Texture",
            resource.name,
            resource.id.raw(),
        ));
    };
    if psxt_path.is_empty() {
        return Err(format!(
            "texture resource '{}' has empty path",
            resource.name
        ));
    }
    let path = if Path::new(psxt_path).is_absolute() {
        PathBuf::from(psxt_path)
    } else {
        project_root.join(psxt_path)
    };
    std::fs::read(&path).map_err(|e| {
        format!(
            "failed to read texture '{}' at {}: {e}",
            resource.name,
            path.display(),
        )
    })
}

/// Remove every file with extension `ext` directly inside `dir`.
/// Used before writing fresh assets so stale `room_NNN.psxw` /
/// `texture_NNN.psxt` files don't survive a rename / removal.
fn purge_directory_files(dir: &Path, ext: &str) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}

/// Purge stale per-model subfolders inside `generated/models/`.
/// Each cook re-creates `model_NNN_<safe>/` folders from scratch,
/// so the simplest safe behaviour is to remove every immediate
/// subdirectory before writing.
fn purge_models_dir(dir: &Path) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            std::fs::remove_dir_all(&path)?;
        }
    }
    Ok(())
}

/// Walk parent links until a Room node is reached.
fn enclosing_room<'a>(
    scene: &'a crate::Scene,
    node: &'a SceneNode,
    node_to_room_index: &std::collections::HashMap<NodeId, u16>,
) -> Option<(&'a SceneNode, u16)> {
    let mut current = node.parent;
    while let Some(parent_id) = current {
        let parent = scene.node(parent_id)?;
        if let Some(&room_index) = node_to_room_index.get(&parent.id) {
            return Some((parent, room_index));
        }
        current = parent.parent;
    }
    None
}

/// Convert a node's editor-space transform to room-local engine units.
fn node_room_local_position(node: &SceneNode, grid: &WorldGrid) -> [i32; 3] {
    let s = grid.sector_size as f32;
    let half_w = grid.width as f32 * 0.5;
    let half_d = grid.depth as f32 * 0.5;
    let x = (node.transform.translation[0] + half_w) * s;
    let y = node.transform.translation[1] * s;
    let z = (node.transform.translation[2] + half_d) * s;
    [x as i32, y as i32, z as i32]
}

/// Convert an editor euler-degrees-Y rotation to a PSX angle
/// unit (`0..4096`).
fn yaw_from_degrees(degrees: f32) -> i16 {
    let normalised = degrees.rem_euclid(360.0);
    let units = normalised * (4096.0 / 360.0);
    units as i16
}

fn cook_error_for_node(name: &str, err: WorldGridCookError) -> String {
    format!("Room '{name}' failed cook: {err}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodeKind, ProjectDocument};

    fn starter_project_root() -> PathBuf {
        crate::default_project_dir()
    }

    fn project_with_one_room() -> ProjectDocument {
        let project = ProjectDocument::starter();
        let scene = project.active_scene();
        let has_room = scene
            .nodes()
            .iter()
            .any(|n| matches!(n.kind, NodeKind::Room { .. }));
        let has_player_spawn = scene
            .nodes()
            .iter()
            .any(|n| matches!(n.kind, NodeKind::SpawnPoint { player: true, .. }));
        assert!(has_room, "starter must contain a Room");
        assert!(has_player_spawn, "starter must contain a player SpawnPoint");
        project
    }

    #[test]
    fn starter_project_validates_and_cooks() {
        let project = project_with_one_room();
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("package returned on ok report");
        assert_eq!(package.rooms.len(), 1);
        assert_eq!(package.room_asset_count(), 1);
        assert!(package.spawn.is_some());
    }

    #[test]
    fn starter_project_emits_player_controller_and_character() {
        let project = project_with_one_room();
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("package returned on ok report");
        assert_eq!(
            package.characters.len(),
            1,
            "starter ships exactly one Character (Wraith Hero)"
        );
        let pc = package
            .player_controller
            .expect("player controller emitted");
        assert_eq!(pc.character, 0);
        assert_eq!(pc.spawn, package.spawn.unwrap());
        let character = &package.characters[0];
        // Wraith model: idle at index 3, walking at index 7.
        assert_eq!(character.idle_clip, 3);
        assert_eq!(character.walk_clip, 7);
        // Run is `running` clip (4); turn is unset.
        assert_eq!(character.run_clip, 4);
        assert_eq!(character.turn_clip, CHARACTER_CLIP_NONE);
    }

    #[test]
    fn player_character_model_is_deduplicated_with_placed_meshinstance() {
        // Starter includes both a Wraith MeshInstance (id 6 in
        // the scene) and a Wraith-Hero Character resource — they
        // share the same Model resource. The cooker must register
        // the model once (in `models`), not twice.
        let project = project_with_one_room();
        let (package, _report) = build_package(&project, &starter_project_root());
        let package = package.expect("starter cooks");
        assert_eq!(
            package.models.len(),
            1,
            "shared model should be registered once across MeshInstance + Character"
        );
        // Player character + placed instance both reference model index 0.
        assert_eq!(package.characters[0].model, 0);
        assert!(package.model_instances.iter().any(|inst| inst.model == 0));
    }

    #[test]
    fn player_character_model_lands_in_room_residency_without_placed_meshinstance() {
        // Simulate a project where the player Character points
        // at a Model that *isn't* also placed as a MeshInstance.
        // The starter has both, so we delete the placed Wraith
        // before cooking and assert residency still picks up the
        // Wraith mesh + atlas + clips via the player path.
        let mut project = project_with_one_room();
        let scene = project.active_scene_mut();
        let placed_ids: Vec<NodeId> = scene
            .nodes()
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::MeshInstance { mesh: Some(_), .. }))
            .map(|n| n.id)
            .collect();
        for id in placed_ids {
            scene.remove_node(id);
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("package returned on ok report");
        // Only the player path should have registered the Wraith
        // — there's no MeshInstance left to pull it in.
        assert!(package.model_instances.is_empty());
        assert_eq!(package.models.len(), 1);
        assert_eq!(package.characters.len(), 1);

        let manifest = render_manifest_source(&package);
        // Asset indexes for the Wraith mesh, atlas, and clips
        // come straight from `package.assets` — every one of
        // them must show up in ROOM_0_REQUIRED_RAM/VRAM.
        let wraith = &package.models[0];
        let mesh_token = format!("AssetId({})", wraith.mesh_asset_index);
        assert!(
            manifest_contains_required(&manifest, "RAM", 0, &mesh_token),
            "RAM missing wraith mesh: {mesh_token}"
        );
        let atlas_token = format!(
            "AssetId({})",
            wraith
                .texture_asset_index
                .expect("starter wraith has atlas")
        );
        assert!(
            manifest_contains_required(&manifest, "VRAM", 0, &atlas_token),
            "VRAM missing wraith atlas: {atlas_token}"
        );
        let cf = wraith.clip_first as usize;
        let cc = wraith.clip_count as usize;
        for clip in &package.model_clips[cf..cf + cc] {
            let tok = format!("AssetId({})", clip.animation_asset_index);
            assert!(
                manifest_contains_required(&manifest, "RAM", 0, &tok),
                "RAM missing clip {}: {tok}",
                clip.name
            );
        }
    }

    #[test]
    fn player_character_model_assets_dedupe_with_placed_meshinstance() {
        // Starter's Wraith is referenced twice: by the placed
        // MeshInstance *and* by the Character. Each asset still
        // shows up exactly once in the manifest's residency
        // slice — the player path mustn't double-add.
        let project = project_with_one_room();
        let (package, _) = build_package(&project, &starter_project_root());
        let package = package.expect("starter cooks");
        let manifest = render_manifest_source(&package);
        let wraith = &package.models[0];

        let mesh_token = format!("AssetId({})", wraith.mesh_asset_index);
        assert_eq!(
            count_required_occurrences(&manifest, "RAM", 0, &mesh_token),
            1,
            "wraith mesh appears more than once in RAM residency"
        );
        let atlas = wraith.texture_asset_index.unwrap();
        let atlas_token = format!("AssetId({atlas})");
        assert_eq!(
            count_required_occurrences(&manifest, "VRAM", 0, &atlas_token),
            1,
            "wraith atlas appears more than once in VRAM residency"
        );
    }

    /// `true` when `ROOM_<idx>_REQUIRED_<kind>` contains `token`.
    fn manifest_contains_required(manifest: &str, kind: &str, idx: u16, token: &str) -> bool {
        count_required_occurrences(manifest, kind, idx, token) > 0
    }

    /// Count occurrences of `token` inside the
    /// `ROOM_<idx>_REQUIRED_<kind>` slice declaration. Robust
    /// enough for residency assertions; not a full Rust parser.
    fn count_required_occurrences(manifest: &str, kind: &str, idx: u16, token: &str) -> usize {
        let header = format!("ROOM_{idx}_REQUIRED_{kind}: &[AssetId] = &[");
        let Some(start) = manifest.find(&header) else {
            return 0;
        };
        let body = &manifest[start + header.len()..];
        let Some(end) = body.find("];") else {
            return 0;
        };
        body[..end].matches(token).count()
    }

    #[test]
    fn rendered_manifest_includes_characters_and_player_controller() {
        let project = project_with_one_room();
        let (package, _) = build_package(&project, &starter_project_root());
        let manifest = render_manifest_source(&package.unwrap());
        assert!(manifest.contains("pub static CHARACTERS:"));
        assert!(manifest.contains("LevelCharacterRecord"));
        assert!(manifest.contains("pub static PLAYER_CONTROLLER:"));
        assert!(manifest.contains("Some(PlayerControllerRecord"));
        assert!(manifest.contains("CHARACTER_CLIP_NONE"));
    }

    #[test]
    fn player_spawn_with_invalid_idle_clip_fails_validation() {
        let mut project = project_with_one_room();
        let scene = project.active_scene();
        let character_id = scene
            .nodes()
            .iter()
            .find_map(|_| {
                project.resources.iter().find_map(|r| match &r.data {
                    crate::ResourceData::Character(_) => Some(r.id),
                    _ => None,
                })
            })
            .expect("starter has a Character");
        // Bump idle clip past the model's clip count so cook
        // validation must reject.
        if let Some(resource) = project.resource_mut(character_id) {
            if let crate::ResourceData::Character(c) = &mut resource.data {
                c.idle_clip = Some(99);
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(report.errors.iter().any(|e| e.contains("idle clip 99")));
    }

    #[test]
    fn player_spawn_without_character_assignment_auto_picks_when_one_exists() {
        // Starter has exactly one Character. Clear the spawn's
        // explicit reference; cooker should auto-pick + warn.
        let mut project = project_with_one_room();
        let scene = project.active_scene();
        let spawn_id = scene
            .nodes()
            .iter()
            .find(|n| matches!(n.kind, NodeKind::SpawnPoint { player: true, .. }))
            .map(|n| n.id)
            .expect("starter has player spawn");
        if let Some(node) = project.active_scene_mut().node_mut(spawn_id) {
            if let NodeKind::SpawnPoint { character, .. } = &mut node.kind {
                *character = None;
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        let package = package.expect("auto-pick should succeed");
        assert!(package.player_controller.is_some());
        assert!(report.warnings.iter().any(|w| w.contains("auto-picked")));
    }

    #[test]
    fn character_with_zero_radius_fails_validation() {
        let mut project = project_with_one_room();
        let character_id = project
            .resources
            .iter()
            .find_map(|r| match &r.data {
                crate::ResourceData::Character(_) => Some(r.id),
                _ => None,
            })
            .expect("starter has a Character");
        if let Some(resource) = project.resource_mut(character_id) {
            if let crate::ResourceData::Character(c) = &mut resource.data {
                c.radius = 0;
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("radius must be > 0")));
    }

    #[test]
    fn legacy_spawn_without_character_field_still_loads() {
        // Older project.ron files lacked `character` on
        // SpawnPoint. `#[serde(default)]` should fill it with
        // `None` so they keep deserializing.
        let ron = r#"(
            name: "Legacy",
            scenes: [(
                name: "Main",
                root: (1),
                next_node_id: 3,
                nodes: [
                    (id: (1), name: "Root", kind: Node3D, transform: (translation: (0.0, 0.0, 0.0), rotation_degrees: (0.0, 0.0, 0.0), scale: (1.0, 1.0, 1.0)), parent: None, children: [(2)]),
                    (id: (2), name: "Spawn", kind: SpawnPoint(player: true), transform: (translation: (0.0, 0.0, 0.0), rotation_degrees: (0.0, 0.0, 0.0), scale: (1.0, 1.0, 1.0)), parent: Some((1)), children: []),
                ],
            )],
            resources: [],
            next_resource_id: 1,
        )"#;
        let project = ProjectDocument::from_ron_str(ron).expect("legacy spawn deserializes");
        let scene = project.active_scene();
        let spawn = scene
            .nodes()
            .iter()
            .find(|n| matches!(n.kind, NodeKind::SpawnPoint { player: true, .. }))
            .expect("spawn round-tripped");
        if let NodeKind::SpawnPoint { character, .. } = &spawn.kind {
            assert!(character.is_none(), "missing field should default to None");
        }
    }

    #[test]
    fn character_resource_roundtrips_through_ron() {
        use crate::CharacterResource;
        let mut project = ProjectDocument::starter();
        let id = project.add_resource(
            "Test Character",
            crate::ResourceData::Character(CharacterResource {
                model: None,
                idle_clip: Some(0),
                walk_clip: Some(1),
                run_clip: None,
                turn_clip: None,
                radius: 200,
                height: 1024,
                walk_speed: 50,
                run_speed: 100,
                turn_speed_degrees_per_second: 240,
                camera_distance: 1500,
                camera_height: 800,
                camera_target_height: 600,
            }),
        );
        let serialized = project.to_ron_string().expect("serializes");
        let reloaded = ProjectDocument::from_ron_str(&serialized).expect("deserializes");
        let resource = reloaded.resource(id).expect("character preserved");
        match &resource.data {
            crate::ResourceData::Character(c) => {
                assert_eq!(c.idle_clip, Some(0));
                assert_eq!(c.walk_clip, Some(1));
                assert_eq!(c.radius, 200);
                assert_eq!(c.walk_speed, 50);
                assert_eq!(c.camera_target_height, 600);
            }
            _ => panic!("character resource lost its variant after round-trip"),
        }
    }

    #[test]
    fn starter_project_emits_three_textures() {
        // Stone room uses floor + brick (room materials, deduped
        // across many cells) plus the obsidian wraith atlas
        // (model). Three distinct texture assets total.
        let project = project_with_one_room();
        let (package, _) = build_package(&project, &starter_project_root());
        let package = package.expect("starter cooks");
        assert_eq!(package.texture_asset_count(), 3);
    }

    #[test]
    fn starter_project_emits_one_model_with_clips() {
        let project = project_with_one_room();
        let (package, _) = build_package(&project, &starter_project_root());
        let package = package.expect("starter cooks");
        assert_eq!(package.models.len(), 1);
        assert_eq!(package.model_instances.len(), 1);
        assert!(package.model_clips.len() >= 1);
        assert_eq!(package.model_mesh_asset_count(), 1);
        assert_eq!(
            package.model_animation_asset_count(),
            package.model_clips.len()
        );
    }

    #[test]
    fn starter_room_material_slice_matches_cook() {
        let project = project_with_one_room();
        let (package, _) = build_package(&project, &starter_project_root());
        let package = package.expect("starter cooks");
        let room = &package.rooms[0];
        // Slice indices are valid.
        let first = room.material_first as usize;
        let count = room.material_count as usize;
        assert!(first + count <= package.materials.len());
        // Each material in the slice belongs to room 0 and has a
        // unique local_slot.
        let slice = &package.materials[first..first + count];
        let mut slots: Vec<u16> = slice.iter().map(|m| m.local_slot).collect();
        slots.sort();
        let mut dedup = slots.clone();
        dedup.dedup();
        assert_eq!(slots, dedup, "duplicate local_slot in room slice");
        for material in slice {
            assert_eq!(material.room, 0);
        }
    }

    #[test]
    fn starter_residency_includes_world_and_textures() {
        let project = project_with_one_room();
        let (package, _) = build_package(&project, &starter_project_root());
        let package = package.expect("starter cooks");

        let room = &package.rooms[0];
        let first = room.material_first as usize;
        let count = room.material_count as usize;
        let mut texture_ids: Vec<usize> = package.materials[first..first + count]
            .iter()
            .map(|m| m.texture_asset_index)
            .collect();
        texture_ids.sort();
        texture_ids.dedup();

        // Sanity: every texture asset index is a Texture asset.
        for &i in &texture_ids {
            assert_eq!(package.assets[i].kind, PlaytestAssetKind::Texture);
        }
        // Room asset is a RoomWorld at the recorded index.
        assert_eq!(
            package.assets[room.world_asset_index].kind,
            PlaytestAssetKind::RoomWorld,
        );
    }

    #[test]
    fn empty_project_fails_validation() {
        let mut project = ProjectDocument::starter();
        project.scenes[0] = crate::Scene::new("Empty");
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(!report.is_ok());
        assert!(report.errors.iter().any(|e| e.contains("Room")));
    }

    #[test]
    fn project_with_no_player_spawn_fails_validation() {
        let mut project = ProjectDocument::starter();
        let scene = project.active_scene_mut();
        let ids: Vec<NodeId> = scene
            .nodes()
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::SpawnPoint { player: true, .. }))
            .map(|n| n.id)
            .collect();
        for id in ids {
            if let Some(node) = scene.node_mut(id) {
                node.kind = NodeKind::SpawnPoint {
                    player: false,
                    character: None,
                };
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(report.errors.iter().any(|e| e.contains("player")));
    }

    #[test]
    fn project_with_multiple_player_spawns_fails_validation() {
        let mut project = ProjectDocument::starter();
        let scene = project.active_scene_mut();
        let room_id = scene
            .nodes()
            .iter()
            .find(|n| matches!(n.kind, NodeKind::Room { .. }))
            .map(|n| n.id)
            .expect("starter has a room");
        scene.add_node(
            room_id,
            "Spawn 2",
            NodeKind::SpawnPoint {
                player: true,
                character: None,
            },
        );
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(report.errors.iter().any(|e| e.contains("exactly one")));
    }

    #[test]
    fn rendered_manifest_imports_psx_level_and_static_blocks() {
        let project = project_with_one_room();
        let (package, _) = build_package(&project, &starter_project_root());
        let src = render_manifest_source(&package.expect("starter cooks"));
        assert!(src.contains("use psx_level::"));
        assert!(src.contains("pub static ASSETS"));
        assert!(src.contains("pub static MATERIALS"));
        assert!(src.contains("pub static ROOMS"));
        assert!(src.contains("pub static ROOM_RESIDENCY"));
        assert!(src.contains("pub static PLAYER_SPAWN"));
        assert!(src.contains("pub static ENTITIES"));
        assert!(src.contains("include_bytes!(\"rooms/"));
        assert!(src.contains("include_bytes!(\"textures/"));
    }

    #[test]
    fn cook_to_dir_writes_manifest_rooms_and_textures() {
        let project = ProjectDocument::starter();
        let dir = std::env::temp_dir().join(format!(
            "psxed-playtest-cook-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let report = cook_to_dir(&project, &starter_project_root(), &dir).expect("cook IO");
        assert!(report.is_ok(), "errors: {:?}", report.errors);

        let manifest =
            std::fs::read_to_string(dir.join(MANIFEST_FILENAME)).expect("manifest written");
        assert!(manifest.contains("rooms/room_000.psxw"));
        assert!(manifest.contains("textures/texture_000.psxt"));

        let blob = std::fs::read(dir.join(ROOMS_DIRNAME).join("room_000.psxw"))
            .expect("room blob written");
        assert_eq!(&blob[0..4], b"PSXW");

        // Texture blobs landed too — the starter has 2.
        assert!(dir
            .join(TEXTURES_DIRNAME)
            .join("texture_000.psxt")
            .is_file());
        assert!(dir
            .join(TEXTURES_DIRNAME)
            .join("texture_001.psxt")
            .is_file());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cook_to_dir_purges_stale_assets() {
        // Drop a fake stale file in textures/ before cooking;
        // the writer should remove it so the generated tree only
        // references files that survive this run.
        let project = ProjectDocument::starter();
        let dir = std::env::temp_dir().join(format!(
            "psxed-playtest-purge-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let textures_dir = dir.join(TEXTURES_DIRNAME);
        std::fs::create_dir_all(&textures_dir).unwrap();
        let stale = textures_dir.join("texture_999.psxt");
        std::fs::write(&stale, b"stale").unwrap();

        let report = cook_to_dir(&project, &starter_project_root(), &dir).expect("cook IO");
        assert!(report.is_ok());
        assert!(!stale.exists(), "stale texture_999.psxt should be purged");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn texture_shared_across_materials_emits_single_asset() {
        // Two materials in the starter both use the floor texture.
        // After cook + package the texture should appear once in
        // ASSETS even though both materials reference it.
        let mut project = ProjectDocument::starter();
        // Find the floor texture id and an existing material to
        // clone-and-retint as a second material referencing the
        // same texture.
        let floor_texture_id = project
            .resources
            .iter()
            .find_map(|r| match &r.data {
                ResourceData::Texture { psxt_path } if psxt_path.ends_with("floor.psxt") => {
                    Some(r.id)
                }
                _ => None,
            })
            .expect("starter has floor.psxt");

        // Reassign every wall material in the room to a new
        // material that *also* points at the floor texture. After
        // cook the world has 2 cooker material slots (floor + the
        // new wall material) but both resolve to the same texture,
        // so playtest should emit 1 texture asset.
        let new_material_id = project.add_resource(
            "FloorOnWalls",
            ResourceData::Material(crate::MaterialResource::opaque(Some(floor_texture_id))),
        );
        let scene = project.active_scene_mut();
        let room_id = scene
            .nodes()
            .iter()
            .find(|n| matches!(n.kind, NodeKind::Room { .. }))
            .map(|n| n.id)
            .expect("starter has a room");
        if let Some(node) = scene.node_mut(room_id) {
            if let NodeKind::Room { grid } = &mut node.kind {
                for sector in grid.sectors.iter_mut().flatten() {
                    for dir in [
                        crate::GridDirection::North,
                        crate::GridDirection::East,
                        crate::GridDirection::South,
                        crate::GridDirection::West,
                    ] {
                        for wall in sector.walls.get_mut(dir).iter_mut() {
                            wall.material = Some(new_material_id);
                        }
                    }
                }
            }
        }

        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("cooks");
        // 2 distinct material slots both reference the same
        // texture (room material dedup); the model atlas adds
        // one more texture so the total is 2 — what we're
        // testing here is that walls don't double-count their
        // shared floor texture, not the absolute count.
        assert_eq!(package.materials.len(), 2);
        assert_eq!(
            package.materials[0].texture_asset_index,
            package.materials[1].texture_asset_index,
        );
    }

    #[test]
    fn missing_texture_path_fails_with_clear_error() {
        // Point a texture resource at a bogus path; cook should
        // refuse and the error should mention the file.
        let mut project = ProjectDocument::starter();
        let target = project
            .resources
            .iter_mut()
            .find_map(|r| match &mut r.data {
                ResourceData::Texture { psxt_path } => Some(psxt_path),
                _ => None,
            })
            .expect("starter has at least one texture");
        *target = "this/does/not/exist.psxt".to_string();

        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report.errors.iter().any(|e| e.contains("does/not/exist")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn missing_model_mesh_path_fails_with_clear_error() {
        // Bend the starter's model resource at a bogus mesh
        // path; cook should refuse rather than silently
        // emitting a Model record without bytes.
        let mut project = ProjectDocument::starter();
        for resource in project.resources.iter_mut() {
            if let ResourceData::Model(model) = &mut resource.data {
                model.model_path = "no/such/model.psxmdl".to_string();
                break;
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("no/such/model.psxmdl")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn animation_clip_override_out_of_range_fails() {
        // Author a per-instance clip override past the model's
        // clip count → cook refuses with an explicit error
        // mentioning the offending node.
        let mut project = ProjectDocument::starter();
        let scene = project.active_scene_mut();
        // Find existing Wraith MeshInstance and bump its clip.
        let id = scene
            .nodes()
            .iter()
            .find(|n| matches!(n.kind, NodeKind::MeshInstance { .. }))
            .map(|n| n.id)
            .expect("starter has a MeshInstance");
        if let Some(node) = scene.node_mut(id) {
            if let NodeKind::MeshInstance { animation_clip, .. } = &mut node.kind {
                *animation_clip = Some(999);
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("clip override 999 out of range")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn model_with_no_atlas_fails_when_placed() {
        // Strip the starter Wraith's texture_path; cook must
        // refuse the placed instance instead of silently
        // dropping it at runtime.
        let mut project = ProjectDocument::starter();
        for resource in project.resources.iter_mut() {
            if let ResourceData::Model(model) = &mut resource.data {
                model.texture_path = None;
                break;
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report.errors.iter().any(|e| e.contains("no atlas")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn model_with_no_clips_fails_when_placed() {
        let mut project = ProjectDocument::starter();
        for resource in project.resources.iter_mut() {
            if let ResourceData::Model(model) = &mut resource.data {
                model.clips.clear();
                model.default_clip = None;
                model.preview_clip = None;
                break;
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("no animation clips")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn starter_project_emits_one_light_record() {
        // Starter Stone Room ships with a "Preview Light" node.
        // It should now appear in `package.lights` with a
        // sensible intensity_q8 derived from the editor's
        // 1.0 intensity float.
        let project = ProjectDocument::starter();
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("starter cooks");
        assert_eq!(package.lights.len(), 1);
        let light = package.lights[0];
        assert_eq!(light.room, 0);
        assert!(light.radius > 0);
        // intensity 1.0 → Q8.8 256.
        assert_eq!(light.intensity_q8, 256);
        // Starter colour authored as (255, 236, 198).
        assert_eq!(light.color, [255, 236, 198]);
    }

    #[test]
    fn light_with_zero_radius_fails() {
        let mut project = ProjectDocument::starter();
        let ids: Vec<NodeId> = project
            .active_scene()
            .nodes()
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Light { .. }))
            .map(|n| n.id)
            .collect();
        let scene = project.active_scene_mut();
        for id in ids {
            if let Some(node) = scene.node_mut(id) {
                if let NodeKind::Light { radius, .. } = &mut node.kind {
                    *radius = 0.0;
                }
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report.errors.iter().any(|e| e.contains("radius")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn light_with_negative_intensity_fails() {
        let mut project = ProjectDocument::starter();
        let ids: Vec<NodeId> = project
            .active_scene()
            .nodes()
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Light { .. }))
            .map(|n| n.id)
            .collect();
        let scene = project.active_scene_mut();
        for id in ids {
            if let Some(node) = scene.node_mut(id) {
                if let NodeKind::Light { intensity, .. } = &mut node.kind {
                    *intensity = -0.5;
                }
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report.errors.iter().any(|e| e.contains("intensity")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn light_radius_converts_sectors_to_world_units() {
        // Author a 4-sector radius; with sector_size=1024 the
        // cooked record must store 4096 world units.
        let mut project = ProjectDocument::starter();
        let ids: Vec<NodeId> = project
            .active_scene()
            .nodes()
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Light { .. }))
            .map(|n| n.id)
            .collect();
        let scene = project.active_scene_mut();
        for id in ids {
            if let Some(node) = scene.node_mut(id) {
                if let NodeKind::Light { radius, .. } = &mut node.kind {
                    *radius = 4.0;
                }
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("cooks");
        assert_eq!(package.lights[0].radius, 4096);
    }

    #[test]
    fn rendered_manifest_emits_lights_block() {
        let project = ProjectDocument::starter();
        let (package, _) = build_package(&project, &starter_project_root());
        let src = render_manifest_source(&package.expect("cooks"));
        assert!(src.contains("PointLightRecord"));
        assert!(src.contains("pub static LIGHTS"));
        assert!(src.contains("intensity_q8"));
    }

    #[test]
    fn out_of_range_model_default_clip_fails_at_cook() {
        // Bend the starter Wraith's default_clip past its clip
        // count; cook must refuse rather than emit a runtime
        // record that resolves to no animation.
        let mut project = ProjectDocument::starter();
        for resource in project.resources.iter_mut() {
            if let ResourceData::Model(model) = &mut resource.data {
                model.default_clip = Some(999);
                break;
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report.errors.iter().any(|e| e.contains("default_clip 999")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn missing_default_clip_resolves_to_clip_zero() {
        // A model with `default_clip: None` plus a populated
        // clip list should cook fine — runtime gets clip 0 as
        // the resolved default. No bind-pose sentinel.
        let mut project = ProjectDocument::starter();
        for resource in project.resources.iter_mut() {
            if let ResourceData::Model(model) = &mut resource.data {
                model.default_clip = None;
                break;
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("cooks");
        let model = &package.models[0];
        assert_eq!(model.default_clip, 0);
        // Sanity: never emit the old u16::MAX sentinel.
        assert!(model.default_clip < model.clip_count);
    }

    #[test]
    fn room_material_must_be_4bpp() {
        // Swap the starter's brick material to point at the
        // model's 8bpp atlas, which lives at the same project.
        // Cook should refuse the room material 8bpp depth.
        let mut project = ProjectDocument::starter();
        // Rewire `brick-wall.psxt` resource to the wraith atlas
        // path so it parses but with the wrong CLUT entry count.
        for resource in project.resources.iter_mut() {
            if let ResourceData::Texture { psxt_path } = &mut resource.data {
                if psxt_path.ends_with("brick-wall.psxt") {
                    *psxt_path = "assets/models/obsidian_wraith/obsidian_wraith_128x128_8bpp.psxt"
                        .to_string();
                }
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report.errors.iter().any(|e| e.contains("must be 4bpp")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn model_atlas_must_be_8bpp() {
        // Swap the wraith atlas to a 4bpp room texture path so
        // the cook runs the depth check on a known-bad atlas.
        let mut project = ProjectDocument::starter();
        for resource in project.resources.iter_mut() {
            if let ResourceData::Model(model) = &mut resource.data {
                model.texture_path = Some("assets/textures/floor.psxt".to_string());
                break;
            }
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(package.is_none());
        assert!(
            report.errors.iter().any(|e| e.contains("must be 8bpp")),
            "errors: {:?}",
            report.errors,
        );
    }

    #[test]
    fn two_instances_of_one_model_dedup_to_one_record() {
        // Add a second MeshInstance that references the same
        // model resource as the starter's Wraith. The cook
        // emits two `model_instances` but only one
        // `models[]` entry.
        let mut project = ProjectDocument::starter();
        // Resolve the starter's Wraith resource id.
        let model_id = project
            .resources
            .iter()
            .find_map(|r| match &r.data {
                ResourceData::Model(_) => Some(r.id),
                _ => None,
            })
            .expect("starter has Model");
        let scene = project.active_scene_mut();
        let room_id = scene
            .nodes()
            .iter()
            .find(|n| matches!(n.kind, NodeKind::Room { .. }))
            .map(|n| n.id)
            .unwrap();
        scene.add_node(
            room_id,
            "Wraith2",
            NodeKind::MeshInstance {
                mesh: Some(model_id),
                material: None,
                animation_clip: None,
            },
        );
        let (package, _) = build_package(&project, &starter_project_root());
        let package = package.expect("cooks");
        assert_eq!(package.models.len(), 1);
        assert_eq!(package.model_instances.len(), 2);
        // Both instances point at the same model index.
        assert_eq!(
            package.model_instances[0].model,
            package.model_instances[1].model
        );
    }

    #[test]
    fn rendered_manifest_emits_model_records() {
        let project = ProjectDocument::starter();
        let (package, _) = build_package(&project, &starter_project_root());
        let src = render_manifest_source(&package.expect("cooks"));
        assert!(src.contains("LevelModelRecord"));
        assert!(src.contains("LevelModelInstanceRecord"));
        assert!(src.contains("LevelModelClipRecord"));
        assert!(src.contains("MODEL_INSTANCES"));
        assert!(src.contains("MODELS"));
        assert!(src.contains("MODEL_CLIPS"));
        assert!(src.contains("AssetKind::ModelMesh"));
        assert!(src.contains("AssetKind::ModelAnimation"));
    }

    /// Helper: starter project with the player spawn moved to
    /// editor coord `(ex, ez)`.
    fn project_with_spawn_at(ex: f32, ez: f32) -> (ProjectDocument, NodeId, NodeId) {
        let mut project = ProjectDocument::starter();
        let (room_id, spawn_id) = {
            let scene = project.active_scene();
            let room = scene
                .nodes()
                .iter()
                .find(|n| matches!(n.kind, crate::NodeKind::Room { .. }))
                .expect("starter has a room");
            let spawn = scene
                .nodes()
                .iter()
                .find(|n| matches!(n.kind, crate::NodeKind::SpawnPoint { player: true, .. }))
                .expect("starter has a player spawn");
            (room.id, spawn.id)
        };
        if let Some(node) = project.active_scene_mut().node_mut(spawn_id) {
            node.transform.translation = [ex, 0.0, ez];
        }
        (project, room_id, spawn_id)
    }

    #[test]
    fn spawn_at_room_centre_lands_at_array_centre() {
        let (project, _, _) = project_with_spawn_at(0.0, 0.0);
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let spawn = package.unwrap().spawn.unwrap();
        assert_eq!((spawn.x, spawn.z), (1536, 1536));
    }

    #[test]
    fn spawn_after_negative_grow_lands_in_same_physical_cell() {
        let (mut project, room_id, _) = project_with_spawn_at(-1.0, 0.0);

        let (pre, _) = build_package(&project, &starter_project_root());
        let pre_spawn = pre.unwrap().spawn.unwrap();
        assert_eq!((pre_spawn.x, pre_spawn.z), (512, 1536));

        let scene = project.active_scene_mut();
        if let Some(node) = scene.node_mut(room_id) {
            if let crate::NodeKind::Room { grid } = &mut node.kind {
                grid.extend_to_include(-1, 0);
            }
        }

        let (post, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let post_spawn = post.unwrap().spawn.unwrap();
        assert_eq!((post_spawn.x, post_spawn.z), (1024, 1536));
    }

    #[test]
    fn entity_after_negative_grow_uses_same_array_relative_formula() {
        let (mut project, room_id, _) = project_with_spawn_at(0.0, 0.0);
        let scene = project.active_scene_mut();
        let entity_id = scene.add_node(
            room_id,
            "Marker",
            crate::NodeKind::SpawnPoint {
                player: false,
                character: None,
            },
        );
        if let Some(node) = scene.node_mut(entity_id) {
            node.transform.translation = [1.0, 0.0, -1.0];
        }
        if let Some(node) = scene.node_mut(room_id) {
            if let crate::NodeKind::Room { grid } = &mut node.kind {
                grid.extend_to_include(0, -1);
            }
        }

        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.unwrap();
        assert_eq!(package.entities.len(), 1);
        let e = package.entities[0];
        assert_eq!((e.x, e.z), (2560, 1024));
    }

    #[test]
    fn empty_package_renders_a_valid_skeleton() {
        let package = PlaytestPackage::default();
        let src = render_manifest_source(&package);
        assert!(src.contains("pub static ASSETS: &[LevelAssetRecord] = &[\n];"));
        assert!(src.contains("pub static MATERIALS: &[LevelMaterialRecord] = &[\n];"));
        assert!(src.contains("pub static ROOMS: &[LevelRoomRecord] = &[\n];"));
        assert!(src.contains("pub static ROOM_RESIDENCY: &[RoomResidencyRecord] = &[\n];"));
        assert!(src.contains("pub static ENTITIES: &[EntityRecord] = &[\n];"));
        assert!(src.contains("pub static PLAYER_SPAWN"));
    }
}
