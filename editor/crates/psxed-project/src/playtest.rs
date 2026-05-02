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

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::world_cook::{cook_world_grid, CookedWorldMaterial, WorldGridCookError};
use crate::{spatial, NodeId, NodeKind, ProjectDocument, ResourceData, ResourceId, SceneNode};

mod assets;
mod manifest;
mod schema;

use assets::{
    expect_room_material_depth, find_resource, load_texture_bytes, resolve_path,
    sanitise_model_dirname,
};

pub use manifest::{cook_to_dir, default_generated_dir, render_manifest_source, write_package};
pub use schema::*;

struct PlayerSpawnCandidate<'a> {
    node: &'a SceneNode,
    room_index: u16,
    character: Option<ResourceId>,
}

/// Build a playtest package from `project`. Validates the scene
/// tree, cooks every Room with non-empty geometry, resolves
/// material textures through `project_root`, and assigns the
/// player spawn.
///
/// On any validation error the returned package is `None`.
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
    // at first insertion (never removed). HashMap is fine -- we
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
                    // Room materials must be 4bpp (16-entry CLUT) --
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
                face_sidedness: cooked_material.face_sidedness,
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
            fog_rgb: grid.fog_color,
            fog_near: grid.fog_near,
            fog_far: grid.fog_far,
            flags: if grid.fog_enabled {
                psx_level::room_flags::FOG_ENABLED
            } else {
                0
            },
        });
    }

    if rooms.is_empty() {
        report.error("every Room is empty — cook needs at least one populated room");
        return (None, report);
    }

    // Pass 3: spawn + entities + model instances + lights.
    let mut player_spawns: Vec<PlayerSpawnCandidate<'_>> = Vec::new();
    let mut entities: Vec<PlaytestEntity> = Vec::new();
    let mut models: Vec<PlaytestModel> = Vec::new();
    let mut model_clips: Vec<PlaytestModelClip> = Vec::new();
    let mut model_instances: Vec<PlaytestModelInstance> = Vec::new();
    let mut lights: Vec<PlaytestLight> = Vec::new();
    // ResourceId → index into `models` for instance dedup.
    let mut model_for_resource: HashMap<ResourceId, u16> = HashMap::new();
    let mut warned_unsupported: HashSet<&'static str> = HashSet::new();

    for node in scene.nodes() {
        if node.id == scene.root || matches!(node.kind, NodeKind::Room { .. }) {
            continue;
        }
        if node.kind.is_component() {
            continue;
        }
        let Some((room_node, room_index)) = enclosing_room(scene, node, &node_to_room_index) else {
            if !matches!(
                node.kind,
                NodeKind::Node | NodeKind::Node3D | NodeKind::Entity | NodeKind::World { .. }
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
            NodeKind::Entity => {
                if let Some((model_resource_id, _material)) = component_model_renderer(scene, node)
                    .and_then(|(model, material)| {
                        model
                            .and_then(|id| {
                                project
                                    .resource(id)
                                    .filter(|r| matches!(r.data, ResourceData::Model(_)))
                                    .map(|_| id)
                            })
                            .map(|id| (id, material))
                    })
                {
                    let clip = component_animator(scene, node).and_then(|anim| anim.clip);
                    if !push_model_instance_for_resource(
                        project,
                        project_root,
                        node.name.as_str(),
                        model_resource_id,
                        clip,
                        room_index,
                        pos,
                        yaw,
                        &mut assets,
                        &mut models,
                        &mut model_clips,
                        &mut model_instances,
                        &mut model_for_resource,
                        &mut report,
                    ) {
                        return (None, report);
                    }
                }

                for light in component_point_lights(scene, node) {
                    if !push_point_light(
                        node.name.as_str(),
                        grid,
                        room_index,
                        pos,
                        light.color,
                        light.intensity,
                        light.radius,
                        &mut lights,
                        &mut report,
                    ) {
                        return (None, report);
                    }
                }

                if let Some(controller) = component_character_controller(scene, node) {
                    if controller.player {
                        player_spawns.push(PlayerSpawnCandidate {
                            node,
                            room_index,
                            character: controller.character,
                        });
                    } else if warned_unsupported.insert("NonPlayerEntity") {
                        report.warn(
                            "Non-player Entity character components are skipped in this pass",
                        );
                    }
                }
            }
            NodeKind::SpawnPoint { player: true, .. } => {
                let NodeKind::SpawnPoint { character, .. } = &node.kind else {
                    unreachable!();
                };
                player_spawns.push(PlayerSpawnCandidate {
                    node,
                    room_index,
                    character: *character,
                });
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
                    if !push_model_instance_for_resource(
                        project,
                        project_root,
                        node.name.as_str(),
                        model_resource_id,
                        *animation_clip,
                        room_index,
                        pos,
                        yaw,
                        &mut assets,
                        &mut models,
                        &mut model_clips,
                        &mut model_instances,
                        &mut model_for_resource,
                        &mut report,
                    ) {
                        return (None, report);
                    }
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
                if !push_point_light(
                    node.name.as_str(),
                    grid,
                    room_index,
                    pos,
                    *color,
                    *intensity,
                    *radius,
                    &mut lights,
                    &mut report,
                ) {
                    return (None, report);
                }
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
            NodeKind::Node
            | NodeKind::Node3D
            | NodeKind::World { .. }
            | NodeKind::Room { .. }
            | NodeKind::ModelRenderer { .. }
            | NodeKind::Animator { .. }
            | NodeKind::Collider { .. }
            | NodeKind::Interactable { .. }
            | NodeKind::CharacterController { .. }
            | NodeKind::AiController { .. }
            | NodeKind::Combat { .. }
            | NodeKind::PointLight { .. } => {}
        }
    }

    let spawn = match player_spawns.len() {
        0 => {
            report.error("playtest needs exactly one SpawnPoint with `player: true` — none found");
            None
        }
        1 => {
            let candidate = &player_spawns[0];
            let node = candidate.node;
            let room_index = candidate.room_index;
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
    // in this pass -- only the player slot consumes them. Once
    // enemies / NPCs surface, the same `register_model_for_instance`
    // dedupe path handles their backing models too.
    let mut characters: Vec<PlaytestCharacter> = Vec::new();
    let player_controller = match (spawn, &player_spawns[..]) {
        (Some(spawn_record), [candidate]) => {
            let spawn_node = candidate.node;
            let resolved = match crate::resolve::resolve_spawn_character(
                project,
                candidate.character,
            ) {
                Ok(resolved) => {
                    if resolved.auto_picked {
                        report.warn(format!(
                            "Player Spawn '{}' had no Character — auto-picked the only one defined",
                            spawn_node.name,
                        ));
                    }
                    Some(resolved.id)
                }
                Err(crate::resolve::SpawnCharacterResolutionError::MissingExplicit(id)) => {
                    report.error(format!(
                        "Player Spawn '{}' references Character #{} which doesn't exist",
                        spawn_node.name,
                        id.raw()
                    ));
                    None
                }
                Err(crate::resolve::SpawnCharacterResolutionError::ExplicitNotCharacter(id)) => {
                    let name = project
                        .resource(id)
                        .map(|r| r.name.as_str())
                        .unwrap_or("<missing>");
                    report.error(format!(
                        "Player Spawn '{}' references resource '{}' which is not a Character",
                        spawn_node.name, name
                    ));
                    None
                }
                Err(crate::resolve::SpawnCharacterResolutionError::NoCharacters) => {
                    report.error(format!(
                        "Player Spawn '{}' has no Character assigned and no Character resources exist",
                        spawn_node.name
                    ));
                    None
                }
                Err(crate::resolve::SpawnCharacterResolutionError::AmbiguousCharacters {
                    count,
                }) => {
                    report.error(format!(
                        "Player Spawn '{}' has no Character assigned and {count} Characters are defined — pick one explicitly",
                        spawn_node.name
                    ));
                    None
                }
            };
            resolved
                .and_then(|id| {
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
                })
                .map(|character_index| PlaytestPlayerController {
                    spawn: spawn_record,
                    character: character_index,
                })
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
#[allow(clippy::too_many_arguments)]
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
    // doesn't ship -- fail loud at cook so the editor surfaces
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
        // Model atlases must be 8bpp (256-entry CLUT) -- the
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

    // Clip assets -- one .psxanim per clip, validated for joint
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

fn push_model_instance_for_resource(
    project: &ProjectDocument,
    project_root: &Path,
    node_name: &str,
    model_resource_id: ResourceId,
    clip_override: Option<u16>,
    room_index: u16,
    pos: [i32; 3],
    yaw: i16,
    assets: &mut Vec<PlaytestAsset>,
    models: &mut Vec<PlaytestModel>,
    model_clips: &mut Vec<PlaytestModelClip>,
    model_instances: &mut Vec<PlaytestModelInstance>,
    model_for_resource: &mut HashMap<ResourceId, u16>,
    report: &mut PlaytestValidationReport,
) -> bool {
    let Some(model_index) = register_model_for_instance(
        project,
        project_root,
        model_resource_id,
        assets,
        models,
        model_clips,
        model_for_resource,
        report,
    ) else {
        return false;
    };
    let model = &models[model_index as usize];
    let clip = match clip_override {
        Some(idx) => {
            if idx >= model.clip_count {
                report.error(format!(
                    "Model instance '{node_name}' clip override {idx} out of range (model has {})",
                    model.clip_count
                ));
                return false;
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
    true
}

#[derive(Clone, Copy)]
struct AnimatorComponent {
    clip: Option<u16>,
}

#[derive(Clone, Copy)]
struct CharacterControllerComponent {
    character: Option<ResourceId>,
    player: bool,
}

#[derive(Clone, Copy)]
struct PointLightComponent {
    color: [u8; 3],
    intensity: f32,
    radius: f32,
}

fn component_model_renderer(
    scene: &crate::Scene,
    host: &SceneNode,
) -> Option<(Option<ResourceId>, Option<ResourceId>)> {
    component_children(scene, host).find_map(|node| match &node.kind {
        NodeKind::ModelRenderer { model, material } => Some((*model, *material)),
        _ => None,
    })
}

fn component_animator(scene: &crate::Scene, host: &SceneNode) -> Option<AnimatorComponent> {
    component_children(scene, host).find_map(|node| match &node.kind {
        NodeKind::Animator { clip, .. } => Some(AnimatorComponent { clip: *clip }),
        _ => None,
    })
}

fn component_character_controller(
    scene: &crate::Scene,
    host: &SceneNode,
) -> Option<CharacterControllerComponent> {
    component_children(scene, host).find_map(|node| match &node.kind {
        NodeKind::CharacterController { character, player } => Some(CharacterControllerComponent {
            character: *character,
            player: *player,
        }),
        _ => None,
    })
}

fn component_point_lights(scene: &crate::Scene, host: &SceneNode) -> Vec<PointLightComponent> {
    component_children(scene, host)
        .filter_map(|node| match &node.kind {
            NodeKind::PointLight {
                color,
                intensity,
                radius,
            } => Some(PointLightComponent {
                color: *color,
                intensity: *intensity,
                radius: *radius,
            }),
            _ => None,
        })
        .collect()
}

fn component_children<'a>(
    scene: &'a crate::Scene,
    host: &'a SceneNode,
) -> impl Iterator<Item = &'a SceneNode> + 'a {
    host.children
        .iter()
        .filter_map(|id| scene.node(*id))
        .filter(|node| node.kind.is_component())
}

fn push_point_light(
    node_name: &str,
    grid: &crate::WorldGrid,
    room_index: u16,
    pos: [i32; 3],
    color: [u8; 3],
    intensity: f32,
    radius: f32,
    lights: &mut Vec<PlaytestLight>,
    report: &mut PlaytestValidationReport,
) -> bool {
    // Reject obviously broken lights at cook time -- radius 0
    // contributes nothing, negative intensity is meaningless.
    // Clamp the rest into the wire format's u16 ranges.
    if radius <= 0.0 {
        report.error(format!(
            "Light '{node_name}' has radius {radius} (must be > 0)"
        ));
        return false;
    }
    if !intensity.is_finite() || intensity < 0.0 {
        report.error(format!(
            "Light '{node_name}' has invalid intensity {intensity}"
        ));
        return false;
    }
    // Editor radius is in *sector units* -- convert to world
    // units (engine units) at cook time so the runtime record
    // stays in one canonical unit regardless of room sector size.
    let radius_world = spatial::light_radius_record_units(grid, radius);
    let intensity_q8 = (intensity * 256.0).clamp(0.0, u16::MAX as f32) as u16;
    lights.push(PlaytestLight {
        room: room_index,
        x: pos[0],
        y: pos[1],
        z: pos[2],
        radius: radius_world,
        intensity_q8,
        color,
    });
    true
}

/// Resolve a path the same way the texture loader does so the
/// model writer / runtime stay in lockstep.
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

/// Convert a node's editor-space transform to cooked room-local
/// engine units.
fn node_room_local_position(node: &SceneNode, grid: &crate::WorldGrid) -> [i32; 3] {
    spatial::node_cooked_room_local_origin(grid, &node.transform)
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
        let has_player_spawn = scene.nodes().iter().any(|n| is_player_spawn_node(scene, n));
        assert!(has_room, "starter must contain a Room");
        assert!(
            has_player_spawn,
            "starter must contain a player spawn entity"
        );
        project
    }

    fn is_player_spawn_node(scene: &crate::Scene, node: &SceneNode) -> bool {
        match &node.kind {
            NodeKind::SpawnPoint { player: true, .. } => true,
            NodeKind::Entity => node.children.iter().any(|id| {
                scene.node(*id).is_some_and(|child| {
                    matches!(
                        child.kind,
                        NodeKind::CharacterController { player: true, .. }
                    )
                })
            }),
            _ => false,
        }
    }

    fn player_spawn_node_id(project: &ProjectDocument) -> NodeId {
        let scene = project.active_scene();
        scene
            .nodes()
            .iter()
            .find(|node| is_player_spawn_node(scene, node))
            .expect("starter has a player spawn entity")
            .id
    }

    fn player_controller_component_id(project: &ProjectDocument) -> NodeId {
        let scene = project.active_scene();
        scene
            .nodes()
            .iter()
            .find(|node| {
                matches!(
                    node.kind,
                    NodeKind::CharacterController { player: true, .. }
                )
            })
            .expect("starter has a player CharacterController")
            .id
    }

    fn demote_player_spawns(project: &mut ProjectDocument) {
        let scene = project.active_scene_mut();
        let ids: Vec<NodeId> = scene
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    node.kind,
                    NodeKind::SpawnPoint { player: true, .. }
                        | NodeKind::CharacterController { player: true, .. }
                )
            })
            .map(|node| node.id)
            .collect();
        for id in ids {
            let Some(node) = scene.node_mut(id) else {
                continue;
            };
            match &mut node.kind {
                NodeKind::SpawnPoint { player, character } if *player => {
                    *player = false;
                    *character = None;
                }
                NodeKind::CharacterController { player, character } if *player => {
                    *player = false;
                    *character = None;
                }
                _ => {}
            }
        }
    }

    fn starter_light_color(project: &ProjectDocument) -> [u8; 3] {
        project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Light { color, .. } => Some(*color),
                NodeKind::PointLight { color, .. } => Some(*color),
                _ => None,
            })
            .expect("starter contains one light")
    }

    fn starter_light_intensity_q8(project: &ProjectDocument) -> u16 {
        let intensity = project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Light { intensity, .. } => Some(*intensity),
                NodeKind::PointLight { intensity, .. } => Some(*intensity),
                _ => None,
            })
            .expect("starter contains one light");
        (intensity * 256.0).clamp(0.0, u16::MAX as f32) as u16
    }

    fn starter_light_ids(project: &ProjectDocument) -> Vec<NodeId> {
        project
            .active_scene()
            .nodes()
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Light { .. } | NodeKind::PointLight { .. }))
            .map(|n| n.id)
            .collect()
    }

    fn placed_model_host_ids(project: &ProjectDocument) -> Vec<NodeId> {
        let scene = project.active_scene();
        scene
            .nodes()
            .iter()
            .filter(|node| match &node.kind {
                NodeKind::MeshInstance { mesh: Some(_), .. } => true,
                NodeKind::Entity => node.children.iter().any(|id| {
                    scene.node(*id).is_some_and(|child| {
                        matches!(child.kind, NodeKind::ModelRenderer { model: Some(_), .. })
                    })
                }),
                _ => false,
            })
            .map(|n| n.id)
            .collect()
    }

    fn set_first_model_instance_clip(project: &mut ProjectDocument, clip_index: u16) {
        let scene = project.active_scene_mut();
        let ids: Vec<NodeId> = scene
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    node.kind,
                    NodeKind::MeshInstance { .. } | NodeKind::Animator { .. }
                )
            })
            .map(|node| node.id)
            .collect();
        for id in ids {
            let Some(node) = scene.node_mut(id) else {
                continue;
            };
            match &mut node.kind {
                NodeKind::MeshInstance { animation_clip, .. } => {
                    *animation_clip = Some(clip_index);
                    return;
                }
                NodeKind::Animator { clip, .. } => {
                    *clip = Some(clip_index);
                    return;
                }
                _ => {}
            }
        }
        panic!("starter has a model animation override node");
    }

    #[test]
    fn tracked_editor_playtest_manifest_is_placeholder() {
        let manifest = std::fs::read_to_string(default_generated_dir().join(MANIFEST_FILENAME))
            .expect("read tracked editor-playtest manifest");
        assert!(
            !manifest.contains("include_bytes!"),
            "tracked placeholder manifest must not reference ignored cooked blobs"
        );
        assert!(manifest.contains("pub static ASSETS: &[LevelAssetRecord] = &[];"));
        assert!(manifest.contains("pub static ROOMS: &[LevelRoomRecord] = &[];"));
        assert!(manifest.contains("pub static ROOM_RESIDENCY: &[RoomResidencyRecord] = &[];"));
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
        // the scene) and a Wraith-Hero Character resource -- they
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
        let placed_ids = placed_model_host_ids(&project);
        let scene = project.active_scene_mut();
        for id in placed_ids {
            scene.remove_node(id);
        }
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("package returned on ok report");
        // Only the player path should have registered the Wraith
        // -- there's no MeshInstance left to pull it in.
        assert!(package.model_instances.is_empty());
        assert_eq!(package.models.len(), 1);
        assert_eq!(package.characters.len(), 1);

        let manifest = render_manifest_source(&package);
        // Asset indexes for the Wraith mesh, atlas, and clips
        // come straight from `package.assets` -- every one of
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
        // slice -- the player path mustn't double-add.
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
        let controller_id = player_controller_component_id(&project);
        if let Some(node) = project.active_scene_mut().node_mut(controller_id) {
            if let NodeKind::CharacterController { character, .. } = &mut node.kind {
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
        assert_eq!(package.model_instances.len(), 2);
        assert!(!package.model_clips.is_empty());
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
        demote_player_spawns(&mut project);
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

        // Texture blobs landed too -- the starter has 2.
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
                    for dir in crate::GridDirection::CARDINAL {
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
        // one more texture so the total is 2 -- what we're
        // testing here is that walls don't double-count their
        // shared floor texture, not the absolute count.
        assert_eq!(package.materials.len(), 2);
        assert_eq!(
            package.materials[0].texture_asset_index,
            package.materials[1].texture_asset_index,
        );
    }

    #[test]
    fn material_sidedness_reaches_playtest_manifest_flags() {
        let mut project = ProjectDocument::starter();
        let material = project
            .resources
            .iter_mut()
            .find_map(|resource| match &mut resource.data {
                ResourceData::Material(material) => Some(material),
                _ => None,
            })
            .expect("starter has a material");
        material.face_sidedness = crate::MaterialFaceSidedness::Back;
        material.sync_legacy_sidedness();

        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("cooks");
        assert!(package
            .materials
            .iter()
            .any(|m| m.face_sidedness == crate::MaterialFaceSidedness::Back));

        let src = render_manifest_source(&package);
        assert!(
            src.contains("flags: 1"),
            "back-sided material should encode FACE_BACK in flags"
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
        set_first_model_instance_clip(&mut project, 999);
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
        // authored intensity float.
        let project = ProjectDocument::starter();
        let expected_color = starter_light_color(&project);
        let expected_intensity_q8 = starter_light_intensity_q8(&project);
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("starter cooks");
        assert_eq!(package.lights.len(), 1);
        let light = package.lights[0];
        assert_eq!(light.room, 0);
        assert!(light.radius > 0);
        assert_eq!(light.intensity_q8, expected_intensity_q8);
        assert_eq!(light.color, expected_color);
    }

    #[test]
    fn light_with_zero_radius_fails() {
        let mut project = ProjectDocument::starter();
        let ids = starter_light_ids(&project);
        let scene = project.active_scene_mut();
        for id in ids {
            if let Some(node) = scene.node_mut(id) {
                match &mut node.kind {
                    NodeKind::Light { radius, .. } | NodeKind::PointLight { radius, .. } => {
                        *radius = 0.0;
                    }
                    _ => {}
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
        let ids = starter_light_ids(&project);
        let scene = project.active_scene_mut();
        for id in ids {
            if let Some(node) = scene.node_mut(id) {
                match &mut node.kind {
                    NodeKind::Light { intensity, .. } | NodeKind::PointLight { intensity, .. } => {
                        *intensity = -0.5;
                    }
                    _ => {}
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
        let ids = starter_light_ids(&project);
        let scene = project.active_scene_mut();
        for id in ids {
            if let Some(node) = scene.node_mut(id) {
                match &mut node.kind {
                    NodeKind::Light { radius, .. } | NodeKind::PointLight { radius, .. } => {
                        *radius = 4.0;
                    }
                    _ => {}
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
        let package = package.expect("cooks");
        let color = package.lights[0].color;
        let src = render_manifest_source(&package);
        assert!(src.contains("PointLightRecord"));
        assert!(src.contains("pub static LIGHTS"));
        assert!(src.contains("intensity_q8"));
        assert!(src.contains(&format!(
            "color: [{}, {}, {}]",
            color[0], color[1], color[2]
        )));
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
        // clip list should cook fine -- runtime gets clip 0 as
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
        assert_eq!(package.model_instances.len(), 3);
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
            (room.id, player_spawn_node_id(&project))
        };
        if let Some(node) = project.active_scene_mut().node_mut(spawn_id) {
            node.transform.translation = [ex, 0.0, ez];
        }
        (project, room_id, spawn_id)
    }

    fn expected_room_local_xz(
        project: &ProjectDocument,
        room_id: NodeId,
        ex: f32,
        ez: f32,
    ) -> (i32, i32) {
        let scene = project.active_scene();
        let room = scene.node(room_id).expect("room exists");
        let crate::NodeKind::Room { grid } = &room.kind else {
            panic!("expected room");
        };
        let transform = crate::Transform3 {
            translation: [ex, 0.0, ez],
            ..crate::Transform3::default()
        };
        let [x, _, z] = spatial::node_cooked_room_local_origin(grid, &transform);
        (x, z)
    }

    #[test]
    fn spawn_at_room_centre_lands_at_array_centre() {
        let (project, room_id, _) = project_with_spawn_at(0.0, 0.0);
        let (package, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let spawn = package.unwrap().spawn.unwrap();
        assert_eq!(
            (spawn.x, spawn.z),
            expected_room_local_xz(&project, room_id, 0.0, 0.0)
        );
    }

    #[test]
    fn spawn_after_negative_grow_lands_in_same_physical_cell() {
        let (mut project, room_id, _) = project_with_spawn_at(-1.0, 0.0);

        let (pre, _) = build_package(&project, &starter_project_root());
        let pre_spawn = pre.unwrap().spawn.unwrap();
        assert_eq!(
            (pre_spawn.x, pre_spawn.z),
            expected_room_local_xz(&project, room_id, -1.0, 0.0)
        );

        let scene = project.active_scene_mut();
        if let Some(node) = scene.node_mut(room_id) {
            if let crate::NodeKind::Room { grid } = &mut node.kind {
                grid.extend_to_include(-1, 0);
            }
        }

        let (post, report) = build_package(&project, &starter_project_root());
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let post_spawn = post.unwrap().spawn.unwrap();
        assert_eq!(
            (post_spawn.x, post_spawn.z),
            expected_room_local_xz(&project, room_id, -1.0, 0.0)
        );
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
        assert_eq!(
            (e.x, e.z),
            expected_room_local_xz(&project, room_id, 1.0, -1.0)
        );
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
