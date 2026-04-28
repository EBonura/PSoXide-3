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
use crate::{NodeId, NodeKind, ProjectDocument, Resource, ResourceData, ResourceId, SceneNode, WorldGrid};

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

/// Coarse asset class — mirrors [`psx_level::AssetKind`] but
/// stays host-side `String`/`Vec` friendly. Converted to the
/// runtime enum at write time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaytestAssetKind {
    /// Cooked `.psxw` room blob.
    RoomWorld,
    /// Cooked `.psxt` texture blob.
    Texture,
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
/// per-room metadata, per-room material slices, and residency.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlaytestPackage {
    /// Master asset table — rooms first, then textures, in
    /// deterministic order.
    pub assets: Vec<PlaytestAsset>,
    /// Cooked rooms with material-slice metadata.
    pub rooms: Vec<PlaytestRoom>,
    /// Material records ordered as `(room, local_slot)`.
    pub materials: Vec<PlaytestMaterial>,
    /// Single player spawn — required.
    pub spawn: Option<PlaytestSpawn>,
    /// Optional entity markers / static meshes.
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
        let material_count = u16::try_from(materials.len() - material_first as usize)
            .unwrap_or(u16::MAX);

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

    // Pass 3: spawn + entities. Identical to the prior implementation.
    let mut player_spawns: Vec<(NodeId, &SceneNode, u16)> = Vec::new();
    let mut entities: Vec<PlaytestEntity> = Vec::new();
    let mut warned_unsupported: HashSet<&'static str> = HashSet::new();

    for node in scene.nodes() {
        if node.id == scene.root || matches!(node.kind, NodeKind::Room { .. }) {
            continue;
        }
        let Some((room_node, room_index)) = enclosing_room(scene, node, &node_to_room_index) else {
            if !matches!(node.kind, NodeKind::Node | NodeKind::Node3D | NodeKind::World) {
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
            NodeKind::SpawnPoint { player: true } => {
                player_spawns.push((node.id, node, room_index));
            }
            NodeKind::SpawnPoint { player: false } => {
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
            NodeKind::MeshInstance { .. } => {
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
            NodeKind::Light { .. } => {
                if warned_unsupported.insert("Light") {
                    report.warn("Light nodes are skipped in this pass (no runtime lighting yet)");
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
            NodeKind::Node | NodeKind::Node3D | NodeKind::World | NodeKind::Room { .. } => {}
        }
    }

    let spawn = match player_spawns.len() {
        0 => {
            report.error(
                "playtest needs exactly one SpawnPoint with `player: true` — none found",
            );
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

    if !report.is_ok() {
        return (None, report);
    }

    (
        Some(PlaytestPackage {
            assets,
            rooms,
            materials,
            spawn,
            entities,
        }),
        report,
    )
}

/// Write `package` and the generated manifest to `generated_dir`.
/// Creates `generated_dir/rooms/` and `generated_dir/textures/`,
/// strips any stale `.psxw` / `.psxt` files inside them, and
/// writes the manifest source last (so a half-failed write
/// doesn't leave a manifest pointing at missing files).
pub fn write_package(
    package: &PlaytestPackage,
    generated_dir: &Path,
) -> std::io::Result<()> {
    let rooms_dir = generated_dir.join(ROOMS_DIRNAME);
    let textures_dir = generated_dir.join(TEXTURES_DIRNAME);
    std::fs::create_dir_all(&rooms_dir)?;
    std::fs::create_dir_all(&textures_dir)?;
    purge_directory_files(&rooms_dir, "psxw")?;
    purge_directory_files(&textures_dir, "psxt")?;

    for asset in &package.assets {
        let dir = match asset.kind {
            PlaytestAssetKind::RoomWorld => &rooms_dir,
            PlaytestAssetKind::Texture => &textures_dir,
        };
        std::fs::write(dir.join(&asset.filename), &asset.bytes)?;
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
        let dir = match asset.kind {
            PlaytestAssetKind::RoomWorld => ROOMS_DIRNAME,
            PlaytestAssetKind::Texture => TEXTURES_DIRNAME,
        };
        let _ = writeln!(
            out,
            "/// {} — {}",
            asset_static_name(asset, i),
            asset.source_label,
        );
        let _ = writeln!(
            out,
            "pub static {}: &[u8] = include_bytes!(\"{dir}/{}\");",
            asset_static_name(asset, i),
            asset.filename,
        );
    }
    out.push('\n');

    out.push_str("/// Master asset table.\n");
    out.push_str("pub static ASSETS: &[LevelAssetRecord] = &[\n");
    for (i, asset) in package.assets.iter().enumerate() {
        let kind = match asset.kind {
            PlaytestAssetKind::RoomWorld => "AssetKind::RoomWorld",
            PlaytestAssetKind::Texture => "AssetKind::Texture",
        };
        let static_name = asset_static_name(asset, i);
        let vram_bytes = match asset.kind {
            PlaytestAssetKind::RoomWorld => 0,
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

    // Per-room residency: required RAM = the room's world asset;
    // required VRAM = the texture assets the room's material
    // slice references, deduplicated and ordered by first appearance.
    for (i, room) in package.rooms.iter().enumerate() {
        let first = room.material_first as usize;
        let count = room.material_count as usize;
        let mut required_vram: Vec<usize> = Vec::with_capacity(count);
        for material in &package.materials[first..first + count] {
            if !required_vram.contains(&material.texture_asset_index) {
                required_vram.push(material.texture_asset_index);
            }
        }

        let _ = writeln!(out, "/// Room {i} required RAM assets.");
        let _ = writeln!(
            out,
            "pub static ROOM_{i}_REQUIRED_RAM: &[AssetId] = &[AssetId({})];",
            room.world_asset_index,
        );
        let _ = writeln!(out, "/// Room {i} required VRAM assets.");
        out.push_str(&format!("pub static ROOM_{i}_REQUIRED_VRAM: &[AssetId] = &["));
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

    out.push_str("/// Entity markers (debug cubes for now).\n");
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
    EntityKind,
    EntityRecord,
    LevelAssetRecord,
    LevelMaterialRecord,
    LevelRoomRecord,
    PlayerSpawnRecord,
    RoomResidencyRecord,
};

";

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
fn load_texture_bytes(resource: &Resource, project_root: &Path) -> Result<Vec<u8>, String> {
    let ResourceData::Texture { psxt_path } = &resource.data else {
        return Err(format!(
            "resource '{}' (#{}) is not a Texture",
            resource.name,
            resource.id.raw(),
        ));
    };
    if psxt_path.is_empty() {
        return Err(format!("texture resource '{}' has empty path", resource.name));
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
            .any(|n| matches!(n.kind, NodeKind::SpawnPoint { player: true }));
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
    fn starter_project_emits_two_textures() {
        // Stone room uses floor + brick — exactly two distinct
        // texture assets, deduplicated even though both walls and
        // both floors share materials.
        let project = project_with_one_room();
        let (package, _) = build_package(&project, &starter_project_root());
        let package = package.expect("starter cooks");
        assert_eq!(package.texture_asset_count(), 2);
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
            .filter(|n| matches!(n.kind, NodeKind::SpawnPoint { player: true }))
            .map(|n| n.id)
            .collect();
        for id in ids {
            if let Some(node) = scene.node_mut(id) {
                node.kind = NodeKind::SpawnPoint { player: false };
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
        scene.add_node(room_id, "Spawn 2", NodeKind::SpawnPoint { player: true });
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

        let manifest = std::fs::read_to_string(dir.join(MANIFEST_FILENAME))
            .expect("manifest written");
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
        // 2 distinct material slots, 1 texture asset.
        assert_eq!(package.materials.len(), 2);
        assert_eq!(package.texture_asset_count(), 1);
        // Both materials reference the same texture asset index.
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
                .find(|n| matches!(n.kind, crate::NodeKind::SpawnPoint { player: true }))
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
            crate::NodeKind::SpawnPoint { player: false },
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
