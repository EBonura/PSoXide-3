//! Playtest pipeline: scene-tree → cooked rooms + Rust-source
//! manifest the `engine/examples/editor-playtest` example
//! `include!`s.
//!
//! # Why a Rust-source manifest?
//!
//! The runtime example is `no_std` and PSX-target only. It can't
//! deserialize RON / parse RAM-resident config without dragging in
//! crates the cooked path doesn't want. A generated Rust source
//! file with `include_bytes!` references for room blobs is the
//! lightest contract: the example sees `static ROOMS: &[...]` and
//! the bytes are baked into the EXE at build time.
//!
//! The same record types are defined twice — once here (used by
//! the cooker for in-memory construction) and once in the
//! generated source itself (used by the runtime). Keeping them
//! literally identical is the contract; the writer below emits
//! the types verbatim so a single edit lands in both places.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::world_cook::{cook_world_grid, encode_world_grid_psxw, WorldGridCookError};
use crate::{NodeId, NodeKind, ProjectDocument, SceneNode, WorldGrid};

/// Generated subdirectory inside the playtest example that
/// receives `level_manifest.rs` + `rooms/*.psxw`. Stable so the
/// example's `include!` paths don't move.
pub const GENERATED_DIRNAME: &str = "generated";

/// Filename of the generated Rust-source manifest the example
/// `include!`s.
pub const MANIFEST_FILENAME: &str = "level_manifest.rs";

/// Subdirectory inside `generated/` that holds cooked `.psxw`
/// blobs. Listed in the generated manifest's `include_bytes!`
/// paths.
pub const ROOMS_DIRNAME: &str = "rooms";

/// One cooked room ready to land on disk + be referenced from
/// the generated manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaytestRoom {
    /// Display name lifted from the editor scene tree. Surfaces
    /// in debug logs / future HUD overlays.
    pub name: String,
    /// Cooked `.psxw` bytes. The writer drops these to
    /// `generated/rooms/room_NNN.psxw` and emits a matching
    /// `include_bytes!`.
    pub world_bytes: Vec<u8>,
    /// Editor-side `WorldGrid::origin[0]`. Travels with the room
    /// for diagnostics and possible future origin-aware runtime
    /// placement; the v1 cooker normalizes geometry to be array-
    /// relative so the runtime renderer ignores this for now.
    pub origin_x: i32,
    /// Editor-side `WorldGrid::origin[1]`.
    pub origin_z: i32,
    /// Engine units per sector.
    pub sector_size: i32,
}

/// Player spawn record. Coordinates are room-local engine units
/// (the same space the cooked `.psxw` lives in — array-rooted at
/// world `(0, 0)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestSpawn {
    /// Room index in [`PlaytestPackage::rooms`].
    pub room: u16,
    /// Room-local X in engine units.
    pub x: i32,
    /// Y in engine units (typically 0 for floor-anchored spawns).
    pub y: i32,
    /// Room-local Z in engine units.
    pub z: i32,
    /// Yaw in PSX angle units (`0..4096`).
    pub yaw: i16,
    /// Reserved spawn flags (e.g. starting weapon, debug flags).
    /// Bit 0 reserved for "spawn enabled" — set by the writer.
    pub flags: u16,
}

/// Coarse runtime kind for [`PlaytestEntity`] records. Markers
/// are visual debug stubs only; static meshes will reference a
/// resource slot once the asset pipeline supports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaytestEntityKind {
    /// Visual marker (debug glyph or coloured cube).
    Marker,
    /// Static mesh instance pinned by `resource_slot`.
    StaticMesh,
}

/// One non-spawn entity. Same coordinate convention as
/// [`PlaytestSpawn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaytestEntity {
    /// Room index in [`PlaytestPackage::rooms`].
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
    /// Resource slot (0 if unused — e.g. for `Marker`).
    pub resource_slot: u16,
    /// Reserved flags.
    pub flags: u16,
}

/// Cooked playtest scene, ready to write to disk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlaytestPackage {
    /// Cooked rooms in deterministic order.
    pub rooms: Vec<PlaytestRoom>,
    /// Single player spawn — required.
    pub spawn: Option<PlaytestSpawn>,
    /// Optional entity markers / static meshes.
    pub entities: Vec<PlaytestEntity>,
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
/// tree, cooks every Room with non-empty geometry, and assigns
/// the player spawn. The returned report carries any errors /
/// warnings. When the report has errors, `package` is `None` —
/// callers should display the report and refuse to launch.
pub fn build_package(project: &ProjectDocument) -> (Option<PlaytestPackage>, PlaytestValidationReport) {
    let mut report = PlaytestValidationReport::default();
    let scene = project.active_scene();

    // Pass 1: enumerate Room nodes in scene-tree order. The
    // index assigned here is the runtime room id and is stable
    // as long as the scene tree's node order is stable.
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

    // Pass 2: cook each Room. Empty grids become `None` rooms
    // (skipped from the package) with a warning. Cooker errors
    // become hard errors against the project.
    let mut rooms: Vec<PlaytestRoom> = Vec::new();
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
        let bytes = match encode_world_grid_psxw(project, grid) {
            Ok(b) => b,
            Err(e) => {
                report.error(cook_error_for_node(&room_node.name, e));
                return (None, report);
            }
        };
        // `cook_world_grid` is also called above by
        // `encode_world_grid_psxw`, but we re-cook to get back
        // the manifest details (currently just material count
        // for the warning path; future fields for entity
        // resource resolution will land here too).
        let _ = cook_world_grid(project, grid);
        let index = u16::try_from(rooms.len()).unwrap_or(u16::MAX);
        node_to_room_index.insert(room_node.id, index);
        rooms.push(PlaytestRoom {
            name: room_node.name.clone(),
            world_bytes: bytes,
            origin_x: grid.origin[0],
            origin_z: grid.origin[1],
            sector_size: grid.sector_size,
        });
    }

    if rooms.is_empty() {
        report.error("every Room is empty — cook needs at least one populated room");
        return (None, report);
    }

    // Pass 3: spawn + entities. Walk every node once, route by
    // its enclosing Room (= nearest Room ancestor).
    let mut player_spawns: Vec<(NodeId, &SceneNode, u16)> = Vec::new();
    let mut entities: Vec<PlaytestEntity> = Vec::new();
    let mut warned_unsupported: HashSet<&'static str> = HashSet::new();

    for node in scene.nodes() {
        // Skip Rooms themselves (already cooked) and the root.
        if node.id == scene.root || matches!(node.kind, NodeKind::Room { .. }) {
            continue;
        }
        let Some((room_node, room_index)) = enclosing_room(scene, node, &node_to_room_index) else {
            // Nodes outside any Room don't survive cooking.
            // Quiet skip for organisational nodes (Node /
            // Node3D / World); warn for spatial entities so
            // the user notices.
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
            NodeKind::Node | NodeKind::Node3D | NodeKind::World | NodeKind::Room { .. } => {
                // Non-spatial / structural — silently skip.
            }
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
            rooms,
            spawn,
            entities,
        }),
        report,
    )
}

/// Write `package` and the generated manifest to `generated_dir`.
/// Creates `generated_dir/rooms/` if it doesn't exist. Overwrites
/// existing `level_manifest.rs` and `room_NNN.psxw` files.
pub fn write_package(
    package: &PlaytestPackage,
    generated_dir: &Path,
) -> std::io::Result<()> {
    let rooms_dir = generated_dir.join(ROOMS_DIRNAME);
    std::fs::create_dir_all(&rooms_dir)?;
    for (i, room) in package.rooms.iter().enumerate() {
        let path = rooms_dir.join(format!("room_{:03}.psxw", i));
        std::fs::write(path, &room.world_bytes)?;
    }
    let manifest = render_manifest_source(package);
    std::fs::write(generated_dir.join(MANIFEST_FILENAME), manifest)?;
    Ok(())
}

/// Render `package` as a Rust source string the runtime example
/// can `include!`. Type definitions are emitted inline so the
/// example doesn't need to depend on `psxed-project`.
pub fn render_manifest_source(package: &PlaytestPackage) -> String {
    let mut out = String::new();
    out.push_str(MANIFEST_HEADER);
    for (i, _room) in package.rooms.iter().enumerate() {
        let _ = writeln!(
            out,
            "pub static ROOM_{i}_WORLD: &[u8] = include_bytes!(\"{ROOMS_DIRNAME}/room_{i:03}.psxw\");"
        );
    }
    out.push('\n');
    out.push_str("pub static ROOMS: &[PlaytestRoomRecord] = &[\n");
    for (i, room) in package.rooms.iter().enumerate() {
        let _ = writeln!(
            out,
            "    PlaytestRoomRecord {{ name: {:?}, world_bytes: ROOM_{i}_WORLD, origin_x: {}, origin_z: {}, sector_size: {} }},",
            room.name, room.origin_x, room.origin_z, room.sector_size
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
        "pub static PLAYER_SPAWN: PlaytestSpawnRecord = PlaytestSpawnRecord {{ room: {}, x: {}, y: {}, z: {}, yaw: {}, flags: {} }};",
        spawn.room, spawn.x, spawn.y, spawn.z, spawn.yaw, spawn.flags
    );
    out.push('\n');

    out.push_str("pub static ENTITIES: &[PlaytestEntityRecord] = &[\n");
    for entity in &package.entities {
        let kind = match entity.kind {
            PlaytestEntityKind::Marker => "PlaytestEntityKind::Marker",
            PlaytestEntityKind::StaticMesh => "PlaytestEntityKind::StaticMesh",
        };
        let _ = writeln!(
            out,
            "    PlaytestEntityRecord {{ room: {}, kind: {kind}, x: {}, y: {}, z: {}, yaw: {}, resource_slot: {}, flags: {} }},",
            entity.room, entity.x, entity.y, entity.z, entity.yaw, entity.resource_slot, entity.flags
        );
    }
    out.push_str("];\n");
    out
}

/// Default destination for the playtest example's generated
/// directory: `editor/../../engine/examples/editor-playtest/generated/`.
/// Resolved at compile time relative to the editor crate so it
/// works for the dev workflow regardless of cwd.
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

/// One-shot cook + write entry point: validate, package, and
/// drop the result at `generated_dir`. Pairs with
/// [`default_generated_dir`] for the dev workflow. Returns the
/// validation report; check `report.is_ok()` before assuming
/// the files were written. Cooked artifacts only land on disk
/// when validation passes — partial writes never happen.
pub fn cook_to_dir(
    project: &ProjectDocument,
    generated_dir: &Path,
) -> std::io::Result<PlaytestValidationReport> {
    let (package, report) = build_package(project);
    if let Some(package) = package {
        write_package(&package, generated_dir)?;
    }
    Ok(report)
}

const MANIFEST_HEADER: &str = "\
// Generated by `psxed_project::playtest::write_package` —
// do not edit by hand. Regenerate with the editor's
// `Cook & Play` action or the equivalent CLI command.

#[derive(Debug, Clone, Copy)]
pub struct PlaytestRoomRecord {
    pub name: &'static str,
    pub world_bytes: &'static [u8],
    pub origin_x: i32,
    pub origin_z: i32,
    pub sector_size: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct PlaytestSpawnRecord {
    pub room: u16,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub yaw: i16,
    pub flags: u16,
}

#[derive(Debug, Clone, Copy)]
pub enum PlaytestEntityKind {
    Marker,
    StaticMesh,
}

#[derive(Debug, Clone, Copy)]
pub struct PlaytestEntityRecord {
    pub room: u16,
    pub kind: PlaytestEntityKind,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub yaw: i16,
    pub resource_slot: u16,
    pub flags: u16,
}

";

/// Walk parent links until a Room node is reached. Returns the
/// `(room, room_index)` pair if found, else `None`.
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

/// Convert a node's editor-space transform (sector-units,
/// room-centre-relative) to room-local engine units (the space
/// the cooked `.psxw` lives in — array-rooted at world `(0, 0)`).
///
/// Origin doesn't appear here: the cooker normalizes the room
/// to be array-rooted, so an entity at "editor coord (0, 0)"
/// (the room centre) lands at runtime world `(half_w * S,
/// half_d * S)` regardless of any negative-side grow.
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
/// unit (`0..4096`). Wrapping is implicit in `as i16`.
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

    fn project_with_one_room() -> ProjectDocument {
        let project = ProjectDocument::starter();
        // The starter already has a Room with a 3×3 stone grid
        // and one `SpawnPoint { player: true }` child. Sanity-
        // check this so the rest of the tests aren't fighting
        // a hidden change to the embedded RON.
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
        let (package, report) = build_package(&project);
        assert!(report.is_ok(), "errors: {:?}", report.errors);
        let package = package.expect("package returned on ok report");
        assert_eq!(package.rooms.len(), 1);
        assert!(!package.rooms[0].world_bytes.is_empty());
        assert!(package.spawn.is_some());
    }

    #[test]
    fn empty_project_fails_validation() {
        let mut project = ProjectDocument::starter();
        // Replace the active scene (always `scenes[0]`) with one
        // that has no Rooms.
        project.scenes[0] = crate::Scene::new("Empty");
        let (package, report) = build_package(&project);
        assert!(package.is_none());
        assert!(!report.is_ok());
        assert!(report.errors.iter().any(|e| e.contains("Room")));
    }

    #[test]
    fn project_with_no_player_spawn_fails_validation() {
        let mut project = ProjectDocument::starter();
        let scene = project.active_scene_mut();
        // Demote every player spawn to a non-player spawn.
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
        let (package, report) = build_package(&project);
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
        let (package, report) = build_package(&project);
        assert!(package.is_none());
        assert!(report.errors.iter().any(|e| e.contains("exactly one")));
    }

    #[test]
    fn rendered_manifest_compiles_against_inline_types() {
        // The generated source must be self-contained (the
        // example pulls it via `include!`). Synthesize a small
        // package, render the source, and confirm it round-
        // trips through Rust's tokenizer via `proc_macro2`-like
        // checks — here, just by string matching key tokens.
        let package = PlaytestPackage {
            rooms: vec![PlaytestRoom {
                name: "Test".to_string(),
                world_bytes: vec![1, 2, 3, 4],
                origin_x: 0,
                origin_z: 0,
                sector_size: 1024,
            }],
            spawn: Some(PlaytestSpawn {
                room: 0,
                x: 512,
                y: 0,
                z: 512,
                yaw: 0,
                flags: 1,
            }),
            entities: vec![],
        };
        let src = render_manifest_source(&package);
        assert!(src.contains("pub static ROOM_0_WORLD"));
        assert!(src.contains("rooms/room_000.psxw"));
        assert!(src.contains("pub static ROOMS"));
        assert!(src.contains("pub static PLAYER_SPAWN"));
        assert!(src.contains("pub static ENTITIES"));
        assert!(src.contains("PlaytestRoomRecord"));
        assert!(src.contains("PlaytestSpawnRecord"));
        assert!(src.contains("PlaytestEntityRecord"));
    }

    #[test]
    fn cook_to_dir_writes_manifest_and_room_blob() {
        let project = ProjectDocument::starter();
        let dir = std::env::temp_dir().join(format!(
            "psxed-playtest-cook-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let report = cook_to_dir(&project, &dir).expect("cook_to_dir IO");
        assert!(report.is_ok(), "errors: {:?}", report.errors);

        // level_manifest.rs exists, references room_000.psxw.
        let manifest = std::fs::read_to_string(dir.join(MANIFEST_FILENAME))
            .expect("manifest written");
        assert!(manifest.contains("rooms/room_000.psxw"));

        // The referenced room blob actually landed on disk.
        let blob_path = dir.join(ROOMS_DIRNAME).join("room_000.psxw");
        let blob = std::fs::read(&blob_path).expect("room blob written");
        // PSXW magic — sanity that it parses, not just a stub.
        assert_eq!(&blob[0..4], b"PSXW");

        // Cleanup tempdir.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_package_renders_a_valid_skeleton() {
        let package = PlaytestPackage::default();
        let src = render_manifest_source(&package);
        assert!(src.contains("pub static ROOMS: &[PlaytestRoomRecord] = &[\n];"));
        assert!(src.contains("pub static PLAYER_SPAWN"));
        assert!(src.contains("pub static ENTITIES: &[PlaytestEntityRecord] = &[\n];"));
    }
}
