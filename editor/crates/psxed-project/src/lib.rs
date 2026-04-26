//! Editor-side project model for PSoXide.
//!
//! This is the authoring model, not the final runtime layout. It keeps a
//! Godot-style scene tree and resource list so the editor can stay pleasant,
//! then later cooker stages flatten it into PS1-friendly world surfaces,
//! texture pages, entity spawns, and engine data.

use std::path::Path;

use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

pub mod world_cook;

pub use world_cook::{
    cook_world_grid, encode_world_grid_psxw, CookedGridHorizontalFace, CookedGridSector,
    CookedGridVerticalFace, CookedGridWalls, CookedWorldGrid, CookedWorldMaterial,
    WorldGridCookError, WorldGridFaceKind,
};

/// Errors raised while reading or writing editor project documents.
#[derive(Debug)]
pub enum ProjectIoError {
    /// Filesystem error.
    Io(std::io::Error),
    /// RON parse error.
    Parse(ron::error::SpannedError),
    /// RON serialization error.
    Serialize(ron::Error),
}

impl std::fmt::Display for ProjectIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "filesystem error: {error}"),
            Self::Parse(error) => write!(f, "project parse error: {error}"),
            Self::Serialize(error) => write!(f, "project serialization error: {error}"),
        }
    }
}

impl std::error::Error for ProjectIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse(error) => Some(error),
            Self::Serialize(error) => Some(error),
        }
    }
}

impl From<std::io::Error> for ProjectIoError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ron::error::SpannedError> for ProjectIoError {
    fn from(error: ron::error::SpannedError) -> Self {
        Self::Parse(error)
    }
}

impl From<ron::Error> for ProjectIoError {
    fn from(error: ron::Error) -> Self {
        Self::Serialize(error)
    }
}

/// Stable identifier for a node inside one scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    /// The root node id every scene starts with.
    pub const ROOT: Self = Self(1);

    /// Return the raw integer value for compact UI/debug display.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Stable identifier for a project resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceId(u64);

impl ResourceId {
    /// Return the raw integer value for compact UI/debug display.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Basic 3D transform used by authored nodes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform3 {
    /// World/local translation in editor units.
    pub translation: [f32; 3],
    /// Euler rotation in degrees, matching common editor UI language.
    pub rotation_degrees: [f32; 3],
    /// Per-axis scale.
    pub scale: [f32; 3],
}

impl Default for Transform3 {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_degrees: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// PS1 semi-transparency mode exposed at editor level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PsxBlendMode {
    /// Opaque textured or flat surface.
    #[default]
    Opaque,
    /// `(background + foreground) / 2`.
    Average,
    /// `background + foreground`, clamped per channel.
    Add,
    /// `background - foreground`, clamped per channel.
    Subtract,
    /// `background + foreground / 4`, clamped per channel.
    AddQuarter,
}

impl PsxBlendMode {
    /// User-facing label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Opaque => "Opaque",
            Self::Average => "Average",
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::AddQuarter => "Add Quarter",
        }
    }
}

/// Authoring material. The cooker maps this to runtime texture/material state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialResource {
    /// Source texture resource, if any.
    pub texture: Option<ResourceId>,
    /// PS1 blend mode.
    pub blend_mode: PsxBlendMode,
    /// Texture modulation tint. `0x80` is neutral for PS1 textured polys.
    pub tint: [u8; 3],
    /// Whether both windings should be emitted at export time.
    pub double_sided: bool,
}

impl MaterialResource {
    /// Build an opaque neutral material.
    pub const fn opaque(texture: Option<ResourceId>) -> Self {
        Self {
            texture,
            blend_mode: PsxBlendMode::Opaque,
            tint: [0x80, 0x80, 0x80],
            double_sided: false,
        }
    }

    /// Build a translucent neutral material.
    pub const fn translucent(texture: Option<ResourceId>, blend_mode: PsxBlendMode) -> Self {
        Self {
            texture,
            blend_mode,
            tint: [0x80, 0x80, 0x80],
            double_sided: true,
        }
    }
}

/// World-grid diagonal split.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridSplit {
    /// Split from north-west to south-east.
    #[default]
    NorthWestSouthEast,
    /// Split from north-east to south-west.
    NorthEastSouthWest,
}

/// Cardinal or diagonal grid edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridDirection {
    /// North edge, -Z.
    North,
    /// East edge, +X.
    East,
    /// South edge, +Z.
    South,
    /// West edge, -X.
    West,
    /// Diagonal from north-west to south-east.
    NorthWestSouthEast,
    /// Diagonal from north-east to south-west.
    NorthEastSouthWest,
}

/// Authored horizontal grid face.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridHorizontalFace {
    /// Corner heights `[NW, NE, SE, SW]` in engine world units.
    pub heights: [i32; 4],
    /// Diagonal split.
    pub split: GridSplit,
    /// Material used by the face.
    pub material: Option<ResourceId>,
    /// Whether character collision treats this face as walkable.
    pub walkable: bool,
}

impl GridHorizontalFace {
    /// Flat face at `height`.
    pub const fn flat(height: i32, material: Option<ResourceId>) -> Self {
        Self {
            heights: [height, height, height, height],
            split: GridSplit::NorthWestSouthEast,
            material,
            walkable: true,
        }
    }
}

/// Authored vertical grid wall.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridVerticalFace {
    /// Corner heights `[bottom-left, bottom-right, top-right, top-left]`.
    pub heights: [i32; 4],
    /// Material used by the wall.
    pub material: Option<ResourceId>,
    /// Whether collision treats this wall as blocking.
    pub solid: bool,
}

impl GridVerticalFace {
    /// Flat wall from `bottom` to `top`.
    pub const fn flat(bottom: i32, top: i32, material: Option<ResourceId>) -> Self {
        Self {
            heights: [bottom, bottom, top, top],
            material,
            solid: true,
        }
    }
}

/// Wall lists for one grid sector.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridWalls {
    /// Walls on the north edge.
    pub north: Vec<GridVerticalFace>,
    /// Walls on the east edge.
    pub east: Vec<GridVerticalFace>,
    /// Walls on the south edge.
    pub south: Vec<GridVerticalFace>,
    /// Walls on the west edge.
    pub west: Vec<GridVerticalFace>,
    /// Diagonal NW-SE walls.
    pub north_west_south_east: Vec<GridVerticalFace>,
    /// Diagonal NE-SW walls.
    pub north_east_south_west: Vec<GridVerticalFace>,
}

impl GridWalls {
    /// Immutable walls for one direction.
    pub fn get(&self, direction: GridDirection) -> &[GridVerticalFace] {
        match direction {
            GridDirection::North => &self.north,
            GridDirection::East => &self.east,
            GridDirection::South => &self.south,
            GridDirection::West => &self.west,
            GridDirection::NorthWestSouthEast => &self.north_west_south_east,
            GridDirection::NorthEastSouthWest => &self.north_east_south_west,
        }
    }

    /// Mutable walls for one direction.
    pub fn get_mut(&mut self, direction: GridDirection) -> &mut Vec<GridVerticalFace> {
        match direction {
            GridDirection::North => &mut self.north,
            GridDirection::East => &mut self.east,
            GridDirection::South => &mut self.south,
            GridDirection::West => &mut self.west,
            GridDirection::NorthWestSouthEast => &mut self.north_west_south_east,
            GridDirection::NorthEastSouthWest => &mut self.north_east_south_west,
        }
    }
}

/// One authored grid sector.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridSector {
    /// Optional floor.
    pub floor: Option<GridHorizontalFace>,
    /// Optional ceiling.
    pub ceiling: Option<GridHorizontalFace>,
    /// Sector edge walls.
    pub walls: GridWalls,
}

impl GridSector {
    /// Empty sector.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Sector with one floor face.
    pub fn with_floor(height: i32, material: Option<ResourceId>) -> Self {
        Self {
            floor: Some(GridHorizontalFace::flat(height, material)),
            ..Self::default()
        }
    }

    /// True if the sector emits any geometry.
    pub fn has_geometry(&self) -> bool {
        self.floor.is_some()
            || self.ceiling.is_some()
            || !self.walls.north.is_empty()
            || !self.walls.east.is_empty()
            || !self.walls.south.is_empty()
            || !self.walls.west.is_empty()
            || !self.walls.north_west_south_east.is_empty()
            || !self.walls.north_east_south_west.is_empty()
    }
}

/// Engine-style grid world authored by a scene node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldGrid {
    /// Width in sectors.
    pub width: u16,
    /// Depth in sectors.
    pub depth: u16,
    /// Engine units per sector.
    pub sector_size: i32,
    /// Flat `[x * depth + z]` sector storage. `None` means no sector.
    pub sectors: Vec<Option<GridSector>>,
    /// Room ambient color used as editor/cooker metadata.
    pub ambient_color: [u8; 3],
    /// Whether PS1 depth cue/fog should be cooked for this grid.
    pub fog_enabled: bool,
}

impl WorldGrid {
    /// Create an empty sparse grid.
    pub fn empty(width: u16, depth: u16, sector_size: i32) -> Self {
        let len = width as usize * depth as usize;
        Self {
            width,
            depth,
            sector_size,
            sectors: vec![None; len],
            ambient_color: [32, 32, 32],
            fog_enabled: true,
        }
    }

    /// Create a rectangular room with floors and perimeter walls.
    pub fn stone_room(
        width: u16,
        depth: u16,
        sector_size: i32,
        floor_material: Option<ResourceId>,
        wall_material: Option<ResourceId>,
    ) -> Self {
        let mut grid = Self::empty(width, depth, sector_size);
        for x in 0..width {
            for z in 0..depth {
                grid.set_floor(x, z, 0, floor_material);
                if z == 0 {
                    grid.add_wall(x, z, GridDirection::North, 0, sector_size, wall_material);
                }
                if x == width.saturating_sub(1) {
                    grid.add_wall(x, z, GridDirection::East, 0, sector_size, wall_material);
                }
                if z == depth.saturating_sub(1) {
                    grid.add_wall(x, z, GridDirection::South, 0, sector_size, wall_material);
                }
                if x == 0 {
                    grid.add_wall(x, z, GridDirection::West, 0, sector_size, wall_material);
                }
            }
        }
        grid
    }

    /// Flat sector index.
    pub fn sector_index(&self, x: u16, z: u16) -> Option<usize> {
        if x < self.width && z < self.depth {
            Some(x as usize * self.depth as usize + z as usize)
        } else {
            None
        }
    }

    /// Immutable sector.
    pub fn sector(&self, x: u16, z: u16) -> Option<&GridSector> {
        self.sector_index(x, z)
            .and_then(|index| self.sectors.get(index)?.as_ref())
    }

    /// Mutable sector, creating it if needed.
    pub fn ensure_sector(&mut self, x: u16, z: u16) -> Option<&mut GridSector> {
        let index = self.sector_index(x, z)?;
        if self.sectors[index].is_none() {
            self.sectors[index] = Some(GridSector::empty());
        }
        self.sectors[index].as_mut()
    }

    /// Set or replace a floor.
    pub fn set_floor(&mut self, x: u16, z: u16, height: i32, material: Option<ResourceId>) {
        if let Some(sector) = self.ensure_sector(x, z) {
            sector.floor = Some(GridHorizontalFace::flat(height, material));
        }
    }

    /// Add a wall to an edge.
    pub fn add_wall(
        &mut self,
        x: u16,
        z: u16,
        direction: GridDirection,
        bottom: i32,
        top: i32,
        material: Option<ResourceId>,
    ) {
        if let Some(sector) = self.ensure_sector(x, z) {
            sector
                .walls
                .get_mut(direction)
                .push(GridVerticalFace::flat(bottom, top, material));
        }
    }

    /// Number of populated sectors.
    pub fn populated_sector_count(&self) -> usize {
        self.sectors.iter().flatten().count()
    }
}

/// Resource payloads available to editor scenes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceData {
    /// Source texture plus intended cooked dimensions/depth.
    Texture {
        /// Project-relative source path.
        source_path: String,
        /// Cooked width in texels.
        width: u16,
        /// Cooked height in texels.
        height: u16,
        /// Indexed texture depth, typically 4 or 8.
        indexed_depth: u8,
    },
    /// Editor material.
    Material(MaterialResource),
    /// Source mesh.
    Mesh {
        /// Project-relative source path.
        source_path: String,
    },
    /// Nested scene/prefab reference.
    Scene {
        /// Project-relative scene path.
        source_path: String,
    },
    /// Script resource.
    Script {
        /// Project-relative script path.
        source_path: String,
    },
    /// Audio resource.
    Audio {
        /// Project-relative audio path.
        source_path: String,
    },
}

impl ResourceData {
    /// User-facing type label.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Texture { .. } => "Texture",
            Self::Material(_) => "Material",
            Self::Mesh { .. } => "Mesh",
            Self::Scene { .. } => "Scene",
            Self::Script { .. } => "Script",
            Self::Audio { .. } => "Audio",
        }
    }
}

/// One named project resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resource {
    /// Stable resource id.
    pub id: ResourceId,
    /// Display name.
    pub name: String,
    /// Payload.
    pub data: ResourceData,
}

/// Node type used by the editor scene tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Plain organisational node.
    Node,
    /// Spatial transform node.
    Node3D,
    /// Room/sector authoring node.
    Room {
        /// Width in sectors.
        width: u16,
        /// Depth in sectors.
        depth: u16,
    },
    /// Engine-style sector grid world.
    GridWorld {
        /// Authored grid-world payload.
        grid: WorldGrid,
    },
    /// Static or dynamic mesh instance.
    MeshInstance {
        /// Mesh resource.
        mesh: Option<ResourceId>,
        /// Material override.
        material: Option<ResourceId>,
    },
    /// Simple authoring light.
    Light {
        /// RGB light colour.
        color: [u8; 3],
        /// Light intensity multiplier.
        intensity: f32,
        /// Approximate editor/runtime radius.
        radius: f32,
    },
    /// Spawn marker.
    SpawnPoint {
        /// Whether this is the player spawn.
        player: bool,
    },
    /// Trigger volume marker.
    Trigger {
        /// Logical trigger id.
        trigger_id: String,
    },
    /// Positional audio source.
    AudioSource {
        /// Audio resource.
        sound: Option<ResourceId>,
        /// Playback radius.
        radius: f32,
    },
}

impl NodeKind {
    /// User-facing label.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Node => "Node",
            Self::Node3D => "Node3D",
            Self::Room { .. } => "Room",
            Self::GridWorld { .. } => "GridWorld",
            Self::MeshInstance { .. } => "MeshInstance",
            Self::Light { .. } => "Light",
            Self::SpawnPoint { .. } => "SpawnPoint",
            Self::Trigger { .. } => "Trigger",
            Self::AudioSource { .. } => "AudioSource",
        }
    }
}

/// A scene-tree node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneNode {
    /// Stable node id.
    pub id: NodeId,
    /// Display name.
    pub name: String,
    /// Node type.
    pub kind: NodeKind,
    /// Local transform.
    pub transform: Transform3,
    /// Parent id, absent only for the scene root.
    pub parent: Option<NodeId>,
    /// Ordered child ids.
    pub children: Vec<NodeId>,
}

impl SceneNode {
    fn new(id: NodeId, parent: Option<NodeId>, name: impl Into<String>, kind: NodeKind) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            transform: Transform3::default(),
            parent,
            children: Vec::new(),
        }
    }
}

/// Owned row used by hierarchy UI without borrowing the scene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRow {
    /// Node id.
    pub id: NodeId,
    /// Tree depth from root.
    pub depth: usize,
    /// Display name.
    pub name: String,
    /// Node kind label.
    pub kind: &'static str,
    /// Number of direct children.
    pub child_count: usize,
}

/// One editor scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    /// Display name.
    pub name: String,
    /// Root node id.
    pub root: NodeId,
    next_node_id: u64,
    nodes: Vec<SceneNode>,
}

impl Scene {
    /// Create a scene with one root `Node3D`.
    pub fn new(name: impl Into<String>) -> Self {
        let root = SceneNode::new(NodeId::ROOT, None, "Root", NodeKind::Node3D);
        Self {
            name: name.into(),
            root: NodeId::ROOT,
            next_node_id: NodeId::ROOT.raw() + 1,
            nodes: vec![root],
        }
    }

    /// All nodes in storage order.
    pub fn nodes(&self) -> &[SceneNode] {
        &self.nodes
    }

    /// Get a node.
    pub fn node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Get a mutable node.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.iter_mut().find(|node| node.id == id)
    }

    /// Add a node under `parent`. Invalid parents fall back to the root.
    pub fn add_node(&mut self, parent: NodeId, name: impl Into<String>, kind: NodeKind) -> NodeId {
        let parent = if self.node(parent).is_some() {
            parent
        } else {
            self.root
        };
        let id = NodeId(self.next_node_id);
        self.next_node_id = self.next_node_id.saturating_add(1);
        self.nodes
            .push(SceneNode::new(id, Some(parent), name, kind));
        if let Some(parent_node) = self.node_mut(parent) {
            parent_node.children.push(id);
        }
        id
    }

    /// Remove a non-root node and its descendants.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        if id == self.root || self.node(id).is_none() {
            return false;
        }

        let mut doomed = Vec::new();
        self.collect_descendants(id, &mut doomed);
        doomed.push(id);

        for node in &mut self.nodes {
            node.children.retain(|child| !doomed.contains(child));
        }
        self.nodes.retain(|node| !doomed.contains(&node.id));
        true
    }

    fn collect_descendants(&self, id: NodeId, out: &mut Vec<NodeId>) {
        if let Some(node) = self.node(id) {
            for child in &node.children {
                self.collect_descendants(*child, out);
                out.push(*child);
            }
        }
    }

    /// Rows in root-first depth-first order.
    pub fn hierarchy_rows(&self) -> Vec<NodeRow> {
        let mut rows = Vec::new();
        self.push_hierarchy_row(self.root, 0, &mut rows);
        rows
    }

    fn push_hierarchy_row(&self, id: NodeId, depth: usize, rows: &mut Vec<NodeRow>) {
        let Some(node) = self.node(id) else {
            return;
        };
        rows.push(NodeRow {
            id,
            depth,
            name: node.name.clone(),
            kind: node.kind.label(),
            child_count: node.children.len(),
        });
        for child in &node.children {
            self.push_hierarchy_row(*child, depth + 1, rows);
        }
    }
}

/// One editor project document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectDocument {
    /// Display name.
    pub name: String,
    /// Open scenes. The first scene is the active scene for now.
    pub scenes: Vec<Scene>,
    /// Project resources.
    pub resources: Vec<Resource>,
    next_resource_id: u64,
}

impl ProjectDocument {
    /// Create an empty project with one scene.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            scenes: vec![Scene::new("Main")],
            resources: Vec::new(),
            next_resource_id: 1,
        }
    }

    /// Create a useful starter project for the embedded editor workspace.
    pub fn starter() -> Self {
        let mut project = Self::new("Untitled PS1 Project");

        let floor_tex = project.add_resource(
            "floor.psxt",
            ResourceData::Texture {
                source_path: "textures/floor.png".to_string(),
                width: 64,
                height: 64,
                indexed_depth: 4,
            },
        );
        let brick_tex = project.add_resource(
            "brick-wall.psxt",
            ResourceData::Texture {
                source_path: "textures/brick-wall.png".to_string(),
                width: 64,
                height: 64,
                indexed_depth: 4,
            },
        );
        let floor_mat = project.add_resource(
            "Floor Material",
            ResourceData::Material(MaterialResource::opaque(Some(floor_tex))),
        );
        let brick_mat = project.add_resource(
            "Brick Material",
            ResourceData::Material(MaterialResource::opaque(Some(brick_tex))),
        );
        let glass_mat = project.add_resource(
            "Average Glass",
            ResourceData::Material(MaterialResource::translucent(
                Some(floor_tex),
                PsxBlendMode::Average,
            )),
        );

        let scene = project.active_scene_mut();
        let root = scene.root;
        scene.add_node(
            root,
            "Stone Room",
            NodeKind::GridWorld {
                grid: WorldGrid::stone_room(3, 3, 1024, Some(floor_mat), Some(brick_mat)),
            },
        );

        let card = scene.add_node(
            root,
            "Material Card",
            NodeKind::MeshInstance {
                mesh: None,
                material: Some(glass_mat),
            },
        );
        if let Some(node) = scene.node_mut(card) {
            node.transform.translation = [0.0, 0.0, 0.0];
            node.transform.scale = [0.9, 1.0, 0.18];
        }

        let spawn = scene.add_node(root, "Player Spawn", NodeKind::SpawnPoint { player: true });
        if let Some(node) = scene.node_mut(spawn) {
            node.transform.translation = [-1.0, 0.0, -0.8];
        }

        let light = scene.add_node(
            root,
            "Preview Light",
            NodeKind::Light {
                color: [255, 236, 198],
                intensity: 1.0,
                radius: 4096.0,
            },
        );
        if let Some(node) = scene.node_mut(light) {
            node.transform.translation = [1.0, 0.0, 0.85];
        }

        project
    }

    /// Active scene.
    pub fn active_scene(&self) -> &Scene {
        &self.scenes[0]
    }

    /// Active scene, mutable.
    pub fn active_scene_mut(&mut self) -> &mut Scene {
        &mut self.scenes[0]
    }

    /// Add a resource and return its id.
    pub fn add_resource(&mut self, name: impl Into<String>, data: ResourceData) -> ResourceId {
        let id = ResourceId(self.next_resource_id);
        self.next_resource_id = self.next_resource_id.saturating_add(1);
        self.resources.push(Resource {
            id,
            name: name.into(),
            data,
        });
        id
    }

    /// Get a resource.
    pub fn resource(&self, id: ResourceId) -> Option<&Resource> {
        self.resources.iter().find(|resource| resource.id == id)
    }

    /// Get a mutable resource.
    pub fn resource_mut(&mut self, id: ResourceId) -> Option<&mut Resource> {
        self.resources.iter_mut().find(|resource| resource.id == id)
    }

    /// Return a resource display name.
    pub fn resource_name(&self, id: ResourceId) -> Option<&str> {
        self.resource(id).map(|resource| resource.name.as_str())
    }

    /// Material resources as `(id, name)` pairs for inspector combo boxes.
    pub fn material_options(&self) -> Vec<(ResourceId, String)> {
        self.resources
            .iter()
            .filter_map(|resource| match &resource.data {
                ResourceData::Material(_) => Some((resource.id, resource.name.clone())),
                _ => None,
            })
            .collect()
    }

    /// Serialize this project to human-readable RON.
    pub fn to_ron_string(&self) -> Result<String, ProjectIoError> {
        let config = PrettyConfig::new()
            .depth_limit(4)
            .separate_tuple_members(true)
            .enumerate_arrays(true);
        ron::ser::to_string_pretty(self, config).map_err(ProjectIoError::Serialize)
    }

    /// Deserialize a project from RON.
    pub fn from_ron_str(source: &str) -> Result<Self, ProjectIoError> {
        ron::from_str(source).map_err(ProjectIoError::Parse)
    }

    /// Save this project to a RON file, creating parent directories.
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), ProjectIoError> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_ron_string()?)?;
        Ok(())
    }

    /// Load a project from a RON file.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, ProjectIoError> {
        let source = std::fs::read_to_string(path)?;
        Self::from_ron_str(&source)
    }
}

impl Default for ProjectDocument {
    fn default() -> Self {
        Self::starter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starter_project_has_scene_tree_and_resources() {
        let project = ProjectDocument::starter();

        assert_eq!(project.scenes.len(), 1);
        assert!(project.resources.len() >= 5);
        assert!(project
            .active_scene()
            .hierarchy_rows()
            .iter()
            .any(|row| row.name == "Material Card"));
        let grid = project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::GridWorld { grid } => Some(grid),
                _ => None,
            })
            .expect("starter should contain a grid-world node");
        assert_eq!(grid.width, 3);
        assert_eq!(grid.depth, 3);
        assert_eq!(grid.populated_sector_count(), 9);
    }

    #[test]
    fn adding_node_preserves_parent_child_relationship() {
        let mut scene = Scene::new("Test");

        let room = scene.add_node(
            scene.root,
            "Room",
            NodeKind::GridWorld {
                grid: WorldGrid::empty(2, 2, 1024),
            },
        );
        let child = scene.add_node(room, "Spawn", NodeKind::SpawnPoint { player: true });

        assert_eq!(scene.node(child).and_then(|node| node.parent), Some(room));
        assert!(scene
            .node(room)
            .is_some_and(|node| node.children.contains(&child)));
    }

    #[test]
    fn removing_node_removes_descendants() {
        let mut scene = Scene::new("Test");
        let parent = scene.add_node(scene.root, "A", NodeKind::Node3D);
        let child = scene.add_node(parent, "B", NodeKind::Node3D);

        assert!(scene.remove_node(parent));
        assert!(scene.node(parent).is_none());
        assert!(scene.node(child).is_none());
        assert!(scene
            .node(scene.root)
            .is_some_and(|root| root.children.is_empty()));
    }

    #[test]
    fn project_roundtrips_through_ron_string() {
        let project = ProjectDocument::starter();
        let ron = project.to_ron_string().unwrap();

        assert!(ron.contains("Material Card"));
        assert_eq!(ProjectDocument::from_ron_str(&ron).unwrap(), project);
    }

    #[test]
    fn project_saves_and_loads_from_disk() {
        let mut project = ProjectDocument::starter();
        project.name = "Disk Test".to_string();

        let dir = std::env::temp_dir().join(format!(
            "psxed-project-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("project.ron");

        project.save_to_path(&path).unwrap();
        assert_eq!(ProjectDocument::load_from_path(&path).unwrap(), project);

        let _ = std::fs::remove_dir_all(dir);
    }
}
