//! Editor-side project model for PSoXide.
//!
//! This is the authoring model, not the final runtime layout. It keeps a
//! Godot-style scene tree and resource list so the editor can stay pleasant,
//! then later cooker stages flatten it into PS1-friendly world surfaces,
//! texture pages, entity spawns, and engine data.

use std::path::{Path, PathBuf};

use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

pub mod model_import;
pub mod playtest;
pub mod resolve;
pub mod spatial;
pub mod world_cook;

/// Embedded copy of the default project's RON, baked at compile
/// time so the editor binary always carries a working starter even
/// if `editor/projects/default/` is absent at runtime. Single source
/// of truth -- edits to the on-disk file propagate to `starter()` on
/// the next build.
const DEFAULT_PROJECT_RON: &str = include_str!("../../../projects/default/project.ron");

/// Source-tree projects directory: `editor/projects/`.
///
/// Captured via `env!("CARGO_MANIFEST_DIR")` at compile time, so it
/// resolves wherever cargo built this crate from. Works for the
/// dev workflow (`cargo run -p frontend` from anywhere in the
/// repo); will need a different strategy when the editor ever
/// ships as a standalone binary.
pub fn projects_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("projects")
}

/// Default project directory (`editor/projects/default/`). Always
/// present in the source tree; user "New Project" copies its
/// contents into a sibling directory.
pub fn default_project_dir() -> PathBuf {
    projects_dir().join("default")
}

/// Enumerate every directory under [`projects_dir`] that contains a
/// `project.ron`. Cheap directory walk, used by the editor's open /
/// switch flow once that lands. Returns an empty Vec rather than
/// erroring when `projects_dir` doesn't exist -- fresh checkout
/// before the dev runs the editor once.
pub fn list_projects() -> std::io::Result<Vec<PathBuf>> {
    let root = projects_dir();
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.join("project.ron").is_file() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

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

/// Which side of an authored face should render.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialFaceSidedness {
    /// Render the face's authored/front winding only.
    Front,
    /// Render only the opposite side.
    Back,
    /// Render both sides.
    #[default]
    Both,
}

impl MaterialFaceSidedness {
    /// User-facing label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Front => "Front",
            Self::Back => "Back",
            Self::Both => "Both",
        }
    }

    /// Convert the old checkbox value into the new enum.
    pub const fn from_double_sided(double_sided: bool) -> Self {
        if double_sided {
            Self::Both
        } else {
            Self::Front
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
    /// Which side(s) of faces using this material should render.
    #[serde(default)]
    pub face_sidedness: MaterialFaceSidedness,
    /// Legacy project field. New code reads/writes
    /// [`face_sidedness`](Self::face_sidedness); this remains so older
    /// `.ron` projects migrate without losing their two-sided setting.
    #[serde(default)]
    pub double_sided: bool,
}

impl MaterialResource {
    /// Build an opaque neutral material.
    pub const fn opaque(texture: Option<ResourceId>) -> Self {
        Self {
            texture,
            blend_mode: PsxBlendMode::Opaque,
            tint: [0x80, 0x80, 0x80],
            face_sidedness: MaterialFaceSidedness::Both,
            double_sided: true,
        }
    }

    /// Build a translucent neutral material.
    pub const fn translucent(texture: Option<ResourceId>, blend_mode: PsxBlendMode) -> Self {
        Self {
            texture,
            blend_mode,
            tint: [0x80, 0x80, 0x80],
            face_sidedness: MaterialFaceSidedness::Both,
            double_sided: true,
        }
    }

    /// Resolved sidedness. Missing `face_sidedness` defaults to
    /// `Both` so old projects keep matching the editor preview, while
    /// legacy `double_sided = true` still upgrades an explicit/front
    /// value to two-sided.
    pub const fn sidedness(&self) -> MaterialFaceSidedness {
        if self.double_sided && matches!(self.face_sidedness, MaterialFaceSidedness::Front) {
            MaterialFaceSidedness::Both
        } else {
            self.face_sidedness
        }
    }

    /// Keep the legacy field aligned after editing `face_sidedness`.
    pub fn sync_legacy_sidedness(&mut self) {
        self.double_sided = matches!(self.face_sidedness, MaterialFaceSidedness::Both);
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

/// Floor / ceiling corner index. Maps directly to the
/// `[NW, NE, SE, SW]` order every height array uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Corner {
    NW,
    NE,
    SE,
    SW,
}

impl Corner {
    /// Index into `[NW, NE, SE, SW]`.
    pub const fn idx(self) -> usize {
        match self {
            Self::NW => 0,
            Self::NE => 1,
            Self::SE => 2,
            Self::SW => 3,
        }
    }

    /// Diagonal-opposite corner. NW ↔ SE, NE ↔ SW. Used by the
    /// vertex-delete pinch flow to find which neighbour the
    /// dropped corner welds to.
    pub const fn diagonal(self) -> Self {
        match self {
            Self::NW => Self::SE,
            Self::NE => Self::SW,
            Self::SE => Self::NW,
            Self::SW => Self::NE,
        }
    }

    /// Diagonal split that keeps a triangle alive when this
    /// corner is dropped. Drop NE / SW → NW-SE keeps one half;
    /// drop NW / SE → NE-SW keeps one half. Picking the *other*
    /// diagonal would put the dropped corner on the cut line,
    /// killing both triangles.
    pub const fn surviving_split(self) -> GridSplit {
        match self {
            Self::NE | Self::SW => GridSplit::NorthWestSouthEast,
            Self::NW | Self::SE => GridSplit::NorthEastSouthWest,
        }
    }
}

/// Wall corner index. Maps to the
/// `[bottom-left, bottom-right, top-right, top-left]` order in
/// every wall heights array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WallCorner {
    BL,
    BR,
    TR,
    TL,
}

impl WallCorner {
    pub const fn idx(self) -> usize {
        match self {
            Self::BL => 0,
            Self::BR => 1,
            Self::TR => 2,
            Self::TL => 3,
        }
    }

    /// `true` when this corner sits at the wall's bottom.
    pub const fn is_bottom(self) -> bool {
        matches!(self, Self::BL | Self::BR)
    }
}

/// Cardinal or diagonal grid edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridDirection {
    /// Editor north edge, +Z.
    North,
    /// East edge, +X.
    East,
    /// Editor south edge, -Z.
    South,
    /// West edge, -X.
    West,
    /// Diagonal from north-west to south-east.
    NorthWestSouthEast,
    /// Diagonal from north-east to south-west.
    NorthEastSouthWest,
}

impl GridDirection {
    /// Cardinal directions in editor perimeter order.
    pub const CARDINAL: [Self; 4] = [Self::North, Self::East, Self::South, Self::West];

    /// Diagonal directions in editor split order.
    pub const DIAGONAL: [Self; 2] = [Self::NorthWestSouthEast, Self::NorthEastSouthWest];

    /// Every authored grid direction.
    pub const ALL: [Self; 6] = [
        Self::North,
        Self::East,
        Self::South,
        Self::West,
        Self::NorthWestSouthEast,
        Self::NorthEastSouthWest,
    ];

    /// `true` for the four perimeter edges.
    pub const fn is_cardinal(self) -> bool {
        matches!(self, Self::North | Self::East | Self::South | Self::West)
    }

    /// Canonical physical edge claimed by this authored cardinal
    /// direction. Editor authoring uses North=+Z and South=-Z;
    /// this key lets opposing-cell wall claims collide without
    /// duplicating the convention in each caller.
    pub const fn physical_edge(self, x: u16, z: u16) -> Option<GridPhysicalEdge> {
        match self {
            Self::North => Some(GridPhysicalEdge {
                x: x as i32,
                z: z as i32 + 1,
                axis: GridEdgeAxis::EastWest,
            }),
            Self::South => Some(GridPhysicalEdge {
                x: x as i32,
                z: z as i32,
                axis: GridEdgeAxis::EastWest,
            }),
            Self::West => Some(GridPhysicalEdge {
                x: x as i32,
                z: z as i32,
                axis: GridEdgeAxis::NorthSouth,
            }),
            Self::East => Some(GridPhysicalEdge {
                x: x as i32 + 1,
                z: z as i32,
                axis: GridEdgeAxis::NorthSouth,
            }),
            Self::NorthWestSouthEast | Self::NorthEastSouthWest => None,
        }
    }
}

/// Axis of a canonical physical edge in editor cell coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GridEdgeAxis {
    /// Edge runs along Z, separating cells across X.
    NorthSouth,
    /// Edge runs along X, separating cells across Z.
    EastWest,
}

/// Canonical integer address of one physical cardinal wall edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPhysicalEdge {
    pub x: i32,
    pub z: i32,
    pub axis: GridEdgeAxis,
}

/// World-space X/Z bounds for one editor grid cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridCellBounds {
    pub x0: i32,
    pub x1: i32,
    pub z0: i32,
    pub z1: i32,
}

impl GridCellBounds {
    /// X/Z position of a horizontal face corner in editor
    /// convention: NW/NE live on the high-Z edge.
    pub const fn horizontal_corner_xz(self, corner: Corner) -> [i32; 2] {
        match corner {
            Corner::NW => [self.x0, self.z1],
            Corner::NE => [self.x1, self.z1],
            Corner::SE => [self.x1, self.z0],
            Corner::SW => [self.x0, self.z0],
        }
    }

    /// Wall bottom-edge endpoints `(BL, BR)` in editor convention.
    pub const fn wall_endpoints_xz(self, direction: GridDirection) -> Option<([i32; 2], [i32; 2])> {
        match direction {
            GridDirection::North => Some(([self.x0, self.z1], [self.x1, self.z1])),
            GridDirection::East => Some(([self.x1, self.z1], [self.x1, self.z0])),
            GridDirection::South => Some(([self.x1, self.z0], [self.x0, self.z0])),
            GridDirection::West => Some(([self.x0, self.z0], [self.x0, self.z1])),
            GridDirection::NorthWestSouthEast | GridDirection::NorthEastSouthWest => None,
        }
    }
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
    /// `Some(corner)` when one corner has been deleted, turning
    /// the face into a triangle. The renderer skips the half
    /// containing the missing corner; `split` is forced to the
    /// surviving diagonal at edit time. Default `None` =
    /// authored as a normal quad.
    #[serde(default)]
    pub dropped_corner: Option<Corner>,
}

impl GridHorizontalFace {
    /// Flat face at `height`.
    pub const fn flat(height: i32, material: Option<ResourceId>) -> Self {
        Self {
            heights: [height, height, height, height],
            split: GridSplit::NorthWestSouthEast,
            material,
            walkable: true,
            dropped_corner: None,
        }
    }

    /// Drop one corner -- the face becomes a visible triangle.
    /// Forces `split` to the diagonal that keeps a triangle
    /// alive (drop NE / SW → NW-SE; drop NW / SE → NE-SW). The
    /// dropped corner's stored height is left untouched so the
    /// user can recover by un-dropping.
    pub fn drop_corner(&mut self, corner: Corner) {
        self.dropped_corner = Some(corner);
        self.split = corner.surviving_split();
    }

    /// Restore the face to a full quad.
    pub fn restore_corner(&mut self) {
        self.dropped_corner = None;
    }

    /// `true` when the face is currently a triangle.
    pub const fn is_triangle(&self) -> bool {
        self.dropped_corner.is_some()
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
    /// `Some(corner)` when one wall corner has been deleted,
    /// turning the wall quad into a triangle. Default `None`.
    #[serde(default)]
    pub dropped_corner: Option<WallCorner>,
}

impl GridVerticalFace {
    /// Flat wall from `bottom` to `top`.
    pub const fn flat(bottom: i32, top: i32, material: Option<ResourceId>) -> Self {
        Self {
            heights: [bottom, bottom, top, top],
            material,
            solid: true,
            dropped_corner: None,
        }
    }

    pub fn drop_corner(&mut self, corner: WallCorner) {
        self.dropped_corner = Some(corner);
    }

    pub fn restore_corner(&mut self) {
        self.dropped_corner = None;
    }

    pub const fn is_triangle(&self) -> bool {
        self.dropped_corner.is_some()
    }
}

/// Wall lists for one grid sector.
///
/// **Ownership rule**: a physical wall between cells `(x, z)` and
/// `(x+1, z)` is the **East** wall of `(x, z)` AND the **West**
/// wall of `(x+1, z)` simultaneously. The editor's PaintWall tool
/// stamps only one side (whichever the user clicked). When both
/// sides claim the same physical edge the cooker rejects the grid
/// with `DuplicatePhysicalWall` -- render-+-collision-correct
/// double walls aren't a thing, and silent-dedup risks the editor
/// and runtime disagreeing about which side won. North/South share
/// `North(x, z)` ↔ `South(x, z+1)` under the same rule.
///
/// Diagonal walls are authoring-only for now: cooker rejects them
/// (`UnsupportedDiagonalWall`) until render / pick / collision
/// agree on their geometry.
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

/// Hard caps on a single room's authoring shape. The cooker
/// rejects past these, and the editor inspector warns as the
/// budget approaches them -- both to keep the cooked `.psxw`
/// inside reasonable PSX-side memory and to surface coordinate
/// safety early (32-sector room × 1024 sector_size = 32 768,
/// right at the i16 cliff; the renderer uses anchor-relative
/// coords now but still respects the cap as belt-and-braces).
pub const MAX_ROOM_WIDTH: u16 = 32;
pub const MAX_ROOM_DEPTH: u16 = 32;
pub const MAX_WALL_STACK: usize = 4;
pub const MAX_ROOM_TRIANGLES: usize = 2048;
pub const MAX_ROOM_BYTES: usize = 64 * 1024;

/// World-unit step every authored vertex height must align to.
///
/// The X / Z grid is locked by construction -- corners are always
/// computed from the cell's array index and `sector_size`. Y is
/// the only free axis, and we constrain it to multiples of this
/// step so the editor can't author noise heights that the runtime
/// quantises away anyway.
///
/// 32 is `sector_size / 32` at the default 1024 -- fine enough that
/// authored slopes look smooth, coarse enough that PS1 i16 vertex
/// jitter never fights the snap.
pub const HEIGHT_QUANTUM: i32 = 32;

/// Snap a vertex height to the nearest [`HEIGHT_QUANTUM`] multiple.
///
/// Round-half-away-from-zero so the snap is symmetric for
/// negative heights -- `snap_height(-15)` returns `0`,
/// `snap_height(-16)` returns `-32`. Plain integer math; no
/// float intermediaries.
pub fn snap_height(y: i32) -> i32 {
    let q = HEIGHT_QUANTUM;
    let half = q / 2;
    if y >= 0 {
        ((y + half) / q) * q
    } else {
        -(((-y + half) / q) * q)
    }
}

/// Snapshot of a [`WorldGrid`]'s authoring footprint + cooked-
/// byte estimate. Cheap to compute (single sector pass); the
/// editor recomputes it whenever the inspector for a Room
/// repaints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorldGridBudget {
    /// Grid width in sectors.
    pub width: u16,
    /// Grid depth in sectors.
    pub depth: u16,
    /// `width * depth`. v1 `.psxw` stores a sector record for
    /// every cell whether it's populated or not, so this is
    /// what the wire-size formula multiplies against.
    pub total_cells: usize,
    /// Cells that have any geometry (floor / ceiling / walls).
    /// Useful for surface-area / drawcall estimates; not the
    /// driver of the byte budget.
    pub populated_cells: usize,
    pub floors: usize,
    pub ceilings: usize,
    pub walls: usize,
    pub triangles: usize,
    /// `.psxw` v1 wire size with 44-byte sectors / 24-byte
    /// walls. v1 stores a record for **every** cell, so this
    /// uses `total_cells`, not `populated_cells`.
    pub psxw_v1_bytes: usize,
    /// Estimated size if we shipped the future compact format
    /// described in `docs/world-format-roadmap.md` (28-byte
    /// sectors, 12-byte walls). Surfaced as a planning aid, not
    /// a contract -- no live `.psxw` is ever this size today.
    pub future_compact_estimated_bytes: usize,
}

impl WorldGridBudget {
    /// `true` if any cap is exceeded. Mirrors the validation the
    /// cooker now enforces -- UI and cooker can't disagree about
    /// what counts as "too big."
    pub fn over_budget(&self) -> bool {
        self.width > MAX_ROOM_WIDTH
            || self.depth > MAX_ROOM_DEPTH
            || self.triangles > MAX_ROOM_TRIANGLES
            || self.psxw_v1_bytes > MAX_ROOM_BYTES
    }
}

const fn default_ambient_color() -> [u8; 3] {
    [32, 32, 32]
}

const fn default_light_color() -> [u8; 3] {
    [255, 240, 200]
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
    /// World offset (in cell units) of cell index `(0, 0)`. Lets the
    /// editor extend the room into negative `X` / `Z` without
    /// renumbering existing cells: a `-X` grow shifts sectors by
    /// `+1` in X, decrements `origin.x` by `1`, and the renderer's
    /// world coord = `(origin + index) * sector_size`. Default
    /// `[0, 0]` for backward compat with already-saved projects.
    #[serde(default)]
    pub origin: [i32; 2],
    /// Room ambient color used as editor/cooker metadata.
    #[serde(default = "default_ambient_color")]
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
            origin: [0, 0],
            ambient_color: default_ambient_color(),
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
                if z == depth.saturating_sub(1) {
                    grid.add_wall(x, z, GridDirection::North, 0, sector_size, wall_material);
                }
                if x == width.saturating_sub(1) {
                    grid.add_wall(x, z, GridDirection::East, 0, sector_size, wall_material);
                }
                if z == 0 {
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

    /// Mutable sector. `None` when out-of-bounds OR the cell hasn't
    /// been authored yet (use `ensure_sector` to create-on-access).
    pub fn sector_mut(&mut self, x: u16, z: u16) -> Option<&mut GridSector> {
        self.sector_index(x, z)
            .and_then(move |index| self.sectors.get_mut(index)?.as_mut())
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

    /// Snapshot of this grid's authoring footprint + cooked-byte
    /// estimate. Used by the editor inspector to surface room
    /// budgets so authors notice when they're approaching PSX
    /// limits before cook time.
    pub fn budget(&self) -> WorldGridBudget {
        let mut b = WorldGridBudget {
            width: self.width,
            depth: self.depth,
            total_cells: (self.width as usize) * (self.depth as usize),
            ..Default::default()
        };
        for sector in self.sectors.iter().flatten() {
            b.populated_cells += 1;
            if sector.floor.is_some() {
                b.floors += 1;
                b.triangles += 2;
            }
            if sector.ceiling.is_some() {
                b.ceilings += 1;
                b.triangles += 2;
            }
            for direction in GridDirection::ALL {
                let count = sector.walls.get(direction).len();
                b.walls += count;
                b.triangles += count * 2;
            }
        }
        // v1 wire layout (matches `psxed_format::world` records):
        //   AssetHeader = 12, WorldHeader = 20, Sector = 44, Wall = 24.
        // v1 stores a sector record for every cell -- empty or not --
        // so the byte count uses `total_cells`. Using
        // `populated_cells` here was the original bug: it under-
        // reported the wire size by ~44 B per empty cell.
        const ASSET_HEADER_BYTES: usize = 12;
        const WORLD_HEADER_BYTES: usize = 20;
        const V1_SECTOR_BYTES: usize = 44;
        const V1_WALL_BYTES: usize = 24;
        // Target compact-format sizes for the planning estimate.
        // See `docs/world-format-roadmap.md`. Plain numeric
        // constants rather than struct sizes so this block doesn't
        // pretend a v2 format exists in code.
        const FUTURE_COMPACT_SECTOR_BYTES: usize = 28;
        const FUTURE_COMPACT_WALL_BYTES: usize = 12;
        b.psxw_v1_bytes = ASSET_HEADER_BYTES
            + WORLD_HEADER_BYTES
            + b.total_cells * V1_SECTOR_BYTES
            + b.walls * V1_WALL_BYTES;
        b.future_compact_estimated_bytes = ASSET_HEADER_BYTES
            + WORLD_HEADER_BYTES
            + b.total_cells * FUTURE_COMPACT_SECTOR_BYTES
            + b.walls * FUTURE_COMPACT_WALL_BYTES;
        b
    }

    /// World-space X coordinate of the left edge of column `sx`
    /// (array index, not world-cell index). Accounts for `origin`
    /// so the renderer and picking always agree on cell positions.
    pub fn cell_world_x(&self, sx: u16) -> i32 {
        (self.origin[0] + sx as i32) * self.sector_size
    }

    /// World-space Z coordinate of the low-Z edge of row `sz`.
    pub fn cell_world_z(&self, sz: u16) -> i32 {
        (self.origin[1] + sz as i32) * self.sector_size
    }

    /// World-space X/Z bounds of cell `(sx, sz)` in editor
    /// convention. `z0` is the low-Z / south edge and `z1` is
    /// the high-Z / north edge.
    pub fn cell_bounds_world(&self, sx: u16, sz: u16) -> GridCellBounds {
        let x0 = self.cell_world_x(sx);
        let z0 = self.cell_world_z(sz);
        GridCellBounds {
            x0,
            x1: x0 + self.sector_size,
            z0,
            z1: z0 + self.sector_size,
        }
    }

    /// World-space `(x, z)` centre of cell `(sx, sz)` in floating
    /// point -- handy for picking, edge inference, and entity
    /// snapping. Mirrors the renderer's cell positioning so all
    /// three pipelines agree on where each cell physically sits.
    pub fn cell_center_world(&self, sx: u16, sz: u16) -> [f32; 2] {
        let s = self.sector_size as f32;
        [
            (self.origin[0] as f32 + sx as f32 + 0.5) * s,
            (self.origin[1] as f32 + sz as f32 + 0.5) * s,
        ]
    }

    /// Geometric centre of the room in world-cell units. After a
    /// negative-side grow this is `(origin + half)` rather than
    /// just `half`, so callers stay correct without each
    /// re-deriving the offset.
    ///
    /// This is the **canonical** editor centre -- every coordinate
    /// helper that bridges editor-viewport units (sector-units,
    /// room-centre-relative) and world-cell / world-space units
    /// goes through this single source of truth.
    pub fn grid_center_cells(&self) -> [f32; 2] {
        [
            self.origin[0] as f32 + self.width as f32 * 0.5,
            self.origin[1] as f32 + self.depth as f32 * 0.5,
        ]
    }

    /// Convert editor-viewport coordinates (sector-units,
    /// room-centre-relative) to world-cell units. The viewport's
    /// `(0, 0)` is the room centre; world-cell `(0, 0)` is the
    /// runtime cell at the room's first array slot pre-grow.
    pub fn editor_to_world_cells(&self, editor: [f32; 2]) -> [f32; 2] {
        let center = self.grid_center_cells();
        [editor[0] + center[0], editor[1] + center[1]]
    }

    /// Inverse of [`Self::editor_to_world_cells`]. World coords
    /// (post-`/sector_size`) returned from a 3D ground-plane hit
    /// land back in the editor's sector-unit space ready to feed
    /// `world_cell_to_array` or stash on a node transform.
    pub fn world_cells_to_editor(&self, world_cells: [f32; 2]) -> [f32; 2] {
        let center = self.grid_center_cells();
        [world_cells[0] - center[0], world_cells[1] - center[1]]
    }

    /// Editor-viewport position → array `(sx, sz)`. Combines
    /// `editor_to_world_cells` + `floor` + `world_cell_to_array`
    /// in one step so callers don't repeat the conversion at
    /// each call site.
    pub fn editor_cells_to_array(&self, editor: [f32; 2]) -> Option<(u16, u16)> {
        let world = self.editor_to_world_cells(editor);
        let wcx = world[0].floor() as i32;
        let wcz = world[1].floor() as i32;
        self.world_cell_to_array(wcx, wcz)
    }

    /// Editor-viewport position → world-space `(x, 0, z)` in
    /// engine units (room-local, origin-aware). Used by the
    /// editor's 3D preview path which renders cells at
    /// `cell_world_x/z` so authored content keeps its visual
    /// position after a negative-side grow.
    pub fn editor_to_room_local(&self, editor: [f32; 2]) -> [f32; 3] {
        let world_cells = self.editor_to_world_cells(editor);
        let s = self.sector_size as f32;
        [world_cells[0] * s, 0.0, world_cells[1] * s]
    }

    /// Inverse of [`Self::editor_to_room_local`] -- world-space
    /// `(x, _, z)` → editor-viewport `(x, z)` (sector-units,
    /// room-centre-relative). The `y` component is dropped:
    /// cell positioning is purely XZ.
    pub fn room_local_to_editor(&self, room_local: [f32; 3]) -> [f32; 2] {
        let s = self.sector_size as f32;
        self.world_cells_to_editor([room_local[0] / s, room_local[2] / s])
    }

    /// Convert a world position to the world-cell coordinate
    /// (which can be negative). The world-cell is the same coord
    /// system the renderer uses; subtract `origin` to get the
    /// array index.
    pub fn world_x_to_cell(&self, world_x: f32) -> i32 {
        (world_x / self.sector_size as f32).floor() as i32
    }

    pub fn world_z_to_cell(&self, world_z: f32) -> i32 {
        (world_z / self.sector_size as f32).floor() as i32
    }

    /// Translate a world-cell coordinate to its array index, or
    /// `None` if the cell isn't currently allocated.
    pub fn world_cell_to_array(&self, wcx: i32, wcz: i32) -> Option<(u16, u16)> {
        let ax = wcx.checked_sub(self.origin[0])?;
        let az = wcz.checked_sub(self.origin[1])?;
        if ax < 0 || az < 0 {
            return None;
        }
        let ax = ax as u32;
        let az = az as u32;
        if ax >= self.width as u32 || az >= self.depth as u32 {
            return None;
        }
        Some((ax as u16, az as u16))
    }

    /// Ensure the world-cell `(wcx, wcz)` is addressable. Grows
    /// the grid in `+X` / `+Z` and / or shifts existing sectors
    /// (with `origin` decrementing in lockstep) when growth is
    /// needed in `-X` / `-Z`. Existing cells keep the same world
    /// position throughout. Returns the resolved array index.
    pub fn extend_to_include(&mut self, wcx: i32, wcz: i32) -> (u16, u16) {
        let rel_x = wcx - self.origin[0];
        let rel_z = wcz - self.origin[1];
        let shift_x = (-rel_x).max(0) as u16;
        let shift_z = (-rel_z).max(0) as u16;
        // The new array width must hold both the shifted existing
        // data ([shift, shift + old_width)) AND the new cell (at
        // shift + max(rel, 0)). Same logic for depth.
        let new_cell_x = (rel_x.max(0) as u16) + shift_x;
        let new_cell_z = (rel_z.max(0) as u16) + shift_z;
        let new_w = (shift_x + self.width).max(new_cell_x + 1);
        let new_d = (shift_z + self.depth).max(new_cell_z + 1);
        if shift_x == 0 && shift_z == 0 && new_w == self.width && new_d == self.depth {
            return (rel_x as u16, rel_z as u16);
        }
        // Rebuild the sector array, shifting existing data by
        // (shift_x, shift_z) so its world position is preserved.
        let new_len = new_w as usize * new_d as usize;
        let mut new_sectors: Vec<Option<GridSector>> = vec![None; new_len];
        for x in 0..self.width {
            for z in 0..self.depth {
                let old_idx = x as usize * self.depth as usize + z as usize;
                let new_x = x as usize + shift_x as usize;
                let new_z = z as usize + shift_z as usize;
                if new_x < new_w as usize && new_z < new_d as usize {
                    let new_idx = new_x * new_d as usize + new_z;
                    new_sectors[new_idx] = self.sectors[old_idx].take();
                }
            }
        }
        self.width = new_w;
        self.depth = new_d;
        self.origin[0] -= shift_x as i32;
        self.origin[1] -= shift_z as i32;
        self.sectors = new_sectors;
        (
            (rel_x + shift_x as i32) as u16,
            (rel_z + shift_z as i32) as u16,
        )
    }

    /// Reshape the grid to `new_width × new_depth`.
    ///
    /// Sectors that lie inside both the old and new bounds keep
    /// their authored content; cells that were outside the old
    /// bounds (a grow operation) come up empty; cells outside the
    /// new bounds (a shrink) are dropped.
    ///
    /// No-op when the dims already match.
    pub fn resize(&mut self, new_width: u16, new_depth: u16) {
        if new_width == self.width && new_depth == self.depth {
            return;
        }
        let new_len = new_width as usize * new_depth as usize;
        let mut new_sectors: Vec<Option<GridSector>> = vec![None; new_len];
        let copy_w = self.width.min(new_width);
        let copy_d = self.depth.min(new_depth);
        for x in 0..copy_w {
            for z in 0..copy_d {
                let old_idx = x as usize * self.depth as usize + z as usize;
                let new_idx = x as usize * new_depth as usize + z as usize;
                new_sectors[new_idx] = self.sectors[old_idx].take();
            }
        }
        self.width = new_width;
        self.depth = new_depth;
        self.sectors = new_sectors;
    }
}

/// One cooked animation clip referenced by a [`ModelResource`].
///
/// `psxanim_path` resolves with the same precedence rules as
/// [`ResourceData::Texture::psxt_path`]: absolute → project-relative →
/// workspace cwd-relative. Stored relative to the project when the
/// editor registers a bundle, so projects move freely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAnimationClip {
    /// Display name surfaced in the inspector (clip dropdown,
    /// scrubber). Derived from the source filename when registered
    /// via a cooked bundle; user-editable.
    pub name: String,
    /// Path to the cooked `.psxanim` artifact.
    pub psxanim_path: String,
}

/// Cooked PSX model bundle: a `.psxmdl` plus optional atlas
/// `.psxt` plus zero or more `.psxanim` clips.
///
/// All paths follow the project-relative resolution rule shared
/// with `Texture` resources. `clips` is ordered deterministically
/// (by file name at registration time); `default_clip` /
/// `preview_clip` index into that list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelResource {
    /// Path to the cooked `.psxmdl` artifact.
    pub model_path: String,
    /// Optional atlas. Required for textured rendering at runtime;
    /// omitting is allowed for placeholder / debug bundles.
    #[serde(default)]
    pub texture_path: Option<String>,
    /// Cooked animation clips, sorted by file name. Empty for
    /// static models (rendered in bind pose).
    #[serde(default)]
    pub clips: Vec<ModelAnimationClip>,
    /// Index into `clips` used at runtime when no per-instance
    /// override is set. `None` means "first clip if any, else
    /// bind pose".
    #[serde(default)]
    pub default_clip: Option<u16>,
    /// Index into `clips` shown in the editor inspector preview.
    /// Falls back to `default_clip` when unset.
    #[serde(default)]
    pub preview_clip: Option<u16>,
    /// Suggested world-space height in engine units (mirrors the
    /// value the cooker stamped into the `.psxmdl` header). Used
    /// by the inspector for sanity checks and by the editor
    /// preview to size selection gizmos.
    #[serde(default = "default_model_world_height")]
    pub world_height: u16,
}

const fn default_model_world_height() -> u16 {
    1024
}

impl ModelResource {
    /// Index of the clip the editor inspector should preview --
    /// `preview_clip` if set, else `default_clip`, else `None`.
    pub fn effective_preview_clip(&self) -> Option<u16> {
        self.preview_clip.or(self.default_clip)
    }

    /// Index of the clip a runtime instance with no override
    /// should play -- `default_clip` if set, else clip 0 if any
    /// clip exists, else `None`.
    pub fn effective_runtime_clip(&self) -> Option<u16> {
        self.default_clip
            .or_else(|| (!self.clips.is_empty()).then_some(0))
    }
}

/// Gameplay metadata layered on top of a Model. The Model owns
/// the `.psxmdl` / `.psxt` / `.psxanim` artifacts; the Character
/// names which clips fill the idle / walk / run / turn roles
/// and pins the controller's capsule + camera defaults.
///
/// Authoring may leave the model unset (the resource still
/// validates to support partial setup); a Character assigned to
/// the player spawn must resolve to a Model with valid idle and
/// walk clips at cook time.
///
/// Engine units throughout -- same convention used by the rest
/// of the runtime (`sector_size = 1024`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharacterResource {
    /// Backing model. `None` is allowed during authoring;
    /// validated at cook time when assigned to the player.
    #[serde(default)]
    pub model: Option<ResourceId>,
    /// Index into the model's clip list -- played when the
    /// character has no movement input. Required for the player.
    #[serde(default)]
    pub idle_clip: Option<u16>,
    /// Index into the model's clip list -- played while walking.
    /// Required for the player.
    #[serde(default)]
    pub walk_clip: Option<u16>,
    /// Index into the model's clip list -- optional run clip.
    #[serde(default)]
    pub run_clip: Option<u16>,
    /// Index into the model's clip list -- optional turn clip.
    #[serde(default)]
    pub turn_clip: Option<u16>,
    /// Capsule radius (engine units). Used by collision +
    /// editor preview gizmo.
    pub radius: u16,
    /// Capsule height (engine units).
    pub height: u16,
    /// Forward walk speed in engine units per frame at 60 Hz.
    pub walk_speed: i32,
    /// Forward run speed in engine units per frame at 60 Hz.
    pub run_speed: i32,
    /// Yaw rate the controller applies when turning.
    pub turn_speed_degrees_per_second: u16,
    /// Distance the third-person camera trails the character.
    pub camera_distance: i32,
    /// Camera vertical offset above the character origin.
    pub camera_height: i32,
    /// Vertical offset of the camera's look-at target above
    /// the character origin (typically slightly above the head
    /// for a comfortable framing).
    pub camera_target_height: i32,
}

impl CharacterResource {
    /// Sensible defaults for a humanoid third-person character.
    /// Sized for the starter project's 1024-unit sector grid.
    pub const fn defaults() -> Self {
        Self {
            model: None,
            idle_clip: None,
            walk_clip: None,
            run_clip: None,
            turn_clip: None,
            radius: 192,
            height: 1024,
            walk_speed: 48,
            run_speed: 96,
            turn_speed_degrees_per_second: 180,
            camera_distance: 2048,
            camera_height: 1024,
            camera_target_height: 768,
        }
    }
}

impl Default for CharacterResource {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Resource payloads available to editor scenes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceData {
    /// Cooked PSXT artifact reference.
    ///
    /// The editor and the runtime both consume the same `.psxt` blob
    /// -- the runtime via `include_bytes!` at compile time, the editor
    /// via `std::fs::read` at refresh time and `psx_asset::Texture::from_bytes`
    /// to extract pixel + CLUT bytes. PNG → PSXT cooking lives in the
    /// `psxed-tex` CLI; the editor's runtime path doesn't touch PNGs.
    Texture {
        /// Path to the cooked `.psxt` artifact. Resolved at refresh
        /// time first as-is (absolute paths), then relative to the
        /// project file's directory, then relative to the workspace
        /// cwd. The starter project ships paths relative to the repo
        /// root so `cargo run -p frontend` from `/repos/psoxide` finds
        /// the canonical `assets/textures/*.psxt`.
        psxt_path: String,
    },
    /// Editor material.
    Material(MaterialResource),
    /// Cooked animated PSX model -- `.psxmdl` + optional `.psxt`
    /// atlas + animation clips. Instantiated in scenes via
    /// [`NodeKind::MeshInstance`] referencing this resource id.
    Model(ModelResource),
    /// Legacy / generic source mesh path. Kept for backward
    /// compatibility; new authoring should use [`ResourceData::Model`].
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
    /// Gameplay character -- Model + role clip mapping +
    /// capsule/camera defaults. Layered on top of a Model
    /// resource; the player spawn references this to resolve
    /// what to render and how the controller behaves.
    Character(CharacterResource),
}

impl ResourceData {
    /// User-facing type label.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Texture { .. } => "Texture",
            Self::Material(_) => "Material",
            Self::Model(_) => "Model",
            Self::Mesh { .. } => "Mesh",
            Self::Scene { .. } => "Scene",
            Self::Script { .. } => "Script",
            Self::Audio { .. } => "Audio",
            Self::Character(_) => "Character",
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
///
/// Hierarchy convention for level authoring:
/// `Scene root → World (macro) → Room (sector grid) → entity nodes`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Plain organisational node.
    Node,
    /// Spatial transform node.
    Node3D,
    /// Macro-node grouping every Room that belongs to one streamed
    /// region. Holds shared metadata (default fog/ambient) the
    /// editor will surface as it grows; for now it's a pure folder.
    World,
    /// One streamed level chunk: a sector grid plus its child
    /// entities. Cooks to a single `.psxw` blob the runtime loads
    /// in isolation.
    Room {
        /// Authored grid-world payload.
        grid: WorldGrid,
    },
    /// Static or dynamic mesh / model instance.
    ///
    /// `mesh` references either a legacy [`ResourceData::Mesh`] or a
    /// cooked [`ResourceData::Model`]. When it points at a Model,
    /// `animation_clip` selects which clip plays -- an explicit
    /// `Some(idx)` overrides the model's `default_clip`; `None`
    /// inherits the model default. Instances of legacy meshes
    /// ignore this field.
    MeshInstance {
        /// Mesh / model resource.
        mesh: Option<ResourceId>,
        /// Material override (legacy mesh path; ignored for Model
        /// resources, which embed material data in the `.psxmdl`).
        material: Option<ResourceId>,
        /// Per-instance animation clip override.
        #[serde(default)]
        animation_clip: Option<u16>,
    },
    /// Simple authoring light.
    Light {
        /// RGB light colour.
        #[serde(default = "default_light_color")]
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
        /// Character resource that drives this spawn. For the
        /// player spawn this picks the player's model + role
        /// clips + controller params. `None` lets the cook step
        /// auto-pick a Character when exactly one exists, or
        /// fail with a clear error otherwise. Non-player spawns
        /// currently ignore this field.
        #[serde(default)]
        character: Option<ResourceId>,
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
    /// Streaming-graph edge: when the player crosses this volume, the
    /// runtime streams the named entry point of `target_room` into
    /// the World's residency set.
    Portal {
        /// Target Room node by id, or `None` when not yet wired.
        target_room: Option<NodeId>,
        /// Entry-portal label on the target room. The runtime matches
        /// this against a same-named Portal in the destination so the
        /// player spawns at the right side of the door.
        target_entry: String,
        /// Identifier this Portal is known by in its own Room. The
        /// matching Portal in another Room sets `target_entry` to
        /// this string.
        entry_name: String,
    },
}

impl NodeKind {
    /// User-facing label.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Node => "Node",
            Self::Node3D => "Node3D",
            Self::World => "World",
            Self::Room { .. } => "Room",
            Self::MeshInstance { .. } => "MeshInstance",
            Self::Light { .. } => "Light",
            Self::SpawnPoint { .. } => "SpawnPoint",
            Self::Trigger { .. } => "Trigger",
            Self::AudioSource { .. } => "AudioSource",
            Self::Portal { .. } => "Portal",
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
    /// Parent node id, or `None` for the scene root.
    pub parent: Option<NodeId>,
    /// Tree depth from root.
    pub depth: usize,
    /// Index of this node inside its parent's `children` list. Used
    /// by the editor's drag-drop machinery so a "drop above this row"
    /// gesture maps cleanly to `move_node(.., parent, sibling_index)`.
    pub sibling_index: usize,
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

    /// `true` when `ancestor` appears anywhere on the parent chain of
    /// `id`. Includes `id` itself in the check, so callers using this
    /// for cycle detection don't need a separate equality test.
    pub fn is_descendant_of(&self, id: NodeId, ancestor: NodeId) -> bool {
        if id == ancestor {
            return true;
        }
        let mut current = self.node(id).and_then(|n| n.parent);
        while let Some(p) = current {
            if p == ancestor {
                return true;
            }
            current = self.node(p).and_then(|n| n.parent);
        }
        false
    }

    /// Move `id` under `new_parent` at `position` in its child list.
    ///
    /// Refuses (returns `false`) when:
    /// * `id` is the scene root,
    /// * `id` or `new_parent` is missing,
    /// * `new_parent` is `id` or any of its descendants -- that would
    ///   form a cycle.
    ///
    /// `position` clamps to the destination's current child count.
    /// Reordering inside the same parent works because `id` is removed
    /// from the child list before `position` is clamped, so dropping
    /// at "the same slot" is a no-op without UI corner cases.
    pub fn move_node(&mut self, id: NodeId, new_parent: NodeId, position: usize) -> bool {
        if id == self.root {
            return false;
        }
        if self.node(id).is_none() || self.node(new_parent).is_none() {
            return false;
        }
        if self.is_descendant_of(new_parent, id) {
            return false;
        }

        let old_parent = self.node(id).and_then(|n| n.parent);
        if let Some(old) = old_parent {
            if let Some(parent) = self.node_mut(old) {
                parent.children.retain(|c| *c != id);
            }
        }
        if let Some(parent) = self.node_mut(new_parent) {
            let pos = position.min(parent.children.len());
            parent.children.insert(pos, id);
        }
        if let Some(node) = self.node_mut(id) {
            node.parent = Some(new_parent);
        }
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
            parent: node.parent,
            depth,
            sibling_index: node
                .parent
                .and_then(|parent_id| self.node(parent_id))
                .and_then(|parent| parent.children.iter().position(|child| *child == id))
                .unwrap_or(0),
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

    /// Deserialize the default project shipped at
    /// `editor/projects/default/project.ron`. The on-disk RON file
    /// is the single source of truth -- the editor reads the exact
    /// same bytes a `cargo run` would, so changes to the default
    /// project are git-trackable and don't require a rebuild.
    ///
    /// Panics only if the embedded RON drifts out of sync with the
    /// `ProjectDocument` schema; the `embedded_default_project_ron_deserializes`
    /// test guards the build-time invariant.
    pub fn starter() -> Self {
        Self::from_ron_str(DEFAULT_PROJECT_RON)
            .expect("editor/projects/default/project.ron is malformed")
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
    fn snap_height_rounds_to_nearest_quantum() {
        assert_eq!(HEIGHT_QUANTUM, 32);
        // Exact multiples are unchanged (positive + negative).
        assert_eq!(snap_height(0), 0);
        assert_eq!(snap_height(32), 32);
        assert_eq!(snap_height(1024), 1024);
        assert_eq!(snap_height(-32), -32);
        assert_eq!(snap_height(-1024), -1024);
        // Below half-quantum rounds down toward zero.
        assert_eq!(snap_height(15), 0);
        assert_eq!(snap_height(-15), 0);
        // At half-quantum (16), away-from-zero on both sides.
        assert_eq!(snap_height(16), 32);
        assert_eq!(snap_height(-16), -32);
        // Above half-quantum rounds up away from zero.
        assert_eq!(snap_height(17), 32);
        assert_eq!(snap_height(-17), -32);
        // Past one quantum the same rule applies -- round to the
        // nearest multiple.
        assert_eq!(snap_height(47), 32);
        assert_eq!(snap_height(48), 64);
        assert_eq!(snap_height(-47), -32);
        assert_eq!(snap_height(-48), -64);
    }

    #[test]
    fn grid_direction_physical_edges_use_editor_z_convention() {
        assert_eq!(
            GridDirection::North.physical_edge(2, 3),
            Some(GridPhysicalEdge {
                x: 2,
                z: 4,
                axis: GridEdgeAxis::EastWest,
            })
        );
        assert_eq!(
            GridDirection::South.physical_edge(2, 3),
            Some(GridPhysicalEdge {
                x: 2,
                z: 3,
                axis: GridEdgeAxis::EastWest,
            })
        );
        assert_eq!(
            GridDirection::East.physical_edge(2, 3),
            Some(GridPhysicalEdge {
                x: 3,
                z: 3,
                axis: GridEdgeAxis::NorthSouth,
            })
        );
        assert_eq!(
            GridDirection::West.physical_edge(2, 3),
            Some(GridPhysicalEdge {
                x: 2,
                z: 3,
                axis: GridEdgeAxis::NorthSouth,
            })
        );
        assert_eq!(GridDirection::NorthWestSouthEast.physical_edge(2, 3), None);
    }

    #[test]
    fn cell_bounds_match_editor_corner_and_wall_convention() {
        let grid = WorldGrid::empty(2, 2, 1024);
        let bounds = grid.cell_bounds_world(1, 1);

        assert_eq!(bounds.horizontal_corner_xz(Corner::NW), [1024, 2048]);
        assert_eq!(bounds.horizontal_corner_xz(Corner::NE), [2048, 2048]);
        assert_eq!(bounds.horizontal_corner_xz(Corner::SE), [2048, 1024]);
        assert_eq!(bounds.horizontal_corner_xz(Corner::SW), [1024, 1024]);

        assert_eq!(
            bounds.wall_endpoints_xz(GridDirection::North),
            Some(([1024, 2048], [2048, 2048]))
        );
        assert_eq!(
            bounds.wall_endpoints_xz(GridDirection::South),
            Some(([2048, 1024], [1024, 1024]))
        );
    }

    #[test]
    fn stone_room_perimeter_uses_editor_direction_convention() {
        let grid = WorldGrid::stone_room(2, 3, 1024, None, None);

        for x in 0..grid.width {
            assert!(!grid
                .sector(x, 0)
                .unwrap()
                .walls
                .get(GridDirection::South)
                .is_empty());
            assert!(grid
                .sector(x, 0)
                .unwrap()
                .walls
                .get(GridDirection::North)
                .is_empty());
            assert!(!grid
                .sector(x, grid.depth - 1)
                .unwrap()
                .walls
                .get(GridDirection::North)
                .is_empty());
            assert!(grid
                .sector(x, grid.depth - 1)
                .unwrap()
                .walls
                .get(GridDirection::South)
                .is_empty());
        }
    }

    #[test]
    fn editor_to_room_local_round_trip_origin_zero() {
        let grid = WorldGrid::stone_room(3, 3, 1024, None, None);
        for editor in [[0.0_f32, 0.0], [1.5, -0.25], [-1.4, 1.49]] {
            let world = grid.editor_to_room_local(editor);
            let back = grid.room_local_to_editor(world);
            assert!(
                (back[0] - editor[0]).abs() < 1e-3,
                "x: {editor:?} → {back:?}"
            );
            assert!(
                (back[1] - editor[1]).abs() < 1e-3,
                "z: {editor:?} → {back:?}"
            );
        }
    }

    #[test]
    fn editor_to_room_local_round_trip_negative_origin() {
        let mut grid = WorldGrid::stone_room(3, 3, 1024, None, None);
        // Force a -2/-3 origin via the public grow path so the
        // test shape matches what auto-grow actually produces.
        grid.extend_to_include(-2, -3);
        assert_eq!(grid.origin, [-2, -3]);

        for editor in [[0.0_f32, 0.0], [2.0, -1.25], [-3.5, 1.0]] {
            let world = grid.editor_to_room_local(editor);
            let back = grid.room_local_to_editor(world);
            assert!(
                (back[0] - editor[0]).abs() < 1e-3,
                "x: {editor:?} → {back:?}"
            );
            assert!(
                (back[1] - editor[1]).abs() < 1e-3,
                "z: {editor:?} → {back:?}"
            );
        }
    }

    #[test]
    fn editor_cells_to_array_resolves_to_correct_cell() {
        // Plain 3×3, origin [0, 0]: editor (0, 0) is room centre,
        // which falls inside cell (1, 1).
        let grid = WorldGrid::stone_room(3, 3, 1024, None, None);
        assert_eq!(grid.editor_cells_to_array([0.0, 0.0]), Some((1, 1)));
        assert_eq!(grid.editor_cells_to_array([-1.4, -1.4]), Some((0, 0)));
        assert_eq!(grid.editor_cells_to_array([1.4, 1.4]), Some((2, 2)));
        // Past the room edge: out of range.
        assert_eq!(grid.editor_cells_to_array([-2.0, 0.0]), None);
    }

    #[test]
    fn editor_cells_to_array_after_negative_grow_is_origin_aware() {
        // Negative-side grow: origin shifts but the previously-
        // existing cells must remain reachable from the same
        // editor coordinates. After `extend_to_include(-1, 0)` on a
        // 3×3 starter the room becomes width=4, depth=3, origin=[-1,0].
        // Old cell at world-cell (0, 0) is now at array (1, 0).
        let mut grid = WorldGrid::stone_room(3, 3, 1024, None, None);
        grid.extend_to_include(-1, 0);
        assert_eq!(grid.origin, [-1, 0]);
        assert_eq!(grid.width, 4);
        // grid_center_cells = [-1 + 2, 0 + 1.5] = [1.0, 1.5]; cell
        // (1, 0) has world-cell centre [0.5, 0.5], so editor centre
        // is [0.5 - 1.0, 0.5 - 1.5] = [-0.5, -1.0].
        assert_eq!(grid.editor_cells_to_array([-0.5, -1.0]), Some((1, 0)));
        // Newly-included cell at array (0, 0) -- world-cell (-1, 0),
        // editor centre [-0.5 - 1.0, -1.0] = [-1.5, -1.0].
        assert_eq!(grid.editor_cells_to_array([-1.5, -1.0]), Some((0, 0)));
    }

    #[test]
    fn cell_center_world_in_editor_units_matches_helper() {
        let mut grid = WorldGrid::stone_room(4, 5, 1024, None, None);
        grid.extend_to_include(-2, -1);
        let s = grid.sector_size as f32;
        for (sx, sz) in [(0u16, 0u16), (1, 2), (3, 4)] {
            let world_centre = grid.cell_center_world(sx, sz);
            let editor = grid.world_cells_to_editor([world_centre[0] / s, world_centre[1] / s]);
            // Same cell via editor_cells_to_array should round-trip.
            assert_eq!(grid.editor_cells_to_array(editor), Some((sx, sz)));
        }
    }

    #[test]
    fn budget_empty_grid_reports_no_geometry() {
        let grid = WorldGrid::empty(3, 3, 1024);
        let b = grid.budget();
        assert_eq!(b.width, 3);
        assert_eq!(b.depth, 3);
        assert_eq!(b.total_cells, 9);
        assert_eq!(b.populated_cells, 0);
        assert_eq!(b.floors, 0);
        assert_eq!(b.ceilings, 0);
        assert_eq!(b.walls, 0);
        assert_eq!(b.triangles, 0);
        // AssetHeader (12) + WorldHeader (20) + 9 sector records.
        // v1 stores a record per cell whether populated or not, so
        // an "empty" 3×3 still costs 9 * 44 = 396 B in sector
        // records on top of the headers.
        assert_eq!(b.psxw_v1_bytes, 12 + 20 + 9 * 44);
        assert_eq!(b.future_compact_estimated_bytes, 12 + 20 + 9 * 28);
        assert!(!b.over_budget());
    }

    #[test]
    fn budget_starter_room_matches_authored_geometry() {
        let grid = WorldGrid::stone_room(3, 3, 1024, None, None);
        let b = grid.budget();
        assert_eq!(b.populated_cells, 9);
        assert_eq!(b.floors, 9);
        assert_eq!(b.ceilings, 0);
        // Perimeter only: 4 sides * 3 cells = 12 walls.
        assert_eq!(b.walls, 12);
        // 2 tris per face: 9 floors + 12 walls = 21 faces.
        assert_eq!(b.triangles, 42);
        // v2 should be strictly smaller than v1 once any geometry
        // exists -- the size delta is the whole point of the v2
        // record reshape.
        assert!(b.future_compact_estimated_bytes < b.psxw_v1_bytes);
        assert!(!b.over_budget());
    }

    #[test]
    fn budget_max_dimension_grid_within_caps() {
        // Floors-only at MAX_ROOM_WIDTH × MAX_ROOM_DEPTH = 32 × 32.
        // Stresses the byte-cap path without going over MAX_ROOM_TRIANGLES.
        let mut grid = WorldGrid::empty(MAX_ROOM_WIDTH, MAX_ROOM_DEPTH, 1024);
        for x in 0..MAX_ROOM_WIDTH {
            for z in 0..MAX_ROOM_DEPTH {
                grid.set_floor(x, z, 0, None);
            }
        }
        let b = grid.budget();
        assert_eq!(b.populated_cells, 1024);
        assert_eq!(b.floors, 1024);
        assert_eq!(b.triangles, 2048);
        assert!(b.triangles <= MAX_ROOM_TRIANGLES);
        // v1: 32 + 1024 * 44 = 45088 -- over the 64KiB cap on the
        // wall-stack-heavy worst case but fine on floors-only.
        // v2: 32 + 1024 * 28 = 28704 -- well under cap.
        assert!(b.future_compact_estimated_bytes <= MAX_ROOM_BYTES);
        assert!(!b.over_budget());
    }

    #[test]
    fn budget_flags_oversized_room_dimensions() {
        // 64×16 fits the byte cap but blows past MAX_ROOM_WIDTH.
        // The old `over_budget` check only watched triangles +
        // bytes; this test pins the new width/depth check that
        // catches asymmetric over-sized rooms.
        let grid = WorldGrid::empty(MAX_ROOM_WIDTH * 2, MAX_ROOM_DEPTH / 2, 1024);
        let b = grid.budget();
        assert!(b.over_budget(), "{b:?}");
    }

    #[test]
    fn extend_to_include_grows_positively_without_shift() {
        let mut grid = WorldGrid::stone_room(3, 3, 1024, None, None);
        let baseline_floor_world = grid.cell_world_x(0); // 0
        let cell = grid.extend_to_include(5, 1);
        assert_eq!(cell, (5, 1));
        assert_eq!(grid.width, 6);
        assert_eq!(grid.depth, 3);
        assert_eq!(grid.origin, [0, 0]);
        // Old (0, 0) data still at array (0, 0), still at world 0.
        assert_eq!(grid.cell_world_x(0), baseline_floor_world);
        assert!(grid.sector(0, 0).is_some());
    }

    #[test]
    fn extend_to_include_grows_negatively_preserving_world_position() {
        let mut grid = WorldGrid::stone_room(3, 3, 1024, None, None);
        let cell = grid.extend_to_include(-2, 0);
        assert_eq!(cell, (0, 0));
        // Two new columns prepended in -X.
        assert_eq!(grid.width, 5);
        assert_eq!(grid.origin[0], -2);
        // Old (0, 0) data is now at array (2, 0), still at world 0.
        assert_eq!(grid.cell_world_x(2), 0);
        assert!(grid.sector(2, 0).is_some());
        // The newly-included cell at array (0, 0) is empty.
        assert!(grid.sector(0, 0).is_none());
    }

    #[test]
    fn embedded_default_project_ron_deserializes() {
        let project = ProjectDocument::from_ron_str(DEFAULT_PROJECT_RON).unwrap();
        assert!(project.resources.iter().any(|r| matches!(
            &r.data,
            ResourceData::Texture { psxt_path } if psxt_path.ends_with("floor.psxt")
        )));
        assert!(project.resources.iter().any(|r| matches!(
            &r.data,
            ResourceData::Texture { psxt_path } if psxt_path.ends_with("brick-wall.psxt")
        )));
        // Starter ships the obsidian wraith model so users can
        // place + playtest a real animated character without
        // running the import flow first.
        let wraith = project
            .resources
            .iter()
            .find_map(|r| match &r.data {
                ResourceData::Model(m) if r.name == "Obsidian Wraith" => Some(m),
                _ => None,
            })
            .expect("starter model resource missing");
        assert!(wraith.model_path.ends_with("obsidian_wraith.psxmdl"));
        assert!(wraith.texture_path.is_some());
        assert!(!wraith.clips.is_empty());
        assert!(wraith.default_clip.is_some());
    }

    #[test]
    fn starter_model_files_present_on_disk() {
        let root = default_project_dir();
        assert!(root
            .join("assets/models/obsidian_wraith/obsidian_wraith.psxmdl")
            .is_file());
        assert!(root
            .join("assets/models/obsidian_wraith/obsidian_wraith_128x128_8bpp.psxt")
            .is_file());
        assert!(root
            .join("assets/models/obsidian_wraith/obsidian_wraith_idle.psxanim")
            .is_file());
    }

    #[test]
    fn projects_dir_resolves_to_real_directory() {
        assert!(projects_dir().is_dir(), "{}", projects_dir().display());
        assert!(default_project_dir().join("project.ron").is_file());
        assert!(default_project_dir()
            .join("assets/textures/brick-wall.psxt")
            .is_file());
        assert!(default_project_dir()
            .join("assets/textures/floor.psxt")
            .is_file());
    }

    #[test]
    fn starter_project_has_scene_tree_and_resources() {
        let project = ProjectDocument::starter();

        assert_eq!(project.scenes.len(), 1);
        // 2 textures + 2 materials = 4.
        assert!(project.resources.len() >= 4);
        assert!(project
            .active_scene()
            .hierarchy_rows()
            .iter()
            .any(|row| row.name == "Stone Room"));
        let grid = project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Room { grid } => Some(grid),
                _ => None,
            })
            .expect("starter should contain a room node");
        assert!(grid.width > 0);
        assert!(grid.depth > 0);
        assert_eq!(
            grid.sectors.len(),
            grid.width as usize * grid.depth as usize
        );
        assert!(grid.populated_sector_count() > 0);
    }

    #[test]
    fn legacy_project_missing_light_color_and_room_ambient_uses_defaults() {
        let source = DEFAULT_PROJECT_RON
            .replace(
                "kind: Light(color: (255, 236, 198), intensity: 1.0, radius: 4.0)",
                "kind: Light(intensity: 1.0, radius: 4.0)",
            )
            .replace(", ambient_color: (32, 32, 32)", "");

        let project = ProjectDocument::from_ron_str(&source).unwrap();
        let light_color = project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Light { color, .. } => Some(*color),
                _ => None,
            })
            .expect("starter has a light");
        assert_eq!(light_color, default_light_color());

        let ambient = project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Room { grid } => Some(grid.ambient_color),
                _ => None,
            })
            .expect("starter has a room");
        assert_eq!(ambient, default_ambient_color());
    }

    #[test]
    fn adding_node_preserves_parent_child_relationship() {
        let mut scene = Scene::new("Test");

        let room = scene.add_node(
            scene.root,
            "Room",
            NodeKind::Room {
                grid: WorldGrid::empty(2, 2, 1024),
            },
        );
        let child = scene.add_node(
            room,
            "Spawn",
            NodeKind::SpawnPoint {
                player: true,
                character: None,
            },
        );

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
    fn move_node_reparents_and_reorders() {
        let mut scene = Scene::new("Test");
        let a = scene.add_node(scene.root, "A", NodeKind::Node3D);
        let b = scene.add_node(scene.root, "B", NodeKind::Node3D);
        let c = scene.add_node(a, "C", NodeKind::Node3D);

        // Reparent c from a to b at position 0.
        assert!(scene.move_node(c, b, 0));
        assert_eq!(scene.node(c).unwrap().parent, Some(b));
        assert!(scene.node(a).unwrap().children.is_empty());
        assert_eq!(scene.node(b).unwrap().children, vec![c]);

        // Reorder b before a at the root.
        assert!(scene.move_node(b, scene.root, 0));
        assert_eq!(scene.node(scene.root).unwrap().children, vec![b, a]);
    }

    #[test]
    fn move_node_rejects_cycles_and_root() {
        let mut scene = Scene::new("Test");
        let a = scene.add_node(scene.root, "A", NodeKind::Node3D);
        let b = scene.add_node(a, "B", NodeKind::Node3D);

        // Cannot reparent a node under itself.
        assert!(!scene.move_node(a, a, 0));
        // Cannot reparent an ancestor under its descendant.
        assert!(!scene.move_node(a, b, 0));
        // Cannot move the root.
        assert!(!scene.move_node(scene.root, a, 0));
    }

    #[test]
    fn project_roundtrips_through_ron_string() {
        let project = ProjectDocument::starter();
        let ron = project.to_ron_string().unwrap();

        assert!(ron.contains("Stone Room"));
        assert_eq!(ProjectDocument::from_ron_str(&ron).unwrap(), project);
    }

    #[test]
    fn model_resource_roundtrips_through_ron_string() {
        let mut project = ProjectDocument::starter();
        let id = project.add_resource(
            "TestModel",
            ResourceData::Model(ModelResource {
                model_path: "assets/models/x/x.psxmdl".to_string(),
                texture_path: Some("assets/models/x/x.psxt".to_string()),
                clips: vec![
                    ModelAnimationClip {
                        name: "idle".to_string(),
                        psxanim_path: "assets/models/x/x_idle.psxanim".to_string(),
                    },
                    ModelAnimationClip {
                        name: "walk".to_string(),
                        psxanim_path: "assets/models/x/x_walk.psxanim".to_string(),
                    },
                ],
                default_clip: Some(0),
                preview_clip: Some(1),
                world_height: 1280,
            }),
        );
        let ron = project.to_ron_string().unwrap();
        let restored = ProjectDocument::from_ron_str(&ron).unwrap();
        assert_eq!(restored, project);
        let resource = restored.resource(id).unwrap();
        match &resource.data {
            ResourceData::Model(m) => {
                assert_eq!(m.clips.len(), 2);
                assert_eq!(m.default_clip, Some(0));
                assert_eq!(m.preview_clip, Some(1));
                assert_eq!(m.world_height, 1280);
                assert_eq!(m.effective_preview_clip(), Some(1));
                assert_eq!(m.effective_runtime_clip(), Some(0));
            }
            _ => panic!("expected Model"),
        }
    }

    #[test]
    fn mesh_instance_with_animation_clip_roundtrips() {
        let mut project = ProjectDocument::starter();
        let scene = project.active_scene_mut();
        let room_id = scene
            .nodes()
            .iter()
            .find(|n| matches!(n.kind, NodeKind::Room { .. }))
            .map(|n| n.id)
            .unwrap();
        let model_resource_id = ResourceId(99);
        scene.add_node(
            room_id,
            "TestWraith",
            NodeKind::MeshInstance {
                mesh: Some(model_resource_id),
                material: None,
                animation_clip: Some(2),
            },
        );
        let ron = project.to_ron_string().unwrap();
        let restored = ProjectDocument::from_ron_str(&ron).unwrap();
        assert_eq!(restored, project);
        // Confirm the new field survives.
        let surviving = restored
            .active_scene()
            .nodes()
            .iter()
            .find(|n| n.name == "TestWraith")
            .unwrap();
        assert!(matches!(
            surviving.kind,
            NodeKind::MeshInstance {
                mesh: Some(_),
                animation_clip: Some(2),
                ..
            }
        ));
    }

    #[test]
    fn legacy_mesh_instance_without_animation_clip_loads() {
        // Synthesize the pre-extension MeshInstance shape -- `animation_clip`
        // missing -- and confirm `#[serde(default)]` lands `None`.
        let ron = r#"
            (
                name: "Legacy",
                next_resource_id: 1,
                resources: [],
                scenes: [
                    Scene(
                        name: "Demo",
                        next_node_id: 3,
                        root: NodeId(1),
                        nodes: [
                            (
                                id: NodeId(1),
                                name: "Root",
                                parent: None,
                                children: [NodeId(2)],
                                kind: Node3D,
                                transform: (translation: (0.0, 0.0, 0.0), rotation_degrees: (0.0, 0.0, 0.0), scale: (1.0, 1.0, 1.0)),
                            ),
                            (
                                id: NodeId(2),
                                name: "OldMesh",
                                parent: Some(NodeId(1)),
                                children: [],
                                kind: MeshInstance(mesh: None, material: None),
                                transform: (translation: (0.0, 0.0, 0.0), rotation_degrees: (0.0, 0.0, 0.0), scale: (1.0, 1.0, 1.0)),
                            ),
                        ],
                    ),
                ],
            )
        "#;
        let project = ProjectDocument::from_ron_str(ron).unwrap();
        let mesh = project
            .active_scene()
            .nodes()
            .iter()
            .find(|n| n.name == "OldMesh")
            .unwrap();
        assert!(matches!(
            mesh.kind,
            NodeKind::MeshInstance {
                mesh: None,
                material: None,
                animation_clip: None,
            }
        ));
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

    #[test]
    fn corner_surviving_split_picks_diagonal_that_keeps_a_triangle() {
        // Drop NE → only the NW-SE diagonal keeps a triangle.
        // Drop NW → only the NE-SW diagonal keeps a triangle.
        assert_eq!(Corner::NE.surviving_split(), GridSplit::NorthWestSouthEast);
        assert_eq!(Corner::SW.surviving_split(), GridSplit::NorthWestSouthEast);
        assert_eq!(Corner::NW.surviving_split(), GridSplit::NorthEastSouthWest);
        assert_eq!(Corner::SE.surviving_split(), GridSplit::NorthEastSouthWest);
    }

    #[test]
    fn drop_corner_marks_face_as_triangle_and_flips_split() {
        let mut face = GridHorizontalFace::flat(0, None);
        face.split = GridSplit::NorthWestSouthEast; // would die if NW dropped
        face.drop_corner(Corner::NW);
        assert!(face.is_triangle());
        assert_eq!(face.dropped_corner, Some(Corner::NW));
        assert_eq!(face.split, GridSplit::NorthEastSouthWest);

        face.restore_corner();
        assert!(!face.is_triangle());
        assert_eq!(face.dropped_corner, None);
    }

    #[test]
    fn drop_corner_on_wall_marks_triangle() {
        let mut wall = GridVerticalFace::flat(0, 64, None);
        wall.drop_corner(WallCorner::TL);
        assert!(wall.is_triangle());
        assert_eq!(wall.dropped_corner, Some(WallCorner::TL));
    }
}
