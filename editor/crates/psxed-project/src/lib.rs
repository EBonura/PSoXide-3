//! Editor-side project model for PSoXide.
//!
//! This is the authoring model, not the final runtime layout. It keeps a
//! Godot-style scene tree and resource list so the editor can stay pleasant,
//! then later cooker stages flatten it into PS1-friendly world surfaces,
//! texture pages, entity spawns, and engine data.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

pub mod model_import;
pub mod playtest;
pub mod resolve;
pub mod spatial;
pub mod streaming;
pub mod texture_import;
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

/// Quarter-turn texture rotation for authored grid faces.
///
/// PS1 textured polygons carry per-corner 8-bit UVs, not a texture
/// matrix, so these rotations are represented by rewriting the UVs
/// sent with each face.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridUvRotation {
    /// No texture rotation.
    #[default]
    Deg0,
    /// Rotate texture coordinates 90 degrees clockwise on the face.
    Deg90,
    /// Rotate texture coordinates 180 degrees.
    Deg180,
    /// Rotate texture coordinates 270 degrees clockwise on the face.
    Deg270,
}

/// Non-destructive texture-coordinate transform for one grid face.
///
/// `offset` is in PS1 texels and is applied after flip/rotation. It
/// wraps in the 8-bit UV coordinate space, which matches packet-level
/// PS1 UVs; runtime room materials use texture-window state so this
/// can repeat a compact material tile without rebaking the texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridUvTransform {
    /// Signed `[u, v]` texel offset.
    #[serde(default)]
    pub offset: [i16; 2],
    /// Optional `[u, v]` UV span in texels. Zero means "use the
    /// source quad's native span" for that axis.
    #[serde(default, skip_serializing_if = "is_default_uv_span")]
    pub span: [u16; 2],
    /// Quarter-turn rotation.
    #[serde(default)]
    pub rotation: GridUvRotation,
    /// Mirror horizontally before rotation.
    #[serde(default)]
    pub flip_u: bool,
    /// Mirror vertically before rotation.
    #[serde(default)]
    pub flip_v: bool,
}

impl GridUvTransform {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        offset: [0, 0],
        span: [0, 0],
        rotation: GridUvRotation::Deg0,
        flip_u: false,
        flip_v: false,
    };

    /// `true` when this transform leaves UVs unchanged.
    pub const fn is_identity(&self) -> bool {
        self.offset[0] == 0
            && self.offset[1] == 0
            && self.span[0] == 0
            && self.span[1] == 0
            && matches!(self.rotation, GridUvRotation::Deg0)
            && !self.flip_u
            && !self.flip_v
    }

    /// Apply the transform to a quad's corner UVs.
    ///
    /// The input order can be any perimeter order (`[NW, NE, SE, SW]`
    /// for floors or `[BL, BR, TR, TL]` for walls); the transform is
    /// computed inside the UV rectangle spanned by those four points.
    pub fn apply_to_quad(self, uvs: [(u8, u8); 4]) -> [(u8, u8); 4] {
        if self.is_identity() {
            return uvs;
        }
        let bounds = uv_bounds(uvs);
        [
            self.apply_one(uvs[0], bounds),
            self.apply_one(uvs[1], bounds),
            self.apply_one(uvs[2], bounds),
            self.apply_one(uvs[3], bounds),
        ]
    }

    fn apply_one(self, uv: (u8, u8), bounds: UvBounds) -> (u8, u8) {
        let width = bounds.max_u - bounds.min_u;
        let height = bounds.max_v - bounds.min_v;
        if width == 0 || height == 0 {
            return (
                wrap_uv(uv.0 as i32 + self.offset[0] as i32),
                wrap_uv(uv.1 as i32 + self.offset[1] as i32),
            );
        }

        let mut u = uv.0 as i32 - bounds.min_u;
        let mut v = uv.1 as i32 - bounds.min_v;
        if self.flip_u {
            u = width - u;
        }
        if self.flip_v {
            v = height - v;
        }

        let (u, v) = match self.rotation {
            GridUvRotation::Deg0 => (u, v),
            GridUvRotation::Deg90 => (
                width - scale_rounded(v, width, height),
                scale_rounded(u, height, width),
            ),
            GridUvRotation::Deg180 => (width - u, height - v),
            GridUvRotation::Deg270 => (
                scale_rounded(v, width, height),
                height - scale_rounded(u, height, width),
            ),
        };
        let span_u = self.effective_span_axis(0, width);
        let span_v = self.effective_span_axis(1, height);
        let u = scale_rounded(u, span_u, width);
        let v = scale_rounded(v, span_v, height);

        (
            wrap_uv(bounds.min_u + u + self.offset[0] as i32),
            wrap_uv(bounds.min_v + v + self.offset[1] as i32),
        )
    }

    fn effective_span_axis(self, axis: usize, fallback: i32) -> i32 {
        let span = self.span[axis];
        if span == 0 {
            fallback
        } else {
            i32::from(span.min(255))
        }
    }
}

impl Default for GridUvTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Debug, Clone, Copy)]
struct UvBounds {
    min_u: i32,
    max_u: i32,
    min_v: i32,
    max_v: i32,
}

fn uv_bounds(uvs: [(u8, u8); 4]) -> UvBounds {
    let mut min_u = uvs[0].0 as i32;
    let mut max_u = min_u;
    let mut min_v = uvs[0].1 as i32;
    let mut max_v = min_v;
    for (u, v) in uvs {
        let u = u as i32;
        let v = v as i32;
        min_u = min_u.min(u);
        max_u = max_u.max(u);
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    UvBounds {
        min_u,
        max_u,
        min_v,
        max_v,
    }
}

fn scale_rounded(value: i32, numerator: i32, denominator: i32) -> i32 {
    if denominator == 0 {
        0
    } else {
        (value.saturating_mul(numerator) + denominator / 2) / denominator
    }
}

fn wrap_uv(value: i32) -> u8 {
    value.rem_euclid(256) as u8
}

pub(crate) fn wrap_tiled_uv_offset_i16(value: i64) -> i16 {
    value.rem_euclid(i64::from(psxed_format::world::TILE_UV)) as i16
}

const fn is_default_uv_span(span: &[u16; 2]) -> bool {
    span[0] == 0 && span[1] == 0
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

    /// Opposite cardinal edge. Diagonals do not have a single
    /// opposite perimeter edge.
    pub const fn opposite_cardinal(self) -> Option<Self> {
        match self {
            Self::North => Some(Self::South),
            Self::East => Some(Self::West),
            Self::South => Some(Self::North),
            Self::West => Some(Self::East),
            Self::NorthWestSouthEast | Self::NorthEastSouthWest => None,
        }
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
    /// Non-destructive texture-coordinate transform.
    #[serde(default, skip_serializing_if = "GridUvTransform::is_identity")]
    pub uv: GridUvTransform,
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
            uv: GridUvTransform::IDENTITY,
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
    /// Non-destructive texture-coordinate transform.
    #[serde(default, skip_serializing_if = "GridUvTransform::is_identity")]
    pub uv: GridUvTransform,
    /// Whether collision treats this wall as blocking.
    pub solid: bool,
    /// `Some(corner)` when one wall corner has been deleted,
    /// turning the wall quad into a triangle. Default `None`.
    #[serde(default)]
    pub dropped_corner: Option<WallCorner>,
}

impl GridVerticalFace {
    /// Wall from explicit per-corner heights in `[BL, BR, TR, TL]`
    /// order.
    pub const fn with_heights(heights: [i32; 4], material: Option<ResourceId>) -> Self {
        Self {
            heights,
            material,
            uv: GridUvTransform::IDENTITY,
            solid: true,
            dropped_corner: None,
        }
    }

    /// Flat wall from `bottom` to `top`.
    pub const fn flat(bottom: i32, top: i32, material: Option<ResourceId>) -> Self {
        Self::with_heights([bottom, bottom, top, top], material)
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

    /// Set this wall's V span so texel density follows the world
    /// grid: `TILE_UV` texels cover one sector-height.
    ///
    /// The wall geometry is not changed. Returns `true` when the
    /// requested span had to be clamped to the PS1 packet UV range.
    pub fn autotile_uv(&mut self, sector_size: i32) -> bool {
        let (span_v, clamped) = uv_span_for_world_span(self.max_vertical_span(), sector_size);
        self.uv.span[0] = 0;
        self.uv.span[1] = stored_uv_span(span_v);
        clamped
    }

    /// Number of runtime wall records needed to draw this wall
    /// without asking one PS1 primitive to encode a V span beyond
    /// the packet's 8-bit UV coordinate range.
    pub fn autotile_segment_count(&self, sector_size: i32) -> usize {
        if !self.should_split_autotile_segments(sector_size) {
            return 1;
        }
        let sector_size = sector_size.max(1) as usize;
        let max_span = self.max_vertical_span().max(0) as usize;
        ((max_span + sector_size - 1) / sector_size).max(1)
    }

    /// Split this wall into sector-height stack entries and retile
    /// each segment so every cooked primitive stays within the
    /// packet's 8-bit V coordinate range.
    pub fn split_into_autotile_segments(&self, sector_size: i32) -> Vec<Self> {
        if !self.should_split_autotile_segments(sector_size) {
            return vec![self.clone()];
        }
        let sector_size = sector_size.max(1);
        let max_span = self.max_vertical_span();
        if max_span == 0 {
            return vec![self.clone()];
        }

        let mut out = Vec::with_capacity(self.autotile_segment_count(sector_size));
        let mut start = 0;
        while start < max_span {
            let end = start.saturating_add(sector_size).min(max_span);
            let mut wall = self.clone();
            wall.heights = self.segment_heights(start, end, max_span);
            let (span_v, _) = uv_span_for_world_span(end.saturating_sub(start), sector_size);
            wall.uv.span[1] = stored_uv_span(span_v);
            let start_v = div_round_i64(
                i64::from(start) * i64::from(psxed_format::world::TILE_UV),
                i64::from(sector_size),
            );
            wall.uv.offset[1] =
                wrap_tiled_uv_offset_i16(i64::from(self.uv.offset[1]).saturating_add(start_v));
            out.push(wall);
            start = end;
        }
        out
    }

    /// Split this wall into sector-height stack entries without
    /// changing its material or UV settings.
    pub fn split_into_height_segments(&self, sector_size: i32) -> Vec<Self> {
        if self.is_triangle() {
            return vec![self.clone()];
        }
        let sector_size = sector_size.max(1);
        let max_span = self.max_vertical_span();
        if max_span == 0 {
            return vec![self.clone()];
        }

        let mut out = Vec::new();
        let mut start = 0;
        while start < max_span {
            let end = start.saturating_add(sector_size).min(max_span);
            let mut wall = self.clone();
            wall.heights = self.segment_heights(start, end, max_span);
            out.push(wall);
            start = end;
        }
        out
    }

    fn should_split_autotile_segments(&self, sector_size: i32) -> bool {
        if self.is_triangle() {
            return false;
        }
        let max_span = self.max_vertical_span();
        let (expected_span, clamped) = uv_span_for_world_span(max_span, sector_size);
        clamped && (self.uv.span[1] == 0 || self.uv.span[1] == stored_uv_span(expected_span))
    }

    fn segment_heights(&self, start: i32, end: i32, max_span: i32) -> [i32; 4] {
        [
            lerp_i32_ratio(
                self.heights[WallCorner::BL.idx()],
                self.heights[WallCorner::TL.idx()],
                start,
                max_span,
            ),
            lerp_i32_ratio(
                self.heights[WallCorner::BR.idx()],
                self.heights[WallCorner::TR.idx()],
                start,
                max_span,
            ),
            lerp_i32_ratio(
                self.heights[WallCorner::BR.idx()],
                self.heights[WallCorner::TR.idx()],
                end,
                max_span,
            ),
            lerp_i32_ratio(
                self.heights[WallCorner::BL.idx()],
                self.heights[WallCorner::TL.idx()],
                end,
                max_span,
            ),
        ]
    }

    fn max_vertical_span(&self) -> i32 {
        let left_span =
            self.heights[WallCorner::TL.idx()].saturating_sub(self.heights[WallCorner::BL.idx()]);
        let right_span =
            self.heights[WallCorner::TR.idx()].saturating_sub(self.heights[WallCorner::BR.idx()]);
        left_span.unsigned_abs().max(right_span.unsigned_abs()) as i32
    }
}

fn uv_span_for_world_span(world_span: i32, sector_size: i32) -> (u16, bool) {
    if world_span <= 0 {
        return (u16::from(psxed_format::world::TILE_UV), false);
    }
    let sector_size = sector_size.max(1);
    let unclamped = div_round_i64(
        i64::from(world_span) * i64::from(psxed_format::world::TILE_UV),
        i64::from(sector_size),
    );
    let texels = unclamped.clamp(1, 255) as u16;
    (texels, unclamped > 255)
}

fn stored_uv_span(span: u16) -> u16 {
    if span == u16::from(psxed_format::world::TILE_UV) {
        0
    } else {
        span
    }
}

fn lerp_i32_ratio(a: i32, b: i32, numerator: i32, denominator: i32) -> i32 {
    if denominator <= 0 {
        return a;
    }
    let delta = i64::from(b).saturating_sub(i64::from(a));
    i64::from(a)
        .saturating_add(div_round_i64(
            delta.saturating_mul(i64::from(numerator)),
            i64::from(denominator),
        ))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn div_round_i64(numerator: i64, denominator: i64) -> i64 {
    if denominator == 0 {
        return 0;
    }
    if numerator >= 0 {
        numerator.saturating_add(denominator / 2) / denominator
    } else {
        numerator.saturating_sub(denominator / 2) / denominator
    }
}

/// Array-sector rectangle enclosing authored grid geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldGridFootprint {
    pub x: u16,
    pub z: u16,
    pub width: u16,
    pub depth: u16,
}

impl WorldGridFootprint {
    pub fn end_x(self) -> u16 {
        self.x + self.width
    }

    pub fn end_z(self) -> u16 {
        self.z + self.depth
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HorizontalSurface {
    Floor,
    Ceiling,
}

impl HorizontalSurface {
    fn edge_heights(self, sector: &GridSector, direction: GridDirection) -> Option<[i32; 2]> {
        let heights = match self {
            Self::Floor => sector.floor.as_ref()?.heights,
            Self::Ceiling => sector.ceiling.as_ref()?.heights,
        };
        horizontal_edge_heights_for_wall(heights, direction)
    }
}

fn horizontal_edge_heights_for_wall(
    heights: [i32; 4],
    direction: GridDirection,
) -> Option<[i32; 2]> {
    match direction {
        GridDirection::North => Some([heights[Corner::NW.idx()], heights[Corner::NE.idx()]]),
        GridDirection::East => Some([heights[Corner::NE.idx()], heights[Corner::SE.idx()]]),
        GridDirection::South => Some([heights[Corner::SE.idx()], heights[Corner::SW.idx()]]),
        GridDirection::West => Some([heights[Corner::SW.idx()], heights[Corner::NW.idx()]]),
        GridDirection::NorthWestSouthEast | GridDirection::NorthEastSouthWest => None,
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
/// World grid size quantum. The editor stores one sector size per
/// World node and snaps it to this step so room/cook math stays
/// integer and PSX-friendly.
pub const WORLD_SECTOR_SIZE_QUANTUM: i32 = 128;
/// Default sector size used by starter/legacy projects.
pub const DEFAULT_WORLD_SECTOR_SIZE: i32 = 1024;
/// Default wall span when no ceiling is authored above the edge.
pub const DEFAULT_WALL_HEIGHT_SECTORS: i32 = 2;
/// Minimum authored sector size.
pub const MIN_WORLD_SECTOR_SIZE: i32 = WORLD_SECTOR_SIZE_QUANTUM;
/// Maximum authored sector size. This is an authoring sanity cap,
/// not a PSX wire-format limit.
pub const MAX_WORLD_SECTOR_SIZE: i32 = 8192;
/// Fixed-point one for authored model resource scale.
pub const MODEL_SCALE_ONE_Q8: u16 = 256;

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

/// Snap a requested World sector size to a positive 128-unit grid.
pub fn snap_world_sector_size(size: i32) -> i32 {
    let clamped = size.clamp(MIN_WORLD_SECTOR_SIZE, MAX_WORLD_SECTOR_SIZE);
    ((clamped + WORLD_SECTOR_SIZE_QUANTUM / 2) / WORLD_SECTOR_SIZE_QUANTUM)
        * WORLD_SECTOR_SIZE_QUANTUM
}

fn default_world_sector_size() -> i32 {
    DEFAULT_WORLD_SECTOR_SIZE
}

fn default_wall_height_for_sector_size(sector_size: i32) -> i32 {
    sector_size.saturating_mul(DEFAULT_WALL_HEIGHT_SECTORS)
}

fn default_model_scale_q8() -> [u16; 3] {
    [MODEL_SCALE_ONE_Q8; 3]
}

fn scale_i32_ratio(value: i32, from: i32, to: i32) -> i32 {
    if from <= 0 || from == to {
        return value;
    }
    (((value as i64) * (to as i64) + (from as i64 / 2)) / (from as i64)) as i32
}

fn scale_u16_ratio(value: u16, from: i32, to: i32) -> u16 {
    scale_i32_ratio(value as i32, from, to).clamp(0, u16::MAX as i32) as u16
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
    /// `width * depth`. `.psxw` stores a sector record for
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
    /// Current `.psxw` wire size. The format stores a record for **every** cell, so this
    /// uses `total_cells`, not `populated_cells`.
    pub psxw_bytes: usize,
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
            || self.psxw_bytes > MAX_ROOM_BYTES
    }
}

const ASSET_HEADER_BYTES: usize = 12;
const WORLD_HEADER_BYTES: usize = 20;
const PSXW_SECTOR_BYTES: usize = psxed_format::world::SectorRecord::SIZE;
const PSXW_WALL_BYTES: usize = psxed_format::world::WallRecord::SIZE;
const FUTURE_COMPACT_SECTOR_BYTES: usize = 28;
const FUTURE_COMPACT_WALL_BYTES: usize = 12;

const fn default_ambient_color() -> [u8; 3] {
    [32, 32, 32]
}

const fn default_fog_color() -> [u8; 3] {
    [24, 28, 34]
}

const fn default_fog_near() -> i32 {
    4096
}

const fn default_fog_far() -> i32 {
    16384
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
    /// Depth-cue far color for this room.
    #[serde(default = "default_fog_color")]
    pub fog_color: [u8; 3],
    /// Start distance for authored fog/depth cue in engine units.
    #[serde(default = "default_fog_near")]
    pub fog_near: i32,
    /// Fully-fogged distance for authored fog/depth cue in engine units.
    #[serde(default = "default_fog_far")]
    pub fog_far: i32,
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
            fog_color: default_fog_color(),
            fog_near: default_fog_near(),
            fog_far: default_fog_far(),
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
        let wall_top = default_wall_height_for_sector_size(sector_size);
        for x in 0..width {
            for z in 0..depth {
                grid.set_floor(x, z, 0, floor_material);
                if z == depth.saturating_sub(1) {
                    grid.add_wall(x, z, GridDirection::North, 0, wall_top, wall_material);
                }
                if x == width.saturating_sub(1) {
                    grid.add_wall(x, z, GridDirection::East, 0, wall_top, wall_material);
                }
                if z == 0 {
                    grid.add_wall(x, z, GridDirection::South, 0, wall_top, wall_material);
                }
                if x == 0 {
                    grid.add_wall(x, z, GridDirection::West, 0, wall_top, wall_material);
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

    /// Add a wall whose bottom edge follows the floor edge under it
    /// and whose top edge follows the ceiling edge when present.
    /// Missing ceilings fall back to a two-sector wall span above
    /// each bottom endpoint.
    pub fn add_wall_aligned_to_surfaces(
        &mut self,
        x: u16,
        z: u16,
        direction: GridDirection,
        material: Option<ResourceId>,
    ) {
        let heights = self.wall_heights_aligned_to_surfaces(x, z, direction);
        if let Some(sector) = self.ensure_sector(x, z) {
            sector
                .walls
                .get_mut(direction)
                .push(GridVerticalFace::with_heights(heights, material));
        }
    }

    /// Candidate wall heights for editor placement on a cardinal
    /// edge. The returned order is `[BL, BR, TR, TL]`.
    pub fn wall_heights_aligned_to_surfaces(
        &self,
        x: u16,
        z: u16,
        direction: GridDirection,
    ) -> [i32; 4] {
        let bottom = self
            .floor_edge_heights_for_wall(x, z, direction)
            .unwrap_or([0, 0]);
        let top = self
            .ceiling_edge_heights_for_wall(x, z, direction)
            .unwrap_or_else(|| {
                let height = default_wall_height_for_sector_size(self.sector_size);
                [
                    bottom[0].saturating_add(height),
                    bottom[1].saturating_add(height),
                ]
            });
        [bottom[0], bottom[1], top[1], top[0]]
    }

    /// Same as [`Self::wall_heights_aligned_to_surfaces`], but
    /// addressed by world-cell coordinates so hover previews can
    /// match clicks that will auto-grow the grid on commit.
    pub fn wall_heights_aligned_to_surfaces_for_world_cell(
        &self,
        wcx: i32,
        wcz: i32,
        direction: GridDirection,
    ) -> [i32; 4] {
        let bottom = self
            .horizontal_edge_heights_for_world_wall(wcx, wcz, direction, HorizontalSurface::Floor)
            .unwrap_or([0, 0]);
        let top = self
            .horizontal_edge_heights_for_world_wall(wcx, wcz, direction, HorizontalSurface::Ceiling)
            .unwrap_or_else(|| {
                let height = default_wall_height_for_sector_size(self.sector_size);
                [
                    bottom[0].saturating_add(height),
                    bottom[1].saturating_add(height),
                ]
            });
        [bottom[0], bottom[1], top[1], top[0]]
    }

    fn floor_edge_heights_for_wall(
        &self,
        x: u16,
        z: u16,
        direction: GridDirection,
    ) -> Option<[i32; 2]> {
        self.horizontal_edge_heights_for_wall(x, z, direction, HorizontalSurface::Floor)
    }

    fn ceiling_edge_heights_for_wall(
        &self,
        x: u16,
        z: u16,
        direction: GridDirection,
    ) -> Option<[i32; 2]> {
        self.horizontal_edge_heights_for_wall(x, z, direction, HorizontalSurface::Ceiling)
    }

    fn horizontal_edge_heights_for_wall(
        &self,
        x: u16,
        z: u16,
        direction: GridDirection,
        surface: HorizontalSurface,
    ) -> Option<[i32; 2]> {
        if let Some(heights) = self
            .sector(x, z)
            .and_then(|sector| surface.edge_heights(sector, direction))
        {
            return Some(heights);
        }

        let (nx, nz, opposite) = self.neighbor_across_cardinal_edge(x, z, direction)?;
        let mut heights = self
            .sector(nx, nz)
            .and_then(|sector| surface.edge_heights(sector, opposite))?;
        heights.swap(0, 1);
        Some(heights)
    }

    fn neighbor_across_cardinal_edge(
        &self,
        x: u16,
        z: u16,
        direction: GridDirection,
    ) -> Option<(u16, u16, GridDirection)> {
        let opposite = direction.opposite_cardinal()?;
        let (nx, nz) = match direction {
            GridDirection::North => (x, z.checked_add(1)?),
            GridDirection::East => (x.checked_add(1)?, z),
            GridDirection::South => (x, z.checked_sub(1)?),
            GridDirection::West => (x.checked_sub(1)?, z),
            GridDirection::NorthWestSouthEast | GridDirection::NorthEastSouthWest => return None,
        };
        (nx < self.width && nz < self.depth).then_some((nx, nz, opposite))
    }

    fn horizontal_edge_heights_for_world_wall(
        &self,
        wcx: i32,
        wcz: i32,
        direction: GridDirection,
        surface: HorizontalSurface,
    ) -> Option<[i32; 2]> {
        if let Some((sx, sz)) = self.world_cell_to_array(wcx, wcz) {
            if let Some(heights) = self
                .sector(sx, sz)
                .and_then(|sector| surface.edge_heights(sector, direction))
            {
                return Some(heights);
            }
        }

        let opposite = direction.opposite_cardinal()?;
        let (nwcx, nwcz) = match direction {
            GridDirection::North => (wcx, wcz.saturating_add(1)),
            GridDirection::East => (wcx.saturating_add(1), wcz),
            GridDirection::South => (wcx, wcz.saturating_sub(1)),
            GridDirection::West => (wcx.saturating_sub(1), wcz),
            GridDirection::NorthWestSouthEast | GridDirection::NorthEastSouthWest => return None,
        };
        let (sx, sz) = self.world_cell_to_array(nwcx, nwcz)?;
        let mut heights = self
            .sector(sx, sz)
            .and_then(|sector| surface.edge_heights(sector, opposite))?;
        heights.swap(0, 1);
        Some(heights)
    }

    /// Number of populated sectors.
    pub fn populated_sector_count(&self) -> usize {
        self.sectors
            .iter()
            .flatten()
            .filter(|sector| sector.has_geometry())
            .count()
    }

    /// Rectangle enclosing every sector that emits authored
    /// geometry. Empty allocated cells are capacity, not room
    /// footprint, so they do not influence bounds or streaming
    /// subdivision.
    pub fn authored_footprint(&self) -> Option<WorldGridFootprint> {
        let mut min_x = self.width;
        let mut min_z = self.depth;
        let mut max_x = 0u16;
        let mut max_z = 0u16;
        let mut found = false;
        for x in 0..self.width {
            for z in 0..self.depth {
                let Some(sector) = self.sector(x, z) else {
                    continue;
                };
                if !sector.has_geometry() {
                    continue;
                }
                found = true;
                min_x = min_x.min(x);
                min_z = min_z.min(z);
                max_x = max_x.max(x);
                max_z = max_z.max(z);
            }
        }
        found.then_some(WorldGridFootprint {
            x: min_x,
            z: min_z,
            width: max_x - min_x + 1,
            depth: max_z - min_z + 1,
        })
    }

    /// Budget for the authored footprint only. This is the number
    /// authors care about when sparse grid allocation has grown past
    /// the currently placed tiles.
    pub fn authored_budget(&self) -> WorldGridBudget {
        self.authored_footprint()
            .and_then(|f| self.budget_for_rect(f.x, f.z, f.width, f.depth))
            .unwrap_or_default()
    }

    /// Snapshot of the allocated grid rectangle + cooked-byte
    /// estimate. Use [`Self::authored_budget`] when empty capacity
    /// should not count as room footprint.
    pub fn budget(&self) -> WorldGridBudget {
        self.budget_for_rect(0, 0, self.width, self.depth)
            .unwrap_or_default()
    }

    /// Snapshot of one rectangular grid area. The rectangle is in
    /// array-sector coordinates, not world-origin-adjusted cells.
    /// Returns `None` for empty or out-of-bounds rectangles.
    pub fn budget_for_rect(
        &self,
        x: u16,
        z: u16,
        width: u16,
        depth: u16,
    ) -> Option<WorldGridBudget> {
        if width == 0 || depth == 0 {
            return None;
        }
        let x1 = x.checked_add(width)?;
        let z1 = z.checked_add(depth)?;
        if x1 > self.width || z1 > self.depth {
            return None;
        }
        let mut b = WorldGridBudget {
            width,
            depth,
            total_cells: (width as usize) * (depth as usize),
            ..Default::default()
        };
        for sx in x..x1 {
            for sz in z..z1 {
                let Some(sector) = self.sector(sx, sz) else {
                    continue;
                };
                if !sector.has_geometry() {
                    continue;
                }
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
                    let count = sector
                        .walls
                        .get(direction)
                        .iter()
                        .map(|wall| wall.autotile_segment_count(self.sector_size))
                        .sum::<usize>();
                    b.walls += count;
                    b.triangles += count * 2;
                }
            }
        }
        // Active wire layout (matches `psxed_format::world` records).
        // `.psxw` stores a sector record for every cell -- empty or
        // not -- so the byte count uses `total_cells`. Using
        // `populated_cells` here was the original bug: it under-
        // reported the wire size by one sector record per empty cell.
        // Target compact-format sizes for the planning estimate.
        // See `docs/world-format-roadmap.md`. Plain numeric
        // constants rather than struct sizes so this block doesn't
        // pretend a v2 format exists in code.
        b.psxw_bytes = ASSET_HEADER_BYTES
            + WORLD_HEADER_BYTES
            + b.total_cells * PSXW_SECTOR_BYTES
            + b.walls * PSXW_WALL_BYTES;
        b.future_compact_estimated_bytes = ASSET_HEADER_BYTES
            + WORLD_HEADER_BYTES
            + b.total_cells * FUTURE_COMPACT_SECTOR_BYTES
            + b.walls * FUTURE_COMPACT_WALL_BYTES;
        Some(b)
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

    /// Change this grid's sector size and scale engine-unit
    /// vertical geometry by the same ratio. X/Z authored positions
    /// are stored in sector units, so they inherit the new physical
    /// size through `sector_size`.
    pub fn rescale_sector_size(&mut self, new_sector_size: i32) {
        let new_sector_size = snap_world_sector_size(new_sector_size);
        let old_sector_size = self.sector_size.max(1);
        if old_sector_size == new_sector_size {
            self.sector_size = new_sector_size;
            self.snap_heights_to_quantum();
            return;
        }
        for sector in self.sectors.iter_mut().flatten() {
            if let Some(face) = &mut sector.floor {
                for h in &mut face.heights {
                    *h = snap_height(scale_i32_ratio(*h, old_sector_size, new_sector_size));
                }
            }
            if let Some(face) = &mut sector.ceiling {
                for h in &mut face.heights {
                    *h = snap_height(scale_i32_ratio(*h, old_sector_size, new_sector_size));
                }
            }
            for direction in GridDirection::ALL {
                for wall in sector.walls.get_mut(direction) {
                    for h in &mut wall.heights {
                        *h = snap_height(scale_i32_ratio(*h, old_sector_size, new_sector_size));
                    }
                }
            }
        }
        self.fog_near = scale_i32_ratio(self.fog_near, old_sector_size, new_sector_size).max(0);
        self.fog_far = scale_i32_ratio(self.fog_far, old_sector_size, new_sector_size)
            .max(self.fog_near + HEIGHT_QUANTUM);
        self.sector_size = new_sector_size;
    }

    /// Snap all authored vertical geometry to the cooker-supported
    /// height quantum. This is load/save normalization for stale or
    /// hand-edited project data; live editor controls call
    /// [`snap_height`] at the point of edit.
    pub fn snap_heights_to_quantum(&mut self) {
        for sector in self.sectors.iter_mut().flatten() {
            if let Some(face) = &mut sector.floor {
                for h in &mut face.heights {
                    *h = snap_height(*h);
                }
            }
            if let Some(face) = &mut sector.ceiling {
                for h in &mut face.heights {
                    *h = snap_height(*h);
                }
            }
            for direction in GridDirection::ALL {
                for wall in sector.walls.get_mut(direction) {
                    for h in &mut wall.heights {
                        *h = snap_height(*h);
                    }
                }
            }
        }
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

/// Named model attachment point, usually bound to a skeleton
/// joint. Runtime composition is:
/// `entity transform × joint pose × socket local transform`.
///
/// Offsets are integer model/engine units and rotations are Q12
/// turn units (`4096 = 360°`) so project data can be cooked
/// directly for the PS1 without preserving floats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachmentSocket {
    /// User-facing socket name (`right_hand_grip`, `back_slot`, …).
    pub name: String,
    /// Joint index in the cooked `.psxmdl` skeleton.
    pub joint: u16,
    /// Local translation relative to the joint pose.
    #[serde(default)]
    pub translation: [i32; 3],
    /// Local Euler rotation in Q12 turns: X / Y / Z, 4096 per turn.
    #[serde(default)]
    pub rotation_q12: [i16; 3],
}

impl AttachmentSocket {
    /// Common right-hand default for humanoid rigs.
    pub fn right_hand_grip() -> Self {
        Self {
            name: default_character_socket(),
            joint: 0,
            translation: [0, 0, 0],
            rotation_q12: [0, 0, 0],
        }
    }
}

/// Pivot on a weapon model that should land on a character socket.
/// A sword normally uses `grip`; a shield might use `handle`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeaponGrip {
    /// User-facing grip/pivot name.
    pub name: String,
    /// Local translation inside the weapon model.
    #[serde(default)]
    pub translation: [i32; 3],
    /// Local Euler rotation in Q12 turns: X / Y / Z, 4096 per turn.
    #[serde(default)]
    pub rotation_q12: [i16; 3],
}

impl Default for WeaponGrip {
    fn default() -> Self {
        Self {
            name: default_weapon_grip(),
            translation: [0, 0, 0],
            rotation_q12: [0, 0, 0],
        }
    }
}

/// Weapon hit volume, stored relative to the weapon grip/pivot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeaponHitShape {
    /// Oriented box hit volume. `half_extents` are local axes.
    Box {
        /// Local center relative to the weapon grip.
        center: [i32; 3],
        /// Half extents in engine/model units.
        half_extents: [u16; 3],
    },
    /// Capsule hit volume, useful for blades, clubs, and spears.
    Capsule {
        /// Local capsule start relative to the weapon grip.
        start: [i32; 3],
        /// Local capsule end relative to the weapon grip.
        end: [i32; 3],
        /// Capsule radius in engine/model units.
        radius: u16,
    },
}

impl Default for WeaponHitShape {
    fn default() -> Self {
        Self::Capsule {
            start: [0, 0, 0],
            end: [0, 512, 0],
            radius: 48,
        }
    }
}

/// One named active hitbox window for a weapon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeaponHitbox {
    /// User-facing hitbox name.
    pub name: String,
    /// Local hit volume.
    #[serde(default)]
    pub shape: WeaponHitShape,
    /// First animation frame where the hitbox is active.
    #[serde(default)]
    pub active_start_frame: u16,
    /// Last animation frame where the hitbox is active.
    #[serde(default)]
    pub active_end_frame: u16,
}

impl Default for WeaponHitbox {
    fn default() -> Self {
        Self {
            name: "Main Hit".to_string(),
            shape: WeaponHitShape::default(),
            active_start_frame: 0,
            active_end_frame: 0,
        }
    }
}

/// Gameplay weapon resource: model reference, grip/pivot, and
/// authored attack hit volumes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeaponResource {
    /// Visual model used for the weapon. `None` is allowed during
    /// authoring so hitboxes can be blocked in before art lands.
    #[serde(default)]
    pub model: Option<ResourceId>,
    /// Which character socket this weapon expects by default.
    #[serde(default = "default_character_socket")]
    pub default_character_socket: String,
    /// Weapon-local grip/pivot that aligns to the character socket.
    #[serde(default)]
    pub grip: WeaponGrip,
    /// Hit volumes authored relative to [`Self::grip`].
    #[serde(default)]
    pub hitboxes: Vec<WeaponHitbox>,
}

impl WeaponResource {
    /// Minimal editable weapon.
    pub fn defaults() -> Self {
        Self {
            model: None,
            default_character_socket: default_character_socket(),
            grip: WeaponGrip::default(),
            hitboxes: vec![WeaponHitbox::default()],
        }
    }
}

impl Default for WeaponResource {
    fn default() -> Self {
        Self::defaults()
    }
}

fn default_character_socket() -> String {
    "right_hand_grip".to_string()
}

fn default_weapon_grip() -> String {
    "grip".to_string()
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
    /// Authored bake-time scale in Q8 fixed point (`256 = 1.0`).
    /// Stored as integers so project data mirrors the PS1/runtime
    /// constraint; any application to mesh data must happen during
    /// cook/import, not as runtime floats.
    #[serde(default = "default_model_scale_q8")]
    pub scale_q8: [u16; 3],
    /// Named sockets used by equipment, VFX, and hitbox authoring.
    #[serde(default)]
    pub attachments: Vec<AttachmentSocket>,
}

const fn default_model_world_height() -> u16 {
    1024
}

impl ModelResource {
    /// Human-readable scale factor for one axis.
    pub fn scale_axis(&self, axis: usize) -> f32 {
        self.scale_q8
            .get(axis)
            .copied()
            .unwrap_or(MODEL_SCALE_ONE_Q8) as f32
            / MODEL_SCALE_ONE_Q8 as f32
    }

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
    /// to extract pixel + CLUT bytes. PNG/JPG/BMP → PSXT cooking lives
    /// in `texture_import` and the `psxed tex` CLI; runtime paths still
    /// consume only cooked blobs.
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
    /// atlas + animation clips. Instantiated in scenes by placing an
    /// [`NodeKind::Entity`] with a [`NodeKind::ModelRenderer`]
    /// component referencing this resource id.
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
    /// Gameplay character profile -- Model + role clip mapping +
    /// capsule/camera defaults. Layered on top of a Model
    /// resource; character-controller components reference this to
    /// resolve what to render and how movement/camera behaviour works.
    Character(CharacterResource),
    /// Equipment/weapon authoring resource. A Weapon references a
    /// Model for visuals and owns grip + hitbox data for combat.
    Weapon(WeaponResource),
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
            Self::Character(_) => "Character Profile",
            Self::Weapon(_) => "Weapon",
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

/// One backing-file move performed by a resource rename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceFileRename {
    /// Previous stored project path.
    pub from: String,
    /// New stored project path.
    pub to: String,
}

/// One backing file deleted with a resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceFileDelete {
    /// Stored project-relative path that was deleted.
    pub path: String,
}

/// Summary returned after renaming a resource and any backing files
/// that are safe for the project to own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceRenameReport {
    /// Files that were physically moved and whose project paths were
    /// updated.
    pub renamed_files: Vec<ResourceFileRename>,
    /// Path fields that were left alone because they were empty,
    /// missing on disk, outside the project root, or otherwise not
    /// safe to move automatically.
    pub skipped_files: Vec<String>,
}

/// Summary returned after removing a resource from the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDeleteReport {
    /// Resource removed from the project's resource table.
    pub removed: Resource,
    /// Number of project references cleared because they pointed at
    /// the removed resource.
    pub cleared_references: usize,
    /// Project-owned backing files physically removed from disk.
    pub deleted_files: Vec<ResourceFileDelete>,
    /// Path fields left alone because they were empty, missing,
    /// outside the project root, or otherwise not safe to delete.
    pub skipped_files: Vec<String>,
}

/// Failure modes for [`ProjectDocument::rename_resource_with_files`].
#[derive(Debug)]
pub enum ResourceRenameError {
    /// No resource with the requested id exists.
    MissingResource(ResourceId),
    /// Empty or whitespace-only names are refused.
    EmptyName,
    /// Two planned file moves would write the same destination.
    DuplicateTarget(PathBuf),
    /// A planned destination already exists.
    TargetExists(PathBuf),
    /// Filesystem operation failed.
    Io {
        /// Source path.
        from: PathBuf,
        /// Destination path.
        to: PathBuf,
        /// Error detail.
        detail: String,
    },
}

impl std::fmt::Display for ResourceRenameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingResource(id) => write!(f, "resource #{} does not exist", id.raw()),
            Self::EmptyName => write!(f, "resource name cannot be empty"),
            Self::DuplicateTarget(path) => {
                write!(f, "multiple files would rename to {}", path.display())
            }
            Self::TargetExists(path) => write!(f, "target already exists: {}", path.display()),
            Self::Io { from, to, detail } => write!(
                f,
                "failed to rename {} to {}: {detail}",
                from.display(),
                to.display()
            ),
        }
    }
}

impl std::error::Error for ResourceRenameError {}

/// Failure modes for [`ProjectDocument::delete_resource_with_files`].
#[derive(Debug)]
pub enum ResourceDeleteError {
    /// No resource with the requested id exists.
    MissingResource(ResourceId),
    /// Filesystem operation failed.
    Io {
        /// File path that could not be removed.
        path: PathBuf,
        /// Error detail.
        detail: String,
    },
}

impl std::fmt::Display for ResourceDeleteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingResource(id) => write!(f, "resource #{} does not exist", id.raw()),
            Self::Io { path, detail } => {
                write!(f, "failed to delete {}: {detail}", path.display())
            }
        }
    }
}

impl std::error::Error for ResourceDeleteError {}

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
    /// Composed world object. The node owns transform/identity;
    /// behaviour is expressed by component-node children such as
    /// [`ModelRenderer`](Self::ModelRenderer),
    /// [`Animator`](Self::Animator), and
    /// [`Collider`](Self::Collider).
    Entity,
    /// Macro-node grouping every Room that belongs to one streamed
    /// region. Owns the global sector size inherited by descendant
    /// Room grids.
    World {
        /// Shared sector size in engine units, snapped to
        /// [`WORLD_SECTOR_SIZE_QUANTUM`].
        #[serde(default = "default_world_sector_size")]
        sector_size: i32,
    },
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
    /// Render a cooked [`ResourceData::Model`] from the transform
    /// on the nearest entity ancestor. This is the component form of
    /// the legacy [`MeshInstance`](Self::MeshInstance) node.
    ModelRenderer {
        /// Model resource.
        model: Option<ResourceId>,
        /// Optional material override for legacy/static paths.
        /// Cooked PSX models currently carry their own atlas and
        /// ignore this field.
        #[serde(default)]
        material: Option<ResourceId>,
    },
    /// Animation component for a model-rendering entity. `clip`
    /// overrides the model default when set; `None` inherits the
    /// model's runtime default.
    Animator {
        /// Per-instance clip override.
        #[serde(default)]
        clip: Option<u16>,
        /// Whether this animation should run automatically in the
        /// editor/playtest runtime.
        #[serde(default = "default_true")]
        autoplay: bool,
    },
    /// Collision component. The first runtime pass only cooks room
    /// grid collision, but keeping authored collider data as a node
    /// makes entity/interactable/NPC architecture explicit now.
    Collider {
        /// Collision shape in engine/editor units.
        #[serde(default)]
        shape: ColliderShape,
        /// Solid colliders block movement; non-solid colliders are
        /// trigger volumes.
        #[serde(default = "default_true")]
        solid: bool,
    },
    /// Interactable component for props such as chests, doors, and
    /// levers. Runtime behaviour is not cooked yet; this is authoring
    /// structure for the upcoming object pass.
    Interactable {
        /// UI prompt or editor-facing affordance.
        #[serde(default)]
        prompt: String,
        /// Logical action id.
        #[serde(default)]
        action: String,
    },
    /// Character/controller component. It binds an entity to a reusable
    /// [`ResourceData::Character`] profile. When `player` is true this is
    /// the component-tree replacement for a legacy player
    /// [`SpawnPoint`](Self::SpawnPoint); non-player character cooking lands
    /// after NPC runtime records exist.
    CharacterController {
        /// Character profile resource.
        #[serde(default)]
        character: Option<ResourceId>,
        /// Whether this controller drives the player.
        #[serde(default)]
        player: bool,
    },
    /// AI marker component for future NPC/enemy runtime records.
    AiController {
        /// Logical AI profile id.
        #[serde(default)]
        behavior: String,
    },
    /// Combat stat component for entity nodes.
    Combat {
        /// Team/faction label.
        #[serde(default)]
        faction: String,
        /// Hit points.
        #[serde(default = "default_combat_health")]
        health: u16,
    },
    /// Equipment component. The parent Entity supplies the animated
    /// character model; this component names the Weapon and which
    /// socket/grip pair should be composed.
    Equipment {
        /// Weapon resource.
        #[serde(default)]
        weapon: Option<ResourceId>,
        /// Character/model socket to follow.
        #[serde(default = "default_character_socket")]
        character_socket: String,
        /// Weapon-local grip/pivot to align to the character socket.
        #[serde(default = "default_weapon_grip")]
        weapon_grip: String,
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
    /// Point-light component form of [`Light`](Self::Light).
    PointLight {
        /// RGB light colour.
        #[serde(default = "default_light_color")]
        color: [u8; 3],
        /// Light intensity multiplier.
        intensity: f32,
        /// Approximate editor/runtime radius in sectors.
        radius: f32,
    },
    /// Spawn marker.
    SpawnPoint {
        /// Whether this is the player spawn.
        player: bool,
        /// Character profile resource that drives this spawn. For the
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
            Self::Entity => "Entity",
            Self::World { .. } => "World",
            Self::Room { .. } => "Room",
            Self::MeshInstance { .. } => "Mesh Instance",
            Self::ModelRenderer { .. } => "Model Renderer",
            Self::Animator { .. } => "Animator",
            Self::Collider { .. } => "Collider",
            Self::Interactable { .. } => "Interactable",
            Self::CharacterController { .. } => "Character Controller",
            Self::AiController { .. } => "AI Controller",
            Self::Combat { .. } => "Combat",
            Self::Equipment { .. } => "Equipment",
            Self::Light { .. } => "Light",
            Self::PointLight { .. } => "Point Light",
            Self::SpawnPoint { .. } => "Spawn Point",
            Self::Trigger { .. } => "Trigger",
            Self::AudioSource { .. } => "Audio Source",
            Self::Portal { .. } => "Portal",
        }
    }

    /// True for behaviour/component nodes that are intended to be
    /// children of an [`Entity`](Self::Entity) host rather than
    /// independent placed objects.
    pub const fn is_component(self: &Self) -> bool {
        matches!(
            self,
            Self::ModelRenderer { .. }
                | Self::Animator { .. }
                | Self::Collider { .. }
                | Self::Interactable { .. }
                | Self::CharacterController { .. }
                | Self::AiController { .. }
                | Self::Combat { .. }
                | Self::Equipment { .. }
                | Self::PointLight { .. }
        )
    }
}

/// Authored collision shape for component-node entities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ColliderShape {
    /// Axis-aligned box, stored as half-extents.
    Box {
        /// Half extents in engine/editor units.
        half_extents: [u16; 3],
    },
    /// Sphere collider.
    Sphere {
        /// Radius in engine/editor units.
        radius: u16,
    },
    /// Upright capsule.
    Capsule {
        /// Radius in engine/editor units.
        radius: u16,
        /// Height in engine/editor units.
        height: u16,
    },
}

impl Default for ColliderShape {
    fn default() -> Self {
        Self::Box {
            half_extents: [256, 256, 256],
        }
    }
}

const fn default_true() -> bool {
    true
}

const fn default_combat_health() -> u16 {
    1
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

    /// Sector size inherited by `id` from the nearest World ancestor.
    pub fn world_sector_size_for_node(&self, id: NodeId) -> Option<i32> {
        let mut current = Some(id);
        while let Some(node_id) = current {
            let node = self.node(node_id)?;
            if let NodeKind::World { sector_size } = &node.kind {
                return Some(snap_world_sector_size(*sector_size));
            }
            current = node.parent;
        }
        None
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

    /// Count project references to `id` from scenes and from other
    /// resources. Backing-file paths are counted separately by the
    /// delete plan because they are owned by the resource itself.
    pub fn resource_reference_count(&self, id: ResourceId) -> usize {
        let mut count = 0;
        for resource in &self.resources {
            count += resource_data_reference_count(&resource.data, id);
        }
        for scene in &self.scenes {
            for node in scene.nodes() {
                count += node_kind_reference_count(&node.kind, id);
            }
        }
        count
    }

    /// Remove a resource from the project and clear references to it.
    pub fn delete_resource(&mut self, id: ResourceId) -> Option<ResourceDeleteReport> {
        let index = self
            .resources
            .iter()
            .position(|resource| resource.id == id)?;
        let removed = self.resources.remove(index);
        let cleared_references = self.clear_resource_references(id);
        Some(ResourceDeleteReport {
            removed,
            cleared_references,
            deleted_files: Vec::new(),
            skipped_files: Vec::new(),
        })
    }

    /// Remove a resource, delete its project-owned backing files, and
    /// clear references to it.
    ///
    /// Files are removed before project data is mutated. Only files
    /// that currently exist under `project_root` are deleted; missing
    /// or external paths are skipped and reported.
    pub fn delete_resource_with_files(
        &mut self,
        id: ResourceId,
        project_root: &Path,
    ) -> Result<ResourceDeleteReport, ResourceDeleteError> {
        let Some(index) = self.resources.iter().position(|resource| resource.id == id) else {
            return Err(ResourceDeleteError::MissingResource(id));
        };
        let plan = plan_resource_file_deletes(&self.resources[index], project_root);
        execute_resource_delete_plan(&plan, project_root)?;

        let mut report = self
            .delete_resource(id)
            .ok_or(ResourceDeleteError::MissingResource(id))?;
        report.deleted_files = plan
            .files
            .iter()
            .map(|op| ResourceFileDelete {
                path: op.stored.clone(),
            })
            .collect();
        report.skipped_files = plan.skipped;
        Ok(report)
    }

    /// Rename a resource and any project-owned backing files whose
    /// names are derived from the resource name.
    ///
    /// File moves are preflighted before project data is mutated:
    /// destinations must not already exist and duplicate destinations
    /// are refused. Only files that already exist under `project_root`
    /// are moved; missing paths and external absolute paths are
    /// preserved and reported as skipped.
    pub fn rename_resource_with_files(
        &mut self,
        id: ResourceId,
        new_name: &str,
        project_root: &Path,
    ) -> Result<ResourceRenameReport, ResourceRenameError> {
        let final_name = new_name.trim();
        if final_name.is_empty() {
            return Err(ResourceRenameError::EmptyName);
        }

        let Some(index) = self.resources.iter().position(|resource| resource.id == id) else {
            return Err(ResourceRenameError::MissingResource(id));
        };

        let resource = self.resources[index].clone();
        let safe_stem = resource_file_stem(final_name, resource_default_stem(&resource.data));
        let mut plan = ResourceRenamePlan::default();
        let mut data = resource.data.clone();

        match &mut data {
            ResourceData::Texture { psxt_path } => {
                plan_path_rename(psxt_path, &safe_stem, "psxt", project_root, &mut plan);
            }
            ResourceData::Model(model) => {
                plan_model_resource_rename(model, &safe_stem, project_root, &mut plan);
            }
            ResourceData::Mesh { source_path }
            | ResourceData::Scene { source_path }
            | ResourceData::Script { source_path }
            | ResourceData::Audio { source_path } => {
                let fallback_ext = resource_default_extension(&resource.data);
                plan_path_rename(
                    source_path,
                    &safe_stem,
                    fallback_ext,
                    project_root,
                    &mut plan,
                );
            }
            ResourceData::Material(_) | ResourceData::Character(_) | ResourceData::Weapon(_) => {}
        }

        execute_resource_rename_plan(&plan)?;

        self.resources[index].name = final_name.to_string();
        self.resources[index].data = data;

        Ok(ResourceRenameReport {
            renamed_files: plan
                .ops
                .iter()
                .map(|op| ResourceFileRename {
                    from: op.from_stored.clone(),
                    to: op.to_stored.clone(),
                })
                .collect(),
            skipped_files: plan.skipped,
        })
    }

    fn clear_resource_references(&mut self, id: ResourceId) -> usize {
        let mut count = 0;
        for resource in &mut self.resources {
            count += clear_resource_data_references(&mut resource.data, id);
        }
        for scene in &mut self.scenes {
            for node in &mut scene.nodes {
                count += clear_node_kind_references(&mut node.kind, id);
            }
        }
        count
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
        let mut project: Self = match ron::from_str(source) {
            Ok(project) => project,
            Err(first_error) => {
                let migrated = migrate_legacy_project_ron(source);
                if migrated == source {
                    return Err(ProjectIoError::Parse(first_error));
                }
                ron::from_str(&migrated).map_err(ProjectIoError::Parse)?
            }
        };
        project.normalize_loaded();
        Ok(project)
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
        let mut normalized = self.clone();
        normalized.normalize_loaded();
        std::fs::write(path, normalized.to_ron_string()?)?;
        Ok(())
    }

    /// Load a project from a RON file.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, ProjectIoError> {
        let source = std::fs::read_to_string(path)?;
        Self::from_ron_str(&source)
    }

    /// Normalize legacy or hand-authored project data after load.
    pub fn normalize_loaded(&mut self) {
        for scene in &mut self.scenes {
            for node in &mut scene.nodes {
                match &mut node.kind {
                    NodeKind::World { sector_size } => {
                        *sector_size = snap_world_sector_size(*sector_size);
                    }
                    _ => {}
                }
            }
            let worlds: Vec<(NodeId, i32)> = scene
                .nodes()
                .iter()
                .filter_map(|node| match &node.kind {
                    NodeKind::World { sector_size } => Some((node.id, *sector_size)),
                    _ => None,
                })
                .collect();
            for (world_id, sector_size) in worlds {
                apply_world_sector_size_to_descendants(
                    scene,
                    world_id,
                    sector_size,
                    sector_size,
                    false,
                );
            }
            let orphan_rooms: Vec<NodeId> = scene
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::Room { .. }))
                .filter(|node| scene.world_sector_size_for_node(node.id).is_none())
                .map(|node| node.id)
                .collect();
            for room_id in orphan_rooms {
                if let Some(node) = scene.node_mut(room_id) {
                    if let NodeKind::Room { grid } = &mut node.kind {
                        grid.rescale_sector_size(grid.sector_size);
                    }
                }
            }
        }
    }

    /// Sector size inherited by `node_id` from its nearest World
    /// ancestor, or the default when no World exists.
    pub fn world_sector_size_for_node(&self, node_id: NodeId) -> i32 {
        self.active_scene()
            .world_sector_size_for_node(node_id)
            .unwrap_or(DEFAULT_WORLD_SECTOR_SIZE)
    }

    /// Update a World node's sector size, snapping to 128-unit
    /// increments and rescaling descendant rooms/components.
    pub fn set_world_sector_size(&mut self, world_id: NodeId, requested: i32) -> Option<i32> {
        let scene = self.active_scene_mut();
        let new_size = snap_world_sector_size(requested);
        let old_size = {
            let world = scene.node_mut(world_id)?;
            let NodeKind::World { sector_size } = &mut world.kind else {
                return None;
            };
            let old_size = snap_world_sector_size(*sector_size);
            *sector_size = new_size;
            old_size
        };
        apply_world_sector_size_to_descendants(
            scene,
            world_id,
            new_size,
            old_size,
            old_size != new_size,
        );
        Some(new_size)
    }
}

fn resource_data_reference_count(data: &ResourceData, id: ResourceId) -> usize {
    match data {
        ResourceData::Material(material) => option_resource_reference_count(material.texture, id),
        ResourceData::Character(character) => option_resource_reference_count(character.model, id),
        ResourceData::Weapon(weapon) => option_resource_reference_count(weapon.model, id),
        ResourceData::Texture { .. }
        | ResourceData::Model(_)
        | ResourceData::Mesh { .. }
        | ResourceData::Scene { .. }
        | ResourceData::Script { .. }
        | ResourceData::Audio { .. } => 0,
    }
}

fn clear_resource_data_references(data: &mut ResourceData, id: ResourceId) -> usize {
    match data {
        ResourceData::Material(material) => clear_option_resource(&mut material.texture, id),
        ResourceData::Character(character) => {
            let cleared = clear_option_resource(&mut character.model, id);
            if cleared > 0 {
                character.idle_clip = None;
                character.walk_clip = None;
                character.run_clip = None;
                character.turn_clip = None;
            }
            cleared
        }
        ResourceData::Weapon(weapon) => clear_option_resource(&mut weapon.model, id),
        ResourceData::Texture { .. }
        | ResourceData::Model(_)
        | ResourceData::Mesh { .. }
        | ResourceData::Scene { .. }
        | ResourceData::Script { .. }
        | ResourceData::Audio { .. } => 0,
    }
}

fn node_kind_reference_count(kind: &NodeKind, id: ResourceId) -> usize {
    match kind {
        NodeKind::Room { grid } => grid_resource_reference_count(grid, id),
        NodeKind::MeshInstance { mesh, material, .. } => {
            option_resource_reference_count(*mesh, id)
                + option_resource_reference_count(*material, id)
        }
        NodeKind::ModelRenderer { model, material } => {
            option_resource_reference_count(*model, id)
                + option_resource_reference_count(*material, id)
        }
        NodeKind::CharacterController { character, .. } => {
            option_resource_reference_count(*character, id)
        }
        NodeKind::Equipment { weapon, .. } => option_resource_reference_count(*weapon, id),
        NodeKind::SpawnPoint { character, .. } => option_resource_reference_count(*character, id),
        NodeKind::AudioSource { sound, .. } => option_resource_reference_count(*sound, id),
        NodeKind::Node
        | NodeKind::Node3D
        | NodeKind::Entity
        | NodeKind::World { .. }
        | NodeKind::Animator { .. }
        | NodeKind::Collider { .. }
        | NodeKind::Interactable { .. }
        | NodeKind::AiController { .. }
        | NodeKind::Combat { .. }
        | NodeKind::Light { .. }
        | NodeKind::PointLight { .. }
        | NodeKind::Trigger { .. }
        | NodeKind::Portal { .. } => 0,
    }
}

fn clear_node_kind_references(kind: &mut NodeKind, id: ResourceId) -> usize {
    match kind {
        NodeKind::Room { grid } => clear_grid_resource_references(grid, id),
        NodeKind::MeshInstance { mesh, material, .. } => {
            clear_option_resource(mesh, id) + clear_option_resource(material, id)
        }
        NodeKind::ModelRenderer { model, material } => {
            clear_option_resource(model, id) + clear_option_resource(material, id)
        }
        NodeKind::CharacterController { character, .. } => clear_option_resource(character, id),
        NodeKind::Equipment { weapon, .. } => clear_option_resource(weapon, id),
        NodeKind::SpawnPoint { character, .. } => clear_option_resource(character, id),
        NodeKind::AudioSource { sound, .. } => clear_option_resource(sound, id),
        NodeKind::Node
        | NodeKind::Node3D
        | NodeKind::Entity
        | NodeKind::World { .. }
        | NodeKind::Animator { .. }
        | NodeKind::Collider { .. }
        | NodeKind::Interactable { .. }
        | NodeKind::AiController { .. }
        | NodeKind::Combat { .. }
        | NodeKind::Light { .. }
        | NodeKind::PointLight { .. }
        | NodeKind::Trigger { .. }
        | NodeKind::Portal { .. } => 0,
    }
}

fn grid_resource_reference_count(grid: &WorldGrid, id: ResourceId) -> usize {
    let mut count = 0;
    for sector in grid.sectors.iter().flatten() {
        if let Some(face) = &sector.floor {
            count += option_resource_reference_count(face.material, id);
        }
        if let Some(face) = &sector.ceiling {
            count += option_resource_reference_count(face.material, id);
        }
        for direction in GridDirection::ALL {
            for wall in sector.walls.get(direction) {
                count += option_resource_reference_count(wall.material, id);
            }
        }
    }
    count
}

fn clear_grid_resource_references(grid: &mut WorldGrid, id: ResourceId) -> usize {
    let mut count = 0;
    for sector in grid.sectors.iter_mut().flatten() {
        if let Some(face) = &mut sector.floor {
            count += clear_option_resource(&mut face.material, id);
        }
        if let Some(face) = &mut sector.ceiling {
            count += clear_option_resource(&mut face.material, id);
        }
        for direction in GridDirection::ALL {
            for wall in sector.walls.get_mut(direction) {
                count += clear_option_resource(&mut wall.material, id);
            }
        }
    }
    count
}

fn option_resource_reference_count(value: Option<ResourceId>, id: ResourceId) -> usize {
    usize::from(value == Some(id))
}

fn clear_option_resource(value: &mut Option<ResourceId>, id: ResourceId) -> usize {
    if *value == Some(id) {
        *value = None;
        1
    } else {
        0
    }
}

fn migrate_legacy_project_ron(source: &str) -> String {
    source
        .replace(
            "kind: World,",
            &format!("kind: World(sector_size: {}),", DEFAULT_WORLD_SECTOR_SIZE),
        )
        .replace("kind: Actor,", "kind: Entity,")
}

fn apply_world_sector_size_to_descendants(
    scene: &mut Scene,
    world_id: NodeId,
    sector_size: i32,
    old_sector_size: i32,
    rescale: bool,
) {
    let ids: Vec<NodeId> = scene
        .nodes()
        .iter()
        .filter(|node| scene.is_descendant_of(node.id, world_id))
        .map(|node| node.id)
        .collect();
    for id in ids {
        let Some(node) = scene.node_mut(id) else {
            continue;
        };
        match &mut node.kind {
            NodeKind::Room { grid } => {
                if rescale {
                    grid.rescale_sector_size(sector_size);
                } else {
                    grid.sector_size = snap_world_sector_size(sector_size);
                    grid.snap_heights_to_quantum();
                }
            }
            NodeKind::Collider { shape, .. } if rescale => {
                rescale_collider_shape(shape, old_sector_size, sector_size);
            }
            _ => {}
        }
    }
}

fn rescale_collider_shape(shape: &mut ColliderShape, from: i32, to: i32) {
    match shape {
        ColliderShape::Box { half_extents } => {
            for axis in half_extents {
                *axis = scale_u16_ratio(*axis, from, to);
            }
        }
        ColliderShape::Sphere { radius } => {
            *radius = scale_u16_ratio(*radius, from, to);
        }
        ColliderShape::Capsule { radius, height } => {
            *radius = scale_u16_ratio(*radius, from, to);
            *height = scale_u16_ratio(*height, from, to);
        }
    }
}

#[derive(Default)]
struct ResourceRenamePlan {
    ops: Vec<ResourcePathRename>,
    skipped: Vec<String>,
}

#[derive(Default)]
struct ResourceDeletePlan {
    files: Vec<ResourcePathDelete>,
    skipped: Vec<String>,
}

struct ResourcePathRename {
    from_abs: PathBuf,
    to_abs: PathBuf,
    from_stored: String,
    to_stored: String,
}

struct ResourcePathDelete {
    abs: PathBuf,
    stored: String,
}

fn plan_resource_file_deletes(resource: &Resource, project_root: &Path) -> ResourceDeletePlan {
    let mut plan = ResourceDeletePlan::default();
    match &resource.data {
        ResourceData::Texture { psxt_path } => {
            plan_path_delete(psxt_path, project_root, &mut plan);
        }
        ResourceData::Model(model) => {
            plan_path_delete(&model.model_path, project_root, &mut plan);
            if let Some(texture_path) = &model.texture_path {
                plan_path_delete(texture_path, project_root, &mut plan);
            }
            for clip in &model.clips {
                plan_path_delete(&clip.psxanim_path, project_root, &mut plan);
            }
        }
        ResourceData::Mesh { source_path }
        | ResourceData::Scene { source_path }
        | ResourceData::Script { source_path }
        | ResourceData::Audio { source_path } => {
            plan_path_delete(source_path, project_root, &mut plan);
        }
        ResourceData::Material(_) | ResourceData::Character(_) | ResourceData::Weapon(_) => {}
    }
    plan
}

fn plan_path_delete(stored: &str, project_root: &Path, plan: &mut ResourceDeletePlan) {
    let trimmed = stored.trim();
    if trimmed.is_empty() {
        return;
    }

    let abs = model_import::resolve_path(trimmed, Some(project_root));
    if !abs.is_file() {
        plan.skipped.push(trimmed.to_string());
        return;
    }
    if !path_is_project_owned(&abs, project_root) {
        plan.skipped.push(trimmed.to_string());
        return;
    }
    if plan.files.iter().any(|op| op.abs == abs) {
        return;
    }
    plan.files.push(ResourcePathDelete {
        stored: relativise_resource_path(&abs, project_root),
        abs,
    });
}

fn execute_resource_delete_plan(
    plan: &ResourceDeletePlan,
    project_root: &Path,
) -> Result<(), ResourceDeleteError> {
    for op in &plan.files {
        std::fs::remove_file(&op.abs).map_err(|error| ResourceDeleteError::Io {
            path: op.abs.clone(),
            detail: error.to_string(),
        })?;
    }
    for op in &plan.files {
        remove_empty_project_parents(op.abs.parent(), project_root);
    }
    Ok(())
}

fn remove_empty_project_parents(mut dir: Option<&Path>, project_root: &Path) {
    while let Some(current) = dir {
        if current == project_root {
            break;
        }
        if std::fs::remove_dir(current).is_err() {
            break;
        }
        dir = current.parent();
    }
}

fn plan_path_rename(
    stored: &mut String,
    safe_stem: &str,
    fallback_ext: &str,
    project_root: &Path,
    plan: &mut ResourceRenamePlan,
) {
    let original = stored.clone();
    let Some(op) = build_path_rename(&original, safe_stem, fallback_ext, project_root, plan) else {
        return;
    };
    *stored = op.to_stored.clone();
    plan.ops.push(op);
}

fn plan_model_resource_rename(
    model: &mut ModelResource,
    safe_stem: &str,
    project_root: &Path,
    plan: &mut ResourceRenamePlan,
) {
    let model_path = model.model_path.clone();
    let model_abs = model_import::resolve_path(&model_path, Some(project_root));
    let model_dir = model_abs.parent().map(Path::to_path_buf);
    let target_dir = model_dir
        .as_deref()
        .map(|dir| model_bundle_target_dir(dir, safe_stem, project_root));

    if let Some(op) = build_path_rename_in_dir(
        &model_path,
        safe_stem,
        "psxmdl",
        target_dir.as_deref(),
        project_root,
        plan,
    ) {
        model.model_path = op.to_stored.clone();
        plan.ops.push(op);
    }

    if let Some(texture_path) = &mut model.texture_path {
        let original = texture_path.clone();
        if let Some(op) = build_path_rename_in_dir(
            &original,
            safe_stem,
            "psxt",
            target_dir.as_deref(),
            project_root,
            plan,
        ) {
            *texture_path = op.to_stored.clone();
            plan.ops.push(op);
        }
    }

    let mut seen_clip_stems = HashSet::new();
    for (index, clip) in model.clips.iter_mut().enumerate() {
        let clip_suffix = resource_file_stem(&clip.name, "clip");
        let mut clip_stem = format!("{safe_stem}_{clip_suffix}");
        if !seen_clip_stems.insert(clip_stem.clone()) {
            clip_stem = format!("{safe_stem}_{index}_{clip_suffix}");
            seen_clip_stems.insert(clip_stem.clone());
        }
        let original = clip.psxanim_path.clone();
        if let Some(op) = build_path_rename_in_dir(
            &original,
            &clip_stem,
            "psxanim",
            target_dir.as_deref(),
            project_root,
            plan,
        ) {
            clip.psxanim_path = op.to_stored.clone();
            plan.ops.push(op);
        }
    }
}

fn build_path_rename(
    stored: &str,
    safe_stem: &str,
    fallback_ext: &str,
    project_root: &Path,
    plan: &mut ResourceRenamePlan,
) -> Option<ResourcePathRename> {
    build_path_rename_in_dir(stored, safe_stem, fallback_ext, None, project_root, plan)
}

fn build_path_rename_in_dir(
    stored: &str,
    safe_stem: &str,
    fallback_ext: &str,
    target_dir: Option<&Path>,
    project_root: &Path,
    plan: &mut ResourceRenamePlan,
) -> Option<ResourcePathRename> {
    let trimmed = stored.trim();
    if trimmed.is_empty() {
        return None;
    }

    let from_abs = model_import::resolve_path(trimmed, Some(project_root));
    if !from_abs.is_file() {
        plan.skipped.push(trimmed.to_string());
        return None;
    }
    if !path_is_project_owned(&from_abs, project_root) {
        plan.skipped.push(trimmed.to_string());
        return None;
    }

    let ext = from_abs
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| !ext.is_empty())
        .unwrap_or(fallback_ext);
    let target_name = format!("{safe_stem}.{ext}");
    let to_abs = target_dir
        .map(|dir| dir.join(&target_name))
        .unwrap_or_else(|| from_abs.with_file_name(target_name));

    if from_abs == to_abs {
        return None;
    }

    Some(ResourcePathRename {
        from_abs,
        to_stored: relativise_resource_path(&to_abs, project_root),
        to_abs,
        from_stored: trimmed.to_string(),
    })
}

fn execute_resource_rename_plan(plan: &ResourceRenamePlan) -> Result<(), ResourceRenameError> {
    let mut targets = HashSet::new();
    for op in &plan.ops {
        if !targets.insert(op.to_abs.clone()) {
            return Err(ResourceRenameError::DuplicateTarget(op.to_abs.clone()));
        }
        if op.to_abs.exists() {
            return Err(ResourceRenameError::TargetExists(op.to_abs.clone()));
        }
    }

    let mut moved: Vec<&ResourcePathRename> = Vec::new();
    let mut created_dirs = Vec::new();
    for op in &plan.ops {
        if let Some(parent) = op.to_abs.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|error| ResourceRenameError::Io {
                    from: op.from_abs.clone(),
                    to: op.to_abs.clone(),
                    detail: error.to_string(),
                })?;
                created_dirs.push(parent.to_path_buf());
            }
        }
        if let Err(error) = std::fs::rename(&op.from_abs, &op.to_abs) {
            for done in moved.iter().rev() {
                let _ = std::fs::rename(&done.to_abs, &done.from_abs);
            }
            for dir in created_dirs.iter().rev() {
                let _ = std::fs::remove_dir(dir);
            }
            return Err(ResourceRenameError::Io {
                from: op.from_abs.clone(),
                to: op.to_abs.clone(),
                detail: error.to_string(),
            });
        }
        moved.push(op);
    }

    for op in &plan.ops {
        if let (Some(from_parent), Some(to_parent)) = (op.from_abs.parent(), op.to_abs.parent()) {
            if from_parent != to_parent {
                let _ = std::fs::remove_dir(from_parent);
            }
        }
    }

    Ok(())
}

fn model_bundle_target_dir(model_dir: &Path, safe_stem: &str, project_root: &Path) -> PathBuf {
    let Ok(relative) = model_dir.strip_prefix(project_root) else {
        return model_dir.to_path_buf();
    };
    let mut components = relative.components();
    let is_imported_bundle = matches!(
        (
            components.next().and_then(|c| c.as_os_str().to_str()),
            components.next().and_then(|c| c.as_os_str().to_str()),
            components.next(),
            components.next()
        ),
        (Some("assets"), Some("models"), Some(_), None)
    );
    if is_imported_bundle {
        project_root.join("assets").join("models").join(safe_stem)
    } else {
        model_dir.to_path_buf()
    }
}

fn path_is_project_owned(path: &Path, project_root: &Path) -> bool {
    path.strip_prefix(project_root).is_ok()
}

fn relativise_resource_path(path: &Path, project_root: &Path) -> String {
    path.strip_prefix(project_root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn resource_file_stem(name: &str, fallback: &str) -> String {
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
        fallback.to_string()
    } else {
        trimmed
    }
}

const fn resource_default_stem(data: &ResourceData) -> &'static str {
    match data {
        ResourceData::Texture { .. } => "texture",
        ResourceData::Material(_) => "material",
        ResourceData::Model(_) => "model",
        ResourceData::Weapon(_) => "weapon",
        ResourceData::Mesh { .. } => "mesh",
        ResourceData::Scene { .. } => "scene",
        ResourceData::Script { .. } => "script",
        ResourceData::Audio { .. } => "audio",
        ResourceData::Character(_) => "character",
    }
}

const fn resource_default_extension(data: &ResourceData) -> &'static str {
    match data {
        ResourceData::Texture { .. } => "psxt",
        ResourceData::Material(_) => "mat",
        ResourceData::Model(_) => "psxmdl",
        ResourceData::Weapon(_) => "weapon",
        ResourceData::Mesh { .. } => "psxmesh",
        ResourceData::Scene { .. } => "room",
        ResourceData::Script { .. } => "script",
        ResourceData::Audio { .. } => "vag",
        ResourceData::Character(_) => "char",
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
    fn normalize_loaded_snaps_room_heights_to_quantum() {
        let mut project = ProjectDocument::starter();
        let room_id = project
            .active_scene()
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))
            .map(|node| node.id)
            .unwrap();
        {
            let room = project.active_scene_mut().node_mut(room_id).unwrap();
            let NodeKind::Room { grid } = &mut room.kind else {
                panic!("expected room");
            };
            let sector = grid.ensure_sector(0, 0).unwrap();
            sector.floor = Some(GridHorizontalFace::flat(17, None));
            let walls = sector.walls.get_mut(GridDirection::West);
            walls.clear();
            walls.push(GridVerticalFace::with_heights([0, 0, 965, 802], None));
        }

        project.normalize_loaded();

        let room = project.active_scene().node(room_id).unwrap();
        let NodeKind::Room { grid } = &room.kind else {
            panic!("expected room");
        };
        let sector = grid.sector(0, 0).unwrap();
        assert_eq!(sector.floor.as_ref().unwrap().heights, [32, 32, 32, 32]);
        assert_eq!(
            sector.walls.get(GridDirection::West)[0].heights,
            [0, 0, 960, 800]
        );
    }

    #[test]
    fn snap_world_sector_size_quantizes_to_128_units() {
        assert_eq!(WORLD_SECTOR_SIZE_QUANTUM, 128);
        assert_eq!(snap_world_sector_size(1), 128);
        assert_eq!(snap_world_sector_size(127), 128);
        assert_eq!(snap_world_sector_size(191), 128);
        assert_eq!(snap_world_sector_size(192), 256);
        assert_eq!(snap_world_sector_size(1500), 1536);
        assert_eq!(
            snap_world_sector_size(MAX_WORLD_SECTOR_SIZE + 1),
            MAX_WORLD_SECTOR_SIZE
        );
    }

    #[test]
    fn changing_world_sector_size_rescales_descendant_room_and_colliders() {
        let mut project = ProjectDocument::new("test");
        let scene = project.active_scene_mut();
        let world = scene.add_node(scene.root, "World", NodeKind::World { sector_size: 1024 });
        let mut grid = WorldGrid::empty(1, 1, 1024);
        grid.set_floor(0, 0, 160, None);
        grid.add_wall(0, 0, GridDirection::North, 0, 1024, None);
        let room = scene.add_node(world, "Room", NodeKind::Room { grid });
        let entity = scene.add_node(room, "Entity", NodeKind::Entity);
        let collider = scene.add_node(
            entity,
            "Collider",
            NodeKind::Collider {
                shape: ColliderShape::Capsule {
                    radius: 128,
                    height: 1024,
                },
                solid: true,
            },
        );

        assert_eq!(project.set_world_sector_size(world, 1500), Some(1536));
        assert_eq!(project.world_sector_size_for_node(entity), 1536);

        let scene = project.active_scene();
        let NodeKind::Room { grid } = &scene.node(room).unwrap().kind else {
            panic!("expected Room");
        };
        assert_eq!(grid.sector_size, 1536);
        let sector = grid.sector(0, 0).unwrap();
        assert_eq!(sector.floor.as_ref().unwrap().heights, [256; 4]);
        assert_eq!(
            sector
                .walls
                .get(GridDirection::North)
                .first()
                .unwrap()
                .heights,
            [0, 0, 1536, 1536]
        );

        let NodeKind::Collider {
            shape: ColliderShape::Capsule { radius, height },
            ..
        } = &scene.node(collider).unwrap().kind
        else {
            panic!("expected capsule collider");
        };
        assert_eq!((*radius, *height), (192, 1536));
    }

    #[test]
    fn saving_normalizes_room_sector_size_to_world() {
        let mut project = ProjectDocument::starter();
        let scene = project.active_scene_mut();
        let room_id = scene
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))
            .map(|node| node.id)
            .unwrap();
        let NodeKind::Room { grid } = &mut scene.node_mut(room_id).unwrap().kind else {
            panic!("expected Room");
        };
        grid.sector_size = 2030;

        let dir = unique_temp_dir("normalize-room-sector-size");
        let path = dir.join("project.ron");
        project.save_to_path(&path).unwrap();

        let saved = std::fs::read_to_string(&path).unwrap();
        let expected_sector_size = project.world_sector_size_for_node(room_id);
        assert!(saved.contains(&format!(
            "kind: World(sector_size: {expected_sector_size}),"
        )));
        assert!(saved.contains(&format!("sector_size: {expected_sector_size}")));
        assert!(!saved.contains("sector_size: 2030"));

        let loaded = ProjectDocument::load_from_path(&path).unwrap();
        let scene = loaded.active_scene();
        let NodeKind::Room { grid } = &scene.node(room_id).unwrap().kind else {
            panic!("expected Room");
        };
        assert_eq!(grid.sector_size, expected_sector_size);

        let _ = std::fs::remove_dir_all(dir);
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
    fn wall_placement_aligns_bottom_edge_to_floor_vertices() {
        let mut grid = WorldGrid::empty(1, 1, 1024);
        let mut floor = GridHorizontalFace::flat(0, None);
        floor.heights = [128, 256, 384, 512];
        grid.ensure_sector(0, 0).unwrap().floor = Some(floor);

        grid.add_wall_aligned_to_surfaces(0, 0, GridDirection::North, None);

        let wall = grid
            .sector(0, 0)
            .unwrap()
            .walls
            .get(GridDirection::North)
            .first()
            .unwrap();
        assert_eq!(wall.heights, [128, 256, 2304, 2176]);
    }

    #[test]
    fn wall_placement_aligns_top_edge_to_ceiling_vertices() {
        let mut grid = WorldGrid::empty(1, 1, 1024);
        let mut floor = GridHorizontalFace::flat(0, None);
        floor.heights = [128, 256, 384, 512];
        let mut ceiling = GridHorizontalFace::flat(1024, None);
        ceiling.heights = [900, 1000, 1100, 1200];
        let sector = grid.ensure_sector(0, 0).unwrap();
        sector.floor = Some(floor);
        sector.ceiling = Some(ceiling);

        grid.add_wall_aligned_to_surfaces(0, 0, GridDirection::East, None);

        let wall = grid
            .sector(0, 0)
            .unwrap()
            .walls
            .get(GridDirection::East)
            .first()
            .unwrap();
        assert_eq!(wall.heights, [256, 384, 1100, 1000]);
    }

    #[test]
    fn off_grid_wall_preview_samples_adjacent_floor_edge() {
        let mut grid = WorldGrid::empty(1, 1, 1024);
        let mut floor = GridHorizontalFace::flat(0, None);
        floor.heights = [128, 256, 384, 512];
        grid.ensure_sector(0, 0).unwrap().floor = Some(floor);

        let heights =
            grid.wall_heights_aligned_to_surfaces_for_world_cell(1, 0, GridDirection::West);

        assert_eq!(heights, [384, 256, 2304, 2432]);
    }

    #[test]
    fn stone_room_perimeter_uses_editor_direction_convention() {
        let grid = WorldGrid::stone_room(2, 3, 1024, None, None);
        let default_wall_height = default_wall_height_for_sector_size(1024);

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
        let south_wall = grid
            .sector(0, 0)
            .unwrap()
            .walls
            .get(GridDirection::South)
            .first()
            .unwrap();
        assert_eq!(
            south_wall.heights,
            [0, 0, default_wall_height, default_wall_height]
        );
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
    fn authored_footprint_ignores_empty_allocation() {
        let mut grid = WorldGrid::empty(8, 6, 1024);
        let _ = grid.ensure_sector(0, 0);
        grid.set_floor(2, 1, 0, None);
        grid.add_wall(5, 4, GridDirection::North, 0, 1024, None);

        let footprint = grid.authored_footprint().expect("authored geometry");
        assert_eq!(
            footprint,
            WorldGridFootprint {
                x: 2,
                z: 1,
                width: 4,
                depth: 4,
            }
        );
        assert_eq!(grid.populated_sector_count(), 2);

        let budget = grid.authored_budget();
        assert_eq!(budget.width, 4);
        assert_eq!(budget.depth, 4);
        assert_eq!(budget.total_cells, 16);
        assert_eq!(budget.populated_cells, 2);
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
        // `.psxw` stores a record per cell whether populated or not.
        assert_eq!(
            b.psxw_bytes,
            12 + 20 + 9 * psxed_format::world::SectorRecord::SIZE
        );
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
        // The future compact estimate should be strictly smaller
        // than the active format once any geometry exists.
        assert!(b.future_compact_estimated_bytes < b.psxw_bytes);
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
        // Active format remains under the byte cap for floors-only;
        // the wall-stack-heavy worst case is what pushes rooms over.
        assert!(b.psxw_bytes <= MAX_ROOM_BYTES);
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
        assert!(project.resources.iter().any(|r| matches!(
            &r.data,
            ResourceData::Texture { psxt_path } if psxt_path.ends_with("dirt.psxt")
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
        assert_eq!(wraith.scale_q8, [MODEL_SCALE_ONE_Q8; 3]);
    }

    #[test]
    fn legacy_world_and_actor_project_ron_migrates_to_world_sector_and_entity() {
        let starter = ProjectDocument::from_ron_str(DEFAULT_PROJECT_RON).unwrap();
        let starter_world_sector_size = starter
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::World { sector_size } => Some(*sector_size),
                _ => None,
            })
            .expect("starter world exists");
        let legacy = DEFAULT_PROJECT_RON
            .replace(
                &format!("kind: World(sector_size: {starter_world_sector_size}),"),
                "kind: World,",
            )
            .replacen("kind: Entity,", "kind: Actor,", 1);

        let project = ProjectDocument::from_ron_str(&legacy).unwrap();
        let scene = project.active_scene();
        let world = scene
            .nodes()
            .iter()
            .find(|node| node.name == "Demo World")
            .expect("starter world exists");
        assert!(matches!(
            &world.kind,
            NodeKind::World { sector_size } if *sector_size == DEFAULT_WORLD_SECTOR_SIZE
        ));
        let migrated = scene
            .nodes()
            .iter()
            .find(|node| node.name == "Wraith Hero")
            .expect("starter player entity exists");
        assert!(matches!(&migrated.kind, NodeKind::Entity));
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
        assert!(default_project_dir()
            .join("assets/textures/dirt.psxt")
            .is_file());
    }

    #[test]
    fn starter_project_has_scene_tree_and_resources() {
        let project = ProjectDocument::starter();

        assert_eq!(project.scenes.len(), 1);
        // Starter includes room textures/materials plus gameplay
        // resources for the animated character and weapon path.
        assert!(project.resources.len() >= 10);
        assert!(project
            .active_scene()
            .hierarchy_rows()
            .iter()
            .any(|row| row.name == "Room"));
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
        let starter = ProjectDocument::from_ron_str(DEFAULT_PROJECT_RON).unwrap();
        let light = starter
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Light {
                    color,
                    intensity,
                    radius,
                }
                | NodeKind::PointLight {
                    color,
                    intensity,
                    radius,
                } => Some((*color, *intensity, *radius)),
                _ => None,
            })
            .expect("starter has a light");
        let room = starter
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Room { grid } => Some(grid),
                _ => None,
            })
            .expect("starter has a room");
        let source = DEFAULT_PROJECT_RON
            .replace(
                &format!(
                    "kind: Light(color: ({}, {}, {}), intensity: {}, radius: {})",
                    light.0[0], light.0[1], light.0[2], light.1, light.2
                ),
                &format!("kind: Light(intensity: {}, radius: {})", light.1, light.2),
            )
            .replace(
                &format!(
                    ", ambient_color: ({}, {}, {})",
                    room.ambient_color[0], room.ambient_color[1], room.ambient_color[2]
                ),
                "",
            )
            .replace(
                &format!(", fog_near: {}, fog_far: {}", room.fog_near, room.fog_far),
                "",
            );

        let project = ProjectDocument::from_ron_str(&source).unwrap();
        let light_color = project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Light { color, .. } => Some(*color),
                NodeKind::PointLight { color, .. } => Some(*color),
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

        let fog = project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Room { grid } => Some((grid.fog_color, grid.fog_near, grid.fog_far)),
                _ => None,
            })
            .expect("starter has a room");
        assert_eq!(
            fog,
            (default_fog_color(), default_fog_near(), default_fog_far())
        );
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

        assert!(ron.contains("Room"));
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
                scale_q8: [
                    MODEL_SCALE_ONE_Q8,
                    MODEL_SCALE_ONE_Q8 * 2,
                    MODEL_SCALE_ONE_Q8,
                ],
                attachments: vec![AttachmentSocket {
                    name: "right_hand_grip".to_string(),
                    joint: 3,
                    translation: [16, 32, -8],
                    rotation_q12: [0, 1024, 0],
                }],
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
                assert_eq!(
                    m.scale_q8,
                    [
                        MODEL_SCALE_ONE_Q8,
                        MODEL_SCALE_ONE_Q8 * 2,
                        MODEL_SCALE_ONE_Q8
                    ]
                );
                assert_eq!(m.effective_preview_clip(), Some(1));
                assert_eq!(m.effective_runtime_clip(), Some(0));
                assert_eq!(m.attachments.len(), 1);
                assert_eq!(m.attachments[0].joint, 3);
                assert_eq!(m.attachments[0].translation, [16, 32, -8]);
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

    fn unique_temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "psxed-project-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn resource_rename_moves_project_owned_texture_file() {
        let root = unique_temp_dir("resource-rename-texture");
        let texture_dir = root.join("assets").join("textures");
        std::fs::create_dir_all(&texture_dir).unwrap();
        std::fs::write(texture_dir.join("floor.psxt"), b"texture").unwrap();

        let mut project = ProjectDocument::new("test");
        let id = project.add_resource(
            "Floor",
            ResourceData::Texture {
                psxt_path: "assets/textures/floor.psxt".to_string(),
            },
        );

        let report = project
            .rename_resource_with_files(id, "Stone Floor", &root)
            .unwrap();

        assert_eq!(project.resource_name(id), Some("Stone Floor"));
        let ResourceData::Texture { psxt_path } = &project.resource(id).unwrap().data else {
            panic!("expected texture");
        };
        assert_eq!(psxt_path, "assets/textures/stone_floor.psxt");
        assert!(!texture_dir.join("floor.psxt").exists());
        assert_eq!(
            std::fs::read(texture_dir.join("stone_floor.psxt")).unwrap(),
            b"texture"
        );
        assert_eq!(report.renamed_files.len(), 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resource_rename_moves_imported_model_bundle_files() {
        let root = unique_temp_dir("resource-rename-model");
        let bundle_dir = root.join("assets").join("models").join("obsidian_wraith");
        std::fs::create_dir_all(&bundle_dir).unwrap();
        std::fs::write(bundle_dir.join("obsidian_wraith.psxmdl"), b"model").unwrap();
        std::fs::write(bundle_dir.join("obsidian_wraith.psxt"), b"atlas").unwrap();
        std::fs::write(bundle_dir.join("obsidian_wraith_idle.psxanim"), b"idle").unwrap();
        std::fs::write(bundle_dir.join("obsidian_wraith_walk.psxanim"), b"walk").unwrap();

        let mut project = ProjectDocument::new("test");
        let id = project.add_resource(
            "Obsidian Wraith",
            ResourceData::Model(ModelResource {
                model_path: "assets/models/obsidian_wraith/obsidian_wraith.psxmdl".to_string(),
                texture_path: Some(
                    "assets/models/obsidian_wraith/obsidian_wraith.psxt".to_string(),
                ),
                clips: vec![
                    ModelAnimationClip {
                        name: "idle".to_string(),
                        psxanim_path: "assets/models/obsidian_wraith/obsidian_wraith_idle.psxanim"
                            .to_string(),
                    },
                    ModelAnimationClip {
                        name: "walk".to_string(),
                        psxanim_path: "assets/models/obsidian_wraith/obsidian_wraith_walk.psxanim"
                            .to_string(),
                    },
                ],
                default_clip: Some(0),
                preview_clip: Some(0),
                world_height: 1024,
                scale_q8: [MODEL_SCALE_ONE_Q8; 3],
                attachments: Vec::new(),
            }),
        );

        let report = project
            .rename_resource_with_files(id, "Hooded Wretch", &root)
            .unwrap();

        let ResourceData::Model(model) = &project.resource(id).unwrap().data else {
            panic!("expected model");
        };
        assert_eq!(
            model.model_path,
            "assets/models/hooded_wretch/hooded_wretch.psxmdl"
        );
        assert_eq!(
            model.texture_path.as_deref(),
            Some("assets/models/hooded_wretch/hooded_wretch.psxt")
        );
        assert_eq!(
            model.clips[0].psxanim_path,
            "assets/models/hooded_wretch/hooded_wretch_idle.psxanim"
        );
        assert_eq!(
            model.clips[1].psxanim_path,
            "assets/models/hooded_wretch/hooded_wretch_walk.psxanim"
        );
        assert_eq!(report.renamed_files.len(), 4);
        assert!(!bundle_dir.exists());
        assert!(root
            .join("assets/models/hooded_wretch/hooded_wretch.psxmdl")
            .exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resource_rename_refuses_existing_target_without_mutating_project() {
        let root = unique_temp_dir("resource-rename-target-exists");
        let texture_dir = root.join("assets").join("textures");
        std::fs::create_dir_all(&texture_dir).unwrap();
        std::fs::write(texture_dir.join("floor.psxt"), b"old").unwrap();
        std::fs::write(texture_dir.join("stone_floor.psxt"), b"target").unwrap();

        let mut project = ProjectDocument::new("test");
        let id = project.add_resource(
            "Floor",
            ResourceData::Texture {
                psxt_path: "assets/textures/floor.psxt".to_string(),
            },
        );

        let error = project
            .rename_resource_with_files(id, "Stone Floor", &root)
            .unwrap_err();

        assert!(matches!(error, ResourceRenameError::TargetExists(_)));
        assert_eq!(project.resource_name(id), Some("Floor"));
        let ResourceData::Texture { psxt_path } = &project.resource(id).unwrap().data else {
            panic!("expected texture");
        };
        assert_eq!(psxt_path, "assets/textures/floor.psxt");
        assert_eq!(
            std::fs::read(texture_dir.join("floor.psxt")).unwrap(),
            b"old"
        );
        assert_eq!(
            std::fs::read(texture_dir.join("stone_floor.psxt")).unwrap(),
            b"target"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn delete_resource_removes_entry_and_clears_references() {
        let root = unique_temp_dir("resource-delete");
        let texture_dir = root.join("assets").join("textures");
        std::fs::create_dir_all(&texture_dir).unwrap();
        std::fs::write(texture_dir.join("target.psxt"), b"texture").unwrap();

        let mut project = ProjectDocument::new("delete-resource");
        let target = project.add_resource(
            "Target",
            ResourceData::Texture {
                psxt_path: "assets/textures/target.psxt".to_string(),
            },
        );
        let material = project.add_resource(
            "Material",
            ResourceData::Material(MaterialResource::opaque(Some(target))),
        );
        let character = project.add_resource(
            "Character",
            ResourceData::Character(CharacterResource {
                model: Some(target),
                idle_clip: Some(0),
                walk_clip: Some(1),
                run_clip: Some(2),
                turn_clip: Some(3),
                ..CharacterResource::defaults()
            }),
        );
        let weapon = project.add_resource(
            "Weapon",
            ResourceData::Weapon(WeaponResource {
                model: Some(target),
                ..WeaponResource::default()
            }),
        );

        let scene = project.active_scene_mut();
        let mut grid = WorldGrid::empty(1, 1, 1024);
        grid.set_floor(0, 0, 0, Some(target));
        grid.add_wall(0, 0, GridDirection::North, 0, 1024, Some(target));
        let room = scene.add_node(scene.root, "Room", NodeKind::Room { grid });
        scene.add_node(
            room,
            "Mesh",
            NodeKind::MeshInstance {
                mesh: Some(target),
                material: Some(target),
                animation_clip: None,
            },
        );
        let entity = scene.add_node(room, "Entity", NodeKind::Entity);
        scene.add_node(
            entity,
            "Renderer",
            NodeKind::ModelRenderer {
                model: Some(target),
                material: Some(target),
            },
        );
        scene.add_node(
            entity,
            "Controller",
            NodeKind::CharacterController {
                character: Some(target),
                player: true,
            },
        );
        scene.add_node(
            entity,
            "Equipment",
            NodeKind::Equipment {
                weapon: Some(target),
                character_socket: "right_hand_grip".to_string(),
                weapon_grip: "grip".to_string(),
            },
        );
        scene.add_node(
            room,
            "Spawn",
            NodeKind::SpawnPoint {
                player: false,
                character: Some(target),
            },
        );
        scene.add_node(
            room,
            "Audio",
            NodeKind::AudioSource {
                sound: Some(target),
                radius: 1.0,
            },
        );

        assert_eq!(project.resource_reference_count(target), 13);
        let report = project
            .delete_resource_with_files(target, &root)
            .expect("resource exists");
        assert_eq!(report.removed.name, "Target");
        assert_eq!(report.cleared_references, 13);
        assert_eq!(
            report.deleted_files,
            vec![ResourceFileDelete {
                path: "assets/textures/target.psxt".to_string(),
            }]
        );
        assert!(report.skipped_files.is_empty());
        assert!(!texture_dir.join("target.psxt").exists());
        assert!(project.resource(target).is_none());
        assert_eq!(project.resource_name(material), Some("Material"));

        let ResourceData::Material(material_data) = &project.resource(material).unwrap().data
        else {
            panic!("expected material");
        };
        assert_eq!(material_data.texture, None);
        let ResourceData::Character(character_data) = &project.resource(character).unwrap().data
        else {
            panic!("expected character");
        };
        assert_eq!(character_data.model, None);
        assert_eq!(character_data.idle_clip, None);
        assert_eq!(character_data.walk_clip, None);
        assert_eq!(character_data.run_clip, None);
        assert_eq!(character_data.turn_clip, None);
        let ResourceData::Weapon(weapon_data) = &project.resource(weapon).unwrap().data else {
            panic!("expected weapon");
        };
        assert_eq!(weapon_data.model, None);

        for node in project.active_scene().nodes() {
            match &node.kind {
                NodeKind::Room { grid } => {
                    let sector = grid.sector(0, 0).unwrap();
                    assert_eq!(sector.floor.as_ref().unwrap().material, None);
                    assert_eq!(
                        sector
                            .walls
                            .get(GridDirection::North)
                            .first()
                            .unwrap()
                            .material,
                        None
                    );
                }
                NodeKind::MeshInstance { mesh, material, .. } => {
                    assert_eq!((*mesh, *material), (None, None));
                }
                NodeKind::ModelRenderer { model, material } => {
                    assert_eq!((*model, *material), (None, None));
                }
                NodeKind::CharacterController { character, .. }
                | NodeKind::SpawnPoint { character, .. } => {
                    assert_eq!(*character, None);
                }
                NodeKind::Equipment { weapon, .. } => {
                    assert_eq!(*weapon, None);
                }
                NodeKind::AudioSource { sound, .. } => {
                    assert_eq!(*sound, None);
                }
                _ => {}
            }
        }

        let _ = std::fs::remove_dir_all(root);
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

    #[test]
    fn grid_uv_transform_rotates_quad_without_rebaking_texture() {
        let transform = GridUvTransform {
            offset: [0, 0],
            span: [0, 0],
            rotation: GridUvRotation::Deg90,
            flip_u: false,
            flip_v: false,
        };

        assert_eq!(
            transform.apply_to_quad([(0, 0), (64, 0), (64, 64), (0, 64)]),
            [(64, 0), (64, 64), (0, 64), (0, 0)]
        );
    }

    #[test]
    fn grid_uv_transform_flips_and_wraps_ps1_uv_offsets() {
        let transform = GridUvTransform {
            offset: [-8, 12],
            span: [0, 0],
            rotation: GridUvRotation::Deg0,
            flip_u: true,
            flip_v: false,
        };

        assert_eq!(
            transform.apply_to_quad([(0, 0), (64, 0), (64, 64), (0, 64)]),
            [(56, 12), (248, 12), (248, 76), (56, 76)]
        );
    }

    #[test]
    fn grid_uv_transform_scales_quad_span_without_rebaking_texture() {
        let transform = GridUvTransform {
            offset: [0, 0],
            span: [0, 32],
            rotation: GridUvRotation::Deg0,
            flip_u: false,
            flip_v: false,
        };

        assert_eq!(
            transform.apply_to_quad([(0, 64), (64, 64), (64, 0), (0, 0)]),
            [(0, 32), (64, 32), (64, 0), (0, 0)]
        );
    }

    #[test]
    fn wall_autotile_sets_double_height_v_span_without_changing_geometry() {
        let mut wall = GridVerticalFace::flat(0, 1536, None);
        let heights = wall.heights;

        let clamped = wall.autotile_uv(768);

        assert!(!clamped);
        assert_eq!(wall.heights, heights);
        assert_eq!(wall.uv.span, [0, 128]);
    }

    #[test]
    fn wall_autotile_uses_partial_v_span_for_short_wall() {
        let mut wall = GridVerticalFace::flat(0, 384, None);

        let clamped = wall.autotile_uv(768);

        assert!(!clamped);
        assert_eq!(wall.heights, [0, 0, 384, 384]);
        assert_eq!(wall.uv.span, [0, 32]);
    }

    #[test]
    fn wall_autotile_clamps_one_quad_to_ps1_uv_range() {
        let mut wall = GridVerticalFace::flat(0, 768 * 5, None);

        let clamped = wall.autotile_uv(768);

        assert!(clamped);
        assert_eq!(wall.heights, [0, 0, 3840, 3840]);
        assert_eq!(wall.uv.span, [0, 255]);
    }

    #[test]
    fn wall_autotile_keeps_one_primitive_when_repeated_uvs_fit_packet() {
        let mut wall = GridVerticalFace::flat(0, 1536, None);
        wall.uv.offset[1] = -5;
        wall.autotile_uv(768);

        let segments = wall.split_into_autotile_segments(768);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].heights, [0, 0, 1536, 1536]);
        assert_eq!(segments[0].uv.span, [0, 128]);
        assert_eq!(segments[0].uv.offset[1], -5);
    }

    #[test]
    fn wall_autotile_segments_restore_clamped_tall_wall_density() {
        let mut wall = GridVerticalFace::flat(0, 768 * 5, None);
        wall.autotile_uv(768);

        let segments = wall.split_into_autotile_segments(768);

        assert_eq!(segments.len(), 5);
        assert!(segments.iter().all(|segment| segment.uv.span == [0, 0]));
        assert_eq!(segments[4].heights, [3072, 3072, 3840, 3840]);
    }

    #[test]
    fn wall_split_height_segments_keeps_uvs_and_sloped_edges_connected() {
        let mut wall = GridVerticalFace::flat(0, 1536, None);
        wall.heights = [0, 384, 1536, 1920];
        wall.uv.span = [12, 96];

        let segments = wall.split_into_height_segments(768);

        assert_eq!(segments.len(), 3);
        assert_eq!(
            [
                segments[0].heights[WallCorner::BL.idx()],
                segments[0].heights[WallCorner::BR.idx()],
            ],
            [0, 384]
        );
        assert_eq!(
            [
                segments[2].heights[WallCorner::TL.idx()],
                segments[2].heights[WallCorner::TR.idx()],
            ],
            [1920, 1536]
        );
        for pair in segments.windows(2) {
            assert_eq!(
                pair[0].heights[WallCorner::TL.idx()],
                pair[1].heights[WallCorner::BL.idx()]
            );
            assert_eq!(
                pair[0].heights[WallCorner::TR.idx()],
                pair[1].heights[WallCorner::BR.idx()]
            );
        }
        assert!(segments.iter().all(|segment| segment.uv.span == [12, 96]));
    }
}
