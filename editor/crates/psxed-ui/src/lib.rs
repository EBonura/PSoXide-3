//! egui editor workspace for PSoXide.
//!
//! The frontend owns the window/Menu. This crate owns the editor panels and
//! the in-memory authoring document they manipulate.

mod history;
mod icons;
mod model_import_preview;
mod play_mode;
mod style;

pub use play_mode::{
    EditorPlaytestRequest, EditorPlaytestStatus, EditorViewport3dMode, EditorViewport3dPresentation,
};

use crate::history::UndoStack;
use crate::style::*;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use egui::{
    Align2, Color32, ColorImage, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Vec2,
};
use psxed_project::{
    snap_height, ColliderShape, GridCellBounds, GridDirection, GridHorizontalFace, GridSplit,
    GridVerticalFace, MaterialFaceSidedness, MaterialResource, NodeId, NodeKind, NodeRow,
    ProjectDocument, PsxBlendMode, Resource, ResourceData, ResourceId, WorldGrid, WorldGridBudget,
    HEIGHT_QUANTUM, MAX_ROOM_BYTES, MAX_ROOM_DEPTH, MAX_ROOM_TRIANGLES, MAX_ROOM_WIDTH,
};

const LEFT_DOCK_MAX_WIDTH: f32 = 420.0;
const LEFT_DOCK_LABEL_CHARS: usize = 34;

/// Discrete action a scene-tree row can produce in one frame.
///
/// The panel iterates rows borrowing `&self.project` immutably; rows
/// describe what they want to happen via this enum, and the panel
/// drains the queue after iteration so all the mutating helpers
/// (`push_undo`, `add_node`, `move_node`, …) can take `&mut self`
/// without fighting the iteration borrow.
enum TreeAction {
    Select(NodeId),
    BeginRename(NodeId),
    CommitRename(NodeId, String),
    CancelRename,
    Delete(NodeId),
    Duplicate(NodeId),
    AddChild {
        parent: NodeId,
        kind: NodeKind,
        name: &'static str,
    },
    /// Move `source` so it becomes a child of `target_parent` at
    /// `position` in that parent's child list. Caller has already
    /// proven the move is non-cyclic; `Scene::move_node` re-checks.
    Reparent {
        source: NodeId,
        target_parent: NodeId,
        position: usize,
    },
}

/// Embedded editor workspace state.
pub struct EditorWorkspace {
    project: ProjectDocument,
    project_dir: PathBuf,
    new_project_dialog_open: bool,
    new_project_name: String,
    new_project_error: Option<String>,
    selected_node: NodeId,
    selected_resource: Option<ResourceId>,
    /// Highlighted sector cell within the active Room. Tracked so the
    /// inspector can show per-cell properties without inflating the
    /// scene-tree node count with a node per sector.
    selected_sector: Option<(u16, u16)>,
    /// Selection mode the Select tool picks at: a whole face,
    /// one of its edges, or one of its corners. Hotkeys 1 / 2 / 3
    /// toggle.
    selection_mode: SelectionMode,
    /// Primitive under the pointer while the Select tool is
    /// active. Updated every frame the panel is hovered and the
    /// tool is Select; cleared when the pointer leaves or another
    /// tool takes over. The renderer outlines this lightly so the
    /// user sees what the next click will pick.
    hovered_primitive: Option<Selection>,
    /// Primitive the user clicked with the Select tool last.
    /// Persists across frames until the user clicks a different
    /// one or switches tools. The renderer outlines it more
    /// boldly than `hovered_primitive`; the inspector reads it to
    /// surface per-primitive properties.
    selected_primitive: Option<Selection>,
    /// Active drag-translate stroke. Set on drag-start over a
    /// primitive in Select mode, mutated by every drag frame, and
    /// cleared on release. Records pre-drag heights of every
    /// physical vertex involved so the apply step is
    /// `snap(pre_y + delta)` -- clean snapping with no error
    /// accumulation.
    primitive_drag: Option<PrimitiveDrag>,
    /// Hovered entity bound under the cursor (Select tool
    /// only). Drives the entity-bounds highlight overlay and
    /// the click→select fast path.
    hovered_entity_node: Option<NodeId>,
    /// Active 3D node-drag stroke. Set when the user
    /// presses on an entity bound; updated each frame the
    /// pointer moves while held; cleared on release.
    node_drag: Option<NodeDrag>,
    /// What the next paint click would target. Cell variant fires
    /// for floor / ceiling / erase / place; Wall variant fires for
    /// PaintWall. World-cell coords let the preview track cells
    /// outside the current grid bounds -- the renderer outlines
    /// them as ghosts at the world position the auto-grow would
    /// place them.
    paint_target_preview: Option<PaintTargetPreview>,
    /// Last paint stamp committed during the current drag. Edge-
    /// aware so dragging across different edges of the same cell
    /// stamps each one (PaintWall sweeping a cell can hit N then
    /// E without the dedupe blocking the second click), but
    /// dwelling on the same edge doesn't re-stack walls. Reset
    /// to `None` whenever a new primary click starts.
    last_paint_stamp: Option<PaintStamp>,
    /// `Some((id, buffer))` while a scene-tree row is in rename mode.
    /// Buffer holds the in-flight string the user is typing; commit /
    /// cancel finalises against the actual node name.
    renaming: Option<(NodeId, String)>,
    /// One-shot flag set when entering rename mode so the next frame
    /// requests focus + selects the text inside the rename TextEdit.
    pending_rename_focus: bool,
    history: UndoStack,
    scene_filter: String,
    file_filter: String,
    resource_search: String,
    resource_filter: ResourceFilter,
    /// `Some((id, buffer))` while the resource inspector's name
    /// field is editing. Committed resource renames may move backing
    /// files, so they happen on focus loss / Enter rather than on
    /// every keystroke.
    resource_renaming: Option<(ResourceId, String)>,
    active_tool: ViewTool,
    /// Kind of node the Place tool drops on a click. Surfaces in
    /// the toolbar as a small picker visible only when
    /// `active_tool == Place`. Player Spawn enforces uniqueness
    /// at place-time; the others are additive markers.
    place_kind: PlaceKind,
    /// Material the next Floor / Wall / Ceiling paint will use, when
    /// set. `None` means "fall back to the name-based heuristic
    /// (`floor → first 'floor' material, brick → first 'brick' …`)" --
    /// which is also what fresh projects start with so painting works
    /// before any material is hand-picked.
    brush_material: Option<ResourceId>,
    snap_to_grid: bool,
    snap_units: u16,
    show_grid: bool,
    view_2d: bool,
    viewport_pan: Vec2,
    viewport_zoom: f32,
    /// Orbit camera for the 3D viewport. Yaw + pitch in 4096-per-turn
    /// Q12 units (matching `psx_engine::WorldCamera::orbit`); radius
    /// in world units. Drag on the 3D panel rotates yaw/pitch; scroll
    /// changes radius.
    viewport_3d_yaw: u16,
    viewport_3d_pitch: u16,
    viewport_3d_radius: i32,
    /// Decoded `.psxt` thumbnails for the resources panel. Built
    /// lazily once per Texture resource (or whenever its `psxt_path`
    /// changes); the egui texture handle stays alive across frames
    /// so the painter can blit it into resource cards without
    /// re-decoding. Keyed on the *Texture* resource id; Materials
    /// follow `material.texture` to the same key.
    texture_thumbs: HashMap<ResourceId, ThumbnailEntry>,
    model_import_dialog: ModelImportDialog,
    model_import_retired_textures: Vec<(u8, egui::TextureHandle)>,
    dirty: bool,
    status: String,
    /// One-shot request emitted by the editor UI for the frontend
    /// to handle. The frontend owns emulator state and build child
    /// processes, so the editor never launches playtest directly.
    pending_playtest_request: Option<EditorPlaytestRequest>,
}

/// One cached `.psxt` thumbnail plus the metadata the inspector
/// reads off the same parse. `signature` is the path the handle was
/// built from -- when the user retypes the path on a Texture
/// resource, the signature mismatches and the cache rebuilds.
struct ThumbnailEntry {
    signature: String,
    handle: egui::TextureHandle,
    stats: PsxtStats,
}

struct ModelImportDialog {
    open: bool,
    source_path: String,
    output_name: String,
    texture_width: i32,
    texture_height: i32,
    animation_fps: i32,
    world_height: i32,
    normalize_root_translation: bool,
    selected_clip: usize,
    preview_yaw_q12: i32,
    preview_pitch_q12: i32,
    preview_radius: i32,
    show_animation_root: bool,
    status: Option<ModelImportStatus>,
    preview: Option<ModelImportPreview>,
}

impl Default for ModelImportDialog {
    fn default() -> Self {
        Self {
            open: false,
            source_path: String::new(),
            output_name: String::new(),
            texture_width: 128,
            texture_height: 128,
            animation_fps: 15,
            world_height: 1024,
            normalize_root_translation: true,
            selected_clip: 0,
            preview_yaw_q12: 340,
            preview_pitch_q12: 350,
            preview_radius: 1536,
            show_animation_root: true,
            status: None,
            preview: None,
        }
    }
}

enum ModelImportStatus {
    Info(String),
    Error(String),
}

struct ModelImportPreview {
    model_bytes: Vec<u8>,
    report: psxed_project::model_import::RigidModelReport,
    atlas: Option<(egui::TextureHandle, PsxtStats)>,
    atlas_image: Option<ColorImage>,
    animated_texture: Option<egui::TextureHandle>,
    world_height: i32,
    clips: Vec<ModelImportClipPreview>,
}

struct ModelImportClipPreview {
    name: String,
    frames: usize,
    bytes: Vec<u8>,
    byte_len: usize,
    root_motion: Option<RootMotionStats>,
}

#[derive(Copy, Clone)]
struct RootMotionStats {
    min: [i32; 3],
    max: [i32; 3],
    mean: [i32; 3],
}

/// Decoded metadata for one `.psxt` blob. Cheap to compute
/// (header parse + lengths); shown in the resource inspector so
/// authors can spot mismatches against an authored target depth /
/// dimensions without leaving the editor.
#[derive(Debug, Clone, Copy)]
struct PsxtStats {
    width: u16,
    height: u16,
    /// 4, 8, or 15 -- mirrors `psxed_format::texture::Depth`'s
    /// numeric form.
    depth_bits: u8,
    /// 16 for 4bpp, 256 for 8bpp, 0 for 15bpp.
    clut_entries: u16,
    pixel_bytes: u32,
    clut_bytes: u32,
    file_bytes: u32,
}

/// Counts shown in the embedded Play readiness status line.
/// Built once per cook from the validated package; stays in
/// the editor crate (not psxed-project) because the status
/// string is editor-facing UI text.
struct PackageSummary {
    rooms: usize,
    assets: usize,
    textures: usize,
    materials: usize,
    models: usize,
    characters: usize,
    lights: usize,
    entities: usize,
    /// Display name of the player's resolved Character, or
    /// `None` when no player controller was emitted.
    player_character: Option<String>,
}

/// Per-stamp paint dedupe key. Two paint events with equal stamps
/// are considered redundant during a single drag -- typically the
/// second is dropped. Edge / stack components let PaintWall stamp
/// multiple edges of the same cell without dwelling on one
/// re-firing the same wall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PaintStamp {
    room: NodeId,
    sx: u16,
    sz: u16,
    tool: ViewTool,
    edge: Option<GridDirection>,
    stack: Option<u8>,
}

/// What the next paint click would target. Carries world-cell
/// coordinates (which can be negative -- outside the current grid)
/// so the renderer can preview cells the next click would auto-
/// create. Stays populated for any paint tool, mirroring the
/// dispatch so what you preview is what you'll paint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintTargetPreview {
    /// Floor / ceiling / erase / place -- outlines the cell.
    Cell {
        world_cell_x: i32,
        world_cell_z: i32,
    },
    /// PaintWall -- outlines the wall on the targeted edge of the
    /// cell. `stack` is the next-free wall slot index for that
    /// edge, used by the renderer to position the ghost above any
    /// existing walls.
    Wall {
        world_cell_x: i32,
        world_cell_z: i32,
        dir: GridDirection,
        stack: u8,
    },
}

/// One pickable surface on the active Room's grid. Floors and
/// ceilings are addressed by sector; walls add a cardinal direction
/// plus a stack index (a single edge can hold multiple stacked walls
/// -- windows / arches -- and each is independently selectable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceKind {
    Floor,
    Ceiling,
    Wall { dir: GridDirection, stack: u8 },
}

/// A face inside the active Room, fully qualified by Room id +
/// sector + face kind. Used by the Select tool's hover / selected
/// state and the per-face inspector that follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceRef {
    pub room: NodeId,
    pub sx: u16,
    pub sz: u16,
    pub kind: FaceKind,
}

// Corner / WallCorner live in `psxed-project` so faces can carry
// `dropped_corner` data with serde support. Re-exported here so
// existing imports (`use psxed_ui::Corner`) keep working.
pub use psxed_project::{Corner, WallCorner};

/// Which of the four edges of a wall quad. Order matches the
/// perimeter walk used by the picker:
/// `Bottom = BL-BR`, `Right = BR-TR`, `Top = TR-TL`,
/// `Left = TL-BL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallEdge {
    Bottom,
    Right,
    Top,
    Left,
}

/// One face-corner. `Selection::Vertex(_)` resolves through
/// [`physical_vertex`] to a `Vec<FaceCornerRef>` listing every
/// face-corner currently sharing the same world position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceCornerRef {
    Floor {
        sx: u16,
        sz: u16,
        corner: Corner,
    },
    Ceiling {
        sx: u16,
        sz: u16,
        corner: Corner,
    },
    Wall {
        sx: u16,
        sz: u16,
        dir: GridDirection,
        stack: u8,
        corner: WallCorner,
    },
}

/// Vertex in a `Selection`. Carries the *seed* corner -- the one
/// the user actually clicked. Resolve to a `PhysicalVertex` to
/// get every coincident face-corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexRef {
    pub room: NodeId,
    pub anchor: VertexAnchor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexAnchor {
    Floor {
        sx: u16,
        sz: u16,
        corner: Corner,
    },
    Ceiling {
        sx: u16,
        sz: u16,
        corner: Corner,
    },
    Wall {
        sx: u16,
        sz: u16,
        dir: GridDirection,
        stack: u8,
        corner: WallCorner,
    },
}

impl VertexAnchor {
    pub const fn as_face_corner(self) -> FaceCornerRef {
        match self {
            Self::Floor { sx, sz, corner } => FaceCornerRef::Floor { sx, sz, corner },
            Self::Ceiling { sx, sz, corner } => FaceCornerRef::Ceiling { sx, sz, corner },
            Self::Wall {
                sx,
                sz,
                dir,
                stack,
                corner,
            } => FaceCornerRef::Wall {
                sx,
                sz,
                dir,
                stack,
                corner,
            },
        }
    }
}

/// Edge in a `Selection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeRef {
    pub room: NodeId,
    pub anchor: EdgeAnchor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeAnchor {
    Floor {
        sx: u16,
        sz: u16,
        dir: GridDirection,
    },
    Ceiling {
        sx: u16,
        sz: u16,
        dir: GridDirection,
    },
    Wall {
        sx: u16,
        sz: u16,
        dir: GridDirection,
        stack: u8,
        edge: WallEdge,
    },
}

/// Tagged selection used by the editor's Select tool. Replaces
/// the previous `selected_face: Option<FaceRef>` so all three
/// modes share one piece of state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    Face(FaceRef),
    Edge(EdgeRef),
    Vertex(VertexRef),
}

impl Selection {
    /// The room this selection belongs to.
    pub const fn room(&self) -> NodeId {
        match self {
            Self::Face(f) => f.room,
            Self::Edge(e) => e.room,
            Self::Vertex(v) => v.room,
        }
    }

    /// Convenience: when the selection is a face, hand it to
    /// callers that still want the old `FaceRef` shape (e.g.
    /// the per-face inspector).
    pub const fn as_face(&self) -> Option<FaceRef> {
        match self {
            Self::Face(f) => Some(*f),
            _ => None,
        }
    }
}

/// Kind label for an [`EntityBounds`]. Drives picking
/// priorities and per-kind gizmo rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityBoundKind {
    /// Model-backed `MeshInstance` with parsed model bounds.
    Model,
    /// Legacy / unbound `MeshInstance` -- fallback box.
    MeshFallback,
    /// `SpawnPoint` (player or non-player).
    SpawnPoint,
    /// `Light`. Marker box only -- radius ring is drawn
    /// separately so a wide-radius light doesn't intercept
    /// every click in the room.
    Light,
    /// `Trigger`.
    Trigger,
    /// `Portal`.
    Portal,
    /// `AudioSource`. Marker box only.
    AudioSource,
}

/// World-space AABB for one selectable scene entity.
/// Coordinates use [`psxed_project::spatial::node_preview_bounds_center`]
/// for entities under a Room, so bounds line up with the same
/// origin-aware preview world used by rendered models, markers, and
/// lights.
#[derive(Debug, Clone, Copy)]
pub struct EntityBounds {
    /// Owning scene-tree node id.
    pub node: NodeId,
    /// Enclosing Room id, if any. Used to filter picking to
    /// the active room.
    pub room: Option<NodeId>,
    /// Bound class for visual styling + picking priority.
    pub kind: EntityBoundKind,
    /// World-space AABB centre.
    pub center: [f32; 3],
    /// World-space half-extents along X / Y / Z. Always
    /// positive.
    pub half_extents: [f32; 3],
    /// Authored Y rotation in degrees. Stored on the bound
    /// so the renderer can draw a facing arrow without
    /// re-walking the scene tree.
    pub yaw_degrees: f32,
}

/// Result of a successful entity-bound pick.
#[derive(Debug, Clone, Copy)]
pub struct EntityBoundHit {
    /// Hit node.
    pub node: NodeId,
    /// Distance from the ray origin to the first hit slab,
    /// in world units. Used to compare against grid hits and
    /// other entity hits.
    pub distance: f32,
    /// World-space hit point along the ray.
    pub point: [f32; 3],
    /// Bounds that produced the hit.
    pub bounds: EntityBounds,
}

/// Slab-intersection ray-vs-AABB. Returns the smallest
/// non-negative `t` for which `origin + t * dir` lands on
/// the box surface (or inside it).
///
/// * `dir` is *not* required to be unit length; the returned
///   `t` is in the same units as `dir`. When the editor uses
///   normalized rays (`camera_ray_for_pointer`), `t` lands in
///   world units.
/// * Box must have positive `half_extents`. Zero-extent boxes
///   never hit.
/// * Rays starting *inside* the box return `t = 0` so callers
///   can still pick something they're standing on.
pub fn ray_intersects_aabb(
    origin: [f32; 3],
    dir: [f32; 3],
    center: [f32; 3],
    half_extents: [f32; 3],
) -> Option<f32> {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    for axis in 0..3 {
        let half = half_extents[axis];
        if half <= 0.0 {
            return None;
        }
        let lo = center[axis] - half;
        let hi = center[axis] + half;
        let o = origin[axis];
        let d = dir[axis];
        if d.abs() < 1e-6 {
            // Ray parallel to this axis -- only hits if origin
            // is between the slabs.
            if o < lo || o > hi {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let t1 = (lo - o) * inv;
            let t2 = (hi - o) * inv;
            let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            if t_near > t_min {
                t_min = t_near;
            }
            if t_far < t_max {
                t_max = t_far;
            }
            if t_min > t_max {
                return None;
            }
        }
    }
    if t_max < 0.0 {
        return None;
    }
    Some(if t_min < 0.0 { 0.0 } else { t_min })
}

/// Intersect a ray with the horizontal plane `y = plane_y`.
/// Used by the entity-drag path to project mouse-move into
/// world-space on the same plane the entity lives on.
/// Returns `None` for parallel rays or hits behind the camera.
pub fn ray_intersects_horizontal_plane(
    origin: [f32; 3],
    dir: [f32; 3],
    plane_y: f32,
) -> Option<[f32; 3]> {
    if dir[1].abs() < 1e-6 {
        return None;
    }
    let t = (plane_y - origin[1]) / dir[1];
    if t < 0.0 {
        return None;
    }
    Some([origin[0] + dir[0] * t, plane_y, origin[2] + dir[2] * t])
}

#[cfg(test)]
mod entity_bounds_tests {
    use super::ray_intersects_aabb as ray_aabb;
    use super::ray_intersects_horizontal_plane as ray_plane;

    #[test]
    fn ray_aabb_hits_centred_box() {
        // Ray along +Z toward origin AABB at distance 10.
        let t = ray_aabb(
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        assert!(t.is_some());
        // Hit should land on the near slab at t = 9.
        assert!((t.unwrap() - 9.0).abs() < 1e-3);
    }

    #[test]
    fn ray_aabb_misses_offset_box() {
        // Box offset to +X by 100 -- a +Z ray at origin misses.
        let t = ray_aabb(
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 1.0],
            [100.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        assert!(t.is_none());
    }

    #[test]
    fn ray_aabb_origin_inside_box_returns_zero() {
        let t = ray_aabb(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
        );
        assert_eq!(t, Some(0.0));
    }

    #[test]
    fn ray_aabb_zero_extent_never_hits() {
        let t = ray_aabb(
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        );
        assert!(t.is_none());
    }

    #[test]
    fn ray_aabb_ray_parallel_to_slab() {
        // Ray on the X axis at Y=10, box at Y=0. Parallel +X
        // ray never enters the Y slab so it must miss.
        let t = ray_aabb(
            [0.0, 10.0, 0.0],
            [1.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
        );
        assert!(t.is_none());
    }

    #[test]
    fn ray_aabb_nearest_of_two_boxes() {
        // Two co-axial boxes; near box at z=10, far box at
        // z=50. Nearest t corresponds to the near box.
        let near = ray_aabb(
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let far = ray_aabb(
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 50.0],
            [1.0, 1.0, 1.0],
        );
        assert!(near.unwrap() < far.unwrap());
    }

    #[test]
    fn ray_plane_hits_horizontal_plane_below() {
        // Camera 100 above origin looking down → +Z forward,
        // -Y up. Hit floor plane y=0 at t=100.
        let p = ray_plane([0.0, 100.0, 0.0], [0.0, -1.0, 0.0], 0.0);
        assert!(p.is_some());
        let p = p.unwrap();
        assert!((p[1] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn ray_plane_misses_when_parallel() {
        let p = ray_plane([0.0, 100.0, 0.0], [1.0, 0.0, 0.0], 0.0);
        assert!(p.is_none());
    }

    #[test]
    fn ray_plane_misses_when_behind_camera() {
        // Ray points away from the plane.
        let p = ray_plane([0.0, 100.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert!(p.is_none());
    }
}

/// Three-mode selection switch -- Blender-style. `Face` keeps
/// the existing whole-face semantics; `Edge` and `Vertex` pick
/// finer primitives via local-UV math on the picked face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionMode {
    #[default]
    Face,
    Edge,
    Vertex,
}

impl SelectionMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Face => "Face",
            Self::Edge => "Edge",
            Self::Vertex => "Vertex",
        }
    }
}

/// In-flight drag-translate stroke. Captured at drag-start
/// over a primitive in Select mode and applied every frame the
/// pointer moves until release.
#[derive(Debug, Clone)]
struct PrimitiveDrag {
    /// What's being dragged -- used to refresh the visual outline
    /// while the geometry under it shifts.
    target: Selection,
    /// Physical vertices to translate -- typically 1 (Vertex
    /// mode), 2 (Edge mode), or 4 (Face mode). Each entry
    /// carries every coincident face-corner so a single drag
    /// applies through the universal-coincidence resolver
    /// without re-walking the grid each frame.
    vertices: Vec<PhysicalVertex>,
    /// Pre-drag Y of each entry in `vertices`, in the same
    /// order. Apply step is `snap(pre_y + total_delta_world)`
    /// for every vertex -- single source of truth for the new
    /// height, no accumulator drift.
    pre_drag_ys: Vec<i32>,
    /// Total mouse-Y travel since drag-start, in screen pixels.
    /// Sign-flipped at apply time (screen +Y is down, world +Y
    /// is up). Stored as `f32` because egui hands per-frame
    /// deltas that way.
    accumulated_pixel_dy: f32,
    /// Whether `push_undo` has fired for this stroke. Lazy:
    /// only fires the first time `accumulated_pixel_dy` causes
    /// a non-zero quantum delta, so a press-without-drag (a
    /// pure click) leaves the undo stack alone.
    snapshot_pushed: bool,
}

/// Active node-drag stroke. Set on press over an entity
/// bound, updated each frame the pointer moves with primary
/// held, cleared on release. The drag is constrained to the
/// horizontal plane the node sits on so X/Z editing is the
/// only motion -- Y stays editable via the inspector.
#[derive(Debug, Clone)]
struct NodeDrag {
    /// The node being dragged.
    node: NodeId,
    /// Editor-space translation when the drag started. The
    /// per-frame update writes `start + delta` so floating
    /// rounding errors don't accumulate.
    start_translation: [f32; 3],
    /// World-space hit point on the node's drag plane at
    /// drag start. Subsequent ray hits on the same plane
    /// yield a delta to add to `start_translation`.
    start_world_hit: [f32; 3],
    /// Plane Y in world units -- locked to the node's current
    /// world Y at drag start.
    drag_plane_y: f32,
    /// `true` once `push_undo` has fired for this stroke.
    /// Pure clicks (press without movement) leave the undo
    /// stack untouched.
    snapshot_pushed: bool,
    /// Cached enclosing room id so per-frame updates don't
    /// re-walk the scene tree.
    room: Option<NodeId>,
}

/// Resolved physical vertex: every face-corner that currently
/// sits at `world` and therefore moves together when the
/// vertex's height is dragged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalVertex {
    /// Integer world position. Every member is at exactly this
    /// `(X, Y, Z)`.
    pub world: [i32; 3],
    /// Face-corners that share the position. Always non-empty
    /// (contains at least the seed).
    pub members: Vec<FaceCornerRef>,
}

/// Snapshot of the editor's 3D viewport camera, handed to the
/// frontend each frame so it can drive the editor-owned `HwRenderer`
/// from the same orbit state the editor's drag input updates.
#[derive(Debug, Clone, Copy)]
pub struct ViewportCameraState {
    /// Yaw, 4096 per full revolution.
    pub yaw_q12: u16,
    /// Pitch, 4096 per full revolution; positive raises the camera
    /// above the target so the view tilts down.
    pub pitch_q12: u16,
    /// Distance from the camera to the orbit target, world units.
    pub radius: i32,
}

/// Floating-point camera basis used by editor picking.
#[derive(Debug, Clone, Copy)]
pub struct ViewportCameraBasis {
    /// Camera position in editor preview world units.
    pub position: [f32; 3],
    /// Forward unit vector.
    pub forward: [f32; 3],
    /// Right unit vector.
    pub right: [f32; 3],
    /// Up unit vector.
    pub up: [f32; 3],
}

impl ViewportCameraState {
    /// Orbit camera basis around a preview-world target.
    pub fn basis(self, target_world: [f32; 3]) -> ViewportCameraBasis {
        let radius = self.radius as f32;
        let cos_p = psx_gte::transform::cos_1_3_12(self.pitch_q12) as f32 / 4096.0;
        let sin_p = psx_gte::transform::sin_1_3_12(self.pitch_q12) as f32 / 4096.0;
        let cos_y = psx_gte::transform::cos_1_3_12(self.yaw_q12) as f32 / 4096.0;
        let sin_y = psx_gte::transform::sin_1_3_12(self.yaw_q12) as f32 / 4096.0;
        let position = [
            target_world[0] + radius * cos_p * sin_y,
            target_world[1] - radius * sin_p,
            target_world[2] + radius * cos_p * cos_y,
        ];
        let forward = normalize3([
            target_world[0] - position[0],
            target_world[1] - position[1],
            target_world[2] - position[2],
        ]);
        let right = normalize3(cross3(forward, [0.0, 1.0, 0.0]));
        let up = cross3(right, forward);
        ViewportCameraBasis {
            position,
            forward,
            right,
            up,
        }
    }

    /// Build a world-space ray from normalized panel coordinates.
    ///
    /// `nx` and `ny` are in `[-1, 1]`, where `0, 0` is the panel
    /// centre. Constants match the editor preview's 320x240,
    /// projection-plane-320 camera.
    pub fn ray_for_normalized_panel_point(
        self,
        target_world: [f32; 3],
        nx: f32,
        ny: f32,
    ) -> ([f32; 3], [f32; 3]) {
        let basis = self.basis(target_world);
        let half_fov_x: f32 = 0.5;
        let half_fov_y: f32 = 0.5 * 240.0 / 320.0;
        let dir = normalize3([
            basis.forward[0]
                + basis.right[0] * (nx * half_fov_x)
                + basis.up[0] * (-ny * half_fov_y),
            basis.forward[1]
                + basis.right[1] * (nx * half_fov_x)
                + basis.up[1] * (-ny * half_fov_y),
            basis.forward[2]
                + basis.right[2] * (nx * half_fov_x)
                + basis.up[2] * (-ny * half_fov_y),
        ]);
        (basis.position, dir)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewTool {
    /// Click to select; press-and-drag on a selected primitive
    /// (face / edge / vertex) translates it vertically in the
    /// 3D viewport. No separate "Move" tool -- the same gesture
    /// handles both select and move.
    Select,
    /// Paint a floor onto the sector under the cursor (Room context).
    PaintFloor,
    /// Paint a wall on the directed edge under the cursor.
    PaintWall,
    /// Paint a ceiling on the sector under the cursor.
    PaintCeiling,
    /// Clear the painted surface under the cursor.
    Erase,
    /// Drop a child entity node into the sector under the cursor.
    /// The kind of node placed is controlled by `place_kind`.
    Place,
}

/// Variants of the `Place` tool. Maps directly onto the
/// `NodeKind` produced by a Place click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaceKind {
    /// `SpawnPoint { player: true }` -- the editor enforces
    /// uniqueness by demoting existing player spawns to
    /// generic spawns at place time.
    PlayerSpawn,
    /// `SpawnPoint { player: false }`. Multiple OK.
    SpawnMarker,
    /// `Entity + ModelRenderer + Animator` referencing a
    /// [`ResourceData::Model`]. Place flow picks the selected Model
    /// resource if set, auto-picks the only Model resource if exactly
    /// one exists, or refuses with an actionable error otherwise.
    ModelInstance,
    /// `Light` with default color / intensity / radius.
    LightMarker,
}

impl PlaceKind {
    const fn label(self) -> &'static str {
        match self {
            Self::PlayerSpawn => "Player Spawn",
            Self::SpawnMarker => "Spawn",
            Self::ModelInstance => "Model",
            Self::LightMarker => "Light",
        }
    }
}

impl ViewTool {
    /// `true` when the tool only makes sense once a Room is the
    /// active context -- viewport clicks should be suppressed
    /// otherwise so we don't paint into thin air.
    const fn requires_room_context(self) -> bool {
        matches!(
            self,
            Self::PaintFloor | Self::PaintWall | Self::PaintCeiling | Self::Erase | Self::Place
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceFilter {
    All,
    Texture,
    Material,
    Model,
    Character,
    Mesh,
    Room,
    Other,
}

impl ResourceFilter {
    const fn label(self) -> &'static str {
        match self {
            Self::All => "All Resources",
            Self::Texture => "Texture",
            Self::Material => "Material",
            Self::Model => "Model",
            Self::Character => "Character",
            Self::Mesh => "Mesh",
            Self::Room => "Room",
            Self::Other => "Other",
        }
    }

    const fn icon(self) -> char {
        match self {
            Self::All => icons::LAYERS,
            Self::Texture => icons::PALETTE,
            Self::Material => icons::BLEND,
            Self::Model => icons::BOX,
            Self::Character => icons::MAP_PIN,
            Self::Mesh => icons::BOX,
            Self::Room => icons::GRID,
            Self::Other => icons::FILE,
        }
    }

    fn matches(self, data: &ResourceData) -> bool {
        match self {
            Self::All => true,
            Self::Texture => matches!(data, ResourceData::Texture { .. }),
            Self::Material => matches!(data, ResourceData::Material(_)),
            Self::Model => matches!(data, ResourceData::Model(_)),
            Self::Character => matches!(data, ResourceData::Character(_)),
            Self::Mesh => matches!(data, ResourceData::Mesh { .. }),
            Self::Room => matches!(data, ResourceData::Scene { .. }),
            Self::Other => matches!(
                data,
                ResourceData::Script { .. } | ResourceData::Audio { .. }
            ),
        }
    }
}

impl EditorWorkspace {
    /// Open the project at `dir`. Errors when `dir/project.ron` is
    /// missing or malformed -- the frontend wraps the error and falls
    /// back to the default project so a real load failure surfaces
    /// in the status bar instead of silently spawning a fresh
    /// starter (which masked the path-resolution bug previously).
    pub fn open_directory(dir: impl Into<PathBuf>) -> Result<Self, String> {
        let dir = dir.into();
        let project_file = dir.join("project.ron");
        let project = ProjectDocument::load_from_path(&project_file)
            .map_err(|error| format!("load {}: {error}", project_file.display()))?;
        let mut workspace = Self::with_project(dir, project);
        workspace.status = format!("Loaded {}", short_path(&workspace.project_dir));
        workspace.select_first_room();
        Ok(workspace)
    }

    /// Construct a workspace around an already-loaded project. Used
    /// by `open_directory` and `create_and_open_project`; not part
    /// of the public API.
    fn with_project(project_dir: PathBuf, project: ProjectDocument) -> Self {
        Self {
            project,
            project_dir,
            new_project_dialog_open: false,
            new_project_name: String::new(),
            new_project_error: None,
            selected_node: NodeId::ROOT,
            selected_resource: None,
            selected_sector: None,
            selection_mode: SelectionMode::default(),
            hovered_primitive: None,
            selected_primitive: None,
            primitive_drag: None,
            hovered_entity_node: None,
            node_drag: None,
            paint_target_preview: None,
            last_paint_stamp: None,
            renaming: None,
            pending_rename_focus: false,
            history: UndoStack::default(),
            scene_filter: String::new(),
            file_filter: String::new(),
            resource_search: String::new(),
            resource_filter: ResourceFilter::All,
            resource_renaming: None,
            active_tool: ViewTool::Select,
            place_kind: PlaceKind::PlayerSpawn,
            brush_material: None,
            snap_to_grid: true,
            snap_units: 16,
            show_grid: true,
            // Default to the 3D preview so the bit-faithful HwRenderer
            // is the first thing the user sees on opening the editor.
            // The 2D top-down view stays one toolbar click away.
            view_2d: false,
            viewport_pan: Vec2::ZERO,
            viewport_zoom: DEFAULT_VIEWPORT_ZOOM,
            // Default orbit: ~22° pitch above the target, looking
            // toward +Z, radius wide enough to frame a 4×4 stone room
            // at the cooker's standard 1024-unit sector size.
            viewport_3d_yaw: 256,
            viewport_3d_pitch: 256,
            viewport_3d_radius: 6144,
            texture_thumbs: HashMap::new(),
            model_import_dialog: ModelImportDialog::default(),
            model_import_retired_textures: Vec::new(),
            dirty: false,
            status: "Editor ready".to_string(),
            pending_playtest_request: None,
        }
    }

    /// Current project document.
    pub fn project(&self) -> &ProjectDocument {
        &self.project
    }

    /// Directory containing this project on disk.
    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// Directory project-relative resource paths resolve against --
    /// always the project's own directory now that every project
    /// owns one. No cwd fallback.
    pub fn project_root(&self) -> &Path {
        &self.project_dir
    }

    /// True when the project has unsaved edits.
    pub const fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Save to `<project_dir>/project.ron`.
    pub fn save(&mut self) -> Result<(), String> {
        let path = self.project_dir.join("project.ron");
        self.project
            .save_to_path(&path)
            .map_err(|error| error.to_string())?;
        self.dirty = false;
        self.status = format!("Saved {}", short_path(&self.project_dir));
        Ok(())
    }

    /// Save only when the project contains unsaved edits.
    pub fn save_if_dirty(&mut self) -> Result<bool, String> {
        if !self.dirty {
            return Ok(false);
        }
        self.save()?;
        Ok(true)
    }

    /// Re-read `<project_dir>/project.ron` from disk, discarding
    /// in-memory edits. Surfaces a load error in the status bar
    /// rather than failing -- the user can still keep editing the
    /// in-memory state.
    pub fn reload(&mut self) {
        let path = self.project_dir.join("project.ron");
        match ProjectDocument::load_from_path(&path) {
            Ok(project) => {
                self.project = project;
                self.selected_node = NodeId::ROOT;
                self.selected_resource = None;
                self.resource_renaming = None;
                self.dirty = false;
                self.status = format!("Reloaded {}", short_path(&self.project_dir));
                self.select_first_room();
            }
            Err(error) => {
                self.status = format!("Reload failed: {error}");
            }
        }
    }

    /// Create `editor/projects/<name>/` by recursive-copy of the
    /// default project, then switch this workspace to it.
    ///
    /// Validates `name`: non-empty, no path separators, no `..`,
    /// no leading `.`, target directory must not already exist.
    /// On success the workspace points at the new project; on
    /// failure the workspace is unchanged.
    pub fn create_and_open_project(&mut self, name: &str) -> Result<(), String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("Project name cannot be empty".to_string());
        }
        if trimmed.contains('/')
            || trimmed.contains('\\')
            || trimmed.starts_with('.')
            || trimmed.contains("..")
        {
            return Err(
                "Project name cannot contain path separators, `..`, or leading `.`".to_string(),
            );
        }
        let target = psxed_project::projects_dir().join(trimmed);
        if target.exists() {
            return Err(format!("{} already exists", short_path(&target)));
        }
        copy_dir_recursive(&psxed_project::default_project_dir(), &target)
            .map_err(|error| format!("copy default project: {error}"))?;
        let opened = Self::open_directory(&target)?;
        *self = opened;
        self.status = format!("Created {}", short_path(&self.project_dir));
        Ok(())
    }

    /// Select the first Room in the active scene if one exists.
    /// Default state is `selected_node = ROOT`, which leaves the
    /// inspector empty and gates the paint tools -- selecting a
    /// concrete Room straight after construction or load makes the
    /// editor immediately useful for the common case (one Room per
    /// project).
    fn select_first_room(&mut self) {
        if let Some(room_id) = self
            .project
            .active_scene()
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))
            .map(|node| node.id)
        {
            self.selected_node = room_id;
            self.frame_3d_on_room(room_id);
        }
    }

    /// Position the orbit camera so `room_id`'s grid fills the
    /// viewport at startup. Pulls back to ~1.6× the room diagonal
    /// in world units, which lands a 3/4 view that shows all four
    /// walls plus the floor without the corners clipping.
    fn frame_3d_on_room(&mut self, room_id: NodeId) {
        let scene = self.project.active_scene();
        let Some(room) = scene.node(room_id) else {
            return;
        };
        let NodeKind::Room { grid } = &room.kind else {
            return;
        };
        let max_side = grid.width.max(grid.depth) as f32;
        let world_extent = max_side * grid.sector_size as f32;
        // 1.6× max-side fits diagonally with a little headroom; the
        // FOV is fixed (focal=320, screen=320×240 → ~53° H-FOV).
        let radius = (world_extent * 1.6).clamp(512.0, 262_144.0);
        self.viewport_3d_radius = radius as i32;
        // Default 3/4 view: yaw 8/64 turn (45° off the +Z axis),
        // pitch 4/64 (~22° looking down). Mirrors the showcase
        // demos' first-frame angle.
        self.viewport_3d_yaw = 256;
        self.viewport_3d_pitch = 256;
    }

    /// Cook every Room in the active scene to a per-Room `.psxw`
    /// blob under `<project_dir>/cooked/`.
    ///
    /// Returns a one-line summary on success. Fails when the project
    /// has not yet been saved (no anchor for `<project_dir>`), or
    /// when any room's grid cooker rejects its inputs (see
    /// `WorldGridCookError`).
    pub fn cook_world_to_disk(&mut self) -> Result<String, String> {
        let cooked_dir = self.project_dir.join("cooked");
        std::fs::create_dir_all(&cooked_dir)
            .map_err(|error| format!("mkdir {}: {error}", cooked_dir.display()))?;

        let scene = self.project.active_scene();
        let rooms: Vec<(NodeId, String)> = scene
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind, NodeKind::Room { .. }))
            .map(|node| (node.id, node.name.clone()))
            .collect();
        if rooms.is_empty() {
            return Err("No Room nodes in the active scene".to_string());
        }

        let mut total_bytes = 0usize;
        let mut written = 0usize;
        for (room_id, room_name) in &rooms {
            let scene = self.project.active_scene();
            let Some(node) = scene.node(*room_id) else {
                continue;
            };
            let NodeKind::Room { grid } = &node.kind else {
                continue;
            };
            let bytes = psxed_project::world_cook::encode_world_grid_psxw(&self.project, grid)
                .map_err(|error| format!("cook \"{room_name}\": {error}"))?;
            let filename = sanitise_room_filename(room_name);
            let path = cooked_dir.join(format!("{filename}.psxw"));
            std::fs::write(&path, &bytes)
                .map_err(|error| format!("write {}: {error}", path.display()))?;
            total_bytes += bytes.len();
            written += 1;
        }

        Ok(format!(
            "Cooked {} room{} ({} KiB) into {}",
            written,
            if written == 1 { "" } else { "s" },
            total_bytes / 1024,
            cooked_dir.display(),
        ))
    }

    /// Cook the active project into the playtest example's
    /// `generated/` directory. Validates the scene tree, cooks
    /// every populated Room into `rooms/room_NNN.psxw`, and
    /// writes a fresh `level_manifest.rs`. Returns a status
    /// string suitable for `self.status`. The "& Play" half is
    /// up to the caller -- the editor doesn't spawn child
    /// processes from this path; instead the status string
    /// hands back the exact command to run.
    #[allow(clippy::too_many_arguments)]
    pub fn cook_playtest_to_disk(&self) -> Result<String, String> {
        let dir = psxed_project::playtest::default_generated_dir();
        // Re-run build_package up front to grab the asset/material
        // counts for the status string. cook_to_dir does this
        // internally too; the duplicate cost is negligible
        // compared to the IO it saves a step later.
        let (package, _report) =
            psxed_project::playtest::build_package(&self.project, &self.project_dir);
        let summary = package.as_ref().map(|p| PackageSummary {
            rooms: p.rooms.len(),
            assets: p.assets.len(),
            textures: p.texture_asset_count(),
            materials: p.materials.len(),
            models: p.models.len(),
            characters: p.characters.len(),
            lights: p.lights.len(),
            entities: p.entities.len(),
            player_character: p
                .player_controller
                .and_then(|pc| p.characters.get(pc.character as usize))
                .and_then(|c| {
                    self.project
                        .resource(c.source_resource)
                        .map(|r| r.name.clone())
                }),
        });

        let report = psxed_project::playtest::cook_to_dir(&self.project, &self.project_dir, &dir)
            .map_err(|e| format!("write playtest output: {e}"))?;
        if !report.is_ok() {
            return Err(format!(
                "playtest validation failed: {}",
                report.errors.join("; ")
            ));
        }
        let warning_suffix = if report.warnings.is_empty() {
            String::new()
        } else {
            format!(
                " ({} warning{})",
                report.warnings.len(),
                if report.warnings.len() == 1 { "" } else { "s" }
            )
        };
        let counts = summary
            .as_ref()
            .map(|s| {
                let player_blurb = match s.player_character.as_deref() {
                    Some(name) => format!(", player: {name}"),
                    None => ", no player".to_string(),
                };
                format!(
                    " — {} room{}, {} model{}, {} character{}{}, {} light{}, {} asset{}, {} texture{}, {} material{}, {} entit{}",
                    s.rooms,
                    if s.rooms == 1 { "" } else { "s" },
                    s.models,
                    if s.models == 1 { "" } else { "s" },
                    s.characters,
                    if s.characters == 1 { "" } else { "s" },
                    player_blurb,
                    s.lights,
                    if s.lights == 1 { "" } else { "s" },
                    s.assets,
                    if s.assets == 1 { "" } else { "s" },
                    s.textures,
                    if s.textures == 1 { "" } else { "s" },
                    s.materials,
                    if s.materials == 1 { "" } else { "s" },
                    s.entities,
                    if s.entities == 1 { "y" } else { "ies" },
                )
            })
            .unwrap_or_default();
        Ok(format!(
            "Playtest cooked → {}{}{}",
            dir.display(),
            counts,
            warning_suffix,
        ))
    }

    /// Drain the one-shot embedded play request emitted by this
    /// frame's UI. The frontend calls this after drawing the editor
    /// and performs the actual cook/build/load/stop work.
    pub fn take_playtest_request(&mut self) -> Option<EditorPlaytestRequest> {
        self.pending_playtest_request.take()
    }

    /// Let the frontend surface embedded play status in the editor's
    /// existing status strip.
    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    /// Draw the full editor workspace.
    ///
    /// `viewport_3d` describes what texture the central 3D viewport
    /// should paint this frame: editable authoring preview or live
    /// embedded playtest output.
    pub fn draw(
        &mut self,
        ctx: &egui::Context,
        viewport_3d: EditorViewport3dPresentation,
        playtest_status: EditorPlaytestStatus,
    ) {
        apply_studio_visuals(ctx);
        self.model_import_retired_textures
            .retain_mut(|(frames, _)| {
                if *frames == 0 {
                    false
                } else {
                    *frames -= 1;
                    true
                }
            });
        let playtest_captured = matches!(
            playtest_status,
            EditorPlaytestStatus::Running {
                input_captured: true
            }
        );
        if !playtest_captured {
            self.handle_global_shortcuts(ctx);
        }
        self.draw_menu_bar(ctx);
        self.draw_action_bar(ctx, playtest_status);
        self.draw_left_dock(ctx);
        self.draw_inspector(ctx);
        self.draw_content_browser(ctx);
        self.draw_viewport(ctx, viewport_3d, playtest_status);
        self.draw_new_project_dialog(ctx);
        self.draw_model_import_dialog(ctx);
    }

    /// Modal for the File → New Project flow. Pops over the editor
    /// when `new_project_dialog_open` is true; submit calls
    /// [`Self::create_and_open_project`] and re-targets the
    /// workspace at the new directory.
    fn draw_new_project_dialog(&mut self, ctx: &egui::Context) {
        if !self.new_project_dialog_open {
            return;
        }
        let mut close = false;
        let mut submit = false;
        egui::Window::new(icons::label(icons::FILE_PLUS, "New Project"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                ui.label("Project name");
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.new_project_name)
                        .hint_text("e.g. test-room"),
                );
                ui.label(
                    RichText::new(format!(
                        "→ editor/projects/{}/",
                        if self.new_project_name.trim().is_empty() {
                            "<name>"
                        } else {
                            self.new_project_name.trim()
                        }
                    ))
                    .color(STUDIO_TEXT_WEAK)
                    .small(),
                );
                if let Some(error) = &self.new_project_error {
                    ui.label(RichText::new(error).color(Color32::from_rgb(0xE0, 0x60, 0x60)));
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                    if ui.button("Create").clicked() {
                        submit = true;
                    }
                });
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    submit = true;
                }
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    close = true;
                }
            });
        if submit {
            let name = self.new_project_name.clone();
            match self.create_and_open_project(&name) {
                Ok(()) => {
                    self.new_project_dialog_open = false;
                    self.new_project_name.clear();
                    self.new_project_error = None;
                }
                Err(error) => {
                    self.new_project_error = Some(error);
                }
            }
        }
        if close {
            self.new_project_dialog_open = false;
            self.new_project_error = None;
        }
    }

    fn draw_model_import_dialog(&mut self, ctx: &egui::Context) {
        if !self.model_import_dialog.open {
            return;
        }

        enum Action {
            BrowseSource,
            Preview,
            Import,
            Close,
        }

        let mut action: Option<Action> = None;
        let dialog = &mut self.model_import_dialog;
        egui::Window::new(icons::label(icons::FILE_PLUS, "Import Model"))
            .collapsible(false)
            .resizable(true)
            .default_width(1160.0)
            .default_height(820.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_size(Vec2::new(980.0, 620.0));
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(300.0);
                        ui.label(RichText::new("Source").strong());
                        ui.label(RichText::new("GLB/glTF path").color(STUDIO_TEXT_WEAK).small());
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut dialog.source_path)
                                    .desired_width(210.0),
                            );
                            if ui
                                .button(icons::label(icons::FOLDER, "Browse"))
                                .on_hover_text("Choose a .glb or .gltf file")
                                .clicked()
                            {
                                action = Some(Action::BrowseSource);
                            }
                        });
                        ui.label(RichText::new("Resource name").color(STUDIO_TEXT_WEAK).small());
                        ui.text_edit_singleline(&mut dialog.output_name);
                        if dialog.output_name.trim().is_empty() {
                            ui.label(
                                RichText::new("Uses the source file name when blank.")
                                    .color(STUDIO_TEXT_WEAK)
                                    .small(),
                            );
                        }

                        ui.separator();
                        ui.label(RichText::new("Bake Settings").strong());
                        ui.horizontal(|ui| {
                            ui.label("Atlas");
                            ui.add(
                                egui::DragValue::new(&mut dialog.texture_width)
                                    .range(16..=512)
                                    .speed(8.0),
                            );
                            ui.label("×");
                            ui.add(
                                egui::DragValue::new(&mut dialog.texture_height)
                                    .range(16..=512)
                                    .speed(8.0),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Anim Hz");
                            ui.add(
                                egui::DragValue::new(&mut dialog.animation_fps)
                                    .range(1..=60)
                                    .speed(1.0),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("World height");
                            ui.add(
                                egui::DragValue::new(&mut dialog.world_height)
                                    .range(128..=8192)
                                    .speed(16.0),
                            );
                        });
                        ui.checkbox(
                            &mut dialog.normalize_root_translation,
                            "Center animation root",
                        )
                        .on_hover_text(
                            "Restores root-joint translation to bind pose while baking clips. Useful for Meshy exports with noisy Hips location keys.",
                        );
                        ui.label(
                            RichText::new("Texture depth: 8bpp indexed")
                                .color(STUDIO_TEXT_WEAK)
                                .small(),
                        );

                        ui.separator();
                        ui.label(RichText::new("Preview").strong());
                        ui.horizontal(|ui| {
                            ui.label("Yaw");
                            ui.add(
                                egui::DragValue::new(&mut dialog.preview_yaw_q12)
                                    .range(0..=4095)
                                    .speed(8.0),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Pitch");
                            ui.add(
                                egui::DragValue::new(&mut dialog.preview_pitch_q12)
                                    .range(64..=960)
                                    .speed(6.0),
                            );
                        });
                        ui.add(
                            egui::Slider::new(&mut dialog.preview_radius, 640..=4096)
                                .text("Distance"),
                        );
                        ui.checkbox(&mut dialog.show_animation_root, "Root marker");
                        if ui.button(icons::label(icons::ROTATE_CCW, "Reset View")).clicked() {
                            dialog.preview_yaw_q12 = 340;
                            dialog.preview_pitch_q12 = 350;
                            dialog.preview_radius = 1536;
                            dialog.show_animation_root = true;
                        }

                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button(icons::label(icons::SCAN, "Cook Preview")).clicked() {
                                action = Some(Action::Preview);
                            }
                            if ui.button(icons::label(icons::PLUS, "Import")).clicked() {
                                action = Some(Action::Import);
                            }
                            if ui.button("Cancel").clicked() {
                                action = Some(Action::Close);
                            }
                        });

                        if let Some(status) = &dialog.status {
                            ui.add_space(6.0);
                            match status {
                                ModelImportStatus::Info(text) => {
                                    ui.label(RichText::new(text).color(STUDIO_TEXT_WEAK).small());
                                }
                                ModelImportStatus::Error(text) => {
                                    ui.label(
                                        RichText::new(text)
                                            .color(Color32::from_rgb(220, 120, 100))
                                            .small(),
                                    );
                                }
                            }
                        }
                    });

                    ui.separator();
                    ui.vertical(|ui| {
                        ui.set_min_width(700.0);
                        if let Some(preview) = &mut dialog.preview {
                            draw_model_import_preview(
                                ui,
                                preview,
                                &mut dialog.selected_clip,
                                &mut dialog.preview_yaw_q12,
                                &mut dialog.preview_pitch_q12,
                                &mut dialog.preview_radius,
                                dialog.show_animation_root,
                            );
                        } else {
                            ui.vertical_centered(|ui| {
                                ui.add_space(160.0);
                                ui.label(RichText::new("Cook a preview").strong());
                                ui.label(
                                    RichText::new(
                                        "The preview shows the cooked model, atlas, clips, and root-motion stats before files are written.",
                                    )
                                    .color(STUDIO_TEXT_WEAK)
                                    .small(),
                                );
                            });
                        }
                    });
                });

                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    action = Some(Action::Close);
                }
            });

        match action {
            Some(Action::BrowseSource) => {
                if self.choose_model_import_source() {
                    self.run_model_import_preview(ctx);
                }
            }
            Some(Action::Preview) => self.run_model_import_preview(ctx),
            Some(Action::Import) => self.commit_model_import(),
            Some(Action::Close) => self.close_model_import_dialog(),
            None => {}
        }
    }

    fn close_model_import_dialog(&mut self) {
        self.model_import_dialog.open = false;
        self.retire_model_import_preview();
    }

    fn retire_model_import_preview(&mut self) {
        if let Some(preview) = self.model_import_dialog.preview.take() {
            if let Some((handle, _)) = preview.atlas {
                self.model_import_retired_textures.push((2, handle));
            }
            if let Some(handle) = preview.animated_texture {
                self.model_import_retired_textures.push((2, handle));
            }
        }
    }

    fn set_model_import_preview(&mut self, preview: ModelImportPreview) {
        self.retire_model_import_preview();
        self.model_import_dialog.preview = Some(preview);
    }

    fn run_model_import_preview(&mut self, ctx: &egui::Context) {
        let source = self.model_import_source_path();
        if source.as_os_str().is_empty() {
            self.model_import_dialog.status = Some(ModelImportStatus::Error(
                "Choose a GLB/glTF source path.".to_string(),
            ));
            return;
        }
        let config = self.model_import_config();
        let world_height = config.world_height as i32;
        match psxed_project::model_import::preview_glb_model(&source, config) {
            Ok(package) => {
                let decoded_atlas = package
                    .texture
                    .as_ref()
                    .and_then(|bytes| decode_psxt_thumbnail(bytes));
                let atlas_image = decoded_atlas.as_ref().map(|(image, _)| image.clone());
                let atlas = decoded_atlas.map(|(image, stats)| {
                    let handle = ctx.load_texture(
                        "model-import-atlas-preview",
                        image,
                        egui::TextureOptions::NEAREST,
                    );
                    (handle, stats)
                });
                let clips = package
                    .clips
                    .iter()
                    .map(|clip| ModelImportClipPreview {
                        name: clip
                            .source_name
                            .as_deref()
                            .unwrap_or(&clip.sanitized_name)
                            .to_string(),
                        frames: clip.frames,
                        byte_len: clip.bytes.len(),
                        bytes: clip.bytes.clone(),
                        root_motion: root_motion_stats(&clip.bytes, 0),
                    })
                    .collect();
                let clip_count = package.clips.len();
                self.set_model_import_preview(ModelImportPreview {
                    model_bytes: package.model,
                    report: package.report,
                    atlas,
                    atlas_image,
                    animated_texture: None,
                    world_height,
                    clips,
                });
                self.model_import_dialog.selected_clip = self
                    .model_import_dialog
                    .selected_clip
                    .min(clip_count.saturating_sub(1));
                self.model_import_dialog.status = Some(ModelImportStatus::Info(format!(
                    "Preview cooked: {clip_count} clip(s){}",
                    if self.model_import_dialog.normalize_root_translation {
                        ", root centered"
                    } else {
                        ""
                    }
                )));
            }
            Err(error) => {
                self.retire_model_import_preview();
                self.model_import_dialog.status =
                    Some(ModelImportStatus::Error(format!("Preview failed: {error}")));
            }
        }
    }

    fn choose_model_import_source(&mut self) -> bool {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Choose GLB/glTF model")
            .add_filter("glTF model", &["glb", "gltf"]);
        let current = self.model_import_source_path();
        if let Some(dir) = Self::path_parent_or_self(&current) {
            dialog = dialog.set_directory(dir);
        } else if self.project_dir.is_dir() {
            dialog = dialog.set_directory(&self.project_dir);
        }

        let Some(path) = dialog.pick_file() else {
            return false;
        };

        self.model_import_dialog.source_path = Self::display_project_path(&path, &self.project_dir);
        if self.model_import_dialog.output_name.trim().is_empty() {
            if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                self.model_import_dialog.output_name = stem.to_string();
            }
        }
        self.retire_model_import_preview();
        self.model_import_dialog.selected_clip = 0;
        self.model_import_dialog.status = Some(ModelImportStatus::Info(format!(
            "Selected source: {}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model")
        )));
        true
    }

    fn commit_model_import(&mut self) {
        let source = self.model_import_source_path();
        if source.as_os_str().is_empty() {
            self.model_import_dialog.status = Some(ModelImportStatus::Error(
                "Choose a GLB/glTF source path.".to_string(),
            ));
            return;
        }
        let output_name = self.model_import_output_name(&source);
        let config = self.model_import_config();
        match psxed_project::model_import::import_glb_model(
            &mut self.project,
            &source,
            &output_name,
            &self.project_dir,
            config,
        ) {
            Ok(id) => {
                self.selected_resource = Some(id);
                self.selected_node = NodeId::ROOT;
                self.selected_primitive = None;
                self.close_model_import_dialog();
                self.status = format!("Imported model {output_name}");
                self.mark_dirty();
            }
            Err(error) => {
                self.model_import_dialog.status =
                    Some(ModelImportStatus::Error(format!("Import failed: {error}")));
            }
        }
    }

    fn model_import_config(&self) -> psxed_project::model_import::RigidModelConfig {
        psxed_project::model_import::RigidModelConfig {
            texture_width: self.model_import_dialog.texture_width.clamp(16, 512) as u16,
            texture_height: self.model_import_dialog.texture_height.clamp(16, 512) as u16,
            texture_depth: psxed_project::model_import::TextureDepth::Bit8,
            animation_fps: self.model_import_dialog.animation_fps.clamp(1, 60) as u16,
            world_height: self.model_import_dialog.world_height.clamp(128, 8192) as u16,
            normalize_root_translation: self.model_import_dialog.normalize_root_translation,
        }
    }

    fn model_import_source_path(&self) -> PathBuf {
        let trimmed = self.model_import_dialog.source_path.trim();
        if trimmed.is_empty() {
            return PathBuf::new();
        }
        let path = Path::new(trimmed);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_dir.join(path)
        }
    }

    fn model_import_output_name(&self, source: &Path) -> String {
        let trimmed = self.model_import_dialog.output_name.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
        source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| "model".to_string())
    }

    fn path_parent_or_self(path: &Path) -> Option<PathBuf> {
        if path.as_os_str().is_empty() {
            return None;
        }
        if path.is_dir() {
            Some(path.to_path_buf())
        } else {
            path.parent().map(Path::to_path_buf)
        }
    }

    fn display_project_path(path: &Path, project_dir: &Path) -> String {
        if let Ok(relative) = path.strip_prefix(project_dir) {
            if !relative.as_os_str().is_empty() {
                return relative.to_string_lossy().into_owned();
            }
        }
        path.to_string_lossy().into_owned()
    }

    /// 3D viewport body -- paints the HwRenderer texture into the
    /// central area's working space and turns pointer input into
    /// orbit-camera updates. Called from `draw_viewport` when the
    /// user has toggled the 2D / 3D switch on the toolbar to 3D.
    fn draw_viewport_3d_body(
        &mut self,
        ui: &mut egui::Ui,
        viewport_3d: EditorViewport3dPresentation,
    ) {
        ui.horizontal(|ui| {
            ui.label(icons::text(icons::BOX, 14.0).color(STUDIO_ACCENT));
            ui.label(RichText::new("3D Preview").strong().color(STUDIO_TEXT));
            ui.separator();
            ui.weak(format!(
                "yaw {} pitch {} r {}",
                self.viewport_3d_yaw, self.viewport_3d_pitch, self.viewport_3d_radius
            ));
        });
        ui.separator();
        let avail = ui.available_size();
        let target_aspect = 320.0_f32 / 240.0;
        let (w, h) = if avail.x / avail.y > target_aspect {
            (avail.y * target_aspect, avail.y)
        } else {
            (avail.x, avail.x / target_aspect)
        };
        let (rect, response) =
            ui.allocate_exact_size(egui::Vec2::new(w, h), egui::Sense::click_and_drag());
        let dnd_active = egui::DragAndDrop::has_any_payload(ui.ctx());
        let resource_drop_hovered = response.dnd_hover_payload::<ResourceId>().is_some();

        // Sims-style: primary button always belongs to the active
        // tool -- click-and-drag floors / walls / entities into the
        // world. Camera orbit lives on middle / secondary so the
        // user can reframe mid-edit without giving up the tool.
        if response.dragged_by(egui::PointerButton::Middle)
            || response.dragged_by(egui::PointerButton::Secondary)
        {
            // 0.5 px → 1 q12-step keeps the orbit responsive without
            // making a small wrist flick spin the camera multiple
            // turns. Earlier 1.5x felt over-eager; this lands around
            // ~3° per 100 pixels of drag.
            const ORBIT_DRAG_STEP: f32 = 0.5;
            let delta = response.drag_delta();
            self.viewport_3d_yaw = self
                .viewport_3d_yaw
                .wrapping_add((delta.x * ORBIT_DRAG_STEP) as i16 as u16);
            self.viewport_3d_pitch = self
                .viewport_3d_pitch
                .wrapping_add((delta.y * ORBIT_DRAG_STEP) as i16 as u16);
        }

        // Hover tracking: every frame the pointer is over the panel,
        // ray-pick which cell it's on so the renderer can stamp a
        // translucent overlay there. Cleared when the pointer leaves.
        // For PaintWall, ALSO track which edge of the cell the pointer
        // is closest to so the renderer can preview the targeted wall
        // edge before the click.
        let hover_world = response
            .hover_pos()
            .and_then(|pointer| self.pick_3d_world(rect, pointer));
        let hover_room = self.active_room_id().or_else(|| {
            self.project
                .active_scene()
                .nodes()
                .iter()
                .find(|n| matches!(n.kind, NodeKind::Room { .. }))
                .map(|n| n.id)
        });
        let paint_tool = matches!(
            self.active_tool,
            ViewTool::PaintFloor
                | ViewTool::PaintWall
                | ViewTool::PaintCeiling
                | ViewTool::Erase
                | ViewTool::Place
        );
        // Face hover ray-tests every floor / wall / ceiling in the
        // active Room and reports the closest hit. Used by Select
        // for the outline UI, AND by paint tools to anchor their
        // dispatch onto the actual face the user clicked rather
        // than the floor-plane projection (which lies under wall
        // surfaces and gets the wrong cell for back-row clicks).
        let face_hit = response
            .hover_pos()
            .and_then(|pointer| self.pick_face_with_hit(rect, pointer));
        // Hover-track via the unified selection. `pick_primitive`
        // (added below) maps the same face-hit to a finer
        // primitive when the selection mode demands one. For now
        // the hover always carries a `Selection::Face` -- the
        // hover preview ignores edge / vertex modes until the
        // mode-aware pick replaces this line.
        self.hovered_primitive =
            face_hit.map(|(face, hit)| self.pick_primitive_from_hit(face, hit));
        // Paint preview: world-cell coords let the ghost outline
        // appear over cells outside the current grid, exactly
        // where the auto-grow would create them.
        self.paint_target_preview = if paint_tool {
            hover_room
                .and_then(|room| self.compute_paint_target_preview(room, face_hit, hover_world))
        } else {
            None
        };
        let dropped_resource = response
            .dnd_release_payload::<ResourceId>()
            .map(|payload| *payload);
        if let Some(resource_id) = dropped_resource {
            self.drop_resource_3d(resource_id, face_hit, hover_world);
        }

        // Primary click / drag: ray-pick the cell under the cursor
        // and dispatch to the active tool. Click starts a fresh
        // drag; drag fires every frame the pointer moves; per-cell
        // dedupe keeps walls / placements from stacking when the
        // pointer dwells inside the same cell across frames.
        if !dnd_active {
            if response.drag_started_by(egui::PointerButton::Primary)
                || response.clicked_by(egui::PointerButton::Primary)
            {
                self.last_paint_stamp = None;
            }

            // Hover-track entity bounds in Select mode so the
            // overlay can highlight the bound under the cursor
            // before the user clicks.
            if matches!(self.active_tool, ViewTool::Select) {
                self.hovered_entity_node = response
                    .hover_pos()
                    .and_then(|p| self.pick_entity_bound(rect, p, self.active_room_id()))
                    .map(|hit| hit.node);
            } else {
                self.hovered_entity_node = None;
            }

            // Select-tool drag-translate. Two distinct drag flows:
            //   1. Entity bound under cursor → start `node_drag`,
            //      move the node on its X/Z plane.
            //   2. Otherwise (face / edge / vertex hit, or empty)
            //      fall back to the existing primitive vertical
            //      drag.
            // Pure clicks (press without movement) just promote
            // the hovered target to the selection -- no undo
            // entry, no mutation. The first drag frame that
            // crosses a threshold lazy-pushes undo so a
            // press-and-release doesn't leave a stale snapshot.
            if matches!(self.active_tool, ViewTool::Select) {
                if response.drag_started_by(egui::PointerButton::Primary) {
                    let entity_hit = response
                        .interact_pointer_pos()
                        .and_then(|p| self.pick_entity_bound(rect, p, self.active_room_id()));
                    if let Some(hit) = entity_hit {
                        self.begin_node_drag(hit, rect);
                    } else {
                        self.begin_primitive_drag();
                    }
                }
                if response.dragged_by(egui::PointerButton::Primary) {
                    if self.node_drag.is_some() {
                        if let Some(p) = response.interact_pointer_pos() {
                            self.update_node_drag(rect, p);
                        }
                    } else {
                        let dy = response.drag_delta().y;
                        self.update_primitive_drag(dy);
                    }
                }
                if response.drag_stopped_by(egui::PointerButton::Primary) {
                    if self.node_drag.is_some() {
                        self.end_node_drag();
                    } else {
                        self.end_primitive_drag();
                    }
                }
                if response.clicked_by(egui::PointerButton::Primary) {
                    // Entity click takes priority over face click --
                    // matches the drag flow above.
                    let entity_hit = response
                        .interact_pointer_pos()
                        .and_then(|p| self.pick_entity_bound(rect, p, self.active_room_id()));
                    if let Some(hit) = entity_hit {
                        self.commit_node_selection(hit.node);
                    } else {
                        self.commit_face_selection();
                    }
                }
            } else {
                let primary_active = response.clicked_by(egui::PointerButton::Primary)
                    || response.dragged_by(egui::PointerButton::Primary);
                if primary_active {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let face_hit = self.pick_face_with_hit(rect, pos);
                        let ground = self.pick_3d_world(rect, pos);
                        self.dispatch_paint_3d(face_hit, ground);
                    }
                }
            }
        } else {
            self.hovered_entity_node = None;
        }

        // Scroll = dolly. ±8% of current radius per wheel notch,
        // clamped so the camera can't pass through the target or
        // escape the world entirely.
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll.abs() > f32::EPSILON {
                let factor = if scroll > 0.0 { 0.92 } else { 1.08 };
                self.viewport_3d_radius =
                    ((self.viewport_3d_radius as f32) * factor).clamp(512.0, 262_144.0) as i32;
            }
        }

        egui::Image::new((viewport_3d.texture, rect.size()))
            .uv(viewport_3d.uv)
            .paint_at(ui, rect);
        if resource_drop_hovered {
            let painter = ui.painter_at(rect);
            painter.rect_stroke(
                rect.shrink(2.0),
                2.0,
                Stroke::new(2.0, STUDIO_ACCENT),
                StrokeKind::Inside,
            );
            painter.text(
                rect.center_top() + Vec2::new(0.0, 16.0),
                Align2::CENTER_TOP,
                "Drop resource into scene",
                FontId::proportional(13.0),
                STUDIO_ACCENT,
            );
        }
    }

    /// Play-mode 3D body -- paints the live emulator framebuffer into
    /// the viewport and suppresses all authoring hit-testing.
    fn draw_viewport_3d_play_body(
        &mut self,
        ui: &mut egui::Ui,
        viewport_3d: EditorViewport3dPresentation,
        playtest_status: EditorPlaytestStatus,
    ) {
        let captured = matches!(
            playtest_status,
            EditorPlaytestStatus::Running {
                input_captured: true
            }
        );
        ui.horizontal(|ui| {
            ui.label(icons::text(icons::PLAY, 14.0).color(STUDIO_ACCENT));
            ui.label(RichText::new("Play Mode").strong().color(STUDIO_TEXT));
            ui.separator();
            if captured {
                ui.weak("input captured");
            } else {
                ui.weak("click viewport to capture input");
            }
        });
        ui.separator();

        let avail = ui.available_size();
        let target_aspect = 320.0_f32 / 240.0;
        let (w, h) = if avail.x / avail.y > target_aspect {
            (avail.y * target_aspect, avail.y)
        } else {
            (avail.x, avail.x / target_aspect)
        };
        let (rect, response) = ui.allocate_exact_size(egui::Vec2::new(w, h), egui::Sense::click());
        if response.clicked() {
            self.pending_playtest_request = Some(EditorPlaytestRequest::CaptureInput);
        }

        egui::Image::new((viewport_3d.texture, rect.size()))
            .uv(viewport_3d.uv)
            .paint_at(ui, rect);

        if !captured {
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, Color32::from_black_alpha(112));
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Click to capture game input",
                FontId::proportional(16.0),
                STUDIO_TEXT,
            );
        }
    }

    /// Snapshot of the orbit camera the frontend needs to drive the
    /// editor's HwRenderer this frame.
    pub fn viewport_3d_camera(&self) -> ViewportCameraState {
        ViewportCameraState {
            yaw_q12: self.viewport_3d_yaw,
            pitch_q12: self.viewport_3d_pitch,
            radius: self.viewport_3d_radius,
        }
    }

    /// Currently-selected scene node. The frontend reads this so the
    /// 3D preview can highlight the selected entity.
    pub fn selected_node_id(&self) -> NodeId {
        self.selected_node
    }

    /// Primitive under the 3D pointer when the Select tool is
    /// active -- face / edge / vertex of a floor, wall, or
    /// ceiling on the active Room. Frontend reads this every
    /// frame to draw a light hover outline.
    pub fn hovered_primitive(&self) -> Option<Selection> {
        self.hovered_primitive
    }

    /// Primitive the user clicked with the Select tool. Frontend
    /// draws a bold outline; the inspector reads it to surface
    /// per-primitive editable fields.
    pub fn selected_primitive(&self) -> Option<Selection> {
        self.selected_primitive
    }

    /// Active selection mode (Face / Edge / Vertex). Hotkeys
    /// 1 / 2 / 3 cycle.
    pub fn selection_mode(&self) -> SelectionMode {
        self.selection_mode
    }

    /// What the next paint click would target. Frontend reads
    /// this every frame for paint tools and outlines either a
    /// cell ghost (Floor / Ceiling / Erase / Place) or a wall
    /// ghost (PaintWall) at the world position the click would
    /// commit to.
    pub fn paint_target_preview(&self) -> Option<PaintTargetPreview> {
        self.paint_target_preview
    }

    /// Scene node whose 3D bounding box currently sits under
    /// the pointer (Select tool only). Frontend reads it each
    /// frame so the editor preview can highlight the box and
    /// the click handler can promote it to a selection.
    pub fn hovered_entity_node(&self) -> Option<NodeId> {
        self.hovered_entity_node
    }

    /// Commit the most recent hover-tracked face to `selected_face`.
    /// Called from the click handler when the Select tool is active,
    /// independent of `dispatch_3d_tool`'s ground-plane sector
    /// requirement so wall / ceiling clicks register even when the
    /// ray-on-Y=0 hit lands beyond the room. Also surfaces the
    /// face's material in the resources panel so the user sees
    /// which material is on the picked surface.
    /// Resolve what the next paint click would target. World-cell
    /// coords (which can be negative) let the preview track cells
    /// outside the current grid -- exactly the cases `auto-grow`
    /// would rescue at click time. Mirrors `run_paint_action` /
    /// `ensure_cell_in_grid` so what you preview is what you'll
    /// paint.
    fn compute_paint_target_preview(
        &self,
        room_id: NodeId,
        face_hit: Option<(FaceRef, [f32; 3])>,
        ground_hit: Option<[f32; 2]>,
    ) -> Option<PaintTargetPreview> {
        let grid = self.room_grid_view(room_id)?;
        let is_paint_wall = matches!(self.active_tool, ViewTool::PaintWall);

        // Cursor over an existing wall while PaintWall is active --
        // the click would replace that exact wall, so preview it
        // directly with its array-derived world cell.
        if is_paint_wall {
            if let Some((
                FaceRef {
                    sx,
                    sz,
                    kind: FaceKind::Wall { dir, stack },
                    ..
                },
                _,
            )) = face_hit
            {
                return Some(PaintTargetPreview::Wall {
                    world_cell_x: grid.origin[0] + sx as i32,
                    world_cell_z: grid.origin[1] + sz as i32,
                    dir,
                    stack,
                });
            }
        }

        // Compute the world cell the cursor is over. Use the face
        // hit when present (works for walls / floors / ceilings of
        // existing cells); otherwise fall back to the floor-plane
        // hit, which can land on cells the grid doesn't cover yet.
        let (world_cell_x, world_cell_z, hit_world) = if let Some((face, hit)) = face_hit {
            (
                grid.origin[0] + face.sx as i32,
                grid.origin[1] + face.sz as i32,
                hit,
            )
        } else {
            let editor = ground_hit?;
            let hit = self.editor_world_to_world3(room_id, editor);
            (
                grid.world_x_to_cell(hit[0]),
                grid.world_z_to_cell(hit[2]),
                hit,
            )
        };

        if is_paint_wall {
            // Cell centre in raw world units -- the inferred edge
            // matches the dispatch's `run_paint_action` because
            // both use the same axis convention.
            let s = grid.sector_size as f32;
            let cell_center_x = (world_cell_x as f32 + 0.5) * s;
            let cell_center_z = (world_cell_z as f32 + 0.5) * s;
            let dir =
                edge_from_world_offset(hit_world[0] - cell_center_x, hit_world[2] - cell_center_z);
            // Stack index points just past any existing walls on
            // that edge -- `add_wall` will append there.
            let stack = grid
                .world_cell_to_array(world_cell_x, world_cell_z)
                .and_then(|(sx, sz)| grid.sector(sx, sz))
                .map(|sector| sector.walls.get(dir).len() as u8)
                .unwrap_or(0);
            Some(PaintTargetPreview::Wall {
                world_cell_x,
                world_cell_z,
                dir,
                stack,
            })
        } else {
            Some(PaintTargetPreview::Cell {
                world_cell_x,
                world_cell_z,
            })
        }
    }

    /// Press on a primitive: select it AND arm a drag. The
    /// drag itself doesn't apply any height change yet -- that
    /// happens in `update_primitive_drag` once the pointer
    /// actually moves. A pure click (no movement) flows through
    /// `commit_face_selection` and never touches `primitive_drag`.
    fn begin_primitive_drag(&mut self) {
        let Some(target) = self.hovered_primitive else {
            return;
        };
        // Make the dragged primitive the selection. The user's
        // mental model: "I clicked it, so it's mine to drag."
        // This also drops `selected_resource` so the inspector
        // doesn't fight for the panel.
        self.selected_primitive = Some(target);
        self.selected_node = NodeId::ROOT;
        if let Selection::Face(face) = target {
            self.selected_resource = self.face_material(face);
        } else {
            self.selected_resource = None;
        }

        // Resolve the physical vertices the drag will translate
        // and snapshot their pre-drag Ys. For Face / Edge this
        // is up to 4 / 2 entries -- the universal-coincidence
        // resolver fans each out to all face-corners that share
        // the world point, so the drag itself only walks this
        // small list.
        let Some(grid) = self.room_grid_view(target.room()) else {
            return;
        };
        let vertices = match drag_corner_seeds(target) {
            Some(seeds) => seeds
                .iter()
                .filter_map(|seed| physical_vertex(grid, *seed))
                .collect::<Vec<_>>(),
            None => Vec::new(),
        };
        if vertices.is_empty() {
            return;
        }
        let pre_drag_ys = vertices.iter().map(|v| v.world[1]).collect();
        self.primitive_drag = Some(PrimitiveDrag {
            target,
            vertices,
            pre_drag_ys,
            accumulated_pixel_dy: 0.0,
            snapshot_pushed: false,
        });
    }

    /// One drag-frame: accumulate mouse-Y travel, convert to a
    /// world-Y delta (snap-aware), and apply to every captured
    /// physical vertex.
    fn update_primitive_drag(&mut self, dy_pixels: f32) {
        if dy_pixels.abs() < f32::EPSILON {
            return;
        }
        // Pixels per HEIGHT_QUANTUM step -- drag 8 px to advance
        // one quantum. With HEIGHT_QUANTUM = 32 and a 1024-unit
        // sector, one full sector of height takes 256 pixels of
        // mouse travel -- comfortable for the orbit-cam panel.
        const PIXELS_PER_QUANTUM: f32 = 8.0;
        let Some(drag) = self.primitive_drag.as_mut() else {
            return;
        };
        // Screen +Y is down; world +Y is up -- invert.
        drag.accumulated_pixel_dy -= dy_pixels;
        let total_quanta = (drag.accumulated_pixel_dy / PIXELS_PER_QUANTUM).round() as i32;
        let world_delta = total_quanta * HEIGHT_QUANTUM;
        // No-op until the drag has crossed a quantum.
        if world_delta == 0 && !drag.snapshot_pushed {
            return;
        }
        // Lazy undo snapshot -- captures pre-drag state once,
        // never on a press-without-movement.
        if !drag.snapshot_pushed {
            drag.snapshot_pushed = true;
            self.push_undo();
        }
        // Re-borrow after the push_undo(&mut self) call.
        let Some(drag) = self.primitive_drag.as_ref() else {
            return;
        };
        // Compute every (vertex, new_y) BEFORE entering the
        // mutable scene borrow so the apply step is one tight
        // loop without re-borrowing.
        let updates: Vec<(PhysicalVertex, i32)> = drag
            .vertices
            .iter()
            .zip(drag.pre_drag_ys.iter())
            .map(|(vertex, pre_y)| {
                let new_y = snap_height(pre_y + world_delta);
                (vertex.clone(), new_y)
            })
            .collect();
        let target = drag.target;
        let scene = self.project.active_scene_mut();
        let Some(node) = scene.node_mut(target.room()) else {
            return;
        };
        let NodeKind::Room { grid } = &mut node.kind else {
            return;
        };
        for (vertex, new_y) in updates {
            apply_vertex_height(grid, &vertex, new_y);
        }
        self.mark_dirty();
    }

    /// Drag released. Just clears the stroke; the heights are
    /// already committed.
    fn end_primitive_drag(&mut self) {
        if let Some(drag) = self.primitive_drag.take() {
            if drag.snapshot_pushed {
                self.status = format!(
                    "Translated {} ({} face-corners followed)",
                    describe_selection(drag.target),
                    drag.vertices.iter().map(|v| v.members.len()).sum::<usize>(),
                );
            }
        }
    }

    /// Promote the hovered primitive to a selection. When the
    /// hover is a face, also pre-load `selected_resource` with
    /// its material so the resource panel surfaces it without a
    /// second click. Edge / vertex modes don't pre-load -- the
    /// inspector renders directly from the selection.
    /// Promote `node` to the active selected node, clearing
    /// any grid primitive selection. Mirrors `commit_face_selection`
    /// for entity bounds -- keeps the inspector and scene tree
    /// in sync with the viewport click.
    fn commit_node_selection(&mut self, node: NodeId) {
        self.selected_node = node;
        self.selected_primitive = None;
        self.selected_resource = None;
        let scene = self.project.active_scene();
        if let Some(n) = scene.node(node) {
            self.status = format!("Selected {} '{}'", n.kind.label(), n.name);
        } else {
            self.status = format!("Selected node #{}", node.raw());
        }
    }

    /// Start a node-drag stroke. The pointer landed on
    /// `hit.bounds`; lock the drag plane to the node's current
    /// world Y, snapshot the start translation, and select the
    /// node so subsequent UI updates show the right inspector.
    /// Undo is lazy -- only pushed once the user actually moves.
    fn begin_node_drag(&mut self, hit: EntityBoundHit, _rect: egui::Rect) {
        // Promote selection so the inspector lands on the
        // dragged node.
        self.commit_node_selection(hit.node);
        let scene = self.project.active_scene();
        let Some(node) = scene.node(hit.node) else {
            return;
        };
        // Lock the drag plane to the node's *current* world Y
        // -- that way the cursor stays attached to the bound
        // even if the camera angle changes mid-drag.
        let drag_plane_y = hit.bounds.center[1];
        self.node_drag = Some(NodeDrag {
            node: hit.node,
            start_translation: node.transform.translation,
            start_world_hit: hit.point,
            drag_plane_y,
            snapshot_pushed: false,
            room: hit.bounds.room,
        });
    }

    /// Per-frame node-drag update. Re-cast the current pointer
    /// onto the locked drag plane, compute the world delta from
    /// the start hit, project that delta back into editor-space
    /// (sectors), and write `start + delta` to the node's
    /// translation. Pushes undo lazily on the first
    /// non-zero-delta frame.
    fn update_node_drag(&mut self, rect: egui::Rect, pointer: egui::Pos2) {
        let Some(drag) = self.node_drag.as_ref() else {
            return;
        };
        let plane_y = drag.drag_plane_y;
        let node_id = drag.node;
        let start_translation = drag.start_translation;
        let start_world_hit = drag.start_world_hit;
        let room_id = drag.room;
        let already_snapshotted = drag.snapshot_pushed;

        let Some((origin, dir)) = self.camera_ray_for_pointer(rect, pointer) else {
            return;
        };
        let Some(world_hit) = ray_intersects_horizontal_plane(origin, dir, plane_y) else {
            return;
        };
        // Convert the world-space delta into editor-space
        // (sectors) using the room's `sector_size`. Without an
        // enclosing Room, fall back to a 1:1 conversion so
        // global nodes still drag.
        let scene = self.project.active_scene();
        let sector_size = room_id
            .and_then(|id| scene.node(id))
            .and_then(|n| match &n.kind {
                NodeKind::Room { grid } => Some(grid.sector_size as f32),
                _ => None,
            })
            .unwrap_or(1.0);
        let delta_world = [
            world_hit[0] - start_world_hit[0],
            world_hit[2] - start_world_hit[2],
        ];
        let new_translation = [
            start_translation[0] + delta_world[0] / sector_size,
            start_translation[1],
            start_translation[2] + delta_world[1] / sector_size,
        ];

        // Lazy undo: first non-zero delta pushes one snapshot.
        if !already_snapshotted
            && (delta_world[0].abs() > f32::EPSILON || delta_world[1].abs() > f32::EPSILON)
        {
            self.push_undo();
            if let Some(d) = self.node_drag.as_mut() {
                d.snapshot_pushed = true;
            }
        }

        if let Some(node) = self.project.active_scene_mut().node_mut(node_id) {
            node.transform.translation = new_translation;
            self.dirty = true;
        }
        self.status = format!(
            "Drag node — ({:.2}, {:.2})",
            new_translation[0], new_translation[2]
        );
    }

    /// End the active node-drag stroke. Idempotent if no drag
    /// is in flight.
    fn end_node_drag(&mut self) {
        self.node_drag = None;
    }

    fn commit_face_selection(&mut self) {
        match self.hovered_primitive {
            Some(selection) => {
                self.selected_primitive = Some(selection);
                self.selected_node = NodeId::ROOT;
                if let Selection::Face(face) = selection {
                    self.selected_resource = self.face_material(face);
                } else {
                    self.selected_resource = None;
                }
                self.status = format!("Selected {}", describe_selection(selection));
            }
            None => {
                self.selected_primitive = None;
                self.selected_resource = None;
                self.status = "Cleared selection".to_string();
            }
        }
    }

    /// 3D paint / move click handler. `face_hit` is the ray-test
    /// result (`pick_face_with_hit`) -- preferred because it
    /// reflects the actual face under the cursor. `ground_hit` is
    /// the floor-plane fallback for empty cells where no face
    /// exists to hover.
    fn dispatch_paint_3d(
        &mut self,
        face_hit: Option<(FaceRef, [f32; 3])>,
        ground_hit: Option<[f32; 2]>,
    ) {
        let Some(room_id) = self.active_room_id() else {
            return;
        };
        let paint_tool = matches!(
            self.active_tool,
            ViewTool::PaintFloor
                | ViewTool::PaintWall
                | ViewTool::PaintCeiling
                | ViewTool::Erase
                | ViewTool::Place
        );
        let (cell, hit_world) = match face_hit {
            Some((face, hit)) => ((face.sx, face.sz), hit),
            None => {
                let Some(world) = ground_hit else {
                    return;
                };
                // Click outside the existing grid? Auto-grow on
                // paint/place clicks so the user can extend a room
                // by stamping a floor in empty space -- Sims-style.
                // Move just bails (it never made sense for it to
                // grow the room).
                let cell = if paint_tool {
                    self.ensure_cell_in_grid(room_id, world)
                } else {
                    self.world_to_sector(room_id, world)
                };
                let Some((sx, sz)) = cell else {
                    return;
                };
                let raw_hit = self.editor_world_to_world3(room_id, world);
                ((sx, sz), raw_hit)
            }
        };
        let (sx, sz) = cell;
        let stamp = self.paint_stamp_for(room_id, sx, sz, face_hit, hit_world);
        if self.last_paint_stamp == Some(stamp) {
            return;
        }
        self.last_paint_stamp = Some(stamp);
        self.selected_sector = Some((sx, sz));
        let tool = self.active_tool;
        self.run_paint_action(tool, room_id, sx, sz, face_hit.map(|(f, _)| f), hit_world)
    }

    fn drop_resource_3d(
        &mut self,
        resource_id: ResourceId,
        face_hit: Option<(FaceRef, [f32; 3])>,
        ground_hit: Option<[f32; 2]>,
    ) {
        if let Some((face, hit_world)) = face_hit {
            self.drop_resource_at_room_hit(resource_id, face.room, hit_world, Some(face));
            return;
        }

        let Some(room_id) = self.active_room_id() else {
            self.status = "Drop needs an active Room".to_string();
            return;
        };
        let Some(editor_world) = ground_hit else {
            self.status = "Drop onto the room floor or an existing face".to_string();
            return;
        };
        let Some((_sx, _sz)) = self.ensure_cell_in_grid(room_id, editor_world) else {
            return;
        };
        let hit_world = self.editor_world_to_world3(room_id, editor_world);
        self.drop_resource_at_room_hit(resource_id, room_id, hit_world, None);
    }

    fn drop_resource_2d(&mut self, resource_id: ResourceId, editor_world: [f32; 2]) {
        let Some(room_id) = self.active_room_id() else {
            self.status = "Drop needs an active Room".to_string();
            return;
        };
        let Some((_sx, _sz)) = self.ensure_cell_in_grid(room_id, editor_world) else {
            return;
        };
        let hit_world = self.editor_world_to_world3(room_id, editor_world);
        self.drop_resource_at_room_hit(resource_id, room_id, hit_world, None);
    }

    fn drop_resource_at_room_hit(
        &mut self,
        resource_id: ResourceId,
        room_id: NodeId,
        hit_world: [f32; 3],
        face: Option<FaceRef>,
    ) {
        let Some(resource) = self.project.resource(resource_id).cloned() else {
            self.status = format!("Resource #{} no longer exists", resource_id.raw());
            return;
        };

        match resource.data {
            ResourceData::Model(_) => {
                self.push_undo();
                let node = self.create_model_entity_at_room_hit(
                    room_id,
                    resource_id,
                    &resource.name,
                    hit_world,
                );
                self.selected_node = node;
                self.selected_resource = None;
                self.selected_primitive = None;
                self.status = format!("Created Entity from {}", resource.name);
                self.mark_dirty();
            }
            ResourceData::Character(character) => {
                self.push_undo();
                let node = self.create_character_actor_at_room_hit(
                    room_id,
                    resource_id,
                    &resource.name,
                    character.model,
                    character.idle_clip,
                    character.radius,
                    character.height,
                    hit_world,
                );
                self.selected_node = node;
                self.selected_resource = None;
                self.selected_primitive = None;
                self.status = format!("Created Actor from {}", resource.name);
                self.mark_dirty();
            }
            ResourceData::Material(_) => {
                let Some(face) = face else {
                    self.status = "Drop Material onto an existing face".to_string();
                    return;
                };
                if self.assign_face_material(face, Some(resource_id)) {
                    self.selected_resource = Some(resource_id);
                    self.selected_primitive = Some(Selection::Face(face));
                    self.status = format!("Assigned {} to {}", resource.name, describe_face(face));
                }
            }
            _ => {
                self.status = format!(
                    "Drag Model or Character resources into the scene; {} is not placeable",
                    resource.data.label()
                );
            }
        }
    }

    fn create_model_entity_at_room_hit(
        &mut self,
        room_id: NodeId,
        model_id: ResourceId,
        name: &str,
        hit_world: [f32; 3],
    ) -> NodeId {
        let editor = self
            .room_grid_view(room_id)
            .map(|grid| grid.room_local_to_editor(hit_world))
            .unwrap_or([hit_world[0], hit_world[2]]);
        let scene = self.project.active_scene_mut();
        let entity = scene.add_node(room_id, name.to_string(), NodeKind::Entity);
        if let Some(node) = scene.node_mut(entity) {
            node.transform.translation = [editor[0], 0.0, editor[1]];
        }
        scene.add_node(
            entity,
            "Model Renderer",
            NodeKind::ModelRenderer {
                model: Some(model_id),
                material: None,
            },
        );
        scene.add_node(
            entity,
            "Animator",
            NodeKind::Animator {
                clip: None,
                autoplay: true,
            },
        );
        entity
    }

    #[allow(clippy::too_many_arguments)]
    fn create_character_actor_at_room_hit(
        &mut self,
        room_id: NodeId,
        character_id: ResourceId,
        name: &str,
        model_id: Option<ResourceId>,
        idle_clip: Option<u16>,
        radius: u16,
        height: u16,
        hit_world: [f32; 3],
    ) -> NodeId {
        let editor = self
            .room_grid_view(room_id)
            .map(|grid| grid.room_local_to_editor(hit_world))
            .unwrap_or([hit_world[0], hit_world[2]]);
        let scene = self.project.active_scene_mut();
        let actor = scene.add_node(room_id, name.to_string(), NodeKind::Actor);
        if let Some(node) = scene.node_mut(actor) {
            node.transform.translation = [editor[0], 0.0, editor[1]];
        }
        if let Some(model_id) = model_id {
            scene.add_node(
                actor,
                "Model Renderer",
                NodeKind::ModelRenderer {
                    model: Some(model_id),
                    material: None,
                },
            );
            scene.add_node(
                actor,
                "Animator",
                NodeKind::Animator {
                    clip: idle_clip,
                    autoplay: true,
                },
            );
        }
        scene.add_node(
            actor,
            "Character Controller",
            NodeKind::CharacterController {
                character: Some(character_id),
                player: false,
            },
        );
        scene.add_node(
            actor,
            "Collider",
            NodeKind::Collider {
                shape: ColliderShape::Capsule { radius, height },
                solid: true,
            },
        );
        actor
    }

    /// Build the dedupe key for the next paint dispatch. PaintWall
    /// records the targeted edge + stack so dragging across edges
    /// of the same cell stamps each one (different stamps), but
    /// dwelling on the same edge during drag dedupes (same stamp).
    /// Other tools key on cell + tool only -- drag-restamping a
    /// floor with the same material is a no-op anyway.
    fn paint_stamp_for(
        &self,
        room_id: NodeId,
        sx: u16,
        sz: u16,
        face_hit: Option<(FaceRef, [f32; 3])>,
        hit_world: [f32; 3],
    ) -> PaintStamp {
        let (edge, stack) = if matches!(self.active_tool, ViewTool::PaintWall) {
            match face_hit {
                Some((
                    FaceRef {
                        kind: FaceKind::Wall { dir, stack },
                        ..
                    },
                    _,
                )) => (Some(dir), Some(stack)),
                _ => {
                    let center = self
                        .room_grid_view(room_id)
                        .map(|grid| grid.cell_center_world(sx, sz))
                        .unwrap_or([0.0, 0.0]);
                    let dir =
                        edge_from_world_offset(hit_world[0] - center[0], hit_world[2] - center[1]);
                    (Some(dir), None)
                }
            }
        } else {
            (None, None)
        };
        PaintStamp {
            room: room_id,
            sx,
            sz,
            tool: self.active_tool,
            edge,
            stack,
        }
    }

    /// Resolve the cell `world` lands in, growing the room's grid
    /// in any direction if the click falls beyond the current
    /// footprint. Negative-side growth re-anchors via
    /// `WorldGrid::origin` so existing geometry keeps its world
    /// position. Returns `None` only when the requested cell sits
    /// past the safety cap.
    fn ensure_cell_in_grid(&mut self, room_id: NodeId, world: [f32; 2]) -> Option<(u16, u16)> {
        const AUTO_GROW_LIMIT: u16 = 64;
        if let Some(cell) = self.world_to_sector(room_id, world) {
            return Some(cell);
        }
        let scene = self.project.active_scene();
        let room = scene.node(room_id)?;
        let NodeKind::Room { grid } = &room.kind else {
            return None;
        };
        // `world` here is already in editor-cell units (sector-units,
        // room-centre-relative -- the 2D viewport's native space).
        // Route through the canonical helper so this stays exactly
        // inverse to `world_to_sector`'s lookup.
        let editor_to_world = grid.editor_to_world_cells(world);
        let wcx = editor_to_world[0].floor() as i32;
        let wcz = editor_to_world[1].floor() as i32;
        // Cap the request before mutating so a wild click can't
        // explode the sector vec. The cap covers the post-grow
        // dimensions in either direction.
        let projected_w = grid.width as i32
            + (wcx - grid.origin[0] - grid.width as i32 + 1).max(0)
            + (grid.origin[0] - wcx).max(0);
        let projected_d = grid.depth as i32
            + (wcz - grid.origin[1] - grid.depth as i32 + 1).max(0)
            + (grid.origin[1] - wcz).max(0);
        if projected_w as u32 > AUTO_GROW_LIMIT as u32
            || projected_d as u32 > AUTO_GROW_LIMIT as u32
        {
            self.status =
                format!("Auto-grow capped at {AUTO_GROW_LIMIT} — resize the grid manually");
            return None;
        }
        self.push_undo();
        let scene = self.project.active_scene_mut();
        let node = scene.node_mut(room_id)?;
        let NodeKind::Room { grid } = &mut node.kind else {
            return None;
        };
        let cell = grid.extend_to_include(wcx, wcz);
        self.status = format!(
            "Grew grid to {}×{} (origin {},{})",
            grid.width, grid.depth, grid.origin[0], grid.origin[1]
        );
        self.mark_dirty();
        Some(cell)
    }

    /// World-space sector size of the named Room, or `None` if the
    /// node isn't a Room.
    fn room_sector_size(&self, room_id: NodeId) -> Option<i32> {
        let node = self.project.active_scene().node(room_id)?;
        match &node.kind {
            NodeKind::Room { grid } => Some(grid.sector_size),
            _ => None,
        }
    }

    /// Convert an editor (sector-units, room-centre-relative) hit
    /// position to a raw world `[x, 0, z]` triple. Thin shim over
    /// `WorldGrid::editor_to_room_local` so `pick_3d_world` and this
    /// stay exact inverses by construction.
    fn editor_world_to_world3(&self, room_id: NodeId, editor: [f32; 2]) -> [f32; 3] {
        self.room_grid_view(room_id)
            .map(|grid| grid.editor_to_room_local(editor))
            .unwrap_or([0.0, 0.0, 0.0])
    }

    /// Borrow the named Room's grid for the duration of `&self`,
    /// or `None` if the node isn't a Room. Avoids the
    /// `node.kind` matching dance at every cell-coord call site.
    fn room_grid_view(&self, room_id: NodeId) -> Option<&WorldGrid> {
        let node = self.project.active_scene().node(room_id)?;
        match &node.kind {
            NodeKind::Room { grid } => Some(grid),
            _ => None,
        }
    }

    /// Apply one paint / erase / place action to `(sx, sz)` in
    /// `room_id`. `picked_face` is set when a face was directly
    /// ray-picked (lets us remove a specific wall stack instead of
    /// the whole sector for Erase). `hit_world` is the world-space
    /// click position for tools that need the in-cell offset
    /// (PaintWall picks the edge from `(dx, dz)` against the cell
    /// centre).
    fn run_paint_action(
        &mut self,
        tool: ViewTool,
        room_id: NodeId,
        sx: u16,
        sz: u16,
        picked_face: Option<FaceRef>,
        hit_world: [f32; 3],
    ) {
        let floor_mat = self
            .brush_material
            .or_else(|| self.default_brush_material("floor"));
        let wall_mat = self
            .brush_material
            .or_else(|| self.default_brush_material("brick"))
            .or(floor_mat);
        let sector_size_i = self.room_sector_size(room_id).unwrap_or(1024);
        let sector_size = sector_size_i as f32;
        let cell_center = self
            .room_grid_view(room_id)
            .map(|grid| grid.cell_center_world(sx, sz))
            .unwrap_or([
                (sx as f32 + 0.5) * sector_size,
                (sz as f32 + 0.5) * sector_size,
            ]);

        if matches!(tool, ViewTool::Place) {
            self.push_undo();
            // World → editor (room-centre-relative, sector units)
            // via `WorldGrid::room_local_to_editor`. Single helper,
            // origin-aware, exact inverse of `editor_to_room_local`.
            let editor = self
                .room_grid_view(room_id)
                .map(|grid| grid.room_local_to_editor(hit_world))
                .unwrap_or([0.0, 0.0]);
            let kind = self.place_kind;
            // Player Spawn is exclusive -- demote any existing
            // player spawns so the cooker sees exactly one.
            if matches!(kind, PlaceKind::PlayerSpawn) {
                let scene = self.project.active_scene_mut();
                let stale: Vec<NodeId> = scene
                    .nodes()
                    .iter()
                    .filter(|n| matches!(n.kind, NodeKind::SpawnPoint { player: true, .. }))
                    .map(|n| n.id)
                    .collect();
                for id in stale {
                    if let Some(node) = scene.node_mut(id) {
                        node.kind = NodeKind::SpawnPoint {
                            player: false,
                            character: None,
                        };
                    }
                }
            }
            let (default_name, node_kind): (String, NodeKind) = match kind {
                PlaceKind::PlayerSpawn => (
                    "Player Spawn".to_string(),
                    NodeKind::SpawnPoint {
                        player: true,
                        character: None,
                    },
                ),
                PlaceKind::SpawnMarker => (
                    "Spawn".to_string(),
                    NodeKind::SpawnPoint {
                        player: false,
                        character: None,
                    },
                ),
                PlaceKind::ModelInstance => {
                    // Resolve which Model resource to bind. Order:
                    // (a) user has a Model selected in the resource
                    //     panel -- use it; (b) exactly one Model
                    //     resource exists project-wide -- auto-pick;
                    //     (c) refuse with an actionable status.
                    match self.resolve_place_model_resource() {
                        Ok((model_id, name)) => {
                            let id = self.create_model_entity_at_room_hit(
                                room_id, model_id, &name, hit_world,
                            );
                            self.selected_node = id;
                            self.selected_resource = None;
                            self.status = format!("Placed Model at {sx},{sz}");
                            self.mark_dirty();
                            return;
                        }
                        Err(message) => {
                            self.status = message;
                            return;
                        }
                    }
                }
                PlaceKind::LightMarker => (
                    "Light".to_string(),
                    NodeKind::Light {
                        color: [255, 240, 200],
                        intensity: 1.0,
                        // Sectors. Matches the Add Child default
                        // and the Room-fill preset -- covers a
                        // typical 4×4 sector room.
                        radius: 4.0,
                    },
                ),
            };
            let id = self
                .project
                .active_scene_mut()
                .add_node(room_id, default_name, node_kind);
            if let Some(node) = self.project.active_scene_mut().node_mut(id) {
                node.transform.translation = [editor[0], 0.0, editor[1]];
            }
            self.selected_node = id;
            self.status = format!("Placed {} at {sx},{sz}", kind.label());
            self.mark_dirty();
            return;
        }

        // Snapshot for undo BEFORE mutating. Each non-Place tool
        // shares the same snapshot point.
        self.push_undo();
        let scene = self.project.active_scene_mut();
        let Some(room) = scene.node_mut(room_id) else {
            return;
        };
        let NodeKind::Room { grid } = &mut room.kind else {
            return;
        };
        let status = match tool {
            ViewTool::PaintFloor => {
                grid.set_floor(sx, sz, 0, floor_mat);
                format!("Painted floor at {sx},{sz}")
            }
            ViewTool::PaintCeiling => {
                if let Some(sector) = grid.ensure_sector(sx, sz) {
                    sector.ceiling = Some(GridHorizontalFace::flat(sector_size_i, floor_mat));
                }
                format!("Painted ceiling at {sx},{sz}")
            }
            ViewTool::PaintWall => {
                // When the ray hit an existing wall: REPLACE its
                // material instead of stacking another wall on top
                // of it (the previous behaviour silently appended,
                // which the user spotted as the click-and-nothing-
                // happens-but-walls-pile-up bug). When the ray hit
                // a floor / ceiling / nothing: infer the edge from
                // the click position relative to the cell centre
                // and append a new wall on that edge.
                if let Some(FaceRef {
                    kind: FaceKind::Wall { dir, stack },
                    ..
                }) = picked_face
                {
                    if let Some(sector) = grid.sector_mut(sx, sz) {
                        if let Some(wall) = sector.walls.get_mut(dir).get_mut(stack as usize) {
                            wall.material = wall_mat;
                            format!(
                                "Repainted {} wall #{stack} at {sx},{sz}",
                                direction_label(dir)
                            )
                        } else {
                            format!(
                                "Wall #{stack} on {} edge of {sx},{sz} is gone",
                                direction_label(dir)
                            )
                        }
                    } else {
                        format!("Cell {sx},{sz} no longer has a sector")
                    }
                } else {
                    let dir = edge_from_world_offset(
                        hit_world[0] - cell_center[0],
                        hit_world[2] - cell_center[1],
                    );
                    grid.add_wall(sx, sz, dir, 0, sector_size_i, wall_mat);
                    format!("Added {} wall at {sx},{sz}", direction_label(dir))
                }
            }
            ViewTool::Erase => {
                // Per-face Erase: a wall ray-pick drops just that
                // wall stack entry; floor/ceiling/no-pick clears
                // the whole sector (mirrors the 2D paint pass).
                match picked_face {
                    Some(FaceRef {
                        kind: FaceKind::Wall { dir, stack },
                        ..
                    }) => {
                        if let Some(sector) = grid.sector_mut(sx, sz) {
                            let walls = sector.walls.get_mut(dir);
                            if (stack as usize) < walls.len() {
                                walls.remove(stack as usize);
                            }
                        }
                        format!("Removed wall at {sx},{sz}")
                    }
                    _ => {
                        if let Some(index) = grid.sector_index(sx, sz) {
                            grid.sectors[index] = None;
                        }
                        format!("Erased sector {sx},{sz}")
                    }
                }
            }
            _ => return,
        };
        self.dirty = true;
        self.status = status;
    }

    /// Material id currently applied to `face`, or `None` if the
    /// face is unassigned / its referent went away.
    fn face_material(&self, face: FaceRef) -> Option<ResourceId> {
        let scene = self.project.active_scene();
        let node = scene.node(face.room)?;
        let NodeKind::Room { grid } = &node.kind else {
            return None;
        };
        let sector = grid.sector(face.sx, face.sz)?;
        match face.kind {
            FaceKind::Floor => sector.floor.as_ref().and_then(|f| f.material),
            FaceKind::Ceiling => sector.ceiling.as_ref().and_then(|c| c.material),
            FaceKind::Wall { dir, stack } => sector
                .walls
                .get(dir)
                .get(stack as usize)
                .and_then(|w| w.material),
        }
    }

    /// Reassign `face`'s material in-place. Marks the project
    /// dirty if the field actually moved. Used by the
    /// resource-card click flow when a face is selected, so
    /// picking a different material in the bottom panel
    /// retargets the selected surface (Sims-style).
    fn assign_face_material(&mut self, face: FaceRef, material: Option<ResourceId>) -> bool {
        if self.face_material(face) == material {
            return false;
        }
        self.push_undo();
        let scene = self.project.active_scene_mut();
        let Some(node) = scene.node_mut(face.room) else {
            return false;
        };
        let NodeKind::Room { grid } = &mut node.kind else {
            return false;
        };
        let Some(sector) = grid.ensure_sector(face.sx, face.sz) else {
            return false;
        };
        let updated = match face.kind {
            FaceKind::Floor => sector
                .floor
                .as_mut()
                .map(|f| f.material = material)
                .is_some(),
            FaceKind::Ceiling => sector
                .ceiling
                .as_mut()
                .map(|c| c.material = material)
                .is_some(),
            FaceKind::Wall { dir, stack } => sector
                .walls
                .get_mut(dir)
                .get_mut(stack as usize)
                .map(|w| w.material = material)
                .is_some(),
        };
        if updated {
            self.mark_dirty();
        }
        updated
    }

    /// Build the camera ray in world units for the given pointer
    /// position, or `None` if the pointer's outside the viewport.
    /// Shared by `pick_3d_world` (ray vs. ground plane) and
    /// `pick_face_at` (ray vs. every face triangle in the active
    /// room) so both agree on every axis convention.
    fn camera_ray_for_pointer(
        &self,
        rect: egui::Rect,
        pointer: egui::Pos2,
    ) -> Option<([f32; 3], [f32; 3])> {
        if !rect.contains(pointer) {
            return None;
        }
        let scene = self.project.active_scene();
        let room = scene
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))?;
        let NodeKind::Room { grid } = &room.kind else {
            return None;
        };

        let target_world = psxed_project::spatial::room_preview_center_f32(grid);
        let nx = (pointer.x - rect.center().x) / (rect.width() * 0.5);
        let ny = (pointer.y - rect.center().y) / (rect.height() * 0.5);
        Some(
            self.viewport_3d_camera()
                .ray_for_normalized_panel_point(target_world, nx, ny),
        )
    }

    /// Map a `(face, world-hit)` pair from `pick_face_with_hit`
    /// to a `Selection`, refining to an edge or vertex of the
    /// face when `selection_mode` demands one. Local-UV math
    /// happens here; the picker's heavy lifting (ray vs every
    /// face) was already paid above.
    fn pick_primitive_from_hit(&self, face: FaceRef, hit: [f32; 3]) -> Selection {
        match self.selection_mode {
            SelectionMode::Face => Selection::Face(face),
            SelectionMode::Edge => self
                .face_edge_at_hit(face, hit)
                .map(Selection::Edge)
                .unwrap_or(Selection::Face(face)),
            SelectionMode::Vertex => self
                .face_vertex_at_hit(face, hit)
                .map(Selection::Vertex)
                .unwrap_or(Selection::Face(face)),
        }
    }

    /// Closest edge of `face` to the world-space hit. Computes
    /// distance to each of the four perimeter line segments in
    /// 3D (so sloped floors / non-rectangular walls still pick
    /// the right edge) and returns the smallest.
    fn face_edge_at_hit(&self, face: FaceRef, hit: [f32; 3]) -> Option<EdgeRef> {
        let corners = self.face_world_corners(face)?;
        let edge_idx = closest_edge_idx(&corners, hit);
        let anchor = match face.kind {
            FaceKind::Floor => EdgeAnchor::Floor {
                sx: face.sx,
                sz: face.sz,
                dir: floor_edge_dir(edge_idx),
            },
            FaceKind::Ceiling => EdgeAnchor::Ceiling {
                sx: face.sx,
                sz: face.sz,
                dir: floor_edge_dir(edge_idx),
            },
            FaceKind::Wall { dir, stack } => EdgeAnchor::Wall {
                sx: face.sx,
                sz: face.sz,
                dir,
                stack,
                edge: wall_edge_idx(edge_idx),
            },
        };
        Some(EdgeRef {
            room: face.room,
            anchor,
        })
    }

    /// Closest corner of `face` to the world-space hit. Distance
    /// computed in world space against the four corner points.
    fn face_vertex_at_hit(&self, face: FaceRef, hit: [f32; 3]) -> Option<VertexRef> {
        let corners = self.face_world_corners(face)?;
        let corner_idx = closest_corner_idx(&corners, hit);
        let anchor = match face.kind {
            FaceKind::Floor => VertexAnchor::Floor {
                sx: face.sx,
                sz: face.sz,
                corner: floor_corner_idx(corner_idx),
            },
            FaceKind::Ceiling => VertexAnchor::Ceiling {
                sx: face.sx,
                sz: face.sz,
                corner: floor_corner_idx(corner_idx),
            },
            FaceKind::Wall { dir, stack } => VertexAnchor::Wall {
                sx: face.sx,
                sz: face.sz,
                dir,
                stack,
                corner: wall_corner_idx(corner_idx),
            },
        };
        Some(VertexRef {
            room: face.room,
            anchor,
        })
    }

    /// Four world-space corners of `face` in canonical
    /// perimeter order -- `[NW, NE, SE, SW]` for floors / ceilings,
    /// `[BL, BR, TR, TL]` for walls. Returns `None` if the face
    /// no longer exists (cell out of bounds, geometry missing).
    fn face_world_corners(&self, face: FaceRef) -> Option<[[f32; 3]; 4]> {
        let scene = self.project.active_scene();
        let room = scene.node(face.room)?;
        let NodeKind::Room { grid } = &room.kind else {
            return None;
        };
        if face.sx >= grid.width || face.sz >= grid.depth {
            return None;
        }
        let sector = grid.sector(face.sx, face.sz)?;
        let bounds = grid.cell_bounds_world(face.sx, face.sz);
        match face.kind {
            FaceKind::Floor => sector
                .floor
                .as_ref()
                .map(|f| horizontal_face_world_corners(bounds, f.heights)),
            FaceKind::Ceiling => sector
                .ceiling
                .as_ref()
                .map(|c| horizontal_face_world_corners(bounds, c.heights)),
            FaceKind::Wall { dir, stack } => {
                let wall = sector.walls.get(dir).get(stack as usize)?;
                wall_face_world_corners(bounds, dir, wall.heights)
            }
        }
    }

    /// Walk every floor / ceiling / wall in the active Room and
    /// return the closest face the camera ray hits. Mirrors the
    /// triangle layout `editor_preview` emits so what the user sees
    /// matches what gets picked. `None` when the pointer is off the
    /// panel or no face is along the ray.
    /// Closest floor / wall / ceiling the camera ray intersects,
    /// along with the world-space hit point. Paint dispatch reads
    /// the hit point to infer which edge of a floor cell the user
    /// clicked when the wall paint tool is active.
    fn pick_face_with_hit(
        &self,
        rect: egui::Rect,
        pointer: egui::Pos2,
    ) -> Option<(FaceRef, [f32; 3])> {
        let (origin, dir) = self.camera_ray_for_pointer(rect, pointer)?;
        let scene = self.project.active_scene();
        let room = scene
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))?;
        let NodeKind::Room { grid } = &room.kind else {
            return None;
        };
        let room_id = room.id;
        let mut best: Option<(FaceRef, f32)> = None;
        let mut consider = |face: FaceRef, t: f32| {
            if !t.is_finite() || t <= 0.0 {
                return;
            }
            if best.is_none_or(|(_, bt)| t < bt) {
                best = Some((face, t));
            }
        };

        for sx in 0..grid.width {
            for sz in 0..grid.depth {
                let Some(sector) = grid.sector(sx, sz) else {
                    continue;
                };
                let bounds = grid.cell_bounds_world(sx, sz);

                if let Some(floor) = &sector.floor {
                    let [nw, ne, se, sw] = horizontal_face_world_corners(bounds, floor.heights);
                    let face = FaceRef {
                        room: room_id,
                        sx,
                        sz,
                        kind: FaceKind::Floor,
                    };
                    for (a, b, c, members) in horizontal_triangles(nw, ne, se, sw, floor.split) {
                        if floor.dropped_corner.is_some_and(|d| members.contains(&d)) {
                            continue;
                        }
                        if let Some(t) = ray_triangle(origin, dir, a, b, c) {
                            consider(face, t);
                        }
                    }
                }
                if let Some(ceiling) = &sector.ceiling {
                    let [nw, ne, se, sw] = horizontal_face_world_corners(bounds, ceiling.heights);
                    let face = FaceRef {
                        room: room_id,
                        sx,
                        sz,
                        kind: FaceKind::Ceiling,
                    };
                    for (a, b, c, members) in horizontal_triangles(nw, ne, se, sw, ceiling.split) {
                        if ceiling.dropped_corner.is_some_and(|d| members.contains(&d)) {
                            continue;
                        }
                        if let Some(t) = ray_triangle(origin, dir, a, b, c) {
                            consider(face, t);
                        }
                    }
                }
                for dir_card in GridDirection::CARDINAL {
                    for (stack_idx, wall) in sector.walls.get(dir_card).iter().enumerate() {
                        let Some([bl, br, tr, tl]) =
                            wall_face_world_corners(bounds, dir_card, wall.heights)
                        else {
                            continue;
                        };
                        let face = FaceRef {
                            room: room_id,
                            sx,
                            sz,
                            kind: FaceKind::Wall {
                                dir: dir_card,
                                stack: stack_idx as u8,
                            },
                        };
                        for (a, b, c, members) in
                            wall_triangles(bl, br, tr, tl, wall.dropped_corner)
                        {
                            if wall.dropped_corner.is_some_and(|d| members.contains(&d)) {
                                continue;
                            }
                            if let Some(t) = ray_triangle(origin, dir, a, b, c) {
                                consider(face, t);
                            }
                        }
                    }
                }
            }
        }

        best.map(|(face, t)| {
            let hit = [
                origin[0] + dir[0] * t,
                origin[1] + dir[1] * t,
                origin[2] + dir[2] * t,
            ];
            (face, hit)
        })
    }

    /// Project a pointer position inside the 3D viewport panel onto
    /// the active Room's ground plane and return the editor's
    /// "1 unit = 1 sector" world coordinates the 2D click handler
    /// already speaks. Lets every paint tool (Floor / Wall / Ceiling
    /// / Erase / Place) work identically in 3D and 2D.
    fn pick_3d_world(&self, rect: egui::Rect, pointer: egui::Pos2) -> Option<[f32; 2]> {
        let (origin, dir) = self.camera_ray_for_pointer(rect, pointer)?;
        let scene = self.project.active_scene();
        let room = scene
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))?;
        let NodeKind::Room { grid } = &room.kind else {
            return None;
        };

        // Ray-plane intersection at world Y=0. The orbit cam never
        // sits exactly on Y=0, but we still guard against a near-
        // zero divisor for numerical safety.
        if dir[1].abs() < 1e-5 {
            return None;
        }
        let t = -origin[1] / dir[1];
        if t < 0.0 {
            return None;
        }
        let hit_world = [origin[0] + dir[0] * t, origin[2] + dir[2] * t];

        // 3D ground hit → editor sector-units, room-centre-relative.
        // `WorldGrid::room_local_to_editor` is the canonical inverse of
        // `editor_to_room_local` and accounts for `origin`, so picking
        // stays correct after a negative-side grow.
        let editor = grid.room_local_to_editor([hit_world[0], 0.0, hit_world[1]]);
        Some(editor)
    }

    /// Top-level keyboard shortcut handler. Cleared via `consume_*`
    /// so child widgets never see the same chord.
    fn handle_global_shortcuts(&mut self, ctx: &egui::Context) {
        let undo_chord = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
        let redo_chord = egui::KeyboardShortcut::new(
            egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
            egui::Key::Z,
        );
        let consume_undo = ctx.input_mut(|input| input.consume_shortcut(&undo_chord));
        let consume_redo = ctx.input_mut(|input| input.consume_shortcut(&redo_chord));
        if consume_redo {
            self.do_redo();
        } else if consume_undo {
            self.do_undo();
        }

        // F2 / Delete only fire when no widget owns focus -- so they
        // don't fight TextEdit content while the user is typing.
        let focus_taken = ctx.memory(|m| m.focused().is_some());
        if !focus_taken {
            let f2 = ctx.input_mut(|i| i.key_pressed(egui::Key::F2));
            if f2 && self.selected_node != NodeId::ROOT {
                self.apply_tree_action(TreeAction::BeginRename(self.selected_node));
            }
            let del = ctx.input_mut(|i| {
                i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)
            });
            if del && self.renaming.is_none() {
                if self.selected_primitive.is_some() {
                    self.delete_selected_primitive();
                } else if self.selected_node != NodeId::ROOT {
                    self.apply_tree_action(TreeAction::Delete(self.selected_node));
                }
            }
            let rot = ctx.input_mut(|i| i.key_pressed(egui::Key::R));
            if rot && self.renaming.is_none() {
                self.rotate_selected_yaw_90();
            }
            // Selection-mode hotkeys: 1 / 2 / 3 = Face / Edge /
            // Vertex (Blender convention). The focus guard above
            // already keeps these from firing while a TextEdit
            // owns focus.
            let num1 = ctx.input_mut(|i| i.key_pressed(egui::Key::Num1));
            if num1 {
                self.set_selection_mode(SelectionMode::Face);
            }
            let num2 = ctx.input_mut(|i| i.key_pressed(egui::Key::Num2));
            if num2 {
                self.set_selection_mode(SelectionMode::Edge);
            }
            let num3 = ctx.input_mut(|i| i.key_pressed(egui::Key::Num3));
            if num3 {
                self.set_selection_mode(SelectionMode::Vertex);
            }
        }
    }

    /// Switch the Select tool's primitive mode. Tries to adapt
    /// the existing selection to the new mode (a face → its NW
    /// corner, a vertex → its parent face) so the user doesn't
    /// lose their place. Falls back to clearing if the current
    /// selection has no natural counterpart.
    fn set_selection_mode(&mut self, mode: SelectionMode) {
        if self.selection_mode == mode {
            return;
        }
        self.selection_mode = mode;
        self.selected_primitive = match (self.selected_primitive, mode) {
            (Some(Selection::Face(face)), SelectionMode::Face) => Some(Selection::Face(face)),
            (Some(Selection::Face(face)), SelectionMode::Edge) => {
                Some(Selection::Edge(face_first_edge(face)))
            }
            (Some(Selection::Face(face)), SelectionMode::Vertex) => {
                Some(Selection::Vertex(face_first_vertex(face)))
            }
            (Some(Selection::Edge(edge)), SelectionMode::Face) => {
                edge_owning_face_ref(edge).map(Selection::Face)
            }
            (Some(Selection::Edge(edge)), SelectionMode::Vertex) => {
                Some(Selection::Vertex(edge_first_vertex(edge)))
            }
            (Some(Selection::Vertex(v)), SelectionMode::Face) => {
                vertex_owning_face_ref(v).map(Selection::Face)
            }
            (Some(Selection::Vertex(v)), SelectionMode::Edge) => {
                Some(Selection::Edge(vertex_first_edge(v)))
            }
            (Some(s), m) if Self::matches_mode(s, m) => Some(s),
            _ => None,
        };
        // Clear the hover too -- its mode is the old one, and
        // the next mouse-move re-pick will repopulate under the
        // new mode anyway.
        self.hovered_primitive = None;
        self.status = format!("Selection mode: {}", mode.label());
    }

    fn matches_mode(selection: Selection, mode: SelectionMode) -> bool {
        matches!(
            (selection, mode),
            (Selection::Face(_), SelectionMode::Face)
                | (Selection::Edge(_), SelectionMode::Edge)
                | (Selection::Vertex(_), SelectionMode::Vertex)
        )
    }

    /// Snap the selected node's Y-rotation up by 90°. No-op on
    /// macro / structural nodes (Root, World, Room, plain
    /// transform-only nodes) since they have no in-world heading.
    /// The `MeshInstance` card and entity markers (spawn / light /
    /// trigger / audio / portal) are all rotatable.
    fn rotate_selected_yaw_90(&mut self) {
        let id = self.selected_node;
        if id == NodeId::ROOT {
            return;
        }
        let scene = self.project.active_scene();
        let Some(node) = scene.node(id) else { return };
        let rotatable = matches!(
            node.kind,
            NodeKind::MeshInstance { .. }
                | NodeKind::SpawnPoint { .. }
                | NodeKind::Light { .. }
                | NodeKind::Trigger { .. }
                | NodeKind::AudioSource { .. }
                | NodeKind::Portal { .. }
        );
        if !rotatable {
            return;
        }
        self.push_undo();
        if let Some(node) = self.project.active_scene_mut().node_mut(id) {
            let next = (node.transform.rotation_degrees[1] + 90.0) % 360.0;
            node.transform.rotation_degrees[1] = next;
            self.status = format!("Rotated {} to {}°", node.name, next as i32);
        }
        self.mark_dirty();
    }

    /// Apply one scene-tree action collected from a row.
    fn apply_tree_action(&mut self, action: TreeAction) {
        match action {
            TreeAction::Select(id) => {
                self.commit_node_selection(id);
                self.renaming = None;
                // No-op when `id` isn't a Room -- keeps the camera
                // put while the user clicks through entity nodes.
                self.frame_3d_on_room(id);
            }
            TreeAction::BeginRename(id) => {
                if let Some(node) = self.project.active_scene().node(id) {
                    let name = node.name.clone();
                    self.commit_node_selection(id);
                    self.renaming = Some((id, name));
                    self.pending_rename_focus = true;
                }
            }
            TreeAction::CommitRename(id, name) => {
                let trimmed = name.trim();
                let final_name = if trimmed.is_empty() {
                    self.project
                        .active_scene()
                        .node(id)
                        .map(|node| node.name.clone())
                        .unwrap_or_default()
                } else {
                    trimmed.to_string()
                };
                let original = self
                    .project
                    .active_scene()
                    .node(id)
                    .map(|node| node.name.clone());
                if original.as_deref() != Some(final_name.as_str()) {
                    self.push_undo();
                    if let Some(node) = self.project.active_scene_mut().node_mut(id) {
                        node.name = final_name.clone();
                    }
                    self.status = format!("Renamed {final_name}");
                    self.mark_dirty();
                }
                self.renaming = None;
            }
            TreeAction::CancelRename => {
                self.renaming = None;
            }
            TreeAction::Delete(id) => {
                self.selected_node = id;
                self.delete_selected();
                self.renaming = None;
            }
            TreeAction::Duplicate(id) => {
                self.selected_node = id;
                self.duplicate_selected();
                self.renaming = None;
            }
            TreeAction::AddChild { parent, kind, name } => {
                self.selected_node = parent;
                self.add_child(kind, name);
            }
            TreeAction::Reparent {
                source,
                target_parent,
                position,
            } => {
                if source == target_parent {
                    return;
                }
                if self
                    .project
                    .active_scene()
                    .is_descendant_of(target_parent, source)
                {
                    self.status = "Cannot reparent: would create a cycle".to_string();
                    return;
                }
                self.push_undo();
                let scene = self.project.active_scene_mut();
                if scene.move_node(source, target_parent, position) {
                    self.selected_node = source;
                    self.status = "Moved node".to_string();
                    self.mark_dirty();
                }
            }
        }
    }

    fn draw_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("psxed_menu_bar")
            .exact_height(35.0)
            .frame(top_bar_frame())
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("New Project…").clicked() {
                            self.open_new_project_dialog();
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Save").clicked() {
                            if let Err(error) = self.save() {
                                self.status = format!("Save failed: {error}");
                            }
                            ui.close_menu();
                        }
                        if ui.button("Reload").clicked() {
                            self.reload();
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new("PSoXide Studio").strong());
                        ui.separator();
                        ui.label(if self.dirty {
                            "Unsaved changes"
                        } else {
                            "Saved"
                        });
                    });
                });
            });
    }

    fn draw_action_bar(&mut self, ctx: &egui::Context, playtest_status: EditorPlaytestStatus) {
        egui::TopBottomPanel::top("psxed_action_bar")
            .exact_height(66.0)
            .frame(top_bar_frame())
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .button(icons::label(icons::FILE_PLUS, "New Project"))
                            .on_hover_text(
                                "Create a new project in editor/projects/<name>/ from the default template.",
                            )
                            .clicked()
                        {
                            self.open_new_project_dialog();
                        }
                        if ui.button(icons::label(icons::SAVE, "Save")).clicked() {
                            if let Err(error) = self.save() {
                                self.status = format!("Save failed: {error}");
                            }
                        }
                        if ui
                            .button(icons::label(icons::ROTATE_CCW, "Reload"))
                            .clicked()
                        {
                            self.reload();
                        }
                        if ui.button(icons::label(icons::FOCUS, "Frame")).clicked() {
                            self.frame_viewport();
                        }
                        if ui.button(icons::label(icons::BLEND, "Cook World")).clicked() {
                            match self.cook_world_to_disk() {
                                Ok(report) => self.status = report,
                                Err(error) => self.status = format!("Cook failed: {error}"),
                            }
                        }
                        let playtest_active = playtest_status.is_active();
                        let play_label = if playtest_active {
                            "Rebuild & Play"
                        } else {
                            "Play"
                        };
                        if ui
                            .button(icons::label(icons::PLAY, play_label))
                            .on_hover_text(
                                "Cook the active scene, build editor-playtest, and run it inside the 3D viewport.",
                            )
                            .clicked()
                        {
                            self.pending_playtest_request = Some(if playtest_active {
                                EditorPlaytestRequest::Rebuild
                            } else {
                                EditorPlaytestRequest::Play
                            });
                        }
                        if playtest_active
                            && ui
                                .button(icons::label(icons::TRASH, "Stop"))
                                .on_hover_text(
                                    "Stop embedded play mode and return the viewport to editing.",
                                )
                                .clicked()
                        {
                            self.pending_playtest_request = Some(EditorPlaytestRequest::Stop);
                        }

                        ui.separator();
                        let project_label = if self.dirty {
                            format!("{} *", self.project.name)
                        } else {
                            self.project.name.clone()
                        };
                        ui.label(project_label);
                    });

                    ui.add_space(1.0);
                    ui.horizontal(|ui| {
                        ui.add_space(6.0);
                        ui.add(
                            egui::Label::new(
                                RichText::new(&self.status).small().color(STUDIO_TEXT_WEAK),
                            )
                            .wrap(),
                        );
                    });
                });
            });
    }

    fn draw_left_dock(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("psxed_left_dock")
            .resizable(true)
            .default_width(280.0)
            .min_width(220.0)
            .max_width(LEFT_DOCK_MAX_WIDTH)
            .frame(dock_frame())
            .show(ctx, |ui| {
                section_frame().show(ui, |ui| self.draw_scene_tree_panel(ui));
                ui.add_space(6.0);
                section_frame().show(ui, |ui| self.draw_filesystem_panel(ui));
            });
    }

    fn draw_scene_tree_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(icons::text(icons::LAYERS, 15.0).color(STUDIO_ACCENT));
            ui.label(RichText::new("Scene").strong().color(STUDIO_TEXT));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(egui::Button::new(icons::text(icons::PLUS, 14.0)))
                    .on_hover_text("Add Node")
                    .clicked()
                {
                    self.add_child(NodeKind::Node3D, "Node3D");
                }
            });
        });
        ui.add(
            egui::TextEdit::singleline(&mut self.scene_filter)
                .hint_text("Filter nodes")
                .desired_width(f32::INFINITY),
        );
        ui.separator();

        let rows = self.project.active_scene().hierarchy_rows();
        let filter = self.scene_filter.to_ascii_lowercase();
        let mut actions: Vec<TreeAction> = Vec::new();
        let selected_node = self.selected_node;
        let renaming = &mut self.renaming;
        let pending_focus = &mut self.pending_rename_focus;
        egui::ScrollArea::vertical()
            .id_salt("psxed_scene_tree")
            .max_height(285.0)
            .show(ui, |ui| {
                for row in &rows {
                    if !filter.is_empty()
                        && !row.name.to_ascii_lowercase().contains(&filter)
                        && !row.kind.to_ascii_lowercase().contains(&filter)
                    {
                        continue;
                    }
                    draw_scene_node_row(
                        ui,
                        row,
                        selected_node == row.id,
                        renaming,
                        pending_focus,
                        &mut actions,
                    );
                }
            });

        for action in actions {
            self.apply_tree_action(action);
        }

        ui.horizontal(|ui| {
            if ui.button(icons::label(icons::COPY, "Duplicate")).clicked() {
                self.duplicate_selected();
            }
            let can_delete = self.selected_node != NodeId::ROOT;
            if ui
                .add_enabled(
                    can_delete,
                    egui::Button::new(icons::label(icons::TRASH, "Delete")),
                )
                .clicked()
            {
                self.delete_selected();
            }
        });

        ui.separator();
        draw_scene_group(
            ui,
            icons::GRID,
            "Rooms",
            count_nodes(&self.project, |kind| matches!(kind, NodeKind::Room { .. })),
        );
        draw_scene_group(
            ui,
            icons::SUN,
            "Lights",
            count_nodes(&self.project, |kind| matches!(kind, NodeKind::Light { .. })),
        );
        draw_scene_group(
            ui,
            icons::MAP_PIN,
            "Spawns",
            count_nodes(&self.project, |kind| {
                matches!(kind, NodeKind::SpawnPoint { .. })
            }),
        );
        draw_scene_group(
            ui,
            icons::BOX,
            "Meshes",
            count_nodes(&self.project, |kind| {
                matches!(kind, NodeKind::MeshInstance { .. })
            }),
        );
    }

    fn draw_filesystem_panel(&mut self, ui: &mut egui::Ui) {
        panel_heading(ui, icons::FOLDER, "FileSystem");
        ui.separator();
        let rows = project_filesystem_rows(&self.project);
        let mut clicked_resource = None;
        egui::ScrollArea::vertical()
            .id_salt("psxed_filesystem")
            .max_height(190.0)
            .show(ui, |ui| {
                let filter = self.file_filter.to_ascii_lowercase();
                for row in &rows {
                    if draw_project_file_row(ui, row, self.selected_resource, &filter) {
                        clicked_resource = row.resource;
                    }
                }
            });
        if let Some(id) = clicked_resource {
            self.selected_resource = Some(id);
            self.selected_node = NodeId::ROOT;
            if let Some(name) = self.project.resource_name(id) {
                self.status = format!("Selected {name}");
            }
        }
        ui.add(
            egui::TextEdit::singleline(&mut self.file_filter)
                .hint_text("Filter files")
                .desired_width(f32::INFINITY),
        );
    }

    fn draw_inspector(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("psxed_inspector")
            .resizable(true)
            .default_width(320.0)
            .min_width(240.0)
            .frame(dock_frame())
            .show(ctx, |ui| {
                panel_heading(ui, icons::SCAN, "Inspector");
                ui.separator();

                // Selection priority: primitive (Select tool's
                // product -- face, edge, or vertex) → resource
                // (clicked in the bottom panel) → node (scene
                // tree row). The primitive branch wins because
                // it's the active edit target during paint and
                // height-edit workflows.
                if let Some(selection) = self.selected_primitive {
                    match selection {
                        Selection::Face(face) => self.draw_face_inspector(ui, face),
                        Selection::Edge(edge) => self.draw_edge_inspector(ui, edge),
                        Selection::Vertex(vertex) => self.draw_vertex_inspector(ui, vertex),
                    }
                    return;
                }

                if let Some(resource_id) = self.selected_resource {
                    self.draw_resource_inspector(ui, resource_id);
                    return;
                }

                let material_options = self.project.material_options();
                let room_options = collect_room_options(&self.project);
                let model_options = collect_model_options(&self.project);
                let character_options = collect_character_options(&self.project);
                let selected = self.selected_node;
                let active_room = self.active_room_id();
                let selected_sector = self.selected_sector;

                let mut changed = false;
                // Picker `→` jump-to requests bubble up here.
                // Applied after both phases release their borrows.
                let mut nav_target: Option<ResourceId> = None;

                // Phase 1: mutate the selected node (transform + kind props).
                {
                    let scene = self.project.active_scene_mut();
                    let Some(node) = scene.node_mut(selected) else {
                        ui.weak("No node selected");
                        return;
                    };

                    ui.horizontal(|ui| {
                        draw_inline_icon(
                            ui,
                            node_lucide_icon(node.kind.label(), node.id == NodeId::ROOT),
                            node_lucide_color(node.kind.label(), node.id == NodeId::ROOT, true),
                        );
                        ui.strong(format!("{} #{}", node.kind.label(), node.id.raw()));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Name");
                        changed |= ui.text_edit_singleline(&mut node.name).changed();
                    });
                    ui.separator();

                    egui::CollapsingHeader::new(icons::label(icons::MOVE, "Transform"))
                        .default_open(true)
                        .show(ui, |ui| {
                            changed |= transform_editor(
                                ui,
                                "Position",
                                &mut node.transform.translation,
                                1.0,
                            );
                            changed |= transform_editor(
                                ui,
                                "Rotation",
                                &mut node.transform.rotation_degrees,
                                1.0,
                            );
                            changed |=
                                transform_editor(ui, "Scale", &mut node.transform.scale, 0.05);
                        });

                    egui::CollapsingHeader::new(icons::label(icons::CIRCLE_DOT, "Node Properties"))
                        .default_open(true)
                        .show(ui, |ui| {
                            changed |= draw_node_kind_editor(
                                ui,
                                &mut node.kind,
                                &material_options,
                                &room_options,
                                &model_options,
                                &character_options,
                                &mut nav_target,
                            );
                        });
                }

                // Phase 2: per-sector inspector. Owns its own borrow of the
                // project so it can edit the active Room's grid.
                if let Some(room_id) = active_room {
                    // Room budget panel -- same data the cooker
                    // uses, surfaced here so authors notice when
                    // they're approaching the PSX cap before cook
                    // time refuses the room.
                    let budget = self.room_grid_view(room_id).map(|grid| grid.budget());
                    if let Some(budget) = budget {
                        draw_room_budget(ui, budget);
                    }
                    if let Some((sx, sz)) = selected_sector {
                        if draw_sector_inspector(
                            ui,
                            &mut self.project,
                            room_id,
                            sx,
                            sz,
                            &material_options,
                            &mut nav_target,
                        ) {
                            changed = true;
                        }
                    } else {
                        egui::CollapsingHeader::new(icons::label(icons::GRID, "Sector"))
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.weak("Click a sector tile to inspect it.");
                            });
                    }
                }

                // Phase 3: read-only diagnostics that just need name / kind.
                let scene = self.project.active_scene();
                let Some(node) = scene.node(selected) else {
                    if changed {
                        self.mark_dirty();
                    }
                    return;
                };

                if matches!(node.kind, NodeKind::Room { .. }) {
                    if let NodeKind::Room { grid } = &node.kind {
                        let budget = grid.budget();
                        draw_room_budget(ui, budget);
                    }
                }

                egui::CollapsingHeader::new(icons::label(icons::BOX, "Render"))
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Draw Mode");
                            ui.label(node_draw_mode(&node.kind));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Ordering");
                            ui.label("World OT");
                        });
                    });

                egui::CollapsingHeader::new(icons::label(icons::SCAN, "PS1 Details"))
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Texture Format");
                            ui.label("4bpp Indexed");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Transform");
                            ui.label("Fixed point");
                        });
                    });

                egui::CollapsingHeader::new(icons::label(icons::WAYPOINT, "Node"))
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Path");
                            ui.label(format!("/Root/{}", node.name));
                        });
                    });

                if changed {
                    self.mark_dirty();
                }

                // Apply any picker `→` jump-to. Phase 1 / 2 borrows
                // are released by the time the closure body reaches
                // here, and the next frame the inspector will see
                // `selected_resource = Some(target)` and route to
                // `draw_resource_inspector`.
                if let Some(target) = nav_target {
                    self.selected_resource = Some(target);
                }
            });
    }

    /// Build the breadcrumb crumbs shown above the face inspector.
    /// Always starts with the face itself; appends a clickable
    /// `Material: <name>` crumb when the face has a material, and
    /// further appends a clickable `Texture: <name>` crumb when
    /// that material wraps one. The chain shortens naturally for
    /// partially-assigned faces.
    fn face_breadcrumb(
        &self,
        face: FaceRef,
        current_material: Option<ResourceId>,
    ) -> Vec<BreadcrumbCrumb> {
        let mut crumbs = vec![BreadcrumbCrumb {
            label: format!("Face: {}", describe_face(face)),
            nav: None,
        }];
        if let Some(material_id) = current_material {
            if let Some(material_resource) = self.project.resource(material_id) {
                crumbs.push(BreadcrumbCrumb {
                    label: format!("Material: {}", material_resource.name),
                    nav: Some(material_id),
                });
                if let ResourceData::Material(m) = &material_resource.data {
                    if let Some(tex_id) = m.texture {
                        if let Some(tex_resource) = self.project.resource(tex_id) {
                            crumbs.push(BreadcrumbCrumb {
                                label: format!("Texture: {}", tex_resource.name),
                                nav: Some(tex_id),
                            });
                        }
                    }
                }
            }
        }
        crumbs
    }

    /// Inspector panel for the face currently selected by the
    /// Select tool. Surfaces material picker, height fields, and a
    /// preview thumbnail of the linked texture so the user can
    /// retarget materials without opening the resources panel.
    /// Edge inspector -- height of both endpoint vertices, with
    /// a "Both" toggle for paired drag. Each endpoint resolves
    /// through `physical_vertex` so a height edit propagates to
    /// every face-corner sharing that physical point.
    fn draw_edge_inspector(&mut self, ui: &mut egui::Ui, edge: EdgeRef) {
        // Resolve both endpoints up front while the project is
        // borrowed immutably. Keeps the edit phase below clear
        // of cross-borrow tangles.
        let (mut endpoint_a, mut endpoint_b) = match self
            .room_grid_view(edge.room)
            .and_then(|grid| edge_endpoints(grid, edge))
        {
            Some(pair) => pair,
            None => {
                ui.weak("Edge target is gone");
                return;
            }
        };

        ui.horizontal(|ui| {
            draw_inline_icon(ui, icons::GRID, STUDIO_ACCENT);
            ui.strong(describe_edge(edge));
        });
        ui.separator();

        let mut new_a = endpoint_a.world[1];
        let mut new_b = endpoint_b.world[1];
        let mut changed_a = false;
        let mut changed_b = false;
        ui.horizontal(|ui| {
            ui.label("    A");
            if ui
                .add(egui::DragValue::new(&mut new_a).speed(HEIGHT_QUANTUM as f32))
                .changed()
            {
                changed_a = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("    B");
            if ui
                .add(egui::DragValue::new(&mut new_b).speed(HEIGHT_QUANTUM as f32))
                .changed()
            {
                changed_b = true;
            }
        });

        if !(changed_a || changed_b) {
            return;
        }

        // Apply against the active project. Push undo BEFORE
        // mutating so the snapshot captures the pre-edit state.
        self.push_undo();
        let new_a = snap_height(new_a);
        let new_b = snap_height(new_b);
        endpoint_a.world[1] = new_a;
        endpoint_b.world[1] = new_b;
        let scene = self.project.active_scene_mut();
        let Some(node) = scene.node_mut(edge.room) else {
            return;
        };
        let NodeKind::Room { grid } = &mut node.kind else {
            return;
        };
        if changed_a {
            apply_vertex_height(grid, &endpoint_a, new_a);
        }
        if changed_b {
            apply_vertex_height(grid, &endpoint_b, new_b);
        }
        let total_members = endpoint_a.members.len() + endpoint_b.members.len();
        self.status = format!(
            "Moved edge endpoints ({} face-corners follow)",
            total_members
        );
        self.mark_dirty();
    }

    /// Vertex inspector -- one Y handle for the whole
    /// physical-vertex group. Lists every member so the user
    /// sees what will move; a `Break` button raises the seed
    /// alone by `HEIGHT_QUANTUM` so the user can split a shared
    /// vertex into two.
    fn draw_vertex_inspector(&mut self, ui: &mut egui::Ui, vertex: VertexRef) {
        let mut physical = match self
            .room_grid_view(vertex.room)
            .and_then(|grid| physical_vertex(grid, vertex.anchor.as_face_corner()))
        {
            Some(pv) => pv,
            None => {
                ui.weak("Vertex target is gone");
                return;
            }
        };

        ui.horizontal(|ui| {
            draw_inline_icon(ui, icons::GRID, STUDIO_ACCENT);
            ui.strong(describe_vertex(vertex));
        });
        ui.label(format!(
            "world {} {} {}",
            physical.world[0], physical.world[1], physical.world[2]
        ));
        ui.separator();

        let mut new_y = physical.world[1];
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("    Y");
            if ui
                .add(egui::DragValue::new(&mut new_y).speed(HEIGHT_QUANTUM as f32))
                .changed()
            {
                changed = true;
            }
        });

        let break_clicked = ui
            .button("Break")
            .on_hover_text(
                "Move this corner alone by one quantum, splitting it from the shared group.",
            )
            .clicked();

        egui::CollapsingHeader::new(format!("Members ({})", physical.members.len()))
            .default_open(false)
            .show(ui, |ui| {
                for member in &physical.members {
                    ui.label(face_corner_label(*member));
                }
            });

        if changed {
            self.push_undo();
            let new_y = snap_height(new_y);
            physical.world[1] = new_y;
            let scene = self.project.active_scene_mut();
            if let Some(node) = scene.node_mut(vertex.room) {
                if let NodeKind::Room { grid } = &mut node.kind {
                    apply_vertex_height(grid, &physical, new_y);
                    self.status = format!(
                        "Moved vertex ({} face-corners follow)",
                        physical.members.len()
                    );
                    self.mark_dirty();
                }
            }
        } else if break_clicked {
            self.push_undo();
            let new_y = snap_height(physical.world[1] + HEIGHT_QUANTUM);
            // Apply only to the seed -- the rest of the group
            // stays put, so they cease being coincident.
            let seed = vertex.anchor.as_face_corner();
            let scene = self.project.active_scene_mut();
            if let Some(node) = scene.node_mut(vertex.room) {
                if let NodeKind::Room { grid } = &mut node.kind {
                    write_face_corner_height(grid, seed, new_y);
                    self.status = "Broke vertex; seed corner moved by one quantum".to_string();
                    self.mark_dirty();
                }
            }
        }
    }

    fn draw_face_inspector(&mut self, ui: &mut egui::Ui, face: FaceRef) {
        // Deferred navigation request: pickers fill this in when
        // the user clicks the `→` jump button. Applied after the
        // mutable scene borrow below releases so we never mutate
        // `self.selected_*` while the project is borrowed.
        let mut nav_target: Option<ResourceId> = None;
        let material_options = self.project.material_options();
        // Snapshot the face's current material id BEFORE we borrow
        // the scene mutably, so the preview lookup below can run
        // without fighting the inspector's `&mut` on resource.data.
        let current_material = self
            .project
            .active_scene()
            .node(face.room)
            .and_then(|node| match &node.kind {
                NodeKind::Room { grid } => Some(grid),
                _ => None,
            })
            .and_then(|grid| grid.sector(face.sx, face.sz))
            .and_then(|sector| match face.kind {
                FaceKind::Floor => sector.floor.as_ref().and_then(|f| f.material),
                FaceKind::Ceiling => sector.ceiling.as_ref().and_then(|c| c.material),
                FaceKind::Wall { dir, stack } => sector
                    .walls
                    .get(dir)
                    .get(stack as usize)
                    .and_then(|w| w.material),
            });
        let preview_thumb = current_material
            .and_then(|id| self.project.resource(id))
            .and_then(|resource| self.texture_thumb_entry(resource))
            .map(|entry| (entry.handle.id(), entry.stats));

        // Build the Face › Material › Texture breadcrumb up
        // front while the project is still only borrowed
        // immutably. Crumbs link to whatever's reachable; the
        // chain auto-shortens when the face has no material or
        // the material has no texture.
        let crumbs = self.face_breadcrumb(face, current_material);

        ui.horizontal(|ui| {
            draw_inline_icon(ui, icons::GRID, STUDIO_ACCENT);
            ui.strong(describe_face(face));
        });
        draw_breadcrumb(ui, &crumbs, &mut nav_target);
        ui.separator();
        draw_psxt_preview_block(ui, preview_thumb);

        let scene = self.project.active_scene_mut();
        let Some(room) = scene.node_mut(face.room) else {
            ui.weak("Selected face's Room is gone");
            return;
        };
        let NodeKind::Room { grid } = &mut room.kind else {
            ui.weak("Selected face's Room kind changed");
            return;
        };
        if face.sx >= grid.width || face.sz >= grid.depth {
            ui.weak("Cell out of grid bounds");
            return;
        }
        let sector_size = grid.sector_size;
        let Some(sector) = grid.ensure_sector(face.sx, face.sz) else {
            ui.weak("Cell not authored");
            return;
        };

        let mut changed = false;
        match face.kind {
            FaceKind::Floor => {
                let Some(face_data) = sector.floor.as_mut() else {
                    ui.weak("Floor was removed");
                    return;
                };
                egui::CollapsingHeader::new(icons::label(icons::BLEND, "Material"))
                    .default_open(true)
                    .show(ui, |ui| {
                        changed |= material_picker(
                            ui,
                            "    Material",
                            &mut face_data.material,
                            &material_options,
                            &mut nav_target,
                        );
                    });
                egui::CollapsingHeader::new(icons::label(icons::MOVE, "Heights"))
                    .default_open(true)
                    .show(ui, |ui| {
                        changed |= height_row("Height", &mut face_data.heights, ui);
                    });
            }
            FaceKind::Ceiling => {
                let Some(face_data) = sector.ceiling.as_mut() else {
                    ui.weak("Ceiling was removed");
                    return;
                };
                egui::CollapsingHeader::new(icons::label(icons::BLEND, "Material"))
                    .default_open(true)
                    .show(ui, |ui| {
                        changed |= material_picker(
                            ui,
                            "    Material",
                            &mut face_data.material,
                            &material_options,
                            &mut nav_target,
                        );
                    });
                egui::CollapsingHeader::new(icons::label(icons::MOVE, "Heights"))
                    .default_open(true)
                    .show(ui, |ui| {
                        changed |= height_row("Height", &mut face_data.heights, ui);
                    });
            }
            FaceKind::Wall { dir, stack } => {
                let walls = sector.walls.get_mut(dir);
                let Some(wall) = walls.get_mut(stack as usize) else {
                    ui.weak("Wall stack entry was removed");
                    return;
                };
                egui::CollapsingHeader::new(icons::label(icons::BLEND, "Material"))
                    .default_open(true)
                    .show(ui, |ui| {
                        changed |= material_picker(
                            ui,
                            "    Material",
                            &mut wall.material,
                            &material_options,
                            &mut nav_target,
                        );
                    });
                egui::CollapsingHeader::new(icons::label(icons::MOVE, "Span"))
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("    Bottom");
                            let mut bot = wall.heights[0];
                            if ui
                                .add(egui::DragValue::new(&mut bot).speed(HEIGHT_QUANTUM as f32))
                                .changed()
                            {
                                let bot = snap_height(bot);
                                wall.heights[0] = bot;
                                wall.heights[1] = bot;
                                changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("    Top");
                            let mut top = wall.heights[2];
                            if ui
                                .add(egui::DragValue::new(&mut top).speed(HEIGHT_QUANTUM as f32))
                                .changed()
                            {
                                let top = snap_height(top);
                                wall.heights[2] = top;
                                wall.heights[3] = top;
                                changed = true;
                            }
                        });
                    });
                let _ = sector_size;
            }
        }

        if changed {
            self.mark_dirty();
        }

        // Apply any deferred nav request from the picker `→`
        // buttons. Safe here because the mutable scene borrow
        // ended at the end of the match block above.
        if let Some(target) = nav_target {
            self.selected_primitive = None;
            self.selected_resource = Some(target);
        }
    }

    /// Breadcrumb crumbs for the resource inspector. A texture
    /// shows just `Texture: <name>` (we don't track which
    /// materials reference it, so there's no parent crumb to
    /// add). A material shows `Material: <name> › Texture: <name>`
    /// when its texture is set. Other resource kinds get a
    /// single self-crumb.
    fn resource_breadcrumb(&self, id: ResourceId) -> Vec<BreadcrumbCrumb> {
        let Some(resource) = self.project.resource(id) else {
            return Vec::new();
        };
        let label_for = |kind: &str, name: &str| format!("{kind}: {name}");
        let mut crumbs = vec![BreadcrumbCrumb {
            label: label_for(resource.data.label(), &resource.name),
            nav: None,
        }];
        if let ResourceData::Material(m) = &resource.data {
            if let Some(tex_id) = m.texture {
                if let Some(tex) = self.project.resource(tex_id) {
                    crumbs.push(BreadcrumbCrumb {
                        label: label_for(tex.data.label(), &tex.name),
                        nav: Some(tex_id),
                    });
                }
            }
        }
        crumbs
    }

    fn draw_resource_inspector(&mut self, ui: &mut egui::Ui, id: ResourceId) {
        // Deferred jump-to navigation, same pattern as
        // `draw_face_inspector`. Applied after the mutable
        // resource borrow releases.
        let mut nav_target: Option<ResourceId> = None;
        // Pull the cached preview before borrowing `self.project`
        // mutably below. `texture_thumb_entry` takes `&self` and
        // walks Texture / Material → cached `.psxt` decode, so this
        // copy is the only way to keep both alive in one inspector.
        let preview_thumb = self
            .project
            .resource(id)
            .and_then(|resource| self.texture_thumb_entry(resource))
            .map(|entry| (entry.handle.id(), entry.stats));
        // Snapshot project_dir for path resolution inside the
        // model inspector (parses .psxmdl / .psxanim for live
        // stats display). Cloned so the mutable resource borrow
        // below doesn't fight `&self.project_dir`.
        let project_root = self.project_dir.clone();
        let texture_options: Vec<(ResourceId, String)> = self
            .project
            .resources
            .iter()
            .filter_map(|r| match &r.data {
                ResourceData::Texture { .. } => Some((r.id, r.name.clone())),
                _ => None,
            })
            .collect();
        // Snapshot Model resources + their clip names so the
        // Character inspector can populate model + clip pickers
        // without borrowing `self.project` while the mutable
        // borrow on `resource_mut` is live.
        let character_ctx = build_character_editor_context(&self.project);

        // Build the breadcrumb before the mutable borrow on
        // `resource_mut` -- we need other resources by id to
        // resolve the material → texture link.
        let crumbs = self.resource_breadcrumb(id);

        let Some((resource_raw_id, current_name, resource_data)) =
            self.project.resource(id).map(|resource| {
                (
                    resource.id.raw(),
                    resource.name.clone(),
                    resource.data.clone(),
                )
            })
        else {
            ui.weak("Resource missing");
            return;
        };

        if !matches!(self.resource_renaming, Some((editing_id, _)) if editing_id == id) {
            self.resource_renaming = Some((id, current_name.clone()));
        }

        let mut rename_commit: Option<String> = None;
        let mut rename_cancelled = false;
        let mut changed = false;
        ui.horizontal(|ui| {
            draw_inline_icon(
                ui,
                resource_lucide_icon(&resource_data),
                resource_lucide_color(&resource_data, true),
            );
            ui.strong(format!("{} #{}", resource_data.label(), resource_raw_id));
        });
        draw_breadcrumb(ui, &crumbs, &mut nav_target);
        ui.horizontal(|ui| {
            ui.label("Name");
            if let Some((_, buffer)) = &mut self.resource_renaming {
                let response = ui.text_edit_singleline(buffer);
                let enter = response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let escape = response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape));
                if escape {
                    *buffer = current_name.clone();
                    rename_cancelled = true;
                } else if response.lost_focus() || enter {
                    rename_commit = Some(buffer.clone());
                }
            }
        });
        if rename_cancelled {
            self.status = "Resource rename cancelled".to_string();
        }
        if let Some(name) = rename_commit {
            self.commit_resource_rename(id, name);
        }

        let Some(resource) = self.project.resource_mut(id) else {
            ui.weak("Resource missing");
            return;
        };

        ui.separator();

        match &mut resource.data {
            ResourceData::Texture { psxt_path } => {
                draw_psxt_preview_block(ui, preview_thumb);
                egui::CollapsingHeader::new(icons::label(icons::FILE, "PSXT"))
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label("Path");
                        changed |= ui.text_edit_singleline(psxt_path).changed();
                        ui.label(
                            RichText::new("Cooked .psxt blob; same artifact the runtime embeds.")
                                .color(STUDIO_TEXT_WEAK)
                                .small(),
                        );
                    });
                if let Some((_, stats)) = preview_thumb {
                    egui::CollapsingHeader::new(icons::label(icons::SCAN, "Info"))
                        .default_open(true)
                        .show(ui, |ui| {
                            draw_psxt_stats(ui, stats);
                        });
                }
            }
            ResourceData::Material(material) => {
                draw_psxt_preview_block(ui, preview_thumb);
                egui::CollapsingHeader::new(icons::label(icons::BLEND, "Material"))
                    .default_open(true)
                    .show(ui, |ui| {
                        changed |= material_texture_picker(
                            ui,
                            &mut material.texture,
                            &texture_options,
                            &mut nav_target,
                        );
                        changed |= blend_mode_editor(ui, &mut material.blend_mode);
                        changed |= color_editor(ui, "Tint", &mut material.tint);
                        let resolved_sides = material.sidedness();
                        if material.face_sidedness != resolved_sides {
                            material.face_sidedness = resolved_sides;
                            material.sync_legacy_sidedness();
                            changed = true;
                        }
                        let before = material.face_sidedness;
                        egui::ComboBox::from_label("Sides")
                            .selected_text(material.face_sidedness.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut material.face_sidedness,
                                    MaterialFaceSidedness::Front,
                                    MaterialFaceSidedness::Front.label(),
                                );
                                ui.selectable_value(
                                    &mut material.face_sidedness,
                                    MaterialFaceSidedness::Back,
                                    MaterialFaceSidedness::Back.label(),
                                );
                                ui.selectable_value(
                                    &mut material.face_sidedness,
                                    MaterialFaceSidedness::Both,
                                    MaterialFaceSidedness::Both.label(),
                                );
                            });
                        if material.face_sidedness != before {
                            material.sync_legacy_sidedness();
                            changed = true;
                        }
                    });
                if let Some((_, stats)) = preview_thumb {
                    egui::CollapsingHeader::new(icons::label(icons::SCAN, "Linked Texture"))
                        .default_open(false)
                        .show(ui, |ui| {
                            draw_psxt_stats(ui, stats);
                        });
                }
            }
            ResourceData::Model(model) => {
                changed |= draw_model_resource_editor(ui, model, &project_root, preview_thumb);
            }
            ResourceData::Character(character) => {
                changed |= draw_character_resource_editor(ui, character, &character_ctx);
            }
            ResourceData::Mesh { source_path }
            | ResourceData::Scene { source_path }
            | ResourceData::Script { source_path }
            | ResourceData::Audio { source_path } => {
                egui::CollapsingHeader::new(icons::label(icons::FILE, "Import"))
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label("Source");
                        changed |= ui.text_edit_singleline(source_path).changed();
                    });
            }
        }

        if changed {
            self.mark_dirty();
        }

        // Apply deferred nav so the user can drill straight
        // into the linked texture.
        if let Some(target) = nav_target {
            self.selected_resource = Some(target);
        }
    }

    fn draw_content_browser(&mut self, ctx: &egui::Context) {
        // Refresh PSXT thumbnail handles up-front so the resource
        // cards rendered below have something to blit instead of the
        // name-keyword procedural fallback. Cheap when nothing's
        // changed -- the signature cache short-circuits per-resource.
        self.refresh_texture_thumbs(ctx);
        egui::TopBottomPanel::bottom("psxed_content_browser")
            .resizable(true)
            .default_height(240.0)
            .min_height(160.0)
            .frame(dock_frame())
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(icons::label(icons::LAYERS, "Resources"));
                    ui.separator();
                    if ui
                        .button(icons::label(icons::PLUS, "Material"))
                        .on_hover_text("Add a new Material resource.")
                        .clicked()
                    {
                        let id = self.project.add_resource(
                            "New Material",
                            ResourceData::Material(MaterialResource::opaque(None)),
                        );
                        self.selected_resource = Some(id);
                        self.status = "Added material".to_string();
                        self.mark_dirty();
                    }
                    if ui
                        .button(icons::label(icons::PLUS, "Texture"))
                        .on_hover_text("Add a Texture resource pointing at a cooked .psxt blob.")
                        .clicked()
                    {
                        let id = self.project.add_resource(
                            "New Texture",
                            ResourceData::Texture {
                                psxt_path: String::new(),
                            },
                        );
                        self.selected_resource = Some(id);
                        self.status = "Added texture".to_string();
                        self.mark_dirty();
                    }
                    if ui
                        .button(icons::label(icons::PLUS, "Model"))
                        .on_hover_text(
                            "Add an empty Model resource. Use the inspector's \
                             'Register Cooked Folder' button to populate it from \
                             an existing bundle, or fill in the paths by hand.",
                        )
                        .clicked()
                    {
                        let id = self.project.add_resource(
                            "New Model",
                            ResourceData::Model(psxed_project::ModelResource {
                                model_path: String::new(),
                                texture_path: None,
                                clips: Vec::new(),
                                default_clip: None,
                                preview_clip: None,
                                world_height: 1024,
                            }),
                        );
                        self.selected_resource = Some(id);
                        self.status = "Added model".to_string();
                        self.mark_dirty();
                    }
                    if ui
                        .button(icons::label(icons::FILE_PLUS, "Import Model"))
                        .on_hover_text(
                            "Open the GLB/glTF model import preview with atlas, clip, and root-centering controls.",
                        )
                        .clicked()
                    {
                        self.open_model_import_dialog();
                    }
                });
                ui.separator();
                self.draw_resources_tab(ui);
            });
    }

    /// Walk every Texture resource and ensure its `.psxt` blob has
    /// been decoded into an egui texture handle the resource cards
    /// can blit. Skips entries whose `psxt_path` matches the cached
    /// signature; rebuilds when the path moves or the file is newly
    /// readable.
    fn refresh_texture_thumbs(&mut self, ctx: &egui::Context) {
        // Clone so the immutable borrow on `self` released here
        // doesn't fight the mutable borrow on `self.texture_thumbs`
        // we need below.
        let project_root = self.project_dir.clone();
        let mut alive: Vec<ResourceId> = Vec::new();
        for resource in self.project.resources.iter() {
            // Texture resources point straight at a `.psxt`;
            // Model resources have a `texture_path` field -- both
            // share the same on-disk format and decoder, so the
            // thumbnail cache treats them uniformly.
            let psxt_path: &str = match &resource.data {
                ResourceData::Texture { psxt_path } => psxt_path.as_str(),
                ResourceData::Model(model) => match &model.texture_path {
                    Some(p) => p.as_str(),
                    None => continue,
                },
                _ => continue,
            };
            alive.push(resource.id);
            if let Some(entry) = self.texture_thumbs.get(&resource.id) {
                if entry.signature == psxt_path {
                    continue;
                }
            }
            if psxt_path.is_empty() {
                self.texture_thumbs.remove(&resource.id);
                continue;
            }
            let abs = if Path::new(psxt_path).is_absolute() {
                PathBuf::from(psxt_path)
            } else {
                project_root.join(psxt_path)
            };
            let Some((image, stats)) = std::fs::read(&abs)
                .ok()
                .and_then(|bytes| decode_psxt_thumbnail(&bytes))
            else {
                self.texture_thumbs.remove(&resource.id);
                continue;
            };
            let handle = ctx.load_texture(
                format!("psxt-thumb-{}", resource.id.raw()),
                image,
                egui::TextureOptions::NEAREST,
            );
            self.texture_thumbs.insert(
                resource.id,
                ThumbnailEntry {
                    signature: psxt_path.to_string(),
                    handle,
                    stats,
                },
            );
        }
        // Drop entries for Texture resources that no longer exist --
        // keeps the cache from growing across delete + re-add.
        self.texture_thumbs.retain(|id, _| alive.contains(id));
    }

    /// Resolve the underlying Texture id for a Material, or the
    /// Texture's own id if `resource` is one. `None` for everything
    /// else.
    fn texture_thumb_id(&self, resource: &Resource) -> Option<egui::TextureId> {
        self.texture_thumb_entry(resource).map(|e| e.handle.id())
    }

    /// Look up the cached thumbnail entry (handle + stats) for a
    /// Texture resource directly, or for a Material via its texture
    /// link. `None` when the link is unset, the file isn't readable,
    /// or the depth is unsupported (15bpp at present).
    fn texture_thumb_entry(&self, resource: &Resource) -> Option<&ThumbnailEntry> {
        let key = match &resource.data {
            ResourceData::Texture { .. } => Some(resource.id),
            ResourceData::Material(mat) => mat.texture,
            _ => None,
        }?;
        self.texture_thumbs.get(&key)
    }

    fn draw_resources_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            section_frame().show(ui, |ui| {
                // Frame inherits the outer `ui.horizontal` layout, so
                // every child widget would otherwise flow on a single
                // row. Force vertical so the filter buttons stack as
                // intended.
                ui.vertical(|ui| {
                    ui.set_width(180.0);
                    panel_heading(ui, icons::SCAN, "Filter");
                    ui.add_space(2.0);
                    ui.selectable_value(
                        &mut self.resource_filter,
                        ResourceFilter::All,
                        icons::label(ResourceFilter::All.icon(), "All"),
                    );
                    for (filter, count) in resource_filter_counts(&self.project) {
                        ui.selectable_value(
                            &mut self.resource_filter,
                            filter,
                            format!("{} ({count})", icons::label(filter.icon(), filter.label())),
                        );
                    }
                });
            });

            ui.add_space(4.0);

            ui.vertical(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.resource_search)
                        .hint_text("Filter resources")
                        .desired_width(460.0),
                );
                let mut clicked = None;
                egui::ScrollArea::horizontal()
                    .id_salt("psxed_resource_cards")
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let search = self.resource_search.to_ascii_lowercase();
                            for resource in self.project.resources.iter().filter(|resource| {
                                resource_matches_filter(
                                    resource,
                                    self.resource_filter,
                                    search.as_str(),
                                )
                            }) {
                                let thumb = self.texture_thumb_id(resource);
                                if draw_resource_card(
                                    ui,
                                    &self.project,
                                    resource,
                                    self.selected_resource == Some(resource.id),
                                    thumb,
                                )
                                .clicked()
                                {
                                    clicked = Some(resource.id);
                                }
                            }
                        });
                    });
                if let Some(id) = clicked {
                    // Sims-style: with a face selected, clicking a
                    // Material card retargets the face's material
                    // rather than swapping the inspector. Texture
                    // / non-Material clicks still navigate normally.
                    let is_material = matches!(
                        self.project.resource(id).map(|r| &r.data),
                        Some(ResourceData::Material(_))
                    );
                    let face_for_assign = self
                        .selected_primitive
                        .and_then(|s| s.as_face())
                        .filter(|_| is_material);
                    if let Some(face) = face_for_assign {
                        if self.assign_face_material(face, Some(id)) {
                            self.status = format!("Assigned material to {}", describe_face(face));
                        }
                        self.selected_resource = Some(id);
                    } else {
                        self.selected_resource = Some(id);
                        self.selected_node = NodeId::ROOT;
                        self.selected_primitive = None;
                    }
                }
            });
        });
    }

    fn draw_viewport(
        &mut self,
        ctx: &egui::Context,
        viewport_3d: EditorViewport3dPresentation,
        playtest_status: EditorPlaytestStatus,
    ) {
        egui::CentralPanel::default()
            .frame(viewport_frame())
            .show(ctx, |ui| {
                self.draw_viewport_tabs(ui);
                ui.separator();
                self.draw_viewport_toolbar(ui);
                ui.separator();

                if viewport_3d.mode == EditorViewport3dMode::Play {
                    self.draw_viewport_3d_play_body(ui, viewport_3d, playtest_status);
                    return;
                }

                if !self.view_2d {
                    self.draw_viewport_3d_body(ui, viewport_3d);
                    return;
                }

                let size = ui.available_size();
                let size = Vec2::new(size.x.max(320.0), size.y.max(240.0));
                let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
                let dnd_active = egui::DragAndDrop::has_any_payload(ui.ctx());
                let resource_drop_hovered = response.dnd_hover_payload::<ResourceId>().is_some();

                if !dnd_active
                    && (response.dragged_by(egui::PointerButton::Middle)
                        || response.dragged_by(egui::PointerButton::Secondary))
                {
                    self.viewport_pan += ui.input(|input| input.pointer.delta());
                }
                if !dnd_active && response.dragged_by(egui::PointerButton::Primary) {
                    self.drag_selected_node(ui.input(|input| input.pointer.delta()));
                }

                if !dnd_active && response.hovered() {
                    let scroll = ui.input(|input| input.raw_scroll_delta.y);
                    if scroll.abs() > f32::EPSILON {
                        let pointer = ui
                            .input(|input| input.pointer.hover_pos())
                            .unwrap_or_else(|| rect.center());
                        let before =
                            ViewportTransform::new(rect, self.viewport_pan, self.viewport_zoom)
                                .screen_to_world(pointer);
                        let zoom_factor = (1.0 + scroll * 0.0015).clamp(0.75, 1.25);
                        self.viewport_zoom = (self.viewport_zoom * zoom_factor)
                            .clamp(MIN_VIEWPORT_ZOOM, MAX_VIEWPORT_ZOOM);
                        let after =
                            ViewportTransform::new(rect, self.viewport_pan, self.viewport_zoom)
                                .world_to_screen(before);
                        self.viewport_pan += pointer - after;
                    }
                }

                let transform = ViewportTransform::new(rect, self.viewport_pan, self.viewport_zoom);
                if let Some(resource_id) = response
                    .dnd_release_payload::<ResourceId>()
                    .map(|payload| *payload)
                {
                    if let Some(pointer) = response.interact_pointer_pos().or(response.hover_pos())
                    {
                        let world = transform.screen_to_world(pointer);
                        self.drop_resource_2d(resource_id, world);
                    }
                }
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 0.0, STUDIO_VIEWPORT);
                if self.show_grid {
                    draw_world_grid(&painter, transform);
                }

                let hits =
                    draw_scene_viewport(&painter, transform, &self.project, self.selected_node);

                if !dnd_active && response.clicked_by(egui::PointerButton::Primary) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let world = transform.screen_to_world(pos);
                        self.handle_viewport_click(world, &hits);
                    }
                }

                draw_viewport_overlay(
                    &painter,
                    rect,
                    &self.project,
                    self.viewport_zoom,
                    self.snap_units,
                );
                draw_axes_gizmo(&painter, rect);
                if resource_drop_hovered {
                    painter.rect_stroke(
                        rect.shrink(2.0),
                        2.0,
                        Stroke::new(2.0, STUDIO_ACCENT),
                        StrokeKind::Inside,
                    );
                    painter.text(
                        rect.center_top() + Vec2::new(0.0, 16.0),
                        Align2::CENTER_TOP,
                        "Drop resource into scene",
                        FontId::proportional(13.0),
                        STUDIO_ACCENT,
                    );
                }
            });
    }

    fn draw_viewport_tabs(&mut self, ui: &mut egui::Ui) {
        // Single-tab placeholder: shows whatever Room is the
        // current edit context. Multi-room tab switching can land
        // alongside File→Open Project; for now just reflect reality
        // instead of the previous hardcoded "Stone Room.room".
        let room_label = self
            .active_room_id()
            .and_then(|id| self.project.active_scene().node(id))
            .map(|node| node.name.as_str())
            .unwrap_or("(no room)");
        ui.horizontal(|ui| {
            let _ = ui.selectable_label(
                true,
                RichText::new(icons::label(icons::GRID, &format!("{room_label}.room"))).strong(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("Project: {}", self.project.name))
                        .color(STUDIO_TEXT_WEAK),
                );
            });
        });
    }

    fn draw_viewport_toolbar(&mut self, ui: &mut egui::Ui) {
        let room_active = self.active_room_id().is_some();
        if !room_active && self.active_tool.requires_room_context() {
            self.active_tool = ViewTool::Select;
        }
        // Two rows: row 1 is "what to do" (tools + the brush
        // material a paint tool will use); row 2 is "how to view"
        // (snap / grid / 2D-vs-3D / zoom). Splitting keeps the
        // toolbar from overflowing on narrow windows and groups
        // controls by intent so each row scans as one decision.
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                {
                    let (tool, label, hint) = (
                        ViewTool::Select,
                        icons::label(icons::POINTER, "Select"),
                        "Click to select; drag the selection vertically to translate it.",
                    );
                    ui.selectable_value(&mut self.active_tool, tool, label)
                        .on_hover_text(hint);
                }
                ui.separator();
                ui.add_enabled_ui(room_active, |ui| {
                    for (tool, label, hint) in [
                        (
                            ViewTool::PaintFloor,
                            icons::label(icons::GRID, "Floor"),
                            "Paint a floor on each cell you click or drag over.",
                        ),
                        (
                            ViewTool::PaintWall,
                            icons::label(icons::BRICK_WALL, "Wall"),
                            "Paint a wall on the edge nearest the click.",
                        ),
                        (
                            ViewTool::PaintCeiling,
                            icons::label(icons::LAYERS, "Ceiling"),
                            "Paint a ceiling on each cell.",
                        ),
                        (
                            ViewTool::Erase,
                            icons::label(icons::TRASH, "Erase"),
                            "Clear floor/walls/ceiling from the cell.",
                        ),
                        (
                            ViewTool::Place,
                            icons::label(icons::PLUS, "Place"),
                            "Drop a SpawnPoint at the clicked cell.",
                        ),
                    ] {
                        ui.selectable_value(&mut self.active_tool, tool, label)
                            .on_hover_text(hint);
                    }
                });
                ui.separator();
                if matches!(self.active_tool, ViewTool::Select) {
                    self.draw_selection_mode_picker(ui);
                } else if matches!(self.active_tool, ViewTool::Place) {
                    self.draw_place_kind_picker(ui);
                } else {
                    self.draw_brush_material_picker(ui);
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut self.snap_to_grid,
                    icons::label(icons::WAYPOINT, "Snap"),
                );
                ui.add(
                    egui::DragValue::new(&mut self.snap_units)
                        .speed(1.0)
                        .range(1..=256),
                );
                ui.separator();
                ui.checkbox(&mut self.show_grid, icons::label(icons::GRID, "Grid"));
                ui.separator();
                ui.selectable_value(&mut self.view_2d, true, icons::label(icons::GRID, "2D"));
                ui.selectable_value(&mut self.view_2d, false, icons::label(icons::BOX, "3D"));
                ui.separator();
                ui.label(RichText::new("Zoom").color(STUDIO_TEXT_WEAK));
                let mut zoom_percent = (self.viewport_zoom / DEFAULT_VIEWPORT_ZOOM * 100.0) as u16;
                if ui
                    .add(
                        egui::DragValue::new(&mut zoom_percent)
                            .speed(1.0)
                            .range(25..=250)
                            .suffix("%"),
                    )
                    .changed()
                {
                    self.viewport_zoom = (zoom_percent as f32 / 100.0 * DEFAULT_VIEWPORT_ZOOM)
                        .clamp(MIN_VIEWPORT_ZOOM, MAX_VIEWPORT_ZOOM);
                }
            });
        });
    }

    /// Toolbar combobox for the active brush material. Selecting
    /// "Auto" leaves `brush_material = None` so paint falls back to
    /// the per-tool name heuristic (`floor → "floor" material,
    /// brick → "brick" material`); picking a specific entry pins
    /// every Floor / Wall / Ceiling stroke to that material.
    /// Toolbar selector for the Place tool's node kind. Shown
    /// only while `active_tool == Place` -- otherwise the brush
    /// material picker takes the same slot.
    /// Toolbar selector for the Select tool's primitive mode.
    /// Visible only while `active_tool == Select`. Mirrors the
    /// `1` / `2` / `3` hotkeys; clicking goes through
    /// `set_selection_mode` so the existing selection adapts.
    fn draw_selection_mode_picker(&mut self, ui: &mut egui::Ui) {
        ui.label(icons::label(icons::POINTER, "Mode"));
        for mode in [
            SelectionMode::Face,
            SelectionMode::Edge,
            SelectionMode::Vertex,
        ] {
            if ui
                .selectable_label(self.selection_mode == mode, mode.label())
                .on_hover_text(match mode {
                    SelectionMode::Face => "Pick a whole face (1).",
                    SelectionMode::Edge => {
                        "Pick the closest edge of the face under the cursor (2)."
                    }
                    SelectionMode::Vertex => {
                        "Pick the closest corner of the face under the cursor (3)."
                    }
                })
                .clicked()
            {
                self.set_selection_mode(mode);
            }
        }
    }

    fn draw_place_kind_picker(&mut self, ui: &mut egui::Ui) {
        ui.label(icons::label(icons::PLUS, "Place"));
        for kind in [
            PlaceKind::PlayerSpawn,
            PlaceKind::SpawnMarker,
            PlaceKind::ModelInstance,
            PlaceKind::LightMarker,
        ] {
            ui.selectable_value(&mut self.place_kind, kind, kind.label());
        }
    }

    /// Pick the Model resource a `PlaceKind::ModelInstance` click
    /// should bind to. Returns `(id, default_node_name)` on
    /// success or an actionable status message on failure. The
    /// caller renders the failure into `self.status` and skips
    /// the place altogether -- never silently substitutes a
    /// generic marker.
    fn resolve_place_model_resource(&self) -> Result<(ResourceId, String), String> {
        // (a) Selected resource is a Model? Use it.
        if let Some(id) = self.selected_resource {
            if let Some(resource) = self.project.resource(id) {
                if matches!(resource.data, ResourceData::Model(_)) {
                    return Ok((id, resource.name.clone()));
                }
            }
        }
        // (b) Exactly one Model resource? Auto-pick.
        let models: Vec<&Resource> = self
            .project
            .resources
            .iter()
            .filter(|r| matches!(r.data, ResourceData::Model(_)))
            .collect();
        match models.len() {
            0 => Err("No Model resources exist. Register or import a model first.".to_string()),
            1 => Ok((models[0].id, models[0].name.clone())),
            n => Err(format!(
                "Select a Model resource before placing a model ({n} available)"
            )),
        }
    }

    fn draw_brush_material_picker(&mut self, ui: &mut egui::Ui) {
        let materials = self.project.material_options();
        let label = match self.brush_material {
            Some(id) => materials
                .iter()
                .find(|(mid, _)| *mid == id)
                .map(|(_, name)| name.clone())
                .unwrap_or_else(|| "(missing)".to_string()),
            None => "Auto".to_string(),
        };
        ui.label(icons::label(icons::PALETTE, "Brush"));
        egui::ComboBox::from_id_salt("brush-material-picker")
            .selected_text(label)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.brush_material, None, "Auto")
                    .on_hover_text(
                    "Pick by tool: Floor uses the first 'floor' material, Wall the first 'brick'.",
                );
                for (id, name) in &materials {
                    ui.selectable_value(&mut self.brush_material, Some(*id), name);
                }
            });
    }

    /// Resolve the Room node that owns the current selection, if any.
    ///
    /// Order: selected face's room → climb the selected node's
    /// Walk the active scene and collect a selectable AABB for
    /// every entity-kind node -- every node that's neither the
    /// scene root, nor a structural Node/World, nor a Room.
    ///
    /// `room_filter` confines the walk to descendants of one
    /// Room (Some(id)) or includes everything (None). The 3D
    /// click handler uses Some(active_room) so a click in the
    /// active room can't pick lights from another room.
    pub fn collect_entity_bounds(&self, room_filter: Option<NodeId>) -> Vec<EntityBounds> {
        let scene = self.project.active_scene();
        let mut out = Vec::new();
        for node in scene.nodes() {
            if node.id == scene.root {
                continue;
            }
            // Find this node's enclosing Room.
            let enclosing_room = enclosing_room_id(scene, node.id);
            if let (Some(want), Some(actual)) = (room_filter, enclosing_room) {
                if want != actual {
                    continue;
                }
            }
            let Some((kind, half_extents)) = entity_bound_kind_and_size(self, node) else {
                continue;
            };
            // World position. Entities under a Room use the
            // canonical room-local convention so bounds line up
            // with the rendered marker / model exactly.
            let center_world = match enclosing_room.and_then(|id| scene.node(id)) {
                Some(room_node) => match &room_node.kind {
                    NodeKind::Room { grid } => psxed_project::spatial::node_preview_bounds_center(
                        grid,
                        &node.transform,
                        half_extents,
                    ),
                    _ => continue,
                },
                None => {
                    // No enclosing Room -- node lives in raw
                    // world space. Use translation directly so
                    // the bound at least lands somewhere
                    // pickable.
                    let p = node.transform.translation;
                    [p[0], p[1] + half_extents[1], p[2]]
                }
            };
            out.push(EntityBounds {
                node: node.id,
                room: enclosing_room,
                kind,
                center: center_world,
                half_extents,
                yaw_degrees: node.transform.rotation_degrees[1],
            });
        }
        out
    }

    /// Pick the nearest entity bound under the camera ray.
    /// Returns the `EntityBoundHit` plus its world distance --
    /// the 3D click handler compares this against grid hits to
    /// pick whichever is closer.
    pub fn pick_entity_bound(
        &self,
        rect: egui::Rect,
        pointer: egui::Pos2,
        room_filter: Option<NodeId>,
    ) -> Option<EntityBoundHit> {
        let (origin, dir) = self.camera_ray_for_pointer(rect, pointer)?;
        let bounds = self.collect_entity_bounds(room_filter);
        let mut best: Option<EntityBoundHit> = None;
        for b in &bounds {
            let Some(t) = ray_intersects_aabb(origin, dir, b.center, b.half_extents) else {
                continue;
            };
            if best.as_ref().is_some_and(|h| h.distance <= t) {
                continue;
            }
            best = Some(EntityBoundHit {
                node: b.node,
                distance: t,
                point: [
                    origin[0] + dir[0] * t,
                    origin[1] + dir[1] * t,
                    origin[2] + dir[2] * t,
                ],
                bounds: *b,
            });
        }
        best
    }

    /// parent chain → fall back to the active scene's first Room.
    /// The fallback keeps paint tools enabled even when the
    /// selection sits outside the scene tree (e.g. a face the user
    /// just picked, which clears `selected_node` to ROOT).
    pub fn active_room_id(&self) -> Option<NodeId> {
        if let Some(selection) = self.selected_primitive {
            return Some(selection.room());
        }
        let scene = self.project.active_scene();
        let mut current = self.selected_node;
        while let Some(node) = scene.node(current) {
            if matches!(node.kind, NodeKind::Room { .. }) {
                return Some(current);
            }
            let Some(parent) = node.parent else { break };
            current = parent;
        }
        scene
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))
            .map(|node| node.id)
    }

    /// Translate a 2D-viewport-space click into a sector cell on
    /// `room`. The viewport draws cells around `node_world(room)`
    /// with 1 unit = 1 sector, so the click is first re-expressed
    /// as editor coords (room-centre-relative) and then routed
    /// through `WorldGrid::editor_cells_to_array`. `origin` enters
    /// the conversion via the canonical helper, keeping 2D and 3D
    /// picks consistent after a negative-side grow.
    fn world_to_sector(&self, room_id: NodeId, world: [f32; 2]) -> Option<(u16, u16)> {
        let scene = self.project.active_scene();
        let room = scene.node(room_id)?;
        let NodeKind::Room { grid } = &room.kind else {
            return None;
        };
        let center = node_world(room);
        let editor = [world[0] - center[0], world[1] - center[1]];
        grid.editor_cells_to_array(editor)
    }

    /// Default material id for a brushed surface, picked by name from
    /// the project's material list. The cooker rejects unassigned
    /// surfaces, so authors are expected to wire real materials in
    /// resources before serious painting -- this fallback at least
    /// keeps the brush usable while iterating.
    fn default_brush_material(&self, needle: &str) -> Option<ResourceId> {
        let lower = needle.to_ascii_lowercase();
        let materials = self.project.material_options();
        materials
            .iter()
            .find(|(_, name)| name.to_ascii_lowercase().contains(&lower))
            .or_else(|| materials.first())
            .map(|(id, _)| *id)
    }

    /// Single dispatch point for primary-button clicks on the viewport.
    fn handle_viewport_click(&mut self, world: [f32; 2], hits: &[ViewportHit]) {
        match self.active_tool {
            ViewTool::Select => {
                if let Some(hit) = hits.iter().rev().find(|hit| hit.contains(world)) {
                    self.selected_node = hit.id;
                    self.selected_resource = None;
                    self.status = format!("Selected {}", hit.name);
                } else {
                    self.selected_resource = None;
                }
                self.refresh_selected_sector(world);
            }
            tool => {
                let Some(room_id) = self.active_room_id() else {
                    return;
                };
                let Some((x, z)) = self.world_to_sector(room_id, world) else {
                    self.selected_sector = None;
                    return;
                };
                self.selected_sector = Some((x, z));
                self.apply_paint(tool, room_id, x, z, world);
            }
        }
    }

    /// Update `selected_sector` from a click position, if there's a
    /// Room in the current selection chain.
    fn refresh_selected_sector(&mut self, world: [f32; 2]) {
        self.selected_sector = self
            .active_room_id()
            .and_then(|room| self.world_to_sector(room, world));
    }

    /// Apply a 2D-viewport click through the same logic as a 3D
    /// click. Old behaviour kept a separate `apply_paint` body
    /// here that diverged from the 3D `run_paint_action` (no
    /// origin awareness, no wall replacement, no `PlaceKind`
    /// dispatch). Now: lift the click into editor coords, pre-
    /// compute a `picked_face` for PaintWall when the inferred
    /// edge already has a wall stack, and hand off.
    fn apply_paint(&mut self, tool: ViewTool, room_id: NodeId, sx: u16, sz: u16, world: [f32; 2]) {
        // 2D `world` is already in editor sector-units (the 2D
        // viewport's native space, room-centre-relative around
        // `node_world(room)`). Convert through the canonical
        // helper to get a room-local 3D position the rest of the
        // paint flow can chew on. `editor_to_room_local` is
        // origin-aware, so this stays correct after a -X / -Z grow.
        let (hit_world, picked_face) = {
            let scene = self.project.active_scene();
            let Some(room) = scene.node(room_id) else {
                return;
            };
            let NodeKind::Room { grid } = &room.kind else {
                return;
            };
            let room_center = node_world(room);
            let editor = [world[0] - room_center[0], world[1] - room_center[1]];
            let hit = grid.editor_to_room_local(editor);

            // For PaintWall: if the inferred edge already has at
            // least one wall, hand `run_paint_action` a `FaceRef`
            // pointing at the top of the stack so it replaces
            // material instead of appending. Empty edge → None
            // → append path.
            let face = if matches!(tool, ViewTool::PaintWall) {
                let centre = grid.cell_center_world(sx, sz);
                let dir = edge_from_world_offset(hit[0] - centre[0], hit[2] - centre[1]);
                grid.sector(sx, sz).and_then(|sector| {
                    let walls = sector.walls.get(dir);
                    let stack = walls.len().checked_sub(1)?;
                    Some(FaceRef {
                        room: room_id,
                        sx,
                        sz,
                        kind: FaceKind::Wall {
                            dir,
                            stack: stack as u8,
                        },
                    })
                })
            } else {
                None
            };
            (hit, face)
        };

        self.run_paint_action(tool, room_id, sx, sz, picked_face, hit_world);
    }

    fn add_child(&mut self, kind: NodeKind, name: &str) {
        self.push_undo();
        let parent = self.selected_node;
        let id = self
            .project
            .active_scene_mut()
            .add_node(parent, name.to_string(), kind);
        self.selected_node = id;
        self.selected_resource = None;
        self.status = format!("Added {name}");
        self.mark_dirty();
    }

    fn duplicate_selected(&mut self) {
        let selected = self.selected_node;
        let Some(source) = self.project.active_scene().node(selected).cloned() else {
            return;
        };
        self.push_undo();
        let parent = source.parent.unwrap_or(NodeId::ROOT);
        let id = self.project.active_scene_mut().add_node(
            parent,
            format!("{} Copy", source.name),
            source.kind,
        );
        if let Some(node) = self.project.active_scene_mut().node_mut(id) {
            node.transform = source.transform;
        }
        self.selected_node = id;
        self.selected_resource = None;
        self.status = "Duplicated node".to_string();
        self.mark_dirty();
    }

    fn delete_selected(&mut self) {
        let selected = self.selected_node;
        if selected == NodeId::ROOT {
            return;
        }
        self.push_undo();
        if self.project.active_scene_mut().remove_node(selected) {
            self.selected_node = NodeId::ROOT;
            self.selected_resource = None;
            self.status = "Deleted node".to_string();
            self.mark_dirty();
        }
    }

    /// Delete dispatch for the active selection:
    /// - Face   → remove the face from its sector.
    /// - Edge   → remove the face that owns the edge.
    /// - Vertex → drop the corner on the seed face, turning it
    ///   into a triangle (split is auto-flipped to the surviving
    ///   diagonal). The other coincident face-corners are left
    ///   untouched.
    fn delete_selected_primitive(&mut self) {
        let Some(selection) = self.selected_primitive else {
            return;
        };
        let outcome = match selection {
            Selection::Face(face) => self.remove_face(face),
            Selection::Edge(edge) => edge_owning_face_ref(edge)
                .map(|face| self.remove_face(face))
                .unwrap_or(DeleteOutcome::Missing),
            Selection::Vertex(vertex) => self.drop_vertex(vertex),
        };
        match outcome {
            DeleteOutcome::Removed(label) => {
                self.selected_primitive = None;
                self.hovered_primitive = None;
                self.status = format!("Deleted {label}");
                self.mark_dirty();
            }
            DeleteOutcome::Triangulated(label) => {
                self.status = format!("Dropped {label}");
                self.mark_dirty();
            }
            DeleteOutcome::Missing => {
                self.status = "Nothing to delete".to_string();
            }
        }
    }

    /// Detach a face from its sector. Floors / ceilings clear the
    /// `Option<>`; walls splice the entry out of the per-direction
    /// `Vec`. Returns `Removed` on success so the caller can update
    /// status / clear the selection.
    fn remove_face(&mut self, face: FaceRef) -> DeleteOutcome {
        self.push_undo();
        let scene = self.project.active_scene_mut();
        let Some(node) = scene.node_mut(face.room) else {
            return DeleteOutcome::Missing;
        };
        let NodeKind::Room { grid } = &mut node.kind else {
            return DeleteOutcome::Missing;
        };
        let Some(sector) = grid.sector_mut(face.sx, face.sz) else {
            return DeleteOutcome::Missing;
        };
        let removed = match face.kind {
            FaceKind::Floor => sector.floor.take().is_some(),
            FaceKind::Ceiling => sector.ceiling.take().is_some(),
            FaceKind::Wall { dir, stack } => {
                let walls = sector.walls.get_mut(dir);
                if (stack as usize) < walls.len() {
                    walls.remove(stack as usize);
                    true
                } else {
                    false
                }
            }
        };
        if removed {
            DeleteOutcome::Removed(describe_face_kind(face.kind))
        } else {
            DeleteOutcome::Missing
        }
    }

    /// Drop a corner from the vertex's seed face. Floors / ceilings
    /// gain a `dropped_corner` and have their split forced to the
    /// surviving diagonal. Walls do the same with `WallCorner`.
    fn drop_vertex(&mut self, vertex: VertexRef) -> DeleteOutcome {
        self.push_undo();
        let scene = self.project.active_scene_mut();
        let Some(node) = scene.node_mut(vertex.room) else {
            return DeleteOutcome::Missing;
        };
        let NodeKind::Room { grid } = &mut node.kind else {
            return DeleteOutcome::Missing;
        };
        let (sx, sz) = match vertex.anchor {
            VertexAnchor::Floor { sx, sz, .. }
            | VertexAnchor::Ceiling { sx, sz, .. }
            | VertexAnchor::Wall { sx, sz, .. } => (sx, sz),
        };
        let Some(sector) = grid.sector_mut(sx, sz) else {
            return DeleteOutcome::Missing;
        };
        match vertex.anchor {
            VertexAnchor::Floor { corner, .. } => {
                let Some(floor) = sector.floor.as_mut() else {
                    return DeleteOutcome::Missing;
                };
                floor.drop_corner(corner);
                DeleteOutcome::Triangulated("floor corner")
            }
            VertexAnchor::Ceiling { corner, .. } => {
                let Some(ceiling) = sector.ceiling.as_mut() else {
                    return DeleteOutcome::Missing;
                };
                ceiling.drop_corner(corner);
                DeleteOutcome::Triangulated("ceiling corner")
            }
            VertexAnchor::Wall {
                dir, stack, corner, ..
            } => {
                let walls = sector.walls.get_mut(dir);
                let Some(wall) = walls.get_mut(stack as usize) else {
                    return DeleteOutcome::Missing;
                };
                wall.drop_corner(corner);
                DeleteOutcome::Triangulated("wall corner")
            }
        }
    }

    fn open_new_project_dialog(&mut self) {
        self.new_project_dialog_open = true;
        self.new_project_name.clear();
        self.new_project_error = None;
    }

    fn open_model_import_dialog(&mut self) {
        self.model_import_dialog.open = true;
        self.model_import_dialog.status = None;
        self.retire_model_import_preview();
        self.model_import_dialog.selected_clip = 0;
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn commit_resource_rename(&mut self, id: ResourceId, name: String) {
        let Some(current_name) = self.project.resource_name(id).map(str::to_string) else {
            self.resource_renaming = None;
            self.status = format!("Resource #{} no longer exists", id.raw());
            return;
        };

        let final_name = name.trim();
        if final_name.is_empty() {
            self.resource_renaming = Some((id, current_name));
            self.status = "Resource name cannot be empty".to_string();
            return;
        }
        if final_name == current_name {
            self.resource_renaming = Some((id, current_name));
            return;
        }

        let before = self.project.clone();
        match self
            .project
            .rename_resource_with_files(id, final_name, &self.project_dir)
        {
            Ok(report) => {
                if report.renamed_files.is_empty() {
                    self.history.record(before);
                } else {
                    self.history.clear();
                }
                self.resource_renaming = Some((id, final_name.to_string()));
                self.mark_dirty();

                let moved = report.renamed_files.len();
                let skipped = report.skipped_files.len();
                self.status = match (moved, skipped) {
                    (0, 0) => format!("Renamed {final_name}"),
                    (m, 0) => format!("Renamed {final_name}; moved {m} file(s)"),
                    (0, s) => format!("Renamed {final_name}; skipped {s} file path(s)"),
                    (m, s) => {
                        format!("Renamed {final_name}; moved {m} file(s), skipped {s} path(s)")
                    }
                };
            }
            Err(error) => {
                self.resource_renaming = Some((id, current_name));
                self.status = format!("Rename failed: {error}");
            }
        }
    }

    /// Snapshot the current project before a discrete mutation.
    /// Call once per user action -- paint click, place, add/delete
    /// node, etc -- so each undo step matches one author intent.
    fn push_undo(&mut self) {
        self.history.record(self.project.clone());
    }

    /// Pop the most recent snapshot back into `project`.
    fn do_undo(&mut self) {
        if let Some(prev) = self.history.undo(self.project.clone()) {
            self.project = prev;
            self.selected_resource = None;
            self.resource_renaming = None;
            self.selected_sector = None;
            if self
                .project
                .active_scene()
                .node(self.selected_node)
                .is_none()
            {
                self.selected_node = NodeId::ROOT;
            }
            self.status = "Undo".to_string();
            self.mark_dirty();
        } else {
            self.status = "Nothing to undo".to_string();
        }
    }

    fn do_redo(&mut self) {
        if let Some(next) = self.history.redo(self.project.clone()) {
            self.project = next;
            self.selected_resource = None;
            self.resource_renaming = None;
            self.selected_sector = None;
            if self
                .project
                .active_scene()
                .node(self.selected_node)
                .is_none()
            {
                self.selected_node = NodeId::ROOT;
            }
            self.status = "Redo".to_string();
            self.mark_dirty();
        } else {
            self.status = "Nothing to redo".to_string();
        }
    }

    fn frame_viewport(&mut self) {
        self.viewport_pan = Vec2::ZERO;
        self.viewport_zoom = DEFAULT_VIEWPORT_ZOOM;
    }

    fn drag_selected_node(&mut self, screen_delta: Vec2) {
        if self.selected_node == NodeId::ROOT || screen_delta == Vec2::ZERO {
            return;
        }

        let world_delta = [
            screen_delta.x / self.viewport_zoom,
            -screen_delta.y / self.viewport_zoom,
        ];
        let mut moved = None;
        if let Some(node) = self.project.active_scene_mut().node_mut(self.selected_node) {
            node.transform.translation[0] += world_delta[0];
            node.transform.translation[2] += world_delta[1];
            moved = Some(node.name.clone());
        }

        if let Some(name) = moved {
            self.status = format!("Moved {name}");
            self.mark_dirty();
        }
    }
}

/// Recursive directory copy. `std` doesn't ship one and we only
/// need it for the New-Project flow, so a 15-line helper is
/// preferable to taking a dep on `fs_extra`.
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn transform_editor(ui: &mut egui::Ui, label: &str, values: &mut [f32; 3], speed: f64) -> bool {
    ui.horizontal(|ui| {
        let mut changed = false;
        ui.label(icons::text(transform_icon(label), 12.0).color(STUDIO_TEXT_WEAK));
        ui.label(label);
        changed |= ui
            .add(
                egui::DragValue::new(&mut values[0])
                    .prefix("X ")
                    .speed(speed),
            )
            .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut values[1])
                    .prefix("Y ")
                    .speed(speed),
            )
            .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut values[2])
                    .prefix("Z ")
                    .speed(speed),
            )
            .changed();
        changed
    })
    .inner
}

fn transform_icon(label: &str) -> char {
    match label {
        "Position" => icons::MOVE,
        "Rotation" => icons::ROTATE_3D,
        "Scale" => icons::SCALE_3D,
        _ => icons::WAYPOINT,
    }
}

fn draw_node_kind_editor(
    ui: &mut egui::Ui,
    kind: &mut NodeKind,
    material_options: &[(ResourceId, String)],
    room_options: &[(NodeId, String)],
    model_options: &[(ResourceId, String, Vec<String>)],
    character_options: &[(ResourceId, String)],
    nav_target: &mut Option<ResourceId>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(icons::text(icons::CIRCLE_DOT, 13.0).color(STUDIO_TEXT_WEAK));
        ui.strong("Node");
    });
    match kind {
        NodeKind::Node | NodeKind::Node3D => {
            ui.weak("Organisational transform node");
        }
        NodeKind::Entity => {
            ui.weak("Entity host. Add component-node children for rendering, collision, interaction, lighting, or logic.");
        }
        NodeKind::Actor => {
            ui.weak("Actor host. Add component-node children for character controller, AI, combat, rendering, and collision.");
        }
        NodeKind::World => {
            ui.weak("Streamed-region group; holds Room children.");
        }
        NodeKind::Room { grid } => {
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::GRID, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label("Grid");
                let mut new_w = grid.width;
                let mut new_d = grid.depth;
                let w_changed = ui
                    .add(
                        egui::DragValue::new(&mut new_w)
                            .speed(0.1)
                            .range(1..=64)
                            .prefix("W "),
                    )
                    .changed();
                let d_changed = ui
                    .add(
                        egui::DragValue::new(&mut new_d)
                            .speed(0.1)
                            .range(1..=64)
                            .prefix("D "),
                    )
                    .changed();
                if w_changed || d_changed {
                    grid.resize(new_w, new_d);
                    changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::WAYPOINT, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label("Sector Size");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut grid.sector_size)
                            .speed(16.0)
                            .range(64..=8192),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::BOX, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label(format!(
                    "{} populated sectors",
                    grid.populated_sector_count()
                ));
            });
            changed |= color_editor(ui, "Ambient Light", &mut grid.ambient_color);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Preset").color(STUDIO_TEXT_WEAK));
                if ui.small_button("Low").clicked() {
                    grid.ambient_color = [32, 32, 32];
                    changed = true;
                }
                if ui.small_button("Neutral").clicked() {
                    grid.ambient_color = [128, 128, 128];
                    changed = true;
                }
                if ui.small_button("Warm").clicked() {
                    grid.ambient_color = [96, 80, 64];
                    changed = true;
                }
            });
            changed |= ui
                .checkbox(&mut grid.fog_enabled, icons::label(icons::SCAN, "Fog"))
                .changed();
        }
        NodeKind::MeshInstance {
            mesh,
            material,
            animation_clip,
        } => {
            // Look up the bound model (if any) so we can show
            // a real clip-name combo for animation_clip.
            let bound_model: Option<&(ResourceId, String, Vec<String>)> =
                mesh.and_then(|id| model_options.iter().find(|(rid, _, _)| *rid == id));

            ui.horizontal(|ui| {
                ui.label(icons::text(icons::BOX, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label(match (mesh, bound_model) {
                    (Some(_), Some((_, name, _))) => format!("Model: {name}"),
                    (Some(id), None) => format!("Mesh resource #{}", id.raw()),
                    (None, _) => "No mesh resource assigned".to_string(),
                });
            });
            ui.separator();
            // Same `material_picker` the face inspector uses, so
            // the `→` jump button is available here too. (Models
            // ignore this field -- material is baked into .psxmdl.)
            changed |= material_picker(ui, "Material", material, material_options, nav_target);

            // Animation clip override. When the bound mesh is a
            // Model, render a clip-name combo so the user picks
            // by name; otherwise fall back to a numeric override
            // for legacy mesh instances.
            ui.horizontal(|ui| {
                ui.label(RichText::new("Animation clip").color(STUDIO_TEXT_WEAK));
                if let Some((_, _, clips)) = bound_model {
                    let preview = match *animation_clip {
                        Some(idx) => clips
                            .get(idx as usize)
                            .map(|n| n.as_str())
                            .unwrap_or("(invalid)")
                            .to_string(),
                        None => "(inherit default)".to_string(),
                    };
                    egui::ComboBox::from_id_salt("mesh_instance_clip")
                        .selected_text(preview)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(animation_clip.is_none(), "(inherit default)")
                                .clicked()
                            {
                                *animation_clip = None;
                                changed = true;
                            }
                            for (i, name) in clips.iter().enumerate() {
                                let label = format!("{i}: {name}");
                                if ui
                                    .selectable_label(*animation_clip == Some(i as u16), label)
                                    .clicked()
                                {
                                    *animation_clip = Some(i as u16);
                                    changed = true;
                                }
                            }
                        });
                    if let Some(idx) = *animation_clip {
                        if (idx as usize) >= clips.len() {
                            ui.colored_label(
                                Color32::from_rgb(220, 160, 80),
                                format!("clip {idx} out of range ({} clips)", clips.len()),
                            );
                        }
                    }
                } else {
                    let mut current = animation_clip.map(|i| i as i32).unwrap_or(-1);
                    let response = ui.add(
                        egui::DragValue::new(&mut current)
                            .speed(0.1)
                            .range(-1..=255)
                            .custom_formatter(|n, _| {
                                if n < 0.0 {
                                    "default".to_string()
                                } else {
                                    format!("{}", n as i32)
                                }
                            }),
                    );
                    if response.changed() {
                        *animation_clip = if current < 0 {
                            None
                        } else {
                            Some(current as u16)
                        };
                        changed = true;
                    }
                }
            });
        }
        NodeKind::ModelRenderer { model, material } => {
            ui.weak("Component: renders a Model from the parent Entity/Actor transform.");
            let bound_model =
                model.and_then(|id| model_options.iter().find(|(rid, _, _)| *rid == id));
            ui.horizontal(|ui| {
                ui.label("Model");
                let preview = bound_model
                    .map(|(_, name, _)| name.as_str())
                    .unwrap_or("(none)");
                egui::ComboBox::from_id_salt("model-renderer-model-picker")
                    .selected_text(preview)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(model.is_none(), "(none)").clicked() {
                            *model = None;
                            changed = true;
                        }
                        for (id, name, _) in model_options {
                            if ui.selectable_label(*model == Some(*id), name).clicked() {
                                *model = Some(*id);
                                changed = true;
                            }
                        }
                    });
            });
            if model.is_some() && bound_model.is_none() {
                ui.colored_label(
                    Color32::from_rgb(220, 120, 100),
                    "Model resource is missing.",
                );
            }
            changed |= material_picker(ui, "Material", material, material_options, nav_target);
        }
        NodeKind::Animator { clip, autoplay } => {
            ui.weak("Component: controls which model animation clip plays on this entity.");
            changed |= ui
                .checkbox(autoplay, icons::label(icons::PLAY, "Autoplay"))
                .changed();
            let mut current = clip.map(|i| i as i32).unwrap_or(-1);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Clip override").color(STUDIO_TEXT_WEAK));
                let response = ui.add(
                    egui::DragValue::new(&mut current)
                        .speed(0.1)
                        .range(-1..=255)
                        .custom_formatter(|n, _| {
                            if n < 0.0 {
                                "inherit".to_string()
                            } else {
                                format!("{}", n as i32)
                            }
                        }),
                );
                if response.changed() {
                    *clip = if current < 0 {
                        None
                    } else {
                        Some(current as u16)
                    };
                    changed = true;
                }
            });
        }
        NodeKind::Collider { shape, solid } => {
            ui.weak("Component: collision authored on an Entity/Actor. Runtime prop collision is not cooked yet.");
            changed |= ui.checkbox(solid, "Solid").changed();
            changed |= collider_shape_editor(ui, shape);
        }
        NodeKind::Interactable { prompt, action } => {
            ui.weak("Component: marks a prop/entity as interactable. Runtime interaction cooking lands later.");
            ui.horizontal(|ui| {
                ui.label("Prompt");
                changed |= ui.text_edit_singleline(prompt).changed();
            });
            ui.horizontal(|ui| {
                ui.label("Action");
                changed |= ui.text_edit_singleline(action).changed();
            });
        }
        NodeKind::CharacterController { character, player } => {
            ui.weak("Component: binds an Actor to a Character profile. Player actors cook into the current playtest controller.");
            changed |= ui
                .checkbox(player, icons::label(icons::MAP_PIN, "Player controlled"))
                .changed();
            changed |= draw_character_selector(ui, character_options, character);
        }
        NodeKind::AiController { behavior } => {
            ui.weak("Component: future NPC/enemy AI profile.");
            ui.horizontal(|ui| {
                ui.label("Behavior");
                changed |= ui.text_edit_singleline(behavior).changed();
            });
        }
        NodeKind::Combat { faction, health } => {
            ui.weak("Component: future combat stats.");
            ui.horizontal(|ui| {
                ui.label("Faction");
                changed |= ui.text_edit_singleline(faction).changed();
            });
            changed |= drag_u16(ui, "Health", health, 0, u16::MAX);
        }
        NodeKind::Light {
            color,
            intensity,
            radius,
        } => {
            changed |= color_editor(ui, "Color", color);
            changed |= ui
                .add(
                    egui::Slider::new(intensity, 0.0..=4.0)
                        .text(icons::label(icons::SUN, "Intensity (× 1.0)")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(radius, 0.0..=8.0)
                        .text(icons::label(icons::WAYPOINT, "Radius (sectors)")),
                )
                .changed();
            // Quick presets -- author-friendly starting points;
            // the user can still drag the sliders below.
            ui.horizontal(|ui| {
                ui.label(RichText::new("Preset").color(STUDIO_TEXT_WEAK));
                if ui.small_button("Torch").clicked() {
                    *color = [0xFF, 0xCC, 0x80];
                    *intensity = 1.0;
                    *radius = 2.0;
                    changed = true;
                }
                if ui.small_button("Room fill").clicked() {
                    *color = [0xFF, 0xF0, 0xD8];
                    *intensity = 0.6;
                    *radius = 4.0;
                    changed = true;
                }
                if ui.small_button("Bright sun").clicked() {
                    *color = [0xFF, 0xFF, 0xF0];
                    *intensity = 2.0;
                    *radius = 8.0;
                    changed = true;
                }
            });
            // Validation warnings -- match what the playtest cooker
            // refuses, so authors see the issue before they cook.
            if *radius <= 0.0 {
                ui.colored_label(
                    Color32::from_rgb(220, 120, 100),
                    "Radius must be > 0 (cook will fail)",
                );
            }
            if !intensity.is_finite() || *intensity < 0.0 {
                ui.colored_label(
                    Color32::from_rgb(220, 120, 100),
                    "Intensity must be finite and ≥ 0 (cook will fail)",
                );
            }
            if *intensity > 4.0 {
                ui.colored_label(
                    Color32::from_rgb(220, 160, 80),
                    "Intensity above 4.0 saturates almost every surface",
                );
            }
        }
        NodeKind::PointLight {
            color,
            intensity,
            radius,
        } => {
            ui.weak("Component: point light emitted from the parent Entity/Actor transform.");
            changed |= color_editor(ui, "Color", color);
            changed |= ui
                .add(
                    egui::Slider::new(intensity, 0.0..=4.0)
                        .text(icons::label(icons::SUN, "Intensity (× 1.0)")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(radius, 0.0..=8.0)
                        .text(icons::label(icons::WAYPOINT, "Radius (sectors)")),
                )
                .changed();
            if *radius <= 0.0 {
                ui.colored_label(
                    Color32::from_rgb(220, 120, 100),
                    "Radius must be > 0 (cook will fail)",
                );
            }
        }
        NodeKind::SpawnPoint { player, character } => {
            changed |= ui
                .checkbox(player, icons::label(icons::MAP_PIN, "Player spawn"))
                .changed();
            if *player {
                changed |= draw_character_selector(ui, character_options, character);
            }
        }
        NodeKind::Trigger { trigger_id } => {
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::SCAN, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label("Trigger ID");
            });
            changed |= ui.text_edit_singleline(trigger_id).changed();
        }
        NodeKind::AudioSource { sound, radius } => {
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::AUDIO_LINES, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label(match sound {
                    Some(id) => format!("Audio resource #{}", id.raw()),
                    None => "No audio resource assigned".to_string(),
                });
            });
            changed |= ui
                .add(
                    egui::Slider::new(radius, 0.0..=16000.0)
                        .text(icons::label(icons::WAYPOINT, "Radius")),
                )
                .changed();
        }
        NodeKind::Portal {
            target_room,
            target_entry,
            entry_name,
        } => {
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::WAYPOINT, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label("Entry name");
                changed |= ui.text_edit_singleline(entry_name).changed();
            });
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::HOUSE, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label("Target room");
                let preview = target_room
                    .and_then(|id| {
                        room_options
                            .iter()
                            .find(|(rid, _)| *rid == id)
                            .map(|(_, name)| name.as_str())
                    })
                    .unwrap_or("(none)");
                egui::ComboBox::from_id_salt("portal_target_room")
                    .selected_text(preview)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(target_room.is_none(), "(none)")
                            .clicked()
                        {
                            *target_room = None;
                            changed = true;
                        }
                        for (id, name) in room_options {
                            if ui
                                .selectable_label(*target_room == Some(*id), name)
                                .clicked()
                            {
                                *target_room = Some(*id);
                                changed = true;
                            }
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::MAP_PIN, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label("Target entry");
                changed |= ui.text_edit_singleline(target_entry).changed();
            });
        }
    }
    changed
}

fn blend_mode_editor(ui: &mut egui::Ui, mode: &mut PsxBlendMode) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(icons::text(icons::BLEND, 12.0).color(STUDIO_TEXT_WEAK));
        ui.label("Blend mode");
    });
    for candidate in [
        PsxBlendMode::Opaque,
        PsxBlendMode::Average,
        PsxBlendMode::Add,
        PsxBlendMode::Subtract,
        PsxBlendMode::AddQuarter,
    ] {
        if ui
            .selectable_label(*mode == candidate, candidate.label())
            .clicked()
            && *mode != candidate
        {
            *mode = candidate;
            changed = true;
        }
    }
    changed
}

fn color_editor(ui: &mut egui::Ui, label: &str, color: &mut [u8; 3]) -> bool {
    ui.horizontal(|ui| {
        let mut changed = false;
        ui.label(icons::text(icons::PALETTE, 12.0).color(STUDIO_TEXT_WEAK));
        ui.label(label);
        changed |= ui.color_edit_button_srgb(color).changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut color[0])
                    .prefix("R ")
                    .range(0..=255),
            )
            .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut color[1])
                    .prefix("G ")
                    .range(0..=255),
            )
            .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut color[2])
                    .prefix("B ")
                    .range(0..=255),
            )
            .changed();
        changed
    })
    .inner
}

fn short_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.display().to_string())
}

fn node_lucide_icon(kind: &str, root: bool) -> char {
    if root {
        return icons::HOUSE;
    }

    match kind {
        "Node3D" => icons::CIRCLE_DOT,
        "Entity" => icons::BOX,
        "Actor" => icons::MAP_PIN,
        "World" => icons::HOUSE,
        "Room" => icons::GRID,
        "MeshInstance" => icons::BOX,
        "ModelRenderer" => icons::BOX,
        "Animator" => icons::PLAY,
        "Collider" => icons::SCALE_3D,
        "Interactable" => icons::POINTER,
        "CharacterController" => icons::MAP_PIN,
        "AiController" => icons::SCAN,
        "Combat" => icons::FOCUS,
        "Light" => icons::SUN,
        "PointLight" => icons::SUN,
        "SpawnPoint" => icons::MAP_PIN,
        "Trigger" => icons::SCAN,
        "AudioSource" => icons::AUDIO_LINES,
        "Portal" => icons::WAYPOINT,
        _ => icons::CIRCLE_DOT,
    }
}

fn node_lucide_color(kind: &str, root: bool, selected: bool) -> Color32 {
    if selected {
        return Color32::WHITE;
    }
    if root {
        return STUDIO_ACCENT;
    }

    match kind {
        "Entity" => Color32::from_rgb(156, 174, 190),
        "Actor" => Color32::from_rgb(110, 190, 142),
        "World" => Color32::from_rgb(232, 152, 96),
        "Room" => Color32::from_rgb(209, 118, 71),
        "MeshInstance" => Color32::from_rgb(156, 174, 190),
        "ModelRenderer" => Color32::from_rgb(134, 168, 196),
        "Animator" => Color32::from_rgb(126, 164, 220),
        "Collider" => Color32::from_rgb(180, 170, 112),
        "Interactable" => Color32::from_rgb(216, 160, 108),
        "CharacterController" => Color32::from_rgb(104, 194, 142),
        "AiController" => Color32::from_rgb(144, 176, 112),
        "Combat" => Color32::from_rgb(220, 110, 110),
        "Light" => Color32::from_rgb(238, 203, 116),
        "PointLight" => Color32::from_rgb(238, 203, 116),
        "SpawnPoint" => Color32::from_rgb(236, 188, 104),
        "Trigger" => Color32::from_rgb(190, 128, 232),
        "AudioSource" => Color32::from_rgb(104, 202, 188),
        "Portal" => Color32::from_rgb(255, 188, 100),
        _ => Color32::from_rgb(141, 160, 180),
    }
}

fn draw_inline_icon(ui: &mut egui::Ui, icon: char, color: Color32) {
    ui.label(icons::text(icon, 16.0).color(color));
}

fn draw_model_import_preview(
    ui: &mut egui::Ui,
    preview: &mut ModelImportPreview,
    selected_clip: &mut usize,
    preview_yaw_q12: &mut i32,
    preview_pitch_q12: &mut i32,
    preview_radius: &mut i32,
    show_animation_root: bool,
) {
    ui.label(RichText::new("Cooked Model").strong());
    if !draw_model_animated_import_preview(
        ui,
        preview,
        *selected_clip,
        preview_yaw_q12,
        preview_pitch_q12,
        preview_radius,
        show_animation_root,
    ) {
        draw_model_wireframe_preview(ui, &preview.model_bytes);
    }

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(RichText::new("Atlas").strong());
            match &preview.atlas {
                Some((handle, stats)) => {
                    draw_psxt_preview_block(ui, Some((handle.id(), *stats)));
                }
                None => {
                    draw_psxt_preview_block(ui, None);
                }
            }
        });
    });

    ui.separator();
    egui::Grid::new("model-import-stats")
        .num_columns(4)
        .spacing([10.0, 3.0])
        .show(ui, |ui| {
            stat_cell(ui, "Source verts", preview.report.source_vertices);
            stat_cell(ui, "Cooked verts", preview.report.cooked_vertices);
            ui.end_row();
            stat_cell(ui, "Faces", preview.report.faces);
            stat_cell(ui, "Parts", preview.report.parts);
            ui.end_row();
            stat_cell(ui, "Joints", preview.report.joints);
            stat_cell(ui, "Local height", preview.report.local_height);
            ui.end_row();
            stat_cell(ui, "Model bytes", preview.report.model_bytes);
            stat_cell(ui, "Anim bytes", preview.report.animation_bytes);
            ui.end_row();
        });

    ui.separator();
    ui.label(RichText::new("Baked Animation Clips").strong());
    if preview.clips.is_empty() {
        ui.weak("No animation clips found in the source.");
        return;
    }
    *selected_clip = (*selected_clip).min(preview.clips.len().saturating_sub(1));
    egui::ScrollArea::vertical()
        .max_height(150.0)
        .show(ui, |ui| {
            for (index, clip) in preview.clips.iter().enumerate() {
                let root = clip
                    .root_motion
                    .map(root_motion_brief)
                    .unwrap_or_else(|| "root n/a".to_string());
                let label = format!(
                    "{}  ·  {} frames  ·  {}  ·  {}",
                    clip.name,
                    clip.frames,
                    human_bytes(clip.byte_len as u32),
                    root
                );
                if ui
                    .selectable_label(*selected_clip == index, label)
                    .clicked()
                {
                    *selected_clip = index;
                }
            }
        });
    if let Some(clip) = preview.clips.get(*selected_clip) {
        egui::CollapsingHeader::new(icons::label(icons::MOVE, "Root Motion"))
            .default_open(true)
            .show(ui, |ui| match clip.root_motion {
                Some(stats) => draw_root_motion_stats(ui, stats),
                None => {
                    ui.weak("Clip could not be parsed for root-motion stats.");
                }
            });
    }
}

fn draw_model_animated_import_preview(
    ui: &mut egui::Ui,
    preview: &mut ModelImportPreview,
    selected_clip: usize,
    preview_yaw_q12: &mut i32,
    preview_pitch_q12: &mut i32,
    preview_radius: &mut i32,
    show_animation_root: bool,
) -> bool {
    let Some(atlas) = preview.atlas_image.as_ref() else {
        return false;
    };
    let Some(clip) = preview.clips.get(selected_clip) else {
        return false;
    };

    let width = ui.available_width().clamp(560.0, 820.0);
    let height = width
        * (model_import_preview::PREVIEW_HEIGHT as f32
            / model_import_preview::PREVIEW_WIDTH as f32);
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::drag());
    if response.dragged() {
        let delta = ui.input(|i| i.pointer.delta());
        *preview_yaw_q12 = (*preview_yaw_q12 + (delta.x * 6.0) as i32).rem_euclid(4096);
        *preview_pitch_q12 = (*preview_pitch_q12 + (delta.y * 4.0) as i32).clamp(64, 960);
    }
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }

    let options = model_import_preview::ImportPreviewOptions {
        world_height: preview.world_height,
        time_seconds: ui.input(|i| i.time),
        yaw_q12: (*preview_yaw_q12).rem_euclid(4096) as u16,
        pitch_q12: (*preview_pitch_q12).rem_euclid(4096) as u16,
        radius: *preview_radius,
        show_animation_root,
    };
    let Some(image) = model_import_preview::render_import_model_preview_with_options(
        &preview.model_bytes,
        &clip.bytes,
        atlas,
        options,
    ) else {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, STUDIO_PANEL);
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "preview failed",
            FontId::proportional(12.0),
            Color32::from_rgb(220, 120, 100),
        );
        return true;
    };

    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(33));

    let texture_id = match &mut preview.animated_texture {
        Some(handle) => {
            handle.set(image, egui::TextureOptions::NEAREST);
            handle.id()
        }
        None => {
            let handle = ui.ctx().load_texture(
                "model-import-animated-preview",
                image,
                egui::TextureOptions::NEAREST,
            );
            let id = handle.id();
            preview.animated_texture = Some(handle);
            id
        }
    };

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, STUDIO_PANEL);
    painter.image(
        texture_id,
        rect,
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
    painter.rect_stroke(
        rect,
        4.0,
        Stroke::new(1.0, STUDIO_BORDER),
        StrokeKind::Inside,
    );
    true
}

fn stat_cell(ui: &mut egui::Ui, label: &str, value: usize) {
    ui.label(RichText::new(label).color(STUDIO_TEXT_WEAK).small());
    ui.label(RichText::new(value.to_string()).monospace());
}

fn draw_model_wireframe_preview(ui: &mut egui::Ui, model_bytes: &[u8]) {
    let width = ui.available_width().min(520.0).max(280.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 280.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, STUDIO_PANEL);
    painter.rect_stroke(
        rect,
        4.0,
        Stroke::new(1.0, STUDIO_BORDER),
        StrokeKind::Inside,
    );

    let Ok(model) = psx_asset::Model::from_bytes(model_bytes) else {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "model parse failed",
            FontId::proportional(12.0),
            Color32::from_rgb(220, 120, 100),
        );
        return;
    };

    let mut projected = Vec::with_capacity(model.vertex_count() as usize);
    let mut min = [f32::INFINITY, f32::INFINITY];
    let mut max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    for i in 0..model.vertex_count() {
        let Some(vertex) = model.vertex(i) else {
            projected.push([0.0, 0.0]);
            continue;
        };
        let x = vertex.position.x as f32;
        let y = vertex.position.y as f32;
        let z = vertex.position.z as f32;
        // Lightweight isometric-ish preview: no renderer, just enough
        // shape to confirm centering, scale, and triangle continuity.
        let p = [x - z * 0.45, -y + z * 0.22];
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
        projected.push(p);
    }
    let span_x = (max[0] - min[0]).max(1.0);
    let span_y = (max[1] - min[1]).max(1.0);
    let scale = ((rect.width() - 28.0) / span_x)
        .min((rect.height() - 28.0) / span_y)
        .max(0.001);
    let to_screen = |p: [f32; 2]| -> Pos2 {
        Pos2::new(
            rect.center().x + (p[0] - (min[0] + max[0]) * 0.5) * scale,
            rect.center().y + (p[1] - (min[1] + max[1]) * 0.5) * scale,
        )
    };

    let face_count = model.face_count();
    let stride = ((face_count as usize) / 900).max(1);
    for face_index in (0..face_count).step_by(stride) {
        let Some(face) = model.face(face_index) else {
            continue;
        };
        let a = projected
            .get(face.corners[0].vertex_index as usize)
            .copied();
        let b = projected
            .get(face.corners[1].vertex_index as usize)
            .copied();
        let c = projected
            .get(face.corners[2].vertex_index as usize)
            .copied();
        let (Some(a), Some(b), Some(c)) = (a, b, c) else {
            continue;
        };
        let stroke = Stroke::new(1.0, Color32::from_rgb(150, 170, 185));
        let pa = to_screen(a);
        let pb = to_screen(b);
        let pc = to_screen(c);
        painter.line_segment([pa, pb], stroke);
        painter.line_segment([pb, pc], stroke);
        painter.line_segment([pc, pa], stroke);
    }
}

fn root_motion_stats(bytes: &[u8], joint_index: u16) -> Option<RootMotionStats> {
    let anim = psx_asset::Animation::from_bytes(bytes).ok()?;
    if joint_index >= anim.joint_count() {
        return None;
    }
    let mut min = [i32::MAX; 3];
    let mut max = [i32::MIN; 3];
    let mut sum = [0i64; 3];
    let mut count = 0i64;
    for frame in 0..anim.frame_count() {
        let pose = anim.pose(frame, joint_index)?;
        let values = [pose.translation.x, pose.translation.y, pose.translation.z];
        for axis in 0..3 {
            min[axis] = min[axis].min(values[axis]);
            max[axis] = max[axis].max(values[axis]);
            sum[axis] += values[axis] as i64;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some(RootMotionStats {
        min,
        max,
        mean: [
            (sum[0] / count) as i32,
            (sum[1] / count) as i32,
            (sum[2] / count) as i32,
        ],
    })
}

fn root_motion_brief(stats: RootMotionStats) -> String {
    let span_x = stats.max[0].saturating_sub(stats.min[0]).abs();
    let span_y = stats.max[1].saturating_sub(stats.min[1]).abs();
    let span_z = stats.max[2].saturating_sub(stats.min[2]).abs();
    format!("root span {span_x}/{span_y}/{span_z}")
}

fn draw_root_motion_stats(ui: &mut egui::Ui, stats: RootMotionStats) {
    egui::Grid::new("model-import-root-motion")
        .num_columns(4)
        .spacing([8.0, 3.0])
        .show(ui, |ui| {
            ui.label("");
            ui.label(RichText::new("min").color(STUDIO_TEXT_WEAK).small());
            ui.label(RichText::new("max").color(STUDIO_TEXT_WEAK).small());
            ui.label(RichText::new("mean").color(STUDIO_TEXT_WEAK).small());
            ui.end_row();
            for (axis, name) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*name);
                ui.label(RichText::new(stats.min[axis].to_string()).monospace());
                ui.label(RichText::new(stats.max[axis].to_string()).monospace());
                ui.label(RichText::new(stats.mean[axis].to_string()).monospace());
                ui.end_row();
            }
        });
    ui.label(
        RichText::new("Values are cooked Q12 pose-translation units for root joint 0.")
            .color(STUDIO_TEXT_WEAK)
            .small(),
    );
}

/// Inspector preview header: a 128×128 image of the linked PSXT
/// (centered, NEAREST-sampled so individual texels are visible at
/// editor scale) above a one-line summary. Falls back to a
/// "no preview" placeholder when the resource has no decoded
/// thumbnail (missing path / unreadable / unsupported depth).
fn draw_psxt_preview_block(ui: &mut egui::Ui, thumb: Option<(egui::TextureId, PsxtStats)>) {
    let preview_size = Vec2::splat(128.0);
    ui.vertical_centered(|ui| match thumb {
        Some((id, stats)) => {
            let (rect, _) = ui.allocate_exact_size(preview_size, Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 4.0, STUDIO_PANEL);
            painter.image(
                id,
                rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
            painter.rect_stroke(
                rect,
                4.0,
                Stroke::new(1.0, STUDIO_BORDER),
                StrokeKind::Inside,
            );
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!(
                    "{}×{}  {}bpp  {}",
                    stats.width,
                    stats.height,
                    stats.depth_bits,
                    human_bytes(stats.file_bytes)
                ))
                .color(STUDIO_TEXT_WEAK)
                .small(),
            );
        }
        None => {
            let (rect, _) = ui.allocate_exact_size(preview_size, Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 4.0, STUDIO_PANEL);
            painter.rect_stroke(
                rect,
                4.0,
                Stroke::new(1.0, STUDIO_BORDER),
                StrokeKind::Inside,
            );
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "no preview",
                FontId::proportional(11.0),
                STUDIO_TEXT_WEAK,
            );
        }
    });
    ui.add_space(6.0);
}

/// Tabular `key -- value` rows summarizing a `.psxt`. Mirrors the
/// fields the cooker writes so authors can sanity-check that their
/// material's texture lines up with the dimensions they expect.
fn draw_psxt_stats(ui: &mut egui::Ui, stats: PsxtStats) {
    let row = |ui: &mut egui::Ui, key: &str, value: String| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(key).color(STUDIO_TEXT_WEAK));
            ui.label(RichText::new(value).monospace());
        });
    };
    row(ui, "Size", format!("{}×{} px", stats.width, stats.height));
    row(
        ui,
        "Depth",
        match stats.depth_bits {
            4 => "4bpp indexed (16-color CLUT)".to_string(),
            8 => "8bpp indexed (256-color CLUT)".to_string(),
            15 => "15bpp direct".to_string(),
            other => format!("{other}bpp (?)"),
        },
    );
    row(ui, "CLUT entries", format!("{}", stats.clut_entries));
    row(ui, "Pixel data", human_bytes(stats.pixel_bytes));
    if stats.clut_bytes > 0 {
        row(ui, "CLUT data", human_bytes(stats.clut_bytes));
    }
    row(ui, "File total", human_bytes(stats.file_bytes));
}

/// Inspector for a [`ResourceData::Model`]. Lets the user edit
/// the model + atlas paths, manage the clip list, choose
/// preview / default clips, and view parsed model + clip
/// statistics. The "Register Cooked Folder" / "Import GLB"
/// helpers run via deferred actions stored in
/// [`ModelInspectorAction`] so the caller can apply them after
/// dropping the mutable resource borrow.
fn draw_model_resource_editor(
    ui: &mut egui::Ui,
    model: &mut psxed_project::ModelResource,
    project_root: &Path,
    preview_thumb: Option<(egui::TextureId, PsxtStats)>,
) -> bool {
    let mut changed = false;

    // Atlas thumbnail block: same panel the Texture inspector
    // uses, but driven from `model.texture_path` via the shared
    // thumbnail cache that already learned about Model atlases.
    if preview_thumb.is_some() {
        draw_psxt_preview_block(ui, preview_thumb);
    }

    egui::CollapsingHeader::new(icons::label(icons::FOLDER, "Bundle helpers"))
        .default_open(false)
        .show(ui, |ui| {
            ui.label(
                RichText::new(
                    "Register a cooked bundle (folder with one .psxmdl, optional .psxt, any number of .psxanim). Paths and clip metadata fill in automatically. Bundle dir resolves against the project root.",
                )
                .color(STUDIO_TEXT_WEAK)
                .small(),
            );
            // egui memory keeps the input + last status across
            // frames without leaking into ModelResource itself.
            let input_id = ui.id().with("model_bundle_input");
            let status_id = ui.id().with("model_bundle_status");
            let mut bundle_dir: String = ui
                .memory_mut(|m| m.data.get_persisted::<String>(input_id))
                .unwrap_or_default();
            ui.horizontal(|ui| {
                ui.label("Bundle dir");
                ui.text_edit_singleline(&mut bundle_dir);
            });
            ui.memory_mut(|m| m.data.insert_persisted(input_id, bundle_dir.clone()));

            if ui
                .button(icons::label(icons::PLUS, "Register Cooked Folder"))
                .on_hover_text(
                    "Walks the directory, validates every blob, and replaces this Model's paths + clip list with the bundle contents.",
                )
                .clicked()
                && !bundle_dir.is_empty()
            {
                let path = if Path::new(&bundle_dir).is_absolute() {
                    PathBuf::from(&bundle_dir)
                } else {
                    project_root.join(&bundle_dir)
                };
                let new_status = match register_bundle_into_model(model, &path, project_root) {
                    Ok(clip_count) => {
                        changed = true;
                        format!("Registered: {clip_count} clip(s)")
                    }
                    Err(e) => format!("Failed: {e}"),
                };
                ui.memory_mut(|m| m.data.insert_persisted(status_id, new_status));
            }

            let status: String = ui
                .memory_mut(|m| m.data.get_persisted::<String>(status_id))
                .unwrap_or_default();
            if !status.is_empty() {
                let color = if status.starts_with("Failed") {
                    Color32::from_rgb(220, 120, 100)
                } else {
                    STUDIO_TEXT_WEAK
                };
                ui.colored_label(color, status);
            }

            ui.label(
                RichText::new(
                    "Use Resources -> Import Model for GLB/glTF preview, root-centering, and bundle import.",
                )
                .color(STUDIO_TEXT_WEAK)
                .small(),
            );
        });

    egui::CollapsingHeader::new(icons::label(icons::BOX, "Model"))
        .default_open(true)
        .show(ui, |ui| {
            ui.label("Cooked .psxmdl path");
            changed |= ui.text_edit_singleline(&mut model.model_path).changed();

            ui.add_space(4.0);
            ui.label("Atlas .psxt path (optional)");
            let mut atlas = model.texture_path.clone().unwrap_or_default();
            let atlas_response = ui.text_edit_singleline(&mut atlas);
            if atlas_response.changed() {
                model.texture_path = if atlas.is_empty() { None } else { Some(atlas) };
                changed = true;
            }

            ui.add_space(4.0);
            ui.label("World height (engine units)");
            let mut h = model.world_height as i32;
            let h_response = ui.add(egui::DragValue::new(&mut h).speed(8.0).range(0..=4096));
            if h_response.changed() {
                model.world_height = h.clamp(0, u16::MAX as i32) as u16;
                changed = true;
            }
        });

    // Live-parse the model + clips for stats. Cheap on every
    // inspector frame for the model sizes this editor targets;
    // a cache lands when authoring scales beyond that.
    let model_path =
        psxed_project::model_import::resolve_path(&model.model_path, Some(project_root));
    let model_stats = std::fs::read(&model_path)
        .ok()
        .and_then(|b| psxed_project::model_import::model_stats_from_bytes(&b).ok());
    if let Some(stats) = &model_stats {
        egui::CollapsingHeader::new(icons::label(icons::SCAN, "Stats"))
            .default_open(true)
            .show(ui, |ui| {
                let row = |ui: &mut egui::Ui, key: &str, value: String| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(key).color(STUDIO_TEXT_WEAK));
                        ui.label(RichText::new(value).monospace());
                    });
                };
                row(ui, "Joints", format!("{}", stats.joint_count));
                row(ui, "Parts", format!("{}", stats.part_count));
                row(ui, "Vertices", format!("{}", stats.vertex_count));
                row(ui, "Faces", format!("{}", stats.face_count));
                row(ui, "Materials", format!("{}", stats.material_count));
                row(
                    ui,
                    "Atlas (header)",
                    format!("{}×{}", stats.texture_width, stats.texture_height),
                );
                row(
                    ui,
                    "Bounds X",
                    format!("{}..{}", stats.bounds_min[0], stats.bounds_max[0]),
                );
                row(
                    ui,
                    "Bounds Y",
                    format!("{}..{}", stats.bounds_min[1], stats.bounds_max[1]),
                );
                row(
                    ui,
                    "Bounds Z",
                    format!("{}..{}", stats.bounds_min[2], stats.bounds_max[2]),
                );
                row(ui, "Model bytes", format!("{}", stats.model_bytes));
            });
    } else if !model.model_path.is_empty() {
        ui.colored_label(
            Color32::from_rgb(220, 120, 100),
            format!("Failed to parse model at {}", model_path.display()),
        );
    }

    egui::CollapsingHeader::new(icons::label(icons::PLAY, "Animation Clips"))
        .default_open(true)
        .show(ui, |ui| {
            // Walk a copy of the indices so we can offer add /
            // remove buttons without confusing the borrow checker.
            let mut to_remove: Option<usize> = None;
            for (i, clip) in model.clips.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("#{i}")).color(STUDIO_TEXT_WEAK));
                    if ui.text_edit_singleline(&mut clip.name).changed() {
                        changed = true;
                    }
                    if ui.small_button(icons::label(icons::TRASH, "")).clicked() {
                        to_remove = Some(i);
                    }
                });
                ui.label(RichText::new("Path").color(STUDIO_TEXT_WEAK).small());
                if ui.text_edit_singleline(&mut clip.psxanim_path).changed() {
                    changed = true;
                }
                // Inline per-clip stats: parse on the fly. Joint
                // mismatch is the most actionable thing to surface
                // -- the cooker rejects the bundle if it persists.
                if let Some(model_stats) = &model_stats {
                    let clip_path = psxed_project::model_import::resolve_path(
                        &clip.psxanim_path,
                        Some(project_root),
                    );
                    if let Ok(bytes) = std::fs::read(&clip_path) {
                        match psxed_project::model_import::animation_stats_from_bytes(
                            &clip.name,
                            &bytes,
                            model_stats.joint_count,
                        ) {
                            Ok(stats) => {
                                let label = format!(
                                    "{} frames @ {} Hz, {} joints",
                                    stats.frame_count, stats.sample_rate_hz, stats.joint_count,
                                );
                                if stats.valid_for_model {
                                    ui.label(RichText::new(label).color(STUDIO_TEXT_WEAK).small());
                                } else {
                                    ui.colored_label(
                                        Color32::from_rgb(220, 160, 80),
                                        format!(
                                            "{label} — joint mismatch (model {})",
                                            model_stats.joint_count
                                        ),
                                    );
                                }
                            }
                            Err(e) => {
                                ui.colored_label(
                                    Color32::from_rgb(220, 120, 100),
                                    format!("parse failed: {e}"),
                                );
                            }
                        }
                    }
                }
                ui.add_space(2.0);
            }
            if let Some(i) = to_remove {
                model.clips.remove(i);
                // Drop indices that referenced the removed clip
                // or anything past it so they stay valid.
                if let Some(d) = model.default_clip {
                    model.default_clip = match (d as usize).cmp(&i) {
                        std::cmp::Ordering::Less => Some(d),
                        std::cmp::Ordering::Equal => None,
                        std::cmp::Ordering::Greater => Some(d.saturating_sub(1)),
                    };
                }
                if let Some(p) = model.preview_clip {
                    model.preview_clip = match (p as usize).cmp(&i) {
                        std::cmp::Ordering::Less => Some(p),
                        std::cmp::Ordering::Equal => None,
                        std::cmp::Ordering::Greater => Some(p.saturating_sub(1)),
                    };
                }
                changed = true;
            }
            if ui.button(icons::label(icons::PLUS, "Add clip")).clicked() {
                model.clips.push(psxed_project::ModelAnimationClip {
                    name: format!("clip_{}", model.clips.len()),
                    psxanim_path: String::new(),
                });
                changed = true;
            }

            ui.separator();
            changed |= clip_picker(ui, "Default clip", &mut model.default_clip, &model.clips);
            changed |= clip_picker(ui, "Preview clip", &mut model.preview_clip, &model.clips);
        });

    changed
}

/// Adapt `psxed_project::model_import::register_cooked_model_bundle`
/// to in-place editing: build a fresh ModelResource from the
/// bundle, then overwrite `target`. Returns the registered clip
/// count so the UI status line can confirm what landed.
fn register_bundle_into_model(
    target: &mut psxed_project::ModelResource,
    bundle_dir: &Path,
    project_root: &Path,
) -> Result<usize, String> {
    // The library helper takes a `&mut ProjectDocument` and
    // pushes a *new* resource. Here we want to overwrite the
    // existing resource's payload in place. Easiest path:
    // build a throwaway scratch project, register into it, then
    // copy the produced ModelResource back over `target`.
    let mut scratch = psxed_project::ProjectDocument::new("scratch");
    let id = psxed_project::model_import::register_cooked_model_bundle(
        &mut scratch,
        bundle_dir,
        "Scratch",
        Some(project_root),
    )
    .map_err(|e| e.to_string())?;
    let resource = scratch.resource(id).ok_or_else(|| "lost id".to_string())?;
    let psxed_project::ResourceData::Model(model) = &resource.data else {
        return Err("scratch resource is not a Model".to_string());
    };
    let clip_count = model.clips.len();
    *target = model.clone();
    Ok(clip_count)
}

/// Combo-box picker for a clip index. `None` means "unset" and
/// shows as `(none)`.
fn clip_picker(
    ui: &mut egui::Ui,
    label: &str,
    current: &mut Option<u16>,
    clips: &[psxed_project::ModelAnimationClip],
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(STUDIO_TEXT_WEAK));
        let preview = match *current {
            Some(i) => clips
                .get(i as usize)
                .map(|c| c.name.as_str())
                .unwrap_or("(invalid)")
                .to_string(),
            None => "(none)".to_string(),
        };
        egui::ComboBox::from_id_salt(label)
            .selected_text(preview)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "(none)").clicked() {
                    *current = None;
                    changed = true;
                }
                for (i, clip) in clips.iter().enumerate() {
                    let label = format!("{}: {}", i, clip.name);
                    if ui
                        .selectable_label(*current == Some(i as u16), label)
                        .clicked()
                    {
                        *current = Some(i as u16);
                        changed = true;
                    }
                }
            });
    });
    changed
}

/// Combo-box picker for a Material's linked texture. `current` is
/// the live `material.texture` field; `options` is every Texture
/// resource in the project. Returns true when the selection moved
/// One segment of the inspector breadcrumb.
///
/// Rendered as plain bold text when `nav` is `None` (the current
/// view, no click target) or as a clickable link otherwise -- a
/// click fires the deferred jump-to that the inspector applies
/// once its mutable borrows release.
struct BreadcrumbCrumb {
    label: String,
    nav: Option<ResourceId>,
}

/// Render an inspector breadcrumb: `Face › Material › Texture`
/// (or any subset). Click any link-style crumb to jump.
fn draw_breadcrumb(
    ui: &mut egui::Ui,
    crumbs: &[BreadcrumbCrumb],
    nav_target: &mut Option<ResourceId>,
) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for (i, crumb) in crumbs.iter().enumerate() {
            if i > 0 {
                ui.label(RichText::new("›").color(STUDIO_TEXT_WEAK));
            }
            match crumb.nav {
                None => {
                    // Current view -- non-interactive, slightly
                    // brighter so the eye lands on "where I am".
                    ui.label(RichText::new(&crumb.label).strong());
                }
                Some(id) => {
                    if ui.link(&crumb.label).clicked() {
                        *nav_target = Some(id);
                    }
                }
            }
        }
    });
}

/// Texture-picker dropdown for the material inspector.
///
/// `jump_to` is an out-param: when the user clicks the `→`
/// button next to the dropdown the picker writes the texture's
/// resource id into it, and the calling inspector applies the
/// navigation after its borrows release. Returns `true` if the
/// picker changed `current`.
fn material_texture_picker(
    ui: &mut egui::Ui,
    current: &mut Option<ResourceId>,
    options: &[(ResourceId, String)],
    jump_to: &mut Option<ResourceId>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Texture");
        let preview = current
            .and_then(|id| {
                options
                    .iter()
                    .find(|(rid, _)| *rid == id)
                    .map(|(_, n)| n.as_str())
            })
            .unwrap_or("(none)");
        egui::ComboBox::from_id_salt("material-texture-picker")
            .selected_text(preview)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "(none)").clicked() {
                    *current = None;
                    changed = true;
                }
                for (id, name) in options {
                    if ui.selectable_label(*current == Some(*id), name).clicked() {
                        *current = Some(*id);
                        changed = true;
                    }
                }
            });
        if let Some(id) = *current {
            if ui
                .small_button("→")
                .on_hover_text("Open this texture in the inspector")
                .clicked()
            {
                *jump_to = Some(id);
            }
        }
    });
    changed
}

/// Snapshot of every Model resource and its clip names. Built
/// before the mutable borrow on a Resource so the Character
/// inspector can populate model + clip dropdowns without
/// fighting the live `&mut CharacterResource`.
struct CharacterEditorContext {
    /// `(model id, model display name, clip names in order)`.
    models: Vec<(ResourceId, String, Vec<String>)>,
}

fn build_character_editor_context(project: &ProjectDocument) -> CharacterEditorContext {
    CharacterEditorContext {
        models: collect_model_options(project),
    }
}

/// Inspector body for `ResourceData::Character`. Combines a
/// model picker, four role-clip pickers (idle / walk / run /
/// turn), capsule sizes, controller speed, and camera params.
/// `Auto Assign Clips By Name` walks the bound model's clip
/// list and matches `idle` / `walk` / `run` / `turn` substrings
/// -- case-insensitive -- into role slots.
fn draw_character_resource_editor(
    ui: &mut egui::Ui,
    character: &mut psxed_project::CharacterResource,
    ctx: &CharacterEditorContext,
) -> bool {
    let mut changed = false;

    // Resolve the bound model + clip list (if any). Used
    // throughout the inspector to surface clip names instead of
    // raw indices.
    let bound = character
        .model
        .and_then(|id| ctx.models.iter().find(|(mid, _, _)| *mid == id));

    egui::CollapsingHeader::new(icons::label(icons::BOX, "Model"))
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Model");
                let preview = bound.map(|(_, name, _)| name.as_str()).unwrap_or("(none)");
                egui::ComboBox::from_id_salt("character-model-picker")
                    .selected_text(preview)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(character.model.is_none(), "(none)")
                            .clicked()
                        {
                            character.model = None;
                            changed = true;
                        }
                        for (id, name, _) in &ctx.models {
                            if ui
                                .selectable_label(character.model == Some(*id), name)
                                .clicked()
                            {
                                character.model = Some(*id);
                                changed = true;
                            }
                        }
                    });
            });
            if character.model.is_some() && bound.is_none() {
                ui.colored_label(
                    Color32::from_rgb(220, 120, 100),
                    "Bound model resource is missing.",
                );
            }
        });

    egui::CollapsingHeader::new(icons::label(icons::PALETTE, "Animation roles"))
        .default_open(true)
        .show(ui, |ui| {
            let clips: &[String] = bound.map(|(_, _, c)| c.as_slice()).unwrap_or(&[]);
            if bound.is_none() {
                ui.colored_label(
                    STUDIO_TEXT_WEAK,
                    "Pick a Model to surface its clip list.",
                );
            }
            changed |= clip_role_picker(ui, "Idle", "character-clip-idle", &mut character.idle_clip, clips);
            changed |= clip_role_picker(ui, "Walk", "character-clip-walk", &mut character.walk_clip, clips);
            changed |= clip_role_picker(ui, "Run", "character-clip-run", &mut character.run_clip, clips);
            changed |= clip_role_picker(ui, "Turn", "character-clip-turn", &mut character.turn_clip, clips);

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let auto_enabled = !clips.is_empty();
                if ui
                    .add_enabled(
                        auto_enabled,
                        egui::Button::new(icons::label(icons::SCAN, "Auto Assign Clips By Name")),
                    )
                    .on_hover_text(
                        "Match clip names against idle/walk/run/turn substrings (case-insensitive).",
                    )
                    .clicked()
                {
                    let before = (
                        character.idle_clip,
                        character.walk_clip,
                        character.run_clip,
                        character.turn_clip,
                    );
                    auto_assign_character_clips(character, clips);
                    if (
                        character.idle_clip,
                        character.walk_clip,
                        character.run_clip,
                        character.turn_clip,
                    ) != before
                    {
                        changed = true;
                    }
                }
            });

            // Inline warnings: required roles missing, or slot
            // points past end of the model's clip list.
            let warn_idx = |label: &str, slot: Option<u16>| -> Option<String> {
                let idx = slot?;
                if (idx as usize) >= clips.len() {
                    Some(format!(
                        "{label} clip index {idx} exceeds model's clip list ({} clips).",
                        clips.len()
                    ))
                } else {
                    None
                }
            };
            for warning in [
                warn_idx("Idle", character.idle_clip),
                warn_idx("Walk", character.walk_clip),
                warn_idx("Run", character.run_clip),
                warn_idx("Turn", character.turn_clip),
            ]
            .into_iter()
            .flatten()
            {
                ui.colored_label(Color32::from_rgb(220, 160, 80), warning);
            }
            if character.model.is_some() && character.idle_clip.is_none() {
                ui.colored_label(
                    Color32::from_rgb(220, 120, 100),
                    "Idle clip is required for the player character.",
                );
            }
            if character.model.is_some() && character.walk_clip.is_none() {
                ui.colored_label(
                    Color32::from_rgb(220, 120, 100),
                    "Walk clip is required for the player character.",
                );
            }
        });

    egui::CollapsingHeader::new(icons::label(icons::SCAN, "Capsule"))
        .default_open(false)
        .show(ui, |ui| {
            changed |= drag_u16(ui, "Radius", &mut character.radius, 1, 4096);
            changed |= drag_u16(ui, "Height", &mut character.height, 1, 8192);
        });

    egui::CollapsingHeader::new(icons::label(icons::LAYERS, "Controller"))
        .default_open(false)
        .show(ui, |ui| {
            changed |= drag_i32(ui, "Walk speed", &mut character.walk_speed, 1, 1024);
            changed |= drag_i32(ui, "Run speed", &mut character.run_speed, 1, 2048);
            changed |= drag_u16(
                ui,
                "Turn speed (deg/s)",
                &mut character.turn_speed_degrees_per_second,
                1,
                720,
            );
        });

    egui::CollapsingHeader::new(icons::label(icons::GRID, "Camera"))
        .default_open(false)
        .show(ui, |ui| {
            changed |= drag_i32(ui, "Distance", &mut character.camera_distance, 1, 16384);
            changed |= drag_i32(ui, "Height", &mut character.camera_height, 0, 16384);
            changed |= drag_i32(
                ui,
                "Target height",
                &mut character.camera_target_height,
                0,
                16384,
            );
        });

    changed
}

/// Helper: clip dropdown for one animation role. Renders the
/// clip's display name, falls back to "(none)" when unset, and
/// flags out-of-range indices in red.
fn clip_role_picker(
    ui: &mut egui::Ui,
    label: &str,
    id_salt: &str,
    slot: &mut Option<u16>,
    clips: &[String],
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let preview = match *slot {
            Some(idx) => clips
                .get(idx as usize)
                .cloned()
                .unwrap_or_else(|| format!("#{idx} (missing)")),
            None => "(none)".to_string(),
        };
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(preview)
            .show_ui(ui, |ui| {
                if ui.selectable_label(slot.is_none(), "(none)").clicked() {
                    *slot = None;
                    changed = true;
                }
                for (idx, name) in clips.iter().enumerate() {
                    let label = format!("{idx}: {name}");
                    if ui
                        .selectable_label(*slot == Some(idx as u16), label)
                        .clicked()
                    {
                        *slot = Some(idx as u16);
                        changed = true;
                    }
                }
            });
    });
    changed
}

/// Heuristic clip-by-name auto-assignment. Case-insensitive
/// substring match. Leaves missing roles unset so the validation
/// warnings still fire.
fn auto_assign_character_clips(character: &mut psxed_project::CharacterResource, clips: &[String]) {
    let lower: Vec<String> = clips.iter().map(|c| c.to_ascii_lowercase()).collect();
    let find_exact = |needles: &[&str]| -> Option<u16> {
        lower
            .iter()
            .position(|name| needles.iter().any(|needle| name == needle))
            .map(|i| i as u16)
    };
    let find_contains = |needle: &str, reject: &[&str]| -> Option<u16> {
        lower
            .iter()
            .position(|name| {
                name.contains(needle) && !reject.iter().any(|blocked| name.contains(blocked))
            })
            .map(|i| i as u16)
    };

    if let Some(i) = find_exact(&["idle"]).or_else(|| find_contains("idle", &[])) {
        character.idle_clip = Some(i);
    }
    if let Some(i) = find_exact(&["walking", "walk", "walk_forward", "forward_walk"])
        .or_else(|| find_contains("walk", &["back", "backward", "unsteady"]))
        .or_else(|| find_contains("walk", &[]))
    {
        character.walk_clip = Some(i);
    }
    if let Some(i) = find_exact(&["running", "run"]).or_else(|| find_contains("run", &[])) {
        character.run_clip = Some(i);
    }
    if let Some(i) = find_exact(&["turn", "turning"]).or_else(|| find_contains("turn", &[])) {
        character.turn_clip = Some(i);
    }
}

/// Player-spawn inspector helper: pick which Character drives
/// this spawn. `(none)` lets the cook step auto-pick when
/// exactly one Character exists.
fn draw_character_selector(
    ui: &mut egui::Ui,
    options: &[(ResourceId, String)],
    current: &mut Option<ResourceId>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Character");
        let preview = current
            .and_then(|id| {
                options
                    .iter()
                    .find(|(rid, _)| *rid == id)
                    .map(|(_, n)| n.as_str())
            })
            .unwrap_or("(none)");
        egui::ComboBox::from_id_salt("player-spawn-character-picker")
            .selected_text(preview)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "(none)").clicked() {
                    *current = None;
                    changed = true;
                }
                for (id, name) in options {
                    if ui.selectable_label(*current == Some(*id), name).clicked() {
                        *current = Some(*id);
                        changed = true;
                    }
                }
            });
    });
    if let Some(id) = *current {
        if !options.iter().any(|(rid, _)| *rid == id) {
            ui.colored_label(
                Color32::from_rgb(220, 120, 100),
                "Selected Character resource is missing.",
            );
        }
    } else if options.is_empty() {
        ui.colored_label(
            STUDIO_TEXT_WEAK,
            "No Character resources defined. Cook will fail unless one is added.",
        );
    } else if options.len() > 1 {
        ui.colored_label(
            STUDIO_TEXT_WEAK,
            "Multiple Characters available — pick one explicitly to avoid Cook failures.",
        );
    } else {
        ui.colored_label(
            STUDIO_TEXT_WEAK,
            format!(
                "Cook will auto-select \"{}\" — only one Character defined.",
                options[0].1
            ),
        );
    }
    changed
}

fn drag_u16(ui: &mut egui::Ui, label: &str, value: &mut u16, min: u16, max: u16) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let mut v = *value as i64;
        if ui
            .add(egui::DragValue::new(&mut v).range(min as i64..=max as i64))
            .changed()
        {
            *value = v.clamp(min as i64, max as i64) as u16;
            changed = true;
        }
    });
    changed
}

fn collider_shape_editor(ui: &mut egui::Ui, shape: &mut ColliderShape) -> bool {
    let mut changed = false;
    let current = match shape {
        ColliderShape::Box { .. } => "Box",
        ColliderShape::Sphere { .. } => "Sphere",
        ColliderShape::Capsule { .. } => "Capsule",
    };
    egui::ComboBox::from_label("Shape")
        .selected_text(current)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(matches!(shape, ColliderShape::Box { .. }), "Box")
                .clicked()
            {
                *shape = ColliderShape::Box {
                    half_extents: [256, 256, 256],
                };
                changed = true;
            }
            if ui
                .selectable_label(matches!(shape, ColliderShape::Sphere { .. }), "Sphere")
                .clicked()
            {
                *shape = ColliderShape::Sphere { radius: 256 };
                changed = true;
            }
            if ui
                .selectable_label(matches!(shape, ColliderShape::Capsule { .. }), "Capsule")
                .clicked()
            {
                *shape = ColliderShape::Capsule {
                    radius: 192,
                    height: 1024,
                };
                changed = true;
            }
        });
    match shape {
        ColliderShape::Box { half_extents } => {
            changed |= drag_u16(ui, "Half X", &mut half_extents[0], 0, 8192);
            changed |= drag_u16(ui, "Half Y", &mut half_extents[1], 0, 8192);
            changed |= drag_u16(ui, "Half Z", &mut half_extents[2], 0, 8192);
        }
        ColliderShape::Sphere { radius } => {
            changed |= drag_u16(ui, "Radius", radius, 0, 8192);
        }
        ColliderShape::Capsule { radius, height } => {
            changed |= drag_u16(ui, "Radius", radius, 0, 8192);
            changed |= drag_u16(ui, "Height", height, 0, 16384);
        }
    }
    changed
}

fn drag_i32(ui: &mut egui::Ui, label: &str, value: &mut i32, min: i32, max: i32) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        if ui
            .add(egui::DragValue::new(value).range(min..=max))
            .changed()
        {
            *value = (*value).clamp(min, max);
            changed = true;
        }
    });
    changed
}

fn human_bytes(n: u32) -> String {
    if n < 1024 {
        format!("{} B", n)
    } else if n < 1024 * 1024 {
        format!("{:.1} KB", (n as f64) / 1024.0)
    } else {
        format!("{:.1} MB", (n as f64) / (1024.0 * 1024.0))
    }
}

fn dock_label_limit(depth: usize) -> usize {
    LEFT_DOCK_LABEL_CHARS
        .saturating_sub(depth.saturating_mul(2))
        .max(18)
}

fn compact_middle(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars || max_chars < 8 {
        return text.to_string();
    }

    let marker = "...";
    let room = max_chars.saturating_sub(marker.len());
    let head = room.saturating_mul(2) / 3;
    let tail = room.saturating_sub(head);
    let mut out = String::with_capacity(text.len().min(max_chars + marker.len()));
    out.extend(text.chars().take(head));
    out.push_str(marker);
    let mut suffix = text.chars().rev().take(tail).collect::<Vec<_>>();
    suffix.reverse();
    out.extend(suffix);
    out
}

fn draw_scene_node_row(
    ui: &mut egui::Ui,
    row: &NodeRow,
    selected: bool,
    renaming: &mut Option<(NodeId, String)>,
    pending_focus: &mut bool,
    actions: &mut Vec<TreeAction>,
) {
    let row_height = 24.0;
    let dnd_active = egui::DragAndDrop::has_any_payload(ui.ctx());

    // Insertion bar above the row: dropping here reorders before the
    // current row inside its parent's child list. Hidden when no drag
    // is in flight so we don't steal mouse hovers.
    if dnd_active && row.id != NodeId::ROOT {
        let (insert_rect, insert_response) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 4.0), Sense::hover());
        if let Some(payload) = insert_response.dnd_release_payload::<NodeId>() {
            actions.push(TreeAction::Reparent {
                source: *payload,
                target_parent: row.parent.unwrap_or(NodeId::ROOT),
                position: row.sibling_index,
            });
        }
        if insert_response.contains_pointer() {
            ui.painter().line_segment(
                [
                    Pos2::new(insert_rect.left() + 4.0, insert_rect.center().y),
                    Pos2::new(insert_rect.right() - 4.0, insert_rect.center().y),
                ],
                Stroke::new(2.0, STUDIO_ACCENT),
            );
        }
    }

    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), row_height),
        Sense::click_and_drag(),
    );
    let painter = ui.painter_at(rect);
    let hovered = response.hovered();

    if selected {
        painter.rect_filled(rect.shrink2(Vec2::new(0.0, 1.0)), 3.0, STUDIO_ACCENT_DIM);
    } else if hovered {
        painter.rect_filled(
            rect.shrink2(Vec2::new(0.0, 1.0)),
            3.0,
            Color32::from_rgba_unmultiplied(42, 58, 70, 120),
        );
    }

    let indent = row.depth as f32 * 16.0;
    let content_left = rect.left() + 4.0 + indent;
    if row.depth > 0 {
        let line_x = rect.left() + 10.0 + (row.depth.saturating_sub(1) as f32 * 16.0);
        painter.line_segment(
            [
                Pos2::new(line_x, rect.top()),
                Pos2::new(line_x, rect.bottom()),
            ],
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(70, 86, 102, 92)),
        );
    }

    let chevron_rect = Rect::from_min_size(
        Pos2::new(content_left, rect.center().y - 5.0),
        Vec2::splat(10.0),
    );
    if row.child_count > 0 {
        painter.text(
            chevron_rect.center(),
            Align2::CENTER_CENTER,
            icons::CHEVRON_DOWN.to_string(),
            icons::font(12.0),
            Color32::from_rgb(160, 174, 188),
        );
    }

    let icon_rect = Rect::from_min_size(
        Pos2::new(content_left + 14.0, rect.center().y - 8.0),
        Vec2::splat(16.0),
    );
    painter.text(
        icon_rect.center(),
        Align2::CENTER_CENTER,
        node_lucide_icon(row.kind, row.id == NodeId::ROOT).to_string(),
        icons::font(15.0),
        node_lucide_color(row.kind, row.id == NodeId::ROOT, selected),
    );

    let text_color = if selected {
        Color32::WHITE
    } else {
        STUDIO_TEXT
    };
    let in_rename = matches!(renaming, Some((id, _)) if *id == row.id);
    let label = if row.id == NodeId::ROOT {
        format!("Root: {}", row.kind)
    } else {
        row.name.clone()
    };
    let display_label = compact_middle(&label, dock_label_limit(row.depth));
    let response = if !in_rename && display_label != label {
        response.on_hover_text(label.clone())
    } else {
        response
    };
    let text_left = icon_rect.right() + 7.0;
    let text_pos = Pos2::new(text_left, rect.center().y);

    if in_rename {
        let edit_rect = Rect::from_min_size(
            Pos2::new(text_left, rect.center().y - 10.0),
            Vec2::new(rect.right() - text_left - 56.0, 20.0),
        );
        if let Some((_, buffer)) = renaming.as_mut() {
            let edit_response = ui.put(
                edit_rect,
                egui::TextEdit::singleline(buffer)
                    .desired_width(edit_rect.width())
                    .margin(egui::Vec2::new(2.0, 1.0)),
            );
            if *pending_focus {
                edit_response.request_focus();
                *pending_focus = false;
            }
            let lost_focus = edit_response.lost_focus();
            let pressed_enter = lost_focus && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let pressed_esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
            if pressed_esc {
                actions.push(TreeAction::CancelRename);
            } else if pressed_enter || lost_focus {
                actions.push(TreeAction::CommitRename(row.id, buffer.clone()));
            }
        }
    } else {
        let name_clip_right = if row.id != NodeId::ROOT {
            (rect.right() - 142.0).max(text_left + 72.0)
        } else {
            rect.right() - 28.0
        };
        painter
            .with_clip_rect(Rect::from_min_max(
                Pos2::new(text_left, rect.top()),
                Pos2::new(name_clip_right, rect.bottom()),
            ))
            .text(
                text_pos,
                Align2::LEFT_CENTER,
                display_label,
                FontId::proportional(13.0),
                text_color,
            );
    }

    if !in_rename && row.id != NodeId::ROOT {
        painter.text(
            Pos2::new(
                (rect.right() - 136.0).max(text_pos.x + 78.0),
                rect.center().y,
            ),
            Align2::LEFT_CENTER,
            row.kind,
            FontId::proportional(11.0),
            if selected {
                Color32::from_rgb(204, 229, 236)
            } else {
                STUDIO_TEXT_WEAK
            },
        );
    }

    if row.child_count > 0 && row.id != NodeId::ROOT {
        let pill = Rect::from_min_size(
            Pos2::new(rect.right() - 50.0, rect.center().y - 8.0),
            Vec2::new(24.0, 16.0),
        );
        painter.rect_filled(pill, 8.0, Color32::from_rgba_unmultiplied(9, 14, 18, 138));
        painter.text(
            pill.center(),
            Align2::CENTER_CENTER,
            row.child_count.to_string(),
            FontId::monospace(10.0),
            STUDIO_TEXT_WEAK,
        );
    }

    let eye_rect = Rect::from_min_size(
        Pos2::new(rect.right() - 22.0, rect.center().y - 6.0),
        Vec2::new(14.0, 12.0),
    );
    painter.text(
        eye_rect.center(),
        Align2::CENTER_CENTER,
        icons::EYE.to_string(),
        icons::font(12.0),
        if selected || hovered {
            Color32::from_rgb(184, 205, 218)
        } else {
            Color32::from_rgb(88, 102, 116)
        },
    );

    if in_rename {
        return;
    }

    // Drag source: only descendants of root can be dragged.
    if row.id != NodeId::ROOT && response.dragged() {
        response.dnd_set_drag_payload::<NodeId>(row.id);
        let label_text = label_for_drag(row);
        let pointer_pos = ui
            .ctx()
            .input(|i| i.pointer.interact_pos())
            .unwrap_or_else(|| rect.center());
        ui.painter().text(
            pointer_pos + Vec2::new(12.0, 0.0),
            Align2::LEFT_CENTER,
            label_text,
            FontId::proportional(12.0),
            STUDIO_ACCENT,
        );
    }

    // Drop on row body → reparent as last child. Highlight while
    // hovered so the user knows where the drop will land.
    if let Some(payload) = response.dnd_hover_payload::<NodeId>() {
        if *payload != row.id {
            ui.painter().rect_stroke(
                rect.shrink2(Vec2::new(2.0, 1.0)),
                3.0,
                Stroke::new(1.5, STUDIO_ACCENT),
                StrokeKind::Inside,
            );
        }
    }
    if let Some(payload) = response.dnd_release_payload::<NodeId>() {
        actions.push(TreeAction::Reparent {
            source: *payload,
            target_parent: row.id,
            position: row.child_count,
        });
    }

    if response.clicked() {
        actions.push(TreeAction::Select(row.id));
    }
    if response.double_clicked() && row.id != NodeId::ROOT {
        actions.push(TreeAction::BeginRename(row.id));
    }

    if row.id != NodeId::ROOT {
        response.context_menu(|ui| {
            ui.menu_button(icons::label(icons::PLUS, "Add Child"), |ui| {
                for (label, kind) in default_addable_kinds() {
                    if ui.button(label).clicked() {
                        actions.push(TreeAction::AddChild {
                            parent: row.id,
                            kind,
                            name: label,
                        });
                        ui.close_menu();
                    }
                }
            });
            if ui.button(icons::label(icons::PALETTE, "Rename")).clicked() {
                actions.push(TreeAction::BeginRename(row.id));
                ui.close_menu();
            }
            if ui.button(icons::label(icons::COPY, "Duplicate")).clicked() {
                actions.push(TreeAction::Duplicate(row.id));
                ui.close_menu();
            }
            ui.separator();
            if ui.button(icons::label(icons::TRASH, "Delete")).clicked() {
                actions.push(TreeAction::Delete(row.id));
                ui.close_menu();
            }
        });
    } else {
        // Even root accepts "Add Child" for top-level Worlds.
        response.context_menu(|ui| {
            ui.menu_button(icons::label(icons::PLUS, "Add Child"), |ui| {
                for (label, kind) in default_addable_kinds() {
                    if ui.button(label).clicked() {
                        actions.push(TreeAction::AddChild {
                            parent: row.id,
                            kind,
                            name: label,
                        });
                        ui.close_menu();
                    }
                }
            });
        });
    }
}

/// Friendly label for the drag-tooltip preview.
fn label_for_drag(row: &NodeRow) -> String {
    if row.name.is_empty() {
        row.kind.to_string()
    } else {
        row.name.clone()
    }
}

/// Default `(menu label, kind template)` pairs for "Add Child" menus.
/// Each menu entry uses the label as the new node's display name.
fn default_addable_kinds() -> [(&'static str, NodeKind); 18] {
    [
        ("World", NodeKind::World),
        (
            "Room",
            NodeKind::Room {
                grid: WorldGrid::empty(4, 4, 1024),
            },
        ),
        ("Entity", NodeKind::Entity),
        ("Actor", NodeKind::Actor),
        ("Node3D", NodeKind::Node3D),
        (
            "ModelRenderer",
            NodeKind::ModelRenderer {
                model: None,
                material: None,
            },
        ),
        (
            "Animator",
            NodeKind::Animator {
                clip: None,
                autoplay: true,
            },
        ),
        (
            "Collider",
            NodeKind::Collider {
                shape: ColliderShape::default(),
                solid: true,
            },
        ),
        (
            "Interactable",
            NodeKind::Interactable {
                prompt: String::new(),
                action: String::new(),
            },
        ),
        (
            "CharacterController",
            NodeKind::CharacterController {
                character: None,
                player: false,
            },
        ),
        (
            "AiController",
            NodeKind::AiController {
                behavior: String::new(),
            },
        ),
        (
            "Combat",
            NodeKind::Combat {
                faction: String::new(),
                health: 1,
            },
        ),
        (
            "PointLight",
            NodeKind::PointLight {
                color: [255, 240, 200],
                intensity: 1.0,
                radius: 4.0,
            },
        ),
        (
            "MeshInstance",
            NodeKind::MeshInstance {
                mesh: None,
                material: None,
                animation_clip: None,
            },
        ),
        (
            "Light",
            NodeKind::Light {
                color: [255, 240, 200],
                intensity: 1.0,
                // Sectors. Matches the Place tool default and
                // the Torch/Room fill presets -- historically
                // this was 4096.0 (4096 sectors!) which lit
                // every room from across the world.
                radius: 4.0,
            },
        ),
        (
            "SpawnPoint",
            NodeKind::SpawnPoint {
                player: false,
                character: None,
            },
        ),
        (
            "Trigger",
            NodeKind::Trigger {
                trigger_id: String::new(),
            },
        ),
        (
            "Portal",
            NodeKind::Portal {
                target_room: None,
                target_entry: String::new(),
                entry_name: String::new(),
            },
        ),
    ]
}

fn node_draw_mode(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::MeshInstance { .. } => "Textured Triangles",
        NodeKind::ModelRenderer { .. } => "Render Component",
        NodeKind::Animator { .. } => "Animation Component",
        NodeKind::Collider { .. } => "Collision Component",
        NodeKind::Interactable { .. } => "Interaction Component",
        NodeKind::CharacterController { .. } => "Character Component",
        NodeKind::AiController { .. } => "AI Component",
        NodeKind::Combat { .. } => "Combat Component",
        NodeKind::World => "Streaming Region",
        NodeKind::Entity => "Entity Host",
        NodeKind::Actor => "Actor Host",
        NodeKind::Room { .. } => "Sector Grid",
        NodeKind::Light { .. } => "Editor Gizmo",
        NodeKind::PointLight { .. } => "Light Component",
        NodeKind::SpawnPoint { .. } => "Spawn Marker",
        NodeKind::Trigger { .. } => "Trigger Volume",
        NodeKind::AudioSource { .. } => "Audio Marker",
        NodeKind::Portal { .. } => "Portal Volume",
        NodeKind::Node | NodeKind::Node3D => "None",
    }
}

fn count_nodes(project: &ProjectDocument, predicate: impl Fn(&NodeKind) -> bool) -> usize {
    project
        .active_scene()
        .nodes()
        .iter()
        .filter(|node| predicate(&node.kind))
        .count()
}

fn draw_scene_group(ui: &mut egui::Ui, icon: char, label: &str, count: usize) {
    egui::CollapsingHeader::new(format!("{} ({count})", icons::label(icon, label)))
        .default_open(false)
        .show(ui, |ui| {
            ui.weak("Filtered collection");
        });
}

#[derive(Debug, Clone)]
struct ProjectFileRow {
    depth: usize,
    name: String,
    folder: bool,
    icon: char,
    resource: Option<ResourceId>,
}

fn project_filesystem_rows(project: &ProjectDocument) -> Vec<ProjectFileRow> {
    let mut rows = Vec::new();
    rows.push(ProjectFileRow {
        depth: 0,
        name: "res://".to_string(),
        folder: true,
        icon: icons::FOLDER,
        resource: None,
    });
    rows.push(ProjectFileRow {
        depth: 1,
        name: "rooms".to_string(),
        folder: true,
        icon: icons::FOLDER,
        resource: None,
    });
    rows.push(ProjectFileRow {
        depth: 2,
        name: format!("{}.room", snake_name(&project.active_scene().name)),
        folder: false,
        icon: icons::GRID,
        resource: None,
    });

    push_resource_folder(project, &mut rows, "materials", ResourceFilter::Material);
    push_resource_folder(project, &mut rows, "textures", ResourceFilter::Texture);
    push_resource_folder(project, &mut rows, "models", ResourceFilter::Model);
    push_resource_folder(project, &mut rows, "meshes", ResourceFilter::Mesh);
    push_resource_folder(project, &mut rows, "spawns", ResourceFilter::Other);
    rows
}

fn push_resource_folder(
    project: &ProjectDocument,
    rows: &mut Vec<ProjectFileRow>,
    folder: &str,
    filter: ResourceFilter,
) {
    rows.push(ProjectFileRow {
        depth: 1,
        name: folder.to_string(),
        folder: true,
        icon: icons::FOLDER,
        resource: None,
    });
    for resource in project
        .resources
        .iter()
        .filter(|resource| filter.matches(&resource.data))
    {
        rows.push(ProjectFileRow {
            depth: 2,
            name: resource_file_name(resource),
            folder: false,
            icon: resource_lucide_icon(&resource.data),
            resource: Some(resource.id),
        });
    }
}

fn draw_project_file_row(
    ui: &mut egui::Ui,
    row: &ProjectFileRow,
    selected_resource: Option<ResourceId>,
    filter: &str,
) -> bool {
    if !row.folder && !filter.is_empty() && !row.name.to_ascii_lowercase().contains(filter) {
        return false;
    }

    let mut clicked = false;
    ui.horizontal(|ui| {
        ui.add_space(row.depth as f32 * 14.0);
        let display_name = compact_middle(&row.name, dock_label_limit(row.depth));
        let label = icons::label(row.icon, &display_name);
        let label_was_compacted = display_name != row.name;
        if row.folder {
            let response = ui.label(RichText::new(label).color(STUDIO_TEXT_WEAK));
            if label_was_compacted {
                response.on_hover_text(row.name.clone());
            }
        } else {
            let response = ui.selectable_label(row.resource == selected_resource, label);
            if response.clicked() {
                clicked = true;
            }
            if label_was_compacted {
                response.on_hover_text(row.name.clone());
            }
        }
    });
    clicked
}

fn resource_file_name(resource: &Resource) -> String {
    match &resource.data {
        ResourceData::Texture { psxt_path } => cooked_name(&resource.name, psxt_path, "psxt"),
        ResourceData::Material(_) => cooked_name(&resource.name, "", "mat"),
        ResourceData::Model(model) => cooked_name(&resource.name, &model.model_path, "psxmdl"),
        ResourceData::Character(_) => cooked_name(&resource.name, "", "char"),
        ResourceData::Mesh { source_path } => cooked_name(&resource.name, source_path, "psxmesh"),
        ResourceData::Scene { source_path } => cooked_name(&resource.name, source_path, "room"),
        ResourceData::Script { source_path } => cooked_name(&resource.name, source_path, "script"),
        ResourceData::Audio { source_path } => cooked_name(&resource.name, source_path, "vag"),
    }
}

fn cooked_name(name: &str, source_path: &str, ext: &str) -> String {
    let stem = Path::new(source_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or(name);
    format!("{}.{}", snake_name(stem), ext)
}

fn snake_name(name: &str) -> String {
    let mut out = String::new();
    let mut previous_was_sep = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            previous_was_sep = false;
        } else if !previous_was_sep {
            out.push('_');
            previous_was_sep = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn resource_filter_counts(project: &ProjectDocument) -> [(ResourceFilter, usize); 7] {
    let mut texture = 0;
    let mut material = 0;
    let mut model = 0;
    let mut character = 0;
    let mut mesh = 0;
    let mut room = 0;
    let mut other = 0;
    for resource in &project.resources {
        match &resource.data {
            ResourceData::Texture { .. } => texture += 1,
            ResourceData::Material(_) => material += 1,
            ResourceData::Model(_) => model += 1,
            ResourceData::Character(_) => character += 1,
            ResourceData::Mesh { .. } => mesh += 1,
            ResourceData::Scene { .. } => room += 1,
            ResourceData::Script { .. } | ResourceData::Audio { .. } => other += 1,
        }
    }
    [
        (ResourceFilter::Texture, texture),
        (ResourceFilter::Material, material),
        (ResourceFilter::Model, model),
        (ResourceFilter::Character, character),
        (ResourceFilter::Mesh, mesh),
        (ResourceFilter::Room, room),
        (ResourceFilter::Other, other),
    ]
}

fn resource_matches_filter(resource: &Resource, filter: ResourceFilter, search: &str) -> bool {
    if !filter.matches(&resource.data) {
        return false;
    }
    if search.is_empty() {
        return true;
    }
    resource.name.to_ascii_lowercase().contains(search)
        || resource.data.label().to_ascii_lowercase().contains(search)
        || resource_source_path(resource)
            .is_some_and(|path| path.to_ascii_lowercase().contains(search))
}

fn resource_source_path(resource: &Resource) -> Option<&str> {
    match &resource.data {
        ResourceData::Texture { psxt_path } => Some(psxt_path.as_str()),
        ResourceData::Model(model) => Some(model.model_path.as_str()),
        ResourceData::Mesh { source_path }
        | ResourceData::Scene { source_path }
        | ResourceData::Script { source_path }
        | ResourceData::Audio { source_path } => Some(source_path.as_str()),
        ResourceData::Material(_) | ResourceData::Character(_) => None,
    }
}

fn resource_lucide_icon(data: &ResourceData) -> char {
    match data {
        ResourceData::Texture { .. } => icons::PALETTE,
        ResourceData::Material(_) => icons::BLEND,
        ResourceData::Model(_) => icons::BOX,
        ResourceData::Character(_) => icons::MAP_PIN,
        ResourceData::Mesh { .. } => icons::BOX,
        ResourceData::Scene { .. } => icons::GRID,
        ResourceData::Script { .. } => icons::FILE,
        ResourceData::Audio { .. } => icons::AUDIO_LINES,
    }
}

fn resource_lucide_color(data: &ResourceData, selected: bool) -> Color32 {
    if selected {
        return Color32::WHITE;
    }

    match data {
        ResourceData::Texture { .. } => Color32::from_rgb(163, 182, 198),
        ResourceData::Material(_) => Color32::from_rgb(208, 112, 162),
        ResourceData::Model(_) => Color32::from_rgb(186, 178, 124),
        ResourceData::Character(_) => Color32::from_rgb(120, 220, 148),
        ResourceData::Mesh { .. } => Color32::from_rgb(156, 174, 190),
        ResourceData::Scene { .. } => Color32::from_rgb(209, 118, 71),
        ResourceData::Script { .. } => Color32::from_rgb(188, 176, 104),
        ResourceData::Audio { .. } => Color32::from_rgb(104, 202, 188),
    }
}

fn draw_resource_card(
    ui: &mut egui::Ui,
    project: &ProjectDocument,
    resource: &Resource,
    selected: bool,
    thumb: Option<egui::TextureId>,
) -> egui::Response {
    let size = Vec2::new(120.0, 155.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    let fill = if selected {
        Color32::from_rgb(21, 50, 62)
    } else {
        STUDIO_PANEL
    };
    painter.rect_filled(rect, 4.0, fill);
    painter.rect_stroke(
        rect,
        4.0,
        Stroke::new(
            if selected { 2.0 } else { 1.0 },
            if selected {
                STUDIO_ACCENT
            } else {
                STUDIO_BORDER
            },
        ),
        StrokeKind::Inside,
    );

    let preview = Rect::from_min_size(rect.min + Vec2::new(12.0, 12.0), Vec2::new(96.0, 76.0));
    draw_resource_preview(&painter, preview, project, resource, thumb);
    painter.rect_stroke(
        preview,
        2.0,
        Stroke::new(1.0, Color32::from_rgb(54, 64, 76)),
        StrokeKind::Inside,
    );
    let badge = Rect::from_min_size(preview.left_top() + Vec2::new(4.0, 4.0), Vec2::splat(22.0));
    painter.rect_filled(badge, 4.0, Color32::from_rgba_unmultiplied(8, 12, 16, 192));
    painter.text(
        badge.center(),
        Align2::CENTER_CENTER,
        resource_lucide_icon(&resource.data).to_string(),
        icons::font(14.0),
        resource_lucide_color(&resource.data, selected),
    );
    painter.text(
        rect.center_top() + Vec2::new(0.0, 98.0),
        Align2::CENTER_TOP,
        &resource.name,
        FontId::monospace(12.0),
        Color32::from_rgb(225, 231, 240),
    );
    painter.text(
        rect.center_bottom() + Vec2::new(0.0, -18.0),
        Align2::CENTER_BOTTOM,
        resource_detail(resource),
        FontId::monospace(10.0),
        STUDIO_TEXT_WEAK,
    );
    if response.dragged() {
        response.dnd_set_drag_payload::<ResourceId>(resource.id);
        let pointer_pos = ui
            .ctx()
            .input(|input| input.pointer.interact_pos())
            .unwrap_or_else(|| rect.center());
        ui.painter().text(
            pointer_pos + Vec2::new(12.0, 0.0),
            Align2::LEFT_CENTER,
            format!("{} {}", resource_lucide_icon(&resource.data), resource.name),
            FontId::proportional(12.0),
            STUDIO_ACCENT,
        );
    }
    response
}

fn draw_resource_preview(
    painter: &egui::Painter,
    preview: Rect,
    project: &ProjectDocument,
    resource: &Resource,
    thumb: Option<egui::TextureId>,
) {
    match &resource.data {
        ResourceData::Material(material) => {
            // Material: blit the linked Texture's decoded thumbnail
            // when available, fall back to the linked texture's
            // procedural pattern, then to the material's own name.
            if let Some(id) = thumb {
                blit_thumb(painter, preview, id);
            } else if let Some(texture) = material.texture.and_then(|id| project.resource(id)) {
                draw_texture_like_preview(painter, preview, texture);
            } else {
                draw_texture_like_preview(painter, preview, resource);
            }
            let tint = Color32::from_rgba_unmultiplied(
                material.tint[0].saturating_mul(2),
                material.tint[1].saturating_mul(2),
                material.tint[2].saturating_mul(2),
                if material.blend_mode == PsxBlendMode::Opaque {
                    55
                } else {
                    110
                },
            );
            painter.rect_filled(preview.shrink(8.0), 2.0, tint);
        }
        ResourceData::Texture { .. } => {
            if let Some(id) = thumb {
                blit_thumb(painter, preview, id);
            } else {
                draw_texture_like_preview(painter, preview, resource);
            }
        }
        _ => {
            painter.rect_filled(preview, 2.0, resource_preview_color(resource));
        }
    }
    draw_palette_strip(painter, preview, palette_for_resource(resource));
}

/// Paint a registered egui texture into the preview rect, full-image
/// UV (no atlasing) and untinted. Used for real `.psxt`-decoded
/// thumbnails so the resource card mirrors the actual texels the
/// runtime would sample.
fn blit_thumb(painter: &egui::Painter, preview: Rect, id: egui::TextureId) {
    painter.image(
        id,
        preview,
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
}

fn draw_texture_like_preview(painter: &egui::Painter, preview: Rect, resource: &Resource) {
    let name = resource.name.to_ascii_lowercase();
    if name.contains("brick") {
        draw_brick_preview(painter, preview, resource_preview_color(resource));
    } else if name.contains("floor") || name.contains("stone") {
        draw_stone_preview(painter, preview, resource_preview_color(resource));
    } else {
        draw_checker_preview(painter, preview, resource_preview_color(resource));
    }
}

fn draw_brick_preview(painter: &egui::Painter, preview: Rect, base: Color32) {
    painter.rect_filled(preview, 2.0, darken(base, 8));
    let row_h = 14.0;
    let rows = (preview.height() / row_h).ceil() as i32;
    for row in 0..rows {
        let top = preview.top() + row as f32 * row_h;
        let y = top.min(preview.bottom());
        painter.line_segment(
            [Pos2::new(preview.left(), y), Pos2::new(preview.right(), y)],
            Stroke::new(1.0, darken(base, 48)),
        );
        let offset = if row % 2 == 0 { 0.0 } else { 18.0 };
        let mut x = preview.left() + offset;
        while x < preview.right() {
            painter.line_segment(
                [
                    Pos2::new(x, top + 2.0),
                    Pos2::new(x, (top + row_h - 2.0).min(preview.bottom())),
                ],
                Stroke::new(1.0, darken(base, 45)),
            );
            x += 36.0;
        }
        let stripe = Rect::from_min_max(
            Pos2::new(preview.left(), top + 2.0),
            Pos2::new(preview.right(), (top + 5.0).min(preview.bottom())),
        );
        painter.rect_filled(
            stripe,
            0.0,
            Color32::from_rgba_unmultiplied(188, 110, 60, 35),
        );
    }
}

fn draw_stone_preview(painter: &egui::Painter, preview: Rect, base: Color32) {
    painter.rect_filled(preview, 2.0, darken(base, 18));
    let cols = 3;
    let rows = 3;
    let cell = Vec2::new(
        preview.width() / cols as f32,
        preview.height() / rows as f32,
    );
    for y in 0..rows {
        for x in 0..cols {
            let min = preview.min + Vec2::new(x as f32 * cell.x, y as f32 * cell.y);
            let rect = Rect::from_min_size(min, cell).shrink(1.0);
            let shade = if (x + y) % 2 == 0 {
                lighten(base, 12)
            } else {
                darken(base, 6)
            };
            painter.rect_filled(rect, 1.0, shade);
            painter.rect_stroke(
                rect,
                1.0,
                Stroke::new(1.0, darken(base, 38)),
                StrokeKind::Inside,
            );
        }
    }
}

fn draw_checker_preview(painter: &egui::Painter, preview: Rect, base: Color32) {
    let alt = Color32::from_rgb(
        base.r().saturating_add(28),
        base.g().saturating_add(24),
        base.b().saturating_add(20),
    );
    let cell = 16.0;
    let cols = (preview.width() / cell).ceil() as i32;
    let rows = (preview.height() / cell).ceil() as i32;
    for y in 0..rows {
        for x in 0..cols {
            let min = preview.min + Vec2::new(x as f32 * cell, y as f32 * cell);
            let rect = Rect::from_min_size(min, Vec2::splat(cell)).intersect(preview);
            painter.rect_filled(rect, 0.0, if (x + y) % 2 == 0 { base } else { alt });
        }
    }
}

/// Decode the bytes of a cooked `.psxt` blob into an egui
/// [`ColorImage`] suitable for [`Context::load_texture`], plus the
/// declared `(width, height)` in texels so the cache can pick a
/// reasonable sample rate.
///
/// Supports 4bpp + 8bpp indexed (the editor's two main authored
/// formats); 15bpp returns `None` and the caller falls back to the
/// procedural pattern. The CLUT's STP bit (bit 15, set by the
/// runtime so semi-transparent draws can mask fully transparent
/// black) is masked out before producing display RGB.
fn decode_psxt_thumbnail(bytes: &[u8]) -> Option<(ColorImage, PsxtStats)> {
    let texture = psx_asset::Texture::from_bytes(bytes).ok()?;
    let width = u8::try_from(texture.width()).ok()?;
    let height = u8::try_from(texture.height()).ok()?;
    let clut_entries = texture.clut_entries() as usize;
    if clut_entries != 16 && clut_entries != 256 {
        // 15bpp / unexpected CLUT count -- fall through.
        return None;
    }
    let clut_bytes = texture.clut_bytes();
    if clut_bytes.len() < clut_entries * 2 {
        return None;
    }
    let stats = PsxtStats {
        width: texture.width(),
        height: texture.height(),
        depth_bits: if clut_entries == 16 { 4 } else { 8 },
        clut_entries: clut_entries as u16,
        pixel_bytes: texture.pixel_bytes().len() as u32,
        clut_bytes: clut_bytes.len() as u32,
        file_bytes: bytes.len() as u32,
    };
    let palette: Vec<Color32> = (0..clut_entries)
        .map(|i| {
            let raw = u16::from_le_bytes([clut_bytes[i * 2], clut_bytes[i * 2 + 1]]) & 0x7FFF;
            let r5 = (raw & 0x1F) as u8;
            let g5 = ((raw >> 5) & 0x1F) as u8;
            let b5 = ((raw >> 10) & 0x1F) as u8;
            Color32::from_rgb(
                (r5 << 3) | (r5 >> 2),
                (g5 << 3) | (g5 >> 2),
                (b5 << 3) | (b5 >> 2),
            )
        })
        .collect();

    let pixel_bytes = texture.pixel_bytes();
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    if clut_entries == 16 {
        // 4bpp: 4 texels per halfword, low nibble first.
        let halfwords_per_row = (width as usize).div_ceil(4);
        for row in 0..height as usize {
            for hw in 0..halfwords_per_row {
                let off = (row * halfwords_per_row + hw) * 2;
                if off + 1 >= pixel_bytes.len() {
                    break;
                }
                let word = u16::from_le_bytes([pixel_bytes[off], pixel_bytes[off + 1]]);
                for nibble in 0..4 {
                    let texel = (word >> (nibble * 4)) & 0xF;
                    if hw * 4 + nibble < width as usize {
                        pixels.push(palette[texel as usize]);
                    }
                }
            }
        }
    } else {
        // 8bpp: 2 texels per halfword, low byte first.
        let halfwords_per_row = (width as usize).div_ceil(2);
        for row in 0..height as usize {
            for hw in 0..halfwords_per_row {
                let off = (row * halfwords_per_row + hw) * 2;
                if off + 1 >= pixel_bytes.len() {
                    break;
                }
                let lo = pixel_bytes[off] as usize;
                let hi = pixel_bytes[off + 1] as usize;
                if hw * 2 < width as usize {
                    pixels.push(palette[lo]);
                }
                if hw * 2 + 1 < width as usize {
                    pixels.push(palette[hi]);
                }
            }
        }
    }
    if pixels.len() != width as usize * height as usize {
        return None;
    }
    Some((
        ColorImage {
            size: [width as usize, height as usize],
            pixels,
        },
        stats,
    ))
}

fn draw_palette_strip(painter: &egui::Painter, preview: Rect, swatches: [Color32; 5]) {
    let width = preview.width() / swatches.len() as f32;
    for (idx, color) in swatches.iter().enumerate() {
        let min = Pos2::new(preview.left() + idx as f32 * width, preview.bottom() - 10.0);
        let rect = Rect::from_min_size(min, Vec2::new(width, 10.0));
        painter.rect_filled(rect, 0.0, *color);
    }
}

fn palette_for_resource(resource: &Resource) -> [Color32; 5] {
    let base = resource_preview_color(resource);
    [
        Color32::from_rgb(28, 30, 34),
        darken(base, 70),
        darken(base, 35),
        base,
        lighten(base, 44),
    ]
}

fn darken(color: Color32, amount: u8) -> Color32 {
    Color32::from_rgb(
        color.r().saturating_sub(amount),
        color.g().saturating_sub(amount),
        color.b().saturating_sub(amount),
    )
}

fn lighten(color: Color32, amount: u8) -> Color32 {
    Color32::from_rgb(
        color.r().saturating_add(amount),
        color.g().saturating_add(amount),
        color.b().saturating_add(amount),
    )
}

fn resource_preview_color(resource: &Resource) -> Color32 {
    let name = resource.name.to_ascii_lowercase();
    if name.contains("brick") {
        Color32::from_rgb(130, 70, 42)
    } else if name.contains("floor") {
        Color32::from_rgb(106, 112, 120)
    } else if name.contains("glass") {
        Color32::from_rgb(80, 150, 165)
    } else {
        match &resource.data {
            ResourceData::Texture { .. } => Color32::from_rgb(92, 116, 140),
            ResourceData::Material(_) => Color32::from_rgb(120, 92, 135),
            ResourceData::Model(_) => Color32::from_rgb(140, 124, 96),
            ResourceData::Character(_) => Color32::from_rgb(96, 144, 110),
            ResourceData::Mesh { .. } => Color32::from_rgb(110, 120, 130),
            ResourceData::Scene { .. } => Color32::from_rgb(92, 130, 106),
            ResourceData::Script { .. } => Color32::from_rgb(128, 126, 80),
            ResourceData::Audio { .. } => Color32::from_rgb(80, 128, 128),
        }
    }
}

fn resource_detail(resource: &Resource) -> &'static str {
    match &resource.data {
        ResourceData::Texture { .. } => "Texture - 4bpp",
        ResourceData::Material(_) => "Material - 4bpp",
        ResourceData::Model(_) => "Model",
        ResourceData::Character(_) => "Character",
        ResourceData::Mesh { .. } => "Mesh",
        ResourceData::Scene { .. } => "Room",
        ResourceData::Script { .. } => "Script",
        ResourceData::Audio { .. } => "Audio",
    }
}

fn draw_viewport_overlay(
    painter: &egui::Painter,
    rect: Rect,
    project: &ProjectDocument,
    zoom: f32,
    snap_units: u16,
) {
    let overlay = Rect::from_min_size(
        rect.left_top() + Vec2::new(12.0, 12.0),
        Vec2::new(132.0, 94.0),
    );
    painter.rect_filled(
        overlay,
        2.0,
        Color32::from_rgba_unmultiplied(14, 20, 26, 224),
    );
    painter.rect_stroke(
        overlay,
        2.0,
        Stroke::new(1.0, STUDIO_BORDER),
        StrokeKind::Inside,
    );
    let lines = [
        "Top Orthographic".to_string(),
        format!("Grid: {snap_units} units"),
        format!("{} nodes", project.active_scene().nodes().len()),
        format!("{} resources", project.resources.len()),
        format!("{:.0} px/unit", zoom),
    ];
    for (idx, line) in lines.iter().enumerate() {
        painter.text(
            overlay.left_top() + Vec2::new(10.0, 10.0 + idx as f32 * 15.0),
            Align2::LEFT_TOP,
            line,
            FontId::monospace(11.0),
            STUDIO_TEXT,
        );
    }
}

fn draw_axes_gizmo(painter: &egui::Painter, rect: Rect) {
    let origin = Pos2::new(rect.left() + 34.0, rect.bottom() - 38.0);
    let x_end = origin + Vec2::new(42.0, 0.0);
    let y_end = origin + Vec2::new(0.0, -42.0);
    let x_stroke = Stroke::new(2.0, Color32::from_rgb(220, 52, 46));
    let y_stroke = Stroke::new(2.0, Color32::from_rgb(108, 220, 92));

    painter.circle_filled(origin, 3.0, STUDIO_ACCENT);
    painter.line_segment([origin, x_end], x_stroke);
    painter.line_segment([origin, y_end], y_stroke);
    painter.line_segment([x_end, x_end + Vec2::new(-7.0, -4.0)], x_stroke);
    painter.line_segment([x_end, x_end + Vec2::new(-7.0, 4.0)], x_stroke);
    painter.line_segment([y_end, y_end + Vec2::new(-4.0, 7.0)], y_stroke);
    painter.line_segment([y_end, y_end + Vec2::new(4.0, 7.0)], y_stroke);
    painter.text(
        x_end + Vec2::new(8.0, 0.0),
        Align2::LEFT_CENTER,
        "X",
        FontId::monospace(12.0),
        Color32::from_rgb(255, 95, 88),
    );
    painter.text(
        y_end + Vec2::new(0.0, -8.0),
        Align2::CENTER_BOTTOM,
        "Y",
        FontId::monospace(12.0),
        Color32::from_rgb(140, 255, 128),
    );
}

#[derive(Debug, Clone, Copy)]
struct ViewportTransform {
    rect: Rect,
    pan: Vec2,
    zoom: f32,
}

impl ViewportTransform {
    fn new(rect: Rect, pan: Vec2, zoom: f32) -> Self {
        Self { rect, pan, zoom }
    }

    fn world_to_screen(self, world: [f32; 2]) -> Pos2 {
        self.rect.center() + self.pan + Vec2::new(world[0] * self.zoom, -world[1] * self.zoom)
    }

    fn screen_to_world(self, screen: Pos2) -> [f32; 2] {
        let delta = screen - self.rect.center() - self.pan;
        [delta.x / self.zoom, -delta.y / self.zoom]
    }

    fn world_rect_to_screen(self, center: [f32; 2], half: [f32; 2]) -> Rect {
        let min = [center[0] - half[0], center[1] - half[1]];
        let max = [center[0] + half[0], center[1] + half[1]];
        let a = self.world_to_screen(min);
        let b = self.world_to_screen(max);
        Rect::from_min_max(
            Pos2::new(a.x.min(b.x), a.y.min(b.y)),
            Pos2::new(a.x.max(b.x), a.y.max(b.y)),
        )
    }

    fn screen_radius(self, radius: f32) -> f32 {
        radius * self.zoom
    }
}

#[derive(Debug, Clone)]
struct ViewportHit {
    id: NodeId,
    name: String,
    shape: HitShape,
}

impl ViewportHit {
    fn rect(id: NodeId, name: impl Into<String>, center: [f32; 2], half: [f32; 2]) -> Self {
        Self {
            id,
            name: name.into(),
            shape: HitShape::Rect { center, half },
        }
    }

    fn circle(id: NodeId, name: impl Into<String>, center: [f32; 2], radius: f32) -> Self {
        Self {
            id,
            name: name.into(),
            shape: HitShape::Circle { center, radius },
        }
    }

    fn contains(&self, world: [f32; 2]) -> bool {
        match self.shape {
            HitShape::Rect { center, half } => {
                world[0] >= center[0] - half[0]
                    && world[0] <= center[0] + half[0]
                    && world[1] >= center[1] - half[1]
                    && world[1] <= center[1] + half[1]
            }
            HitShape::Circle { center, radius } => {
                let dx = world[0] - center[0];
                let dz = world[1] - center[1];
                dx * dx + dz * dz <= radius * radius
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum HitShape {
    Rect { center: [f32; 2], half: [f32; 2] },
    Circle { center: [f32; 2], radius: f32 },
}

fn draw_world_grid(painter: &egui::Painter, transform: ViewportTransform) {
    let rect = transform.rect;
    let top_left = transform.screen_to_world(rect.left_top());
    let bottom_right = transform.screen_to_world(rect.right_bottom());
    let min_x = top_left[0].min(bottom_right[0]).floor() as i32 - 1;
    let max_x = top_left[0].max(bottom_right[0]).ceil() as i32 + 1;
    let min_z = top_left[1].min(bottom_right[1]).floor() as i32 - 1;
    let max_z = top_left[1].max(bottom_right[1]).ceil() as i32 + 1;

    let minor = Stroke::new(1.0, Color32::from_rgb(20, 43, 52));
    let major = Stroke::new(1.0, Color32::from_rgb(31, 63, 75));
    let axis = Stroke::new(1.0, Color32::from_rgb(58, 91, 103));

    for x in min_x..=max_x {
        let stroke = if x == 0 {
            axis
        } else if x % 4 == 0 {
            major
        } else {
            minor
        };
        let a = transform.world_to_screen([x as f32, min_z as f32]);
        let b = transform.world_to_screen([x as f32, max_z as f32]);
        painter.line_segment([a, b], stroke);
    }

    for z in min_z..=max_z {
        let stroke = if z == 0 {
            axis
        } else if z % 4 == 0 {
            major
        } else {
            minor
        };
        let a = transform.world_to_screen([min_x as f32, z as f32]);
        let b = transform.world_to_screen([max_x as f32, z as f32]);
        painter.line_segment([a, b], stroke);
    }
}

fn draw_scene_viewport(
    painter: &egui::Painter,
    transform: ViewportTransform,
    project: &ProjectDocument,
    selected: NodeId,
) -> Vec<ViewportHit> {
    let scene = project.active_scene();
    let mut hits = Vec::new();

    for node in scene.nodes() {
        if matches!(node.kind, NodeKind::Room { .. }) {
            draw_room(painter, transform, project, node, selected, &mut hits);
        }
    }

    for node in scene.nodes() {
        match &node.kind {
            NodeKind::MeshInstance { .. } => {
                draw_mesh_marker(painter, transform, project, node, selected, &mut hits);
            }
            NodeKind::SpawnPoint { .. } => {
                draw_spawn_marker(painter, transform, node, selected, &mut hits);
            }
            NodeKind::Light {
                color,
                intensity,
                radius,
            } => {
                draw_light_marker(
                    painter, transform, node, selected, *color, *intensity, *radius, &mut hits,
                );
            }
            NodeKind::Trigger { .. } => {
                draw_simple_marker(
                    painter,
                    transform,
                    node,
                    selected,
                    "T",
                    Color32::from_rgb(180, 116, 230),
                    &mut hits,
                );
            }
            NodeKind::AudioSource { .. } => {
                draw_simple_marker(
                    painter,
                    transform,
                    node,
                    selected,
                    "A",
                    Color32::from_rgb(70, 190, 165),
                    &mut hits,
                );
            }
            NodeKind::Portal { .. } => {
                draw_simple_marker(
                    painter,
                    transform,
                    node,
                    selected,
                    "P",
                    Color32::from_rgb(255, 188, 100),
                    &mut hits,
                );
            }
            NodeKind::Node | NodeKind::Node3D if node.id != scene.root => {
                draw_simple_marker(
                    painter,
                    transform,
                    node,
                    selected,
                    "N",
                    Color32::from_rgb(110, 124, 150),
                    &mut hits,
                );
            }
            _ => {}
        }
    }

    hits
}

fn draw_room(
    painter: &egui::Painter,
    transform: ViewportTransform,
    project: &ProjectDocument,
    node: &psxed_project::SceneNode,
    selected: NodeId,
    hits: &mut Vec<ViewportHit>,
) {
    let NodeKind::Room { grid } = &node.kind else {
        return;
    };

    let center = node_world(node);
    let half = [grid.width as f32 * 0.5, grid.depth as f32 * 0.5];
    let outline = transform.world_rect_to_screen(center, half);
    hits.push(ViewportHit::rect(node.id, node.name.clone(), center, half));
    painter.rect_filled(outline, 0.0, darken(STUDIO_ROOM_FLOOR, 28));

    for x in 0..grid.width {
        for z in 0..grid.depth {
            let Some(sector) = grid.sector(x, z) else {
                continue;
            };
            let tile_center = [
                center[0] - half[0] + x as f32 + 0.5,
                center[1] - half[1] + z as f32 + 0.5,
            ];
            let screen_rect = transform.world_rect_to_screen(tile_center, [0.5, 0.5]);
            if !screen_rect.intersects(transform.rect) {
                continue;
            }
            let floor_material = sector.floor.as_ref().and_then(|floor| floor.material);
            let floor_color = material_color(project, floor_material, SurfaceRole::Floor);
            draw_floor_tile(painter, screen_rect, floor_color, x as i32, z as i32);
            hits.push(ViewportHit::rect(
                node.id,
                format!("{} sector {},{}", node.name, x, z),
                tile_center,
                [0.5, 0.5],
            ));
            draw_grid_sector_walls(painter, transform, project, center, half, x, z, sector);
        }
    }

    painter.rect_stroke(
        outline,
        0.0,
        selected_stroke(selected == node.id),
        StrokeKind::Outside,
    );
    painter.text(
        transform.world_to_screen([center[0] - half[0], center[1] + half[1]])
            + Vec2::new(8.0, -8.0),
        Align2::LEFT_BOTTOM,
        &node.name,
        FontId::monospace(12.0),
        Color32::from_rgb(230, 235, 245),
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_grid_sector_walls(
    painter: &egui::Painter,
    transform: ViewportTransform,
    project: &ProjectDocument,
    center: [f32; 2],
    half: [f32; 2],
    x: u16,
    z: u16,
    sector: &psxed_project::GridSector,
) {
    let wall_thickness = 0.18;
    for direction in GridDirection::CARDINAL {
        let walls = sector.walls.get(direction);
        if walls.is_empty() {
            continue;
        }
        let material = walls.first().and_then(|wall| wall.material);
        let wall_color = material_color(project, material, SurfaceRole::Wall);
        let min_x = center[0] - half[0] + x as f32;
        let min_z = center[1] - half[1] + z as f32;
        let (wall_center, wall_half) = match direction {
            GridDirection::North => (
                [min_x + 0.5, min_z + 1.0 + wall_thickness * 0.5],
                [0.5, wall_thickness * 0.5],
            ),
            GridDirection::East => (
                [min_x + 1.0 + wall_thickness * 0.5, min_z + 0.5],
                [wall_thickness * 0.5, 0.5],
            ),
            GridDirection::South => (
                [min_x + 0.5, min_z - wall_thickness * 0.5],
                [0.5, wall_thickness * 0.5],
            ),
            GridDirection::West => (
                [min_x - wall_thickness * 0.5, min_z + 0.5],
                [wall_thickness * 0.5, 0.5],
            ),
            _ => unreachable!(),
        };
        let screen_rect = transform.world_rect_to_screen(wall_center, wall_half);
        if screen_rect.intersects(transform.rect) {
            draw_wall_band(painter, screen_rect, wall_color);
            painter.rect_stroke(
                screen_rect,
                0.0,
                Stroke::new(1.0, Color32::from_rgb(84, 58, 44)),
                StrokeKind::Inside,
            );
        }
    }

    for (direction, nw_to_se) in [
        (GridDirection::NorthWestSouthEast, true),
        (GridDirection::NorthEastSouthWest, false),
    ] {
        if sector.walls.get(direction).is_empty() {
            continue;
        }
        let min_x = center[0] - half[0] + x as f32;
        let min_z = center[1] - half[1] + z as f32;
        let a = if nw_to_se {
            transform.world_to_screen([min_x, min_z + 1.0])
        } else {
            transform.world_to_screen([min_x + 1.0, min_z + 1.0])
        };
        let b = if nw_to_se {
            transform.world_to_screen([min_x + 1.0, min_z])
        } else {
            transform.world_to_screen([min_x, min_z])
        };
        painter.line_segment([a, b], Stroke::new(4.0, STUDIO_ROOM_WALL));
        painter.line_segment([a, b], Stroke::new(1.0, Color32::from_rgb(84, 58, 44)));
    }
}

fn draw_floor_tile(painter: &egui::Painter, rect: Rect, base: Color32, ix: i32, iz: i32) {
    let tint = if (ix + iz) % 2 == 0 {
        lighten(base, 8)
    } else {
        darken(base, 5)
    };
    painter.rect_filled(rect, 0.0, tint);
    painter.rect_stroke(
        rect,
        0.0,
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(44, 56, 65, 168)),
        StrokeKind::Inside,
    );

    if rect.width() < 28.0 || rect.height() < 28.0 {
        return;
    }

    let mid_x = rect.center().x
        + if ix % 2 == 0 {
            -rect.width() * 0.12
        } else {
            rect.width() * 0.10
        };
    let mid_y = rect.center().y
        + if iz % 2 == 0 {
            rect.height() * 0.08
        } else {
            -rect.height() * 0.10
        };
    let crack = Stroke::new(1.0, Color32::from_rgba_unmultiplied(70, 80, 88, 150));
    painter.line_segment(
        [
            Pos2::new(mid_x, rect.top() + 5.0),
            Pos2::new(mid_x, rect.bottom() - 5.0),
        ],
        crack,
    );
    painter.line_segment(
        [
            Pos2::new(rect.left() + 5.0, mid_y),
            Pos2::new(rect.right() - 5.0, mid_y),
        ],
        crack,
    );
}

fn draw_wall_band(painter: &egui::Painter, rect: Rect, base: Color32) {
    painter.rect_filled(rect, 0.0, darken(base, 4));
    let highlight = Stroke::new(1.0, Color32::from_rgba_unmultiplied(166, 92, 50, 160));
    let shadow = Stroke::new(1.0, Color32::from_rgba_unmultiplied(72, 42, 30, 180));

    if rect.width() >= rect.height() {
        let rows = (rect.height() / 7.0).max(2.0) as i32;
        for row in 1..rows {
            let y = rect.top() + row as f32 * rect.height() / rows as f32;
            painter.line_segment(
                [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                if row % 2 == 0 { highlight } else { shadow },
            );
        }
        let cols = (rect.width() / 42.0).max(3.0) as i32;
        for col in 1..cols {
            let x = rect.left() + col as f32 * rect.width() / cols as f32;
            painter.line_segment(
                [
                    Pos2::new(x, rect.top() + 3.0),
                    Pos2::new(x, rect.bottom() - 3.0),
                ],
                shadow,
            );
        }
    } else {
        let cols = (rect.width() / 7.0).max(2.0) as i32;
        for col in 1..cols {
            let x = rect.left() + col as f32 * rect.width() / cols as f32;
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                if col % 2 == 0 { highlight } else { shadow },
            );
        }
        let rows = (rect.height() / 42.0).max(3.0) as i32;
        for row in 1..rows {
            let y = rect.top() + row as f32 * rect.height() / rows as f32;
            painter.line_segment(
                [
                    Pos2::new(rect.left() + 3.0, y),
                    Pos2::new(rect.right() - 3.0, y),
                ],
                shadow,
            );
        }
    }
}

fn draw_mesh_marker(
    painter: &egui::Painter,
    transform: ViewportTransform,
    project: &ProjectDocument,
    node: &psxed_project::SceneNode,
    selected: NodeId,
    hits: &mut Vec<ViewportHit>,
) {
    let NodeKind::MeshInstance { material, .. } = node.kind else {
        return;
    };
    let center = node_world(node);
    let half = [
        node.transform.scale[0].abs().max(0.35) * 0.5,
        node.transform.scale[2].abs().max(0.18) * 0.5,
    ];
    let rect = transform.world_rect_to_screen(center, half);
    let color = material_color(project, material, SurfaceRole::Object);
    let translucent = material_is_translucent(project, material);
    painter.rect_filled(rect, 0.0, color);
    if translucent {
        draw_glass_marker(painter, rect);
    }
    painter.rect_stroke(
        rect,
        0.0,
        selected_stroke(selected == node.id),
        StrokeKind::Outside,
    );
    painter.text(
        rect.center_top() + Vec2::new(0.0, -6.0),
        Align2::CENTER_BOTTOM,
        &node.name,
        FontId::monospace(11.0),
        Color32::from_rgb(232, 238, 246),
    );
    hits.push(ViewportHit::rect(node.id, node.name.clone(), center, half));
}

fn draw_glass_marker(painter: &egui::Painter, rect: Rect) {
    painter.rect_filled(
        rect.shrink(1.0),
        0.0,
        Color32::from_rgba_unmultiplied(190, 230, 232, 34),
    );
    let sheen = Stroke::new(1.0, Color32::from_rgba_unmultiplied(222, 252, 255, 120));
    let step = 18.0;
    let mut x = rect.left() - rect.height();
    while x < rect.right() {
        painter.line_segment(
            [
                Pos2::new(x, rect.bottom()),
                Pos2::new((x + rect.height()).min(rect.right()), rect.top()),
            ],
            sheen,
        );
        x += step;
    }
}

fn draw_spawn_marker(
    painter: &egui::Painter,
    transform: ViewportTransform,
    node: &psxed_project::SceneNode,
    selected: NodeId,
    hits: &mut Vec<ViewportHit>,
) {
    draw_simple_marker(
        painter,
        transform,
        node,
        selected,
        "P",
        Color32::from_rgb(82, 184, 118),
        hits,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_light_marker(
    painter: &egui::Painter,
    transform: ViewportTransform,
    node: &psxed_project::SceneNode,
    selected: NodeId,
    color: [u8; 3],
    intensity: f32,
    radius: f32,
    hits: &mut Vec<ViewportHit>,
) {
    let center = node_world(node);
    let world_radius = (radius / 4096.0).clamp(0.45, 2.5) * intensity.max(0.25);
    let screen_center = transform.world_to_screen(center);
    painter.circle_filled(
        screen_center,
        transform.screen_radius(world_radius),
        Color32::from_rgba_unmultiplied(color[0], color[1], color[2], 28),
    );
    draw_simple_marker(
        painter,
        transform,
        node,
        selected,
        "L",
        Color32::from_rgb(color[0], color[1], color[2]),
        hits,
    );
}

fn draw_simple_marker(
    painter: &egui::Painter,
    transform: ViewportTransform,
    node: &psxed_project::SceneNode,
    selected: NodeId,
    label: &str,
    fill: Color32,
    hits: &mut Vec<ViewportHit>,
) {
    let center = node_world(node);
    let screen = transform.world_to_screen(center);
    let radius = 0.18;
    painter.circle_filled(screen, transform.screen_radius(radius).max(8.0), fill);
    painter.circle_stroke(
        screen,
        transform.screen_radius(radius).max(8.0),
        selected_stroke(selected == node.id),
    );
    painter.text(
        screen,
        Align2::CENTER_CENTER,
        label,
        FontId::monospace(12.0),
        Color32::from_rgb(8, 10, 13),
    );
    painter.text(
        screen + Vec2::new(0.0, 16.0),
        Align2::CENTER_TOP,
        &node.name,
        FontId::monospace(10.0),
        Color32::from_rgb(220, 228, 238),
    );
    hits.push(ViewportHit::circle(
        node.id,
        node.name.clone(),
        center,
        radius.max(8.0 / transform.zoom),
    ));
}

#[derive(Debug, Clone, Copy)]
enum SurfaceRole {
    Floor,
    Wall,
    Object,
}

fn material_color(
    project: &ProjectDocument,
    material: Option<ResourceId>,
    role: SurfaceRole,
) -> Color32 {
    let Some(id) = material else {
        return match role {
            SurfaceRole::Floor => STUDIO_ROOM_FLOOR,
            SurfaceRole::Wall => STUDIO_ROOM_WALL,
            SurfaceRole::Object => Color32::from_rgb(125, 155, 190),
        };
    };
    let Some(resource) = project.resource(id) else {
        return Color32::from_rgb(150, 80, 120);
    };

    let name = resource.name.to_ascii_lowercase();
    let mut color = if name.contains("brick") {
        Color32::from_rgb(126, 72, 43)
    } else if name.contains("floor") || name.contains("stone") {
        STUDIO_ROOM_FLOOR
    } else if name.contains("glass") {
        Color32::from_rgba_unmultiplied(122, 176, 198, 118)
    } else {
        match role {
            SurfaceRole::Floor => STUDIO_ROOM_FLOOR,
            SurfaceRole::Wall => STUDIO_ROOM_WALL,
            SurfaceRole::Object => Color32::from_rgb(125, 155, 190),
        }
    };

    if let ResourceData::Material(material) = &resource.data {
        if material.blend_mode != PsxBlendMode::Opaque {
            color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 132);
        }
    }

    color
}

fn material_is_translucent(project: &ProjectDocument, material: Option<ResourceId>) -> bool {
    material
        .and_then(|id| project.resource(id))
        .is_some_and(|resource| match &resource.data {
            ResourceData::Material(material) => material.blend_mode != PsxBlendMode::Opaque,
            _ => false,
        })
}

fn selected_stroke(selected: bool) -> Stroke {
    if selected {
        Stroke::new(3.0, STUDIO_GOLD)
    } else {
        Stroke::new(1.0, Color32::from_rgb(70, 84, 108))
    }
}

fn node_world(node: &psxed_project::SceneNode) -> [f32; 2] {
    [node.transform.translation[0], node.transform.translation[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq <= f32::EPSILON {
        return [0.0, 0.0, 1.0];
    }
    let inv = len_sq.sqrt().recip();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Outcome of a primitive delete. `Removed` => the face is gone
/// (user selected a face or edge). `Triangulated` => one corner
/// was dropped and the face is still alive as a triangle.
/// `Missing` => the face / sector wasn't where the selection
/// thought it was (stale state) -- caller should leave things
/// alone.
enum DeleteOutcome {
    Removed(&'static str),
    Triangulated(&'static str),
    Missing,
}

/// Short label for a face kind, used in delete status messages.
const fn describe_face_kind(kind: FaceKind) -> &'static str {
    match kind {
        FaceKind::Floor => "floor",
        FaceKind::Ceiling => "ceiling",
        FaceKind::Wall { .. } => "wall",
    }
}

/// One-line human description of a `FaceRef` for status messages
/// and the inspector header. Walls include their cardinal direction
/// + stack index since a single edge can carry several stacked
///   walls (windows / arches).
fn describe_face(face: FaceRef) -> String {
    match face.kind {
        FaceKind::Floor => format!("Floor at {},{}", face.sx, face.sz),
        FaceKind::Ceiling => format!("Ceiling at {},{}", face.sx, face.sz),
        FaceKind::Wall { dir, stack } => {
            format!(
                "{} wall #{stack} at {},{}",
                direction_label(dir),
                face.sx,
                face.sz
            )
        }
    }
}

/// Status-line / breadcrumb text for any `Selection`. Falls
/// through to `describe_face` for face selections so existing
/// face copy stays identical.
pub fn describe_selection(selection: Selection) -> String {
    match selection {
        Selection::Face(face) => describe_face(face),
        Selection::Edge(edge) => describe_edge(edge),
        Selection::Vertex(vertex) => describe_vertex(vertex),
    }
}

fn describe_edge(edge: EdgeRef) -> String {
    match edge.anchor {
        EdgeAnchor::Floor { sx, sz, dir } => {
            format!("Floor {} edge at {sx},{sz}", direction_label(dir))
        }
        EdgeAnchor::Ceiling { sx, sz, dir } => {
            format!("Ceiling {} edge at {sx},{sz}", direction_label(dir))
        }
        EdgeAnchor::Wall {
            sx,
            sz,
            dir,
            stack,
            edge,
        } => format!(
            "{} wall #{stack} {} edge at {sx},{sz}",
            direction_label(dir),
            wall_edge_label(edge),
        ),
    }
}

fn describe_vertex(vertex: VertexRef) -> String {
    match vertex.anchor {
        VertexAnchor::Floor { sx, sz, corner } => {
            format!("Floor {} vertex at {sx},{sz}", corner_label(corner))
        }
        VertexAnchor::Ceiling { sx, sz, corner } => {
            format!("Ceiling {} vertex at {sx},{sz}", corner_label(corner))
        }
        VertexAnchor::Wall {
            sx,
            sz,
            dir,
            stack,
            corner,
        } => format!(
            "{} wall #{stack} {} vertex at {sx},{sz}",
            direction_label(dir),
            wall_corner_label(corner),
        ),
    }
}

/// Adapter: face → its first edge (north for floor / ceiling,
/// bottom for walls). Used by mode-switch logic so a face
/// selection naturally promotes into one of its edges.
fn face_first_edge(face: FaceRef) -> EdgeRef {
    let anchor = match face.kind {
        FaceKind::Floor => EdgeAnchor::Floor {
            sx: face.sx,
            sz: face.sz,
            dir: GridDirection::North,
        },
        FaceKind::Ceiling => EdgeAnchor::Ceiling {
            sx: face.sx,
            sz: face.sz,
            dir: GridDirection::North,
        },
        FaceKind::Wall { dir, stack } => EdgeAnchor::Wall {
            sx: face.sx,
            sz: face.sz,
            dir,
            stack,
            edge: WallEdge::Bottom,
        },
    };
    EdgeRef {
        room: face.room,
        anchor,
    }
}

/// Adapter: face → its first vertex (NW for floor / ceiling,
/// BL for walls).
fn face_first_vertex(face: FaceRef) -> VertexRef {
    let anchor = match face.kind {
        FaceKind::Floor => VertexAnchor::Floor {
            sx: face.sx,
            sz: face.sz,
            corner: Corner::NW,
        },
        FaceKind::Ceiling => VertexAnchor::Ceiling {
            sx: face.sx,
            sz: face.sz,
            corner: Corner::NW,
        },
        FaceKind::Wall { dir, stack } => VertexAnchor::Wall {
            sx: face.sx,
            sz: face.sz,
            dir,
            stack,
            corner: WallCorner::BL,
        },
    };
    VertexRef {
        room: face.room,
        anchor,
    }
}

/// Adapter: edge → the face that owns it. Used when the user
/// switches from Edge mode back to Face mode and we don't want
/// to lose context.
fn edge_owning_face_ref(edge: EdgeRef) -> Option<FaceRef> {
    let kind = match edge.anchor {
        EdgeAnchor::Floor { .. } => FaceKind::Floor,
        EdgeAnchor::Ceiling { .. } => FaceKind::Ceiling,
        EdgeAnchor::Wall { dir, stack, .. } => FaceKind::Wall { dir, stack },
    };
    let (sx, sz) = match edge.anchor {
        EdgeAnchor::Floor { sx, sz, .. }
        | EdgeAnchor::Ceiling { sx, sz, .. }
        | EdgeAnchor::Wall { sx, sz, .. } => (sx, sz),
    };
    Some(FaceRef {
        room: edge.room,
        sx,
        sz,
        kind,
    })
}

/// Adapter: edge → its first endpoint vertex. Edge perimeter
/// convention: floor / ceiling north = NW-NE, east = NE-SE,
/// south = SE-SW, west = SW-NW. Wall bottom = BL-BR, right =
/// BR-TR, top = TR-TL, left = TL-BL. The "first" vertex is
/// the leading corner of that walk.
fn edge_first_vertex(edge: EdgeRef) -> VertexRef {
    let anchor = match edge.anchor {
        EdgeAnchor::Floor { sx, sz, dir } => VertexAnchor::Floor {
            sx,
            sz,
            corner: edge_first_floor_corner(dir),
        },
        EdgeAnchor::Ceiling { sx, sz, dir } => VertexAnchor::Ceiling {
            sx,
            sz,
            corner: edge_first_floor_corner(dir),
        },
        EdgeAnchor::Wall {
            sx,
            sz,
            dir,
            stack,
            edge,
        } => VertexAnchor::Wall {
            sx,
            sz,
            dir,
            stack,
            corner: edge_first_wall_corner(edge),
        },
    };
    VertexRef {
        room: edge.room,
        anchor,
    }
}

const fn edge_first_floor_corner(dir: GridDirection) -> Corner {
    match dir {
        GridDirection::North => Corner::NW,
        GridDirection::East => Corner::NE,
        GridDirection::South => Corner::SE,
        GridDirection::West => Corner::SW,
        // Diagonals shouldn't reach this code path (cooker
        // rejects them); pick NW arbitrarily so the function
        // stays total.
        _ => Corner::NW,
    }
}

const fn edge_first_wall_corner(edge: WallEdge) -> WallCorner {
    match edge {
        WallEdge::Bottom => WallCorner::BL,
        WallEdge::Right => WallCorner::BR,
        WallEdge::Top => WallCorner::TR,
        WallEdge::Left => WallCorner::TL,
    }
}

/// Adapter: vertex → owning face.
fn vertex_owning_face_ref(vertex: VertexRef) -> Option<FaceRef> {
    let kind = match vertex.anchor {
        VertexAnchor::Floor { .. } => FaceKind::Floor,
        VertexAnchor::Ceiling { .. } => FaceKind::Ceiling,
        VertexAnchor::Wall { dir, stack, .. } => FaceKind::Wall { dir, stack },
    };
    let (sx, sz) = match vertex.anchor {
        VertexAnchor::Floor { sx, sz, .. }
        | VertexAnchor::Ceiling { sx, sz, .. }
        | VertexAnchor::Wall { sx, sz, .. } => (sx, sz),
    };
    Some(FaceRef {
        room: vertex.room,
        sx,
        sz,
        kind,
    })
}

/// Adapter: vertex → one of the two edges it sits on. Picks
/// the first walking the perimeter from this corner.
fn vertex_first_edge(vertex: VertexRef) -> EdgeRef {
    let anchor = match vertex.anchor {
        VertexAnchor::Floor { sx, sz, corner } => EdgeAnchor::Floor {
            sx,
            sz,
            dir: floor_corner_first_edge(corner),
        },
        VertexAnchor::Ceiling { sx, sz, corner } => EdgeAnchor::Ceiling {
            sx,
            sz,
            dir: floor_corner_first_edge(corner),
        },
        VertexAnchor::Wall {
            sx,
            sz,
            dir,
            stack,
            corner,
        } => EdgeAnchor::Wall {
            sx,
            sz,
            dir,
            stack,
            edge: wall_corner_first_edge(corner),
        },
    };
    EdgeRef {
        room: vertex.room,
        anchor,
    }
}

const fn floor_corner_first_edge(corner: Corner) -> GridDirection {
    match corner {
        Corner::NW => GridDirection::North,
        Corner::NE => GridDirection::East,
        Corner::SE => GridDirection::South,
        Corner::SW => GridDirection::West,
    }
}

const fn wall_corner_first_edge(corner: WallCorner) -> WallEdge {
    match corner {
        WallCorner::BL => WallEdge::Bottom,
        WallCorner::BR => WallEdge::Right,
        WallCorner::TR => WallEdge::Top,
        WallCorner::TL => WallEdge::Left,
    }
}

const fn corner_label(corner: Corner) -> &'static str {
    match corner {
        Corner::NW => "NW",
        Corner::NE => "NE",
        Corner::SE => "SE",
        Corner::SW => "SW",
    }
}

const fn wall_corner_label(corner: WallCorner) -> &'static str {
    match corner {
        WallCorner::BL => "bottom-left",
        WallCorner::BR => "bottom-right",
        WallCorner::TR => "top-right",
        WallCorner::TL => "top-left",
    }
}

const fn wall_edge_label(edge: WallEdge) -> &'static str {
    match edge {
        WallEdge::Bottom => "bottom",
        WallEdge::Right => "right",
        WallEdge::Top => "top",
        WallEdge::Left => "left",
    }
}

fn direction_label(dir: GridDirection) -> &'static str {
    match dir {
        GridDirection::North => "North",
        GridDirection::East => "East",
        GridDirection::South => "South",
        GridDirection::West => "West",
        _ => "Diag",
    }
}

/// Pick the cardinal `GridDirection` for a wall edge given the
/// click offset from the cell's world-space centre. Mirrors the
/// renderer's `WallEdge` mapping in `editor_preview`:
/// `North = +Z`, `East = +X`, `South = -Z`, `West = -X`. The
/// dominant axis decides; ties favour the X axis.
fn edge_from_world_offset(dx: f32, dz: f32) -> GridDirection {
    psxed_project::spatial::editor_wall_direction_from_offset(dx, dz)
}

/// World-space integer position of `corner` in the room
/// described by `grid`. Returns `None` if the addressed face
/// no longer exists (cell out of bounds, geometry missing).
pub fn face_corner_world(grid: &WorldGrid, corner: FaceCornerRef) -> Option<[i32; 3]> {
    match corner {
        FaceCornerRef::Floor { sx, sz, corner } => {
            if sx >= grid.width || sz >= grid.depth {
                return None;
            }
            let face = grid.sector(sx, sz)?.floor.as_ref()?;
            Some(floor_corner_world(grid, sx, sz, corner, face.heights))
        }
        FaceCornerRef::Ceiling { sx, sz, corner } => {
            if sx >= grid.width || sz >= grid.depth {
                return None;
            }
            let face = grid.sector(sx, sz)?.ceiling.as_ref()?;
            Some(floor_corner_world(grid, sx, sz, corner, face.heights))
        }
        FaceCornerRef::Wall {
            sx,
            sz,
            dir,
            stack,
            corner,
        } => {
            if sx >= grid.width || sz >= grid.depth {
                return None;
            }
            let walls = grid.sector(sx, sz)?.walls.get(dir);
            let wall = walls.get(stack as usize)?;
            wall_corner_world(grid, sx, sz, dir, corner, wall.heights)
        }
    }
}

fn floor_corner_world(
    grid: &WorldGrid,
    sx: u16,
    sz: u16,
    corner: Corner,
    heights: [i32; 4],
) -> [i32; 3] {
    let [x, z] = grid.cell_bounds_world(sx, sz).horizontal_corner_xz(corner);
    [x, heights[corner.idx()], z]
}

fn horizontal_face_world_corners(bounds: GridCellBounds, heights: [i32; 4]) -> [[f32; 3]; 4] {
    let nw = bounds.horizontal_corner_xz(Corner::NW);
    let ne = bounds.horizontal_corner_xz(Corner::NE);
    let se = bounds.horizontal_corner_xz(Corner::SE);
    let sw = bounds.horizontal_corner_xz(Corner::SW);
    [
        [nw[0] as f32, heights[Corner::NW.idx()] as f32, nw[1] as f32],
        [ne[0] as f32, heights[Corner::NE.idx()] as f32, ne[1] as f32],
        [se[0] as f32, heights[Corner::SE.idx()] as f32, se[1] as f32],
        [sw[0] as f32, heights[Corner::SW.idx()] as f32, sw[1] as f32],
    ]
}

fn wall_corner_world(
    grid: &WorldGrid,
    sx: u16,
    sz: u16,
    dir: GridDirection,
    corner: WallCorner,
    heights: [i32; 4],
) -> Option<[i32; 3]> {
    let (bl, br) = grid.cell_bounds_world(sx, sz).wall_endpoints_xz(dir)?;
    let [x, z] = match corner {
        // BL / TL share the BL endpoint; BR / TR share BR.
        WallCorner::BL | WallCorner::TL => bl,
        WallCorner::BR | WallCorner::TR => br,
    };
    Some([x, heights[corner.idx()], z])
}

fn wall_face_world_corners(
    bounds: GridCellBounds,
    dir: GridDirection,
    heights: [i32; 4],
) -> Option<[[f32; 3]; 4]> {
    let (bl, br) = bounds.wall_endpoints_xz(dir)?;
    Some([
        [
            bl[0] as f32,
            heights[WallCorner::BL.idx()] as f32,
            bl[1] as f32,
        ],
        [
            br[0] as f32,
            heights[WallCorner::BR.idx()] as f32,
            br[1] as f32,
        ],
        [
            br[0] as f32,
            heights[WallCorner::TR.idx()] as f32,
            br[1] as f32,
        ],
        [
            bl[0] as f32,
            heights[WallCorner::TL.idx()] as f32,
            bl[1] as f32,
        ],
    ])
}

/// Universal coincidence resolver. Returns the physical vertex
/// containing `seed` -- every face-corner whose current world
/// position equals the seed's world position. Walks every
/// floor / ceiling / wall corner in the grid (`O(faces × 4)`,
/// runs in microseconds for 32×32 rooms).
pub fn physical_vertex(grid: &WorldGrid, seed: FaceCornerRef) -> Option<PhysicalVertex> {
    let world = face_corner_world(grid, seed)?;
    let mut members = Vec::new();
    for sx in 0..grid.width {
        for sz in 0..grid.depth {
            let Some(sector) = grid.sector(sx, sz) else {
                continue;
            };
            if sector.floor.is_some() {
                for c in [Corner::NW, Corner::NE, Corner::SE, Corner::SW] {
                    let r = FaceCornerRef::Floor { sx, sz, corner: c };
                    if face_corner_world(grid, r) == Some(world) {
                        members.push(r);
                    }
                }
            }
            if sector.ceiling.is_some() {
                for c in [Corner::NW, Corner::NE, Corner::SE, Corner::SW] {
                    let r = FaceCornerRef::Ceiling { sx, sz, corner: c };
                    if face_corner_world(grid, r) == Some(world) {
                        members.push(r);
                    }
                }
            }
            for dir in GridDirection::CARDINAL {
                for (stack_idx, _) in sector.walls.get(dir).iter().enumerate() {
                    for c in [
                        WallCorner::BL,
                        WallCorner::BR,
                        WallCorner::TR,
                        WallCorner::TL,
                    ] {
                        let r = FaceCornerRef::Wall {
                            sx,
                            sz,
                            dir,
                            stack: stack_idx as u8,
                            corner: c,
                        };
                        if face_corner_world(grid, r) == Some(world) {
                            members.push(r);
                        }
                    }
                }
            }
        }
    }
    Some(PhysicalVertex { world, members })
}

/// Face-corner seeds for a drag-translate stroke. The drag
/// engine resolves each seed through `physical_vertex` so all
/// coincident face-corners follow.
///
/// - Face: 4 corners of the face (preserves slope; same Δ on
///   each).
/// - Edge: 2 endpoint corners.
/// - Vertex: 1 corner.
fn drag_corner_seeds(selection: Selection) -> Option<Vec<FaceCornerRef>> {
    Some(match selection {
        Selection::Face(face) => match face.kind {
            FaceKind::Floor => [Corner::NW, Corner::NE, Corner::SE, Corner::SW]
                .iter()
                .map(|c| FaceCornerRef::Floor {
                    sx: face.sx,
                    sz: face.sz,
                    corner: *c,
                })
                .collect(),
            FaceKind::Ceiling => [Corner::NW, Corner::NE, Corner::SE, Corner::SW]
                .iter()
                .map(|c| FaceCornerRef::Ceiling {
                    sx: face.sx,
                    sz: face.sz,
                    corner: *c,
                })
                .collect(),
            FaceKind::Wall { dir, stack } => [
                WallCorner::BL,
                WallCorner::BR,
                WallCorner::TR,
                WallCorner::TL,
            ]
            .iter()
            .map(|c| FaceCornerRef::Wall {
                sx: face.sx,
                sz: face.sz,
                dir,
                stack,
                corner: *c,
            })
            .collect(),
        },
        Selection::Edge(edge) => {
            let (a, b) = edge_endpoint_corners(edge);
            vec![a, b]
        }
        Selection::Vertex(vertex) => vec![vertex.anchor.as_face_corner()],
    })
}

/// Pair of physical vertices that bound `edge`. Returns `None`
/// if either endpoint can't be resolved (face removed, cell
/// out of bounds).
pub fn edge_endpoints(grid: &WorldGrid, edge: EdgeRef) -> Option<(PhysicalVertex, PhysicalVertex)> {
    let (a, b) = edge_endpoint_corners(edge);
    let pa = physical_vertex(grid, a)?;
    let pb = physical_vertex(grid, b)?;
    Some((pa, pb))
}

/// Endpoint face-corners of `edge` as `(start, end)`. Order
/// matches the perimeter walk used elsewhere -- north = NW→NE,
/// east = NE→SE, etc.
fn edge_endpoint_corners(edge: EdgeRef) -> (FaceCornerRef, FaceCornerRef) {
    match edge.anchor {
        EdgeAnchor::Floor { sx, sz, dir } => {
            let (ca, cb) = floor_edge_endpoints(dir);
            (
                FaceCornerRef::Floor { sx, sz, corner: ca },
                FaceCornerRef::Floor { sx, sz, corner: cb },
            )
        }
        EdgeAnchor::Ceiling { sx, sz, dir } => {
            let (ca, cb) = floor_edge_endpoints(dir);
            (
                FaceCornerRef::Ceiling { sx, sz, corner: ca },
                FaceCornerRef::Ceiling { sx, sz, corner: cb },
            )
        }
        EdgeAnchor::Wall {
            sx,
            sz,
            dir,
            stack,
            edge,
        } => {
            let (ca, cb) = wall_edge_endpoints(edge);
            (
                FaceCornerRef::Wall {
                    sx,
                    sz,
                    dir,
                    stack,
                    corner: ca,
                },
                FaceCornerRef::Wall {
                    sx,
                    sz,
                    dir,
                    stack,
                    corner: cb,
                },
            )
        }
    }
}

const fn floor_edge_endpoints(dir: GridDirection) -> (Corner, Corner) {
    match dir {
        GridDirection::North => (Corner::NW, Corner::NE),
        GridDirection::East => (Corner::NE, Corner::SE),
        GridDirection::South => (Corner::SE, Corner::SW),
        GridDirection::West => (Corner::SW, Corner::NW),
        // Diagonals -- pick the two corners on the diagonal so
        // the inspector can at least show something. Picker
        // doesn't produce these because the cooker rejects
        // diagonal walls.
        _ => (Corner::NW, Corner::SE),
    }
}

const fn wall_edge_endpoints(edge: WallEdge) -> (WallCorner, WallCorner) {
    match edge {
        WallEdge::Bottom => (WallCorner::BL, WallCorner::BR),
        WallEdge::Right => (WallCorner::BR, WallCorner::TR),
        WallEdge::Top => (WallCorner::TR, WallCorner::TL),
        WallEdge::Left => (WallCorner::TL, WallCorner::BL),
    }
}

/// Inspector member-list label.
fn face_corner_label(corner: FaceCornerRef) -> String {
    match corner {
        FaceCornerRef::Floor { sx, sz, corner } => {
            format!("Floor ({sx},{sz}) {}", corner_label(corner))
        }
        FaceCornerRef::Ceiling { sx, sz, corner } => {
            format!("Ceiling ({sx},{sz}) {}", corner_label(corner))
        }
        FaceCornerRef::Wall {
            sx,
            sz,
            dir,
            stack,
            corner,
        } => format!(
            "{} wall #{stack} ({sx},{sz}) {}",
            direction_label(dir),
            wall_corner_label(corner)
        ),
    }
}

/// Apply a new Y to every member of `vertex`. X / Z are
/// preserved by construction -- `face_corner_world` returns the
/// current `(X, Y, Z)` and we only ever rewrite the corner's
/// height array entry.
pub fn apply_vertex_height(grid: &mut WorldGrid, vertex: &PhysicalVertex, new_y: i32) {
    let new_y = snap_height(new_y);
    for member in &vertex.members {
        write_face_corner_height(grid, *member, new_y);
    }
}

fn write_face_corner_height(grid: &mut WorldGrid, corner: FaceCornerRef, new_y: i32) {
    match corner {
        FaceCornerRef::Floor { sx, sz, corner } => {
            if let Some(sector) = grid.sector_mut(sx, sz) {
                if let Some(face) = sector.floor.as_mut() {
                    face.heights[corner.idx()] = new_y;
                }
            }
        }
        FaceCornerRef::Ceiling { sx, sz, corner } => {
            if let Some(sector) = grid.sector_mut(sx, sz) {
                if let Some(face) = sector.ceiling.as_mut() {
                    face.heights[corner.idx()] = new_y;
                }
            }
        }
        FaceCornerRef::Wall {
            sx,
            sz,
            dir,
            stack,
            corner,
        } => {
            if let Some(sector) = grid.sector_mut(sx, sz) {
                if let Some(wall) = sector.walls.get_mut(dir).get_mut(stack as usize) {
                    wall.heights[corner.idx()] = new_y;
                }
            }
        }
    }
}

/// Decompose a horizontal face into its two ray-test
/// triangles, tagged with the corners they're built from. The
/// pick path skips a triangle whose member list contains the
/// face's `dropped_corner`.
type HorizontalTri = ([f32; 3], [f32; 3], [f32; 3], [Corner; 3]);
fn horizontal_triangles(
    nw: [f32; 3],
    ne: [f32; 3],
    se: [f32; 3],
    sw: [f32; 3],
    split: GridSplit,
) -> [HorizontalTri; 2] {
    match split {
        GridSplit::NorthWestSouthEast => [
            (nw, ne, se, [Corner::NW, Corner::NE, Corner::SE]),
            (nw, se, sw, [Corner::NW, Corner::SE, Corner::SW]),
        ],
        GridSplit::NorthEastSouthWest => [
            (nw, ne, sw, [Corner::NW, Corner::NE, Corner::SW]),
            (ne, se, sw, [Corner::NE, Corner::SE, Corner::SW]),
        ],
    }
}

/// Wall-quad triangle decomposition. The diagonal flips when
/// the dropped corner sits on the BL-TR line -- `BL` / `TR`
/// trigger the BR-TL diagonal.
type WallTri = ([f32; 3], [f32; 3], [f32; 3], [WallCorner; 3]);
fn wall_triangles(
    bl: [f32; 3],
    br: [f32; 3],
    tr: [f32; 3],
    tl: [f32; 3],
    dropped: Option<WallCorner>,
) -> [WallTri; 2] {
    let use_br_tl = matches!(dropped, Some(WallCorner::BL) | Some(WallCorner::TR));
    if use_br_tl {
        [
            (bl, br, tl, [WallCorner::BL, WallCorner::BR, WallCorner::TL]),
            (br, tr, tl, [WallCorner::BR, WallCorner::TR, WallCorner::TL]),
        ]
    } else {
        [
            (bl, br, tr, [WallCorner::BL, WallCorner::BR, WallCorner::TR]),
            (bl, tr, tl, [WallCorner::BL, WallCorner::TR, WallCorner::TL]),
        ]
    }
}

/// Index of the corner closest (3D distance) to `hit` among
/// the four `corners`. Caller is responsible for the corner
/// ordering convention -- `[NW, NE, SE, SW]` for floors /
/// ceilings, `[BL, BR, TR, TL]` for walls.
fn closest_corner_idx(corners: &[[f32; 3]; 4], hit: [f32; 3]) -> usize {
    let mut best = 0usize;
    let mut best_d2 = f32::INFINITY;
    for (i, c) in corners.iter().enumerate() {
        let d2 = dist2_3d(*c, hit);
        if d2 < best_d2 {
            best = i;
            best_d2 = d2;
        }
    }
    best
}

/// Index of the edge (perimeter walk: 0..3) closest to `hit`.
/// Edge `i` runs `corners[i] → corners[(i+1) % 4]`.
fn closest_edge_idx(corners: &[[f32; 3]; 4], hit: [f32; 3]) -> usize {
    let mut best = 0usize;
    let mut best_d2 = f32::INFINITY;
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let d2 = point_segment_dist2(hit, a, b);
        if d2 < best_d2 {
            best = i;
            best_d2 = d2;
        }
    }
    best
}

fn dist2_3d(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Squared distance from point `p` to the segment `a-b`.
fn point_segment_dist2(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if len2 > 0.0 {
        ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    dist2_3d(q, p)
}

/// Floor / ceiling corner index `0..3` → `Corner` (NW, NE,
/// SE, SW in perimeter order).
const fn floor_corner_idx(idx: usize) -> Corner {
    match idx {
        0 => Corner::NW,
        1 => Corner::NE,
        2 => Corner::SE,
        _ => Corner::SW,
    }
}

const fn wall_corner_idx(idx: usize) -> WallCorner {
    match idx {
        0 => WallCorner::BL,
        1 => WallCorner::BR,
        2 => WallCorner::TR,
        _ => WallCorner::TL,
    }
}

/// Floor / ceiling edge index `0..3` → cardinal `GridDirection`.
const fn floor_edge_dir(idx: usize) -> GridDirection {
    match idx {
        0 => GridDirection::North,
        1 => GridDirection::East,
        2 => GridDirection::South,
        _ => GridDirection::West,
    }
}

const fn wall_edge_idx(idx: usize) -> WallEdge {
    match idx {
        0 => WallEdge::Bottom,
        1 => WallEdge::Right,
        2 => WallEdge::Top,
        _ => WallEdge::Left,
    }
}

/// Möller–Trumbore ray-triangle intersection. Returns the ray
/// parameter `t` of the front-side hit, or `None` for misses /
/// back-face hits / degenerate triangles. Front-side only because
/// the editor draws every face once per OT slot -- picking the back
/// of a wall would land on the *opposite* room's geometry, which
/// reads as a bug to the user.
fn ray_triangle(
    orig: [f32; 3],
    dir: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> Option<f32> {
    let edge1 = sub3(v1, v0);
    let edge2 = sub3(v2, v0);
    let h = cross3(dir, edge2);
    let a = dot3(edge1, h);
    if a.abs() < 1e-6 {
        return None;
    }
    let f = 1.0 / a;
    let s = sub3(orig, v0);
    let u = f * dot3(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross3(s, edge1);
    let v = f * dot3(dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * dot3(edge2, q);
    if t > 1e-3 {
        Some(t)
    } else {
        None
    }
}

/// Lowercase + non-alphanumeric → `_`, matching the cooker's clip
/// naming convention so room files line up with the asset cooker.
fn sanitise_room_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "room".to_string()
    } else {
        trimmed
    }
}

/// Walk the active scene and collect every Room node as an
/// `(id, display name)` pair, used by Portal pickers.
/// Walk parent links until a `NodeKind::Room` is found.
/// Returns its `NodeId` or `None` if `node_id` lives outside
/// any Room.
fn enclosing_room_id(scene: &psxed_project::Scene, node_id: NodeId) -> Option<NodeId> {
    let mut current = scene.node(node_id)?.parent;
    while let Some(parent_id) = current {
        let parent = scene.node(parent_id)?;
        if matches!(parent.kind, NodeKind::Room { .. }) {
            return Some(parent_id);
        }
        current = parent.parent;
    }
    None
}

/// Per-kind half-extents in world units. Picked so:
/// - bounds are big enough to click reliably at typical
///   editor zoom levels,
/// - small enough that a Light or AudioSource doesn't block
///   selection of nearby grid faces,
/// - distinct enough to read at a glance.
///
/// `None` for node kinds that don't get a 3D bound (Room,
/// World, Node, Node3D -- the structural / non-spatial ones).
fn entity_bound_kind_and_size(
    workspace: &EditorWorkspace,
    node: &psxed_project::SceneNode,
) -> Option<(EntityBoundKind, [f32; 3])> {
    match &node.kind {
        NodeKind::Room { .. } | NodeKind::World | NodeKind::Node | NodeKind::Node3D => None,
        NodeKind::ModelRenderer { .. }
        | NodeKind::Animator { .. }
        | NodeKind::Collider { .. }
        | NodeKind::Interactable { .. }
        | NodeKind::CharacterController { .. }
        | NodeKind::AiController { .. }
        | NodeKind::Combat { .. }
        | NodeKind::PointLight { .. } => None,
        NodeKind::Entity | NodeKind::Actor => {
            if let Some(model) = entity_model_resource(workspace, node) {
                let h = (model.world_height as f32).max(256.0);
                let half_h = h * 0.5;
                let half_xz = (h / 3.0).max(192.0);
                return Some((EntityBoundKind::Model, [half_xz, half_h, half_xz]));
            }
            Some((EntityBoundKind::MeshFallback, [256.0, 256.0, 256.0]))
        }
        NodeKind::MeshInstance { mesh, .. } => {
            // Model-backed instance: scale the bound to the
            // model's `world_height` so a Wraith reads as a
            // standing humanoid box, not a marker cube. Falls
            // back to a fixed mesh box for unbound instances.
            if let Some(id) = mesh {
                if let Some(resource) = workspace.project.resource(*id) {
                    if let psxed_project::ResourceData::Model(model) = &resource.data {
                        let h = (model.world_height as f32).max(256.0);
                        let half_h = h * 0.5;
                        // Square footprint sized as roughly
                        // a third of the model height -- wide
                        // enough to click, tight enough that
                        // adjacent models don't overlap.
                        let half_xz = (h / 3.0).max(192.0);
                        return Some((EntityBoundKind::Model, [half_xz, half_h, half_xz]));
                    }
                }
            }
            Some((EntityBoundKind::MeshFallback, [256.0, 256.0, 256.0]))
        }
        NodeKind::SpawnPoint { .. } => Some((EntityBoundKind::SpawnPoint, [128.0, 256.0, 128.0])),
        NodeKind::Light { .. } => Some((EntityBoundKind::Light, [128.0, 128.0, 128.0])),
        NodeKind::Trigger { .. } => Some((EntityBoundKind::Trigger, [256.0, 256.0, 256.0])),
        NodeKind::Portal { .. } => Some((EntityBoundKind::Portal, [256.0, 256.0, 64.0])),
        NodeKind::AudioSource { .. } => Some((EntityBoundKind::AudioSource, [128.0, 128.0, 128.0])),
    }
}

fn entity_model_resource<'a>(
    workspace: &'a EditorWorkspace,
    node: &psxed_project::SceneNode,
) -> Option<&'a psxed_project::ModelResource> {
    let scene = workspace.project.active_scene();
    node.children
        .iter()
        .filter_map(|id| scene.node(*id))
        .find_map(|child| match &child.kind {
            NodeKind::ModelRenderer {
                model: Some(id), ..
            } => workspace
                .project
                .resource(*id)
                .and_then(|resource| match &resource.data {
                    ResourceData::Model(model) => Some(model),
                    _ => None,
                }),
            NodeKind::CharacterController {
                character: Some(id),
                ..
            } => workspace
                .project
                .resource(*id)
                .and_then(|resource| match &resource.data {
                    ResourceData::Character(character) => character.model.and_then(|model_id| {
                        workspace
                            .project
                            .resource(model_id)
                            .and_then(|model_resource| match &model_resource.data {
                                ResourceData::Model(model) => Some(model),
                                _ => None,
                            })
                    }),
                    _ => None,
                }),
            _ => None,
        })
}

fn collect_room_options(project: &ProjectDocument) -> Vec<(NodeId, String)> {
    project
        .active_scene()
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::Room { .. }))
        .map(|node| (node.id, node.name.clone()))
        .collect()
}

/// Collect every Model resource as a `(resource id, display
/// name, clip names)` row. The MeshInstance inspector uses
/// this to render its clip-name combo box.
fn collect_model_options(project: &ProjectDocument) -> Vec<(ResourceId, String, Vec<String>)> {
    project
        .resources
        .iter()
        .filter_map(|r| match &r.data {
            ResourceData::Model(m) => Some((
                r.id,
                r.name.clone(),
                m.clips.iter().map(|c| c.name.clone()).collect(),
            )),
            _ => None,
        })
        .collect()
}

/// Collect every Character resource as `(id, name)`. The
/// player-spawn inspector uses this to populate its picker.
fn collect_character_options(project: &ProjectDocument) -> Vec<(ResourceId, String)> {
    project
        .resources
        .iter()
        .filter_map(|r| match &r.data {
            ResourceData::Character(_) => Some((r.id, r.name.clone())),
            _ => None,
        })
        .collect()
}

/// Per-cell inspector for one sector inside the active Room.
///
/// Renders a CollapsingHeader with floor/ceiling toggles, a single
/// flat height per face (corner authoring lands later), a material
/// Render the room-budget summary block. Counts on the left,
/// hard caps on the right; rows tinted red once they cross.
/// Surfaces both `.psxw` v1 and v2-estimate sizes so authors
/// can see what the format change is buying.
fn draw_room_budget(ui: &mut egui::Ui, budget: WorldGridBudget) {
    let over = budget.over_budget();
    let header = if over {
        icons::label(icons::TRASH, "Budget — over limit")
    } else {
        icons::label(icons::SCAN, "Budget")
    };
    egui::CollapsingHeader::new(header)
        .default_open(over)
        .show(ui, |ui| {
            let row = |ui: &mut egui::Ui, key: &str, val: String, hot: bool| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(key).color(STUDIO_TEXT_WEAK));
                    let txt = RichText::new(val).monospace();
                    ui.label(if hot {
                        txt.color(Color32::from_rgb(0xE0, 0x60, 0x60))
                    } else {
                        txt
                    });
                });
            };
            row(
                ui,
                "Cells",
                format!(
                    "{} populated / {} total",
                    budget.populated_cells, budget.total_cells
                ),
                budget.total_cells > (MAX_ROOM_WIDTH as usize) * (MAX_ROOM_DEPTH as usize),
            );
            row(ui, "Floors", format!("{}", budget.floors), false);
            row(ui, "Ceilings", format!("{}", budget.ceilings), false);
            row(ui, "Walls", format!("{}", budget.walls), false);
            row(
                ui,
                "Triangles",
                format!("{} / {}", budget.triangles, MAX_ROOM_TRIANGLES),
                budget.triangles > MAX_ROOM_TRIANGLES,
            );
            row(
                ui,
                ".psxw v1",
                format!(
                    "{} / {}",
                    human_bytes(budget.psxw_v1_bytes as u32),
                    human_bytes(MAX_ROOM_BYTES as u32)
                ),
                budget.psxw_v1_bytes > MAX_ROOM_BYTES,
            );
            row(
                ui,
                ".psxw v2 est.",
                human_bytes(budget.future_compact_estimated_bytes as u32).to_string(),
                budget.future_compact_estimated_bytes > MAX_ROOM_BYTES,
            );
        });
}

/// dropdown for each face, and a row of toggles for the four
/// cardinal walls. Returns `true` if any field changed so the
/// workspace can mark the project dirty in one place.
fn draw_sector_inspector(
    ui: &mut egui::Ui,
    project: &mut ProjectDocument,
    room_id: NodeId,
    sx: u16,
    sz: u16,
    material_options: &[(ResourceId, String)],
    nav_target: &mut Option<ResourceId>,
) -> bool {
    let scene = project.active_scene_mut();
    let Some(room) = scene.node_mut(room_id) else {
        return false;
    };
    let NodeKind::Room { grid } = &mut room.kind else {
        return false;
    };
    if sx >= grid.width || sz >= grid.depth {
        return false;
    }
    let sector_size = grid.sector_size;
    let mut changed = false;

    egui::CollapsingHeader::new(icons::label(icons::GRID, "Sector"))
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Cell");
                ui.monospace(format!("{sx}, {sz}"));
            });

            let sector = grid.ensure_sector(sx, sz);
            let Some(sector) = sector else {
                ui.weak("Cell out of grid bounds");
                return;
            };

            // Floor row: enabled toggle + height + material picker.
            ui.horizontal(|ui| {
                let mut has_floor = sector.floor.is_some();
                if ui.checkbox(&mut has_floor, "Floor").changed() {
                    sector.floor = if has_floor {
                        Some(GridHorizontalFace::flat(0, None))
                    } else {
                        None
                    };
                    changed = true;
                }
            });
            if let Some(floor) = sector.floor.as_mut() {
                changed |= height_row("    Height", &mut floor.heights, ui);
                changed |= material_picker(
                    ui,
                    "    Material",
                    &mut floor.material,
                    material_options,
                    nav_target,
                );
            }

            ui.separator();

            // Ceiling row.
            ui.horizontal(|ui| {
                let mut has_ceiling = sector.ceiling.is_some();
                if ui.checkbox(&mut has_ceiling, "Ceiling").changed() {
                    sector.ceiling = if has_ceiling {
                        Some(GridHorizontalFace::flat(sector_size, None))
                    } else {
                        None
                    };
                    changed = true;
                }
            });
            if let Some(ceiling) = sector.ceiling.as_mut() {
                changed |= height_row("    Height", &mut ceiling.heights, ui);
                changed |= material_picker(
                    ui,
                    "    Material",
                    &mut ceiling.material,
                    material_options,
                    nav_target,
                );
            }

            ui.separator();
            ui.label(icons::label(icons::BRICK_WALL, "Walls"));
            for (label, dir) in [
                ("North", GridDirection::North),
                ("East", GridDirection::East),
                ("South", GridDirection::South),
                ("West", GridDirection::West),
            ] {
                changed |= wall_stack_row(
                    label,
                    sector.walls.get_mut(dir),
                    sector_size,
                    material_options,
                    nav_target,
                    ui,
                );
            }
        });

    changed
}

/// Stack-of-walls editor for a single sector edge (N/E/S/W).
///
/// PSX rooms commonly stack walls to model windows / arches: one
/// wall from `0..window_bottom`, another from `window_top..ceiling`.
/// The data model already allows N walls per edge -- this UI surfaces
/// it. Each wall row carries its own `[bottom, top]` and material;
/// `+` adds a new wall on top of the previous one (or `0..ceil` for
/// the first), `×` removes that row.
fn wall_stack_row(
    edge_label: &str,
    walls: &mut Vec<GridVerticalFace>,
    sector_size: i32,
    material_options: &[(ResourceId, String)],
    nav_target: &mut Option<ResourceId>,
    ui: &mut egui::Ui,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(edge_label);
        if ui
            .small_button("+")
            .on_hover_text("Add wall stack")
            .clicked()
        {
            // New wall sits above the highest existing wall, or
            // spans the full sector when this edge is empty.
            let bottom = walls
                .iter()
                .map(|w| w.heights[2].max(w.heights[3]))
                .max()
                .unwrap_or(0);
            let top = (bottom + sector_size).max(bottom + 1);
            walls.push(GridVerticalFace::flat(bottom, top, None));
            changed = true;
        }
    });
    let mut remove_at: Option<usize> = None;
    for (i, wall) in walls.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("    #{i}"));
            ui.label("bot");
            // bottom height = heights[0] = heights[1]; top = heights[2] = heights[3].
            let mut bot = wall.heights[0];
            let mut top = wall.heights[2];
            if ui
                .add(egui::DragValue::new(&mut bot).speed(HEIGHT_QUANTUM as f32))
                .changed()
            {
                let bot = snap_height(bot);
                wall.heights[0] = bot;
                wall.heights[1] = bot;
                changed = true;
            }
            ui.label("top");
            if ui
                .add(egui::DragValue::new(&mut top).speed(HEIGHT_QUANTUM as f32))
                .changed()
            {
                let top = snap_height(top);
                wall.heights[2] = top;
                wall.heights[3] = top;
                changed = true;
            }
            if ui.small_button("×").on_hover_text("Remove wall").clicked() {
                remove_at = Some(i);
            }
        });
        let pick_label = format!("    #{i} mat");
        changed |= material_picker(
            ui,
            &pick_label,
            &mut wall.material,
            material_options,
            nav_target,
        );
    }
    if let Some(i) = remove_at {
        walls.remove(i);
        changed = true;
    }
    changed
}

/// Editable row for a `[NW, NE, SE, SW]` corner-height array.
///
/// Renders one DragValue when the four corners agree (the common
/// "flat floor" case) and switches to a 2×2 grid of independent
/// DragValues -- laid out NW-NE / SW-SE so the on-screen position
/// matches the world-space corner -- once the heights diverge or
/// the user clicks the "Slope" toggle. Returns `true` whenever any
/// corner changed so the caller can mark the project dirty.
fn height_row(label: &str, heights: &mut [i32; 4], ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    let mut sloped =
        !(heights[0] == heights[1] && heights[1] == heights[2] && heights[2] == heights[3]);

    ui.horizontal(|ui| {
        ui.label(label);
        if ui
            .toggle_value(&mut sloped, "Slope")
            .on_hover_text("Edit each corner height independently.")
            .changed()
            && !sloped
        {
            // Collapse back to the NW value so the floor is flat
            // again -- predictable, matches how `flat()` builds.
            heights[1] = heights[0];
            heights[2] = heights[0];
            heights[3] = heights[0];
            changed = true;
        }
    });

    // DragValue speed must equal HEIGHT_QUANTUM so each "tick" of
    // mouse drag advances by one snap step. Combined with the
    // `snap_height` post-clamp, the value visibly walks
    // 0 → 32 → 64 → … without intermediate noise.
    if sloped {
        // 2×2 grid: NW NE on top row (z+), SW SE on bottom (z−).
        // The order in `heights` is [NW, NE, SE, SW] -- index map:
        //   top row: [0]=NW, [1]=NE
        //   bottom:  [3]=SW, [2]=SE
        egui::Grid::new(format!("{label}-corners"))
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                for &idx in &[0usize, 1, 3, 2] {
                    if ui
                        .add(egui::DragValue::new(&mut heights[idx]).speed(HEIGHT_QUANTUM as f32))
                        .changed()
                    {
                        heights[idx] = snap_height(heights[idx]);
                        changed = true;
                    }
                    if idx == 1 {
                        ui.end_row();
                    }
                }
                ui.end_row();
            });
    } else {
        ui.horizontal(|ui| {
            // Indent so the field aligns with the per-corner grid above.
            ui.label("    ");
            let mut h = heights[0];
            if ui
                .add(egui::DragValue::new(&mut h).speed(HEIGHT_QUANTUM as f32))
                .changed()
            {
                let snapped = snap_height(h);
                *heights = [snapped; 4];
                changed = true;
            }
        });
    }

    changed
}

/// Material picker used by the sector / face inspector.
///
/// `jump_to` is an out-param: clicking the `→` button writes
/// the selected material's resource id into it. The caller
/// applies the navigation after its borrows release. Returns
/// `true` if the picker changed `current`.
fn material_picker(
    ui: &mut egui::Ui,
    label: &str,
    current: &mut Option<ResourceId>,
    options: &[(ResourceId, String)],
    jump_to: &mut Option<ResourceId>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let preview = current
            .and_then(|id| {
                options
                    .iter()
                    .find(|(rid, _)| *rid == id)
                    .map(|(_, n)| n.as_str())
            })
            .unwrap_or("(none)");
        egui::ComboBox::from_id_salt(label.to_string())
            .selected_text(preview)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "(none)").clicked() {
                    *current = None;
                    changed = true;
                }
                for (id, name) in options {
                    if ui.selectable_label(*current == Some(*id), name).clicked() {
                        *current = Some(*id);
                        changed = true;
                    }
                }
            });
        if let Some(id) = *current {
            if ui
                .small_button("→")
                .on_hover_text("Open this material in the inspector")
                .clicked()
            {
                *jump_to = Some(id);
            }
        }
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_directory_saves_and_reloads_project() {
        let dir = std::env::temp_dir().join(format!(
            "psxed-ui-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let project_file = dir.join("project.ron");
        std::fs::write(
            &project_file,
            ProjectDocument::starter().to_ron_string().unwrap(),
        )
        .unwrap();

        let mut workspace = EditorWorkspace::open_directory(&dir).unwrap();
        assert!(!workspace.is_dirty());
        assert_eq!(workspace.project_root(), dir);
        workspace.save().unwrap();
        assert!(project_file.is_file());

        let loaded = EditorWorkspace::open_directory(&dir).unwrap();
        assert!(!loaded.is_dirty());
        assert_eq!(
            loaded.project().resources.len(),
            workspace.project().resources.len()
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn open_directory_errors_when_project_ron_missing() {
        let dir = std::env::temp_dir().join(format!(
            "psxed-ui-test-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let err = match EditorWorkspace::open_directory(&dir) {
            Ok(_) => panic!("expected open_directory to fail on missing project.ron"),
            Err(e) => e,
        };
        assert!(err.contains("project.ron"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn create_and_open_project_validates_name() {
        let mut ws = EditorWorkspace::open_directory(psxed_project::default_project_dir()).unwrap();
        assert!(ws.create_and_open_project("").is_err());
        assert!(ws.create_and_open_project("with/slash").is_err());
        assert!(ws.create_and_open_project("..").is_err());
        assert!(ws.create_and_open_project(".hidden").is_err());
        // "default" is a real existing dir, so this hits the "already exists" branch.
        assert!(ws.create_and_open_project("default").is_err());
    }

    #[test]
    fn viewport_transform_roundtrips_world_and_screen_points() {
        let transform = ViewportTransform::new(
            Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(300.0, 200.0)),
            Vec2::new(12.0, -8.0),
            40.0,
        );

        let world = [1.25, -0.5];
        let screen = transform.world_to_screen(world);
        let roundtrip = transform.screen_to_world(screen);

        assert!((roundtrip[0] - world[0]).abs() < 0.001);
        assert!((roundtrip[1] - world[1]).abs() < 0.001);
    }

    #[test]
    fn viewport_hits_rectangles_and_circles() {
        let rect = ViewportHit::rect(NodeId::ROOT, "Rect", [0.0, 0.0], [1.0, 0.5]);
        assert!(rect.contains([0.25, 0.25]));
        assert!(!rect.contains([1.25, 0.25]));

        let circle = ViewportHit::circle(NodeId::ROOT, "Circle", [2.0, 2.0], 0.5);
        assert!(circle.contains([2.25, 2.25]));
        assert!(!circle.contains([2.6, 2.0]));
    }

    fn starter_player_actor(scene: &psxed_project::Scene) -> &psxed_project::SceneNode {
        scene
            .nodes()
            .iter()
            .find(|node| {
                matches!(node.kind, NodeKind::Actor)
                    && node.children.iter().any(|id| {
                        scene.node(*id).is_some_and(|child| {
                            matches!(
                                child.kind,
                                NodeKind::CharacterController { player: true, .. }
                            )
                        })
                    })
            })
            .expect("starter has a player Actor")
    }

    #[test]
    fn dragging_selected_node_moves_it_in_xz_space() {
        let mut workspace =
            EditorWorkspace::open_directory(psxed_project::default_project_dir()).unwrap();
        let spawn = starter_player_actor(workspace.project.active_scene()).id;
        let start = workspace
            .project
            .active_scene()
            .node(spawn)
            .unwrap()
            .transform
            .translation;

        workspace.selected_node = spawn;
        workspace.drag_selected_node(Vec2::new(96.0, -48.0));

        let node = workspace.project.active_scene().node(spawn).unwrap();
        assert!((node.transform.translation[0] - (start[0] + 1.0)).abs() < 0.001);
        assert!((node.transform.translation[2] - (start[2] + 0.5)).abs() < 0.001);
        assert!(workspace.is_dirty());
    }

    #[test]
    fn scene_tree_select_clears_inspector_shadow_selection() {
        let mut workspace =
            EditorWorkspace::open_directory(psxed_project::default_project_dir()).unwrap();
        let scene = workspace.project.active_scene();
        let room = scene
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))
            .expect("starter scene has a Room")
            .id;
        let spawn = starter_player_actor(scene).id;
        let resource = workspace
            .project
            .resources
            .first()
            .expect("starter project has resources")
            .id;

        workspace.selected_node = NodeId::ROOT;
        workspace.selected_resource = Some(resource);
        workspace.selected_primitive = Some(Selection::Face(FaceRef {
            room,
            sx: 0,
            sz: 0,
            kind: FaceKind::Floor,
        }));

        workspace.apply_tree_action(TreeAction::Select(spawn));

        assert_eq!(workspace.selected_node, spawn);
        assert_eq!(workspace.selected_primitive, None);
        assert_eq!(workspace.selected_resource, None);
    }

    #[test]
    fn collect_entity_bounds_covers_starter_scene_entities() {
        let workspace =
            EditorWorkspace::open_directory(psxed_project::default_project_dir()).unwrap();
        let bounds = workspace.collect_entity_bounds(workspace.active_room_id());
        assert!(
            !bounds.is_empty(),
            "starter scene should expose at least one selectable entity bound"
        );
        let scene = workspace.project.active_scene();
        // Player Actor always exists in the starter project; its
        // bound should land in the active Room with a positive
        // half-extent on every axis.
        let spawn = starter_player_actor(scene);
        let spawn_bound = bounds
            .iter()
            .find(|b| b.node == spawn.id)
            .expect("player actor bound was emitted");
        assert!(matches!(
            spawn_bound.kind,
            EntityBoundKind::Model | EntityBoundKind::MeshFallback
        ));
        assert!(spawn_bound.half_extents[0] > 0.0);
        assert!(spawn_bound.half_extents[1] > 0.0);
        assert!(spawn_bound.half_extents[2] > 0.0);
    }

    #[test]
    fn dropping_model_resource_creates_component_entity() {
        let mut workspace =
            EditorWorkspace::open_directory(psxed_project::default_project_dir()).unwrap();
        let room = workspace.active_room_id().expect("starter has a room");
        let model_id = workspace
            .project
            .resources
            .iter()
            .find(|resource| matches!(resource.data, ResourceData::Model(_)))
            .expect("starter has a model")
            .id;

        workspace.drop_resource_at_room_hit(model_id, room, [512.0, 0.0, 512.0], None);

        let scene = workspace.project.active_scene();
        let entity = scene
            .node(workspace.selected_node)
            .expect("new entity is selected");
        assert!(matches!(entity.kind, NodeKind::Entity));
        assert!(entity.children.iter().any(|id| {
            scene.node(*id).is_some_and(|child| {
                matches!(
                    child.kind,
                    NodeKind::ModelRenderer {
                        model: Some(id),
                        ..
                    } if id == model_id
                )
            })
        }));
        assert!(entity.children.iter().any(|id| {
            scene
                .node(*id)
                .is_some_and(|child| matches!(child.kind, NodeKind::Animator { .. }))
        }));
        assert!(workspace.is_dirty());
    }

    #[test]
    fn dropping_character_resource_creates_actor_components() {
        let mut workspace =
            EditorWorkspace::open_directory(psxed_project::default_project_dir()).unwrap();
        let room = workspace.active_room_id().expect("starter has a room");
        let character_id = workspace
            .project
            .resources
            .iter()
            .find(|resource| matches!(resource.data, ResourceData::Character(_)))
            .expect("starter has a character")
            .id;

        workspace.drop_resource_at_room_hit(character_id, room, [512.0, 0.0, 512.0], None);

        let scene = workspace.project.active_scene();
        let actor = scene
            .node(workspace.selected_node)
            .expect("new actor is selected");
        assert!(matches!(actor.kind, NodeKind::Actor));
        assert!(actor.children.iter().any(|id| {
            scene.node(*id).is_some_and(|child| {
                matches!(
                    child.kind,
                    NodeKind::CharacterController {
                        character: Some(id),
                        player: false
                    } if id == character_id
                )
            })
        }));
        assert!(actor.children.iter().any(|id| {
            scene
                .node(*id)
                .is_some_and(|child| matches!(child.kind, NodeKind::Collider { .. }))
        }));
        assert!(workspace.is_dirty());
    }

    #[test]
    fn pick_entity_bound_returns_node_when_ray_hits_centre() {
        let workspace =
            EditorWorkspace::open_directory(psxed_project::default_project_dir()).unwrap();
        let bounds = workspace.collect_entity_bounds(workspace.active_room_id());
        let target = bounds
            .iter()
            .find(|b| {
                matches!(
                    b.kind,
                    EntityBoundKind::Model | EntityBoundKind::MeshFallback
                )
            })
            .copied()
            .expect("starter player Actor produces a bound");
        // Cast a ray straight at the bound's centre from far
        // outside it; ray_intersects_aabb is the primitive
        // pick_entity_bound calls into.
        let origin = [
            target.center[0] - 4096.0,
            target.center[1],
            target.center[2],
        ];
        let dir = [1.0, 0.0, 0.0];
        let t = ray_intersects_aabb(origin, dir, target.center, target.half_extents);
        assert!(t.is_some(), "ray straight at bound centre must hit");
    }

    #[test]
    fn project_filesystem_rows_are_generated_from_resources() {
        let project = ProjectDocument::starter();
        let rows = project_filesystem_rows(&project);

        assert!(rows.iter().any(|row| row.name == "res://"));
        assert!(rows.iter().any(|row| row.name == "main.room"));
        assert!(rows.iter().any(|row| row.name == "floor.psxt"));
        assert!(rows.iter().any(|row| row.name == "brick_wall.psxt"));
        assert!(rows
            .iter()
            .any(|row| row.name == "brick.mat" && row.resource.is_some()));
    }

    #[test]
    fn compact_middle_keeps_long_asset_names_dock_sized() {
        let name = "meshy_ai_obsidian_wraith_biped_meshy_ai_meshy_merged_animations.psxmdl";
        let compact = compact_middle(name, 32);

        assert!(compact.chars().count() <= 32);
        assert!(compact.starts_with("meshy_ai"));
        assert!(compact.ends_with(".psxmdl"));
        assert!(compact.contains("..."));
    }

    #[test]
    fn resource_filter_and_search_match_expected_resources() {
        let project = ProjectDocument::starter();
        let floor_texture = project
            .resources
            .iter()
            .find(|resource| resource.name == "floor.psxt")
            .unwrap();
        let brick_material = project
            .resources
            .iter()
            .find(|resource| resource.name == "Brick")
            .unwrap();

        assert!(resource_matches_filter(
            floor_texture,
            ResourceFilter::Texture,
            "floor"
        ));
        assert!(!resource_matches_filter(
            floor_texture,
            ResourceFilter::Material,
            "floor"
        ));
        assert!(resource_matches_filter(
            brick_material,
            ResourceFilter::Material,
            "brick"
        ));
    }

    fn cell_with_floor(material: Option<psxed_project::ResourceId>) -> psxed_project::GridSector {
        psxed_project::GridSector {
            floor: Some(psxed_project::GridHorizontalFace::flat(0, material)),
            ceiling: None,
            walls: psxed_project::GridWalls::default(),
        }
    }

    fn populated_grid(width: u16, depth: u16) -> WorldGrid {
        let mut grid = WorldGrid::empty(width, depth, 1024);
        for sx in 0..width {
            for sz in 0..depth {
                if let Some(s) = grid.ensure_sector(sx, sz) {
                    *s = cell_with_floor(None);
                }
            }
        }
        grid
    }

    #[test]
    fn physical_vertex_isolated_corner_returns_self_only() {
        let grid = populated_grid(1, 1);
        let seed = FaceCornerRef::Floor {
            sx: 0,
            sz: 0,
            corner: Corner::NW,
        };
        let pv = physical_vertex(&grid, seed).unwrap();
        assert_eq!(pv.members, vec![seed]);
    }

    #[test]
    fn physical_vertex_interior_grid_corner_returns_four_floors() {
        let grid = populated_grid(2, 2);
        // Cell (0, 0) NE shares its world position with three
        // other cells' corresponding corners.
        let seed = FaceCornerRef::Floor {
            sx: 0,
            sz: 0,
            corner: Corner::NE,
        };
        let pv = physical_vertex(&grid, seed).unwrap();
        assert_eq!(pv.members.len(), 4, "{:?}", pv.members);
        // Spot-check that the expected siblings are in the set.
        assert!(pv.members.contains(&FaceCornerRef::Floor {
            sx: 1,
            sz: 0,
            corner: Corner::NW,
        }));
        assert!(pv.members.contains(&FaceCornerRef::Floor {
            sx: 0,
            sz: 1,
            corner: Corner::SE,
        }));
        assert!(pv.members.contains(&FaceCornerRef::Floor {
            sx: 1,
            sz: 1,
            corner: Corner::SW,
        }));
    }

    #[test]
    fn physical_vertex_skips_unpopulated_cells() {
        // 2×2 grid with only three cells populated. The corner
        // they all share should yield exactly 3 members.
        let mut grid = WorldGrid::empty(2, 2, 1024);
        for (sx, sz) in [(0u16, 0u16), (1, 0), (0, 1)] {
            if let Some(s) = grid.ensure_sector(sx, sz) {
                *s = cell_with_floor(None);
            }
        }
        let seed = FaceCornerRef::Floor {
            sx: 0,
            sz: 0,
            corner: Corner::NE,
        };
        let pv = physical_vertex(&grid, seed).unwrap();
        assert_eq!(pv.members.len(), 3);
    }

    #[test]
    fn apply_vertex_height_writes_every_member() {
        let mut grid = populated_grid(2, 2);
        let seed = FaceCornerRef::Floor {
            sx: 0,
            sz: 0,
            corner: Corner::NE,
        };
        let pv = physical_vertex(&grid, seed).unwrap();
        apply_vertex_height(&mut grid, &pv, 64);
        for member in &pv.members {
            let world = face_corner_world(&grid, *member).unwrap();
            assert_eq!(world[1], 64, "{:?}", member);
        }
    }

    #[test]
    fn apply_vertex_height_break_action_separates_seed() {
        let mut grid = populated_grid(2, 2);
        let seed = FaceCornerRef::Floor {
            sx: 0,
            sz: 0,
            corner: Corner::NE,
        };
        // Capture the pre-break member set so we can confirm
        // exactly one corner left (the seed) when the break
        // mutates only the seed's height.
        let before = physical_vertex(&grid, seed).unwrap();
        assert_eq!(before.members.len(), 4);
        // Move only the seed by writing directly via the helper.
        write_face_corner_height(&mut grid, seed, 32);
        // Re-resolve from a former neighbour. Should now contain
        // 3 members (the seed has departed).
        let neighbour = FaceCornerRef::Floor {
            sx: 1,
            sz: 0,
            corner: Corner::NW,
        };
        let after = physical_vertex(&grid, neighbour).unwrap();
        assert_eq!(after.members.len(), 3);
        assert!(!after.members.contains(&seed));
    }

    #[test]
    fn closest_corner_idx_picks_nearest_corner() {
        let corners = [
            [0.0_f32, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
        ];
        // Each quadrant of the unit square should resolve to
        // the nearest corner.
        assert_eq!(closest_corner_idx(&corners, [1.0, 0.0, 1.0]), 0);
        assert_eq!(closest_corner_idx(&corners, [9.0, 0.0, 1.0]), 1);
        assert_eq!(closest_corner_idx(&corners, [9.0, 0.0, 9.0]), 2);
        assert_eq!(closest_corner_idx(&corners, [1.0, 0.0, 9.0]), 3);
    }

    #[test]
    fn closest_edge_idx_picks_nearest_edge() {
        let corners = [
            [0.0_f32, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
        ];
        // (5, 0, 0.5) → near edge 0 (corners 0–1).
        assert_eq!(closest_edge_idx(&corners, [5.0, 0.0, 0.5]), 0);
        // (9.5, 0, 5) → near edge 1 (corners 1–2).
        assert_eq!(closest_edge_idx(&corners, [9.5, 0.0, 5.0]), 1);
        // (5, 0, 9.5) → near edge 2 (corners 2–3).
        assert_eq!(closest_edge_idx(&corners, [5.0, 0.0, 9.5]), 2);
        // (0.5, 0, 5) → near edge 3 (corners 3–0).
        assert_eq!(closest_edge_idx(&corners, [0.5, 0.0, 5.0]), 3);
    }
}
