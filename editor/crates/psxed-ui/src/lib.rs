//! egui editor workspace for PSoXide.
//!
//! The frontend owns the window/Menu. This crate owns the editor panels and
//! the in-memory authoring document they manipulate.

mod icons;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use egui::{
    Align2, Color32, ColorImage, CornerRadius, FontId, Frame, Margin, Pos2, Rect, RichText, Sense,
    Stroke, StrokeKind, Vec2,
};
use psxed_project::{
    GridDirection, GridHorizontalFace, GridVerticalFace, MaterialResource, NodeId, NodeKind,
    NodeRow, ProjectDocument, PsxBlendMode, Resource, ResourceData, ResourceId, WorldGrid,
    WorldGridBudget, MAX_ROOM_BYTES, MAX_ROOM_DEPTH, MAX_ROOM_TRIANGLES, MAX_ROOM_WIDTH,
};

/// Maximum undo / redo snapshots retained.
const UNDO_CAPACITY: usize = 64;

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
    AddChild { parent: NodeId, kind: NodeKind, name: &'static str },
    /// Move `source` so it becomes a child of `target_parent` at
    /// `position` in that parent's child list. Caller has already
    /// proven the move is non-cyclic; `Scene::move_node` re-checks.
    Reparent {
        source: NodeId,
        target_parent: NodeId,
        position: usize,
    },
}

/// Snapshot-based undo. Each entry is a full `ProjectDocument` clone;
/// for hand-authored level data this is cheap and avoids the
/// command-pattern bookkeeping that operation-based undo demands.
#[derive(Default)]
struct UndoStack {
    undo: std::collections::VecDeque<ProjectDocument>,
    redo: std::collections::VecDeque<ProjectDocument>,
}

impl UndoStack {
    fn record(&mut self, snapshot: ProjectDocument) {
        if self.undo.len() == UNDO_CAPACITY {
            self.undo.pop_front();
        }
        self.undo.push_back(snapshot);
        self.redo.clear();
    }

    fn undo(&mut self, current: ProjectDocument) -> Option<ProjectDocument> {
        let prev = self.undo.pop_back()?;
        if self.redo.len() == UNDO_CAPACITY {
            self.redo.pop_front();
        }
        self.redo.push_back(current);
        Some(prev)
    }

    fn redo(&mut self, current: ProjectDocument) -> Option<ProjectDocument> {
        let next = self.redo.pop_back()?;
        if self.undo.len() == UNDO_CAPACITY {
            self.undo.pop_front();
        }
        self.undo.push_back(current);
        Some(next)
    }
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
    /// Face under the pointer while the Select tool is active —
    /// floors, walls, ceilings of the active Room. Updated every
    /// frame the panel is hovered and the tool is Select; cleared
    /// when the pointer leaves or another tool takes over. The
    /// renderer outlines this face lightly so the user sees what
    /// the next click will pick.
    hovered_face: Option<FaceRef>,
    /// Face the user clicked with the Select tool last. Persists
    /// across frames until the user clicks a different face or
    /// switches tools. The renderer outlines it more boldly than
    /// `hovered_face`; the inspector panel reads it to surface
    /// per-face properties (material, heights, …).
    selected_face: Option<FaceRef>,
    /// What the next paint click would target. Cell variant fires
    /// for floor / ceiling / erase / place; Wall variant fires for
    /// PaintWall. World-cell coords let the preview track cells
    /// outside the current grid bounds — the renderer outlines
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
    active_tool: ViewTool,
    /// Kind of node the Place tool drops on a click. Surfaces in
    /// the toolbar as a small picker visible only when
    /// `active_tool == Place`. Player Spawn enforces uniqueness
    /// at place-time; the others are additive markers.
    place_kind: PlaceKind,
    /// Material the next Floor / Wall / Ceiling paint will use, when
    /// set. `None` means "fall back to the name-based heuristic
    /// (`floor → first 'floor' material, brick → first 'brick' …`)" —
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
    dirty: bool,
    status: String,
}

/// One cached `.psxt` thumbnail plus the metadata the inspector
/// reads off the same parse. `signature` is the path the handle was
/// built from — when the user retypes the path on a Texture
/// resource, the signature mismatches and the cache rebuilds.
struct ThumbnailEntry {
    signature: String,
    handle: egui::TextureHandle,
    stats: PsxtStats,
}

/// Decoded metadata for one `.psxt` blob. Cheap to compute
/// (header parse + lengths); shown in the resource inspector so
/// authors can spot mismatches against an authored target depth /
/// dimensions without leaving the editor.
#[derive(Debug, Clone, Copy)]
struct PsxtStats {
    width: u16,
    height: u16,
    /// 4, 8, or 15 — mirrors `psxed_format::texture::Depth`'s
    /// numeric form.
    depth_bits: u8,
    /// 16 for 4bpp, 256 for 8bpp, 0 for 15bpp.
    clut_entries: u16,
    pixel_bytes: u32,
    clut_bytes: u32,
    file_bytes: u32,
}

/// Per-stamp paint dedupe key. Two paint events with equal stamps
/// are considered redundant during a single drag — typically the
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
/// coordinates (which can be negative — outside the current grid)
/// so the renderer can preview cells the next click would auto-
/// create. Stays populated for any paint tool, mirroring the
/// dispatch so what you preview is what you'll paint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintTargetPreview {
    /// Floor / ceiling / erase / place — outlines the cell.
    Cell { world_cell_x: i32, world_cell_z: i32 },
    /// PaintWall — outlines the wall on the targeted edge of the
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
/// — windows / arches — and each is independently selectable).
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewTool {
    /// Click to select; arrow-key nudge moves the selected node.
    Select,
    /// Drag a selected node in the viewport plane.
    Move,
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
    /// `SpawnPoint { player: true }` — the editor enforces
    /// uniqueness by demoting existing player spawns to
    /// generic spawns at place time.
    PlayerSpawn,
    /// `SpawnPoint { player: false }`. Multiple OK.
    SpawnMarker,
    /// `MeshInstance` with no resource yet — useful as a
    /// physical entity placeholder during authoring.
    EntityMarker,
    /// `Light` with default color / intensity / radius.
    LightMarker,
}

impl PlaceKind {
    const fn label(self) -> &'static str {
        match self {
            Self::PlayerSpawn => "Player Spawn",
            Self::SpawnMarker => "Spawn",
            Self::EntityMarker => "Entity",
            Self::LightMarker => "Light",
        }
    }
}

impl ViewTool {
    /// `true` when the tool only makes sense once a Room is the
    /// active context — viewport clicks should be suppressed
    /// otherwise so we don't paint into thin air.
    const fn requires_room_context(self) -> bool {
        matches!(
            self,
            Self::PaintFloor
                | Self::PaintWall
                | Self::PaintCeiling
                | Self::Erase
                | Self::Place
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceFilter {
    All,
    Texture,
    Material,
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
            Self::Mesh => matches!(data, ResourceData::Mesh { .. }),
            Self::Room => matches!(data, ResourceData::Scene { .. }),
            Self::Other => matches!(
                data,
                ResourceData::Script { .. } | ResourceData::Audio { .. }
            ),
        }
    }
}

const DEFAULT_VIEWPORT_ZOOM: f32 = 96.0;
const MIN_VIEWPORT_ZOOM: f32 = 24.0;
const MAX_VIEWPORT_ZOOM: f32 = 220.0;

const STUDIO_BG: Color32 = Color32::from_rgb(12, 16, 21);
const STUDIO_TOP_BAR: Color32 = Color32::from_rgb(18, 22, 28);
const STUDIO_PANEL: Color32 = Color32::from_rgb(17, 22, 28);
const STUDIO_PANEL_DARK: Color32 = Color32::from_rgb(13, 17, 22);
const STUDIO_INPUT: Color32 = Color32::from_rgb(11, 15, 20);
const STUDIO_VIEWPORT: Color32 = Color32::from_rgb(12, 24, 31);
const STUDIO_BORDER: Color32 = Color32::from_rgb(41, 51, 63);
const STUDIO_BORDER_DARK: Color32 = Color32::from_rgb(28, 36, 45);
const STUDIO_TEXT: Color32 = Color32::from_rgb(220, 229, 238);
const STUDIO_TEXT_WEAK: Color32 = Color32::from_rgb(142, 154, 168);
const STUDIO_ACCENT: Color32 = Color32::from_rgb(45, 177, 207);
const STUDIO_ACCENT_DIM: Color32 = Color32::from_rgb(17, 82, 101);
const STUDIO_GOLD: Color32 = Color32::from_rgb(238, 197, 119);
const STUDIO_ROOM_FLOOR: Color32 = Color32::from_rgb(119, 132, 143);
const STUDIO_ROOM_WALL: Color32 = Color32::from_rgb(126, 73, 43);

fn apply_studio_visuals(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut(|style| {
        style.spacing.item_spacing = Vec2::new(6.0, 4.0);
        style.spacing.button_padding = Vec2::new(8.0, 3.0);
        style.spacing.interact_size = Vec2::new(30.0, 22.0);
        style.spacing.window_margin = Margin::same(6);
        style.spacing.menu_margin = Margin::symmetric(8, 5);
        style.spacing.indent = 16.0;
        style.visuals = studio_visuals();
    });
}

fn studio_visuals() -> egui::Visuals {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(STUDIO_TEXT);
    visuals.panel_fill = STUDIO_PANEL_DARK;
    visuals.window_fill = STUDIO_PANEL;
    visuals.window_stroke = Stroke::new(1.0, STUDIO_BORDER);
    visuals.window_corner_radius = CornerRadius::same(3);
    visuals.menu_corner_radius = CornerRadius::same(3);
    visuals.faint_bg_color = Color32::from_rgb(20, 27, 35);
    visuals.extreme_bg_color = STUDIO_INPUT;
    visuals.code_bg_color = Color32::from_rgb(16, 21, 27);
    visuals.hyperlink_color = STUDIO_ACCENT;
    visuals.selection.bg_fill = STUDIO_ACCENT_DIM;
    visuals.selection.stroke = Stroke::new(1.0, STUDIO_ACCENT);
    visuals.button_frame = true;
    visuals.collapsing_header_frame = false;
    visuals.indent_has_left_vline = true;
    visuals.slider_trailing_fill = true;

    visuals.widgets.noninteractive.bg_fill = STUDIO_PANEL;
    visuals.widgets.noninteractive.weak_bg_fill = STUDIO_PANEL_DARK;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, STUDIO_BORDER);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, STUDIO_TEXT);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(3);

    visuals.widgets.inactive.bg_fill = Color32::from_rgb(20, 28, 36);
    visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(18, 26, 34);
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, STUDIO_BORDER);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, STUDIO_TEXT);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(3);

    visuals.widgets.hovered.bg_fill = Color32::from_rgb(28, 46, 57);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(23, 38, 48);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(65, 95, 112));
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(3);

    visuals.widgets.active.bg_fill = STUDIO_ACCENT_DIM;
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(18, 92, 113);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, STUDIO_ACCENT);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.active.corner_radius = CornerRadius::same(3);

    visuals.widgets.open.bg_fill = Color32::from_rgb(20, 30, 38);
    visuals.widgets.open.weak_bg_fill = Color32::from_rgb(20, 30, 38);
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, STUDIO_BORDER);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, STUDIO_TEXT);
    visuals.widgets.open.corner_radius = CornerRadius::same(3);
    visuals
}

fn top_bar_frame() -> Frame {
    Frame::new()
        .fill(STUDIO_TOP_BAR)
        .stroke(Stroke::new(1.0, STUDIO_BORDER_DARK))
        .inner_margin(Margin::symmetric(10, 4))
}

fn dock_frame() -> Frame {
    Frame::new()
        .fill(STUDIO_PANEL_DARK)
        .stroke(Stroke::new(1.0, STUDIO_BORDER))
        .inner_margin(Margin::symmetric(6, 6))
}

fn section_frame() -> Frame {
    Frame::new()
        .fill(STUDIO_PANEL)
        .stroke(Stroke::new(1.0, STUDIO_BORDER))
        .corner_radius(CornerRadius::same(3))
        .inner_margin(Margin::symmetric(8, 7))
}

fn viewport_frame() -> Frame {
    Frame::new()
        .fill(STUDIO_BG)
        .stroke(Stroke::new(1.0, STUDIO_BORDER))
        .inner_margin(Margin::same(4))
}

fn panel_heading(ui: &mut egui::Ui, icon: char, label: &str) {
    ui.horizontal(|ui| {
        ui.label(icons::text(icon, 15.0).color(STUDIO_ACCENT));
        ui.label(RichText::new(label).strong().color(STUDIO_TEXT));
    });
}

impl EditorWorkspace {
    /// Open the project at `dir`. Errors when `dir/project.ron` is
    /// missing or malformed — the frontend wraps the error and falls
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
            hovered_face: None,
            selected_face: None,
            paint_target_preview: None,
            last_paint_stamp: None,
            renaming: None,
            pending_rename_focus: false,
            history: UndoStack::default(),
            scene_filter: String::new(),
            file_filter: String::new(),
            resource_search: String::new(),
            resource_filter: ResourceFilter::All,
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
            dirty: false,
            status: "Editor ready".to_string(),
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

    /// Directory project-relative resource paths resolve against —
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
    /// rather than failing — the user can still keep editing the
    /// in-memory state.
    pub fn reload(&mut self) {
        let path = self.project_dir.join("project.ron");
        match ProjectDocument::load_from_path(&path) {
            Ok(project) => {
                self.project = project;
                self.selected_node = NodeId::ROOT;
                self.selected_resource = None;
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
    /// inspector empty and gates the paint tools — selecting a
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
    /// up to the caller — the editor doesn't spawn child
    /// processes from this path; instead the status string
    /// hands back the exact command to run.
    pub fn cook_playtest_to_disk(&self) -> Result<String, String> {
        let dir = psxed_project::playtest::default_generated_dir();
        let report = psxed_project::playtest::cook_to_dir(&self.project, &dir)
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
            format!(" ({} warning{})", report.warnings.len(),
                if report.warnings.len() == 1 { "" } else { "s" })
        };
        Ok(format!(
            "Playtest cooked → {}{}.  Run: make run-editor-playtest",
            dir.display(),
            warning_suffix,
        ))
    }

    /// Draw the full editor workspace.
    ///
    /// `viewport_3d_tex` is the host's `egui::TextureId` for the
    /// editor's dedicated `HwRenderer` target — the frontend renders
    /// authored scene data into it once per frame and we paint it as
    /// an Image when the user toggles the central viewport into 3D
    /// mode. The texture stays bit-faithful to what the PS1 would
    /// draw because it's produced by the same `psx-gpu-render`
    /// `HwRenderer` the emulator uses.
    pub fn draw(&mut self, ctx: &egui::Context, viewport_3d_tex: egui::TextureId) {
        apply_studio_visuals(ctx);
        self.handle_global_shortcuts(ctx);
        self.draw_menu_bar(ctx);
        self.draw_action_bar(ctx);
        self.draw_left_dock(ctx);
        self.draw_inspector(ctx);
        self.draw_content_browser(ctx);
        self.draw_viewport(ctx, viewport_3d_tex);
        self.draw_new_project_dialog(ctx);
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

    /// 3D viewport body — paints the HwRenderer texture into the
    /// central area's working space and turns pointer input into
    /// orbit-camera updates. Called from `draw_viewport` when the
    /// user has toggled the 2D / 3D switch on the toolbar to 3D.
    fn draw_viewport_3d_body(&mut self, ui: &mut egui::Ui, viewport_3d_tex: egui::TextureId) {
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
        let (rect, response) = ui.allocate_exact_size(
            egui::Vec2::new(w, h),
            egui::Sense::click_and_drag(),
        );

        // Sims-style: primary button always belongs to the active
        // tool — click-and-drag floors / walls / entities into the
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
        self.hovered_face = face_hit.map(|(face, _)| face);
        // Paint preview: world-cell coords let the ghost outline
        // appear over cells outside the current grid, exactly
        // where the auto-grow would create them.
        self.paint_target_preview = if paint_tool {
            hover_room.and_then(|room| {
                self.compute_paint_target_preview(room, face_hit, hover_world)
            })
        } else {
            None
        };

        // Primary click / drag: ray-pick the cell under the cursor
        // and dispatch to the active tool. Click starts a fresh
        // drag; drag fires every frame the pointer moves; per-cell
        // dedupe keeps walls / placements from stacking when the
        // pointer dwells inside the same cell across frames.
        if response.drag_started_by(egui::PointerButton::Primary)
            || response.clicked_by(egui::PointerButton::Primary)
        {
            self.last_paint_stamp = None;
        }
        let primary_active = response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary);
        if primary_active {
            // Select and paint tools both read the hover ray-test
            // result so the click lands on the actual face under
            // the cursor, not the floor-plane projection (which is
            // wrong for back-row walls and gets the cell-edge
            // direction backwards because of editor / world Z
            // convention drift). Falls back to ground-plane only
            // for the empty-cell paint case where there's no face.
            if matches!(self.active_tool, ViewTool::Select) {
                self.commit_face_selection();
            } else if let Some(pos) = response.interact_pointer_pos() {
                let face_hit = self.pick_face_with_hit(rect, pos);
                let ground = self.pick_3d_world(rect, pos);
                self.dispatch_paint_3d(face_hit, ground);
            }
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

        let uv = egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(640.0 / 2048.0, 480.0 / 1024.0),
        );
        egui::Image::new((viewport_3d_tex, rect.size()))
            .uv(uv)
            .paint_at(ui, rect);
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

    /// Face under the 3D pointer when the Select tool is active —
    /// floors / walls / ceilings of the active Room. Frontend reads
    /// this every frame to draw a light hover outline.
    pub fn hovered_face(&self) -> Option<FaceRef> {
        self.hovered_face
    }

    /// Face the user picked with the Select tool. Frontend draws a
    /// bold outline; the inspector reads it to surface per-face
    /// editable fields (material, heights, …).
    pub fn selected_face(&self) -> Option<FaceRef> {
        self.selected_face
    }

    /// What the next paint click would target. Frontend reads
    /// this every frame for paint tools and outlines either a
    /// cell ghost (Floor / Ceiling / Erase / Place) or a wall
    /// ghost (PaintWall) at the world position the click would
    /// commit to.
    pub fn paint_target_preview(&self) -> Option<PaintTargetPreview> {
        self.paint_target_preview
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
    /// outside the current grid — exactly the cases `auto-grow`
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

        // Cursor over an existing wall while PaintWall is active —
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
            // Cell centre in raw world units — the inferred edge
            // matches the dispatch's `run_paint_action` because
            // both use the same axis convention.
            let s = grid.sector_size as f32;
            let cell_center_x = (world_cell_x as f32 + 0.5) * s;
            let cell_center_z = (world_cell_z as f32 + 0.5) * s;
            let dir =
                edge_from_world_offset(hit_world[0] - cell_center_x, hit_world[2] - cell_center_z);
            // Stack index points just past any existing walls on
            // that edge — `add_wall` will append there.
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

    fn commit_face_selection(&mut self) {
        if let Some(face) = self.hovered_face {
            self.selected_face = Some(face);
            self.selected_node = NodeId::ROOT;
            self.selected_resource = self.face_material(face);
            self.status = format!("Selected {}", describe_face(face));
        } else {
            self.selected_face = None;
            self.selected_resource = None;
            self.status = "Cleared face selection".to_string();
        }
    }

    /// 3D paint / move click handler. `face_hit` is the ray-test
    /// result (`pick_face_with_hit`) — preferred because it
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
                // by stamping a floor in empty space — Sims-style.
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
        match self.active_tool {
            ViewTool::Move => self.snap_selected_to_cell(room_id, sx, sz),
            tool => self.run_paint_action(
                tool,
                room_id,
                sx,
                sz,
                face_hit.map(|(f, _)| f),
                hit_world,
            ),
        }
    }

    /// Build the dedupe key for the next paint dispatch. PaintWall
    /// records the targeted edge + stack so dragging across edges
    /// of the same cell stamps each one (different stamps), but
    /// dwelling on the same edge during drag dedupes (same stamp).
    /// Other tools key on cell + tool only — drag-restamping a
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
                    let dir = edge_from_world_offset(
                        hit_world[0] - center[0],
                        hit_world[2] - center[1],
                    );
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
    fn ensure_cell_in_grid(
        &mut self,
        room_id: NodeId,
        world: [f32; 2],
    ) -> Option<(u16, u16)> {
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
        // room-centre-relative — the 2D viewport's native space).
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
            // Player Spawn is exclusive — demote any existing
            // player spawns so the cooker sees exactly one.
            if matches!(kind, PlaceKind::PlayerSpawn) {
                let scene = self.project.active_scene_mut();
                let stale: Vec<NodeId> = scene
                    .nodes()
                    .iter()
                    .filter(|n| matches!(n.kind, NodeKind::SpawnPoint { player: true }))
                    .map(|n| n.id)
                    .collect();
                for id in stale {
                    if let Some(node) = scene.node_mut(id) {
                        node.kind = NodeKind::SpawnPoint { player: false };
                    }
                }
            }
            let (default_name, node_kind) = match kind {
                PlaceKind::PlayerSpawn => ("Player Spawn", NodeKind::SpawnPoint { player: true }),
                PlaceKind::SpawnMarker => ("Spawn", NodeKind::SpawnPoint { player: false }),
                PlaceKind::EntityMarker => (
                    "Entity",
                    NodeKind::MeshInstance {
                        mesh: None,
                        material: None,
                    },
                ),
                PlaceKind::LightMarker => (
                    "Light",
                    NodeKind::Light {
                        color: [255, 240, 200],
                        intensity: 1.0,
                        radius: 1.0,
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
                            format!("Repainted {} wall #{stack} at {sx},{sz}", direction_label(dir))
                        } else {
                            format!("Wall #{stack} on {} edge of {sx},{sz} is gone", direction_label(dir))
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
            FaceKind::Floor => sector.floor.as_mut().map(|f| f.material = material).is_some(),
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

    /// Snap the selected entity's translation to the centre of cell
    /// `(sx, sz)` inside `room_id`. Editor convention: 1 transform
    /// unit = 1 sector, with the room at its own translation; cell
    /// centres land in editor sector-unit space via the canonical
    /// `WorldGrid::cell_center_world` (origin-aware) post-divide.
    fn snap_selected_to_cell(&mut self, room_id: NodeId, sx: u16, sz: u16) {
        if self.selected_node == NodeId::ROOT {
            return;
        }
        let editor = {
            let scene = self.project.active_scene();
            let Some(room) = scene.node(room_id) else {
                return;
            };
            let NodeKind::Room { grid } = &room.kind else {
                return;
            };
            // World-space centre of the cell, divided into editor
            // sector-units, then re-expressed room-centre-relative.
            let world_centre = grid.cell_center_world(sx, sz);
            let s = grid.sector_size as f32;
            grid.world_cells_to_editor([world_centre[0] / s, world_centre[1] / s])
        };
        let new_x = editor[0];
        let new_z = editor[1];
        let mut moved = None;
        if let Some(node) = self.project.active_scene_mut().node_mut(self.selected_node) {
            // Don't move Rooms / Worlds via cell-snap — only entities.
            if matches!(node.kind, NodeKind::Room { .. } | NodeKind::World) {
                return;
            }
            node.transform.translation[0] = new_x;
            node.transform.translation[2] = new_z;
            moved = Some(node.name.clone());
        }
        if let Some(name) = moved {
            self.status = format!("Moved {name} to {sx},{sz}");
            self.mark_dirty();
        }
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

        let yaw = self.viewport_3d_yaw;
        let pitch = self.viewport_3d_pitch;
        let radius = self.viewport_3d_radius as f32;
        let cos_p = psx_gte::transform::cos_1_3_12(pitch) as f32 / 4096.0;
        let sin_p = psx_gte::transform::sin_1_3_12(pitch) as f32 / 4096.0;
        let cos_y = psx_gte::transform::cos_1_3_12(yaw) as f32 / 4096.0;
        let sin_y = psx_gte::transform::sin_1_3_12(yaw) as f32 / 4096.0;
        // Geometric centre in world coords. Editor `(0, 0)` is the
        // room centre by definition, so `editor_to_room_local`
        // delivers the right point — origin-aware via the canonical
        // helper, no ad-hoc multiplication at this call site.
        let target_world = grid.editor_to_room_local([0.0, 0.0]);
        let cam_pos = [
            target_world[0] + radius * cos_p * sin_y,
            target_world[1] - radius * sin_p,
            target_world[2] + radius * cos_p * cos_y,
        ];
        let forward = normalize3([
            target_world[0] - cam_pos[0],
            target_world[1] - cam_pos[1],
            target_world[2] - cam_pos[2],
        ]);
        let right = normalize3(cross3(forward, [0.0, 1.0, 0.0]));
        let up = cross3(right, forward);

        let nx = (pointer.x - rect.center().x) / (rect.width() * 0.5);
        let ny = (pointer.y - rect.center().y) / (rect.height() * 0.5);
        // PSX projection plane H = 320, framebuffer = 320×240. The
        // ray's right/up offsets are tan(half-FOV) at panel-edge:
        // 160/320 = 0.5 horizontally, 120/320 = 0.375 vertically.
        // Hardcoding 0.5 for both (the previous shape) over-shoots
        // the vertical ray angle by exactly the 4:3 aspect ratio,
        // which is why floor picking worked but walls — whose Y
        // range is narrow — never got hit.
        let half_fov_x: f32 = 0.5;
        let half_fov_y: f32 = 0.5 * 240.0 / 320.0;
        let dir = normalize3([
            forward[0] + right[0] * (nx * half_fov_x) + up[0] * (-ny * half_fov_y),
            forward[1] + right[1] * (nx * half_fov_x) + up[1] * (-ny * half_fov_y),
            forward[2] + right[2] * (nx * half_fov_x) + up[2] * (-ny * half_fov_y),
        ]);
        Some((cam_pos, dir))
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
        let s = grid.sector_size as f32;

        let mut best: Option<(FaceRef, f32)> = None;
        let mut consider = |face: FaceRef, t: f32| {
            if !t.is_finite() || t <= 0.0 {
                return;
            }
            if best.map_or(true, |(_, bt)| t < bt) {
                best = Some((face, t));
            }
        };

        for sx in 0..grid.width {
            for sz in 0..grid.depth {
                let Some(sector) = grid.sector(sx, sz) else {
                    continue;
                };
                let x0 = grid.cell_world_x(sx) as f32;
                let x1 = x0 + s;
                let z0 = grid.cell_world_z(sz) as f32;
                let z1 = z0 + s;

                if let Some(floor) = &sector.floor {
                    let h = floor.heights;
                    let nw = [x0, h[0] as f32, z1];
                    let ne = [x1, h[1] as f32, z1];
                    let se = [x1, h[2] as f32, z0];
                    let sw = [x0, h[3] as f32, z0];
                    let face = FaceRef {
                        room: room_id,
                        sx,
                        sz,
                        kind: FaceKind::Floor,
                    };
                    if let Some(t) = ray_triangle(origin, dir, nw, sw, ne) {
                        consider(face, t);
                    }
                    if let Some(t) = ray_triangle(origin, dir, ne, sw, se) {
                        consider(face, t);
                    }
                }
                if let Some(ceiling) = &sector.ceiling {
                    let h = ceiling.heights;
                    let nw = [x0, h[0] as f32, z1];
                    let ne = [x1, h[1] as f32, z1];
                    let se = [x1, h[2] as f32, z0];
                    let sw = [x0, h[3] as f32, z0];
                    let face = FaceRef {
                        room: room_id,
                        sx,
                        sz,
                        kind: FaceKind::Ceiling,
                    };
                    if let Some(t) = ray_triangle(origin, dir, nw, ne, sw) {
                        consider(face, t);
                    }
                    if let Some(t) = ray_triangle(origin, dir, ne, se, sw) {
                        consider(face, t);
                    }
                }
                for dir_card in [
                    GridDirection::North,
                    GridDirection::East,
                    GridDirection::South,
                    GridDirection::West,
                ] {
                    for (stack_idx, wall) in sector.walls.get(dir_card).iter().enumerate() {
                        let (bl_xy, br_xy) = wall_xy_endpoints(dir_card, x0, x1, z0, z1);
                        let h = wall.heights;
                        let bl = [bl_xy.0, h[0] as f32, bl_xy.1];
                        let br = [br_xy.0, h[1] as f32, br_xy.1];
                        let tr = [br_xy.0, h[2] as f32, br_xy.1];
                        let tl = [bl_xy.0, h[3] as f32, bl_xy.1];
                        let face = FaceRef {
                            room: room_id,
                            sx,
                            sz,
                            kind: FaceKind::Wall {
                                dir: dir_card,
                                stack: stack_idx as u8,
                            },
                        };
                        if let Some(t) = ray_triangle(origin, dir, bl, br, tr) {
                            consider(face, t);
                        }
                        if let Some(t) = ray_triangle(origin, dir, bl, tr, tl) {
                            consider(face, t);
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
        let scene = self.project.active_scene();
        let room = scene
            .nodes()
            .iter()
            .find(|node| matches!(node.kind, NodeKind::Room { .. }))?;
        let NodeKind::Room { grid } = &room.kind else {
            return None;
        };

        // Camera basis from the orbit state. Mirrors the math in
        // `editor_preview::setup_gte_for_camera` so picking and
        // rendering agree on every axis convention.
        let yaw = self.viewport_3d_yaw;
        let pitch = self.viewport_3d_pitch;
        let radius = self.viewport_3d_radius as f32;
        let cos_p = psx_gte::transform::cos_1_3_12(pitch) as f32 / 4096.0;
        let sin_p = psx_gte::transform::sin_1_3_12(pitch) as f32 / 4096.0;
        let cos_y = psx_gte::transform::cos_1_3_12(yaw) as f32 / 4096.0;
        let sin_y = psx_gte::transform::sin_1_3_12(yaw) as f32 / 4096.0;

        // Geometric centre in world coords. Editor `(0, 0)` is the
        // room centre by definition, so `editor_to_room_local`
        // delivers the right point — origin-aware via the canonical
        // helper, no ad-hoc multiplication at this call site.
        let target_world = grid.editor_to_room_local([0.0, 0.0]);
        let cam_pos = [
            target_world[0] + radius * cos_p * sin_y,
            target_world[1] - radius * sin_p,
            target_world[2] + radius * cos_p * cos_y,
        ];

        // World-space camera basis: forward = target - camera, with
        // +Y up. `right` is forward × up, `up` recomputed from those
        // two so the basis is orthonormal in floating point.
        let forward = normalize3([
            target_world[0] - cam_pos[0],
            target_world[1] - cam_pos[1],
            target_world[2] - cam_pos[2],
        ]);
        let right = normalize3(cross3(forward, [0.0, 1.0, 0.0]));
        let up = cross3(right, forward);

        // Map the pointer to PSX 320×240 NDC, clamped to the panel
        // rect since clicks beyond it shouldn't pick.
        if !rect.contains(pointer) {
            return None;
        }
        // Pixel offset from panel centre, normalised so ±1 covers
        // the half-FOV. Focal=320, half-screen=160 → tan(half_fov)
        // = 0.5; we bake that into the multiplier below.
        let nx = (pointer.x - rect.center().x) / (rect.width() * 0.5);
        let ny = (pointer.y - rect.center().y) / (rect.height() * 0.5);
        // Same FOV constants `camera_ray_for_pointer` uses; see the
        // comment there. Hardcoded for the PSX framebuffer's
        // 320×240 over PROJ_H=320.
        let half_fov_x: f32 = 0.5;
        let half_fov_y: f32 = 0.5 * 240.0 / 320.0;
        let dir = normalize3([
            forward[0] + right[0] * (nx * half_fov_x) + up[0] * (-ny * half_fov_y),
            forward[1] + right[1] * (nx * half_fov_x) + up[1] * (-ny * half_fov_y),
            forward[2] + right[2] * (nx * half_fov_x) + up[2] * (-ny * half_fov_y),
        ]);

        // Ray-plane intersection at world Y=0. The orbit cam never
        // sits exactly on Y=0, but we still guard against a near-
        // zero divisor for numerical safety.
        if dir[1].abs() < 1e-5 {
            return None;
        }
        let t = -cam_pos[1] / dir[1];
        if t < 0.0 {
            return None;
        }
        let hit_world = [cam_pos[0] + dir[0] * t, cam_pos[2] + dir[2] * t];

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

        // F2 / Delete only fire when no widget owns focus — so they
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
            if del && self.selected_node != NodeId::ROOT && self.renaming.is_none() {
                self.apply_tree_action(TreeAction::Delete(self.selected_node));
            }
            let rot = ctx.input_mut(|i| i.key_pressed(egui::Key::R));
            if rot && self.renaming.is_none() {
                self.rotate_selected_yaw_90();
            }
        }
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
                self.selected_node = id;
                self.selected_resource = None;
                self.renaming = None;
                // No-op when `id` isn't a Room — keeps the camera
                // put while the user clicks through entity nodes.
                self.frame_3d_on_room(id);
            }
            TreeAction::BeginRename(id) => {
                if let Some(node) = self.project.active_scene().node(id) {
                    self.renaming = Some((id, node.name.clone()));
                    self.pending_rename_focus = true;
                    self.selected_node = id;
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

    fn draw_action_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("psxed_action_bar")
            .exact_height(43.0)
            .frame(top_bar_frame())
            .show(ctx, |ui| {
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
                    if ui
                        .button(icons::label(icons::PLAY, "Cook & Play"))
                        .on_hover_text(
                            "Cook the active scene into editor-playtest's generated dir. \
                             Run `make run-editor-playtest` afterwards to launch.",
                        )
                        .clicked()
                    {
                        match self.cook_playtest_to_disk() {
                            Ok(report) => self.status = report,
                            Err(error) => self.status = format!("Cook & Play failed: {error}"),
                        }
                    }

                    ui.separator();
                    let project_label = if self.dirty {
                        format!("{} *", self.project.name)
                    } else {
                        self.project.name.clone()
                    };
                    ui.label(project_label);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(&self.status).color(STUDIO_TEXT_WEAK));
                    });
                });
            });
    }

    fn draw_left_dock(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("psxed_left_dock")
            .resizable(true)
            .default_width(280.0)
            .min_width(220.0)
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

                // Selection priority: face (Select tool product) →
                // resource (clicked in the bottom panel) → node
                // (scene tree row). Faces win because they're the
                // active edit target during paint workflows.
                if let Some(face) = self.selected_face {
                    self.draw_face_inspector(ui, face);
                    return;
                }

                if let Some(resource_id) = self.selected_resource {
                    self.draw_resource_inspector(ui, resource_id);
                    return;
                }

                let material_options = self.project.material_options();
                let room_options = collect_room_options(&self.project);
                let selected = self.selected_node;
                let active_room = self.active_room_id();
                let selected_sector = self.selected_sector;

                let mut changed = false;

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
                            );
                        });
                }

                // Phase 2: per-sector inspector. Owns its own borrow of the
                // project so it can edit the active Room's grid.
                if let Some(room_id) = active_room {
                    // Room budget panel — same data the cooker
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
            });
    }

    /// Inspector panel for the face currently selected by the
    /// Select tool. Surfaces material picker, height fields, and a
    /// preview thumbnail of the linked texture so the user can
    /// retarget materials without opening the resources panel.
    fn draw_face_inspector(&mut self, ui: &mut egui::Ui, face: FaceRef) {
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

        ui.horizontal(|ui| {
            draw_inline_icon(ui, icons::GRID, STUDIO_ACCENT);
            ui.strong(describe_face(face));
        });
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
                        );
                    });
                egui::CollapsingHeader::new(icons::label(icons::MOVE, "Span"))
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("    Bottom");
                            let mut bot = wall.heights[0];
                            if ui.add(egui::DragValue::new(&mut bot).speed(8.0)).changed() {
                                wall.heights[0] = bot;
                                wall.heights[1] = bot;
                                changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("    Top");
                            let mut top = wall.heights[2];
                            if ui.add(egui::DragValue::new(&mut top).speed(8.0)).changed() {
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
    }

    fn draw_resource_inspector(&mut self, ui: &mut egui::Ui, id: ResourceId) {
        // Pull the cached preview before borrowing `self.project`
        // mutably below. `texture_thumb_entry` takes `&self` and
        // walks Texture / Material → cached `.psxt` decode, so this
        // copy is the only way to keep both alive in one inspector.
        let preview_thumb = self
            .project
            .resource(id)
            .and_then(|resource| self.texture_thumb_entry(resource))
            .map(|entry| (entry.handle.id(), entry.stats));
        let texture_options: Vec<(ResourceId, String)> = self
            .project
            .resources
            .iter()
            .filter_map(|r| match &r.data {
                ResourceData::Texture { .. } => Some((r.id, r.name.clone())),
                _ => None,
            })
            .collect();

        let Some(resource) = self.project.resource_mut(id) else {
            ui.weak("Resource missing");
            return;
        };

        let mut changed = false;
        ui.horizontal(|ui| {
            draw_inline_icon(
                ui,
                resource_lucide_icon(&resource.data),
                resource_lucide_color(&resource.data, true),
            );
            ui.strong(format!("{} #{}", resource.data.label(), resource.id.raw()));
        });
        ui.horizontal(|ui| {
            ui.label("Name");
            changed |= ui.text_edit_singleline(&mut resource.name).changed();
        });
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
                            RichText::new(
                                "Cooked .psxt blob; same artifact the runtime embeds.",
                            )
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
                        );
                        changed |= blend_mode_editor(ui, &mut material.blend_mode);
                        changed |= color_editor(ui, "Tint", &mut material.tint);
                        changed |= ui
                            .checkbox(&mut material.double_sided, "Double sided")
                            .changed();
                    });
                if let Some((_, stats)) = preview_thumb {
                    egui::CollapsingHeader::new(icons::label(icons::SCAN, "Linked Texture"))
                        .default_open(false)
                        .show(ui, |ui| {
                            draw_psxt_stats(ui, stats);
                        });
                }
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
    }

    fn draw_content_browser(&mut self, ctx: &egui::Context) {
        // Refresh PSXT thumbnail handles up-front so the resource
        // cards rendered below have something to blit instead of the
        // name-keyword procedural fallback. Cheap when nothing's
        // changed — the signature cache short-circuits per-resource.
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
                        .on_hover_text(
                            "Add a Texture resource pointing at a cooked .psxt blob.",
                        )
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
            let ResourceData::Texture { psxt_path } = &resource.data else {
                continue;
            };
            alive.push(resource.id);
            if let Some(entry) = self.texture_thumbs.get(&resource.id) {
                if entry.signature == *psxt_path {
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
                    signature: psxt_path.clone(),
                    handle,
                    stats,
                },
            );
        }
        // Drop entries for Texture resources that no longer exist —
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
                    if let (true, Some(face)) = (is_material, self.selected_face) {
                        if self.assign_face_material(face, Some(id)) {
                            self.status =
                                format!("Assigned material to {}", describe_face(face));
                        }
                        self.selected_resource = Some(id);
                    } else {
                        self.selected_resource = Some(id);
                        self.selected_node = NodeId::ROOT;
                        self.selected_face = None;
                    }
                }
            });
        });
    }

    fn draw_viewport(&mut self, ctx: &egui::Context, viewport_3d_tex: egui::TextureId) {
        egui::CentralPanel::default()
            .frame(viewport_frame())
            .show(ctx, |ui| {
                self.draw_viewport_tabs(ui);
                ui.separator();
                self.draw_viewport_toolbar(ui);
                ui.separator();

                if !self.view_2d {
                    self.draw_viewport_3d_body(ui, viewport_3d_tex);
                    return;
                }

                let size = ui.available_size();
                let size = Vec2::new(size.x.max(320.0), size.y.max(240.0));
                let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());

                if response.dragged_by(egui::PointerButton::Middle)
                    || response.dragged_by(egui::PointerButton::Secondary)
                {
                    self.viewport_pan += ui.input(|input| input.pointer.delta());
                }
                if response.dragged_by(egui::PointerButton::Primary) {
                    self.drag_selected_node(ui.input(|input| input.pointer.delta()));
                }

                if response.hovered() {
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
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 0.0, STUDIO_VIEWPORT);
                if self.show_grid {
                    draw_world_grid(&painter, transform);
                }

                let hits =
                    draw_scene_viewport(&painter, transform, &self.project, self.selected_node);

                if response.clicked_by(egui::PointerButton::Primary) {
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
                for (tool, label, hint) in [
                    (
                        ViewTool::Select,
                        icons::label(icons::POINTER, "Select"),
                        "Click a face (floor / wall / ceiling) in the viewport.",
                    ),
                    (
                        ViewTool::Move,
                        icons::label(icons::MOVE, "Move"),
                        "Drag the selected entity onto another cell.",
                    ),
                ] {
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
                if matches!(self.active_tool, ViewTool::Place) {
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
                let mut zoom_percent =
                    (self.viewport_zoom / DEFAULT_VIEWPORT_ZOOM * 100.0) as u16;
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
    /// only while `active_tool == Place` — otherwise the brush
    /// material picker takes the same slot.
    fn draw_place_kind_picker(&mut self, ui: &mut egui::Ui) {
        ui.label(icons::label(icons::PLUS, "Place"));
        for kind in [
            PlaceKind::PlayerSpawn,
            PlaceKind::SpawnMarker,
            PlaceKind::EntityMarker,
            PlaceKind::LightMarker,
        ] {
            ui.selectable_value(&mut self.place_kind, kind, kind.label());
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
    /// parent chain → fall back to the active scene's first Room.
    /// The fallback keeps paint tools enabled even when the
    /// selection sits outside the scene tree (e.g. a face the user
    /// just picked, which clears `selected_node` to ROOT).
    fn active_room_id(&self) -> Option<NodeId> {
        if let Some(face) = self.selected_face {
            return Some(face.room);
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
    /// resources before serious painting — this fallback at least
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
            ViewTool::Select | ViewTool::Move => {
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
    fn apply_paint(
        &mut self,
        tool: ViewTool,
        room_id: NodeId,
        sx: u16,
        sz: u16,
        world: [f32; 2],
    ) {
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

    fn open_new_project_dialog(&mut self) {
        self.new_project_dialog_open = true;
        self.new_project_name.clear();
        self.new_project_error = None;
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Snapshot the current project before a discrete mutation.
    /// Call once per user action — paint click, place, add/delete
    /// node, etc — so each undo step matches one author intent.
    fn push_undo(&mut self) {
        self.history.record(self.project.clone());
    }

    /// Pop the most recent snapshot back into `project`.
    fn do_undo(&mut self) {
        if let Some(prev) = self.history.undo(self.project.clone()) {
            self.project = prev;
            self.selected_resource = None;
            self.selected_sector = None;
            if self.project.active_scene().node(self.selected_node).is_none() {
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
            self.selected_sector = None;
            if self.project.active_scene().node(self.selected_node).is_none() {
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
            changed |= color_editor(ui, "Ambient", &mut grid.ambient_color);
            changed |= ui
                .checkbox(&mut grid.fog_enabled, icons::label(icons::SCAN, "Fog"))
                .changed();
        }
        NodeKind::MeshInstance { mesh, material } => {
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::BOX, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label(match mesh {
                    Some(id) => format!("Mesh resource #{}", id.raw()),
                    None => "No mesh resource assigned".to_string(),
                });
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(icons::text(icons::BLEND, 12.0).color(STUDIO_TEXT_WEAK));
                ui.label("Material");
            });
            if ui.selectable_label(material.is_none(), "None").clicked() && material.is_some() {
                *material = None;
                changed = true;
            }
            for (id, name) in material_options {
                if ui.selectable_label(*material == Some(*id), name).clicked()
                    && *material != Some(*id)
                {
                    *material = Some(*id);
                    changed = true;
                }
            }
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
                        .text(icons::label(icons::SUN, "Intensity")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(radius, 0.0..=16000.0)
                        .text(icons::label(icons::WAYPOINT, "Radius")),
                )
                .changed();
        }
        NodeKind::SpawnPoint { player } => {
            changed |= ui
                .checkbox(player, icons::label(icons::MAP_PIN, "Player spawn"))
                .changed();
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
                        if ui.selectable_label(target_room.is_none(), "(none)").clicked() {
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
        "World" => icons::HOUSE,
        "Room" => icons::GRID,
        "MeshInstance" => icons::BOX,
        "Light" => icons::SUN,
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
        "World" => Color32::from_rgb(232, 152, 96),
        "Room" => Color32::from_rgb(209, 118, 71),
        "MeshInstance" => Color32::from_rgb(156, 174, 190),
        "Light" => Color32::from_rgb(238, 203, 116),
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

/// Tabular `key — value` rows summarizing a `.psxt`. Mirrors the
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

/// Combo-box picker for a Material's linked texture. `current` is
/// the live `material.texture` field; `options` is every Texture
/// resource in the project. Returns true when the selection moved
/// so the caller can mark the project dirty.
fn material_texture_picker(
    ui: &mut egui::Ui,
    current: &mut Option<ResourceId>,
    options: &[(ResourceId, String)],
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Texture");
        let preview = current
            .and_then(|id| options.iter().find(|(rid, _)| *rid == id).map(|(_, n)| n.as_str()))
            .unwrap_or("(none)");
        egui::ComboBox::from_id_salt("material-texture-picker")
            .selected_text(preview)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "(none)").clicked() {
                    *current = None;
                    changed = true;
                }
                for (id, name) in options {
                    if ui
                        .selectable_label(*current == Some(*id), name)
                        .clicked()
                    {
                        *current = Some(*id);
                        changed = true;
                    }
                }
            });
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
        let (insert_rect, insert_response) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), 4.0),
            Sense::hover(),
        );
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
            let pressed_enter =
                lost_focus && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let pressed_esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
            if pressed_esc {
                actions.push(TreeAction::CancelRename);
            } else if pressed_enter || lost_focus {
                actions.push(TreeAction::CommitRename(row.id, buffer.clone()));
            }
        }
    } else {
        painter.text(
            text_pos,
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(13.0),
            text_color,
        );
    }

    if !in_rename && row.id != NodeId::ROOT {
        painter.text(
            Pos2::new(text_pos.x + 118.0, rect.center().y),
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
fn default_addable_kinds() -> [(&'static str, NodeKind); 8] {
    [
        ("World", NodeKind::World),
        (
            "Room",
            NodeKind::Room {
                grid: WorldGrid::empty(4, 4, 1024),
            },
        ),
        ("Node3D", NodeKind::Node3D),
        (
            "MeshInstance",
            NodeKind::MeshInstance {
                mesh: None,
                material: None,
            },
        ),
        (
            "Light",
            NodeKind::Light {
                color: [255, 240, 200],
                intensity: 1.0,
                radius: 4096.0,
            },
        ),
        ("SpawnPoint", NodeKind::SpawnPoint { player: false }),
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
        NodeKind::World => "Streaming Region",
        NodeKind::Room { .. } => "Sector Grid",
        NodeKind::Light { .. } => "Editor Gizmo",
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
        let label = icons::label(row.icon, &row.name);
        if row.folder {
            ui.label(RichText::new(label).color(STUDIO_TEXT_WEAK));
        } else if ui
            .selectable_label(row.resource == selected_resource, label)
            .clicked()
        {
            clicked = true;
        }
    });
    clicked
}

fn resource_file_name(resource: &Resource) -> String {
    match &resource.data {
        ResourceData::Texture { psxt_path } => {
            cooked_name(&resource.name, psxt_path, "psxt")
        }
        ResourceData::Material(_) => cooked_name(&resource.name, "", "mat"),
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

fn resource_filter_counts(project: &ProjectDocument) -> [(ResourceFilter, usize); 5] {
    let mut texture = 0;
    let mut material = 0;
    let mut mesh = 0;
    let mut room = 0;
    let mut other = 0;
    for resource in &project.resources {
        match &resource.data {
            ResourceData::Texture { .. } => texture += 1,
            ResourceData::Material(_) => material += 1,
            ResourceData::Mesh { .. } => mesh += 1,
            ResourceData::Scene { .. } => room += 1,
            ResourceData::Script { .. } | ResourceData::Audio { .. } => other += 1,
        }
    }
    [
        (ResourceFilter::Texture, texture),
        (ResourceFilter::Material, material),
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
        ResourceData::Mesh { source_path }
        | ResourceData::Scene { source_path }
        | ResourceData::Script { source_path }
        | ResourceData::Audio { source_path } => Some(source_path.as_str()),
        ResourceData::Material(_) => None,
    }
}

fn resource_lucide_icon(data: &ResourceData) -> char {
    match data {
        ResourceData::Texture { .. } => icons::PALETTE,
        ResourceData::Material(_) => icons::BLEND,
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
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
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
    } else if name.contains("glass") {
        draw_checker_preview(painter, preview, resource_preview_color(resource));
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
        // 15bpp / unexpected CLUT count — fall through.
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
            Color32::from_rgb((r5 << 3) | (r5 >> 2), (g5 << 3) | (g5 >> 2), (b5 << 3) | (b5 >> 2))
        })
        .collect();

    let pixel_bytes = texture.pixel_bytes();
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    if clut_entries == 16 {
        // 4bpp: 4 texels per halfword, low nibble first.
        let halfwords_per_row = (width as usize + 3) / 4;
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
        let halfwords_per_row = (width as usize + 1) / 2;
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
    for direction in [
        GridDirection::North,
        GridDirection::East,
        GridDirection::South,
        GridDirection::West,
    ] {
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
                [min_x + 0.5, min_z - wall_thickness * 0.5],
                [0.5, wall_thickness * 0.5],
            ),
            GridDirection::East => (
                [min_x + 1.0 + wall_thickness * 0.5, min_z + 0.5],
                [wall_thickness * 0.5, 0.5],
            ),
            GridDirection::South => (
                [min_x + 0.5, min_z + 1.0 + wall_thickness * 0.5],
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
            transform.world_to_screen([min_x, min_z])
        } else {
            transform.world_to_screen([min_x + 1.0, min_z])
        };
        let b = if nw_to_se {
            transform.world_to_screen([min_x + 1.0, min_z + 1.0])
        } else {
            transform.world_to_screen([min_x, min_z + 1.0])
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

/// One-line human description of a `FaceRef` for status messages
/// and the inspector header. Walls include their cardinal direction
/// + stack index since a single edge can carry several stacked
/// walls (windows / arches).
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
    if dz.abs() > dx.abs() {
        if dz >= 0.0 {
            GridDirection::North
        } else {
            GridDirection::South
        }
    } else if dx >= 0.0 {
        GridDirection::East
    } else {
        GridDirection::West
    }
}

/// World-space (x, z) endpoints of a wall on the given cardinal
/// edge of a sector, packed as `(bottom-left, bottom-right)`. The
/// pairing matches `editor_preview::push_wall_face` so picking and
/// rendering agree on which corner is which.
fn wall_xy_endpoints(
    dir: GridDirection,
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
) -> ((f32, f32), (f32, f32)) {
    match dir {
        GridDirection::North => ((x0, z1), (x1, z1)),
        GridDirection::East => ((x1, z1), (x1, z0)),
        GridDirection::South => ((x1, z0), (x0, z0)),
        GridDirection::West => ((x0, z0), (x0, z1)),
        // Diagonals aren't authored from the toolbar yet — picking
        // them is a follow-up. Pick a degenerate edge so the
        // ray-test never hits.
        _ => ((x0, z0), (x0, z0)),
    }
}

/// Möller–Trumbore ray-triangle intersection. Returns the ray
/// parameter `t` of the front-side hit, or `None` for misses /
/// back-face hits / degenerate triangles. Front-side only because
/// the editor draws every face once per OT slot — picking the back
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
fn collect_room_options(project: &ProjectDocument) -> Vec<(NodeId, String)> {
    project
        .active_scene()
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::Room { .. }))
        .map(|node| (node.id, node.name.clone()))
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
            row(
                ui,
                "Floors",
                format!("{}", budget.floors),
                false,
            );
            row(
                ui,
                "Ceilings",
                format!("{}", budget.ceilings),
                false,
            );
            row(
                ui,
                "Walls",
                format!("{}", budget.walls),
                false,
            );
            row(
                ui,
                "Triangles",
                format!("{} / {}", budget.triangles, MAX_ROOM_TRIANGLES),
                budget.triangles > MAX_ROOM_TRIANGLES,
            );
            row(
                ui,
                ".psxw v1",
                format!("{} / {}", human_bytes(budget.psxw_v1_bytes as u32), human_bytes(MAX_ROOM_BYTES as u32)),
                budget.psxw_v1_bytes > MAX_ROOM_BYTES,
            );
            row(
                ui,
                ".psxw v2 est.",
                format!("{}", human_bytes(budget.future_compact_estimated_bytes as u32)),
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
                changed |= material_picker(ui, "    Material", &mut floor.material, material_options);
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
                changed |= wall_stack_row(label, sector.walls.get_mut(dir), sector_size, material_options, ui);
            }
        });

    changed
}

/// Stack-of-walls editor for a single sector edge (N/E/S/W).
///
/// PSX rooms commonly stack walls to model windows / arches: one
/// wall from `0..window_bottom`, another from `window_top..ceiling`.
/// The data model already allows N walls per edge — this UI surfaces
/// it. Each wall row carries its own `[bottom, top]` and material;
/// `+` adds a new wall on top of the previous one (or `0..ceil` for
/// the first), `×` removes that row.
fn wall_stack_row(
    edge_label: &str,
    walls: &mut Vec<GridVerticalFace>,
    sector_size: i32,
    material_options: &[(ResourceId, String)],
    ui: &mut egui::Ui,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(edge_label);
        if ui.small_button("+").on_hover_text("Add wall stack").clicked() {
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
            if ui.add(egui::DragValue::new(&mut bot).speed(8.0)).changed() {
                wall.heights[0] = bot;
                wall.heights[1] = bot;
                changed = true;
            }
            ui.label("top");
            if ui.add(egui::DragValue::new(&mut top).speed(8.0)).changed() {
                wall.heights[2] = top;
                wall.heights[3] = top;
                changed = true;
            }
            if ui.small_button("×").on_hover_text("Remove wall").clicked() {
                remove_at = Some(i);
            }
        });
        let pick_label = format!("    #{i} mat");
        changed |= material_picker(ui, &pick_label, &mut wall.material, material_options);
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
/// DragValues — laid out NW-NE / SW-SE so the on-screen position
/// matches the world-space corner — once the heights diverge or
/// the user clicks the "Slope" toggle. Returns `true` whenever any
/// corner changed so the caller can mark the project dirty.
fn height_row(label: &str, heights: &mut [i32; 4], ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    let mut sloped = !(heights[0] == heights[1]
        && heights[1] == heights[2]
        && heights[2] == heights[3]);

    ui.horizontal(|ui| {
        ui.label(label);
        if ui
            .toggle_value(&mut sloped, "Slope")
            .on_hover_text("Edit each corner height independently.")
            .changed()
        {
            if !sloped {
                // Collapse back to the NW value so the floor is flat
                // again — predictable, matches how `flat()` builds.
                heights[1] = heights[0];
                heights[2] = heights[0];
                heights[3] = heights[0];
                changed = true;
            }
        }
    });

    if sloped {
        // 2×2 grid: NW NE on top row (z+), SW SE on bottom (z−).
        // The order in `heights` is [NW, NE, SE, SW] — index map:
        //   top row: [0]=NW, [1]=NE
        //   bottom:  [3]=SW, [2]=SE
        egui::Grid::new(format!("{label}-corners"))
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                for &idx in &[0usize, 1] {
                    if ui
                        .add(egui::DragValue::new(&mut heights[idx]).speed(8.0))
                        .changed()
                    {
                        changed = true;
                    }
                }
                ui.end_row();
                for &idx in &[3usize, 2] {
                    if ui
                        .add(egui::DragValue::new(&mut heights[idx]).speed(8.0))
                        .changed()
                    {
                        changed = true;
                    }
                }
                ui.end_row();
            });
    } else {
        ui.horizontal(|ui| {
            // Indent so the field aligns with the per-corner grid above.
            ui.label("    ");
            let mut h = heights[0];
            if ui.add(egui::DragValue::new(&mut h).speed(8.0)).changed() {
                *heights = [h; 4];
                changed = true;
            }
        });
    }

    changed
}

/// Shared single-material picker used by the sector inspector.
fn material_picker(
    ui: &mut egui::Ui,
    label: &str,
    current: &mut Option<ResourceId>,
    options: &[(ResourceId, String)],
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let preview = current
            .and_then(|id| options.iter().find(|(rid, _)| *rid == id).map(|(_, n)| n.as_str()))
            .unwrap_or("(none)");
        egui::ComboBox::from_id_salt(label.to_string())
            .selected_text(preview)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "(none)").clicked() {
                    *current = None;
                    changed = true;
                }
                for (id, name) in options {
                    if ui
                        .selectable_label(*current == Some(*id), name)
                        .clicked()
                    {
                        *current = Some(*id);
                        changed = true;
                    }
                }
            });
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

    #[test]
    fn dragging_selected_node_moves_it_in_xz_space() {
        let mut workspace =
            EditorWorkspace::open_directory(psxed_project::default_project_dir()).unwrap();
        let spawn = workspace
            .project
            .active_scene()
            .nodes()
            .iter()
            .find(|node| node.name == "Player Spawn")
            .unwrap()
            .id;
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
}
