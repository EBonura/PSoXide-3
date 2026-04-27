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
    project_path: Option<PathBuf>,
    selected_node: NodeId,
    selected_resource: Option<ResourceId>,
    /// Highlighted sector cell within the active Room. Tracked so the
    /// inspector can show per-cell properties without inflating the
    /// scene-tree node count with a node per sector.
    selected_sector: Option<(u16, u16)>,
    /// Cell currently under the 3D viewport's pointer. Updated each
    /// frame the panel is hovered; rendered as a translucent overlay
    /// so the user sees exactly what cell paint / place tools will hit.
    hovered_3d_sector: Option<(u16, u16)>,
    /// Wall-edge currently under the pointer when the PaintWall tool
    /// is active — `(sx, sz, dir_index)` where `dir_index` is the
    /// `GridDirection` enum discriminant (0=N, 1=E, 2=S, 3=W). The
    /// frontend overlays a thin strip on this edge so the user sees
    /// which boundary the next click will target. Cleared whenever
    /// the active tool isn't PaintWall or the pointer leaves.
    hovered_3d_edge: Option<(u16, u16, u8)>,
    /// Last cell `apply_paint` has run on during the current drag.
    /// Used to dedupe per-frame paint events so click-drag with
    /// Wall / Place doesn't stack duplicate walls / spawn N entities
    /// in the same cell. Reset to `None` whenever a new primary
    /// click starts.
    last_paint_cell: Option<(NodeId, u16, u16)>,
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

/// One cached `.psxt` thumbnail. `signature` is the path the
/// handle was built from — when the user retypes the path on a
/// Texture resource, the signature mismatches and the cache rebuilds.
struct ThumbnailEntry {
    signature: String,
    handle: egui::TextureHandle,
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
    /// Yaw the selected node around its origin.
    Rotate,
    /// Resize the selected node uniformly.
    Scale,
    /// Paint a floor onto the sector under the cursor (Room context).
    PaintFloor,
    /// Paint a wall on the directed edge under the cursor.
    PaintWall,
    /// Paint a ceiling on the sector under the cursor.
    PaintCeiling,
    /// Clear the painted surface under the cursor.
    Erase,
    /// Drop a child entity node into the sector under the cursor.
    Place,
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
    /// Create a workspace with a starter project.
    pub fn new() -> Self {
        let mut workspace = Self::new_blank();
        workspace.select_first_room();
        workspace
    }

    /// Bare construction without the post-init selection bump. Used
    /// internally by `new` / `open_or_new`; external callers should
    /// stick to `new` so the inspector + paint tools have a Room
    /// selected from the first frame.
    fn new_blank() -> Self {
        Self {
            project: ProjectDocument::starter(),
            project_path: None,
            selected_node: NodeId::ROOT,
            selected_resource: None,
            selected_sector: None,
            hovered_3d_sector: None,
            hovered_3d_edge: None,
            last_paint_cell: None,
            renaming: None,
            pending_rename_focus: false,
            history: UndoStack::default(),
            scene_filter: String::new(),
            file_filter: String::new(),
            resource_search: String::new(),
            resource_filter: ResourceFilter::All,
            active_tool: ViewTool::Select,
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

    /// Create a workspace backed by `project_path`, loading the project
    /// if it already exists or seeding a starter project otherwise.
    pub fn open_or_new(project_path: impl Into<PathBuf>) -> Self {
        let path = project_path.into();
        let mut workspace = Self::new();
        workspace.project_path = Some(path.clone());

        if path.exists() {
            if let Err(error) = workspace.load_from_path(&path) {
                workspace.status = format!("Could not load project: {error}");
                workspace.dirty = false;
            }
        } else {
            workspace.dirty = true;
            workspace.status = format!("New project at {}", short_path(&path));
        }

        workspace
    }

    /// Current project document.
    pub fn project(&self) -> &ProjectDocument {
        &self.project
    }

    /// Directory project-relative paths resolve against. Returns the
    /// project file's parent directory when one is configured, falling
    /// back to the current working directory otherwise — which is
    /// what the in-tree starter relies on so `cargo run -p frontend`
    /// from the repo root finds `assets/textures/*.psxt`.
    pub fn project_root(&self) -> PathBuf {
        self.project_path
            .as_ref()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    /// True when the project has unsaved edits.
    pub const fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Set or replace the path used by [`EditorWorkspace::save`].
    pub fn set_project_path(&mut self, path: impl Into<PathBuf>) {
        self.project_path = Some(path.into());
    }

    /// Save to the currently configured project path.
    pub fn save(&mut self) -> Result<(), String> {
        let path = self
            .project_path
            .clone()
            .ok_or_else(|| "no editor project path configured".to_string())?;
        self.save_to_path(path)
    }

    /// Save only when the project contains unsaved edits.
    pub fn save_if_dirty(&mut self) -> Result<bool, String> {
        if !self.dirty {
            return Ok(false);
        }
        self.save()?;
        Ok(true)
    }

    /// Save to a specific path and make that the current project path.
    pub fn save_to_path(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        self.project
            .save_to_path(path)
            .map_err(|error| error.to_string())?;
        self.project_path = Some(path.to_path_buf());
        self.dirty = false;
        self.status = format!("Saved {}", short_path(path));
        Ok(())
    }

    /// Load a project from disk and make that the current project path.
    pub fn load_from_path(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        self.project = ProjectDocument::load_from_path(path).map_err(|error| error.to_string())?;
        self.project_path = Some(path.to_path_buf());
        self.selected_node = NodeId::ROOT;
        self.selected_resource = None;
        self.dirty = false;
        self.status = format!("Loaded {}", short_path(path));
        self.select_first_room();
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
        let project_path = self
            .project_path
            .clone()
            .ok_or("Save the project first so cooked rooms have a destination")?;
        let cooked_dir = project_path
            .parent()
            .map(|parent| parent.join("cooked"))
            .ok_or("Project path has no parent directory")?;
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
            let delta = response.drag_delta();
            self.viewport_3d_yaw = self
                .viewport_3d_yaw
                .wrapping_add((delta.x * 1.5) as i16 as u16);
            self.viewport_3d_pitch = self
                .viewport_3d_pitch
                .wrapping_add((delta.y * 1.5) as i16 as u16);
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
        self.hovered_3d_sector = hover_world
            .and_then(|world| hover_room.and_then(|room| self.world_to_sector(room, world)));
        self.hovered_3d_edge =
            if matches!(self.active_tool, ViewTool::PaintWall) {
                hover_world.and_then(|world| {
                    hover_room.and_then(|room| self.hovered_wall_edge(room, world))
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
            self.last_paint_cell = None;
        }
        let primary_active = response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary);
        if primary_active {
            if let Some(pos) = response.interact_pointer_pos() {
                if let Some(world) = self.pick_3d_world(rect, pos) {
                    self.dispatch_3d_tool(world);
                }
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

    /// Cell under the 3D pointer last frame, or `None` when the
    /// pointer is off-panel. Frontend overlays a translucent quad on
    /// it so paint tools have a Sims-style "you'll hit this cell"
    /// affordance.
    pub fn hovered_3d_sector(&self) -> Option<(u16, u16)> {
        self.hovered_3d_sector
    }

    /// Wall-edge under the 3D pointer last frame, or `None` when the
    /// PaintWall tool isn't active or the pointer's off-panel.
    /// `(sx, sz, dir_index)` where `dir_index` matches the
    /// `GridDirection` discriminant `(0=N, 1=E, 2=S, 3=W)`.
    pub fn hovered_3d_edge(&self) -> Option<(u16, u16, u8)> {
        self.hovered_3d_edge
    }

    /// Pick the cell + nearest cardinal edge at `world` within `room`.
    /// Mirrors the click-time logic in `apply_paint(PaintWall)` so
    /// the hover preview and the actual placement target agree.
    fn hovered_wall_edge(&self, room: NodeId, world: [f32; 2]) -> Option<(u16, u16, u8)> {
        let (sx, sz) = self.world_to_sector(room, world)?;
        let scene = self.project.active_scene();
        let node = scene.node(room)?;
        let NodeKind::Room { grid } = &node.kind else {
            return None;
        };
        let room_center = node_world(node);
        let half = [grid.width as f32 * 0.5, grid.depth as f32 * 0.5];
        let cell_center = [
            room_center[0] - half[0] + sx as f32 + 0.5,
            room_center[1] - half[1] + sz as f32 + 0.5,
        ];
        let dx = world[0] - cell_center[0];
        let dz = world[1] - cell_center[1];
        let dir: u8 = if dz.abs() > dx.abs() {
            if dz < 0.0 { 0 /* North */ } else { 2 /* South */ }
        } else if dx < 0.0 {
            3 /* West */
        } else {
            1 /* East */
        };
        Some((sx, sz, dir))
    }

    /// 3D-viewport tool dispatch: routes a picked world position to
    /// the right per-tool handler, deduping per cell so click-drag
    /// stays idempotent within a cell.
    fn dispatch_3d_tool(&mut self, world: [f32; 2]) {
        let Some(room_id) = self.active_room_id().or_else(|| {
            self.project
                .active_scene()
                .nodes()
                .iter()
                .find(|n| matches!(n.kind, NodeKind::Room { .. }))
                .map(|n| n.id)
        }) else {
            return;
        };
        let Some((sx, sz)) = self.world_to_sector(room_id, world) else {
            return;
        };
        if self.last_paint_cell == Some((room_id, sx, sz)) {
            return;
        }
        self.last_paint_cell = Some((room_id, sx, sz));
        // Surfacing the clicked cell in the inspector is useful for
        // every tool, not just paint — Select / Move users want to
        // see what cell they're on too.
        self.selected_sector = Some((sx, sz));

        match self.active_tool {
            ViewTool::Move => self.snap_selected_to_cell(room_id, sx, sz),
            // Select picks the placeable child node nearest to the
            // pointer's projected ground-plane hit; falls back to
            // selecting the Room itself if no entity is close.
            ViewTool::Select => self.pick_entity_at(room_id, world),
            // Rotate / Scale aren't Sims-shaped — skip in 3D.
            ViewTool::Rotate | ViewTool::Scale => {}
            // Paint / Place / Erase: call `apply_paint` directly so
            // the 3D path doesn't re-pick through the 2D click
            // handler's hit-test machinery.
            tool => {
                self.apply_paint(tool, room_id, sx, sz, world);
            }
        }
    }

    /// Hit-test placeable child nodes against the picked ground
    /// position; selects the closest within half a sector. Falls
    /// back to selecting the Room when no entity is close enough.
    fn pick_entity_at(&mut self, room_id: NodeId, world: [f32; 2]) {
        let scene = self.project.active_scene();
        let mut best: Option<(NodeId, String, f32)> = None;
        for node in scene.nodes() {
            // Skip the things that can't be 3D-picked: the Room
            // itself, its World parent, the scene root, plain
            // organisational nodes.
            if !matches!(
                node.kind,
                NodeKind::SpawnPoint { .. }
                    | NodeKind::MeshInstance { .. }
                    | NodeKind::Light { .. }
                    | NodeKind::Trigger { .. }
                    | NodeKind::Portal { .. }
                    | NodeKind::AudioSource { .. }
            ) {
                continue;
            }
            let entity_x = node.transform.translation[0];
            let entity_z = node.transform.translation[2];
            // World coords here are in editor "1 unit = 1 sector"
            // space; the room's geometry origin is at (-half_w,
            // -half_d). Entity transforms are stored relative to
            // that anchor too, so the difference is direct.
            let dx = entity_x - world[0];
            let dz = entity_z - world[1];
            let dist = (dx * dx + dz * dz).sqrt();
            if dist > 0.5 {
                continue;
            }
            if let Some((_, _, best_dist)) = best {
                if dist >= best_dist {
                    continue;
                }
            }
            best = Some((node.id, node.name.clone(), dist));
        }
        match best {
            Some((id, name, _)) => {
                self.selected_node = id;
                self.selected_resource = None;
                self.status = format!("Selected {name}");
            }
            None => {
                self.selected_node = room_id;
                self.selected_resource = None;
            }
        }
    }

    /// Snap the selected entity's translation to the centre of cell
    /// `(sx, sz)` inside `room_id`. Editor convention: 1 transform
    /// unit = 1 sector, with the room at its own translation; cell
    /// centres land at integer + 0.5 minus half the grid extents so
    /// the room renders symmetric around its anchor.
    fn snap_selected_to_cell(&mut self, room_id: NodeId, sx: u16, sz: u16) {
        if self.selected_node == NodeId::ROOT {
            return;
        }
        // Read the room's grid dims first (immutable borrow).
        let (half_w, half_d) = {
            let scene = self.project.active_scene();
            let Some(room) = scene.node(room_id) else {
                return;
            };
            let NodeKind::Room { grid } = &room.kind else {
                return;
            };
            (grid.width as f32 * 0.5, grid.depth as f32 * 0.5)
        };
        let new_x = (sx as f32) + 0.5 - half_w;
        let new_z = (sz as f32) + 0.5 - half_d;
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

        let target_world = [
            (grid.width as f32 * grid.sector_size as f32) * 0.5,
            0.0,
            (grid.depth as f32 * grid.sector_size as f32) * 0.5,
        ];
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
        let aspect = rect.width() / rect.height();
        // PSX framebuffer aspect is 320/240 = 4:3; if the panel rect
        // is wider/narrower than that, the painted texture is
        // letterboxed. We still want the click to track what the
        // user sees, so scale the X half-FOV by the actual aspect
        // relative to 4:3.
        let target_aspect = 320.0 / 240.0;
        let half_fov_x = 0.5 * (aspect / target_aspect).max(0.001);
        let half_fov_y = 0.5;
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

        // The 2D click handler thinks in editor "viewport units"
        // where 1 = 1 sector and the origin is the room's *centre*
        // (the room sits at its `transform.translation`; cells lay
        // out symmetrically around it). The 3D geometry is drawn
        // corner-rooted in actual world units, so we both divide by
        // sector_size and subtract the half-extents to translate.
        let unit = grid.sector_size as f32;
        let half_w = grid.width as f32 * 0.5;
        let half_d = grid.depth as f32 * 0.5;
        Some([hit_world[0] / unit - half_w, hit_world[1] / unit - half_d])
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
                    for menu in ["File", "Edit", "View", "Project", "Build", "Help"] {
                        ui.menu_button(menu, |_ui| {});
                    }
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
                    if ui.button(icons::label(icons::FILE_PLUS, "New")).clicked() {
                        self.reset_project();
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
                        self.reload_project();
                    }
                    if ui.button(icons::label(icons::FOCUS, "Frame")).clicked() {
                        self.frame_viewport();
                    }
                    if ui.button(icons::label(icons::PLAY, "Playtest")).clicked() {
                        self.status = "Playtest preview pending scene cook".to_string();
                    }
                    if ui.button(icons::label(icons::BLEND, "Cook World")).clicked() {
                        match self.cook_world_to_disk() {
                            Ok(report) => self.status = report,
                            Err(error) => self.status = format!("Cook failed: {error}"),
                        }
                    }

                    ui.separator();
                    let project_label = if self.dirty {
                        format!("{} *", self.project.name)
                    } else {
                        self.project.name.clone()
                    };
                    ui.label(project_label);
                    ui.separator();

                    if ui
                        .button(icons::label(icons::CIRCLE_DOT, "Add Node"))
                        .clicked()
                    {
                        self.add_child(NodeKind::Node3D, "Node3D");
                    }
                    if ui.button(icons::label(icons::GRID, "Add Room")).clicked() {
                        self.add_grid_world_child();
                    }
                    if ui.button(icons::label(icons::BOX, "Add Mesh")).clicked() {
                        let material = self.project.material_options().first().map(|(id, _)| *id);
                        self.add_child(
                            NodeKind::MeshInstance {
                                mesh: None,
                                material,
                            },
                            "MeshInstance",
                        );
                    }
                    if ui.button(icons::label(icons::SUN, "Add Light")).clicked() {
                        self.add_child(
                            NodeKind::Light {
                                color: [255, 240, 210],
                                intensity: 1.0,
                                radius: 4096.0,
                            },
                            "Light",
                        );
                    }
                    if ui
                        .button(icons::label(icons::MAP_PIN, "Add Spawn"))
                        .clicked()
                    {
                        self.add_child(NodeKind::SpawnPoint { player: true }, "Player Spawn");
                    }

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
                    let project = &self.project;
                    egui::CollapsingHeader::new(icons::label(icons::SCAN, "Budget"))
                        .default_open(false)
                        .show(ui, |ui| {
                            draw_room_budget(ui, project, selected);
                        });
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

    fn draw_resource_inspector(&mut self, ui: &mut egui::Ui, id: ResourceId) {
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
            }
            ResourceData::Material(material) => {
                egui::CollapsingHeader::new(icons::label(icons::BLEND, "Material"))
                    .default_open(true)
                    .show(ui, |ui| {
                        changed |= blend_mode_editor(ui, &mut material.blend_mode);
                        changed |= color_editor(ui, "Tint", &mut material.tint);
                        changed |= ui
                            .checkbox(&mut material.double_sided, "Double sided")
                            .changed();
                        ui.label(match material.texture {
                            Some(texture) => format!("Texture resource #{}", texture.raw()),
                            None => "No texture assigned".to_string(),
                        });
                    });
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
        let project_root = self.project_root();
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
            let Some(image) = std::fs::read(&abs)
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
        let key = match &resource.data {
            ResourceData::Texture { .. } => Some(resource.id),
            ResourceData::Material(mat) => mat.texture,
            _ => None,
        }?;
        self.texture_thumbs.get(&key).map(|e| e.handle.id())
    }

    fn draw_resources_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            section_frame().show(ui, |ui| {
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
                    self.selected_resource = Some(id);
                    self.selected_node = NodeId::ROOT;
                }
            });

            ui.add_space(4.0);

            section_frame().show(ui, |ui| {
                ui.set_width(230.0);
                panel_heading(ui, icons::PALETTE, "Texture Atlas (PS1)");
                draw_atlas_preview(ui, &self.project);
                ui.label(RichText::new("Atlas: default.atlas (256x256)").color(STUDIO_TEXT_WEAK));
                ui.label(RichText::new("Mode: 4bpp Indexed").color(STUDIO_TEXT_WEAK));
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
        ui.horizontal(|ui| {
            let _ = ui.selectable_label(
                true,
                RichText::new(icons::label(icons::GRID, "Stone Room.room")).strong(),
            );
            if ui
                .add(egui::Button::new(icons::text(icons::PLUS, 14.0)))
                .on_hover_text("New scene")
                .clicked()
            {
                self.status = "New scene tabs pending document support".to_string();
            }
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
        ui.horizontal(|ui| {
            for (tool, label, hint) in [
                (
                    ViewTool::Select,
                    icons::label(icons::POINTER, "Select"),
                    "Click an entity in the viewport to select it.",
                ),
                (
                    ViewTool::Move,
                    icons::label(icons::MOVE, "Move"),
                    "Drag the selected entity onto another cell.",
                ),
                (
                    ViewTool::Rotate,
                    icons::label(icons::ROTATE_3D, "Rotate"),
                    "Rotate (2D viewport only for now).",
                ),
                (
                    ViewTool::Scale,
                    icons::label(icons::SCALE_3D, "Scale"),
                    "Scale (2D viewport only for now).",
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
            self.draw_brush_material_picker(ui);
            ui.separator();
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
            ui.selectable_value(&mut self.view_2d, true, icons::label(icons::GRID, "2D"));
            ui.selectable_value(&mut self.view_2d, false, icons::label(icons::BOX, "3D"));
            ui.separator();
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
    }

    /// Toolbar combobox for the active brush material. Selecting
    /// "Auto" leaves `brush_material = None` so paint falls back to
    /// the per-tool name heuristic (`floor → "floor" material,
    /// brick → "brick" material`); picking a specific entry pins
    /// every Floor / Wall / Ceiling stroke to that material.
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
    /// Returns the selected node itself when it's a Room, otherwise
    /// climbs the parent chain. Used to scope paint tools to a Room
    /// context — placing entities or painting only makes sense once
    /// the tool knows which `WorldGrid` it's editing.
    fn active_room_id(&self) -> Option<NodeId> {
        let scene = self.project.active_scene();
        let mut current = self.selected_node;
        loop {
            let node = scene.node(current)?;
            if matches!(node.kind, NodeKind::Room { .. }) {
                return Some(current);
            }
            current = node.parent?;
        }
    }

    /// Translate a viewport-space click into a sector cell on `room`.
    ///
    /// The Room renders its grid centred on its node transform, with
    /// each cell exactly one viewport unit, so the inverse mapping is
    /// trivial: subtract the room corner, floor, and bound-check.
    fn world_to_sector(&self, room_id: NodeId, world: [f32; 2]) -> Option<(u16, u16)> {
        let scene = self.project.active_scene();
        let room = scene.node(room_id)?;
        let NodeKind::Room { grid } = &room.kind else {
            return None;
        };
        let center = node_world(room);
        let half = [grid.width as f32 * 0.5, grid.depth as f32 * 0.5];
        let lx = (world[0] - (center[0] - half[0])).floor();
        let lz = (world[1] - (center[1] - half[1])).floor();
        if lx < 0.0 || lz < 0.0 {
            return None;
        }
        let x = lx as u32;
        let z = lz as u32;
        if x >= grid.width as u32 || z >= grid.depth as u32 {
            return None;
        }
        Some((x as u16, z as u16))
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
            ViewTool::Select | ViewTool::Move | ViewTool::Rotate | ViewTool::Scale => {
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

    /// Apply a paint/erase/place tool to one sector. Marks the project
    /// dirty so save/cook prompts work; status bar reflects the action.
    fn apply_paint(
        &mut self,
        tool: ViewTool,
        room_id: NodeId,
        sx: u16,
        sz: u16,
        world: [f32; 2],
    ) {
        self.push_undo();
        // Toolbar picker overrides the name-based default. The picker
        // lives in the workspace state so the choice survives across
        // tool switches and across cells in a single drag-paint.
        let floor_mat = self
            .brush_material
            .or_else(|| self.default_brush_material("floor"));
        let wall_mat = self
            .brush_material
            .or_else(|| self.default_brush_material("brick"))
            .or(floor_mat);
        let room_center;
        let sector_center;
        {
            let scene = self.project.active_scene();
            let Some(room) = scene.node(room_id) else {
                return;
            };
            let NodeKind::Room { grid } = &room.kind else {
                return;
            };
            room_center = node_world(room);
            let half = [grid.width as f32 * 0.5, grid.depth as f32 * 0.5];
            sector_center = [
                room_center[0] - half[0] + sx as f32 + 0.5,
                room_center[1] - half[1] + sz as f32 + 0.5,
            ];
        }

        if matches!(tool, ViewTool::Place) {
            let id = self.project.active_scene_mut().add_node(
                room_id,
                "Spawn",
                NodeKind::SpawnPoint { player: false },
            );
            if let Some(node) = self.project.active_scene_mut().node_mut(id) {
                node.transform.translation = [
                    world[0] - room_center[0],
                    0.0,
                    world[1] - room_center[1],
                ];
            }
            self.selected_node = id;
            self.status = format!("Placed entity at {sx},{sz}");
            self.mark_dirty();
            return;
        }

        let scene = self.project.active_scene_mut();
        let Some(room) = scene.node_mut(room_id) else {
            return;
        };
        let NodeKind::Room { grid } = &mut room.kind else {
            return;
        };
        let sector_size = grid.sector_size;
        let status = match tool {
            ViewTool::PaintFloor => {
                grid.set_floor(sx, sz, 0, floor_mat);
                format!("Painted floor at {sx},{sz}")
            }
            ViewTool::PaintCeiling => {
                if let Some(sector) = grid.ensure_sector(sx, sz) {
                    sector.ceiling = Some(GridHorizontalFace::flat(sector_size, floor_mat));
                }
                format!("Painted ceiling at {sx},{sz}")
            }
            ViewTool::PaintWall => {
                let dx = world[0] - sector_center[0];
                let dz = world[1] - sector_center[1];
                let dir = if dz.abs() > dx.abs() {
                    if dz < 0.0 {
                        GridDirection::North
                    } else {
                        GridDirection::South
                    }
                } else if dx < 0.0 {
                    GridDirection::West
                } else {
                    GridDirection::East
                };
                grid.add_wall(sx, sz, dir, 0, sector_size, wall_mat);
                format!("Added {dir:?} wall at {sx},{sz}")
            }
            ViewTool::Erase => {
                if let Some(index) = grid.sector_index(sx, sz) {
                    grid.sectors[index] = None;
                }
                format!("Erased sector {sx},{sz}")
            }
            _ => return,
        };
        self.status = status;
        self.mark_dirty();
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

    fn add_grid_world_child(&mut self) {
        let material_options = self.project.material_options();
        let floor = material_options
            .iter()
            .find(|(_, name)| name.to_ascii_lowercase().contains("floor"))
            .or_else(|| material_options.first())
            .map(|(id, _)| *id);
        let wall = material_options
            .iter()
            .find(|(_, name)| name.to_ascii_lowercase().contains("brick"))
            .or_else(|| material_options.first())
            .map(|(id, _)| *id);
        self.add_child(
            NodeKind::Room {
                grid: WorldGrid::stone_room(4, 4, 1024, floor, wall),
            },
            "Room",
        );
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

    fn reset_project(&mut self) {
        self.project = ProjectDocument::starter();
        self.selected_node = NodeId::ROOT;
        self.selected_resource = None;
        self.status = "New starter project".to_string();
        self.frame_viewport();
        self.mark_dirty();
        self.select_first_room();
    }

    fn reload_project(&mut self) {
        let Some(path) = self.project_path.clone() else {
            self.status = "Reload unavailable: no project path".to_string();
            return;
        };
        if let Err(error) = self.load_from_path(path) {
            self.status = format!("Reload failed: {error}");
        }
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

impl Default for EditorWorkspace {
    fn default() -> Self {
        Self::new()
    }
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
fn decode_psxt_thumbnail(bytes: &[u8]) -> Option<ColorImage> {
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
    Some(ColorImage {
        size: [width as usize, height as usize],
        pixels,
    })
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

fn draw_atlas_preview(ui: &mut egui::Ui, project: &ProjectDocument) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(210.0, 86.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, Color32::from_rgb(12, 16, 20));
    let colors: Vec<_> = project
        .resources
        .iter()
        .filter(|resource| {
            matches!(
                resource.data,
                ResourceData::Texture { .. } | ResourceData::Material(_)
            )
        })
        .map(resource_preview_color)
        .collect();
    let cell = 16.0;
    for y in 0..5 {
        for x in 0..13 {
            let idx = (x + y * 3) % colors.len().max(1);
            let min = rect.min + Vec2::new(x as f32 * cell, y as f32 * cell);
            painter.rect_filled(
                Rect::from_min_size(min, Vec2::splat(cell - 1.0)),
                0.0,
                colors
                    .get(idx)
                    .copied()
                    .unwrap_or_else(|| Color32::from_rgb(64, 72, 84)),
            );
        }
    }
    painter.rect_stroke(
        rect,
        2.0,
        Stroke::new(1.0, Color32::from_rgb(58, 70, 86)),
        StrokeKind::Inside,
    );
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

/// Streaming-budget readout for one Room.
///
/// Cooks the room's grid in-memory and reports populated sector count,
/// distinct material count, and the cooked `.psxw` byte total — the
/// number that determines whether the streamer can fit the room in
/// its residency budget.
fn draw_room_budget(ui: &mut egui::Ui, project: &ProjectDocument, room_id: NodeId) {
    let Some(room) = project.active_scene().node(room_id) else {
        return;
    };
    let NodeKind::Room { grid } = &room.kind else {
        return;
    };

    let total_cells = grid.width as usize * grid.depth as usize;
    let populated = grid.populated_sector_count();
    ui.horizontal(|ui| {
        ui.label("Populated sectors");
        ui.monospace(format!("{populated} / {total_cells}"));
    });

    match psxed_project::world_cook::cook_world_grid(project, grid) {
        Ok(cooked) => {
            ui.horizontal(|ui| {
                ui.label("Materials");
                ui.monospace(cooked.materials.len().to_string());
            });
            match cooked.to_psxw_bytes() {
                Ok(bytes) => {
                    let kib = bytes.len() as f32 / 1024.0;
                    ui.horizontal(|ui| {
                        ui.label("Cooked .psxw");
                        ui.monospace(format!("{} B  ({kib:.2} KiB)", bytes.len()));
                    });
                }
                Err(error) => {
                    ui.colored_label(
                        Color32::from_rgb(220, 90, 90),
                        format!("encode failed: {error}"),
                    );
                }
            }
        }
        Err(error) => {
            ui.colored_label(
                Color32::from_rgb(220, 90, 90),
                format!("cook failed: {error}"),
            );
        }
    }
}

/// Per-cell inspector for one sector inside the active Room.
///
/// Renders a CollapsingHeader with floor/ceiling toggles, a single
/// flat height per face (corner authoring lands later), a material
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
    fn open_or_new_saves_and_reloads_default_project() {
        let dir = std::env::temp_dir().join(format!(
            "psxed-ui-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("workspace.ron");

        let mut workspace = EditorWorkspace::open_or_new(&path);
        assert!(workspace.is_dirty());
        workspace.save().unwrap();
        assert!(!workspace.is_dirty());
        assert!(path.is_file());

        let loaded = EditorWorkspace::open_or_new(&path);
        assert!(!loaded.is_dirty());
        assert_eq!(
            loaded.project().resources.len(),
            workspace.project().resources.len()
        );

        let _ = std::fs::remove_dir_all(dir);
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
        let mut workspace = EditorWorkspace::new();
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
