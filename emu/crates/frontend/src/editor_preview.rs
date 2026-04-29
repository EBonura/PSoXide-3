//! Editor 3D viewport — Phase 1 sector renderer.
//!
//! Walks the editor's active Room and feeds the editor-owned
//! [`HwRenderer`](psx_gpu_render::HwRenderer) the same way runtime
//! PS1 code does:
//!
//! 1. Configure the GTE for an orbit camera (RT / TR / OFX / OFY / H).
//! 2. For every populated sector with a floor, project the four
//!    corners through the host GTE shim ([`psx_gte::scene::project_vertex`]).
//! 3. Emit two `TriFlat` packets per floor, coloured from the
//!    sector's material base colour.
//! 4. Insert each packet into an `OrderingTable` keyed on average
//!    depth.
//! 5. Walk the OT via `iter_packets`, build a `GpuCmdLogEntry` log,
//!    hand it to `psx-gpu-render::HwRenderer::render_frame`.
//!
//! That's the same path PS1 software follows (minus DMA) — so the
//! editor preview is bit-identical to the renderer the emulator runs
//! against the actual cooked `.psxw`.
//!
//! Walls / ceilings / textures land in later phases. This phase is
//! the smallest end-to-end loop that proves authored Room data
//! drives the editor renderer.

use std::sync::Mutex;

use emulator_core::gpu::GpuCmdLogEntry;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::TriFlat;
use psx_gpu::prim::TriTextured;
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene as gte_scene;

use psxed_project::{
    GridDirection, GridSplit, NodeKind, ProjectDocument, ResourceData, ResourceId, WorldGrid,
};

use crate::editor_textures::{EditorTextures, MaterialSlot};
use psxed_ui::ViewportCameraState;

/// Maximum sectors we'll attempt to render in one preview pass.
/// 64×64 grid would already be enormous for PSX (~16 MiB cooked); a
/// 4096-cap caps the per-frame primitive count at a comfortable
/// number for the host renderer.
const TRI_CAP: usize = 4096;
/// Ordering-table depth — tradeoff between Z resolution and the
/// per-frame chain-walk cost. 256 slots is plenty for an orbit-camera
/// view where the front-to-back range is a small multiple of the
/// sector size.
const OT_DEPTH: usize = 256;

/// Default screen geometry — matches the PSX 320×240 framebuffer the
/// editor's HwRenderer is sized to display.
const SCREEN_W: i32 = 320;
const SCREEN_H: i32 = 240;
const SCREEN_CX: i32 = SCREEN_W / 2;
const SCREEN_CY: i32 = SCREEN_H / 2;
/// Projection-plane distance (focal length). Bigger = narrower FOV.
const PROJ_H: i32 = 320;

/// Per-frame scratch — primitives **and** OT must live in the same
/// memory region. `OrderingTable` stores 24-bit chain pointers (the
/// PS1 DMA encoding); `iter_packets` reconstructs full addresses by
/// OR-ing the OT slot's high 40 bits over the 24-bit chain entries.
/// That only works if every chained primitive sits in the same 16 MB
/// window as the OT itself — heap-allocated `Vec<TriFlat>` lives in
/// a totally separate region on host and segfaults on dereference.
/// Keeping the array inline alongside the OT in the static fixes
/// that and matches PS1's flat 2 MB main RAM layout.
struct PreviewScratch {
    ot: OrderingTable<OT_DEPTH>,
    tris: [TriFlat; TRI_CAP],
    tex_tris: [TriTextured; TRI_CAP],
    /// `0` = next free slot in `tris` (flat-shaded);
    /// `tex_used` = next free slot in `tex_tris`.
    used: usize,
    tex_used: usize,
    /// GP0(02h) fill-rectangle packet: 1 tag word + 3 data words
    /// (`opcode|color`, `pack_xy(x, y)`, `pack_xy(w, h)`). Must live
    /// in the same static as the OT for the same reason the prim
    /// arrays do — `iter_packets` reconstructs full pointers from
    /// the OT's 24-bit chain encoding plus the OT struct's high
    /// address bits, so chained packets must share that 16 MB
    /// region.
    clear_packet: [u32; 4],
}

const EMPTY_TRI: TriFlat = TriFlat::new([(0, 0), (0, 0), (0, 0)], 0, 0, 0);
const EMPTY_TEX_TRI: TriTextured = TriTextured::new(
    [(0, 0), (0, 0), (0, 0)],
    [(0, 0), (0, 0), (0, 0)],
    0,
    0,
    (0x80, 0x80, 0x80),
);

static SCRATCH: Mutex<PreviewScratch> = Mutex::new(PreviewScratch {
    ot: OrderingTable::new(),
    tris: [EMPTY_TRI; TRI_CAP],
    tex_tris: [EMPTY_TEX_TRI; TRI_CAP],
    used: 0,
    tex_used: 0,
    clear_packet: [0; 4],
});

/// Build a fresh `cmd_log` rendering the project's first Room from
/// `camera`'s orbit angles.
///
/// Returns an empty log if the project has no Rooms — the editor
/// renderer will then paint a black panel, which is the correct "no
/// scene to show" affordance.
#[allow(clippy::too_many_arguments)]
pub fn build_phase1_cmd_log(
    project: &ProjectDocument,
    camera: ViewportCameraState,
    selected: psxed_project::NodeId,
    hovered_primitive: Option<psxed_ui::Selection>,
    selected_primitive: Option<psxed_ui::Selection>,
    paint_target_preview: Option<psxed_ui::PaintTargetPreview>,
    entity_bounds: &[psxed_ui::EntityBounds],
    hovered_entity_node: Option<psxed_project::NodeId>,
    textures: &EditorTextures,
    assets: &crate::editor_assets::EditorAssets,
) -> Vec<GpuCmdLogEntry> {
    let Some((room_id, grid, target)) = first_room_grid(project) else {
        return Vec::new();
    };

    let mut scratch = SCRATCH.lock().expect("editor preview scratch mutex");
    scratch.used = 0;
    scratch.tex_used = 0;
    scratch.ot.clear();

    push_clear(&mut scratch);
    let world_camera = setup_gte_for_camera(camera, target);
    walk_room(project, room_id, grid, textures, &mut scratch);
    walk_entities(project, grid, selected, &mut scratch);
    walk_light_gizmos(project, grid, selected, &mut scratch);

    // Selection / hover / paint overlays drawn before models —
    // they project through the camera GTE matrix that
    // `setup_gte_for_camera` installed. Models render after,
    // overwriting per-joint GTE state. We re-install the
    // camera state below before drawing entity bounds so they
    // pick up the same camera basis instead of the last
    // model joint matrix.
    if let Some(selection) = selected_primitive {
        push_selection_outline(grid, selection, OutlineRole::Selected, &mut scratch);
    }
    if let Some(selection) = hovered_primitive {
        if Some(selection) != selected_primitive {
            push_selection_outline(grid, selection, OutlineRole::Hover, &mut scratch);
        }
    }
    if let Some(preview) = paint_target_preview {
        push_paint_preview(grid, preview, &mut scratch);
    }

    walk_model_instances(
        project,
        grid,
        textures,
        assets,
        selected,
        &world_camera,
        &mut scratch,
    );

    // Re-prime the GTE with the camera matrix — model
    // rendering left it set to the last joint's view, which
    // would project entity bound lines into junk.
    let _ = setup_gte_for_camera(camera, target);
    walk_entity_bounds(entity_bounds, selected, hovered_entity_node, &mut scratch);

    // SAFETY: `scratch.tris` lives until end of this function (the
    // mutex guard keeps it alive); the OT chain pointers reference
    // packets inside that vec and are stable while the lock is held.
    unsafe { psx_gpu_render::build_cmd_log(&scratch.ot) }
}

/// First Room node in the active scene plus the world-space point we
/// want the orbit camera to look at (centre of the grid at floor
/// height). `None` if no Room exists.
fn first_room_grid(
    project: &ProjectDocument,
) -> Option<(psxed_project::NodeId, &WorldGrid, [i32; 3])> {
    let scene = project.active_scene();
    let room = scene
        .nodes()
        .iter()
        .find(|node| matches!(node.kind, NodeKind::Room { .. }))?;
    let NodeKind::Room { grid } = &room.kind else {
        return None;
    };
    // Geometric centre in world coords accounts for `origin` so the
    // camera reframes naturally as the grid grows in any direction.
    let center_x = grid.origin[0] * grid.sector_size + (grid.width as i32 * grid.sector_size) / 2;
    let center_z = grid.origin[1] * grid.sector_size + (grid.depth as i32 * grid.sector_size) / 2;
    Some((room.id, grid, [center_x, 0, center_z]))
}

/// Configure the host-side GTE so subsequent `project_vertex` /
/// `project_triangle` calls produce screen-space coords for the
/// requested orbit camera.
fn setup_gte_for_camera(camera: ViewportCameraState, target: [i32; 3]) -> psx_engine::WorldCamera {
    // Orbit camera world position: target offset by radius along the
    // unit vector implied by yaw + pitch. Pitch positive = camera
    // raised above the target, looking down.
    let r = camera.radius;
    let cos_p = psx_gte::transform::cos_1_3_12(camera.pitch_q12) as i32;
    let sin_p = psx_gte::transform::sin_1_3_12(camera.pitch_q12) as i32;
    let cos_y = psx_gte::transform::cos_1_3_12(camera.yaw_q12) as i32;
    let sin_y = psx_gte::transform::sin_1_3_12(camera.yaw_q12) as i32;
    let horiz = (r * cos_p) >> 12;
    let cam_x = target[0] + ((horiz * sin_y) >> 12);
    let cam_y = target[1] - ((r * sin_p) >> 12);
    let cam_z = target[2] + ((horiz * cos_y) >> 12);

    // View rotation: world →camera. Built so that:
    //   row0 = right (= +X in camera space)
    //   row1 = -up   (PSX screen Y points down, so we flip)
    //   row2 = forward (= +Z in camera space; camera looks toward target)
    // Matches `psx_engine::render3d::camera_gte_view_matrix`.
    let view = Mat3I16 {
        m: [
            [clamp_i16(cos_y), 0, clamp_i16(-sin_y)],
            [
                clamp_i16(-((sin_y * sin_p) >> 12)),
                clamp_i16(-cos_p),
                clamp_i16(-((cos_y * sin_p) >> 12)),
            ],
            [
                clamp_i16(-((sin_y * cos_p) >> 12)),
                clamp_i16(sin_p),
                clamp_i16(-((cos_y * cos_p) >> 12)),
            ],
        ],
    };

    // Vertex emit will subtract `target` from each world coord
    // (see `world_to_view`), so anything inside ±i16 of the
    // camera target is safe to GTE-project. Compose the GTE
    // translation around that anchor: view·(anchor - cam_world)
    // = view·(-cam_local) where cam_local lives entirely within
    // the (small) orbit-radius range. This is the fix for the
    // 64×1024 = 65 536 > i16::MAX overflow the prior code's
    // saturating clamp would silently corrupt.
    let cam_local = [cam_x - target[0], cam_y - target[1], cam_z - target[2]];
    let tr = Vec3I32::new(
        -dot_view_world(view.m[0], cam_local),
        -dot_view_world(view.m[1], cam_local),
        -dot_view_world(view.m[2], cam_local),
    );

    set_view_anchor(target);
    gte_scene::load_rotation(&view);
    gte_scene::load_translation(tr);
    gte_scene::set_screen_offset(SCREEN_CX << 16, SCREEN_CY << 16);
    gte_scene::set_projection_plane(PROJ_H as u16);

    // Build a `WorldCamera` matching the same basis so the
    // model preview path can compose joint transforms with
    // `psx_engine::compute_joint_view_transform`.
    psx_engine::WorldCamera::from_basis(
        psx_engine::WorldProjection::new(SCREEN_CX as i16, SCREEN_CY as i16, PROJ_H, 32),
        psx_engine::WorldVertex::new(cam_x, cam_y, cam_z),
        sin_y,
        cos_y,
        sin_p,
        cos_p,
    )
}

/// Shared anchor that `world_to_view` subtracts from each vertex
/// before squashing to `i16`. Set per-frame by
/// `setup_gte_for_camera` to the camera target so the emitted
/// vertices stay anchor-relative — the GTE absorbs the offset via
/// its translation register. Without this, a single 32-sector
/// room (32 × 1024 = 32 768) sits exactly on the i16 cliff.
static VIEW_ANCHOR: std::sync::Mutex<[i32; 3]> = std::sync::Mutex::new([0, 0, 0]);

fn set_view_anchor(anchor: [i32; 3]) {
    if let Ok(mut a) = VIEW_ANCHOR.lock() {
        *a = anchor;
    }
}

fn view_anchor() -> [i32; 3] {
    VIEW_ANCHOR.lock().map(|a| *a).unwrap_or([0, 0, 0])
}

/// `view_row · world_pos` with the >>12 the GTE does internally for
/// matrix * world products.
fn dot_view_world(row: [i16; 3], v: [i32; 3]) -> i32 {
    let a = (row[0] as i32).saturating_mul(v[0]);
    let b = (row[1] as i32).saturating_mul(v[1]);
    let c = (row[2] as i32).saturating_mul(v[2]);
    a.saturating_add(b).saturating_add(c) >> 12
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// One placed light flattened into world coords + the
/// pre-multiplied colour×intensity_q8 the lighting accumulator
/// uses. Built once per frame from `walk_room`.
#[derive(Copy, Clone)]
struct PreviewLight {
    position: [i32; 3],
    radius: i32,
    /// `color × intensity` in 0..65535 per channel — already
    /// includes the Q8.8 multiplier so the per-face math
    /// reduces to one >>8 shift per attenuated channel.
    weighted_color: [u32; 3],
}

/// Walk every populated sector and emit triangles for floors,
/// ceilings, and the walls on each cardinal edge. Faces whose
/// material has a texture in the editor cache draw textured;
/// everything else falls back to flat shading. Light
/// accumulation happens per-face: the shade walks every
/// `PreviewLight` once, attenuates by distance to the face
/// centre, and modulates the base material colour.
fn walk_room(
    project: &ProjectDocument,
    room_id: psxed_project::NodeId,
    grid: &WorldGrid,
    textures: &EditorTextures,
    scratch: &mut PreviewScratch,
) {
    let s = grid.sector_size;
    let lights = collect_preview_lights(project, room_id, grid);
    let ambient = grid.ambient_color;
    for x in 0..grid.width {
        for z in 0..grid.depth {
            let Some(sector) = grid.sector(x, z) else {
                continue;
            };
            // Corner heights: [NW, NE, SE, SW] in `GridHorizontalFace`.
            // World coords with +X east, +Z south, +Y up.
            // `cell_world_x/z` add `grid.origin` so cells stay at the
            // same world position when the room grows in -X / -Z.
            let x0 = grid.cell_world_x(x);
            let x1 = x0 + s;
            let z0 = grid.cell_world_z(z);
            let z1 = z0 + s;

            if let Some(floor) = sector.floor.as_ref() {
                let center = horizontal_face_center([x0, x1, z0, z1], floor.heights);
                let shade = light_face(
                    face_shade(project, floor.material, FALLBACK_FLOOR, textures),
                    center,
                    &lights,
                    ambient,
                );
                push_horizontal_face(
                    scratch,
                    [x0, x1, z0, z1],
                    floor.heights,
                    floor.split,
                    floor.dropped_corner,
                    shade,
                    /* flip_winding */ false,
                );
            }
            if let Some(ceiling) = sector.ceiling.as_ref() {
                let center = horizontal_face_center([x0, x1, z0, z1], ceiling.heights);
                let shade = light_face(
                    face_shade(project, ceiling.material, FALLBACK_CEILING, textures),
                    center,
                    &lights,
                    ambient,
                );
                push_horizontal_face(
                    scratch,
                    [x0, x1, z0, z1],
                    ceiling.heights,
                    ceiling.split,
                    ceiling.dropped_corner,
                    shade,
                    // Ceiling normal points down; flipping the winding
                    // keeps backface-cullers happy and pins the inside
                    // surface as the visible side once we add culling.
                    /* flip_winding */
                    true,
                );
            }
            for &(direction, edge) in &[
                (GridDirection::North, WallEdge::North),
                (GridDirection::East, WallEdge::East),
                (GridDirection::South, WallEdge::South),
                (GridDirection::West, WallEdge::West),
            ] {
                for face in sector.walls.get(direction) {
                    let center = wall_face_center([x0, x1, z0, z1], edge, face.heights);
                    let shade = light_face(
                        face_shade(project, face.material, FALLBACK_WALL, textures),
                        center,
                        &lights,
                        ambient,
                    );
                    push_wall_face(
                        scratch,
                        [x0, x1, z0, z1],
                        edge,
                        face.heights,
                        face.dropped_corner,
                        shade,
                    );
                }
            }

            if scratch.used >= TRI_CAP || scratch.tex_used >= TRI_CAP {
                return;
            }
        }
    }
}

/// Per-face render description: either a texture sample with a
/// per-material tint, or a flat RGB. Resolved up-front so each
/// face's tri emit doesn't re-walk the resource table.
#[derive(Copy, Clone)]
enum FaceShade {
    Flat(u8, u8, u8),
    Textured {
        slot: MaterialSlot,
        tint: (u8, u8, u8),
    },
}

fn face_shade(
    project: &ProjectDocument,
    material: Option<ResourceId>,
    fallback: (u8, u8, u8),
    textures: &EditorTextures,
) -> FaceShade {
    let tint = material_color(project, material, fallback);
    if let Some(id) = material {
        if let Some(slot) = textures.slot(id) {
            return FaceShade::Textured { slot, tint };
        }
    }
    FaceShade::Flat(tint.0, tint.1, tint.2)
}

/// Walk every Light node in `project` whose enclosing room is
/// the active grid and pre-multiply its colour×intensity_q8.
/// Lights authored outside any Room (no enclosing parent) are
/// skipped silently — the cooker warns about those, the
/// preview just doesn't render them.
fn collect_preview_lights(
    project: &ProjectDocument,
    room_id: psxed_project::NodeId,
    grid: &WorldGrid,
) -> Vec<PreviewLight> {
    let s = grid.sector_size;
    let scene = project.active_scene();
    let mut out = Vec::new();
    for node in scene.nodes() {
        let NodeKind::Light {
            color,
            intensity,
            radius,
        } = &node.kind
        else {
            continue;
        };
        if *radius <= 0.0 || !intensity.is_finite() || *intensity < 0.0 {
            continue;
        }
        // Filter by enclosing Room — a light authored under
        // some other Room must not bleed into this one.
        if !is_descendant_of_room(scene, node.id, room_id) {
            continue;
        }
        let pos = node.transform.translation;
        // World position uses the same convention as
        // `walk_entities` / `walk_model_instances`: editor
        // translation is in *sectors relative to the room
        // origin*, so multiply by `sector_size` and shift by
        // half-grid. Room `origin` is baked into
        // `cell_world_x/z` for grid faces; entities use the
        // half-grid offset.
        let world = [
            ((pos[0] * s as f32) as i32).saturating_add((grid.width as i32 * s) / 2),
            (pos[1] * s as f32) as i32,
            ((pos[2] * s as f32) as i32).saturating_add((grid.depth as i32 * s) / 2),
        ];
        // Editor `radius` is in sector units; convert to
        // engine units once here so the per-face attenuation
        // math stays in world space.
        let radius_engine = (radius * s as f32) as i32;
        // Pre-multiply colour × intensity into u32 channels;
        // intensity scaled by 256 (Q8.8) keeps the per-face
        // accumulator in integer math.
        let intensity_q8 = (intensity * 256.0).clamp(0.0, u16::MAX as f32) as u32;
        let weighted_color = [
            color[0] as u32 * intensity_q8,
            color[1] as u32 * intensity_q8,
            color[2] as u32 * intensity_q8,
        ];
        out.push(PreviewLight {
            position: world,
            radius: radius_engine,
            weighted_color,
        });
    }
    out
}

/// Walk parent links from `node_id` looking for `room_id`.
/// `true` if `room_id` itself is on the chain. Used to confine
/// per-room lights to the room they were authored under.
fn is_descendant_of_room(
    scene: &psxed_project::Scene,
    node_id: psxed_project::NodeId,
    room_id: psxed_project::NodeId,
) -> bool {
    let mut current = Some(node_id);
    while let Some(id) = current {
        if id == room_id {
            return true;
        }
        current = scene.node(id).and_then(|n| n.parent);
    }
    false
}

/// Centre of a horizontal face (floor / ceiling) — average X /
/// Z of the bounds, mean of the four corner heights for Y.
fn horizontal_face_center(bounds: [i32; 4], heights: [i32; 4]) -> [i32; 3] {
    let [x0, x1, z0, z1] = bounds;
    let cx = (x0 + x1) / 2;
    let cz = (z0 + z1) / 2;
    let cy = (heights[0] as i64 + heights[1] as i64 + heights[2] as i64 + heights[3] as i64) / 4;
    [cx, cy as i32, cz]
}

/// Centre of a wall face — midpoint of the wall's bottom edge
/// in X/Z, midpoint of the four corner heights for Y. Wall
/// edges run along one of the cell's four cardinal sides; the
/// `WallEdge` picks which.
fn wall_face_center(bounds: [i32; 4], edge: WallEdge, heights: [i32; 4]) -> [i32; 3] {
    let [x0, x1, z0, z1] = bounds;
    let (cx, cz) = match edge {
        WallEdge::North => ((x0 + x1) / 2, z1),
        WallEdge::East => (x1, (z0 + z1) / 2),
        WallEdge::South => ((x0 + x1) / 2, z0),
        WallEdge::West => (x0, (z0 + z1) / 2),
    };
    let cy = (heights[0] as i64 + heights[1] as i64 + heights[2] as i64 + heights[3] as i64) / 4;
    [cx, cy as i32, cz]
}

/// Apply per-face lighting: ambient + linear-attenuation sum
/// of every `PreviewLight` whose radius covers `face_center`.
/// Final RGB clamps to 8 bits and modulates the input shade.
/// Lighting convention (PSX-neutral):
///
/// * `light_rgb` is in `0..=255` per channel.
/// * `128` = neutral — material renders at its base brightness.
/// * `0`   = pitch black.
/// * `255` = saturated overbright (clamped at the modulate
///   step).
///
/// Both the editor preview and the runtime use this scale.
/// Final colour = `base * light_rgb / 128`, clamped to `255`.
pub(crate) const LIGHTING_NEUTRAL: u32 = 128;
pub(crate) const LIGHTING_MAX: u32 = 255;

fn light_face(
    base: FaceShade,
    face_center: [i32; 3],
    lights: &[PreviewLight],
    ambient: [u8; 3],
) -> FaceShade {
    let base_color = match base {
        FaceShade::Flat(r, g, b) => (r, g, b),
        FaceShade::Textured { tint, .. } => tint,
    };
    // Start at room ambient — *not* `ambient * 256`. The
    // accumulator is the same 0..255 light_rgb space the
    // modulate step expects: ambient = neutral 128 produces
    // unmodified base material; ambient < 128 produces a
    // dimmer surface; ambient > 128 brightens.
    let mut light_rgb: [u32; 3] = [ambient[0] as u32, ambient[1] as u32, ambient[2] as u32];
    for light in lights {
        let dx = (face_center[0] - light.position[0]) as i64;
        let dy = (face_center[1] - light.position[1]) as i64;
        let dz = (face_center[2] - light.position[2]) as i64;
        let d2 = dx * dx + dy * dy + dz * dz;
        let r = light.radius as i64;
        if r <= 0 || d2 >= r * r {
            continue;
        }
        // Linear falloff: weight in Q8 — `0..=256` where 256
        // means "at the centre".
        let d = isqrt_i64(d2);
        let weight_q8 = (((r - d) << 8) / r) as u32;
        for (c, channel) in light_rgb.iter_mut().enumerate() {
            // weighted_color[c] = color[c] * intensity_q8 (up to
            // 255 * u16::MAX). One contribution lands as
            //   color * intensity * weight  → in 0..=255 typical
            //   = (weighted_color * weight_q8) >> 16
            let contrib = (light.weighted_color[c] as u64) * (weight_q8 as u64);
            *channel = channel.saturating_add((contrib >> 16) as u32);
        }
    }
    // Saturate light_rgb at 255 before modulating. Guarantees
    // `base * light / 128` can't exceed `base * 2`, which the
    // 8-bit clamp catches.
    for channel in &mut light_rgb {
        if *channel > LIGHTING_MAX {
            *channel = LIGHTING_MAX;
        }
    }
    let modulate = |base: u8, light: u32| -> u8 {
        let blended = (base as u32 * light) / LIGHTING_NEUTRAL;
        blended.min(255) as u8
    };
    let r = modulate(base_color.0, light_rgb[0]);
    let g = modulate(base_color.1, light_rgb[1]);
    let b = modulate(base_color.2, light_rgb[2]);
    match base {
        FaceShade::Flat(_, _, _) => FaceShade::Flat(r, g, b),
        FaceShade::Textured { slot, .. } => FaceShade::Textured {
            slot,
            tint: (r, g, b),
        },
    }
}

/// Integer square root for `i64`. Cheap iterative method;
/// runs once per (face × light) so it's not in a hot inner
/// loop. Returns 0 for negative input.
fn isqrt_i64(value: i64) -> i64 {
    if value <= 0 {
        return 0;
    }
    let mut x = value as u64;
    let mut r: u64 = 0;
    let mut bit: u64 = 1u64 << 62;
    while bit > x {
        bit >>= 2;
    }
    while bit != 0 {
        if x >= r + bit {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    r as i64
}

/// Project the four corners of a sector-aligned horizontal face
/// and emit one or two triangles. `heights` is `[NW, NE, SE, SW]`.
/// `flip_winding=true` reverses the vertex order for ceilings.
/// `dropped_corner=Some(c)` makes the face a triangle: the half
/// containing `c` is skipped (`split` must already be on the
/// diagonal that keeps the other half alive — `Corner::surviving_split`
/// enforces this at the data layer).
fn push_horizontal_face(
    scratch: &mut PreviewScratch,
    bounds: [i32; 4],
    heights: [i32; 4],
    split: GridSplit,
    dropped_corner: Option<psxed_project::Corner>,
    shade: FaceShade,
    flip_winding: bool,
) {
    use psxed_project::Corner;
    let [x0, x1, z0, z1] = bounds;
    let p_nw = gte_scene::project_vertex(world_to_view([x0, heights[0], z1]));
    let p_ne = gte_scene::project_vertex(world_to_view([x1, heights[1], z1]));
    let p_se = gte_scene::project_vertex(world_to_view([x1, heights[2], z0]));
    let p_sw = gte_scene::project_vertex(world_to_view([x0, heights[3], z0]));
    let (uv_nw, uv_ne, uv_se, uv_sw);
    let max_u;
    let max_v;
    if let FaceShade::Textured { slot, .. } = shade {
        max_u = slot.width.saturating_sub(1);
        max_v = slot.height.saturating_sub(1);
        uv_nw = (0u8, 0u8);
        uv_ne = (max_u, 0);
        uv_se = (max_u, max_v);
        uv_sw = (0, max_v);
    } else {
        max_u = 0;
        max_v = 0;
        uv_nw = (0, 0);
        uv_ne = (0, 0);
        uv_se = (0, 0);
        uv_sw = (0, 0);
    }
    let _ = max_u;
    let _ = max_v;

    // Per split, pick the two triangles. Triangle A is the
    // perimeter walk's "first" half, B the "second" — under
    // each split the dropped corner exists in exactly one of
    // them so we can skip cleanly.
    let (tri_a, tri_b) = match split {
        GridSplit::NorthWestSouthEast => (
            // (NW, NE, SE) and (NW, SE, SW) — diagonal NW–SE.
            (
                [p_nw, p_ne, p_se],
                [uv_nw, uv_ne, uv_se],
                [Corner::NW, Corner::NE, Corner::SE],
            ),
            (
                [p_nw, p_se, p_sw],
                [uv_nw, uv_se, uv_sw],
                [Corner::NW, Corner::SE, Corner::SW],
            ),
        ),
        GridSplit::NorthEastSouthWest => (
            // (NW, NE, SW) and (NE, SE, SW) — diagonal NE–SW.
            (
                [p_nw, p_ne, p_sw],
                [uv_nw, uv_ne, uv_sw],
                [Corner::NW, Corner::NE, Corner::SW],
            ),
            (
                [p_ne, p_se, p_sw],
                [uv_ne, uv_se, uv_sw],
                [Corner::NE, Corner::SE, Corner::SW],
            ),
        ),
    };

    let triangle_contains =
        |members: [Corner; 3], target: Corner| -> bool { members.contains(&target) };
    let emit_triangle = |scratch: &mut PreviewScratch,
                         verts: [psx_gte::scene::Projected; 3],
                         uvs: [(u8, u8); 3]| {
        if flip_winding {
            // Ceilings: forward `[0, 1, 2]` walk (CW from above
            // = CCW from below) so the inward normal points down.
            emit_face_tri(scratch, verts, uvs, shade);
        } else {
            // Floors: reverse to `[0, 2, 1]` (CCW from above),
            // matching the legacy non-flip winding.
            emit_face_tri(
                scratch,
                [verts[0], verts[2], verts[1]],
                [uvs[0], uvs[2], uvs[1]],
                shade,
            );
        }
    };

    let skip_a = dropped_corner
        .map(|c| triangle_contains(tri_a.2, c))
        .unwrap_or(false);
    let skip_b = dropped_corner
        .map(|c| triangle_contains(tri_b.2, c))
        .unwrap_or(false);
    if !skip_a {
        emit_triangle(scratch, tri_a.0, tri_a.1);
    }
    if !skip_b {
        emit_triangle(scratch, tri_b.0, tri_b.1);
    }
}

/// Which edge of the sector this wall sits on. The renderer needs
/// the four corner positions in a consistent order so heights[bl,
/// br, tr, tl] line up with the right world-space corners.
#[derive(Copy, Clone)]
enum WallEdge {
    North,
    East,
    South,
    West,
}

/// Build the four world-space corners of a wall face on `edge`
/// and emit one or two triangles. `heights` is the
/// `GridVerticalFace` `[bl, br, tr, tl]` quad. `dropped_corner`
/// makes the face a triangle: BR / TL skip the second triangle
/// of the BL-TR diagonal split; BL / TR fall through to the
/// other diagonal.
fn push_wall_face(
    scratch: &mut PreviewScratch,
    bounds: [i32; 4],
    edge: WallEdge,
    heights: [i32; 4],
    dropped_corner: Option<psxed_project::WallCorner>,
    shade: FaceShade,
) {
    use psxed_project::WallCorner;
    let [x0, x1, z0, z1] = bounds;
    // For each cardinal edge, "left" and "right" are picked so an
    // observer standing inside the sector sees the wall the right
    // way up.
    let (bl_xy, br_xy, tr_xy, tl_xy) = match edge {
        WallEdge::North => ((x0, z1), (x1, z1), (x1, z1), (x0, z1)),
        WallEdge::East => ((x1, z1), (x1, z0), (x1, z0), (x1, z1)),
        WallEdge::South => ((x1, z0), (x0, z0), (x0, z0), (x1, z0)),
        WallEdge::West => ((x0, z0), (x0, z1), (x0, z1), (x0, z0)),
    };
    let p_bl = gte_scene::project_vertex(world_to_view([bl_xy.0, heights[0], bl_xy.1]));
    let p_br = gte_scene::project_vertex(world_to_view([br_xy.0, heights[1], br_xy.1]));
    let p_tr = gte_scene::project_vertex(world_to_view([tr_xy.0, heights[2], tr_xy.1]));
    let p_tl = gte_scene::project_vertex(world_to_view([tl_xy.0, heights[3], tl_xy.1]));
    let (uv_bl, uv_br, uv_tr, uv_tl) = if let FaceShade::Textured { slot, .. } = shade {
        let u_max = slot.width.saturating_sub(1);
        let v_max = slot.height.saturating_sub(1);
        ((0, v_max), (u_max, v_max), (u_max, 0), (0, 0))
    } else {
        ((0, 0), (0, 0), (0, 0), (0, 0))
    };

    // Two diagonals to choose between. `BlTr` is the renderer's
    // legacy split; switch to `BrTl` only when BL or TR is the
    // dropped corner so a triangle survives.
    let use_br_tl = matches!(dropped_corner, Some(WallCorner::BL) | Some(WallCorner::TR));
    let (tri_a, tri_b) = if use_br_tl {
        (
            // (BL, BR, TL) and (BR, TR, TL) — diagonal BR-TL.
            (
                [p_bl, p_br, p_tl],
                [uv_bl, uv_br, uv_tl],
                [WallCorner::BL, WallCorner::BR, WallCorner::TL],
            ),
            (
                [p_br, p_tr, p_tl],
                [uv_br, uv_tr, uv_tl],
                [WallCorner::BR, WallCorner::TR, WallCorner::TL],
            ),
        )
    } else {
        (
            // (BL, BR, TR) and (BL, TR, TL) — diagonal BL-TR.
            (
                [p_bl, p_br, p_tr],
                [uv_bl, uv_br, uv_tr],
                [WallCorner::BL, WallCorner::BR, WallCorner::TR],
            ),
            (
                [p_bl, p_tr, p_tl],
                [uv_bl, uv_tr, uv_tl],
                [WallCorner::BL, WallCorner::TR, WallCorner::TL],
            ),
        )
    };

    let skip =
        |members: [WallCorner; 3]| -> bool { dropped_corner.is_some_and(|c| members.contains(&c)) };
    if !skip(tri_a.2) {
        emit_face_tri(scratch, tri_a.0, tri_a.1, shade);
    }
    if !skip(tri_b.2) {
        emit_face_tri(scratch, tri_b.0, tri_b.1, shade);
    }
}

/// Walk every placeable child node and stamp a small screen-space
/// marker so the user can see where they sit inside the room.
///
/// The room geometry uses the GTE-projected world coords; markers
/// project the same way so they read as "here is this thing in the
/// world", but the corners are drawn at fixed pixel offsets around
/// the projected centre — a billboarded square that doesn't shrink
/// with distance, the way Godot's editor sprites work.
fn walk_entities(
    project: &ProjectDocument,
    grid: &WorldGrid,
    selected: psxed_project::NodeId,
    scratch: &mut PreviewScratch,
) {
    let s = grid.sector_size;
    let scene = project.active_scene();
    for node in scene.nodes() {
        // Skip Model-backed MeshInstances — `walk_model_instances`
        // renders them as real textured models. Without this guard
        // they'd get *both* a marker square and the real model on
        // top of each other.
        if let NodeKind::MeshInstance { mesh: Some(id), .. } = &node.kind {
            if let Some(resource) = project.resource(*id) {
                if matches!(resource.data, ResourceData::Model(_)) {
                    continue;
                }
            }
        }
        let Some(kind_color) = entity_marker_color(&node.kind) else {
            continue;
        };
        // Editor convention: child node transforms are in "sectors
        // relative to the room origin." 2D viewport draws them at
        // exactly translation; 3D viewport must scale by sector_size
        // to match the room geometry below it.
        let pos = node.transform.translation;
        let entity_world = [
            ((pos[0] * s as f32) as i32).saturating_add((grid.width as i32 * s) / 2),
            (pos[1] * s as f32) as i32,
            ((pos[2] * s as f32) as i32).saturating_add((grid.depth as i32 * s) / 2),
        ];
        let projected = gte_scene::project_vertex(world_to_view(entity_world));
        if projected.sz == 0 {
            continue;
        }

        let is_selected = node.id == selected;
        let half = if is_selected { 9 } else { 6 };
        let (mut r, mut g, mut b) = kind_color;
        if is_selected {
            // Brighten selected markers so they stand out on top of
            // any colour scheme.
            r = r.saturating_add(0x40);
            g = g.saturating_add(0x40);
            b = b.saturating_add(0x40);
        }

        let cx = projected.sx;
        let cy = projected.sy;
        let p_tl = synth(cx - half, cy - half, projected.sz);
        let p_tr = synth(cx + half, cy - half, projected.sz);
        let p_br = synth(cx + half, cy + half, projected.sz);
        let p_bl = synth(cx - half, cy + half, projected.sz);
        push_tri(scratch, [p_tl, p_bl, p_tr], (r, g, b));
        push_tri(scratch, [p_tr, p_bl, p_br], (r, g, b));

        if is_selected {
            // Outline ring for selected entity: four thin tris
            // forming an offset square one pixel beyond the marker.
            let ring = half + 2;
            let outline = (0xFF, 0xFF, 0xFF);
            let r_tl = synth(cx - ring, cy - ring, projected.sz);
            let r_tr = synth(cx + ring, cy - ring, projected.sz);
            let r_br = synth(cx + ring, cy + ring, projected.sz);
            let r_bl = synth(cx - ring, cy + ring, projected.sz);
            push_tri(scratch, [r_tl, p_tl, r_tr], outline);
            push_tri(scratch, [r_tr, p_tl, p_tr], outline);
            push_tri(scratch, [r_tr, p_tr, r_br], outline);
            push_tri(scratch, [r_br, p_tr, p_br], outline);
            push_tri(scratch, [r_br, p_br, r_bl], outline);
            push_tri(scratch, [r_bl, p_br, p_bl], outline);
            push_tri(scratch, [r_bl, p_bl, r_tl], outline);
            push_tri(scratch, [r_tl, p_bl, p_tl], outline);
        }
    }
}

/// Cap on placed Model-backed MeshInstance nodes the editor
/// preview will render in one frame. Excess instances skip
/// silently (the manifest hasn't filtered them) — keeps a
/// runaway scene from busting the per-frame budget.
const MAX_PREVIEW_MODEL_INSTANCES: usize = 8;
/// Cap on joints any one previewed model can carry. Matches
/// the runtime `JOINT_CAP` so a model that renders in
/// editor-playtest also renders here.
const PREVIEW_JOINT_CAP: usize = 32;

/// Per-frame tick used to advance animation phase for the
/// editor's looping model preview. Bumped once per
/// `build_phase1_cmd_log` call. PSX angle / phase math needs
/// monotonic ticks rather than wall-clock, and the editor
/// frame rate fluctuates on host — so this is "preview
/// frames", not real-time. Good enough for inspector preview.
static PREVIEW_TICK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// One placed model the preview pass should render. Resolved
/// once per call to `walk_model_instances` so the per-instance
/// loop only does projection + emit work.
struct PreviewModelInstance<'a> {
    /// Cached parsed model. Owns no allocation; references
    /// bytes the caller keeps alive.
    model: psx_asset::Model<'a>,
    /// Cached parsed animation clip. Resolved through the
    /// preview / default clip rule.
    animation: psx_asset::Animation<'a>,
    /// Atlas slot in the editor's model atlas region.
    atlas: MaterialSlot,
    /// Render origin (room-local engine units). Model placement
    /// stays floor-anchored in `InstanceMeta`; this is lifted to
    /// the cooked model's centre before drawing.
    origin: psx_engine::WorldVertex,
    /// Y-axis rotation matrix derived from the node's yaw.
    instance_rotation: Mat3I16,
}

/// Render every Model-backed `MeshInstance` in the scene as a
/// real textured animated model. Mirrors the runtime path in
/// `editor-playtest`: parse `.psxmdl` + `.psxt` + `.psxanim`,
/// upload atlas (lazily — done by `EditorTextures::refresh_models`),
/// compose joint transforms via `compute_joint_view_transform`,
/// project per-vertex, emit textured triangles into the OT.
///
/// Models with bad/missing data are skipped silently — the
/// editor inspector + cook validation surface those errors
/// elsewhere; the preview just keeps drawing what it can.
fn walk_model_instances(
    project: &ProjectDocument,
    grid: &WorldGrid,
    textures: &EditorTextures,
    assets: &crate::editor_assets::EditorAssets,
    selected: psxed_project::NodeId,
    camera: &psx_engine::WorldCamera,
    scratch: &mut PreviewScratch,
) {
    // Bump the global preview tick once per frame so the
    // animation loops at a stable rate regardless of how many
    // instances we render.
    let tick = PREVIEW_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // The persistent `EditorAssets` cache owns mesh + animation
    // bytes. We only borrow into it here; nothing in this loop
    // touches the filesystem. Per-instance state (which clip is
    // active, where it lives in the world) lives in
    // `instances_meta`.
    let scene = project.active_scene();
    let mut instances_meta: Vec<InstanceMeta> = Vec::new();

    for node in scene.nodes() {
        if instances_meta.len() >= MAX_PREVIEW_MODEL_INSTANCES {
            break;
        }
        let NodeKind::MeshInstance {
            mesh: Some(mesh_id),
            animation_clip,
            ..
        } = &node.kind
        else {
            continue;
        };
        let Some(model_resource) = project.resource(*mesh_id) else {
            continue;
        };
        let ResourceData::Model(model) = &model_resource.data else {
            continue;
        };
        // Atlas required — runtime contract.
        if model.texture_path.is_none() {
            continue;
        }
        // Atlas slot must already be uploaded (refresh_models
        // ran earlier in the frame). Skip if not — lets the
        // user know visually that the atlas is broken.
        let Some(atlas_slot) = textures.model_atlas_slot(*mesh_id) else {
            continue;
        };

        // Resolve clip: explicit instance override → preview clip → default.
        let clip_local = animation_clip
            .or(model.preview_clip)
            .or(model.default_clip)
            .unwrap_or(0);
        if (clip_local as usize) >= model.clips.len() {
            continue;
        }

        // World position: same convention `walk_entities` uses
        // for marker nodes — translation is in sectors relative
        // to the room centre, so multiply by sector_size and
        // shift by half-grid.
        let s = grid.sector_size;
        let pos = node.transform.translation;
        let origin = psx_engine::WorldVertex::new(
            ((pos[0] * s as f32) as i32).saturating_add((grid.width as i32 * s) / 2),
            (pos[1] * s as f32) as i32,
            ((pos[2] * s as f32) as i32).saturating_add((grid.depth as i32 * s) / 2),
        );

        let yaw_q12 = yaw_to_q12(node.transform.rotation_degrees[1]);
        let instance_rotation = yaw_rotation_q12(yaw_q12);

        instances_meta.push(InstanceMeta {
            mesh_id: *mesh_id,
            clip_local,
            origin,
            instance_rotation,
            atlas: atlas_slot,
            is_selected: node.id == selected,
            yaw_q12,
            world_height: model.world_height as i32,
        });
    }

    // Player-spawn preview: render the player's character at
    // the spawn so level designers see where the player starts
    // *and* what they look like. Reuses the same model render
    // path — no separate player renderer.
    walk_player_spawn_preview(project, grid, textures, selected, &mut instances_meta);

    // Resolve parsed model + animation per instance straight
    // out of the cache. Each meta carries its own
    // `(mesh_id, clip_local)` pair so two instances of the
    // same model with different clips resolve to two different
    // animation entries — fixes the prior shared-buffer bug
    // where whichever clip got loaded first won.
    let mut instances: Vec<PreviewModelInstance> = Vec::new();
    for meta in &instances_meta {
        let Some(mesh_bytes) = assets.mesh_bytes(meta.mesh_id) else {
            continue;
        };
        let Some(anim_bytes) = assets.clip_bytes(meta.mesh_id, meta.clip_local) else {
            continue;
        };
        let Ok(model) = psx_asset::Model::from_bytes(mesh_bytes) else {
            continue;
        };
        let Ok(animation) = psx_asset::Animation::from_bytes(anim_bytes) else {
            continue;
        };
        instances.push(PreviewModelInstance {
            model,
            animation,
            atlas: meta.atlas,
            origin: floor_anchored_model_origin(meta.origin, meta.world_height),
            instance_rotation: meta.instance_rotation,
        });
    }

    // Gizmos first while GTE still holds the camera matrix —
    // `draw_preview_model_instance` overrides rotation/translation
    // per joint so any project_vertex after a model render uses
    // joint-space, not world-space.
    for meta in &instances_meta {
        if meta.is_selected {
            draw_model_selection_gizmo(meta, scratch);
        }
    }
    for instance in instances {
        draw_preview_model_instance(camera, tick, &instance, scratch);
    }
}

/// For every Player Spawn (`SpawnPoint { player: true, .. }`),
/// resolve its `character` link to a Model + idle clip and
/// queue an `InstanceMeta` so the same render path placed
/// model instances follow renders the character at the spawn.
/// `(mesh_id, clip_local)` is the cache key — different player
/// idle clips and different placed-instance clips each resolve
/// to their own animation entry.
///
/// Resolution rule mirrors the cooker:
/// 1. Explicit `character` assignment wins.
/// 2. If unset and exactly one Character resource exists,
///    auto-pick it.
/// 3. Otherwise skip the preview (the cook step's validation
///    will surface the missing character).
fn walk_player_spawn_preview(
    project: &ProjectDocument,
    grid: &WorldGrid,
    textures: &EditorTextures,
    selected: psxed_project::NodeId,
    instances_meta: &mut Vec<InstanceMeta>,
) {
    let scene = project.active_scene();
    for node in scene.nodes() {
        if instances_meta.len() >= MAX_PREVIEW_MODEL_INSTANCES {
            break;
        }
        let NodeKind::SpawnPoint {
            player: true,
            character,
        } = &node.kind
        else {
            continue;
        };
        let Some(character_id) = resolve_player_spawn_character(project, *character) else {
            continue;
        };
        let Some(character_resource) = project.resource(character_id) else {
            continue;
        };
        let ResourceData::Character(char_resource) = &character_resource.data else {
            continue;
        };
        let Some(model_id) = char_resource.model else {
            continue;
        };
        let Some(model_resource) = project.resource(model_id) else {
            continue;
        };
        let ResourceData::Model(model) = &model_resource.data else {
            continue;
        };
        if model.texture_path.is_none() {
            continue;
        }
        let Some(atlas_slot) = textures.model_atlas_slot(model_id) else {
            continue;
        };

        // Idle clip drives the preview loop — the spec wants
        // designers to see "what would the player be doing
        // standing still here". Falls through to the model's
        // preview / default clip if the Character has no idle
        // assigned, so the surface still renders even when the
        // Character is mid-author.
        let clip_local = char_resource
            .idle_clip
            .or(model.preview_clip)
            .or(model.default_clip)
            .unwrap_or(0);
        if (clip_local as usize) >= model.clips.len() {
            continue;
        }

        let s = grid.sector_size;
        let pos = node.transform.translation;
        let origin = psx_engine::WorldVertex::new(
            ((pos[0] * s as f32) as i32).saturating_add((grid.width as i32 * s) / 2),
            (pos[1] * s as f32) as i32,
            ((pos[2] * s as f32) as i32).saturating_add((grid.depth as i32 * s) / 2),
        );
        let yaw_q12 = yaw_to_q12(node.transform.rotation_degrees[1]);
        let instance_rotation = yaw_rotation_q12(yaw_q12);

        instances_meta.push(InstanceMeta {
            mesh_id: model_id,
            clip_local,
            origin,
            instance_rotation,
            atlas: atlas_slot,
            // Spawn node is selected, not the model — but the
            // preview gizmo still helps designers see *which*
            // spawn they have selected.
            is_selected: node.id == selected,
            yaw_q12,
            world_height: model.world_height as i32,
        });
    }
}

/// Resolve a Player Spawn's character reference, applying the
/// "auto-pick the only one" rule when no explicit character is
/// set. `None` means the editor preview can't render a player
/// model — typically because the project has zero or multiple
/// Characters and the spawn is mid-author.
fn resolve_player_spawn_character(
    project: &ProjectDocument,
    explicit: Option<ResourceId>,
) -> Option<ResourceId> {
    if let Some(id) = explicit {
        return Some(id);
    }
    let mut found: Option<ResourceId> = None;
    for r in &project.resources {
        if matches!(r.data, ResourceData::Character(_)) {
            if found.is_some() {
                return None;
            }
            found = Some(r.id);
        }
    }
    found
}

/// Selection gizmo for a placed model: a cyan vertical line
/// at the origin (visible against any backdrop) plus a yellow
/// forward arrow showing the yaw direction. The model itself
/// draws underneath the gizmo via the OT depth slot system.
///
/// Restores the camera GTE rotation/translation before
/// projecting because `draw_preview_model_instance` left the
/// GTE primed with the *last part's* joint transform.
fn draw_model_selection_gizmo(meta: &InstanceMeta, scratch: &mut PreviewScratch) {
    // Re-prime the GTE with the camera transform — model
    // rendering left it set to the last joint's view.
    // `world_to_view` already does the anchor subtract so we
    // just need rotation+translation back to camera basis.
    // Cheap: re-derive from VIEW_ANCHOR + the existing camera
    // matrix is harder than just calling project_vertex with
    // the camera setup. Skip the explicit restore and use
    // the existing set_view_anchor → world_to_view pipeline
    // by projecting via gte_scene::project_vertex with the
    // camera matrix re-loaded explicitly.
    //
    // Pragmatic shortcut: emit screen-space lines built from
    // worldspace endpoints projected with `gte_scene::project_vertex`
    // after we restore the camera transform via setup_gte_for_camera.
    // We don't have access to the camera state here, so the gizmo
    // routes through the same world_to_view + project_vertex path
    // the room geometry uses *before* model rendering kicks in.
    // To make this work we run gizmo emit *before* model render
    // in the caller; for now route it through and accept that
    // gizmos may use the last-joint transform if rendered after
    // the model. We'll fix ordering in the caller.

    let height = meta.world_height.max(256);
    let origin_w = [meta.origin.x, meta.origin.y, meta.origin.z];
    let top_w = [meta.origin.x, meta.origin.y - height, meta.origin.z];
    let mid_w = [meta.origin.x, meta.origin.y - height / 4, meta.origin.z];
    let len = (height / 3).max(128);
    let s = psx_gte::transform::sin_1_3_12(meta.yaw_q12) as i32;
    let c = psx_gte::transform::cos_1_3_12(meta.yaw_q12) as i32;
    let forward_w = [
        meta.origin.x + ((s * len) >> 12),
        meta.origin.y - height / 4,
        meta.origin.z + ((c * len) >> 12),
    ];

    let origin_p = gte_scene::project_vertex(world_to_view(origin_w));
    let top_p = gte_scene::project_vertex(world_to_view(top_w));
    let mid_p = gte_scene::project_vertex(world_to_view(mid_w));
    let forward_p = gte_scene::project_vertex(world_to_view(forward_w));

    let cyan = FaceOutlineStyle {
        rgb: (0x40, 0xC8, 0xE8),
        thickness_px: 3,
    };
    let yellow = FaceOutlineStyle {
        rgb: (0xF0, 0xC8, 0x40),
        thickness_px: 3,
    };
    if origin_p.sz != 0 && top_p.sz != 0 {
        push_screen_line(scratch, origin_p, top_p, cyan);
    }
    if mid_p.sz != 0 && forward_p.sz != 0 {
        push_screen_line(scratch, mid_p, forward_p, yellow);
    }
}

struct InstanceMeta {
    mesh_id: ResourceId,
    /// Clip index inside the model's clip list. Two instances
    /// of the same model with different clip overrides carry
    /// different `clip_local` values, which keys the
    /// `EditorAssets::clip_bytes` lookup so each instance's
    /// animation lands separately.
    clip_local: u16,
    origin: psx_engine::WorldVertex,
    instance_rotation: Mat3I16,
    atlas: MaterialSlot,
    /// `true` when the placed instance is the currently
    /// selected scene node. Drives the selection gizmo.
    is_selected: bool,
    /// Yaw in PSX angle units, retained for the facing arrow.
    yaw_q12: u16,
    /// Approximate world-space height for the facing arrow's
    /// vertical extent. Lifted from `ModelResource::world_height`.
    world_height: i32,
}

fn floor_anchored_model_origin(
    origin: psx_engine::WorldVertex,
    world_height: i32,
) -> psx_engine::WorldVertex {
    psx_engine::WorldVertex::new(
        origin.x,
        origin
            .y
            .saturating_add(model_origin_floor_lift(world_height)),
        origin.z,
    )
}

fn model_origin_floor_lift(world_height: i32) -> i32 {
    // Imported model vertices are normalized around their bounds
    // centre, while editor placements describe the floor contact
    // point. The model path's projected Y convention needs the
    // render origin offset by +half height for that floor anchor.
    world_height.max(0) / 2
}

/// Convert editor-Y rotation in degrees to PSX angle units
/// (Q12, 4096 per turn). Matches the playtest writer's
/// `yaw_from_degrees`.
fn yaw_to_q12(degrees: f32) -> u16 {
    let normalised = degrees.rem_euclid(360.0);
    (normalised * (4096.0 / 360.0)) as i32 as u16
}

/// Y-axis rotation matrix in Q12. Mirrors `yaw_rotation_matrix`
/// in editor-playtest's runtime.
fn yaw_rotation_q12(yaw_q12: u16) -> Mat3I16 {
    let s = psx_gte::transform::sin_1_3_12(yaw_q12);
    let c = psx_gte::transform::cos_1_3_12(yaw_q12);
    Mat3I16 {
        m: [[c, 0, s], [0, 0x1000, 0], [-s, 0, c]],
    }
}

/// Per-instance projection + face emit. Loops every part,
/// composes the joint view transform via the engine helper,
/// reloads the GTE per-part, projects vertices, emits one
/// textured triangle per face.
///
/// IMPORTANT: this clobbers the GTE rotation/translation
/// registers, so any caller relying on the camera-target
/// transform set by `setup_gte_for_camera` must restore it
/// before projecting non-model geometry. Today
/// `walk_model_instances` runs after every overlay (room +
/// entities + selection / hover / paint previews), so nothing
/// post-model depends on the camera transform.
fn draw_preview_model_instance(
    camera: &psx_engine::WorldCamera,
    tick: u32,
    instance: &PreviewModelInstance<'_>,
    scratch: &mut PreviewScratch,
) {
    let local_to_world =
        psx_engine::LocalToWorldScale::from_q12(instance.model.local_to_world_q12());
    let frame_q12 = instance.animation.phase_at_tick_q12(tick, 60);

    // Joint view transforms — one per joint, capped.
    let joint_count = (instance.model.joint_count() as usize).min(PREVIEW_JOINT_CAP);
    let mut joint_view_transforms: [psx_engine::JointViewTransform; PREVIEW_JOINT_CAP] =
        [psx_engine::JointViewTransform::ZERO; PREVIEW_JOINT_CAP];
    for (joint, joint_view_transform) in joint_view_transforms
        .iter_mut()
        .enumerate()
        .take(joint_count)
    {
        if let Some(pose) = instance.animation.pose_looped_q12(frame_q12, joint as u16) {
            let (rotation, translation) = psx_engine::compute_joint_view_transform(
                *camera,
                pose,
                instance.instance_rotation,
                local_to_world,
                instance.origin,
            );
            *joint_view_transform = psx_engine::JointViewTransform {
                rotation,
                translation,
            };
        }
    }

    // Per-part projection + face emit.
    //
    // Editor-preview limitation: skin-blend vertices (those
    // with `vertex.is_blend() == true`, i.e. carrying a
    // secondary joint + blend weight) project from the primary
    // joint only here. The runtime's `submit_textured_model`
    // implements the secondary-joint LERP path; the editor
    // preview takes the cheaper single-joint shortcut. For
    // models that use only rigid skinning (one joint per
    // vertex — current Wraith / Hooded Wretch rigs) the editor
    // preview matches the runtime exactly. For models with
    // secondary joint weights the preview will diverge at
    // those vertices; placement / clip / atlas validation
    // remain correct.
    let mut projected: Vec<psx_gte::scene::Projected> = Vec::with_capacity(64);
    for part_index in 0..instance.model.part_count() {
        let Some(part) = instance.model.part(part_index) else {
            break;
        };
        let primary_joint = part.joint_index() as usize;
        if primary_joint >= joint_count {
            continue;
        }
        let primary = joint_view_transforms[primary_joint];

        gte_scene::load_rotation(&primary.rotation);
        gte_scene::load_translation(primary.translation);

        let part_vertex_count = part.vertex_count() as usize;
        projected.clear();
        let first_vertex = part.first_vertex();
        for local in 0..part_vertex_count {
            let global = first_vertex.saturating_add(local as u16);
            let Some(v) = instance.model.vertex(global) else {
                break;
            };
            projected.push(gte_scene::project_vertex(v.position));
        }

        // Per-face triangle emit. UV pairs come straight from
        // the model vertex table; tint stays neutral.
        let first_face = part.first_face();
        let face_count = part.face_count();
        for face in 0..face_count {
            let Some((ia, ib, ic)) = instance.model.face(first_face + face) else {
                break;
            };
            let local_indices = [
                (ia as i32 - first_vertex as i32) as usize,
                (ib as i32 - first_vertex as i32) as usize,
                (ic as i32 - first_vertex as i32) as usize,
            ];
            if local_indices.iter().any(|&i| i >= projected.len()) {
                continue;
            }
            let p = [
                projected[local_indices[0]],
                projected[local_indices[1]],
                projected[local_indices[2]],
            ];
            // Skip near-plane / behind-camera triangles.
            if p[0].sz == 0 || p[1].sz == 0 || p[2].sz == 0 {
                continue;
            }
            // UV lookup straight off the model's vertex table.
            let uv = |idx: u16| -> (u8, u8) {
                instance
                    .model
                    .vertex(idx)
                    .map(|v| (v.uv.0, v.uv.1))
                    .unwrap_or((0, 0))
            };
            let uvs = [uv(ia), uv(ib), uv(ic)];
            push_tex_tri(scratch, p, uvs, instance.atlas, (0x80, 0x80, 0x80));
        }
    }
}

/// Draw a horizontal radius ring at floor level for every
/// `NodeKind::Light` in the scene. Selected lights get a
/// thicker, brighter ring; unselected lights get a thin one
/// in the light's own colour. Marker squares + selection
/// gizmos are still drawn by `walk_entities` (the ring is
/// additive).
fn walk_light_gizmos(
    project: &ProjectDocument,
    grid: &WorldGrid,
    selected: psxed_project::NodeId,
    scratch: &mut PreviewScratch,
) {
    let s = grid.sector_size;
    let scene = project.active_scene();
    for node in scene.nodes() {
        let NodeKind::Light {
            color,
            intensity: _,
            radius,
        } = &node.kind
        else {
            continue;
        };
        let pos = node.transform.translation;
        let center_world = [
            ((pos[0] * s as f32) as i32).saturating_add((grid.width as i32 * s) / 2),
            (pos[1] * s as f32) as i32,
            ((pos[2] * s as f32) as i32).saturating_add((grid.depth as i32 * s) / 2),
        ];
        // Light radius is authored in *sector units*; scale to
        // engine units so the ring matches the light's actual
        // attenuation footprint.
        let radius_engine = (radius * s as f32) as i32;
        if radius_engine <= 0 {
            continue;
        }

        let is_selected = node.id == selected;
        let style = if is_selected {
            FaceOutlineStyle {
                rgb: (0xFF, 0xE0, 0x80),
                thickness_px: 3,
            }
        } else {
            // Tint the unlit ring toward the authored colour
            // so multiple lights in a room read at a glance.
            FaceOutlineStyle {
                rgb: (color[0].max(0x40), color[1].max(0x40), color[2].max(0x40)),
                thickness_px: 1,
            }
        };
        push_horizontal_ring(scratch, center_world, radius_engine, 16, style);
    }
}

/// Wireframe AABB + facing arrow per selectable scene entity.
/// Bounds are gathered by `EditorWorkspace::collect_entity_bounds`
/// — every entity-kind node (model, spawn, light, trigger,
/// portal, audio source, legacy mesh) carries an AABB the user
/// can click to select and drag to move. This pass renders the
/// box wireframe so the user can *see* what they're picking;
/// the picker itself runs in psxed-ui and never reads from the
/// editor preview.
///
/// Idle bounds draw thin and muted so they don't dominate the
/// viewport over the room they sit in. Hover and selected reuse
/// the room face palette for cross-tool consistency: yellow for
/// hover, cyan-bold for selected.
fn walk_entity_bounds(
    bounds: &[psxed_ui::EntityBounds],
    selected: psxed_project::NodeId,
    hovered: Option<psxed_project::NodeId>,
    scratch: &mut PreviewScratch,
) {
    for b in bounds {
        let is_selected = b.node == selected;
        let is_hovered = hovered == Some(b.node);
        let style = entity_bound_style(b.kind, is_selected, is_hovered);
        push_aabb_wireframe(scratch, b.center, b.half_extents, style);

        // Yaw arrow only for kinds with meaningful facing —
        // models and spawn points point at where they'll
        // render / face. Lights / triggers / portals / audio
        // are either omnidirectional or carry their own
        // direction gizmo elsewhere (light radius ring).
        if matches!(
            b.kind,
            psxed_ui::EntityBoundKind::Model
                | psxed_ui::EntityBoundKind::SpawnPoint
                | psxed_ui::EntityBoundKind::MeshFallback
        ) {
            push_facing_arrow(scratch, b.center, b.half_extents, b.yaw_degrees, style);
        }
    }
}

/// Pick the outline style for one bound. Selected wins over
/// hover; idle uses a muted kind-tinted thin line so multiple
/// boxes in a busy room read at a glance without dominating.
fn entity_bound_style(
    kind: psxed_ui::EntityBoundKind,
    selected: bool,
    hovered: bool,
) -> FaceOutlineStyle {
    if selected {
        return FACE_OUTLINE_SELECTED;
    }
    if hovered {
        return FACE_OUTLINE_HOVER;
    }
    let rgb = match kind {
        psxed_ui::EntityBoundKind::Model => (0xC0, 0xC8, 0xD0),
        psxed_ui::EntityBoundKind::MeshFallback => (0x90, 0x98, 0xA0),
        psxed_ui::EntityBoundKind::SpawnPoint => (0x60, 0xE0, 0x80),
        psxed_ui::EntityBoundKind::Light => (0xFF, 0xD8, 0x70),
        psxed_ui::EntityBoundKind::Trigger => (0xC8, 0x80, 0xE0),
        psxed_ui::EntityBoundKind::Portal => (0xFF, 0xB0, 0x60),
        psxed_ui::EntityBoundKind::AudioSource => (0x70, 0xD8, 0xC0),
    };
    FaceOutlineStyle {
        rgb,
        thickness_px: 1,
    }
}

/// Project the 8 corners of a world-space AABB and emit the 12
/// edges as `push_screen_line` segments. Coordinates are stored
/// `f32` in the bound; rounded to `i32` here because the GTE
/// shim wants integer world coords.
fn push_aabb_wireframe(
    scratch: &mut PreviewScratch,
    center: [f32; 3],
    half_extents: [f32; 3],
    style: FaceOutlineStyle,
) {
    let cx = center[0].round() as i32;
    let cy = center[1].round() as i32;
    let cz = center[2].round() as i32;
    let hx = half_extents[0].round() as i32;
    let hy = half_extents[1].round() as i32;
    let hz = half_extents[2].round() as i32;
    if hx <= 0 || hy <= 0 || hz <= 0 {
        return;
    }
    let lo = [cx - hx, cy - hy, cz - hz];
    let hi = [cx + hx, cy + hy, cz + hz];
    // Corner index encoding: bit0 = X (lo/hi), bit1 = Y, bit2 = Z.
    let corner = |i: usize| -> [i32; 3] {
        [
            if i & 1 != 0 { hi[0] } else { lo[0] },
            if i & 2 != 0 { hi[1] } else { lo[1] },
            if i & 4 != 0 { hi[2] } else { lo[2] },
        ]
    };
    let p: [_; 8] = std::array::from_fn(|i| gte_scene::project_vertex(world_to_view(corner(i))));
    // 12 edges of a box: 4 along X, 4 along Y, 4 along Z. Pairs
    // of corners that differ in exactly one bit.
    const EDGES: [(usize, usize); 12] = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7), // along X
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7), // along Y
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // along Z
    ];
    for (a, b) in EDGES {
        if p[a].sz == 0 || p[b].sz == 0 {
            continue;
        }
        push_screen_line(scratch, p[a], p[b], style);
    }
}

/// Draw a forward-pointing arrow from the bound centre out
/// past the front face, indicating where the entity faces.
/// Length scales with the bound's horizontal extent so big
/// models get a visible arrow and tiny markers don't grow
/// horns.
fn push_facing_arrow(
    scratch: &mut PreviewScratch,
    center: [f32; 3],
    half_extents: [f32; 3],
    yaw_degrees: f32,
    style: FaceOutlineStyle,
) {
    let yaw_q12 = yaw_to_q12(yaw_degrees);
    let s = psx_gte::transform::sin_1_3_12(yaw_q12) as i32;
    let c = psx_gte::transform::cos_1_3_12(yaw_q12) as i32;
    // Arrow length = bound's horizontal half-extent + a small
    // overshoot so the head sits clearly outside the box.
    let reach = (half_extents[0].max(half_extents[2]) * 1.5).max(96.0) as i32;
    let cx = center[0].round() as i32;
    let cy = center[1].round() as i32;
    let cz = center[2].round() as i32;
    let tip = [cx + ((s * reach) >> 12), cy, cz + ((c * reach) >> 12)];
    let p_origin = gte_scene::project_vertex(world_to_view([cx, cy, cz]));
    let p_tip = gte_scene::project_vertex(world_to_view(tip));
    if p_origin.sz != 0 && p_tip.sz != 0 {
        push_screen_line(scratch, p_origin, p_tip, style);
    }
}

/// Project a horizontal `segments`-sided polygon at world
/// `center` with `radius` into screen space and emit one
/// `push_screen_line` per edge. Used for light radius gizmos
/// and any future ground-plane affordances.
fn push_horizontal_ring(
    scratch: &mut PreviewScratch,
    center: [i32; 3],
    radius: i32,
    segments: u16,
    style: FaceOutlineStyle,
) {
    if segments < 3 || radius <= 0 {
        return;
    }
    let mut prev_world = [center[0] + radius, center[1], center[2]];
    let mut prev_proj = gte_scene::project_vertex(world_to_view(prev_world));
    for i in 1..=segments {
        // PSX trig uses 4096 units per turn; sample the unit
        // circle around the light origin once per segment.
        let angle_q12 = ((i as u32 * 4096) / segments as u32) as u16;
        let s = psx_gte::transform::sin_1_3_12(angle_q12) as i32;
        let c = psx_gte::transform::cos_1_3_12(angle_q12) as i32;
        let next_world = [
            center[0] + ((c * radius) >> 12),
            center[1],
            center[2] + ((s * radius) >> 12),
        ];
        let next_proj = gte_scene::project_vertex(world_to_view(next_world));
        if prev_proj.sz != 0 && next_proj.sz != 0 {
            push_screen_line(scratch, prev_proj, next_proj, style);
        }
        prev_world = next_world;
        prev_proj = next_proj;
    }
    let _ = prev_world; // silence the unused-final-assignment lint
}

fn synth(sx: i16, sy: i16, sz: u16) -> psx_gte::scene::Projected {
    psx_gte::scene::Projected { sx, sy, sz }
}

/// Marker colour per node kind, or `None` for nodes that aren't
/// placeable in 3D space (the World macro, the Room itself, plain
/// transform-only nodes).
fn entity_marker_color(kind: &NodeKind) -> Option<(u8, u8, u8)> {
    match kind {
        NodeKind::SpawnPoint { player: true, .. } => Some((0x60, 0xE0, 0x80)),
        NodeKind::SpawnPoint { player: false, .. } => Some((0x60, 0xB8, 0xF0)),
        NodeKind::MeshInstance { .. } => Some((0xC0, 0xC8, 0xD0)),
        NodeKind::Light { .. } => Some((0xFF, 0xD8, 0x70)),
        NodeKind::Trigger { .. } => Some((0xC8, 0x80, 0xE0)),
        NodeKind::Portal { .. } => Some((0xFF, 0xB0, 0x60)),
        NodeKind::AudioSource { .. } => Some((0x70, 0xD8, 0xC0)),
        NodeKind::Room { .. } | NodeKind::World | NodeKind::Node | NodeKind::Node3D => None,
    }
}

const FALLBACK_FLOOR: (u8, u8, u8) = (0xB0, 0xA0, 0x88);
const FALLBACK_WALL: (u8, u8, u8) = (0x88, 0x70, 0x58);
const FALLBACK_CEILING: (u8, u8, u8) = (0x60, 0x60, 0x70);

/// Pick the GP0 RGB triple to paint a face with.
///
/// Authored `MaterialResource::tint` defaults to PSX-neutral
/// `(0x80, 0x80, 0x80)` because that's the right value when sampling
/// a textured polygon (output = texel × tint / 128). For the editor's
/// pre-textured flat-shaded preview that means every face renders the
/// same dull grey — useless for distinguishing materials. Mirror the
/// 2D viewport's approach: derive a colour from the material's name
/// so a project's "Floor Material" / "Brick Material" / "Glass" all
/// land at distinct, recognisable hues until real texturing arrives.
fn material_color(
    project: &ProjectDocument,
    material: Option<ResourceId>,
    fallback: (u8, u8, u8),
) -> (u8, u8, u8) {
    let Some(id) = material else {
        return fallback;
    };
    let Some(resource) = project.resource(id) else {
        return fallback;
    };
    let name = resource.name.to_ascii_lowercase();
    if name.contains("brick") {
        (0xC8, 0x70, 0x40)
    } else if name.contains("floor") || name.contains("stone") {
        (0xB6, 0xAC, 0x96)
    } else if name.contains("glass") {
        (0x70, 0xA8, 0xC0)
    } else if name.contains("wood") {
        (0x90, 0x60, 0x40)
    } else if name.contains("metal") {
        (0x90, 0x96, 0x9A)
    } else if let ResourceData::Material(mat) = &resource.data {
        // Author actually tinted the material away from neutral — use
        // the tint directly. The mid-grey default falls through to
        // the role-specific fallback below.
        if mat.tint != [0x80, 0x80, 0x80] {
            let [r, g, b] = mat.tint;
            (r, g, b)
        } else {
            fallback
        }
    } else {
        fallback
    }
}

/// Squash a world-space i32 corner into the i16 the GTE V0 register
/// expects. Subtracts the per-frame view anchor (= camera target)
/// first so the emitted coord is anchor-relative. With sector_size
/// 1024, this gives ±32 sectors of headroom from the camera target
/// before clamp truncation kicks in — comfortably the editor's
/// budget cap.
///
/// Debug builds assert; release silently saturates rather than
/// crashing the editor mid-paint.
fn world_to_view(world: [i32; 3]) -> Vec3I16 {
    let a = view_anchor();
    let lx = world[0] - a[0];
    let ly = world[1] - a[1];
    let lz = world[2] - a[2];
    debug_assert!(
        (i16::MIN as i32..=i16::MAX as i32).contains(&lx)
            && (i16::MIN as i32..=i16::MAX as i32).contains(&ly)
            && (i16::MIN as i32..=i16::MAX as i32).contains(&lz),
        "vertex {:?} (anchor-relative {:?}) overflows i16 — room too big or camera anchor wrong",
        world,
        [lx, ly, lz]
    );
    Vec3I16::new(clamp_i16(lx), clamp_i16(ly), clamp_i16(lz))
}

/// Render the paint-target ghost outline. Cell ghosts trace the
/// floor footprint of the would-be cell; wall ghosts use
/// `push_face_outline` with a synthetic `FaceRef` whose world cell
/// might lie outside the current grid — `push_face_outline`'s
/// missing-data fallback supplies default heights for the ghost
/// case. World-cell coords let both work for cells the grid
/// hasn't allocated yet; the outline appears exactly where the
/// auto-grow would create the cell on click.
fn push_paint_preview(
    grid: &WorldGrid,
    preview: psxed_ui::PaintTargetPreview,
    scratch: &mut PreviewScratch,
) {
    match preview {
        psxed_ui::PaintTargetPreview::Cell {
            world_cell_x,
            world_cell_z,
        } => push_cell_ghost_outline(grid, world_cell_x, world_cell_z, scratch),
        psxed_ui::PaintTargetPreview::Wall {
            world_cell_x,
            world_cell_z,
            dir,
            stack,
        } => {
            // Translate world cell → array (when in bounds) so
            // existing wall data is read for the outline; for
            // off-grid ghosts we pass a synthetic array index that
            // can't collide with any real wall and let
            // `push_face_outline` fall back to default heights.
            let (sx, sz) = grid
                .world_cell_to_array(world_cell_x, world_cell_z)
                .unwrap_or((u16::MAX, u16::MAX));
            // Fake a FaceRef. `room` field is unused by
            // push_face_outline; safe to fill with anything.
            let face = psxed_ui::FaceRef {
                room: psxed_project::NodeId::ROOT,
                sx,
                sz,
                kind: psxed_ui::FaceKind::Wall { dir, stack },
            };
            // For off-grid wall ghosts we have to project the
            // outline ourselves — `push_face_outline` short-
            // circuits when sx/sz are out of grid bounds.
            if sx == u16::MAX || sz == u16::MAX {
                push_ghost_wall_outline(grid, world_cell_x, world_cell_z, dir, scratch);
            } else {
                push_face_outline(grid, face, FACE_OUTLINE_WALL_PAINT, scratch);
            }
        }
    }
}

/// Outline a cell at world-cell `(wcx, wcz)`. Draws four screen-
/// space lines along the cell's edges, lifted slightly above
/// `y = 0` so the strokes don't z-fight any existing floor at the
/// same world position.
fn push_cell_ghost_outline(grid: &WorldGrid, wcx: i32, wcz: i32, scratch: &mut PreviewScratch) {
    let s = grid.sector_size;
    let x0 = wcx * s;
    let x1 = x0 + s;
    let z0 = wcz * s;
    let z1 = z0 + s;
    let y = 4;
    let nw = gte_scene::project_vertex(world_to_view([x0, y, z1]));
    let ne = gte_scene::project_vertex(world_to_view([x1, y, z1]));
    let se = gte_scene::project_vertex(world_to_view([x1, y, z0]));
    let sw = gte_scene::project_vertex(world_to_view([x0, y, z0]));
    if [nw, ne, se, sw].iter().any(|p| p.sz == 0) {
        return;
    }
    for (a, b) in [(nw, ne), (ne, se), (se, sw), (sw, nw)] {
        push_screen_line(scratch, a, b, FACE_OUTLINE_HOVER);
    }
}

/// Outline a wall at world-cell `(wcx, wcz)` on edge `dir` with
/// default heights `[0, 0, sector_size, sector_size]`. Used when
/// `push_face_outline`'s array-bound check rejects an off-grid
/// ghost so the user still sees where the wall will land.
fn push_ghost_wall_outline(
    grid: &WorldGrid,
    wcx: i32,
    wcz: i32,
    dir: GridDirection,
    scratch: &mut PreviewScratch,
) {
    let s = grid.sector_size;
    let x0 = wcx * s;
    let x1 = x0 + s;
    let z0 = wcz * s;
    let z1 = z0 + s;
    let (bl_xy, br_xy) = wall_xy_for(dir, x0, x1, z0, z1);
    const LIFT: i32 = 4;
    let (nx, nz) = wall_inward_normal(dir);
    let corners = [
        [bl_xy.0 + LIFT * nx, 0, bl_xy.1 + LIFT * nz],
        [br_xy.0 + LIFT * nx, 0, br_xy.1 + LIFT * nz],
        [br_xy.0 + LIFT * nx, s, br_xy.1 + LIFT * nz],
        [bl_xy.0 + LIFT * nx, s, bl_xy.1 + LIFT * nz],
    ];
    let projected: [psx_gte::scene::Projected; 4] = [
        gte_scene::project_vertex(world_to_view(corners[0])),
        gte_scene::project_vertex(world_to_view(corners[1])),
        gte_scene::project_vertex(world_to_view(corners[2])),
        gte_scene::project_vertex(world_to_view(corners[3])),
    ];
    if projected.iter().any(|p| p.sz == 0) {
        return;
    }
    for i in 0..4 {
        push_screen_line(
            scratch,
            projected[i],
            projected[(i + 1) % 4],
            FACE_OUTLINE_WALL_PAINT,
        );
    }
}

/// Hover and Selected outline styling. RGB plus screen-space line
/// thickness in pixels; selected reads bolder so the user can spot
/// it across a busy room. Thicknesses are at least 2 px because
/// `push_screen_line` carries the line as two screen-space tris
/// whose half-width gets truncated to integer pixels — anything
/// less collapses to a degenerate zero-width strip.
const FACE_OUTLINE_HOVER: FaceOutlineStyle = FaceOutlineStyle {
    rgb: (0xFF, 0xE0, 0x60),
    thickness_px: 2,
};
const FACE_OUTLINE_SELECTED: FaceOutlineStyle = FaceOutlineStyle {
    rgb: (0x60, 0xC8, 0xFF),
    thickness_px: 4,
};
/// PaintWall hover preview — green for "this would be added /
/// replaced". 3 px so it reads through the `FACE_OUTLINE_HOVER`
/// yellow when both fire on the same face.
const FACE_OUTLINE_WALL_PAINT: FaceOutlineStyle = FaceOutlineStyle {
    rgb: (0x60, 0xFF, 0x90),
    thickness_px: 3,
};

#[derive(Copy, Clone)]
struct FaceOutlineStyle {
    rgb: (u8, u8, u8),
    thickness_px: i16,
}

/// Hover vs Selected — outline style picker for the unified
/// selection dispatch. Hover uses the lighter yellow; selected
/// uses the bolder cyan. Same constants the original face-only
/// path consumed.
#[derive(Copy, Clone)]
enum OutlineRole {
    Hover,
    Selected,
}

impl OutlineRole {
    fn face_style(self) -> FaceOutlineStyle {
        match self {
            Self::Hover => FACE_OUTLINE_HOVER,
            Self::Selected => FACE_OUTLINE_SELECTED,
        }
    }
}

/// Dispatch a `Selection` to the appropriate outline helper.
/// Each variant gets its own screen-space overlay: face → 4
/// edge lines, edge → 1 line, vertex → cross.
fn push_selection_outline(
    grid: &WorldGrid,
    selection: psxed_ui::Selection,
    role: OutlineRole,
    scratch: &mut PreviewScratch,
) {
    match selection {
        psxed_ui::Selection::Face(face) => {
            push_face_outline(grid, face, role.face_style(), scratch);
        }
        psxed_ui::Selection::Edge(edge) => {
            push_edge_outline(grid, edge, role.face_style(), scratch);
        }
        psxed_ui::Selection::Vertex(vertex) => {
            push_vertex_outline(grid, vertex, role.face_style(), scratch);
        }
    }
}

/// One thick screen-space line between the edge's two world
/// endpoints. Lifted slightly off the surface (same `LIFT` as
/// `push_face_outline`) so it doesn't z-fight the geometry it
/// outlines.
fn push_edge_outline(
    grid: &WorldGrid,
    edge: psxed_ui::EdgeRef,
    style: FaceOutlineStyle,
    scratch: &mut PreviewScratch,
) {
    let Some((a, b)) = edge_world_endpoints(grid, edge) else {
        return;
    };
    let projected_a = gte_scene::project_vertex(world_to_view(a));
    let projected_b = gte_scene::project_vertex(world_to_view(b));
    if projected_a.sz == 0 || projected_b.sz == 0 {
        return;
    }
    push_screen_line(scratch, projected_a, projected_b, style);
}

/// Small screen-space cross at the vertex's world position.
/// The cross is drawn as four short line segments offset along
/// world axes so its on-screen size scales naturally with
/// distance — close vertices read clearly, far ones don't
/// dominate the viewport.
fn push_vertex_outline(
    grid: &WorldGrid,
    vertex: psxed_ui::VertexRef,
    style: FaceOutlineStyle,
    scratch: &mut PreviewScratch,
) {
    let Some(world) = vertex_world_position(grid, vertex) else {
        return;
    };
    // Half-extent of the cross in world units. ~32 reads as a
    // few px in the viewport at orbit distances we use.
    const ARM: i32 = 32;
    let arms = [
        (
            [world[0] - ARM, world[1], world[2]],
            [world[0] + ARM, world[1], world[2]],
        ),
        (
            [world[0], world[1] - ARM, world[2]],
            [world[0], world[1] + ARM, world[2]],
        ),
        (
            [world[0], world[1], world[2] - ARM],
            [world[0], world[1], world[2] + ARM],
        ),
    ];
    for (a, b) in arms {
        let pa = gte_scene::project_vertex(world_to_view(a));
        let pb = gte_scene::project_vertex(world_to_view(b));
        if pa.sz == 0 || pb.sz == 0 {
            continue;
        }
        push_screen_line(scratch, pa, pb, style);
    }
}

fn edge_world_endpoints(grid: &WorldGrid, edge: psxed_ui::EdgeRef) -> Option<([i32; 3], [i32; 3])> {
    use psxed_ui::{EdgeAnchor, FaceCornerRef};
    let (a, b) = match edge.anchor {
        EdgeAnchor::Floor { sx, sz, dir } => (
            FaceCornerRef::Floor {
                sx,
                sz,
                corner: floor_edge_a(dir),
            },
            FaceCornerRef::Floor {
                sx,
                sz,
                corner: floor_edge_b(dir),
            },
        ),
        EdgeAnchor::Ceiling { sx, sz, dir } => (
            FaceCornerRef::Ceiling {
                sx,
                sz,
                corner: floor_edge_a(dir),
            },
            FaceCornerRef::Ceiling {
                sx,
                sz,
                corner: floor_edge_b(dir),
            },
        ),
        EdgeAnchor::Wall {
            sx,
            sz,
            dir,
            stack,
            edge: e,
        } => (
            FaceCornerRef::Wall {
                sx,
                sz,
                dir,
                stack,
                corner: wall_edge_a(e),
            },
            FaceCornerRef::Wall {
                sx,
                sz,
                dir,
                stack,
                corner: wall_edge_b(e),
            },
        ),
    };
    Some((
        psxed_ui::face_corner_world(grid, a)?,
        psxed_ui::face_corner_world(grid, b)?,
    ))
}

fn vertex_world_position(grid: &WorldGrid, vertex: psxed_ui::VertexRef) -> Option<[i32; 3]> {
    psxed_ui::face_corner_world(grid, vertex.anchor.as_face_corner())
}

const fn floor_edge_a(dir: GridDirection) -> psxed_ui::Corner {
    match dir {
        GridDirection::North => psxed_ui::Corner::NW,
        GridDirection::East => psxed_ui::Corner::NE,
        GridDirection::South => psxed_ui::Corner::SE,
        GridDirection::West => psxed_ui::Corner::SW,
        _ => psxed_ui::Corner::NW,
    }
}

const fn floor_edge_b(dir: GridDirection) -> psxed_ui::Corner {
    match dir {
        GridDirection::North => psxed_ui::Corner::NE,
        GridDirection::East => psxed_ui::Corner::SE,
        GridDirection::South => psxed_ui::Corner::SW,
        GridDirection::West => psxed_ui::Corner::NW,
        _ => psxed_ui::Corner::SE,
    }
}

const fn wall_edge_a(edge: psxed_ui::WallEdge) -> psxed_ui::WallCorner {
    match edge {
        psxed_ui::WallEdge::Bottom => psxed_ui::WallCorner::BL,
        psxed_ui::WallEdge::Right => psxed_ui::WallCorner::BR,
        psxed_ui::WallEdge::Top => psxed_ui::WallCorner::TR,
        psxed_ui::WallEdge::Left => psxed_ui::WallCorner::TL,
    }
}

const fn wall_edge_b(edge: psxed_ui::WallEdge) -> psxed_ui::WallCorner {
    match edge {
        psxed_ui::WallEdge::Bottom => psxed_ui::WallCorner::BR,
        psxed_ui::WallEdge::Right => psxed_ui::WallCorner::TR,
        psxed_ui::WallEdge::Top => psxed_ui::WallCorner::TL,
        psxed_ui::WallEdge::Left => psxed_ui::WallCorner::BL,
    }
}

/// Stamp four short, screen-space-thick line segments along the
/// edges of a picked face. Drawing in screen space (after GTE
/// projection) keeps the outline a constant pixel weight regardless
/// of perspective, which matches Godot / Unity's "selection halo"
/// look. Lines pinned to OT slot 0 so they paint on top of every
/// floor / wall / ceiling.
fn push_face_outline(
    grid: &WorldGrid,
    face: psxed_ui::FaceRef,
    style: FaceOutlineStyle,
    scratch: &mut PreviewScratch,
) {
    if face.sx >= grid.width || face.sz >= grid.depth {
        return;
    }
    let sector = grid.sector(face.sx, face.sz);
    let s = grid.sector_size;
    let x0 = grid.cell_world_x(face.sx);
    let x1 = x0 + s;
    let z0 = grid.cell_world_z(face.sz);
    let z1 = z0 + s;
    // Lift a hair off the surface so the outline doesn't z-fight
    // the face it's marking. Sloped floors keep their relative
    // outline position because we lift each corner by the same
    // amount along the local up axis.
    const LIFT: i32 = 4;
    let corners = match face.kind {
        psxed_ui::FaceKind::Floor => sector.and_then(|s| s.floor.as_ref()).map(|f| {
            let h = f.heights;
            [
                [x0, h[0] + LIFT, z1],
                [x1, h[1] + LIFT, z1],
                [x1, h[2] + LIFT, z0],
                [x0, h[3] + LIFT, z0],
            ]
        }),
        psxed_ui::FaceKind::Ceiling => sector.and_then(|s| s.ceiling.as_ref()).map(|c| {
            let h = c.heights;
            [
                [x0, h[0] - LIFT, z1],
                [x1, h[1] - LIFT, z1],
                [x1, h[2] - LIFT, z0],
                [x0, h[3] - LIFT, z0],
            ]
        }),
        psxed_ui::FaceKind::Wall { dir, stack } => {
            // Default ghost heights span the full sector when the
            // wall doesn't exist yet — used by the PaintWall
            // hover preview to outline where a brand-new wall
            // would land.
            let h = sector
                .and_then(|s| s.walls.get(dir).get(stack as usize))
                .map(|wall| wall.heights)
                .unwrap_or([0, 0, s, s]);
            let (bl_xy, br_xy) = wall_xy_for(dir, x0, x1, z0, z1);
            // Inset along the wall's inward normal so the outline
            // sits inside the room rather than z-fighting the
            // wall surface when viewed from inside.
            let (nx, nz) = wall_inward_normal(dir);
            Some([
                [bl_xy.0 + LIFT * nx, h[0], bl_xy.1 + LIFT * nz],
                [br_xy.0 + LIFT * nx, h[1], br_xy.1 + LIFT * nz],
                [br_xy.0 + LIFT * nx, h[2], br_xy.1 + LIFT * nz],
                [bl_xy.0 + LIFT * nx, h[3], bl_xy.1 + LIFT * nz],
            ])
        }
    };
    let Some(corners) = corners else { return };
    let projected: [psx_gte::scene::Projected; 4] = [
        gte_scene::project_vertex(world_to_view(corners[0])),
        gte_scene::project_vertex(world_to_view(corners[1])),
        gte_scene::project_vertex(world_to_view(corners[2])),
        gte_scene::project_vertex(world_to_view(corners[3])),
    ];
    // Skip outlines whose corners didn't project — `project_vertex`
    // returns `sz == 0` for behind-camera or near-plane-clipped
    // points, which would produce nonsense screen lines.
    if projected.iter().any(|p| p.sz == 0) {
        return;
    }
    for i in 0..4 {
        let a = projected[i];
        let b = projected[(i + 1) % 4];
        push_screen_line(scratch, a, b, style);
    }
}

/// World-space (x, z) endpoints of a wall on the given cardinal
/// edge — mirrors `push_wall_face` so picking, paint preview, and
/// outline rendering all agree.
fn wall_xy_for(dir: GridDirection, x0: i32, x1: i32, z0: i32, z1: i32) -> ((i32, i32), (i32, i32)) {
    match dir {
        GridDirection::North => ((x0, z1), (x1, z1)),
        GridDirection::East => ((x1, z1), (x1, z0)),
        GridDirection::South => ((x1, z0), (x0, z0)),
        GridDirection::West => ((x0, z0), (x0, z1)),
        _ => ((x0, z0), (x0, z0)),
    }
}

/// Inward-facing normal (sign of x, sign of z) for the wall on
/// `dir`. Used to inset the outline so it sits *inside* the room,
/// not behind the wall plane.
fn wall_inward_normal(dir: GridDirection) -> (i32, i32) {
    match dir {
        GridDirection::North => (0, -1),
        GridDirection::East => (-1, 0),
        GridDirection::South => (0, 1),
        GridDirection::West => (1, 0),
        _ => (0, 0),
    }
}

/// Round-away-from-zero, clamped so any non-zero perpendicular is
/// at least ±1 pixel. Keeps thin diagonal lines from collapsing to
/// degenerate zero-width strips after `as i16` truncation.
fn round_perp(value: f32) -> i16 {
    if value > 0.0 {
        (value + 0.5).max(1.0) as i16
    } else if value < 0.0 {
        (value - 0.5).min(-1.0) as i16
    } else {
        0
    }
}

/// Emit two triangles forming a `thickness_px`-pixel-wide line
/// between two screen-projected vertices. The OT slot is pinned at
/// 0 so the line draws on top of everything.
fn push_screen_line(
    scratch: &mut PreviewScratch,
    a: psx_gte::scene::Projected,
    b: psx_gte::scene::Projected,
    style: FaceOutlineStyle,
) {
    // Perpendicular to (b - a) in screen space, normalised to
    // `thickness_px / 2`. Width 0 (parallel a/b) collapses the line
    // to a single screen point, which we just skip.
    let dx = (b.sx as f32) - (a.sx as f32);
    let dy = (b.sy as f32) - (a.sy as f32);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        return;
    }
    let half = (style.thickness_px as f32) * 0.5;
    // Round-away-from-zero on both components so a diagonal screen
    // edge with a sub-pixel perpendicular doesn't collapse the line
    // to zero width — `0.707 as i16` truncates to 0, so a 2 px
    // hover line on any tilted edge would otherwise disappear. We
    // also clamp the magnitude to ≥ 1 px on the dominant axis so
    // very thin slopes still render visibly.
    let nx_f = -dy / len * half;
    let ny_f = dx / len * half;
    let nx = round_perp(nx_f);
    let ny = round_perp(ny_f);
    let a_lo = synth(a.sx.saturating_add(-nx), a.sy.saturating_add(-ny), a.sz);
    let a_hi = synth(a.sx.saturating_add(nx), a.sy.saturating_add(ny), a.sz);
    let b_lo = synth(b.sx.saturating_add(-nx), b.sy.saturating_add(-ny), b.sz);
    let b_hi = synth(b.sx.saturating_add(nx), b.sy.saturating_add(ny), b.sz);
    push_tri_at_slot(scratch, [a_lo, a_hi, b_lo], style.rgb, 0);
    push_tri_at_slot(scratch, [b_lo, a_hi, b_hi], style.rgb, 0);
}

/// Lower-level [`push_tri`] that pins the OT slot rather than
/// computing it from screen-space depth. Used for fixed-layer
/// overlays (hover highlight, gizmos, …).
fn push_tri_at_slot(
    scratch: &mut PreviewScratch,
    p: [psx_gte::scene::Projected; 3],
    rgb: (u8, u8, u8),
    slot: usize,
) {
    if scratch.used >= TRI_CAP {
        return;
    }
    let idx = scratch.used;
    scratch.tris[idx] = TriFlat::new(
        [(p[0].sx, p[0].sy), (p[1].sx, p[1].sy), (p[2].sx, p[2].sy)],
        rgb.0,
        rgb.1,
        rgb.2,
    );
    scratch.used = idx + 1;
    let packet_ptr: *mut TriFlat = &mut scratch.tris[idx];
    unsafe {
        scratch.ot.insert(
            slot.min(OT_DEPTH - 1),
            packet_ptr.cast::<u32>(),
            TriFlat::WORDS,
        );
    }
}

/// Stamp a GP0(02h) fill-rectangle into `scratch.clear_packet` and
/// link it into the back-most OT slot so it runs first when DMA
/// walks the chain — which is the same pattern PS1 software uses to
/// "clear" the framebuffer at the start of every frame, since the
/// HwRenderer (faithfully) preserves VRAM across frames the way real
/// hardware does.
fn push_clear(scratch: &mut PreviewScratch) {
    // PSX VRAM coords; the editor's HwRenderer renders the same
    // 320×240 sub-rect that the runtime frame-buffer would land in.
    let color_word = 0x0200_0000_u32; // opcode 0x02, RGB = 0 (black)
    let xy_word = 0u32; // top-left at (0, 0)
    let wh_word = ((240u32) << 16) | 320u32; // pack_xy(320, 240)
                                             // word[0] is rewritten by `OrderingTable::insert` with the
                                             // chain tag — leave it at 0 here.
    scratch.clear_packet[1] = color_word;
    scratch.clear_packet[2] = xy_word;
    scratch.clear_packet[3] = wh_word;
    let ptr: *mut u32 = scratch.clear_packet.as_mut_ptr();
    unsafe {
        scratch.ot.insert(OT_DEPTH - 1, ptr, 3);
    }
}

/// Per-face emit: routes to the flat or textured pool based on
/// `shade`, packing UVs only when textured.
fn emit_face_tri(
    scratch: &mut PreviewScratch,
    p: [psx_gte::scene::Projected; 3],
    uvs: [(u8, u8); 3],
    shade: FaceShade,
) {
    match shade {
        FaceShade::Flat(r, g, b) => push_tri(scratch, p, (r, g, b)),
        FaceShade::Textured { slot, tint } => push_tex_tri(scratch, p, uvs, slot, tint),
    }
}

/// Compose a [`TriTextured`] sampling `slot`'s tpage / CLUT, stash
/// it in the static `tex_tris` arena, and chain it into the OT.
///
/// `tint` modulates the texel: PSX hardware computes
/// `output = texel * tint / 0x80`, so `(0x80, 0x80, 0x80)` is a
/// pass-through and `(0xFF, 0x60, 0x40)` saturates a grey texel
/// toward terracotta. The editor uses the material-name keyword
/// colour as the tint so a procedural-grey brick texture still
/// reads as red until real cooked textures land.
fn push_tex_tri(
    scratch: &mut PreviewScratch,
    p: [psx_gte::scene::Projected; 3],
    uvs: [(u8, u8); 3],
    slot: MaterialSlot,
    tint: (u8, u8, u8),
) {
    if scratch.tex_used >= TRI_CAP {
        return;
    }
    let idx = scratch.tex_used;
    scratch.tex_tris[idx] = TriTextured::new(
        [(p[0].sx, p[0].sy), (p[1].sx, p[1].sy), (p[2].sx, p[2].sy)],
        uvs,
        slot.clut_word,
        slot.tpage_word,
        tint,
    );
    let avg_sz = (p[0].sz as u32 + p[1].sz as u32 + p[2].sz as u32) / 3;
    let slot_idx = ((avg_sz as usize) >> 6).clamp(1, OT_DEPTH - 2);
    let packet_ptr: *mut TriTextured = &mut scratch.tex_tris[idx];
    unsafe {
        scratch
            .ot
            .insert(slot_idx, packet_ptr.cast::<u32>(), TriTextured::WORDS);
    }
    scratch.tex_used = idx + 1;
}

/// Compose a [`TriFlat`] from three projected vertices, store it in
/// the next slot of the static `tris` array, and link it into the
/// OT keyed on average screen-space depth.
fn push_tri(scratch: &mut PreviewScratch, p: [psx_gte::scene::Projected; 3], rgb: (u8, u8, u8)) {
    if scratch.used >= TRI_CAP {
        return;
    }
    let idx = scratch.used;
    scratch.tris[idx] = TriFlat::new(
        [(p[0].sx, p[0].sy), (p[1].sx, p[1].sy), (p[2].sx, p[2].sy)],
        rgb.0,
        rgb.1,
        rgb.2,
    );
    scratch.used = idx + 1;
    let avg_sz = (p[0].sz as u32 + p[1].sz as u32 + p[2].sz as u32) / 3;
    // Map sz (Q0, range up to ~32K for our scenes) into the OT
    // depth band. Smaller sz = closer = drawn last, so map to a
    // lower OT slot index. We reserve slot OT_DEPTH-1 for the
    // per-frame fill-rect clear and slot 0 for the hover overlay
    // (drawn last so it tops everything), so geometry rides the
    // 1..OT_DEPTH-1 band exclusively.
    let slot = ((avg_sz as usize) >> 6).clamp(1, OT_DEPTH - 2);
    let packet_ptr: *mut TriFlat = &mut scratch.tris[idx];
    unsafe {
        scratch
            .ot
            .insert(slot, packet_ptr.cast::<u32>(), TriFlat::WORDS);
    }
}

#[cfg(test)]
mod tests {
    use super::{floor_anchored_model_origin, light_face, FaceShade, PreviewLight};
    use psx_engine::WorldVertex;

    fn flat(r: u8, g: u8, b: u8) -> FaceShade {
        FaceShade::Flat(r, g, b)
    }

    fn unpack(shade: FaceShade) -> (u8, u8, u8) {
        match shade {
            FaceShade::Flat(r, g, b) => (r, g, b),
            FaceShade::Textured { tint, .. } => tint,
        }
    }

    #[test]
    fn floor_anchored_model_origin_offsets_by_half_world_height() {
        let origin = floor_anchored_model_origin(WorldVertex::new(10, 0, 20), 1024);
        assert_eq!(origin, WorldVertex::new(10, 512, 20));
    }

    #[test]
    fn floor_anchored_model_origin_ignores_negative_height() {
        let origin = floor_anchored_model_origin(WorldVertex::new(10, 32, 20), -128);
        assert_eq!(origin, WorldVertex::new(10, 32, 20));
    }

    #[test]
    fn light_face_no_lights_ambient_32_is_not_white() {
        // Regression: pre-fix the `ambient * 256` bug saturated
        // every face to 255. With the new convention an unlit
        // face at ambient 32 should render at ~32, not white.
        let base = flat(128, 128, 128);
        let lit = light_face(base, [0, 0, 0], &[], [32, 32, 32]);
        let (r, g, b) = unpack(lit);
        assert!(r < 64 && g < 64 && b < 64, "got ({r}, {g}, {b})");
    }

    #[test]
    fn light_face_ambient_128_is_neutral() {
        // 128 ambient is the neutral PSX-tint value; an unlit
        // 128-base material should land back at exactly 128.
        let lit = light_face(flat(128, 128, 128), [0, 0, 0], &[], [128, 128, 128]);
        assert_eq!(unpack(lit), (128, 128, 128));
    }

    #[test]
    fn light_face_zero_ambient_zero_lights_black() {
        let lit = light_face(flat(255, 255, 255), [0, 0, 0], &[], [0, 0, 0]);
        assert_eq!(unpack(lit), (0, 0, 0));
    }

    #[test]
    fn light_face_point_light_inside_radius_brightens() {
        // White light at the face centre with neutral base
        // should land at saturating-bright since contribution
        // (255 × 256 × 256) / 65536 = 255 dominates ambient.
        let light = PreviewLight {
            position: [0, 0, 0],
            radius: 100,
            // intensity_q8 = 256, color (255, 255, 255).
            weighted_color: [255 * 256, 255 * 256, 255 * 256],
        };
        let lit = light_face(flat(128, 128, 128), [0, 0, 0], &[light], [32, 32, 32]);
        let (r, g, b) = unpack(lit);
        assert!(r > 200 && g > 200 && b > 200, "got ({r}, {g}, {b})");
    }

    #[test]
    fn light_face_point_light_outside_radius_zero() {
        // Place the face well outside the light's radius; the
        // contribution must be exactly zero. Output should
        // match the no-lights case.
        let light = PreviewLight {
            position: [0, 0, 0],
            radius: 100,
            weighted_color: [255 * 256, 255 * 256, 255 * 256],
        };
        let lit = light_face(flat(128, 128, 128), [10000, 0, 0], &[light], [32, 32, 32]);
        let baseline = light_face(flat(128, 128, 128), [10000, 0, 0], &[], [32, 32, 32]);
        assert_eq!(unpack(lit), unpack(baseline));
    }

    #[test]
    fn light_face_two_lights_accumulate_and_clamp() {
        let l = PreviewLight {
            position: [0, 0, 0],
            radius: 100,
            weighted_color: [255 * 256, 255 * 256, 255 * 256],
        };
        let lit = light_face(flat(255, 255, 255), [0, 0, 0], &[l, l], [128, 128, 128]);
        let (r, g, b) = unpack(lit);
        // Even with two saturating lights, output never
        // exceeds 255 per channel.
        assert_eq!((r, g, b), (255, 255, 255));
    }
}
