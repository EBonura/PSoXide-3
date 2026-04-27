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
use psx_gte::math::{Mat3I16, Vec3I16, Vec3I32};
use psx_gte::scene as gte_scene;
use psx_gpu::prim::TriTextured;
use psxed_project::{
    GridDirection, NodeKind, ProjectDocument, ResourceData, ResourceId, WorldGrid,
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
pub fn build_phase1_cmd_log(
    project: &ProjectDocument,
    camera: ViewportCameraState,
    selected: psxed_project::NodeId,
    hovered_face: Option<psxed_ui::FaceRef>,
    selected_face: Option<psxed_ui::FaceRef>,
    paint_target_preview: Option<psxed_ui::PaintTargetPreview>,
    textures: &EditorTextures,
) -> Vec<GpuCmdLogEntry> {
    let Some((grid, target)) = first_room_grid(project) else {
        return Vec::new();
    };

    let mut scratch = SCRATCH.lock().expect("editor preview scratch mutex");
    scratch.used = 0;
    scratch.tex_used = 0;
    scratch.ot.clear();

    push_clear(&mut scratch);
    setup_gte_for_camera(camera, target);
    walk_room(project, grid, textures, &mut scratch);
    walk_entities(project, grid, selected, &mut scratch);
    // Select-tool face outlines. Selected drawn first so its
    // bolder lines sit *under* a possibly-co-located hover — that
    // way switching focus from selected → hover via mouse-move
    // doesn't make the bold outline strobe.
    if let Some(face) = selected_face {
        push_face_outline(grid, face, FACE_OUTLINE_SELECTED, &mut scratch);
    }
    if let Some(face) = hovered_face {
        if Some(face) != selected_face {
            push_face_outline(grid, face, FACE_OUTLINE_HOVER, &mut scratch);
        }
    }
    // Paint preview: ghost outline of the cell or wall the next
    // click would create / replace. Works for cells outside the
    // current grid by reading world-cell coords directly rather
    // than via array indices.
    if let Some(preview) = paint_target_preview {
        push_paint_preview(grid, preview, &mut scratch);
    }

    // SAFETY: `scratch.tris` lives until end of this function (the
    // mutex guard keeps it alive); the OT chain pointers reference
    // packets inside that vec and are stable while the lock is held.
    unsafe { psx_gpu_render::build_cmd_log(&scratch.ot) }
}

/// First Room node in the active scene plus the world-space point we
/// want the orbit camera to look at (centre of the grid at floor
/// height). `None` if no Room exists.
fn first_room_grid(project: &ProjectDocument) -> Option<(&WorldGrid, [i32; 3])> {
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
    let center_x =
        grid.origin[0] * grid.sector_size + (grid.width as i32 * grid.sector_size) / 2;
    let center_z =
        grid.origin[1] * grid.sector_size + (grid.depth as i32 * grid.sector_size) / 2;
    Some((grid, [center_x, 0, center_z]))
}

/// Configure the host-side GTE so subsequent `project_vertex` /
/// `project_triangle` calls produce screen-space coords for the
/// requested orbit camera.
fn setup_gte_for_camera(camera: ViewportCameraState, target: [i32; 3]) {
    // Orbit camera world position: target offset by radius along the
    // unit vector implied by yaw + pitch. Pitch positive = camera
    // raised above the target, looking down.
    let r = camera.radius as i32;
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

    // GTE translation: view × (-camera_pos). Computed manually as i32
    // dot products since we need the full 32-bit range.
    let cam_world = [cam_x, cam_y, cam_z];
    let tr = Vec3I32::new(
        -dot_view_world(view.m[0], cam_world),
        -dot_view_world(view.m[1], cam_world),
        -dot_view_world(view.m[2], cam_world),
    );

    gte_scene::load_rotation(&view);
    gte_scene::load_translation(tr);
    gte_scene::set_screen_offset((SCREEN_CX as i32) << 16, (SCREEN_CY as i32) << 16);
    gte_scene::set_projection_plane(PROJ_H as u16);
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

/// Walk every populated sector and emit triangles for floors,
/// ceilings, and the walls on each cardinal edge. Faces whose
/// material has a texture in the editor cache draw textured;
/// everything else falls back to flat shading.
fn walk_room(
    project: &ProjectDocument,
    grid: &WorldGrid,
    textures: &EditorTextures,
    scratch: &mut PreviewScratch,
) {
    let s = grid.sector_size;
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
                push_horizontal_face(
                    scratch,
                    [x0, x1, z0, z1],
                    floor.heights,
                    face_shade(project, floor.material, FALLBACK_FLOOR, textures),
                    /* flip_winding */ false,
                );
            }
            if let Some(ceiling) = sector.ceiling.as_ref() {
                push_horizontal_face(
                    scratch,
                    [x0, x1, z0, z1],
                    ceiling.heights,
                    face_shade(project, ceiling.material, FALLBACK_CEILING, textures),
                    // Ceiling normal points down; flipping the winding
                    // keeps backface-cullers happy and pins the inside
                    // surface as the visible side once we add culling.
                    /* flip_winding */ true,
                );
            }
            for &(direction, edge) in &[
                (GridDirection::North, WallEdge::North),
                (GridDirection::East, WallEdge::East),
                (GridDirection::South, WallEdge::South),
                (GridDirection::West, WallEdge::West),
            ] {
                for face in sector.walls.get(direction) {
                    push_wall_face(
                        scratch,
                        [x0, x1, z0, z1],
                        edge,
                        face.heights,
                        face_shade(project, face.material, FALLBACK_WALL, textures),
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

/// Project the four corners of a sector-aligned horizontal face and
/// emit two triangles for it. `heights` is `[NW, NE, SE, SW]`.
/// `flip_winding=true` reverses the vertex order for ceilings.
fn push_horizontal_face(
    scratch: &mut PreviewScratch,
    bounds: [i32; 4],
    heights: [i32; 4],
    shade: FaceShade,
    flip_winding: bool,
) {
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
        // Unused but the compiler still needs values; arbitrary.
        max_u = 0;
        max_v = 0;
        uv_nw = (0, 0);
        uv_ne = (0, 0);
        uv_se = (0, 0);
        uv_sw = (0, 0);
    }
    let _ = max_u;
    let _ = max_v;
    if flip_winding {
        emit_face_tri(scratch, [p_nw, p_ne, p_sw], [uv_nw, uv_ne, uv_sw], shade);
        emit_face_tri(scratch, [p_ne, p_se, p_sw], [uv_ne, uv_se, uv_sw], shade);
    } else {
        emit_face_tri(scratch, [p_nw, p_sw, p_ne], [uv_nw, uv_sw, uv_ne], shade);
        emit_face_tri(scratch, [p_ne, p_sw, p_se], [uv_ne, uv_sw, uv_se], shade);
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

/// Build the four world-space corners of a wall face on `edge` and
/// emit two triangles for it. `heights` is the `GridVerticalFace`
/// `[bottom_left, bottom_right, top_right, top_left]` quad.
fn push_wall_face(
    scratch: &mut PreviewScratch,
    bounds: [i32; 4],
    edge: WallEdge,
    heights: [i32; 4],
    shade: FaceShade,
) {
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
    emit_face_tri(scratch, [p_bl, p_br, p_tr], [uv_bl, uv_br, uv_tr], shade);
    emit_face_tri(scratch, [p_bl, p_tr, p_tl], [uv_bl, uv_tr, uv_tl], shade);
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

fn synth(sx: i16, sy: i16, sz: u16) -> psx_gte::scene::Projected {
    psx_gte::scene::Projected { sx, sy, sz }
}

/// Marker colour per node kind, or `None` for nodes that aren't
/// placeable in 3D space (the World macro, the Room itself, plain
/// transform-only nodes).
fn entity_marker_color(kind: &NodeKind) -> Option<(u8, u8, u8)> {
    match kind {
        NodeKind::SpawnPoint { player: true } => Some((0x60, 0xE0, 0x80)),
        NodeKind::SpawnPoint { player: false } => Some((0x60, 0xB8, 0xF0)),
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
/// expects. Cooked PS1 levels live within ±i16 by design (sectors
/// are 1024 units, even a 64-sector room fits).
fn world_to_view(world: [i32; 3]) -> Vec3I16 {
    Vec3I16::new(
        clamp_i16(world[0]),
        clamp_i16(world[1]),
        clamp_i16(world[2]),
    )
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
fn push_cell_ghost_outline(
    grid: &WorldGrid,
    wcx: i32,
    wcz: i32,
    scratch: &mut PreviewScratch,
) {
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
fn wall_xy_for(
    dir: GridDirection,
    x0: i32,
    x1: i32,
    z0: i32,
    z1: i32,
) -> ((i32, i32), (i32, i32)) {
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
        [
            (p[0].sx, p[0].sy),
            (p[1].sx, p[1].sy),
            (p[2].sx, p[2].sy),
        ],
        rgb.0,
        rgb.1,
        rgb.2,
    );
    scratch.used = idx + 1;
    let packet_ptr: *mut TriFlat = &mut scratch.tris[idx];
    unsafe {
        scratch
            .ot
            .insert(slot.min(OT_DEPTH - 1), packet_ptr.cast::<u32>(), TriFlat::WORDS);
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
        [
            (p[0].sx, p[0].sy),
            (p[1].sx, p[1].sy),
            (p[2].sx, p[2].sy),
        ],
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
fn push_tri(
    scratch: &mut PreviewScratch,
    p: [psx_gte::scene::Projected; 3],
    rgb: (u8, u8, u8),
) {
    if scratch.used >= TRI_CAP {
        return;
    }
    let idx = scratch.used;
    scratch.tris[idx] = TriFlat::new(
        [
            (p[0].sx, p[0].sy),
            (p[1].sx, p[1].sy),
            (p[2].sx, p[2].sy),
        ],
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
