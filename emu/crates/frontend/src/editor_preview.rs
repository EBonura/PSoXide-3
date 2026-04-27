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
use psxed_project::{NodeKind, ProjectDocument, WorldGrid};
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
    used: usize,
}

const EMPTY_TRI: TriFlat = TriFlat::new([(0, 0), (0, 0), (0, 0)], 0, 0, 0);

static SCRATCH: Mutex<PreviewScratch> = Mutex::new(PreviewScratch {
    ot: OrderingTable::new(),
    tris: [EMPTY_TRI; TRI_CAP],
    used: 0,
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
) -> Vec<GpuCmdLogEntry> {
    let Some((grid, target)) = first_room_grid(project) else {
        return Vec::new();
    };

    let mut scratch = SCRATCH.lock().expect("editor preview scratch mutex");
    scratch.used = 0;
    scratch.ot.clear();

    setup_gte_for_camera(camera, target);

    walk_floors(grid, target, &mut scratch);

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
    let center_x = (grid.width as i32 * grid.sector_size) / 2;
    let center_z = (grid.depth as i32 * grid.sector_size) / 2;
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

/// Walk every populated sector and emit two flat triangles per floor.
fn walk_floors(grid: &WorldGrid, _target: [i32; 3], scratch: &mut PreviewScratch) {
    let s = grid.sector_size;
    for x in 0..grid.width {
        for z in 0..grid.depth {
            let Some(sector) = grid.sector(x, z) else {
                continue;
            };
            let Some(floor) = sector.floor.as_ref() else {
                continue;
            };
            // Corner heights: [NW, NE, SE, SW] in `GridHorizontalFace`.
            // World coords with +X east, +Z south, +Y up.
            let x0 = (x as i32) * s;
            let x1 = ((x as i32) + 1) * s;
            let z0 = (z as i32) * s;
            let z1 = ((z as i32) + 1) * s;
            let nw = world_to_view([x0, floor.heights[0], z1]);
            let ne = world_to_view([x1, floor.heights[1], z1]);
            let se = world_to_view([x1, floor.heights[2], z0]);
            let sw = world_to_view([x0, floor.heights[3], z0]);
            // Each `project_vertex` returns sx/sy/sz directly out of
            // the GTE; we need depth for the OT key and screen coords
            // for the prim packet.
            let p_nw = gte_scene::project_vertex(nw);
            let p_ne = gte_scene::project_vertex(ne);
            let p_se = gte_scene::project_vertex(se);
            let p_sw = gte_scene::project_vertex(sw);

            // Two-triangle floor split along the NW-SE diagonal.
            let (r, g, b) = floor_color_for(x, z);
            push_tri(scratch, [p_nw, p_sw, p_ne], (r, g, b));
            push_tri(scratch, [p_ne, p_sw, p_se], (r, g, b));

            if scratch.used >= TRI_CAP {
                return;
            }
        }
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

/// Lay each tile out in a checkerboard so the grid is visible even
/// before per-sector materials are surfaced. Sector materials feed in
/// during a later phase; for Phase 1 we just want to *see* the
/// authored shape clearly.
fn floor_color_for(x: u16, z: u16) -> (u8, u8, u8) {
    if (x + z) % 2 == 0 {
        (0xC0, 0xB0, 0x90)
    } else {
        (0x90, 0x80, 0x70)
    }
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
    // lower OT slot index.
    let slot = ((avg_sz as usize) >> 6).min(OT_DEPTH - 1);
    let packet_ptr: *mut TriFlat = &mut scratch.tris[idx];
    unsafe {
        scratch
            .ot
            .insert(slot, packet_ptr.cast::<u32>(), TriFlat::WORDS);
    }
}
