//! Shared editor/runtime spatial conversion helpers.
//!
//! The editor has two room-space targets that are intentionally
//! different:
//!
//! - **Preview world** keeps authored cells at their physical
//!   editor-world coordinates. Room `origin` is part of the result,
//!   so growing a room toward negative X/Z does not visually move
//!   existing geometry, lights, models, or selection bounds.
//! - **Cooked room local** matches the compact `.psxw` layout. Room
//!   geometry is array-rooted at `(0, 0)`, so entity records use the
//!   current array centre and ignore `origin`; `origin` is emitted in
//!   the manifest only as editor metadata.
//!
//! Keeping both conversions named here is deliberate: call sites
//! should choose the space they need rather than re-derive a formula.

use crate::{GridCellBounds, GridDirection, Transform3, WorldGrid};

/// Integer room-space position `[x, y, z]`.
pub type RoomPoint = [i32; 3];

/// Floating room-space position `[x, y, z]`.
pub type RoomPointF = [f32; 3];

/// Origin of a node in editor preview world space.
///
/// This is origin-aware and should be used by editor 3D rendering,
/// picking/bounds, gizmos, and any other authoring-surface feature
/// that must line up with what the user sees.
pub fn node_preview_origin(grid: &WorldGrid, transform: &Transform3) -> RoomPoint {
    let xz = grid.editor_to_room_local([transform.translation[0], transform.translation[2]]);
    [
        xz[0] as i32,
        (transform.translation[1] * grid.sector_size as f32) as i32,
        xz[2] as i32,
    ]
}

/// Floating-point form of [`node_preview_origin`].
pub fn node_preview_origin_f32(grid: &WorldGrid, transform: &Transform3) -> RoomPointF {
    let xz = grid.editor_to_room_local([transform.translation[0], transform.translation[2]]);
    [
        xz[0],
        transform.translation[1] * grid.sector_size as f32,
        xz[2],
    ]
}

/// Centre of a selectable node bound in editor preview world space.
pub fn node_preview_bounds_center(
    grid: &WorldGrid,
    transform: &Transform3,
    half_extents: [f32; 3],
) -> RoomPointF {
    let origin = node_preview_origin_f32(grid, transform);
    [origin[0], origin[1] + half_extents[1], origin[2]]
}

/// Origin of a node in cooked `.psxw` room-local space.
///
/// This intentionally ignores [`WorldGrid::origin`]. The cooked room
/// geometry is array-rooted, so a node authored at editor `(0, 0)`
/// belongs at the centre of the current array.
pub fn node_cooked_room_local_origin(grid: &WorldGrid, transform: &Transform3) -> RoomPoint {
    let s = grid.sector_size as f32;
    [
        ((transform.translation[0] + grid.width as f32 * 0.5) * s) as i32,
        (transform.translation[1] * s) as i32,
        ((transform.translation[2] + grid.depth as f32 * 0.5) * s) as i32,
    ]
}

/// Geometric centre of a room in editor preview world space.
pub fn room_preview_center(grid: &WorldGrid) -> RoomPoint {
    let center = room_preview_center_f32(grid);
    [center[0] as i32, 0, center[2] as i32]
}

/// Floating-point geometric centre of a room in editor preview world
/// space.
pub fn room_preview_center_f32(grid: &WorldGrid) -> RoomPointF {
    grid.editor_to_room_local([0.0, 0.0])
}

/// Convert an authored light radius, expressed in sectors, to engine
/// world units.
pub fn light_radius_engine_units(grid: &WorldGrid, radius_sectors: f32) -> i32 {
    (radius_sectors * grid.sector_size as f32) as i32
}

/// Convert an authored light radius to the manifest wire format.
pub fn light_radius_record_units(grid: &WorldGrid, radius_sectors: f32) -> u16 {
    (radius_sectors * grid.sector_size as f32).clamp(1.0, u16::MAX as f32) as u16
}

/// Pick the editor cardinal wall edge from a point's offset relative
/// to a cell centre. Editor convention: North = +Z, South = -Z.
pub fn editor_wall_direction_from_offset(dx: f32, dz: f32) -> GridDirection {
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

/// Inward-facing X/Z normal for an editor cardinal wall edge.
pub const fn editor_wall_inward_normal(direction: GridDirection) -> Option<[i32; 2]> {
    match direction {
        GridDirection::North => Some([0, -1]),
        GridDirection::East => Some([-1, 0]),
        GridDirection::South => Some([0, 1]),
        GridDirection::West => Some([1, 0]),
        GridDirection::NorthWestSouthEast | GridDirection::NorthEastSouthWest => None,
    }
}

/// Build cell bounds from a world-cell coordinate. Used for off-grid
/// paint ghosts before the grid has been grown to contain the cell.
pub const fn cell_bounds_from_world_cell(
    world_cell_x: i32,
    world_cell_z: i32,
    sector_size: i32,
) -> GridCellBounds {
    let x0 = world_cell_x * sector_size;
    let z0 = world_cell_z * sector_size;
    GridCellBounds {
        x0,
        x1: x0 + sector_size,
        z0,
        z1: z0 + sector_size,
    }
}

/// Wall outline corners in editor preview world space, optionally
/// inset by `lift` along the wall's inward normal.
pub fn editor_wall_outline_corners(
    bounds: GridCellBounds,
    direction: GridDirection,
    heights: [i32; 4],
    lift: i32,
) -> Option<[RoomPoint; 4]> {
    let (bl, br) = bounds.wall_endpoints_xz(direction)?;
    let [nx, nz] = editor_wall_inward_normal(direction)?;
    Some([
        [bl[0] + lift * nx, heights[0], bl[1] + lift * nz],
        [br[0] + lift * nx, heights[1], br[1] + lift * nz],
        [br[0] + lift * nx, heights[2], br[1] + lift * nz],
        [bl[0] + lift * nx, heights[3], bl[1] + lift * nz],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_origin_accounts_for_negative_room_origin() {
        let mut grid = WorldGrid::stone_room(4, 7, 1024, None, None);
        grid.origin = [-1, -3];
        let transform = Transform3 {
            translation: [0.5, 0.25, -1.0],
            ..Transform3::default()
        };

        let origin = node_preview_origin(&grid, &transform);
        let expected_xz = grid.editor_to_room_local([0.5, -1.0]);
        assert_eq!(origin, [expected_xz[0] as i32, 256, expected_xz[2] as i32]);
        assert_ne!(
            origin,
            [
                ((transform.translation[0] + grid.width as f32 * 0.5) * 1024.0) as i32,
                256,
                ((transform.translation[2] + grid.depth as f32 * 0.5) * 1024.0) as i32,
            ]
        );
    }

    #[test]
    fn cooked_room_local_origin_is_array_rooted() {
        let mut grid = WorldGrid::stone_room(4, 7, 1024, None, None);
        grid.origin = [-1, -3];
        let transform = Transform3 {
            translation: [0.5, 0.25, -1.0],
            ..Transform3::default()
        };

        assert_eq!(
            node_cooked_room_local_origin(&grid, &transform),
            [2560, 256, 2560]
        );
    }

    #[test]
    fn wall_direction_and_outline_share_editor_convention() {
        assert_eq!(
            editor_wall_direction_from_offset(0.1, 0.9),
            GridDirection::North
        );
        assert_eq!(
            editor_wall_direction_from_offset(0.9, 0.9),
            GridDirection::East
        );

        let bounds = GridCellBounds {
            x0: 0,
            x1: 1024,
            z0: 0,
            z1: 1024,
        };
        assert_eq!(
            editor_wall_outline_corners(bounds, GridDirection::North, [0, 0, 1024, 1024], 4),
            Some([
                [0, 0, 1020],
                [1024, 0, 1020],
                [1024, 1024, 1020],
                [0, 1024, 1020]
            ])
        );
    }
}
