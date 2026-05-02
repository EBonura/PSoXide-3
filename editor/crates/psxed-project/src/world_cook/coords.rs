//! Editor-to-runtime coordinate conversion helpers.

use super::*;

/// Editor-authored horizontal corners use the preview convention
/// `NW=(x0,z1), NE=(x1,z1), SE=(x1,z0), SW=(x0,z0)`. The compact
/// `.psxw` runtime format is array-rooted with `NW=(x0,z0)`.
/// Flip the Z axis at cook time so runtime rendering/collision match
/// the editor's 3D preview.
pub(super) const fn runtime_horizontal_heights(heights: [i32; 4]) -> [i32; 4] {
    [
        heights[world::CORNER_SW],
        heights[world::CORNER_SE],
        heights[world::CORNER_NE],
        heights[world::CORNER_NW],
    ]
}

pub(super) const fn runtime_horizontal_uvs(uvs: [(u8, u8); 4]) -> [(u8, u8); 4] {
    [
        uvs[world::CORNER_SW],
        uvs[world::CORNER_SE],
        uvs[world::CORNER_NE],
        uvs[world::CORNER_NW],
    ]
}

pub(super) const fn runtime_horizontal_split(split: GridSplit) -> GridSplit {
    match split {
        GridSplit::NorthWestSouthEast => GridSplit::NorthEastSouthWest,
        GridSplit::NorthEastSouthWest => GridSplit::NorthWestSouthEast,
    }
}

/// Editor cardinal walls follow the same preview convention as
/// horizontal faces: North is the +Z edge and South is the -Z edge.
/// The `.psxw` runtime format uses North=-Z / South=+Z, so swap the
/// Z-facing directions while preserving East/West.
pub(super) const fn runtime_wall_direction(direction: GridDirection) -> GridDirection {
    match direction {
        GridDirection::North => GridDirection::South,
        GridDirection::South => GridDirection::North,
        other => other,
    }
}

/// Convert editor wall corner order to the runtime wall order for the
/// same physical face. In editor preview terms every cardinal wall's
/// bottom edge runs opposite the runtime format, so BL/BR and TL/TR
/// swap as pairs.
pub(super) const fn runtime_wall_heights(heights: [i32; 4]) -> [i32; 4] {
    [
        heights[world::WALL_BOTTOM_RIGHT],
        heights[world::WALL_BOTTOM_LEFT],
        heights[world::WALL_TOP_LEFT],
        heights[world::WALL_TOP_RIGHT],
    ]
}

pub(super) const fn runtime_wall_uvs(uvs: [(u8, u8); 4]) -> [(u8, u8); 4] {
    [
        uvs[world::WALL_BOTTOM_RIGHT],
        uvs[world::WALL_BOTTOM_LEFT],
        uvs[world::WALL_TOP_LEFT],
        uvs[world::WALL_TOP_RIGHT],
    ]
}
