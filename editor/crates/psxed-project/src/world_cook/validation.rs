//! Cook-time validation for authored world grids.

use super::*;

pub(super) fn validate_grid_shape(grid: &WorldGrid) -> Result<(), WorldGridCookError> {
    if grid.sector_size != world::SECTOR_SIZE {
        return Err(WorldGridCookError::UnsupportedSectorSize {
            expected: world::SECTOR_SIZE,
            actual: grid.sector_size,
        });
    }
    if grid.width == 0 || grid.depth == 0 {
        return Err(WorldGridCookError::InvalidDimensions {
            width: grid.width,
            depth: grid.depth,
        });
    }
    let expected = grid.width as usize * grid.depth as usize;
    if grid.sectors.len() != expected {
        return Err(WorldGridCookError::SectorStorageLenMismatch {
            expected,
            actual: grid.sectors.len(),
        });
    }
    Ok(())
}

/// Enforce the runtime caps the inspector surfaces. Cooker and
/// inspector now share `WorldGridBudget::over_budget` semantics:
/// what the editor warned about will fail at cook time.
pub(super) fn validate_grid_budget(grid: &WorldGrid) -> Result<(), WorldGridCookError> {
    if grid.width > MAX_ROOM_WIDTH {
        return Err(WorldGridCookError::RoomDimensionExceeded {
            axis: 'X',
            value: grid.width,
            limit: MAX_ROOM_WIDTH,
        });
    }
    if grid.depth > MAX_ROOM_DEPTH {
        return Err(WorldGridCookError::RoomDimensionExceeded {
            axis: 'Z',
            value: grid.depth,
            limit: MAX_ROOM_DEPTH,
        });
    }
    // Per-edge wall stack -- caught at the source rather than
    // after wall records have been laid out, so the error
    // message points at the authored sector.
    for x in 0..grid.width {
        for z in 0..grid.depth {
            let Some(sector) = grid.sector(x, z) else {
                continue;
            };
            for direction in GridDirection::CARDINAL {
                let count = sector.walls.get(direction).len();
                if count > MAX_WALL_STACK {
                    return Err(WorldGridCookError::WallStackExceeded {
                        x,
                        z,
                        direction,
                        count,
                        limit: MAX_WALL_STACK,
                    });
                }
            }
        }
    }
    let budget = grid.budget();
    if budget.triangles > MAX_ROOM_TRIANGLES {
        return Err(WorldGridCookError::RoomTriangleBudgetExceeded {
            triangles: budget.triangles,
            limit: MAX_ROOM_TRIANGLES,
        });
    }
    if budget.psxw_v1_bytes > MAX_ROOM_BYTES {
        return Err(WorldGridCookError::RoomByteBudgetExceeded {
            bytes: budget.psxw_v1_bytes,
            limit: MAX_ROOM_BYTES,
        });
    }
    Ok(())
}

/// Reject duplicate physical walls. The wall between cells
/// `(x, z)` and `(x+1, z)` is the same vertical face whether
/// it's authored as `East(x, z)` or `West(x+1, z)`. Authoring
/// both would double-render and double-collide, so the cooker
/// fails loud and points at both cells. A future "low-index-
/// cell wins" normalization pass is fine but should be an
/// explicit, opted-in cooker step.
pub(super) fn validate_no_duplicate_walls(grid: &WorldGrid) -> Result<(), WorldGridCookError> {
    use std::collections::HashMap;

    // Each entry: which (x, z, dir) was the first to claim a
    // given physical edge. Diagonals don't share with anyone
    // (and the cooker rejects them anyway via cook_walls), so
    // they don't enter the map.
    let mut claims = HashMap::new();
    for x in 0..grid.width {
        for z in 0..grid.depth {
            let Some(sector) = grid.sector(x, z) else {
                continue;
            };
            for direction in GridDirection::CARDINAL {
                if sector.walls.get(direction).is_empty() {
                    continue;
                }
                let Some(edge) = direction.physical_edge(x, z) else {
                    continue;
                };
                if let Some(&(other_x, other_z, other_direction)) = claims.get(&edge) {
                    return Err(WorldGridCookError::DuplicatePhysicalWall {
                        x,
                        z,
                        direction,
                        other_x,
                        other_z,
                        other_direction,
                    });
                }
                claims.insert(edge, (x, z, direction));
            }
        }
    }
    Ok(())
}

/// Reject any authored vertex height that isn't a multiple of
/// [`HEIGHT_QUANTUM`]. The editor snaps every drag through
/// `snap_height`, so values that survive to here come from
/// programmatic writes, hand-edited RON, or projects authored
/// before the quantum landed. Catching them at cook time keeps
/// the runtime free of jitter the snap was meant to remove.
pub(super) fn validate_quantized_heights(grid: &WorldGrid) -> Result<(), WorldGridCookError> {
    for x in 0..grid.width {
        for z in 0..grid.depth {
            let Some(sector) = grid.sector(x, z) else {
                continue;
            };
            if let Some(face) = &sector.floor {
                check_face_heights(x, z, WorldGridFaceKind::Floor, &face.heights)?;
            }
            if let Some(face) = &sector.ceiling {
                check_face_heights(x, z, WorldGridFaceKind::Ceiling, &face.heights)?;
            }
            for direction in GridDirection::CARDINAL {
                for wall in sector.walls.get(direction) {
                    check_face_heights(x, z, WorldGridFaceKind::Wall(direction), &wall.heights)?;
                }
            }
        }
    }
    Ok(())
}

fn check_face_heights(
    x: u16,
    z: u16,
    face: WorldGridFaceKind,
    heights: &[i32; 4],
) -> Result<(), WorldGridCookError> {
    for &h in heights {
        if h % HEIGHT_QUANTUM != 0 {
            return Err(WorldGridCookError::HeightNotQuantized {
                x,
                z,
                face,
                value: h,
                quantum: HEIGHT_QUANTUM,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_wall_heights(
    wall: &GridVerticalFace,
    x: u16,
    z: u16,
    direction: GridDirection,
) -> Result<(), WorldGridCookError> {
    let left_valid = wall.heights[world::WALL_TOP_LEFT] >= wall.heights[world::WALL_BOTTOM_LEFT];
    let right_valid = wall.heights[world::WALL_TOP_RIGHT] >= wall.heights[world::WALL_BOTTOM_RIGHT];
    if left_valid && right_valid {
        Ok(())
    } else {
        Err(WorldGridCookError::InvalidWallHeights {
            x,
            z,
            direction,
            heights: wall.heights,
        })
    }
}
