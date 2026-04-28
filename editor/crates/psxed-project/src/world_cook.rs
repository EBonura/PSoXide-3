//! Validation and cooking for authored grid worlds.
//!
//! The output here is still an owned host-side manifest. A later writer can
//! turn it into a compact `.psxw` blob or static Rust arrays for
//! `psx_engine::GridWorld`.

use std::collections::HashMap;

use psxed_format::world;

use crate::{
    GridDirection, GridHorizontalFace, GridSector, GridSplit, GridVerticalFace, GridWalls,
    MaterialResource, ProjectDocument, PsxBlendMode, ResourceData, ResourceId, WorldGrid,
    HEIGHT_QUANTUM, MAX_ROOM_BYTES, MAX_ROOM_DEPTH, MAX_ROOM_TRIANGLES, MAX_ROOM_WIDTH,
    MAX_WALL_STACK,
};

/// Authored grid face referenced by a cooker diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldGridFaceKind {
    /// Sector floor face.
    Floor,
    /// Sector ceiling face.
    Ceiling,
    /// Sector wall in the given direction.
    Wall(GridDirection),
}

impl std::fmt::Display for WorldGridFaceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Floor => f.write_str("floor"),
            Self::Ceiling => f.write_str("ceiling"),
            Self::Wall(direction) => write!(f, "{direction:?} wall"),
        }
    }
}

/// Errors raised while validating and cooking an authored grid world.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldGridCookError {
    /// Runtime currently supports only the canonical PS1 sector scale.
    UnsupportedSectorSize {
        /// Required sector size.
        expected: i32,
        /// Authored sector size.
        actual: i32,
    },
    /// A grid must have non-zero dimensions.
    InvalidDimensions {
        /// Authored width.
        width: u16,
        /// Authored depth.
        depth: u16,
    },
    /// Flat sector storage does not match `width * depth`.
    SectorStorageLenMismatch {
        /// Expected sector count.
        expected: usize,
        /// Actual sector count.
        actual: usize,
    },
    /// A renderable face has no assigned material.
    UnassignedMaterial {
        /// Sector X coordinate.
        x: u16,
        /// Sector Z coordinate.
        z: u16,
        /// Face that needs a material.
        face: WorldGridFaceKind,
    },
    /// A face references a missing project resource.
    MissingMaterial {
        /// Missing resource id.
        id: ResourceId,
    },
    /// A face references a resource that is not a material.
    ResourceIsNotMaterial {
        /// Resource id.
        id: ResourceId,
    },
    /// Material slot ids must fit in the runtime `u16` slot type.
    TooManyMaterials {
        /// Number of slots requested.
        count: usize,
    },
    /// Sector records must fit in the world format's `u16` count.
    TooManySectors {
        /// Number of sectors requested.
        count: usize,
    },
    /// Wall records must fit in the world format's `u16` count.
    TooManyWalls {
        /// Number of walls requested.
        count: usize,
    },
    /// Encoded payload length must fit in the common asset header.
    EncodedWorldTooLarge {
        /// Encoded payload length.
        bytes: usize,
    },
    /// A wall has one or both top corners below its bottom corners.
    InvalidWallHeights {
        /// Sector X coordinate.
        x: u16,
        /// Sector Z coordinate.
        z: u16,
        /// Wall direction.
        direction: GridDirection,
        /// Authored wall heights.
        heights: [i32; 4],
    },
    /// Diagonal walls (`NorthWestSouthEast` / `NorthEastSouthWest`)
    /// aren't supported by the v1 cooker / runtime. The data model
    /// has the slots so authoring can land later, but render +
    /// picking + collision aren't consistent yet — better to fail
    /// loud than ship half-working diagonals.
    UnsupportedDiagonalWall {
        /// Sector X coordinate.
        x: u16,
        /// Sector Z coordinate.
        z: u16,
        /// Diagonal direction.
        direction: GridDirection,
    },
    /// Grid width / depth past the runtime cap. Caught at cook
    /// time so the editor's inspector warning lines up with a
    /// hard failure when the user actually ships.
    RoomDimensionExceeded {
        /// `'X'` or `'Z'`.
        axis: char,
        /// Authored size.
        value: u16,
        /// Cap.
        limit: u16,
    },
    /// Triangle count past `MAX_ROOM_TRIANGLES`.
    RoomTriangleBudgetExceeded {
        /// Triangles the cooker would emit.
        triangles: usize,
        /// Cap.
        limit: usize,
    },
    /// Cooked `.psxw` payload past `MAX_ROOM_BYTES`.
    RoomByteBudgetExceeded {
        /// Estimated wire size.
        bytes: usize,
        /// Cap.
        limit: usize,
    },
    /// Wall stack on a single edge past `MAX_WALL_STACK`.
    WallStackExceeded {
        /// Sector X coordinate.
        x: u16,
        /// Sector Z coordinate.
        z: u16,
        /// Edge.
        direction: GridDirection,
        /// Authored wall count on this edge.
        count: usize,
        /// Cap.
        limit: usize,
    },
    /// Two cells claim the same physical wall — `East(x, z)`
    /// and `West(x+1, z)` describe the same vertical face, and
    /// authoring both would double-render and double-collide.
    /// The cooker rejects rather than silently dedup-pick: a
    /// future "low-index-cell wins" rule is fine but should be
    /// an explicit cooker pass, not implicit.
    DuplicatePhysicalWall {
        /// First wall's sector.
        x: u16,
        /// First wall's sector.
        z: u16,
        /// First wall's direction.
        direction: GridDirection,
        /// Second wall's sector.
        other_x: u16,
        /// Second wall's sector.
        other_z: u16,
        /// Second wall's direction.
        other_direction: GridDirection,
    },
    /// A vertex height is not aligned to [`HEIGHT_QUANTUM`]. The
    /// editor snaps every drag through `snap_height`; values
    /// that slip through come from programmatic writes, hand-
    /// edited RON, or a legacy project predating the quantum.
    HeightNotQuantized {
        /// Sector X coordinate.
        x: u16,
        /// Sector Z coordinate.
        z: u16,
        /// Which face the height belongs to.
        face: WorldGridFaceKind,
        /// Offending height.
        value: i32,
        /// Required step.
        quantum: i32,
    },
    /// A face has a `dropped_corner` set — the editor models it
    /// as a triangle, but the v1 `.psxw` wire format only carries
    /// quad faces. Render / pick / collision in the editor live
    /// preview already honour the drop, but the cooked runtime
    /// payload doesn't have a slot for it yet.
    TriangleFaceNotSupported {
        /// Sector X coordinate.
        x: u16,
        /// Sector Z coordinate.
        z: u16,
        /// Which face is triangulated.
        face: WorldGridFaceKind,
    },
}

impl std::fmt::Display for WorldGridCookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSectorSize { expected, actual } => {
                write!(f, "unsupported sector size {actual}; expected {expected}")
            }
            Self::InvalidDimensions { width, depth } => {
                write!(f, "invalid grid dimensions {width} x {depth}")
            }
            Self::SectorStorageLenMismatch { expected, actual } => {
                write!(f, "grid stores {actual} sectors; expected {expected}")
            }
            Self::UnassignedMaterial { x, z, face } => {
                write!(f, "sector {x},{z} {face} has no material")
            }
            Self::MissingMaterial { id } => write!(f, "missing material resource #{}", id.raw()),
            Self::ResourceIsNotMaterial { id } => {
                write!(f, "resource #{} is not a material", id.raw())
            }
            Self::TooManyMaterials { count } => {
                write!(
                    f,
                    "world uses {count} material slots; maximum is {}",
                    u16::MAX
                )
            }
            Self::TooManySectors { count } => {
                write!(
                    f,
                    "world uses {count} sector records; maximum is {}",
                    u16::MAX
                )
            }
            Self::TooManyWalls { count } => {
                write!(
                    f,
                    "world uses {count} wall records; maximum is {}",
                    u16::MAX
                )
            }
            Self::EncodedWorldTooLarge { bytes } => {
                write!(f, "encoded world payload is too large: {bytes} bytes")
            }
            Self::InvalidWallHeights {
                x,
                z,
                direction,
                heights,
            } => write!(
                f,
                "sector {x},{z} {direction:?} wall has invalid heights {heights:?}"
            ),
            Self::UnsupportedDiagonalWall { x, z, direction } => write!(
                f,
                "sector {x},{z} has a {direction:?} diagonal wall — diagonals aren't supported by the v1 cooker / runtime yet"
            ),
            Self::RoomDimensionExceeded { axis, value, limit } => write!(
                f,
                "room {axis} = {value} exceeds the runtime cap of {limit} sectors"
            ),
            Self::RoomTriangleBudgetExceeded { triangles, limit } => write!(
                f,
                "room would emit {triangles} triangles; cap is {limit}"
            ),
            Self::RoomByteBudgetExceeded { bytes, limit } => write!(
                f,
                "cooked room is {bytes} bytes; cap is {limit}"
            ),
            Self::WallStackExceeded {
                x,
                z,
                direction,
                count,
                limit,
            } => write!(
                f,
                "sector {x},{z} {direction:?} edge has {count} walls; cap is {limit}"
            ),
            Self::DuplicatePhysicalWall {
                x,
                z,
                direction,
                other_x,
                other_z,
                other_direction,
            } => write!(
                f,
                "physical wall claimed twice: ({x},{z}) {direction:?} and ({other_x},{other_z}) {other_direction:?}"
            ),
            Self::HeightNotQuantized {
                x,
                z,
                face,
                value,
                quantum,
            } => write!(
                f,
                "sector {x},{z} {face} has height {value} not aligned to {quantum}-unit quantum"
            ),
            Self::TriangleFaceNotSupported { x, z, face } => write!(
                f,
                "sector {x},{z} {face} is a triangle (dropped corner); v1 .psxw only carries quad faces"
            ),
        }
    }
}

impl std::error::Error for WorldGridCookError {}

/// One material slot used by a cooked world grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CookedWorldMaterial {
    /// Runtime slot id.
    pub slot: u16,
    /// Source editor resource id.
    pub source: ResourceId,
    /// Source texture resource, if any.
    pub texture: Option<ResourceId>,
    /// PS1 semi-transparency mode.
    pub blend_mode: PsxBlendMode,
    /// Texture modulation tint.
    pub tint: [u8; 3],
    /// Whether the cooker should emit both windings.
    pub double_sided: bool,
}

impl CookedWorldMaterial {
    fn from_resource(slot: u16, source: ResourceId, material: &MaterialResource) -> Self {
        Self {
            slot,
            source,
            texture: material.texture,
            blend_mode: material.blend_mode,
            tint: material.tint,
            double_sided: material.double_sided,
        }
    }
}

/// Cooked horizontal face with resolved runtime material slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CookedGridHorizontalFace {
    /// Corner heights `[NW, NE, SE, SW]`.
    pub heights: [i32; 4],
    /// Diagonal split.
    pub split: GridSplit,
    /// Runtime material slot.
    pub material: u16,
    /// Whether collision treats this face as walkable.
    pub walkable: bool,
}

/// Cooked vertical wall with resolved runtime material slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CookedGridVerticalFace {
    /// Corner heights `[bottom-left, bottom-right, top-right, top-left]`.
    pub heights: [i32; 4],
    /// Runtime material slot.
    pub material: u16,
    /// Whether collision treats this wall as blocking.
    pub solid: bool,
}

/// Cooked wall lists for one sector.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CookedGridWalls {
    /// Walls on the north edge.
    pub north: Vec<CookedGridVerticalFace>,
    /// Walls on the east edge.
    pub east: Vec<CookedGridVerticalFace>,
    /// Walls on the south edge.
    pub south: Vec<CookedGridVerticalFace>,
    /// Walls on the west edge.
    pub west: Vec<CookedGridVerticalFace>,
    /// Diagonal NW-SE walls.
    pub north_west_south_east: Vec<CookedGridVerticalFace>,
    /// Diagonal NE-SW walls.
    pub north_east_south_west: Vec<CookedGridVerticalFace>,
}

impl CookedGridWalls {
    fn get_mut(&mut self, direction: GridDirection) -> &mut Vec<CookedGridVerticalFace> {
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

/// One cooked sector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CookedGridSector {
    /// Optional floor face.
    pub floor: Option<CookedGridHorizontalFace>,
    /// Optional ceiling face.
    pub ceiling: Option<CookedGridHorizontalFace>,
    /// Sector wall faces.
    pub walls: CookedGridWalls,
}

/// Cooked owned grid-world manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CookedWorldGrid {
    /// Width in sectors.
    pub width: u16,
    /// Depth in sectors.
    pub depth: u16,
    /// Engine units per sector.
    pub sector_size: i32,
    /// Flat `[x * depth + z]` sector storage.
    pub sectors: Vec<Option<CookedGridSector>>,
    /// Material slots referenced by cooked faces.
    pub materials: Vec<CookedWorldMaterial>,
    /// Room ambient color.
    pub ambient_color: [u8; 3],
    /// Whether PS1 depth cue/fog should be enabled.
    pub fog_enabled: bool,
}

impl CookedWorldGrid {
    /// Number of sectors that contain cooked geometry.
    pub fn populated_sector_count(&self) -> usize {
        self.sectors.iter().flatten().count()
    }

    /// Encode this cooked grid into the `.psxw` byte layout.
    pub fn to_psxw_bytes(&self) -> Result<Vec<u8>, WorldGridCookError> {
        encode_cooked_world_grid_psxw(self)
    }
}

/// Validate and cook one editor-authored world grid.
pub fn cook_world_grid(
    project: &ProjectDocument,
    grid: &WorldGrid,
) -> Result<CookedWorldGrid, WorldGridCookError> {
    validate_grid_shape(grid)?;
    validate_grid_budget(grid)?;
    validate_no_duplicate_walls(grid)?;
    validate_quantized_heights(grid)?;

    let mut materials = Vec::new();
    let mut material_slots = HashMap::new();
    let mut sectors = Vec::with_capacity(grid.sectors.len());

    for x in 0..grid.width {
        for z in 0..grid.depth {
            let sector = grid
                .sector(x, z)
                .map(|sector| {
                    cook_sector(project, sector, x, z, &mut materials, &mut material_slots)
                })
                .transpose()?
                .flatten();
            sectors.push(sector);
        }
    }

    Ok(CookedWorldGrid {
        width: grid.width,
        depth: grid.depth,
        sector_size: grid.sector_size,
        sectors,
        materials,
        ambient_color: grid.ambient_color,
        fog_enabled: grid.fog_enabled,
    })
}

/// Validate, cook, and encode one editor-authored grid as `.psxw` bytes.
pub fn encode_world_grid_psxw(
    project: &ProjectDocument,
    grid: &WorldGrid,
) -> Result<Vec<u8>, WorldGridCookError> {
    cook_world_grid(project, grid)?.to_psxw_bytes()
}

fn validate_grid_shape(grid: &WorldGrid) -> Result<(), WorldGridCookError> {
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
fn validate_grid_budget(grid: &WorldGrid) -> Result<(), WorldGridCookError> {
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
    // Per-edge wall stack — caught at the source rather than
    // after wall records have been laid out, so the error
    // message points at the authored sector.
    for x in 0..grid.width {
        for z in 0..grid.depth {
            let Some(sector) = grid.sector(x, z) else {
                continue;
            };
            for direction in [
                GridDirection::North,
                GridDirection::East,
                GridDirection::South,
                GridDirection::West,
            ] {
                let count = sector.walls.get(direction).len();
                if count > MAX_WALL_STACK as usize {
                    return Err(WorldGridCookError::WallStackExceeded {
                        x,
                        z,
                        direction,
                        count,
                        limit: MAX_WALL_STACK as usize,
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
fn validate_no_duplicate_walls(grid: &WorldGrid) -> Result<(), WorldGridCookError> {
    use std::collections::HashMap;

    // Each entry: which (x, z, dir) was the first to claim a
    // given physical edge. Diagonals don't share with anyone
    // (and the cooker rejects them anyway via cook_walls), so
    // they don't enter the map.
    let mut claims: HashMap<PhysicalEdge, (u16, u16, GridDirection)> = HashMap::new();
    for x in 0..grid.width {
        for z in 0..grid.depth {
            let Some(sector) = grid.sector(x, z) else {
                continue;
            };
            for direction in [
                GridDirection::North,
                GridDirection::East,
                GridDirection::South,
                GridDirection::West,
            ] {
                if sector.walls.get(direction).is_empty() {
                    continue;
                }
                let Some(edge) = canonical_edge(x, z, direction) else {
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

/// The vertical face between two cells, addressed canonically
/// so opposing-cell descriptions collide on insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PhysicalEdge {
    x: i32,
    z: i32,
    axis: EdgeAxis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EdgeAxis {
    /// Edge runs north-south (separates two cells across the X
    /// axis): West / East walls land here.
    NorthSouth,
    /// Edge runs east-west (separates across Z): North / South
    /// walls land here.
    EastWest,
}

fn canonical_edge(x: u16, z: u16, dir: GridDirection) -> Option<PhysicalEdge> {
    match dir {
        GridDirection::North => Some(PhysicalEdge {
            x: x as i32,
            z: z as i32,
            axis: EdgeAxis::EastWest,
        }),
        GridDirection::South => Some(PhysicalEdge {
            x: x as i32,
            z: z as i32 + 1,
            axis: EdgeAxis::EastWest,
        }),
        GridDirection::West => Some(PhysicalEdge {
            x: x as i32,
            z: z as i32,
            axis: EdgeAxis::NorthSouth,
        }),
        GridDirection::East => Some(PhysicalEdge {
            x: x as i32 + 1,
            z: z as i32,
            axis: EdgeAxis::NorthSouth,
        }),
        GridDirection::NorthWestSouthEast | GridDirection::NorthEastSouthWest => None,
    }
}

/// Reject any authored vertex height that isn't a multiple of
/// [`HEIGHT_QUANTUM`]. The editor snaps every drag through
/// `snap_height`, so values that survive to here come from
/// programmatic writes, hand-edited RON, or projects authored
/// before the quantum landed. Catching them at cook time keeps
/// the runtime free of jitter the snap was meant to remove.
fn validate_quantized_heights(grid: &WorldGrid) -> Result<(), WorldGridCookError> {
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
            for direction in [
                GridDirection::North,
                GridDirection::East,
                GridDirection::South,
                GridDirection::West,
            ] {
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

fn cook_sector(
    project: &ProjectDocument,
    sector: &GridSector,
    x: u16,
    z: u16,
    materials: &mut Vec<CookedWorldMaterial>,
    material_slots: &mut HashMap<ResourceId, u16>,
) -> Result<Option<CookedGridSector>, WorldGridCookError> {
    if !sector.has_geometry() {
        return Ok(None);
    }

    // The cooker writes one record per face today, but the
    // fields in that record fall into two concerns the runtime
    // already separates via `psx_engine::RoomRender` /
    // `psx_engine::RoomCollision`:
    //
    //   render    — heights, splits, material slot
    //   collision — heights, splits, walkable / solid bits
    //
    // Heights and splits feed both views. Materials are render-
    // only (no collision branches on tpage). `walkable` /
    // `solid` are collision-only (no draw call branches on
    // walkability). When v2 lands and the byte format splits
    // render and collision into distinct tables, this function
    // peels into `cook_sector_render` + `cook_sector_collision`
    // — the call structure is already grouped that way below.

    // -------- render-relevant cook --------
    let floor = sector
        .floor
        .as_ref()
        .map(|face| {
            cook_horizontal_face(
                project,
                face,
                x,
                z,
                WorldGridFaceKind::Floor,
                materials,
                material_slots,
            )
        })
        .transpose()?;
    let ceiling = sector
        .ceiling
        .as_ref()
        .map(|face| {
            cook_horizontal_face(
                project,
                face,
                x,
                z,
                WorldGridFaceKind::Ceiling,
                materials,
                material_slots,
            )
        })
        .transpose()?;
    let walls = cook_walls(project, &sector.walls, x, z, materials, material_slots)?;

    // -------- collision-relevant cook --------
    // `walkable` / `solid` are forwarded through the cooked
    // structs by `cook_horizontal_face` / `cook_walls` already
    // — listed here only so the future v2 split has an obvious
    // landing spot for any extra collision-only data
    // (sector logic offset, traversal portal slots, …).

    Ok(Some(CookedGridSector {
        floor,
        ceiling,
        walls,
    }))
}

fn cook_horizontal_face(
    project: &ProjectDocument,
    face: &GridHorizontalFace,
    x: u16,
    z: u16,
    kind: WorldGridFaceKind,
    materials: &mut Vec<CookedWorldMaterial>,
    material_slots: &mut HashMap<ResourceId, u16>,
) -> Result<CookedGridHorizontalFace, WorldGridCookError> {
    if face.is_triangle() {
        return Err(WorldGridCookError::TriangleFaceNotSupported { x, z, face: kind });
    }
    Ok(CookedGridHorizontalFace {
        heights: face.heights,
        split: face.split,
        material: material_slot(
            project,
            face.material,
            x,
            z,
            kind,
            materials,
            material_slots,
        )?,
        walkable: face.walkable,
    })
}

fn cook_walls(
    project: &ProjectDocument,
    walls: &GridWalls,
    x: u16,
    z: u16,
    materials: &mut Vec<CookedWorldMaterial>,
    material_slots: &mut HashMap<ResourceId, u16>,
) -> Result<CookedGridWalls, WorldGridCookError> {
    let mut cooked = CookedGridWalls::default();
    // Diagonal walls fail loud: the data model has the slots
    // (so authoring + serialization works), but render / pick /
    // collision aren't consistent yet — better to refuse to
    // cook than ship half-supported geometry.
    for direction in [
        GridDirection::NorthWestSouthEast,
        GridDirection::NorthEastSouthWest,
    ] {
        if !walls.get(direction).is_empty() {
            return Err(WorldGridCookError::UnsupportedDiagonalWall { x, z, direction });
        }
    }
    // Cardinal walls only. Duplicate physical walls (east of
    // (x,z) and west of (x+1,z) describe the same face) are
    // already rejected upstream by `validate_no_duplicate_walls`
    // — by the time we reach this loop the grid is guaranteed
    // to claim each physical edge from at most one side.
    for direction in [
        GridDirection::North,
        GridDirection::East,
        GridDirection::South,
        GridDirection::West,
    ] {
        for wall in walls.get(direction) {
            if wall.is_triangle() {
                return Err(WorldGridCookError::TriangleFaceNotSupported {
                    x,
                    z,
                    face: WorldGridFaceKind::Wall(direction),
                });
            }
            validate_wall_heights(wall, x, z, direction)?;
            let material = material_slot(
                project,
                wall.material,
                x,
                z,
                WorldGridFaceKind::Wall(direction),
                materials,
                material_slots,
            )?;
            cooked.get_mut(direction).push(CookedGridVerticalFace {
                heights: wall.heights,
                material,
                solid: wall.solid,
            });
        }
    }
    Ok(cooked)
}

fn validate_wall_heights(
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

fn material_slot(
    project: &ProjectDocument,
    material: Option<ResourceId>,
    x: u16,
    z: u16,
    face: WorldGridFaceKind,
    materials: &mut Vec<CookedWorldMaterial>,
    material_slots: &mut HashMap<ResourceId, u16>,
) -> Result<u16, WorldGridCookError> {
    let id = material.ok_or(WorldGridCookError::UnassignedMaterial { x, z, face })?;
    if let Some(slot) = material_slots.get(&id).copied() {
        return Ok(slot);
    }

    let resource = project
        .resources
        .iter()
        .find(|resource| resource.id == id)
        .ok_or(WorldGridCookError::MissingMaterial { id })?;
    let ResourceData::Material(material) = &resource.data else {
        return Err(WorldGridCookError::ResourceIsNotMaterial { id });
    };
    if materials.len() >= u16::MAX as usize {
        return Err(WorldGridCookError::TooManyMaterials {
            count: materials.len() + 1,
        });
    }

    let slot = materials.len() as u16;
    materials.push(CookedWorldMaterial::from_resource(slot, id, material));
    material_slots.insert(id, slot);
    Ok(slot)
}

fn encode_cooked_world_grid_psxw(cooked: &CookedWorldGrid) -> Result<Vec<u8>, WorldGridCookError> {
    if cooked.sectors.len() > u16::MAX as usize {
        return Err(WorldGridCookError::TooManySectors {
            count: cooked.sectors.len(),
        });
    }
    if cooked.materials.len() > u16::MAX as usize {
        return Err(WorldGridCookError::TooManyMaterials {
            count: cooked.materials.len(),
        });
    }

    let mut sector_records = Vec::with_capacity(cooked.sectors.len() * world::SectorRecord::SIZE);
    let mut wall_records = Vec::new();

    for sector in &cooked.sectors {
        let first_wall_index = wall_records.len() / world::WallRecord::SIZE;
        let first_wall = checked_u16(
            first_wall_index,
            WorldGridCookError::TooManyWalls {
                count: first_wall_index,
            },
        )?;
        let wall_start = wall_records.len() / world::WallRecord::SIZE;
        if let Some(sector) = sector {
            encode_sector_walls(sector, &mut wall_records)?;
        }
        let wall_end = wall_records.len() / world::WallRecord::SIZE;
        let wall_count = checked_u16(
            wall_end - wall_start,
            WorldGridCookError::TooManyWalls {
                count: wall_end - wall_start,
            },
        )?;
        encode_sector_record(sector.as_ref(), first_wall, wall_count, &mut sector_records);
    }

    let wall_record_count = wall_records.len() / world::WallRecord::SIZE;

    let payload_len = world::WorldHeader::SIZE + sector_records.len() + wall_records.len();
    if payload_len > u32::MAX as usize {
        return Err(WorldGridCookError::EncodedWorldTooLarge { bytes: payload_len });
    }

    let mut out = Vec::with_capacity(psxed_format::AssetHeader::SIZE + payload_len);
    out.extend_from_slice(&world::MAGIC);
    out.extend_from_slice(&world::VERSION.to_le_bytes());
    out.extend_from_slice(&world::flags::RESERVED.to_le_bytes());
    out.extend_from_slice(&(payload_len as u32).to_le_bytes());

    out.extend_from_slice(&cooked.width.to_le_bytes());
    out.extend_from_slice(&cooked.depth.to_le_bytes());
    out.extend_from_slice(&cooked.sector_size.to_le_bytes());
    out.extend_from_slice(&(cooked.sectors.len() as u16).to_le_bytes());
    out.extend_from_slice(&(cooked.materials.len() as u16).to_le_bytes());
    out.extend_from_slice(&(wall_record_count as u16).to_le_bytes());
    out.extend_from_slice(&cooked.ambient_color);
    out.push(if cooked.fog_enabled {
        world::world_flags::FOG_ENABLED
    } else {
        0
    });
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&sector_records);
    out.extend_from_slice(&wall_records);
    Ok(out)
}

fn checked_u16(value: usize, error: WorldGridCookError) -> Result<u16, WorldGridCookError> {
    if value > u16::MAX as usize {
        Err(error)
    } else {
        Ok(value as u16)
    }
}

fn encode_sector_record(
    sector: Option<&CookedGridSector>,
    first_wall: u16,
    wall_count: u16,
    out: &mut Vec<u8>,
) {
    let mut flags = 0u8;
    let mut floor_split = world::split::NORTH_WEST_SOUTH_EAST;
    let mut ceiling_split = world::split::NORTH_WEST_SOUTH_EAST;
    let mut floor_material = world::NO_MATERIAL;
    let mut ceiling_material = world::NO_MATERIAL;
    let mut floor_heights = [0; 4];
    let mut ceiling_heights = [0; 4];

    if let Some(sector) = sector {
        if let Some(floor) = sector.floor {
            flags |= world::sector_flags::HAS_FLOOR;
            if floor.walkable {
                flags |= world::sector_flags::FLOOR_WALKABLE;
            }
            floor_split = split_id(floor.split);
            floor_material = floor.material;
            floor_heights = floor.heights;
        }
        if let Some(ceiling) = sector.ceiling {
            flags |= world::sector_flags::HAS_CEILING;
            if ceiling.walkable {
                flags |= world::sector_flags::CEILING_WALKABLE;
            }
            ceiling_split = split_id(ceiling.split);
            ceiling_material = ceiling.material;
            ceiling_heights = ceiling.heights;
        }
    }

    out.push(flags);
    out.push(floor_split);
    out.push(ceiling_split);
    out.push(0);
    out.extend_from_slice(&floor_material.to_le_bytes());
    out.extend_from_slice(&ceiling_material.to_le_bytes());
    out.extend_from_slice(&first_wall.to_le_bytes());
    out.extend_from_slice(&wall_count.to_le_bytes());
    for height in floor_heights {
        out.extend_from_slice(&height.to_le_bytes());
    }
    for height in ceiling_heights {
        out.extend_from_slice(&height.to_le_bytes());
    }
}

fn encode_sector_walls(
    sector: &CookedGridSector,
    out: &mut Vec<u8>,
) -> Result<(), WorldGridCookError> {
    for (direction, walls) in [
        (GridDirection::North, sector.walls.north.as_slice()),
        (GridDirection::East, sector.walls.east.as_slice()),
        (GridDirection::South, sector.walls.south.as_slice()),
        (GridDirection::West, sector.walls.west.as_slice()),
        (
            GridDirection::NorthWestSouthEast,
            sector.walls.north_west_south_east.as_slice(),
        ),
        (
            GridDirection::NorthEastSouthWest,
            sector.walls.north_east_south_west.as_slice(),
        ),
    ] {
        for wall in walls {
            if out.len() / world::WallRecord::SIZE >= u16::MAX as usize {
                return Err(WorldGridCookError::TooManyWalls {
                    count: (out.len() / world::WallRecord::SIZE) + 1,
                });
            }
            out.push(direction_id(direction));
            out.push(if wall.solid {
                world::wall_flags::SOLID
            } else {
                0
            });
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&wall.material.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            for height in wall.heights {
                out.extend_from_slice(&height.to_le_bytes());
            }
        }
    }
    Ok(())
}

const fn split_id(split: GridSplit) -> u8 {
    match split {
        GridSplit::NorthWestSouthEast => world::split::NORTH_WEST_SOUTH_EAST,
        GridSplit::NorthEastSouthWest => world::split::NORTH_EAST_SOUTH_WEST,
    }
}

const fn direction_id(direction: GridDirection) -> u8 {
    match direction {
        GridDirection::North => world::direction::NORTH,
        GridDirection::East => world::direction::EAST,
        GridDirection::South => world::direction::SOUTH,
        GridDirection::West => world::direction::WEST,
        GridDirection::NorthWestSouthEast => world::direction::NORTH_WEST_SOUTH_EAST,
        GridDirection::NorthEastSouthWest => world::direction::NORTH_EAST_SOUTH_WEST,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Corner, NodeKind, ResourceData, WallCorner};

    fn starter_grid(project: &ProjectDocument) -> WorldGrid {
        project
            .active_scene()
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                NodeKind::Room { grid } => Some(grid.clone()),
                _ => None,
            })
            .expect("starter project should contain a room")
    }

    #[test]
    fn cooks_starter_grid_to_material_slots() {
        let project = ProjectDocument::starter();
        let grid = starter_grid(&project);

        let cooked = cook_world_grid(&project, &grid).unwrap();

        assert_eq!(cooked.width, 3);
        assert_eq!(cooked.depth, 3);
        assert_eq!(cooked.sector_size, world::SECTOR_SIZE);
        assert_eq!(cooked.populated_sector_count(), 9);
        assert_eq!(cooked.materials.len(), 2);
        assert_eq!(cooked.materials[0].slot, 0);
        assert_eq!(cooked.materials[1].slot, 1);
        assert!(cooked.sectors[0].as_ref().unwrap().floor.is_some());
        assert_eq!(
            cooked.sectors[0].as_ref().unwrap().floor.unwrap().material,
            0
        );
        assert_eq!(
            cooked.sectors[0]
                .as_ref()
                .unwrap()
                .walls
                .north
                .first()
                .unwrap()
                .material,
            1
        );
    }

    #[test]
    fn rejects_unsupported_sector_size() {
        let project = ProjectDocument::starter();
        let mut grid = starter_grid(&project);
        grid.sector_size = 512;

        assert_eq!(
            cook_world_grid(&project, &grid).unwrap_err(),
            WorldGridCookError::UnsupportedSectorSize {
                expected: world::SECTOR_SIZE,
                actual: 512,
            }
        );
    }

    #[test]
    fn rejects_bad_sector_storage_len() {
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::empty(2, 2, world::SECTOR_SIZE);
        grid.sectors.pop();

        assert_eq!(
            cook_world_grid(&project, &grid).unwrap_err(),
            WorldGridCookError::SectorStorageLenMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn rejects_unassigned_surface_material() {
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.set_floor(0, 0, 0, None);

        assert_eq!(
            cook_world_grid(&project, &grid).unwrap_err(),
            WorldGridCookError::UnassignedMaterial {
                x: 0,
                z: 0,
                face: WorldGridFaceKind::Floor,
            }
        );
    }

    #[test]
    fn rejects_non_material_resources() {
        let project = ProjectDocument::starter();
        let texture = project
            .resources
            .iter()
            .find_map(|resource| match resource.data {
                ResourceData::Texture { .. } => Some(resource.id),
                _ => None,
            })
            .expect("starter project should contain a texture");
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.set_floor(0, 0, 0, Some(texture));

        assert_eq!(
            cook_world_grid(&project, &grid).unwrap_err(),
            WorldGridCookError::ResourceIsNotMaterial { id: texture }
        );
    }

    #[test]
    fn strips_empty_authored_sectors() {
        let project = ProjectDocument::starter();
        let grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);

        let cooked = cook_world_grid(&project, &grid).unwrap();

        assert_eq!(cooked.sectors, vec![None]);
        assert!(cooked.materials.is_empty());
    }

    #[test]
    fn cooked_starter_round_trips_through_psx_asset() {
        // End-to-end: cook the starter grid, parse the .psxw bytes
        // through `psx_asset::World`, and verify the runtime view
        // decodes the same dimensions / sector / wall counts the
        // cooker reported. Regresses the v1 byte layout against
        // both producer and consumer in one assertion.
        let project = ProjectDocument::starter();
        let grid = starter_grid(&project);
        let cooked = cook_world_grid(&project, &grid).unwrap();
        let bytes = encode_world_grid_psxw(&project, &grid).unwrap();

        let world = psx_asset::World::from_bytes(&bytes).expect("psxw parses");
        assert_eq!(world.width(), cooked.width);
        assert_eq!(world.depth(), cooked.depth);
        assert_eq!(world.sector_size(), cooked.sector_size);
        assert_eq!(world.material_count() as usize, cooked.materials.len());

        // Sector (0,0) — the starter has a floor + perimeter walls
        // here, so this exercises both decoded paths.
        let sector = world.sector(0, 0).expect("starter (0,0) populated");
        assert!(sector.has_floor());
        assert!(sector.wall_count() > 0);
    }

    #[test]
    fn starter_cook_pins_floor_to_slot_zero_and_brick_to_slot_one() {
        // The runtime side of `engine/examples/showcase-room`
        // bakes in slot 0 = floor texture, slot 1 = brick-wall
        // texture. The cooker assigns slots in first-use order
        // while iterating sectors `[x * depth + z]`. If a future
        // reshape flips that order, both this test and the
        // example's build.rs assertion fail loud.
        let project = ProjectDocument::starter();
        let grid = starter_grid(&project);
        let cooked = cook_world_grid(&project, &grid).unwrap();

        assert_eq!(cooked.materials.len(), 2);

        let psxt_path_for_slot = |slot: usize| -> String {
            let texture_id = cooked.materials[slot]
                .texture
                .expect("starter material has a texture");
            let texture = project.resource(texture_id).expect("texture in resources");
            match &texture.data {
                ResourceData::Texture { psxt_path } => psxt_path.clone(),
                _ => panic!("slot {slot} resource isn't a texture"),
            }
        };

        assert!(psxt_path_for_slot(0).ends_with("floor.psxt"));
        assert!(psxt_path_for_slot(1).ends_with("brick-wall.psxt"));
    }

    #[test]
    fn rejects_oversized_room_dimensions() {
        let project = ProjectDocument::starter();
        let grid = WorldGrid::empty(MAX_ROOM_WIDTH + 1, 4, world::SECTOR_SIZE);
        match cook_world_grid(&project, &grid) {
            Err(WorldGridCookError::RoomDimensionExceeded { axis, value, limit }) => {
                assert_eq!(axis, 'X');
                assert_eq!(value, MAX_ROOM_WIDTH + 1);
                assert_eq!(limit, MAX_ROOM_WIDTH);
            }
            other => panic!("expected RoomDimensionExceeded, got {other:?}"),
        }
    }

    #[test]
    fn rejects_excessive_wall_stacks() {
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        for _ in 0..(MAX_WALL_STACK as usize + 1) {
            grid.add_wall(
                0,
                0,
                GridDirection::North,
                0,
                world::SECTOR_SIZE,
                None,
            );
        }
        match cook_world_grid(&project, &grid) {
            Err(WorldGridCookError::WallStackExceeded {
                x,
                z,
                direction,
                count,
                limit,
            }) => {
                assert_eq!((x, z), (0, 0));
                assert_eq!(direction, GridDirection::North);
                assert_eq!(count, MAX_WALL_STACK as usize + 1);
                assert_eq!(limit, MAX_WALL_STACK as usize);
            }
            other => panic!("expected WallStackExceeded, got {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_physical_walls() {
        // East(0, 0) and West(1, 0) describe the same physical
        // wall. Authoring both must surface a clear cook error.
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::empty(2, 1, world::SECTOR_SIZE);
        grid.add_wall(0, 0, GridDirection::East, 0, world::SECTOR_SIZE, None);
        grid.add_wall(1, 0, GridDirection::West, 0, world::SECTOR_SIZE, None);
        match cook_world_grid(&project, &grid) {
            Err(WorldGridCookError::DuplicatePhysicalWall {
                x,
                z,
                direction,
                other_x,
                other_z,
                other_direction,
            }) => {
                // The cooker walks (x, z) major; East(0,0) is
                // claimed first, so the duplicate is West(1,0).
                assert_eq!((x, z, direction), (1, 0, GridDirection::West));
                assert_eq!(
                    (other_x, other_z, other_direction),
                    (0, 0, GridDirection::East)
                );
            }
            other => panic!("expected DuplicatePhysicalWall, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_quantized_heights() {
        // Floor with one corner off the quantum grid. Using
        // `set_floor` with a snap-aligned value, then mutating
        // a single corner directly so the test bypasses the
        // editor's drag snap (which is the runtime path the
        // cooker validator catches).
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::stone_room(1, 1, world::SECTOR_SIZE, None, None);
        if let Some(sector) = grid.sector_mut(0, 0) {
            if let Some(floor) = sector.floor.as_mut() {
                floor.heights[0] = 17; // Not a multiple of 32.
            }
        }
        match cook_world_grid(&project, &grid) {
            Err(WorldGridCookError::HeightNotQuantized {
                x,
                z,
                face,
                value,
                quantum,
            }) => {
                assert_eq!((x, z), (0, 0));
                assert_eq!(face, WorldGridFaceKind::Floor);
                assert_eq!(value, 17);
                assert_eq!(quantum, HEIGHT_QUANTUM);
            }
            other => panic!("expected HeightNotQuantized, got {other:?}"),
        }
    }

    #[test]
    fn rejects_triangle_floor() {
        // Drop a corner on the starter floor and confirm the
        // cooker refuses it: v1 .psxw doesn't carry triangles.
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::stone_room(1, 1, world::SECTOR_SIZE, None, None);
        if let Some(sector) = grid.sector_mut(0, 0) {
            if let Some(floor) = sector.floor.as_mut() {
                floor.drop_corner(Corner::NW);
            }
        }
        match cook_world_grid(&project, &grid) {
            Err(WorldGridCookError::TriangleFaceNotSupported { x, z, face }) => {
                assert_eq!((x, z), (0, 0));
                assert_eq!(face, WorldGridFaceKind::Floor);
            }
            other => panic!("expected TriangleFaceNotSupported, got {other:?}"),
        }
    }

    #[test]
    fn rejects_triangle_wall() {
        // Wall-only grid (no floor) so the cooker's first stop is
        // the wall validator; otherwise UnassignedMaterial on the
        // floor fires before we reach the wall path.
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.add_wall(0, 0, GridDirection::North, 0, world::SECTOR_SIZE, None);
        if let Some(sector) = grid.sector_mut(0, 0) {
            let walls = sector.walls.get_mut(GridDirection::North);
            if let Some(wall) = walls.first_mut() {
                wall.drop_corner(WallCorner::TL);
            }
        }
        match cook_world_grid(&project, &grid) {
            Err(WorldGridCookError::TriangleFaceNotSupported { x, z, face }) => {
                assert_eq!((x, z), (0, 0));
                assert_eq!(face, WorldGridFaceKind::Wall(GridDirection::North));
            }
            other => panic!("expected TriangleFaceNotSupported, got {other:?}"),
        }
    }

    #[test]
    fn rejects_diagonal_walls() {
        // Author a NW-SE diagonal wall and confirm the cooker
        // refuses it. Render / pick / collision aren't consistent
        // for diagonals yet; failing loud at cook time beats
        // shipping half-supported geometry.
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        // No floor — the diagonal-wall check has to fire before
        // any material validation in the rest of the sector.
        grid.add_wall(
            0,
            0,
            GridDirection::NorthWestSouthEast,
            0,
            world::SECTOR_SIZE,
            None,
        );

        match cook_world_grid(&project, &grid) {
            Err(WorldGridCookError::UnsupportedDiagonalWall { x, z, direction }) => {
                assert_eq!((x, z), (0, 0));
                assert_eq!(direction, GridDirection::NorthWestSouthEast);
            }
            other => panic!("expected UnsupportedDiagonalWall, got {other:?}"),
        }
    }

    #[test]
    fn encodes_starter_grid_as_psxw_blob() {
        let project = ProjectDocument::starter();
        let grid = starter_grid(&project);

        let bytes = encode_world_grid_psxw(&project, &grid).unwrap();

        assert_eq!(&bytes[0..4], &world::MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), world::VERSION);
        let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(
            payload_len as usize,
            bytes.len() - psxed_format::AssetHeader::SIZE
        );

        let header = psxed_format::AssetHeader::SIZE;
        assert_eq!(u16::from_le_bytes([bytes[header], bytes[header + 1]]), 3);
        assert_eq!(
            u16::from_le_bytes([bytes[header + 2], bytes[header + 3]]),
            3
        );
        assert_eq!(
            i32::from_le_bytes([
                bytes[header + 4],
                bytes[header + 5],
                bytes[header + 6],
                bytes[header + 7],
            ]),
            world::SECTOR_SIZE
        );
        assert_eq!(
            u16::from_le_bytes([bytes[header + 8], bytes[header + 9]]),
            9
        );
        assert_eq!(
            u16::from_le_bytes([bytes[header + 10], bytes[header + 11]]),
            2
        );
        assert_eq!(
            u16::from_le_bytes([bytes[header + 12], bytes[header + 13]]),
            12
        );

        let sector = header + world::WorldHeader::SIZE;
        assert_eq!(
            bytes[sector],
            world::sector_flags::HAS_FLOOR | world::sector_flags::FLOOR_WALKABLE
        );
        assert_eq!(
            u16::from_le_bytes([bytes[sector + 4], bytes[sector + 5]]),
            0
        );
        assert_eq!(
            u16::from_le_bytes([bytes[sector + 10], bytes[sector + 11]]),
            2
        );
    }
}
