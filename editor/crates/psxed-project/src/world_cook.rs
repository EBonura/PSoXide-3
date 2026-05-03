//! Validation and cooking for authored grid worlds.
//!
//! The output here is still an owned host-side manifest. A later writer can
//! turn it into a compact `.psxw` blob or static Rust arrays for
//! `psx_engine::GridWorld`.

use std::collections::HashMap;

use psxed_format::world;

use crate::{
    snap_world_sector_size, GridDirection, GridHorizontalFace, GridSector, GridSplit,
    GridVerticalFace, GridWalls, MaterialFaceSidedness, MaterialResource, ProjectDocument,
    PsxBlendMode, ResourceData, ResourceId, WorldGrid, HEIGHT_QUANTUM, MAX_ROOM_BYTES,
    MAX_ROOM_DEPTH, MAX_ROOM_TRIANGLES, MAX_ROOM_WIDTH, MAX_WALL_STACK, WORLD_SECTOR_SIZE_QUANTUM,
};

mod coords;
mod encode;
mod materials;
mod validation;

use coords::{
    runtime_horizontal_heights, runtime_horizontal_split, runtime_horizontal_uvs,
    runtime_wall_direction, runtime_wall_heights, runtime_wall_uvs,
};
use encode::encode_cooked_world_grid_psxw;
use materials::material_slot;
use validation::{
    validate_grid_budget, validate_grid_shape, validate_no_duplicate_walls,
    validate_quantized_heights, validate_wall_heights,
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
    /// Sector size must be snapped to the editor/runtime sector quantum.
    UnsupportedSectorSize {
        /// Required sector-size quantum.
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
    /// picking + collision aren't consistent yet -- better to fail
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
    /// Two cells claim the same physical wall -- `East(x, z)`
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
    /// A face has a `dropped_corner` set -- the editor models it
    /// as a triangle, but the current `.psxw` wire format only carries
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
                write!(
                    f,
                    "unsupported sector size {actual}; expected {expected}-unit increments"
                )
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
                "sector {x},{z} {face} is a triangle (dropped corner); .psxw only carries quad faces"
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
    /// Which side(s) should render.
    pub face_sidedness: MaterialFaceSidedness,
}

impl CookedWorldMaterial {
    fn from_resource(slot: u16, source: ResourceId, material: &MaterialResource) -> Self {
        Self {
            slot,
            source,
            texture: material.texture,
            blend_mode: material.blend_mode,
            tint: material.tint,
            face_sidedness: material.sidedness(),
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
    /// Runtime UVs `[NW, NE, SE, SW]`.
    pub uvs: [(u8, u8); 4],
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
    /// Runtime UVs `[bottom-left, bottom-right, top-right, top-left]`.
    pub uvs: [(u8, u8); 4],
    /// Whether collision treats this wall as blocking.
    pub solid: bool,
}

/// Cooked wall lists for one sector.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CookedGridWalls {
    /// Walls on the runtime north edge (`.psxw` North = -Z).
    pub north: Vec<CookedGridVerticalFace>,
    /// Walls on the runtime east edge.
    pub east: Vec<CookedGridVerticalFace>,
    /// Walls on the runtime south edge (`.psxw` South = +Z).
    pub south: Vec<CookedGridVerticalFace>,
    /// Walls on the runtime west edge.
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
    /// Authored depth-cue far color. Kept in the cook model even
    /// while the current `.psxw` payload only persists the enable flag.
    pub fog_color: [u8; 3],
    /// Authored fog start distance in engine units.
    pub fog_near: i32,
    /// Authored fog end distance in engine units.
    pub fog_far: i32,
}

impl CookedWorldGrid {
    /// Number of sectors that contain cooked geometry.
    pub fn populated_sector_count(&self) -> usize {
        self.sectors.iter().flatten().count()
    }

    /// Number of cooked wall records across all sectors.
    pub fn wall_count(&self) -> u16 {
        self.sectors
            .iter()
            .flatten()
            .map(|sector| {
                sector.walls.north.len()
                    + sector.walls.east.len()
                    + sector.walls.south.len()
                    + sector.walls.west.len()
                    + sector.walls.north_west_south_east.len()
                    + sector.walls.north_east_south_west.len()
            })
            .sum::<usize>() as u16
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
                    cook_sector(
                        project,
                        sector,
                        x,
                        z,
                        grid.sector_size,
                        &mut materials,
                        &mut material_slots,
                    )
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
        fog_color: grid.fog_color,
        fog_near: grid.fog_near,
        fog_far: grid.fog_far,
    })
}

/// Validate, cook, and encode one editor-authored grid as `.psxw` bytes.
pub fn encode_world_grid_psxw(
    project: &ProjectDocument,
    grid: &WorldGrid,
) -> Result<Vec<u8>, WorldGridCookError> {
    cook_world_grid(project, grid)?.to_psxw_bytes()
}

fn cook_sector(
    project: &ProjectDocument,
    sector: &GridSector,
    x: u16,
    z: u16,
    sector_size: i32,
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
    //   render    -- heights, splits, material slot
    //   collision -- heights, splits, walkable / solid bits
    //
    // Heights and splits feed both views. Materials are render-
    // only (no collision branches on tpage). `walkable` /
    // `solid` are collision-only (no draw call branches on
    // walkability). When v2 lands and the byte format splits
    // render and collision into distinct tables, this function
    // peels into `cook_sector_render` + `cook_sector_collision`
    // -- the call structure is already grouped that way below.

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
    let walls = cook_walls(
        project,
        &sector.walls,
        x,
        z,
        sector_size,
        materials,
        material_slots,
    )?;

    // -------- collision-relevant cook --------
    // `walkable` / `solid` are forwarded through the cooked
    // structs by `cook_horizontal_face` / `cook_walls` already
    // -- listed here only so the future v2 split has an obvious
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
        heights: runtime_horizontal_heights(face.heights),
        split: runtime_horizontal_split(face.split),
        material: material_slot(
            project,
            face.material,
            x,
            z,
            kind,
            materials,
            material_slots,
        )?,
        uvs: runtime_horizontal_uvs(face.uv.apply_to_quad(world::FLOOR_UVS)),
        walkable: face.walkable,
    })
}

fn cook_walls(
    project: &ProjectDocument,
    walls: &GridWalls,
    x: u16,
    z: u16,
    sector_size: i32,
    materials: &mut Vec<CookedWorldMaterial>,
    material_slots: &mut HashMap<ResourceId, u16>,
) -> Result<CookedGridWalls, WorldGridCookError> {
    let mut cooked = CookedGridWalls::default();
    // Diagonal walls fail loud: the data model has the slots
    // (so authoring + serialization works), but render / pick /
    // collision aren't consistent yet -- better to refuse to
    // cook than ship half-supported geometry.
    for direction in GridDirection::DIAGONAL {
        if !walls.get(direction).is_empty() {
            return Err(WorldGridCookError::UnsupportedDiagonalWall { x, z, direction });
        }
    }
    // Cardinal walls only. Duplicate physical walls (east of
    // (x,z) and west of (x+1,z) describe the same face) are
    // already rejected upstream by `validate_no_duplicate_walls`
    // -- by the time we reach this loop the grid is guaranteed
    // to claim each physical edge from at most one side.
    for direction in GridDirection::CARDINAL {
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
            let runtime_direction = runtime_wall_direction(direction);
            for mut segment in wall.split_into_autotile_segments(sector_size) {
                validate_wall_heights(&segment, x, z, direction)?;
                if segment.uv.span[1] == 0 {
                    segment.autotile_uv(sector_size);
                    segment.uv.offset[1] =
                        crate::wrap_tiled_uv_offset_i16(i64::from(segment.uv.offset[1]));
                }
                cooked
                    .get_mut(runtime_direction)
                    .push(CookedGridVerticalFace {
                        heights: runtime_wall_heights(segment.heights),
                        material,
                        uvs: runtime_wall_uvs(segment.uv.apply_to_quad(world::WALL_UVS)),
                        solid: segment.solid,
                    });
            }
        }
    }
    Ok(cooked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Corner, GridUvRotation, GridUvTransform, NodeKind, ResourceData, WallCorner};

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

    fn first_floor_material(grid: &WorldGrid) -> ResourceId {
        grid.sectors
            .iter()
            .flatten()
            .find_map(|sector| sector.floor.as_ref()?.material)
            .expect("starter floor has material")
    }

    fn texture_named(project: &ProjectDocument, suffix: &str) -> ResourceId {
        project
            .resources
            .iter()
            .find_map(|resource| match &resource.data {
                ResourceData::Texture { psxt_path } if psxt_path.ends_with(suffix) => {
                    Some(resource.id)
                }
                _ => None,
            })
            .expect("starter texture exists")
    }

    fn material_for_texture(
        project: &mut ProjectDocument,
        name: &str,
        texture: ResourceId,
    ) -> ResourceId {
        project.add_resource(
            name,
            ResourceData::Material(crate::MaterialResource::opaque(Some(texture))),
        )
    }

    fn first_populated_cooked_sector(cooked: &CookedWorldGrid) -> &CookedGridSector {
        cooked
            .sectors
            .iter()
            .flatten()
            .next()
            .expect("starter has populated sector")
    }

    fn authored_geometry_sector_count(grid: &WorldGrid) -> usize {
        grid.sectors
            .iter()
            .flatten()
            .filter(|sector| sector.has_geometry())
            .count()
    }

    fn cooked_wall_count(sector: &CookedGridSector) -> usize {
        sector.walls.north.len()
            + sector.walls.east.len()
            + sector.walls.south.len()
            + sector.walls.west.len()
            + sector.walls.north_west_south_east.len()
            + sector.walls.north_east_south_west.len()
    }

    fn max_uv_v_span(uvs: [(u8, u8); 4]) -> u8 {
        let min_v = uvs.iter().map(|(_, v)| *v).min().unwrap();
        let max_v = uvs.iter().map(|(_, v)| *v).max().unwrap();
        max_v - min_v
    }

    #[test]
    fn cooks_starter_grid_to_material_slots() {
        let project = ProjectDocument::starter();
        let grid = starter_grid(&project);

        let cooked = cook_world_grid(&project, &grid).unwrap();

        assert_eq!(cooked.width, grid.width);
        assert_eq!(cooked.depth, grid.depth);
        assert_eq!(cooked.sector_size, grid.sector_size);
        assert_eq!(
            cooked.populated_sector_count(),
            authored_geometry_sector_count(&grid)
        );
        assert_eq!(cooked.materials.len(), 3);
        for (idx, material) in cooked.materials.iter().enumerate() {
            assert_eq!(material.slot, idx as u16);
        }
        let first_sector = first_populated_cooked_sector(&cooked);
        assert!(first_sector.floor.is_some());
        assert_eq!(first_sector.floor.unwrap().material, 0);
    }

    #[test]
    fn cook_flips_horizontal_faces_to_runtime_z_convention() {
        let project = ProjectDocument::starter();
        let material = first_floor_material(&starter_grid(&project));
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.set_floor(0, 0, 0, Some(material));
        let floor = grid
            .sector_mut(0, 0)
            .and_then(|sector| sector.floor.as_mut())
            .expect("floor exists");
        floor.heights = [32, 64, 96, 128];
        floor.split = GridSplit::NorthWestSouthEast;

        let cooked = cook_world_grid(&project, &grid).unwrap();
        let cooked_floor = cooked.sectors[0].as_ref().unwrap().floor.unwrap();

        assert_eq!(cooked_floor.heights, [128, 96, 64, 32]);
        assert_eq!(cooked_floor.split, GridSplit::NorthEastSouthWest);
    }

    #[test]
    fn cook_bakes_horizontal_uv_transform_to_runtime_uvs() {
        let project = ProjectDocument::starter();
        let material = first_floor_material(&starter_grid(&project));
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.set_floor(0, 0, 0, Some(material));
        let floor_uv = GridUvTransform {
            offset: [3, -2],
            span: [0, 0],
            rotation: GridUvRotation::Deg90,
            flip_u: false,
            flip_v: true,
        };
        grid.sector_mut(0, 0)
            .and_then(|sector| sector.floor.as_mut())
            .expect("floor exists")
            .uv = floor_uv;

        let cooked = cook_world_grid(&project, &grid).unwrap();
        let cooked_floor = cooked.sectors[0].as_ref().unwrap().floor.unwrap();
        let bytes = encode_world_grid_psxw(&project, &grid).unwrap();
        let world = psx_asset::World::from_bytes(&bytes).expect("psxw parses");

        assert_eq!(
            cooked_floor.uvs,
            runtime_horizontal_uvs(floor_uv.apply_to_quad(world::FLOOR_UVS))
        );
        assert_eq!(
            world.sector(0, 0).unwrap().floor_uvs().corners(),
            cooked_floor.uvs
        );
    }

    #[test]
    fn cook_flips_cardinal_walls_to_runtime_convention() {
        let project = ProjectDocument::starter();
        let material = first_floor_material(&starter_grid(&project));
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.add_wall(
            0,
            0,
            GridDirection::North,
            0,
            world::SECTOR_SIZE,
            Some(material),
        );
        let wall = grid
            .sector_mut(0, 0)
            .and_then(|sector| sector.walls.get_mut(GridDirection::North).first_mut())
            .expect("north wall exists");
        wall.heights = [32, 64, 96, 128];

        let cooked = cook_world_grid(&project, &grid).unwrap();
        let cooked_walls = &cooked.sectors[0].as_ref().unwrap().walls;

        assert!(cooked_walls.north.is_empty());
        assert_eq!(cooked_walls.south[0].heights, [64, 32, 128, 96]);
    }

    #[test]
    fn cook_bakes_wall_uv_transform_to_runtime_uvs() {
        let project = ProjectDocument::starter();
        let material = first_floor_material(&starter_grid(&project));
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.add_wall(
            0,
            0,
            GridDirection::North,
            0,
            world::SECTOR_SIZE,
            Some(material),
        );
        let wall_uv = GridUvTransform {
            offset: [-5, 7],
            span: [0, 0],
            rotation: GridUvRotation::Deg180,
            flip_u: true,
            flip_v: false,
        };
        grid.sector_mut(0, 0)
            .and_then(|sector| sector.walls.get_mut(GridDirection::North).first_mut())
            .expect("north wall exists")
            .uv = wall_uv;

        let cooked = cook_world_grid(&project, &grid).unwrap();
        let cooked_wall = cooked.sectors[0]
            .as_ref()
            .unwrap()
            .walls
            .south
            .first()
            .copied()
            .expect("north editor wall cooks to runtime south");
        let bytes = encode_world_grid_psxw(&project, &grid).unwrap();
        let world = psx_asset::World::from_bytes(&bytes).expect("psxw parses");
        let sector = world.sector(0, 0).unwrap();
        let parsed_wall = world.sector_wall(sector, 0).unwrap();

        assert_eq!(
            cooked_wall.uvs,
            runtime_wall_uvs(wall_uv.apply_to_quad(world::WALL_UVS))
        );
        assert_eq!(parsed_wall.uvs().corners(), cooked_wall.uvs);
    }

    #[test]
    fn cook_autotiles_implicit_tall_wall_without_splitting_when_uv_fits_packet() {
        let project = ProjectDocument::starter();
        let material = first_floor_material(&starter_grid(&project));
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.add_wall(
            0,
            0,
            GridDirection::North,
            0,
            world::SECTOR_SIZE * 2,
            Some(material),
        );

        let cooked = cook_world_grid(&project, &grid).unwrap();
        let cooked_sector = cooked.sectors[0].as_ref().unwrap();
        let bytes = encode_world_grid_psxw(&project, &grid).unwrap();
        let parsed_world = psx_asset::World::from_bytes(&bytes).expect("psxw parses");
        let parsed_sector = parsed_world.sector(0, 0).unwrap();
        let parsed_wall = parsed_world.sector_wall(parsed_sector, 0).unwrap();

        assert_eq!(cooked_wall_count(cooked_sector), 1);
        assert_eq!(parsed_sector.wall_count(), 1);
        assert_eq!(
            max_uv_v_span(cooked_sector.walls.south[0].uvs),
            world::TILE_UV * 2
        );
        assert_eq!(
            max_uv_v_span(parsed_wall.uvs().corners()),
            world::TILE_UV * 2
        );
    }

    #[test]
    fn cook_splits_autotiled_wall_only_when_uv_exceeds_packet_range() {
        let project = ProjectDocument::starter();
        let material = first_floor_material(&starter_grid(&project));
        let mut grid = WorldGrid::empty(1, 1, world::SECTOR_SIZE);
        grid.add_wall(
            0,
            0,
            GridDirection::North,
            0,
            world::SECTOR_SIZE * 5,
            Some(material),
        );

        let cooked = cook_world_grid(&project, &grid).unwrap();
        let cooked_sector = cooked.sectors[0].as_ref().unwrap();
        let bytes = encode_world_grid_psxw(&project, &grid).unwrap();
        let parsed_world = psx_asset::World::from_bytes(&bytes).expect("psxw parses");
        let parsed_sector = parsed_world.sector(0, 0).unwrap();

        assert_eq!(cooked_wall_count(cooked_sector), 5);
        assert_eq!(parsed_sector.wall_count(), 5);
        for wall in &cooked_sector.walls.south {
            assert!(max_uv_v_span(wall.uvs) <= world::TILE_UV);
        }
        for index in 0..5 {
            let parsed_wall = parsed_world.sector_wall(parsed_sector, index).unwrap();
            assert!(max_uv_v_span(parsed_wall.uvs().corners()) <= world::TILE_UV);
        }
    }

    #[test]
    fn cooks_quantized_non_default_sector_size() {
        let project = ProjectDocument::starter();
        let mut grid = starter_grid(&project);
        grid.rescale_sector_size(1536);

        let cooked = cook_world_grid(&project, &grid).unwrap();

        assert_eq!(cooked.sector_size, 1536);
    }

    #[test]
    fn rejects_non_quantized_sector_size() {
        let project = ProjectDocument::starter();
        let mut grid = starter_grid(&project);
        grid.sector_size = 513;

        assert_eq!(
            cook_world_grid(&project, &grid).unwrap_err(),
            WorldGridCookError::UnsupportedSectorSize {
                expected: WORLD_SECTOR_SIZE_QUANTUM,
                actual: 513,
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
        let mut project = ProjectDocument::starter();
        let floor_texture = texture_named(&project, "floor.psxt");
        let brick_texture = texture_named(&project, "brick-wall.psxt");
        let floor = material_for_texture(&mut project, "Floor Slot", floor_texture);
        let wall = material_for_texture(&mut project, "Brick Slot", brick_texture);
        let grid = WorldGrid::stone_room(2, 2, 1024, Some(floor), Some(wall));
        let cooked = cook_world_grid(&project, &grid).unwrap();
        let bytes = encode_world_grid_psxw(&project, &grid).unwrap();

        let world = psx_asset::World::from_bytes(&bytes).expect("psxw parses");
        assert_eq!(world.width(), cooked.width);
        assert_eq!(world.depth(), cooked.depth);
        assert_eq!(world.sector_size(), cooked.sector_size);
        assert_eq!(world.material_count() as usize, cooked.materials.len());
        assert_eq!(world.ambient_color(), grid.ambient_color);

        // Sector (0,0) -- the starter has a floor + perimeter walls
        // in its first populated cell, so this exercises both
        // decoded paths without assuming a fixed starter shape.
        let (sx, sz, _) = cooked
            .sectors
            .iter()
            .enumerate()
            .find_map(|(index, sector)| {
                let sector = sector.as_ref()?;
                Some((
                    (index / cooked.depth as usize) as u16,
                    (index % cooked.depth as usize) as u16,
                    sector,
                ))
            })
            .expect("starter has populated sector");
        let sector = world.sector(sx, sz).expect("starter sector populated");
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
        let mut project = ProjectDocument::starter();
        let floor_texture = texture_named(&project, "floor.psxt");
        let brick_texture = texture_named(&project, "brick-wall.psxt");
        let floor = material_for_texture(&mut project, "Floor Slot", floor_texture);
        let wall = material_for_texture(&mut project, "Brick Slot", brick_texture);
        let grid = WorldGrid::stone_room(2, 2, 1024, Some(floor), Some(wall));
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
        for _ in 0..(MAX_WALL_STACK + 1) {
            grid.add_wall(0, 0, GridDirection::North, 0, world::SECTOR_SIZE, None);
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
                assert_eq!(count, MAX_WALL_STACK + 1);
                assert_eq!(limit, MAX_WALL_STACK);
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
    fn rejects_duplicate_physical_walls_on_editor_north_south_edges() {
        // Editor convention is North=+Z and South=-Z. These two
        // authored records both claim the edge between rows 0 and 1.
        let project = ProjectDocument::starter();
        let mut grid = WorldGrid::empty(1, 2, world::SECTOR_SIZE);
        grid.add_wall(0, 0, GridDirection::North, 0, world::SECTOR_SIZE, None);
        grid.add_wall(0, 1, GridDirection::South, 0, world::SECTOR_SIZE, None);

        match cook_world_grid(&project, &grid) {
            Err(WorldGridCookError::DuplicatePhysicalWall {
                x,
                z,
                direction,
                other_x,
                other_z,
                other_direction,
            }) => {
                assert_eq!((x, z, direction), (0, 1, GridDirection::South));
                assert_eq!(
                    (other_x, other_z, other_direction),
                    (0, 0, GridDirection::North)
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
        // cooker refuses it: .psxw doesn't carry triangles.
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
        // No floor -- the diagonal-wall check has to fire before
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
        let cooked = cook_world_grid(&project, &grid).unwrap();

        let bytes = encode_world_grid_psxw(&project, &grid).unwrap();

        assert_eq!(&bytes[0..4], &world::MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), world::VERSION);
        let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(
            payload_len as usize,
            bytes.len() - psxed_format::AssetHeader::SIZE
        );

        let header = psxed_format::AssetHeader::SIZE;
        assert_eq!(
            u16::from_le_bytes([bytes[header], bytes[header + 1]]),
            grid.width
        );
        assert_eq!(
            u16::from_le_bytes([bytes[header + 2], bytes[header + 3]]),
            grid.depth
        );
        assert_eq!(
            i32::from_le_bytes([
                bytes[header + 4],
                bytes[header + 5],
                bytes[header + 6],
                bytes[header + 7],
            ]),
            grid.sector_size
        );
        assert_eq!(
            u16::from_le_bytes([bytes[header + 8], bytes[header + 9]]),
            grid.width * grid.depth
        );
        assert_eq!(
            u16::from_le_bytes([bytes[header + 10], bytes[header + 11]]),
            cooked.materials.len() as u16
        );
        assert_eq!(
            u16::from_le_bytes([bytes[header + 12], bytes[header + 13]]),
            cooked.wall_count()
        );
        assert_eq!(
            [bytes[header + 14], bytes[header + 15], bytes[header + 16]],
            grid.ambient_color
        );

        let first_sector_index = cooked
            .sectors
            .iter()
            .position(|sector| sector.is_some())
            .expect("starter has populated sector");
        let sector =
            header + world::WorldHeader::SIZE + first_sector_index * world::SectorRecord::SIZE;
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
            cooked_wall_count(first_populated_cooked_sector(&cooked)) as u16
        );
    }
}
