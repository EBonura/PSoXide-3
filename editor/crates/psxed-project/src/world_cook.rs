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
    // Cardinal walls only. The cooker keeps each cell's locally-
    // authored walls — the wall-ownership normalization
    // (deduping the east wall of (x,z) against the west wall of
    // (x+1,z)) lives in `normalize_shared_walls` so the runtime
    // never gets two coplanar walls on the same physical edge.
    for direction in [
        GridDirection::North,
        GridDirection::East,
        GridDirection::South,
        GridDirection::West,
    ] {
        for wall in walls.get(direction) {
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
    use crate::{NodeKind, ResourceData};

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
