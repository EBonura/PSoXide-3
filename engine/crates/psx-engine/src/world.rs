//! Grid-world runtime data.
//!
//! The editor-facing model is free to be pleasant and dynamic. The engine
//! model is intentionally static: rooms are grids of sectors, sectors own
//! optional floor/ceiling faces plus edge walls, and every collection is a
//! borrowed slice suitable for cooked, ROM-backed data.

use crate::WorldVertex;

/// World units per grid sector.
///
/// Bonnie-32 used TR-style 1024-unit sectors. Keeping that unit at engine
/// level gives the editor, collision, and render cooker a shared scale.
pub const GRID_SECTOR_SIZE: i32 = 1024;

/// Runtime material slot used by cooked world geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldMaterialId(pub u16);

/// Cardinal or diagonal sector edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridDirection {
    /// North edge, -Z.
    North,
    /// East edge, +X.
    East,
    /// South edge, +Z.
    South,
    /// West edge, -X.
    West,
    /// Diagonal from north-west to south-east.
    NorthWestSouthEast,
    /// Diagonal from north-east to south-west.
    NorthEastSouthWest,
}

/// Diagonal split used for a quad face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridSplit {
    /// Split from north-west to south-east.
    NorthWestSouthEast,
    /// Split from north-east to south-west.
    NorthEastSouthWest,
}

impl GridSplit {
    /// First triangle corner indices in `[NW, NE, SE, SW]` order.
    pub const fn triangle_a(self) -> [usize; 3] {
        match self {
            Self::NorthWestSouthEast => [0, 1, 2],
            Self::NorthEastSouthWest => [0, 1, 3],
        }
    }

    /// Second triangle corner indices in `[NW, NE, SE, SW]` order.
    pub const fn triangle_b(self) -> [usize; 3] {
        match self {
            Self::NorthWestSouthEast => [0, 2, 3],
            Self::NorthEastSouthWest => [1, 2, 3],
        }
    }
}

impl Default for GridSplit {
    fn default() -> Self {
        Self::NorthWestSouthEast
    }
}

/// Horizontal sector face, used for floors and ceilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridHorizontalFace {
    /// Corner heights `[NW, NE, SE, SW]` in world units.
    pub heights: [i32; 4],
    /// Diagonal split.
    pub split: GridSplit,
    /// Runtime material slot.
    pub material: WorldMaterialId,
    /// Whether character collision treats this face as walkable.
    pub walkable: bool,
}

impl GridHorizontalFace {
    /// Create a flat face at `height`.
    pub const fn flat(height: i32, material: WorldMaterialId) -> Self {
        Self {
            heights: [height, height, height, height],
            split: GridSplit::NorthWestSouthEast,
            material,
            walkable: true,
        }
    }

    /// Average face height.
    pub const fn average_height(self) -> i32 {
        (self.heights[0] + self.heights[1] + self.heights[2] + self.heights[3]) / 4
    }

    /// True when every corner has the same height.
    pub const fn is_flat(self) -> bool {
        self.heights[0] == self.heights[1]
            && self.heights[0] == self.heights[2]
            && self.heights[0] == self.heights[3]
    }

    /// Heights along one edge, ordered left-to-right as seen from inside.
    pub const fn edge_heights(self, direction: GridDirection) -> (i32, i32) {
        match direction {
            GridDirection::North => (self.heights[0], self.heights[1]),
            GridDirection::East => (self.heights[1], self.heights[2]),
            GridDirection::South => (self.heights[3], self.heights[2]),
            GridDirection::West => (self.heights[0], self.heights[3]),
            GridDirection::NorthWestSouthEast => (self.heights[0], self.heights[2]),
            GridDirection::NorthEastSouthWest => (self.heights[1], self.heights[3]),
        }
    }
}

/// Vertical face on a sector edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridVerticalFace {
    /// Corner heights `[bottom-left, bottom-right, top-right, top-left]`.
    pub heights: [i32; 4],
    /// Runtime material slot.
    pub material: WorldMaterialId,
    /// Whether collision treats this face as blocking.
    pub solid: bool,
}

impl GridVerticalFace {
    /// Create a flat vertical wall.
    pub const fn flat(bottom: i32, top: i32, material: WorldMaterialId) -> Self {
        Self {
            heights: [bottom, bottom, top, top],
            material,
            solid: true,
        }
    }

    /// Average bottom height.
    pub const fn bottom(self) -> i32 {
        (self.heights[0] + self.heights[1]) / 2
    }

    /// Average top height.
    pub const fn top(self) -> i32 {
        (self.heights[2] + self.heights[3]) / 2
    }
}

/// Wall lists for every sector edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridWalls<'a> {
    /// Walls on the north edge.
    pub north: &'a [GridVerticalFace],
    /// Walls on the east edge.
    pub east: &'a [GridVerticalFace],
    /// Walls on the south edge.
    pub south: &'a [GridVerticalFace],
    /// Walls on the west edge.
    pub west: &'a [GridVerticalFace],
    /// Diagonal NW-SE walls.
    pub north_west_south_east: &'a [GridVerticalFace],
    /// Diagonal NE-SW walls.
    pub north_east_south_west: &'a [GridVerticalFace],
}

impl<'a> GridWalls<'a> {
    /// Empty edge wall lists.
    pub const EMPTY: Self = Self {
        north: &[],
        east: &[],
        south: &[],
        west: &[],
        north_west_south_east: &[],
        north_east_south_west: &[],
    };

    /// Walls for one direction.
    pub const fn get(self, direction: GridDirection) -> &'a [GridVerticalFace] {
        match direction {
            GridDirection::North => self.north,
            GridDirection::East => self.east,
            GridDirection::South => self.south,
            GridDirection::West => self.west,
            GridDirection::NorthWestSouthEast => self.north_west_south_east,
            GridDirection::NorthEastSouthWest => self.north_east_south_west,
        }
    }
}

impl Default for GridWalls<'_> {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// One authored sector in a room grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSector<'a> {
    /// Optional floor face.
    pub floor: Option<GridHorizontalFace>,
    /// Optional ceiling face.
    pub ceiling: Option<GridHorizontalFace>,
    /// Walls on sector edges.
    pub walls: GridWalls<'a>,
}

impl<'a> GridSector<'a> {
    /// Empty sector.
    pub const EMPTY: Self = Self {
        floor: None,
        ceiling: None,
        walls: GridWalls::EMPTY,
    };

    /// Sector with a single floor.
    pub const fn with_floor(height: i32, material: WorldMaterialId) -> Self {
        Self {
            floor: Some(GridHorizontalFace::flat(height, material)),
            ceiling: None,
            walls: GridWalls::EMPTY,
        }
    }

    /// True when this sector emits any world geometry.
    pub const fn has_geometry(self) -> bool {
        self.floor.is_some()
            || self.ceiling.is_some()
            || !self.walls.north.is_empty()
            || !self.walls.east.is_empty()
            || !self.walls.south.is_empty()
            || !self.walls.west.is_empty()
            || !self.walls.north_west_south_east.is_empty()
            || !self.walls.north_east_south_west.is_empty()
    }
}

impl Default for GridSector<'_> {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Integer grid coordinate inside a room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridCoord {
    /// X sector coordinate.
    pub x: u16,
    /// Z sector coordinate.
    pub z: u16,
}

/// Floor sample at a world-space X/Z point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridFloorSample {
    /// Sector containing the point.
    pub coord: GridCoord,
    /// Floor face used for the sample.
    pub face: GridHorizontalFace,
    /// Interpolated floor height in world units.
    pub height: i32,
}

/// One cooked grid room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridRoom<'a> {
    /// World-space room origin.
    pub origin: WorldVertex,
    /// Width in sectors.
    pub width: u16,
    /// Depth in sectors.
    pub depth: u16,
    /// Flat `[x * depth + z]` sector storage. `None` means no sector.
    pub sectors: &'a [Option<GridSector<'a>>],
}

impl<'a> GridRoom<'a> {
    /// Create a room over cooked sector storage.
    pub const fn new(
        origin: WorldVertex,
        width: u16,
        depth: u16,
        sectors: &'a [Option<GridSector<'a>>],
    ) -> Self {
        Self {
            origin,
            width,
            depth,
            sectors,
        }
    }

    /// Flat sector index for a coordinate.
    pub const fn sector_index(self, coord: GridCoord) -> Option<usize> {
        if coord.x < self.width && coord.z < self.depth {
            Some(coord.x as usize * self.depth as usize + coord.z as usize)
        } else {
            None
        }
    }

    /// Sector at a coordinate.
    pub fn sector(self, coord: GridCoord) -> Option<GridSector<'a>> {
        self.sector_index(coord)
            .and_then(|index| self.sectors.get(index).copied().flatten())
    }

    /// Floor under a world-space X/Z point.
    pub fn floor_at(self, x: i32, z: i32) -> Option<GridFloorSample> {
        let coord = self.world_to_grid(x, z)?;
        let sector = self.sector(coord)?;
        let face = sector.floor?;
        let local_x =
            (x - (self.origin.x + coord.x as i32 * GRID_SECTOR_SIZE)).clamp(0, GRID_SECTOR_SIZE);
        let local_z =
            (z - (self.origin.z + coord.z as i32 * GRID_SECTOR_SIZE)).clamp(0, GRID_SECTOR_SIZE);
        Some(GridFloorSample {
            coord,
            face,
            height: face.height_at_local(local_x, local_z),
        })
    }

    /// Walls on a sector edge, or an empty slice for absent sectors.
    pub fn walls(self, coord: GridCoord, direction: GridDirection) -> &'a [GridVerticalFace] {
        self.sector(coord)
            .map(|sector| sector.walls.get(direction))
            .unwrap_or(&[])
    }

    /// Convert a sector coordinate to the world-space north-west corner.
    pub const fn grid_to_world(self, coord: GridCoord) -> WorldVertex {
        WorldVertex::new(
            self.origin.x + coord.x as i32 * GRID_SECTOR_SIZE,
            self.origin.y,
            self.origin.z + coord.z as i32 * GRID_SECTOR_SIZE,
        )
    }

    /// Convert a world X/Z position to a sector coordinate.
    pub const fn world_to_grid(self, x: i32, z: i32) -> Option<GridCoord> {
        let local_x = x - self.origin.x;
        let local_z = z - self.origin.z;
        if local_x < 0 || local_z < 0 {
            return None;
        }
        let grid_x = (local_x / GRID_SECTOR_SIZE) as u16;
        let grid_z = (local_z / GRID_SECTOR_SIZE) as u16;
        if grid_x < self.width && grid_z < self.depth {
            Some(GridCoord {
                x: grid_x,
                z: grid_z,
            })
        } else {
            None
        }
    }
}

impl GridHorizontalFace {
    /// Interpolated height at local sector coordinates.
    ///
    /// `local_x` and `local_z` are clamped to `0..=GRID_SECTOR_SIZE`.
    pub fn height_at_local(self, local_x: i32, local_z: i32) -> i32 {
        let u = local_x.clamp(0, GRID_SECTOR_SIZE);
        let v = local_z.clamp(0, GRID_SECTOR_SIZE);
        let [nw, ne, se, sw] = self.heights;
        match self.split {
            GridSplit::NorthWestSouthEast => {
                if v <= u {
                    nw.saturating_add(mul_sector(height_delta(ne, nw), u - v))
                        .saturating_add(mul_sector(height_delta(se, nw), v))
                } else {
                    nw.saturating_add(mul_sector(height_delta(se, sw), u))
                        .saturating_add(mul_sector(height_delta(sw, nw), v))
                }
            }
            GridSplit::NorthEastSouthWest => {
                if u + v <= GRID_SECTOR_SIZE {
                    nw.saturating_add(mul_sector(height_delta(ne, nw), u))
                        .saturating_add(mul_sector(height_delta(sw, nw), v))
                } else {
                    sw.saturating_add(mul_sector(height_delta(se, sw), u))
                        .saturating_add(mul_sector(height_delta(ne, se), GRID_SECTOR_SIZE - v))
                }
            }
        }
    }
}

fn height_delta(to: i32, from: i32) -> i32 {
    to.saturating_sub(from)
}

fn mul_sector(delta: i32, amount: i32) -> i32 {
    delta.saturating_mul(amount) / GRID_SECTOR_SIZE
}

/// Complete cooked grid-world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridWorld<'a> {
    /// Rooms in the world.
    pub rooms: &'a [GridRoom<'a>],
}

impl<'a> GridWorld<'a> {
    /// Empty grid-world.
    pub const EMPTY: Self = Self { rooms: &[] };

    /// Create a world from rooms.
    pub const fn new(rooms: &'a [GridRoom<'a>]) -> Self {
        Self { rooms }
    }

    /// Room by index.
    pub fn room(self, index: usize) -> Option<GridRoom<'a>> {
        self.rooms.get(index).copied()
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;

    const MAT_FLOOR: WorldMaterialId = WorldMaterialId(1);
    const MAT_WALL: WorldMaterialId = WorldMaterialId(2);
    const NORTH_WALL: [GridVerticalFace; 1] = [GridVerticalFace::flat(0, 1024, MAT_WALL)];
    const SECTORS: [Option<GridSector<'static>>; 2] = [
        Some(GridSector {
            floor: Some(GridHorizontalFace::flat(0, MAT_FLOOR)),
            ceiling: None,
            walls: GridWalls {
                north: &NORTH_WALL,
                ..GridWalls::EMPTY
            },
        }),
        None,
    ];
    const ROOM: GridRoom<'static> = GridRoom::new(WorldVertex::ZERO, 1, 2, &SECTORS);

    #[test]
    fn room_maps_world_positions_to_sectors() {
        assert_eq!(ROOM.world_to_grid(12, 1030), Some(GridCoord { x: 0, z: 1 }));
        assert_eq!(ROOM.world_to_grid(-1, 0), None);
        assert_eq!(ROOM.world_to_grid(0, 2048), None);
    }

    #[test]
    fn sector_preserves_floor_and_wall_data() {
        let sector = ROOM.sector(GridCoord { x: 0, z: 0 }).unwrap();
        assert!(sector.has_geometry());
        assert_eq!(sector.floor.unwrap().average_height(), 0);
        assert_eq!(sector.walls.get(GridDirection::North)[0].top(), 1024);
        assert!(ROOM.sector(GridCoord { x: 0, z: 1 }).is_none());
    }

    #[test]
    fn floor_at_samples_flat_floor_and_empty_cells() {
        let sample = ROOM.floor_at(12, 12).unwrap();
        assert_eq!(sample.coord, GridCoord { x: 0, z: 0 });
        assert_eq!(sample.height, 0);
        assert!(ROOM.floor_at(12, 1030).is_none());
    }

    #[test]
    fn walls_returns_sector_edge_or_empty_slice() {
        assert_eq!(
            ROOM.walls(GridCoord { x: 0, z: 0 }, GridDirection::North)
                .len(),
            1
        );
        assert!(ROOM
            .walls(GridCoord { x: 0, z: 1 }, GridDirection::North)
            .is_empty());
    }

    #[test]
    fn height_at_local_respects_nw_se_split() {
        let face = GridHorizontalFace {
            heights: [0, 1024, 2048, 1024],
            split: GridSplit::NorthWestSouthEast,
            material: MAT_FLOOR,
            walkable: true,
        };
        assert_eq!(face.height_at_local(0, 0), 0);
        assert_eq!(face.height_at_local(1024, 0), 1024);
        assert_eq!(face.height_at_local(1024, 1024), 2048);
        assert_eq!(face.height_at_local(0, 1024), 1024);
        assert_eq!(face.height_at_local(512, 512), 1024);
    }

    #[test]
    fn height_at_local_respects_ne_sw_split() {
        let face = GridHorizontalFace {
            heights: [0, 1024, 2048, 1024],
            split: GridSplit::NorthEastSouthWest,
            material: MAT_FLOOR,
            walkable: true,
        };
        assert_eq!(face.height_at_local(0, 0), 0);
        assert_eq!(face.height_at_local(1024, 0), 1024);
        assert_eq!(face.height_at_local(1024, 1024), 2048);
        assert_eq!(face.height_at_local(0, 1024), 1024);
        assert_eq!(face.height_at_local(512, 512), 1024);
    }
}
