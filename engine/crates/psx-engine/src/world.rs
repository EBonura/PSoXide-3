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

/// One cooked grid room — **authoring / test helper**, not the
/// resident PSX runtime format.
///
/// The room body holds `&[Option<GridSector<'a>>]` where each
/// `GridSector<'a>` further holds six borrowed slices. Convenient
/// for tests that build a world in static const data and for
/// engine-side code that wants direct access; **but six-pointer
/// pre-decoded sectors are not what we want resident in PSX
/// memory at scale**. The PSX target shape is `psx_asset::World<'a>`
/// — flat byte tables decoded by-value on demand. Don't grow new
/// runtime systems on top of `GridRoom`; build them on
/// `psx_asset::World`.
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
///
/// Same authoring / test caveat as [`GridRoom`]: this is the
/// engine-side helper, not the PSX-resident shape. PSX-resident
/// world data is `psx_asset::World<'a>` (one room) plus a thin
/// runtime wrapper — see [`RuntimeRoom`].
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

/// PSX-resident wrapper over a parsed `.psxw` blob.
///
/// Compared with [`GridRoom`], this type holds **only** the
/// zero-copy `psx_asset::World<'a>` view — no pre-decoded sector
/// array, no `Option<GridSector>` slice, no per-sector borrows.
/// Sectors and walls decode by value on demand:
///
/// ```ignore
/// let blob: &[u8] = include_bytes!("level1.psxw");
/// let room = RuntimeRoom::from_bytes(blob)?;
/// for x in 0..room.width() {
///     for z in 0..room.depth() {
///         if let Some(sector) = room.sector(x, z) {
///             // …decode walls by value:
///             for i in 0..sector.wall_count() {
///                 if let Some(wall) = room.sector_wall(sector, i) {
///                     // …
///                 }
///             }
///         }
///     }
/// }
/// ```
///
/// New runtime systems (collision, rendering, AI floor sampling)
/// should grow on this type rather than `GridRoom` — `GridRoom`
/// stays for tests and authoring helpers only.
#[derive(Copy, Clone, Debug)]
pub struct RuntimeRoom<'a> {
    inner: psx_asset::World<'a>,
}

impl<'a> RuntimeRoom<'a> {
    /// Parse a cooked `.psxw` blob into a runtime view.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, psx_asset::ParseError> {
        Ok(Self {
            inner: psx_asset::World::from_bytes(bytes)?,
        })
    }

    /// Wrap an already-parsed `World`. Used when the parse and
    /// the engine-side wrap happen in different layers.
    pub const fn from_world(world: psx_asset::World<'a>) -> Self {
        Self { inner: world }
    }

    /// Underlying byte-level view. Borrow it when you need the
    /// raw slice access (validation, debug dump, …).
    pub const fn world(&self) -> &psx_asset::World<'a> {
        &self.inner
    }

    /// Width in grid sectors.
    pub fn width(&self) -> u16 {
        self.inner.width()
    }

    /// Depth in grid sectors.
    pub fn depth(&self) -> u16 {
        self.inner.depth()
    }

    /// Engine units per sector.
    pub fn sector_size(&self) -> i32 {
        self.inner.sector_size()
    }

    /// Sector by `(x, z)` cell index, or `None` for empty cells
    /// or out-of-range coords.
    pub fn sector(&self, x: u16, z: u16) -> Option<psx_asset::WorldSector> {
        self.inner.sector(x, z)
    }

    /// Wall record by sector-local wall index. Skip the array-
    /// decode dance the caller would otherwise do over
    /// `sector.first_wall + i`.
    pub fn sector_wall(
        &self,
        sector: psx_asset::WorldSector,
        local_index: u16,
    ) -> Option<psx_asset::WorldWall> {
        self.inner.sector_wall(sector, local_index)
    }

    /// Render-side facade, see [`RoomRender`].
    pub const fn render(&self) -> RoomRender<'a, '_> {
        RoomRender { room: self }
    }

    /// Collision-side facade, see [`RoomCollision`].
    pub const fn collision(&self) -> RoomCollision<'a, '_> {
        RoomCollision { room: self }
    }
}

// ============================================================
// Render-vs-collision facades
// ============================================================
//
// Tomb Raider runs render and collision off the same on-disk
// room data, but with two distinct read paths: the renderer
// walks tr_face4 / tr_face3 lists with materials and lighting;
// collision walks tr_room_sector heights and traversal portals
// with no concept of texture pages. The two systems literally
// cannot fetch each other's fields at the API level.
//
// `RoomRender` / `RoomCollision` give us the same discipline
// over a single `RuntimeRoom`. Both views are zero-cost
// `Copy` borrows; the v1 byte format keeps render + collision
// fields in one record so today's cooker writes both streams
// in one pass — but a caller that says
// `room.render().sector(...)` cannot accidentally branch on
// `floor_walkable`, and a caller that says
// `room.collision().sector(...)` cannot accidentally read a
// material slot.

/// Render-side view over a [`RuntimeRoom`].
///
/// Exposes only the fields a draw pass cares about: heights and
/// splits for vertex emission, materials for tpage / clut lookup,
/// world-level lighting state. Collision-only state
/// (`walkable`, `solid`, traversal portals) is intentionally
/// **not** reachable through this view.
#[derive(Copy, Clone, Debug)]
pub struct RoomRender<'a, 'b> {
    room: &'b RuntimeRoom<'a>,
}

impl<'a, 'b> RoomRender<'a, 'b> {
    /// Width in grid sectors.
    pub fn width(self) -> u16 {
        self.room.width()
    }

    /// Depth in grid sectors.
    pub fn depth(self) -> u16 {
        self.room.depth()
    }

    /// Engine units per sector.
    pub fn sector_size(self) -> i32 {
        self.room.sector_size()
    }

    /// Room ambient RGB color.
    pub fn ambient_color(self) -> [u8; 3] {
        self.room.world().ambient_color()
    }

    /// Whether fog / depth cue is enabled for this world.
    pub fn fog_enabled(self) -> bool {
        self.room.world().fog_enabled()
    }

    /// Sector at `(x, z)` for render purposes, or `None` for
    /// empty cells.
    pub fn sector(self, x: u16, z: u16) -> Option<SectorRender> {
        self.room.sector(x, z).map(SectorRender)
    }

    /// Wall record by sector-local index, render view.
    pub fn sector_wall(self, sector: SectorRender, local_index: u16) -> Option<WallRender> {
        self.room.sector_wall(sector.0, local_index).map(WallRender)
    }
}

/// Collision-side view over a [`RuntimeRoom`].
///
/// Exposes only the fields a movement / floor-sample query
/// cares about: heights for surface sampling, splits for
/// triangulation of the height grid, walkable / solid bits for
/// stop-or-pass decisions. Render-only state (materials,
/// lighting, fog) is intentionally **not** reachable through
/// this view.
#[derive(Copy, Clone, Debug)]
pub struct RoomCollision<'a, 'b> {
    room: &'b RuntimeRoom<'a>,
}

impl<'a, 'b> RoomCollision<'a, 'b> {
    /// Width in grid sectors.
    pub fn width(self) -> u16 {
        self.room.width()
    }

    /// Depth in grid sectors.
    pub fn depth(self) -> u16 {
        self.room.depth()
    }

    /// Engine units per sector.
    pub fn sector_size(self) -> i32 {
        self.room.sector_size()
    }

    /// Sector at `(x, z)` for collision purposes, or `None` for
    /// empty cells.
    pub fn sector(self, x: u16, z: u16) -> Option<SectorCollision> {
        self.room.sector(x, z).map(SectorCollision)
    }

    /// Wall record by sector-local index, collision view.
    pub fn sector_wall(self, sector: SectorCollision, local_index: u16) -> Option<WallCollision> {
        self.room
            .sector_wall(sector.0, local_index)
            .map(WallCollision)
    }
}

/// Render-side projection of one decoded sector.
#[derive(Copy, Clone, Debug)]
pub struct SectorRender(psx_asset::WorldSector);

impl SectorRender {
    /// `true` if this sector emits a floor face.
    pub fn has_floor(self) -> bool {
        self.0.has_floor()
    }

    /// `true` if this sector emits a ceiling face.
    pub fn has_ceiling(self) -> bool {
        self.0.has_ceiling()
    }

    /// Floor diagonal split id.
    pub fn floor_split(self) -> u8 {
        self.0.floor_split()
    }

    /// Ceiling diagonal split id.
    pub fn ceiling_split(self) -> u8 {
        self.0.ceiling_split()
    }

    /// Floor material slot, if any.
    pub fn floor_material(self) -> Option<u16> {
        self.0.floor_material()
    }

    /// Ceiling material slot, if any.
    pub fn ceiling_material(self) -> Option<u16> {
        self.0.ceiling_material()
    }

    /// Floor corner heights `[NW, NE, SE, SW]` for vertex emission.
    pub fn floor_heights(self) -> [i32; 4] {
        self.0.floor_heights()
    }

    /// Ceiling corner heights `[NW, NE, SE, SW]` for vertex emission.
    pub fn ceiling_heights(self) -> [i32; 4] {
        self.0.ceiling_heights()
    }

    /// First global wall index for this sector.
    pub fn first_wall(self) -> u16 {
        self.0.first_wall()
    }

    /// Number of walls belonging to this sector.
    pub fn wall_count(self) -> u16 {
        self.0.wall_count()
    }
}

/// Collision-side projection of one decoded sector.
#[derive(Copy, Clone, Debug)]
pub struct SectorCollision(psx_asset::WorldSector);

impl SectorCollision {
    /// `true` if this sector has a floor surface to sample.
    pub fn has_floor(self) -> bool {
        self.0.has_floor()
    }

    /// `true` if this sector has a ceiling surface for clearance.
    pub fn has_ceiling(self) -> bool {
        self.0.has_ceiling()
    }

    /// `true` if the floor face is walkable.
    pub fn floor_walkable(self) -> bool {
        self.0.floor_walkable()
    }

    /// Floor diagonal split id (decides the triangulation used
    /// to interpolate height samples).
    pub fn floor_split(self) -> u8 {
        self.0.floor_split()
    }

    /// Ceiling diagonal split id.
    pub fn ceiling_split(self) -> u8 {
        self.0.ceiling_split()
    }

    /// Floor corner heights `[NW, NE, SE, SW]`.
    pub fn floor_heights(self) -> [i32; 4] {
        self.0.floor_heights()
    }

    /// Ceiling corner heights `[NW, NE, SE, SW]`.
    pub fn ceiling_heights(self) -> [i32; 4] {
        self.0.ceiling_heights()
    }

    /// First global wall index for this sector.
    pub fn first_wall(self) -> u16 {
        self.0.first_wall()
    }

    /// Number of walls belonging to this sector.
    pub fn wall_count(self) -> u16 {
        self.0.wall_count()
    }
}

/// Render-side projection of one decoded wall.
#[derive(Copy, Clone, Debug)]
pub struct WallRender(psx_asset::WorldWall);

impl WallRender {
    /// Direction id.
    pub fn direction(self) -> u8 {
        self.0.direction()
    }

    /// Material slot.
    pub fn material(self) -> u16 {
        self.0.material()
    }

    /// Wall heights `[bottom-left, bottom-right, top-right, top-left]`.
    pub fn heights(self) -> [i32; 4] {
        self.0.heights()
    }
}

/// Collision-side projection of one decoded wall.
#[derive(Copy, Clone, Debug)]
pub struct WallCollision(psx_asset::WorldWall);

impl WallCollision {
    /// Direction id.
    pub fn direction(self) -> u8 {
        self.0.direction()
    }

    /// `true` when this wall blocks character movement.
    pub fn solid(self) -> bool {
        self.0.solid()
    }

    /// Wall heights `[bottom-left, bottom-right, top-right, top-left]`
    /// for slab-vs-character clearance checks.
    pub fn heights(self) -> [i32; 4] {
        self.0.heights()
    }
}

// Compile-time guarantee that `RuntimeRoom` and its render /
// collision facades stay zero-allocation `Copy` types. Any
// future change that adds an owned field (Vec, String, …) will
// break the build here, which is the whole point.
const _: () = {
    const fn _assert_copy<T: Copy>() {}
    _assert_copy::<RuntimeRoom<'static>>();
    _assert_copy::<RoomRender<'static, 'static>>();
    _assert_copy::<RoomCollision<'static, 'static>>();
    _assert_copy::<SectorRender>();
    _assert_copy::<SectorCollision>();
    _assert_copy::<WallRender>();
    _assert_copy::<WallCollision>();
};

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
