//! Render helpers built around PS1 ordering tables.
//!
//! The SDK-level [`psx_gpu::ot::OrderingTable`] is intentionally a
//! thin hardware wrapper. This module adds the engine-facing shape:
//! begin a frame, add primitives at depth slots, submit once, and use
//! fixed backing arenas for primitive packets without depending on an
//! allocator.

use psx_gpu::{
    ot::OrderingTable,
    prim::{
        LineMono, QuadFlat, QuadGouraud, QuadTextured, RectFlat, Sprite, TriFlat, TriGouraud,
        TriTextured, TriTexturedGouraud,
    },
};

/// GPU primitive packet that can be inserted into an ordering table.
///
/// The associated `WORDS` value is the number of data words following
/// the packet tag. SDK primitive structs expose this as an inherent
/// constant; this trait lets engine render helpers use it without
/// every call site repeating the constant manually.
pub trait GpuPacket {
    /// Number of data words after the tag word.
    const WORDS: u8;
}

macro_rules! impl_gpu_packet {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl GpuPacket for $ty {
                const WORDS: u8 = <$ty>::WORDS;
            }
        )+
    };
}

impl_gpu_packet!(
    TriFlat,
    TriGouraud,
    QuadFlat,
    RectFlat,
    QuadGouraud,
    LineMono,
    TriTextured,
    TriTexturedGouraud,
    QuadTextured,
    Sprite,
);

/// A clamped ordering-table slot.
///
/// Higher slot indices are farther from the camera and are submitted
/// earlier by the PS1 linked-list DMA walk. Lower slots draw later
/// and therefore appear in front.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DepthSlot {
    index: usize,
}

impl DepthSlot {
    /// Build a depth slot from a raw index.
    ///
    /// The value is clamped by [`OtFrame::add_slot`] against the
    /// actual ordering-table depth.
    pub const fn new(index: usize) -> Self {
        Self { index }
    }

    /// Raw slot index.
    pub const fn index(self) -> usize {
        self.index
    }
}

/// Linear mapping from camera-space depth to OT slots.
///
/// `near` maps to slot `0` (front) and `far` maps to the last OT slot
/// (back). Depths outside the range clamp.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DepthRange {
    near: i32,
    far: i32,
}

impl DepthRange {
    /// Create a range where `near` is front and `far` is back.
    ///
    /// If `far <= near`, every value maps to slot `0`; this avoids
    /// division by zero and makes invalid ranges fail visually
    /// conservative.
    pub const fn new(near: i32, far: i32) -> Self {
        Self { near, far }
    }

    /// Front depth.
    pub const fn near(self) -> i32 {
        self.near
    }

    /// Back depth.
    pub const fn far(self) -> i32 {
        self.far
    }

    /// Map `depth` into an OT slot for a table with `DEPTH` slots.
    pub fn slot<const DEPTH: usize>(self, depth: i32) -> DepthSlot {
        DepthBand::whole().slot::<DEPTH>(self, depth)
    }
}

/// Inclusive subset of an ordering table reserved for one render layer.
///
/// Engines often reserve the farthest slot for backgrounds and the
/// nearest slots for overlays/effects. A band lets a scene map
/// camera-space depth into only the slots allocated to world
/// geometry, keeping those layers from fighting each other.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DepthBand {
    front: usize,
    back: usize,
}

impl DepthBand {
    /// Build an inclusive OT slot band.
    ///
    /// `front` is the nearest slot in the band; `back` is the
    /// farthest. Values are clamped by [`slot`](Self::slot) against
    /// the actual ordering-table depth.
    pub const fn new(front: usize, back: usize) -> Self {
        Self { front, back }
    }

    /// The whole ordering table.
    pub const fn whole() -> Self {
        Self {
            front: 0,
            back: usize::MAX,
        }
    }

    /// Nearest slot requested by this band.
    pub const fn front(self) -> usize {
        self.front
    }

    /// Farthest slot requested by this band.
    pub const fn back(self) -> usize {
        self.back
    }

    /// Map `depth` through `range` into this inclusive band.
    pub fn slot<const DEPTH: usize>(self, range: DepthRange, depth: i32) -> DepthSlot {
        if DEPTH == 0 {
            return DepthSlot::new(0);
        }

        let max_slot = DEPTH - 1;
        let front = self.front.min(max_slot);
        let back = self.back.min(max_slot);
        if back <= front || range.far <= range.near || depth <= range.near {
            return DepthSlot::new(front);
        }
        if depth >= range.far {
            return DepthSlot::new(back);
        }

        let span = range.far - range.near;
        let offset = depth - range.near;
        let band_slots = (back - front) as i32;
        DepthSlot::new(front + ((offset.saturating_mul(band_slots)) / span) as usize)
    }
}

/// One frame's mutable view of an ordering table.
///
/// Constructing this with [`begin`](Self::begin) clears the table for
/// the current frame. Calling [`submit`](Self::submit) consumes the
/// frame view, which keeps call sites honest: all inserts happen
/// before the DMA submission.
#[must_use = "call submit() to send the ordering table to the GPU"]
pub struct OtFrame<'a, const DEPTH: usize> {
    ot: &'a mut OrderingTable<DEPTH>,
}

impl<'a, const DEPTH: usize> OtFrame<'a, DEPTH> {
    /// Clear `ot` and begin a new frame.
    ///
    /// `DEPTH` must be greater than zero; the underlying SDK
    /// ordering table has the same requirement.
    pub fn begin(ot: &'a mut OrderingTable<DEPTH>) -> Self {
        debug_assert!(DEPTH > 0);
        ot.clear();
        Self { ot }
    }

    /// Insert a primitive at a raw OT slot.
    pub fn add<T>(&mut self, slot: usize, prim: &mut T, words: u8) {
        debug_assert!(words <= 15);
        self.ot.add(slot, prim, words);
    }

    /// Insert a raw primitive packet pointer at a raw OT slot.
    ///
    /// # Safety
    /// `packet_ptr` must point at the first word of a live GPU packet
    /// that remains writable until the ordering-table DMA has consumed
    /// it. `words` is the number of data words following the tag word.
    pub unsafe fn add_raw(&mut self, slot: usize, packet_ptr: *mut u32, words: u8) {
        debug_assert!(words <= 15);
        unsafe { self.ot.insert(slot, packet_ptr, words) };
    }

    /// Insert a primitive at a typed OT slot.
    pub fn add_slot<T>(&mut self, slot: DepthSlot, prim: &mut T, words: u8) {
        self.add(slot.index(), prim, words);
    }

    /// Insert a raw primitive packet pointer at a typed OT slot.
    ///
    /// # Safety
    /// Same requirements as [`add_raw`](Self::add_raw).
    pub unsafe fn add_raw_slot(&mut self, slot: DepthSlot, packet_ptr: *mut u32, words: u8) {
        unsafe { self.add_raw(slot.index(), packet_ptr, words) };
    }

    /// Insert a known SDK GPU packet at a raw OT slot.
    pub fn add_packet<T: GpuPacket>(&mut self, slot: usize, prim: &mut T) {
        self.add(slot, prim, T::WORDS);
    }

    /// Insert a known SDK GPU packet at a typed OT slot.
    pub fn add_packet_slot<T: GpuPacket>(&mut self, slot: DepthSlot, prim: &mut T) {
        self.add_slot(slot, prim, T::WORDS);
    }

    /// Map camera-space `depth` through `range` and insert the
    /// primitive into the resulting OT slot.
    pub fn add_depth<T>(&mut self, range: DepthRange, depth: i32, prim: &mut T, words: u8) {
        self.add_slot(range.slot::<DEPTH>(depth), prim, words);
    }

    /// Map camera-space `depth` through `range` and insert a known
    /// SDK GPU packet into the resulting OT slot.
    pub fn add_packet_depth<T: GpuPacket>(&mut self, range: DepthRange, depth: i32, prim: &mut T) {
        self.add_packet_slot(range.slot::<DEPTH>(depth), prim);
    }

    /// Submit this frame's ordering table via DMA linked-list mode.
    pub fn submit(self) {
        self.ot.submit();
    }
}

/// Fixed backing storage for primitive packets.
///
/// PS1 render packets must live in RAM until the OT DMA walk has
/// consumed them. This arena wraps caller-owned storage and writes
/// primitives sequentially each frame. Call [`reset`](Self::reset)
/// before reusing it for a new frame.
pub struct PrimitiveArena<'a, T> {
    storage: &'a mut [T],
    len: usize,
}

impl<'a, T> PrimitiveArena<'a, T> {
    /// Wrap caller-owned primitive storage.
    pub fn new(storage: &'a mut [T]) -> Self {
        Self { storage, len: 0 }
    }

    /// Clear the arena for reuse. Existing values are left in memory
    /// and will be overwritten by subsequent [`push`](Self::push)
    /// calls.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Number of primitives written this frame.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no primitives have been written this frame.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Backing storage capacity.
    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    /// Remaining primitive slots.
    pub fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.len)
    }

    /// Write `prim` and return a mutable reference suitable for OT
    /// insertion. Returns `None` if the arena is full.
    pub fn push(&mut self, prim: T) -> Option<&mut T> {
        let idx = self.push_index(prim)?;
        Some(&mut self.storage[idx])
    }

    /// Write `prim` and return its arena index.
    ///
    /// This is useful for render passes that need to build packets
    /// first, sort draw commands, and only then borrow the packets
    /// for ordering-table insertion.
    pub fn push_index(&mut self, prim: T) -> Option<usize> {
        if self.len >= self.storage.len() {
            return None;
        }
        let idx = self.len;
        self.len += 1;
        self.storage[idx] = prim;
        Some(idx)
    }

    /// Borrow a primitive previously written by [`push_index`](Self::push_index).
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        Some(&mut self.storage[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_range_clamps_and_scales() {
        let range = DepthRange::new(100, 900);
        assert_eq!(range.slot::<8>(0).index(), 0);
        assert_eq!(range.slot::<8>(100).index(), 0);
        assert_eq!(range.slot::<8>(500).index(), 3);
        assert_eq!(range.slot::<8>(899).index(), 6);
        assert_eq!(range.slot::<8>(900).index(), 7);
        assert_eq!(range.slot::<8>(1200).index(), 7);
    }

    #[test]
    fn depth_band_reserves_front_and_back_slots() {
        let range = DepthRange::new(0, 4000);
        let band = DepthBand::new(2, 5);
        assert_eq!(band.slot::<8>(range, -100).index(), 2);
        assert_eq!(band.slot::<8>(range, 0).index(), 2);
        assert_eq!(band.slot::<8>(range, 2000).index(), 3);
        assert_eq!(band.slot::<8>(range, 3999).index(), 4);
        assert_eq!(band.slot::<8>(range, 4000).index(), 5);
        assert_eq!(band.slot::<8>(range, 9000).index(), 5);
    }

    #[test]
    fn depth_band_clamps_to_table_depth() {
        let range = DepthRange::new(0, 100);
        let band = DepthBand::new(6, 99);
        assert_eq!(band.slot::<8>(range, 0).index(), 6);
        assert_eq!(band.slot::<8>(range, 100).index(), 7);
    }

    #[test]
    fn invalid_depth_range_maps_front() {
        let range = DepthRange::new(100, 100);
        assert_eq!(range.slot::<8>(500).index(), 0);
    }

    #[test]
    fn primitive_arena_pushes_until_full() {
        let mut storage = [0u16; 2];
        let mut arena = PrimitiveArena::new(&mut storage);

        assert!(arena.is_empty());
        assert_eq!(arena.capacity(), 2);
        assert_eq!(arena.remaining(), 2);

        *arena.push(7).expect("slot 0") = 8;
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.remaining(), 1);

        arena.push(9).expect("slot 1");
        assert_eq!(arena.push(10), None);
        assert_eq!(arena.len(), 2);

        arena.reset();
        assert!(arena.is_empty());
        assert_eq!(arena.remaining(), 2);
    }

    #[test]
    fn primitive_arena_can_reborrow_by_index() {
        let mut storage = [0u16; 2];
        let mut arena = PrimitiveArena::new(&mut storage);

        let idx = arena.push_index(7).expect("slot 0");
        *arena.get_mut(idx).expect("reborrow") = 12;

        assert_eq!(*arena.get_mut(idx).expect("slot still live"), 12);
        assert!(arena.get_mut(1).is_none());
    }
}
