//! Ordering Table: depth-sorted linked-list of GPU primitives.
//!
//! The PS1 has no Z-buffer. Games sort primitives back-to-front
//! (painter's algorithm) by inserting them into an OT slot indexed
//! by depth. Each OT slot is the head of a linked list; primitives
//! prepend themselves so the most-recently-inserted draws first
//! within a slot.
//!
//! Once a frame's primitives are inserted, the whole OT is shipped
//! to GPU GP0 via DMA channel 2 in linked-list mode. The DMA walker
//! follows the `next` pointers embedded in each packet's first word
//! until it hits `0x00FFFFFF` (end of chain).
//!
//! Each 32-bit OT entry (and primitive header) is:
//!
//! ```text
//!   bits 0..=23: address of next packet (24-bit, masked into RAM)
//!   bits 24..=31: word count (0..=15) of this packet's data
//! ```
//!
//! An "empty OT" has every entry pointing at its predecessor,
//! ending in `0x00FFFFFF`. Submitting such an OT sends nothing to
//! GP0. As primitives are added, their packets chain in.

use core::ptr;

/// Fixed-size OT. `N` depth slots. Typical values: 256, 1024, 4096.
#[repr(C, align(4))]
pub struct OrderingTable<const N: usize> {
    entries: [u32; N],
}

impl<const N: usize> OrderingTable<N> {
    /// Create a table with every slot being a chain terminator.
    /// Call [`clear`](Self::clear) before submitting — that wires
    /// up the inter-slot chain so DMA walks across all `N` slots.
    pub const fn new() -> Self {
        Self {
            entries: [0x00FF_FFFF; N],
        }
    }

    /// Reset every slot for a fresh frame. Entry `[0]` is the
    /// terminator (farthest from camera); each higher slot points
    /// to the slot below. Submission starts at `[N-1]` so the
    /// DMA walker visits `[N-1] → [N-2] → … → [0] → end`.
    pub fn clear(&mut self) {
        // Slot 0 is the sentinel; chain walks stop here.
        self.entries[0] = 0x00FF_FFFF;
        for i in 1..N {
            let prev = &self.entries[i - 1] as *const u32 as u32 & 0x00FF_FFFF;
            self.entries[i] = prev;
        }
    }

    /// Prepend a primitive packet into the depth-`z` slot. `packet_ptr`
    /// must point at the packet's tag word (first `u32`); `words` is
    /// the count of data words that follow the tag (≤ 15).
    ///
    /// # Safety
    /// Caller guarantees that `[packet_ptr .. packet_ptr + 1 + words]`
    /// is live, writable, 4-byte-aligned RAM for the duration of the
    /// OT submission. Primitives returned by the builders in
    /// [`crate::prim`] satisfy this.
    pub unsafe fn insert(&mut self, z: usize, packet_ptr: *mut u32, words: u8) {
        let z = z.min(N - 1);
        let old_head = self.entries[z] & 0x00FF_FFFF;
        let tag = ((words as u32) << 24) | old_head;
        unsafe { ptr::write_volatile(packet_ptr, tag) };
        let pkt_addr = packet_ptr as u32 & 0x00FF_FFFF;
        self.entries[z] = pkt_addr;
    }

    /// Insert a primitive struct. The struct must be `#[repr(C)]`
    /// with its first field being the tag `u32`. `words` is the
    /// number of data words that follow the tag.
    pub fn add<T>(&mut self, z: usize, prim: &mut T, words: u8) {
        unsafe { self.insert(z, prim as *mut T as *mut u32, words) };
    }

    /// Pointer to the slot where DMA starts (`[N-1]`). Passed to
    /// [`submit_via_dma`] as the linked-list entry point.
    #[inline]
    pub fn submit_head(&self) -> *const u32 {
        &self.entries[N - 1] as *const u32
    }

    /// Submit the whole table to GPU via DMA channel 2 linked-list
    /// mode and wait for completion. Forwards to
    /// [`crate::submit_linked_list`].
    pub fn submit(&self) {
        crate::submit_linked_list(self.submit_head());
    }
}

impl<const N: usize> Default for OrderingTable<N> {
    fn default() -> Self {
        Self::new()
    }
}
