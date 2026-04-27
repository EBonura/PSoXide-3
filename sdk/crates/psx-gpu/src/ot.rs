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

    /// Walk the linked chain in DMA submission order, producing one
    /// `(packet_ptr, words)` pair per primitive packet.
    ///
    /// Used by the editor's host-side preview to convert an OT into a
    /// `psx-gpu-render` command log without DMAing through real
    /// hardware. The hardware DMA walker follows the same pointers in
    /// the same order, so the iterator output is bit-equivalent to
    /// what the GPU would consume.
    ///
    /// # Safety
    /// Every chained packet must be live for the lifetime of the
    /// returned iterator — exactly the same invariant `submit()`
    /// requires. Primitives produced by [`crate::prim::*`] paired with
    /// a `PrimitiveArena` satisfy this; bespoke chains must guarantee
    /// the same.
    pub unsafe fn iter_packets(&self) -> OtPacketIter {
        // The submit head holds the address of the first chained
        // packet (its low 24 bits). PS1 hardware masks to 24 bits
        // because RAM is 2 MB and packet pointers can omit the high
        // byte. On host the same masking still recovers the address
        // because all OT-chained primitives live in the same arena
        // whose pointer fits in 24 bits relative to a stable base —
        // [`PrimitiveArena`] enforces that.
        OtPacketIter {
            next: self.entries[N - 1] & 0x00FF_FFFF,
            base_high: (self.submit_head() as usize) & !0x00FF_FFFF,
        }
    }
}

/// Walks an [`OrderingTable`]'s chain in DMA submission order.
///
/// Each `next()` returns the pointer to a packet and the number of
/// data words that follow its tag (so the full packet occupies
/// `1 + words` u32s starting at the returned pointer). The terminal
/// `0x00FFFFFF` marker stops iteration cleanly.
pub struct OtPacketIter {
    next: u32,
    base_high: usize,
}

impl Iterator for OtPacketIter {
    type Item = (*const u32, u8);

    fn next(&mut self) -> Option<Self::Item> {
        // Walk the chain, skipping empty stepping-stones — OT slots
        // that hold `words=0` because they were never targeted by an
        // `insert`. The DMA hardware silently no-ops through those
        // and only forwards entries with actual packet data, so this
        // iterator presents the same view to the cmd-log adapter.
        loop {
            if self.next == 0x00FF_FFFF {
                return None;
            }
            let ptr = (self.base_high | self.next as usize) as *const u32;
            // SAFETY: ptr was reached by walking the chain that
            // `iter_packets`'s caller swore was live; tag word is
            // always present in any chained slot.
            let tag = unsafe { ptr::read_volatile(ptr) };
            let words = ((tag >> 24) & 0xFF) as u8;
            self.next = tag & 0x00FF_FFFF;
            if words > 0 {
                return Some((ptr, words));
            }
        }
    }
}

impl<const N: usize> Default for OrderingTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(target_arch = "mips")))]
mod tests {
    use super::*;

    /// Build a primitive packet by hand (one tag word + N data words),
    /// insert it, and walk the chain. The iterator must report the
    /// same `(ptr, words)` pair we inserted.
    #[test]
    fn iter_packets_walks_a_single_inserted_primitive() {
        let mut ot: OrderingTable<8> = OrderingTable::new();
        ot.clear();
        // Packet layout: [tag, w0, w1, w2] — 3 data words after the tag.
        let mut packet: [u32; 4] = [0; 4];
        packet[1] = 0xAAAA_BBBB;
        packet[2] = 0xCCCC_DDDD;
        packet[3] = 0xEEEE_FFFF;
        unsafe {
            ot.insert(2, packet.as_mut_ptr(), 3);
        }

        let mut iter = unsafe { ot.iter_packets() };
        let entry = iter.next().expect("one entry");
        assert_eq!(entry.0 as usize, packet.as_ptr() as usize);
        assert_eq!(entry.1, 3);
        assert!(iter.next().is_none());
    }

    /// Two primitives in different slots — chain walks both; later
    /// inserts (lower slot) come first because `clear()` chains
    /// high-to-low and the DMA head is `[N-1]`.
    #[test]
    fn iter_packets_walks_multiple_slots_in_dma_order() {
        let mut ot: OrderingTable<8> = OrderingTable::new();
        ot.clear();
        let mut a: [u32; 2] = [0, 0xA];
        let mut b: [u32; 2] = [0, 0xB];
        unsafe {
            // a is in a deeper (further from camera) slot than b, so b
            // should appear first when walking from the head.
            ot.insert(2, a.as_mut_ptr(), 1);
            ot.insert(5, b.as_mut_ptr(), 1);
        }

        let mut iter = unsafe { ot.iter_packets() };
        // DMA walker starts at [N-1] = [7] and chains down to [0].
        // b lives in slot 5, a in slot 2 — both should appear, b first.
        let first = iter.next().expect("first entry").0 as usize;
        let second = iter.next().expect("second entry").0 as usize;
        assert!(iter.next().is_none());
        assert_eq!(first, b.as_ptr() as usize);
        assert_eq!(second, a.as_ptr() as usize);
    }

    /// Multiple primitives in the same slot chain via the most-
    /// recently-inserted-first rule.
    #[test]
    fn iter_packets_chains_primitives_within_one_slot() {
        let mut ot: OrderingTable<4> = OrderingTable::new();
        ot.clear();
        let mut first: [u32; 2] = [0, 0x1111];
        let mut second: [u32; 2] = [0, 0x2222];
        unsafe {
            ot.insert(1, first.as_mut_ptr(), 1);
            ot.insert(1, second.as_mut_ptr(), 1);
        }

        let mut iter = unsafe { ot.iter_packets() };
        // `second` was inserted last and prepends to the chain head;
        // it walks first.
        let head = iter.next().expect("first").0 as usize;
        let tail = iter.next().expect("second").0 as usize;
        assert!(iter.next().is_none());
        assert_eq!(head, second.as_ptr() as usize);
        assert_eq!(tail, first.as_ptr() as usize);
    }
}
