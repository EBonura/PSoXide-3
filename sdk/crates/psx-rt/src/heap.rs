//! Bump allocator gated behind the `alloc` feature.
//!
//! A tiny `GlobalAlloc` that never frees -- fine for PS1 homebrew that
//! uses a permanent arena for assets and scratch buffers. Replace
//! with a real allocator (`linked_list_allocator`, `talc`, …) when the
//! engine needs deallocation.

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;

struct BumpAllocator {
    state: UnsafeCell<BumpState>,
}

struct BumpState {
    next: usize,
    end: usize,
}

// Single-threaded environment (interrupts masked during alloc, no
// SMP on PS1) -- `Sync` is sound for the bump allocator.
unsafe impl Sync for BumpAllocator {}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let state = &mut *self.state.get();
            let align = layout.align();
            let size = layout.size();
            let aligned = (state.next + align - 1) & !(align - 1);
            let end = aligned + size;
            if end > state.end {
                return core::ptr::null_mut();
            }
            state.next = end;
            aligned as *mut u8
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator -- nothing to release until a full reset.
    }
}

#[global_allocator]
static ALLOCATOR: BumpAllocator = BumpAllocator {
    state: UnsafeCell::new(BumpState { next: 0, end: 0 }),
};

/// Seed the allocator from `start`, spanning `size` bytes.
///
/// # Safety
/// Called exactly once from [`crate::_start`] with a heap range that
/// doesn't overlap anything in use.
pub unsafe fn init(start: usize, size: usize) {
    unsafe {
        let state = &mut *ALLOCATOR.state.get();
        state.next = start;
        state.end = start + size;
    }
}
