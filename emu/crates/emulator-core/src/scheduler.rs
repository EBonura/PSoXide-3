//! Unified event scheduler — port of Redux's
//! `m_regs.interrupt` + `m_regs.intTargets` + `lowestTarget` from
//! `src/core/r3000a.h::scheduleInterrupt`.
//!
//! Background: every subsystem that needs to fire an IRQ "N cycles
//! from now" — CDROM command acks, GPU DMA completions, SPU async,
//! MDEC decode finish, etc. — registers an event with this
//! scheduler. On every branchTest-equivalent (end of each CPU
//! delay-slot retirement in `Cpu::step`) we ask the scheduler which
//! slots are due and dispatch their handlers.
//!
//! Before this module, each subsystem had its own ad-hoc timer
//! (`Bus::next_vblank_cycle`, `Bus::pending_dma_completions`,
//! `CdRom::pending`). That let subsystem timings drift
//! independently of Redux's tuned per-slot constants. Centralising
//! the queue is the groundwork for Redux-accurate per-slot cycle
//! deltas in the next sessions; this session only lays down the
//! infrastructure with no subsystems migrated yet.
//!
//! Design decisions:
//!
//! - **Fixed-slot enum, not a binary heap.** Redux has exactly 14
//!   interrupt slots — each a singleton (at most one outstanding
//!   event per subsystem-event-kind). We follow suit: `targets[16]`
//!   indexed by [`EventSlot`], + one `u32` bitmap of active slots.
//!   Enumerating the handful of active slots on each tick is
//!   O(popcount), cheaper than a heap insert/pop once you amortise
//!   across many schedule-and-cancel sequences.
//! - **`lowest_target` cache.** Early-exit for the common case
//!   where nothing's due — callers can skip the slot walk entirely
//!   with a single comparison.
//! - **No handler callbacks stored here.** Redux uses C++ function
//!   pointers; we keep the data pure and dispatch in [`Bus`] via a
//!   `match` on the returned `EventSlot`. Avoids box-dyn / lifetime
//!   noise and makes the firing order deterministic.

/// A scheduled-event slot. Names mirror Redux's `PSXINT_*` constants
/// 1:1 so cross-referencing the Redux source stays trivial. The
/// discriminants are the bit positions in [`Scheduler::active`] so
/// a slot index is `slot as u32`.
///
/// Subsystems currently don't use any of these — they still run
/// their own timers. The names / indices are stable so migrations
/// in future sessions don't churn identifiers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EventSlot {
    /// Controller / memory-card SIO0 command complete.
    Sio = 0,
    /// Link-cable SIO1.
    Sio1 = 1,
    /// CDROM command response (first- and second-response delays).
    Cdr = 2,
    /// CDROM sector-data-ready from ReadN / ReadS.
    CdRead = 3,
    /// GPU DMA (channel 2) transfer complete.
    GpuDma = 4,
    /// MDEC output DMA (channel 1) complete.
    MdecOutDma = 5,
    /// SPU DMA (channel 4) complete.
    SpuDma = 6,
    /// MDEC input DMA (channel 0) complete.
    MdecInDma = 7,
    /// GPU OTC DMA (channel 6) complete.
    GpuOtcDma = 8,
    /// CDROM DMA (channel 3) complete.
    CdrDma = 9,
    /// CDROM Play (CdlPlay / CdlStop second response).
    CdrPlay = 10,
    /// CDROM decoded-buffer interrupt.
    CdrDbuf = 11,
    /// CDROM lid-open / RESCAN_CD transitions.
    CdrLid = 12,
    /// SPU async (periodic mix callback).
    SpuAsync = 13,
    /// VBlank — not a Redux slot (Redux drives VBlank off its counter
    /// base-rate), but we wire it here so our existing VBlank
    /// scheduler can migrate in a later session without adding a
    /// sibling queue.
    VBlank = 14,
}

/// Total slots in the `targets` array / `active` bitmap. One more
/// than the highest [`EventSlot`] discriminant; the extra entry
/// keeps indexing safe if a future slot is added.
pub const SLOT_COUNT: usize = 16;

impl EventSlot {
    /// Convert a raw slot index (from e.g. bitmap iteration) back
    /// to a typed slot. Returns `None` for indices outside the
    /// defined range.
    pub fn from_index(idx: u32) -> Option<Self> {
        Some(match idx {
            0 => Self::Sio,
            1 => Self::Sio1,
            2 => Self::Cdr,
            3 => Self::CdRead,
            4 => Self::GpuDma,
            5 => Self::MdecOutDma,
            6 => Self::SpuDma,
            7 => Self::MdecInDma,
            8 => Self::GpuOtcDma,
            9 => Self::CdrDma,
            10 => Self::CdrPlay,
            11 => Self::CdrDbuf,
            12 => Self::CdrLid,
            13 => Self::SpuAsync,
            14 => Self::VBlank,
            _ => return None,
        })
    }

    /// Bitmap position — `1 << self.bit()` gives the mask.
    #[inline]
    pub fn bit(self) -> u32 {
        self as u32
    }
}

/// Singleton-per-slot event scheduler. Each slot may hold at most
/// one outstanding event; re-scheduling the same slot replaces the
/// previous deadline (matches Redux's behaviour — see
/// `scheduleInterrupt` which unconditionally overwrites
/// `m_regs.intTargets[interrupt]`).
pub struct Scheduler {
    /// Absolute bus-cycle at which each slot's event is due.
    /// Meaningful only when the slot's bit is set in `active`.
    targets: [u64; SLOT_COUNT],
    /// Bitmap of currently-pending slots.
    active: u32,
    /// Smallest `targets[i]` among active slots — cached so the
    /// common "nothing's due" path is one compare, not a walk.
    /// `u64::MAX` when no slot is active.
    lowest_target: u64,
    /// Cumulative count of [`Scheduler::schedule`] calls. Diagnostic
    /// only — lets tests + the probe CLI see how busy the queue has
    /// been without inspecting handler counts per subsystem.
    total_scheduled: u64,
    /// Cumulative count of [`Scheduler::take_due`] returns that
    /// were `Some`. Matches Redux's "events fired" quantity.
    total_fired: u64,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    /// Fresh scheduler with no pending events.
    pub const fn new() -> Self {
        Self {
            targets: [0; SLOT_COUNT],
            active: 0,
            lowest_target: u64::MAX,
            total_scheduled: 0,
            total_fired: 0,
        }
    }

    /// Register an event for `slot` to fire at `now + delta` bus
    /// cycles. Replaces any previous pending event for the same
    /// slot (the old deadline is discarded without firing).
    ///
    /// Mirrors Redux's `scheduleInterrupt(interrupt, eCycle)`. `now`
    /// is the current bus-cycle count (caller passes `bus.cycles`).
    pub fn schedule(&mut self, slot: EventSlot, now: u64, delta: u64) {
        let target = now.saturating_add(delta);
        self.targets[slot as usize] = target;
        self.active |= 1 << slot.bit();
        if target < self.lowest_target {
            self.lowest_target = target;
        }
        self.total_scheduled = self.total_scheduled.saturating_add(1);
    }

    /// Is this slot currently pending?
    #[inline]
    pub fn is_pending(&self, slot: EventSlot) -> bool {
        self.active & (1 << slot.bit()) != 0
    }

    /// Bitmap of all currently-pending slots. Diagnostic.
    #[inline]
    pub fn pending_bitmap(&self) -> u32 {
        self.active
    }

    /// Fetch the scheduled deadline for `slot`, or `None` if the
    /// slot isn't pending.
    pub fn target(&self, slot: EventSlot) -> Option<u64> {
        if self.is_pending(slot) {
            Some(self.targets[slot as usize])
        } else {
            None
        }
    }

    /// Cancel any pending event for `slot`. No-op if nothing was
    /// scheduled. Callers use this when a subsystem's state
    /// transition invalidates an in-flight event (e.g. CDROM
    /// Pause cancels the pending sector-ready IRQ).
    pub fn cancel(&mut self, slot: EventSlot) {
        let mask = 1 << slot.bit();
        if self.active & mask != 0 {
            self.active &= !mask;
            self.recompute_lowest();
        }
    }

    /// Remove and return `slot` when its target is `<= now`.
    ///
    /// Most Redux scheduled interrupts are strict (`target < cycle`),
    /// which is what [`Scheduler::take_due`] models. Root counters are
    /// different: `branchTest` calls `Counters::update()` when
    /// `cycle >= m_psxNextCounter`. VBlank is currently represented as
    /// a scheduler slot in PSoXide, so the bus uses this helper to keep
    /// counter/VBlank timing inclusive without weakening DMA/CDROM
    /// interrupt timing.
    pub fn take_slot_due_inclusive(&mut self, slot: EventSlot, now: u64) -> Option<u64> {
        let mask = 1 << slot.bit();
        if self.active & mask == 0 {
            return None;
        }
        let target = self.targets[slot as usize];
        if target > now {
            return None;
        }
        self.active &= !mask;
        self.recompute_lowest();
        self.total_fired = self.total_fired.saturating_add(1);
        Some(target)
    }

    /// Remove and return the earliest-deadline slot whose target is
    /// `< now`, along with that original target cycle. `None` if
    /// nothing's due. Callers invoke this in a loop to drain every
    /// due event on each tick; the returned target is what a
    /// periodic handler (VBlank, SPU async) uses to reschedule its
    /// *next* event — scheduling from `now` instead would drift the
    /// period every time the drain lagged the target.
    ///
    /// Returning slots in earliest-target order (not lowest-bit
    /// order) matches Redux's `branchTest`, which uses
    /// `lowestTarget` to pick the next event — important when two
    /// events share a target cycle and the handlers interact.
    pub fn take_due(&mut self, now: u64) -> Option<(EventSlot, u64)> {
        // Redux's `branchTest` only enters the per-slot walk when
        // `lowestTarget < cycle`, not `<=`. That one-cycle strictness
        // matters: a DMA scheduled for cycle 46247457 must *not*
        // latch its IRQ on the exact branch-test where the CPU cycle
        // first equals 46247457. It becomes visible on the next
        // branch-test after the CPU has moved past the target.
        if self.active == 0 || now <= self.lowest_target {
            return None;
        }

        // Find the slot with the smallest target that's <= now.
        let mut best_idx: Option<u32> = None;
        let mut best_target = u64::MAX;
        let mut bits = self.active;
        while bits != 0 {
            let idx = bits.trailing_zeros();
            let target = self.targets[idx as usize];
            if target <= now && target < best_target {
                best_target = target;
                best_idx = Some(idx);
            }
            bits &= bits - 1; // clear lowest set bit
        }

        let idx = best_idx?;
        self.active &= !(1 << idx);
        self.recompute_lowest();
        self.total_fired = self.total_fired.saturating_add(1);
        EventSlot::from_index(idx).map(|s| (s, best_target))
    }

    /// Like [`Scheduler::take_due`], but ignores slots whose bits are
    /// set in `excluded_mask`. Used by the per-instruction bias tick
    /// for events that Redux only services from `branchTest`.
    pub fn take_due_excluding(&mut self, now: u64, excluded_mask: u32) -> Option<(EventSlot, u64)> {
        let active = self.active & !excluded_mask;
        if active == 0 {
            return None;
        }

        let mut best_idx: Option<u32> = None;
        let mut best_target = u64::MAX;
        let mut bits = active;
        while bits != 0 {
            let idx = bits.trailing_zeros();
            let target = self.targets[idx as usize];
            if target <= now && target < best_target {
                best_target = target;
                best_idx = Some(idx);
            }
            bits &= bits - 1;
        }
        if now <= best_target {
            return None;
        }

        let idx = best_idx?;
        self.active &= !(1 << idx);
        self.recompute_lowest();
        self.total_fired = self.total_fired.saturating_add(1);
        EventSlot::from_index(idx).map(|s| (s, best_target))
    }

    /// Look at what the scheduler would fire next, without removing
    /// it. Useful for tests and diagnostic printouts.
    pub fn peek_due(&self, now: u64) -> Option<EventSlot> {
        if self.active == 0 || now <= self.lowest_target {
            return None;
        }
        let mut best_idx: Option<u32> = None;
        let mut best_target = u64::MAX;
        let mut bits = self.active;
        while bits != 0 {
            let idx = bits.trailing_zeros();
            let target = self.targets[idx as usize];
            if target <= now && target < best_target {
                best_target = target;
                best_idx = Some(idx);
            }
            bits &= bits - 1;
        }
        best_idx.and_then(EventSlot::from_index)
    }

    /// Next scheduled target across all active slots. `u64::MAX` if
    /// nothing's pending. Matches Redux's `m_regs.lowestTarget` —
    /// handy for a future "fast-forward to next event" optimisation
    /// in the step loop.
    #[inline]
    pub fn lowest_target(&self) -> u64 {
        self.lowest_target
    }

    /// Cumulative count of scheduled events since construction.
    /// Diagnostic.
    pub fn total_scheduled(&self) -> u64 {
        self.total_scheduled
    }

    /// Cumulative count of fired events since construction.
    pub fn total_fired(&self) -> u64 {
        self.total_fired
    }

    fn recompute_lowest(&mut self) {
        if self.active == 0 {
            self.lowest_target = u64::MAX;
            return;
        }
        let mut lowest = u64::MAX;
        let mut bits = self.active;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            let target = self.targets[idx];
            if target < lowest {
                lowest = target;
            }
            bits &= bits - 1;
        }
        self.lowest_target = lowest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_scheduler_has_nothing_pending() {
        let s = Scheduler::new();
        assert_eq!(s.pending_bitmap(), 0);
        assert_eq!(s.lowest_target(), u64::MAX);
        assert!(s.peek_due(0).is_none());
        assert!(s.peek_due(u64::MAX).is_none());
    }

    #[test]
    fn schedule_marks_slot_pending_and_records_target() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::VBlank, 100, 500);
        assert!(s.is_pending(EventSlot::VBlank));
        assert_eq!(s.target(EventSlot::VBlank), Some(600));
        assert_eq!(s.lowest_target(), 600);
        assert_eq!(s.total_scheduled(), 1);
    }

    #[test]
    fn take_due_before_deadline_returns_none() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::Cdr, 100, 500);
        assert!(s.take_due(599).is_none());
        // Still pending.
        assert!(s.is_pending(EventSlot::Cdr));
    }

    #[test]
    fn take_due_strictly_after_deadline_fires_and_clears() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::Cdr, 100, 500);
        assert!(
            s.take_due(600).is_none(),
            "exact target must wait one branchTest"
        );
        assert_eq!(s.take_due(601), Some((EventSlot::Cdr, 600)));
        assert!(!s.is_pending(EventSlot::Cdr));
        // Draining again returns None.
        assert!(s.take_due(601).is_none());
        assert_eq!(s.total_fired(), 1);
    }

    #[test]
    fn multiple_slots_fire_in_target_order_not_bit_order() {
        // Schedule GpuDma (bit 4) first with a later target, then
        // Sio (bit 0) with an earlier one. take_due should return
        // Sio first because its target is smaller.
        let mut s = Scheduler::new();
        s.schedule(EventSlot::GpuDma, 100, 1000);
        s.schedule(EventSlot::Sio, 100, 200);
        assert_eq!(s.take_due(5000), Some((EventSlot::Sio, 300)));
        assert_eq!(s.take_due(5000), Some((EventSlot::GpuDma, 1100)));
        assert!(s.take_due(5000).is_none());
    }

    #[test]
    fn re_scheduling_same_slot_replaces_deadline() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::CdRead, 100, 1000);
        s.schedule(EventSlot::CdRead, 100, 500);
        assert_eq!(s.target(EventSlot::CdRead), Some(600));
        assert_eq!(s.lowest_target(), 600);
    }

    #[test]
    fn cancel_removes_slot_and_recomputes_lowest() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::Sio, 0, 100);
        s.schedule(EventSlot::Cdr, 0, 200);
        assert_eq!(s.lowest_target(), 100);
        s.cancel(EventSlot::Sio);
        assert!(!s.is_pending(EventSlot::Sio));
        assert_eq!(s.lowest_target(), 200);
    }

    #[test]
    fn cancel_of_non_pending_slot_is_noop() {
        let mut s = Scheduler::new();
        s.cancel(EventSlot::Cdr);
        assert_eq!(s.pending_bitmap(), 0);
    }

    #[test]
    fn lowest_target_collapses_to_max_when_drained() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::VBlank, 0, 100);
        s.take_due(101);
        assert_eq!(s.lowest_target(), u64::MAX);
    }

    #[test]
    fn peek_due_does_not_mutate() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::GpuDma, 0, 50);
        assert!(
            s.peek_due(50).is_none(),
            "exact target must still look pending"
        );
        assert_eq!(s.peek_due(100), Some(EventSlot::GpuDma));
        // Still pending after peek.
        assert!(s.is_pending(EventSlot::GpuDma));
    }

    #[test]
    fn from_index_round_trips_all_defined_slots() {
        for raw in 0..=14u32 {
            let slot = EventSlot::from_index(raw).unwrap();
            assert_eq!(slot.bit(), raw);
        }
        assert!(EventSlot::from_index(15).is_none());
        assert!(EventSlot::from_index(99).is_none());
    }

    #[test]
    fn simultaneous_deadlines_both_fire() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::Sio, 0, 100);
        s.schedule(EventSlot::VBlank, 0, 100);
        assert!(s.take_due(100).is_none(), "exact target must not fire yet");
        let first = s.take_due(101);
        let second = s.take_due(101);
        let third = s.take_due(101);
        assert!(first.is_some() && second.is_some());
        assert!(third.is_none());
        // Order is irrelevant — both should have fired, and both
        // should report target 100.
        let mut fired = [first.unwrap().0, second.unwrap().0];
        fired.sort_by_key(|s| s.bit());
        assert_eq!(fired, [EventSlot::Sio, EventSlot::VBlank]);
        assert_eq!(first.unwrap().1, 100);
        assert_eq!(second.unwrap().1, 100);
    }

    #[test]
    fn take_due_reports_original_target_for_periodic_reschedule() {
        // VBlank + SPU async need to reschedule from the *original*
        // target so long drain lags don't accumulate drift. If we
        // fire at now=700 but the target was 500, the next period
        // should start at 500, not 700.
        let mut s = Scheduler::new();
        s.schedule(EventSlot::VBlank, 0, 500);
        let fired = s.take_due(700).unwrap();
        assert_eq!(fired.0, EventSlot::VBlank);
        assert_eq!(fired.1, 500); // original target, not now=700
    }

    #[test]
    fn saturating_schedule_on_large_delta_does_not_wrap() {
        let mut s = Scheduler::new();
        s.schedule(EventSlot::SpuAsync, u64::MAX - 10, 100);
        assert_eq!(s.target(EventSlot::SpuAsync), Some(u64::MAX));
    }
}
