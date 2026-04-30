//! Bounded ring buffer of recent MMIO accesses, gated behind the
//! `trace-mmio` Cargo feature.
//!
//! When the feature is off (the default), every type and method below
//! compiles down to zero-sized storage and no-op calls -- `Bus` carries
//! an empty `MmioTrace` and the hot path is untouched. When it's on,
//! `Bus` records each read/write that lands in the MMIO window so the
//! `smoke_draw` example (or any other diagnostic) can dump the tail.
//!
//! The two-layer shape (outer `MmioTrace` wrapper + inner
//! `MmioTraceInner` that only exists with the feature) lets call sites
//! stay cfg-free: they just call `bus.mmio_trace.record(...)` and the
//! compiler elides the whole call when disabled. This keeps the
//! "compile-time gated, not runtime-flagged" contract from MEMORY.md.

/// Which variant of access was recorded. Always available so call-site
/// code doesn't need to be cfg-gated; the enum carries no payload so
/// the zero-feature build still compiles.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum MmioKind {
    R8,
    R16,
    R32,
    W8,
    W16,
    W32,
}

impl MmioKind {
    /// Short 3-char tag for log dumps.
    pub fn tag(self) -> &'static str {
        match self {
            MmioKind::R8 => "r8 ",
            MmioKind::R16 => "r16",
            MmioKind::R32 => "r32",
            MmioKind::W8 => "w8 ",
            MmioKind::W16 => "w16",
            MmioKind::W32 => "w32",
        }
    }
}

/// One recorded MMIO access.
#[cfg(feature = "trace-mmio")]
#[derive(Copy, Clone, Debug)]
pub struct MmioEntry {
    /// Cumulative CPU cycles at the moment of the access.
    pub cycle: u64,
    /// Read vs. write, and byte width.
    pub kind: MmioKind,
    /// Physical address.
    pub addr: u32,
    /// Value read (for reads) or written (for writes). Padded to u32.
    pub value: u32,
}

/// Ring-buffer handle exposed on `Bus`. No-op shell without the feature.
#[derive(Default)]
pub struct MmioTrace {
    #[cfg(feature = "trace-mmio")]
    inner: MmioTraceInner,
}

impl MmioTrace {
    /// Ring capacity when the feature is enabled. 8192 entries × ~24 B
    /// ≈ 200 KiB -- small enough to carry around, large enough to catch
    /// a few BIOS event-loop iterations.
    #[cfg(feature = "trace-mmio")]
    pub const CAPACITY: usize = 8192;

    /// Build an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an access. Collapses to `{}` when `trace-mmio` is off.
    #[inline]
    pub fn record(&mut self, cycle: u64, kind: MmioKind, addr: u32, value: u32) {
        #[cfg(feature = "trace-mmio")]
        self.inner.push(cycle, kind, addr, value);
        #[cfg(not(feature = "trace-mmio"))]
        {
            let _ = (cycle, kind, addr, value);
        }
    }

    /// Number of recorded entries (0 without the feature).
    #[inline]
    pub fn len(&self) -> usize {
        #[cfg(feature = "trace-mmio")]
        {
            self.inner.len()
        }
        #[cfg(not(feature = "trace-mmio"))]
        {
            0
        }
    }

    /// True when no entries are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate the recorded entries in chronological order (oldest first).
    /// Returns an empty iterator without the feature.
    #[cfg(feature = "trace-mmio")]
    pub fn iter_chronological(&self) -> impl Iterator<Item = &MmioEntry> + '_ {
        self.inner.iter_chronological()
    }
}

// --- Feature-on machinery -------------------------------------------------

#[cfg(feature = "trace-mmio")]
#[derive(Default)]
struct MmioTraceInner {
    entries: Vec<MmioEntry>,
    head: usize,
    filled: bool,
}

#[cfg(feature = "trace-mmio")]
impl MmioTraceInner {
    fn push(&mut self, cycle: u64, kind: MmioKind, addr: u32, value: u32) {
        let e = MmioEntry {
            cycle,
            kind,
            addr,
            value,
        };
        if self.entries.len() < MmioTrace::CAPACITY {
            self.entries.push(e);
            self.head = self.entries.len() % MmioTrace::CAPACITY;
        } else {
            self.entries[self.head] = e;
            self.head = (self.head + 1) % MmioTrace::CAPACITY;
            self.filled = true;
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn iter_chronological(&self) -> impl Iterator<Item = &MmioEntry> + '_ {
        // Before wrap: Vec is already in insertion order.
        // After wrap: rotate at head so oldest comes first.
        let (a, b) = if self.filled {
            self.entries.split_at(self.head)
        } else {
            (&self.entries[..], &[][..])
        };
        b.iter().chain(a.iter())
    }
}

#[cfg(all(test, feature = "trace-mmio"))]
mod tests {
    use super::*;

    #[test]
    fn records_and_returns_in_order() {
        let mut t = MmioTrace::new();
        t.record(1, MmioKind::R32, 0x1F80_1070, 0xDEAD);
        t.record(2, MmioKind::W32, 0x1F80_1074, 0xBEEF);
        let v: Vec<_> = t.iter_chronological().collect();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].cycle, 1);
        assert_eq!(v[1].cycle, 2);
    }

    #[test]
    fn wraps_at_capacity() {
        let mut t = MmioTrace::new();
        // Push capacity + 10 entries; oldest should be dropped.
        for i in 0..(MmioTrace::CAPACITY + 10) as u64 {
            t.record(i, MmioKind::R8, i as u32, 0);
        }
        assert_eq!(t.len(), MmioTrace::CAPACITY);
        let first = t.iter_chronological().next().unwrap();
        assert_eq!(
            first.cycle, 10,
            "oldest 10 entries should have been dropped"
        );
        let last = t.iter_chronological().last().unwrap();
        assert_eq!(last.cycle, (MmioTrace::CAPACITY + 10 - 1) as u64);
    }
}

#[cfg(all(test, not(feature = "trace-mmio")))]
mod tests {
    use super::*;

    #[test]
    fn no_op_without_feature() {
        let mut t = MmioTrace::new();
        t.record(1, MmioKind::R32, 0x1F80_1070, 0xDEAD);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }
}
