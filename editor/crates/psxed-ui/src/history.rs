//! Snapshot-based undo / redo for the editor workspace.
//!
//! Each entry is a full [`ProjectDocument`] clone -- for
//! hand-authored level data the snapshots are cheap and avoid
//! the command-pattern bookkeeping that operation-based undo
//! demands. Capacity is bounded so a long edit session can't
//! grow the history without limit.

use std::collections::VecDeque;

use psxed_project::ProjectDocument;

/// Maximum number of snapshots retained on either the undo or
/// the redo stack. Hitting the cap drops the oldest entry --
/// matches IDE-style "you can always undo a few steps but the
/// stack stays bounded" behaviour.
pub const UNDO_CAPACITY: usize = 64;

/// Snapshot-based undo / redo stack. The editor pushes a
/// `ProjectDocument` clone onto [`UndoStack::record`] before
/// every mutating action; [`UndoStack::undo`] /
/// [`UndoStack::redo`] swap the current document with the
/// next entry on the matching stack.
#[derive(Default)]
pub(crate) struct UndoStack {
    undo: VecDeque<ProjectDocument>,
    redo: VecDeque<ProjectDocument>,
}

impl UndoStack {
    /// Push the *pre-mutation* `snapshot` onto the undo stack
    /// and clear the redo stack -- any new edit forks history.
    pub(crate) fn record(&mut self, snapshot: ProjectDocument) {
        if self.undo.len() == UNDO_CAPACITY {
            self.undo.pop_front();
        }
        self.undo.push_back(snapshot);
        self.redo.clear();
    }

    /// Drop all undo / redo entries. Used after filesystem-backed
    /// operations because snapshots only capture project metadata,
    /// not file moves that have already happened on disk.
    pub(crate) fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    /// Pop the most recent undo entry and stash `current` on
    /// the redo stack. Returns the previous snapshot the
    /// caller should swap into the live document, or `None`
    /// when there's nothing to undo.
    pub(crate) fn undo(&mut self, current: ProjectDocument) -> Option<ProjectDocument> {
        let prev = self.undo.pop_back()?;
        if self.redo.len() == UNDO_CAPACITY {
            self.redo.pop_front();
        }
        self.redo.push_back(current);
        Some(prev)
    }

    /// Inverse of [`Self::undo`]: move one step forward in
    /// history if the user previously undid something.
    pub(crate) fn redo(&mut self, current: ProjectDocument) -> Option<ProjectDocument> {
        let next = self.redo.pop_back()?;
        if self.undo.len() == UNDO_CAPACITY {
            self.undo.pop_front();
        }
        self.undo.push_back(current);
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psxed_project::ProjectDocument;

    fn doc(name: &str) -> ProjectDocument {
        ProjectDocument::new(name)
    }

    #[test]
    fn undo_returns_previous_snapshot_and_pushes_current_to_redo() {
        let mut stack = UndoStack::default();
        stack.record(doc("v1"));
        let restored = stack.undo(doc("v2")).expect("undo entry exists");
        assert_eq!(restored.name, "v1");
        // Live state moved onto the redo stack -- redo should
        // hand it back.
        let redone = stack.redo(doc("v1")).expect("redo entry exists");
        assert_eq!(redone.name, "v2");
    }

    #[test]
    fn record_clears_redo_history() {
        let mut stack = UndoStack::default();
        stack.record(doc("v1"));
        let _ = stack.undo(doc("v2"));
        // A new edit *after* an undo forks history -- redo
        // should yield nothing.
        stack.record(doc("v2'"));
        assert!(stack.redo(doc("live")).is_none());
    }

    #[test]
    fn capacity_drops_oldest_undo_entry() {
        let mut stack = UndoStack::default();
        for i in 0..(UNDO_CAPACITY + 5) {
            stack.record(doc(&format!("v{i}")));
        }
        // Drain the stack -- the first entry should be
        // `v5` (oldest 5 dropped).
        let mut last = None;
        while let Some(prev) = stack.undo(doc("live")) {
            last = Some(prev.name);
        }
        assert_eq!(last.as_deref(), Some("v5"));
    }
}
