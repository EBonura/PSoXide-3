//! Editor 3D viewport — Phase 0 spike scene.
//!
//! Builds a hardcoded three-triangle GP0 command log via the same
//! pipeline the runtime uses on PS1: `psx-gpu::prim` packets ➜
//! `OrderingTable::insert` ➜ `psx_gpu_render::build_cmd_log` ➜
//! `HwRenderer::render_frame`. The point is to validate the
//! cross-workspace plumbing end-to-end — once this paints colored
//! triangles in the editor panel, every subsequent phase (real Room
//! geometry, gizmos, textures, model preview, picking) is plumbing
//! on top of a known-working foundation.
//!
//! The scene is sized for the PSX 320×240 logical display: each
//! triangle anchors at a different screen-space corner so it's
//! immediately obvious that all three packets reached the renderer.

use std::sync::Mutex;

use emulator_core::gpu::GpuCmdLogEntry;
use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::TriFlat;

/// Phase 0 scene scratch: three flat-coloured triangles + a tiny OT
/// they're inserted into.
///
/// Stays in static storage so the OT chain pointers remain valid for
/// the duration of `build_cmd_log` — primitives moved out of an OT
/// would invalidate the linked-list addresses written into them by
/// `insert`. Wrapped in a `Mutex` so the cell-internals access is
/// sound even though only the editor renderer touches it today.
struct PreviewScene {
    ot: OrderingTable<8>,
    tris: [TriFlat; 3],
    initialised: bool,
}

static PREVIEW: Mutex<PreviewScene> = Mutex::new(PreviewScene {
    ot: OrderingTable::new(),
    tris: [
        TriFlat::new([(0, 0), (0, 0), (0, 0)], 0, 0, 0),
        TriFlat::new([(0, 0), (0, 0), (0, 0)], 0, 0, 0),
        TriFlat::new([(0, 0), (0, 0), (0, 0)], 0, 0, 0),
    ],
    initialised: false,
});

/// Build (or rebuild) the Phase 0 command log.
///
/// First call wires the three triangles into the static OT; subsequent
/// calls just walk the cached chain. Returns a fresh `Vec` because the
/// `HwRenderer::render_frame` API takes `&[GpuCmdLogEntry]` and the
/// log contents are tiny (≈ 5 entries × a few words).
pub fn build_phase0_cmd_log() -> Vec<GpuCmdLogEntry> {
    let mut scene = PREVIEW.lock().expect("editor preview scene mutex");

    if !scene.initialised {
        // Rebuild the triangles in place at known screen positions so
        // we can see all three at once. PSX screen coords are i16 with
        // the (0,0) origin at the top-left of the draw area; the
        // engine's default 320×240 frame fits comfortably inside an
        // i16's range.
        scene.tris[0] = TriFlat::new(
            [(40, 200), (160, 30), (280, 200)],
            0xFF, 0x40, 0x40, // warm red — fills most of the panel
        );
        scene.tris[1] = TriFlat::new(
            [(20, 60), (90, 20), (90, 100)],
            0x40, 0xC0, 0xFF, // cyan — top-left wedge
        );
        scene.tris[2] = TriFlat::new(
            [(230, 100), (300, 60), (300, 140)],
            0xC0, 0xFF, 0x40, // lime — top-right wedge
        );

        scene.ot.clear();
        // Distinct depth slots so the OT chain has actual structure
        // for `iter_packets` to walk. Lower z draws first (back).
        let tris_ptr: *mut TriFlat = scene.tris.as_mut_ptr();
        unsafe {
            scene
                .ot
                .insert(0, tris_ptr.add(0).cast::<u32>(), TriFlat::WORDS);
            scene
                .ot
                .insert(2, tris_ptr.add(1).cast::<u32>(), TriFlat::WORDS);
            scene
                .ot
                .insert(4, tris_ptr.add(2).cast::<u32>(), TriFlat::WORDS);
        }
        scene.initialised = true;
    }

    // SAFETY: `scene.tris` and `scene.ot` are alive for the duration
    // of this function (they're behind a static + a held lock). The
    // OT only references their addresses; no aliasing concerns since
    // we own the lock.
    unsafe { psx_gpu_render::build_cmd_log(&scene.ot) }
}
