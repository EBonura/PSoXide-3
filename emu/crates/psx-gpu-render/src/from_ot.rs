//! Convert a host-side [`OrderingTable`] into a [`GpuCmdLogEntry`] log.
//!
//! On hardware the DMA chain walker reads each packet's tag word, then
//! ships the following data words to GP0 over the GPU command FIFO.
//! On host the editor's preview path uses the same `OrderingTable` +
//! `psx-gpu::prim` data structures the runtime does, but instead of
//! DMAing it walks the chain in software and copies each packet's
//! data words into the [`GpuCmdLogEntry`] format `psx-gpu-render`'s
//! translator already consumes when it renders the live emulator.
//!
//! The adapter is the only piece between "scene authored in the
//! editor" and "wgpu draw calls" — everything else (projection,
//! primitive composition, OT insertion) is the same code that ships
//! on PS1.

use core::ptr;

use emulator_core::gpu::GpuCmdLogEntry;
use psx_gpu::ot::OrderingTable;

/// Walk `ot` in DMA submission order and produce one
/// [`GpuCmdLogEntry`] per primitive packet.
///
/// Each entry's `fifo` is a freshly-cloned `Vec<u32>` containing the
/// `words` data words that follow the OT tag (i.e. the actual GP0
/// command starting at its opcode byte). `opcode` is the top byte of
/// the first FIFO word; `index` is the packet's position in walk
/// order.
///
/// # Safety
/// Same invariant `OrderingTable::iter_packets` requires: every
/// chained packet must be live, writable, 4-byte-aligned RAM for the
/// duration of the call. Primitives produced by the SDK's
/// [`PrimitiveArena`](psx_gpu::PrimitiveArena) satisfy this; bespoke
/// chains must too.
pub unsafe fn build_cmd_log<const N: usize>(ot: &OrderingTable<N>) -> Vec<GpuCmdLogEntry> {
    let mut log = Vec::new();
    // SAFETY: contract above forwards directly to iter_packets.
    let iter = unsafe { ot.iter_packets() };
    for (index, (packet_ptr, words)) in iter.enumerate() {
        let mut fifo = Vec::with_capacity(words as usize);
        for offset in 1..=(words as usize) {
            // SAFETY: packet[1..=words] is alive per the iter contract;
            // each word is u32-aligned because OT primitives are u32
            // aligned by `repr(C, align(4))`.
            let word = unsafe { ptr::read_volatile(packet_ptr.add(offset)) };
            fifo.push(word);
        }
        let opcode = fifo
            .first()
            .map(|&first| ((first >> 24) & 0xFF) as u8)
            .unwrap_or(0);
        log.push(GpuCmdLogEntry {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            opcode,
            fifo,
        });
    }
    log
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-build a one-primitive OT and round-trip it through the
    /// adapter; the resulting cmd-log entry must carry the expected
    /// opcode and data words.
    #[test]
    fn build_cmd_log_extracts_opcode_and_data_words() {
        let mut ot: OrderingTable<8> = OrderingTable::new();
        ot.clear();

        // Hand-built packet imitating a flat triangle:
        //   packet[0] = tag (set by `insert`)
        //   packet[1] = 0x20RRGGBB (opcode 0x20, mono triangle)
        //   packet[2..4] = vertex words
        let mut packet: [u32; 5] = [0, 0x2080_4020, 0x0001_0002, 0x0003_0004, 0x0005_0006];
        unsafe {
            ot.insert(2, packet.as_mut_ptr(), 4);
        }

        let log = unsafe { build_cmd_log(&ot) };
        assert_eq!(log.len(), 1);
        let entry = &log[0];
        assert_eq!(entry.index, 0);
        assert_eq!(entry.opcode, 0x20);
        assert_eq!(
            entry.fifo,
            vec![0x2080_4020, 0x0001_0002, 0x0003_0004, 0x0005_0006]
        );
    }

    /// Two primitives in the same OT — the cmd-log preserves DMA
    /// walk order (most-recently-inserted in a slot first).
    #[test]
    fn build_cmd_log_preserves_dma_order() {
        let mut ot: OrderingTable<4> = OrderingTable::new();
        ot.clear();

        let mut a: [u32; 2] = [0, 0xAA00_0000];
        let mut b: [u32; 2] = [0, 0xBB00_0000];
        unsafe {
            ot.insert(1, a.as_mut_ptr(), 1);
            ot.insert(1, b.as_mut_ptr(), 1);
        }

        let log = unsafe { build_cmd_log(&ot) };
        assert_eq!(log.len(), 2);
        // `b` was inserted last and prepends to the chain head.
        assert_eq!(log[0].opcode, 0xBB);
        assert_eq!(log[1].opcode, 0xAA);
    }
}
