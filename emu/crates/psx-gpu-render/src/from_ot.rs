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
//! editor" and "wgpu draw calls" -- everything else (projection,
//! primitive composition, OT insertion) is the same code that ships
//! on PS1.

use core::ptr;

use emulator_core::gpu::{gp0_command_word_count, GpuCmdLogEntry};
use psx_gpu::ot::OrderingTable;

/// Walk `ot` in DMA submission order and produce one
/// [`GpuCmdLogEntry`] per GP0 command.
///
/// Each OT packet may contain more than one GP0 command; for example
/// windowed textured triangles are emitted as `E2 + 24h` inside one
/// DMA packet so the texture-window state stays adjacent to the
/// primitive. Each returned entry's `fifo` contains exactly one decoded
/// GP0 command, matching the emulator's live command log.
///
/// # Safety
/// Same invariant `OrderingTable::iter_packets` requires: every
/// chained packet must be live, writable, 4-byte-aligned RAM for the
/// duration of the call. Primitives produced by the SDK's
/// [`PrimitiveArena`](psx_gpu::PrimitiveArena) satisfy this; bespoke
/// chains must too.
pub unsafe fn build_cmd_log<const N: usize>(ot: &OrderingTable<N>) -> Vec<GpuCmdLogEntry> {
    let mut log = Vec::new();
    let mut command_index = 0u32;
    // SAFETY: contract above forwards directly to iter_packets.
    let iter = unsafe { ot.iter_packets() };
    for (packet_ptr, words) in iter {
        let mut packet = Vec::with_capacity(words as usize);
        for offset in 1..=(words as usize) {
            // SAFETY: packet[1..=words] is alive per the iter contract;
            // each word is u32-aligned because OT primitives are u32
            // aligned by `repr(C, align(4))`.
            let word = unsafe { ptr::read_volatile(packet_ptr.add(offset)) };
            packet.push(word);
        }

        let mut offset = 0usize;
        while offset < packet.len() {
            let opcode = ((packet[offset] >> 24) & 0xFF) as u8;
            let command_words = gp0_command_word_count(opcode)
                .max(1)
                .min(packet.len() - offset);
            let fifo = packet[offset..offset + command_words].to_vec();
            log.push(GpuCmdLogEntry {
                index: command_index,
                opcode,
                fifo,
            });
            command_index = command_index.saturating_add(1);
            offset += command_words;
        }
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

    /// Two primitives in the same OT -- the cmd-log preserves DMA
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

    /// A single DMA packet can legally contain multiple GP0 commands.
    /// The host renderer consumes command-log entries, so the adapter
    /// must split the FIFO words before translation.
    #[test]
    fn build_cmd_log_splits_multi_command_packets() {
        let mut ot: OrderingTable<8> = OrderingTable::new();
        ot.clear();

        let mut packet: [u32; 9] = [
            0,
            0xE204_2318,
            0x2480_8080,
            0x0001_0002,
            0x0000_0000,
            0x0003_0004,
            0x0000_0000,
            0x0005_0006,
            0x0000_0000,
        ];
        unsafe {
            ot.insert(2, packet.as_mut_ptr(), 8);
        }

        let log = unsafe { build_cmd_log(&ot) };
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].opcode, 0xE2);
        assert_eq!(log[0].fifo, vec![0xE204_2318]);
        assert_eq!(log[1].opcode, 0x24);
        assert_eq!(log[1].fifo.len(), 7);
        assert_eq!(log[1].index, 1);
    }
}
