# DMA controller

## MMIO layout

### Per channel (7 channels, stride 0x10)

| Offset | Name | Purpose |
|---|---|---|
| `+0x0` | `MADR` (base) | RAM address (source/destination) |
| `+0x4` | `BCR` (block control) | Count + block size, format depends on sync mode |
| `+0x8` | `CHCR` (channel control) | Direction, step, sync mode, start/busy |

Channels live at `0x1F80_1080 + 0x10 * ch` for `ch` in `0..=6`.

### Global

| Address | Name | Purpose |
|---|---|---|
| `0x1F80_10F0` | `DPCR` | Per-channel enables + priority bits |
| `0x1F80_10F4` | `DICR` | IRQ enables + IRQ-pending flags |

## Channel identity

| ch | Consumer | Notes |
|---|---|---|
| 0 | MDEC-in | macroblock data into the decoder |
| 1 | MDEC-out | decoded 16bpp data out of the decoder |
| 2 | GPU | GP0 command words from RAM |
| 3 | CD-ROM | sector data from the drive controller |
| 4 | SPU | ADPCM samples into SPU RAM |
| 5 | PIO | extension port |
| 6 | OTC | ordering-table clear (see below) |

## CHCR bit layout (relevant subset)

| Bit | Name | Meaning |
|---|---|---|
| 0 | Direction | 0 = RAM → device, 1 = device → RAM |
| 1 | Step | 0 = `+4` after each word, 1 = `-4` |
| 9:8 | Sync mode | 0 = manual, 1 = block/request, 2 = linked list |
| 24 | Start/trigger | Software sets to begin; cleared by hardware on completion |
| 28 | Enable | Typically set alongside bit 24; cleared by hardware on completion |

## Sync modes

### Manual (`0`)

Single-word transfer — software writes MADR, asserts bit 24, a single word moves, bit 24 clears. Rare on PS1.

### Block / request (`1`)

Transfers `BS` (block size, low 16 bits of `BCR`) words per request. Either a fixed count (`BA` × `BS` words total) or request-paced (drives `DMARequest` from the device). GPU uses this for direct command-list dumps; SPU for sample DMA.

### Linked list (`2`)

Walks a linked list in RAM. Each node header: `0xSS_AAAAAA` — `SS` is the word count to ship (little-endian read, so the upper byte), `AAAAAA` is the physical address of the next node. Terminator `AAAAAA = 0xFFFFFF`. Used by the GPU channel to ship ordering tables.

## OTC channel (channel 6)

OTC = "Ordering Table Clear". Unlike the other channels it doesn't interact with any external device — it just fills a RAM region with the linked-list terminator pattern that the GPU channel will then walk in reverse.

Layout produced by a count-`N` OTC at base `B` (all words 32-bit, addresses word-aligned):

| Address | Value |
|---|---|
| `B` (base, highest) | `0x00FF_FFFF` (terminator) |
| `B - 4` | `B` |
| `B - 8` | `B - 4` |
| … | … |
| `B - 4(N-1)` | `B - 4(N-2)` |

Walking forward from `B - 4(N-1)` through the linked `addr = mem[addr]` chain hits the terminator at `B`.

OTC uses sync mode 0 (manual count) with step bit 1 (decrement).

## Rust shape

```rust
pub struct DmaChannel {
    pub base: u32,           // MADR — 24-bit masked
    pub block_control: u32,  // BCR
    pub channel_control: u32, // CHCR
}

pub struct Dma {
    pub channels: [DmaChannel; 7],
    pub dpcr: u32,
    pub dicr: u32,
}

impl Dma {
    pub const BASE: u32 = 0x1F80_1080;
    pub const END:  u32 = 0x1F80_10F8;
    pub fn contains(phys: u32) -> bool;
    pub fn read32(&self, phys: u32) -> u32;
    pub fn write32(&mut self, phys: u32, value: u32);
    /// Run the OTC (ch 6) transfer if its start bit is set.
    pub fn run_otc(&mut self, ram: &mut [u8]) -> bool;
}
```

`Bus::write32` calls `Dma::run_otc` after every DMA-register write so an OTC kick is observable immediately without waiting for a scheduler tick. The other channels (GPU, SPU, CD-ROM, MDEC) have their CHCR start bits *recorded* but no transfer fires — each lands with its consumer subsystem.

## DICR — incomplete

DICR carries per-channel IRQ enable bits + flag bits + a master-enable bit + a force-flag bit. Correct modelling needs to:

1. Track per-channel IRQ flags (bits 24..=30).
2. Raise `IrqSource::Dma` when any enabled channel completes *and* the master enable is on.
3. Handle write-acknowledge: writing 1 to a flag bit clears it (opposite polarity from `I_STAT`'s AND-acknowledge, yes, really).

Currently `DICR` just stores what's written — sufficient until a DMA channel needs to signal completion. When OTC or the GPU channel needs to fire an IRQ, this gets a proper implementation.

## References

- Nocash PSX-SPX — "DMA Channels"
- PCSX-Redux `src/core/psxdma.cc` — the dispatch that splits block / linked-list / OTC
- `emulator_core::dma` — our impl + the OTC unit tests
