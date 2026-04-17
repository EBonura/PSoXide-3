# SPU (Sound Processing Unit)

## MMIO layout

A huge window (`0x1F80_1C00..0x1F80_1E00`, 512 bytes) packed with
24 voice control blocks plus global mix / reverb / transfer
registers. Phase 3a types only the two the BIOS polls during
cold-init; everything else still round-trips through the echo
buffer.

### Phase 3a: typed

| Address | Size | Name | Access | Notes |
|---|---|---|---|---|
| `0x1F80_1DAA` | 16 | `SPUCNT` | R/W | Control register |
| `0x1F80_1DAE` | 16 | `SPUSTAT` | R only | Status — mirrors `SPUCNT[5:0]` + misc internal bits |

### The rest (echo buffer)

| Range | What |
|---|---|
| `0x1F80_1C00..=0x1F80_1D7F` | 24 voices × 16-byte blocks (volume L/R, sample rate, start addr, ADSR, repeat addr) |
| `0x1F80_1D80..=0x1F80_1D9E` | Global L/R volume, reverb volumes, key-on/off, noise mode, reverb mode, channel-status mirrors |
| `0x1F80_1DA0..=0x1F80_1DBF` | Reverb work area, IRQ address, data transfer regs, SPUCNT/SPUSTAT, CD input volumes, extern input volumes |
| `0x1F80_1DC0..=0x1F80_1DFF` | 32 × 16-bit reverb coefficients |

## SPUCNT bit layout

| Bit | Name | Meaning |
|---|---|---|
| 0 | CD Enable | Route CD audio through SPU mixer |
| 1 | CD Reverb | Apply reverb to CD audio |
| 2 | External Enable | External audio input |
| 3 | External Reverb | Apply reverb to external audio |
| 5:4 | Transfer Mode | 0 = stop, 1 = manual, 2 = DMA write, 3 = DMA read |
| 6 | IRQ Enable | Enable IRQ on specific SPU RAM address |
| 7 | Reverb Master Enable | |
| 9:8 | Noise Frequency Step | |
| 13:10 | Noise Frequency Shift | |
| 14 | Mute | |
| 15 | SPU Enable | Master enable (BIOS sets this early — `0x8010`) |

## SPUSTAT bit layout

| Bit | Meaning |
|---|---|
| 5:0 | **Current SPU mode** — mirror of `SPUCNT[5:0]` (delayed by a few cycles on real hw; we model as instant) |
| 6 | IRQ9 flag |
| 7 | Data transfer busy |
| 9:8 | Data transfer mode (read from SPUCNT[5:4] as a convenience) |
| 10 | DMA R/W request |
| 11 | DMA write request |
| 12 | DMA read request |
| 13 | Data transfer busy flag |
| 15:14 | Capture-buffer half-flag |

The crucial bit is the low 6: when the BIOS writes `SPUCNT = 0x8010`
(SPU enabled, mute reverb) and then reads `SPUSTAT`, it expects to
see bit 4 reflected (→ `0x0010`). Phase 3a's fix was exactly this.

## SPU RAM

512 KiB separate from main RAM. Holds ADPCM sample data, reverb
work buffer, and capture buffers (left channel from voice 1/3, right
channel from voice 0/2). Accessed through the data-transfer register
sequence (write transfer address, set transfer mode in SPUCNT,
write/read bytes via `0x1F80_1DA8`).

## IRQ

`IrqSource::Spu` fires when:
- The SPU hits a configured IRQ address during playback
- (SPU-internal events are complex; full modelling waits on Milestone F/H)

## What's missing

- 24-voice ADPCM decoder + ADSR envelope
- Reverb (massive chunk — convolution engine with 32 tap coefficients)
- Capture buffers
- CD audio input path (requires CD-ROM controller)
- Data transfer to/from SPU RAM (manual + DMA channel 4)
- IRQ on SPU-RAM-address match
- All the per-voice registers doing something when the CPU writes them

This is the largest pending subsystem by far. Milestone F (Crash's
intro FMV with XA audio) is the first gate where we need real SPU.

## Rust shape (Phase 3a)

```rust
pub struct Spu {
    spucnt: u16,
}

impl Spu {
    pub fn contains(phys: u32) -> bool; // just the two typed addresses
    pub fn read16/read32/write16/write32 (...);
    pub fn spucnt(&self) -> u16;
    pub fn spustat(&self) -> u16; // derived
}
```

## References

- Nocash PSX-SPX — "Sound Processing Unit (SPU)" (easily the longest section)
- PCSX-Redux `src/spu/` — the reference implementation
- PSoXide-2 `emulator/spu/` — prior art, worth reading for voice structure
- `emulator_core::spu` — our Phase 3a stub
