# Root counters (Timer 0 / 1 / 2)

## MMIO layout

Three counters with identical three-register shape at stride 0x10.

| Address | Size | Name | Access | Notes |
|---|---|---|---|---|
| `0x1F80_1100` | 16 | Timer 0 counter | R/W | Increments on its clock source |
| `0x1F80_1104` | 16 | Timer 0 mode    | R/W | Control + status bits |
| `0x1F80_1108` | 16 | Timer 0 target  | R/W | Compare value |
| `0x1F80_1110` / `0x1F80_1114` / `0x1F80_1118` | 16 | Timer 1 | | |
| `0x1F80_1120` / `0x1F80_1124` / `0x1F80_1128` | 16 | Timer 2 | | |

Accessible as 16- or 32-bit; upper bits read as 0. Writes at
`byte` granularity are possible on hardware but our Bus routes
these through the echo buffer (not likely to matter -- all
software I've seen uses word-sized access).

## Clock sources per timer

| Timer | Mode bits 9:8 = 0 | = 1 | = 2 | = 3 |
|---|---|---|---|---|
| 0 | System clock | Dot clock (GPU) | System clock | Dot clock |
| 1 | System clock | H-blank | System clock | H-blank |
| 2 | System clock | System clock | System clock / 8 | System clock / 8 |

The **dot clock** advances at the pixel rate -- 5.37 MHz at
320×240, 6.72 MHz at 384×240, etc. Timer 0 with dot-clock source
is how games measure horizontal pixel positions.

**H-blank** ticks once per scanline (~15.73 kHz NTSC / ~15.63 kHz
PAL). Timer 1 with h-blank source is how the BIOS derives
VSync timing without trusting the GPU's VBlank IRQ.

## Mode / control bits

| Bit | Name | Meaning |
|---|---|---|
| 0 | Sync enable | 0 = free-run; 1 = sync-gated |
| 2:1 | Sync mode | Gate / pause / reset behaviour during HBlank/VBlank |
| 3 | Reset mode | 0 = reset at 0xFFFF; 1 = reset at target |
| 4 | Target IRQ | Fire IRQ on target reached |
| 5 | Overflow IRQ | Fire IRQ on counter = 0xFFFF |
| 6 | IRQ repeat | 0 = one-shot; 1 = repeat |
| 7 | IRQ toggle | 0 = pulse; 1 = toggle |
| 9:8 | Clock source | Per-timer (see table above) |
| 10 | IRQ flag | 0 = IRQ fired (latched); 1 = inactive |
| 11 | Reached target | Read-only status |
| 12 | Reached 0xFFFF | Read-only status |
| 15:13 | -- | Unused |

Writing the mode register is surprisingly effectful: it **resets
the counter to 0** as a side effect, and re-arms the IRQ latch.

## Ticking

The emulator advances root counters from the bus cycle clock. To keep
the hot CPU path cheap, `Bus` synchronizes the timer bank lazily at
scheduler drains and timer MMIO accesses instead of ticking all three
counters every instruction.

Timer 0 converts cycles through the current GPU dot-clock divisor when
bits 9:8 select dot clock. Timer 1 converts cycles to HBlank ticks when
bits 9:8 select HBlank. Timer 2 supports the system-clock and system/8
sources, with Redux-compatible stop behavior for sync modes 1/2.

Redux does not implement Timer 0/1 sync gate/reset semantics beyond the
clock-source and IRQ effects. PSoXide mirrors that: Timer 1 mode
`0x0103`, used by the SDK `vsync()` helper, continues counting across
VBlank instead of resetting on the VBlank IRQ edge.

## Rust shape

```rust
pub struct Timer {
    pub counter: u32,   // 16-bit meaningful
    pub mode:    u32,
    pub target:  u32,
    // plus accumulator and diagnostic cycle fields
}

pub struct Timers {
    pub timers: [Timer; 3],
}

impl Timers {
    pub const BASE:   u32 = 0x1F80_1100;
    pub const SIZE:   u32 = 0x30;
    pub const STRIDE: u32 = 0x10;
    pub fn contains(phys: u32) -> bool;
    pub fn advance_to(&mut self, now: u64, hsync_period: u64, dot_clock_divisor: u64) -> u8;
    pub fn read32(&mut self, phys: u32) -> u32;
    pub fn write32(&mut self, phys: u32, value: u32, now: u64);
}
```

## References

- Nocash PSX-SPX -- "Timers"
- PCSX-Redux `src/core/psxcounters.cc` -- the real ticking implementation
- `emulator_core::timers` -- PSoXide's timed counter implementation
