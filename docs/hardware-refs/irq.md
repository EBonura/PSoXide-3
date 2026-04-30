# IRQ controller

## MMIO layout

| Address | Size | Name | Access | Notes |
|---|---|---|---|---|
| `0x1F80_1070` | 32 | `I_STAT` | R / W (ack) | Pending interrupts. Write is AND-acknowledge. |
| `0x1F80_1074` | 32 | `I_MASK` | R / W | Enabled interrupts. Direct overwrite. |

Upper bits are hardware-reserved -- the 11 meaningful positions are 0..=10.

## Source bits

| Bit | Source | Implemented |
|---|---|---|
| 0 | VBlank | source wiring pending (cycle model) |
| 1 | GPU | needs GPU command decoder |
| 2 | CD-ROM | pending |
| 3 | DMA | pending DICR model |
| 4 | Timer 0 (dot / system clock) | pending cycle model |
| 5 | Timer 1 (system / hblank) | pending cycle model |
| 6 | Timer 2 (system / system/8) | pending cycle model |
| 7 | Controller / MemCard (SIO0) | pending |
| 8 | SIO | pending |
| 9 | SPU | pending full SPU |
| 10 | Lightpen | rarely used, low priority |

## Write semantics

**`I_STAT` is an AND-acknowledge.** Writing value `v` replaces `I_STAT` with `I_STAT & v`:

- Any bit set in `v` is **preserved**.
- Any bit cleared in `v` is **cleared**.

Software typically acknowledges source `s` by writing `!(1 << s)` -- every bit set to 1 *except* the one being acknowledged.

**`I_MASK` is a straight write.** The written value becomes the new mask (with reserved bits dropped).

## CPU integration

The IRQ controller's `(I_STAT & I_MASK) != 0` signal is the single external-interrupt pin the R3000 sees, wired into `COP0.CAUSE.IP[2]` (bit 10).

`Cpu::step` mirrors this every instruction:

```rust
fn sync_external_interrupt(&mut self, bus: &Bus) {
    if bus.external_interrupt_pending() {
        self.cop0[13] |= 1 << 10;     // CAUSE.IP[2] = 1
    } else {
        self.cop0[13] &= !(1 << 10);  // CAUSE.IP[2] = 0
    }
}
```

The interrupt is actually taken iff:
- `SR.IE` (bit 0) is set (globally enabled)
- `SR.IM[2]` (bit 10) is set (hardware-int-pin mask allows it)

When taken, the CPU enters exception `ExceptionCode::Interrupt` (ExcCode = 0), pushes the KU/IE stack, sets `EPC`, clears any pending branch target, and redirects PC to the exception vector (`0xBFC0_0180` while `SR.BEV` is set, the BIOS's boot default).

## Reset state

- `I_STAT` = 0 (no pending)
- `I_MASK` = 0 (none enabled)
- `CAUSE.IP[2]` = 0

The BIOS's first IRQ-controller touch is a write to `I_MASK` establishing which sources it wants to hear from.

## Rust shape

```rust
pub enum IrqSource { VBlank = 0, Gpu = 1, ... Lightpen = 10 }

pub struct Irq { stat: u32, mask: u32 }
impl Irq {
    pub const VALID_BITS: u32 = 0x7FF;
    pub fn raise(&mut self, s: IrqSource);      // sets I_STAT bit
    pub fn pending(&self) -> bool;              // (stat & mask) != 0
    pub fn write_stat(&mut self, v: u32);       // AND-ack
    pub fn write_mask(&mut self, v: u32);       // direct
}
```

Peripherals own the `raise` side -- Timer N raises `IrqSource::Timer(N)`, the GPU raises `IrqSource::Gpu` when it needs to, etc. The CPU owns the `pending` side via its per-step sync.

## References

- Nocash PSX-SPX -- "Interrupts"
- PCSX-Redux `src/core/psxhw.cc` -- the `readHardwareRegister` / `writeHardwareRegister` paths for 0x1F801070/74
- `emulator_core::irq` -- our impl + the acknowledge-semantics unit tests
