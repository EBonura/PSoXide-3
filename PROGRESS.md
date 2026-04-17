# Progress log

A rolling record of what's been built, in roughly chronological
order. Older entries get terser as they age. For design rationale
and the roadmap, see [docs/milestones.md](docs/milestones.md).

## Current parity ceiling

**At least 12,761,002 bit-identical instructions** against PCSX-
Redux with SCPH1001 — the highest step count we've confirmed.
A 50M probe is being re-run against the latest fixes; the real
ceiling is likely substantially higher.

## 2026-04-17

### CPU

- Expanded decoder coverage to everything the BIOS emits in its
  first ~3M instructions: REGIMM branches (BLTZ/BGEZ/BLTZAL/BGEZAL),
  JAL/JR/JALR, BLEZ/BGTZ, SLTI/SLTIU/ANDI/XORI, sub-word loads +
  stores (LB/LH/LBU/LHU/SB/SH), shift family (SLL/SRL/SRA + V
  variants), HI/LO + MULT/MULTU/DIV/DIVU/MFHI/MFLO/MTHI/MTLO, COP0
  MFC0/RFE.
- SYSCALL + BREAK + full exception-entry plumbing: CAUSE.ExcCode,
  CAUSE.BD, EPC (with delay-slot `-4` backup), SR KU/IE stack push,
  exception vector select from `SR.BEV`.
- Interrupt dispatch (ExcCode 0) wired through `CAUSE.IP[2]` — the
  controller can raise an external interrupt and the CPU will take
  it as soon as `SR.IE` + `SR.IM[2]` allow.
- **R3000 load-delay squash** on same-register writeback collision:
  if a load-delay-slot instruction writes the load's target
  non-load-wise, the load's writeback is cancelled. Fixed the 12.7M
  parity divergence that was caused by us committing the load
  anyway.

### Peripherals (typed register scaffolding)

- **IRQ** — `I_STAT` + `I_MASK`, AND-acknowledge semantics,
  11 source bits, CPU mirror into `CAUSE.IP[2]`.
- **Timers** — three root counters at their standard addresses,
  counter/mode/target triplet per timer. No ticking yet.
- **DMA** — 7 channels + DPCR + DICR. **OTC (channel 6) actually
  transfers** — fills an ordering table with the linked-list
  terminator pattern. Other channels accept writes but don't
  transfer.
- **GPU** — GP0/GP1/GPUSTAT/GPUREAD ports wired. Status returns
  the always-ready soft-GPU pattern (bits 26/27/28 forced,
  bit 25 derived from DMA direction). GP1 0x00/0x03/0x04 decoded;
  GP0 writes accepted but discarded. Owns VRAM (migrated from
  frontend).
- **SPU** — SPUCNT + SPUSTAT mirror typed; the rest of SPU MMIO
  routes through the echo buffer. Enough to unblock BIOS cold-init.

### Bus

- Typed dispatch for all four peripherals ahead of the MMIO echo
  buffer fallback. `try_read8` for safe diagnostic access. Byte +
  half + word access paths for SPU (BIOS uses `SH`).

### Frontend

- Full `winit + wgpu + egui` app with charcoal/teal theme, VT323
  monospace, Lucide icons.
- Side panels: register viewer (GPRs + PC/HI/LO + COP0 with
  bit-level decode + breakpoints list + 64-entry exec history with
  disassembly), memory viewer (hex + disasm modes, quick-jump
  buttons, PC row highlight, BP toggle).
- Bottom panel: VRAM viewer — live 1024×512 texture upload, 5→8-bit
  expansion with proper full-range mapping.
- Overlay: Menu menu (Game / Debug / System categories, animated
  slide), HUD bar (FPS + Mips + frame time + tick + RUNNING/PAUSED).
- Continuous run at N instr/frame (slider-tunable), breakpoints
  halt with gdb semantics.
- MIPS R3000 disassembler with full coverage of implemented opcodes
  and graceful unknowns.

### Tooling

- PCSX-Redux parity oracle (headless, Lua-scripted, batched stepping).
- Parity ladder: 1 / 20 / 100 / 500 / 2k / 5k / 50k / 200k / 1M /
  2.735M / probe rungs.
- `make run` / `make parity` / `make test` / `make check` /
  `make lint`.

### Documentation

- README with wordmark and player icon.
- [docs/milestones.md](docs/milestones.md) codifies the canary
  ladder.
- [docs/frontend.md](docs/frontend.md) architecture reference.
- [docs/hardware-refs/](docs/hardware-refs/) specs for IRQ, DMA,
  Timers, GPU, SPU.

## Upcoming

Tracked loosely; see [docs/milestones.md](docs/milestones.md) for
the full ladder.

### Pre-Milestone-A work

1. **Cycle model + VBlank generation** (~1 day). Unblocks BIOS
   `waitVSync`. Will break parity deliberately — we pivot to
   capability-based validation here.
2. **DMA channel 2 (GPU)** (~1 day). Linked-list walker + feed
   commands into GP0.
3. **GP0 command FIFO + decoder** (~3–5 days). Draw-mode commands
   first, then primitives.
4. **Minimal rasterizer** (~1 week). Flat triangles, textured
   triangles, fill rect. Enough to display the Sony logo.

### Milestone A — Sony logo boots

When the four items above land, we expect the boot sequence to
reach the Sony logo. Freeze a frame hash; it becomes Milestone A's
regression test.
