# Progress log

A rolling record of what's been built, in roughly chronological
order. Older entries get terser as they age. For design rationale
and the roadmap, see [docs/milestones.md](docs/milestones.md).

## Current parity ceiling

**At least 19,474,543 bit-identical instructions** against PCSX-
Redux with SCPH1001 (no-disc BIOS boot) — state matches
(tick + pc + instr + gprs) for every step up to that point in
the cached 50M-record trace. First divergence at step 19,474,544
is a DMA-IRQ scheduling-order issue (see `probe_cycle_first_divergence`).

At the Sony-logo frame (100M steps), the visible display-area
hash is **byte-exact with Redux** (`0xa3ac6881044333d0`,
0 / 611840 diverging bytes) — proven across all 8 game discs in
the collection via `probe_all_games_100m`.

## 2026-04-19

### GPU renderer — pixel-exact with Redux

- **Scanline-delta rasterizer** (`7e19add`) — full rewrite of all
  four triangle paths (flat / shaded / textured / textured-shaded)
  to match Redux's `drawPoly3Gi` / `drawPoly3TGEx{4,8}i` byte-for-
  byte. Replaces the per-pixel barycentric interpolator whose
  integer rounding diverged from Redux on narrow triangles. 615
  insertions / 255 deletions in `emu/crates/emulator-core/src/gpu.rs`.
- **Sony-logo diverge 3.48% → 0.00%** (`9f9d6ea`). Three Redux-parity
  fixes: dither matrix switched to Redux's coefficient-threshold
  model, `blend_pixel` Average fixed to per-channel `(bg>>1) + (fg>>1)`,
  top-left fill rule applied (now subsumed by the scanline walk).
- **CLUT-0 transparency rule** (`31f40fa`) — `sample_texture`
  now treats the resolved 16-bit CLUT value `0x0000` as transparent,
  not just index 0. Fixes the "TM on a black box" artifact.
- **Pixel-ownership tracer** (`8fbfb4b`) — opt-in `Gpu::pixel_owner`
  buffer + `cmd_log`. After a run, any pixel's owning GP0 packet can
  be looked up in O(1). `probe_pixel_trace` uses it.
- **LWL shift-inversion fix** (`71979f1`) — our LWL used
  `(addr & 3) * 8` when hardware uses `(3 - (addr & 3)) * 8`.
  Corrupted every unaligned word load. Crash Bandicoot's string
  loop overwrote the saved `$ra` on stack → wild jump to
  0x09070026 at step 208M. Fixed, with four truth-table tests
  matching Redux's `LWL_MASK[4]` / `LWL_SHIFT[4]`.
- **Scale-mode toggle** (`dae2c38`) — `Fit` vs `Integer` (pixel-
  perfect) in the toolbar MONITOR icon.
- **Wireframe mode** (`78939db`) — toolbar toggle, routes triangles
  through the line rasterizer.

### Cycle-parity investigation

- **DMA trigger policy** (`cd41b88`) — `bus.write32` now only calls
  `maybe_run_dma()` when the CHCR register is written with bit 24
  set (matching Redux's `psxhw.cc` dispatcher). Previously every
  DMA-register write re-ran every channel whose start bit was
  still high, producing duplicate schedule events. `probe_dma_schedules`:
  went from 9 spurious schedules → 5 correct schedules in the same
  window.
- **Cycle-drift probe** (`fb5e19f`) — `probe_cycle_parity` compares
  our `bus.cycles()` to Redux's tick at matched step counts.
  Headlines: BIOS 100M -0.99%, BIOS 500M **0.000%** exact, Crash
  100M -1.47%, Crash 300M -2.96%, Crash 900M -6.03%. BIOS code
  matches Redux's per-instruction cost model; disc-active paths
  accumulate drift.
- **Instruction-level first-divergence probe** (`605f0d7`,
  `bec5da2`) — `probe_cycle_first_divergence` + `probe_isr_trace`
  + `probe_dma_timeline` walk the cached Redux trace in lock-step
  with our emulator, folding IRQ handlers into single records the
  way Redux's oracle does. Pinpoints step 19,474,544 as the first
  full divergence (2488 cycles, DMA IRQ fires late in ours).
- **All-games 100M scorecard** (`a431d55`) — `probe_all_games_100m`
  runs every disc in `~/Downloads/ps1 games/` to 100M steps and
  verifies the display hash matches Redux's canonical
  `0xa3ac6881044333d0`. Currently 9/9.
- **Parity test suite** (`22fa220`, `fb5e19f`) — per-game Sony-logo
  parity tests (`parity_gt2_sony_logo`, `parity_mgs_sony_logo`,
  `parity_re2_sony_logo`, `parity_wipeout{1,2097,3}_sony_logo`)
  with 0.05% pixel-divergence ceiling.

### Milestones

- **A**: 100M Sony-logo, self + Redux-verified display hash pinned.
- **B**: 500M BIOS shell, self-regression pinned (Redux hash TBD).
- **D**: Crash-Bandicoot progression guard (runs 300M instructions,
  asserts final PC is in legit RAM/BIOS/scratchpad) + 600M VRAM
  self-hash. Tekken 3 licensed-screen VRAM self-hash at 800M.
- Milestone A now carries a Redux-verified correctness assertion,
  so any rendering regression breaks the test immediately.

## 2026-04-17 / 18 (catch-up — pre-scanline-rasterizer era)

### GPU — rasterizer + primitives

- Full rasterizer for flat, Gouraud, textured, and textured-Gouraud
  triangles and quads. 4bpp/8bpp CLUT + 15bpp direct colour modes.
- Four semi-transparency blend modes (average, add, sub, quarter)
  with per-texel mask-bit gating matching PSX-SPX.
- 4×4 Bayer dither (`dither_rgb`) for Gouraud + textured-Gouraud
  when `GP0 0xE1` bit 9 is set. Later replaced with Redux's
  coefficient-threshold model for byte-exact parity.
- Mask-bit logic: `GP0 0xE6` set-mask / check-before-draw fully
  implemented; `plot_pixel` respects both.
- Texture-window masking (`GP0 0xE2`): `U' = (U & ~mask) | (offset & mask)`
  per axis.
- GP0 0x64..=0x7F sprite / textured-rectangle family with flip-x /
  flip-y from draw-mode status bits.
- GP0 0x40..=0x5F line family: single + polyline, mono + shaded.
- 24-bit display mode (GP1 0x08 bit 4) with RGBA8 pixel accessor.
- GPU busy-flag pacing: `charge_busy(pixels/2)` per primitive,
  decayed at 32 units/cycle — matches real hardware's "a few
  scanlines to finish the batch" polling pattern.

### SPU — full ADPCM synthesis

- 24 voices with per-voice ADSR envelopes, Gaussian interpolation
  (4-point, 1024-entry coefficient table), pitch modulation,
  ADPCM block decoder.
- Voice volume-sweep envelopes (not just static snapshots) —
  per-voice volume sweeps independent of ADSR.
- Main / CD / ext / reverb volume-pair envelopes.
- CD audio input mix + XA ADPCM decoder.
- Gaussian OOB fix: clamp `vl` to 1020 when `frac` leaks past 0x1000
  on block boundaries at high pitch.

### MDEC — full decoder

- IDCT + YUV→RGB + `DataReadRAW` + `DataReadRLE` + block
  compression/output format, enough to decode real game cutscenes.

### CDROM — commands + XA + timing

- `CdlReadS` (0x1B) + `CdlGetlocL` / `CdlGetlocP` (0x10 / 0x11)
  + SetMode speed toggle.
- Full XA ADPCM decoder on interleaved sectors with SPU mixing.
- Redux-matched timing constants (read period, seek, stat-ready).

### GTE

- 19 of the 20 commands covering matrix × vector, rotation, perspective
  project, normal colour, light colour, etc.
- **BK bias fix** (`NCS`/`NCDS`/`NCCS` stage 1) — Redux's GTE has
  no BK bias in stage 1, matching hardware. Previous revision double-
  counted the background colour.

### Timers

- Three root counters with all sync modes:
  - Timer 0 dot-clock source + hblank.
  - Timer 1 VBlank pause/reset (sync-mode 3 unlocks on VBlank).
  - Timer 2 stop-counter sync-modes.

### DMA — unified scheduler

- Channels 0-6 with per-channel completion events routed through a
  single `EventSlot` scheduler shared with VBlank + SPU async.
- Linked-list walker for GPU channel + block-mode for MDEC /
  SPU / CDROM.
- DICR master-edge IRQ raise; per-channel enable bits.

### Pad — DualShock + rumble + port 2

- Digital + analog pad with stick axes.
- Command `0x4D` + `0x42` TX rumble motor mapping.
- Port-2 attach + button plumbing.

### HLE BIOS

- Syscall trampoline at the standard 0xA0 / 0xB0 / 0xC0 vectors.
- Dispatches the common functions (puts, fopen, etc.) against a
  minimal memfile backing.

### Frontend

- Top-bar icon controls (XMB + wireframe + memory + VRAM + monitor-
  scale + registers toggles). HUD consolidated into the toolbar.
- Audio output via `cpal` — 44.1 kHz stereo ring buffer, nearest-
  neighbour resample when the host doesn't accept 44.1 kHz.
- Gamepad input via `gilrs` (digital + analog sticks, Xbox-layout
  to PSX-button mapping).

### SDK — scaffold + 5 examples

- Six crates: `psx-sdk`, `psx-hw`, `psx-rt`, `psx-io`, `psx-gpu`,
  `psx-gte`, `psx-pad`.
- Examples: `hello-tri` (direct GP0 triangle), `hello-tex` (textured
  sprite), `hello-ot` (DMA linked list), `hello-input` (pad poll),
  `hello-gte` (GTE perspective transform).
- `PSOXIDE_EXE=…` side-loads a built example into the frontend,
  skipping the BIOS reset vector; HLE BIOS + digital pad auto-enabled.

### Tooling

- `tools/mkisopsx` — writes a bootable PSX ISO with SYSTEM.CNF
  pointing at a chosen EXE.
- `tools/psx-exe-pack` — patches PS1-EXE headers.
- `parity-oracle` — PCSX-Redux headless harness with Lua-scripted
  stepping, cached-trace store at
  `target/parity-cache/redux-{bios-hash}-{N}.bin`.

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
- Overlay: XMB menu (Game / Debug / System categories, animated
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
the full ladder. The pre-Milestone-A work that was listed in this
section previously (cycle model / VBlank / DMA2 / GP0 FIFO /
rasterizer) has all landed — Milestone A passes with Redux-
verified byte-exact parity on the Sony-logo frame.

### Cycle-accuracy push (post-Milestone-A)

The open investigation is closing the cycle-accuracy gap that
causes rendered frames at deeper step counts to land on slightly
different animation instants than Redux's (renderer byte-exact,
virtual-clock not). Current state from `probe_cycle_parity`:
BIOS 500M is 0.000% (exact); Crash 900M is -6.03%.

1. **Extra DMA schedules** (`probe_dma_timeline`). Our BIOS boot
   issues 5 DMA schedule events in a window Redux issues 3 ISR
   folds for. Root-causing the two extras is the next thread —
   likely either a spurious CHCR path or a missing scheduler-
   overwrite guard.
2. **Redux MMIO-write logging**. To diff our CHCR/MADR/BCR write
   sequences against Redux's directly, the oracle needs to emit
   MMIO events into the cached trace. Currently only instruction
   records are captured.
3. **SPU / CDROM / IRQ timing audits** (later). Each subsystem's
   scheduled-event cycles need Redux-exact computation; gaps
   compound in disc-active paths (Crash's drift doubles from 100M
   to 300M to 900M).

### Milestone-C MVP

SDK scaffolding is in place; an end-to-end "own-built EXE renders
a triangle, VRAM hash locked" regression test still needs
wiring — probably as `milestone_c_sdk_hello_tri` in
`emu/crates/emulator-core/tests/milestones.rs`, loading the
compiled `sdk/examples/hello-tri` binary via the `PSOXIDE_EXE`
side-load path and hashing the VRAM after N frames.

### Milestones E–K

Crash title-screen render (E), intro FMV (F), MDEC-heavy
cutscenes (G), multi-track audio-CD (H), 60 fps timing-critical
(I), PAL timing (J), GT2 heaviest-load (K). Each requires a
different subsystem to reach Redux-verified correctness first.
