# PCSX-Redux oracle

PSoXide does not include PCSX-Redux source or binaries. The oracle
harness in `emu/crates/parity-oracle` launches an external patched
Redux build and talks to `lua/oracle.lua` over stdin/stdout.

The current expected Redux line is:

- repo: `https://github.com/EBonura/pcsx-redux.git`
- branch: `psoxide-bindings`
- known local commit: `23cdc6ab` (`Expose audio drain hooks for PSoXide oracle`)

Set the binary explicitly when possible:

```bash
export PSOXIDE_REDUX_BIN=/Users/ebonura/Desktop/repos/pcsx-redux/pcsx-redux
export PSOXIDE_BIOS=/absolute/path/to/SCPH1001.BIN
```

If `PSOXIDE_REDUX_BIN` is not set, `OracleConfig` tries the local
`/Users/ebonura/Desktop/repos/pcsx-redux` fallback paths.

## Required bindings

`oracle.lua` requires a Redux build that exposes these Lua surfaces:

- stepping and execution: `PCSX.stepIn()`, `PCSX.runExecute()`,
  `PCSX.setQuietPauseResume()`
- memory/register access: `PCSX.getMemPtr()`, `PCSX.getRomPtr()`,
  `PCSX.getScratchPtr()`, `PCSX.getRegisters()`, `PCSX.getCPUCycles()`,
  `PCSX.invalidateCache()`
- display/audio capture: `PCSX.GPU.takeScreenShot()`,
  `PCSX.drainAudioFrames()`
- controller overrides: `PCSX.SIO0.slots[port].pads[1].setOverride()`
  and `clearOverride()`

Run the quick protocol checks with:

```bash
make oracle-smoke
```

## Side-loaded EXE parity

Side-loaded SDK canaries normally use PSoXide's HLE BIOS path. That is
good for fast local frame probes, but it is not a Redux parity setup:
Redux runs the real BIOS ROM, and PSoXide's HLE dispatcher deliberately
does not match the real BIOS instruction stream.

The Redux side-load oracle uses this contract instead:

1. Start both emulators from reset with the same BIOS.
2. Run the real BIOS warmup on both sides. The default is
   `DISC_FAST_BOOT_WARMUP_STEPS` (`10_000_000` steps).
3. Copy the same PSX-EXE payload into RAM.
4. Clear the EXE BSS range.
5. Seed PC, GP, and SP/FP from the EXE header.
6. Invalidate Redux's instruction cache.
7. Acknowledge any already-latched VBlank IRQ, advance both emulators
   to future VBlank checkpoints, and compare visible display hashes.

Run the side-load parity probe with:

```bash
make oracle-side-load
```

By default the ignored test matrix covers:

- `hello-tri` -- direct GP0 Gouraud triangle
- `hello-tex` -- texture upload / CLUT / textured drawing
- `hello-ot` -- GPU DMA linked-list ordering table
- `hello-input` -- pad polling path with no buttons pressed

Override inputs with:

```bash
export PSOXIDE_ORACLE_EXE=/absolute/path/to/example.exe
export PSOXIDE_ORACLE_EXE_HELLO_TEX=/absolute/path/to/hello-tex.exe
export PSOXIDE_ORACLE_EXE_WARMUP=10000000
export PSOXIDE_ORACLE_EXE_MAX_STEPS_PER_FRAME=2000000
export PSOXIDE_ORACLE_EXE_FRAMES=3
export PSOXIDE_ORACLE_EXE_HELLO_TEX_FRAMES=3
export PSOXIDE_ORACLE_EXE_SETTLE_STEPS=0
```

The default side-load target checks three VBlank checkpoints per
example, enough to prove double-buffer display-start transitions
against Redux. Raise `PSOXIDE_ORACLE_EXE_FRAMES` while investigating
deeper frame drift; any mismatch fails the test and prints both hashes.
`PSOXIDE_ORACLE_EXE_SETTLE_STEPS` is normally `0`; set it only when
probing whether a mismatch is
edge-sampling phase or actual display/GPU state.

The SDK `vsync()` helper polls Timer 1 with HBlank clock source. For
Redux parity, PSoXide intentionally mirrors Redux's counter model:
Timer 1's VBlank sync/reset mode bits do not reset the counter at the
VBlank IRQ edge. Side-loaded double-buffer samples depend on this phase.

`hello-gte` is intentionally not in the green side-load matrix yet. It
currently diverges on frame 1 against Redux even after matching Redux's
flat-line raster tie-breaks, so the next parity investigation should
split its remaining gap between COP2 projection state and GP0 line edge
cases.

## Commercial disc smoke

The bounded commercial-disc smoke test is ignored and requires a local
disc path. PSoXide does not vendor or name copyrighted disc assets.

```bash
export PSOXIDE_DISC=/absolute/path/to/game.cue
export PSOXIDE_BIOS=/absolute/path/to/SCPH1001.BIN
make oracle-disc-smoke
```

`PSOXIDE_DISC` may point at a `.cue` or raw `.bin`; the local emulator
uses the same CUE/BIN loader path as the interactive
`disc_vram_parity` example. Redux uses the configured
`PSOXIDE_REDUX_BIN` discovery rules documented above.

The default checkpoint is intentionally conservative:

```bash
export PSOXIDE_ORACLE_DISC_STEPS=1000000
```

The test skips cleanly when `PSOXIDE_DISC`, the disc file, the BIOS, or
the patched Redux binary is missing. When all inputs are present, it
boots BIOS plus disc in both emulators, advances to the checkpoint, and
requires the visible display dimensions and hash to match Redux.

## Managing Redux changes

Keep Redux changes in the Redux fork/branch. Keep PSoXide changes in
this repo. Do not vendor a Redux binary into PSoXide.

When a new oracle feature needs a Redux C++ Lua binding:

1. Commit the binding in `EBonura/pcsx-redux:psoxide-bindings`.
2. Rebuild Redux.
3. Update this document's known commit.
4. Update `oracle.lua`'s required-binding header.
5. Add or update a PSoXide smoke test that fails clearly when the
   binding is missing.

The PSoXide commit should document the Redux commit it expects. The
operator chooses that build through `PSOXIDE_REDUX_BIN`.
