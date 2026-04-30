# Milestone ladder

PSoXide's progress is organized around named-game canaries. Each milestone locks in one specific capability, identified by a single game (or BIOS state), and, once passed, becomes a regression test that never re-breaks. The ladder is intentionally forward-loaded: the first few milestones are tight and early; the later ones are open-ended research.

## Why this shape

Previous attempts (`psoxide`, `PSoXide-2`) showed that "support all PS1 games" is a rapidly exploding goalpost. A game list is infinite; a named canary is finite. Instead of chasing compatibility percentages, each milestone names exactly one game that must reach exactly one observable state. When it does, we freeze a frame-hash / VRAM-dump / CPU-trace golden for it and the canary stays in the suite forever.

Commercial-game compatibility is treated as *validation*, not as a goal -- if the engine's output games run on real PS1 hardware, the same emulator must run real PS1 games. The two are coupled on purpose.

## The ladder

| # | Milestone | Canary | What it proves |
|---|---|---|---|
| **A** | BIOS boots to Sony logo | SCPH1001 (US) | CPU + GPU init, DMA chain, first pixels |
| **B** | BIOS boots to shell (no disc) | SCPH1001 | CD-ROM "no disc" path, menu rendering |
| **C** | Homebrew SDK triangle renders | own SDK | **MVP** -- full stack works end-to-end |
| **D** | BIOS disc-check passes | Crash Bandicoot | CD-ROM read + `SYSTEM.CNF` parse |
| **E** | Title screen renders | Crash Bandicoot | GTE + basic DMA + SPU init |
| **F** | Intro FMV plays | Crash Bandicoot | MDEC + XA audio + CD streaming |
| **G** | Complex MDEC cutscenes | Metal Gear Solid | Heavy MDEC, VAG audio, long CD streams |
| **H** | Multi-track audio CD | Resident Evil 2 | CD-DA tracks + pre-rendered backgrounds |
| **I** | 60 fps timing-critical | Tekken 3 | The hardest commercial title -- previous project's ceiling |
| **J** | PAL region + timing | WipEout 2097 | 50 Hz path, different timer constants |
| **K** | Stretch: heaviest GTE + streaming | Gran Turismo 2 | Full-system load |

## BIOS regions on hand

- `SCPH1001.BIN` -- US, NTSC
- `SCPH5500.BIN` -- JP
- `SCPH5501.BIN` -- US (later revision)
- `SCPH5502.BIN` -- EU, PAL

`SCPH1001` is the default for Milestones A / B / D–I. PAL timing (Milestone J) uses `SCPH5502`. Region-sensitive tests may run against all four.

## What's not a milestone

No "simple 2D" rung -- the library is 3D-heavy, so Milestone E (Crash title screen) is already exercising GTE + SPU + DMA + CD-ROM simultaneously. Milestones C → E are tightly coupled: reaching E is the first real proof the core is correct.

No compatibility percentage. If a game not on the ladder happens to work, good; if it doesn't, that's not a regression.

## Current status

As of 2026-04-19 (see `emu/crates/emulator-core/tests/milestones.rs` for the frozen-hash tests):

- **A -- BIOS Sony logo ✅ passes.** 100M-step VRAM hash pinned. Display-area hash is now **byte-exact with PCSX-Redux**: `0xa3ac6881044333d0` on both sides (0 / 611840 diverging bytes). Proven across all 8 game discs in the collection via `probe_all_games_100m`.
- **B -- BIOS shell (no disc) ✅ passes.** 500M-step hash pinned as a self-regression; Redux-verified hash at 500M is a TBD capture because the oracle's `run N` costs ~25s per million steps.
- **C -- Homebrew SDK triangle.** Scaffolding complete: six SDK crates (`psx-sdk`, `psx-hw`, `psx-rt`, `psx-io`, `psx-gpu`, `psx-gte`, `psx-pad`) plus five examples (`hello-tri`, `hello-tex`, `hello-ot`, `hello-input`, `hello-gte`). The examples build and side-load via `PSOXIDE_EXE=…`; a true end-to-end MVP test (load → draw → hash on real hardware) is still pending.
- **D -- BIOS disc-check ⚠ partial.** `milestone_d_bios_accepts_licensed_disc` passes after the LWL shift-inversion fix (commit `71979f1`) -- Crash Bandicoot no longer wild-jumps at step 208M, and the guard runs 300M instructions of Crash before snapshotting. `milestone_d_tekken_licensed_screen` also passes (Tekken holds stably on the SCEA license screen). **Remaining gap**: cycle-accuracy drift makes our rendered frame at Crash 300M a different animation step than Redux's -- the renderer is byte-exact on a static frame, but we're at a different virtual clock than Redux (≈ -6% at Crash 900M, per `probe_cycle_parity`).
- **E–K** -- not yet reached.

**Instruction-level parity ceiling**: the cached Redux trace at `target/parity-cache/redux-*-50000000.bin` matches our emulator lock-step up to step **19,474,543** (pc + instr + gprs + tick). The first full divergence at 19,474,544 is a DMA-IRQ scheduling-order difference (captured by `probe_cycle_first_divergence`). Before that, everything -- registers, memory accesses, cycle counts -- matches Redux exactly.

**Active investigation threads**:
- Closing the Crash cycle-accuracy gap (currently -6% at 900M) needs identifying why our BIOS issues 5 DMA schedules in a window Redux issues 3 -- the extra schedules overwrite scheduler targets and delay IRQ fire. `probe_dma_schedules` + `probe_dma_timeline` + `probe_isr_trace` are the tools for this.
- Redux-verified 500M / D display hashes are pending fresh capture (oracle is slow; not blocking).

See [../PROGRESS.md](../PROGRESS.md) for the chronological log.
