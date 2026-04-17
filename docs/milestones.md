# Milestone ladder

PSoXide's progress is organized around named-game canaries. Each milestone locks in one specific capability, identified by a single game (or BIOS state), and, once passed, becomes a regression test that never re-breaks. The ladder is intentionally forward-loaded: the first few milestones are tight and early; the later ones are open-ended research.

## Why this shape

Previous attempts (`psoxide`, `PSoXide-2`) showed that "support all PS1 games" is a rapidly exploding goalpost. A game list is infinite; a named canary is finite. Instead of chasing compatibility percentages, each milestone names exactly one game that must reach exactly one observable state. When it does, we freeze a frame-hash / VRAM-dump / CPU-trace golden for it and the canary stays in the suite forever.

Commercial-game compatibility is treated as *validation*, not as a goal — if the engine's output games run on real PS1 hardware, the same emulator must run real PS1 games. The two are coupled on purpose.

## The ladder

| # | Milestone | Canary | What it proves |
|---|---|---|---|
| **A** | BIOS boots to Sony logo | SCPH1001 (US) | CPU + GPU init, DMA chain, first pixels |
| **B** | BIOS boots to shell (no disc) | SCPH1001 | CD-ROM "no disc" path, menu rendering |
| **C** | Homebrew SDK triangle renders | own SDK | **MVP** — full stack works end-to-end |
| **D** | BIOS disc-check passes | Crash Bandicoot | CD-ROM read + `SYSTEM.CNF` parse |
| **E** | Title screen renders | Crash Bandicoot | GTE + basic DMA + SPU init |
| **F** | Intro FMV plays | Crash Bandicoot | MDEC + XA audio + CD streaming |
| **G** | Complex MDEC cutscenes | Metal Gear Solid | Heavy MDEC, VAG audio, long CD streams |
| **H** | Multi-track audio CD | Resident Evil 2 | CD-DA tracks + pre-rendered backgrounds |
| **I** | 60 fps timing-critical | Tekken 3 | The hardest commercial title — previous project's ceiling |
| **J** | PAL region + timing | WipEout 2097 | 50 Hz path, different timer constants |
| **K** | Stretch: heaviest GTE + streaming | Gran Turismo 2 | Full-system load |

## BIOS regions on hand

- `SCPH1001.BIN` — US, NTSC
- `SCPH5500.BIN` — JP
- `SCPH5501.BIN` — US (later revision)
- `SCPH5502.BIN` — EU, PAL

`SCPH1001` is the default for Milestones A / B / D–I. PAL timing (Milestone J) uses `SCPH5502`. Region-sensitive tests may run against all four.

## What's not a milestone

No "simple 2D" rung — the library is 3D-heavy, so Milestone E (Crash title screen) is already exercising GTE + SPU + DMA + CD-ROM simultaneously. Milestones C → E are tightly coupled: reaching E is the first real proof the core is correct.

No compatibility percentage. If a game not on the ladder happens to work, good; if it doesn't, that's not a regression.

## Current status

As of the last commit on `main`:

- **Before A**: CPU + SYSCALL + exception machinery + typed register scaffolding for IRQ / Timers / DMA / GPU / SPU.
- **Parity**: bit-identical against PCSX-Redux through at least 10M BIOS instructions with SCPH1001 (`make parity`).
- **First real DMA**: OTC channel, ordering-table initialization.
- **Next**: cycle model + VBlank generation — the piece that breaks parity deliberately and pivots the harness from "bit-identical to instruction N" to "canary X reaches state Y".

See [PROGRESS.md](../PROGRESS.md) (not yet written) for a running activity log.
