<p align="center">
  <img src="assets/branding/logo-wordmark.svg" alt="PSoXide" width="420">
</p>

<h1 align="center">PSoXide</h1>

<p align="center">
  <strong>A Rust-native PlayStation 1 development platform — full-stack rewrite, third attempt.</strong>
</p>

<p align="center">
  <img src="assets/branding/logo-icon-player.png" alt="PSoXide player" width="120">
</p>

PSoXide aims at an emulator + SDK + engine co-designed around the same hardware model, so asset formats, BIOS call shapes, and GPU command layouts stay coherent across the stack. The ultimate goal is an engine tuned tightly enough to push the PS1 to its limits — down to the kind of work FromSoftware did on their own PS1 RPGs (King's Field, Shadow Tower).

This is attempt number three. It starts honest: a CPU interpreter whose every instruction is validated bit-identically against PCSX-Redux, peripherals that grow one typed subsystem at a time, and a frontend designed for debugging from day one.

## Status

Early — and deliberately so. The current milestones:

- **CPU**: all R3000A opcodes the BIOS needs for its first 2.735 million instructions, validated bit-identically against PCSX-Redux (`make parity`). SYSCALL / BREAK exception machinery, COP0 stack push, delay-slot-aware EPC.
- **Peripherals**: typed register scaffolding for IRQ controller, Timers (Root Counters 0/1/2), DMA controller (7 channels + DPCR/DICR), GPU (GPUSTAT + GP0/GP1 ports + VRAM). No cycle-accurate ticking yet — that lands alongside VBlank generation.
- **Frontend**: winit + wgpu + egui desktop app with a PS3-style Menu menu, live register/VRAM/memory panels, breakpoints, exec history, continuous-run mode, and an HUD overlay. See [docs/frontend.md](docs/frontend.md) for the architecture.
- **SDK / engine**: not started yet. They land after Milestone A (Sony logo boots) per the plan.

The next target is **Milestone A**: boot through to the Sony logo. That needs DMA transfers wired up, a GPU command decoder, a rasterizer, and some cycle model to drive VBlank. Weeks of work; visible progress at the end.

## Repository Layout

```text
.
├── crates/                 no_std target-agnostic shared crates
│   ├── psx-hw              memory map, register addresses, HW constants
│   ├── psx-iso             BIN/CUE + ISO9660 (placeholder; CD-ROM lands later)
│   └── psx-trace           per-instruction trace record format (emu + oracle emit it)
│
├── emu/                    host workspace (emulator + frontend + parity oracle)
│   ├── crates/
│   │   ├── emulator-core   CPU + Bus + Vram + IRQ + Timers + DMA + Gpu
│   │   ├── frontend        winit + wgpu + egui desktop app
│   │   └── parity-oracle   headless PCSX-Redux harness for trace comparison
│   └── Cargo.toml          host-side workspace, stable toolchain
│
├── sdk/                    PS1-targeted workspace (empty until Milestone C)
├── tests/                  top-level integration tests (parity goldens, etc.)
├── docs/                   design notes, hardware refs
├── assets/                 branding, static resources
└── Makefile                common entry points
```

## Workspaces

| Workspace | Path | What it contains | Target |
|-----------|------|------------------|--------|
| Root | `/` | shared `no_std` crates (`psx-hw`, `psx-iso`, `psx-trace`) | any |
| Emulator | `emu/` | `emulator-core`, `frontend`, `parity-oracle` | host |
| SDK | `sdk/` | (empty — lands with Milestone C) | `mipsel-sony-psx` |

Splitting into separate workspaces lets the SDK target `mipsel-sony-psx` with `build-std` on nightly without polluting the host workspace's toolchain.

## Development

### Requirements

- Rust stable (for `emu/` and root)
- A PS1 BIOS image at `PSOXIDE_BIOS` (default: `/Users/ebonura/Downloads/ps1 bios/SCPH1001.BIN`)
- [PCSX-Redux](https://github.com/grumpycoders/pcsx-redux) built from source for parity testing (with patched-in `stepIn` / `runExecute` / `setQuietPauseResume` Lua bindings)

### Common commands

```bash
make check        # cargo check across both workspaces
make test         # fast unit tests (both workspaces, excludes canaries)
make canaries     # commercial-game canary tests (Milestones D–K)
make fmt          # format both workspaces
make lint         # clippy -D warnings
make run          # launch the desktop frontend
make parity       # step emulator + Redux and assert bit-identical traces
```

### Canary milestone ladder

Validation is organized around named-game canaries rather than "support all games." Each milestone locks in a specific capability, then becomes a regression test:

| # | Milestone | Canary |
|---|---|---|
| A | BIOS boots to Sony logo | SCPH1001 |
| B | BIOS boots to shell (no disc) | SCPH1001 |
| C | Homebrew SDK triangle renders | your SDK |
| D | BIOS disc-check passes | Crash Bandicoot |
| E | Title screen renders | Crash Bandicoot |
| F | Intro FMV plays | Crash Bandicoot |
| G | Complex MDEC cutscenes | Metal Gear Solid |
| H | Multi-track audio CD | Resident Evil 2 |
| I | 60 fps timing-critical | Tekken 3 |
| J | PAL region + timing | WipEout 2097 |
| K | Stretch: heaviest GTE | Gran Turismo 2 |

## Design principles

- **Parity first on the CPU**, then pivot to subsystem-capability on peripherals. Bit-identical traces against a real emulator catch bugs cheaply at the instruction level; peripherals are too stateful for that to scale.
- **Debug instrumentation is compile-time gated.** Cargo features, not runtime flags. Release binaries have zero debug paths — direct lesson from psoxide's perf-degrading trace accumulators.
- **No UI in the core.** `emulator-core` outputs state; the frontend is a separate consumer. Makes VRAM regression debuggable as "was the state wrong, or was the rendering wrong?"
- **Commercial-game compatibility is validation, not a goal.** If Crash boots, the engine can target real hardware with confidence. If it doesn't, there's an accuracy gap the engine would inherit silently.

## Prior art

PSoXide is the third attempt; the first two taught the architectural lessons:

- **[psoxide](../psoxide)**: mature CPU, full SDK, working frontend. Stalled on the exponential-tail work of commercial-game parity. Its Menu menu and HUD bar patterns live on here.
- **[PSoXide-2](../PSoXide-2)**: rewrite attempt that got to boot + flat triangles but never reached full stack. Its Redux-oracle harness and debug-panel layout are the direct ancestors of the current tooling.

## License

GPL-2.0-or-later
