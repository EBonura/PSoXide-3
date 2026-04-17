# Frontend architecture

The desktop frontend lives in [`emu/crates/frontend`](../emu/crates/frontend). It's a single-threaded wgpu + egui app that drives the emulator core directly — no `Arc<Mutex<_>>`, no message-passing, no separate render thread. The UI reads emulator state in place each frame.

## Why single-threaded?

Both prior attempts (`psoxide`, `PSoXide-2`) experimented with threaded architectures and consistently paid more in debugging cost than they saved in performance. A 33.87 MHz PS1 CPU is not going to out-run a modern host in any foreseeable scenario; serializing CPU stepping and rendering on the same thread keeps the debugger honest and the code boring.

## Stack

| Crate | Version | Role |
|---|---|---|
| `winit` | 0.30 | Window + event loop |
| `wgpu` | 24 | GPU surface + texture upload |
| `egui` | 0.31 | Immediate-mode UI |
| `egui-wgpu` | 0.31 | egui backend against wgpu |
| `egui-winit` | 0.31 | egui input from winit events |
| `pollster` | 0.3 | Block-on for wgpu async setup |

These are pinned deliberately — both priors independently converged on this set, and any bump is a conscious decision, not drift.

## Module layout

```text
src/
├── main.rs           # winit ApplicationHandler shell, event dispatch, run loop
├── app.rs            # AppState — emulator + UI state, no Arc/Mutex
├── gfx.rs            # Graphics — wgpu surface + egui renderer + VRAM texture
├── theme.rs          # charcoal/teal palette, VT323 + Lucide fonts, section helpers
├── icons.rs          # Lucide codepoint constants
└── ui/
    ├── mod.rs        # draw_layout — composes panels in layer order
    ├── registers.rs  # CPU + COP0 + history + breakpoints side panel
    ├── memory.rs     # hex+ASCII viewer, quick-jump, BP toggle
    ├── vram.rs       # 1024×512 VRAM as an egui::Image
    ├── hud.rs        # FPS / frame-time / tick overlay
    └── xmb.rs        # XrossMediaBar menu, Painter-drawn on Middle layer
```

## Layer order, outside-in

1. **Central panel** — the future PS1 framebuffer. Currently placeholder text.
2. **Register side panel** (left, `egui::SidePanel::left`).
3. **Memory side panel** (right, `egui::SidePanel::right`, hidden by default).
4. **VRAM bottom panel** (`egui::TopBottomPanel::bottom`).
5. **XMB overlay** on `egui::Order::Middle` — dims background, slides animated category icons.
6. **HUD bar** on `egui::Order::Foreground` — always on top, above XMB.

Each panel is its own module, so adding a new one is about 150 lines and touching `ui/mod.rs`' layout-orchestration function.

## Data flow per frame

```text
winit event
  → Shell::window_event
    → (keyboard?) merge_key() into pending XmbInput
    → (redraw?) → run_frame:
        1. dt from Instant
        2. XmbState::update(input) → Option<MenuAction>
        3. ui::apply_menu_action (run/step/reset/toggle panels)
        4. run loop: bus + cpu → exec_history ring, breakpoint check
        5. Graphics::prepare_vram(state.bus?.gpu.vram)
        6. Graphics::render(|ctx| ui::draw_layout(...))
```

Key pattern: `run_frame` destructures `state` so `state.bus`, `state.cpu`, and `state.exec_history` are three disjoint field borrows Rust accepts simultaneously. A `&mut self` method on `AppState` would block that.

## VRAM upload

`Graphics` owns a persistent `wgpu::Texture` (1024×512, `Rgba8UnormSrgb`, `TEXTURE_BINDING | COPY_DST`) registered with the egui-wgpu renderer as a native texture once at startup. Every frame, `prepare_vram` decodes the 16bpp VRAM into an RGBA8 scratch buffer (full-range `(v<<3)|(v>>2)` expansion, not the naive `v<<3` that loses 8% of white brightness) and `queue.write_texture`s it onto the persistent target.

The VRAM panel then renders the single `egui::Image` referencing this texture — all three panels (game view, VRAM view, and future framebuffer clip) will eventually share the same upload by differing only in their `uv` rect.

## XMB mechanics

Ported from `psoxide-1`'s `xmb.rs`, trimmed to three categories (Game / Debug / System) plus infrastructure for expansion. Drawn entirely through `egui::Painter` on a middle layer — no high-level widgets — which keeps it snappy and position-locked.

- `anim_x` interpolates toward `target_x` at `10/dt`, yielding the signature horizontal slide.
- Selection uses a 3-pixel accent-color bar on the left edge of the item rect.
- Input: arrows navigate, Enter/Space confirms, Escape toggles open/closed.
- Gamepad support will land alongside the controller subsystem.

## Keyboard shortcuts

| Key | Action |
|---|---|
| Esc | Toggle XMB (and back-out when navigating) |
| ↑ ↓ ← → | Navigate XMB (items / categories) |
| Enter / Space | Confirm item |

## Debugging loop

The frontend is designed to double as a live debugger:

1. Pause (XMB → Game → Pause, or just open the XMB with Esc).
2. Scroll the memory viewer with the quick-jump buttons to interesting addresses.
3. Toggle a breakpoint with "Set BP" at the current viewer address; the row highlights in the accent color.
4. Resume. The run loop checks `breakpoints.contains(&cpu.pc())` before each `cpu.step` and pauses on match.
5. The register panel shows last 64 retired instructions, live COP0 state, and all active breakpoints.

## What's intentionally absent

- **No gamepad**. Lands with the controller subsystem (SIO0).
- **No audio output**. Lands with SPU voice playback.
- **No framebuffer display**. Lands with the GPU rasterizer — the central panel stays placeholder until then.
- **No save states**. Will require a serializable-state contract across all subsystems.

All of these are part of the canary ladder's later milestones; the frontend scaffolding is in place to slot them in when they land.
