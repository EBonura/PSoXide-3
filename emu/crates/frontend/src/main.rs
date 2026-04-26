//! PSoXide desktop frontend.
//!
//! Modular layout:
//! - `theme`   — fonts, colors, framed-section helpers.
//! - `icons`   — Lucide codepoint constants.
//! - `gfx`     — winit window + wgpu surface + egui-wgpu plumbing.
//! - `app`     — top-level state, UI orchestration entry point.
//! - `ui/*`    — individual panels (central, registers, vram, menu, hud).

#![warn(missing_docs)]

mod app;
mod audio;
mod cli;
mod disasm;
mod gfx;
mod icons;
mod input;
mod theme;
mod ui;

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::app::AppState;
use crate::cli::Cli;
use crate::gfx::Graphics;
use crate::ui::{menu::MenuInput, MenuOutcome};

use emulator_core::{button, spu::SAMPLE_CYCLES};
use psoxide_settings::settings::{InputBinding, PortBindings};

/// Default window size when not running fullscreen. Chosen big
/// enough to show the Menu + a framebuffer comfortably on a
/// standard laptop display.
const INITIAL_WIDTH: u32 = 1600;
const INITIAL_HEIGHT: u32 = 1000;
/// Keep the toolbar usable: full debug controls + boot toggle +
/// volume slider + transport buttons need roughly 700 logical px on
/// Retina displays, and the initial window is already larger.
const MIN_WIDTH: u32 = 1400;
const MIN_HEIGHT: u32 = 700;
/// Frontend run cadence target. The toolbar, "advance one frame"
/// control, and sample pump all assume an NTSC-ish 60 Hz shell.
const TARGET_FRAME_DT: f32 = 1.0 / 60.0;
/// Don't try to catch up an arbitrarily long stall in one redraw;
/// cap the burst so a debugger stop or window drag doesn't spend
/// seconds chewing through delayed emu frames.
const MAX_CATCHUP_FRAMES: u32 = 4;

fn main() {
    // Argument parsing first — if a subcommand is present, we
    // dispatch through the headless CLI and never open a window.
    // Clap's derive API panics with a nicely-formatted message on
    // bad arguments, which is exactly what a CLI user expects.
    let cli = Cli::parse();
    if cli.command.is_some() {
        if let Err(e) = cli::run(cli) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }

    // `--config-dir` also applies to the GUI path — lets testers
    // point the app at a scratch directory without touching their
    // real settings. Ditto `--fullscreen`.
    let config_dir = cli.config_dir;
    let fullscreen = cli.fullscreen;
    let gpu_compute = cli.gpu_compute;

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = Shell::new(config_dir, fullscreen, gpu_compute);
    event_loop.run_app(&mut app).expect("event loop");
}

struct Shell {
    graphics: Option<Graphics>,
    state: AppState,
    pending_input: MenuInput,
    last_frame: Instant,
    /// Live port-1 pad mask. Key press/release events toggle bits
    /// here; the shell flushes it into `bus.set_port1_buttons` each
    /// frame before running CPU steps so the guest always sees the
    /// latest state.
    pad1_mask: u16,
    /// Whether to open the window in borderless-fullscreen mode.
    /// Decision is made at startup via CLI flag and then captured
    /// here; changing it at runtime would need a window recreation.
    fullscreen: bool,
    /// Host audio output. `None` when no device is available
    /// (headless CI, devices that can't open a stereo stream).
    /// Emulation keeps running regardless — silence is fine.
    audio: Option<audio::AudioOut>,
    /// Host input router — tracks every connected gamepad, emits
    /// merged PSX pad-1 masks, detects the Select+Start chord
    /// that opens the Menu, and logs connect / disconnect events
    /// for diagnosing missing controllers. Always constructible;
    /// a failed gilrs init just produces empty frames so the
    /// keyboard path keeps working.
    input: input::InputRouter,
    /// Wall-clock debt waiting to be converted into emulated
    /// "frames". Without this, the current `ControlFlow::Poll`
    /// shell runs the guest as fast as redraws can arrive, which
    /// massively overfills the audio queue and produces crackle
    /// from dropped samples.
    emu_frame_accum: f32,
    /// Residual emulated CPU cycles that haven't yet been converted
    /// into SPU sample ticks. Redux clocks the SPU at 44.1 kHz from
    /// the PSX master clock (768 cycles/sample); tying audio to host
    /// redraws instead produces under/over-runs on anything that
    /// isn't an exact 60 Hz render loop.
    audio_cycle_accum: u64,
    /// Phase C — when `Some`, the experimental compute-shader
    /// rasterizer is shadowing the CPU rasterizer: each frame the
    /// CPU's `cmd_log` is drained and replayed onto the GPU compute
    /// path, and the display reads from the GPU's VRAM.
    compute_backend: Option<psx_gpu_compute::ComputeBackend>,
    /// Whether to display the GPU compute output instead of the CPU
    /// VRAM. Toggled at runtime by F12. Independent of whether the
    /// compute backend is active — when off, GPU still runs (so it
    /// stays in sync) but the user sees CPU output.
    display_gpu_compute: bool,
}

impl Default for Shell {
    fn default() -> Self {
        Self::new(None, false, false)
    }
}

impl Shell {
    fn new(config_dir: Option<std::path::PathBuf>, fullscreen: bool, gpu_compute: bool) -> Self {
        let audio = audio::AudioOut::open();
        if let Some(a) = audio.as_ref() {
            eprintln!("[audio] opened host stream @ {} Hz", a.host_sample_rate());
        } else {
            eprintln!("[audio] no host output device available — running silent");
        }
        let input = input::InputRouter::new();
        if input.is_connected() {
            eprintln!(
                "[input] already-connected pads: {}",
                input.connected_names()
            );
        } else {
            eprintln!("[input] no pads connected at startup — watching for hot-plug");
        }
        // The compute backend gets its own headless wgpu device.
        // We *could* share the main `Graphics` device for zero-copy
        // VRAM-to-display, but that needs `Arc<Device>` plumbing
        // throughout `Graphics` — bigger refactor for a marginal
        // perf win in an opt-in shadow path. Per-frame VRAM bounces
        // through CPU memory, which costs ~1 MiB read + 1 MiB write
        // and is invisible next to the rasterizer cost.
        let compute_backend = if gpu_compute {
            eprintln!("[gpu-compute] enabling shadow compute rasterizer");
            Some(psx_gpu_compute::ComputeBackend::new_headless())
        } else {
            None
        };
        Self {
            graphics: None,
            state: AppState::with_config_dir(config_dir),
            pending_input: MenuInput::default(),
            last_frame: Instant::now(),
            pad1_mask: 0,
            fullscreen,
            audio,
            input,
            emu_frame_accum: 0.0,
            audio_cycle_accum: 0,
            compute_backend,
            display_gpu_compute: gpu_compute,
        }
    }
}

/// Map a winit logical key to a PSX digital-pad bitmask using the
/// persisted port-1 bindings. Returns `None` for keys that aren't
/// bound.
fn key_to_pad_button(key: &Key, bindings: &PortBindings) -> Option<u16> {
    [
        (button::UP, &bindings.up),
        (button::DOWN, &bindings.down),
        (button::LEFT, &bindings.left),
        (button::RIGHT, &bindings.right),
        (button::CROSS, &bindings.cross),
        (button::CIRCLE, &bindings.circle),
        (button::SQUARE, &bindings.square),
        (button::TRIANGLE, &bindings.triangle),
        (button::L1, &bindings.l1),
        (button::R1, &bindings.r1),
        (button::L2, &bindings.l2),
        (button::R2, &bindings.r2),
        (button::START, &bindings.start),
        (button::SELECT, &bindings.select),
    ]
    .into_iter()
    .find_map(|(mask, binding)| binding_matches_key(binding, key).then_some(mask))
}

fn binding_matches_key(binding: &InputBinding, key: &Key) -> bool {
    match (binding, key) {
        (InputBinding::Unbound, _) => false,
        (InputBinding::Character(expected), Key::Character(actual)) => actual
            .chars()
            .next()
            .is_some_and(|c| c.eq_ignore_ascii_case(expected)),
        (InputBinding::Named(expected), Key::Named(actual)) => {
            named_key_label(actual).is_some_and(|name| expected.eq_ignore_ascii_case(name))
        }
        _ => false,
    }
}

fn named_key_label(key: &NamedKey) -> Option<&'static str> {
    match key {
        NamedKey::ArrowUp => Some("ArrowUp"),
        NamedKey::ArrowDown => Some("ArrowDown"),
        NamedKey::ArrowLeft => Some("ArrowLeft"),
        NamedKey::ArrowRight => Some("ArrowRight"),
        NamedKey::Enter => Some("Enter"),
        NamedKey::Backspace => Some("Backspace"),
        NamedKey::Shift => Some("Shift"),
        NamedKey::Space => Some("Space"),
        NamedKey::Tab => Some("Tab"),
        NamedKey::Escape => Some("Escape"),
        _ => None,
    }
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.graphics.is_some() {
            return;
        }

        // Borderless-fullscreen on the primary monitor when
        // `--fullscreen` was passed. Falls back to a windowed
        // 1600×1000 otherwise so development on a laptop with
        // panels + a terminal remains bearable.
        let mut attrs = Window::default_attributes()
            .with_title("PSoXide")
            .with_inner_size(winit::dpi::PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT))
            .with_min_inner_size(winit::dpi::PhysicalSize::new(MIN_WIDTH, MIN_HEIGHT));
        if self.fullscreen {
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        self.graphics = Some(pollster::block_on(Graphics::new(window)));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gfx) = self.graphics.as_mut() else {
            return;
        };

        let consumed = gfx.egui_winit.on_window_event(&gfx.window, &event).consumed;

        match event {
            WindowEvent::CloseRequested => {
                // Flush any dirty memory card so save progress
                // survives a window-close. A hard crash still
                // loses whatever hasn't been flushed — the run
                // loop could call this periodically; for now
                // graceful exit is enough.
                if let Err(e) = self.state.flush_memcard_port1() {
                    eprintln!("[frontend] memcard flush on exit: {e}");
                }
                if let Err(e) = self.state.save_editor_project() {
                    eprintln!("[frontend] editor save on exit: {e}");
                }
                // Persist current settings (BIOS path, library
                // root, etc.) so the next launch picks up any
                // user tweaks without needing a manual save step.
                if let Err(e) = self.state.save_settings() {
                    eprintln!("[frontend] settings save on exit: {e}");
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gfx.resize(size);
                gfx.window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                // Pad state tracks both press AND release continuously
                // so held buttons keep polling as "pressed". Auto-repeat
                // events are ignored — the key is already down, and the
                // BIOS polls every frame anyway.
                if !repeat && !self.state.workspace.is_editor() {
                    if let Some(mask) =
                        key_to_pad_button(&logical_key, &self.state.settings.input.port1)
                    {
                        match state {
                            ElementState::Pressed => self.pad1_mask |= mask,
                            ElementState::Released => self.pad1_mask &= !mask,
                        }
                    }
                }
                // The Menu *does* honour OS-level key-repeat: holding
                // down-arrow scrolls through a long Examples list one
                // row per repeat tick, matching GUI-standard behaviour.
                // Only press events (including repeats) trigger menu
                // navigation; releases don't.
                if state == ElementState::Pressed {
                    self.pending_input = merge_key(self.pending_input, &logical_key);
                }
                // F12 — toggle the display source between the CPU
                // rasterizer's VRAM and the compute backend's. Only
                // meaningful when the compute backend is active
                // (i.e. `--gpu-compute` was passed). No-op otherwise.
                if state == ElementState::Pressed && !repeat {
                    if matches!(&logical_key, Key::Named(NamedKey::F12)) {
                        self.display_gpu_compute = !self.display_gpu_compute;
                        eprintln!(
                            "[gpu-compute] display source: {}",
                            if self.display_gpu_compute {
                                "GPU compute"
                            } else {
                                "CPU rasterizer"
                            }
                        );
                    }
                }
                gfx.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                let mut input = std::mem::take(&mut self.pending_input);

                // Poll the gamepad router BEFORE doing anything
                // else for this frame: the event drain is what
                // lets gilrs notice hot-plugged Bluetooth pads, so
                // we can't gate it on run state. We then merge the
                // frame's edges into `MenuInput` and keep the merged
                // mask handy for the run branch further down.
                let pad_frame = self.input.poll();
                if !pad_frame.notices.is_empty() {
                    let msg = pad_frame
                        .notices
                        .iter()
                        .map(|notice| notice.message())
                        .collect::<Vec<_>>()
                        .join(" · ");
                    self.state.status_message_set(msg);
                }
                if pad_frame.toggle_menu {
                    // Select+Start is the gamepad equivalent of
                    // Escape — route it into the same `toggle_open`
                    // path so there's exactly one place that decides
                    // what "PS button" does based on current state.
                    input.toggle_open = true;
                }
                // When the Menu is open OR currently paused, the
                // gamepad doubles as the menu navigator. D-pad /
                // left-stick edges become up/down/left/right, Cross
                // is Enter, Circle is Back. `|=` so keyboard and
                // pad can both contribute — last-one-wins semantics
                // don't matter at this granularity.
                input.up |= pad_frame.menu_up;
                input.down |= pad_frame.menu_down;
                input.left |= pad_frame.menu_left;
                input.right |= pad_frame.menu_right;
                input.confirm |= pad_frame.menu_confirm;
                input.back |= pad_frame.menu_back;

                // Escape is the "PS button" — it toggles between
                // "game running" and "game paused + menu open".
                // Intercept it here so the Menu doesn't also interpret
                // it as a navigation input. The user pressed Escape
                // (or Select+Start, now) to swap contexts, not to
                // press "back" on whatever menu item happened to
                // be highlighted.
                if input.toggle_open {
                    input.toggle_open = false;
                    input.back = false;
                    if self.state.running {
                        // Game mode → menu mode: pause and open overlay.
                        self.state.running = false;
                        self.state.menu.sync_run_label(false);
                        self.state.menu.open = true;
                    } else if self.state.menu.open {
                        // Menu mode → game mode: resume if we have a
                        // live game to resume; otherwise just close
                        // the overlay.
                        self.state.menu.open = false;
                        if self.state.bus.is_some() && self.state.current_game.is_some() {
                            self.state.running = true;
                            self.state.menu.sync_run_label(true);
                        }
                    } else {
                        // No game running and Menu already closed —
                        // Escape just opens the menu.
                        self.state.menu.open = true;
                    }
                }

                if let Some(action) = self.state.menu.update(&input) {
                    if ui::apply_menu_action(&mut self.state, action) == MenuOutcome::Quit {
                        if let Err(e) = self.state.flush_memcard_port1() {
                            eprintln!("[frontend] memcard flush on quit: {e}");
                        }
                        if let Err(e) = self.state.save_editor_project() {
                            eprintln!("[frontend] editor save on quit: {e}");
                        }
                        if let Err(e) = self.state.save_settings() {
                            eprintln!("[frontend] settings save on quit: {e}");
                        }
                        event_loop.exit();
                        return;
                    }
                }

                // Arm GPU command capture before stepping so the HW /
                // compute sidecars see the frame that is about to run.
                // Re-arming clears the log, so only do this once per
                // Bus lifetime.
                if let Some(bus) = self.state.bus.as_mut() {
                    if self.compute_backend.is_some() {
                        if bus.gpu.pixel_owner.is_none() {
                            bus.gpu.enable_pixel_tracer();
                        }
                    } else if !bus.gpu.cmd_log_enabled() {
                        bus.gpu.enable_cmd_log();
                    }
                }

                // Run loop: retire one NTSC frame's worth of PSX cycles
                // if we're in run mode. Any execution error auto-pauses
                // and surfaces via the register panel. History captures
                // only the tail via `push_history`'s ring-buffer semantics.
                if self.state.running {
                    self.emu_frame_accum = (self.emu_frame_accum + dt).min(0.25);
                    let frames_to_run =
                        ((self.emu_frame_accum / TARGET_FRAME_DT) as u32).min(MAX_CATCHUP_FRAMES);
                    // Merge the current keyboard-derived pad mask with
                    // gamepad input before stepping, so the game/homebrew
                    // sees fresh input this frame. `pad_frame.pad1_mask`
                    // already has the Select+Start chord stripped for
                    // the frame the chord fires — prevents in-game
                    // handlers from seeing the "open menu" combo.
                    let pad_mask = self.pad1_mask | pad_frame.pad1_mask;
                    let (rx, ry) = pad_frame.right_stick;
                    let (lx, ly) = pad_frame.left_stick;
                    if let Some(bus) = self.state.bus.as_mut() {
                        bus.set_port1_buttons(emulator_core::ButtonState::from_bits(pad_mask));
                        // Forward analog sticks to the pad's analog
                        // axes so DualShock-aware games see joystick
                        // motion. Byte range is 0..=0xFF with 0x80 =
                        // centre; gilrs gives us -1.0..=1.0. Y axis
                        // is inverted on host gamepads (up = positive).
                        let map = |v: f32| ((v.clamp(-1.0, 1.0) * 127.0) + 128.0) as u8;
                        bus.set_port1_sticks(map(rx), map(-ry), map(lx), map(-ly));
                    }
                    for _ in 0..frames_to_run {
                        let cycles_before =
                            self.state.bus.as_ref().map(|bus| bus.cycles()).unwrap_or(0);
                        app::step_one_frame(&mut self.state);

                        // Pump the SPU by however much emulated time the
                        // CPU just advanced, not by "one host redraw".
                        // This keeps audio pacing tied to the PSX master
                        // clock even on 120 Hz / 144 Hz hosts or slow
                        // frames, matching the SPU's 768-cycles/sample
                        // timing model.
                        let effective_audio_volume = self.state.effective_audio_volume();
                        if let Some(bus) = self.state.bus.as_mut() {
                            let cycles_after = bus.cycles();
                            self.audio_cycle_accum = self
                                .audio_cycle_accum
                                .saturating_add(cycles_after.saturating_sub(cycles_before));
                            let sample_count = (self.audio_cycle_accum / SAMPLE_CYCLES) as usize;
                            self.audio_cycle_accum %= SAMPLE_CYCLES;
                            if sample_count > 0 {
                                bus.run_spu_samples(sample_count);
                            }
                            if let Some(audio) = self.audio.as_ref() {
                                audio.set_volume(effective_audio_volume);
                                let samples = bus.spu.drain_audio();
                                if !samples.is_empty() {
                                    audio.push_samples(&samples);
                                }
                                // Surface the cpal ring depth in the HUD.
                                self.state.hud.set_audio_queue_len(audio.queue_len());
                            } else {
                                // No output device — drain and discard so the
                                // SPU's internal queue doesn't grow unbounded.
                                let _ = bus.spu.drain_audio();
                            }
                        }
                    }
                    self.emu_frame_accum -= (frames_to_run as f32) * TARGET_FRAME_DT;
                } else {
                    self.emu_frame_accum = 0.0;
                    // Throw away any fractional carry when emulation is
                    // paused or no game is running so a later launch or
                    // resume doesn't inherit cycles from an older run.
                    self.audio_cycle_accum = 0;
                }

                let state = &mut self.state;

                let frame_log = if let Some(bus) = state.bus.as_mut() {
                    std::mem::take(&mut bus.gpu.cmd_log)
                } else {
                    Vec::new()
                };

                // Phase C: drain the CPU rasterizer's `cmd_log` and
                // replay each GP0 packet onto the compute backend.
                // This runs for every frame the bus advanced (or
                // not, when paused — in which case `cmd_log` will
                // be empty and the loop is a no-op).
                if let (Some(backend), Some(bus)) =
                    (self.compute_backend.as_mut(), state.bus.as_mut())
                {
                    // Sync VRAM so any uploads / FMV writes / VRAM-to-
                    // VRAM copies are reflected on the compute side
                    // before we replay this frame's draw commands.
                    backend.sync_vram_from_cpu(bus.gpu.vram.words());
                    for entry in &frame_log {
                        backend.replay_packet(entry);
                    }
                    // pixel_owner needs resetting too — we don't use
                    // its data here, but its `current_cmd_index`
                    // would otherwise drift past u32::MAX over time.
                    if let Some(owner) = bus.gpu.pixel_owner.as_mut() {
                        owner.fill(u32::MAX);
                    }
                }

                gfx.prepare_vram(state.bus.as_ref().map(|b| &b.gpu.vram));
                gfx.prepare_24bpp_display(state.bus.as_ref().map(|b| &b.gpu));

                // Match the HW renderer's internal scale to the
                // current Native↔Window mode + framebuffer pixel budget.
                // Cheap when stable; reallocates the VRAM-shaped
                // target on change (which clears it — the next
                // cmd_log replay paints a fresh frame).
                let scale_mode = match state.scale_mode {
                    app::ScaleMode::Native => psx_gpu_render::ScaleMode::Native,
                    app::ScaleMode::Window => psx_gpu_render::ScaleMode::Window,
                };
                let display_size = state
                    .bus
                    .as_ref()
                    .map(|b| {
                        let area = b.gpu.display_area();
                        ((area.width as u32).max(320), (area.height as u32).max(240))
                    })
                    .unwrap_or((320, 240));
                gfx.update_hw_scale(scale_mode, state.framebuffer_present_size_px, display_size);

                // Drive the hardware renderer once per frame. The
                // VRAM-shaped target persists across frames the way
                // PSX VRAM does; the framebuffer panel UV-samples
                // the active display sub-rect.
                if let Some(bus) = state.bus.as_mut() {
                    let vram_words = bus.gpu.vram.words().to_vec();
                    gfx.render_hw_frame(&bus.gpu, &frame_log, &vram_words);
                } else {
                    let empty_log: Vec<emulator_core::gpu::GpuCmdLogEntry> = Vec::new();
                    let empty_vram: Vec<u16> = vec![0; 1024 * 512];
                    let dummy_gpu = emulator_core::Gpu::new();
                    gfx.render_hw_frame(&dummy_gpu, &empty_log, &empty_vram);
                }

                let vram_tex = gfx.vram_texture_id();
                let use_24bpp_display = state
                    .bus
                    .as_ref()
                    .map(|b| b.gpu.display_area().bpp24)
                    .unwrap_or(false);
                let (display_tex, framebuffer_source) = if use_24bpp_display {
                    (
                        gfx.display_texture_id(),
                        ui::framebuffer::FramebufferSource::CpuDisplay,
                    )
                } else {
                    (
                        gfx.hw_texture_id(),
                        ui::framebuffer::FramebufferSource::HardwareVram,
                    )
                };
                gfx.render(|ctx| {
                    app::build_ui(ctx, state, vram_tex, display_tex, framebuffer_source, dt)
                });
            }
            _ => {
                if !consumed {
                    gfx.window.request_redraw();
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gfx) = self.graphics.as_ref() {
            gfx.window.request_redraw();
        }
    }
}

/// OR a keypress into the next-frame Menu input. `Escape` both toggles
/// the overlay and acts as back when navigating; the combined semantics
/// are handled inside `MenuState::update`.
fn merge_key(mut input: MenuInput, key: &Key) -> MenuInput {
    match key {
        Key::Named(NamedKey::ArrowUp) => input.up = true,
        Key::Named(NamedKey::ArrowDown) => input.down = true,
        Key::Named(NamedKey::ArrowLeft) => input.left = true,
        Key::Named(NamedKey::ArrowRight) => input.right = true,
        Key::Named(NamedKey::Enter) | Key::Named(NamedKey::Space) => input.confirm = true,
        Key::Named(NamedKey::Escape) => {
            input.toggle_open = true;
            input.back = true;
        }
        _ => {}
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_mapping_uses_default_settings() {
        let bindings = PortBindings::default();

        assert_eq!(
            key_to_pad_button(&Key::Character("x".into()), &bindings),
            Some(button::CROSS)
        );
        assert_eq!(
            key_to_pad_button(&Key::Character("c".into()), &bindings),
            Some(button::CIRCLE)
        );
        assert_eq!(
            key_to_pad_button(&Key::Character("z".into()), &bindings),
            Some(button::SQUARE)
        );
        assert_eq!(
            key_to_pad_button(&Key::Named(NamedKey::Backspace), &bindings),
            Some(button::SELECT)
        );
    }

    #[test]
    fn keyboard_mapping_honors_rebound_button() {
        let mut bindings = PortBindings::default();
        bindings.cross = InputBinding::Character('j');

        assert_eq!(
            key_to_pad_button(&Key::Character("j".into()), &bindings),
            Some(button::CROSS)
        );
        assert_eq!(
            key_to_pad_button(&Key::Character("x".into()), &bindings),
            None
        );
    }
}
