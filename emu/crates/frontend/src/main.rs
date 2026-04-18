//! PSoXide-3 desktop frontend.
//!
//! Modular layout:
//! - `theme`   — fonts, colors, framed-section helpers.
//! - `icons`   — Lucide codepoint constants.
//! - `gfx`     — winit window + wgpu surface + egui-wgpu plumbing.
//! - `app`     — top-level state, UI orchestration entry point.
//! - `ui/*`    — individual panels (central, registers, vram, xmb, hud).

#![warn(missing_docs)]

mod app;
mod cli;
mod disasm;
mod gfx;
mod icons;
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
use crate::ui::{xmb::XmbInput, MenuOutcome};

use emulator_core::button;

/// Default window size when not running fullscreen. Chosen big
/// enough to show the XMB + a framebuffer comfortably on a
/// standard laptop display.
const INITIAL_WIDTH: u32 = 1600;
const INITIAL_HEIGHT: u32 = 1000;

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

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = Shell::new(config_dir, fullscreen);
    event_loop.run_app(&mut app).expect("event loop");
}

struct Shell {
    graphics: Option<Graphics>,
    state: AppState,
    pending_input: XmbInput,
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
}

impl Default for Shell {
    fn default() -> Self {
        Self::new(None, false)
    }
}

impl Shell {
    fn new(config_dir: Option<std::path::PathBuf>, fullscreen: bool) -> Self {
        Self {
            graphics: None,
            state: AppState::with_config_dir(config_dir),
            pending_input: XmbInput::default(),
            last_frame: Instant::now(),
            pad1_mask: 0,
            fullscreen,
        }
    }
}

/// Map a winit logical key to a PSX digital-pad bitmask. Returns
/// `None` for keys that aren't bound.
///
/// Bindings: arrows = D-pad, Z = Cross, X = Circle, A = Square,
/// S = Triangle, Enter = Start, Shift = Select, Q/W = L1/R1,
/// E/R = L2/R2. Lowercase vs. uppercase doesn't matter — winit
/// gives us the logical key post-modifier, we compare on the
/// character directly.
fn key_to_pad_button(key: &Key) -> Option<u16> {
    match key {
        Key::Named(NamedKey::ArrowUp) => Some(button::UP),
        Key::Named(NamedKey::ArrowDown) => Some(button::DOWN),
        Key::Named(NamedKey::ArrowLeft) => Some(button::LEFT),
        Key::Named(NamedKey::ArrowRight) => Some(button::RIGHT),
        Key::Named(NamedKey::Enter) => Some(button::START),
        Key::Named(NamedKey::Shift) => Some(button::SELECT),
        Key::Character(s) => match s.as_str() {
            "z" | "Z" => Some(button::CROSS),
            "x" | "X" => Some(button::CIRCLE),
            "a" | "A" => Some(button::SQUARE),
            "s" | "S" => Some(button::TRIANGLE),
            "q" | "Q" => Some(button::L1),
            "w" | "W" => Some(button::R1),
            "e" | "E" => Some(button::L2),
            "r" | "R" => Some(button::R2),
            _ => None,
        },
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
            .with_title("PSoXide-3")
            .with_inner_size(winit::dpi::PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT));
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
                if !repeat {
                    if let Some(mask) = key_to_pad_button(&logical_key) {
                        match state {
                            ElementState::Pressed => self.pad1_mask |= mask,
                            ElementState::Released => self.pad1_mask &= !mask,
                        }
                    }
                }
                // The XMB only reacts to presses, not releases.
                if state == ElementState::Pressed && !repeat {
                    self.pending_input = merge_key(self.pending_input, &logical_key);
                }
                gfx.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                let input = std::mem::take(&mut self.pending_input);
                if let Some(action) = self.state.xmb.update(&input) {
                    if ui::apply_menu_action(&mut self.state, action) == MenuOutcome::Quit {
                        if let Err(e) = self.state.flush_memcard_port1() {
                            eprintln!("[frontend] memcard flush on quit: {e}");
                        }
                        event_loop.exit();
                        return;
                    }
                }

                // Run loop: retire `run_steps_per_frame` instructions this
                // frame if we're in run mode. Any execution error auto-
                // pauses and surfaces via the register panel. History
                // captures only the tail via `push_history`'s ring-buffer
                // semantics, so a 100k-instruction frame doesn't allocate.
                if self.state.running {
                    // Flush the current keyboard-derived pad mask into
                    // the guest just before stepping, so the game/
                    // homebrew sees fresh input this frame.
                    if let Some(bus) = self.state.bus.as_mut() {
                        bus.set_port1_buttons(emulator_core::ButtonState::from_bits(
                            self.pad1_mask,
                        ));
                    }
                    run_frame(&mut self.state);
                }

                let state = &mut self.state;
                gfx.prepare_vram(state.bus.as_ref().map(|b| &b.gpu.vram));
                let vram_tex = gfx.vram_texture_id();
                gfx.render(|ctx| app::build_ui(ctx, state, vram_tex, dt));
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

/// Retire up to `run_steps_per_frame` instructions. Any execution
/// error auto-pauses, reopens the XMB, and surfaces the stopped
/// state via the register panel. Hitting a breakpoint does the
/// same. Split out so the borrow checker sees `state.bus`,
/// `state.cpu`, and `state.exec_history` as disjoint field borrows
/// instead of one big `&mut state`.
fn run_frame(state: &mut AppState) {
    let steps = state.run_steps_per_frame;
    let Some(bus) = state.bus.as_mut() else {
        state.running = false;
        state.xmb.sync_run_label(false);
        return;
    };

    for _ in 0..steps {
        // Breakpoint check happens BEFORE stepping so the paused PC
        // is the BP address itself — the instruction at that PC has
        // not yet executed.
        if state.breakpoints.contains(&state.cpu.pc()) {
            state.running = false;
            state.xmb.sync_run_label(false);
            state.xmb.open = true;
            break;
        }

        match state.cpu.step(bus) {
            Ok(record) => {
                app::push_history(&mut state.exec_history, record);
            }
            Err(_) => {
                state.running = false;
                state.xmb.sync_run_label(false);
                state.xmb.open = true;
                break;
            }
        }
    }
}

/// OR a keypress into the next-frame XMB input. `Escape` both toggles
/// the overlay and acts as back when navigating; the combined semantics
/// are handled inside `XmbState::update`.
fn merge_key(mut input: XmbInput, key: &Key) -> XmbInput {
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
