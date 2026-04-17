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
mod disasm;
mod gfx;
mod icons;
mod theme;
mod ui;

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::app::AppState;
use crate::gfx::Graphics;
use crate::ui::{menu::MenuInput, MenuOutcome};

const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 800;

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = Shell::default();
    event_loop.run_app(&mut app).expect("event loop");
}

struct Shell {
    graphics: Option<Graphics>,
    state: AppState,
    pending_input: MenuInput,
    last_frame: Instant,
}

impl Default for Shell {
    fn default() -> Self {
        Self {
            graphics: None,
            state: AppState::default(),
            pending_input: MenuInput::default(),
            last_frame: Instant::now(),
        }
    }
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.graphics.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("PSoXide")
            .with_inner_size(winit::dpi::PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT));
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
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gfx.resize(size);
                gfx.window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                self.pending_input = merge_key(self.pending_input, &logical_key);
                gfx.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                let input = std::mem::take(&mut self.pending_input);
                if let Some(action) = self.state.menu.update(&input) {
                    if ui::apply_menu_action(&mut self.state, action) == MenuOutcome::Quit {
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
/// error auto-pauses, reopens the Menu, and surfaces the stopped
/// state via the register panel. Hitting a breakpoint does the
/// same. Split out so the borrow checker sees `state.bus`,
/// `state.cpu`, and `state.exec_history` as disjoint field borrows
/// instead of one big `&mut state`.
fn run_frame(state: &mut AppState) {
    let steps = state.run_steps_per_frame;
    let Some(bus) = state.bus.as_mut() else {
        state.running = false;
        state.menu.sync_run_label(false);
        return;
    };

    for _ in 0..steps {
        // Breakpoint check happens BEFORE stepping so the paused PC
        // is the BP address itself — the instruction at that PC has
        // not yet executed.
        if state.breakpoints.contains(&state.cpu.pc()) {
            state.running = false;
            state.menu.sync_run_label(false);
            state.menu.open = true;
            break;
        }

        match state.cpu.step(bus) {
            Ok(record) => {
                app::push_history(&mut state.exec_history, record);
            }
            Err(_) => {
                state.running = false;
                state.menu.sync_run_label(false);
                state.menu.open = true;
                break;
            }
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
