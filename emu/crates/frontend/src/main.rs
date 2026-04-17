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
mod gfx;
mod icons;
mod theme;
mod ui;

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::app::AppState;
use crate::gfx::Graphics;

const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 800;

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = Shell::default();
    event_loop.run_app(&mut app).expect("event loop");
}

#[derive(Default)]
struct Shell {
    graphics: Option<Graphics>,
    state: AppState,
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
            WindowEvent::RedrawRequested => {
                let state = &mut self.state;
                gfx.render(|ctx| app::build_ui(ctx, state));
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
