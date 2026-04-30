//! Window icon loading for the desktop frontend.

use winit::window::Icon;

const APP_ICON_PNG: &[u8] = include_bytes!("../../../../assets/branding/logo-icon-player.png");

/// Decode the tracked PSoXide application icon for use by winit.
pub(crate) fn load_window_icon() -> Option<Icon> {
    let image = image::load_from_memory(APP_ICON_PNG).ok()?.into_rgba8();
    let (width, height) = image.dimensions();
    Icon::from_rgba(image.into_raw(), width, height).ok()
}

/// Set the process-level application icon where the platform exposes one.
#[cfg(target_os = "macos")]
pub(crate) fn set_application_icon() {
    use objc2::{AnyThread, MainThreadMarker};
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::NSData;

    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let data = NSData::with_bytes(APP_ICON_PNG);
    let Some(image) = NSImage::initWithData(NSImage::alloc(), &data) else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);

    // SAFETY: AppKit requires this call on the main thread and with a live
    // NSImage. The marker proves the former; `image` is retained for the call.
    unsafe {
        app.setApplicationIconImage(Some(&image));
    }
}

/// Set the process-level application icon where the platform exposes one.
#[cfg(not(target_os = "macos"))]
pub(crate) fn set_application_icon() {}
