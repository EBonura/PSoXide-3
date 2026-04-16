//! Serial I/O: controller / memory-card port (`SIO0`) and debug serial (`SIO1`).
//!
//! `SIO0` is the interface to the two controller ports and two memory-card
//! slots. `SIO1` is a more conventional async serial port used mostly by
//! debugging dev-kits; absent on retail cables.
//!
//! To be populated: data/stat/mode/ctrl/baud register layouts, controller
//! protocol framing, memory-card protocol framing, select/chip-enable
//! behaviour.
//!
//! Reference: nocash PSX-SPX "Controllers / Memory Cards" section.

/// `SIO0` register base: controllers and memory cards.
pub const SIO0_BASE: u32 = 0x1F80_1040;

/// `SIO1` register base: debug / standard serial.
pub const SIO1_BASE: u32 = 0x1F80_1050;
