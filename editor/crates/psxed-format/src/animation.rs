//! On-disk layout for cooked rigid-skeletal animations (`.psxanim`).
//!
//! The animation format stores already-sampled fixed-point joint pose
//! matrices. Runtime code can index a frame, fetch the joint record for
//! each model part, and submit transformed triangles without evaluating
//! glTF channels, quaternions, or keyframe interpolation on the PS1.
//!
//! # File layout
//!
//! ```text
//!   AssetHeader (12 bytes)
//!     magic       = b"PSXA"
//!     version     = VERSION
//!     flags       = reserved
//!     payload_len = everything after this header
//!
//!   AnimationHeader (8 bytes)
//!     joint_count     u16
//!     frame_count     u16
//!     sample_rate_hz  u16
//!     _reserved       u16
//!
//!   Pose table: frame_count × joint_count × 30 bytes
//!     matrix:      i16[9] Q3.12, column-major 3×3
//!     translation: i32[3] model-local units
//! ```
//!
//! The pose matrix maps model-space vertices into the sampled animated
//! pose. Translation uses the same model-local unit scale as the
//! matching `.psxmdl` vertices, so local precision can be much denser
//! than world/grid precision.

/// ASCII magic identifying the `.psxanim` animation format.
pub const MAGIC: [u8; 4] = *b"PSXA";

/// Current animation format revision.
pub const VERSION: u16 = 1;

/// Byte layout of the animation payload header.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct AnimationHeader {
    /// Number of joint pose records per frame.
    pub joint_count: u16,
    /// Number of sampled frames.
    pub frame_count: u16,
    /// Integer sample rate in Hz.
    pub sample_rate_hz: u16,
    /// Reserved. Writers set to zero; readers ignore.
    pub _reserved: u16,
}

impl AnimationHeader {
    /// Size of the animation header in bytes (always 8).
    pub const SIZE: usize = 8;

    /// Build an animation header.
    pub const fn new(joint_count: u16, frame_count: u16, sample_rate_hz: u16) -> Self {
        Self {
            joint_count,
            frame_count,
            sample_rate_hz,
            _reserved: 0,
        }
    }
}

/// Size of one joint pose record in bytes.
pub const POSE_RECORD_SIZE: usize = 30;
