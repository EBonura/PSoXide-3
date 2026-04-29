//! Fixed-size particle pool. Covers the "explosion burst" case
//! 99% of arcade games want:
//!
//! - Spawn N coloured particles at a point, each with a random
//!   velocity spread.
//! - Each frame: integrate position + velocity, apply gravity,
//!   decrement TTL.
//! - Render as `RectFlat`s into an ordering table, with colour
//!   and size tapering as TTL approaches zero.
//!
//! Const-generic pool size — each consumer picks what fits its
//! scene budget:
//!
//! ```ignore
//! static mut PARTICLES: ParticlePool<48> = ParticlePool::new();
//! ```
//!
//! Values beyond what you pass in on spawn are 0-initialised;
//! `ttl == 0` marks a slot as empty. No allocation, ever.

use psx_gpu::ot::OrderingTable;
use psx_gpu::prim::RectFlat;

use crate::rng::LcgRng;

/// One live particle.
///
/// Positions are in screen-space pixels (i16). Velocities are
/// Q4.4 sub-pixel — `vx = 32` means 2 px / frame. That keeps
/// enough resolution for slow drift + fast bursts out of the
/// same i16 channel.
#[derive(Copy, Clone, Debug)]
pub struct Particle {
    /// Top-left x position in pixels.
    pub x: i16,
    /// Top-left y position in pixels.
    pub y: i16,
    /// Velocity along x, Q4.4 pixels/frame.
    pub vx: i16,
    /// Velocity along y, Q4.4 pixels/frame. Positive = downward.
    pub vy: i16,
    /// Colour at spawn — the renderer scales this by
    /// `ttl / spawn_ttl` for a fade-out.
    pub r: u8,
    /// Green at spawn.
    pub g: u8,
    /// Blue at spawn.
    pub b: u8,
    /// Frames remaining. `0` = slot empty.
    pub ttl: u8,
    /// TTL originally passed at spawn. Kept so the fade ratio is
    /// correct when different bursts have different lifespans.
    pub spawn_ttl: u8,
}

impl Particle {
    /// A dead / empty slot. Useful for static-array init.
    pub const fn empty() -> Self {
        Self {
            x: 0,
            y: 0,
            vx: 0,
            vy: 0,
            r: 0,
            g: 0,
            b: 0,
            ttl: 0,
            spawn_ttl: 0,
        }
    }

    /// Is this particle currently alive and worth rendering?
    pub const fn alive(&self) -> bool {
        self.ttl != 0
    }
}

/// Fixed-capacity pool of `N` particles.
///
/// Uses a linear search for free slots on spawn. `N` up to ~128
/// is comfortably fast on a 33 MHz MIPS — if you're routinely
/// saturating a 256+ pool you want a proper object pool with a
/// free-list, which lives at the engine layer.
pub struct ParticlePool<const N: usize> {
    particles: [Particle; N],
}

impl<const N: usize> ParticlePool<N> {
    /// A brand-new empty pool. Const-safe for static-array init.
    pub const fn new() -> Self {
        Self {
            particles: [Particle::empty(); N],
        }
    }

    /// Spawn `count` particles at `centre` with velocities
    /// scattered within `[-velocity_range, +velocity_range]` on
    /// each axis (Q4.4). Colour and TTL are uniform across the
    /// burst.
    ///
    /// Finds free slots via linear scan; if the pool is full the
    /// burst is truncated silently. Callers that care about
    /// guaranteed-spawns should size the pool to
    /// `max_concurrent_bursts * count_per_burst`.
    ///
    /// `rng` is stepped for each spawned particle, seeding two
    /// signed values per axis.
    pub fn spawn_burst(
        &mut self,
        rng: &mut LcgRng,
        centre: (i16, i16),
        color: (u8, u8, u8),
        count: usize,
        velocity_range: i16,
        ttl: u8,
    ) {
        let mut spawned = 0;
        for slot in self.particles.iter_mut() {
            if slot.ttl != 0 {
                continue;
            }
            let vx = rng.signed(velocity_range);
            let vy = rng.signed(velocity_range);
            *slot = Particle {
                x: centre.0,
                y: centre.1,
                vx,
                vy,
                r: color.0,
                g: color.1,
                b: color.2,
                ttl,
                spawn_ttl: ttl,
            };
            spawned += 1;
            if spawned >= count {
                return;
            }
        }
    }

    /// One simulation step for every live particle:
    /// - Position += velocity (Q4.4 integrated as `>> 4`).
    /// - `vy += gravity` (Q4.4, so 1 = 0.0625 px/frame² falling).
    /// - `ttl -= 1` — particle expires when it hits zero.
    pub fn update(&mut self, gravity: i16) {
        for p in self.particles.iter_mut() {
            if p.ttl == 0 {
                continue;
            }
            p.x = p.x.wrapping_add(p.vx / 16);
            p.y = p.y.wrapping_add(p.vy / 16);
            p.vy = p.vy.saturating_add(gravity);
            p.ttl -= 1;
        }
    }

    /// Clear every slot. Use when resetting between levels so
    /// leftover particles don't float in from the previous scene.
    pub fn clear(&mut self) {
        for p in self.particles.iter_mut() {
            *p = Particle::empty();
        }
    }

    /// Render every live particle as a `RectFlat` into the caller's
    /// buffer, inserting each into OT slot `z`.
    ///
    /// `rects` must have space for all live particles; the function
    /// writes starting at index 0 and returns the number of rects
    /// used — callers typically track a running `idx` and advance
    /// it by this value.
    ///
    /// `shake` is a per-frame vertex offset applied uniformly to
    /// every particle (convenience for callers that apply the same
    /// shake to ball / paddle / etc.).
    ///
    /// The particle's original colour is scaled by `ttl /
    /// spawn_ttl` so bursts fade out gracefully; size tapers from
    /// 3 px (first half of life) to 2 px (second half).
    pub fn render_into_ot<const OT_N: usize>(
        &self,
        ot: &mut OrderingTable<OT_N>,
        rects: &mut [RectFlat],
        z: u8,
        shake: (i16, i16),
    ) -> usize {
        let mut written = 0;
        for p in self.particles.iter() {
            if p.ttl == 0 {
                continue;
            }
            if written >= rects.len() {
                break;
            }
            let denom = p.spawn_ttl.max(1) as u16;
            let scale = p.ttl as u16;
            let r = ((p.r as u16 * scale) / denom) as u8;
            let g = ((p.g as u16 * scale) / denom) as u8;
            let b = ((p.b as u16 * scale) / denom) as u8;
            let size = if (p.ttl as u16) * 2 > denom { 3 } else { 2 };
            rects[written] = RectFlat::new(p.x + shake.0, p.y + shake.1, size, size, r, g, b);
            ot.add(z as usize, &mut rects[written], RectFlat::WORDS);
            written += 1;
        }
        written
    }

    /// How many slots currently hold live particles. O(N) scan —
    /// useful for debugging budgets + tests, not hot-path code.
    pub fn live_count(&self) -> usize {
        self.particles.iter().filter(|p| p.ttl != 0).count()
    }

    /// Immutable view of the raw particle array. Debugging
    /// helper; most consumers don't need this.
    pub fn particles(&self) -> &[Particle; N] {
        &self.particles
    }
}

impl<const N: usize> Default for ParticlePool<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pool_is_empty() {
        let pool: ParticlePool<16> = ParticlePool::new();
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn spawn_burst_fills_slots() {
        let mut pool: ParticlePool<16> = ParticlePool::new();
        let mut rng = LcgRng::new(42);
        pool.spawn_burst(&mut rng, (100, 100), (255, 0, 0), 6, 32, 30);
        assert_eq!(pool.live_count(), 6);
    }

    #[test]
    fn spawn_burst_truncates_at_capacity() {
        let mut pool: ParticlePool<4> = ParticlePool::new();
        let mut rng = LcgRng::new(1);
        pool.spawn_burst(&mut rng, (0, 0), (255, 255, 255), 16, 32, 30);
        assert_eq!(pool.live_count(), 4);
    }

    #[test]
    fn update_decrements_ttl() {
        let mut pool: ParticlePool<4> = ParticlePool::new();
        let mut rng = LcgRng::new(1);
        pool.spawn_burst(&mut rng, (0, 0), (255, 255, 255), 2, 0, 3);
        assert_eq!(pool.live_count(), 2);
        pool.update(0);
        pool.update(0);
        pool.update(0);
        assert_eq!(pool.live_count(), 0, "all slots should expire after TTL");
    }

    #[test]
    fn clear_empties_pool() {
        let mut pool: ParticlePool<8> = ParticlePool::new();
        let mut rng = LcgRng::new(1);
        pool.spawn_burst(&mut rng, (0, 0), (255, 255, 255), 4, 0, 30);
        pool.clear();
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn update_integrates_velocity() {
        let mut pool: ParticlePool<2> = ParticlePool::new();
        // Manually place a particle with known vx so we know the
        // expected motion.
        pool.particles[0] = Particle {
            x: 10,
            y: 20,
            vx: 32, // Q4.4 = 2 px/frame
            vy: 16, // Q4.4 = 1 px/frame
            r: 0,
            g: 0,
            b: 0,
            ttl: 5,
            spawn_ttl: 5,
        };
        pool.update(0);
        assert_eq!(pool.particles[0].x, 12);
        assert_eq!(pool.particles[0].y, 21);
        assert_eq!(pool.particles[0].ttl, 4);
    }
}
