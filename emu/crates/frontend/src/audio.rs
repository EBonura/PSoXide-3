//! Host-side audio output — cpal output stream fed by a lock-free
//! ring buffer whose producer is the emulation thread.
//!
//! Design:
//!
//! - The SPU produces samples on the emulation thread via
//!   `Bus::run_spu_samples(n)` + `Bus::spu.drain_audio()` → `(i16, i16)`
//!   pairs at 44.1 kHz stereo. The shell pumps this every frame.
//! - A cpal output stream runs on an OS-provided audio thread and pulls
//!   samples out of a shared ring buffer on each callback. If the
//!   producer falls behind, the callback writes silence instead of
//!   blocking (audio underrun → brief pop, no deadlock).
//! - The ring buffer is a [`std::sync::Mutex<VecDeque<(i16, i16)>>`].
//!   Not lock-free but cheap at audio-block granularity (512-sample
//!   blocks × ~86 blocks/sec = 172 locks/sec — negligible).
//!
//! We keep the cpal host + stream + config alive inside [`AudioOut`].
//! Dropping the struct stops the stream.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Target sample rate — PSX SPU native rate. Host may negotiate up
/// (48 kHz is common); cpal handles any rate we ask for or tells us
/// the default.
const TARGET_SAMPLE_RATE: u32 = 44_100;

/// Shared producer/consumer queue. Producer = emulation thread;
/// consumer = cpal callback. `Arc<Mutex<...>>` is overkill for the
/// real-time audio path but cpal's callback lives longer than the
/// main thread's stack and must own its state.
///
/// `VecDeque<(i16, i16)>` stores interleaved stereo; the callback
/// pops `(l, r)` pairs and interleaves them into cpal's f32 output.
pub type SampleQueue = Arc<Mutex<std::collections::VecDeque<(i16, i16)>>>;

/// Live audio output. Owns the cpal stream (which runs on an OS
/// audio thread) and exposes the producer handle to the shell's
/// per-frame SPU drain.
pub struct AudioOut {
    /// Sample producer — the shell clones this to push drained
    /// SPU samples after each CPU frame.
    queue: SampleQueue,
    /// Kept alive so the cpal stream keeps running. Dropping it
    /// stops the audio thread.
    _stream: cpal::Stream,
    /// Sample rate actually negotiated with the host. Differs from
    /// [`TARGET_SAMPLE_RATE`] when the OS device doesn't accept
    /// 44.1 kHz (e.g. macOS CoreAudio often wants 48 kHz). The
    /// callback linearly resamples the SPU stream to avoid the
    /// zippery crackle nearest-neighbour introduces at 48 kHz.
    host_sample_rate: u32,
}

impl AudioOut {
    /// Spin up the host audio stream. Returns `None` when no output
    /// device is available (headless CI, WSL without PulseAudio).
    /// The shell treats `None` as "audio silenced" — emulation
    /// still runs, you just don't hear anything.
    pub fn open() -> Option<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        // Pick a supported stereo output config; prefer 44.1 kHz to
        // avoid resampling, fall back to whatever the device offers.
        let supported = device.supported_output_configs().ok()?;
        let mut chosen: Option<cpal::SupportedStreamConfig> = None;
        for cfg in supported {
            if cfg.channels() != 2 {
                continue;
            }
            let min = cfg.min_sample_rate().0;
            let max = cfg.max_sample_rate().0;
            if (min..=max).contains(&TARGET_SAMPLE_RATE) {
                chosen = Some(cfg.with_sample_rate(cpal::SampleRate(TARGET_SAMPLE_RATE)));
                break;
            }
            if chosen.is_none() {
                chosen = Some(cfg.with_max_sample_rate());
            }
        }
        let config = chosen?;
        let host_sample_rate = config.sample_rate().0;
        let sample_format = config.sample_format();
        let stream_config: cpal::StreamConfig = config.into();

        let queue: SampleQueue = Arc::new(Mutex::new(std::collections::VecDeque::with_capacity(
            16_384,
        )));

        // Ratio between PSX 44.1 kHz and the host's actual rate.
        // E.g. host @ 48 kHz, PSX @ 44.1 kHz → ratio = 44100/48000 ≈ 0.919,
        // so every 1000 host samples advance ~919 source samples.
        let pull_rate = TARGET_SAMPLE_RATE as f32 / host_sample_rate as f32;

        let queue_cb = Arc::clone(&queue);
        let err_fn = |e| eprintln!("[audio] stream error: {e}");

        let stream = match sample_format {
            cpal::SampleFormat::F32 => device
                .build_output_stream(
                    &stream_config,
                    {
                        let mut resampler = LinearResampler::new();
                        move |out: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                            let mut q = queue_cb.lock().unwrap();
                            for frame in out.chunks_mut(2) {
                                let (l, r) = resampler.next(&mut q, pull_rate);
                                frame[0] = (l as f32) / 32768.0;
                                if frame.len() > 1 {
                                    frame[1] = (r as f32) / 32768.0;
                                }
                            }
                        }
                    },
                    err_fn,
                    None,
                )
                .ok()?,
            cpal::SampleFormat::I16 => {
                let queue_cb = Arc::clone(&queue);
                device
                    .build_output_stream(
                        &stream_config,
                        {
                            let mut resampler = LinearResampler::new();
                            move |out: &mut [i16], _info: &cpal::OutputCallbackInfo| {
                                let mut q = queue_cb.lock().unwrap();
                                for frame in out.chunks_mut(2) {
                                    let (l, r) = resampler.next(&mut q, pull_rate);
                                    frame[0] = l;
                                    if frame.len() > 1 {
                                        frame[1] = r;
                                    }
                                }
                            }
                        },
                        err_fn,
                        None,
                    )
                    .ok()?
            }
            // Other formats (U16, etc.) — not common on modern
            // hosts; gracefully fail and let the shell run silent.
            _ => return None,
        };
        stream.play().ok()?;

        Some(Self {
            queue,
            _stream: stream,
            host_sample_rate,
        })
    }

    /// Push drained SPU samples into the ring. The shell calls this
    /// after each `run_spu_samples` pump. Discards oldest samples
    /// when the queue grows past a ~0.5-second backlog — prevents
    /// unbounded growth when the emulator runs faster than real time
    /// (fast-forward, rewind).
    pub fn push_samples(&self, samples: &[(i16, i16)]) {
        let mut q = self.queue.lock().unwrap();
        // Cap the backlog. Anything beyond ~0.5 s is audible lag
        // and we're better off dropping it.
        let cap = (TARGET_SAMPLE_RATE as usize) / 2;
        let overflow = (q.len() + samples.len()).saturating_sub(cap);
        for _ in 0..overflow {
            q.pop_front();
        }
        q.extend(samples.iter().copied());
    }

    /// Host's negotiated sample rate. Diagnostic — shown in the
    /// HUD so users can confirm audio is actually running.
    pub fn host_sample_rate(&self) -> u32 {
        self.host_sample_rate
    }

    /// Current queue depth in stereo samples. Diagnostic — very
    /// high values mean the CPU is overrunning real-time; very low
    /// means we're starving the callback.
    pub fn queue_len(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }
}

/// Tiny stateful sample-rate converter used by the CPAL callback.
/// It keeps its interpolation phase across callbacks; resetting this
/// state per callback is audible as periodic ticks on hosts whose
/// native rate is 48 kHz.
struct LinearResampler {
    phase: f32,
    prev: (i16, i16),
    next: (i16, i16),
    primed: bool,
}

impl LinearResampler {
    fn new() -> Self {
        Self {
            phase: 0.0,
            prev: (0, 0),
            next: (0, 0),
            primed: false,
        }
    }

    fn next(
        &mut self,
        queue: &mut std::collections::VecDeque<(i16, i16)>,
        pull_rate: f32,
    ) -> (i16, i16) {
        if !self.primed {
            self.prev = queue.pop_front().unwrap_or((0, 0));
            self.next = queue.pop_front().unwrap_or(self.prev);
            self.primed = true;
        }

        let out = lerp_sample(self.prev, self.next, self.phase);
        self.phase += pull_rate;
        while self.phase >= 1.0 {
            self.prev = self.next;
            self.next = queue.pop_front().unwrap_or((0, 0));
            self.phase -= 1.0;
        }
        out
    }
}

fn lerp_sample(a: (i16, i16), b: (i16, i16), t: f32) -> (i16, i16) {
    (lerp_i16(a.0, b.0, t), lerp_i16(a.1, b.1, t))
}

fn lerp_i16(a: i16, b: i16, t: f32) -> i16 {
    let value = a as f32 + (b as f32 - a as f32) * t;
    value.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}
