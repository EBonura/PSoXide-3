//! Subprocess lifecycle and pipe protocol for a headless PCSX-Redux.
//!
//! The harness owns three pipes into Redux:
//!
//! - **stdin** -- the harness writes newline-terminated commands.
//! - **stdout** -- Redux's own log output *and* our protocol responses
//!   share this stream; responses are distinguished by a `#PSX3:` line
//!   prefix and pulled out by the drain thread.
//! - **stderr** -- captured to a rolling buffer for diagnosis.
//!
//! The drain thread classifies every stdout line:
//! - starts with `#PSX3:` → stripped and forwarded on the response channel
//! - anything else → appended to the rolling diagnostic log
//!
//! Callers never see Redux's own chatter; they see typed protocol
//! responses through [`ReduxProcess::wait_for_response`].

use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use psx_trace::InstructionRecord;
use tempfile::TempDir;

use crate::{OracleConfig, OracleError};

/// Sentinel prefix used to distinguish harness protocol responses from
/// Redux's own stdout chatter. Must match [`oracle.lua`].
const PROTOCOL_PREFIX: &str = "#PSX3:";

/// Maximum bytes retained in the rolling stdout/stderr logs.
const BUFFER_CAPACITY: usize = 64 * 1024;

/// Granularity for mid-wait child-death polling. A response that arrives
/// within this window is still prompt; a dead child is noticed within it.
const POLL_CHUNK: Duration = Duration::from_millis(1);

/// A live (or recently terminated) Redux subprocess.
pub struct ReduxProcess {
    child: Child,
    /// Tempdir Redux runs in; auto-removed on drop.
    _run_dir: TempDir,
    /// Writing half of the child's stdin. `None` once closed.
    stdin: Option<ChildStdin>,
    /// Non-protocol stdout lines (Redux's own log output).
    stdout_log: Arc<Mutex<RollingBuffer>>,
    /// Everything Redux wrote to stderr.
    stderr_log: Arc<Mutex<RollingBuffer>>,
    /// Parsed protocol responses awaiting consumption.
    responses: Receiver<String>,
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
}

/// What a terminated process produced (for diagnostic reporting).
#[derive(Debug, Clone)]
pub struct Capture {
    /// Exit code (`None` if killed by signal -- the normal case for our SIGKILL).
    pub exit_code: Option<i32>,
    /// Rolling tail of non-protocol stdout (Redux's own log).
    pub stdout: String,
    /// Rolling tail of stderr.
    pub stderr: String,
}

/// Snapshot of Redux's currently-visible display area, returned by
/// [`ReduxProcess::display_hash`]. Pairs an FNV-1a-64 hash of the
/// pixel bytes with the dimensions + bpp so a consumer can sanity-
/// check matching resolution before asserting pixel parity.
///
/// An empty display (pre-boot, before any display command has run)
/// is represented by `width = 0, height = 0, byte_len = 0` and the
/// FNV-1a offset basis `0xCBF2_9CE4_8422_2325` as the hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayHash {
    /// FNV-1a-64 of the screenshot's raw pixel bytes.
    pub hash: u64,
    /// Display width in pixels (from `GP1 0x06 / 0x07` range).
    pub width: u32,
    /// Display height in pixels.
    pub height: u32,
    /// Redux's `bpp` enum value (opaque to us -- use `byte_len`
    /// and `width` to derive bytes-per-pixel).
    pub bpp: u32,
    /// Total byte count of the pixel buffer (width × height ×
    /// bytes-per-pixel).
    pub byte_len: usize,
}

/// Coarse CPU state checkpoint emitted by Redux during a silent run.
///
/// `state_hash` is an FNV-1a-64 digest of the software-visible GPR and
/// COP2 register files at the checkpoint. It is intentionally small so
/// long local-library sweeps can verify instruction-state lockstep
/// without shipping a full JSON trace over the pipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateCheckpoint {
    /// Redux-style user-side step count from the start of the run.
    pub step: u64,
    /// Redux CPU cycle counter at this checkpoint.
    pub tick: u64,
    /// Program counter after the checkpoint step retires.
    pub pc: u32,
    /// FNV-1a-64 over GPR + software-visible COP2 state.
    pub state_hash: u64,
}

impl ReduxProcess {
    /// Spawn Redux with the configured flags and start draining its
    /// output streams.
    pub fn launch(config: &OracleConfig) -> Result<Self, OracleError> {
        ensure_file(&config.bios)?;
        ensure_file(&config.lua_script)?;
        if let Some(disc) = &config.disc {
            ensure_file(disc)?;
        }

        let run_dir = tempfile::Builder::new()
            .prefix("psoxide3-redux-")
            .tempdir()?;
        let binary = stage_redux_binary(&config.binary, run_dir.path())?;

        let bios = config.bios.to_string_lossy().into_owned();
        let lua = config.lua_script.to_string_lossy().into_owned();

        let mut cmd = Command::new(&binary);
        cmd.args([
            "-no-ui",
            "-no-gui-log",
            "-stdout",
            "-portable",
            "-interpreter",
            // Required for Lua-registered breakpoints to fire. Redux's
            // interpreter skips `checkBP` unless this flag is set
            // (see psxinterpreter.cc: `execBlock<debug, ...>`).
            "-debugger",
            "-bios",
            &bios,
            "-dofile",
            &lua,
        ]);
        if let Some(disc) = &config.disc {
            let disc = disc.to_string_lossy().into_owned();
            cmd.args(["-iso", &disc]);
        }
        cmd.current_dir(run_dir.path())
            // SDL's dummy audio driver silences Redux completely. Real
            // audio output has no value for parity testing and actively
            // annoys anyone running the suite locally.
            .env("SDL_AUDIODRIVER", "dummy")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(OracleError::Spawn)?;

        let stdin = child.stdin.take();
        let stdout_pipe = child.stdout.take().expect("stdout was piped");
        let stderr_pipe = child.stderr.take().expect("stderr was piped");

        let stdout_log = Arc::new(Mutex::new(RollingBuffer::new(BUFFER_CAPACITY)));
        let stderr_log = Arc::new(Mutex::new(RollingBuffer::new(BUFFER_CAPACITY)));

        let (tx, rx) = mpsc::channel();

        let stdout_thread = spawn_stdout_drain(Arc::clone(&stdout_log), tx, stdout_pipe);
        let stderr_thread = spawn_stderr_drain(Arc::clone(&stderr_log), stderr_pipe);

        Ok(Self {
            child,
            _run_dir: run_dir,
            stdin,
            stdout_log,
            stderr_log,
            responses: rx,
            stdout_thread: Some(stdout_thread),
            stderr_thread: Some(stderr_thread),
        })
    }

    /// Send a newline-terminated command to Redux's stdin.
    pub fn send_command(&mut self, command: &str) -> Result<(), OracleError> {
        let stdin = self.stdin.as_mut().ok_or(OracleError::StdinClosed)?;
        writeln!(stdin, "{command}")?;
        stdin.flush()?;
        Ok(())
    }

    /// Block until the next protocol response arrives, the child dies,
    /// or the timeout elapses.
    ///
    /// Polls child liveness every [`POLL_CHUNK`] so a Redux SIGSEGV is
    /// surfaced as [`OracleError::EarlyExit`] within ~100 ms instead of
    /// blocking for the full timeout.
    pub fn wait_for_response(&mut self, timeout: Duration) -> Result<String, OracleError> {
        // Fast path: if a response is already queued, return it without
        // touching any syscall. Only when the channel is empty do we
        // fall back to the polling loop that also watches for child
        // death. This matters because the drain thread pushes responses
        // in bursts and the main-thread recv is the hot path.
        if let Ok(line) = self.responses.try_recv() {
            return Ok(line);
        }

        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(self.timeout_error(timeout));
            }
            let chunk = remaining.min(POLL_CHUNK);

            match self.responses.recv_timeout(chunk) {
                Ok(line) => return Ok(line),

                Err(RecvTimeoutError::Timeout) => {
                    if let Some(status) = self.child.try_wait()? {
                        if let Ok(line) = self.responses.try_recv() {
                            return Ok(line);
                        }
                        return Err(self.early_exit_error(status.code()));
                    }
                }

                Err(RecvTimeoutError::Disconnected) => {
                    let status = self.child.try_wait()?.and_then(|s| s.code());
                    return Err(self.early_exit_error(status));
                }
            }
        }
    }

    /// Advance Redux's CPU by exactly `n` instructions, returning one
    /// [`InstructionRecord`] per step.
    ///
    /// Each step uses a per-instruction breakpoint at `PC + 4`. This is
    /// correct for ordinary instructions but wrong for branches with
    /// delay slots -- see the note in `oracle.lua`.
    pub fn step(
        &mut self,
        n: u32,
        per_step_timeout: Duration,
    ) -> Result<Vec<InstructionRecord>, OracleError> {
        self.send_command(&format!("step {n}"))?;
        let mut records = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let line = self.wait_for_response(per_step_timeout)?;
            let record =
                InstructionRecord::from_json_line(&line).map_err(|e| OracleError::Protocol {
                    expected: "InstructionRecord JSON".to_string(),
                    got: format!("{line} (parse error: {e})"),
                })?;
            records.push(record);
        }
        Ok(records)
    }

    /// Like [`step`], but invokes `on_record` for each record as it
    /// arrives instead of accumulating a `Vec`. Use this when `n` is
    /// large enough that holding every record in memory would exhaust
    /// RAM -- e.g. a 100 M-record trace is 14 GiB as `Vec`, but only
    /// kilobytes at a time if the callback writes each record straight
    /// to disk.
    ///
    /// `progress` (if `Some`) is called roughly every time the record
    /// count crosses a multiple of the given interval. Use it to log
    /// heartbeat lines on long runs so the operator can tell the
    /// harness is still alive.
    pub fn step_streaming<F>(
        &mut self,
        n: u64,
        per_step_timeout: Duration,
        progress_interval: Option<u64>,
        mut on_record: F,
    ) -> Result<(), OracleError>
    where
        F: FnMut(&InstructionRecord) -> Result<(), OracleError>,
    {
        // Redux's `step N` command accepts a 32-bit count; for longer
        // runs we issue it in chunks so the protocol's internal loop
        // stays bounded. A 100 M trace split into 100 × 1 M chunks
        // still fits comfortably in the pipe's flow control.
        const CHUNK: u64 = 1_000_000;
        let mut emitted = 0u64;
        let mut remaining = n;
        while remaining > 0 {
            let take = remaining.min(CHUNK);
            self.send_command(&format!("step {take}"))?;
            for _ in 0..take {
                let line = self.wait_for_response(per_step_timeout)?;
                let record = InstructionRecord::from_json_line(&line).map_err(|e| {
                    OracleError::Protocol {
                        expected: "InstructionRecord JSON".to_string(),
                        got: format!("{line} (parse error: {e})"),
                    }
                })?;
                on_record(&record)?;
                emitted += 1;
                if let Some(step) = progress_interval {
                    if emitted % step == 0 {
                        eprintln!(
                            "[oracle] streamed {emitted}/{n} records ({:.1}%)",
                            100.0 * emitted as f64 / n as f64
                        );
                    }
                }
            }
            remaining -= take;
        }
        Ok(())
    }

    /// Run `n` user-side steps silently while capturing every
    /// CDROM IRQ's `(step, tick, type)` as it latches. Used by the
    /// probe that walks the first K CDROM IRQs side-by-side
    /// between ours and Redux to find which specific IRQ fires at
    /// a divergent cycle.
    ///
    /// `max_log` bounds the number of entries returned -- the Lua
    /// loop stops emitting past that but keeps running to `n`.
    pub fn log_cdrom_irqs(
        &mut self,
        n: u64,
        max_log: u32,
        timeout: Duration,
    ) -> Result<Vec<(u64, u64, u8)>, OracleError> {
        self.send_command(&format!("log_cdrom_irqs {n} {max_log}"))?;
        let mut out = Vec::with_capacity(max_log as usize);
        loop {
            let line = self.wait_for_response(timeout)?;
            if let Some(rest) = line.strip_prefix("log_cdrom_irqs ok ") {
                let _ = rest; // emitted=... sugar, ignored
                return Ok(out);
            }
            if let Some(rest) = line.strip_prefix("cdrom_irq ") {
                let mut step = 0u64;
                let mut tick = 0u64;
                let mut ty = 0u8;
                for part in rest.split_whitespace() {
                    if let Some(v) = part.strip_prefix("step=") {
                        step = v.parse().unwrap_or(0);
                    } else if let Some(v) = part.strip_prefix("tick=") {
                        tick = v.parse().unwrap_or(0);
                    } else if let Some(v) = part.strip_prefix("type=") {
                        ty = v.parse().unwrap_or(0);
                    }
                }
                out.push((step, tick, ty));
                continue;
            }
            if let Some(rest) = line.strip_prefix("err log_cdrom_irqs") {
                return Err(OracleError::Protocol {
                    expected: "log_cdrom_irqs ok".to_string(),
                    got: format!("err log_cdrom_irqs{rest}"),
                });
            }
            // Unknown line -- protocol error.
            return Err(OracleError::Protocol {
                expected: "cdrom_irq ... or log_cdrom_irqs ok".to_string(),
                got: line,
            });
        }
    }

    /// Coarse-grained divergence probe: run `n` user-side steps
    /// silently, invoking `on_checkpoint(step, tick, pc)` every
    /// `interval` steps. Returns the final Redux tick.
    ///
    /// A 100 M-step `run_checkpoint` with `interval = 10_000` emits
    /// ~10 K checkpoints (a few hundred KiB on the wire) instead of
    /// 14 GiB of full trace records, and finishes in ~30 s instead of
    /// ~40 min. Its job is to localize divergence to an `interval`-
    /// step window; a follow-up full `step()` inside that window
    /// pinpoints the exact instruction.
    pub fn run_checkpoint<F>(
        &mut self,
        n: u64,
        interval: u64,
        timeout: Duration,
        mut on_checkpoint: F,
    ) -> Result<u64, OracleError>
    where
        F: FnMut(u64, u64, u32) -> Result<(), OracleError>,
    {
        if interval == 0 || n == 0 {
            return Err(OracleError::Protocol {
                expected: "n > 0, interval > 0".to_string(),
                got: format!("n={n} interval={interval}"),
            });
        }
        self.send_command(&format!("run_checkpoint {n} {interval}"))?;
        let expected_checkpoints = n / interval;
        for _ in 0..expected_checkpoints {
            let line = self.wait_for_response(timeout)?;
            let (step, tick, pc) = parse_checkpoint(&line)?;
            on_checkpoint(step, tick, pc)?;
        }
        // Final `run_checkpoint ok` summary line with the terminal
        // state. It's always emitted even if `n % interval != 0`.
        let final_line = self.wait_for_response(timeout)?;
        let rest = final_line
            .strip_prefix("run_checkpoint ok ")
            .ok_or_else(|| OracleError::Protocol {
                expected: "run_checkpoint ok ...".to_string(),
                got: final_line.clone(),
            })?;
        let (_step, tick, _pc) = parse_kv_triple(rest, &final_line)?;
        Ok(tick)
    }

    /// Coarse-grained lockstep probe: run `n` user-side steps silently,
    /// invoking `on_checkpoint` every `interval` steps with PC, cycle
    /// count, and a compact digest of CPU-visible state.
    ///
    /// This is the preferred first pass for whole-library parity work:
    /// if every checkpoint matches, the run is very likely instruction-
    /// state aligned; if one diverges, the caller can rerun the prior
    /// window with full [`InstructionRecord`] tracing.
    pub fn run_state_checkpoint<F>(
        &mut self,
        n: u64,
        interval: u64,
        timeout: Duration,
        mut on_checkpoint: F,
    ) -> Result<StateCheckpoint, OracleError>
    where
        F: FnMut(StateCheckpoint) -> Result<(), OracleError>,
    {
        if interval == 0 || n == 0 {
            return Err(OracleError::Protocol {
                expected: "n > 0, interval > 0".to_string(),
                got: format!("n={n} interval={interval}"),
            });
        }
        self.send_command(&format!("run_state_checkpoint {n} {interval}"))?;
        let expected_checkpoints = n / interval;
        for _ in 0..expected_checkpoints {
            let line = self.wait_for_response(timeout)?;
            let checkpoint = parse_state_checkpoint(&line)?;
            on_checkpoint(checkpoint)?;
        }
        let final_line = self.wait_for_response(timeout)?;
        let rest = final_line
            .strip_prefix("run_state_checkpoint ok ")
            .ok_or_else(|| OracleError::Protocol {
                expected: "run_state_checkpoint ok ...".to_string(),
                got: final_line.clone(),
            })?;
        parse_state_kv(rest, &final_line)
    }

    /// `run_checkpoint`, but with a VBlank-timed pad schedule applied
    /// on Redux's side. `base_mask` is the always-held mask and
    /// `pulses` is a list of `(mask, start_vblank, frames)` tuples
    /// combined with it while the given VBlank window is active.
    #[allow(clippy::too_many_arguments)]
    pub fn run_checkpoint_pad<F>(
        &mut self,
        n: u64,
        interval: u64,
        port: u32,
        base_mask: u16,
        pulses: &[(u16, u64, u64)],
        timeout: Duration,
        mut on_checkpoint: F,
    ) -> Result<u64, OracleError>
    where
        F: FnMut(u64, u64, u32) -> Result<(), OracleError>,
    {
        if interval == 0 || n == 0 || port == 0 {
            return Err(OracleError::Protocol {
                expected: "n > 0, interval > 0, port > 0".to_string(),
                got: format!("n={n} interval={interval} port={port}"),
            });
        }
        let pulse_spec = if pulses.is_empty() {
            "-".to_string()
        } else {
            pulses
                .iter()
                .map(|(mask, start_vblank, frames)| format!("{mask}@{start_vblank}+{frames}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        self.send_command(&format!(
            "run_checkpoint_pad {n} {interval} {port} {base_mask} {pulse_spec}"
        ))?;
        let expected_checkpoints = n / interval;
        for _ in 0..expected_checkpoints {
            let line = self.wait_for_response(timeout)?;
            let (step, tick, pc) = parse_checkpoint(&line)?;
            on_checkpoint(step, tick, pc)?;
        }
        let final_line = self.wait_for_response(timeout)?;
        let rest = final_line
            .strip_prefix("run_checkpoint_pad ok ")
            .ok_or_else(|| OracleError::Protocol {
                expected: "run_checkpoint_pad ok ...".to_string(),
                got: final_line.clone(),
            })?;
        let (_step, tick, _pc) = parse_kv_triple(rest, &final_line)?;
        Ok(tick)
    }

    /// Silently advance Redux by `n` instructions without emitting
    /// per-step records. The caller typically follows this with a
    /// `vram_hash` / `regs` / `peek32` query to capture only the
    /// final state -- avoiding the per-step Lua-stdout overhead that
    /// makes `step()` cost ~25 s per million instructions. Good for
    /// milestone tests where intermediate trace records aren't needed.
    pub fn run(&mut self, n: u64, timeout: Duration) -> Result<u64, OracleError> {
        self.send_command(&format!("run {n}"))?;
        let line = self.wait_for_response(timeout)?;
        // Format: `run ok tick=<cycles>`
        if let Some(rest) = line.strip_prefix("run ok tick=") {
            rest.trim()
                .parse::<u64>()
                .map_err(|e| OracleError::Protocol {
                    expected: "cycle count".to_string(),
                    got: format!("{line} (parse error: {e})"),
                })
        } else {
            Err(OracleError::Protocol {
                expected: "run ok tick=<cycles>".to_string(),
                got: line,
            })
        }
    }

    /// Run `n` user-side steps silently while draining Redux's mixed
    /// audio output to `path` as raw little-endian stereo s16 PCM.
    ///
    /// `chunk_steps` controls how often the Lua side drains the SPU's
    /// internal audio stream. Smaller values reduce the risk of stream
    /// overflow during audio-heavy scenes; larger values reduce
    /// protocol overhead.
    pub fn run_audio_capture(
        &mut self,
        n: u64,
        chunk_steps: u64,
        path: &std::path::Path,
        timeout: Duration,
    ) -> Result<(u64, u64), OracleError> {
        self.send_command(&format!(
            "run_audio_capture {n} {chunk_steps} {}",
            path.display()
        ))?;
        let line = self.wait_for_response(timeout)?;
        let Some(rest) = line.strip_prefix("run_audio_capture ok ") else {
            return Err(OracleError::Protocol {
                expected: "run_audio_capture ok tick=<cycles> frames=<n>".to_string(),
                got: line,
            });
        };

        let mut tick = None;
        let mut frames = None;
        for part in rest.split_whitespace() {
            if let Some(v) = part.strip_prefix("tick=") {
                tick = v.parse::<u64>().ok();
            } else if let Some(v) = part.strip_prefix("frames=") {
                frames = v.parse::<u64>().ok();
            }
        }
        match (tick, frames) {
            (Some(tick), Some(frames)) => Ok((tick, frames)),
            _ => Err(OracleError::Protocol {
                expected: "tick=<cycles> frames=<n>".to_string(),
                got: line,
            }),
        }
    }

    /// Save Redux's current screenshot to `path` as raw
    /// little-endian 15bpp pixel bytes, with a `<path>.txt` sidecar
    /// describing the dimensions. Used to diff byte-by-byte against
    /// our own display buffer when a hash mismatch needs explaining
    /// (e.g. edge-cropping, V-range differences).
    pub fn screenshot_save(
        &mut self,
        path: &std::path::Path,
        timeout: Duration,
    ) -> Result<(), OracleError> {
        self.send_command(&format!("screenshot_save {}", path.display()))?;
        let line = self.wait_for_response(timeout)?;
        if line.starts_with("screenshot_save ok ") {
            Ok(())
        } else {
            Err(OracleError::Protocol {
                expected: "screenshot_save ok ...".to_string(),
                got: line,
            })
        }
    }

    /// FNV-1a-64 of Redux's visible display area ("screenshot")
    /// plus its dimensions. Used by milestone tests to verify we
    /// render the same pixels as Redux at a given instruction
    /// count, not just that our emulator is self-consistent
    /// run-to-run. The hash is computed server-side (LuaJIT
    /// `uint64_t`) so only the header + 16 hex chars cross the
    /// pipe instead of the full pixel buffer.
    ///
    /// Returns an `Err` if Redux's build of the PCSX Lua API
    /// doesn't expose a screenshot accessor -- callers can fall
    /// back to a determinism-only check.
    pub fn display_hash(&mut self, timeout: Duration) -> Result<DisplayHash, OracleError> {
        self.send_command("vram_hash")?;
        let line = self.wait_for_response(timeout)?;
        // Success: `vram_hash <16-hex> w=<W> h=<H> bpp=<B> len=<N>`
        // Error:   `err vram_hash: <reason>`
        let Some(rest) = line.strip_prefix("vram_hash ") else {
            return Err(OracleError::Protocol {
                expected: "vram_hash <hex> ...".to_string(),
                got: line,
            });
        };
        let mut parts = rest.split_whitespace();
        let hex = parts.next().ok_or_else(|| OracleError::Protocol {
            expected: "hash hex".to_string(),
            got: line.clone(),
        })?;
        let hash = u64::from_str_radix(hex, 16).map_err(|e| OracleError::Protocol {
            expected: "16-hex-char hash".to_string(),
            got: format!("{line} ({e})"),
        })?;
        let mut out = DisplayHash {
            hash,
            width: 0,
            height: 0,
            bpp: 0,
            byte_len: 0,
        };
        for p in parts {
            if let Some(v) = p.strip_prefix("w=") {
                out.width = v.parse().unwrap_or(0);
            } else if let Some(v) = p.strip_prefix("h=") {
                out.height = v.parse().unwrap_or(0);
            } else if let Some(v) = p.strip_prefix("bpp=") {
                out.bpp = v.parse().unwrap_or(0);
            } else if let Some(v) = p.strip_prefix("len=") {
                out.byte_len = v.parse().unwrap_or(0);
            }
        }
        Ok(out)
    }

    /// Read one 32-bit MMIO / RAM / BIOS word through Redux's view
    /// of the system. Mirrors the oracle's `peek32` command.
    pub fn peek32(&mut self, addr: u32, timeout: Duration) -> Result<u32, OracleError> {
        self.send_command(&format!("peek32 {addr}"))?;
        let line = self.wait_for_response(timeout)?;
        let Some(rest) = line.strip_prefix("peek32 ") else {
            return Err(OracleError::Protocol {
                expected: "peek32 <value>".to_string(),
                got: line,
            });
        };
        rest.trim()
            .parse::<i64>()
            .map(|v| v as u32)
            .map_err(|e| OracleError::Protocol {
                expected: "peek32 integer".to_string(),
                got: format!("{rest} ({e})"),
            })
    }

    /// Convenience: consume Lua's initial `ready` announcement and
    /// round-trip a `handshake` command.
    ///
    /// On success the oracle knows Redux is up, Lua is running the
    /// read-loop, and the pipe protocol is healthy.
    pub fn handshake(&mut self, timeout: Duration) -> Result<(), OracleError> {
        let ready = self.wait_for_response(timeout)?;
        if ready != "ready" {
            return Err(OracleError::Protocol {
                expected: "ready".to_string(),
                got: ready,
            });
        }
        self.send_command("handshake")?;
        let ack = self.wait_for_response(timeout)?;
        if ack != "handshake ok" {
            return Err(OracleError::Protocol {
                expected: "handshake ok".to_string(),
                got: ack,
            });
        }
        Ok(())
    }

    /// Kill the child (if still running) and collect diagnostic output.
    pub fn terminate(mut self) -> Capture {
        // Drop stdin first so Lua's `io.lines()` hits EOF if it's still
        // reading. Redux itself ignores this, but it's good hygiene.
        self.stdin = None;

        let _ = self.child.kill();
        let exit_code = self.child.wait().ok().and_then(|s| s.code());
        self.join_drain_threads();
        Capture {
            exit_code,
            stdout: self.stdout_log.lock().unwrap().snapshot(),
            stderr: self.stderr_log.lock().unwrap().snapshot(),
        }
    }

    fn timeout_error(&self, timeout: Duration) -> OracleError {
        OracleError::Timeout {
            marker: "protocol response".to_string(),
            timeout_ms: timeout.as_millis() as u64,
            tail: self.stdout_log.lock().unwrap().snapshot(),
        }
    }

    fn early_exit_error(&self, status: Option<i32>) -> OracleError {
        OracleError::EarlyExit {
            status,
            stdout_tail: self.stdout_log.lock().unwrap().snapshot(),
            stderr_tail: self.stderr_log.lock().unwrap().snapshot(),
        }
    }

    fn join_drain_threads(&mut self) {
        if let Some(t) = self.stdout_thread.take() {
            let _ = t.join();
        }
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for ReduxProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Parse `chk step=X tick=Y pc=Z` checkpoint lines.
fn parse_checkpoint(line: &str) -> Result<(u64, u64, u32), OracleError> {
    let rest = line
        .strip_prefix("chk ")
        .ok_or_else(|| OracleError::Protocol {
            expected: "chk step=... tick=... pc=...".to_string(),
            got: line.to_string(),
        })?;
    parse_kv_triple(rest, line)
}

/// Parse `chk step=X tick=Y pc=Z state=H` checkpoint lines.
fn parse_state_checkpoint(line: &str) -> Result<StateCheckpoint, OracleError> {
    let rest = line
        .strip_prefix("chk ")
        .ok_or_else(|| OracleError::Protocol {
            expected: "chk step=... tick=... pc=... state=...".to_string(),
            got: line.to_string(),
        })?;
    parse_state_kv(rest, line)
}

/// Parse `step=X tick=Y pc=Z state=H` fragments.
fn parse_state_kv(rest: &str, full_line: &str) -> Result<StateCheckpoint, OracleError> {
    let mut step = None;
    let mut tick = None;
    let mut pc = None;
    let mut state_hash = None;
    for part in rest.split_whitespace() {
        if let Some(v) = part.strip_prefix("step=") {
            step = v.parse().ok();
        } else if let Some(v) = part.strip_prefix("tick=") {
            tick = v.parse().ok();
        } else if let Some(v) = part.strip_prefix("pc=") {
            pc = v.parse().ok();
        } else if let Some(v) = part.strip_prefix("state=") {
            state_hash = u64::from_str_radix(v, 16).ok();
        }
    }
    match (step, tick, pc, state_hash) {
        (Some(step), Some(tick), Some(pc), Some(state_hash)) => Ok(StateCheckpoint {
            step,
            tick,
            pc,
            state_hash,
        }),
        _ => Err(OracleError::Protocol {
            expected: "step=... tick=... pc=... state=<16-hex>".to_string(),
            got: full_line.to_string(),
        }),
    }
}

/// Parse `step=X tick=Y pc=Z` fragments (shared between `chk ...`
/// and the final `run_checkpoint ok ...` line).
fn parse_kv_triple(rest: &str, full_line: &str) -> Result<(u64, u64, u32), OracleError> {
    let mut step = None;
    let mut tick = None;
    let mut pc = None;
    for part in rest.split_whitespace() {
        if let Some(v) = part.strip_prefix("step=") {
            step = v.parse().ok();
        } else if let Some(v) = part.strip_prefix("tick=") {
            tick = v.parse().ok();
        } else if let Some(v) = part.strip_prefix("pc=") {
            pc = v.parse().ok();
        }
    }
    match (step, tick, pc) {
        (Some(s), Some(t), Some(p)) => Ok((s, t, p)),
        _ => Err(OracleError::Protocol {
            expected: "step=... tick=... pc=...".to_string(),
            got: full_line.to_string(),
        }),
    }
}

fn ensure_file(path: &std::path::Path) -> Result<(), OracleError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(OracleError::MissingFile {
            path: path.to_path_buf(),
            source: io::Error::new(io::ErrorKind::NotFound, "file does not exist"),
        })
    }
}

fn stage_redux_binary(
    source: &std::path::Path,
    run_dir: &std::path::Path,
) -> Result<std::path::PathBuf, OracleError> {
    let staged = run_dir.join("pcsx-redux-oracle");
    fs::copy(source, &staged)?;

    // On macOS, launching `/path/to/pcsx-redux/pcsx-redux` can make
    // the kernel/code-signing stack treat the whole checkout as a
    // malformed bundle because the executable name matches the parent
    // directory. The process then sits forever in "launched-suspended"
    // state before Lua or even `main` runs. Running a copied executable
    // with a neutral filename avoids that bundle heuristic.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(source)?.permissions();
        perms.set_mode(perms.mode() | 0o700);
        fs::set_permissions(&staged, perms)?;
    }

    Ok(staged)
}

/// Stdout drain: classifies each line by the protocol prefix.
fn spawn_stdout_drain<R: Read + Send + 'static>(
    log: Arc<Mutex<RollingBuffer>>,
    responses: Sender<String>,
    pipe: R,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let reader = BufReader::new(pipe);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(payload) = line.strip_prefix(PROTOCOL_PREFIX) {
                // If the receiver has been dropped the process is
                // shutting down; losing this line is fine.
                let _ = responses.send(payload.to_string());
            } else {
                log.lock().unwrap().push_line(&line);
            }
        }
    })
}

/// Plain stderr drain -- everything goes to the diagnostic buffer.
fn spawn_stderr_drain<R: Read + Send + 'static>(
    log: Arc<Mutex<RollingBuffer>>,
    pipe: R,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let reader = BufReader::new(pipe);
        for line in reader.lines().map_while(Result::ok) {
            log.lock().unwrap().push_line(&line);
        }
    })
}

/// Bounded-size text buffer. Oldest lines roll off when `capacity` is
/// exceeded. Line boundaries are preserved so the tail is always a valid
/// set of complete lines.
struct RollingBuffer {
    data: String,
    capacity: usize,
}

impl RollingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: String::with_capacity(capacity),
            capacity,
        }
    }

    fn push_line(&mut self, line: &str) {
        self.data.push_str(line);
        self.data.push('\n');
        while self.data.len() > self.capacity {
            match self.data.find('\n') {
                Some(idx) => {
                    self.data.drain(..=idx);
                }
                None => {
                    self.data.clear();
                    break;
                }
            }
        }
    }

    fn snapshot(&self) -> String {
        self.data.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::RollingBuffer;

    #[test]
    fn rolling_buffer_drops_oldest_lines_when_full() {
        let mut buf = RollingBuffer::new(20);
        buf.push_line("first line xx"); // 14 bytes
        buf.push_line("second line"); // 12 more, total 26 → first rolls off
        assert!(!buf.snapshot().contains("first"));
        assert!(buf.snapshot().contains("second"));
    }

    #[test]
    fn rolling_buffer_preserves_line_boundaries_after_trim() {
        let mut buf = RollingBuffer::new(8);
        buf.push_line("abcdef");
        buf.push_line("ghijkl");
        assert!(buf.snapshot().ends_with("ghijkl\n"));
        assert!(!buf.snapshot().contains("abc"));
    }
}
