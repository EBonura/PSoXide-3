//! Subprocess lifecycle and pipe protocol for a headless PCSX-Redux.
//!
//! The harness owns three pipes into Redux:
//!
//! - **stdin** — the harness writes newline-terminated commands.
//! - **stdout** — Redux's own log output *and* our protocol responses
//!   share this stream; responses are distinguished by a `#PSX3:` line
//!   prefix and pulled out by the drain thread.
//! - **stderr** — captured to a rolling buffer for diagnosis.
//!
//! The drain thread classifies every stdout line:
//! - starts with `#PSX3:` → stripped and forwarded on the response channel
//! - anything else → appended to the rolling diagnostic log
//!
//! Callers never see Redux's own chatter; they see typed protocol
//! responses through [`ReduxProcess::wait_for_response`].

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
const POLL_CHUNK: Duration = Duration::from_millis(100);

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
    /// Exit code (`None` if killed by signal — the normal case for our SIGKILL).
    pub exit_code: Option<i32>,
    /// Rolling tail of non-protocol stdout (Redux's own log).
    pub stdout: String,
    /// Rolling tail of stderr.
    pub stderr: String,
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

        let bios = config.bios.to_string_lossy().into_owned();
        let lua = config.lua_script.to_string_lossy().into_owned();

        let mut cmd = Command::new(&config.binary);
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
                        // Drain the channel of any response that landed
                        // just before the child died before declaring failure.
                        if let Ok(line) = self.responses.try_recv() {
                            return Ok(line);
                        }
                        return Err(self.early_exit_error(status.code()));
                    }
                }

                Err(RecvTimeoutError::Disconnected) => {
                    // Drain thread closed its sender — child's stdout hit
                    // EOF, meaning the process is gone or about to be.
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
    /// delay slots — see the note in `oracle.lua`.
    pub fn step(
        &mut self,
        n: u32,
        per_step_timeout: Duration,
    ) -> Result<Vec<InstructionRecord>, OracleError> {
        self.send_command(&format!("step {n}"))?;
        let mut records = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let line = self.wait_for_response(per_step_timeout)?;
            let record = InstructionRecord::from_json_line(&line).map_err(|e| {
                OracleError::Protocol {
                    expected: "InstructionRecord JSON".to_string(),
                    got: format!("{line} (parse error: {e})"),
                }
            })?;
            records.push(record);
        }
        Ok(records)
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

/// Plain stderr drain — everything goes to the diagnostic buffer.
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
