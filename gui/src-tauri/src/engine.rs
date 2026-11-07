use crate::EngineSpec;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const PROTOCOL: u64 = 1;
pub const SERVE_ARG: &str = "serve";
const STDERR_TAIL: usize = 12;
const CRASH_WINDOW: Duration = Duration::from_secs(60);
const CRASH_LIMIT: usize = 3;
const EXIT_POLL_MS: u64 = 25;
const READ_ONLY: [&str; 9] = ["meetings", "search", "report", "stats", "devices", "models", "check", "meeting.state", "ping"];
const NOT_RUNNING: &str = "Die Engine wird beendet.";
const REPEATED_CRASH: &str = "Die Engine ist mehrfach hintereinander abgestürzt und wird nicht automatisch neu gestartet. Der nächste Befehl versucht es noch einmal.";

pub trait Detached {
    fn detached(&mut self) -> &mut Self;
}

impl Detached for Command {
    #[cfg(windows)]
    fn detached(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        self.creation_flags(0x0800_0000)
    }

    #[cfg(not(windows))]
    fn detached(&mut self) -> &mut Self {
        self
    }
}

pub fn kill_tree(child: &mut Child) {
    if cfg!(windows) {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .detached()
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[derive(Clone, Debug)]
pub struct Timeouts {
    pub startup: Duration,
    pub startup_uv: Duration,
    pub query: Duration,
    pub slow: Duration,
    pub probe: Duration,
    pub shutdown: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Timeouts {
            startup: Duration::from_secs(90),
            startup_uv: Duration::from_secs(300),
            query: Duration::from_secs(30),
            slow: Duration::from_secs(180),
            probe: Duration::from_secs(5),
            shutdown: Duration::from_secs(3),
        }
    }
}

impl Timeouts {
    fn of(&self, method: &str) -> Duration {
        match method {
            "devices" | "models" | "check" => self.slow,
            "ping" => self.probe,
            _ => self.query,
        }
    }
}

pub type Notify = Arc<dyn Fn(&str, Value) + Send + Sync>;
pub type Locate = Box<dyn Fn() -> Result<EngineSpec, String> + Send + Sync>;

enum Failure {
    Rpc(String),
    Lost(String),
    Timeout(String),
}

impl Failure {
    fn text(self) -> String {
        match self {
            Failure::Rpc(t) | Failure::Lost(t) | Failure::Timeout(t) => t,
        }
    }
}

type Reply = Result<Value, Failure>;

struct Conn {
    child: Mutex<Child>,
    stdin: Mutex<Option<ChildStdin>>,
    pending: Mutex<HashMap<u64, Sender<Reply>>>,
    tail: Mutex<VecDeque<String>>,
    errors: Mutex<Option<JoinHandle<()>>>,
    alive: AtomicBool,
    ready: AtomicBool,
    expected: AtomicBool,
    pid: u32,
    info: Mutex<Value>,
    exit_text: Mutex<Option<String>>,
}

impl Conn {
    fn request(&self, id: u64, method: &str, params: Value, timeout: Duration) -> Reply {
        let (tx, rx) = mpsc::channel();
        if !self.alive.load(Ordering::SeqCst) {
            return Err(Failure::Lost(self.lost_text(None)));
        }
        self.pending.lock().unwrap().insert(id, tx);
        if !self.alive.load(Ordering::SeqCst) {
            self.pending.lock().unwrap().remove(&id);
            return Err(Failure::Lost(self.lost_text(None)));
        }
        let line = format!("{}\n", json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        let written = match self.stdin.lock().unwrap().as_mut() {
            Some(stdin) => stdin.write_all(line.as_bytes()).and_then(|_| stdin.flush()).is_ok(),
            None => false,
        };
        if !written {
            self.pending.lock().unwrap().remove(&id);
            return Err(Failure::Lost(self.lost_text(None)));
        }
        match rx.recv_timeout(timeout) {
            Ok(reply) => reply,
            Err(RecvTimeoutError::Timeout) => {
                self.pending.lock().unwrap().remove(&id);
                Err(Failure::Timeout(format!("Die Engine antwortet nicht (Zeitüberschreitung bei „{method}“ nach {} s).", timeout.as_secs())))
            }
            Err(RecvTimeoutError::Disconnected) => Err(Failure::Lost(self.lost_text(None))),
        }
    }

    fn lost_text(&self, code: Option<i32>) -> String {
        if let Some(text) = self.exit_text.lock().unwrap().clone().filter(|_| code.is_none()) {
            return text;
        }
        let tail = self.tail.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n");
        let head = match code {
            Some(code) => format!("Die Engine wurde unerwartet beendet (Code {code})."),
            None => "Die Engine wurde unerwartet beendet.".to_string(),
        };
        if tail.is_empty() { head } else { format!("{head} Letzte Meldungen:\n{tail}") }
    }

    fn settle(&self, text: &str) {
        for (_, sender) in self.pending.lock().unwrap().drain() {
            let _ = sender.send(Err(Failure::Lost(text.to_string())));
        }
    }

    fn close_input(&self) {
        self.stdin.lock().unwrap().take();
    }

    fn kill(&self) {
        self.expected.store(true, Ordering::SeqCst);
        kill_tree(&mut self.child.lock().unwrap());
    }

    fn finished(&self, within: Duration) -> bool {
        let start = Instant::now();
        loop {
            if matches!(self.child.lock().unwrap().try_wait(), Ok(Some(_))) {
                return true;
            }
            if start.elapsed() >= within {
                return false;
            }
            std::thread::sleep(Duration::from_millis(EXIT_POLL_MS));
        }
    }
}

pub struct EngineClient {
    locate: Locate,
    notify: Notify,
    log: Option<PathBuf>,
    timeouts: Timeouts,
    slot: Mutex<Option<Arc<Conn>>>,
    next_id: AtomicU64,
    closing: AtomicBool,
    crashes: Mutex<VecDeque<Instant>>,
}

impl EngineClient {
    pub fn new(locate: Locate, notify: Notify, log: Option<PathBuf>, timeouts: Timeouts) -> Arc<EngineClient> {
        Arc::new(EngineClient {
            locate,
            notify,
            log,
            timeouts,
            slot: Mutex::new(None),
            next_id: AtomicU64::new(1),
            closing: AtomicBool::new(false),
            crashes: Mutex::new(VecDeque::new()),
        })
    }

    #[cfg(test)]
    pub fn pid(&self) -> Option<u32> {
        self.slot.lock().unwrap().as_ref().filter(|c| c.alive.load(Ordering::SeqCst)).map(|c| c.pid)
    }

    #[cfg(test)]
    pub fn info(&self) -> Option<Value> {
        self.slot.lock().unwrap().as_ref().filter(|c| c.alive.load(Ordering::SeqCst)).map(|c| c.info.lock().unwrap().clone())
    }

    fn spawn(self: &Arc<Self>, spec: &EngineSpec) -> Result<Arc<Conn>, String> {
        let mut child = Command::new(&spec.program)
            .args(&spec.args)
            .arg(SERVE_ARG)
            .detached()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Die Engine konnte nicht gestartet werden: {e}"))?;
        let stdout = child.stdout.take().ok_or("Kein stdout der Engine")?;
        let stderr = child.stderr.take().ok_or("Kein stderr der Engine")?;
        let conn = Arc::new(Conn {
            stdin: Mutex::new(child.stdin.take()),
            pid: child.id(),
            child: Mutex::new(child),
            pending: Mutex::new(HashMap::new()),
            tail: Mutex::new(VecDeque::new()),
            errors: Mutex::new(None),
            alive: AtomicBool::new(true),
            ready: AtomicBool::new(false),
            expected: AtomicBool::new(false),
            info: Mutex::new(Value::Null),
            exit_text: Mutex::new(None),
        });
        let mut log = self.log.as_ref().and_then(|p| std::fs::File::create(p).ok());
        let tail = conn.clone();
        *conn.errors.lock().unwrap() = Some(std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Some(file) = log.as_mut() {
                    let _ = writeln!(file, "{line}");
                }
                let mut lines = tail.tail.lock().unwrap();
                lines.push_back(line);
                if lines.len() > STDERR_TAIL {
                    lines.pop_front();
                }
            }
        }));
        let (client, reader) = (self.clone(), conn.clone());
        std::thread::spawn(move || client.read_loop(reader, stdout));
        Ok(conn)
    }

    fn read_loop(self: Arc<Self>, conn: Arc<Conn>, stdout: std::process::ChildStdout) {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(message) = serde_json::from_str::<Value>(line.trim()) else { continue };
            if message.get("method").is_none() {
                let Some(id) = message.get("id").and_then(Value::as_u64) else { continue };
                let Some(sender) = conn.pending.lock().unwrap().remove(&id) else { continue };
                let reply = match message.get("error") {
                    Some(error) => Err(Failure::Rpc(Self::error_text(error))),
                    None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
                };
                let _ = sender.send(reply);
            } else if let Some(method) = message["method"].as_str() {
                (self.notify)(method, message.get("params").cloned().unwrap_or(Value::Null));
            }
        }
        self.gone(&conn);
    }

    fn error_text(error: &Value) -> String {
        let message = error["message"].as_str().unwrap_or("Unbekannter Fehler der Engine.");
        match error["data"]["detail"].as_str().map(str::trim) {
            Some(detail) if !detail.is_empty() && detail != message.trim() => format!("{message}\n{detail}"),
            _ => message.to_string(),
        }
    }

    fn gone(self: &Arc<Self>, conn: &Arc<Conn>) {
        let code = loop {
            match conn.child.lock().unwrap().try_wait() {
                Ok(None) => {}
                Ok(Some(status)) => break status.code(),
                Err(_) => break None,
            }
            std::thread::sleep(Duration::from_millis(EXIT_POLL_MS));
        };
        if let Some(thread) = conn.errors.lock().unwrap().take() {
            let _ = thread.join();
        }
        let text = conn.lost_text(code);
        *conn.exit_text.lock().unwrap() = Some(text.clone());
        conn.alive.store(false, Ordering::SeqCst);
        conn.settle(&text);
        if conn.expected.load(Ordering::SeqCst) || !conn.ready.load(Ordering::SeqCst) || self.closing.load(Ordering::SeqCst) {
            return;
        }
        let tail = conn.tail.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n");
        (self.notify)("engine.crashed", json!({ "code": code, "stderr": tail, "pid": conn.pid }));
        let repeated = {
            let mut crashes = self.crashes.lock().unwrap();
            crashes.push_back(Instant::now());
            crashes.retain(|t| t.elapsed() < CRASH_WINDOW);
            crashes.len() >= CRASH_LIMIT
        };
        if repeated {
            return (self.notify)("engine.restarted", json!({ "ok": false, "error": REPEATED_CRASH }));
        }
        let client = self.clone();
        std::thread::spawn(move || {
            let outcome = client.ensure();
            let payload = match outcome {
                Ok(_) => json!({ "ok": true }),
                Err(error) => json!({ "ok": false, "error": error }),
            };
            (client.notify)("engine.restarted", payload);
        });
    }

    fn handshake(&self, conn: &Arc<Conn>, spec: &EngineSpec) -> Result<(), String> {
        let timeout = if spec.source == "uv" { self.timeouts.startup_uv } else { self.timeouts.startup };
        let params = json!({ "protocol": PROTOCOL, "client": { "name": "meetscribe-gui", "version": env!("CARGO_PKG_VERSION") } });
        match conn.request(self.next_id.fetch_add(1, Ordering::SeqCst), "initialize", params, timeout) {
            Ok(info) => {
                *conn.info.lock().unwrap() = info;
                conn.ready.store(true, Ordering::SeqCst);
                Ok(())
            }
            Err(Failure::Rpc(text)) => Err(text),
            Err(Failure::Lost(text)) => Err(format!("Die Engine ließ sich nicht starten. {text}")),
            Err(Failure::Timeout(_)) => {
                let tail = conn.tail.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n");
                Err(format!("Die Engine hat sich nicht gemeldet (Zeitüberschreitung nach {} s).{}", timeout.as_secs(), if tail.is_empty() { String::new() } else { format!(" Letzte Meldungen:\n{tail}") }))
            }
        }
    }

    pub fn start(self: &Arc<Self>) -> Result<(), String> {
        self.ensure().map(|_| ())
    }

    fn ensure(self: &Arc<Self>) -> Result<Arc<Conn>, String> {
        let mut slot = self.slot.lock().unwrap();
        if let Some(conn) = slot.as_ref().filter(|c| c.alive.load(Ordering::SeqCst)) {
            return Ok(conn.clone());
        }
        if self.closing.load(Ordering::SeqCst) {
            return Err(NOT_RUNNING.into());
        }
        let spec = (self.locate)()?;
        let conn = self.spawn(&spec)?;
        if let Err(error) = self.handshake(&conn, &spec) {
            conn.kill();
            return Err(error);
        }
        *slot = Some(conn.clone());
        Ok(conn)
    }

    pub fn call(self: &Arc<Self>, method: &str, params: Value) -> Result<Value, String> {
        let timeout = self.timeouts.of(method);
        let mut retried = false;
        loop {
            let conn = self.ensure()?;
            match conn.request(self.next_id.fetch_add(1, Ordering::SeqCst), method, params.clone(), timeout) {
                Ok(value) => return Ok(value),
                Err(Failure::Lost(_)) if !retried && READ_ONLY.contains(&method) => retried = true,
                Err(Failure::Timeout(text)) => {
                    self.probe(conn);
                    return Err(text);
                }
                Err(failure) => return Err(failure.text()),
            }
        }
    }

    fn probe(self: &Arc<Self>, conn: Arc<Conn>) {
        let client = self.clone();
        std::thread::spawn(move || {
            let id = client.next_id.fetch_add(1, Ordering::SeqCst);
            if conn.request(id, "ping", json!({}), client.timeouts.probe).is_err() {
                kill_tree(&mut conn.child.lock().unwrap());
            }
        });
    }

    pub fn abort(&self) {
        if let Some(conn) = self.slot.lock().unwrap().take() {
            conn.kill();
        }
    }

    pub fn shutdown(&self) {
        self.closing.store(true, Ordering::SeqCst);
        let Some(conn) = self.slot.lock().unwrap().take().filter(|c| c.alive.load(Ordering::SeqCst)) else { return };
        conn.expected.store(true, Ordering::SeqCst);
        let _ = conn.request(self.next_id.fetch_add(1, Ordering::SeqCst), "shutdown", json!({}), self.timeouts.shutdown);
        conn.close_input();
        if !conn.finished(self.timeouts.shutdown) {
            conn.kill();
        }
    }
}
