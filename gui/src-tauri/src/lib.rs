use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use tauri_plugin_dialog::DialogExt;

const MOBILE: bool = cfg!(any(target_os = "android", target_os = "ios"));
const EXE: &str = if cfg!(windows) { ".exe" } else { "" };
const VENV_BIN: &str = if cfg!(windows) { "Scripts" } else { "bin" };
const ENGINE_NAME: &str = "meetscribe";
const UV_NAME: &str = "uv";
const MOBILE_NOTICE: &str = "Auf diesem Gerät läuft keine Engine. Aufnehmen, Besprechungen und Suche brauchen die Engine auf einem PC.";
const ENV_CONFIG: &str = "MEETSCRIBE_GUI_CONFIG";
const SETTINGS_FILE: &str = "settings.json";
const PENDING_FILE: &str = "pending.json";
const LOG_FILE: &str = "engine.log";
const HARDWARE: [&str; 5] = ["auto", "npu", "gpu", "igpu", "cpu"];
const AUTO: &str = "auto";
const DOWNLOAD_LOG_FILE: &str = "download.log";
const DOWNLOAD_EVENT_NAME: &str = "download-event";
const DOWNLOAD_EXIT_NAME: &str = "download-exit";
const AUDIO_EXTENSIONS: [&str; 9] = ["wav", "flac", "mp3", "m4a", "ogg", "opus", "aac", "mp4", "webm"];
const STDERR_TAIL: usize = 12;
const POLL_MS: u64 = 50;
const DISCARD_TRIES: usize = 4;
const DISCARD_WAIT_MS: u64 = 300;
const HEARTBEAT_EVENT: &str = "alive";
const STILL_RUNNING: &str = "Die frühere Verarbeitung läuft vermutlich noch. Bitte einen Moment warten und es dann erneut versuchen.";
const EVENT_NAME: &str = "engine-event";
const EXIT_NAME: &str = "engine-exit";
const CLOSE_NAME: &str = "close-requested";

trait Detached {
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default)]
struct Settings {
    engine: String,
    repo: String,
    device: String,
    backend: String,
    model: String,
    mic: String,
    diarize: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { engine: String::new(), repo: String::new(), device: AUTO.into(), backend: AUTO.into(), model: AUTO.into(), mic: String::new(), diarize: true }
    }
}

fn safe_choice(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with(['-', '.'])
        && value.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | ':'))
}

fn check_choices(device: &str, backend: &str, model: &str) -> Result<(), String> {
    if !HARDWARE.contains(&device) {
        return Err(format!("Unbekanntes Gerät: {device}"));
    }
    for (label, value) in [("Backend", backend), ("Modell", model)] {
        if !safe_choice(value) {
            return Err(format!("Ungültige Angabe bei {label}: {value}"));
        }
    }
    Ok(())
}

impl Settings {
    fn load(path: &Path) -> Settings {
        std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        check_choices(&self.device, &self.backend, &self.model)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("Einstellungen nicht gespeichert: {e}"))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("Einstellungen nicht gespeichert: {e}"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct EngineSpec {
    program: PathBuf,
    args: Vec<String>,
    source: String,
}

impl EngineSpec {
    fn direct(program: impl Into<PathBuf>, source: &str) -> EngineSpec {
        EngineSpec { program: program.into(), args: vec![], source: source.into() }
    }
}

struct Locator<'a> {
    settings: &'a Settings,
    exists: &'a dyn Fn(&Path) -> bool,
    path_var: String,
    exe_dir: Option<PathBuf>,
    build_repo: PathBuf,
}

impl<'a> Locator<'a> {
    fn find_in_path(&self, name: &str) -> Option<PathBuf> {
        std::env::split_paths(&self.path_var).map(|dir| dir.join(name)).find(|p| (self.exists)(p))
    }

    fn repo(&self) -> PathBuf {
        let saved = self.settings.repo.trim();
        if saved.is_empty() { self.build_repo.clone() } else { PathBuf::from(saved) }
    }

    fn locate(&self) -> Result<EngineSpec, String> {
        if MOBILE {
            return Err(MOBILE_NOTICE.into());
        }
        let name = format!("{ENGINE_NAME}{EXE}");
        let saved = self.settings.engine.trim();
        if !saved.is_empty() {
            return if (self.exists)(Path::new(saved)) {
                Ok(EngineSpec::direct(saved, "gespeicherter Pfad"))
            } else {
                Err(format!("Der gespeicherte Engine-Pfad existiert nicht: {saved}. In den Einstellungen korrigieren oder leer lassen."))
            };
        }
        let repo = self.repo();
        let has_project = (self.exists)(&repo.join("pyproject.toml"));
        let saved_repo = self.settings.repo.trim();
        if !saved_repo.is_empty() && !has_project {
            return Err(format!("Der gespeicherte Projektordner enthält keine pyproject.toml: {saved_repo}. In den Einstellungen korrigieren oder leer lassen."));
        }
        let venv = repo.join(".venv").join(VENV_BIN).join(&name);
        if (self.exists)(&venv) {
            return Ok(EngineSpec::direct(venv, "Projektordner"));
        }
        if let Some(found) = self.find_in_path(&name) {
            return Ok(EngineSpec::direct(found, "PATH"));
        }
        if let Some(beside) = self.exe_dir.as_ref().map(|d| d.join(&name)).filter(|p| (self.exists)(p)) {
            return Ok(EngineSpec::direct(beside, "neben der Anwendung"));
        }
        match self.find_in_path(&format!("{UV_NAME}{EXE}")) {
            Some(uv) if has_project => Ok(EngineSpec {
                program: uv,
                args: ["run", "--project", &repo.to_string_lossy(), ENGINE_NAME].map(String::from).to_vec(),
                source: "uv".into(),
            }),
            _ => Err("Engine nicht gefunden. In den Einstellungen den Pfad zu meetscribe oder den Projektordner eintragen; im Projektordner zuerst 'uv sync' ausführen.".into()),
        }
    }
}

fn locate_engine(settings: &Settings) -> Result<EngineSpec, String> {
    let exists = |p: &Path| p.exists();
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    let build_repo = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).map(Path::to_path_buf).unwrap_or_default();
    Locator { settings, exists: &exists, path_var: std::env::var("PATH").unwrap_or_default(), exe_dir, build_repo }.locate()
}

struct Engine {
    spec: EngineSpec,
}

impl Engine {
    fn command(&self, args: &[String]) -> Command {
        let mut cmd = Command::new(&self.spec.program);
        cmd.args(&self.spec.args).args(args).detached();
        cmd
    }

    fn run(&self, args: &[&str]) -> Result<String, String> {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        let output = self
            .command(&args)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("Engine konnte nicht gestartet werden: {e}"))?;
        if output.status.success() {
            return Ok(String::from_utf8_lossy(&output.stdout).to_string());
        }
        let err = String::from_utf8_lossy(&output.stderr);
        let text = err.trim();
        Err(if text.is_empty() { format!("Engine endete mit Code {}", output.status.code().unwrap_or(-1)) } else { text.to_string() })
    }

    fn json(&self, args: &[&str]) -> Result<Value, String> {
        parse_json_output(&self.run(args)?)
    }

    fn meeting_state(&self, meeting_id: &str) -> Result<MeetingState, String> {
        let listing = self.json(&["--json", "meetings"])?;
        let found = listing.as_array().into_iter().flatten().find(|m| m["meeting_id"] == meeting_id);
        Ok(match found {
            None => MeetingState::Absent,
            Some(m) if m["complete"].as_bool() == Some(false) => MeetingState::Partial,
            Some(_) => MeetingState::Complete,
        })
    }

    fn discard(&self, meeting_id: &str) -> bool {
        for _ in 0..DISCARD_TRIES {
            if self.run(&["discard", "--", meeting_id]).is_ok() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(DISCARD_WAIT_MS));
        }
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum MeetingState {
    Absent,
    Partial,
    Complete,
}

fn parse_json_output(text: &str) -> Result<Value, String> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str(trimmed) {
        return Ok(value);
    }
    trimmed
        .lines()
        .rev()
        .filter(|l| l.starts_with('{') || l.starts_with('['))
        .find_map(|l| serde_json::from_str(l).ok())
        .ok_or_else(|| "Antwort der Engine ist kein JSON.".to_string())
}

fn parse_event(line: &str) -> Option<Value> {
    let line = line.trim();
    if !line.starts_with('{') {
        return None;
    }
    let value: Value = serde_json::from_str(line).ok()?;
    value.get("event")?.as_str()?;
    Some(value)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Stage {
    Preflight,
    Recording,
    Processing,
    Finished,
}

impl Stage {
    fn after(self, event: &Value) -> Stage {
        match event["event"].as_str() {
            Some("recording") => Stage::Recording,
            Some("transcribing") => Stage::Processing,
            Some("done") | Some("error") => Stage::Finished,
            _ => self,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct RunInfo {
    audio: Option<String>,
    input: Option<String>,
    meeting_id: Option<String>,
    warnings: Vec<String>,
}

impl RunInfo {
    fn absorb(&mut self, event: &Value) {
        let text = |key: &str| event[key].as_str().map(String::from);
        match event["event"].as_str() {
            Some("recording") => {
                self.input = text("input");
                self.audio = text("path").or(self.audio.take());
            }
            Some("transcribing") => {
                self.audio = text("audio").or(self.audio.take());
                self.meeting_id = text("meeting_id");
            }
            Some("warning") => self.warnings.extend(text("message")),
            _ => {}
        }
    }
}

fn kill_tree(child: &mut Child) {
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

type Sink = Arc<dyn Fn(&str, Value) + Send + Sync>;

struct Session {
    child: Mutex<Child>,
    stdin: Mutex<Option<ChildStdin>>,
    stage: Mutex<(Stage, Instant)>,
    info: Mutex<RunInfo>,
    finished: AtomicBool,
}

impl Session {
    fn spawn(mut cmd: Command, sink: Sink, log: Option<PathBuf>) -> Result<Arc<Session>, String> {
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Engine konnte nicht gestartet werden: {e}"))?;
        let stdout = child.stdout.take().ok_or("Kein stdout der Engine")?;
        let stderr = child.stderr.take().ok_or("Kein stderr der Engine")?;
        let stdin = child.stdin.take();
        let session = Arc::new(Session {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            stage: Mutex::new((Stage::Preflight, Instant::now())),
            info: Mutex::new(RunInfo::default()),
            finished: AtomicBool::new(false),
        });
        let tail = Arc::new(Mutex::new(Vec::<String>::new()));
        let err_tail = tail.clone();
        let mut log_file = log.as_ref().and_then(|p| std::fs::File::create(p).ok());
        let err_thread = std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Some(file) = log_file.as_mut() {
                    let _ = writeln!(file, "{line}");
                }
                let mut lines = err_tail.lock().unwrap();
                lines.push(line);
                if lines.len() > STDERR_TAIL {
                    lines.remove(0);
                }
            }
        });
        let runner = session.clone();
        std::thread::spawn(move || {
            let mut terminal = false;
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(event) = parse_event(&line).filter(|e| e["event"] != HEARTBEAT_EVENT) {
                    runner.info.lock().unwrap().absorb(&event);
                    let mut stage = runner.stage.lock().unwrap();
                    let next = stage.0.after(&event);
                    if next != stage.0 {
                        *stage = (next, Instant::now());
                    }
                    terminal |= next == Stage::Finished;
                    drop(stage);
                    sink(EVENT_NAME, event);
                }
            }
            let code = loop {
                if let Ok(Some(status)) = runner.child.lock().unwrap().try_wait() {
                    break status.code();
                }
                std::thread::sleep(Duration::from_millis(POLL_MS));
            };
            let _ = err_thread.join();
            *runner.stage.lock().unwrap() = (Stage::Finished, Instant::now());
            runner.stdin.lock().unwrap().take();
            runner.finished.store(true, Ordering::SeqCst);
            let info = runner.info.lock().unwrap().clone();
            let log = log.map(|p| p.to_string_lossy().to_string());
            sink(EXIT_NAME, json!({ "code": code, "terminal": terminal, "stderr": tail.lock().unwrap().join("\n"), "audio": info.audio, "log": log }));
        });
        Ok(session)
    }

    fn stage(&self) -> Stage {
        self.stage.lock().unwrap().0
    }

    fn info(&self) -> RunInfo {
        self.info.lock().unwrap().clone()
    }

    fn active(&self) -> bool {
        !self.finished.load(Ordering::SeqCst)
    }

    fn state(&self) -> Value {
        let (stage, since) = *self.stage.lock().unwrap();
        let info = self.info();
        json!({
            "active": self.active(),
            "stage": stage,
            "elapsed_s": since.elapsed().as_secs(),
            "audio": info.audio,
            "input": info.input,
            "warnings": info.warnings,
        })
    }

    fn stop(&self) -> Result<(), String> {
        if self.stage() != Stage::Recording {
            return Err("Es läuft keine Aufnahme.".into());
        }
        let mut guard = self.stdin.lock().unwrap();
        let stdin = guard.as_mut().ok_or("Die Engine nimmt keine Eingaben mehr an.")?;
        stdin.write_all(b"\n").and_then(|_| stdin.flush()).map_err(|e| format!("Stopp nicht übermittelt: {e}"))
    }

    fn kill(&self) -> Result<(), String> {
        if self.stage() == Stage::Recording {
            return Err("Die Aufnahme läuft noch. Erst beenden, damit das Audio gespeichert wird.".into());
        }
        kill_tree(&mut self.child.lock().unwrap());
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
struct StartOptions {
    title: Option<String>,
    audio: Option<String>,
    mic: Option<String>,
    device: String,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    model: Option<String>,
    diarize: bool,
}

impl StartOptions {
    fn clean(value: &Option<String>) -> Option<String> {
        value.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(String::from)
    }

    fn audio(&self) -> Option<String> {
        Self::clean(&self.audio)
    }

    fn choice(value: &Option<String>) -> String {
        Self::clean(value).unwrap_or_else(|| AUTO.into())
    }

    fn validate(&self) -> Result<(), String> {
        check_choices(&self.device, &Self::choice(&self.backend), &Self::choice(&self.model))?;
        match self.audio() {
            Some(audio) if !Path::new(&audio).is_file() => Err(format!("Audiodatei nicht gefunden: {audio}")),
            _ => Ok(()),
        }
    }

    fn args(&self) -> Vec<String> {
        let audio = self.audio();
        let mut args: Vec<String> = ["start", "--events", "--no-open", "--device", &self.device].map(String::from).to_vec();
        for (flag, value) in [("--backend", Self::choice(&self.backend)), ("--model", Self::choice(&self.model))] {
            if value != AUTO {
                args.extend([flag.into(), value]);
            }
        }
        if audio.is_none() {
            args.push("--stdin-stop".into());
            if let Some(mic) = Self::clean(&self.mic) {
                args.extend(["--mic".into(), mic]);
            }
        }
        if let Some(title) = Self::clean(&self.title) {
            args.extend(["--title".into(), title]);
        }
        if !self.diarize {
            args.push("--no-diarize".into());
        }
        if let Some(audio) = audio {
            args.extend(["--".into(), audio]);
        }
        args
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct Pending {
    audio: String,
    meeting_id: Option<String>,
    title: Option<String>,
    device: String,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    model: Option<String>,
    diarize: bool,
}

struct PendingStore {
    path: PathBuf,
}

impl PendingStore {
    fn read(&self) -> Option<Pending> {
        serde_json::from_str(&std::fs::read_to_string(&self.path).ok()?).ok()
    }

    fn load(&self) -> Option<Pending> {
        let pending = self.read()?;
        if Path::new(&pending.audio).is_file() {
            return Some(pending);
        }
        self.clear();
        None
    }

    fn clear(&self) {
        let _ = std::fs::remove_file(&self.path);
    }

    fn write(&self, pending: &Pending) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(pending) {
            let _ = std::fs::write(&self.path, text);
        }
    }

    fn settle(&self, engine: &Engine, meeting_id: &str) -> Result<MeetingState, String> {
        match engine.meeting_state(meeting_id)? {
            MeetingState::Partial if !engine.discard(meeting_id) => Err(STILL_RUNNING.into()),
            state => Ok(state),
        }
    }

    fn reconcile(&self, engine: &Engine, audio: &str) -> Result<(), String> {
        let Some(id) = self.load().filter(|p| p.audio == audio).and_then(|p| p.meeting_id) else { return Ok(()) };
        if self.settle(engine, &id)? == MeetingState::Complete {
            self.clear();
            return Err(format!("Diese Verarbeitung ist bereits abgeschlossen ({id}). Das Ergebnis steht unter „Besprechungen“."));
        }
        Ok(())
    }

    fn unfinished(&self, engine: Result<Engine, String>) -> Option<Pending> {
        let pending = self.load()?;
        let finished = match (pending.meeting_id.as_deref(), engine) {
            (Some(id), Ok(engine)) => engine.meeting_state(id) == Ok(MeetingState::Complete),
            _ => false,
        };
        if finished {
            self.clear();
            return None;
        }
        Some(pending)
    }

    fn forget(&self, engine: Result<Engine, String>) -> Result<(), String> {
        if let (Some(id), Ok(engine)) = (self.read().and_then(|p| p.meeting_id), engine) {
            self.settle(&engine, &id)?;
        }
        self.clear();
        Ok(())
    }

    fn observe(&self, event: &Value, options: &StartOptions) {
        let text = |key: &str| event[key].as_str().map(String::from);
        match event["event"].as_str() {
            Some("done") => self.clear(),
            Some("error") if event["meeting_id"].is_string() => self.clear(),
            Some("recording") | Some("transcribing") | Some("error") => {
                let audio = text("audio").or_else(|| text("path"));
                let Some(audio) = audio else { return };
                self.write(&Pending {
                    audio,
                    meeting_id: text("meeting_id"),
                    title: StartOptions::clean(&options.title),
                    device: options.device.clone(),
                    backend: StartOptions::clean(&options.backend),
                    model: StartOptions::clean(&options.model),
                    diarize: options.diarize,
                });
            }
            _ => {}
        }
    }
}

#[derive(Debug, Deserialize)]
struct SpeakerName {
    label: String,
    name: String,
}

struct AppState {
    settings_path: PathBuf,
    pending: PendingStore,
    log_path: PathBuf,
    download_log_path: PathBuf,
    session: Mutex<Option<Arc<Session>>>,
    download: Mutex<Option<Arc<Session>>>,
}

impl AppState {
    fn launch(&self, spawn: impl FnOnce() -> Result<Arc<Session>, String>) -> Result<(), String> {
        let mut slot = self.session.lock().unwrap();
        if slot.as_ref().is_some_and(|s| s.active()) {
            return Err("Es läuft bereits eine Sitzung.".into());
        }
        let download = self.download.lock().unwrap();
        if download.as_ref().is_some_and(|s| s.active()) {
            return Err("Während eines Modell-Downloads startet keine Aufnahme oder Verarbeitung.".into());
        }
        *slot = Some(spawn()?);
        Ok(())
    }

    fn settings(&self) -> Settings {
        Settings::load(&self.settings_path)
    }

    fn engine(&self) -> Result<Engine, String> {
        locate_engine(&self.settings()).map(|spec| Engine { spec })
    }

    fn active_session(&self) -> Option<Arc<Session>> {
        self.session.lock().unwrap().clone().filter(|s| s.active())
    }

    fn active_download(&self) -> Option<Arc<Session>> {
        self.download.lock().unwrap().clone().filter(|s| s.active())
    }

    fn start_download(&self, spawn: impl FnOnce() -> Result<Arc<Session>, String>) -> Result<(), String> {
        let session = self.session.lock().unwrap();
        if session.as_ref().is_some_and(|s| s.active()) {
            return Err("Während einer Aufnahme oder Verarbeitung wird nichts heruntergeladen.".into());
        }
        let mut slot = self.download.lock().unwrap();
        if slot.as_ref().is_some_and(|s| s.active()) {
            return Err("Es läuft bereits ein Download.".into());
        }
        *slot = Some(spawn()?);
        Ok(())
    }
}

fn download_args(backend: &str, model: &str) -> Result<Vec<String>, String> {
    for (label, value) in [("Backend", backend), ("Modell", model)] {
        if !safe_choice(value) || value == AUTO {
            return Err(format!("Ungültige Angabe bei {label}: {value}"));
        }
    }
    Ok(["setup", "--events", "--backend", backend, "--model", model].map(String::from).to_vec())
}

fn config_dir(app: &tauri::App) -> PathBuf {
    std::env::var_os(ENV_CONFIG)
        .map(PathBuf::from)
        .or_else(|| app.path().app_config_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn check_row(id: &str, label: &str, ok: bool, fix: Option<String>, detail: Option<String>) -> Value {
    json!({ "id": id, "label": label, "ok": ok, "fix": fix, "detail": detail })
}

#[tauri::command]
fn platform_info() -> Value {
    json!({ "mobile": MOBILE, "notice": MOBILE.then_some(MOBILE_NOTICE) })
}

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings()
}

#[tauri::command]
fn save_settings(state: State<'_, AppState>, settings: Settings) -> Result<(), String> {
    settings.save(&state.settings_path)
}

#[tauri::command]
fn engine_info(state: State<'_, AppState>) -> Result<EngineSpec, String> {
    locate_engine(&state.settings())
}

#[tauri::command]
async fn list_meetings(state: State<'_, AppState>) -> Result<Value, String> {
    state.engine()?.json(&["--json", "meetings"])
}

#[tauri::command]
async fn search_text(state: State<'_, AppState>, term: String) -> Result<Value, String> {
    if term.trim().is_empty() {
        return Ok(json!([]));
    }
    state.engine()?.json(&["--json", "search", "--limit", "50", "--", term.trim()])
}

#[tauri::command]
async fn get_report(state: State<'_, AppState>, meeting_id: String) -> Result<Value, String> {
    state.engine()?.json(&["--json", "report", "--", &meeting_id])
}

#[tauri::command]
async fn list_models(state: State<'_, AppState>) -> Result<Value, String> {
    state.engine()?.json(&["--json", "models"])
}

#[tauri::command]
async fn download_model(app: AppHandle, state: State<'_, AppState>, backend: String, model: String) -> Result<(), String> {
    let args = download_args(&backend, &model)?;
    let engine = state.engine()?;
    state.start_download(|| {
        let sink: Sink = Arc::new(move |name, payload| {
            let _ = app.emit(if name == EVENT_NAME { DOWNLOAD_EVENT_NAME } else { DOWNLOAD_EXIT_NAME }, payload);
        });
        Session::spawn(engine.command(&args), sink, Some(state.download_log_path.clone()))
    })
}

#[tauri::command]
fn download_state(state: State<'_, AppState>) -> Value {
    json!({ "active": state.active_download().is_some() })
}

#[tauri::command]
async fn list_devices(state: State<'_, AppState>) -> Result<Value, String> {
    state.engine()?.json(&["--json", "devices"])
}

#[tauri::command]
async fn run_checks(state: State<'_, AppState>) -> Result<Value, String> {
    let label = "Engine (meetscribe)";
    let spec = match locate_engine(&state.settings()) {
        Ok(spec) => spec,
        Err(fix) => return Ok(json!([check_row("engine", label, false, Some(fix), None)])),
    };
    let detail = format!("{} ({})", spec.program.display(), spec.source);
    match (Engine { spec }).json(&["--json", "check"]) {
        Ok(Value::Array(items)) => Ok(Value::Array([vec![check_row("engine", label, true, None, Some(detail))], items].concat())),
        Ok(_) => Ok(json!([check_row("engine", label, false, Some("Die Engine antwortet nicht wie erwartet.".into()), Some(detail))])),
        Err(err) => {
            let fix = format!("Die Engine lässt sich nicht prüfen: {}", err.chars().take(300).collect::<String>());
            Ok(json!([check_row("engine", label, false, Some(fix), Some(detail))]))
        }
    }
}

#[tauri::command]
async fn start_session(app: AppHandle, state: State<'_, AppState>, options: StartOptions) -> Result<(), String> {
    options.validate()?;
    let engine = state.engine()?;
    let args = options.args();
    state.launch(|| {
        if let Some(audio) = options.audio() {
            state.pending.reconcile(&engine, &audio)?;
        }
        let pending = PendingStore { path: state.pending.path.clone() };
        let sink: Sink = Arc::new(move |name, payload| {
            if name == EVENT_NAME {
                pending.observe(&payload, &options);
            }
            let _ = app.emit(name, payload);
        });
        Session::spawn(engine.command(&args), sink, Some(state.log_path.clone()))
    })
}

#[tauri::command]
fn stop_session(state: State<'_, AppState>) -> Result<(), String> {
    state.active_session().ok_or("Keine Sitzung aktiv.")?.stop()
}

#[tauri::command]
fn session_state(state: State<'_, AppState>) -> Value {
    match state.active_session() {
        Some(session) => session.state(),
        None => json!({ "active": false, "stage": "finished" }),
    }
}

#[tauri::command]
async fn pending_run(state: State<'_, AppState>) -> Result<Option<Pending>, String> {
    Ok(state.pending.unfinished(state.engine()))
}

#[tauri::command]
async fn discard_pending(state: State<'_, AppState>) -> Result<(), String> {
    state.pending.forget(state.engine())
}

#[tauri::command]
async fn save_speakers(state: State<'_, AppState>, meeting_id: String, report: String, names: Vec<SpeakerName>) -> Result<Value, String> {
    let engine = state.engine()?;
    if !report.to_lowercase().ends_with(".md") {
        return Err("Protokollpfad muss auf .md enden.".into());
    }
    for item in names.iter().filter(|n| !n.name.trim().is_empty()) {
        engine.run(&["speakers", "set", "--meeting", &meeting_id, "--", &item.label, item.name.trim()])?;
    }
    engine.json(&["--json", "report", "--out", &report, "--", &meeting_id])
}

#[tauri::command]
fn open_report(path: String) -> Result<(), String> {
    let target = Path::new(&path);
    let is_md = target.extension().map(|e| e.eq_ignore_ascii_case("md")).unwrap_or(false);
    if !is_md || !target.is_file() {
        return Err(format!("Protokoll nicht gefunden: {path}"));
    }
    if MOBILE {
        return Err("Auf diesem Gerät lässt sich das Protokoll nicht öffnen.".into());
    }
    let opener = if cfg!(windows) { "explorer.exe" } else if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    Command::new(opener).arg(target).detached().spawn().map(|_| ()).map_err(|e| format!("Protokoll nicht geöffnet: {e}"))
}

#[tauri::command]
async fn pick_audio(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .add_filter("Audio und Video", &AUDIO_EXTENSIONS)
        .blocking_pick_file()
        .and_then(|f| f.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
async fn exit_app(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    if let Some(session) = state.active_session() {
        session.kill()?;
        if let (Some(id), Ok(engine)) = (session.info().meeting_id, state.engine()) {
            engine.discard(&id);
        }
    }
    if let Some(download) = state.active_download() {
        let _ = download.kill();
    }
    app.exit(0);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let dir = config_dir(app);
            app.manage(AppState {
                settings_path: dir.join(SETTINGS_FILE),
                pending: PendingStore { path: dir.join(PENDING_FILE) },
                log_path: dir.join(LOG_FILE),
                download_log_path: dir.join(DOWNLOAD_LOG_FILE),
                session: Mutex::new(None),
                download: Mutex::new(None),
            });
            let _ = std::fs::create_dir_all(&dir);
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let state = window.app_handle().state::<AppState>();
                if let Some(session) = state.active_session() {
                    api.prevent_close();
                    let _ = window.emit(CLOSE_NAME, json!({ "stage": session.stage(), "audio": session.info().audio }));
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            platform_info,
            get_settings,
            save_settings,
            engine_info,
            list_meetings,
            search_text,
            get_report,
            list_devices,
            list_models,
            download_model,
            download_state,
            run_checks,
            start_session,
            stop_session,
            session_state,
            pending_run,
            discard_pending,
            save_speakers,
            open_report,
            pick_audio,
            exit_app
        ])
        .run(tauri::generate_context!())
        .expect("Meetscribe konnte nicht gestartet werden");
}

#[cfg(test)]
mod tests;
