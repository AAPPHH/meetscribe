mod engine;

use engine::{Detached, EngineClient, Notify, Timeouts};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State, WindowEvent};
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
const DOWNLOAD_EVENT_NAME: &str = "download-event";
const DOWNLOAD_EXIT_NAME: &str = "download-exit";
const AUDIO_EXTENSIONS: [&str; 9] = ["wav", "flac", "mp3", "m4a", "ogg", "opus", "aac", "mp4", "webm"];
const HEARTBEAT_EVENT: &str = "alive";
const STILL_RUNNING: &str = "Die frühere Verarbeitung läuft vermutlich noch. Bitte einen Moment warten und es dann erneut versuchen.";
const SESSION_BUSY: &str = "Es läuft bereits eine Sitzung.";
const NO_SESSION: &str = "Keine Sitzung aktiv.";
const RECORDING_KEPT: &str = "Die Aufnahme läuft noch. Erst beenden, damit das Audio gespeichert wird.";
const EVENT_NAME: &str = "engine-event";
const EXIT_NAME: &str = "engine-exit";
const CLOSE_NAME: &str = "close-requested";
const NOTICE_NAME: &str = "engine-notice";
const RESTARTED: &str = "Die Engine war abgestürzt und wurde neu gestartet.";
const RESTARTED_RUN: &str = "Die Engine war abgestürzt und wurde neu gestartet. Die laufende Verarbeitung wurde abgebrochen, das Audio bleibt erhalten.";
const NOT_RESTARTED: &str = "Die Engine ist abgestürzt und ließ sich nicht neu starten";

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

#[derive(Clone, Copy, Debug, PartialEq)]
enum MeetingState {
    Absent,
    Partial,
    Complete,
}

impl MeetingState {
    fn parse(value: &Value) -> MeetingState {
        match value["state"].as_str() {
            Some("complete") => MeetingState::Complete,
            Some("partial") => MeetingState::Partial,
            _ => MeetingState::Absent,
        }
    }
}

fn meeting_state(engine: &Arc<EngineClient>, meeting_id: &str) -> Result<MeetingState, String> {
    engine.call("meeting.state", json!({ "meeting_id": meeting_id })).map(|v| MeetingState::parse(&v))
}

fn settle_meeting(engine: &Arc<EngineClient>, meeting_id: &str) -> Result<MeetingState, String> {
    engine.call("meeting.settle", json!({ "meeting_id": meeting_id })).map(|v| MeetingState::parse(&v))
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

    fn params(&self) -> Value {
        let mut params = json!({ "device": self.device, "diarize": self.diarize });
        match (self.audio(), Self::clean(&self.mic)) {
            (Some(audio), _) => params["audio"] = json!(audio),
            (None, Some(mic)) => params["mic"] = json!(mic),
            _ => {}
        }
        for (key, value) in [("backend", Self::choice(&self.backend)), ("model", Self::choice(&self.model))] {
            if value != AUTO {
                params[key] = json!(value);
            }
        }
        if let Some(title) = Self::clean(&self.title) {
            params["title"] = json!(title);
        }
        params
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

    fn reconcile(&self, engine: &Arc<EngineClient>, audio: &str) -> Result<(), String> {
        let Some(id) = self.load().filter(|p| p.audio == audio).and_then(|p| p.meeting_id) else { return Ok(()) };
        if settle_meeting(engine, &id)? == MeetingState::Complete {
            self.clear();
            return Err(format!("Diese Verarbeitung ist bereits abgeschlossen ({id}). Das Ergebnis steht unter „Besprechungen“."));
        }
        Ok(())
    }

    fn unfinished(&self, engine: &Arc<EngineClient>) -> Option<Pending> {
        let pending = self.load()?;
        let finished = pending.meeting_id.as_deref().is_some_and(|id| meeting_state(engine, id) == Ok(MeetingState::Complete));
        if finished {
            self.clear();
            return None;
        }
        Some(pending)
    }

    fn forget(&self, engine: Option<&Arc<EngineClient>>) -> Result<(), String> {
        if let (Some(id), Some(engine)) = (self.read().and_then(|p| p.meeting_id), engine) {
            if let Err(error) = settle_meeting(engine, &id) {
                if error == STILL_RUNNING {
                    return Err(error);
                }
            }
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

struct Tracker {
    active: bool,
    stage: Stage,
    since: Instant,
    info: RunInfo,
    options: Option<StartOptions>,
}

impl Tracker {
    fn idle() -> Tracker {
        Tracker { active: false, stage: Stage::Finished, since: Instant::now(), info: RunInfo::default(), options: None }
    }

    fn begin(&mut self, options: StartOptions) -> Result<(), String> {
        if self.active {
            return Err(SESSION_BUSY.into());
        }
        *self = Tracker { active: true, stage: Stage::Preflight, options: Some(options), ..Tracker::idle() };
        Ok(())
    }

    fn absorb(&mut self, event: &Value) {
        self.info.absorb(event);
        let next = self.stage.after(event);
        if next != self.stage {
            self.stage = next;
            self.since = Instant::now();
        }
    }

    fn finish(&mut self) -> RunInfo {
        self.active = false;
        self.stage = Stage::Finished;
        self.info.clone()
    }

    fn state(&self) -> Value {
        if !self.active {
            return json!({ "active": false, "stage": "finished" });
        }
        json!({
            "active": true,
            "stage": self.stage,
            "elapsed_s": self.since.elapsed().as_secs(),
            "audio": self.info.audio,
            "input": self.info.input,
            "warnings": self.info.warnings,
        })
    }
}

type Emit<'a> = &'a dyn Fn(&str, Value);

struct Core {
    pending: PendingStore,
    log_path: PathBuf,
    run: Mutex<Tracker>,
    download: AtomicBool,
    interrupted: AtomicBool,
}

impl Core {
    fn new(dir: &Path) -> Core {
        Core { pending: PendingStore { path: dir.join(PENDING_FILE) }, log_path: dir.join(LOG_FILE), run: Mutex::new(Tracker::idle()), download: AtomicBool::new(false), interrupted: AtomicBool::new(false) }
    }

    fn payload(params: &Value) -> Value {
        let mut event = params.clone();
        if let Some(map) = event.as_object_mut() {
            map.remove("run");
        }
        event
    }

    fn exit_payload(&self, params: &Value, info: &RunInfo, terminal: bool, stderr: &str) -> Value {
        json!({ "code": params["code"], "terminal": terminal, "stderr": stderr, "audio": info.audio, "log": self.log_path.to_string_lossy() })
    }

    fn route(&self, emit: Emit, method: &str, params: Value) {
        match method {
            "run.event" => {
                let event = Self::payload(&params);
                if event["event"] == HEARTBEAT_EVENT {
                    return;
                }
                {
                    let mut run = self.run.lock().unwrap();
                    run.absorb(&event);
                    if let Some(options) = run.options.clone() {
                        self.pending.observe(&event, &options);
                    }
                }
                emit(EVENT_NAME, event);
            }
            "run.exit" => {
                let info = self.run.lock().unwrap().finish();
                emit(EXIT_NAME, self.exit_payload(&params, &info, params["terminal"].as_bool().unwrap_or(false), ""));
            }
            "download.event" => emit(DOWNLOAD_EVENT_NAME, Self::payload(&params)),
            "download.exit" => {
                self.download.store(false, Ordering::SeqCst);
                emit(DOWNLOAD_EXIT_NAME, json!({ "code": params["code"], "terminal": params["terminal"], "stderr": params["stderr"], "audio": null, "log": self.log_path.to_string_lossy() }));
            }
            "engine.crashed" => {
                let stderr = params["stderr"].as_str().unwrap_or("");
                let mut run = self.run.lock().unwrap();
                if run.active {
                    self.interrupted.store(true, Ordering::SeqCst);
                    let info = run.finish();
                    emit(EXIT_NAME, self.exit_payload(&params, &info, false, stderr));
                }
                drop(run);
                if self.download.swap(false, Ordering::SeqCst) {
                    emit(DOWNLOAD_EXIT_NAME, json!({ "code": params["code"], "terminal": false, "stderr": stderr, "audio": null, "log": self.log_path.to_string_lossy() }));
                }
            }
            "engine.restarted" => {
                let text = if params["ok"] == true {
                    if self.interrupted.swap(false, Ordering::SeqCst) { RESTARTED_RUN.to_string() } else { RESTARTED.to_string() }
                } else {
                    format!("{NOT_RESTARTED}: {}", params["error"].as_str().unwrap_or("unbekannter Grund"))
                };
                emit(NOTICE_NAME, json!({ "text": text }));
            }
            _ => {}
        }
    }
}

struct AppState {
    settings_path: PathBuf,
    core: Arc<Core>,
    engine: Arc<EngineClient>,
}

impl AppState {
    fn settings(&self) -> Settings {
        Settings::load(&self.settings_path)
    }
}

fn download_params(backend: &str, model: &str) -> Result<Value, String> {
    for (label, value) in [("Backend", backend), ("Modell", model)] {
        if !safe_choice(value) || value == AUTO {
            return Err(format!("Ungültige Angabe bei {label}: {value}"));
        }
    }
    Ok(json!({ "backend": backend, "model": model }))
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

async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(work).await.map_err(|e| e.to_string())
}

async fn ask(state: &State<'_, AppState>, method: &'static str, params: Value) -> Result<Value, String> {
    let engine = state.engine.clone();
    blocking(move || engine.call(method, params)).await?
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
    ask(&state, "meetings", json!({})).await
}

#[tauri::command]
async fn search_text(state: State<'_, AppState>, term: String) -> Result<Value, String> {
    if term.trim().is_empty() {
        return Ok(json!([]));
    }
    ask(&state, "search", json!({ "term": term.trim(), "limit": 50 })).await
}

#[tauri::command]
async fn get_report(state: State<'_, AppState>, meeting_id: String) -> Result<Value, String> {
    ask(&state, "report", json!({ "meeting_id": meeting_id })).await
}

#[tauri::command]
async fn list_models(state: State<'_, AppState>) -> Result<Value, String> {
    ask(&state, "models", json!({})).await
}

#[tauri::command]
async fn download_model(state: State<'_, AppState>, backend: String, model: String) -> Result<(), String> {
    let params = download_params(&backend, &model)?;
    if state.core.download.swap(true, Ordering::SeqCst) {
        return Err("Es läuft bereits ein Download.".into());
    }
    if let Err(error) = ask(&state, "download.start", params).await {
        state.core.download.store(false, Ordering::SeqCst);
        return Err(error);
    }
    Ok(())
}

#[tauri::command]
fn download_state(state: State<'_, AppState>) -> Value {
    json!({ "active": state.core.download.load(Ordering::SeqCst) })
}

#[tauri::command]
async fn list_devices(state: State<'_, AppState>) -> Result<Value, String> {
    ask(&state, "devices", json!({})).await
}

#[tauri::command]
async fn run_checks(state: State<'_, AppState>) -> Result<Value, String> {
    let label = "Engine (meetscribe)";
    let spec = match locate_engine(&state.settings()) {
        Ok(spec) => spec,
        Err(fix) => return Ok(json!([check_row("engine", label, false, Some(fix), None)])),
    };
    let detail = format!("{} ({})", spec.program.display(), spec.source);
    match ask(&state, "check", json!({})).await {
        Ok(Value::Array(items)) => Ok(Value::Array([vec![check_row("engine", label, true, None, Some(detail))], items].concat())),
        Ok(_) => Ok(json!([check_row("engine", label, false, Some("Die Engine antwortet nicht wie erwartet.".into()), Some(detail))])),
        Err(err) => {
            let fix = format!("Die Engine lässt sich nicht prüfen: {}", err.chars().take(300).collect::<String>());
            Ok(json!([check_row("engine", label, false, Some(fix), Some(detail))]))
        }
    }
}

fn launch(core: &Core, engine: &Arc<EngineClient>, options: StartOptions) -> Result<(), String> {
    options.validate()?;
    if let Some(audio) = options.audio() {
        core.pending.reconcile(engine, &audio)?;
    }
    core.run.lock().unwrap().begin(options.clone())?;
    engine.call("run.start", options.params()).map(|_| ()).inspect_err(|_| {
        core.run.lock().unwrap().finish();
    })
}

fn leave(core: &Core, engine: &Arc<EngineClient>) -> Result<(), String> {
    let (active, stage, meeting_id) = {
        let run = core.run.lock().unwrap();
        (run.active, run.stage, run.info.meeting_id.clone())
    };
    if active && stage == Stage::Recording {
        return Err(RECORDING_KEPT.to_string());
    }
    if active || core.download.load(Ordering::SeqCst) {
        engine.abort();
        core.run.lock().unwrap().finish();
        core.download.store(false, Ordering::SeqCst);
    }
    if let Some(id) = meeting_id.filter(|_| active) {
        let _ = settle_meeting(engine, &id);
    }
    Ok(())
}

#[tauri::command]
async fn start_session(state: State<'_, AppState>, options: StartOptions) -> Result<(), String> {
    let (core, engine) = (state.core.clone(), state.engine.clone());
    blocking(move || launch(&core, &engine, options)).await?
}

#[tauri::command]
async fn stop_session(state: State<'_, AppState>) -> Result<(), String> {
    if !state.core.run.lock().unwrap().active {
        return Err(NO_SESSION.into());
    }
    ask(&state, "run.stop", json!({})).await.map(|_| ())
}

#[tauri::command]
fn session_state(state: State<'_, AppState>) -> Value {
    state.core.run.lock().unwrap().state()
}

#[tauri::command]
async fn pending_run(state: State<'_, AppState>) -> Result<Option<Pending>, String> {
    let (core, engine) = (state.core.clone(), state.engine.clone());
    blocking(move || core.pending.unfinished(&engine)).await
}

#[tauri::command]
async fn discard_pending(state: State<'_, AppState>) -> Result<(), String> {
    let (core, engine) = (state.core.clone(), state.engine.clone());
    blocking(move || core.pending.forget(Some(&engine))).await?
}

#[tauri::command]
async fn save_speakers(state: State<'_, AppState>, meeting_id: String, report: String, names: Vec<SpeakerName>) -> Result<Value, String> {
    if !report.to_lowercase().ends_with(".md") {
        return Err("Protokollpfad muss auf .md enden.".into());
    }
    for item in names.iter().filter(|n| !n.name.trim().is_empty()) {
        ask(&state, "speakers.set", json!({ "meeting_id": meeting_id, "label": item.label, "name": item.name.trim() })).await?;
    }
    ask(&state, "report", json!({ "meeting_id": meeting_id, "out": report })).await
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
    let (core, engine) = (state.core.clone(), state.engine.clone());
    blocking(move || leave(&core, &engine)).await??;
    app.exit(0);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let dir = config_dir(app);
            let _ = std::fs::create_dir_all(&dir);
            let core = Arc::new(Core::new(&dir));
            let (handle, router) = (app.handle().clone(), core.clone());
            let notify: Notify = Arc::new(move |method, params| router.route(&|name, payload| drop(handle.emit(name, payload)), method, params));
            let settings_path = dir.join(SETTINGS_FILE);
            let located = settings_path.clone();
            let engine = EngineClient::new(Box::new(move || locate_engine(&Settings::load(&located))), notify, Some(dir.join(LOG_FILE)), Timeouts::default());
            app.manage(AppState { settings_path, core, engine: engine.clone() });
            if !MOBILE {
                std::thread::spawn(move || drop(engine.start()));
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let state = window.app_handle().state::<AppState>();
                let (active, stage, audio) = {
                    let run = state.core.run.lock().unwrap();
                    (run.active, run.stage, run.info.audio.clone())
                };
                if active {
                    api.prevent_close();
                    let _ = window.emit(CLOSE_NAME, json!({ "stage": stage, "audio": audio }));
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
        .build(tauri::generate_context!())
        .expect("Meetscribe konnte nicht gestartet werden");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            handle.state::<AppState>().engine.shutdown();
        }
    });
}

#[cfg(test)]
mod tests;
