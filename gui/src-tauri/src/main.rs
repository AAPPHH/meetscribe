#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use tauri_plugin_dialog::DialogExt;

const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const DEFAULT_ENGINE: &str = r"C:\clones\meetscribe\.venv\Scripts\meetscribe.exe";
const DEFAULT_REPO: &str = r"C:\clones\meetscribe";
const ENV_CONFIG: &str = "MEETSCRIBE_GUI_CONFIG";
const SETTINGS_FILE: &str = "settings.json";
const DEVICES: [&str; 3] = ["npu", "gpu", "cpu"];
const AUDIO_EXTENSIONS: [&str; 9] = ["wav", "flac", "mp3", "m4a", "ogg", "opus", "aac", "mp4", "webm"];
const STDERR_TAIL: usize = 12;
const POLL_MS: u64 = 50;
const EVENT_NAME: &str = "engine-event";
const EXIT_NAME: &str = "engine-exit";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default)]
struct Settings {
    engine: String,
    repo: String,
    device: String,
    mic: String,
    diarize: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { engine: String::new(), repo: String::new(), device: "npu".into(), mic: String::new(), diarize: true }
    }
}

impl Settings {
    fn load(path: &Path) -> Settings {
        std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        if !DEVICES.contains(&self.device.as_str()) {
            return Err(format!("Unbekanntes Gerät: {}", self.device));
        }
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

struct Locator<'a> {
    settings: &'a Settings,
    exists: &'a dyn Fn(&Path) -> bool,
    path_var: String,
}

impl<'a> Locator<'a> {
    fn find_in_path(&self, name: &str) -> Option<PathBuf> {
        std::env::split_paths(&self.path_var).map(|dir| dir.join(name)).find(|p| (self.exists)(p))
    }

    fn repo(&self) -> PathBuf {
        PathBuf::from(if self.settings.repo.trim().is_empty() { DEFAULT_REPO } else { self.settings.repo.trim() })
    }

    fn locate(&self) -> Result<EngineSpec, String> {
        let saved = self.settings.engine.trim();
        let candidates = [(saved, "gespeicherter Pfad"), (DEFAULT_ENGINE, "Standardpfad")];
        for (candidate, source) in candidates {
            if !candidate.is_empty() && (self.exists)(Path::new(candidate)) {
                return Ok(EngineSpec { program: PathBuf::from(candidate), args: vec![], source: source.into() });
            }
        }
        let repo = self.repo();
        match self.find_in_path("uv.exe") {
            Some(uv) if (self.exists)(&repo.join("pyproject.toml")) => Ok(EngineSpec {
                program: uv,
                args: ["run", "--project", &repo.to_string_lossy(), "meetscribe"].map(String::from).to_vec(),
                source: "uv".into(),
            }),
            _ => Err(format!(
                "Engine nicht gefunden. Pfad zu meetscribe.exe in den Einstellungen eintragen oder im Projektordner {} 'uv sync' ausführen.",
                repo.display()
            )),
        }
    }
}

fn locate_engine(settings: &Settings) -> Result<EngineSpec, String> {
    let exists = |p: &Path| p.exists();
    Locator { settings, exists: &exists, path_var: std::env::var("PATH").unwrap_or_default() }.locate()
}

struct Engine {
    spec: EngineSpec,
}

impl Engine {
    fn command(&self, args: &[String]) -> Command {
        let mut cmd = Command::new(&self.spec.program);
        cmd.args(&self.spec.args).args(args).creation_flags(CREATE_NO_WINDOW);
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

type Sink = Arc<dyn Fn(&str, Value) + Send + Sync>;

struct Session {
    child: Mutex<Child>,
    stdin: Mutex<Option<ChildStdin>>,
    stage: Mutex<Stage>,
    finished: AtomicBool,
}

impl Session {
    fn spawn(mut cmd: Command, sink: Sink) -> Result<Arc<Session>, String> {
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
            stage: Mutex::new(Stage::Preflight),
            finished: AtomicBool::new(false),
        });
        let tail = Arc::new(Mutex::new(Vec::<String>::new()));
        let err_tail = tail.clone();
        let err_thread = std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
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
                if let Some(event) = parse_event(&line) {
                    let mut stage = runner.stage.lock().unwrap();
                    *stage = stage.after(&event);
                    terminal |= *stage == Stage::Finished;
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
            *runner.stage.lock().unwrap() = Stage::Finished;
            runner.stdin.lock().unwrap().take();
            runner.finished.store(true, Ordering::SeqCst);
            sink(EXIT_NAME, json!({ "code": code, "terminal": terminal, "stderr": tail.lock().unwrap().join("\n") }));
        });
        Ok(session)
    }

    fn stage(&self) -> Stage {
        *self.stage.lock().unwrap()
    }

    fn active(&self) -> bool {
        !self.finished.load(Ordering::SeqCst)
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
        let _ = self.child.lock().unwrap().kill();
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct StartOptions {
    title: Option<String>,
    audio: Option<String>,
    mic: Option<String>,
    device: String,
    diarize: bool,
}

fn start_args(o: &StartOptions) -> Result<Vec<String>, String> {
    if !DEVICES.contains(&o.device.as_str()) {
        return Err(format!("Unbekanntes Gerät: {}", o.device));
    }
    let clean = |v: &Option<String>| v.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(String::from);
    let audio = clean(&o.audio);
    let mut args: Vec<String> = vec!["start".into()];
    args.extend(audio.clone());
    args.extend(["--events", "--no-open", "--device", &o.device].map(String::from));
    if audio.is_none() {
        args.push("--stdin-stop".into());
        if let Some(mic) = clean(&o.mic) {
            args.extend(["--mic".into(), mic]);
        }
    }
    if let Some(title) = clean(&o.title) {
        args.extend(["--title".into(), title]);
    }
    if !o.diarize {
        args.push("--no-diarize".into());
    }
    Ok(args)
}

#[derive(Debug, Deserialize)]
struct SpeakerName {
    label: String,
    name: String,
}

struct AppState {
    settings_path: PathBuf,
    session: Mutex<Option<Arc<Session>>>,
}

impl AppState {
    fn settings(&self) -> Settings {
        Settings::load(&self.settings_path)
    }

    fn engine(&self) -> Result<Engine, String> {
        locate_engine(&self.settings()).map(|spec| Engine { spec })
    }

    fn active_session(&self) -> Option<Arc<Session>> {
        self.session.lock().unwrap().clone().filter(|s| s.active())
    }
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
    let mut rows = vec![check_row("engine", label, true, None, Some(detail))];
    match (Engine { spec }).json(&["--json", "check"]) {
        Ok(Value::Array(items)) => rows.extend(items),
        Ok(_) => rows.push(check_row("check", "Prüfung", false, Some("Unerwartete Antwort der Engine.".into()), None)),
        Err(err) => rows.push(check_row("check", "Prüfung", false, Some(err), None)),
    }
    Ok(Value::Array(rows))
}

#[tauri::command]
async fn start_session(app: AppHandle, state: State<'_, AppState>, options: StartOptions) -> Result<(), String> {
    if state.active_session().is_some() {
        return Err("Es läuft bereits eine Sitzung.".into());
    }
    let engine = state.engine()?;
    let args = start_args(&options)?;
    let handle = app.clone();
    let sink: Sink = Arc::new(move |name, payload| {
        let _ = handle.emit(name, payload);
    });
    let session = Session::spawn(engine.command(&args), sink)?;
    *state.session.lock().unwrap() = Some(session);
    Ok(())
}

#[tauri::command]
fn stop_session(state: State<'_, AppState>) -> Result<(), String> {
    state.active_session().ok_or("Keine Sitzung aktiv.")?.stop()
}

#[tauri::command]
fn session_state(state: State<'_, AppState>) -> Value {
    match state.active_session() {
        Some(session) => json!({ "active": true, "stage": session.stage() }),
        None => json!({ "active": false, "stage": "finished" }),
    }
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
    Command::new("explorer.exe")
        .arg(target)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Protokoll nicht geöffnet: {e}"))
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
fn exit_app(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    if let Some(session) = state.active_session() {
        session.kill()?;
    }
    app.exit(0);
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let dir = config_dir(app);
            app.manage(AppState { settings_path: dir.join(SETTINGS_FILE), session: Mutex::new(None) });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let state = window.app_handle().state::<AppState>();
                if let Some(session) = state.active_session() {
                    api.prevent_close();
                    let _ = window.emit("close-requested", json!({ "stage": session.stage() }));
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_settings,
            save_settings,
            engine_info,
            list_meetings,
            search_text,
            get_report,
            list_devices,
            run_checks,
            start_session,
            stop_session,
            session_state,
            save_speakers,
            open_report,
            pick_audio,
            exit_app
        ])
        .run(tauri::generate_context!())
        .expect("Meetscribe konnte nicht gestartet werden");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn locator<'a>(settings: &'a Settings, present: &'a dyn Fn(&Path) -> bool, path_var: &str) -> Locator<'a> {
        Locator { settings, exists: present, path_var: path_var.to_string() }
    }

    fn present_from(paths: &'static [&'static str]) -> impl Fn(&Path) -> bool {
        let set: HashSet<PathBuf> = paths.iter().map(PathBuf::from).collect();
        move |p| set.contains(p)
    }

    #[test]
    fn event_lines_skip_noise_and_keep_events() {
        assert_eq!(parse_event(r#"  {"event":"recording","input":"DJI"}  "#).unwrap()["input"], "DJI");
        assert!(parse_event("Warnung der Bibliothek").is_none());
        assert!(parse_event(r#"{"kein":"event"}"#).is_none());
        assert!(parse_event(r#"{"event":"kaputt"#).is_none());
        assert!(parse_event(r#"{"event":5}"#).is_none());
        assert!(parse_event("").is_none());
    }

    #[test]
    fn stages_follow_events() {
        let ev = |name: &str| json!({ "event": name });
        let mut stage = Stage::Preflight;
        for (name, expected) in [
            ("warning", Stage::Preflight),
            ("recording", Stage::Recording),
            ("warning", Stage::Recording),
            ("transcribing", Stage::Processing),
            ("done", Stage::Finished),
        ] {
            stage = stage.after(&ev(name));
            assert_eq!(stage, expected);
        }
        assert_eq!(Stage::Processing.after(&ev("error")), Stage::Finished);
    }

    #[test]
    fn json_output_tolerates_leading_noise() {
        assert_eq!(parse_json_output("[1,2]\n").unwrap(), json!([1, 2]));
        assert_eq!(parse_json_output("Rauschen\n{\"a\":1}\n").unwrap(), json!({ "a": 1 }));
        assert!(parse_json_output("nur Text").is_err());
    }

    #[test]
    fn saved_engine_wins_then_default_then_uv() {
        let settings = Settings { engine: r"D:\x\meetscribe.exe".into(), ..Settings::default() };
        let present = present_from(&[r"D:\x\meetscribe.exe", DEFAULT_ENGINE]);
        let spec = locator(&settings, &present, "").locate().unwrap();
        assert_eq!((spec.program, spec.source.as_str()), (PathBuf::from(r"D:\x\meetscribe.exe"), "gespeicherter Pfad"));
        let present = present_from(&[DEFAULT_ENGINE]);
        let spec = locator(&settings, &present, "").locate().unwrap();
        assert_eq!((spec.program, spec.source.as_str()), (PathBuf::from(DEFAULT_ENGINE), "Standardpfad"));
        let spec = locator(&Settings::default(), &present, "").locate().unwrap();
        assert_eq!(spec.source, "Standardpfad");
    }

    #[test]
    fn uv_fallback_uses_project_and_requires_pyproject() {
        let settings = Settings::default();
        let uv = r"C:\tools\uv.exe";
        let pyproject = format!(r"{DEFAULT_REPO}\pyproject.toml");
        let present = |p: &Path| p == Path::new(uv) || p == Path::new(&pyproject);
        let spec = locator(&settings, &present, r"C:\win;C:\tools").locate().unwrap();
        assert_eq!(spec.program, PathBuf::from(uv));
        assert_eq!(spec.args, ["run", "--project", DEFAULT_REPO, "meetscribe"]);
        let custom = Settings { repo: r"E:\repo".into(), ..Settings::default() };
        let present = |p: &Path| p == Path::new(uv) || p == Path::new(r"E:\repo\pyproject.toml");
        assert_eq!(locator(&custom, &present, r"C:\tools").locate().unwrap().args[2], r"E:\repo");
        let only_uv = |p: &Path| p == Path::new(uv);
        let err = locator(&settings, &only_uv, r"C:\tools").locate().unwrap_err();
        assert!(err.contains("Engine nicht gefunden") && err.contains("uv sync"));
        let none = |_: &Path| false;
        assert!(locator(&settings, &none, r"C:\tools").locate().is_err());
    }

    #[test]
    fn start_args_cover_recording_and_file_modes() {
        let base = StartOptions { title: Some("  Runde ".into()), audio: None, mic: Some("DJI".into()), device: "gpu".into(), diarize: false };
        assert_eq!(
            start_args(&base).unwrap(),
            ["start", "--events", "--no-open", "--device", "gpu", "--stdin-stop", "--mic", "DJI", "--title", "Runde", "--no-diarize"]
        );
        let file = StartOptions { title: None, audio: Some(r"C:\a b\x.wav".into()), mic: Some("DJI".into()), device: "npu".into(), diarize: true };
        assert_eq!(start_args(&file).unwrap(), ["start", r"C:\a b\x.wav", "--events", "--no-open", "--device", "npu"]);
        let blank = StartOptions { title: Some(" ".into()), audio: Some("".into()), mic: Some("".into()), device: "cpu".into(), diarize: true };
        assert_eq!(start_args(&blank).unwrap(), ["start", "--events", "--no-open", "--device", "cpu", "--stdin-stop"]);
        let bad = StartOptions { device: "tpu".into(), ..blank };
        assert!(start_args(&bad).is_err());
    }

    #[test]
    fn settings_roundtrip_and_defaults() {
        let dir = std::env::temp_dir().join(format!("meetscribe-gui-test-{}", std::process::id()));
        let path = dir.join(SETTINGS_FILE);
        assert_eq!(Settings::load(&path), Settings::default());
        let settings = Settings { engine: "x".into(), mic: "DJI".into(), device: "gpu".into(), diarize: false, ..Settings::default() };
        settings.save(&path).unwrap();
        assert_eq!(Settings::load(&path), settings);
        std::fs::write(&path, r#"{"device":"cpu"}"#).unwrap();
        assert_eq!(Settings::load(&path), Settings { device: "cpu".into(), ..Settings::default() });
        assert!(Settings { device: "tpu".into(), ..Settings::default() }.save(&path).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    fn fake_engine() -> PathBuf {
        std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("examples").join("fake_engine.exe")
    }

    fn wait_for(limit: usize, mut done: impl FnMut() -> bool) {
        for _ in 0..limit {
            if done() {
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    fn run_fake(mode: &str, args: &[&str], stop_after: Option<&str>) -> (Vec<(String, Value)>, Arc<Session>) {
        let mut cmd = Command::new(fake_engine());
        cmd.args(args).env("FAKE_ENGINE_MODE", mode).creation_flags(CREATE_NO_WINDOW);
        let seen = Arc::new(Mutex::new(Vec::<(String, Value)>::new()));
        let log = seen.clone();
        let session = Session::spawn(cmd, Arc::new(move |name, payload| log.lock().unwrap().push((name.to_string(), payload)))).unwrap();
        if let Some(name) = stop_after {
            wait_for(200, || seen.lock().unwrap().iter().any(|(_, v)| v["event"] == name));
            wait_for(200, || session.stage() == Stage::Recording);
            session.stop().unwrap();
        }
        wait_for(400, || !session.active());
        let events = seen.lock().unwrap().clone();
        (events, session)
    }

    fn names(events: &[(String, Value)]) -> Vec<String> {
        events.iter().map(|(n, v)| if n == EXIT_NAME { "exit".to_string() } else { v["event"].as_str().unwrap().to_string() }).collect()
    }

    #[test]
    fn session_stop_line_ends_recording_and_processing_continues() {
        let (events, session) = run_fake("", &["start", "--events", "--stdin-stop"], Some("recording"));
        assert_eq!(names(&events), ["preflight", "recording", "warning", "transcribing", "done", "exit"]);
        let exit = &events.last().unwrap().1;
        assert_eq!((exit["code"].as_i64(), exit["terminal"].as_bool()), (Some(0), Some(true)));
        assert!(!session.active() && session.stage() == Stage::Finished);
        assert_eq!(events[4].1["labels"][0]["sample"], "Guten Morgen zusammen.");
        assert!(session.stop().is_err());
    }

    #[test]
    fn session_reports_failure_events_and_exit_code() {
        let (events, _) = run_fake("preflight_fail", &["start", "--events"], None);
        assert_eq!(names(&events), ["preflight", "error", "exit"]);
        assert_eq!(events[2].1["code"], 1);
        let (events, _) = run_fake("process_fail", &["start", "x.wav", "--events"], None);
        assert_eq!(names(&events), ["preflight", "transcribing", "error", "exit"]);
        assert!(events[2].1["retry"].as_str().unwrap().contains("--events"));
        assert_eq!(events[2].1["audio"], r"C:\fake\aufnahme.flac");
    }

    #[test]
    fn kill_is_refused_while_recording_but_works_after() {
        let mut cmd = Command::new(fake_engine());
        cmd.args(["start", "--events", "--stdin-stop"]).creation_flags(CREATE_NO_WINDOW);
        let session = Session::spawn(cmd, Arc::new(|_, _| {})).unwrap();
        wait_for(200, || session.stage() == Stage::Recording);
        assert!(session.kill().is_err());
        session.stop().unwrap();
        wait_for(200, || session.stage() != Stage::Recording);
        assert!(session.kill().is_ok());
    }

    #[test]
    fn engine_json_queries_run_through_the_command_line() {
        let engine = Engine { spec: EngineSpec { program: fake_engine(), args: vec![], source: "test".into() } };
        assert_eq!(engine.json(&["--json", "devices"]).unwrap()["inputs"][0]["name"], "Fake Mikrofon");
        assert_eq!(engine.json(&["--json", "report", "--", "fake"]).unwrap()["markdown"], "# Fake\n\nProtokoll");
        assert!(engine.run(&["speakers", "set", "--meeting", "fake", "--", "A", "B"]).is_ok());
        let broken = Engine { spec: EngineSpec { program: PathBuf::from(r"C:\gibt\es\nicht.exe"), args: vec![], source: "test".into() } };
        assert!(broken.json(&["--json", "meetings"]).unwrap_err().contains("nicht gestartet"));
    }
}
