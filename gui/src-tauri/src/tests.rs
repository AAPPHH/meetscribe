use super::*;
use engine::PROTOCOL;
use std::collections::HashSet;
use std::time::Duration;

const BUILD_REPO: &str = r"C:\build\repo";

fn locator<'a>(settings: &'a Settings, present: &'a dyn Fn(&Path) -> bool, path_var: &str) -> Locator<'a> {
    Locator { settings, exists: present, path_var: path_var.to_string(), exe_dir: Some(PathBuf::from(r"C:\app")), build_repo: PathBuf::from(BUILD_REPO) }
}

fn present_from(paths: &[&str]) -> impl Fn(&Path) -> bool {
    let set: HashSet<PathBuf> = paths.iter().map(PathBuf::from).collect();
    move |p| set.contains(p)
}

fn found(settings: &Settings, paths: &[&str], path_var: &str) -> (PathBuf, String) {
    let present = present_from(paths);
    let spec = locator(settings, &present, path_var).locate().unwrap();
    (spec.program, spec.source)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("meetscribe-gui-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
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
fn run_info_remembers_audio_input_meeting_and_warnings() {
    let mut info = RunInfo::default();
    info.absorb(&json!({ "event": "recording", "input": "DJI", "path": r"C:\a\x.flac" }));
    info.absorb(&json!({ "event": "warning", "message": "leise" }));
    assert_eq!((info.input.as_deref(), info.audio.as_deref()), (Some("DJI"), Some(r"C:\a\x.flac")));
    info.absorb(&json!({ "event": "transcribing", "audio": r"C:\a\x.flac", "meeting_id": "m-1" }));
    info.absorb(&json!({ "event": "error", "message": "x" }));
    assert_eq!((info.meeting_id.as_deref(), info.warnings.as_slice()), (Some("m-1"), ["leise".to_string()].as_slice()));
}

#[test]
fn saved_engine_wins_and_a_missing_one_is_an_error_not_a_silent_fallback() {
    let settings = Settings { engine: r"D:\x\meetscribe.exe".into(), ..Settings::default() };
    let venv = r"C:\build\repo\.venv\Scripts\meetscribe.exe";
    assert_eq!(found(&settings, &[r"D:\x\meetscribe.exe", venv], ""), (PathBuf::from(r"D:\x\meetscribe.exe"), "gespeicherter Pfad".into()));
    let present = present_from(&[venv, r"C:\build\repo\pyproject.toml"]);
    let err = locator(&settings, &present, "").locate().unwrap_err();
    assert!(err.contains(r"D:\x\meetscribe.exe") && err.contains("existiert nicht"));
}

#[test]
fn engine_is_searched_in_project_then_path_then_beside_the_app_then_uv() {
    let none = Settings::default();
    let venv = r"C:\build\repo\.venv\Scripts\meetscribe.exe";
    assert_eq!(found(&none, &[venv, r"C:\bin\meetscribe.exe"], r"C:\bin"), (PathBuf::from(venv), "Projektordner".into()));
    assert_eq!(found(&none, &[r"C:\bin\meetscribe.exe", r"C:\app\meetscribe.exe"], r"C:\x;C:\bin"), (PathBuf::from(r"C:\bin\meetscribe.exe"), "PATH".into()));
    assert_eq!(found(&none, &[r"C:\app\meetscribe.exe"], ""), (PathBuf::from(r"C:\app\meetscribe.exe"), "neben der Anwendung".into()));
    let custom = Settings { repo: r"E:\repo".into(), ..Settings::default() };
    assert_eq!(found(&custom, &[r"E:\repo\pyproject.toml", r"E:\repo\.venv\Scripts\meetscribe.exe"], "").1, "Projektordner");
    let present = present_from(&[r"C:\tools\uv.exe", r"E:\repo\pyproject.toml"]);
    let spec = locator(&custom, &present, r"C:\win;C:\tools").locate().unwrap();
    assert_eq!(spec.program, PathBuf::from(r"C:\tools\uv.exe"));
    assert_eq!(spec.args, ["run", "--project", r"E:\repo", "meetscribe"]);
    assert_eq!(spec.source, "uv");
}

#[test]
fn the_mcp_command_sits_beside_the_engine_carries_the_archive_variables_and_is_never_guessed() {
    let direct = EngineSpec::direct(r"C:\build\repo\.venv\Scripts\meetscribe.exe", "Projektordner");
    let present = present_from(&[r"C:\build\repo\.venv\Scripts\meetscribe-mcp.exe"]);
    let env = |key: &str| match key {
        "MEETSCRIBE_DATA" => Some(r"D:\Archiv".to_string()),
        "MEETSCRIBE_MODELS" => Some("  ".to_string()),
        _ => None,
    };
    let spec = direct.mcp(&present, &env).unwrap();
    assert_eq!(spec.program, PathBuf::from(r"C:\build\repo\.venv\Scripts\meetscribe-mcp.exe"));
    assert!(spec.args.is_empty());
    assert_eq!(spec.env, BTreeMap::from([("MEETSCRIBE_DATA".to_string(), r"D:\Archiv".to_string())]));
    let err = direct.mcp(&|_: &Path| false, &|_: &str| None).unwrap_err();
    assert!(err.contains("meetscribe-mcp.exe") && err.contains("uv sync"));
    let via_uv = EngineSpec { program: PathBuf::from(r"C:\tools\uv.exe"), args: ["run", "--project", r"E:\repo", "meetscribe"].map(String::from).to_vec(), source: "uv".into() };
    let spec = via_uv.mcp(&|_: &Path| false, &|_: &str| None).unwrap();
    assert_eq!(spec.program, PathBuf::from(r"C:\tools\uv.exe"));
    assert_eq!(spec.args, ["run", "--project", r"E:\repo", "meetscribe-mcp"]);
}

#[test]
fn missing_engine_message_names_no_developer_folder_and_a_bad_saved_project_is_reported() {
    let none = |_: &Path| false;
    let err = locator(&Settings::default(), &none, r"C:\tools").locate().unwrap_err();
    assert!(err.contains("Engine nicht gefunden") && err.contains("uv sync") && !err.contains("clones") && !err.contains(BUILD_REPO));
    let only_uv = |p: &Path| p == Path::new(r"C:\tools\uv.exe");
    assert!(locator(&Settings::default(), &only_uv, r"C:\tools").locate().is_err());
    let bad = Settings { repo: r"E:\leer".into(), ..Settings::default() };
    let err = locator(&bad, &only_uv, r"C:\tools").locate().unwrap_err();
    assert!(err.contains(r"E:\leer") && err.contains("pyproject.toml"));
}

#[test]
fn missing_or_misspelled_audio_is_refused_in_german_before_any_engine_starts() {
    let dir = scratch("audio");
    let real = dir.join("a.wav");
    std::fs::write(&real, b"x").unwrap();
    let options = |audio: &str| StartOptions { title: None, audio: Some(audio.into()), mic: None, device: "npu".into(), backend: None, model: None, diarize: true };
    assert!(options(&real.to_string_lossy()).validate().is_ok());
    for bad in [dir.join("fehlt.wav").to_string_lossy().to_string(), dir.to_string_lossy().to_string(), "--events".to_string()] {
        let err = options(&bad).validate().unwrap_err();
        assert!(err.starts_with("Audiodatei nicht gefunden:") && err.contains(&bad), "{err}");
    }
    assert!(StartOptions { audio: None, ..options("") }.validate().is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn start_params_cover_recording_and_file_modes_and_keep_automatic_choices_out() {
    let base = StartOptions { title: Some("  Runde ".into()), audio: None, mic: Some("DJI".into()), device: "gpu".into(), backend: None, model: None, diarize: false };
    assert_eq!(base.params(), json!({ "device": "gpu", "diarize": false, "mic": "DJI", "title": "Runde" }));
    let file = StartOptions { title: None, audio: Some(r"C:\a b\x.wav".into()), mic: Some("DJI".into()), device: "npu".into(), backend: None, model: None, diarize: true };
    assert_eq!(file.params(), json!({ "device": "npu", "diarize": true, "audio": r"C:\a b\x.wav" }));
    let blank = StartOptions { title: Some(" ".into()), audio: Some("".into()), mic: Some("".into()), device: "cpu".into(), backend: None, model: None, diarize: true };
    assert_eq!(blank.params(), json!({ "device": "cpu", "diarize": true }));
    assert!(StartOptions { device: "tpu".into(), ..blank }.validate().is_err());
    let auto = StartOptions { backend: Some("auto".into()), model: Some(" auto ".into()), ..base.clone() };
    assert_eq!(auto.params(), base.params());
    let chosen = StartOptions { device: "igpu".into(), backend: Some("faster-whisper".into()), model: Some("small:int8".into()), mic: None, title: None, ..base.clone() };
    assert_eq!(chosen.params(), json!({ "device": "igpu", "diarize": false, "backend": "faster-whisper", "model": "small:int8" }));
    assert!(chosen.validate().is_ok());
    for bad in ["--events", "-x", ".hidden", "a b", "a;b", r"C:\x"] {
        assert!(StartOptions { model: Some(bad.into()), ..base.clone() }.validate().is_err(), "{bad}");
        assert!(StartOptions { backend: Some(bad.into()), ..base.clone() }.validate().is_err(), "{bad}");
    }
    for device in ["auto", "npu", "gpu", "igpu", "cpu"] {
        assert!(StartOptions { device: device.into(), ..base.clone() }.validate().is_ok(), "{device}");
    }
}

#[test]
fn downloads_pass_exactly_one_checked_model_to_the_engine() {
    assert_eq!(download_params("openvino", "small:int8").unwrap(), json!({ "backend": "openvino", "model": "small:int8" }));
    for (backend, model) in [("auto", "small"), ("openvino", "auto"), ("openvino", "--x"), ("", "small"), ("openvino", "a b")] {
        assert!(download_params(backend, model).is_err(), "{backend} {model}");
    }
}

#[test]
fn settings_roundtrip_and_defaults() {
    let dir = scratch("settings");
    let path = dir.join(SETTINGS_FILE);
    assert_eq!(Settings::load(&path), Settings::default());
    let settings = Settings { engine: "x".into(), mic: "DJI".into(), device: "gpu".into(), backend: "faster-whisper".into(), model: "small".into(), diarize: false, ..Settings::default() };
    settings.save(&path).unwrap();
    assert_eq!(Settings::load(&path), settings);
    assert_eq!((Settings::default().device.as_str(), Settings::default().backend.as_str(), Settings::default().model.as_str()), ("auto", "auto", "auto"));
    std::fs::write(&path, r#"{"device":"gpu","mic":"DJI"}"#).unwrap();
    assert_eq!(Settings::load(&path), Settings { device: "gpu".into(), mic: "DJI".into(), ..Settings::default() });
    assert!(Settings { backend: "--x".into(), ..Settings::default() }.save(&path).is_err());
    assert!(Settings { model: "".into(), ..Settings::default() }.save(&path).is_err());
    std::fs::write(&path, r#"{"device":"cpu"}"#).unwrap();
    assert_eq!(Settings::load(&path), Settings { device: "cpu".into(), ..Settings::default() });
    assert!(Settings { device: "tpu".into(), ..Settings::default() }.save(&path).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn pending_run_follows_the_event_stream() {
    let dir = scratch("pending");
    let audio = dir.join("a.flac");
    std::fs::write(&audio, b"x").unwrap();
    let audio_text = audio.to_string_lossy().to_string();
    let store = PendingStore { path: dir.join(PENDING_FILE) };
    let options = StartOptions { title: Some("Runde".into()), audio: None, mic: None, device: "gpu".into(), backend: None, model: None, diarize: false };
    assert!(store.load().is_none());
    store.observe(&json!({ "event": "preflight" }), &options);
    assert!(store.load().is_none());
    store.observe(&json!({ "event": "recording", "path": audio_text }), &options);
    let expected = Pending { audio: audio_text.clone(), meeting_id: None, title: Some("Runde".into()), device: "gpu".into(), backend: None, model: None, diarize: false };
    let chosen = StartOptions { backend: Some("faster-whisper".into()), model: Some("small".into()), ..options.clone() };
    store.observe(&json!({ "event": "recording", "path": audio_text }), &chosen);
    assert_eq!(store.load().map(|p| (p.backend, p.model)), Some((Some("faster-whisper".into()), Some("small".into()))));
    std::fs::write(&store.path, format!(r#"{{"audio":{},"meeting_id":null,"title":null,"device":"npu","diarize":true}}"#, json!(audio_text))).unwrap();
    assert_eq!(store.load().map(|p| (p.device, p.backend, p.model)), Some(("npu".into(), None, None)));
    store.clear();
    store.observe(&json!({ "event": "recording", "path": audio_text }), &options);
    assert_eq!(store.load(), Some(expected.clone()));
    store.observe(&json!({ "event": "transcribing", "audio": audio_text, "meeting_id": "m-1" }), &options);
    assert_eq!(store.load().unwrap().meeting_id.as_deref(), Some("m-1"));
    store.observe(&json!({ "event": "error", "audio": audio_text, "meeting_id": null }), &options);
    assert_eq!(store.load(), Some(expected));
    store.observe(&json!({ "event": "done", "audio": audio_text }), &options);
    assert!(store.load().is_none());
    store.observe(&json!({ "event": "recording", "path": audio_text }), &options);
    store.observe(&json!({ "event": "error", "audio": audio_text, "meeting_id": "gespeichert" }), &options);
    assert!(store.load().is_none());
    store.observe(&json!({ "event": "recording", "path": audio_text }), &options);
    std::fs::remove_file(&audio).unwrap();
    assert!(store.load().is_none() && !store.path.exists());
    let _ = std::fs::remove_dir_all(dir);
}

fn fake_engine() -> PathBuf {
    std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("examples").join(format!("fake_engine{EXE}"))
}

fn wait_for(limit: usize, mut done: impl FnMut() -> bool) -> bool {
    for _ in 0..limit {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    false
}

fn process_alive(pid: u64) -> bool {
    let out = Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"]).detached().output().unwrap();
    String::from_utf8_lossy(&out.stdout).contains(&format!("\"{pid}\""))
}

type Seen = Arc<Mutex<Vec<(String, Value)>>>;

fn fake_spec(args: &[&str]) -> EngineSpec {
    EngineSpec { program: fake_engine(), args: args.iter().map(|a| a.to_string()).collect(), source: "test".into() }
}

fn quick() -> Timeouts {
    Timeouts { startup: Duration::from_secs(3), query: Duration::from_millis(400), probe: Duration::from_millis(400), shutdown: Duration::from_secs(2), ..Timeouts::default() }
}

struct Rig {
    client: Arc<EngineClient>,
    notes: Seen,
}

impl Rig {
    fn new(args: &[&str], timeouts: Timeouts) -> Rig {
        let notes: Seen = Arc::new(Mutex::new(vec![]));
        let sink = notes.clone();
        let notify: Notify = Arc::new(move |method, params| sink.lock().unwrap().push((method.to_string(), params)));
        let spec = fake_spec(args);
        Rig { client: EngineClient::new(Box::new(move || Ok(spec.clone())), notify, None, timeouts), notes }
    }

    fn pid(&self) -> u64 {
        self.client.call("ping", json!({})).unwrap()["pid"].as_u64().unwrap()
    }

    fn noted(&self, method: &str) -> Vec<Value> {
        self.notes.lock().unwrap().iter().filter(|(m, _)| m == method).map(|(_, p)| p.clone()).collect()
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.client.shutdown();
    }
}

#[test]
fn replies_are_matched_by_id_even_when_they_arrive_out_of_order() {
    let rig = Rig::new(&[], Timeouts::default());
    let order = Arc::new(Mutex::new(Vec::<String>::new()));
    let threads: Vec<_> = [("a", 600), ("b", 10), ("c", 200)]
        .into_iter()
        .map(|(tag, ms)| {
            let (client, order) = (rig.client.clone(), order.clone());
            std::thread::spawn(move || {
                if tag != "a" {
                    std::thread::sleep(Duration::from_millis(100));
                }
                let reply = client.call("test.slow", json!({ "tag": tag, "ms": ms })).unwrap();
                order.lock().unwrap().push(reply["tag"].as_str().unwrap().to_string());
                reply["tag"].as_str().unwrap().to_string() == tag
            })
        })
        .collect();
    assert!(threads.into_iter().all(|t| t.join().unwrap()));
    assert_eq!(*order.lock().unwrap(), ["b", "c", "a"]);
    for n in 0..20 {
        assert_eq!(rig.client.call("test.echo", json!({ "n": n })).unwrap()["n"], n);
    }
}

#[test]
fn noise_on_stdout_and_unknown_notifications_do_not_disturb_the_protocol() {
    let rig = Rig::new(&[], Timeouts::default());
    assert_eq!(rig.client.call("devices", json!({})).unwrap()["inputs"][0]["name"], "Fake Mikrofon");
    assert_eq!(rig.client.call("report", json!({ "meeting_id": "fake" })).unwrap()["markdown"], "# Fake\n\nProtokoll");
    assert_eq!(rig.client.info().unwrap()["protocol"], PROTOCOL);
    assert!(rig.client.call("gibt.es.nicht", json!({})).unwrap_err().contains("Unbekannte Methode"));
}

#[test]
fn a_crash_fails_the_call_with_the_engine_output_then_restarts_and_says_so() {
    let rig = Rig::new(&[], Timeouts::default());
    let before = rig.pid();
    let error = rig.client.call("test.crash", json!({})).unwrap_err();
    assert!(error.contains("unerwartet beendet") && error.contains("Code 7") && error.contains("Absturz im Test"), "{error}");
    assert!(wait_for(200, || !rig.noted("engine.restarted").is_empty()));
    assert_eq!(rig.noted("engine.crashed")[0]["code"], 7);
    assert!(rig.noted("engine.crashed")[0]["stderr"].as_str().unwrap().contains("Absturz im Test"));
    assert_eq!(rig.noted("engine.restarted")[0]["ok"], true);
    let after = rig.pid();
    assert_ne!(before, after);
    assert!(wait_for(100, || !process_alive(before)));
}

#[test]
fn a_read_only_query_is_repeated_once_when_the_engine_dies_under_it() {
    let marker = scratch("once").join("marker");
    let rig = Rig::new(&["--once", &marker.to_string_lossy()], Timeouts::default());
    let before = rig.pid();
    let hits = rig.client.call("search", json!({ "term": "crash!" })).unwrap();
    assert_eq!(hits[0]["text"], "crash!");
    assert_ne!(rig.pid(), before);
    assert_eq!(rig.noted("engine.crashed").len(), 1);
    assert!(marker.exists());
}

#[test]
fn a_timeout_is_reported_and_an_engine_that_stays_silent_is_replaced() {
    let rig = Rig::new(&[], quick());
    let before = rig.pid();
    let error = rig.client.call("test.hang", json!({})).unwrap_err();
    assert!(error.contains("Zeitüberschreitung") && error.contains("test.hang"), "{error}");
    assert!(wait_for(200, || !rig.noted("engine.restarted").is_empty()));
    assert_eq!(rig.noted("engine.crashed").len(), 1);
    assert_ne!(rig.pid(), before);
    assert!(wait_for(100, || !process_alive(before)));
}

#[test]
fn a_slow_but_healthy_engine_keeps_running_after_a_timeout() {
    let rig = Rig::new(&[], quick());
    let before = rig.pid();
    assert!(rig.client.call("test.slow", json!({ "tag": "x", "ms": 900 })).unwrap_err().contains("Zeitüberschreitung"));
    std::thread::sleep(Duration::from_millis(1200));
    assert!(rig.noted("engine.crashed").is_empty());
    assert_eq!(rig.pid(), before);
}

#[test]
fn startup_failures_say_why_in_german_and_never_loop() {
    let broken = Rig::new(&["--mode", "exit_at_start"], quick());
    let error = broken.client.start().unwrap_err();
    assert!(error.contains("ließ sich nicht starten") && error.contains("ModuleNotFoundError: meetscribe") && error.contains("Code 3"), "{error}");
    let old = Rig::new(&["--mode", "old_protocol"], quick());
    assert!(old.client.start().unwrap_err().contains("Protokollversionen"));
    let silent = Rig::new(&["--mode", "silent_start"], Timeouts { startup: Duration::from_millis(400), ..quick() });
    assert!(silent.client.start().unwrap_err().contains("nicht gemeldet"));
    let missing = EngineClient::new(Box::new(|| Ok(EngineSpec::direct(r"C:\gibt\es\nicht.exe", "test"))), Arc::new(|_, _| {}), None, quick());
    assert!(missing.start().unwrap_err().contains("nicht gestartet werden"));
    std::thread::sleep(Duration::from_millis(300));
    assert!(broken.noted("engine.crashed").is_empty() && old.noted("engine.restarted").is_empty() && silent.noted("engine.crashed").is_empty());
}

#[test]
fn an_engine_that_cannot_be_located_reports_the_locator_message() {
    let client = EngineClient::new(Box::new(|| Err("Engine nicht gefunden. Pfad eintragen.".into())), Arc::new(|_, _| {}), None, quick());
    assert_eq!(client.call("meetings", json!({})).unwrap_err(), "Engine nicht gefunden. Pfad eintragen.");
}

#[test]
fn shutdown_leaves_no_engine_behind_and_refuses_new_starts() {
    let rig = Rig::new(&[], Timeouts::default());
    let pid = rig.pid();
    assert!(process_alive(pid.into()));
    rig.client.shutdown();
    assert!(wait_for(100, || !process_alive(pid.into())));
    assert!(rig.client.call("meetings", json!({})).unwrap_err().contains("wird beendet"));
    assert!(rig.noted("engine.crashed").is_empty());
}

#[test]
fn repeated_crashes_stop_the_automatic_restarts() {
    let rig = Rig::new(&[], Timeouts::default());
    for n in 1..=3 {
        let _ = rig.client.call("test.crash", json!({}));
        assert!(wait_for(400, || rig.noted("engine.restarted").len() == n));
    }
    let restarted = rig.noted("engine.restarted");
    assert_eq!((restarted[0]["ok"].as_bool(), restarted[1]["ok"].as_bool(), restarted[2]["ok"].as_bool()), (Some(true), Some(true), Some(false)));
    assert!(restarted[2]["error"].as_str().unwrap().contains("mehrfach"));
    assert!(rig.client.call("ping", json!({})).is_ok());
}

struct App {
    core: Arc<Core>,
    engine: Arc<EngineClient>,
    seen: Seen,
    dir: PathBuf,
}

impl App {
    fn new(name: &str, args: &[&str]) -> App {
        let dir = scratch(name);
        let core = Arc::new(Core::new(&dir));
        let seen: Seen = Arc::new(Mutex::new(vec![]));
        let (router, sink) = (core.clone(), seen.clone());
        let notify: Notify = Arc::new(move |method, params| router.route(&|name, payload| sink.lock().unwrap().push((name.to_string(), payload)), method, params));
        let spec = fake_spec(args);
        let engine = EngineClient::new(Box::new(move || Ok(spec.clone())), notify, Some(dir.join(LOG_FILE)), Timeouts::default());
        App { core, engine, seen, dir }
    }

    fn options(&self, audio: bool) -> StartOptions {
        let file = self.dir.join("a.wav");
        std::fs::write(&file, b"x").unwrap();
        StartOptions { title: None, audio: audio.then(|| file.to_string_lossy().to_string()), mic: None, device: "npu".into(), backend: None, model: None, diarize: true }
    }

    fn names(&self) -> Vec<String> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|(name, v)| match name.as_str() {
                EVENT_NAME | DOWNLOAD_EVENT_NAME => v["event"].as_str().unwrap().to_string(),
                EXIT_NAME | DOWNLOAD_EXIT_NAME => "exit".to_string(),
                other => other.to_string(),
            })
            .collect()
    }

    fn last(&self, name: &str) -> Value {
        self.seen.lock().unwrap().iter().rev().find(|(n, _)| n == name).map(|(_, v)| v.clone()).unwrap()
    }

    fn idle(&self) -> bool {
        wait_for(400, || !self.core.run.lock().unwrap().active)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.engine.shutdown();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn a_recording_runs_through_the_engine_and_a_stop_line_ends_the_recording() {
    let app = App::new("run_record", &[]);
    launch(&app.core, &app.engine, app.options(false)).unwrap();
    assert!(wait_for(400, || app.core.run.lock().unwrap().stage == Stage::Recording));
    let state = app.core.run.lock().unwrap().state();
    assert_eq!((state["active"].as_bool(), state["stage"].as_str(), state["input"].as_str()), (Some(true), Some("recording"), Some("Fake Mikrofon")));
    assert_eq!(state["warnings"].as_array().unwrap().len(), 1);
    assert_eq!(launch(&app.core, &app.engine, app.options(false)).unwrap_err(), SESSION_BUSY);
    app.engine.call("run.stop", json!({})).unwrap();
    assert!(app.idle());
    assert_eq!(app.names(), ["preflight", "recording", "warning", "transcribing", "done", "exit"]);
    let exit = app.last(EXIT_NAME);
    assert_eq!((exit["code"].as_i64(), exit["terminal"].as_bool()), (Some(0), Some(true)));
    assert_eq!(exit["audio"], r"C:\fake\aufnahme.flac");
    assert_eq!(app.core.run.lock().unwrap().state()["active"], false);
    assert_eq!(app.last(EVENT_NAME)["labels"][0]["sample"], "Guten Morgen zusammen.");
    assert!(app.engine.call("run.stop", json!({})).unwrap_err().contains("keine Aufnahme"));
}

#[test]
fn failures_arrive_as_error_events_and_heartbeats_never_reach_the_frontend() {
    let app = App::new("run_fail", &["--mode", "process_fail"]);
    launch(&app.core, &app.engine, app.options(true)).unwrap();
    assert!(app.idle());
    assert_eq!(app.names(), ["preflight", "transcribing", "error", "exit"]);
    let error = app.last(EVENT_NAME);
    assert_eq!(error["audio"], r"C:\fake\aufnahme.flac");
    assert!(error["retry"].as_str().unwrap().contains("--events"));
    let early = App::new("run_early", &["--mode", "preflight_fail"]);
    launch(&early.core, &early.engine, early.options(false)).unwrap();
    assert!(early.idle());
    assert_eq!(early.names(), ["preflight", "error", "exit"]);
    assert_eq!(early.last(EXIT_NAME)["code"], 1);
}

#[test]
fn a_crash_during_a_run_ends_it_for_the_frontend_with_the_audio_and_restarts_the_engine() {
    let app = App::new("run_crash", &["--mode", "crash_run"]);
    launch(&app.core, &app.engine, app.options(false)).unwrap();
    assert!(wait_for(400, || app.names().contains(&"exit".to_string())));
    let exit = app.last(EXIT_NAME);
    assert_eq!((exit["code"].as_i64().unwrap() != 0, exit["terminal"].as_bool()), (true, Some(false)));
    assert_eq!(exit["audio"], r"C:\fake\aufnahme.flac");
    assert!(exit["stderr"].as_str().unwrap().contains("Ursache zeile 1"));
    assert_eq!(exit["log"], app.dir.join(LOG_FILE).to_string_lossy().to_string());
    assert!(wait_for(400, || app.names().contains(&NOTICE_NAME.to_string())));
    assert_eq!(app.last(NOTICE_NAME)["text"], RESTARTED_RUN);
    assert!(!app.core.run.lock().unwrap().active);
    assert!(app.engine.call("ping", json!({})).is_ok());
    assert!(std::fs::read_to_string(app.dir.join(LOG_FILE)).is_ok());
}

#[test]
fn a_crash_while_idle_only_posts_the_plain_notice() {
    let app = App::new("idle_crash", &[]);
    let _ = app.engine.call("test.crash", json!({}));
    assert!(wait_for(400, || app.names().contains(&NOTICE_NAME.to_string())));
    assert_eq!(app.last(NOTICE_NAME)["text"], RESTARTED);
    assert!(!app.names().contains(&"exit".to_string()));
}

#[test]
fn leaving_during_processing_kills_the_engine_tree_and_discards_the_partial_meeting_through_a_fresh_engine() {
    let app = App::new("leave", &["--mode", "tree"]);
    launch(&app.core, &app.engine, app.options(true)).unwrap();
    assert!(wait_for(400, || app.core.run.lock().unwrap().stage == Stage::Processing));
    let child = app.seen.lock().unwrap().iter().find_map(|(_, v)| v["child_pid"].as_u64()).unwrap();
    let engine_pid = app.engine.pid().unwrap();
    assert!(process_alive(child) && process_alive(engine_pid.into()));
    leave(&app.core, &app.engine).unwrap();
    assert!(wait_for(200, || !process_alive(child)), "Kindprozess {child} lebt weiter");
    assert!(wait_for(100, || !process_alive(engine_pid.into())));
    assert!(!app.core.run.lock().unwrap().active);
    assert!(!app.names().contains(&"exit".to_string()) && !app.names().contains(&NOTICE_NAME.to_string()));
    assert!(app.engine.call("ping", json!({})).is_ok());
}

#[test]
fn leaving_is_refused_while_recording_and_the_engine_stays() {
    let app = App::new("leave_record", &[]);
    launch(&app.core, &app.engine, app.options(false)).unwrap();
    assert!(wait_for(400, || app.core.run.lock().unwrap().stage == Stage::Recording));
    assert_eq!(leave(&app.core, &app.engine).unwrap_err(), RECORDING_KEPT);
    assert!(app.engine.pid().is_some());
    app.engine.call("run.stop", json!({})).unwrap();
    assert!(app.idle());
}

#[test]
fn a_download_streams_progress_then_done_and_reports_failures_with_detail() {
    let app = App::new("download", &[]);
    app.core.download.store(true, Ordering::SeqCst);
    app.engine.call("download.start", download_params("openvino", "small:int8").unwrap()).unwrap();
    assert!(wait_for(400, || !app.core.download.load(Ordering::SeqCst)));
    assert_eq!(app.names(), ["download", "done", "exit"]);
    assert_eq!(app.seen.lock().unwrap()[0].1["total"], 257000000);
    assert!(app.seen.lock().unwrap()[0].1.get("run").is_none());
    let failing = App::new("download_fail", &["--mode", "download_fail"]);
    failing.engine.call("download.start", download_params("openvino", "small").unwrap()).unwrap();
    assert!(wait_for(400, || failing.names().contains(&"exit".to_string())));
    assert_eq!(failing.names(), ["download", "error", "exit"]);
    assert!(failing.seen.lock().unwrap()[1].1["detail"].as_str().unwrap().contains("offline"));
    assert_eq!(failing.last(DOWNLOAD_EXIT_NAME)["terminal"], true);
}

#[test]
fn the_tracker_allows_one_session_and_reports_its_state_for_a_reloaded_view() {
    let mut tracker = Tracker::idle();
    let options = StartOptions { title: None, audio: None, mic: None, device: "cpu".into(), backend: None, model: None, diarize: true };
    assert_eq!(tracker.state()["active"], false);
    tracker.begin(options.clone()).unwrap();
    assert_eq!(tracker.begin(options.clone()).unwrap_err(), SESSION_BUSY);
    tracker.absorb(&json!({ "event": "recording", "input": "DJI", "path": r"C:\a.flac" }));
    assert_eq!((tracker.state()["stage"].as_str(), tracker.state()["audio"].as_str()), (Some("recording"), Some(r"C:\a.flac")));
    tracker.absorb(&json!({ "event": "transcribing", "audio": r"C:\a.flac", "meeting_id": "m-1" }));
    assert_eq!(tracker.finish().meeting_id.as_deref(), Some("m-1"));
    tracker.begin(options).unwrap();
}

fn pending_engine(name: &str) -> (Rig, PendingStore, String, PathBuf) {
    let dir = scratch(name);
    let audio = dir.join("a.flac");
    std::fs::write(&audio, b"x").unwrap();
    (Rig::new(&[], Timeouts::default()), PendingStore { path: dir.join(PENDING_FILE) }, audio.to_string_lossy().to_string(), dir)
}

fn remember(store: &PendingStore, audio: &str, meeting_id: Option<&str>) {
    store.write(&Pending { audio: audio.into(), meeting_id: meeting_id.map(String::from), title: None, device: "gpu".into(), backend: None, model: None, diarize: false });
}

#[test]
fn meeting_state_tells_absent_partial_and_complete_apart() {
    let rig = Rig::new(&[], Timeouts::default());
    assert_eq!(meeting_state(&rig.client, "m-fertig"), Ok(MeetingState::Complete));
    assert_eq!(meeting_state(&rig.client, "m-teil"), Ok(MeetingState::Partial));
    assert_eq!(meeting_state(&rig.client, "m-weg"), Ok(MeetingState::Absent));
    assert_eq!(settle_meeting(&rig.client, "m-teil"), Ok(MeetingState::Absent));
    assert_eq!(settle_meeting(&rig.client, "m-fail"), Err(STILL_RUNNING.to_string()));
}

#[test]
fn a_run_that_finished_after_the_window_died_is_not_offered_and_not_started_twice() {
    let (rig, store, audio, dir) = pending_engine("finished");
    remember(&store, &audio, Some("m-fertig"));
    let err = store.reconcile(&rig.client, &audio).unwrap_err();
    assert!(err.contains("bereits abgeschlossen") && err.contains("m-fertig"));
    assert!(!store.path.exists());
    remember(&store, &audio, Some("m-fertig"));
    assert!(store.unfinished(&rig.client).is_none() && !store.path.exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_partial_run_is_discarded_before_the_retry_and_a_run_that_still_holds_it_blocks_the_retry() {
    let (rig, store, audio, dir) = pending_engine("partial");
    remember(&store, &audio, Some("m-teil"));
    assert_eq!(store.unfinished(&rig.client).unwrap().meeting_id.as_deref(), Some("m-teil"));
    assert!(store.reconcile(&rig.client, &audio).is_ok());
    remember(&store, &audio, Some("m-fail"));
    assert_eq!(store.reconcile(&rig.client, &audio).unwrap_err(), STILL_RUNNING);
    assert!(store.path.exists());
    remember(&store, &audio, Some("m-teil"));
    assert!(store.reconcile(&rig.client, r"C:\ein\anderes\audio.flac").is_ok() && !store.path.exists());
    remember(&store, &audio, None);
    assert!(store.reconcile(&rig.client, &audio).is_ok() && store.path.exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_crashed_run_is_cleaned_up_when_another_file_starts_and_a_run_that_still_holds_it_blocks_that_start() {
    let (rig, store, audio, dir) = pending_engine("other");
    let other = r"C:\ein\anderes\audio.flac";
    for id in ["m-teil", "m-fertig", "m-weg"] {
        remember(&store, &audio, Some(id));
        assert!(store.reconcile(&rig.client, other).is_ok() && !store.path.exists(), "{id}");
    }
    remember(&store, &audio, Some("m-fail"));
    assert_eq!(store.reconcile(&rig.client, other).unwrap_err(), STILL_RUNNING);
    assert_eq!(store.read().unwrap().meeting_id.as_deref(), Some("m-fail"));
    remember(&store, &audio, None);
    assert!(store.reconcile(&rig.client, other).is_ok() && store.path.exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn forgetting_removes_the_partial_meeting_but_never_a_complete_one() {
    let (rig, store, audio, dir) = pending_engine("forget");
    remember(&store, &audio, Some("m-fail"));
    assert_eq!(store.forget(Some(&rig.client)).unwrap_err(), STILL_RUNNING);
    assert!(store.path.exists());
    for id in [Some("m-teil"), Some("m-fertig"), Some("m-weg"), None] {
        remember(&store, &audio, id);
        assert!(store.forget(Some(&rig.client)).is_ok() && !store.path.exists());
    }
    remember(&store, &audio, Some("m-teil"));
    assert!(store.forget(None).is_ok() && !store.path.exists());
    let _ = std::fs::remove_dir_all(dir);
}
