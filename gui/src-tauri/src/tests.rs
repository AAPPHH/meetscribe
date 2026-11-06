use super::*;
use std::collections::HashSet;

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
fn json_output_tolerates_leading_noise() {
    assert_eq!(parse_json_output("[1,2]\n").unwrap(), json!([1, 2]));
    assert_eq!(parse_json_output("Rauschen\n{\"a\":1}\n").unwrap(), json!({ "a": 1 }));
    assert!(parse_json_output("nur Text").is_err());
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
fn start_args_cover_recording_and_file_modes_and_keep_audio_out_of_option_parsing() {
    let base = StartOptions { title: Some("  Runde ".into()), audio: None, mic: Some("DJI".into()), device: "gpu".into(), diarize: false };
    assert_eq!(
        base.args(),
        ["start", "--events", "--no-open", "--device", "gpu", "--stdin-stop", "--mic", "DJI", "--title", "Runde", "--no-diarize"]
    );
    let file = StartOptions { title: None, audio: Some(r"C:\a b\x.wav".into()), mic: Some("DJI".into()), device: "npu".into(), diarize: true };
    assert_eq!(file.args(), ["start", "--events", "--no-open", "--device", "npu", "--", r"C:\a b\x.wav"]);
    let tricky = StartOptions { title: Some("T".into()), audio: Some("--events".into()), ..file.clone() };
    assert_eq!(tricky.args(), ["start", "--events", "--no-open", "--device", "npu", "--title", "T", "--", "--events"]);
    let blank = StartOptions { title: Some(" ".into()), audio: Some("".into()), mic: Some("".into()), device: "cpu".into(), diarize: true };
    assert_eq!(blank.args(), ["start", "--events", "--no-open", "--device", "cpu", "--stdin-stop"]);
    assert!(StartOptions { device: "tpu".into(), ..blank }.validate().is_err());
}

#[test]
fn missing_or_misspelled_audio_is_refused_in_german_before_any_engine_starts() {
    let dir = scratch("audio");
    let real = dir.join("a.wav");
    std::fs::write(&real, b"x").unwrap();
    let options = |audio: &str| StartOptions { title: None, audio: Some(audio.into()), mic: None, device: "npu".into(), diarize: true };
    assert!(options(&real.to_string_lossy()).validate().is_ok());
    for bad in [dir.join("fehlt.wav").to_string_lossy().to_string(), dir.to_string_lossy().to_string(), "--events".to_string()] {
        let err = options(&bad).validate().unwrap_err();
        assert!(err.starts_with("Audiodatei nicht gefunden:") && err.contains(&bad), "{err}");
    }
    assert!(StartOptions { audio: None, ..options("") }.validate().is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn settings_roundtrip_and_defaults() {
    let dir = scratch("settings");
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

#[test]
fn pending_run_follows_the_event_stream() {
    let dir = scratch("pending");
    let audio = dir.join("a.flac");
    std::fs::write(&audio, b"x").unwrap();
    let audio_text = audio.to_string_lossy().to_string();
    let store = PendingStore { path: dir.join(PENDING_FILE) };
    let options = StartOptions { title: Some("Runde".into()), audio: None, mic: None, device: "gpu".into(), diarize: false };
    assert!(store.load().is_none());
    store.observe(&json!({ "event": "preflight" }), &options);
    assert!(store.load().is_none());
    store.observe(&json!({ "event": "recording", "path": audio_text }), &options);
    let expected = Pending { audio: audio_text.clone(), meeting_id: None, title: Some("Runde".into()), device: "gpu".into(), diarize: false };
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

fn fake_command(mode: &str, args: &[&str]) -> Command {
    let mut cmd = Command::new(fake_engine());
    cmd.args(args).env("FAKE_ENGINE_MODE", mode).detached();
    cmd
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

fn run_fake(mode: &str, args: &[&str], stop_after: Option<&str>) -> (Vec<(String, Value)>, Arc<Session>) {
    let seen = Arc::new(Mutex::new(Vec::<(String, Value)>::new()));
    let log = seen.clone();
    let session = Session::spawn(fake_command(mode, args), Arc::new(move |name, payload| log.lock().unwrap().push((name.to_string(), payload))), None).unwrap();
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
    assert_eq!(exit["audio"], r"C:\fake\aufnahme.flac");
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
    assert_eq!(events[3].1["audio"], r"C:\fake\aufnahme.flac");
}

#[test]
fn crash_exit_still_carries_the_audio_path_and_stderr_is_logged_in_full() {
    let dir = scratch("crash");
    let log = dir.join(LOG_FILE);
    let seen = Arc::new(Mutex::new(Vec::<(String, Value)>::new()));
    let sink = seen.clone();
    let session = Session::spawn(fake_command("crash", &["start", "--events", "--stdin-stop"]), Arc::new(move |n, p| sink.lock().unwrap().push((n.to_string(), p))), Some(log.clone())).unwrap();
    assert!(wait_for(400, || !session.active()));
    let exit = seen.lock().unwrap().last().unwrap().1.clone();
    assert_eq!((exit["code"].as_i64().unwrap() != 0, exit["terminal"].as_bool()), (true, Some(false)));
    assert_eq!(exit["audio"], r"C:\fake\aufnahme.flac");
    assert_eq!(exit["log"], log.to_string_lossy().to_string());
    let text = std::fs::read_to_string(&log).unwrap();
    assert!(text.contains("Ursache zeile 1") && text.contains("Ursache zeile 20"));
    assert!(exit["stderr"].as_str().unwrap().lines().count() <= STDERR_TAIL);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn session_state_reports_stage_audio_input_and_warnings_for_a_reloaded_view() {
    let session = Session::spawn(fake_command("", &["start", "--events", "--stdin-stop"]), Arc::new(|_, _| {}), None).unwrap();
    assert!(wait_for(200, || session.stage() == Stage::Recording && !session.info().warnings.is_empty()));
    let state = session.state();
    assert_eq!((state["active"].as_bool(), state["stage"].as_str(), state["input"].as_str()), (Some(true), Some("recording"), Some("Fake Mikrofon")));
    assert_eq!(state["audio"], r"C:\fake\aufnahme.flac");
    assert_eq!(state["warnings"].as_array().unwrap().len(), 1);
    assert!(state["elapsed_s"].as_u64().is_some());
    session.stop().unwrap();
    wait_for(400, || !session.active());
}

#[test]
fn kill_is_refused_while_recording_but_works_after() {
    let session = Session::spawn(fake_command("", &["start", "--events", "--stdin-stop"]), Arc::new(|_, _| {}), None).unwrap();
    wait_for(200, || session.stage() == Stage::Recording);
    assert!(session.kill().is_err());
    session.stop().unwrap();
    wait_for(200, || session.stage() != Stage::Recording);
    assert!(session.kill().is_ok());
}

#[test]
fn kill_takes_the_whole_process_tree_with_it() {
    let seen = Arc::new(Mutex::new(Vec::<(String, Value)>::new()));
    let log = seen.clone();
    let session = Session::spawn(fake_command("tree", &["start", "x.wav", "--events"]), Arc::new(move |n, p| log.lock().unwrap().push((n.to_string(), p))), None).unwrap();
    assert!(wait_for(400, || session.stage() == Stage::Processing));
    let child = seen.lock().unwrap().iter().find_map(|(_, v)| v["child_pid"].as_u64()).unwrap();
    assert!(process_alive(child));
    session.kill().unwrap();
    assert!(wait_for(200, || !session.active()));
    let gone = wait_for(200, || !process_alive(child));
    if !gone {
        let _ = Command::new("taskkill").args(["/PID", &child.to_string(), "/F"]).detached().status();
    }
    assert!(gone, "Kindprozess {child} lebt nach dem Abbruch weiter");
    assert_eq!(session.info().meeting_id.as_deref(), Some("fake-1"));
}

#[test]
fn concurrent_launches_leave_exactly_one_session() {
    let dir = scratch("launch");
    let state = Arc::new(AppState {
        settings_path: dir.join(SETTINGS_FILE),
        pending: PendingStore { path: dir.join(PENDING_FILE) },
        log_path: dir.join(LOG_FILE),
        session: Mutex::new(None),
    });
    let threads: Vec<_> = (0..6)
        .map(|_| {
            let state = state.clone();
            std::thread::spawn(move || state.launch(|| Session::spawn(fake_command("sleep", &["start", "--events"]), Arc::new(|_, _| {}), None)))
        })
        .collect();
    let results: Vec<Result<(), String>> = threads.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert!(results.iter().filter_map(|r| r.as_ref().err()).all(|e| e == "Es läuft bereits eine Sitzung."));
    state.active_session().unwrap().kill().unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn engine_json_queries_run_through_the_command_line() {
    let engine = Engine { spec: EngineSpec::direct(fake_engine(), "test") };
    assert_eq!(engine.json(&["--json", "devices"]).unwrap()["inputs"][0]["name"], "Fake Mikrofon");
    assert_eq!(engine.json(&["--json", "report", "--", "fake"]).unwrap()["markdown"], "# Fake\n\nProtokoll");
    assert!(engine.run(&["speakers", "set", "--meeting", "fake", "--", "A", "B"]).is_ok());
    let broken = Engine { spec: EngineSpec::direct(r"C:\gibt\es\nicht.exe", "test") };
    assert!(broken.json(&["--json", "meetings"]).unwrap_err().contains("nicht gestartet"));
}

#[test]
fn discard_asks_the_engine_to_drop_exactly_that_meeting_and_reports_refusal() {
    let engine = Engine { spec: EngineSpec::direct(fake_engine(), "test") };
    assert!(engine.discard("m-teil"));
    assert!(!engine.discard("m-fail"));
}

const LISTING: &str = r#"[{"meeting_id":"m-fertig","complete":true},{"meeting_id":"m-teil","complete":false},{"meeting_id":"m-fail","complete":false}]"#;

struct PendingCase {
    dir: PathBuf,
    store: PendingStore,
    audio: String,
}

impl PendingCase {
    fn new(name: &str) -> PendingCase {
        let dir = scratch(name);
        let audio = dir.join("a.flac");
        std::fs::write(&audio, b"x").unwrap();
        PendingCase { store: PendingStore { path: dir.join(PENDING_FILE) }, audio: audio.to_string_lossy().to_string(), dir }
    }

    fn engine(&self) -> Engine {
        let file = self.dir.join("meetings.json");
        std::fs::write(&file, LISTING).unwrap();
        let args = vec!["--meetings-file".to_string(), file.to_string_lossy().to_string()];
        Engine { spec: EngineSpec { program: fake_engine(), args, source: "test".into() } }
    }

    fn remember(&self, meeting_id: Option<&str>) {
        self.store.write(&Pending { audio: self.audio.clone(), meeting_id: meeting_id.map(String::from), title: None, device: "gpu".into(), diarize: false });
    }
}

#[test]
fn meeting_state_tells_absent_partial_and_complete_apart() {
    let engine = PendingCase::new("state").engine();
    assert_eq!(engine.meeting_state("m-fertig"), Ok(MeetingState::Complete));
    assert_eq!(engine.meeting_state("m-teil"), Ok(MeetingState::Partial));
    assert_eq!(engine.meeting_state("m-weg"), Ok(MeetingState::Absent));
    let broken = Engine { spec: EngineSpec::direct(r"C:\gibt\es\nicht.exe", "test") };
    assert!(broken.meeting_state("m-teil").is_err());
}

#[test]
fn a_run_that_finished_after_the_window_died_is_not_offered_and_not_started_twice() {
    let case = PendingCase::new("finished");
    case.remember(Some("m-fertig"));
    let err = case.store.reconcile(&case.engine(), &case.audio).unwrap_err();
    assert!(err.contains("bereits abgeschlossen") && err.contains("m-fertig"));
    assert!(!case.store.path.exists());
    case.remember(Some("m-fertig"));
    assert!(case.store.unfinished(Ok(case.engine())).is_none() && !case.store.path.exists());
    let _ = std::fs::remove_dir_all(&case.dir);
}

#[test]
fn a_partial_run_is_discarded_before_the_retry_and_a_run_that_still_holds_it_blocks_the_retry() {
    let case = PendingCase::new("partial");
    case.remember(Some("m-teil"));
    assert_eq!(case.store.unfinished(Ok(case.engine())).unwrap().meeting_id.as_deref(), Some("m-teil"));
    assert!(case.store.reconcile(&case.engine(), &case.audio).is_ok());
    case.remember(Some("m-fail"));
    assert_eq!(case.store.reconcile(&case.engine(), &case.audio).unwrap_err(), STILL_RUNNING);
    assert!(case.store.path.exists());
    case.remember(Some("m-teil"));
    assert!(case.store.reconcile(&case.engine(), r"C:\ein\anderes\audio.flac").is_ok());
    case.remember(None);
    assert!(case.store.reconcile(&case.engine(), &case.audio).is_ok() && case.store.path.exists());
    let _ = std::fs::remove_dir_all(&case.dir);
}

#[test]
fn forgetting_removes_the_partial_meeting_but_never_a_complete_one() {
    let case = PendingCase::new("forget");
    case.remember(Some("m-fail"));
    assert_eq!(case.store.forget(Ok(case.engine())).unwrap_err(), STILL_RUNNING);
    assert!(case.store.path.exists());
    for id in [Some("m-teil"), Some("m-fertig"), Some("m-weg"), None] {
        case.remember(id);
        assert!(case.store.forget(Ok(case.engine())).is_ok() && !case.store.path.exists());
    }
    case.remember(Some("m-teil"));
    assert!(case.store.forget(Err("keine Engine".into())).is_ok() && !case.store.path.exists());
    let _ = std::fs::remove_dir_all(&case.dir);
}

#[test]
fn heartbeat_events_never_reach_the_frontend() {
    let (events, _) = run_fake("", &["start", "x.wav", "--events"], None);
    assert_eq!(names(&events), ["preflight", "transcribing", "done", "exit"]);
}
