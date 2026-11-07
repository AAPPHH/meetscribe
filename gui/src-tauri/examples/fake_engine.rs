use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const AUDIO: &str = r"C:\fake\aufnahme.flac";
const STILL_RUNNING: &str = "Die frühere Verarbeitung läuft vermutlich noch. Bitte einen Moment warten und es dann erneut versuchen.";

struct Out(Mutex<()>);

impl Out {
    fn send(&self, message: Value) {
        let _guard = self.0.lock().unwrap();
        let mut stdout = std::io::stdout().lock();
        let _ = writeln!(stdout, "{message}");
        let _ = stdout.flush();
    }

    fn note(&self, method: &str, params: Value) {
        self.send(json!({ "jsonrpc": "2.0", "method": method, "params": params }));
    }

    fn event(&self, mut event: Value) {
        event["run"] = json!(1);
        self.note("run.event", event);
    }

    fn reply(&self, id: &Value, result: Result<Value, (i64, &str)>) {
        match result {
            Ok(result) => self.send(json!({ "jsonrpc": "2.0", "id": id, "result": result })),
            Err((code, message)) => self.send(json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })),
        }
    }
}

struct Fake {
    out: Out,
    mode: String,
    listing: Option<String>,
    once: Option<String>,
    recording: AtomicBool,
    stop: AtomicBool,
    active: AtomicBool,
}

impl Fake {
    fn sleep(ms: u64) {
        std::thread::sleep(Duration::from_millis(ms));
    }

    fn finish(&self, code: i64, terminal: bool) {
        self.active.store(false, Ordering::SeqCst);
        self.out.note("run.exit", json!({ "run": 1, "code": code, "terminal": terminal }));
    }

    fn run(&self, params: &Value) {
        let mode = self.mode.as_str();
        self.out.event(json!({ "event": "preflight" }));
        if mode == "preflight_fail" {
            self.out.event(json!({ "event": "error", "message": "ffmpeg nicht gefunden.", "audio": null, "meeting_id": null, "retry": null, "exit_code": 1 }));
            return self.finish(1, true);
        }
        if mode == "sleep" {
            return Self::sleep(60_000);
        }
        if mode == "tree" {
            let child = Command::new(std::env::current_exe().unwrap())
                .args(["--mode", "childsleep"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            self.out.event(json!({ "event": "transcribing", "audio": AUDIO, "meeting_id": "fake-1", "child_pid": child.id() }));
            return Self::sleep(60_000);
        }
        if params["audio"].is_null() {
            self.recording.store(true, Ordering::SeqCst);
            self.out.event(json!({ "event": "recording", "input": "Fake Mikrofon", "path": AUDIO }));
            self.out.event(json!({ "event": "warning", "message": "Eingang 'Fake Mikrofon': kein Audio seit 6 s (Test)" }));
            if mode == "crash_run" {
                eprintln!("Ursache zeile 1");
                eprintln!("Ursache zeile 2");
                std::process::exit(-1073741819);
            }
            while !self.stop.load(Ordering::SeqCst) {
                Self::sleep(10);
            }
            self.recording.store(false, Ordering::SeqCst);
        }
        self.out.event(json!({ "event": "transcribing", "audio": AUDIO, "meeting_id": "fake-1" }));
        self.out.event(json!({ "event": "alive" }));
        Self::sleep(400);
        if mode == "process_fail" {
            self.out.event(json!({ "event": "error", "message": "Verarbeitung fehlgeschlagen: Test", "detail": "Ursache im Detail", "audio": AUDIO, "meeting_id": null, "retry": "meetscribe start x --events", "exit_code": 1 }));
            return self.finish(1, true);
        }
        self.out.event(json!({
            "event": "done", "meeting_id": "fake", "report": r"C:\fake\protokoll.md", "audio": AUDIO, "segments": 2, "speakers": 2,
            "labels": [{ "label": "SPEAKER_00", "sample": "Guten Morgen zusammen." }, { "label": "SPEAKER_01", "sample": "Hallo, schoen dass ihr da seid." }]
        }));
        self.finish(0, true);
    }

    fn download(&self) {
        self.out.note("download.event", json!({ "run": 1, "event": "download", "repo": "OpenVINO/whisper-small-int8-ov", "bytes": 1000000, "total": 257000000 }));
        if self.mode == "download_fail" {
            self.out.note("download.event", json!({ "run": 1, "event": "error", "message": "Der Download ist fehlgeschlagen.", "detail": "whisper OpenVINO/whisper-small-int8-ov: FEHLER OSError: offline", "exit_code": 1 }));
            return self.out.note("download.exit", json!({ "run": 1, "code": 1, "terminal": true, "stderr": "" }));
        }
        self.out.note("download.event", json!({ "run": 1, "event": "done", "messages": ["whisper OpenVINO/whisper-small-int8-ov: ok"] }));
        self.out.note("download.exit", json!({ "run": 1, "code": 0, "terminal": true, "stderr": "" }));
    }

    fn handle(self: &Arc<Self>, id: Value, method: &str, params: Value) {
        let listing: Value = self.listing.as_deref().and_then(|t| serde_json::from_str(t).ok()).unwrap_or(json!([]));
        if method == "search" && params["term"] == "crash!" {
            if let Some(marker) = self.once.as_deref().filter(|m| !Path::new(m).exists()) {
                let _ = std::fs::write(marker, b"x");
                eprintln!("Absturz mitten in der Abfrage");
                std::process::exit(9);
            }
        }
        let meeting = params["meeting_id"].as_str().unwrap_or("");
        let state = match meeting {
            "m-fertig" => "complete",
            "m-teil" | "m-fail" => "partial",
            _ => "absent",
        };
        let result = match method {
            "test.echo" => Ok(params.clone()),
            "test.slow" => {
                Self::sleep(params["ms"].as_u64().unwrap_or(0));
                Ok(json!({ "tag": params["tag"], "pid": std::process::id() }))
            }
            "test.crash" => {
                eprintln!("Absturz im Test");
                std::process::exit(7);
            }
            "test.hang" => loop {
                Self::sleep(1000);
            },
            "ping" => Ok(json!({ "pid": std::process::id() })),
            "meetings" => Ok(listing),
            "search" => Ok(json!([{ "meeting_id": "fake", "seg_id": 0, "start_s": 1.0, "display_name": "A", "text": params["term"] }])),
            "report" => Ok(json!({ "meeting_id": params["meeting_id"], "path": params["out"], "markdown": "# Fake\n\nProtokoll" })),
            "models" => Ok(json!({ "default_model": "large-v3-turbo", "backends": [], "hardware": [], "models": [], "choices": [] })),
            "devices" => Ok(json!({ "inputs": [{ "index": 0, "name": "Fake Mikrofon", "api": "MME", "channels": 1, "rate": 16000 }], "inputs_error": null, "openvino": [{ "device": "CPU", "name": "Fake CPU" }] })),
            "check" => Ok(json!([{ "id": "ffmpeg", "label": "ffmpeg", "ok": true, "fix": null }])),
            "speakers.set" => Ok(json!({ "meeting_id": params["meeting_id"], "label": params["label"], "name": params["name"] })),
            "meeting.state" => Ok(json!({ "meeting_id": meeting, "state": state })),
            "meeting.settle" if meeting == "m-fail" => Err((-32010, STILL_RUNNING)),
            "meeting.settle" => Ok(json!({ "meeting_id": meeting, "state": if state == "partial" { "discarded" } else { state } })),
            "run.start" => {
                if self.active.swap(true, Ordering::SeqCst) {
                    Err((-32010, "Es läuft bereits eine Sitzung."))
                } else {
                    self.stop.store(false, Ordering::SeqCst);
                    let (fake, params) = (self.clone(), params.clone());
                    std::thread::spawn(move || fake.run(&params));
                    Ok(json!({ "run": 1 }))
                }
            }
            "run.stop" if self.recording.load(Ordering::SeqCst) => {
                self.stop.store(true, Ordering::SeqCst);
                Ok(json!({ "run": 1 }))
            }
            "run.stop" => Err((-32000, "Es läuft keine Aufnahme.")),
            "download.start" => {
                let fake = self.clone();
                std::thread::spawn(move || {
                    Self::sleep(50);
                    fake.download();
                });
                Ok(json!({ "run": 1 }))
            }
            "shutdown" => {
                self.out.reply(&id, Ok(json!({ "ok": true })));
                std::process::exit(0);
            }
            _ => Err((-32601, "Unbekannte Methode")),
        };
        self.out.reply(&id, result);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let option = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let mode = option("--mode").unwrap_or_default();
    if mode == "childsleep" {
        return Fake::sleep(60_000);
    }
    if mode == "exit_at_start" {
        eprintln!("Traceback: kaputte Installation");
        eprintln!("ModuleNotFoundError: meetscribe");
        std::process::exit(3);
    }
    let fake = Arc::new(Fake { out: Out(Mutex::new(())), mode: mode.clone(), listing: option("--listing"), once: option("--once"), recording: AtomicBool::new(false), stop: AtomicBool::new(false), active: AtomicBool::new(false) });
    fake.out.send(json!({ "unerwartet": "Rauschen einer Bibliothek" }));
    println!("Rauschen, das kein JSON ist");
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let Ok(message) = serde_json::from_str::<Value>(&line) else { continue };
        let (id, method, params) = (message["id"].clone(), message["method"].as_str().unwrap_or("").to_string(), message["params"].clone());
        if method == "initialize" {
            if mode == "silent_start" {
                Fake::sleep(60_000);
            }
            if mode == "old_protocol" {
                fake.out.reply(&id, Err((-32001, "Engine und Oberfläche sprechen verschiedene Protokollversionen (Engine: 0, Oberfläche: 1). Bitte beide auf denselben Stand bringen.")));
            } else {
                fake.out.reply(&id, Ok(json!({ "protocol": 1, "engine": { "name": "fake", "version": "0", "pid": std::process::id() }, "methods": [] })));
            }
            continue;
        }
        if method == "test.hang" {
            fake.out.note("test.noise", json!({ "hang": true }));
        }
        let worker = fake.clone();
        let handle = std::thread::spawn(move || worker.handle(id, &method, params));
        if message["method"] == "test.hang" {
            let _ = handle.join();
        }
    }
}
