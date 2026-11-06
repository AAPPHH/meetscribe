use std::io::{BufRead, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

fn emit(line: &str) {
    println!("{line}");
    std::io::stdout().flush().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let mode = std::env::var("FAKE_ENGINE_MODE").unwrap_or_default();
    if args.first().map(String::as_str) == Some("discard") {
        let accepted = args.len() == 3 && args[1] == "--" && args[2] != "m-fail";
        std::process::exit(if accepted { 0 } else { 1 });
    }
    if args.first().map(String::as_str) != Some("start") {
        let text = if has("devices") {
            r#"{"inputs":[{"index":0,"name":"Fake Mikrofon","api":"MME","channels":1,"rate":16000}],"inputs_error":null,"openvino":[{"device":"CPU","name":"Fake CPU"}]}"#
        } else if has("report") {
            r##"{"meeting_id":"fake","path":null,"markdown":"# Fake\n\nProtokoll"}"##
        } else if has("check") {
            r#"[{"id":"ffmpeg","label":"ffmpeg","ok":true,"fix":null}]"#
        } else if has("speakers") {
            "fake gesetzt"
        } else {
            "[]"
        };
        emit(text);
        return;
    }
    emit("Rauschen einer Bibliothek");
    emit(r#"{"event":"preflight"}"#);
    if mode == "preflight_fail" {
        emit(r#"{"event":"error","message":"ffmpeg nicht gefunden.","audio":null,"meeting_id":null,"retry":null,"exit_code":1}"#);
        std::process::exit(1);
    }
    if mode == "sleep" {
        std::thread::sleep(Duration::from_secs(60));
        return;
    }
    if mode == "tree" {
        let child = Command::new(std::env::current_exe().unwrap())
            .arg("start")
            .env("FAKE_ENGINE_MODE", "sleep")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        emit(&format!(r#"{{"event":"transcribing","audio":"C:\\fake\\aufnahme.flac","meeting_id":"fake-1","child_pid":{}}}"#, child.id()));
        std::thread::sleep(Duration::from_secs(60));
        return;
    }
    if has("--stdin-stop") {
        emit(r#"{"event":"recording","input":"Fake Mikrofon","path":"C:\\fake\\aufnahme.flac"}"#);
        emit(r#"{"event":"warning","message":"Eingang 'Fake Mikrofon': kein Audio seit 6 s (Test)"}"#);
        if mode == "crash" {
            for n in 1..=20 {
                eprintln!("Ursache zeile {n}");
            }
            std::process::exit(-1073741819);
        }
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).unwrap();
    }
    emit(r#"{"event":"transcribing","audio":"C:\\fake\\aufnahme.flac","meeting_id":"fake-1"}"#);
    std::thread::sleep(Duration::from_millis(400));
    if mode == "process_fail" {
        emit(r#"{"event":"error","message":"Verarbeitung fehlgeschlagen: Test","detail":"Ursache im Detail","audio":"C:\\fake\\aufnahme.flac","meeting_id":null,"retry":"meetscribe start C:\\fake\\aufnahme.flac --device npu --no-open --events","exit_code":1}"#);
        std::process::exit(1);
    }
    emit(r#"{"event":"done","meeting_id":"fake","report":"C:\\fake\\protokoll.md","audio":"C:\\fake\\aufnahme.flac","segments":2,"speakers":2,"labels":[{"label":"SPEAKER_00","sample":"Guten Morgen zusammen."},{"label":"SPEAKER_01","sample":"Hallo, schoen dass ihr da seid."}]}"#);
}
