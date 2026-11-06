use std::io::{BufRead, Write};

fn emit(line: &str) {
    println!("{line}");
    std::io::stdout().flush().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let mode = std::env::var("FAKE_ENGINE_MODE").unwrap_or_default();
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
    if has("--stdin-stop") {
        emit(r#"{"event":"recording","input":"Fake Mikrofon","path":"C:\\fake\\aufnahme.flac"}"#);
        emit(r#"{"event":"warning","message":"Eingang 'Fake Mikrofon': kein Audio seit 6 s (Test)"}"#);
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).unwrap();
    }
    emit(r#"{"event":"transcribing","audio":"C:\\fake\\aufnahme.flac"}"#);
    std::thread::sleep(std::time::Duration::from_millis(400));
    if mode == "process_fail" {
        emit(r#"{"event":"error","message":"Verarbeitung fehlgeschlagen: Test","audio":"C:\\fake\\aufnahme.flac","meeting_id":null,"retry":"meetscribe start C:\\fake\\aufnahme.flac --device npu --no-open --events","exit_code":1}"#);
        std::process::exit(1);
    }
    emit(r#"{"event":"done","meeting_id":"fake","report":"C:\\fake\\protokoll.md","audio":"C:\\fake\\aufnahme.flac","segments":2,"speakers":2,"labels":[{"label":"SPEAKER_00","sample":"Guten Morgen zusammen."},{"label":"SPEAKER_01","sample":"Hallo, schoen dass ihr da seid."}]}"#);
}
