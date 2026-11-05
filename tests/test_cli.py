import os
import shutil
import signal
import subprocess
import sys
import threading
import types
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from meetscribe import cli, pipeline
from meetscribe.store import HotFile, Store, make_config

REAL_RECORDER = pipeline.Recorder

WORDS = [
    ("Grüße", 0.2, 0.6), ("aus", 0.6, 0.8), ("München", 0.8, 1.2),
    ("Schön", 1.7, 2.0), ("dass", 2.0, 2.2), ("Sie", 2.2, 2.4), ("da", 2.4, 2.6),
]
TURNS = [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}, {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"}]
OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
LAPTOP = {"name": "Mikrofonarray (Intel Smart Sound)", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 16000.0}
DJI = {"name": "DJI MIC MINI", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 16000.0}
ES_CONTINUOUS_SYSTEM = pipeline.AWAKE_FLAGS["continuous"] | pipeline.AWAKE_FLAGS["system_required"]


class FakeEngine:
    fail_devices = ()
    constructed = []
    silent = False

    def __init__(self, device, models, language, override=None):
        self.device, self.model = device, "fake-model"
        FakeEngine.constructed.append(device)

    def transcribe(self, audio):
        if self.device in FakeEngine.fail_devices:
            raise RuntimeError("DEVICE_LOST")
        if FakeEngine.silent:
            return {"words": [], "chunks": []}
        return {"words": [{"text": t, "start_s": s, "end_s": e} for t, s, e in WORDS], "chunks": []}


class FakeDiarizer:
    calls = []

    def __init__(self, models):
        pass

    def diarize(self, audio, **options):
        FakeDiarizer.calls.append(options)
        return TURNS


class FakeRecorder:
    started = []
    interrupt_with = "signal"
    problem_text = None
    seconds = 1.0
    overruns = 0

    def __init__(self, device, path, hostapis=()):
        self.device, self.path, self.rate, self.hostapi = device, path, 16000, hostapis[device["hostapi"]]

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.path, np.full(16000, 0.1, dtype="float32"), 16000)
        FakeRecorder.started.append(self.path)
        if FakeRecorder.interrupt_with == "signal":
            threading.Timer(0.05, signal.raise_signal, [signal.SIGINT]).start()

    def problem(self):
        return FakeRecorder.problem_text

    def stop(self):
        return {"path": str(self.path), "seconds": FakeRecorder.seconds, "overruns": FakeRecorder.overruns, "error": None}


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    for spec in pipeline.MODELS.values():
        (models / spec["dir"]).mkdir(parents=True, exist_ok=True)
        (models / spec["dir"] / "openvino_model.xml").write_text("x")
    monkeypatch.setenv("MEETSCRIBE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("MEETSCRIBE_MODELS", str(models))
    monkeypatch.delenv("MEETSCRIBE_DEVICE", raising=False)
    for name in (*OFFLINE, "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    FakeEngine.fail_devices, FakeEngine.constructed, FakeEngine.silent, FakeDiarizer.calls = (), [], False, []
    loaded = []
    monkeypatch.setattr(pipeline, "WhisperEngine", FakeEngine)
    monkeypatch.setattr(pipeline, "Diarizer", FakeDiarizer)
    monkeypatch.setattr(pipeline, "Recorder", FakeRecorder)
    FakeRecorder.started, FakeRecorder.interrupt_with = [], "signal"
    FakeRecorder.problem_text, FakeRecorder.seconds, FakeRecorder.overruns = None, 1.0, 0
    monkeypatch.setattr(pipeline.AudioDevices, "sounddevice_picker", staticmethod(lambda: pipeline.InputPicker([LAPTOP, DJI], ["MME"])))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setenv("MEETSCRIBE_REPORTS", str(tmp_path / "protokolle"))
    monkeypatch.setattr(pipeline.AudioLoader, "load", staticmethod(lambda path, rate=None: loaded.append(path) or np.full(48000, 0.1, dtype="float32")))
    audio = tmp_path / "Besprechung März.wav"
    audio.write_bytes(b"x")
    return {"tmp": tmp_path, "models": models, "audio": audio, "loaded": loaded, "store": Store(make_config(tmp_path / "data"))}


def add_pyannote(env):
    target = env["models"] / pipeline.SETTINGS["diarization"]["dir"]
    target.mkdir(parents=True)
    (target / "config.yaml").write_text("x")


def invoke(*args, **kwargs):
    return CliRunner().invoke(cli.main, [str(a) for a in args], **kwargs)


def transcribe(env, *extra, meeting_id="m1"):
    return invoke("transcribe", env["audio"], "--title", "Wochenbesprechung", "--date", "2026-03-04",
                  "--meeting-id", meeting_id, *extra)


def test_happy_path_transcribe_search_speakers_report_rotate(env):
    add_pyannote(env)
    result = transcribe(env, "--speakers", "2")
    assert result.exit_code == 0, result.output
    assert "m1: 2 Segmente, 2 Sprecher" in result.output and "angefordert: NPU, genutzt: NPU" in result.output
    assert FakeDiarizer.calls == [{"speakers": 2, "min_speakers": None, "max_speakers": None}]
    assert HotFile(env["store"].hot_path("m1")).is_complete()
    first = invoke("search", "münchen")
    assert first.exit_code == 0 and "Grüße aus München" in first.output and "SPEAKER_00" in first.output
    assert "SPEAKER_01" in invoke("speakers", "show", "--meeting", "m1").output
    assert invoke("speakers", "set", "SPEAKER_00", "Anna", "--meeting", "m1").exit_code == 0
    named = invoke("search", "münchen", "--speaker", "Anna")
    assert "Anna" in named.output and "Grüße" in named.output
    report = invoke("report", "m1", "--out", env["tmp"] / "out" / "m1.md")
    text = (env["tmp"] / "out" / "m1.md").read_text(encoding="utf-8")
    assert report.exit_code == 0 and "**[00:00] Anna:** Grüße aus München" in text and "SPEAKER_01" in text
    assert "Wochenbesprechung" in invoke("meetings").output
    assert "Anna" in invoke("stats", "--meeting", "m1").output
    assert invoke("sql", "SELECT count(*) AS n, any_value(device) AS d FROM segments").output.split()[-2:] == ["2", "NPU"]
    before = invoke("sql", "SELECT seg_id, speaker, text, device FROM segments ORDER BY seg_id").output
    rotated = invoke("rotate")
    assert "rotiert: m1" in rotated.output and not env["store"].hot_files()
    assert invoke("sql", "SELECT seg_id, speaker, text, device FROM segments ORDER BY seg_id").output == before
    assert invoke("search", "münchen", "--speaker", "Anna").output == named.output
    assert invoke("speakers", "unset", "SPEAKER_00", "--meeting", "m1").exit_code == 0
    assert "Anna" not in invoke("report", "m1").output


def test_missing_pyannote_stops_before_heavy_work(env):
    result = transcribe(env)
    assert result.exit_code != 0
    assert "Diarisierungsmodell fehlt" in result.output and "--no-diarize" in result.output and "setup" in result.output
    assert env["loaded"] == [] and FakeEngine.constructed == [] and not env["store"].hot_files()


def test_no_diarize_uses_unknown(env):
    result = transcribe(env, "--no-diarize")
    assert result.exit_code == 0, result.output
    assert FakeDiarizer.calls == []
    assert invoke("sql", "SELECT DISTINCT speaker FROM segments").output.split()[-1] == "UNKNOWN"


def test_missing_whisper_model_points_to_setup(env):
    for path in (env["models"] / pipeline.MODELS["npu"]["dir"]).iterdir():
        path.unlink()
    result = transcribe(env, "--no-diarize")
    assert result.exit_code != 0 and "meetscribe setup" in result.output and env["loaded"] == []


def test_fallback_reported_and_per_row_device(env):
    FakeEngine.fail_devices = ("npu",)
    result = transcribe(env, "--no-diarize")
    assert result.exit_code == 0, result.output
    assert result.output.count("NPU fehlgeschlagen") == 1
    assert "angefordert: NPU, genutzt: GPU" in result.output
    assert invoke("sql", "SELECT DISTINCT device FROM segments").output.split()[-1] == "GPU"


def test_no_fallback_flag_fails(env):
    FakeEngine.fail_devices = ("npu",)
    result = transcribe(env, "--no-diarize", "--no-fallback")
    assert result.exit_code != 0 and "--no-fallback" in result.output
    assert FakeEngine.constructed == ["npu"]
    assert not env["store"].hot_path("m1").exists()


def test_all_devices_failing_exits_nonzero_without_eos(env):
    FakeEngine.fail_devices = ("npu", "gpu", "cpu")
    result = transcribe(env, "--no-diarize")
    assert result.exit_code != 0 and "Alle Geraete fehlgeschlagen" in result.output
    assert not env["store"].hot_path("m1").exists()


def test_device_from_env_and_flag(env):
    result = invoke("transcribe", env["audio"], "--no-diarize", "--meeting-id", "b", env={"MEETSCRIBE_DEVICE": "gpu"})
    assert "angefordert: GPU" in result.output and FakeEngine.constructed[-1] == "gpu"
    result = invoke("transcribe", env["audio"], "--no-diarize", "--meeting-id", "c", "--device", "cpu", env={"MEETSCRIBE_DEVICE": "gpu"})
    assert "angefordert: CPU" in result.output and FakeEngine.constructed[-1] == "cpu"


def test_default_meeting_id_from_date_and_stem(env):
    for _ in range(2):
        result = invoke("transcribe", env["audio"], "--no-diarize", "--date", "2026-03-04")
        assert result.exit_code == 0, result.output
    assert sorted(p.stem for p in env["store"].hot_files()) == ["20260304-besprechung-maerz", "20260304-besprechung-maerz-2"]


def test_duplicate_and_unknown_meeting_errors(env):
    assert transcribe(env, "--no-diarize").exit_code == 0
    again = transcribe(env, "--no-diarize")
    assert again.exit_code != 0 and "existiert bereits" in again.output and "Traceback" not in again.output
    for args in (("report", "nope"), ("speakers", "show", "--meeting", "nope"), ("speakers", "set", "SPEAKER_99", "X", "--meeting", "m1"), ("sql", "SELEC 1")):
        result = invoke(*args)
        assert result.exit_code != 0 and result.exception.__class__ is SystemExit, args


def test_offline_env_set_for_commands_except_setup(env, monkeypatch):
    assert invoke("meetings").exit_code == 0
    assert all(os.environ[name] == "1" for name in OFFLINE)
    for name in OFFLINE:
        monkeypatch.setenv(name, "1")
    seen = {}

    def fake_download(repo, target, token=None):
        seen[repo] = (token, {name: os.environ.get(name) for name in OFFLINE})

    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(fake_download))
    monkeypatch.setattr(pipeline.ModelSetup, "warm", lambda self, device="npu": None)
    result = invoke("setup")
    assert result.exit_code == 0, result.output
    assert set(seen) == {spec["repo"] for spec in pipeline.MODELS.values()} and len(seen) == 2
    assert seen[pipeline.MODELS["npu"]["repo"]] == (None, {name: None for name in OFFLINE})
    assert "pyannote" in result.output and "fehlt" in result.output


def test_setup_with_token_downloads_pyannote(env, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(lambda repo, target, token=None: calls.append((repo, token))))
    result = invoke("setup", env={"HF_TOKEN": "tok"})
    assert result.exit_code == 0
    assert (pipeline.SETTINGS["diarization"]["repo"], "tok") in calls


def test_setup_warms_npu_and_tolerates_failure(env, monkeypatch):
    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(lambda repo, target, token=None: None))
    ok = invoke("setup")
    assert ok.exit_code == 0 and "Compile-Cache" in ok.output and FakeEngine.constructed == ["npu"]
    FakeEngine.fail_devices = ("npu",)
    bad = invoke("setup")
    assert bad.exit_code == 0 and bad.output.count("Warnung: NPU-Cache nicht aufgewaermt") == 1


def test_setup_whisper_failure_is_error(env, monkeypatch):
    def broken(repo, target, token=None):
        raise OSError("offline")

    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(broken))
    assert invoke("setup").exit_code != 0


def test_umlauts_survive_piped_output(env):
    assert transcribe(env, "--no-diarize").exit_code == 0
    result = invoke("sql", "SELECT text FROM segments WHERE text LIKE '%München%'")
    assert "München" in result.output


def test_rotate_skips_incomplete_unless_forced(env):
    with pytest.raises(KeyError):
        with env["store"].writer("m1", title="t", meeting_date="2026-03-04", language="de") as writer:
            writer.append([{"start_s": 0.0, "end_s": 1.0, "speaker": "UNKNOWN", "text": "x"}])
            raise KeyError
    assert "incomplete" in invoke("rotate").output
    assert "rotiert: m1" not in invoke("rotate").output


def test_setup_warm_failure_is_only_a_warning(env, monkeypatch):
    FakeEngine.fail_devices = ("npu",)
    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(lambda repo, target, token=None: None))
    result = invoke("setup")
    assert result.exit_code == 0 and "Warnung: NPU-Cache nicht aufgewaermt" in result.output


def test_last_meeting_is_default_for_report_and_speakers(env):
    assert invoke("report").exit_code != 0 and "Keine Besprechung" in invoke("speakers", "show").output
    assert transcribe(env, "--no-diarize", meeting_id="old").exit_code == 0
    add_pyannote(env)
    assert transcribe(env, "--speakers", "2", meeting_id="new").exit_code == 0
    assert "SPEAKER_01" in invoke("speakers", "show").output
    assert invoke("speakers", "set", "SPEAKER_00", "Anna").exit_code == 0
    assert "Anna" in invoke("report").output and "Anna" not in invoke("report", "old").output
    assert invoke("speakers", "unset", "SPEAKER_00").exit_code == 0
    assert "Anna" not in invoke("report").output


def start(*extra, **kwargs):
    return invoke("start", "--title", "Wochenplanung", *extra, **kwargs)


def reports(env):
    return sorted((env["tmp"] / "protokolle").glob("*.md"))


def recordings(env):
    return sorted((env["tmp"] / "data" / "aufnahmen").glob("*.flac"))


def test_start_preflight_failure_records_nothing(env, monkeypatch):
    result = start()
    assert result.exit_code != 0 and "Diarisierungsmodell fehlt" in result.output
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    missing = start("--no-diarize")
    assert missing.exit_code != 0 and "ffmpeg nicht gefunden" in missing.output
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(pipeline.AudioDevices, "sounddevice_picker", staticmethod(lambda: pipeline.InputPicker([DJI], ["MME"])))
    nomic = start("--no-diarize", "--mic", "laptop")
    assert nomic.exit_code != 0 and "Kein Audio-Eingang" in nomic.output
    for path in (env["models"] / pipeline.MODELS["npu"]["dir"]).iterdir():
        path.unlink()
    nomodel = start("--no-diarize")
    assert nomodel.exit_code != 0 and "meetscribe setup" in nomodel.output
    assert FakeRecorder.started == [] and not recordings(env) and env["loaded"] == [] and not reports(env)


def test_start_auto_mic_prefers_dji_else_laptop(env, monkeypatch):
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 0, result.output
    assert "Eingang: DJI MIC MINI [MME]" in result.output
    monkeypatch.setattr(pipeline.AudioDevices, "sounddevice_picker", staticmethod(lambda: pipeline.InputPicker([LAPTOP], ["MME"])))
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 0, result.output
    assert "Eingang: Mikrofonarray" in result.output
    explicit = start("--no-diarize", "--no-open", "--mic", "mikrofonarray")
    assert "Eingang: Mikrofonarray" in explicit.output


@pytest.mark.parametrize("mode", ["signal", "keyboard"])
def test_start_ctrl_c_ends_recording_and_continues(env, monkeypatch, mode):
    FakeRecorder.interrupt_with = mode
    if mode == "keyboard":
        def interrupted(self, timeout):
            raise KeyboardInterrupt

        monkeypatch.setattr(pipeline.Interrupt, "wait", interrupted)
    before = signal.getsignal(signal.SIGINT)
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output and "Aborted" not in result.output
    assert pipeline.RECORDING["lid_notice"] in result.output and "Aufnahme gespeichert" in result.output
    assert len(recordings(env)) == 1 and len(reports(env)) == 1
    assert signal.getsignal(signal.SIGINT) is before
    assert "UNKNOWN" in reports(env)[0].read_text(encoding="utf-8")


def test_start_failure_after_recording_prints_retry_and_keeps_audio(env):
    FakeEngine.fail_devices = ("npu", "gpu", "cpu")
    result = start("--no-diarize", "--speakers", "2")
    assert result.exit_code == 1
    audio = recordings(env)[0]
    assert f"Audio erhalten: {audio}" in result.output
    retry = [line for line in result.output.splitlines() if line.startswith("Erneut versuchen:")][0]
    assert "meetscribe start" in retry and str(audio) in retry and "--title Wochenplanung --speakers 2 --no-diarize" in retry
    assert audio.exists() and not reports(env)
    FakeEngine.fail_devices = ()
    ok = invoke("start", audio, "--no-diarize", "--no-open", "--title", "Wochenplanung", "--speakers", "2")
    assert ok.exit_code == 0, ok.output
    assert audio.exists() and len(recordings(env)) == 1 and len(reports(env)) == 1


def test_start_abort_leaves_no_partial_meeting(env):
    FakeEngine.fail_devices = ("npu", "gpu", "cpu")
    assert start("--no-diarize").exit_code == 1
    assert not env["store"].hot_files()
    FakeEngine.fail_devices = ()
    audio = recordings(env)[0]
    assert invoke("start", audio, "--no-diarize", "--no-open", "--title", "Wochenplanung").exit_code == 0
    ids = [line.split()[0] for line in invoke("meetings").output.splitlines()[2:]]
    assert ids == [f"{date.today():%Y%m%d}-wochenplanung"]


def test_start_without_speech_explains_and_keeps_audio(env):
    FakeEngine.silent = True
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 1 and "Traceback" not in result.output and "Keine Sprache erkannt" in result.output
    assert f"Audio erhalten: {recordings(env)[0]}" in result.output
    assert not env["store"].hot_files() and not reports(env)
    transcribed = transcribe(env, "--no-diarize")
    assert transcribed.exit_code == 1 and "Keine Sprache erkannt" in transcribed.output and not env["store"].hot_files()


def test_start_failure_with_spaces_quotes_retry(env):
    FakeEngine.fail_devices = ("npu", "gpu", "cpu")
    result = invoke("start", env["audio"], "--no-diarize", "--title", "Mein Treffen")
    assert result.exit_code == 1 and '--title "Mein Treffen"' in result.output and f'"{env["audio"]}"' in result.output
    assert env["audio"].exists() and FakeRecorder.started == []


def test_start_tty_names_speakers_and_opens_report(env, monkeypatch):
    add_pyannote(env)
    opened = []
    monkeypatch.setattr(cli, "stdin_is_tty", lambda: True)
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)
    result = start("--speakers", "2", input="Anna\n\n")
    assert result.exit_code == 0, result.output
    assert 'SPEAKER_00: "Grüße aus München"' in result.output and 'SPEAKER_01: "Schön dass Sie da"' in result.output
    text = reports(env)[0].read_text(encoding="utf-8")
    assert "Anna" in text and "SPEAKER_01" in text and "München" in text
    assert opened == [reports(env)[0]]
    assert invoke("speakers", "set", "SPEAKER_01", "Bert").exit_code == 0
    final = invoke("report").output
    assert "Anna" in final and "Bert" in final and "SPEAKER_0" not in final
    opened.clear()
    assert start("--no-open", "--speakers", "2", input="\n\n").exit_code == 0
    assert opened == []


def test_start_non_tty_skips_naming_and_opening(env, monkeypatch):
    add_pyannote(env)
    opened = []
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)
    result = start("--speakers", "2")
    assert result.exit_code == 0, result.output
    assert "Name fuer" not in result.output and opened == []
    assert "SPEAKER_00" in reports(env)[0].read_text(encoding="utf-8")


def test_start_report_name_collision_gets_suffix(env):
    for _ in range(3):
        assert start("--no-diarize", "--no-open").exit_code == 0
    stamp = f"{date.today():%Y-%m-%d}_wochenplanung"
    assert sorted(p.name for p in reports(env)) == sorted([f"{stamp}.md", f"{stamp}-2.md", f"{stamp}-3.md"])
    assert len(recordings(env)) == 3


def test_start_with_audiofile_does_not_record(env):
    result = invoke("start", env["audio"], "--no-diarize", "--no-open")
    assert result.exit_code == 0, result.output
    assert FakeRecorder.started == [] and not recordings(env) and len(reports(env)) == 1
    assert "besprechung-maerz" in reports(env)[0].name
    assert "Grüße aus München" in reports(env)[0].read_bytes().decode("utf-8")


def test_start_rotate_failure_is_only_a_warning(env, monkeypatch):
    calls = []

    def failing(self, older_than_days=0, force=False):
        calls.append(older_than_days)
        raise OSError("disk")

    monkeypatch.setattr(type(env["store"]), "rotate", failing)
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 0 and calls == [30]
    assert result.output.count("Warnung: Rotation fehlgeschlagen: OSError: disk") == 1 and len(reports(env)) == 1


def test_keep_awake_wraps_start_transcribe_and_setup(env, monkeypatch):
    calls = []
    released = [ES_CONTINUOUS_SYSTEM, pipeline.AWAKE_FLAGS["continuous"]]
    monkeypatch.setattr(pipeline.KeepAwake, "native", staticmethod(lambda: calls.append))
    assert start("--no-diarize", "--no-open").exit_code == 0
    assert calls == released
    calls.clear()
    assert transcribe(env, "--no-diarize").exit_code == 0
    assert calls == released
    calls.clear()
    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(lambda repo, target, token=None: None))
    assert invoke("setup").exit_code == 0
    assert calls == released
    calls.clear()
    FakeEngine.fail_devices = ("npu", "gpu", "cpu")
    assert start("--no-diarize").exit_code == 1
    assert calls == released


@pytest.mark.skipif(shutil.which("git") is None, reason="git fehlt")
def test_gitignore_covers_media_and_keeps_project_files():
    root = Path(__file__).resolve().parent.parent
    names = ["a.ogg", "a.opus", "a.aac", "a.mp4", "a.wma", "a.webm", "a.mov", "a.mkv", ".env.local", "protokoll-x.md", "data/x.arrow"]
    ignored = subprocess.run(["git", "check-ignore", *names], cwd=root, capture_output=True, text=True).stdout.split()
    assert sorted(ignored) == sorted(names)
    kept = subprocess.run(["git", "check-ignore", "README.md", "pyproject.toml"], cwd=root, capture_output=True, text=True)
    assert kept.returncode == 1 and kept.stdout == ""


def test_sql_cannot_fetch_extensions_but_reads_local_files(env):
    assert transcribe(env, "--no-diarize").exit_code == 0
    assert invoke("rotate").exit_code == 0
    remote = invoke("sql", "SELECT * FROM read_parquet('https://example.invalid/x.parquet')")
    assert remote.exit_code == 1 and "Traceback" not in remote.output and "requires the extension httpfs" in remote.output
    relock = invoke("sql", "SET autoload_known_extensions=true")
    assert relock.exit_code == 1 and "Traceback" not in relock.output
    parquet = next((env["tmp"] / "data" / "cold").rglob("*.parquet")).as_posix()
    direct = invoke("sql", f"SELECT count(*) AS n FROM read_parquet('{parquet}')")
    view = invoke("sql", "SELECT count(*) AS n FROM segments_named")
    assert direct.exit_code == 0 and direct.output == view.output and int(view.output.split()[-1]) > 0


def test_start_with_interrupted_diarization_setup_records_nothing(env):
    add_pyannote(env)
    (env["models"] / pipeline.SETTINGS["diarization"]["dir"] / "config.yaml").write_text("segmentation: $model/segmentation\n")
    result = start()
    assert result.exit_code != 0 and "unvollstaendig" in result.output and "meetscribe setup" in result.output and "--no-diarize" in result.output
    assert FakeRecorder.started == [] and not recordings(env)
    assert start("--no-diarize", "--no-open").exit_code == 0


def test_start_warns_once_per_real_rotation_error_and_stays_quiet_for_skips(env, monkeypatch):
    monkeypatch.setitem(pipeline.START, "rotate_days", 0)
    with pytest.raises(KeyError):
        with env["store"].writer("broken", title="t", meeting_date="2026-03-04", language="de") as writer:
            writer.append([{"start_s": 0.0, "end_s": 1.0, "speaker": "UNKNOWN", "text": "x"}])
            raise KeyError
    with env["store"].writer("hollow", title="t", meeting_date="2026-03-04", language="de"):
        pass
    quiet = start("--no-diarize", "--no-open")
    assert quiet.exit_code == 0 and "Warnung" not in quiet.output

    def failing(self, path, table):
        raise OSError("disk voll")

    monkeypatch.setattr(type(env["store"]), "_rotate_one", failing)
    for number in (1, 2):
        with env["store"].writer(f"ready-{number}", title="t", meeting_date="2026-03-04", language="de") as writer:
            writer.append([{"start_s": 0.0, "end_s": 1.0, "speaker": "UNKNOWN", "text": "x"}])
    result = start("--no-diarize", "--no-open")
    warnings = [line for line in result.output.splitlines() if line.startswith("Warnung")]
    assert result.exit_code == 0 and len(warnings) == 3 and all("disk voll" in line for line in warnings)
    assert sum("ready-" in line for line in warnings) == 2 and not any("broken" in line or "hollow" in line for line in warnings)


def test_start_open_failure_warns_with_path_and_still_rotates(env, monkeypatch):
    rotated = []

    def refuse(path):
        raise OSError("keine Zuordnung")

    monkeypatch.setattr(cli, "stdin_is_tty", lambda: True)
    monkeypatch.setattr(os, "startfile", refuse, raising=False)
    monkeypatch.setattr(type(env["store"]), "rotate", lambda self, older_than_days=0, force=False: rotated.append(older_than_days) or {"rotated": [], "skipped": []})
    result = start("--no-diarize", input="\n")
    assert result.exit_code == 0, result.output
    assert f"Warnung: Protokoll konnte nicht geoeffnet werden ({reports(env)[0]}): OSError: keine Zuordnung" in result.output
    assert rotated == [30] and len(reports(env)) == 1


def test_start_transcribes_partial_recording_with_warning(env):
    FakeRecorder.problem_text = "kein Audio seit 6 s (Geraet getrennt oder blockiert)"
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 0, result.output
    assert result.output.count("Warnung: Eingang") == 1 and "Warnung: Aufnahme unvollstaendig" in result.output
    assert len(recordings(env)) == 1 and len(reports(env)) == 1


def test_start_without_usable_recording_fails_and_keeps_audio(env):
    FakeRecorder.problem_text, FakeRecorder.seconds = "Schreibfehler, dieser Eingang wird nicht weiter aufgezeichnet (OSError: disk)", 0.0
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 1 and "Traceback" not in result.output and "Keine brauchbare Aufnahme gespeichert" in result.output
    assert len(recordings(env)) == 1 and not reports(env) and not env["store"].hot_files() and env["loaded"] == []


def test_start_prints_overruns_only_when_present(env):
    assert "Ueberlaeufe" not in start("--no-diarize", "--no-open").output
    FakeRecorder.overruns = 3
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 0 and any(f"Ueberlaeufe: 3 ({path})" in result.output for path in recordings(env))


def test_record_reports_problems_after_live_warning(env):
    FakeRecorder.problem_text = "kein Audio seit 6 s (Geraet getrennt oder blockiert)"
    result = invoke("record", "--out", env["tmp"] / "r.flac", "--seconds", "5")
    assert result.exit_code == 1 and result.output.count("kein Audio seit 6 s") == 2 and "Probleme waehrend der Aufnahme" in result.output


class FakeStream:
    fail = None
    instances = []

    def __init__(self, **options):
        if FakeStream.fail == "open":
            raise OSError("Error opening InputStream: Invalid device [PaErrorCode -9996]")
        self.callback = options["callback"]
        self.timer = threading.Timer(0, int)
        FakeStream.instances.append(self)

    def start(self):
        if FakeStream.fail == "start":
            raise OSError("Error starting stream: Device unavailable [PaErrorCode -9985]")
        for _ in range(3):
            self.callback(np.full((1600, 2), 0.1, dtype="float32"), 1600, None, types.SimpleNamespace(input_overflow=False))
        self.timer = threading.Timer(0.3, signal.raise_signal, [signal.SIGINT])
        self.timer.start()

    def stop(self):
        self.timer.cancel()

    def close(self):
        pass


@pytest.fixture
def real_recorder(env, monkeypatch):
    FakeStream.fail, FakeStream.instances = None, []
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=FakeStream, WasapiSettings=lambda **kw: kw))
    monkeypatch.setattr(pipeline, "Recorder", REAL_RECORDER)
    monkeypatch.setattr(pipeline.AudioDevices, "sounddevice_picker", staticmethod(lambda: pipeline.InputPicker([{**LAPTOP, "index": 0}], ["MME"])))
    return env


@pytest.mark.parametrize("stage", ["open", "start"])
def test_start_stream_open_failure_aborts_cleanly_and_a_later_start_works(real_recorder, stage):
    env = real_recorder
    FakeStream.fail = stage
    result = start("--no-diarize", "--no-open")
    assert result.exit_code == 1 and "Traceback" not in result.output
    assert "Datenschutz" in result.output and "Mikrofon" in result.output and "belegt" in result.output
    assert "Aufnahme laeuft" not in result.output and "Aufnahme gespeichert" not in result.output
    assert not list((env["tmp"] / "data" / "aufnahmen").glob("*")) and not reports(env)
    FakeStream.fail = None
    again = start("--no-diarize", "--no-open")
    assert again.exit_code == 0, again.output
    assert again.output.index("Aufnahme laeuft") < again.output.index("Aufnahme gespeichert")
    assert len(recordings(env)) == 1 and sf.info(recordings(env)[0]).frames == 4800 and len(reports(env)) == 1


def test_record_stream_open_failure_leaves_no_file(real_recorder):
    FakeStream.fail = "start"
    out = real_recorder["tmp"] / "r.flac"
    result = invoke("record", "--out", out, "--seconds", "1")
    assert result.exit_code == 1 and "Datenschutz" in result.output and "Aufnahme laeuft" not in result.output and not out.exists()
    FakeStream.fail = None
    assert invoke("record", "--out", out, "--seconds", "5").exit_code == 0 and out.exists()
