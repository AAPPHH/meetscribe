import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import click
import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from meetscribe import backends, cli, pipeline
from meetscribe.store import HotFile, Query, Store, make_config

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
NPU_DIR, INT8_DIR = "whisper-large-v3-turbo-fp16-ov", "whisper-large-v3-turbo-int8-ov"
HARDWARE_ROWS = [
    {"id": "cpu", "device": "CPU", "name": "Test CPU", "type": "integrated", "cuda": None, "backends": ["openvino", "faster-whisper"]},
    {"id": "igpu", "device": "GPU", "name": "Test Arc", "type": "integrated", "cuda": None, "backends": ["openvino"]},
    {"id": "npu", "device": "NPU", "name": "Test NPU", "type": "integrated", "cuda": None, "backends": ["openvino"]},
]


class FakeEngine(backends.OpenVinoEngine):
    fail_devices = ()
    constructed = []
    silent = False

    def __init__(self, step, models, language):
        self.device, self.model = step.label, "fake-model"
        self.hardware = step.hardware
        FakeEngine.constructed.append(step.hardware)

    def transcribe(self, audio):
        if self.hardware in FakeEngine.fail_devices:
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
    for name in (NPU_DIR, INT8_DIR):
        (models / name).mkdir(parents=True, exist_ok=True)
        (models / name / "openvino_model.xml").write_text("x")
        (models / name / "openvino_model.bin").write_bytes(b"x")
    monkeypatch.setenv("MEETSCRIBE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("MEETSCRIBE_MODELS", str(models))
    for name in ("MEETSCRIBE_DEVICE", "MEETSCRIBE_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    for name in (*OFFLINE, "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    FakeEngine.fail_devices, FakeEngine.constructed, FakeEngine.silent, FakeDiarizer.calls = (), [], False, []
    loaded = []
    monkeypatch.setitem(backends.ENGINES, "openvino", FakeEngine)
    monkeypatch.setattr(backends.Hardware, "detect", classmethod(lambda cls: [dict(r, backends=list(r["backends"])) for r in HARDWARE_ROWS]))
    monkeypatch.setattr(backends.Catalog, "installed", staticmethod(lambda backend: backend == "openvino"))
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
    assert "m1: 2 Segmente, 2 Sprecher" in result.output and "angefordert: Automatisch, genutzt: NPU" in result.output
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


def clear_models(env, *names):
    for name in names:
        for path in (env["models"] / name).iterdir():
            path.unlink()


def test_missing_whisper_model_points_to_setup(env):
    clear_models(env, NPU_DIR, INT8_DIR)
    result = transcribe(env, "--no-diarize")
    assert result.exit_code != 0 and "meetscribe setup --model large-v3-turbo:fp16 --backend openvino" in result.output and env["loaded"] == []


def test_auto_skips_hardware_without_model_and_says_so(env):
    clear_models(env, NPU_DIR)
    result = transcribe(env, "--no-diarize")
    assert result.exit_code == 0, result.output
    assert "NPU mit OpenVINO übersprungen" in result.output and "genutzt: iGPU" in result.output
    assert FakeEngine.constructed == ["igpu"]
    explicit = transcribe(env, "--no-diarize", "--device", "npu", meeting_id="m2")
    assert explicit.exit_code != 0 and "fehlt" in explicit.output and FakeEngine.constructed == ["igpu"]


def test_fallback_reported_and_per_row_device(env):
    FakeEngine.fail_devices = ("npu",)
    result = transcribe(env, "--no-diarize")
    assert result.exit_code == 0, result.output
    assert result.output.count("NPU fehlgeschlagen") == 1
    assert "angefordert: Automatisch, genutzt: iGPU" in result.output
    assert invoke("sql", "SELECT DISTINCT device FROM segments").output.split()[-1] == "iGPU"


def test_no_fallback_flag_fails(env):
    FakeEngine.fail_devices = ("npu",)
    result = transcribe(env, "--no-diarize", "--no-fallback")
    assert result.exit_code != 0 and "--no-fallback" in result.output
    assert FakeEngine.constructed == ["npu"]
    assert not env["store"].hot_path("m1").exists()


def test_all_devices_failing_exits_nonzero_without_eos(env):
    FakeEngine.fail_devices = ("npu", "igpu", "cpu")
    result = transcribe(env, "--no-diarize")
    assert result.exit_code != 0 and "Alle Geräte fehlgeschlagen" in result.output
    assert not env["store"].hot_path("m1").exists()


def test_device_from_env_and_flag(env):
    result = invoke("transcribe", env["audio"], "--no-diarize", "--meeting-id", "b", env={"MEETSCRIBE_DEVICE": "gpu"})
    assert "angefordert: GPU" in result.output and FakeEngine.constructed[-1] == "igpu"
    assert result.output.count("keine dedizierte Grafikkarte") == 1
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
    monkeypatch.setattr(pipeline.ModelSetup, "warm", lambda self, step: None)
    result = invoke("setup")
    assert result.exit_code == 0, result.output
    assert set(seen) == {"OpenVINO/whisper-large-v3-turbo-fp16-ov", "OpenVINO/whisper-large-v3-turbo-int8-ov"}
    assert seen["OpenVINO/whisper-large-v3-turbo-fp16-ov"] == (None, {name: None for name in OFFLINE})
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
    clear_models(env, NPU_DIR, INT8_DIR)
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
    FakeEngine.fail_devices = ("npu", "igpu", "cpu")
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
    FakeEngine.fail_devices = ("npu", "igpu", "cpu")
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
    FakeEngine.fail_devices = ("npu", "igpu", "cpu")
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


def assert_saved_and_recoverable(env, result, code):
    meeting = f"{date.today():%Y%m%d}-wochenplanung"
    assert result.exit_code == code and "Traceback" not in result.output and "Aborted" not in result.output
    assert f"Audio erhalten: {recordings(env)[0]}" in result.output
    assert f"Besprechung gespeichert: {meeting}" in result.output and f"meetscribe report {meeting}" in result.output
    assert env["store"].exists(meeting) and invoke("report", meeting).exit_code == 0


def test_start_report_write_oserror_keeps_meeting_and_explains_recovery(env):
    blocker = env["tmp"] / "protokolle"
    blocker.write_text("im Weg")
    result = start("--no-diarize", "--no-open")
    assert_saved_and_recoverable(env, result, 1)
    assert "Nachbearbeitung fehlgeschlagen" in result.output and not reports(env)


def test_start_speaker_name_oserror_keeps_meeting_and_explains_recovery(env, monkeypatch):
    def locked(self, meeting_id, label, name):
        raise PermissionError("speakers.parquet gesperrt")

    add_pyannote(env)
    monkeypatch.setattr(cli, "stdin_is_tty", lambda: True)
    monkeypatch.setattr(type(env["store"].speakers), "set", locked)
    result = start("--speakers", "2", "--no-open", input="Anna\n")
    assert_saved_and_recoverable(env, result, 1)
    assert "PermissionError: speakers.parquet gesperrt" in result.output


@pytest.mark.parametrize("stage", ["write_report", "rotate"])
def test_start_ctrl_c_after_transcription_exits_130_and_keeps_meeting(env, monkeypatch, stage):
    def interrupted(self, *args):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline.StartFlow, stage, interrupted)
    result = start("--no-diarize", "--no-open")
    assert_saved_and_recoverable(env, result, 130)
    assert "Nachbearbeitung abgebrochen" in result.output


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
    FakeEngine.fail_devices = ("npu", "igpu", "cpu")
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


@pytest.mark.parametrize("statement", [
    "INSTALL tpch FROM '{url}'", "FORCE INSTALL tpch FROM '{url}'", "LOAD httpfs", "SELECT 1; LOAD httpfs", "UPDATE EXTENSIONS",
])
def test_sql_refuses_extension_statements_without_any_request(env, statement):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_error(404)

        log_message = lambda *args: None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = invoke("sql", statement.format(url=f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()
        server.server_close()
    assert result.exit_code == 1 and "Traceback" not in result.output and "gesperrt" in result.output and requests == []
    assert invoke("sql", "SELECT 1 AS n").exit_code == 0


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
    assert result.exit_code != 0 and "unvollständig" in result.output and "Sprecher trennen" in result.output and "meetscribe setup" in result.output and "segmentation" in result.output and "--no-diarize" in result.output
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


def test_search_help_does_not_promise_full_text_search():
    text = invoke("search", "--help").output
    assert "Volltext" not in text and "Teilstring" in text


def parsed(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_json_switch_covers_read_commands(env):
    add_pyannote(env)
    assert transcribe(env, "--speakers", "2").exit_code == 0
    assert invoke("speakers", "set", "SPEAKER_00", "Anna").exit_code == 0
    meetings = parsed(invoke("--json", "meetings"))
    assert [m["meeting_id"] for m in meetings] == ["m1"] and meetings[0]["segments"] == 2 and meetings[0]["meeting_date"] == "2026-03-04"
    hits = parsed(invoke("--json", "search", "münchen"))
    assert len(hits) == 1 and hits[0]["display_name"] == "Anna" and hits[0]["speaker"] == "SPEAKER_00" and hits[0]["text"].startswith("Grüße")
    stats = parsed(invoke("--json", "stats"))
    assert {r["speaker"] for r in stats} == {"Anna", "SPEAKER_01"} and all(isinstance(r["share"], float) and r["labels"] for r in stats)
    shown = parsed(invoke("--json", "speakers", "show"))
    assert {r["label"]: r["name"] for r in shown} == {"SPEAKER_00": "Anna", "SPEAKER_01": "SPEAKER_01"}
    document = parsed(invoke("--json", "report"))
    assert document["meeting_id"] == "m1" and document["path"] is None and "Anna" in document["markdown"]
    target = env["tmp"] / "out" / "p.md"
    written = parsed(invoke("--json", "report", "m1", "--out", target))
    assert written["path"] == str(target) and target.read_text(encoding="utf-8") == written["markdown"]
    table = parsed(invoke("--json", "sql", "SELECT 1 AS a, 2 AS a, 'ü' AS b, 'nan'::DOUBLE AS c, DATE '2026-01-02' AS d"))
    assert table == {"columns": ["a", "a", "b", "c", "d"], "rows": [[1, 2, "ü", None, "2026-01-02"]]}
    assert parsed(invoke("--json", "meetings")) == meetings
    assert parsed(invoke("--json", "search", "gibtesnicht")) == []


def test_json_output_is_ascii_and_default_output_unchanged(env):
    assert transcribe(env, "--no-diarize").exit_code == 0
    raw = invoke("--json", "search", "München").stdout
    assert raw.isascii() and "\u00fc" not in raw and "M\u00fcnchen" in json.loads(raw)[0]["text"]
    human = invoke("search", "München").output
    assert human.startswith("meeting_id") and "München" in human and not human.startswith("[")
    assert invoke("meetings").output.startswith("meeting_id")
    assert invoke("report").output.startswith("#")


def test_json_devices_lists_inputs_and_openvino(env, monkeypatch):
    monkeypatch.setattr(pipeline.AudioDevices, "openvino_devices", staticmethod(lambda: [{"device": "NPU", "name": "Intel NPU"}]))
    data = parsed(invoke("--json", "devices"))
    assert data["openvino"] == [{"device": "NPU", "name": "Intel NPU"}] and data["inputs_error"] is None
    assert [d["name"] for d in data["inputs"]] == [LAPTOP["name"], DJI["name"]]
    assert data["inputs"][0] == {"index": 0, "name": LAPTOP["name"], "api": "MME", "channels": 2, "rate": 16000}
    human = invoke("devices").output
    assert "kanaele" in human and "Test NPU" in human and "faster-whisper" in human

    def broken():
        raise OSError("kein Treiber")

    monkeypatch.setattr(pipeline.AudioDevices, "sounddevice_picker", staticmethod(broken))
    data = parsed(invoke("--json", "devices"))
    assert data["inputs"] == [] and "kein Treiber" in data["inputs_error"] and data["openvino"]


def test_check_reports_missing_pieces_with_fixes(env, monkeypatch):
    rows = {r["id"]: r for r in parsed(invoke("--json", "check"))}
    assert rows["ffmpeg"]["ok"] and rows["pyannote"]["ok"] is False and "meetscribe setup" in rows["pyannote"]["fix"]
    assert "„Sprecher trennen“ ausschalten" in rows["pyannote"]["fix"] and "ausführen" in rows["pyannote"]["fix"] and "ausfuehren" not in rows["pyannote"]["fix"]
    assert rows["hf_token"]["ok"] is False and "HF_TOKEN" in rows["hf_token"]["fix"]
    assert all(r["ok"] and r["fix"] is None for k, r in rows.items() if k.startswith("whisper:"))
    human = invoke("check")
    assert human.exit_code == 1 and "NEIN" in human.output
    add_pyannote(env)
    (env["models"] / NPU_DIR / "openvino_model.xml").unlink()
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    rows = {r["id"]: r for r in parsed(invoke("--json", "check"))}
    assert rows["pyannote"]["ok"] and rows["hf_token"]["ok"] and not rows["ffmpeg"]["ok"] and "PATH" in rows["ffmpeg"]["fix"]
    assert rows[f"whisper:{NPU_DIR}"]["ok"] is False


def events_of(result):
    return [json.loads(line) for line in result.stdout.splitlines()]


def names(events):
    return [e["event"] for e in events]


class StdinFeed:
    def __init__(self, monkeypatch, payload, close):
        self.read_end, self.write_end = os.pipe()
        self.payload, self.close = payload, close
        self.fired = []
        monkeypatch.setitem(pipeline.START, "stdin_fd", self.read_end)
        FakeRecorder.interrupt_with = "none"
        self.watchdog = threading.Timer(8, lambda: (self.fired.append(True), signal.raise_signal(signal.SIGINT)))
        self.writer = threading.Thread(target=self.feed, daemon=True)

    def feed(self):
        deadline = time.monotonic() + 8
        while not FakeRecorder.started and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.3)
        if self.payload:
            os.write(self.write_end, self.payload)
        if self.close:
            os.close(self.write_end)

    def __enter__(self):
        self.watchdog.start()
        self.writer.start()
        return self

    def __exit__(self, *exc):
        self.watchdog.cancel()
        self.writer.join()
        for fd in (self.read_end, *([] if self.close else [self.write_end])):
            os.close(fd)
        assert not self.fired
        return False


def test_start_events_with_audiofile_emit_stage_events_without_prompts(env, monkeypatch):
    add_pyannote(env)
    opened = []
    monkeypatch.setattr(cli, "stdin_is_tty", lambda: True)
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)
    result = start("--events", "--speakers", "2", env["audio"], input="Anna\n\n")
    assert result.exit_code == 0, result.output
    events = events_of(result)
    assert names(events) == ["preflight", "transcribing", "done"]
    assert events[1]["audio"] == str(env["audio"])
    assert events[1]["meeting_id"] == events[2]["meeting_id"]
    done = events[2]
    assert done["meeting_id"].endswith("-wochenplanung")
    assert done["report"] == str(reports(env)[0]) and done["audio"] == str(env["audio"]) and done["segments"] == 2 and done["speakers"] == 2
    assert done["labels"] == [{"label": "SPEAKER_00", "sample": "Grüße aus München"}, {"label": "SPEAKER_01", "sample": "Schön dass Sie da"}]
    assert opened == [] and "Name fuer" not in result.output
    assert "Anna" not in reports(env)[0].read_text(encoding="utf-8")


def test_start_events_recording_event_names_input_and_path(env):
    result = start("--events", "--no-diarize")
    assert result.exit_code == 0, result.output
    events = events_of(result)
    assert names(events) == ["preflight", "recording", "transcribing", "done"]
    assert events[1]["input"] == DJI["name"] and events[1]["path"] == str(recordings(env)[0])
    assert events[2]["audio"] == str(recordings(env)[0]) and events[3]["audio"] == str(recordings(env)[0]) and events[3]["labels"] == []
    assert all(line.isascii() for line in result.stdout.splitlines())
    both = events_of(start("--events", "--no-diarize", "--mic", "both"))
    assert both[1]["input"] == f"{DJI['name']} + {LAPTOP['name']}" and both[1]["path"].endswith(".mix.flac")


def test_start_stdin_stop_ends_recording_on_line(env, monkeypatch):
    with StdinFeed(monkeypatch, b"stop\n", close=False):
        result = start("--events", "--stdin-stop", "--no-diarize")
    assert result.exit_code == 0, result.output
    assert names(events_of(result)) == ["preflight", "recording", "transcribing", "done"]
    assert len(recordings(env)) == 1 and len(reports(env)) == 1


def test_start_stdin_stop_ends_recording_when_stdin_closes_and_keeps_audio(env, monkeypatch):
    with StdinFeed(monkeypatch, b"", close=True):
        result = start("--events", "--stdin-stop", "--no-diarize")
    assert result.exit_code == 0, result.output
    assert names(events_of(result)) == ["preflight", "recording", "transcribing", "done"]
    assert len(recordings(env)) == 1 and sf.info(recordings(env)[0]).frames > 0


def test_start_stdin_stop_keeps_audio_when_processing_fails(env, monkeypatch):
    FakeEngine.fail_devices = ("npu", "igpu", "cpu")
    with StdinFeed(monkeypatch, b"", close=True):
        result = start("--events", "--stdin-stop", "--no-diarize")
    assert result.exit_code == 1
    error = events_of(result)[-1]
    assert error["event"] == "error" and error["audio"] == str(recordings(env)[0]) and error["meeting_id"] is None and error["exit_code"] == 1
    assert error["retry"].startswith("meetscribe start ") and str(recordings(env)[0]) in error["retry"] and "--events" in error["retry"]
    assert recordings(env)[0].exists() and not reports(env)


def test_start_stdin_stop_works_without_events_and_prints_no_json(env, monkeypatch):
    with StdinFeed(monkeypatch, b"x\n", close=False):
        result = start("--stdin-stop", "--no-diarize", "--no-open")
    assert result.exit_code == 0 and "Aufnahme gespeichert" in result.output and not result.stdout.lstrip().startswith("{")


def test_start_events_preflight_failure_is_an_error_event_without_audio(env):
    result = start("--events")
    assert result.exit_code == 1 and names(events_of(result)) == ["preflight", "error"]
    error = events_of(result)[-1]
    assert "Diarisierungsmodell fehlt" in error["message"] and error["audio"] is None and error["retry"] is None and error["exit_code"] == 1
    assert FakeRecorder.started == [] and not recordings(env)


def test_start_events_post_processing_failure_names_saved_meeting(env, monkeypatch):
    def broken(self, meeting_id):
        raise OSError("gesperrt")

    monkeypatch.setattr(pipeline.StartFlow, "write_report", broken)
    result = start("--events", "--no-diarize", env["audio"])
    assert result.exit_code == 1 and names(events_of(result))[-1] == "error"
    error = events_of(result)[-1]
    assert error["meeting_id"] and error["retry"] == f"meetscribe report {error['meeting_id']}" and error["audio"] == str(env["audio"])


def test_start_events_warnings_for_fallback_and_input_problems(env):
    FakeEngine.fail_devices = ("npu",)
    FakeRecorder.problem_text = "kein Audio seit 6 s (Geraet getrennt oder blockiert)"
    FakeRecorder.overruns = 2
    result = start("--events", "--no-diarize")
    assert result.exit_code == 0, result.output
    warnings = [e["message"] for e in events_of(result) if e["event"] == "warning"]
    assert any(m.startswith("Eingang") for m in warnings) and any(m.startswith("NPU fehlgeschlagen") for m in warnings)
    assert any(m.startswith("Aufnahme unvollstaendig") for m in warnings) and any(m.startswith("Ueberlaeufe: 2") for m in warnings)
    assert all(not m.startswith("Warnung") for m in warnings)


def test_start_events_keep_stdout_clean_of_foreign_prints(env, monkeypatch):
    original = pipeline.StartFlow.process

    def noisy(self, audio):
        print("Fremdausgabe einer Bibliothek")
        return original(self, audio)

    monkeypatch.setattr(pipeline.StartFlow, "process", noisy)
    result = start("--events", "--no-diarize", env["audio"])
    assert result.exit_code == 0 and all(line.startswith("{") for line in result.stdout.splitlines())


def test_values_reach_the_engine_unexpanded_on_the_real_entry_point(env, tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "a.txt").write_text("x")
    (cwd / "b.txt").write_text("x")
    run_env = {**os.environ, "MEETSCRIBE_DATA": str(tmp_path / "data"), "USERNAME": "Anna", "PYTHONUTF8": "1"}

    def entry(*args):
        return subprocess.run([sys.executable, "-c", "from meetscribe.cli import main; main()", *args], cwd=cwd, env=run_env, capture_output=True, text=True)

    star = entry("--json", "search", "--", "*")
    assert (star.returncode, json.loads(star.stdout)) == (0, [])
    literal = entry("--json", "sql", "SELECT '%USERNAME%' AS a, '~' AS b")
    assert json.loads(literal.stdout)["rows"] == [["%USERNAME%", "~"]]


@pytest.mark.parametrize("target", ["missing.wav", "."])
def test_start_events_reports_unusable_audio_path_in_german(env, target):
    path = env["tmp"] / target
    result = invoke("start", "--events", "--no-diarize", path)
    assert result.exit_code == 1 and names(events_of(result)) == ["preflight", "error"]
    error = events_of(result)[-1]
    assert error["message"].startswith("Audiodatei nicht gefunden:") and error["audio"] is None
    assert "Usage" not in result.output and not recordings(env) and FakeRecorder.started == []


def test_start_rejects_overlong_title_before_recording(env):
    result = invoke("start", "--events", "--no-diarize", "--title", "x" * 400)
    assert result.exit_code == 1 and names(events_of(result)) == ["preflight", "error"]
    assert "Titel ist zu lang" in events_of(result)[-1]["message"] and FakeRecorder.started == [] and not recordings(env)
    assert invoke("start", "--events", "--no-diarize", "--title", "x" * 80, env["audio"]).exit_code == 0


class DeadPipe:
    def write(self, data):
        raise OSError(22, "Invalid argument")

    def flush(self):
        raise OSError(22, "Invalid argument")


def slow_job(env, monkeypatch, seconds=6):
    original = pipeline.TranscribeJob.run

    def run(self):
        with self.store.writer(self.requested_id, title="x", meeting_date="2026-03-04", language="de", discard_on_error=True) as writer:
            writer.append([{"start_s": 0.0, "end_s": 1.0, "speaker": "S", "text": "Anfang"}])
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                time.sleep(0.01)
        return original(self)

    monkeypatch.setattr(pipeline.TranscribeJob, "run", run)


def test_dead_event_pipe_aborts_before_processing_and_keeps_the_audio(env):
    flow = pipeline.StartFlow(env["store"], audio=env["audio"], title="Rohr", diarize=False, reporter=pipeline.EventReporter(DeadPipe()))
    with pytest.raises(click.exceptions.Exit):
        flow.run()
    assert flow.reporter.broken and env["store"].incomplete() == [] and Query(env["store"]).meetings() == []
    assert env["audio"].exists() and not reports(env) and not env["loaded"]


def test_gui_killed_during_recording_saves_the_audio_and_does_not_process_it(env, monkeypatch):
    flow = pipeline.StartFlow(env["store"], mic="dji", diarize=False, reporter=pipeline.EventReporter(DeadPipe()), stdin_stop=True)
    with StdinFeed(monkeypatch, b"", close=True), pytest.raises(click.exceptions.Exit):
        flow.run()
    assert len(recordings(env)) == 1 and sf.info(recordings(env)[0]).frames > 0
    assert Query(env["store"]).meetings() == [] and env["store"].incomplete() == [] and not reports(env) and not env["loaded"]


def test_event_pipe_dying_during_processing_aborts_and_removes_the_partial_meeting(env, monkeypatch):
    monkeypatch.setitem(pipeline.START, "heartbeat_s", 0.05)
    slow_job(env, monkeypatch)

    class Pipe:
        lines = []

        def write(self, data):
            if '"alive"' in data:
                raise OSError(22, "Invalid argument")
            Pipe.lines.append(data)

        def flush(self):
            pass

    flow = pipeline.StartFlow(env["store"], audio=env["audio"], title="Waise", diarize=False, reporter=pipeline.EventReporter(Pipe()))
    started = time.monotonic()
    with pytest.raises(click.exceptions.Exit) as caught:
        flow.run()
    assert caught.value.exit_code == 130 and time.monotonic() - started < 4
    assert env["store"].incomplete() == [] and Query(env["store"]).meetings() == [] and not reports(env) and env["audio"].exists()
    assert any('"transcribing"' in line for line in Pipe.lines)


def test_second_start_is_refused_while_another_one_holds_the_lock(env):
    with pipeline.RunLock(env["store"].config["data_dir"] / pipeline.START["lock_file"]):
        result = invoke("start", "--events", "--no-diarize", env["audio"])
    assert result.exit_code == 1 and names(events_of(result)) == ["error"]
    assert "läuft bereits" in events_of(result)[0]["message"] and env["store"].incomplete() == [] and not reports(env)
    assert invoke("start", "--events", "--no-diarize", env["audio"]).exit_code == 0
    assert len(Query(env["store"]).meetings()) == 1


def test_error_event_carries_unclipped_detail_and_done_counts_stored_speakers(env, monkeypatch):
    long_path = "C:\\" + "ordner\\" * 40 + "x.wav"

    def broken(path, rate=None):
        raise RuntimeError(f"Audio konnte nicht dekodiert werden ({long_path})\nUrsache: kaputt")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline.AudioLoader, "load", staticmethod(broken))
        failed = events_of(invoke("start", "--events", "--no-diarize", env["audio"]))[-1]
    assert long_path in failed["detail"] and "Ursache: kaputt" in failed["detail"] and long_path in failed["message"]
    done = events_of(invoke("start", "--events", "--no-diarize", env["audio"]))[-1]
    assert done["event"] == "done" and done["speakers"] == 1 and done["labels"] == []


def truncated_meeting(env, meeting_id="m-teil"):
    with pytest.raises(KeyError):
        with env["store"].writer(meeting_id, title="Teil", meeting_date="2026-03-04", language="de") as writer:
            writer.append([{"start_s": 0.0, "end_s": 1.0, "speaker": "UNKNOWN", "text": "halb"}])
            raise KeyError
    return meeting_id


def test_truncated_meeting_is_flagged_in_listing_and_report_and_can_be_discarded(env):
    assert transcribe(env, "--no-diarize").exit_code == 0
    truncated_meeting(env)
    flags = {m["meeting_id"]: m["complete"] for m in parsed(invoke("--json", "meetings"))}
    assert flags == {"m1": True, "m-teil": False}
    assert "NEIN" in invoke("meetings").output
    assert "unvollständig" in invoke("report", "m-teil").output and "unvollständig" not in invoke("report", "m1").output
    refused = invoke("discard", "m1")
    assert refused.exit_code == 1 and "vollstaendig" in refused.output and env["store"].exists("m1")
    assert invoke("discard", "m-teil").exit_code == 0 and not env["store"].exists("m-teil")
    assert invoke("discard", "m-teil").exit_code == 1


def test_whisper_model_needs_nonempty_graph_and_weights(tmp_path):
    folder = tmp_path / "w"
    folder.mkdir()
    assert not backends.OpenVinoEngine.ready(folder)
    (folder / "leer.xml").write_text("")
    assert not backends.OpenVinoEngine.ready(folder)
    (folder / "leer.xml").write_text("<net/>")
    assert not backends.OpenVinoEngine.ready(folder)
    (folder / "leer.bin").write_bytes(b"")
    assert not backends.OpenVinoEngine.ready(folder)
    (folder / "leer.bin").write_bytes(b"w")
    assert backends.OpenVinoEngine.ready(folder)


def test_check_marks_speaker_rows_as_diarize_scope(env):
    rows = {r["id"]: r for r in parsed(invoke("--json", "check"))}
    assert rows["pyannote"]["scope"] == rows["hf_token"]["scope"] == "diarize"
    assert rows["ffmpeg"]["scope"] is None


def test_start_events_carry_the_plan_and_the_real_hardware(env):
    events = events_of(start("--events", "--no-diarize", "--device", "igpu", env["audio"]))
    assert names(events) == ["preflight", "transcribing", "done"]
    assert {k: events[1][k] for k in ("backend", "device", "model", "variant")} == {"backend": "openvino", "device": "iGPU", "model": "large-v3-turbo", "variant": "int8"}
    ran = events[2]["ran"]
    assert ran["devices"] == ["iGPU"] and ran["backend"] == "openvino" and ran["requested"] == "iGPU" and ran["fallback"] is False
    assert ran["text"] == "iGPU · large-v3-turbo · OpenVINO" and ran["stored"] == ["fake-model"]


def test_start_events_show_a_fallback_as_warning_and_in_the_result(env):
    FakeEngine.fail_devices = ("npu",)
    events = events_of(start("--events", "--no-diarize", env["audio"]))
    warning = next(e for e in events if e["event"] == "warning")
    assert "NPU fehlgeschlagen" in warning["message"] and "Rückfall auf iGPU" in warning["message"]
    ran = events[-1]["ran"]
    assert ran["fallback"] is True and ran["devices"] == ["iGPU"] and ran["requested"] == "Automatisch"
    assert ran["text"].startswith("iGPU · ")


def test_start_refuses_impossible_choices_before_recording(env, monkeypatch):
    uninstalled = start("--no-diarize", "--backend", "faster-whisper", "--device", "cpu")
    assert uninstalled.exit_code != 0 and "faster-whisper ist nicht installiert. Installieren mit: uv sync --extra faster-whisper" in uninstalled.output
    monkeypatch.setattr(backends.Catalog, "installed", staticmethod(lambda backend: True))
    cases = (
        (("--device", "npu", "--model", "large-v3-turbo:int8"), "int8 läuft auf der NPU nicht"),
        (("--backend", "faster-whisper", "--device", "gpu"), "Keine NVIDIA-Karte (CUDA) gefunden"),
        (("--backend", "faster-whisper", "--device", "npu"), "nur auf der CPU und auf NVIDIA-Karten"),
        (("--model", "riesig"), "Unbekanntes Modell 'riesig'"),
    )
    for flags, reason in cases:
        result = start("--no-diarize", *flags)
        assert result.exit_code != 0 and reason in result.output, (flags, result.output)
    refused = events_of(start("--events", "--no-diarize", "--backend", "faster-whisper", "--device", "gpu"))
    assert names(refused) == ["preflight", "error"] and "NVIDIA" in refused[1]["message"]
    assert FakeRecorder.started == [] and not recordings(env) and FakeEngine.constructed == []


def test_start_no_fallback_flag_and_choices_reach_the_retry_command(env):
    FakeEngine.fail_devices = ("npu", "igpu", "cpu")
    events = events_of(start("--events", "--no-diarize", "--device", "igpu", "--model", "large-v3-turbo:int8", "--no-fallback", env["audio"]))
    error = events[-1]
    assert error["event"] == "error" and FakeEngine.constructed == ["igpu"]
    assert "--device igpu" in error["retry"] and "--model large-v3-turbo:int8" in error["retry"] and "--no-fallback" in error["retry"]
    assert "--backend" not in error["retry"]


def test_legacy_gpu_choice_keeps_working_from_env_and_flag(env):
    result = start("--no-diarize", "--no-open", env["audio"], env={"MEETSCRIBE_DEVICE": "gpu"})
    assert result.exit_code == 0, result.output
    assert result.output.count("keine dedizierte Grafikkarte") == 1 and FakeEngine.constructed == ["igpu"]
    again = start("--no-diarize", "--no-open", "--device", "gpu", env["audio"])
    assert again.output.count("keine dedizierte Grafikkarte") == 1


def test_backend_choice_from_env_and_the_default_backend_is_stored(env):
    result = invoke("transcribe", env["audio"], "--no-diarize", "--meeting-id", "m1", env={"MEETSCRIBE_BACKEND": "openvino"})
    assert result.exit_code == 0, result.output
    assert invoke("sql", "SELECT DISTINCT backend FROM segments").output.split()[-1] == "openvino"
    assert "openvino" in parsed(invoke("--json", "sql", "SELECT backend FROM segments"))["rows"][0]
    assert "**Backend:** openvino" in invoke("report").output


def test_models_command_lists_catalog_with_state_size_and_fit(env):
    text = invoke("models")
    assert text.exit_code == 0, text.output
    lines = {tuple(line.split()[:3]): line for line in text.output.splitlines() if line.split()[:1] and line.split()[0] in backends.CATALOG["models"]}
    turbo = lines[("large-v3-turbo", "openvino", "fp16")].split()
    assert turbo[3:5] == ["1628", "MB"] and turbo[5] == "ja"
    assert "ungeprüft" in lines[("small", "openvino", "fp16")] and "nein" in lines[("small", "faster-whisper", "ct2")]
    data = parsed(invoke("models", "--json"))
    assert data["default_model"] == "large-v3-turbo" and len(data["models"]) == 18 and len(data["choices"]) == 105
    assert [b["id"] for b in data["backends"]] == ["openvino", "faster-whisper"] and data["backends"][1]["installed"] is False
    assert [h["id"] for h in data["hardware"]] == ["npu", "gpu", "igpu", "cpu"]
    assert data["hardware"][1]["present"] is False and "dedizierte" in data["hardware"][1]["reason"] and data["hardware"][0]["present"] is True
    assert parsed(invoke("--json", "models"))["default_model"] == "large-v3-turbo"
    auto = next(c for c in data["choices"] if (c["backend"], c["hardware"], c["model"]) == ("auto", "auto", "auto"))
    assert auto["ok"] and auto["resolved"]["device"] == "NPU" and auto["installed"] is True


def test_devices_json_reports_hardware_with_type_and_serving_backends(env, monkeypatch):
    monkeypatch.setattr(pipeline.AudioDevices, "openvino_devices", staticmethod(lambda: []))
    data = parsed(invoke("devices", "--json"))
    by = {r["device"]: r for r in data["hardware"]}
    assert by["GPU"]["id"] == "igpu" and by["GPU"]["type"] == "integrated" and by["CPU"]["backends"] == ["openvino", "faster-whisper"]
    assert [k["present"] for k in data["kinds"]] == [True, False, True, True]
    assert data["backends"][0]["installed"] is True and "uv sync --extra faster-whisper" in data["backends"][1]["reason"]
    assert parsed(invoke("--json", "devices"))["hardware"] == data["hardware"]


def fake_download(seen):
    def download(repo, target, token=None):
        seen.append(repo)
        target.mkdir(parents=True, exist_ok=True)
        for name in ("openvino_model.xml", "openvino_model.bin"):
            (target / name).write_bytes(b"x" * 10)

    return staticmethod(download)


def test_setup_model_downloads_only_that_model(env, monkeypatch):
    seen, warmed = [], []
    monkeypatch.setattr(pipeline.ModelSetup, "download", fake_download(seen))
    monkeypatch.setattr(pipeline.ModelSetup, "warm", lambda self, step: warmed.append(step))
    result = invoke("setup", "--model", "small", "--device", "cpu")
    assert result.exit_code == 0, result.output
    assert seen == ["OpenVINO/whisper-small-int8-ov"] and warmed == [] and "pyannote" not in result.output
    assert (env["models"] / "whisper-small-int8-ov" / "openvino_model.xml").is_file()
    assert invoke("setup", "--model", "riesig").exit_code != 0


def test_setup_events_stream_progress_then_done_or_error(env, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.ModelSetup, "download", fake_download(seen))
    done = invoke("setup", "--events", "--model", "tiny:fp16")
    events = events_of(done)
    assert names(events)[-1] == "done" and events[-1]["messages"][0].startswith("whisper OpenVINO/whisper-tiny-fp16-ov: ok")
    assert any(e["event"] == "download" and e["repo"] == "OpenVINO/whisper-tiny-fp16-ov" and e["total"] == 84_000_000 for e in events)
    assert all(line.isascii() for line in done.stdout.splitlines())
    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(lambda repo, target, token=None: (_ for _ in ()).throw(OSError("Netz nicht erreichbar"))))
    failed = invoke("setup", "--events", "--model", "base:fp16")
    last = events_of(failed)[-1]
    assert failed.exit_code == 1 and last["event"] == "error" and "Netz nicht erreichbar" in last["detail"]
    unknown = invoke("setup", "--events", "--model", "riesig")
    assert unknown.exit_code == 1 and "Unbekanntes Modell" in events_of(unknown)[-1]["message"]


def test_setup_without_model_does_what_it_did_before(env, monkeypatch):
    seen, warmed = [], []
    monkeypatch.setattr(pipeline.ModelSetup, "download", fake_download(seen))
    monkeypatch.setattr(pipeline.ModelSetup, "warm", lambda self, step: warmed.append(step.hardware))
    result = invoke("setup")
    assert result.exit_code == 0 and seen == ["OpenVINO/whisper-large-v3-turbo-fp16-ov", "OpenVINO/whisper-large-v3-turbo-int8-ov"]
    assert warmed == ["npu"] and "pyannote" in result.output
