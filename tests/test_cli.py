import os

import numpy as np
import pytest
from click.testing import CliRunner

from meetscribe import cli, pipeline
from meetscribe.store import HotFile, Store, make_config

WORDS = [
    ("Grüße", 0.2, 0.6), ("aus", 0.6, 0.8), ("München", 0.8, 1.2),
    ("Schön", 1.7, 2.0), ("dass", 2.0, 2.2), ("Sie", 2.2, 2.4), ("da", 2.4, 2.6),
]
TURNS = [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}, {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"}]
OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


class FakeEngine:
    fail_devices = ()
    constructed = []

    def __init__(self, device, models, language, override=None):
        self.device, self.model = device, "fake-model"
        FakeEngine.constructed.append(device)

    def transcribe(self, audio):
        if self.device in FakeEngine.fail_devices:
            raise RuntimeError("DEVICE_LOST")
        return {"words": [{"text": t, "start_s": s, "end_s": e} for t, s, e in WORDS], "chunks": []}


class FakeDiarizer:
    calls = []

    def __init__(self, models):
        pass

    def diarize(self, audio, **options):
        FakeDiarizer.calls.append(options)
        return TURNS


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / pipeline.MODELS["npu"]["dir"]).mkdir(parents=True)
    (models / pipeline.MODELS["npu"]["dir"] / "openvino_model.xml").write_text("x")
    monkeypatch.setenv("MEETSCRIBE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("MEETSCRIBE_MODELS", str(models))
    monkeypatch.delenv("MEETSCRIBE_DEVICE", raising=False)
    for name in (*OFFLINE, "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    FakeEngine.fail_devices, FakeEngine.constructed, FakeDiarizer.calls = (), [], []
    loaded = []
    monkeypatch.setattr(pipeline, "WhisperEngine", FakeEngine)
    monkeypatch.setattr(pipeline, "Diarizer", FakeDiarizer)
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
    assert "SPEAKER_01" in invoke("speakers", "show", "m1").output
    assert invoke("speakers", "set", "m1", "SPEAKER_00", "Anna").exit_code == 0
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
    assert invoke("speakers", "unset", "m1", "SPEAKER_00").exit_code == 0
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
    assert not HotFile(env["store"].hot_path("m1")).is_complete()


def test_all_devices_failing_exits_nonzero_without_eos(env):
    FakeEngine.fail_devices = ("npu", "gpu", "cpu")
    result = transcribe(env, "--no-diarize")
    assert result.exit_code != 0 and "Alle Geraete fehlgeschlagen" in result.output
    assert not HotFile(env["store"].hot_path("m1")).is_complete()


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
    for args in (("report", "nope"), ("speakers", "show", "nope"), ("speakers", "set", "m1", "SPEAKER_99", "X"), ("sql", "SELEC 1")):
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
    result = invoke("setup")
    assert result.exit_code == 0, result.output
    assert set(seen) == {pipeline.MODELS["npu"]["repo"]}
    assert seen[pipeline.MODELS["npu"]["repo"]] == (None, {name: None for name in OFFLINE})
    assert "pyannote" in result.output and "fehlt" in result.output


def test_setup_with_token_downloads_pyannote(env, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(lambda repo, target, token=None: calls.append((repo, token))))
    result = invoke("setup", env={"HF_TOKEN": "tok"})
    assert result.exit_code == 0
    assert (pipeline.SETTINGS["diarization"]["repo"], "tok") in calls


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
    FakeEngine.fail_devices = ("npu",)
    transcribe(env, "--no-diarize", "--no-fallback")
    assert "incomplete" in invoke("rotate").output
    assert "rotiert: m1" not in invoke("rotate").output
