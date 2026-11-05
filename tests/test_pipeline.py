import shutil
import signal
import threading

import numpy as np
import pytest
import soundfile as sf

from meetscribe import pipeline
from meetscribe.pipeline import (
    ASRFailure, ASRRunner, Aligner, AudioError, AudioLoader, DeviceError, InputPicker, Interrupt, KeepAwake, Mixer, Models, Naming,
    RecordingSession, TranscribeJob, Transcriber, Windows,
)
from meetscribe.store import HotFile, Query, ReportFile, Store, make_config

RATE = 16000
DEVICES = [
    {"name": "Mikrofonarray (Intel® Smart Sound Technologie)", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100.0},
    {"name": "Mikrofonarray (Intel® Smart Sound Technologie)", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 48000.0},
    {"name": "Lautsprecher", "hostapi": 1, "max_input_channels": 0, "default_samplerate": 48000.0},
    {"name": "DJI MIC MINI", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 48000.0},
    {"name": "DJI MIC MINI", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 48000.0},
    {"name": "Headset", "hostapi": 1, "max_input_channels": 1, "default_samplerate": 16000.0},
]
HOSTAPIS = ["MME", "Windows WASAPI"]


def word(text, start, end):
    return {"text": text, "start_s": start, "end_s": end}


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def noise(seconds, gaps=(), seed=0):
    audio = (np.random.default_rng(seed).random(int(seconds * RATE)) - 0.5).astype("float32")
    for a, b in gaps:
        audio[int(a * RATE):int(b * RATE)] = 0
    return audio


def small_windows():
    return Windows(window_s=2.0, search_s=0.2, frame_s=0.01, rate=RATE)


class FakeEngine:
    calls = 0

    def __init__(self, device, fail_on=(), fail_construct=False):
        if fail_construct:
            raise RuntimeError("compile failed\nsecond line")
        self.device, self.model, self.fail_on = device, f"fake-{device}", fail_on

    def transcribe(self, audio):
        FakeEngine.calls += 1
        if FakeEngine.calls in self.fail_on:
            raise RuntimeError("ZE_RESULT_ERROR_DEVICE_LOST\nmore")
        return {"words": [word(f"a{FakeEngine.calls}", 0.5, 0.9), word(f"b{FakeEngine.calls}", 1.0, 1.2)], "chunks": []}


@pytest.fixture(autouse=True)
def reset_calls():
    FakeEngine.calls = 0


@pytest.fixture
def store(tmp_path):
    return Store(make_config(tmp_path / "data"))


def run_windows(store, factory, devices=("npu", "gpu", "cpu"), fallback=True, audio=None, turns=()):
    runner = ASRRunner(devices, factory, fallback)
    transcriber = Transcriber(runner, windows=small_windows())
    audio = noise(5.0) if audio is None else audio
    with store.writer("m1", title="t", meeting_date="2026-01-02", language="de") as writer:
        summary = transcriber.run(audio, list(turns), writer)
    return summary, runner


def test_windows_cut_at_quietest_point():
    audio = noise(5.0, gaps=[(2.05, 2.1), (3.85, 3.9)])
    windows = small_windows().split(audio)
    assert len(windows) == 3
    assert windows[0][1] == windows[1][0] and windows[1][1] == windows[2][0]
    assert 2.05 * RATE <= windows[0][1] <= 2.1 * RATE
    assert 3.85 * RATE <= windows[1][1] <= 3.9 * RATE
    assert windows[0][0] == 0 and windows[-1][1] == len(audio)


def test_windows_short_empty_and_tail():
    assert small_windows().split(noise(1.5)) == [(0, int(1.5 * RATE))]
    assert small_windows().split(np.zeros(0, dtype="float32")) == []
    assert len(small_windows().split(noise(4.1))) == 2


def test_windows_cuts_stay_within_search_range():
    windows = small_windows().split(noise(9.0, seed=3))
    for k, (_, end) in enumerate(windows[:-1], start=1):
        assert abs(end - 2.0 * k * RATE) <= 0.2 * RATE + 1


def test_aligner_empty_inputs():
    assert Aligner().align([], [turn(0, 1, "A")]) == []
    segments = Aligner().align([word("hi", 0, 1)], [])
    assert segments[0]["speaker"] == "UNKNOWN" and segments[0]["text"] == "hi"


def test_aligner_max_overlap_and_straddle():
    turns = [turn(0, 2.0, "A"), turn(2.0, 5.0, "B")]
    words = [word("eins", 0.1, 0.9), word("zwei", 1.5, 2.3), word("drei", 1.9, 3.1), word("vier", 3.2, 3.8)]
    segments = Aligner().align(words, turns)
    assert [(s["speaker"], s["text"]) for s in segments] == [("A", "eins zwei"), ("B", "drei vier")]
    assert segments[0]["words"][1]["text"] == "zwei"
    assert segments[0]["start_s"] == 0.1 and segments[0]["end_s"] == 2.3


def test_aligner_gap_without_turn_uses_nearest():
    turns = [turn(0, 1, "A"), turn(10, 11, "B")]
    segments = Aligner(max_gap_s=100).align([word("x", 2, 3), word("y", 8, 9)], turns)
    assert [s["speaker"] for s in segments] == ["A", "B"]


def test_aligner_splits_on_long_gap_and_keeps_words():
    segments = Aligner(max_gap_s=1.0).align([word("a", 0, 1), word("b", 1.5, 2), word("c", 5, 6)], [turn(0, 20, "A")])
    assert [s["text"] for s in segments] == ["a b", "c"]
    assert all(len(s["words"]) == len(s["text"].split()) for s in segments)


def test_aligner_splits_after_sentence_end_once_long_enough():
    words = [word("Eins.", 0, 3), word("Zwei.", 3, 6), word("Drei", 6, 9), word("vier.", 9, 11), word("Fuenf.", 11, 12)]
    segments = Aligner(min_segment_s=8.0).align(words, [])
    assert [s["text"] for s in segments] == ["Eins. Zwei. Drei vier.", "Fuenf."]
    assert [s["text"] for s in Aligner(min_segment_s=1000).align(words, [])] == ["Eins. Zwei. Drei vier. Fuenf."]


def test_aligner_overlapping_turns_sum_per_speaker():
    turns = [turn(0, 3, "A"), turn(1, 4, "B"), turn(2, 4, "B")]
    assert Aligner().align([word("x", 1.0, 4.0)], turns)[0]["speaker"] == "B"


def test_fallback_mid_run_keeps_rows_and_device(store, capfd):
    def factory(device):
        return FakeEngine(device, fail_on=(2,) if device == "npu" else ())

    summary, _ = run_windows(store, factory)
    err = capfd.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "NPU" in err[0] and "DEVICE_LOST" in err[0]
    assert summary == {"segments": 3, "used": ["NPU", "GPU"]}
    assert HotFile(store.hot_path("m1")).is_complete()
    rows = Query(store).execute("SELECT seg_id, start_s, device, model FROM segments ORDER BY seg_id")[1]
    assert [r[2] for r in rows] == ["NPU", "GPU", "GPU"]
    assert [r[3] for r in rows] == ["fake-npu", "fake-gpu", "fake-gpu"]
    starts = [r[1] for r in rows]
    assert starts == sorted(starts) and starts[1] > 2.0 and starts[2] > 4.0


def test_fallback_on_construction_failure(store, capfd):
    summary, _ = run_windows(store, lambda d: FakeEngine(d, fail_construct=d in ("npu", "gpu")))
    assert len(capfd.readouterr().err.strip().splitlines()) == 2
    assert summary["used"] == ["CPU"]
    assert {r[0] for r in Query(store).execute("SELECT device FROM segments")[1]} == {"CPU"}


def test_no_fallback_fails_and_keeps_earlier_rows(store, capfd):
    with pytest.raises(ASRFailure):
        run_windows(store, lambda d: FakeEngine(d, fail_on=(2,)), devices=("npu",), fallback=False)
    assert len(capfd.readouterr().err.strip().splitlines()) == 1
    table, complete = HotFile(store.hot_path("m1")).read()
    assert not complete and table.num_rows == 1


def test_all_devices_fail_leaves_no_eos(store, capfd):
    with pytest.raises(ASRFailure):
        run_windows(store, lambda d: FakeEngine(d, fail_construct=True))
    assert len(capfd.readouterr().err.strip().splitlines()) == 3
    assert not HotFile(store.hot_path("m1")).is_complete()


def test_words_clamped_monotonic_across_windows():
    words = Transcriber.words({"words": [word("a", 1.0, 5.0), word("  ", 1, 2), word("b", 0.2, 0.4)]}, 10.0, 12.0, 10.5)
    assert [w["text"] for w in words] == ["a", "b"]
    assert words[0]["start_s"] == 11.0 and words[0]["end_s"] == 12.0
    assert words[1]["start_s"] == 11.0


def test_words_fall_back_to_chunks():
    assert Transcriber.words({"words": [], "chunks": [word(" Hallo Welt ", 0, 1)]}, 2.0, 9.0, 0.0)[0]["text"] == "Hallo Welt"


def test_silent_window_skipped(store):
    audio = np.concatenate([noise(2.0), np.zeros(2 * RATE, dtype="float32"), noise(1.0)])
    summary, _ = run_windows(store, lambda d: FakeEngine(d), audio=audio)
    assert summary["segments"] == 2 and FakeEngine.calls == 2


@pytest.mark.parametrize("device,fallback,expected", [
    ("npu", True, ("npu", "gpu", "cpu")), ("gpu", True, ("gpu", "cpu")), ("cpu", True, ("cpu",)), ("gpu", False, ("gpu",)),
])
def test_device_chain(tmp_path, device, fallback, expected):
    job = TranscribeJob(None, "a.wav", device.upper(), "de", models=Models(tmp_path), fallback=fallback)
    assert job.chain == expected


def test_model_paths(tmp_path):
    models = Models(tmp_path)
    assert models.whisper_dir("npu") == tmp_path / pipeline.MODELS["npu"]["dir"]
    assert "fp16" in pipeline.MODELS["npu"]["dir"] and "int8" in pipeline.MODELS["gpu"]["dir"] == pipeline.MODELS["cpu"]["dir"]
    assert models.whisper_dir("cpu", "custom") == tmp_path / "custom"
    assert models.cache_dir == tmp_path / "ov_cache"
    assert not models.diarization_ready()


def test_naming_slug_and_suffix(store):
    assert Naming.slug("Wöchentliches Meeting – Straße_1") == "woechentliches-meeting-strasse-1"
    assert Naming.slug("Team Sync!") == "team-sync"
    first = Naming.meeting_id(store, "2026-03-04", "Team Sync")
    assert first == "20260304-team-sync"
    with store.writer(first, title="x", meeting_date="2026-03-04") as writer:
        writer.append([{"start_s": 0, "end_s": 1, "speaker": "A", "text": "hi"}])
    assert Naming.meeting_id(store, "2026-03-04", "Team Sync") == "20260304-team-sync-2"


def test_picker_prefers_wasapi_and_matches_patterns():
    picker = InputPicker(DEVICES, HOSTAPIS)
    assert picker.pick("dji")["index"] == 4
    assert picker.pick("LAPTOP")["index"] == 1
    assert picker.pick("headset")["index"] == 5
    assert InputPicker(DEVICES[:1], HOSTAPIS).pick("laptop")["index"] == 0


def test_picker_missing_lists_existing():
    picker = InputPicker([d for d in DEVICES if "DJI" not in d["name"]], HOSTAPIS)
    with pytest.raises(DeviceError) as info:
        picker.pick("dji")
    assert "Headset" in info.value.message and "Lautsprecher" not in info.value.message


def test_recording_paths(tmp_path):
    picker = InputPicker(DEVICES, HOSTAPIS)
    assert RecordingSession("dji", tmp_path / "m", picker).paths() == {"dji": tmp_path / "m.flac"}
    both = RecordingSession("both", tmp_path / "m.flac", picker).paths()
    assert both == {"dji": tmp_path / "m.dji.flac", "laptop": tmp_path / "m.laptop.flac", "mix": tmp_path / "m.mix.flac"}


def test_mixer_stretches_to_equal_length(tmp_path):
    a, b = tmp_path / "a.flac", tmp_path / "b.flac"

    def tone(n, freq, rate):
        return (0.4 * np.sin(2 * np.pi * freq * np.arange(n) / rate)).astype("float32")

    sf.write(a, np.stack([tone(48000 * 4, 220, 48000)] * 2, axis=1), 48000)
    sf.write(b, tone(44100 * 3, 330, 44100), 44100)
    data, rate = sf.read(Mixer.mix([a, b], tmp_path / "mix.flac"), dtype="float32")
    assert rate == 48000 and data.ndim == 1 and len(data) == 48000 * 4
    assert 0.2 < np.abs(data).max() <= 0.81 and np.abs(data[-100:]).max() > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_audio_loader_decodes_to_mono_16k(tmp_path):
    path = tmp_path / "in.wav"
    sf.write(path, np.zeros((8000, 2), dtype="float32") + 0.1, 8000)
    audio = AudioLoader.load(path)
    assert audio.dtype == np.float32 and audio.ndim == 1 and abs(len(audio) - 16000) < 64


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_audio_loader_decode_failure(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not audio")
    with pytest.raises(AudioError):
        AudioLoader.load(path)


def test_audio_loader_without_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    with pytest.raises(AudioError, match="ffmpeg"):
        AudioLoader.load(tmp_path / "x.wav")


def test_keep_awake_sets_and_releases_flags():
    calls = []
    with KeepAwake(calls.append):
        assert calls == [pipeline.AWAKE_FLAGS["continuous"] | pipeline.AWAKE_FLAGS["system_required"]]
    assert calls[-1] == pipeline.AWAKE_FLAGS["continuous"] and len(calls) == 2


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_keep_awake_released_on_exceptions(error):
    calls = []
    with pytest.raises(error):
        with KeepAwake(calls.append):
            raise error("boom")
    assert calls[-1] == pipeline.AWAKE_FLAGS["continuous"] and len(calls) == 2


def test_keep_awake_is_noop_without_windows(monkeypatch):
    monkeypatch.setattr(pipeline.sys, "platform", "linux")
    assert KeepAwake.native() is None
    with KeepAwake():
        pass


def test_interrupt_sets_flag_and_restores_handler():
    before = signal.getsignal(signal.SIGINT)
    with Interrupt() as interrupt:
        assert not interrupt.wait(0)
        threading.Timer(0.02, signal.raise_signal, [signal.SIGINT]).start()
        assert any(interrupt.wait(0.05) for _ in range(40))
    assert signal.getsignal(signal.SIGINT) is before


def test_interrupt_outside_main_thread_is_tolerated():
    seen = []
    thread = threading.Thread(target=lambda: seen.append(Interrupt().__enter__().installed))
    thread.start()
    thread.join()
    assert seen == [False]


def test_auto_picks_dji_else_laptop():
    assert InputPicker(DEVICES, HOSTAPIS).auto() == "dji"
    assert InputPicker([d for d in DEVICES if "DJI" not in d["name"]], HOSTAPIS).auto() == "laptop"


def test_report_file_suffix_on_collision(tmp_path):
    files = ReportFile(tmp_path / "p")
    paths = [files.write("2026-03-04", "x", "ä") for _ in range(3)]
    assert [p.name for p in paths] == ["2026-03-04_x.md", "2026-03-04_x-2.md", "2026-03-04_x-3.md"]
    assert paths[0].read_bytes() == "ä".encode("utf-8")
