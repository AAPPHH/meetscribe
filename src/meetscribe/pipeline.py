import gc
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from bisect import bisect_left
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path

import click
import numpy as np
import soundfile as sf

from meetscribe.store import PROJECT_ROOT, Query, Report, ReportFile, as_date

DEVICES = ("npu", "gpu", "cpu")
WHISPER_INT8 = {"repo": "OpenVINO/whisper-large-v3-turbo-int8-ov", "dir": "whisper-large-v3-turbo-int8-ov"}
WHISPER_FP16 = {"repo": "OpenVINO/whisper-large-v3-turbo-fp16-ov", "dir": "whisper-large-v3-turbo-fp16-ov"}
MODELS = {"npu": dict(WHISPER_FP16), "gpu": dict(WHISPER_INT8), "cpu": dict(WHISPER_INT8)}
SETTINGS = {
    "default_device": "npu",
    "device_order": DEVICES,
    "language": "de",
    "sample_rate": 16000,
    "window_s": 300.0,
    "search_s": 5.0,
    "frame_s": 0.05,
    "max_gap_s": 1.5,
    "min_segment_s": 8.0,
    "sentence_end": ".!?",
    "silence_peak": 1e-4,
    "unknown_speaker": "UNKNOWN",
    "models_env": "MEETSCRIBE_MODELS",
    "cache_subdir": "ov_cache",
    "reason_chars": 120,
    "diarization": {
        "repo": "pyannote/speaker-diarization-community-1",
        "dir": "pyannote-speaker-diarization-community-1",
        "config": "config.yaml",
        "token_env": "HF_TOKEN",
    },
}
UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"})
RECORDING = {
    "patterns": {
        "dji": [r"dji", r"mic\s*mini", r"mic\s*2\b", r"wireless\s*mic"],
        "laptop": [r"intel.*smart\s*sound", r"mikrofonarray", r"microphone\s*array"],
    },
    "hostapi_preference": ["Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS"],
    "flush_s": 2.0,
    "mix_block_s": 30.0,
    "default_suffix": ".flac",
    "poll_s": 0.2,
    "lid_notice": "Hinweis: Deckel offen lassen, sonst kann der Rechner trotz Wachhalten in den Ruhezustand gehen.",
}
START = {
    "audio_subdir": "aufnahmen",
    "stamp": "%Y-%m-%d_%H%M%S",
    "rotate_days": 30,
    "sentence_chars": 140,
    "program": "meetscribe",
}
AWAKE_FLAGS = {"continuous": 0x80000000, "system_required": 0x00000001}


class PipelineError(click.ClickException):
    pass


class AudioError(PipelineError):
    pass


class DeviceError(PipelineError):
    pass


class ASRFailure(PipelineError):
    pass


class Models:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get(SETTINGS["models_env"]) or PROJECT_ROOT / "models")

    @property
    def cache_dir(self):
        return self.root / SETTINGS["cache_subdir"]

    @property
    def diarization_dir(self):
        return self.root / SETTINGS["diarization"]["dir"]

    def whisper_dir(self, device, override=None):
        if override:
            path = Path(override)
            return path if path.exists() else self.root / override
        return self.root / MODELS[device]["dir"]

    @staticmethod
    def whisper_ready(path):
        return path.is_dir() and any(path.glob("*.xml"))

    def diarization_ready(self):
        return (self.diarization_dir / SETTINGS["diarization"]["config"]).is_file()


class ModelSetup:
    def __init__(self, models, token=None):
        self.models = models
        self.token = token

    @staticmethod
    def download(repo, target, token=None):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, local_dir=str(target), token=token)

    def run(self):
        messages, failed = [], False
        for repo, name in {spec["repo"]: spec["dir"] for spec in MODELS.values()}.items():
            target = self.models.root / name
            try:
                self.download(repo, target)
                messages.append(f"whisper {repo}: ok ({target})")
            except Exception as exc:
                failed = True
                messages.append(f"whisper {repo}: FEHLER {Reason.of(exc)}")
        diar = SETTINGS["diarization"]
        if self.token:
            try:
                self.download(diar["repo"], self.models.diarization_dir, self.token)
                messages.append(f"pyannote {diar['repo']}: ok ({self.models.diarization_dir})")
            except Exception as exc:
                messages.append(f"pyannote {diar['repo']}: FEHLER {Reason.of(exc)}")
        elif not self.models.diarization_ready():
            messages.append(
                f"pyannote {diar['repo']}: fehlt. {diar['token_env']} setzen (Zugriff auf huggingface.co "
                "freischalten) und 'meetscribe setup' erneut ausfuehren; ohne Modell nur mit --no-diarize."
            )
        return messages, failed

    def warm(self, device="npu"):
        engine = WhisperEngine(device, self.models, SETTINGS["language"])
        engine.transcribe(np.zeros(SETTINGS["sample_rate"], dtype="float32"))


class KeepAwake:
    def __init__(self, call=None):
        self.call = call or self.native()

    @staticmethod
    def native():
        if sys.platform != "win32":
            return None
        import ctypes
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint32]
        function.restype = ctypes.c_uint32
        return function

    def __enter__(self):
        if self.call:
            self.call(AWAKE_FLAGS["continuous"] | AWAKE_FLAGS["system_required"])
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.call:
            self.call(AWAKE_FLAGS["continuous"])
        return False


class Interrupt:
    def __init__(self):
        self.event = threading.Event()
        self.previous = None
        self.installed = False

    def _handler(self, signum, frame):
        self.event.set()

    def __enter__(self):
        try:
            self.previous = signal.signal(signal.SIGINT, self._handler)
            self.installed = True
        except ValueError:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.installed:
            signal.signal(signal.SIGINT, self.previous if self.previous is not None else signal.default_int_handler)
        return False

    def wait(self, timeout):
        return self.event.wait(timeout)


class Reason:
    @staticmethod
    def of(exc):
        lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
        text = lines[0] if lines else ""
        text = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
        return text[: SETTINGS["reason_chars"]]


class AudioLoader:
    @staticmethod
    def require():
        if shutil.which("ffmpeg") is None:
            raise AudioError("ffmpeg nicht gefunden. Bitte ffmpeg installieren und in den PATH aufnehmen.")

    @classmethod
    def load(cls, path, rate=None):
        rate = rate or SETTINGS["sample_rate"]
        cls.require()
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["unbekannter Fehler"]
            raise AudioError(f"Audio konnte nicht dekodiert werden ({path}): {tail[0]}")
        audio = np.frombuffer(proc.stdout, dtype="<f4").copy()
        if audio.size == 0:
            raise AudioError(f"Audio enthaelt keine Daten: {path}")
        return audio


class Windows:
    def __init__(self, window_s=None, search_s=None, frame_s=None, rate=None):
        self.window_s = window_s or SETTINGS["window_s"]
        self.search_s = SETTINGS["search_s"] if search_s is None else search_s
        self.frame_s = frame_s or SETTINGS["frame_s"]
        self.rate = rate or SETTINGS["sample_rate"]

    def quietest(self, audio, center):
        search, frame = int(self.search_s * self.rate), max(int(self.frame_s * self.rate), 1)
        low, high = max(center - search, 0), min(center + search, len(audio))
        count = (high - low) // frame
        if count < 1:
            return center
        energy = np.square(audio[low : low + count * frame]).reshape(count, frame).mean(axis=1)
        return low + int(np.argmin(energy)) * frame + frame // 2

    def split(self, audio):
        size, search, total = int(self.window_s * self.rate), int(self.search_s * self.rate), len(audio)
        cuts, boundary = [], size
        while boundary + search < total:
            cuts.append(self.quietest(audio, boundary))
            boundary += size
        edges = [0, *cuts, total]
        return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


class Aligner:
    def __init__(self, max_gap_s=None, unknown=None, min_segment_s=None):
        self.max_gap_s = SETTINGS["max_gap_s"] if max_gap_s is None else max_gap_s
        self.min_segment_s = SETTINGS["min_segment_s"] if min_segment_s is None else min_segment_s
        self.unknown = unknown or SETTINGS["unknown_speaker"]

    @staticmethod
    def index(turns):
        ordered = sorted(turns, key=lambda t: (t["start"], t["end"]))
        reach, top = [], float("-inf")
        for turn in ordered:
            top = max(top, turn["end"])
            reach.append(top)
        return ordered, [t["start"] for t in ordered], reach

    def speaker(self, word, index):
        ordered, starts, reach = index
        if not ordered:
            return self.unknown
        start, end = word["start_s"], word["end_s"]
        overlap = {}
        for i in range(bisect_left(starts, end) - 1, -1, -1):
            if reach[i] <= start:
                break
            turn = ordered[i]
            amount = min(end, turn["end"]) - max(start, turn["start"])
            if amount > 0:
                overlap[turn["speaker"]] = overlap.get(turn["speaker"], 0.0) + amount
        if overlap:
            return max(overlap, key=overlap.get)
        return min(ordered, key=lambda t: max(t["start"] - end, start - t["end"], 0.0))["speaker"]

    def closed(self, segment):
        return segment["end_s"] - segment["start_s"] >= self.min_segment_s and segment["words"][-1]["text"].endswith(tuple(SETTINGS["sentence_end"]))

    def align(self, words, turns):
        index = self.index(turns)
        segments = []
        for word in sorted(words, key=lambda w: (w["start_s"], w["end_s"])):
            speaker = self.speaker(word, index)
            current = segments[-1] if segments else None
            if current and current["speaker"] == speaker and word["start_s"] - current["end_s"] <= self.max_gap_s and not self.closed(current):
                current["words"].append(dict(word))
                current["end_s"] = max(current["end_s"], word["end_s"])
            else:
                segments.append({"start_s": word["start_s"], "end_s": word["end_s"], "speaker": speaker, "words": [dict(word)]})
        for segment in segments:
            segment["text"] = " ".join(w["text"] for w in segment["words"])
        return segments


class WhisperEngine:
    def __init__(self, device, models, language, override=None):
        path = models.whisper_dir(device, override)
        if not Models.whisper_ready(path):
            raise FileNotFoundError(f"Modell fehlt: {path}")
        import openvino_genai
        self.device, self.language, self.model = device, language, path.name
        self.pipe = openvino_genai.WhisperPipeline(
            str(path), device.upper(), word_timestamps=True, CACHE_DIR=str(models.cache_dir)
        )

    def transcribe(self, audio):
        result = self.pipe.generate(
            audio.tolist(), language=f"<|{self.language}|>", task="transcribe", return_timestamps=True, word_timestamps=True
        )
        return {
            "words": [{"text": w.word, "start_s": w.start_ts, "end_s": w.end_ts} for w in (result.words or [])],
            "chunks": [{"text": c.text, "start_s": c.start_ts, "end_s": c.end_ts} for c in (result.chunks or [])],
        }


class ASRRunner:
    def __init__(self, devices, factory, fallback=True):
        self.devices = list(devices)
        self.factory = factory
        self.fallback = fallback
        self.index = 0
        self.engine = None
        self.used = []

    def transcribe(self, audio):
        while True:
            device = self.devices[self.index]
            try:
                if self.engine is None:
                    self.engine = self.factory(device)
                result = self.engine.transcribe(audio)
            except Exception as exc:
                self.engine = None
                gc.collect()
                click.echo(f"{device.upper()} fehlgeschlagen: {Reason.of(exc)}", err=True)
                self.index += 1
                if not self.fallback:
                    raise ASRFailure(f"Transkription auf {device.upper()} fehlgeschlagen (--no-fallback).") from exc
                if self.index >= len(self.devices):
                    raise ASRFailure("Alle Geraete fehlgeschlagen: " + ", ".join(d.upper() for d in self.devices)) from exc
                continue
            if device not in self.used:
                self.used.append(device)
            return self.engine, result


class Transcriber:
    def __init__(self, runner, aligner=None, windows=None):
        self.runner = runner
        self.aligner = aligner or Aligner()
        self.windows = windows or Windows()

    @staticmethod
    def words(result, offset, limit, floor):
        raw = result.get("words") or result.get("chunks") or []
        words = []
        for item in raw:
            text = item["text"].strip()
            if not text:
                continue
            start = min(max(offset + item["start_s"], floor), limit)
            end = min(max(offset + item["end_s"], start), limit)
            words.append({"text": text, "start_s": start, "end_s": end})
            floor = start
        return words

    def run(self, audio, turns, writer):
        rate, floor, count = self.windows.rate, 0.0, 0
        for first, last in self.windows.split(audio):
            chunk = audio[first:last]
            if float(np.max(np.abs(chunk))) < SETTINGS["silence_peak"]:
                continue
            engine, result = self.runner.transcribe(chunk)
            words = self.words(result, first / rate, last / rate, floor)
            segments = self.aligner.align(words, turns)
            for segment in segments:
                segment.update(device=engine.device.upper(), model=engine.model)
            count += writer.append(segments)
            if words:
                floor = words[-1]["start_s"]
        return {"segments": count, "used": [d.upper() for d in self.runner.used]}


class Diarizer:
    def __init__(self, models):
        self.models = models

    @staticmethod
    def annotation(output):
        for name in ("exclusive_speaker_diarization", "speaker_diarization"):
            found = getattr(output, name, None)
            if found is not None:
                return found
        return output

    def diarize(self, audio, rate=None, speakers=None, min_speakers=None, max_speakers=None):
        import torch
        from pyannote.audio import Pipeline
        options = {"num_speakers": speakers, "min_speakers": min_speakers, "max_speakers": max_speakers}
        pipeline = Pipeline.from_pretrained(str(self.models.diarization_dir))
        try:
            pipeline.to(torch.device("cpu"))
            source = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": rate or SETTINGS["sample_rate"]}
            annotation = self.annotation(pipeline(source, **{k: v for k, v in options.items() if v is not None}))
            return [
                {"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
                for turn, _, label in annotation.itertracks(yield_label=True)
            ]
        finally:
            del pipeline
            gc.collect()


class Naming:
    @staticmethod
    def slug(text):
        text = text.translate(UMLAUTS)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "meeting"

    @classmethod
    def meeting_id(cls, store, day, stem):
        base = f"{as_date(day):%Y%m%d}-{cls.slug(stem)}"
        candidate, number = base, 1
        while store.exists(candidate):
            number += 1
            candidate = f"{base}-{number}"
        return candidate


class TranscribeJob:
    def __init__(self, store, audio, device, language, models=None, model=None, title=None, day=None, meeting_id=None,
                 diarize=True, fallback=True, speakers=None, min_speakers=None, max_speakers=None):
        self.store = store
        self.audio = Path(audio)
        self.device = device.lower()
        self.language = language
        self.models = models or Models()
        self.model = model
        self.title = title if title is not None else self.audio.stem
        self.day = as_date(day) if day else date.today()
        self.requested_id = meeting_id
        self.diarize = diarize
        self.fallback = fallback
        self.speaker_options = {"speakers": speakers, "min_speakers": min_speakers, "max_speakers": max_speakers}

    @property
    def chain(self):
        order = SETTINGS["device_order"]
        return order[order.index(self.device):] if self.fallback else (self.device,)

    def check(self):
        if self.diarize and not self.models.diarization_ready():
            diar = SETTINGS["diarization"]
            raise PipelineError(
                f"Diarisierungsmodell fehlt ({self.models.diarization_dir}). Bitte {diar['token_env']} setzen "
                f"(Zugriff auf {diar['repo']} bei huggingface.co akzeptieren) und 'meetscribe setup' ausfuehren, "
                "oder mit --no-diarize ohne Sprechertrennung transkribieren."
            )
        path = self.models.whisper_dir(self.device, self.model)
        if not Models.whisper_ready(path):
            raise PipelineError(f"Whisper-Modell fehlt ({path}). Bitte 'meetscribe setup' ausfuehren.")

    def run(self):
        self.check()
        meeting_id = self.requested_id or Naming.meeting_id(self.store, self.day, self.audio.stem)
        try:
            exists = self.store.exists(meeting_id)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        if exists:
            raise PipelineError(f"meeting_id existiert bereits: {meeting_id}")
        audio = AudioLoader.load(self.audio)
        turns = Diarizer(self.models).diarize(audio, **self.speaker_options) if self.diarize else []
        gc.collect()
        runner = ASRRunner(self.chain, lambda d: WhisperEngine(d, self.models, self.language, self.model), self.fallback)
        with self.store.writer(meeting_id, title=self.title, meeting_date=self.day, language=self.language) as writer:
            summary = Transcriber(runner).run(audio, turns, writer)
        return {**summary, "meeting_id": meeting_id, "requested": self.device.upper(), "speakers": len({t["speaker"] for t in turns})}


class InputPicker:
    def __init__(self, devices, hostapis):
        self.devices = [{**d, "index": d.get("index", i)} for i, d in enumerate(devices)]
        self.hostapis = [h["name"] if isinstance(h, dict) else h for h in hostapis]

    def inputs(self):
        return [d for d in self.devices if d["max_input_channels"] > 0]

    def rank(self, device):
        preference = RECORDING["hostapi_preference"]
        api = self.hostapis[device["hostapi"]]
        return preference.index(api) if api in preference else len(preference)

    def listing(self):
        return [f"{d['name']} [{self.hostapis[d['hostapi']]}]" for d in self.inputs()]

    def auto(self):
        for spec in RECORDING["patterns"]:
            try:
                self.pick(spec)
                return spec
            except DeviceError:
                continue
        return "laptop"

    def pick(self, spec):
        patterns = RECORDING["patterns"].get(spec.lower()) or [re.escape(spec)]
        found = [d for d in self.inputs() if any(re.search(p, d["name"], re.I) for p in patterns)]
        if not found:
            existing = "\n".join(f"  {line}" for line in self.listing()) or "  (keine Eingabegeraete)"
            raise DeviceError(f"Kein Audio-Eingang passend zu '{spec}' gefunden. Vorhanden:\n{existing}")
        return min(found, key=lambda d: (self.rank(d), d["index"]))


class Recorder:
    def __init__(self, device, path, hostapis=()):
        self.device = device
        self.path = Path(path)
        self.rate = int(device["default_samplerate"])
        self.hostapi = hostapis[device["hostapi"]] if hostapis else ""
        self.queue = queue.Queue()
        self.overruns = 0
        self.frames = 0
        self.error = None
        self.stream = None
        self.file = None
        self.thread = None

    def _callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            self.overruns += 1
        self.queue.put(indata.copy())

    def _drain(self):
        last = time.monotonic()
        while True:
            block = self.queue.get()
            if block is None:
                return
            try:
                self.file.write(block)
                self.frames += len(block)
                if time.monotonic() - last > RECORDING["flush_s"]:
                    self.file.flush()
                    last = time.monotonic()
            except Exception as exc:
                self.error = self.error or exc

    def start(self):
        import sounddevice as sd
        rate, channels = self.rate, int(self.device["max_input_channels"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = sf.SoundFile(self.path, "w", samplerate=rate, channels=channels)
        extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in self.hostapi else None
        try:
            self.stream = sd.InputStream(
                device=self.device["index"], channels=channels, samplerate=rate, dtype="float32",
                callback=self._callback, extra_settings=extra,
            )
            self.stream.start()
        except Exception:
            self.file.close()
            raise
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.thread is not None:
            self.queue.put(None)
            self.thread.join()
        if self.file is not None:
            self.file.close()
        return {"path": str(self.path), "seconds": self.frames / self.rate, "overruns": self.overruns, "error": self.error}


class Mixer:
    @staticmethod
    def stretched(handle, total, start, stop):
        length = len(handle)
        if length == 0:
            return np.zeros(stop - start, dtype="float32")
        positions = np.arange(start, stop) * ((length - 1) / max(total - 1, 1))
        low = int(positions[0])
        high = min(int(np.ceil(positions[-1])) + 1, length)
        handle.seek(low)
        data = handle.read(high - low, dtype="float32", always_2d=True).mean(axis=1)
        return np.interp(positions - low, np.arange(len(data)), data)

    @classmethod
    def mix(cls, paths, out):
        infos = [sf.info(p) for p in paths]
        rate = max(i.samplerate for i in infos)
        total = max(round(i.duration * rate) for i in infos)
        block = int(RECORDING["mix_block_s"] * rate)
        with ExitStack() as stack:
            handles = [stack.enter_context(sf.SoundFile(p)) for p in paths]
            target = stack.enter_context(sf.SoundFile(out, "w", samplerate=rate, channels=1))
            for start in range(0, total, block):
                stop = min(start + block, total)
                target.write(np.mean([cls.stretched(h, total, start, stop) for h in handles], axis=0).astype("float32"))
        return Path(out)


class RecordingSession:
    def __init__(self, mic, out, picker):
        self.mic = mic
        self.out = Path(out) if Path(out).suffix else Path(out).with_suffix(RECORDING["default_suffix"])
        self.picker = picker
        self.recorders = None

    @property
    def specs(self):
        return list(RECORDING["patterns"]) if self.mic.lower() == "both" else [self.mic]

    def paths(self):
        if self.mic.lower() != "both":
            return {self.mic: self.out}
        stem, suffix = self.out.with_suffix(""), self.out.suffix
        return {**{spec: Path(f"{stem}.{spec}{suffix}") for spec in self.specs}, "mix": Path(f"{stem}.mix{suffix}")}

    def prepare(self):
        paths = self.paths()
        self.recorders = [Recorder(self.picker.pick(spec), paths[spec], self.picker.hostapis) for spec in self.specs]
        return self.recorders

    def run(self, seconds=None):
        paths = self.paths()
        recorders = self.recorders or self.prepare()
        started = []
        with Interrupt() as interrupt:
            try:
                for recorder in recorders:
                    recorder.start()
                    started.append(recorder)
                deadline = time.monotonic() + seconds if seconds else None
                try:
                    while not interrupt.wait(RECORDING["poll_s"]) and (deadline is None or time.monotonic() < deadline):
                        pass
                except KeyboardInterrupt:
                    pass
            finally:
                reports = [r.stop() for r in started]
            if "mix" in paths and len(started) == len(recorders):
                Mixer.mix([r.path for r in recorders], paths["mix"])
        return reports


class StartFlow:
    def __init__(self, store, audio=None, mic=None, device="npu", title=None, speakers=None, diarize=True,
                 open_report=True, tty=False, models=None, picker=None):
        self.store = store
        self.audio = Path(audio) if audio else None
        self.mic = mic
        self.device = device.lower()
        self.title = title
        self.speakers = speakers
        self.diarize = diarize
        self.open_report = open_report
        self.tty = tty
        self.models = models or Models()
        self.picker = picker
        self.session = None
        self.job = None

    def retry_command(self, audio):
        parts = [START["program"], "start", str(audio), "--device", self.device]
        for flag, value in (("--title", self.title), ("--speakers", self.speakers)):
            if value is not None:
                parts += [flag, str(value)]
        parts += [flag for flag, active in (("--no-diarize", not self.diarize), ("--no-open", not self.open_report)) if active]
        return subprocess.list2cmdline(parts)

    def recording_path(self):
        folder = self.store.config["data_dir"] / START["audio_subdir"]
        stem = datetime.now().strftime(START["stamp"])
        candidate, number = folder / f"{stem}{RECORDING['default_suffix']}", 1
        while candidate.exists():
            number += 1
            candidate = folder / f"{stem}-{number}{RECORDING['default_suffix']}"
        return candidate

    def preflight(self):
        AudioLoader.require()
        planned = self.audio or self.recording_path()
        self.job = TranscribeJob(
            self.store, planned, self.device, SETTINGS["language"], models=self.models, title=self.title,
            diarize=self.diarize, speakers=self.speakers,
        )
        self.job.check()
        if self.audio is not None:
            return
        try:
            picker = self.picker or AudioDevices.sounddevice_picker()
        except Exception as exc:
            raise DeviceError(f"Audio-Eingaenge nicht lesbar: {Reason.of(exc)}") from exc
        self.session = RecordingSession(self.mic or picker.auto(), planned, picker)
        for recorder in self.session.prepare():
            click.echo(f"Eingang: {recorder.device['name']} [{recorder.hostapi}]")

    def record(self):
        click.echo(RECORDING["lid_notice"])
        click.echo("Aufnahme laeuft, Ende mit Strg+C.")
        reports = self.session.run()
        path = self.session.paths().get("mix") or self.session.out
        click.echo(f"Aufnahme gespeichert: {path} ({max(r['seconds'] for r in reports):.0f} s)")
        errors = [str(r["error"]) for r in reports if r["error"]]
        if errors:
            raise PipelineError(f"Schreibfehler waehrend der Aufnahme ({path}): " + "; ".join(errors))
        return path

    def fail(self, audio, exc):
        interrupted = isinstance(exc, KeyboardInterrupt)
        click.echo("Verarbeitung abgebrochen." if interrupted else f"Verarbeitung fehlgeschlagen: {Reason.of(exc)}", err=True)
        click.echo(f"Audio erhalten: {audio}", err=True)
        click.echo(f"Erneut versuchen: {self.retry_command(audio)}", err=True)
        raise click.exceptions.Exit(130 if interrupted else 1)

    def process(self, audio):
        self.job.audio = Path(audio)
        self.job.requested_id = Naming.meeting_id(self.store, self.job.day, self.title or self.job.audio.stem)
        started = time.monotonic()
        try:
            result = self.job.run()
        except (Exception, KeyboardInterrupt) as exc:
            self.fail(audio, exc)
        click.echo(
            f"{result['meeting_id']}: {result['segments']} Segmente, {result['speakers']} Sprecher, "
            f"Geraet: {' -> '.join(result['used']) or '-'}, {time.monotonic() - started:.0f} s"
        )
        return result["meeting_id"]

    def name_speakers(self, meeting_id):
        if not self.tty:
            return
        _, rows = Query(self.store).execute(
            "SELECT speaker, arg_max(text, end_s - start_s) FROM segments WHERE meeting_id = ? AND speaker <> ? "
            "GROUP BY speaker ORDER BY speaker",
            [meeting_id, SETTINGS["unknown_speaker"]],
        )
        try:
            for label, text in rows:
                click.echo(f"{label}: \"{' '.join(text.split())[: START['sentence_chars']]}\"")
                name = click.prompt(f"Name fuer {label} (Enter = ueberspringen)", default="", show_default=False).strip()
                if name:
                    self.store.speakers.set(meeting_id, label, name)
        except click.Abort:
            click.echo()

    def write_report(self, meeting_id):
        text = Report(Query(self.store)).render(meeting_id)
        path = ReportFile().write(self.job.day, Naming.slug(self.title or self.job.audio.stem), text)
        click.echo(f"Protokoll: {path}")
        return path

    def rotate(self):
        try:
            self.store.rotate(older_than_days=START["rotate_days"])
        except Exception as exc:
            click.echo(f"Warnung: Rotation fehlgeschlagen: {Reason.of(exc)}", err=True)

    def run(self):
        self.preflight()
        audio = self.record() if self.audio is None else self.audio
        meeting_id = self.process(audio)
        self.name_speakers(meeting_id)
        path = self.write_report(meeting_id)
        if self.open_report and self.tty and hasattr(os, "startfile"):
            os.startfile(path)
        self.rotate()
        return path


class AudioDevices:
    @staticmethod
    def sounddevice_picker():
        import sounddevice as sd
        return InputPicker([dict(d) for d in sd.query_devices()], [dict(h) for h in sd.query_hostapis()])

    @staticmethod
    def openvino_devices():
        import openvino as ov
        core = ov.Core()
        rows = []
        for name in core.available_devices:
            try:
                full = core.get_property(name, "FULL_DEVICE_NAME")
            except Exception:
                full = ""
            rows.append({"device": name, "name": full})
        return rows
