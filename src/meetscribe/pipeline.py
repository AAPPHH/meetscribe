import gc
import os
import _thread
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
from contextlib import ExitStack, suppress
from datetime import date, datetime
from pathlib import Path

import click
import numpy as np
import soundfile as sf

from meetscribe import backends
from meetscribe.backends import CATALOG, Catalog, Hardware, Selection
from meetscribe.store import PROJECT_ROOT, Json, Query, Report, ReportFile, as_date

SETTINGS = {
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
    "download_poll_s": 0.5,
    "reason_chars": 400,
    "detail_chars": 4000,
    "diarization": {
        "repo": "pyannote/speaker-diarization-community-1",
        "dir": "pyannote-speaker-diarization-community-1",
        "config": "config.yaml",
        "token_env": "HF_TOKEN",
        "model_ref": r"\$model/([^\s'\",\]}#]+)",
        "comment": r"(?:^|\s)#.*",
        "default_files": ["pytorch_model.bin"],
        "ref_files": {"plda": ["plda.npz", "xvec_transform.npz"]},
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
    "stall_s": 5.0,
    "lid_notice": "Hinweis: Deckel offen lassen, sonst kann der Rechner trotz Wachhalten in den Ruhezustand gehen.",
}
START = {
    "audio_subdir": "aufnahmen",
    "stamp": "%Y-%m-%d_%H%M%S",
    "rotate_days": 30,
    "sentence_chars": 140,
    "program": "meetscribe",
    "stdin_fd": 0,
    "title_chars": 80,
    "lock_file": "start.lock",
    "heartbeat_s": 2.0,
    "abort_grace_s": 30.0,
}
AWAKE_FLAGS = {"continuous": 0x80000000, "system_required": 0x00000001}


class PipelineError(click.ClickException):
    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail

    def format_message(self):
        return f"{self.message}\n{self.detail}" if self.detail else self.message


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

    @staticmethod
    def ready(backend, path):
        return backends.ENGINES[backend].ready(path)

    def diarization_missing(self):
        diar = SETTINGS["diarization"]
        root = self.diarization_dir
        config = root / diar["config"]
        if not config.is_file():
            return [config]
        lines = (re.sub(diar["comment"], "", line) for line in config.read_text(encoding="utf-8", errors="replace").splitlines())
        found = dict.fromkeys(ref for line in lines for ref in re.findall(diar["model_ref"], line))
        return [path for ref in found for path in self.absent(root, ref)]

    @staticmethod
    def absent(root, ref):
        diar = SETTINGS["diarization"]
        path = root / ref
        if path.is_file():
            return []
        if not path.is_dir():
            return [path]
        return [path / name for name in diar["ref_files"].get(ref, diar["default_files"]) if not (path / name).is_file()]

    def diarization_ready(self):
        return not self.diarization_missing()


class DownloadProgress:
    def __init__(self, target, total_mb, reporter, repo):
        self.target, self.total, self.reporter, self.repo = Path(target), int((total_mb or 0) * 1_000_000), reporter, repo
        self.done = threading.Event()
        self.thread = None
        self.last = -1

    def size(self):
        return sum(f.stat().st_size for f in self.target.rglob("*") if f.is_file()) if self.target.is_dir() else 0

    def emit(self):
        size = self.size()
        if size != self.last:
            self.last = size
            self.reporter.event("download", repo=self.repo, bytes=size, total=self.total)

    def _run(self):
        while not self.done.wait(SETTINGS["download_poll_s"]):
            self.emit()

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.done.set()
        self.thread.join()
        if exc_type is None:
            self.emit()
        return False


class ModelSetup:
    def __init__(self, models, token=None, reporter=None):
        self.models = models
        self.token = token
        self.reporter = reporter or Reporter()

    @staticmethod
    def download(repo, target, token=None):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, local_dir=str(target), token=token)

    def targets(self, model=None, backend=None, device=None):
        backend = (backend or "auto").lower()
        backend = CATALOG["backend_order"][0] if backend == "auto" else backend
        if backend not in CATALOG["backends"]:
            raise backends.SelectionError(f"Unbekanntes Backend '{backend}'. Verfügbar: {', '.join(CATALOG['backend_order'])}.")
        name, variant = Selection.split(model or CATALOG["default_model"])
        name = CATALOG["default_model"] if name == "auto" else name
        if name not in CATALOG["models"]:
            raise backends.SelectionError(f"Unbekanntes Modell '{name}'. Verfügbar: {', '.join(CATALOG['models'])}.")
        variants = Catalog.variants(backend, name)
        if variant and variant not in variants:
            raise backends.SelectionError(f"Unbekannte Variante '{variant}' für {name}. Verfügbar: {', '.join(variants)}.")
        kinds = [device] if device and device != "auto" else list(CATALOG["hardware_order"])
        chosen = [variant] if variant else [spec["default"] for kind in kinds if (spec := Catalog.spec(backend, kind))]
        return backend, name, list(dict.fromkeys(chosen))

    def fetch(self, backend, model, variant):
        entry = Catalog.entry(backend, model, variant)
        target = self.models.root / Catalog.dir_name(entry)
        with DownloadProgress(target, entry["mb"], self.reporter, entry["repo"]):
            self.download(entry["repo"], target)
        if not self.models.ready(backend, target):
            raise PipelineError(f"Der Download von {entry['repo']} ist unvollständig: {target}")
        return entry["repo"], target

    def run(self, model=None, backend=None, device=None):
        messages, failed = [], False
        explicit = any((model, backend, device))
        backend, name, variants = self.targets(model, backend, device)
        for variant in variants:
            repo = Catalog.entry(backend, name, variant)["repo"]
            try:
                _, target = self.fetch(backend, name, variant)
                messages.append(f"whisper {repo}: ok ({target})")
            except Exception as exc:
                failed = True
                messages.append(f"whisper {repo}: FEHLER {Reason.of(exc)}")
        if explicit:
            return messages, failed
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

    def warm(self, step):
        engine = backends.ENGINES[step.backend](step, self.models, SETTINGS["language"])
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


class Reporter:
    machine = False
    broken = False

    def say(self, text="", err=False):
        click.echo(text, err=err)

    def warn(self, text, prefix="Warnung: "):
        click.echo(prefix + text, err=True)

    def event(self, name, **data):
        pass


class EventReporter(Reporter):
    machine = True

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.broken = False

    def say(self, text="", err=False):
        pass

    def warn(self, text, prefix=None):
        self.event("warning", message=text)

    def event(self, name, **data):
        if self.broken:
            return
        try:
            click.echo(Json.dumps({"event": name, **data}), file=self.stream)
            self.stream.flush()
        except (OSError, ValueError):
            self.broken = True


class StdinWatcher:
    def __init__(self, fd=None):
        self.fd = START["stdin_fd"] if fd is None else fd
        self.event = threading.Event()

    def _read(self):
        try:
            while (chunk := os.read(self.fd, 4096)) and b"\n" not in chunk:
                pass
        except OSError:
            pass
        self.event.set()

    def start(self):
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def is_set(self):
        return self.event.is_set()


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

    @staticmethod
    def detail(exc):
        text = getattr(exc, "detail", None) or (exc.format_message() if isinstance(exc, click.ClickException) else str(exc))
        return text.strip()[: SETTINGS["detail_chars"]] or type(exc).__name__


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


class ASRRunner:
    def __init__(self, steps, factory, fallback=True, reporter=None):
        self.reporter = reporter or Reporter()
        self.steps = list(steps)
        self.factory = factory
        self.fallback = fallback
        self.index = 0
        self.engine = None
        self.used = []

    def transcribe(self, audio):
        while True:
            step = self.steps[self.index]
            try:
                if self.engine is None:
                    self.engine = self.factory(step)
                result = self.engine.transcribe(audio)
            except Exception as exc:
                self.engine = None
                gc.collect()
                self.index += 1
                if not self.fallback:
                    self.reporter.warn(f"{step.label} fehlgeschlagen: {Reason.of(exc)}", prefix="")
                    raise ASRFailure(f"Transkription auf {step.label} fehlgeschlagen (--no-fallback).") from exc
                if self.index >= len(self.steps):
                    self.reporter.warn(f"{step.label} fehlgeschlagen: {Reason.of(exc)}", prefix="")
                    raise ASRFailure("Alle Geräte fehlgeschlagen: " + ", ".join(s.label for s in self.steps)) from exc
                following = self.steps[self.index]
                self.reporter.event("fallback", **{"from": step.label, "to": following.label, "backend": following.backend})
                self.reporter.warn(f"{step.label} fehlgeschlagen: {Reason.of(exc)}. Rückfall auf {following.label} ({Catalog.backend_label(following.backend)}).", prefix="")
                continue
            if step.label not in self.used:
                self.used.append(step.label)
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
        rate, floor, count, speakers, ran = self.windows.rate, 0.0, 0, set(), []
        for first, last in self.windows.split(audio):
            chunk = audio[first:last]
            if float(np.max(np.abs(chunk))) < SETTINGS["silence_peak"]:
                continue
            engine, result = self.runner.transcribe(chunk)
            words = self.words(result, first / rate, last / rate, floor)
            segments = self.aligner.align(words, turns)
            if (engine.backend, engine.model) not in ran:
                ran.append((engine.backend, engine.model))
            for segment in segments:
                segment.update(device=engine.device, model=engine.model, backend=engine.backend)
            count += writer.append(segments)
            speakers.update(segment["speaker"] for segment in segments)
            if words:
                floor = words[-1]["start_s"]
        return {"segments": count, "speakers": len(speakers), "used": list(self.runner.used), "engines": ran}


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
                 diarize=True, fallback=True, speakers=None, min_speakers=None, max_speakers=None, reporter=None,
                 backend="auto", hardware=None):
        self.reporter = reporter or Reporter()
        self.store = store
        self.audio = Path(audio)
        self.device = (device or "auto").lower()
        self.backend = (backend or "auto").lower()
        self.language = language
        self.models = models or Models()
        self.model = model
        self.title = title if title is not None else self.audio.stem
        self.day = as_date(day) if day else date.today()
        self.requested_id = meeting_id
        self.diarize = diarize
        self.fallback = fallback
        self.hardware = hardware
        self.speaker_options = {"speakers": speakers, "min_speakers": min_speakers, "max_speakers": max_speakers}
        self.cached = None

    @property
    def plan(self):
        if self.cached is None:
            self.hardware = self.hardware or Hardware()
            self.cached = Selection(self.models, self.hardware).resolve(self.backend, self.device, self.model or "auto", self.fallback)
            for notice in self.cached.notices:
                self.reporter.warn(notice, prefix="")
        return self.cached

    def check(self):
        missing = self.models.diarization_missing() if self.diarize else []
        if missing:
            diar = SETTINGS["diarization"]
            raise PipelineError(
                "Diarisierungsmodell fehlt oder ist unvollständig: Die Sprechertrennung ist noch nicht eingerichtet. "
                "Schnelle Lösung: in den Einstellungen „Sprecher trennen“ ausschalten (Kommandozeile: --no-diarize). "
                f"Dauerhaft: {diar['token_env']} setzen und 'meetscribe setup' ausführen.",
                f"Fehlende Dateien: {', '.join(str(p) for p in missing)}\n"
                f"Für 'meetscribe setup' den Zugriff auf {diar['repo']} bei huggingface.co akzeptieren und {diar['token_env']} setzen.",
            )
        step = self.plan.primary
        if not step.installed:
            variant = "" if step.variant == "custom" else f":{step.variant}"
            raise PipelineError(
                f"Das Modell {self.plan.model} ({Catalog.backend_label(step.backend)}, {step.variant}) fehlt: {step.path}. "
                f"Bitte 'meetscribe setup --model {self.plan.model}{variant} --backend {step.backend}' ausführen."
            )

    def announce(self):
        step = self.plan.primary
        return {"backend": step.backend, "device": step.label, "model": self.plan.model, "variant": step.variant}

    def ran(self, summary):
        step = self.plan.primary
        used = summary["used"]
        fallback = bool(used) and used[0] != step.label or len(used) > 1
        backends_used = list(dict.fromkeys(b for b, _ in summary["engines"]))
        text = " · ".join([
            " → ".join(used) or step.label, self.plan.model, " + ".join(Catalog.backend_label(b) for b in backends_used) or Catalog.backend_label(step.backend),
        ])
        return {
            "backend": step.backend, "backends": backends_used, "devices": used, "model": self.plan.model, "variant": step.variant,
            "stored": [m for _, m in summary["engines"]], "requested": self.plan.requested, "fallback": fallback, "text": text,
        }

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
        runner = ASRRunner(self.plan.chain, lambda step: backends.ENGINES[step.backend](step, self.models, self.language), self.fallback, self.reporter)
        with self.store.writer(meeting_id, title=self.title, meeting_date=self.day, language=self.language, discard_on_error=True) as writer:
            summary = Transcriber(runner).run(audio, turns, writer)
        if not summary["segments"]:
            self.store.discard(meeting_id)
        return {**summary, "meeting_id": meeting_id, "requested": self.plan.requested, "ran": self.ran(summary)}


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
        self.last_audio = time.monotonic()
        self.stopped = False

    @property
    def name(self):
        return self.device["name"]

    def _callback(self, indata, frames, time_info, status):
        self.last_audio = time.monotonic()
        if status.input_overflow:
            self.overruns += 1
        self.queue.put(indata.copy())

    def _drain(self):
        last = time.monotonic()
        while True:
            block = self.queue.get()
            if block is None:
                return
            if self.error is not None:
                continue
            try:
                self.file.write(block)
                self.frames += len(block)
                if time.monotonic() - last > RECORDING["flush_s"]:
                    self.file.flush()
                    last = time.monotonic()
            except Exception as exc:
                self.error = exc

    def problem(self):
        if self.error is not None:
            return f"Schreibfehler, dieser Eingang wird nicht weiter aufgezeichnet ({Reason.of(self.error)})"
        silent = time.monotonic() - self.last_audio
        if not self.stopped and silent > RECORDING["stall_s"]:
            return f"kein Audio seit {silent:.0f} s (Geraet getrennt oder blockiert)"
        return None

    def open_error(self, exc):
        return AudioError(
            f"Audioeingang '{self.name}' konnte nicht geoeffnet werden: {Reason.of(exc)}. Bitte in den Windows-Einstellungen unter "
            "Datenschutz und Sicherheit > Mikrofon den Zugriff fuer Apps und Desktop-Apps erlauben und pruefen, ob ein anderes "
            "Programm das Geraet belegt."
        )

    def start(self):
        import sounddevice as sd
        channels = int(self.device["max_input_channels"])
        extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in self.hostapi else None
        try:
            self.stream = sd.InputStream(
                device=self.device["index"], channels=channels, samplerate=self.rate, dtype="float32",
                callback=self._callback, extra_settings=extra,
            )
        except Exception as exc:
            raise self.open_error(exc) from exc
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = sf.SoundFile(self.path, "w", samplerate=self.rate, channels=channels)
        except Exception as exc:
            self.stream.close()
            raise AudioError(f"Aufnahmedatei konnte nicht angelegt werden ({self.path}): {Reason.of(exc)}") from exc
        try:
            self.last_audio = time.monotonic()
            self.stream.start()
        except Exception as exc:
            self.stream.close()
            self.file.close()
            self.path.unlink(missing_ok=True)
            raise self.open_error(exc) from exc
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped = True
        if self.stream is not None:
            for step in (self.stream.stop, self.stream.close):
                with suppress(Exception):
                    step()
        if self.thread is not None:
            self.queue.put(None)
            self.thread.join()
        if self.file is not None:
            try:
                self.file.close()
            except Exception as exc:
                self.error = self.error or exc
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
    def __init__(self, mic, out, picker, reporter=None):
        self.reporter = reporter or Reporter()
        self.mic = mic
        self.out = Path(out) if Path(out).suffix else Path(out).with_suffix(RECORDING["default_suffix"])
        self.picker = picker
        self.recorders = None
        self.problems = {}
        self.final = None

    @property
    def mixing(self):
        return self.mic.lower() == "both"

    @property
    def specs(self):
        return list(RECORDING["patterns"]) if self.mixing else [self.mic]

    @property
    def mix_path(self):
        return Path(f"{self.out.with_suffix('')}.mix{self.out.suffix}")

    def paths(self):
        if not self.mixing:
            return {self.mic: self.out}
        stem, suffix = self.out.with_suffix(""), self.out.suffix
        return {**{spec: Path(f"{stem}.{spec}{suffix}") for spec in self.specs}, "mix": self.mix_path}

    def prepare(self):
        paths = self.paths()
        for path in paths.values():
            if path.exists():
                raise AudioError(f"Zieldatei existiert bereits und wird nicht ueberschrieben: {path}")
        self.recorders = [Recorder(self.picker.pick(spec), paths[spec], self.picker.hostapis) for spec in self.specs]
        return self.recorders

    def begin(self, recorders):
        started = []
        try:
            for recorder in recorders:
                recorder.start()
                started.append(recorder)
        except BaseException:
            for recorder in started:
                recorder.stop()
                recorder.path.unlink(missing_ok=True)
            raise
        return started

    def check(self, started):
        healthy = []
        for recorder in started:
            problem = recorder.problem()
            if problem is not None and recorder not in self.problems:
                self.problems[recorder] = problem
                self.reporter.warn(f"Eingang '{recorder.device['name']}': {problem}")
            if problem is None:
                healthy.append(recorder)
        return [r for r in healthy if r not in self.problems]

    def settle(self, started, reports):
        usable = [(r, rep) for r, rep in zip(started, reports) if rep["seconds"] > 0 and r.path.exists()]
        if self.mixing and len(usable) == len(self.recorders) and not self.problems:
            try:
                Mixer.mix([r.path for r, _ in usable], self.mix_path)
                return self.mix_path
            except Exception as exc:
                self.mix_path.unlink(missing_ok=True)
                self.reporter.warn(f"Mix nicht moeglich ({Reason.of(exc)}), es wird eine Einzelaufnahme verwendet.")
        best = max(usable, key=lambda pair: (pair[0] not in self.problems, pair[1]["seconds"]), default=None)
        return best[0].path if best else None

    def run(self, seconds=None, announce=None, stop=None):
        recorders = self.recorders or self.prepare()
        self.problems, self.final = {}, None
        with Interrupt() as interrupt:
            started = self.begin(recorders)
            try:
                if announce:
                    announce()
                deadline = time.monotonic() + seconds if seconds else None
                try:
                    while not interrupt.wait(RECORDING["poll_s"]) and not (stop and stop.is_set()) and (deadline is None or time.monotonic() < deadline):
                        if not self.check(started):
                            break
                except KeyboardInterrupt:
                    pass
            finally:
                self.check(started)
                reports = [r.stop() for r in started]
                self.check(started)
            self.final = self.settle(started, reports)
        return [{**report, "problem": self.problems.get(r)} for r, report in zip(started, reports)]


class RunLock:
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def _lock(self, take):
        self.file.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK if take else msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if take else fcntl.LOCK_UN)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a+b")
        try:
            self._lock(True)
        except OSError:
            self.file.close()
            raise PipelineError(
                "Es läuft bereits eine Aufnahme oder Verarbeitung (nach hartem Beenden der Oberfläche läuft die alte noch kurz weiter). "
                "Bitte einen Moment warten und dann erneut versuchen."
            ) from None
        return self

    def __exit__(self, exc_type, exc, tb):
        with suppress(OSError):
            self._lock(False)
        self.file.close()
        return False


class Heartbeat:
    def __init__(self, reporter, interval=None, grace=None, interrupt=_thread.interrupt_main, leave=os._exit):
        self.reporter = reporter
        self.interval = START["heartbeat_s"] if interval is None else interval
        self.grace = START["abort_grace_s"] if grace is None else grace
        self.interrupt = interrupt
        self.leave = leave
        self.done = threading.Event()
        self.thread = None

    def _run(self):
        while not self.done.wait(self.interval):
            self.reporter.event("alive")
            if self.reporter.broken:
                self.interrupt()
                if not self.done.wait(self.grace):
                    self.leave(130)
                return

    def __enter__(self):
        if self.reporter.machine:
            if self.reporter.broken:
                raise PipelineError("Die Oberfläche ist nicht mehr erreichbar, die Verarbeitung wird nicht gestartet. Das Audio bleibt erhalten.")
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.done.set()
        if self.thread is not None:
            self.thread.join()
        return False


class StartFlow:
    def __init__(self, store, audio=None, mic=None, device="auto", title=None, speakers=None, diarize=True,
                 open_report=True, tty=False, models=None, picker=None, reporter=None, stdin_stop=False,
                 backend="auto", model=None, fallback=True, hardware=None):
        self.reporter = reporter or Reporter()
        self.store = store
        self.audio = Path(audio) if audio else None
        self.mic = mic
        self.device = (device or "auto").lower()
        self.backend = (backend or "auto").lower()
        self.model = model
        self.fallback = fallback
        self.hardware = hardware
        self.title = title
        self.speakers = speakers
        self.diarize = diarize
        self.open_report = open_report and not self.reporter.machine
        self.tty = tty and not self.reporter.machine
        self.models = models or Models()
        self.picker = picker
        self.watcher = StdinWatcher() if stdin_stop else None
        self.session = None
        self.job = None
        self.result = None

    def retry_command(self, audio):
        parts = [START["program"], "start", str(audio)]
        for flag, value in (("--backend", self.backend), ("--device", self.device), ("--model", self.model), ("--title", self.title), ("--speakers", self.speakers)):
            if value is not None and value != "auto":
                parts += [flag, str(value)]
        parts += [
            flag for flag, active in (
                ("--no-fallback", not self.fallback), ("--no-diarize", not self.diarize), ("--no-open", not self.open_report), ("--events", self.reporter.machine),
            ) if active
        ]
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
        self.reporter.event("preflight")
        if self.audio is not None and not self.audio.is_file():
            raise AudioError(f"Audiodatei nicht gefunden: {self.audio}")
        if self.title and len(self.title) > START["title_chars"]:
            raise PipelineError(f"Der Titel ist zu lang ({len(self.title)} Zeichen, höchstens {START['title_chars']}).")
        AudioLoader.require()
        planned = self.audio or self.recording_path()
        self.job = TranscribeJob(
            self.store, planned, self.device, SETTINGS["language"], models=self.models, model=self.model, title=self.title,
            diarize=self.diarize, speakers=self.speakers, reporter=self.reporter, backend=self.backend,
            fallback=self.fallback, hardware=self.hardware,
        )
        self.job.check()
        if self.audio is not None:
            return
        try:
            picker = self.picker or AudioDevices.sounddevice_picker()
        except Exception as exc:
            raise DeviceError(f"Audio-Eingaenge nicht lesbar: {Reason.of(exc)}") from exc
        self.session = RecordingSession(self.mic or picker.auto(), planned, picker, self.reporter)
        for recorder in self.session.prepare():
            self.reporter.say(f"Eingang: {recorder.device['name']} [{recorder.hostapi}]")

    def announce(self):
        self.reporter.say(RECORDING["lid_notice"])
        self.reporter.say("Aufnahme laeuft, Ende mit Strg+C.")
        names = " + ".join(r.device["name"] for r in self.session.recorders)
        self.reporter.event("recording", input=names, path=str(self.session.mix_path if self.session.mixing else self.session.out))
        if self.watcher:
            self.watcher.start()

    def record(self):
        reports = self.session.run(announce=self.announce, stop=self.watcher)
        for report in reports:
            if report["overruns"]:
                self.reporter.say(f"Ueberlaeufe: {report['overruns']} ({report['path']})")
                self.reporter.event("warning", message=f"Ueberlaeufe: {report['overruns']} ({report['path']})")
        path = self.session.final
        if path is None:
            details = "; ".join(f"{r['path']}: {r['problem'] or 'leer'}" for r in reports)
            raise PipelineError(f"Keine brauchbare Aufnahme gespeichert ({details}). Vorhandene Dateien bleiben erhalten.")
        used = reports if path == self.session.mix_path else [r for r in reports if Path(r["path"]) == path]
        self.reporter.say(f"Aufnahme gespeichert: {path} ({max(r['seconds'] for r in used):.0f} s)")
        if self.session.problems:
            self.reporter.warn(f"Aufnahme unvollstaendig, transkribiert wird nur {path}.")
        return path

    def fail(self, audio, exc, saved=None):
        interrupted = isinstance(exc, KeyboardInterrupt)
        stage = "Nachbearbeitung" if saved else "Verarbeitung"
        message = f"{stage} abgebrochen." if interrupted else f"{stage} fehlgeschlagen: {Reason.of(exc)}"
        retry = f"{START['program']} report {saved}" if saved else self.retry_command(audio)
        self.reporter.say(message, err=True)
        self.reporter.say(f"Audio erhalten: {audio}", err=True)
        if saved:
            self.reporter.say(f"Besprechung gespeichert: {saved}", err=True)
            self.reporter.say(f"Protokoll erneut erzeugen: {retry} (nicht erneut start, das legt eine zweite Besprechung an)", err=True)
        else:
            self.reporter.say(f"Erneut versuchen: {retry}", err=True)
        code = 130 if interrupted else 1
        self.reporter.event("error", message=message, detail=Reason.detail(exc), audio=str(audio), meeting_id=saved, retry=retry, exit_code=code)
        raise click.exceptions.Exit(code)

    def process(self, audio):
        self.job.audio = Path(audio)
        self.job.requested_id = Naming.meeting_id(self.store, self.job.day, self.title or self.job.audio.stem)
        started = time.monotonic()
        self.reporter.event("transcribing", audio=str(audio), meeting_id=self.job.requested_id, **self.job.announce())
        try:
            with Heartbeat(self.reporter):
                self.result = self.job.run()
            if not self.result["segments"]:
                raise PipelineError("Keine Sprache erkannt, kein Protokoll erstellt")
        except (Exception, KeyboardInterrupt) as exc:
            self.fail(audio, exc)
        result = self.result
        self.reporter.say(
            f"{result['meeting_id']}: {result['segments']} Segmente, {result['speakers']} Sprecher, "
            f"Geraet: {' -> '.join(result['used']) or '-'}, {result['ran']['text']}, {time.monotonic() - started:.0f} s"
        )
        return result["meeting_id"]

    def samples(self, meeting_id):
        _, rows = Query(self.store).execute(
            "SELECT speaker, arg_max(text, end_s - start_s) FROM segments WHERE meeting_id = ? AND speaker <> ? "
            "GROUP BY speaker ORDER BY speaker",
            [meeting_id, SETTINGS["unknown_speaker"]],
        )
        return [(label, " ".join(text.split())[: START["sentence_chars"]]) for label, text in rows]

    def name_speakers(self, meeting_id):
        if not self.tty:
            return
        try:
            for label, text in self.samples(meeting_id):
                click.echo(f"{label}: \"{text}\"")
                name = click.prompt(f"Name fuer {label} (Enter = ueberspringen)", default="", show_default=False).strip()
                if name:
                    self.store.speakers.set(meeting_id, label, name)
        except click.Abort:
            click.echo()

    def write_report(self, meeting_id):
        text = Report(Query(self.store)).render(meeting_id)
        path = ReportFile().write(self.job.day, Naming.slug(self.title or self.job.audio.stem), text)
        self.reporter.say(f"Protokoll: {path}")
        return path

    def rotate(self):
        try:
            result = self.store.rotate(older_than_days=START["rotate_days"])
        except Exception as exc:
            self.reporter.warn(f"Rotation fehlgeschlagen: {Reason.of(exc)}")
            return
        for skipped in result["skipped"]:
            if skipped["reason"].startswith("error:"):
                self.reporter.warn(f"Rotation von {skipped['meeting_id']} fehlgeschlagen: {skipped['reason'][len('error:'):].strip()}")

    def show(self, path):
        if not (self.open_report and self.tty and hasattr(os, "startfile")):
            return
        try:
            os.startfile(path)
        except Exception as exc:
            self.reporter.warn(f"Protokoll konnte nicht geoeffnet werden ({path}): {Reason.of(exc)}")

    def execute(self):
        self.preflight()
        audio = self.record() if self.audio is None else self.audio
        meeting_id = self.process(audio)
        try:
            self.name_speakers(meeting_id)
            path = self.write_report(meeting_id)
            self.show(path)
            self.rotate()
            self.reporter.event(
                "done", meeting_id=meeting_id, report=str(path), audio=str(audio), segments=self.result["segments"],
                speakers=self.result["speakers"], ran=self.result["ran"], labels=[{"label": l, "sample": t} for l, t in self.samples(meeting_id)],
            )
        except (Exception, KeyboardInterrupt) as exc:
            self.fail(audio, exc, saved=meeting_id)
        return path

    def run(self):
        try:
            with RunLock(self.store.config["data_dir"] / START["lock_file"]):
                return self.execute()
        except click.exceptions.Exit:
            raise
        except (Exception, KeyboardInterrupt) as exc:
            if not self.reporter.machine:
                raise
            code = 130 if isinstance(exc, KeyboardInterrupt) else getattr(exc, "exit_code", 1)
            message = exc.message if isinstance(exc, click.ClickException) else Reason.of(exc)
            self.reporter.event("error", message=message, detail=Reason.detail(exc), audio=None, meeting_id=None, retry=None, exit_code=code)
            raise click.exceptions.Exit(code) from exc


class SetupCheck:
    def __init__(self, models=None, token=None):
        self.models = models or Models()
        self.token = os.environ.get(SETTINGS["diarization"]["token_env"]) if token is None else token

    @staticmethod
    def row(key, label, ok, fix, scope=None):
        return {"id": key, "label": label, "ok": bool(ok), "fix": None if ok else fix, "scope": scope}

    def run(self):
        diar = SETTINGS["diarization"]
        rows = [self.row("ffmpeg", "ffmpeg", shutil.which("ffmpeg"), "ffmpeg installieren und in den PATH aufnehmen.")]
        dirs = {}
        backend, model = CATALOG["backend_order"][0], CATALOG["default_model"]
        for kind in CATALOG["hardware_order"]:
            entry = Catalog.entry(backend, model, Catalog.spec(backend, kind)["default"])
            dirs.setdefault(Catalog.dir_name(entry), []).append(Catalog.label(kind))
        for name, devices in dirs.items():
            ready = self.models.ready(backend, self.models.root / name)
            rows.append(self.row(f"whisper:{name}", f"Whisper-Modell ({'/'.join(devices)})", ready, f"'meetscribe setup' ausführen; es lädt {name} nach {self.models.root}."))
        ready = self.models.diarization_ready()
        rows.append(self.row("pyannote", "pyannote-Modell (Sprechertrennung)", ready, f"Schnelle Lösung: in den Einstellungen „Sprecher trennen“ ausschalten. Dauerhaft: {diar['token_env']} setzen und 'meetscribe setup' ausführen.", "diarize"))
        rows.append(self.row("hf_token", f"{diar['token_env']} (nur für 'meetscribe setup')", ready or self.token, f"Nur für die Einrichtung der Sprechertrennung: {diar['token_env']} setzen (Zugriff auf huggingface.co/{diar['repo']} freischalten) und 'meetscribe setup' ausführen.", "diarize"))
        return rows


class AudioDevices:
    @staticmethod
    def sounddevice_picker():
        import sounddevice as sd
        return InputPicker([dict(d) for d in sd.query_devices()], [dict(h) for h in sd.query_hostapis()])

    @staticmethod
    def hardware_rows():
        return Hardware().rows

    @staticmethod
    def openvino_devices():
        core = backends.QuietImport.module("openvino").Core()
        rows = []
        for name in core.available_devices:
            try:
                full = core.get_property(name, "FULL_DEVICE_NAME")
            except Exception:
                full = ""
            rows.append({"device": name, "name": full})
        return rows
