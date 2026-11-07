import gc
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from importlib import metadata
from inspect import signature
from pathlib import Path

import click
import duckdb

from meetscribe import backends, pipeline
from meetscribe.backends import BACKEND_CHOICES, CATALOG, Catalog, Hardware, Selection
from meetscribe.store import Json, MeetingNotFound, Query, Report, Store, make_config

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYANNOTE_METRICS_ENABLED": "0",
}
SERVICE = {
    "protocol": 1,
    "name": "meetscribe-engine",
    "idle_release_s": 600.0,
    "workers": 8,
    "discard_tries": 4,
    "discard_wait_s": 0.3,
    "stderr_tail": 12,
    "shutdown_wait_s": 5.0,
    "choice": r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    "setup_command": [sys.executable, "-c", "from meetscribe.cli import main; main()"],
}
CODES = {"parse": -32700, "request": -32600, "method": -32601, "params": -32602, "internal": -32603, "failed": -32000, "protocol": -32001, "uninitialized": -32002, "busy": -32010}
TEXT = {
    "still_running": "Die frühere Verarbeitung läuft vermutlich noch. Bitte einen Moment warten und es dann erneut versuchen.",
    "session_busy": "Es läuft bereits eine Sitzung.",
    "download_blocks_run": "Während eines Modell-Downloads startet keine Aufnahme oder Verarbeitung.",
    "run_blocks_download": "Während einer Aufnahme oder Verarbeitung wird nichts heruntergeladen.",
    "download_busy": "Es läuft bereits ein Download.",
    "no_recording": "Es läuft keine Aufnahme.",
    "no_run": "Keine Sitzung aktiv.",
    "client_gone": "Die Oberfläche ist nicht mehr erreichbar.",
    "protocol": "Engine und Oberfläche sprechen verschiedene Protokollversionen (Engine: {engine}, Oberfläche: {client}). Bitte beide auf denselben Stand bringen.",
    "uninitialized": "Die Verbindung ist nicht eingerichtet: zuerst initialize aufrufen.",
    "method": "Unbekannte Methode: {method}",
    "params": "Ungültige Parameter für {method}: {reason}",
    "request": "Ungültige Anfrage.",
    "parse": "Die Anfrage ist kein gültiges JSON.",
    "internal": "Interner Fehler der Engine: {reason}",
    "choice": "Ungültige Angabe bei {label}: {value}",
    "no_meeting": "Keine Besprechung vorhanden.",
    "unknown_meeting": "Unbekannte meeting_id: {meeting_id}",
    "complete_kept": "{meeting_id} ist vollstaendig und wird nicht verworfen.",
    "discard_failed": "{meeting_id} konnte nicht verworfen werden: {reason}",
    "unknown_label": "Unbekanntes Label '{label}'. Vorhanden: {labels}",
    "no_mapping": "Keine Zuordnung fuer {label} in {meeting_id}.",
    "audio_missing": "Die Audiodatei existiert nicht oder ist keine Datei: {audio}",
}
RUN_OPTIONS = ("audio", "title", "mic", "device", "backend", "model", "diarize", "speakers", "fallback")
TRANSCRIPT = {"default_rows": 200, "max_rows": 1000}
SPEAKERS_SQL = (
    "SELECT speaker AS label, any_value(display_name) AS name, sum(end_s - start_s) AS talk_s, count(*) AS segments "
    "FROM segments_named WHERE meeting_id = ? GROUP BY speaker ORDER BY speaker"
)


class ServiceError(pipeline.PipelineError):
    def __init__(self, message, detail=None, kind="failed"):
        super().__init__(message, detail)
        self.kind = kind


class RpcError(Exception):
    def __init__(self, kind, message, data=None):
        super().__init__(message)
        self.code = CODES[kind]
        self.message = message
        self.data = data


def hardware_report(hardware):
    rows = []
    for kind in CATALOG["hardware_order"]:
        found = [r for r in hardware.rows if r["id"] == kind]
        rows.append({
            "id": kind, "label": Catalog.label(kind), "present": bool(found), "name": found[0]["name"] if found else None,
            "type": found[0]["type"] if found else None, "devices": [r["device"] for r in found],
            "backends": sorted({b for r in found for b in r["backends"]}), "reason": None if found else CATALOG["hardware"][kind]["absent"],
        })
    return rows


def backend_report():
    return [{"id": b, "label": Catalog.backend_label(b), "installed": Catalog.installed(b), "reason": None if Catalog.installed(b) else Catalog.missing_reason(b)} for b in CATALOG["backend_order"]]


def engine_version():
    try:
        return metadata.version("meetscribe")
    except metadata.PackageNotFoundError:
        return "unbekannt"


class EngineCache:
    def __init__(self, idle_s=None):
        self.idle_s = SERVICE["idle_release_s"] if idle_s is None else idle_s
        self.items = {}
        self.lock = threading.RLock()
        self.timer = None
        self.busy = 0
        self.hits = 0

    @staticmethod
    def key(step, language):
        return (step.backend, str(step.path), step.row["device"], step.row.get("cuda"), step.variant, language, tuple(sorted(step.options.items())))

    def get(self, step, models, language):
        key = self.key(step, language)
        with self.lock:
            engine = self.items.get(key)
            if engine is not None:
                self.hits += 1
                return engine
        engine = backends.ENGINES[step.backend](step, models, language)
        with self.lock:
            self.items[key] = engine
        return engine

    def drop(self, step, models, language):
        with self.lock:
            self.items.pop(self.key(step, language), None)
        gc.collect()

    def _cancel(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def hold(self):
        with self.lock:
            self.busy += 1
            self._cancel()

    def settle(self):
        with self.lock:
            self.busy = max(self.busy - 1, 0)
            if self.busy or not self.items:
                return
            if self.idle_s <= 0:
                return self.release()
            self._cancel()
            self.timer = threading.Timer(self.idle_s, self.release)
            self.timer.daemon = True
            self.timer.start()

    def release(self):
        with self.lock:
            if self.busy:
                return
            self.items.clear()
            self._cancel()
        gc.collect()

    @property
    def loaded(self):
        return len(self.items)


class RunState:
    def __init__(self, ident, options):
        self.id = ident
        self.options = options
        self.stop = pipeline.StopSignal()
        self.cancel = pipeline.StopSignal()
        self.stage = "preflight"
        self.since = time.monotonic()
        self.audio = self.input = self.meeting_id = None
        self.warnings = []
        self.terminal = False
        self.code = None
        self.finished = threading.Event()

    @property
    def active(self):
        return not self.finished.is_set()

    def absorb(self, name, data):
        stage = self.stage
        if name == "recording":
            stage, self.input, self.audio = "recording", data.get("input"), data.get("path") or self.audio
        elif name == "transcribing":
            stage, self.audio, self.meeting_id = "processing", data.get("audio") or self.audio, data.get("meeting_id")
        elif name == "warning":
            self.warnings.append(data.get("message"))
        elif name in ("done", "error"):
            stage, self.terminal = "finished", True
        if stage != self.stage:
            self.stage, self.since = stage, time.monotonic()

    def finish(self, code):
        self.code, self.stage = code, "finished"
        self.finished.set()

    def snapshot(self):
        return {
            "run": self.id, "active": self.active, "stage": self.stage, "elapsed_s": int(time.monotonic() - self.since),
            "audio": self.audio, "input": self.input, "meeting_id": self.meeting_id, "warnings": list(self.warnings),
        }


class RunReporter(pipeline.EventReporter):
    def __init__(self, run, wire, method="run.event"):
        self.run, self.wire, self.method = run, wire, method

    @property
    def broken(self):
        return self.wire.closed

    def event(self, name, **data):
        self.run.absorb(name, data)
        if not self.broken:
            self.wire.notify(self.method, {"run": self.run.id, "event": name, **data})


class Download:
    def __init__(self, ident, backend, model):
        self.id = ident
        self.command = [*SERVICE["setup_command"], "setup", "--events", "--backend", backend, "--model", model]
        self.child = None
        self.tail = deque(maxlen=SERVICE["stderr_tail"])
        self.terminal = False
        self.finished = threading.Event()

    @property
    def active(self):
        return not self.finished.is_set()

    def start(self, wire):
        env = {k: v for k, v in os.environ.items() if k not in OFFLINE_ENV}
        self.child = subprocess.Popen(
            self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, encoding="utf-8", errors="replace",
        )
        threading.Thread(target=self._relay, args=(wire,), daemon=True, name="download").start()

    def _drain(self):
        for line in self.child.stderr:
            self.tail.append(line.rstrip())
            print(line.rstrip(), file=sys.stderr, flush=True)

    def _relay(self, wire):
        errors = threading.Thread(target=self._drain, daemon=True)
        errors.start()
        for line in self.child.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and isinstance(event.get("event"), str):
                self.terminal |= event["event"] in ("done", "error")
                wire.notify("download.event", {"run": self.id, **event})
        code = self.child.wait()
        errors.join()
        self.finished.set()
        wire.notify("download.exit", {"run": self.id, "code": code, "terminal": self.terminal, "stderr": "\n".join(self.tail)})

    def kill(self):
        if self.child is not None and self.child.poll() is None:
            self.child.kill()


class Service:
    def __init__(self, store=None, engines=None, sandbox=False):
        self.sandbox = sandbox
        self.store = store or Store(make_config())
        self.engines = engines or pipeline.FreshEngines()
        self.lock = threading.RLock()
        self.slot = threading.Lock()
        self.query = None
        self.hardware = None
        self.run = None
        self.download = None
        self.counter = 0

    @contextmanager
    def reading(self):
        with self.lock:
            if self.query is None:
                self.query = Query(self.store, sandbox=self.sandbox)
            yield self.query

    def warm(self):
        with suppress(Exception):
            self.meetings()

    def require(self, meeting_id=None):
        if meeting_id is None:
            with self.reading() as query:
                meeting_id = query.latest_meeting()
            if meeting_id is None:
                raise ServiceError(TEXT["no_meeting"])
            return meeting_id
        try:
            known = self.store.exists(meeting_id)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        if not known:
            raise ServiceError(TEXT["unknown_meeting"].format(meeting_id=meeting_id))
        return meeting_id

    def meetings(self):
        with self.reading() as query:
            return query.meetings()

    def search(self, term, meeting=None, speaker=None, limit=20):
        with self.reading() as query:
            return query.search(term, meeting_id=meeting, speaker=speaker, limit=limit)

    def stats(self, meeting=None):
        with self.reading() as query:
            return query.speaker_stats(meeting)

    def sql(self, query):
        import duckdb
        try:
            with self.reading() as reader:
                columns, rows = reader.execute(query)
        except duckdb.Error as exc:
            raise ServiceError(str(exc)) from exc
        return {"columns": list(columns), "rows": [list(r) for r in rows]}

    def select(self, query, limit=None):
        try:
            with self.reading() as reader:
                columns, rows, more = reader.select(query, limit)
        except duckdb.Error as exc:
            raise ServiceError(str(exc)) from exc
        return {"columns": list(columns), "rows": [list(r) for r in rows], "truncated": more}

    def transcript(self, meeting_id=None, speaker=None, offset=0, limit=None):
        meeting_id = self.require(meeting_id)
        limit = min(max(int(limit or TRANSCRIPT["default_rows"]), 1), TRANSCRIPT["max_rows"])
        offset = max(int(offset or 0), 0)
        with self.reading() as query:
            info = next(m for m in query.meetings() if m["meeting_id"] == meeting_id)
            where = "meeting_id = ?" + (" AND (speaker = ? OR display_name = ?)" if speaker else "")
            params = [meeting_id, *([speaker, speaker] if speaker else [])]
            total = query.execute(f"SELECT count(*) FROM segments_named WHERE {where}", params)[1][0][0]
            segments = query._records(
                f"SELECT seg_id, start_s, end_s, speaker AS label, display_name AS speaker, text FROM segments_named WHERE {where} "
                "ORDER BY start_s, seg_id LIMIT ? OFFSET ?", [*params, limit, offset],
            )
        return {
            "meeting_id": meeting_id, "title": info["title"], "date": info["meeting_date"], "complete": info["complete"],
            "total": total, "offset": offset, "next_offset": offset + len(segments) if offset + len(segments) < total else None, "segments": segments,
        }

    def speakers_show(self, meeting_id=None):
        meeting_id = self.require(meeting_id)
        with self.reading() as query:
            columns, rows = query.execute(SPEAKERS_SQL, [meeting_id])
        return [dict(zip(columns, row)) for row in rows]

    def labels(self, meeting_id):
        with self.reading() as query:
            return [r[0] for r in query.execute("SELECT DISTINCT speaker FROM segments WHERE meeting_id = ? ORDER BY 1", [meeting_id])[1]]

    def speakers_set(self, label, name, meeting_id=None):
        meeting_id = self.require(meeting_id)
        labels = self.labels(meeting_id)
        if label not in labels:
            raise ServiceError(TEXT["unknown_label"].format(label=label, labels=", ".join(labels)))
        try:
            with self.lock:
                self.store.speakers.set(meeting_id, label, name)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        return {"meeting_id": meeting_id, "label": label, "name": name.strip()}

    def speakers_unset(self, label, meeting_id=None):
        meeting_id = self.require(meeting_id)
        with self.lock:
            removed = self.store.speakers.unset(meeting_id, label)
        if not removed:
            raise ServiceError(TEXT["no_mapping"].format(label=label, meeting_id=meeting_id))
        return {"meeting_id": meeting_id, "label": label}

    def report(self, meeting_id=None, out=None):
        meeting_id = self.require(meeting_id)
        try:
            with self.reading() as query:
                text = Report(query).render(meeting_id)
        except MeetingNotFound as exc:
            raise ServiceError(str(exc)) from exc
        target = Path(out) if out else None
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return {"meeting_id": meeting_id, "path": str(target) if target else None, "markdown": text}

    def rotate(self, older_than_days=0, force=False):
        with self.lock:
            return self.store.rotate(older_than_days=older_than_days, force=force)

    def meeting_state(self, meeting_id):
        try:
            known = self.store.exists(meeting_id)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        if not known:
            return {"meeting_id": meeting_id, "state": "absent"}
        with self.reading() as query:
            return {"meeting_id": meeting_id, "state": "partial" if query.is_incomplete(meeting_id) else "complete"}

    def discard(self, meeting_id):
        meeting_id = self.require(meeting_id)
        if meeting_id not in self.store.incomplete():
            raise ServiceError(TEXT["complete_kept"].format(meeting_id=meeting_id))
        try:
            with self.lock:
                self.store.discard(meeting_id)
        except OSError as exc:
            raise ServiceError(TEXT["discard_failed"].format(meeting_id=meeting_id, reason=exc)) from exc
        return {"meeting_id": meeting_id}

    def settle(self, meeting_id):
        state = self.meeting_state(meeting_id)
        if state["state"] != "partial":
            return state
        run = self.run
        if run is not None and run.active and run.meeting_id == meeting_id:
            raise ServiceError(TEXT["still_running"], kind="busy")
        for _ in range(SERVICE["discard_tries"]):
            try:
                self.discard(meeting_id)
                return {"meeting_id": meeting_id, "state": "discarded"}
            except ServiceError:
                time.sleep(SERVICE["discard_wait_s"])
        raise ServiceError(TEXT["still_running"], kind="busy")

    def devices(self):
        inputs, error = [], None
        try:
            busy = self.run is not None and self.run.stage == "recording"
            picker = pipeline.AudioDevices.sounddevice_picker(refresh=False) if busy else pipeline.AudioDevices.sounddevice_picker()
            inputs = [
                {"index": d["index"], "name": d["name"], "api": picker.hostapis[d["hostapi"]], "channels": d["max_input_channels"], "rate": int(d["default_samplerate"])}
                for d in picker.inputs()
            ]
        except Exception as exc:
            error = f"Audio-Eingaenge nicht lesbar: {pipeline.Reason.of(exc)}"
        self.hardware = Hardware()
        return {
            "inputs": inputs, "inputs_error": error, "openvino": pipeline.AudioDevices.openvino_devices(),
            "hardware": self.hardware.rows, "kinds": hardware_report(self.hardware), "backends": backend_report(),
        }

    def models(self):
        hardware = self.hardware or Hardware()
        selection = Selection(pipeline.Models(), hardware)
        return {
            "default_model": CATALOG["default_model"], "backends": backend_report(), "hardware": hardware_report(hardware),
            "models": selection.listing(), "choices": selection.matrix(),
        }

    def check(self):
        return pipeline.SetupCheck().run()

    def flow(self, **options):
        return pipeline.StartFlow(self.store, engines=self.engines, hardware=self.hardware, **options)

    @staticmethod
    def choice(label, value):
        if value is not None and not re.match(SERVICE["choice"], str(value)):
            raise ServiceError(TEXT["choice"].format(label=label, value=value))
        return value

    def claim(self, kind, build):
        with self.slot:
            if self.run is not None and self.run.active:
                raise ServiceError(TEXT["session_busy"] if kind == "run" else TEXT["run_blocks_download"], kind="busy")
            if self.download is not None and self.download.active:
                raise ServiceError(TEXT["download_blocks_run"] if kind == "run" else TEXT["download_busy"], kind="busy")
            self.counter += 1
            item = build(self.counter)
            setattr(self, kind, item)
            return item

    def begin_run(self, options, wire, submit):
        options = {k: v for k, v in options.items() if k in RUN_OPTIONS}
        for label, key in (("Backend", "backend"), ("Modell", "model")):
            if options.get(key) not in (None, "auto"):
                self.choice(label, options[key])
        run = self.claim("run", lambda ident: RunState(ident, options))
        self.engines.hold()
        submit(lambda: self.execute(run, wire))
        return {"run": run.id}

    def execute(self, run, wire):
        reporter = RunReporter(run, wire)
        flow, code = None, 0
        try:
            flow = self.flow(reporter=reporter, stop=run.stop, cancel=run.cancel, **run.options)
            with pipeline.KeepAwake():
                flow.run()
        except click.exceptions.Exit as exc:
            code = exc.exit_code
        except BaseException as exc:
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            if not run.terminal:
                reporter.event("error", message=pipeline.Reason.of(exc), detail=pipeline.Reason.detail(exc), audio=run.audio, meeting_id=None, retry=None, exit_code=code)
        finally:
            if flow is not None and flow.job is not None and flow.job.hardware is not None:
                self.hardware = flow.job.hardware
            run.finish(code)
            wire.notify("run.exit", {"run": run.id, "code": code, "terminal": run.terminal})
            self.engines.settle()
        return code

    def stop_run(self):
        run = self.run
        if run is None or not run.active:
            raise ServiceError(TEXT["no_run"])
        if run.stage != "recording":
            raise ServiceError(TEXT["no_recording"])
        run.stop.set()
        return {"run": run.id}

    def cancel_run(self):
        run = self.run
        if run is None or not run.active:
            raise ServiceError(TEXT["no_run"])
        run.cancel.set()
        run.stop.set()
        return {"run": run.id}

    def begin_transcription(self, audio, wire, submit, **options):
        path = Path(audio)
        if not path.is_file():
            raise ServiceError(TEXT["audio_missing"].format(audio=audio))
        options = {k: v for k, v in options.items() if k in RUN_OPTIONS and k not in ("audio", "mic") and v is not None}
        return self.begin_run({**options, "audio": str(path.resolve())}, wire, submit)

    def run_state(self):
        run = self.run
        return run.snapshot() if run is not None else {"active": False, "stage": "finished"}

    def begin_download(self, backend, model, wire):
        for label, value in (("Backend", backend), ("Modell", model)):
            if value == "auto":
                raise ServiceError(TEXT["choice"].format(label=label, value=value))
            self.choice(label, value)
        download = self.claim("download", lambda ident: Download(ident, backend, model))
        download.start(wire)
        return {"run": download.id}

    def download_state(self):
        download = self.download
        return {"active": download is not None and download.active}

    def close(self):
        if self.download is not None:
            self.download.kill()
        if self.run is not None:
            self.run.stop.set()
            self.run.cancel.set()
        self.engines.release()


class Wire:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.lock = threading.Lock()
        self.closed = False

    @classmethod
    def stdio(cls):
        writer = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        return cls(sys.stdin.buffer, writer)

    def send(self, message):
        data = (Json.dumps(message) + "\n").encode("ascii")
        with self.lock:
            if self.closed:
                return False
            try:
                self.writer.write(data)
                self.writer.flush()
            except (OSError, ValueError):
                self.closed = True
            return not self.closed

    def notify(self, method, params):
        return self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def lines(self):
        while True:
            try:
                line = self.reader.readline()
            except (OSError, ValueError):
                return
            if not line:
                return
            if line.strip():
                yield line

    def close(self):
        with self.lock:
            self.closed = True


class RpcServer:
    def __init__(self, service, wire, leave=os._exit):
        self.service = service
        self.wire = wire
        self.leave = leave
        self.pool = ThreadPoolExecutor(SERVICE["workers"], thread_name_prefix="rpc")
        self.jobs = queue.Queue()
        self.ready = False
        self.started = time.monotonic()
        self.methods = {
            "meetings": service.meetings, "search": service.search, "stats": service.stats, "report": service.report, "sql": service.sql,
            "speakers.show": service.speakers_show, "speakers.set": service.speakers_set, "speakers.unset": service.speakers_unset,
            "devices": service.devices, "models": service.models, "check": service.check, "rotate": service.rotate,
            "discard": service.discard, "meeting.state": service.meeting_state, "meeting.settle": service.settle,
            "run.start": self.run_start, "run.stop": service.stop_run, "run.state": service.run_state,
            "download.start": self.download_start, "download.state": service.download_state,
            "ping": self.ping, "shutdown": self.shutdown,
        }

    def run_start(self, **options):
        return self.service.begin_run(options, self.wire, self.jobs.put)

    def download_start(self, backend, model):
        return self.service.begin_download(backend, model, self.wire)

    def ping(self):
        run, download = self.service.run, self.service.download
        return {
            "pid": os.getpid(), "uptime_s": round(time.monotonic() - self.started, 1), "run_active": bool(run and run.active),
            "download_active": bool(download and download.active), "models_loaded": getattr(self.service.engines, "loaded", 0),
        }

    def shutdown(self):
        run = self.service.run
        if run is not None and run.active:
            run.stop.set()
            self.watchdog(SERVICE["shutdown_wait_s"], 0)
        return {"ok": True}

    def watchdog(self, seconds, code):
        timer = threading.Timer(seconds, self.leave, [code])
        timer.daemon = True
        timer.start()

    def initialize(self, protocol=None, client=None):
        if protocol != SERVICE["protocol"]:
            raise RpcError("protocol", TEXT["protocol"].format(engine=SERVICE["protocol"], client=protocol), {"engine": SERVICE["protocol"], "client": protocol})
        self.ready = True
        return {
            "protocol": SERVICE["protocol"], "engine": {"name": SERVICE["name"], "version": engine_version(), "pid": os.getpid()},
            "methods": sorted([*self.methods, "initialize"]),
        }

    def dispatch(self, method, params):
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise RpcError("params", TEXT["params"].format(method=method, reason="Parameter müssen ein Objekt sein."))
        if method == "initialize":
            return self.initialize(**params)
        if not self.ready:
            raise RpcError("uninitialized", TEXT["uninitialized"])
        target = self.methods.get(method)
        if target is None:
            raise RpcError("method", TEXT["method"].format(method=method))
        try:
            signature(target).bind(**params)
        except TypeError as exc:
            raise RpcError("params", TEXT["params"].format(method=method, reason=exc)) from exc
        return target(**params)

    @staticmethod
    def failure(exc):
        if isinstance(exc, RpcError):
            return {"code": exc.code, "message": exc.message, **({"data": exc.data} if exc.data else {})}
        if isinstance(exc, click.ClickException):
            detail = pipeline.Reason.detail(exc)
            return {"code": CODES["busy" if getattr(exc, "kind", "") == "busy" else "failed"], "message": exc.message, "data": {"detail": detail, "kind": getattr(exc, "kind", "failed")}}
        if isinstance(exc, (MeetingNotFound, ValueError)) or type(exc).__module__.startswith("duckdb"):
            return {"code": CODES["failed"], "message": str(exc), "data": {"kind": type(exc).__name__}}
        traceback.print_exception(exc, file=sys.stderr)
        return {"code": CODES["internal"], "message": TEXT["internal"].format(reason=pipeline.Reason.of(exc)), "data": {"detail": pipeline.Reason.detail(exc)}}

    def handle(self, message):
        valid = isinstance(message, dict) and message.get("jsonrpc") == "2.0" and isinstance(message.get("method"), str)
        ident = message.get("id") if isinstance(message, dict) else None
        try:
            if not valid or isinstance(ident, (dict, list, bool)):
                raise RpcError("request", TEXT["request"])
            reply = {"result": self.dispatch(message["method"], message.get("params"))}
        except Exception as exc:
            reply = {"error": self.failure(exc)}
        if not valid or "id" in message:
            self.wire.send({"jsonrpc": "2.0", **reply, "id": ident})
        if valid and message["method"] == "shutdown" and "result" in reply:
            self.jobs.put(None)

    def read(self):
        for line in self.wire.lines():
            try:
                message = json.loads(line)
            except ValueError:
                self.wire.send({"jsonrpc": "2.0", "error": {"code": CODES["parse"], "message": TEXT["parse"]}, "id": None})
                continue
            self.pool.submit(self.handle, message)
        self.gone()

    def gone(self):
        self.wire.close()
        run = self.service.run
        if run is not None and run.active:
            run.stop.set()
            self.watchdog(pipeline.START["abort_grace_s"] + SERVICE["shutdown_wait_s"], 130)
        self.jobs.put(None)

    def serve(self):
        threading.Thread(target=self.read, daemon=True, name="rpc-reader").start()
        threading.Thread(target=self.service.warm, daemon=True, name="warm").start()
        while (job := self.jobs.get()) is not None:
            try:
                job()
            except KeyboardInterrupt:
                pass
            except Exception:
                traceback.print_exc()
        self.service.close()
        self.pool.shutdown(wait=False, cancel_futures=True)
