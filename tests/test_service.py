import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from meetscribe import backends, pipeline, service
from meetscribe.service import EngineCache, RpcServer, Service, ServiceError, Wire
from test_cli import FakeEngine, FakeRecorder, env, transcribe, truncated_meeting  # noqa: F401
from test_store import make_orphan, start_writer

LAUNCHER = Path(__file__).parent / "fake_engine_process.py"


class SlowEngine(FakeEngine):
    gate = None

    def transcribe(self, audio):
        if SlowEngine.gate is not None:
            SlowEngine.gate.wait(15)
        time.sleep(0.3)
        return super().transcribe(audio)


class Peer:
    def __init__(self, reader, writer):
        self.writer = writer
        self.notes = []
        self.replies = {}
        self.cond = threading.Condition()
        self.counter = 0
        threading.Thread(target=self.pump, args=(reader,), daemon=True).start()

    def pump(self, reader):
        for line in reader:
            message = json.loads(line)
            with self.cond:
                if "id" in message and "method" not in message:
                    self.replies[message["id"]] = message
                else:
                    self.notes.append(message)
                self.cond.notify_all()

    def send(self, message):
        self.writer.write((json.dumps(message) + "\n").encode())
        self.writer.flush()

    def post(self, method, **params):
        self.counter += 1
        self.send({"jsonrpc": "2.0", "id": self.counter, "method": method, "params": params})
        return self.counter

    def wait(self, ident, timeout=20):
        with self.cond:
            assert self.cond.wait_for(lambda: ident in self.replies, timeout), f"keine Antwort auf {ident}"
            return self.replies.pop(ident)

    def raw(self, method, **params):
        return self.wait(self.post(method, **params))

    def call(self, method, **params):
        reply = self.raw(method, **params)
        assert "error" not in reply, reply
        return reply["result"]

    def fail(self, method, **params):
        reply = self.raw(method, **params)
        assert "error" in reply, reply
        return reply["error"]

    def until(self, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                found = [n for n in self.notes if predicate(n)]
                if found:
                    return found[0]
                assert self.cond.wait(max(deadline - time.monotonic(), 0.01)) or time.monotonic() < deadline, "Ereignis blieb aus"

    def events(self, method="run.event"):
        with self.cond:
            return [n["params"] for n in self.notes if n.get("method") == method]

    def event_names(self, alive=False):
        return [e["event"] for e in self.events() if alive or e["event"] != "alive"]

    def finished(self, timeout=20):
        return self.until(lambda n: n.get("method") == "run.exit", timeout)["params"]

    def handshake(self):
        return self.call("initialize", protocol=1, client={"name": "test"})


class Harness(Peer):
    def __init__(self, svc):
        to_server, from_client = os.pipe()
        from_server, to_client = os.pipe()
        self.left = []
        self.wire = Wire(os.fdopen(to_server, "rb"), os.fdopen(to_client, "wb"))
        self.server = RpcServer(svc, self.wire, leave=self.left.append)
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()
        super().__init__(os.fdopen(from_server, "rb"), os.fdopen(from_client, "wb"))

    def close(self):
        self.writer.close()
        self.thread.join(10)
        assert not self.thread.is_alive()


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setitem(pipeline.START, "heartbeat_s", 0.05)
    monkeypatch.setitem(backends.ENGINES, "openvino", SlowEngine)
    SlowEngine.gate = None
    yield
    if SlowEngine.gate is not None:
        SlowEngine.gate.set()
    SlowEngine.gate = None


@pytest.fixture
def server(env, fast):
    cache = EngineCache(idle_s=60)
    harness = Harness(Service(env["store"], cache))
    harness.cache = cache
    harness.env = env
    harness.handshake()
    yield harness
    SlowEngine.gate and SlowEngine.gate.set()
    harness.close()


def run_options(env, **extra):
    return {"audio": str(env["audio"]), "device": "auto", "diarize": False, "title": "Runde", **extra}


def test_handshake_must_come_first_and_versions_must_match(env, fast):
    harness = Harness(Service(env["store"]))
    assert harness.fail("meetings")["code"] == -32002
    wrong = harness.fail("initialize", protocol=2)
    assert wrong["code"] == -32001 and "Protokollversionen" in wrong["message"] and wrong["data"] == {"engine": 1, "client": 2}
    assert harness.fail("initialize")["code"] == -32001
    hello = harness.handshake()
    assert hello["protocol"] == 1 and hello["engine"]["name"] == "meetscribe-engine" and hello["engine"]["pid"] == os.getpid()
    assert {"meetings", "search", "report", "stats", "speakers.show", "speakers.set", "devices", "models", "check", "run.start", "run.stop",
            "run.state", "download.start", "meeting.settle", "meeting.state", "discard", "shutdown", "initialize"} <= set(hello["methods"])
    harness.close()


def test_malformed_traffic_gets_standard_errors_and_notifications_get_no_answer(server):
    assert server.fail("gibt.es.nicht")["code"] == -32601
    assert server.fail("search")["code"] == -32602
    assert server.fail("search", term="x", zuviel=1)["code"] == -32602
    for line, ident, code in ((b"das ist kein json\n", None, -32700), (b"[1, 2]\n", None, -32600), (b'{"jsonrpc": "2.0", "method": 5, "id": 77}\n', 77, -32600)):
        server.writer.write(line)
        server.writer.flush()
        with server.cond:
            assert server.cond.wait_for(lambda: ident in server.replies, 10)
            reply = server.replies.pop(ident)
        assert reply["error"]["code"] == code and reply["jsonrpc"] == "2.0"
    server.send({"jsonrpc": "2.0", "method": "ping"})
    assert server.call("ping")["pid"] == os.getpid()
    assert server.replies == {}


def test_queries_match_the_command_line_and_follow_new_and_rotated_files(server):
    env = server.env
    assert server.call("meetings") == []
    assert transcribe(env, "--no-diarize").exit_code == 0
    first = server.call("meetings")
    assert [m["meeting_id"] for m in first] == ["m1"] and first[0]["complete"] is True
    assert transcribe(env, "--no-diarize", meeting_id="m2").exit_code == 0
    assert {m["meeting_id"] for m in server.call("meetings")} == {"m1", "m2"}
    hits = server.call("search", term="münchen", limit=5)
    assert {h["meeting_id"] for h in hits} == {"m1", "m2"} and hits[0]["display_name"] == "UNKNOWN"
    assert server.call("stats", meeting="m1")[0]["segments"] == 1
    server.call("speakers.set", label="UNKNOWN", name="Anna", meeting_id="m1")
    assert server.call("speakers.show", meeting_id="m1")[0]["name"] == "Anna"
    assert "Anna" in server.call("report", meeting_id="m1")["markdown"]
    assert server.call("rotate", older_than_days=0)["rotated"] == ["m1", "m2"]
    assert not env["store"].hot_files() and len(env["store"].cold_files()) == 2
    assert {m["meeting_id"] for m in server.call("meetings")} == {"m1", "m2"}
    assert "Anna" in server.call("report", meeting_id="m1", out=str(env["tmp"] / "x" / "m1.md"))["markdown"]
    assert (env["tmp"] / "x" / "m1.md").read_text(encoding="utf-8").startswith("# Wochenbesprechung")
    assert server.call("sql", query="SELECT count(*) AS n FROM segments") == {"columns": ["n"], "rows": [[2]]}
    assert "gesperrt" in server.fail("sql", query="INSTALL httpfs")["message"]
    assert server.fail("report", meeting_id="gibtsnicht")["message"] == "Unbekannte meeting_id: gibtsnicht"
    assert "invalid meeting_id" in server.fail("report", meeting_id="../x")["message"]
    assert server.call("models")["default_model"] == "large-v3-turbo"
    assert {r["id"] for r in server.call("check")} >= {"ffmpeg", "pyannote"}


def test_devices_report_inputs_and_hardware(server, monkeypatch):
    monkeypatch.setattr(pipeline.AudioDevices, "openvino_devices", staticmethod(lambda: [{"device": "NPU", "name": "Intel NPU"}]))
    report = server.call("devices")
    assert [d["name"] for d in report["inputs"]][0].startswith("Mikrofonarray")
    assert {r["id"] for r in report["hardware"]} == {"cpu", "igpu", "npu"} and report["inputs_error"] is None


def test_a_run_streams_todays_events_and_the_second_run_reuses_the_loaded_engine(server):
    env = server.env
    assert server.call("run.start", **run_options(env)) == {"run": 1}
    exit_one = server.finished()
    assert exit_one == {"run": 1, "code": 0, "terminal": True}
    names = server.event_names()
    assert names[:2] == ["preflight", "transcribing"] and names[-1] == "done" and {e["run"] for e in server.events()} == {1}
    done = server.events()[-1]
    assert done["segments"] == 1 and done["ran"]["load_s"] >= 0 and done["meeting_id"].endswith("-runde")
    assert FakeEngine.constructed == ["npu"] and server.cache.loaded == 1 and server.cache.hits == 0
    server.call("run.start", **run_options(env))
    assert server.until(lambda n: n.get("method") == "run.exit" and n["params"]["run"] == 2)
    assert FakeEngine.constructed == ["npu"] and server.cache.hits == 1
    assert len(server.call("meetings")) == 2
    assert server.call("run.state")["stage"] == "finished"


def test_fallback_and_alive_arrive_as_events_and_a_failed_engine_leaves_the_cache(server):
    FakeEngine.fail_devices = ("npu",)
    server.call("run.start", **run_options(server.env))
    server.finished()
    names = server.event_names(alive=True)
    assert "fallback" in names and "alive" in names and names[-1] == "done"
    fallback = next(e for e in server.events() if e["event"] == "fallback")
    assert (fallback["from"], fallback["to"]) == ("NPU", "iGPU")
    assert server.cache.loaded == 1


def test_queries_are_answered_while_a_transcription_runs_and_a_second_start_is_refused(server):
    SlowEngine.gate = threading.Event()
    server.call("run.start", **run_options(server.env))
    server.until(lambda n: n.get("params", {}).get("event") == "transcribing")
    started = time.monotonic()
    for _ in range(5):
        assert isinstance(server.call("meetings"), list)
        assert server.call("search", term="a", limit=3) is not None
    assert time.monotonic() - started < 5
    assert server.call("ping")["run_active"] is True and server.call("run.state")["stage"] == "processing"
    busy = server.fail("run.start", **run_options(server.env))
    assert busy["code"] == -32010 and busy["message"] == "Es läuft bereits eine Sitzung."
    assert server.fail("download.start", backend="openvino", model="small")["message"].startswith("Während einer Aufnahme oder Verarbeitung")
    held = server.events()[1]["meeting_id"]
    assert server.call("meeting.state", meeting_id=held)["state"] == "partial"
    assert server.fail("meeting.settle", meeting_id=held)["message"].startswith("Die frühere Verarbeitung läuft vermutlich noch")
    SlowEngine.gate.set()
    assert server.finished()["terminal"] is True
    assert server.call("meeting.settle", meeting_id=held)["state"] == "complete"


def test_parallel_requests_each_get_their_own_answer(server):
    transcribe(server.env, "--no-diarize")
    ids = [server.post("search", term="münchen", limit=n) for n in range(1, 21)] + [server.post("ping") for _ in range(10)]
    replies = [server.wait(i) for i in ids]
    assert all("result" in r for r in replies)
    assert [len(r["result"]) for r in replies[:20]] == [1] * 20
    assert all(r["result"]["pid"] == os.getpid() for r in replies[20:])


def test_recording_ends_on_stop_and_stop_outside_a_recording_is_refused(server):
    FakeRecorder.interrupt_with = "none"
    assert server.fail("run.stop")["message"] == "Keine Sitzung aktiv."
    server.call("run.start", device="auto", diarize=False)
    server.until(lambda n: n.get("params", {}).get("event") == "recording")
    assert server.call("run.state")["stage"] == "recording"
    server.call("run.stop")
    assert server.finished()["terminal"] is True
    assert server.event_names() == ["preflight", "recording", "transcribing", "done"]
    assert server.fail("run.stop")["message"] == "Keine Sitzung aktiv."
    assert len(list((server.env["tmp"] / "data" / "aufnahmen").glob("*.flac"))) == 1


def test_run_failures_are_error_events_with_the_audio_kept(server):
    FakeEngine.fail_devices = ("npu", "igpu", "cpu")
    server.call("run.start", **run_options(server.env))
    assert server.finished() == {"run": 1, "code": 1, "terminal": True}
    error = server.events()[-1]
    assert error["event"] == "error" and error["audio"] == str(server.env["audio"]) and error["exit_code"] == 1
    assert server.env["audio"].exists() and server.call("meetings") == []
    assert server.fail("run.start", backend="--x")["message"] == "Ungültige Angabe bei Backend: --x"


def test_the_start_lock_still_refuses_a_second_process(server):
    with pipeline.RunLock(server.env["tmp"] / "data" / pipeline.START["lock_file"]):
        server.call("run.start", **run_options(server.env))
        server.finished()
    error = server.events()[-1]
    assert error["event"] == "error" and "Es läuft bereits eine Aufnahme oder Verarbeitung" in error["message"]


def test_settle_discards_a_partial_meeting_and_never_a_complete_one(server):
    env = server.env
    transcribe(env, "--no-diarize")
    truncated_meeting(env)
    assert server.call("meeting.state", meeting_id="m-teil")["state"] == "partial"
    assert server.call("meeting.state", meeting_id="m1")["state"] == "complete"
    assert server.call("meeting.state", meeting_id="m-weg")["state"] == "absent"
    assert server.call("meeting.settle", meeting_id="m1")["state"] == "complete"
    assert server.call("meeting.settle", meeting_id="m-teil")["state"] == "discarded"
    assert server.call("meeting.settle", meeting_id="m-teil")["state"] == "absent"
    assert "wird nicht verworfen" in server.fail("discard", meeting_id="m1")["message"]
    assert server.call("meetings")[0]["meeting_id"] == "m1"


def download_script(tmp_path, body):
    script = tmp_path / "setup_child.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def test_download_relays_progress_from_a_child_that_may_use_the_network(server, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setitem(service.SERVICE, "setup_command", download_script(server.env["tmp"], (
        "import json, os, sys\n"
        "print('Rauschen')\n"
        "print(json.dumps({'event': 'download', 'repo': 'a/b', 'bytes': 5, 'total': 10, 'args': sys.argv[1:], 'offline': os.environ.get('HF_HUB_OFFLINE')}), flush=True)\n"
        "print('Fortschrittszeile', file=sys.stderr)\n"
        "print(json.dumps({'event': 'done', 'messages': ['ok']}))\n"
    )))
    assert server.fail("download.start", backend="auto", model="small")["code"] == -32000
    assert server.fail("download.start", backend="openvino", model="--x")["code"] == -32000
    assert server.call("download.start", backend="openvino", model="small:int8") == {"run": 1}
    exit_info = server.until(lambda n: n.get("method") == "download.exit")["params"]
    progress = server.events("download.event")
    assert [e["event"] for e in progress] == ["download", "done"]
    assert progress[0]["args"] == ["setup", "--events", "--backend", "openvino", "--model", "small:int8"] and progress[0]["offline"] is None
    assert exit_info["code"] == 0 and exit_info["terminal"] is True and "Fortschrittszeile" in exit_info["stderr"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert server.call("download.state") == {"active": False}


def test_a_download_excludes_runs_and_a_failing_download_reports_its_exit(server, monkeypatch):
    monkeypatch.setitem(service.SERVICE, "setup_command", download_script(server.env["tmp"], (
        "import json, sys, time\n"
        "print(json.dumps({'event': 'error', 'message': 'Der Download ist fehlgeschlagen.', 'detail': 'offline', 'exit_code': 1}), flush=True)\n"
        "time.sleep(1)\n"
        "sys.exit(1)\n"
    )))
    server.call("download.start", backend="openvino", model="small")
    assert server.fail("download.start", backend="openvino", model="small")["message"] == "Es läuft bereits ein Download."
    assert server.fail("run.start", **run_options(server.env))["message"] == "Während eines Modell-Downloads startet keine Aufnahme oder Verarbeitung."
    exit_info = server.until(lambda n: n.get("method") == "download.exit")["params"]
    assert (exit_info["code"], exit_info["terminal"]) == (1, True)
    assert server.call("run.start", **run_options(server.env)) == {"run": 2}
    server.finished()


def test_shutdown_ends_the_serve_loop_after_answering(env, fast):
    harness = Harness(Service(env["store"]))
    harness.handshake()
    assert harness.call("shutdown") == {"ok": True}
    harness.thread.join(10)
    assert not harness.thread.is_alive()


def test_engine_cache_keeps_a_pipeline_until_it_has_been_idle_long_enough(monkeypatch):
    built = []

    class Dummy:
        def __init__(self, step, models, language):
            built.append(step.row["device"])

    monkeypatch.setitem(backends.ENGINES, "openvino", Dummy)
    step = type("Step", (), {"backend": "openvino", "path": Path("m"), "row": {"device": "NPU"}, "variant": "fp16", "options": {}})()
    cache = EngineCache(idle_s=0.3)
    cache.hold()
    first = cache.get(step, None, "de")
    assert cache.get(step, None, "de") is first and built == ["NPU"] and cache.hits == 1
    cache.settle()
    time.sleep(0.1)
    assert cache.loaded == 1
    cache.hold()
    time.sleep(0.5)
    assert cache.loaded == 1
    cache.settle()
    deadline = time.monotonic() + 5
    while cache.loaded and time.monotonic() < deadline:
        time.sleep(0.05)
    assert cache.loaded == 0 and cache.get(step, None, "de") is not first
    cache.drop(step, None, "de")
    assert cache.loaded == 0
    never = EngineCache(idle_s=0)
    never.hold()
    never.get(step, None, "de")
    never.settle()
    assert never.loaded == 0


def spawn(tmp_path, **extra):
    env = {
        **os.environ, "MEETSCRIBE_DATA": str(tmp_path / "data"), "MEETSCRIBE_MODELS": str(tmp_path / "models"),
        "MEETSCRIBE_REPORTS": str(tmp_path / "protokolle"), "PYTHONIOENCODING": "utf-8", **extra,
    }
    for name in ("MEETSCRIBE_DEVICE", "MEETSCRIBE_BACKEND"):
        env.pop(name, None)
    log = open(tmp_path / "engine.err", "wb")
    proc = subprocess.Popen([sys.executable, str(LAUNCHER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env)
    proc.log = log
    peer = Peer(proc.stdout, proc.stdin)
    peer.proc = proc
    return peer


@pytest.fixture
def engine_process(env):
    started = []

    def make(**extra):
        peer = spawn(env["tmp"], **extra)
        started.append(peer)
        peer.handshake()
        return peer

    yield make
    for peer in started:
        if peer.proc.poll() is None:
            peer.proc.kill()
        peer.proc.wait(10)
        for stream in (peer.proc.stdin, peer.proc.stdout, peer.proc.log):
            stream.close()


def audio_run(peer, env):
    return peer.call("run.start", **run_options(env))


def test_foreign_output_on_stdout_never_reaches_the_protocol(engine_process, env):
    peer = engine_process()
    audio_run(peer, env)
    assert peer.finished()["terminal"] is True
    assert peer.event_names()[-1] == "done"
    assert len(peer.call("meetings")) == 1
    assert peer.call("shutdown") == {"ok": True}
    assert peer.proc.wait(15) == 0


def test_a_hard_kill_during_a_run_keeps_the_audio_and_the_next_engine_cleans_up(engine_process, env):
    peer = engine_process(FAKE_SLEEP_S="60")
    audio_run(peer, env)
    meeting_id = peer.until(lambda n: n.get("params", {}).get("event") == "transcribing")["params"]["meeting_id"]
    deadline = time.monotonic() + 20
    while not env["store"].hot_path(meeting_id).exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    peer.proc.kill()
    peer.proc.wait(10)
    assert env["audio"].exists()
    fresh = Harness(Service(env["store"]))
    fresh.handshake()
    assert fresh.call("meeting.state", meeting_id=meeting_id)["state"] == "absent"
    assert fresh.call("meeting.settle", meeting_id=meeting_id)["state"] == "absent"
    assert not env["store"].hot_path(meeting_id).exists() and env["audio"].exists() and fresh.call("meetings") == []
    fresh.close()


def test_the_next_engine_reaps_a_hard_killed_orphan_keeps_the_audio_and_the_retry_gives_one_meeting(engine_process, env):
    orphan = make_orphan(env["store"])
    assert env["store"].incomplete() == [orphan]
    peer = engine_process()
    assert env["store"].incomplete() == [] and peer.call("meeting.state", meeting_id=orphan)["state"] == "absent"
    assert env["audio"].exists()
    audio_run(peer, env)
    assert peer.finished()["terminal"] is True
    meetings = peer.call("meetings")
    assert len(meetings) == 1 and meetings[0]["complete"] is True


def test_an_engine_started_while_another_process_writes_leaves_that_file_alone(engine_process, env):
    writer = start_writer(env["store"])
    try:
        peer = engine_process()
        assert env["store"].hot_path("orphan").exists() and peer.call("meeting.state", meeting_id="orphan")["state"] == "partial"
    finally:
        writer.stdin.close()
        assert writer.wait(30) == 0
        writer.stdout.close()


def test_closing_the_pipe_during_processing_aborts_the_run_and_removes_the_partial_meeting(engine_process, env):
    peer = engine_process(FAKE_SLEEP_S="60", FAKE_GRACE_S="1")
    audio_run(peer, env)
    peer.until(lambda n: n.get("params", {}).get("event") == "transcribing")
    peer.proc.stdin.close()
    assert peer.proc.wait(40) in (0, 130)
    assert env["audio"].exists() and Service(env["store"]).meetings() == []


def test_closing_the_pipe_during_a_recording_saves_the_audio_and_processes_nothing(engine_process, env):
    peer = engine_process()
    peer.call("run.start", device="auto", diarize=False)
    peer.until(lambda n: n.get("params", {}).get("event") == "recording")
    peer.proc.stdin.close()
    assert peer.proc.wait(30) in (0, 130)
    assert len(list((env["tmp"] / "data" / "aufnahmen").glob("*.flac"))) == 1
    assert Service(env["store"]).meetings() == []


def test_the_engine_exits_cleanly_when_its_client_goes_away_while_idle(engine_process):
    peer = engine_process()
    peer.proc.stdin.close()
    assert peer.proc.wait(15) == 0


def test_many_queries_in_a_row_stay_fast_once_warm(engine_process, env):
    transcribe(env, "--no-diarize")
    peer = engine_process()
    peer.call("meetings")
    started = time.perf_counter()
    for _ in range(20):
        peer.call("meetings")
        peer.call("search", term="münchen", limit=50)
    assert (time.perf_counter() - started) / 40 < 0.25


class CollectWire:
    closed = False

    def __init__(self):
        self.notes = []

    def notify(self, method, params):
        self.notes.append((method, params))
        return True

    def events(self):
        return [p for m, p in self.notes if m == "run.event" and p["event"] != "alive"]


def in_thread(job):
    threading.Thread(target=job, daemon=True).start()


def test_cancel_stops_processing_between_windows_and_leaves_no_partial_meeting(env, fast, monkeypatch):
    monkeypatch.setitem(pipeline.SETTINGS, "window_s", 10.0)
    monkeypatch.setattr(pipeline.AudioLoader, "load", staticmethod(lambda path, rate=None: np.full(16000 * 60, 0.1, dtype="float32")))
    SlowEngine.gate = threading.Event()
    svc, wire = Service(env["store"], EngineCache(idle_s=60)), CollectWire()
    run = svc.begin_transcription(env["audio"], wire, in_thread, diarize=False)
    deadline = time.monotonic() + 15
    while svc.run.stage != "processing" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert svc.run.stage == "processing" and svc.cancel_run() == run
    SlowEngine.gate.set()
    assert svc.run.finished.wait(20) and svc.run.code == 130
    error = wire.events()[-1]
    assert error["event"] == "error" and error["message"] == "Verarbeitung abgebrochen." and error["exit_code"] == 130
    assert not env["store"].hot_files() and svc.meetings() == []
    assert ("run.exit", {"run": 1, "code": 130, "terminal": True}) in wire.notes
    with pytest.raises(ServiceError, match="Keine Sitzung aktiv"):
        svc.cancel_run()
    assert svc.begin_transcription(env["audio"], wire, in_thread, diarize=False) == {"run": 2}
    SlowEngine.gate = None
    assert svc.run.finished.wait(30)


def test_stopping_a_recording_does_not_cancel_the_processing_that_follows(env, fast):
    FakeRecorder.interrupt_with = "none"
    svc, wire = Service(env["store"], EngineCache(idle_s=60)), CollectWire()
    svc.begin_run({"device": "auto", "diarize": False}, wire, in_thread)
    deadline = time.monotonic() + 15
    while svc.run.stage != "recording" and time.monotonic() < deadline:
        time.sleep(0.02)
    svc.stop_run()
    assert svc.run.finished.wait(20) and svc.run.code == 0 and wire.events()[-1]["event"] == "done"


@pytest.mark.parametrize("audio", ["fehlt.wav", "."])
def test_transcription_needs_an_existing_audio_file_and_never_records(env, fast, audio):
    svc = Service(env["store"], EngineCache(idle_s=60))
    with pytest.raises(ServiceError, match="existiert nicht oder ist keine Datei"):
        svc.begin_transcription(env["tmp"] / audio, CollectWire(), in_thread)
    assert svc.run is None and FakeRecorder.started == []


def test_transcript_pages_through_segments_with_names(env, fast):
    assert transcribe(env, "--no-diarize").exit_code == 0
    svc = Service(env["store"])
    svc.speakers_set("UNKNOWN", "Anna", "m1")
    full = svc.transcript("m1")
    assert full["total"] == 1 and full["next_offset"] is None and full["segments"][0]["speaker"] == "Anna" and full["segments"][0]["label"] == "UNKNOWN"
    assert svc.transcript(None, speaker="Anna")["total"] == 1 and svc.transcript("m1", speaker="Bernd")["segments"] == []
    assert svc.transcript("m1", offset=5)["segments"] == []
    with pytest.raises(ServiceError, match="Unbekannte meeting_id"):
        svc.transcript("gibtsnicht")


def test_sandboxed_service_selects_read_only(env):
    assert transcribe(env, "--no-diarize").exit_code == 0
    svc = Service(env["store"], sandbox=True)
    assert svc.select("SELECT count(*) AS n FROM segments") == {"columns": ["n"], "rows": [[1]], "truncated": False}
    with pytest.raises(ServiceError, match="SELECT"):
        svc.select("DROP VIEW segments")
    assert svc.select("SELECT * FROM range(10)", 3)["truncated"] is True


@pytest.mark.parametrize("audio", [r"\\192.0.2.1\share\x.wav", "//192.0.2.1/share/x.wav", r"\\?\UNC\192.0.2.1\share\x.wav", r"\\.\pipe\x"])
def test_transcription_refuses_network_paths_without_touching_them(env, fast, monkeypatch, audio):
    touched = []
    for name in ("is_file", "resolve", "exists", "stat"):
        monkeypatch.setattr(Path, name, lambda self, *a, **k: touched.append(str(self)) or False)
    svc = Service(env["store"], EngineCache(idle_s=60))
    with pytest.raises(ServiceError, match="Netzwerkpfade und Netzlaufwerke sind gesperrt"):
        svc.begin_transcription(audio, CollectWire(), in_thread)
    monkeypatch.undo()
    assert svc.run is None and touched == []


def test_transcription_refuses_a_mapped_network_drive(env, fast, monkeypatch):
    import ctypes
    seen = []
    monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", lambda root: seen.append(root) or 4, raising=False)
    svc = Service(env["store"], EngineCache(idle_s=60))
    with pytest.raises(ServiceError, match="Netzlaufwerke"):
        svc.begin_transcription(r"Z:\meeting.wav", CollectWire(), in_thread)
    assert seen == ["Z:\\"]


def test_a_meeting_without_segments_yet_gets_the_same_german_answer_everywhere(env):
    svc = Service(env["store"])
    with env["store"].writer("m-lauft", title="T", meeting_date="2026-03-04", language="de"):
        for call in (svc.transcript, svc.report):
            with pytest.raises(ServiceError, match="noch keine Segmente"):
                call("m-lauft")
        assert svc.stats("m-lauft") == [] and svc.meetings() == []


def test_closing_the_service_discards_the_running_meeting_at_once_not_after_the_window(env, fast, monkeypatch):
    monkeypatch.setitem(pipeline.SETTINGS, "window_s", 10.0)
    monkeypatch.setattr(pipeline.AudioLoader, "load", staticmethod(lambda path, rate=None: np.full(16000 * 60, 0.1, dtype="float32")))
    SlowEngine.gate = threading.Event()
    svc, wire = Service(env["store"], EngineCache(idle_s=60)), CollectWire()
    svc.begin_transcription(env["audio"], wire, in_thread, diarize=False)
    deadline = time.monotonic() + 15
    while not env["store"].hot_files() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert env["store"].hot_files()
    svc.close()
    assert not env["store"].hot_files() and not svc.run.finished.is_set()
    SlowEngine.gate.set()
    assert svc.run.finished.wait(20) and svc.run.code == 130
    assert not env["store"].hot_files() and svc.meetings() == []
    SlowEngine.gate = None


def test_closing_the_service_after_some_windows_leaves_no_incomplete_meeting(env, fast, monkeypatch):
    monkeypatch.setitem(pipeline.SETTINGS, "window_s", 10.0)
    monkeypatch.setattr(pipeline.AudioLoader, "load", staticmethod(lambda path, rate=None: np.full(16000 * 60, 0.1, dtype="float32")))
    SlowEngine.gate = None
    svc, wire = Service(env["store"], EngineCache(idle_s=60)), CollectWire()
    svc.begin_transcription(env["audio"], wire, in_thread, diarize=False)
    deadline = time.monotonic() + 15
    while not svc.meetings() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert svc.meetings() and not svc.meetings()[0]["complete"]
    svc.close()
    assert not env["store"].hot_files() and svc.meetings() == []
    assert svc.run.finished.wait(20) and svc.run.code == 130 and not env["store"].hot_files()


@pytest.mark.parametrize("sandbox, wanted, unwanted", [
    (True, "diarize=false", ("--no-diarize", "Kommandozeile", "meetscribe setup")),
    (False, "Einstellungen", ("--no-diarize", "Kommandozeile", "meetscribe setup")),
])
def test_the_missing_diarization_model_names_the_way_out_of_the_surface_that_asked(env, fast, sandbox, wanted, unwanted):
    svc, wire = Service(env["store"], EngineCache(idle_s=60), sandbox=sandbox, surface="mcp" if sandbox else "app"), CollectWire()
    svc.begin_transcription(env["audio"], wire, in_thread, diarize=True)
    assert svc.run.finished.wait(20)
    error = wire.events()[-1]
    assert error["event"] == "error" and wanted in error["message"]
    assert not [word for word in unwanted if word in error["message"] or word in error["detail"]]


def test_the_start_lock_message_names_the_app_and_claude_not_a_hard_kill(tmp_path):
    with pipeline.RunLock(tmp_path / "start.lock"):
        with pytest.raises(pipeline.PipelineError) as busy:
            pipeline.RunLock(tmp_path / "start.lock").__enter__()
    assert "in der App oder über Claude" in busy.value.message and "nach hartem Beenden der Oberfläche läuft" not in busy.value.message


def make_junction(link, substitute):
    import ctypes
    import struct
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    kernel.DeviceIoControl.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    os.mkdir(link)
    name = substitute.encode("utf-16-le")
    body = struct.pack("<HHHH", 0, len(name), len(name) + 2, 0) + name + b"\0\0\0\0"
    buffer = struct.pack("<IHH", 0xA0000003, len(body), 0) + body
    handle = kernel.CreateFileW(str(link), 0x40000000, 7, None, 3, 0x02200000, None)
    ok = handle not in (None, ctypes.c_void_p(-1).value) and kernel.DeviceIoControl(handle, 0x900A4, buffer, len(buffer), None, 0, ctypes.byref(wintypes.DWORD()), None)
    error = ctypes.get_last_error()
    if handle not in (None, ctypes.c_void_p(-1).value):
        kernel.CloseHandle(handle)
    if not ok:
        os.rmdir(link)
        raise OSError(error, "junction")


@pytest.fixture
def links(tmp_path):
    made = []

    def make(name, substitute):
        link = tmp_path / name
        make_junction(link, substitute)
        made.append(link)
        return link
    yield make
    for link in made:
        os.rmdir(link)


needs_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows-Verknüpfungen")


@needs_windows
def test_local_path_refuses_junctions_to_a_network_target_on_every_component(links, tmp_path):
    real = tmp_path / "echt"
    real.mkdir()
    (real / "a.wav").write_bytes(b"x")
    network = links("netz", "\\??\\UNC\\192.0.2.1\\share")
    deep = tmp_path / "tief"
    deep.mkdir()
    inner = links("tief\\sprung", "\\??\\UNC\\192.0.2.1\\share\\x")
    via_local = links("lokal", "\\??\\" + str(real))
    chained = links("kette", "\\??\\" + str(network))
    backslashed = {k: str(v) for k, v in {"net": network, "inner": inner}.items()}
    assert service.LocalPath.remote(network / "a.wav") and service.LocalPath.remote(network) and service.LocalPath.remote(str(network / "a" / "b" / "c.wav"))
    assert service.LocalPath.remote(str(network).replace("\\", "/") + "/a.wav") and service.LocalPath.remote(backslashed["inner"] + "\\deep\\f.wav")
    assert service.LocalPath.remote(chained / "a.wav") and service.LocalPath.remote(network / ".." / "netz" / "a.wav")
    assert not service.LocalPath.remote(real / "a.wav") and not service.LocalPath.remote(via_local / "a.wav") and not service.LocalPath.remote(tmp_path / "gibt-es-nicht.wav")
    assert not service.LocalPath.remote(r"C:\Windows\notepad.exe") and service.LocalPath.remote("\\\\192.0.2.1\\share\\a.wav") and service.LocalPath.remote("//server/share/a.wav")
    assert deep.exists()


@needs_windows
def test_local_path_follows_a_link_loop_to_a_refusal_not_a_hang(links, tmp_path):
    first = links("eins", "\\??\\" + str(tmp_path / "zwei"))
    links("zwei", "\\??\\" + str(first))
    assert service.LocalPath.remote(first / "a.wav")


@needs_windows
def test_local_path_resolves_subst_drives_by_their_target_and_refuses_remote_drive_types(tmp_path, monkeypatch):
    real = tmp_path / "echt"
    real.mkdir()
    (real / "a.wav").write_bytes(b"x")
    letter = next(c for c in "XYZWVUTSRQPONM" if not os.path.exists(c + ":\\"))
    assert subprocess.run(["subst", letter + ":", str(real)], capture_output=True).returncode == 0
    try:
        assert not service.LocalPath.remote(letter + ":\\a.wav")
        (real / "sub").mkdir()
        assert not service.LocalPath.remote(letter + ":\\sub\\b.wav")
    finally:
        subprocess.run(["subst", letter + ":", "/D"], check=True, capture_output=True)
    assert not os.path.exists(letter + ":\\")
    monkeypatch.setattr(service.LocalPath, "drive_type", staticmethod(lambda root: service.DRIVE_REMOTE))
    assert service.LocalPath.remote("C:\\Windows\\notepad.exe")


@needs_windows
def test_local_path_reads_a_symlink_the_same_way_as_a_junction(monkeypatch, tmp_path):
    file = tmp_path / "link.wav"
    file.write_bytes(b"x")
    targets = {str(file): "\\\\?\\UNC\\192.0.2.1\\share\\a.wav"}
    real = os.readlink
    monkeypatch.setattr(service.os, "readlink", lambda path: targets[str(path)] if str(path) in targets else real(path))
    assert service.LocalPath.remote(file)
    targets[str(file)] = "other.wav"
    (tmp_path / "other.wav").write_bytes(b"x")
    assert not service.LocalPath.remote(file)
    targets[str(file)] = "..\\" + tmp_path.name + "\\other.wav"
    assert not service.LocalPath.remote(file)
    targets[str(file)] = "\\??\\UNC\\192.0.2.1\\share\\a.wav"
    assert service.LocalPath.remote(file)


@needs_windows
def test_begin_transcription_refuses_a_link_to_a_network_before_touching_the_file(links, env):
    network = links("netz", "\\??\\UNC\\192.0.2.1\\share")
    svc = Service(env["store"])
    with pytest.raises(ServiceError, match="Netzwerkpfade und Netzlaufwerke sind gesperrt"):
        svc.begin_transcription(str(network / "a.wav"), Wire(None, None), lambda work: None)
    assert svc.run is None
