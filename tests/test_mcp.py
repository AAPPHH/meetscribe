import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import anyio
import numpy as np
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from meetscribe import mcp_server, pipeline
from meetscribe.mcp_server import TOOLS, McpApp
from meetscribe.service import EngineCache, Service
from meetscribe.store import Store, make_config
from test_cli import FakeRecorder, env, transcribe  # noqa: F401
from test_service import SlowEngine, fast  # noqa: F401
from test_store import SPEAKER_WORKER, write

TESTS = Path(__file__).parent
GUARD = TESTS / "socket_guard_server.py"
EXPECTED_TOOLS = {
    "list_meetings", "search", "get_transcript", "get_minutes", "speaker_stats", "set_speaker_name", "transcribe_start", "transcribe_status",
    "transcribe_cancel", "list_models", "list_hardware", "query_sql",
}
FORBIDDEN_WORDS = ("record", "setup", "download", "delete", "discard", "rotate", "install", "mic")
SQL_REFUSED = [
    "CREATE TABLE z (a INT)", "DROP VIEW segments", "DELETE FROM speakers", "INSERT INTO speakers VALUES ('m1', 'x', 'y')", "UPDATE speakers SET name = 'x'",
    "COPY segments TO 'leak.csv'", "COPY (SELECT 1) TO 'leak.csv'", "ATTACH ':memory:' AS z", "INSTALL httpfs", "LOAD httpfs", "SET threads = 1", "RESET threads",
    "PRAGMA database_list", "EXPORT DATABASE 'leak'", "CALL pragma_version()", "SELECT 1; DROP VIEW segments", "BEGIN", "COMMIT", "VACUUM", "   ",
    "SELECT * FROM read_csv('C:/Windows/win.ini')", "SELECT * FROM read_text('C:/Windows/win.ini')", "SELECT * FROM glob('C:/*')",
    "SELECT * FROM read_parquet('C:/Windows/*.parquet')", "SELECT * FROM query('DROP VIEW segments')", "FROM read_blob('C:/Windows/win.ini')",
]


class Caller:
    def __init__(self, client):
        self.client = client

    async def raw(self, tool, /, **arguments):
        return await self.client.call_tool(tool, arguments)

    async def ok(self, tool, /, **arguments):
        result = await self.raw(tool, **arguments)
        assert not result.is_error, result.content[0].text
        return result.structured_content if result.structured_content is not None else result.content[0].text

    async def fail(self, tool, /, **arguments):
        result = await self.raw(tool, **arguments)
        assert result.is_error, result.structured_content
        return result.content[0].text


def drive(app, script):
    async def main():
        async with Client(app.server) as client:
            return await script(Caller(client))
    return anyio.run(main)


@pytest.fixture
def app(env, fast):
    transcribe(env, "--no-diarize")
    write(env["store"], "m2", title="Planung", day="2026-03-05")
    return McpApp(Service(env["store"], EngineCache(idle_s=60), sandbox=True, surface="mcp"))


def test_exactly_the_planned_tools_with_german_descriptions_and_typed_inputs(app):
    async def script(call):
        return (await call.client.list_tools()).tools
    tools = {t.name: t for t in drive(app, script)}
    assert set(tools) == EXPECTED_TOOLS == set(TOOLS)
    assert not [n for n in tools if any(w in n for w in FORBIDDEN_WORDS)]
    for tool in tools.values():
        assert len(tool.description) > 40 and tool.annotations.open_world_hint is False
        assert all(p.get("description") for p in tool.input_schema.get("properties", {}).values())
    assert tools["query_sql"].annotations.read_only_hint and not tools["set_speaker_name"].annotations.read_only_hint
    start = tools["transcribe_start"].input_schema
    assert start["required"] == ["audio"] and start["properties"]["device"]["enum"] == ["auto", "npu", "gpu", "igpu", "cpu"]
    assert start["properties"]["diarize"]["type"] == "boolean" and "mic" not in start["properties"]
    assert tools["search"].input_schema["properties"]["limit"]["maximum"] == 200
    assert tools["transcribe_status"].input_schema["required"] == ["job"]
    for name in ("search", "get_transcript", "get_minutes", "query_sql"):
        assert "unvertrauenswürdig" in tools[name].description
    assert "unvertrauenswürdig" in app.server.instructions


def test_read_tools_list_search_transcript_minutes_and_stats(app):
    async def script(call):
        meetings = (await call.ok("list_meetings"))["meetings"]
        hits = (await call.ok("search", term="MÜNCHEN"))["hits"]
        only = (await call.ok("search", term="sch", meeting_id="m2", speaker="SPEAKER_01", limit=1))["hits"]
        page = await call.ok("get_transcript", meeting_id="m2", limit=2)
        rest = await call.ok("get_transcript", meeting_id="m2", offset=page["next_offset"])
        latest = await call.ok("get_transcript")
        minutes = await call.ok("get_minutes", meeting_id="m2")
        stats = (await call.ok("speaker_stats", meeting_id="m2"))["speakers"]
        everyone = (await call.ok("speaker_stats"))["speakers"]
        return meetings, hits, only, page, rest, latest, minutes, stats, everyone
    meetings, hits, only, page, rest, latest, minutes, stats, everyone = drive(app, script)
    assert {m["meeting_id"] for m in meetings} == {"m1", "m2"} and all(m["complete"] for m in meetings) and meetings[0]["meeting_date"] == "2026-03-05"
    assert {h["meeting_id"] for h in hits} == {"m1", "m2"} and any("Grüße aus München" in h["text"] for h in hits)
    assert len(only) == 1 and only[0]["speaker"] == "SPEAKER_01"
    assert page["total"] == 3 and len(page["segments"]) == 2 and page["next_offset"] == 2 and "unvertrauenswürdig" in page["hinweis"]
    assert len(rest["segments"]) == 1 and rest["next_offset"] is None and rest["segments"][0]["text"] == "Fangen wir an."
    assert latest["meeting_id"] in ("m1", "m2") and latest["total"] >= 1
    assert minutes.startswith("# Planung – 05.03.2026") and "## Teilnehmer" in minutes and "**[00:04] SPEAKER_01:**" in minutes
    assert {s["speaker"] for s in stats} == {"SPEAKER_00", "SPEAKER_01"} and sum(s["share"] for s in stats) == pytest.approx(1.0)
    assert {s["meeting_id"] for s in everyone} == {"m1", "m2"}


def test_speaker_names_flow_into_every_view_and_bad_input_is_a_tool_error(app):
    async def script(call):
        done = await call.ok("set_speaker_name", label="SPEAKER_00", name="Anna Beispiel", meeting_id="m2")
        return (
            done, await call.ok("get_minutes", meeting_id="m2"), (await call.ok("get_transcript", meeting_id="m2"))["segments"][0],
            (await call.ok("search", term="münchen", speaker="Anna Beispiel"))["hits"], (await call.ok("speaker_stats", meeting_id="m2"))["speakers"],
            await call.fail("set_speaker_name", label="SPEAKER_77", name="X", meeting_id="m2"),
            await call.fail("set_speaker_name", label="SPEAKER_00", name="   ", meeting_id="m2"),
            await call.fail("get_minutes", meeting_id="gibtsnicht"), await call.fail("get_minutes", meeting_id="../x"),
            await call.fail("get_transcript", meeting_id="m2", limit=0), await call.fail("search", term=""),
        )
    done, minutes, first, hits, stats, bad_label, blank, unknown, traversal, zero, empty = drive(app, script)
    assert done == {"meeting_id": "m2", "label": "SPEAKER_00", "name": "Anna Beispiel"}
    assert "| Anna Beispiel |" in minutes and "**[00:00] Anna Beispiel:**" in minutes
    assert first["speaker"] == "Anna Beispiel" and first["label"] == "SPEAKER_00"
    assert len(hits) == 1 and "Anna Beispiel" in {s["speaker"] for s in stats}
    assert "Unbekanntes Label" in bad_label and "SPEAKER_00" in bad_label and "must not be empty" in blank
    assert "Unbekannte meeting_id: gibtsnicht" in unknown
    assert all(text for text in (traversal, zero, empty))


def test_sql_reads_and_limits_but_refuses_everything_else(app, env):
    def files():
        return sorted(str(p.relative_to(env["tmp"])) for p in env["tmp"].rglob("*") if p.is_file() and p.suffix != ".lock")
    before = files()

    async def script(call):
        counted = await call.ok("query_sql", sql="SELECT count(*) AS n, count(DISTINCT meeting_id) AS m FROM segments")
        limited = await call.ok("query_sql", sql="SELECT * FROM range(5000)", limit=5)
        default = await call.ok("query_sql", sql="SELECT * FROM range(5000)")
        joined = await call.ok("query_sql", sql="SELECT meeting_id, display_name FROM segments_named WHERE seg_id = 0 ORDER BY meeting_id")
        refused = {sql: await call.fail("query_sql", sql=sql) for sql in SQL_REFUSED}
        after = await call.ok("query_sql", sql="SELECT count(*) AS n FROM segments")
        too_big = await call.fail("query_sql", sql="SELECT 1", limit=100000)
        return counted, limited, default, joined, refused, after, too_big
    counted, limited, default, joined, refused, after, too_big = drive(app, script)
    assert counted == {"columns": ["n", "m"], "rows": [[4, 2]], "truncated": False}
    assert len(limited["rows"]) == 5 and limited["truncated"] is True
    assert len(default["rows"]) == 200 and default["truncated"] is True
    assert joined["rows"] == [["m1", "UNKNOWN"], ["m2", "SPEAKER_00"]]
    assert all(refused.values()) and len(refused) == len(SQL_REFUSED) and after["rows"] == [[4]] and too_big
    assert files() == before


def test_models_and_hardware_without_microphones(app):
    async def script(call):
        return await call.ok("list_models"), await call.ok("list_hardware")
    models, hardware = drive(app, script)
    assert models["default_model"] == "large-v3-turbo" and {m["backend"] for m in models["models"]} >= {"openvino"}
    assert {b["id"] for b in models["backends"]} >= {"openvino"}
    kinds = {h["id"]: h for h in hardware["hardware"]}
    assert kinds["npu"]["present"] and kinds["cpu"]["present"] and set(hardware) == {"hardware", "backends"}


def test_transcription_job_runs_to_the_end_with_status_and_a_stored_meeting(app, env):
    async def script(call):
        started = await call.ok("transcribe_start", audio=str(env["audio"]), title="Neue Runde", diarize=False)
        waiting = await call.ok("transcribe_status", job=started["job"])
        final = await call.ok("transcribe_status", job=started["job"], wait_s=30)
        meetings = (await call.ok("list_meetings"))["meetings"]
        minutes = await call.ok("get_minutes", meeting_id=final["meeting_id"])
        return started, waiting, final, meetings, minutes
    started, waiting, final, meetings, minutes = drive(app, script)
    assert started == {"job": 1, "state": "running"} and waiting["state"] in ("running", "done")
    assert final["state"] == "done" and final["segments"] == 1 and final["speakers"] == 1 and final["meeting_id"].endswith("-neue-runde")
    assert final["labels"] == [] and final["ran"]["devices"] == ["NPU"]
    assert final["plan"]["device"] == "NPU" and final["report_file"].endswith(".md")
    assert any(m["meeting_id"] == final["meeting_id"] and m["title"] == "Neue Runde" for m in meetings) and "Neue Runde" in minutes
    assert FakeRecorder.started == []


def test_transcription_refuses_bad_paths_a_second_job_and_unknown_jobs(app, env):
    SlowEngine.gate = threading.Event()

    async def script(call):
        missing = await call.fail("transcribe_start", audio=str(env["tmp"] / "fehlt.wav"))
        folder = await call.fail("transcribe_start", audio=str(env["tmp"]))
        started = await call.ok("transcribe_start", audio=str(env["audio"]), diarize=False)
        busy = await call.fail("transcribe_start", audio=str(env["audio"]), diarize=False)
        running = await call.ok("transcribe_status", job=started["job"])
        unknown = await call.fail("transcribe_status", job=99)
        unknown_cancel = await call.fail("transcribe_cancel", job=99)
        SlowEngine.gate.set()
        final = await call.ok("transcribe_status", job=started["job"], wait_s=30)
        late_cancel = await call.fail("transcribe_cancel", job=started["job"])
        return missing, folder, busy, running, unknown, unknown_cancel, final, late_cancel
    missing, folder, busy, running, unknown, unknown_cancel, final, late_cancel = drive(app, script)
    assert "existiert nicht oder ist keine Datei" in missing and "existiert nicht oder ist keine Datei" in folder
    assert "bereits eine Sitzung" in busy and running["state"] == "running" and running["stage"] in ("preflight", "processing")
    assert "Unbekannte Job-ID 99" in unknown and "Unbekannte Job-ID" in unknown_cancel
    assert final["state"] == "done" and "läuft nicht mehr" in late_cancel


def test_cancel_ends_the_job_discards_the_partial_meeting_and_keeps_the_audio(app, env, monkeypatch):
    monkeypatch.setitem(pipeline.SETTINGS, "window_s", 10.0)
    monkeypatch.setattr(pipeline.AudioLoader, "load", staticmethod(lambda path, rate=None: np.full(16000 * 60, 0.1, dtype="float32")))
    SlowEngine.gate = threading.Event()

    async def script(call):
        started = await call.ok("transcribe_start", audio=str(env["audio"]), diarize=False)
        for _ in range(300):
            if (await call.ok("transcribe_status", job=started["job"]))["stage"] == "processing":
                break
            await anyio.sleep(0.05)
        cancelling = await call.ok("transcribe_cancel", job=started["job"])
        SlowEngine.gate.set()
        final = await call.ok("transcribe_status", job=started["job"], wait_s=30)
        return cancelling, final, (await call.ok("list_meetings"))["meetings"]
    cancelling, final, meetings = drive(app, script)
    assert cancelling == {"job": 1, "state": "cancelling"} and final["state"] == "cancelled" and final["exit_code"] == 130
    assert final["message"] == "Verarbeitung abgebrochen." and {m["meeting_id"] for m in meetings} == {"m1", "m2"}
    assert env["audio"].exists() and not [p for p in env["store"].hot_files() if p.stem not in ("m1", "m2")]


def test_cancel_during_the_only_window_still_ends_with_no_meeting(app, env):
    SlowEngine.gate = threading.Event()

    async def script(call):
        started = await call.ok("transcribe_start", audio=str(env["audio"]), diarize=False)
        await anyio.sleep(0.5)
        await call.ok("transcribe_cancel", job=started["job"])
        SlowEngine.gate.set()
        return await call.ok("transcribe_status", job=started["job"], wait_s=30), (await call.ok("list_meetings"))["meetings"]
    final, meetings = drive(app, script)
    assert final["state"] == "cancelled" and {m["meeting_id"] for m in meetings} == {"m1", "m2"}
    assert {p.stem for p in env["store"].hot_files()} == {"m1", "m2"}


def test_a_foreign_start_lock_fails_the_job_with_the_reason_and_frees_afterwards(app, env):
    lock = env["store"].config["data_dir"] / pipeline.START["lock_file"]
    code = "import sys; from meetscribe.pipeline import RunLock; lock = RunLock(sys.argv[1]).__enter__(); print('held', flush=True); sys.stdin.readline()"
    holder = subprocess.Popen([sys.executable, "-c", code, str(lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    async def script(call):
        started = await call.ok("transcribe_start", audio=str(env["audio"]), diarize=False)
        return await call.ok("transcribe_status", job=started["job"], wait_s=30)
    try:
        assert holder.stdout.readline().strip() == "held"
        final = drive(app, script)
    finally:
        holder.stdin.close()
        holder.wait(10)
    assert final["state"] == "failed" and "Es läuft bereits eine Aufnahme oder Verarbeitung" in final["message"] and final["exit_code"] == 1
    assert drive(app, script)["state"] == "done"


def test_main_goes_offline_and_serves_stdio_with_the_sandboxed_service(monkeypatch):
    seen = {}
    monkeypatch.setattr(McpApp, "run", lambda self: seen.update(sandbox=self.service.sandbox, engines=type(self.service.engines).__name__))
    monkeypatch.setattr(mcp_server.os, "_exit", lambda code: seen.update(code=code))
    for name in mcp_server.OFFLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    mcp_server.main()
    assert seen == {"sandbox": True, "engines": "EngineCache", "code": 0}
    assert all(os.environ[k] == v for k, v in mcp_server.OFFLINE_ENV.items())


def stdio_params(data, log):
    env = {**os.environ, "MEETSCRIBE_DATA": str(data), "MEETSCRIBE_SOCKET_LOG": str(log), "MEETSCRIBE_REPORTS": str(data.parent / "protokolle")}
    return StdioServerParameters(command=sys.executable, args=[str(GUARD)], env=env)


def test_real_stdio_session_works_without_any_network_access_and_shares_the_archive(tmp_path):
    data, log = tmp_path / "data", tmp_path / "net.log"
    store = Store(make_config(data))
    write(store, "m1", title="Abgelegt", day="2026-03-04")
    write(store, "m2", title="Rotiert", day="2026-03-05")
    store.rotate(force=True)
    write(store, "m3", title="Heiss", day="2026-03-06")
    worker = subprocess.Popen([sys.executable, "-c", SPEAKER_WORKER, str(data), "W", "12", str(time.time() + 4.0)])

    async def session():
        async with Client(stdio_params(data, log)) as client:
            call = Caller(client)
            tools = {t.name for t in (await client.list_tools()).tools}
            meetings = (await call.ok("list_meetings"))["meetings"]
            hits = (await call.ok("search", term="münchen"))["hits"]
            minutes = await call.ok("get_minutes", meeting_id="m2")
            names = [await call.ok("set_speaker_name", label="SPEAKER_01" if i % 2 else "SPEAKER_00", name=f"Mcp{i}", meeting_id=f"m{i % 3 + 1}") for i in range(12)]
            sql = await call.ok("query_sql", sql="SELECT count(*) AS n FROM speakers")
            refused = await call.fail("query_sql", sql="SELECT * FROM read_csv('C:/Windows/win.ini')")
            hardware = await call.ok("list_hardware")
            return tools, meetings, hits, minutes, names, sql, refused, hardware
    tools, meetings, hits, minutes, names, sql, refused, hardware = anyio.run(session)
    assert worker.wait(60) == 0
    assert tools == EXPECTED_TOOLS and {m["meeting_id"] for m in meetings} == {"m1", "m2", "m3"} and len(hits) == 3
    assert minutes.startswith("# Rotiert") and len(names) == 12 and refused and "hardware" in hardware
    mapping = {(r["meeting_id"], r["label"]) for r in store.speakers.all().to_pylist()}
    assert len({k for k in mapping if k[1].startswith("W_")}) == 12 and len({k for k in mapping if not k[1].startswith("W_")}) == 6
    assert sql["rows"][0][0] >= 6
    assert not log.exists() or log.read_text() == ""


def test_the_socket_guard_really_blocks_and_records(tmp_path):
    log = tmp_path / "net.log"
    code = (
        f"import sys; sys.path.insert(0, {str(TESTS)!r}); import socket, socket_guard_server\n"
        "ok = 0\n"
        "for attempt in (lambda: socket.create_connection(('example.com', 80), 1), lambda: socket.socket().connect(('93.184.216.34', 80)), lambda: socket.getaddrinfo('example.com', 80)):\n"
        "    try: attempt()\n"
        "    except OSError: ok += 1\n"
        "a, b = socket.socketpair(); a.send(b'x'); print(ok, b.recv(1))"
    )
    done = subprocess.run([sys.executable, "-c", code], env={**os.environ, "MEETSCRIBE_SOCKET_LOG": str(log)}, capture_output=True, text=True, timeout=60)
    assert done.stdout.strip() == "3 b'x'", done.stderr
    assert {line.split()[0] for line in log.read_text().splitlines()} >= {"getaddrinfo", "connect"}


def test_transcribe_start_refuses_a_network_path_with_a_german_reason_and_starts_nothing(app, env):
    async def script(call):
        refused = await call.fail("transcribe_start", audio=r"\\192.0.2.1\share\x.wav")
        return refused, await call.fail("transcribe_status", job=1)
    refused, status = drive(app, script)
    assert "Netzwerkpfade und Netzlaufwerke sind gesperrt" in refused and "Unbekannte Job-ID" in status
    assert app.service.run is None


def test_reading_a_running_job_before_its_first_window_gives_a_clear_answer_in_every_tool(app, env):
    SlowEngine.gate = threading.Event()

    async def script(call):
        started = await call.ok("transcribe_start", audio=str(env["audio"]), diarize=False)
        meeting_id = None
        for _ in range(300):
            meeting_id = (await call.ok("transcribe_status", job=started["job"]))["meeting_id"]
            if meeting_id and env["store"].hot_path(meeting_id).exists():
                break
            await anyio.sleep(0.05)
        answers = (
            await call.fail("get_transcript", meeting_id=meeting_id), await call.fail("get_minutes", meeting_id=meeting_id),
            await call.ok("speaker_stats", meeting_id=meeting_id),
        )
        SlowEngine.gate.set()
        await call.ok("transcribe_status", job=started["job"], wait_s=30)
        return answers
    transcript, minutes, stats = drive(app, script)
    assert "noch keine Segmente" in transcript and "noch keine Segmente" in minutes and stats == {"speakers": []}
    SlowEngine.gate = None


def test_a_missing_diarization_model_names_the_tool_argument_not_a_command_line(app, env):
    async def script(call):
        started = await call.ok("transcribe_start", audio=str(env["audio"]))
        return await call.ok("transcribe_status", job=started["job"], wait_s=30)
    final = drive(app, script)
    assert final["state"] == "failed" and "diarize=false" in final["message"] and "--no-diarize" not in final["message"] + final["detail"]
