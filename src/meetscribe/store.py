import json
import math
import os
import re
import sys
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EOS = b"\xff\xff\xff\xff\x00\x00\x00\x00"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
WORD_TYPE = pa.struct([("text", pa.string()), ("start_s", pa.float64()), ("end_s", pa.float64())])
SCHEMA = pa.schema([
    ("meeting_id", pa.string()),
    ("title", pa.string()),
    ("meeting_date", pa.date32()),
    ("seg_id", pa.int32()),
    ("start_s", pa.float64()),
    ("end_s", pa.float64()),
    ("speaker", pa.string()),
    ("text", pa.string()),
    ("language", pa.string()),
    ("model", pa.string()),
    ("device", pa.string()),
    ("backend", pa.string()),
    ("words", pa.list_(WORD_TYPE)),
    ("created_at", pa.timestamp("us", tz="UTC")),
])
SPEAKER_SCHEMA = pa.schema([("meeting_id", pa.string()), ("label", pa.string()), ("name", pa.string())])
COLUMNS = ", ".join(SCHEMA.names)
REQUIRED = ("start_s", "end_s", "speaker", "text")
DUCKDB_SETTINGS = {
    "TimeZone": "'UTC'",
    "autoinstall_known_extensions": "false",
    "autoload_known_extensions": "false",
}
SANDBOX = {"memory_limit": "'1GB'", "threads": "2", "enable_external_access": "false"}
SQL_LIMITS = {"default_rows": 200, "max_rows": 1000, "timeout_s": 20.0}
LOCKS = {"speakers_wait_s": 15.0, "rotate_wait_s": 60.0, "poll_s": 0.05, "busy_tries": 40, "busy_wait_s": 0.05}
TEXT = {
    "select_only": "Erlaubt ist genau eine lesende SELECT-Abfrage ({kind}).",
    "timeout": "Die Abfrage wurde nach {seconds:g} s abgebrochen.",
    "busy": "Die Datei ist gerade gesperrt: {path}",
}


def make_config(data_dir=None):
    base = Path(data_dir or os.environ.get("MEETSCRIBE_DATA") or PROJECT_ROOT / "data")
    return {
        "data_dir": base,
        "hot_dir": base / "hot",
        "cold_dir": base / "cold",
        "archive_dir": base / "archive",
        "speakers_file": base / "speakers.parquet",
    }


CONFIG = make_config()


class DuplicateMeetingError(Exception):
    pass


class MeetingNotFound(LookupError):
    pass


class ExtensionBlocked(duckdb.Error):
    pass


class ReadOnlyViolation(duckdb.Error):
    pass


class VerifyError(RuntimeError):
    pass


def as_date(value):
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    return value if isinstance(value, date) else date.fromisoformat(value)


def conform(table):
    columns = [table.column(n) if n in table.column_names else pa.nulls(table.num_rows) for n in SCHEMA.names]
    return pa.Table.from_arrays(columns, names=SCHEMA.names).cast(SCHEMA)


def retry_busy(action):
    for attempt in range(LOCKS["busy_tries"]):
        try:
            return action()
        except PermissionError:
            if attempt == LOCKS["busy_tries"] - 1:
                raise
            time.sleep(LOCKS["busy_wait_s"])


def read_parquet(path):
    def read():
        with open(path, "rb") as handle:
            return pq.read_table(handle)
    return retry_busy(read)


def atomic_parquet(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    pq.write_table(table, tmp)
    if read_parquet(tmp).num_rows != table.num_rows:
        tmp.unlink()
        raise VerifyError(f"row count mismatch for {path.name}")
    retry_busy(lambda: os.replace(tmp, path))


def sql_literal(path):
    return "'" + path.as_posix().replace("'", "''") + "'"


class LockBusy(OSError):
    pass


class FileLock:
    def __init__(self, path, wait_s=0.0):
        self.path = Path(path)
        self.wait_s = wait_s
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
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                self._lock(True)
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self.file.close()
                    raise LockBusy(TEXT["busy"].format(path=self.path)) from None
                time.sleep(LOCKS["poll_s"])

    def __exit__(self, exc_type, exc, tb):
        try:
            self._lock(False)
        except OSError:
            pass
        self.file.close()
        return False


class HotFile:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        data = self.path.read_bytes()
        batches, complete = [], False
        try:
            reader = ipc.open_stream(pa.BufferReader(data))
        except (pa.ArrowException, OSError):
            return SCHEMA.empty_table(), False
        try:
            while True:
                batches.append(reader.read_next_batch())
        except StopIteration:
            complete = data.endswith(EOS)
        except (pa.ArrowException, OSError):
            pass
        table = pa.Table.from_batches(batches, schema=reader.schema)
        return conform(table), complete

    def is_complete(self):
        return self.read()[1]


class SpeakerMap:
    def __init__(self, store):
        self.path = store.config["speakers_file"]
        self.lock_path = store.config["data_dir"] / "speakers.lock"

    def locked(self):
        return FileLock(self.lock_path, LOCKS["speakers_wait_s"])

    def all(self):
        return read_parquet(self.path) if self.path.exists() else SPEAKER_SCHEMA.empty_table()

    def _mapping(self):
        return {(r["meeting_id"], r["label"]): r["name"] for r in self.all().to_pylist()}

    def _write(self, mapping):
        rows = [{"meeting_id": m, "label": l, "name": n} for (m, l), n in sorted(mapping.items())]
        atomic_parquet(pa.Table.from_pylist(rows, schema=SPEAKER_SCHEMA), self.path)

    def set(self, meeting_id, label, name):
        name = name.strip()
        if not name:
            raise ValueError("name must not be empty")
        with self.locked():
            mapping = self._mapping()
            mapping[(meeting_id, label)] = name
            self._write(mapping)

    def unset(self, meeting_id, label):
        with self.locked():
            mapping = self._mapping()
            removed = mapping.pop((meeting_id, label), None) is not None
            if removed:
                self._write(mapping)
        return removed


class Store:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.speakers = SpeakerMap(self)

    @property
    def hot_dir(self):
        return self.config["hot_dir"]

    @property
    def cold_dir(self):
        return self.config["cold_dir"]

    @property
    def archive_dir(self):
        return self.config["archive_dir"]

    def check_id(self, meeting_id):
        if not isinstance(meeting_id, str) or not ID_PATTERN.match(meeting_id):
            raise ValueError(f"invalid meeting_id: {meeting_id!r}")

    def hot_path(self, meeting_id):
        self.check_id(meeting_id)
        return self.hot_dir / f"{meeting_id}.arrow"

    def hot_files(self):
        return sorted(self.hot_dir.glob("*.arrow"))

    def cold_files(self):
        return sorted(self.cold_dir.glob("year=*/month=*/*.parquet"))

    def exists(self, meeting_id):
        return self.hot_path(meeting_id).exists() or any(p.stem == meeting_id for p in self.cold_files())

    def writer(self, meeting_id, **meta):
        return TranscriptWriter(meeting_id, store=self, **meta)

    def discard(self, meeting_id):
        self.hot_path(meeting_id).unlink(missing_ok=True)

    def incomplete(self):
        return [p.stem for p in self.hot_files() if not HotFile(p).is_complete()]

    def cold_path(self, meeting_id, day):
        return self.cold_dir / f"year={day.year:04d}" / f"month={day.month:02d}" / f"{meeting_id}.parquet"

    def rotate(self, older_than_days=0, force=False):
        with FileLock(self.config["data_dir"] / "rotate.lock", LOCKS["rotate_wait_s"]):
            return self._rotate(older_than_days, force)

    def _rotate(self, older_than_days, force):
        result = {"rotated": [], "skipped": []}
        for tmp in self.cold_dir.glob("**/*.tmp"):
            tmp.unlink(missing_ok=True)
        cutoff = time.time() - older_than_days * 86400
        for path in self.hot_files():
            meeting_id = path.stem
            table, complete = HotFile(path).read()
            reason = None
            if not complete and not force:
                reason = "incomplete"
            elif older_than_days > 0 and path.stat().st_mtime > cutoff:
                reason = "too_recent"
            elif table.num_rows == 0:
                reason = "empty"
            if reason is None:
                try:
                    self._rotate_one(path, table)
                except (OSError, pa.ArrowException, VerifyError) as exc:
                    reason = f"error: {exc}"
            if reason is None:
                result["rotated"].append(meeting_id)
            else:
                result["skipped"].append({"meeting_id": meeting_id, "reason": reason})
        return result

    def _rotate_one(self, path, table):
        day = table.column("meeting_date")[0].as_py()
        atomic_parquet(table, self.cold_path(path.stem, day))
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        os.replace(path, self.archive_dir / path.name)


class TranscriptWriter:
    def __init__(self, meeting_id, title="", meeting_date=None, language=None, model=None, device=None, backend=None,
                 store=None, discard_on_error=False):
        self.store = store or Store()
        self.discard_on_error = discard_on_error
        self.path = self.store.hot_path(meeting_id)
        self.meta = {
            "meeting_id": meeting_id,
            "title": title,
            "meeting_date": as_date(meeting_date),
            "language": language,
            "model": model,
            "device": device,
            "backend": backend,
        }
        self.next_id = 0
        self.file = None
        self.stream = None

    def __enter__(self):
        if self.store.exists(self.meta["meeting_id"]):
            raise DuplicateMeetingError(f"meeting_id already exists: {self.meta['meeting_id']}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file = open(self.path, "xb")
        except FileExistsError:
            raise DuplicateMeetingError(f"meeting_id already exists: {self.meta['meeting_id']}") from None
        self.stream = ipc.new_stream(self.file, SCHEMA)
        self.file.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        failed = exc_type is not None
        try:
            try:
                if not failed:
                    self.stream.close()
            finally:
                self.file.close()
        except BaseException:
            failed = True
            raise
        finally:
            if failed and self.discard_on_error:
                self.store.discard(self.meta["meeting_id"])
        return False

    def _row(self, seg, now):
        row = {**self.meta, "seg_id": self.next_id, "created_at": now, **seg}
        missing = [k for k in REQUIRED if row.get(k) is None]
        if missing:
            raise ValueError(f"segment is missing {missing}")
        if row.get("words") is not None:
            row["words"] = [{k: w[k] for k in WORD_TYPE.names} for w in row["words"]]
        self.next_id = row["seg_id"] + 1
        return row

    def append(self, segments):
        now = datetime.now(timezone.utc)
        saved = self.next_id
        try:
            rows = [self._row(seg, now) for seg in segments]
            if not rows:
                return 0
            self.stream.write_batch(pa.RecordBatch.from_pylist(rows, schema=SCHEMA))
            self.file.flush()
        except Exception:
            self.next_id = saved
            raise
        return len(rows)


class Query:
    def __init__(self, store=None, sandbox=False):
        self.store = store or Store()
        self.con = duckdb.connect(":memory:")
        settings = dict(DUCKDB_SETTINGS)
        if sandbox:
            settings["allowed_directories"] = f"[{sql_literal(self.store.cold_dir.resolve())}]"
            settings.update(SANDBOX)
        for setting, value in {**settings, "lock_configuration": "true"}.items():
            self.con.execute(f"SET {setting}={value}")
        self.con.register("empty_segments", SCHEMA.empty_table())
        self._signature = None
        self.refresh()

    def _state(self):
        paths = [*self.store.hot_files(), *self.store.cold_files(), self.store.config["speakers_file"]]
        state = []
        for p in paths:
            try:
                st = p.stat()
                state.append((str(p), st.st_size, st.st_mtime_ns))
            except FileNotFoundError:
                state.append((str(p), None, None))
        return tuple(state)

    def refresh(self):
        state = self._state()
        if state == self._signature:
            return
        reads = []
        for path in self.store.hot_files():
            try:
                reads.append((path.stem, *HotFile(path).read()))
            except FileNotFoundError:
                continue
        tables = [table for _, table, _ in reads]
        self.con.register("hot_raw", pa.concat_tables(tables) if tables else SCHEMA.empty_table())
        self.con.register("incomplete_ids", pa.table({"meeting_id": pa.array([m for m, _, complete in reads if not complete], pa.string())}))
        self.con.register("speakers", self.store.speakers.all())
        source = "empty_segments"
        if self.store.cold_files():
            files = ", ".join(sql_literal(f) for f in self.store.cold_files())
            source = f"(SELECT * FROM read_parquet([{files}], hive_partitioning=true, union_by_name=true) UNION ALL BY NAME SELECT * FROM empty_segments)"
        for sql in (
            f"CREATE OR REPLACE VIEW cold_raw AS SELECT {COLUMNS} FROM {source}",
            f"CREATE OR REPLACE VIEW segments AS SELECT {COLUMNS} FROM hot_raw h "
            f"WHERE NOT EXISTS (SELECT 1 FROM cold_raw c WHERE c.meeting_id = h.meeting_id) "
            f"UNION ALL SELECT {COLUMNS} FROM cold_raw",
            "CREATE OR REPLACE VIEW segments_named AS SELECT s.*, COALESCE(p.name, s.speaker) AS display_name "
            "FROM segments s LEFT JOIN speakers p ON p.meeting_id = s.meeting_id AND p.label = s.speaker",
        ):
            self.con.execute(sql)
        self._signature = state

    def _guard(self, sql):
        try:
            blocked = any(s.type == duckdb.StatementType.LOAD for s in self.con.extract_statements(sql))
        except ValueError:
            blocked = True
        if blocked:
            raise ExtensionBlocked("Erweiterungen installieren, laden oder aktualisieren ist gesperrt (meetscribe greift nur mit setup auf das Netz zu).")

    def execute(self, sql, params=None):
        self._guard(sql)
        self.refresh()
        relation = self.con.sql(sql, params=params or [])
        if relation is None:
            return [], []
        table = relation.to_arrow_table()
        return table.column_names, list(zip(*(c.to_pylist() for c in table.columns)))

    def select(self, sql, limit=None, timeout_s=None):
        limit = min(max(int(limit or SQL_LIMITS["default_rows"]), 1), SQL_LIMITS["max_rows"])
        try:
            statements = self.con.extract_statements(sql)
        except duckdb.Error as exc:
            raise ReadOnlyViolation(str(exc)) from exc
        kinds = [s.type for s in statements]
        if kinds != [duckdb.StatementType.SELECT]:
            raise ReadOnlyViolation(TEXT["select_only"].format(kind=", ".join(k.name for k in kinds) or "leer"))
        self.refresh()
        seconds = SQL_LIMITS["timeout_s"] if timeout_s is None else timeout_s
        cursor = self.con
        timer = threading.Timer(seconds, cursor.interrupt)
        try:
            cursor.execute("BEGIN TRANSACTION READ ONLY")
            timer.start()
            table = cursor.sql(sql).limit(limit + 1).to_arrow_table()
            columns = table.column_names
            rows = list(zip(*(c.to_pylist() for c in table.columns)))
        except duckdb.InterruptException as exc:
            raise duckdb.Error(TEXT["timeout"].format(seconds=seconds)) from exc
        finally:
            timer.cancel()
            try:
                cursor.execute("ROLLBACK")
            except duckdb.Error:
                pass
        return columns, rows[:limit], len(rows) > limit

    def _records(self, sql, params=None):
        columns, rows = self.execute(sql, params)
        return [dict(zip(columns, row)) for row in rows]

    @staticmethod
    def _where(**filters):
        clauses = [sql for sql, value in filters.values() if value is not None]
        params = [value for sql, value in filters.values() if value is not None for _ in range(sql.count("?"))]
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    @staticmethod
    def _like(term):
        return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"

    def search(self, term, meeting_id=None, speaker=None, limit=50):
        where, params = self._where(
            term=("lower(text) LIKE lower(?) ESCAPE '\\'", self._like(term)),
            meeting_id=("meeting_id = ?", meeting_id),
            speaker=("(speaker = ? OR display_name = ?)", speaker),
        )
        return self._records(
            "SELECT meeting_id, seg_id, start_s, end_s, speaker, display_name, text, "
            "meeting_id NOT IN (SELECT meeting_id FROM incomplete_ids) AS complete FROM segments_named"
            f"{where} ORDER BY meeting_date, meeting_id, seg_id LIMIT ?",
            [*params, int(limit)],
        )

    def meetings(self):
        return self._records(
            "SELECT meeting_id, any_value(title) AS title, any_value(meeting_date) AS meeting_date, "
            "count(*)::BIGINT AS segments, count(DISTINCT speaker)::BIGINT AS speakers, max(end_s) AS duration_s, "
            "string_agg(DISTINCT language, ', ' ORDER BY language) AS language, "
            "string_agg(DISTINCT model, ', ' ORDER BY model) AS model, "
            "string_agg(DISTINCT device, ', ' ORDER BY device) AS device, "
            "string_agg(DISTINCT backend, ', ' ORDER BY backend) AS backend, "
            "meeting_id NOT IN (SELECT meeting_id FROM incomplete_ids) AS complete "
            "FROM segments GROUP BY meeting_id ORDER BY meeting_date DESC, max(created_at) DESC, meeting_id"
        )

    def is_incomplete(self, meeting_id):
        return bool(self._records("SELECT meeting_id FROM incomplete_ids WHERE meeting_id = ?", [meeting_id]))

    def latest_meeting(self):
        rows = self._records("SELECT meeting_id FROM segments GROUP BY meeting_id ORDER BY max(created_at) DESC, meeting_id DESC LIMIT 1")
        return rows[0]["meeting_id"] if rows else None

    def speaker_stats(self, meeting_id=None):
        where, params = self._where(meeting_id=("meeting_id = ?", meeting_id))
        return self._records(
            "WITH t AS (SELECT meeting_id, display_name, speaker, GREATEST(end_s - start_s, 0) AS dur, "
            "CASE WHEN words IS NOT NULL AND len(words) > 0 THEN len(words) WHEN trim(text) = '' THEN 0 "
            "ELSE len(regexp_split_to_array(trim(text), '\\s+')) END AS n "
            f"FROM segments_named{where}) "
            "SELECT meeting_id, display_name AS speaker, list_sort(list(DISTINCT speaker)) AS labels, "
            "sum(dur) AS talk_s, sum(dur) / NULLIF(sum(sum(dur)) OVER (PARTITION BY meeting_id), 0) AS share, "
            "count(*)::BIGINT AS segments, sum(n)::BIGINT AS words "
            "FROM t GROUP BY meeting_id, display_name ORDER BY meeting_id, talk_s DESC, speaker",
            params,
        )


class Json:
    @classmethod
    def clean(cls, value):
        if isinstance(value, dict):
            return {str(k): cls.clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.clean(v) for v in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value

    @staticmethod
    def default(value):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    @classmethod
    def dumps(cls, data):
        return json.dumps(cls.clean(data), ensure_ascii=True, default=cls.default)


class Report:
    def __init__(self, query):
        self.query = query

    @staticmethod
    def clock(seconds):
        hours, rest = divmod(int(seconds or 0), 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def percent(share):
        return f"{(share or 0) * 100:.1f}".replace(".", ",") + " %"

    @staticmethod
    def cell(value):
        return str(value).replace("|", "\\|")

    def render(self, meeting_id):
        info = self.query._records(
            "SELECT any_value(title) AS title, any_value(meeting_date) AS meeting_date, count(*) AS n, "
            "max(end_s) AS duration_s, string_agg(DISTINCT language, ', ' ORDER BY language) AS language, "
            "string_agg(DISTINCT model, ', ' ORDER BY model) AS model, "
            "string_agg(DISTINCT device, ', ' ORDER BY device) AS device, "
            "string_agg(DISTINCT backend, ', ' ORDER BY backend) AS backend FROM segments WHERE meeting_id = ?",
            [meeting_id],
        )[0]
        if not info["n"]:
            raise MeetingNotFound(f"Unbekannte meeting_id: {meeting_id}")
        title = info["title"] or meeting_id
        lines = [f"# {title} – {info['meeting_date']:%d.%m.%Y}", ""]
        facts = (
            ("Dauer", self.clock(info["duration_s"])),
            ("Sprache", info["language"]),
            ("Modell", info["model"]),
            ("Gerät", info["device"]),
            ("Backend", info["backend"]),
        )
        lines += [f"- **{label}:** {value}" for label, value in facts if value]
        if self.query.is_incomplete(meeting_id):
            lines.append("- **Status:** unvollständig, die Verarbeitung wurde abgebrochen")
        lines += ["", "## Teilnehmer", "", "| Name | Redezeit | Redeanteil | Wörter |", "|---|---|---|---|"]
        for s in self.query.speaker_stats(meeting_id):
            lines.append(
                f"| {self.cell(s['speaker'])} | {self.clock(s['talk_s'])} | {self.percent(s['share'])} | {s['words']} |"
            )
        lines += ["", "## Transkript", ""]
        _, blocks = self.query.execute(
            "WITH o AS (SELECT display_name, start_s, trim(text) AS text, "
            "row_number() OVER (ORDER BY start_s, seg_id) AS rn FROM segments_named "
            "WHERE meeting_id = ? AND trim(text) <> ''), "
            "g AS (SELECT *, rn - row_number() OVER (PARTITION BY display_name ORDER BY rn) AS grp FROM o) "
            "SELECT display_name, min(start_s), string_agg(text, ' ' ORDER BY rn) FROM g "
            "GROUP BY display_name, grp ORDER BY min(rn)",
            [meeting_id],
        )
        for name, start, text in blocks:
            lines += [f"**[{self.clock(start)}] {name}:** {text}", ""]
        return "\n".join(lines).rstrip("\n") + "\n"


class ReportFile:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get("MEETSCRIBE_REPORTS") or PROJECT_ROOT / "protokolle")

    def target(self, day, slug):
        base = f"{as_date(day):%Y-%m-%d}_{slug}"
        candidate, number = self.directory / f"{base}.md", 1
        while candidate.exists():
            number += 1
            candidate = self.directory / f"{base}-{number}.md"
        return candidate

    def write(self, day, slug, text):
        path = self.target(day, slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
