import os
import shutil
from datetime import date

import pytest

from meetscribe.store import (
    CONFIG, DuplicateMeetingError, EOS, HotFile, MeetingNotFound, Query, Report, Store, make_config,
)

ALL_ROWS = "SELECT * FROM segments ORDER BY meeting_id, seg_id"


def segs(offset=0.0):
    return [
        {"start_s": offset, "end_s": offset + 4.0, "speaker": "SPEAKER_00", "text": "Gr\u00fc\u00dfe aus M\u00fcnchen, \u00fcber alles.",
         "words": [{"text": "Gr\u00fc\u00dfe", "start_s": offset, "end_s": offset + 1.0, "prob": 0.9}, {"text": "aus", "start_s": offset + 1.0, "end_s": offset + 2.0}]},
        {"start_s": offset + 4.0, "end_s": offset + 6.0, "speaker": "SPEAKER_01", "text": "Sch\u00f6n, dass Sie da sind."},
        {"start_s": offset + 6.0, "end_s": offset + 9.0, "speaker": "SPEAKER_01", "text": "Fangen wir an."},
    ]


@pytest.fixture
def store(tmp_path):
    return Store(make_config(tmp_path / "data"))


def write(store, meeting_id="m1", title="Wochenbesprechung", day="2026-03-04", batches=(0.0,), **meta):
    meta = {"language": "de", "model": "whisper-small", "device": "NPU", **meta}
    with store.writer(meeting_id, title=title, meeting_date=day, **meta) as w:
        for offset in batches:
            w.append(segs(offset))
    return store.hot_path(meeting_id)


def rows(store, sql=ALL_ROWS):
    return Query(store).execute(sql)[1]


def test_config_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MEETSCRIBE_DATA", str(tmp_path / "x"))
    cfg = make_config()
    assert cfg["data_dir"] == tmp_path / "x"
    assert cfg["hot_dir"] == tmp_path / "x" / "hot"
    assert cfg["cold_dir"].name == "cold" and cfg["archive_dir"].name == "archive"
    monkeypatch.delenv("MEETSCRIBE_DATA")
    assert make_config()["data_dir"].name == "data" and CONFIG["hot_dir"].name == "hot"


def test_roundtrip_umlauts(store):
    path = write(store, batches=(0.0, 10.0))
    assert path.read_bytes().endswith(EOS)
    table, complete = HotFile(path).read()
    assert complete and table.num_rows == 6
    assert table.column("seg_id").to_pylist() == [0, 1, 2, 3, 4, 5]
    cols, data = Query(store).execute(ALL_ROWS)
    first = dict(zip(cols, data[0]))
    assert first["text"] == "Gr\u00fc\u00dfe aus M\u00fcnchen, \u00fcber alles."
    assert first["meeting_date"] == date(2026, 3, 4)
    assert first["created_at"].tzinfo is not None
    assert [w["text"] for w in first["words"]] == ["Gr\u00fc\u00dfe", "aus"]
    assert data[1][cols.index("words")] is None
    assert [r["seg_id"] for r in Query(store).search("SCHÖN")] == [1, 4]


def test_truncated_hot_file(store):
    path = write(store, batches=(0.0, 10.0))
    full = path.read_bytes()
    with store.writer("probe") as w:
        w.append(segs())
        one = store.hot_path("probe").stat().st_size
        w.append(segs())
        two = store.hot_path("probe").stat().st_size
    assert one < two
    probe = store.hot_path("probe").read_bytes()
    target = store.hot_path("cut")
    target.write_bytes(probe[: two - 5])
    table, complete = HotFile(target).read()
    assert not complete and table.num_rows == 3
    target.write_bytes(probe[: one])
    table, complete = HotFile(target).read()
    assert not complete and table.num_rows == 3
    target.write_bytes(probe[:-8])
    table, complete = HotFile(target).read()
    assert not complete and table.num_rows == 6
    for size in (0, 3, 20, 100):
        target.write_bytes(probe[:size])
        table, complete = HotFile(target).read()
        assert not complete and table.num_rows == 0
    previous = 0
    for size in range(0, len(full) + 1, 7):
        target.write_bytes(full[:size])
        table, complete = HotFile(target).read()
        assert table.num_rows >= previous
        previous = table.num_rows
        assert complete == (size == len(full))
    target.write_bytes(full[: len(full) - 9])
    for other in ("m1", "probe"):
        store.hot_path(other).unlink()
    assert Query(store).execute("SELECT count(*) FROM segments")[1][0][0] == 3
    assert set(store.incomplete()) == {"cut"}


def test_crashed_writer_has_no_eos(store):
    with pytest.raises(RuntimeError):
        with store.writer("boom") as w:
            w.append(segs())
            raise RuntimeError("crash")
    table, complete = HotFile(store.hot_path("boom")).read()
    assert not complete and table.num_rows == 3


def test_live_read_while_writing(store):
    with store.writer("live", meeting_date="2026-01-01") as w:
        w.append(segs())
        table, complete = HotFile(store.hot_path("live")).read()
        assert table.num_rows == 3 and not complete
        assert rows(store, "SELECT count(*) FROM segments")[0][0] == 3


def test_rotate_equivalence(store):
    write(store, "m1", day="2026-03-04")
    write(store, "m2", day="2026-04-09", batches=(0.0, 20.0))
    before = rows(store)
    assert len(before) == 9
    result = store.rotate()
    assert sorted(result["rotated"]) == ["m1", "m2"] and result["skipped"] == []
    assert store.hot_files() == []
    assert sorted(p.name for p in store.archive_dir.glob("*.arrow")) == ["m1.arrow", "m2.arrow"]
    assert (store.cold_dir / "year=2026" / "month=03" / "m1.parquet").exists()
    assert (store.cold_dir / "year=2026" / "month=04" / "m2.parquet").exists()
    after = rows(store)
    assert after == before
    cols = Query(store).execute("SELECT * FROM segments")[0]
    assert cols == ["meeting_id", "title", "meeting_date", "seg_id", "start_s", "end_s", "speaker", "text",
                    "language", "model", "device", "words", "created_at"]
    assert store.rotate() == {"rotated": [], "skipped": []}


def test_same_query_instance_sees_rotation(store):
    write(store)
    query = Query(store)
    before = query.execute(ALL_ROWS)[1]
    store.rotate()
    assert query.execute(ALL_ROWS)[1] == before


def test_crash_both_hot_and_cold(store):
    write(store)
    before = rows(store)
    store.rotate()
    shutil.copy(store.archive_dir / "m1.arrow", store.hot_path("m1"))
    cold = next(store.cold_dir.glob("**/*.parquet"))
    (cold.parent / "stale.parquet.tmp").write_bytes(b"junk")
    assert rows(store) == before
    assert rows(store, "SELECT count(*) FROM segments")[0][0] == 3
    assert rows(store, "SELECT count(*) FROM segments_named")[0][0] == 3
    assert store.rotate() == {"rotated": ["m1"], "skipped": []}
    assert store.hot_files() == [] and not list(store.cold_dir.glob("**/*.tmp"))
    assert rows(store) == before


def test_crash_before_cold_replace(store):
    write(store)
    before = rows(store)
    cold_dir = store.cold_dir / "year=2026" / "month=03"
    cold_dir.mkdir(parents=True)
    (cold_dir / "m1.parquet.tmp").write_bytes(b"half written")
    assert rows(store) == before
    assert store.rotate()["rotated"] == ["m1"]
    assert rows(store) == before


def test_rotate_skips_incomplete_unless_forced(store):
    write(store, "ok")
    with pytest.raises(RuntimeError):
        with store.writer("bad", meeting_date="2026-03-05") as w:
            w.append(segs())
            raise RuntimeError
    result = store.rotate()
    assert result["rotated"] == ["ok"]
    assert result["skipped"] == [{"meeting_id": "bad", "reason": "incomplete"}]
    assert [p.stem for p in store.hot_files()] == ["bad"]
    assert store.rotate(force=True)["rotated"] == ["bad"]
    assert rows(store, "SELECT count(*) FROM segments")[0][0] == 6


def test_rotate_older_than_days(store):
    path = write(store)
    assert store.rotate(older_than_days=1)["skipped"] == [{"meeting_id": "m1", "reason": "too_recent"}]
    old = path.stat().st_mtime - 3 * 86400
    os.utime(path, (old, old))
    assert store.rotate(older_than_days=1)["rotated"] == ["m1"]


def test_empty_views(store):
    query = Query(store)
    cols, data = query.execute("SELECT * FROM segments")
    assert data == [] and cols[0] == "meeting_id" and "year" not in cols
    assert query.execute("SELECT * FROM segments_named")[1] == []
    assert query.meetings() == [] and query.speaker_stats() == [] and query.search("x") == []
    assert query.execute("SELECT count(*) FROM speakers")[1] == [(0,)]
    store.hot_dir.mkdir(parents=True)
    write(store)
    only_hot = Query(store)
    assert only_hot.execute("SELECT count(*) FROM segments")[1] == [(3,)]
    store.rotate()
    only_cold = Query(store)
    assert only_cold.execute("SELECT count(*) FROM segments")[1] == [(3,)]
    assert only_cold.execute("SELECT * FROM segments")[0] == cols


def test_speaker_mapping_leaves_raw_files_untouched(store):
    write(store, "m1")
    write(store, "m2", day="2026-04-01")
    store.rotate()
    write(store, "m3", day="2026-05-01")
    raw = {p: p.read_bytes() for p in [*store.hot_files(), *store.cold_dir.glob("**/*.parquet")]}
    assert len(raw) == 3
    store.speakers.set("m1", "SPEAKER_00", "Anna Schmidt")
    store.speakers.set("m3", "SPEAKER_01", "Bj\u00f6rn")
    store.speakers.set("m3", "SPEAKER_01", "Bj\u00f6rn K\u00fchn")
    query = Query(store)
    named = query.execute(
        "SELECT meeting_id, speaker, display_name FROM segments_named WHERE seg_id IN (0, 1) ORDER BY meeting_id, seg_id")[1]
    assert named == [
        ("m1", "SPEAKER_00", "Anna Schmidt"), ("m1", "SPEAKER_01", "SPEAKER_01"),
        ("m2", "SPEAKER_00", "SPEAKER_00"), ("m2", "SPEAKER_01", "SPEAKER_01"),
        ("m3", "SPEAKER_00", "SPEAKER_00"), ("m3", "SPEAKER_01", "Bj\u00f6rn K\u00fchn"),
    ]
    assert query.execute("SELECT count(*) FROM speakers")[1] == [(2,)]
    assert {p: p.read_bytes() for p in raw} == raw
    assert store.speakers.unset("m1", "SPEAKER_00") and not store.speakers.unset("m1", "SPEAKER_00")
    assert query.execute("SELECT display_name FROM segments_named WHERE meeting_id = 'm1' AND seg_id = 0")[1] == [("SPEAKER_00",)]
    assert {p: p.read_bytes() for p in raw} == raw
    assert not list(store.config["data_dir"].glob("*.tmp"))
    with pytest.raises(ValueError):
        store.speakers.set("m1", "SPEAKER_00", "  ")


def test_speaker_stats(store):
    write(store)
    store.speakers.set("m1", "SPEAKER_01", "Bea")
    stats = Query(store).speaker_stats("m1")
    assert [s["speaker"] for s in stats] == ["Bea", "SPEAKER_00"]
    by = {s["speaker"]: s for s in stats}
    assert by["SPEAKER_00"]["talk_s"] == 4.0 and by["Bea"]["talk_s"] == 5.0
    assert by["Bea"]["segments"] == 2 and by["SPEAKER_00"]["segments"] == 1
    assert by["SPEAKER_00"]["words"] == 2 and by["Bea"]["words"] == 8
    assert abs(by["Bea"]["share"] - 5 / 9) < 1e-9 and by["Bea"]["labels"] == ["SPEAKER_01"]
    assert len(Query(store).speaker_stats()) == 2


def test_search_like_wildcards_are_literal(store):
    with store.writer("w", meeting_date="2026-01-01") as w:
        w.append([
            {"start_s": 0, "end_s": 1, "speaker": "A", "text": "100% sicher"},
            {"start_s": 1, "end_s": 2, "speaker": "A", "text": "a_b und axb"},
            {"start_s": 2, "end_s": 3, "speaker": "B", "text": "Pfad C:\\temp\\x"},
            {"start_s": 3, "end_s": 4, "speaker": "B", "text": "Gro\u00dfe \u00dcberraschung \u00fcber M\u00dcLLER"},
        ])
    query = Query(store)
    assert [r["text"] for r in query.search("%")] == ["100% sicher"]
    assert [r["text"] for r in query.search("_")] == ["a_b und axb"]
    assert [r["text"] for r in query.search("a_b")] == ["a_b und axb"]
    assert [r["text"] for r in query.search("a%b")] == []
    assert [r["text"] for r in query.search("\\temp")] == ["Pfad C:\\temp\\x"]
    assert [r["seg_id"] for r in query.search("\u00dcBER")] == [3]
    assert [r["seg_id"] for r in query.search("m\u00fcller")] == [3]
    assert [r["seg_id"] for r in query.search("SICHER", meeting_id="w", speaker="A")] == [0]
    assert query.search("sicher", speaker="B") == [] and query.search("sicher", meeting_id="nope") == []
    assert len(query.search("", limit=2)) == 2


def test_search_sql_injection_shapes(store):
    write(store)
    query = Query(store)
    for term in ("'; DROP VIEW segments; --", "%' OR '1'='1", "\" OR 1=1 --", "x') UNION SELECT 1 --"):
        assert query.search(term) == []
        assert query.search("Fangen", meeting_id=term, speaker=term) == []
    assert len(query.search("fangen")) == 1
    assert query.execute("SELECT count(*) FROM segments")[1] == [(3,)]


def test_free_sql_with_params(store):
    write(store)
    cols, data = Query(store).execute("SELECT speaker, count(*) AS n FROM segments WHERE meeting_id = ? GROUP BY speaker ORDER BY speaker", ["m1"])
    assert cols == ["speaker", "n"] and data == [("SPEAKER_00", 1), ("SPEAKER_01", 2)]


def test_path_with_space_and_apostrophe(tmp_path):
    store = Store(make_config(tmp_path / "O'Brien data [x]" / "my files"))
    write(store, "m1")
    write(store, "m2", day="2026-05-05")
    before = rows(store)
    assert store.rotate(force=False)["rotated"] == ["m1", "m2"]
    write(store, "m3", day="2026-06-06")
    after = rows(store)
    assert len(after) == 9 and after[:6] == before
    assert len(Query(store).search("fangen")) == 3
    store.speakers.set("m1", "SPEAKER_00", "Zo\u00eb")
    assert Query(store).search("Gr\u00fc\u00dfe", meeting_id="m1")[0]["display_name"] == "Zo\u00eb"


def test_meetings_listing(store):
    write(store, "a", title="Alt", day="2026-01-01")
    write(store, "b", title="Neu", day="2026-02-01", language="en")
    store.rotate()
    write(store, "c", title="Heiss", day="2026-03-01")
    listing = Query(store).meetings()
    assert [m["meeting_id"] for m in listing] == ["c", "b", "a"]
    assert listing[1]["language"] == "en" and listing[0]["segments"] == 3 and listing[0]["speakers"] == 2
    assert listing[0]["duration_s"] == 9.0 and listing[0]["meeting_date"] == date(2026, 3, 1)


def test_report_content(store):
    with store.writer("r1", title="Projekt K\u00fcche", meeting_date="2026-03-04", language="de",
                      model="whisper-small", device="NPU") as w:
        w.append([
            {"start_s": 0, "end_s": 4, "speaker": "SPEAKER_00", "text": "Guten Morgen."},
            {"start_s": 4, "end_s": 8, "speaker": "SPEAKER_00", "text": "Wir starten."},
            {"start_s": 8, "end_s": 12, "speaker": "SPEAKER_01", "text": "Sch\u00f6n, dass Sie da sind."},
            {"start_s": 12, "end_s": 14, "speaker": "SPEAKER_00", "text": "  "},
            {"start_s": 3725, "end_s": 3730, "speaker": "SPEAKER_00", "text": "Sp\u00e4ter noch ein Punkt."},
        ])
    store.speakers.set("r1", "SPEAKER_00", "Anna | Chefin")
    store.speakers.set("r1", "SPEAKER_01", "Bj\u00f6rn")
    text = Report(Query(store)).render("r1")
    assert text.startswith("# Projekt K\u00fcche \u2013 04.03.2026\n")
    assert "- **Dauer:** 1:02:10" in text
    assert "- **Sprache:** de" in text and "- **Modell:** whisper-small" in text and "- **Ger\u00e4t:** NPU" in text
    assert "| Anna \| Chefin | 00:15 | 78,9 % | 8 |" in text
    assert "| Björn | 00:04 | 21,1 % | 5 |" in text
    assert "**[00:00] Anna | Chefin:** Guten Morgen. Wir starten." in text
    assert "**[00:08] Bj\u00f6rn:** Sch\u00f6n, dass Sie da sind." in text
    assert "**[1:02:05] Anna | Chefin:** Sp\u00e4ter noch ein Punkt." in text
    assert text.count("Guten Morgen") == 1 and text.index("[00:00]") < text.index("[00:08]") < text.index("[1:02:05]")
    assert text.endswith("\n")
    store.rotate()
    assert Report(Query(store)).render("r1") == text


def test_report_unknown_meeting(store):
    write(store)
    with pytest.raises(MeetingNotFound, match="nope"):
        Report(Query(store)).render("nope")
    with pytest.raises(MeetingNotFound):
        Report(Query(Store(make_config(store.config["data_dir"] / "leer")))).render("m1")


def test_refuses_duplicate_meeting_id(store):
    path = write(store)
    content = path.read_bytes()
    with pytest.raises(DuplicateMeetingError):
        with store.writer("m1") as w:
            w.append(segs())
    assert path.read_bytes() == content
    store.rotate()
    with pytest.raises(DuplicateMeetingError):
        with store.writer("m1"):
            pass
    assert store.hot_files() == []


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a\\b", ".hidden", "a b", "x:y"])
def test_invalid_meeting_id(store, bad):
    with pytest.raises(ValueError):
        store.writer(bad)


def test_append_validation(store):
    with store.writer("v", meeting_date="2026-01-01") as w:
        with pytest.raises(ValueError):
            w.append([{"start_s": 0, "end_s": 1, "text": "ohne Sprecher"}])
        assert w.append([]) == 0
        assert w.append([{"start_s": 0, "end_s": 1, "speaker": "A", "text": "ok"}]) == 1
    assert rows(store, "SELECT seg_id FROM segments") == [(0,)]


class FailingClose:
    def __init__(self, inner, target):
        self.inner, self.target = inner, target

    def close(self):
        if self.target == "stream":
            raise OSError("disk full")
        self.inner.close()
        raise OSError("disk full")

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.mark.parametrize("target", ["stream", "file"])
@pytest.mark.parametrize("discard", [True, False])
def test_close_failure_discards_only_when_requested(store, target, discard):
    with pytest.raises(OSError, match="disk full"):
        with store.writer("c1", title="t", meeting_date="2026-03-04", discard_on_error=discard) as w:
            w.append(segs())
            setattr(w, target, FailingClose(getattr(w, target), target))
    assert store.hot_path("c1").exists() is not discard
    if discard:
        assert store.incomplete() == [] and not store.exists("c1")
