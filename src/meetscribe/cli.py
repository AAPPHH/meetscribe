import os
import sys
from pathlib import Path

import click

from meetscribe.store import MeetingNotFound, Query, Report, Store, make_config

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYANNOTE_METRICS_ENABLED": "0",
}
DEVICE_CHOICES = ("npu", "gpu", "cpu")
TABLE = {"clip": 100, "float_digits": 2}


def reconfigure_streams():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


class Table:
    def __init__(self, columns, rows, clip=TABLE["clip"]):
        self.columns = list(columns)
        self.rows = [[self.cell(v, clip) for v in row] for row in rows]

    @staticmethod
    def cell(value, clip):
        if value is None:
            return ""
        if isinstance(value, float):
            text = f"{value:.{TABLE['float_digits']}f}"
        elif isinstance(value, (list, tuple)):
            text = ", ".join(str(v) for v in value)
        else:
            text = str(value)
        text = " ".join(text.split())
        return text if not clip or len(text) <= clip else text[: clip - 1] + "…"

    def render(self):
        if not self.rows:
            return "(keine Treffer)"
        widths = [max(len(c), *(len(r[i]) for r in self.rows)) for i, c in enumerate(self.columns)]
        line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
        return "\n".join([line(self.columns), line("-" * w for w in widths), *(line(r) for r in self.rows)])

    def echo(self):
        click.echo(self.render())


def echo_records(records, columns=None, clip=TABLE["clip"]):
    columns = columns or (list(records[0]) if records else [])
    Table(columns, [[r[c] for c in columns] for r in records], clip).echo()


def stdin_is_tty():
    return sys.stdin.isatty()


def get_store():
    return Store(make_config())


def require_meeting(store, meeting_id):
    if meeting_id is None:
        meeting_id = Query(store).latest_meeting()
        if meeting_id is None:
            raise click.ClickException("Keine Besprechung vorhanden.")
        return meeting_id
    try:
        known = store.exists(meeting_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not known:
        raise click.ClickException(f"Unbekannte meeting_id: {meeting_id}")
    return meeting_id


@click.group(help="Lokale Besprechungsabschriften: Aufnahme, Transkription, Archiv und Protokoll.")
@click.pass_context
def main(ctx):
    reconfigure_streams()
    if ctx.invoked_subcommand != "setup":
        os.environ.update(OFFLINE_ENV)


@main.command(help="Modelle herunterladen (einziger Befehl mit Netzzugriff).")
def setup():
    for name in OFFLINE_ENV:
        os.environ.pop(name, None)
    from meetscribe import pipeline
    models = pipeline.Models()
    setup_job = pipeline.ModelSetup(models, os.environ.get("HF_TOKEN"))
    messages, failed = setup_job.run()
    for message in messages:
        click.echo(message)
    if pipeline.Models.whisper_ready(models.whisper_dir("npu")):
        click.echo("NPU-Compile-Cache wird aufgewaermt; das dauert einmalig mehrere Minuten.")
        try:
            with pipeline.KeepAwake():
                setup_job.warm("npu")
        except Exception as exc:
            click.echo(f"Warnung: NPU-Cache nicht aufgewaermt: {pipeline.Reason.of(exc)}", err=True)
    if failed:
        raise click.ClickException("Whisper-Download fehlgeschlagen.")


@main.command(help="Audio-Eingaenge und OpenVINO-Geraete auflisten.")
def devices():
    from meetscribe import pipeline
    try:
        picker = pipeline.AudioDevices.sounddevice_picker()
        Table(
            ["index", "name", "api", "kanaele", "rate"],
            [[d["index"], d["name"], picker.hostapis[d["hostapi"]], d["max_input_channels"], int(d["default_samplerate"])]
             for d in picker.inputs()],
        ).echo()
    except Exception as exc:
        click.echo(f"Audio-Eingaenge nicht lesbar: {pipeline.Reason.of(exc)}", err=True)
    click.echo()
    Table(["openvino", "name"], [[r["device"], r["name"]] for r in pipeline.AudioDevices.openvino_devices()]).echo()


@main.command(help="Aufnahme bis Strg+C oder --seconds.")
@click.option("--mic", default="laptop", show_default=True, help="dji, laptop, both oder Namensteil des Geraets.")
@click.option("--out", "out", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Zieldatei (.flac).")
@click.option("--seconds", type=float, default=None, help="Aufnahmedauer in Sekunden.")
def record(mic, out, seconds):
    from meetscribe import pipeline
    session = pipeline.RecordingSession(mic, out, pipeline.AudioDevices.sounddevice_picker())
    for recorder in session.prepare():
        click.echo(f"{recorder.device['name']} [{recorder.hostapi}], {recorder.device['max_input_channels']} Kanaele, {recorder.rate} Hz")

    def announce():
        click.echo("Aufnahme laeuft, Ende mit Strg+C." if seconds is None else f"Aufnahme laeuft {seconds:g} s.")
        click.echo(pipeline.RECORDING["lid_notice"])

    with pipeline.KeepAwake():
        reports = session.run(seconds, announce)
    for report in reports:
        click.echo(f"{report['path']}: {report['seconds']:.1f} s, Ueberlaeufe: {report['overruns']}")
    if session.final == session.mix_path:
        click.echo(f"{session.mix_path}: Mix")
    problems = [f"{r['path']}: {r['problem']}" for r in reports if r["problem"]]
    if problems:
        raise click.ClickException("Probleme waehrend der Aufnahme: " + "; ".join(problems))


@main.command(help="Audio transkribieren und im Archiv ablegen.")
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--device", type=click.Choice(DEVICE_CHOICES, case_sensitive=False), envvar="MEETSCRIBE_DEVICE",
              default="npu", show_default=True, help="Gewuenschtes Geraet; Fallback npu -> gpu -> cpu.")
@click.option("--model", default=None, help="Modellverzeichnis (Pfad oder Name unter models/).")
@click.option("--language", default="de", show_default=True)
@click.option("--title", default=None)
@click.option("--date", "day", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, help="Datum JJJJ-MM-TT.")
@click.option("--meeting-id", default=None)
@click.option("--no-diarize", is_flag=True, help="Keine Sprechertrennung (Sprecher UNKNOWN).")
@click.option("--no-fallback", is_flag=True, help="Bei Geraetefehler abbrechen statt weiterzumachen.")
@click.option("--speakers", type=int, default=None)
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
def transcribe(audio, device, model, language, title, day, meeting_id, no_diarize, no_fallback, speakers, min_speakers, max_speakers):
    from meetscribe import pipeline
    from meetscribe.store import DuplicateMeetingError
    job = pipeline.TranscribeJob(
        get_store(), audio, device, language, model=model, title=title, day=day.date() if day else None,
        meeting_id=meeting_id, diarize=not no_diarize, fallback=not no_fallback,
        speakers=speakers, min_speakers=min_speakers, max_speakers=max_speakers,
    )
    try:
        with pipeline.KeepAwake():
            result = job.run()
    except DuplicateMeetingError as exc:
        raise click.ClickException(str(exc)) from exc
    if not result["segments"]:
        raise click.ClickException("Keine Sprache erkannt, nichts gespeichert.")
    click.echo(
        f"{result['meeting_id']}: {result['segments']} Segmente, {result['speakers']} Sprecher, "
        f"Geraet angefordert: {result['requested']}, genutzt: {' -> '.join(result['used']) or '-'}"
    )


@main.command(help="Aufnehmen (oder AUDIOFILE nehmen), transkribieren, Sprecher benennen, Protokoll schreiben.")
@click.argument("audio", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--title", default=None)
@click.option("--mic", default=None, help="dji, laptop, both oder Namensteil; Standard: DJI, sonst Laptop.")
@click.option("--device", type=click.Choice(DEVICE_CHOICES, case_sensitive=False), envvar="MEETSCRIBE_DEVICE",
              default="npu", show_default=True, help="Gewuenschtes Geraet; Fallback npu -> gpu -> cpu.")
@click.option("--speakers", type=int, default=None)
@click.option("--no-diarize", is_flag=True, help="Keine Sprechertrennung (Sprecher UNKNOWN).")
@click.option("--no-open", is_flag=True, help="Protokoll nicht oeffnen.")
def start(audio, title, mic, device, speakers, no_diarize, no_open):
    from meetscribe import pipeline
    flow = pipeline.StartFlow(
        get_store(), audio=audio, mic=mic, device=device, title=title, speakers=speakers,
        diarize=not no_diarize, open_report=not no_open, tty=stdin_is_tty(),
    )
    with pipeline.KeepAwake():
        flow.run()


@main.command(help="Heisse Dateien ins Parquet-Archiv verschieben.")
@click.option("--older-than-days", type=float, default=0, show_default=True)
@click.option("--force", is_flag=True, help="Auch unvollstaendige Dateien rotieren.")
def rotate(older_than_days, force):
    result = get_store().rotate(older_than_days=older_than_days, force=force)
    click.echo(f"rotiert: {', '.join(result['rotated']) or '-'}")
    for skipped in result["skipped"]:
        click.echo(f"uebersprungen: {skipped['meeting_id']} ({skipped['reason']})")


@main.command(help="Teilstring-Suche (ohne Gross-/Kleinschreibung) in den Segmenten.")
@click.argument("term")
@click.option("--meeting", default=None)
@click.option("--speaker", default=None)
@click.option("--limit", type=int, default=20, show_default=True)
def search(term, meeting, speaker, limit):
    rows = Query(get_store()).search(term, meeting_id=meeting, speaker=speaker, limit=limit)
    echo_records(rows, ["meeting_id", "seg_id", "start_s", "display_name", "text"])


@main.command(help="Alle Besprechungen auflisten.")
def meetings():
    echo_records(Query(get_store()).meetings(), ["meeting_id", "title", "meeting_date", "segments", "speakers", "duration_s", "device"])


@main.command(help="Redezeit und Woerter je Sprecher.")
@click.option("--meeting", default=None)
def stats(meeting):
    rows = Query(get_store()).speaker_stats(meeting)
    for row in rows:
        row["share"] = None if row["share"] is None else f"{row['share'] * 100:.1f} %"
    echo_records(rows, ["meeting_id", "speaker", "labels", "talk_s", "share", "segments", "words"])


@main.group(name="speakers", help="Sprecherzuordnung anzeigen und aendern.")
def speakers_group():
    pass


MEETING_OPTION = click.option("--meeting", "meeting_id", default=None, help="Besprechung; Standard: die zuletzt erstellte.")


@speakers_group.command(name="show")
@MEETING_OPTION
def speakers_show(meeting_id):
    store = get_store()
    meeting_id = require_meeting(store, meeting_id)
    columns, rows = Query(store).execute(
        "SELECT speaker AS label, any_value(display_name) AS name, sum(end_s - start_s) AS talk_s, count(*) AS segments "
        "FROM segments_named WHERE meeting_id = ? GROUP BY speaker ORDER BY speaker",
        [meeting_id],
    )
    Table(columns, rows).echo()


def known_labels(store, meeting_id):
    return [r[0] for r in Query(store).execute("SELECT DISTINCT speaker FROM segments WHERE meeting_id = ? ORDER BY 1", [meeting_id])[1]]


@speakers_group.command(name="set")
@click.argument("label")
@click.argument("name")
@MEETING_OPTION
def speakers_set(label, name, meeting_id):
    store = get_store()
    meeting_id = require_meeting(store, meeting_id)
    labels = known_labels(store, meeting_id)
    if label not in labels:
        raise click.ClickException(f"Unbekanntes Label '{label}'. Vorhanden: {', '.join(labels)}")
    try:
        store.speakers.set(meeting_id, label, name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{meeting_id} {label} = {name.strip()}")


@speakers_group.command(name="unset")
@click.argument("label")
@MEETING_OPTION
def speakers_unset(label, meeting_id):
    store = get_store()
    meeting_id = require_meeting(store, meeting_id)
    if not store.speakers.unset(meeting_id, label):
        raise click.ClickException(f"Keine Zuordnung fuer {label} in {meeting_id}.")
    click.echo(f"{meeting_id} {label} entfernt")


@main.command(help="Markdown-Protokoll erzeugen (Standard: letzte Besprechung).")
@click.argument("meeting_id", required=False)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Zieldatei statt Ausgabe.")
def report(meeting_id, out):
    store = get_store()
    meeting_id = require_meeting(store, meeting_id)
    try:
        text = Report(Query(store)).render(meeting_id)
    except MeetingNotFound as exc:
        raise click.ClickException(str(exc)) from exc
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        click.echo(f"geschrieben: {out}")
    else:
        click.echo(text, nl=False)


@main.command(name="sql", help="SQL-Abfrage auf segments, segments_named und speakers.")
@click.argument("query")
def sql(query):
    import duckdb
    try:
        columns, rows = Query(get_store()).execute(query)
    except duckdb.Error as exc:
        raise click.ClickException(str(exc)) from exc
    Table(columns, rows, clip=None).echo()
