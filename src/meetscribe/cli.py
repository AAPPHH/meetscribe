import contextlib
import os
import sys
from pathlib import Path

import click

from meetscribe.backends import BACKEND_CHOICES, CATALOG, HARDWARE_CHOICES, Catalog, Selection
from meetscribe.service import OFFLINE_ENV, EngineCache, RpcServer, Service, Wire
from meetscribe.store import Json, Store, make_config

TABLE = {"clip": 100, "float_digits": 2}
STATUS_WORDS = {"ok": "ok", "unverified": "ungeprüft", "refused": "nein"}


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


class Output:
    def __init__(self, machine=False):
        self.machine = machine

    def dump(self, data):
        click.echo(Json.dumps(data))

    def table(self, columns, rows, clip=TABLE["clip"]):
        if self.machine:
            self.dump({"columns": list(columns), "rows": [list(r) for r in rows]})
        else:
            Table(columns, rows, clip).echo()

    def records(self, records, columns=None, formats=None, clip=TABLE["clip"]):
        if self.machine:
            return self.dump(records)
        columns = columns or (list(records[0]) if records else [])
        formats = formats or {}
        shown = lambda r, c: formats[c](r[c]) if c in formats and r[c] is not None else r[c]
        Table(columns, [[shown(r, c) for c in columns] for r in records], clip).echo()


def out():
    return click.get_current_context().obj


def stdin_is_tty():
    return sys.stdin.isatty()


def get_store():
    return Store(make_config())


def get_service():
    return Service(get_store())


def get_service_with_cache():
    return Service(get_store(), EngineCache())


def selection_options(command):
    options = [
        click.option("--backend", type=click.Choice(BACKEND_CHOICES, case_sensitive=False), envvar="MEETSCRIBE_BACKEND", default="auto",
                     show_default=True, help="Spracherkennungs-Backend (siehe 'meetscribe models')."),
        click.option("--device", type=click.Choice(HARDWARE_CHOICES, case_sensitive=False), envvar="MEETSCRIBE_DEVICE", default="auto",
                     show_default=True, help="Hardware; auto wählt npu, gpu, igpu, cpu in dieser Reihenfolge, Rückfall bei Fehlern."),
        click.option("--model", default=None, help="Modell aus 'meetscribe models' (optional mit :Variante) oder Modellverzeichnis; Standard large-v3-turbo."),
    ]
    for option in reversed(options):
        command = option(command)
    return command


JSON_FLAG = click.option("--json", "as_json", is_flag=True, help="Maschinenlesbare JSON-Ausgabe.")


def want_json(as_json):
    if as_json:
        out().machine = True


class Cli(click.Group):
    def main(self, *args, **kwargs):
        kwargs.setdefault("windows_expand_args", False)
        return super().main(*args, **kwargs)


@click.group(cls=Cli, help="Lokale Besprechungsabschriften: Aufnahme, Transkription, Archiv und Protokoll.")
@click.option("--json", "machine", is_flag=True, help="Maschinenlesbare JSON-Ausgabe (meetings, search, stats, speakers show, report, sql, devices).")
@click.pass_context
def main(ctx, machine):
    ctx.obj = Output(machine)
    reconfigure_streams()
    if ctx.invoked_subcommand != "setup":
        os.environ.update(OFFLINE_ENV)


@main.command(help="Modelle herunterladen (einziger Befehl mit Netzzugriff).")
@click.option("--model", default=None, help="Nur dieses Modell laden (siehe 'meetscribe models'), optional mit :Variante.")
@click.option("--backend", type=click.Choice(BACKEND_CHOICES, case_sensitive=False), default=None, help="Backend des Modells (Standard openvino).")
@click.option("--device", type=click.Choice(HARDWARE_CHOICES, case_sensitive=False), default=None, help="Nur die Variante für diese Hardware laden.")
@click.option("--events", is_flag=True, help="Fortschritt und Ergebnis als JSON-Zeilen auf stdout.")
def setup(model, backend, device, events):
    for name in OFFLINE_ENV:
        os.environ.pop(name, None)
    from meetscribe import pipeline
    reporter = pipeline.EventReporter() if events else pipeline.Reporter()
    models = pipeline.Models()
    setup_job = pipeline.ModelSetup(models, os.environ.get("HF_TOKEN"), reporter)
    with contextlib.redirect_stdout(sys.stderr) if events else contextlib.nullcontext():
        try:
            messages, failed = setup_job.run(model, backend, device)
        except click.ClickException as exc:
            if not events:
                raise
            reporter.event("error", message=exc.message, detail=pipeline.Reason.detail(exc), exit_code=1)
            raise click.exceptions.Exit(1) from exc
        for message in messages:
            click.echo(message)
    if events:
        if failed:
            reporter.event("error", message="Der Download ist fehlgeschlagen.", detail="\n".join(messages), exit_code=1)
            raise click.exceptions.Exit(1)
        return reporter.event("done", messages=messages)
    if not any((model, backend, device)):
        try:
            plan = Selection(models).resolve("openvino", "npu", fallback=False)
        except click.ClickException:
            plan = None
        if plan and plan.installed:
            click.echo("NPU-Compile-Cache wird aufgewaermt; das dauert einmalig mehrere Minuten.")
            try:
                with pipeline.KeepAwake():
                    setup_job.warm(plan.primary)
            except Exception as exc:
                click.echo(f"Warnung: NPU-Cache nicht aufgewaermt: {pipeline.Reason.of(exc)}", err=True)
    if failed:
        raise click.ClickException("Whisper-Download fehlgeschlagen.")


@main.command(help="Audio-Eingaenge und erkannte Hardware (NPU, GPU, iGPU, CPU) auflisten.")
@JSON_FLAG
def devices(as_json):
    want_json(as_json)
    report = get_service().devices()
    if out().machine:
        return out().dump(report)
    if report["inputs_error"]:
        click.echo(report["inputs_error"], err=True)
    else:
        Table(["index", "name", "api", "kanaele", "rate"], [[d["index"], d["name"], d["api"], d["channels"], d["rate"]] for d in report["inputs"]]).echo()
    click.echo()
    Table(["hardware", "geraet", "name", "typ", "backends"], [[Catalog.label(r["id"]), r["device"], r["name"], r["type"], ", ".join(r["backends"]) or "-"] for r in report["hardware"]]).echo()
    click.echo()
    Table(["backend", "installiert", "hinweis"], [[b["label"], "ja" if b["installed"] else "nein", b["reason"]] for b in report["backends"]]).echo()


@main.command(name="models", help="Modelle aus dem Katalog mit Installationsstand, Groesse und Eignung je Hardware.")
@JSON_FLAG
def models_command(as_json):
    want_json(as_json)
    report = get_service().models()
    if out().machine:
        return out().dump(report)
    rows, kinds = report["models"], CATALOG["hardware_order"]
    Table(
        ["modell", "backend", "variante", "groesse", "installiert", *[Catalog.label(k) for k in kinds]],
        [[r["model"], r["backend"], r["variant"], f"{r['size_mb']} MB", "ja" if r["installed"] else "nein", *[STATUS_WORDS[r["hardware"][k]["status"]] for k in kinds]] for r in rows],
    ).echo()
    click.echo("ok = geeignet, ungeprüft = nicht gemessen, nein = geht nicht (Grund: meetscribe models --json). Laden: meetscribe setup --model NAME [--backend B].")


@main.command(name="check", help="Voraussetzungen pruefen (ffmpeg, Modelle, HF_TOKEN).")
def check_setup():
    rows = get_service().check()
    if out().machine:
        return out().dump(rows)
    Table(["pruefung", "ok", "loesung"], [[r["label"], "ja" if r["ok"] else "NEIN", r["fix"]] for r in rows]).echo()
    if not all(r["ok"] for r in rows):
        raise click.exceptions.Exit(1)


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
@selection_options
@click.option("--language", default="de", show_default=True)
@click.option("--title", default=None)
@click.option("--date", "day", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, help="Datum JJJJ-MM-TT.")
@click.option("--meeting-id", default=None)
@click.option("--no-diarize", is_flag=True, help="Keine Sprechertrennung (Sprecher UNKNOWN).")
@click.option("--no-fallback", is_flag=True, help="Bei Geraetefehler abbrechen statt weiterzumachen.")
@click.option("--speakers", type=int, default=None)
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
def transcribe(audio, backend, device, model, language, title, day, meeting_id, no_diarize, no_fallback, speakers, min_speakers, max_speakers):
    from meetscribe import pipeline
    from meetscribe.store import DuplicateMeetingError
    job = pipeline.TranscribeJob(
        get_store(), audio, device, language, model=model, backend=backend, title=title, day=day.date() if day else None,
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
        f"Geraet angefordert: {result['requested']}, genutzt: {' -> '.join(result['used']) or '-'} ({result['ran']['text']})"
    )


@main.command(help="Aufnehmen (oder AUDIOFILE nehmen), transkribieren, Sprecher benennen, Protokoll schreiben.")
@click.argument("audio", required=False, type=click.Path(path_type=Path))
@click.option("--title", default=None)
@click.option("--mic", default=None, help="dji, laptop, both oder Namensteil; Standard: DJI, sonst Laptop.")
@selection_options
@click.option("--no-fallback", is_flag=True, help="Bei Geraetefehler abbrechen statt auf die naechste Hardware auszuweichen.")
@click.option("--speakers", type=int, default=None)
@click.option("--no-diarize", is_flag=True, help="Keine Sprechertrennung (Sprecher UNKNOWN).")
@click.option("--no-open", is_flag=True, help="Protokoll nicht oeffnen.")
@click.option("--events", is_flag=True, help="Ereignisse als JSON-Zeilen auf stdout, ohne Rueckfragen und ohne Oeffnen des Protokolls.")
@click.option("--stdin-stop", is_flag=True, help="Aufnahme beenden, sobald eine Zeile auf stdin ankommt oder stdin geschlossen wird.")
def start(audio, title, mic, backend, device, model, no_fallback, speakers, no_diarize, no_open, events, stdin_stop):
    from meetscribe import pipeline
    reporter = pipeline.EventReporter() if events else None
    flow = get_service().flow(
        audio=audio, mic=mic, device=device, backend=backend, model=model, fallback=not no_fallback, title=title, speakers=speakers,
        diarize=not no_diarize, open_report=not no_open, tty=stdin_is_tty(), reporter=reporter, stdin_stop=stdin_stop,
    )
    with contextlib.redirect_stdout(sys.stderr) if events else contextlib.nullcontext(), pipeline.KeepAwake():
        flow.run()


@main.command(help="Heisse Dateien ins Parquet-Archiv verschieben.")
@click.option("--older-than-days", type=float, default=0, show_default=True)
@click.option("--force", is_flag=True, help="Auch unvollstaendige Dateien rotieren.")
def rotate(older_than_days, force):
    result = get_service().rotate(older_than_days=older_than_days, force=force)
    click.echo(f"rotiert: {', '.join(result['rotated']) or '-'}")
    for skipped in result["skipped"]:
        click.echo(f"uebersprungen: {skipped['meeting_id']} ({skipped['reason']})")


@main.command(help="Unvollstaendige Besprechung verwerfen (Rest einer abgebrochenen Verarbeitung).")
@click.argument("meeting_id")
def discard(meeting_id):
    click.echo(f"{get_service().discard(meeting_id)['meeting_id']} verworfen")


@main.command(help="Teilstring-Suche (ohne Gross-/Kleinschreibung) in den Segmenten.")
@click.argument("term")
@click.option("--meeting", default=None)
@click.option("--speaker", default=None)
@click.option("--limit", type=int, default=20, show_default=True)
def search(term, meeting, speaker, limit):
    rows = get_service().search(term, meeting=meeting, speaker=speaker, limit=limit)
    out().records(rows, ["meeting_id", "seg_id", "start_s", "display_name", "text"])


@main.command(help="Alle Besprechungen auflisten.")
def meetings():
    out().records(get_service().meetings(), ["meeting_id", "title", "meeting_date", "segments", "speakers", "duration_s", "device", "complete"], formats={"complete": lambda v: "ja" if v else "NEIN"})


@main.command(help="Redezeit und Woerter je Sprecher.")
@click.option("--meeting", default=None)
def stats(meeting):
    rows = get_service().stats(meeting)
    out().records(rows, ["meeting_id", "speaker", "labels", "talk_s", "share", "segments", "words"], formats={"share": lambda v: f"{v * 100:.1f} %"})


@main.group(name="speakers", help="Sprecherzuordnung anzeigen und aendern.")
def speakers_group():
    pass


MEETING_OPTION = click.option("--meeting", "meeting_id", default=None, help="Besprechung; Standard: die zuletzt erstellte.")


@speakers_group.command(name="show")
@MEETING_OPTION
def speakers_show(meeting_id):
    out().records(get_service().speakers_show(meeting_id))


@speakers_group.command(name="set")
@click.argument("label")
@click.argument("name")
@MEETING_OPTION
def speakers_set(label, name, meeting_id):
    done = get_service().speakers_set(label, name, meeting_id)
    click.echo(f"{done['meeting_id']} {label} = {done['name']}")


@speakers_group.command(name="unset")
@click.argument("label")
@MEETING_OPTION
def speakers_unset(label, meeting_id):
    click.echo(f"{get_service().speakers_unset(label, meeting_id)['meeting_id']} {label} entfernt")


@main.command(help="Markdown-Protokoll erzeugen (Standard: letzte Besprechung).")
@click.argument("meeting_id", required=False)
@click.option("--out", "target", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Zieldatei statt Ausgabe.")
def report(meeting_id, target):
    done = get_service().report(meeting_id, target)
    if out().machine:
        return out().dump(done)
    if target:
        click.echo(f"geschrieben: {target}")
    else:
        click.echo(done["markdown"], nl=False)


@main.command(name="sql", help="SQL-Abfrage auf segments, segments_named und speakers.")
@click.argument("query")
def sql(query):
    result = get_service().sql(query)
    out().table(result["columns"], result["rows"], clip=None)


@main.command(help="Langlebige Engine für die App: JSON-RPC 2.0 über stdin/stdout, eine Nachricht je Zeile.")
def serve():
    RpcServer(get_service_with_cache(), Wire.stdio()).serve()
    sys.stderr.flush()
    os._exit(0)
