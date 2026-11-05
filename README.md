# meetscribe

## Zweck

meetscribe nimmt Besprechungen auf (oder nimmt eine vorhandene Audiodatei), transkribiert sie lokal auf Deutsch, trennt Sprecher, legt das Ergebnis in einem durchsuchbaren Archiv ab und schreibt ein Markdown-Protokoll. Alles läuft auf dem eigenen Rechner. Die Spracherkennung (Whisper large-v3-turbo über OpenVINO) läuft bevorzugt auf der NPU eines Intel Core Ultra, mit Rückfall auf die integrierte GPU und die CPU.

Das Protokoll enthält Kopfdaten, eine Teilnehmertabelle (Redezeit, Anteil, Wörter) und das Transkript nach Sprechern. Es enthält keine Zusammenfassung und keine Aufgabenliste (siehe Hinweise).

## Schnellstart

```
meetscribe start
```

Das prüft zuerst alles Nötige, nimmt auf (Ende mit Strg+C), transkribiert, fragt nach Sprechernamen, schreibt das Protokoll nach `protokolle\` und öffnet es. Voraussetzung ist die einmalige Installation unten.

Mit vorhandener Datei, etwa einer Handyaufnahme:

```
meetscribe start besprechung.m4a --title "Projektrunde"
```

## Voraussetzungen

- Windows 11 (64 Bit). Entwickelt und gemessen nur auf Windows 11 mit Core Ultra 7 258V, Arc 140V, 32 GB RAM.
- Intel Core Ultra mit NPU. Ohne NPU läuft es über den Fallback auf GPU oder CPU.
- [uv](https://docs.astral.sh/uv/) (holt Python 3.12 selbst).
- ffmpeg im PATH (`ffmpeg -version` muss funktionieren), z. B. `winget install Gyan.FFmpeg`. Wird zum Dekodieren jeder Audiodatei gebraucht.
- Intel-NPU-Treiber und Intel-Arc-Grafiktreiber. Beides wird nicht automatisch installiert.
  - NPU-Treiber: https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html (aktuell 32.0.100.5540; gemessen wurde mit 32.0.100.4240).
  - Arc-Grafiktreiber: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html (gemessen mit 32.0.101.8132). Der generische Intel-Treiber überschreibt den Treiber des Geräteherstellers; im Zweifel dessen Seite prüfen.
- Für die Sprechertrennung (nur einmalig bei `setup`): ein Hugging-Face-Konto, auf https://huggingface.co/pyannote/speaker-diarization-community-1 die Nutzungsbedingungen akzeptieren, und ein Zugriffstoken (Lesezugriff) in der Umgebungsvariable `HF_TOKEN`. Ohne das geht nur `--no-diarize`.

## Installation

Im Projektordner:

```
uv tool install --editable .
meetscribe setup
```

Die Installation muss editierbar sein: Modelle (`models\`), Daten (`data\`) und Protokolle (`protokolle\`) liegen relativ zum Projektordner, nicht relativ zum Aufrufort. Ist `meetscribe` danach nicht im PATH, hilft `uv tool update-shell` und ein neues Terminal.

Token für den Setup setzen, z. B. in PowerShell:

```
$env:HF_TOKEN = "hf_..."
meetscribe setup
```

`setup` ist der einzige Befehl mit Netzzugriff. Er lädt:

- `OpenVINO/whisper-large-v3-turbo-fp16-ov` (NPU) und `OpenVINO/whisper-large-v3-turbo-int8-ov` (GPU/CPU) nach `models\`,
- bei gesetztem `HF_TOKEN` das Diarisierungsmodell `pyannote/speaker-diarization-community-1`,
- und wärmt danach den NPU-Compile-Cache (`models\ov_cache`) auf. Das dauert einmalig mehrere Minuten (beim ersten Lauf ohne Cache gemessen: 265 s); danach lädt das Modell in etwa 3 bis 4 s.

Speicherbedarf, gemessen im Projektordner: virtuelle Umgebung 1,4 GB, Whisper fp16 1,6 GB, Whisper int8 0,8 GB, Compile-Cache 3,8 GB. Die Größe des Diarisierungsmodells wurde nicht gemessen. Dazu kommen Aufnahmen: Audio wird nie gelöscht.

Schlägt der Whisper-Download fehl, endet `setup` mit Fehler. Fehlt nur das Diarisierungsmodell, gibt `setup` eine Meldung aus und endet ohne Fehler.

## Nutzung

### start

`meetscribe start [AUDIO]` ist der normale Weg.

Ablauf:

1. Vorprüfung vor jeder Aufnahme: ffmpeg, Whisper-Modell, Diarisierungsmodell (außer mit `--no-diarize`) und, wenn aufgenommen wird, der Audioeingang. Schlägt etwas fehl, bricht `start` mit einer deutschen Meldung ab und nichts wird aufgenommen.
2. Aufnahme bis Strg+C. Der Eingang wird automatisch gewählt (DJI, sonst Laptopmikrofon) und als `Eingang: <Name> [API]` ausgegeben. Strg+C beendet nur die Aufnahme; die Verarbeitung läuft danach normal weiter.
3. Transkription und Sprechertrennung. Wird keine Sprache erkannt (stummes Mikrofon, leere Datei), gibt es kein Protokoll und keine Besprechung im Archiv; `start` meldet das, nennt das erhaltene Audio und endet mit Exit-Code 1.
4. Benennung der Sprecher, nur wenn die Eingabe ein Terminal ist: je Sprecherlabel wird der längste Satz gezeigt, Enter überspringt. Ohne Terminal entfällt das still. `UNKNOWN` wird nicht abgefragt.
5. Protokoll nach `protokolle\<Datum>_<Titel>.md` (bei gleichem Namen `-2`, `-3`, ...; UTF-8). Es wird mit dem Standardprogramm geöffnet, wenn die Eingabe ein Terminal ist und `--no-open` fehlt.
6. Zuletzt wird archiviert: Besprechungen, die älter als 30 Tage sind, wandern ins Parquet-Archiv. Schlägt das fehl, gibt es eine Warnzeile, der Lauf gilt trotzdem als erfolgreich.

Die Aufnahme liegt unter `data\aufnahmen\<JJJJ-MM-TT_HHMMSS>.flac` (bei Kollision `-2`) und wird nie gelöscht. Schlägt die Verarbeitung fehl oder wird sie mit Strg+C abgebrochen, gibt `start` `Audio erhalten: <Pfad>` und eine fertige Zeile `Erneut versuchen: meetscribe start <Pfad> ...` aus und endet mit Exit-Code 1 (bei Strg+C 130). Eine dabei halb geschriebene Besprechung wird wieder entfernt (auch bei `transcribe`), der erneute Versuch bekommt dieselbe ID.

Optionen: `--title`, `--mic dji|laptop|both|<Namensteil>`, `--device npu|gpu|cpu`, `--speakers N` (Sprecherzahl vorgeben), `--no-diarize`, `--no-open`. Titel ohne Angabe: Dateiname der Aufnahme bzw. der Datei. Datum der Besprechung ist immer das heutige Datum. Die Sprache ist bei `start` fest Deutsch.

### Vorhandene Datei

`meetscribe start datei.m4a` überspringt Aufnahme und Eingangsprüfung. Alles, was ffmpeg lesen kann, funktioniert. Die Datei wird nicht verändert oder kopiert.

### Nachschlagen

```
meetscribe meetings
meetscribe search Budget
meetscribe search Budget --speaker Anna --limit 50
meetscribe report
meetscribe report 20261105-projektrunde --out C:\temp\protokoll.md
```

`search` ist eine Teilstring-Suche ohne Beachtung der Groß-/Kleinschreibung (kein Index, keine Wortstämme) über alle Segmente; `--meeting` und `--speaker` (Label oder Name) schränken ein. `report` ohne ID nimmt die zuletzt erstellte Besprechung und schreibt auf die Konsole oder mit `--out` in eine Datei.

### Weitere Befehle

| Befehl | Zweck |
|---|---|
| `meetscribe devices` | Audioeingänge (Index, Name, API, Kanäle, Rate) und OpenVINO-Geräte auflisten; prüft z. B. ob der DJI-Empfänger sichtbar ist |
| `meetscribe record --out datei.flac [--mic ...] [--seconds N]` | nur aufnehmen, Standard-Mikrofon `laptop` |
| `meetscribe transcribe datei [...]` | nur transkribieren und ablegen, ohne Protokoll und ohne Sprecherabfrage |
| `meetscribe rotate [--older-than-days N] [--force]` | heiße Dateien ins Archiv verschieben (Standard 0 Tage: alle vollständigen) |
| `meetscribe stats [--meeting ID]` | Redezeit, Anteil, Segmente und Wörter je Sprecher |
| `meetscribe speakers show\|set\|unset` | Sprecherzuordnung ansehen und ändern |
| `meetscribe sql "..."` | beliebige SQL-Abfrage |

`transcribe` kennt zusätzlich `--model`, `--language`, `--date`, `--meeting-id`, `--min-speakers`, `--max-speakers` und `--no-fallback`; `start` kennt diese nicht. `rotate` wird von `transcribe` nicht automatisch aufgerufen, nur von `start`.

### Gerätewahl und Fallback

`--device npu|gpu|cpu` (Standard `npu`, auch über die Umgebungsvariable `MEETSCRIBE_DEVICE`) wählt den Startpunkt. Bei einem Fehler fällt die Transkription in der Reihenfolge NPU, GPU, CPU zurück; `--device gpu` fällt also nur auf die CPU zurück. Jeder Rückfall wird auf stderr gemeldet. Mit `--no-fallback` (nur `transcribe`) bricht der Lauf stattdessen ab. Welches Gerät tatsächlich lief, steht in der Ausgabe und je Segment in der Spalte `device`.

Es gibt keine dedizierte GPU; `gpu` bedeutet die integrierte Arc-Grafik. NPU nutzt das fp16-Modell, GPU und CPU das int8-Modell.

### Sprecher benennen

```
meetscribe speakers show
meetscribe speakers set SPEAKER_00 Anna
meetscribe speakers unset SPEAKER_00
meetscribe report
```

Ohne `--meeting ID` gilt die zuletzt erstellte Besprechung. Die Zuordnung steht in einer eigenen Datei; die Rohdaten bleiben unverändert.

### sql

Verfügbare Sichten: `segments` (Rohdaten, heiß und kalt vereint), `segments_named` (zusätzlich `display_name`: zugeordneter Name, sonst das Label) und `speakers` (Zuordnungstabelle).

```
meetscribe sql "SELECT display_name, count(*) AS segmente, round(sum(end_s - start_s)) AS sekunden FROM segments_named WHERE meeting_date >= DATE '2026-11-01' GROUP BY display_name ORDER BY sekunden DESC"
```

Schema von `segments`:

| Spalte | Typ |
|---|---|
| meeting_id | VARCHAR (Format `JJJJMMTT-titel`) |
| title | VARCHAR |
| meeting_date | DATE |
| seg_id | INTEGER |
| start_s, end_s | DOUBLE (Sekunden ab Aufnahmebeginn) |
| speaker | VARCHAR (`SPEAKER_00`, ... oder `UNKNOWN`) |
| text | VARCHAR |
| language, model, device | VARCHAR |
| words | Liste von (text, start_s, end_s), Zeitstempel je Wort |
| created_at | TIMESTAMP WITH TIME ZONE (UTC) |

`speakers`: `meeting_id`, `label`, `name`.

## Mikrofone

- DJI Mic Mini 2: der Empfänger wird per USB-C angeschlossen und erscheint als Audioeingang. `--mic dji` und die automatische Wahl suchen im Gerätenamen nach `dji`, `mic mini`, `mic 2` oder `wireless mic`. Heißt der Empfänger anders, mit `meetscribe devices` den Namen nachsehen und `--mic <Namensteil>` verwenden.
- Laptopmikrofon: gefunden über Namen wie `Intel Smart Sound`, `Mikrofonarray` oder `Microphone Array`; sonst über `--mic <Namensteil>`.
- Gibt es mehrere Einträge für dasselbe Gerät, wird die API in der Reihenfolge WASAPI, DirectSound, MME, WDM-KS bevorzugt.
- `--mic both` nimmt DJI und Laptop gleichzeitig in getrennte Dateien (`<Name>.dji.flac`, `<Name>.laptop.flac`) und erzeugt `<Name>.mix.flac`. Der Mix bildet je Datei den Kanalmittelwert und streckt beide Aufnahmen linear auf die Länge der längeren, um einen Taktunterschied der beiden Geräte auszugleichen; dann werden sie gemittelt. Das setzt voraus, dass beide gleichzeitig starten. Transkribiert wird der Mix.
- Handyaufnahme: Datei übertragen und `meetscribe start datei.m4a`.
- Die Aufnahme behält alle Kanäle des Eingangs; die Transkription arbeitet auf einem Mono-Mix (ffmpeg, 16 kHz). Sender TX1 und TX2 des DJI-Systems kommen möglicherweise als linker und rechter Kanal an; das ist nicht geprüft. Die Kanäle sind in der FLAC-Datei erhalten, werden aber nicht getrennt ausgewertet.
- Deckel offen lassen. Während Aufnahme, Transkription und Setup-Warmup hält meetscribe den Rechner per `SetThreadExecutionState` wach. Das verhindert nur den Ruhezustand durch Inaktivität, nicht das Zuklappen des Deckels, den Netzschalter oder einen kritischen Akkustand. `record` und `start` weisen darauf hin.

## Geräte und Modelle

Spracherkennung läuft über `openvino_genai.WhisperPipeline`. Gründe, die sich aus dem Code ergeben: eine API für NPU, GPU und CPU (nur der Gerätename ändert sich), Wortzeitstempel direkt aus der Pipeline, ein Compile-Cache über `CACHE_DIR`, und fertig konvertierte Modelle aus der Hugging-Face-Organisation `OpenVINO`. Vergleichsmessungen mit whisper.cpp oder optimum-intel wurden nicht gemacht. PyTorch wird nur für die Sprechertrennung (pyannote, immer auf der CPU) gebraucht.

Ablauf bei langem Audio: Das Audio wird in Fenster von etwa 5 min geteilt, geschnitten an der leisesten Stelle in ±5 s um die Grenze. Fenster ohne Signal werden übersprungen. Wörter werden dem Sprecher mit der größten zeitlichen Überlappung zugeordnet und zu Segmenten zusammengefasst (Pause bis 1,5 s; ein Segment wird nach einem Satzende beendet, sobald es mindestens 8 s lang ist).

Messungen (Core Ultra 7 258V, Arc 140V, NPU-Treiber 32.0.100.4240, 32 GB RAM). RTF = Rechenzeit der Erkennung geteilt durch Audiodauer, kleiner ist schneller. WER gegen das bekannte Skript.

| Gerät, Modell | Audio | RTF | WER | Spitzen-RAM |
|---|---|---|---|---|
| NPU, fp16, erster Lauf (Compile 265 s) | 80 s | 0,090 | 1,80 % (3 von 167 Wörtern) | 4,5 GB |
| NPU, fp16, Cache vorhanden (Compile 3,9 s) | 80 s | 0,089 | 1,80 % | 4,9 GB |
| NPU, fp16, Cache vorhanden | 12,0 min (3 Fenster) | 0,096 | 1,93 % (29 von 1503 Wörtern) | 4,9 GB |
| iGPU, int8 | kurze Datei | 0,066 | 1,8 % | nicht erfasst |
| iGPU, int8 | 12 min | 0,043 | 1,9 % | 1,9 GB |
| CPU, int8 | nicht getrennt festgehalten | 0,19 bis 0,45 | 1,8 % | 2,1 GB |

Die NPU-Zahlen messen die Erkennung ohne Modellladen (Spitzen-RAM ist der Arbeitssatz des Prozesses). Bei den iGPU- und CPU-Zeilen ist nicht festgehalten, ob Ladezeit enthalten war. Bei der Geschwindigkeit liegt die iGPU in diesen Messungen vor der NPU. In allen NPU-Läufen liefen alle Segmente auf der NPU, ohne Fallback und ohne Gerätefehler. Die Fehler der WER sind Zahlenformat-Unterschiede (Ziffern gegen ausgeschriebene Zahlen).

Das Testaudio ist synthetische Windows-TTS-Sprache unter sauberen Bedingungen. Die WER bei echten Besprechungen mit Raumhall, Nebengeräuschen und Dialekt wird schlechter sein; wie viel, wurde nicht gemessen.

NPU-Besonderheiten:

- Auf der NPU wird das fp16-Modell verwendet. Das int8-Modell stürzt auf der NPU ab.
- Das int4-Modell von FluidInference lässt sich nicht laden. Das Verzeichnis `models\whisper-large-v3-turbo-int4-ov-npu` ist ein Überrest dieses Versuchs und wird weder von `setup` geladen noch vom Programm benutzt.
- Ob ein neuerer NPU-Treiber (Intel listet derzeit 32.0.100.5540) int8 auf der NPU ermöglicht, ist nicht geprüft. Für fp16 ist er nicht nötig.

## Datenhaltung

Standardmäßig unter dem Projektordner; `MEETSCRIBE_DATA`, `MEETSCRIBE_MODELS` und `MEETSCRIBE_REPORTS` verlegen Daten, Modelle bzw. Protokolle.

| Pfad | Inhalt |
|---|---|
| `data\hot\<id>.arrow` | frische Besprechungen, Arrow IPC, wird beim Transkribieren segmentweise geschrieben |
| `data\cold\year=JJJJ\month=MM\<id>.parquet` | archivierte Besprechungen, Parquet |
| `data\archive\<id>.arrow` | Original der heißen Datei nach dem Archivieren (nicht gelöscht) |
| `data\speakers.parquet` | Sprecherzuordnung |
| `data\aufnahmen\` | Aufnahmen von `start` |
| `protokolle\` | Markdown-Protokolle |
| `models\` | Whisper-Modelle, Diarisierungsmodell, `ov_cache\` |

DuckDB liest heiße und kalte Dateien als eine Sicht (`segments`) und verbindet die Sprecherzuordnung (`segments_named`). Namen ändern nur `speakers.parquet`, nie die Rohdaten. Eine unvollständige heiße Datei (nur noch bei hartem Prozessabbruch, etwa Absturz oder Stromausfall; bei Fehlern und Strg+C wird sie entfernt) wird bei `rotate` ohne `--force` übersprungen. `start` rotiert zuletzt automatisch alles, was vollständig und älter als 30 Tage ist (Alter nach Änderungsdatum der Datei).

Audio wird nicht automatisch gelöscht, auch nicht nach der Archivierung. `data\`, `models\`, `protokolle\` und gängige Audioendungen stehen in `.gitignore`.

## Hinweise

- Offline: Nur `setup` greift auf das Netz zu. Alle anderen Befehle setzen die Hugging-Face-Offline-Variablen und senden nichts. Es gibt keine Cloud-Dienste.
- Aufnahme und Einwilligung: In Deutschland ist das heimliche Aufnehmen des nichtöffentlich gesprochenen Wortes strafbar (§ 201 StGB). Vor der Aufnahme sollten alle Beteiligten zustimmen. Das ist keine Rechtsberatung.
- Personenbezogene Daten liegen unverschlüsselt auf der Platte: Aufnahmen, Transkripte, Protokolle, Sprecherzuordnung. Wer das braucht, muss das Laufwerk verschlüsseln.
- Kein Sprachmodell im Einsatz: Das Protokoll enthält Transkript und Redezeiten, aber keine Zusammenfassung, keine Beschlüsse und keine Aufgaben.
- Sprechertrennung: Die Label (`SPEAKER_00`, ...) gelten nur innerhalb einer Besprechung; es gibt keine Wiedererkennung über Besprechungen hinweg. Bei ähnlichen Stimmen oder wechselnder Entfernung zum Mikrofon werden Sprecher verwechselt oder aufgespalten; `--speakers N` hilft, wenn die Zahl bekannt ist. Die Trennung läuft auf der CPU; ihre Dauer wurde nicht gemessen.
- Überlappende Rede: Ein Wort wird genau einem Sprecher zugeordnet, und Whisper erkennt einen einzigen Sprachstrom. Wo mehrere gleichzeitig sprechen, geht Text verloren oder landet beim falschen Sprecher.
- Die Transkription ist nicht fehlerfrei; Eigennamen, Fachbegriffe und Zahlen vor Verwendung prüfen.

## Bekannte offene Punkte

Nicht an echter Hardware oder echten Eingaben geprüft:

- Sprechertrennung mit dem echten pyannote-Modell, auch ohne Token oder lokalen Snapshot: Alle echten Läufe liefen mit `--no-diarize`. Die Diarisierung ist nur in Tests (ohne echtes Modell) abgedeckt.
- Der echte DJI-Empfänger: Die Namensmuster für den USB-Eingang (`dji`, `mic mini`, `mic 2`, `wireless mic`) sind Annahmen. Ebenso unbekannt: ob TX1 und TX2 als linker und rechter Kanal ankommen.
- `--mic both` auf echter Hardware, einschließlich des streckenden Mixes.
- Aufnahme mit dem echten Laptopmikrofon und die Namensmuster dafür.
- Die interaktive Sprecherabfrage in einer echten Konsole (nur mit simuliertem Terminal getestet) und das Öffnen des Protokolls mit `os.startfile` (nur über einen Ersatz aufgerufen).
- `setup` gegen das Netz.
- Keep-awake: Aufruf und Freigabe sind getestet, dass es Modern Standby tatsächlich verhindert, wurde nicht beobachtet.
- Der NPU-Compile-Cache: Der erste NPU-Lauf brauchte 265 s, obwohl `models\ov_cache` schon existierte. Wie oft der Cache nach `setup` trifft, ist unklar. Er ist inzwischen 3,8 GB groß. Ob ein Treiberwechsel ihn ungültig macht, wurde nicht geprüft.
- Die WER stammt aus synthetischem Testaudio (siehe oben); echte Besprechungen sind nicht vermessen.
- Messungen nur auf einem Gerät (Core Ultra 7 258V). Andere Core-Ultra-Generationen sind nicht geprüft.
- Strg+C während der Aufnahme wurde mit einem simulierten Rekorder und einem echten Konsolen-Ereignis geprüft, nicht mit echter Aufnahme.

Tests: `uv run pytest -q` (97 Tests, alle grün; sie prüfen Logik und Abläufe mit Ersatzobjekten, nicht die oben genannten Hardwarefälle).
