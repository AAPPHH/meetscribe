import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { afterEach, test } from "node:test";
import { JSDOM } from "jsdom";

const HTML = readFileSync(new URL("../src/index.html", import.meta.url), "utf8").replace(/<script[\s\S]*?<\/script>/, "");
const BUNDLE = new URL("./.build/main.js", import.meta.url).href;
const SETTINGS = { engine: "", repo: "", device: "auto", backend: "auto", model: "auto", mic: "", diarize: true };
const AUDIO = "C:\\Daten\\aufnahmen\\2026-11-06_101500.flac";
let runs = 0;
let dom = null;

const flush = () => new Promise((resolve) => setTimeout(resolve, 30));

async function boot(handlers = {}, settings = {}) {
  dom = new JSDOM(HTML, { url: "http://localhost/", pretendToBeVisual: true });
  dom.window.HTMLElement.prototype.scrollIntoView = () => {};
  Object.assign(globalThis, { window: dom.window, document: dom.window.document });
  const harness = {
    calls: [],
    listeners: {},
    handlers: {
      get_settings: () => ({ ...SETTINGS, ...settings }),
      platform_info: () => ({ mobile: false, notice: null }),
      session_state: () => ({ active: false, stage: "finished" }),
      pending_run: () => null,
      run_checks: () => [],
      list_meetings: () => [],
      ...handlers,
    },
  };
  globalThis.__tauri = harness;
  await import(`${BUNDLE}?run=${runs++}`);
  await flush();
  const $ = (id) => document.getElementById(id);
  const emit = async (name, payload) => {
    (harness.listeners[name] ?? []).forEach((callback) => callback({ payload }));
    await flush();
  };
  const click = async (id) => {
    $(id).click();
    await flush();
  };
  return { $, emit, click, harness, called: (cmd) => harness.calls.filter((c) => c.cmd === cmd) };
}

afterEach(() => {
  dom?.window.close();
  dom = null;
});

const meeting = (id, date, extra = {}) => ({ meeting_id: id, title: id, meeting_date: date, segments: 1, speakers: 1, duration_s: 60, ...extra });

test("Besprechungen erscheinen in der Reihenfolge der Engine, neueste zuerst", async () => {
  const ui = await boot({ list_meetings: () => [meeting("neu", "2026-05-01"), meeting("mitte", "2026-03-04"), meeting("alt", "2026-01-01")] });
  document.querySelector('[data-tab="meetings"]').click();
  await flush();
  const dates = [...document.querySelectorAll("#meetings-table tbody tr td:first-child")].map((td) => td.textContent);
  assert.deepEqual(dates, ["2026-05-01", "2026-03-04", "2026-01-01"]);
});

test("unvollständige Besprechungen sind markiert", async () => {
  await boot({ list_meetings: () => [meeting("ganz", "2026-05-01", { complete: true }), meeting("teil", "2026-05-02", { complete: false })] });
  document.querySelector('[data-tab="meetings"]').click();
  await flush();
  const tags = [...document.querySelectorAll("#meetings-table .tag")].map((t) => t.textContent);
  assert.deepEqual(tags, ["unvollständig"]);
});

test("Protokolle zeigen die Teilnehmertabelle als Tabelle statt als Rohtext", async () => {
  const markdown = "# Titel\n\n## Teilnehmer\n\n| Name | Redezeit |\n|---|---|\n| Anna \\| B | 01:00 |\n| UNKNOWN | 02:00 |\n\n## Transkript\n\n**[00:00] Anna:** Hallo\n";
  const ui = await boot({ list_meetings: () => [meeting("m", "2026-05-01")], get_report: () => ({ markdown }) });
  document.querySelector('[data-tab="meetings"]').click();
  await flush();
  document.querySelector("#meetings-table tbody tr").click();
  await flush();
  const report = ui.$("report-view");
  const rows = [...report.querySelectorAll("table tr")].map((tr) => [...tr.children].map((c) => c.textContent));
  assert.deepEqual(rows, [["Name", "Redezeit"], ["Anna | B", "01:00"], ["UNKNOWN", "02:00"]]);
  assert.equal(report.querySelectorAll("th").length, 2);
  assert.ok(!report.textContent.includes("|---|") && !report.textContent.includes("| Name"));
  assert.ok(report.querySelector("p strong"));
});

async function startRun(ui) {
  await ui.click("start");
  assert.equal(ui.called("start_session").length, 1);
}

test("ein Absturz der Engine behält den Pfad zum Audio und bietet Erneut versuchen an", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  await ui.emit("engine-event", { event: "transcribing", audio: AUDIO, meeting_id: "m-1" });
  await ui.emit("engine-exit", { code: -1073741819, terminal: false, stderr: "Ursache", audio: AUDIO, log: "C:\\cfg\\engine.log" });
  assert.ok(ui.$("error-audio").textContent.includes(AUDIO));
  assert.equal(ui.$("retry").hidden, false);
  assert.ok(ui.$("error-detail").textContent.includes("C:\\cfg\\engine.log") && ui.$("error-detail").textContent.includes("Ursache"));
  await ui.click("retry");
  assert.equal(ui.called("start_session").at(-1).args.options.audio, AUDIO);
});

test("ein Absturz während der Aufnahme nennt die Aufnahmedatei aus dem Ereignis", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  await ui.emit("engine-event", { event: "recording", input: "DJI", path: AUDIO });
  await ui.emit("engine-exit", { code: -1, terminal: false, stderr: "", audio: null, log: null });
  assert.ok(ui.$("error-audio").textContent.includes(AUDIO));
  assert.equal(ui.$("retry").hidden, false);
});

test("ein Neustart der Engine nach einem Absturz erscheint als Hinweis, der sich schließen lässt", async () => {
  const ui = await boot();
  assert.equal(ui.$("engine-notice").hidden, true);
  await ui.emit("engine-notice", { text: "Die Engine war abgestürzt und wurde neu gestartet." });
  assert.equal(ui.$("engine-notice").hidden, false);
  assert.equal(ui.$("engine-notice-text").textContent, "Die Engine war abgestürzt und wurde neu gestartet.");
  await ui.click("engine-notice-close");
  assert.equal(ui.$("engine-notice").hidden, true);
});

test("Fehlerereignisse zeigen die Einzelheiten ungekürzt", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  await ui.emit("engine-event", { event: "error", message: "Verarbeitung fehlgeschlagen", detail: "Ursache: Datei kaputt\nzweite Zeile", audio: AUDIO, meeting_id: null });
  assert.equal(ui.$("error-more").hidden, false);
  assert.equal(ui.$("error-detail").textContent, "Ursache: Datei kaputt\nzweite Zeile");
});

test("nach einem Neuladen mitten in der Aufnahme kehrt die Ansicht mit Stopp-Knopf zurück", async () => {
  const ui = await boot({
    session_state: () => ({ active: true, stage: "recording", elapsed_s: 65, audio: AUDIO, input: "DJI MIC", warnings: ["leise"] }),
  });
  assert.equal(ui.$("view-running").hidden, false);
  assert.equal(ui.$("view-idle").hidden, true);
  assert.equal(ui.$("stop").hidden, false);
  assert.equal(ui.$("stop").disabled, false);
  assert.match(ui.$("elapsed").textContent, /^01:0[5-9]$/);
  assert.ok(ui.$("input-name").textContent.includes("DJI MIC"));
  assert.ok(ui.$("audio-path").textContent.includes(AUDIO));
  assert.equal(ui.$("warnings").children.length, 1);
  await ui.click("stop");
  assert.equal(ui.called("stop_session").length, 1);
});

test("nach einem Neuladen in der Verarbeitung gibt es keinen Stopp-Knopf, aber den Stand", async () => {
  const ui = await boot({ session_state: () => ({ active: true, stage: "processing", elapsed_s: 5, audio: AUDIO, warnings: [] }) });
  assert.equal(ui.$("view-running").hidden, false);
  assert.equal(ui.$("stop").hidden, true);
  assert.equal(ui.$("stage").textContent, "Transkription und Sprechertrennung");
  assert.ok(ui.$("audio-path").textContent.includes(AUDIO));
});

test("eine unfertige Verarbeitung wird beim Start angeboten und lässt sich wiederholen oder vergessen", async () => {
  const pending = { audio: AUDIO, meeting_id: "m-1", title: "Runde", device: "gpu", diarize: false };
  let current = pending;
  const ui = await boot({ pending_run: () => current, start_session: () => null, discard_pending: () => (current = null) });
  assert.equal(ui.$("pending").hidden, false);
  assert.ok(ui.$("pending-audio").textContent.includes(AUDIO));
  await ui.click("pending-retry");
  assert.deepEqual(ui.called("start_session")[0].args.options, { title: "Runde", audio: AUDIO, mic: null, device: "gpu", backend: "auto", model: "auto", diarize: false });
  await ui.emit("engine-event", { event: "transcribing", audio: AUDIO });
  assert.equal(ui.$("stage").textContent, "Transkription");
  await ui.emit("engine-event", { event: "done", meeting_id: "m-1", report: "C:\\r.md", segments: 2, speakers: 1, labels: [] });
  current = null;
  await ui.click("again");
  assert.equal(ui.$("pending").hidden, true);
});

test("Vergessen räumt die unfertige Verarbeitung weg", async () => {
  let current = { audio: AUDIO, meeting_id: null, title: null, device: "npu", diarize: true };
  const ui = await boot({ pending_run: () => current, discard_pending: () => (current = null) });
  await ui.click("pending-discard");
  assert.equal(ui.called("discard_pending").length, 1);
  assert.equal(ui.$("pending").hidden, true);
});

async function closeDialog(ui, stage, audio) {
  await ui.emit("close-requested", { stage, audio });
  return ui.$("modal-text").textContent;
}

test("der Schließen-Dialog sagt in der Vorbereitung nicht, das Audio sei gespeichert", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  const body = await closeDialog(ui, "preflight", null);
  assert.ok(body.includes("noch nichts aufgenommen") && !body.includes("Audio ist gespeichert") && !body.includes("Erneut versuchen"));
});

test("der Schließen-Dialog nennt in der Verarbeitung den Pfad und was beim nächsten Start geht", async () => {
  const ui = await boot({ start_session: () => null, exit_app: () => null });
  await startRun(ui);
  const body = await closeDialog(ui, "processing", AUDIO);
  assert.ok(body.includes(AUDIO) && body.includes("Erneut versuchen") && body.includes("verwirft"));
  assert.equal(ui.$("modal-confirm").textContent, "Abbrechen und schließen");
  await ui.click("modal-confirm");
  assert.equal(ui.called("exit_app").length, 1);
});

test("der Schließen-Dialog in der Aufnahme nennt den Aufnahmepfad", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  const body = await closeDialog(ui, "recording", AUDIO);
  assert.ok(body.includes(AUDIO) && body.startsWith("Das Fenster bleibt offen"));
});

test("endet die Sitzung bei offenem Dialog, verschwindet der Dialog", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  await ui.emit("engine-event", { event: "recording", input: "DJI", path: AUDIO });
  await closeDialog(ui, "recording", AUDIO);
  assert.equal(ui.$("modal").hidden, false);
  await ui.emit("engine-exit", { code: -1, terminal: false, stderr: "", audio: AUDIO, log: null });
  assert.equal(ui.$("modal").hidden, true);
  assert.equal(ui.called("stop_session").length, 0);
  assert.equal(ui.$("view-error").hidden, false);
});

test("Neue Aufnahme übernimmt den vorherigen Titel nicht still", async () => {
  const ui = await boot({ start_session: () => null });
  ui.$("title").value = "Projektrunde";
  await startRun(ui);
  await ui.emit("engine-event", { event: "done", meeting_id: "m", report: "C:\\r.md", segments: 1, speakers: 1, labels: [] });
  await ui.click("again");
  assert.equal(ui.$("title").value, "");
  assert.equal(ui.$("view-idle").hidden, false);
});

test("ohne Sprechertrennung heißt die Stufe Transkription und das Ergebnis nennt keine Sprecherzahl", async () => {
  const ui = await boot({ start_session: () => null }, { diarize: false });
  await startRun(ui);
  await ui.emit("engine-event", { event: "transcribing", audio: AUDIO });
  assert.equal(ui.$("stage").textContent, "Transkription");
  await ui.emit("engine-event", { event: "done", meeting_id: "m", report: "C:\\r.md", segments: 3, speakers: 1, labels: [] });
  assert.ok(ui.$("result-info").textContent.startsWith("3 Abschnitte. Protokoll"));
  assert.ok(!ui.$("result-info").textContent.includes("Sprecher"));
});

const ROWS = [
  { id: "ffmpeg", label: "ffmpeg", ok: true, fix: null, scope: null },
  { id: "pyannote", label: "pyannote", ok: false, fix: "setup", scope: "diarize" },
  { id: "hf_token", label: "HF_TOKEN", ok: false, fix: "token", scope: "diarize" },
];

test("fehlende Sprechertrennungs-Modelle zählen nur bei eingeschalteter Trennung als Mangel", async () => {
  const off = await boot({ run_checks: () => ROWS }, { diarize: false });
  assert.equal(off.$("setup-warning").hidden, true);
  assert.equal(off.$("check-badge").hidden, true);
  assert.ok([...document.querySelectorAll("#check-list li")].slice(1).every((li) => li.classList.contains("ok")));
  dom.window.close();
  const on = await boot({ run_checks: () => ROWS });
  assert.equal(on.$("setup-warning").hidden, false);
  dom.window.close();
  const broken = await boot({ run_checks: () => [{ id: "ffmpeg", label: "ffmpeg", ok: false, fix: "installieren", scope: null }] }, { diarize: false });
  assert.equal(broken.$("setup-warning").hidden, false);
});

test("auf Android sind Aufnahme und Dateiwahl gesperrt und der Grund steht da", async () => {
  const ui = await boot({ platform_info: () => ({ mobile: true, notice: "Auf diesem Gerät läuft keine Engine." }) });
  assert.equal(ui.$("start").disabled, true);
  assert.equal(ui.$("platform-notice").hidden, false);
  assert.equal(ui.$("platform-notice").textContent, "Auf diesem Gerät läuft keine Engine.");
});

test("der Audiopfad steht während der Aufnahme auf der Seite und bleibt, wenn der Dialog Weiter aufnehmen gewählt wird", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  assert.ok(!document.body.textContent.includes("aufnahme.flac"));
  await ui.emit("engine-event", { event: "recording", input: "DJI MIC", path: AUDIO });
  assert.ok(ui.$("view-running").textContent.includes(AUDIO));
  assert.ok(ui.$("input-name").textContent.includes("DJI MIC"));
  await closeDialog(ui, "recording", AUDIO);
  await ui.click("modal-keep");
  assert.ok(ui.$("view-running").textContent.includes(AUDIO));
  await ui.emit("engine-event", { event: "transcribing", audio: AUDIO, meeting_id: "m" });
  assert.ok(ui.$("view-running").textContent.includes(AUDIO));
});

test("Zurück nach einem Fehler lässt Titel und Audiopfad stehen, Neue Aufnahme leert sie", async () => {
  const ui = await boot({ start_session: () => null });
  ui.$("title").value = "Projektrunde";
  ui.$("audio").value = AUDIO;
  await startRun(ui);
  await ui.emit("engine-event", { event: "error", message: "Diarisierungsmodell fehlt", audio: null, meeting_id: null });
  await ui.click("error-back");
  assert.equal(ui.$("view-idle").hidden, false);
  assert.equal(ui.$("title").value, "Projektrunde");
  assert.equal(ui.$("audio").value, AUDIO);
});

test("Einzelheiten erscheinen nicht, wenn sie die Meldung nur wiederholen", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  await ui.emit("engine-event", { event: "error", message: "Es hat nicht geklappt.", detail: " Es hat nicht geklappt. ", audio: null, meeting_id: null });
  assert.equal(ui.$("error-more").hidden, true);
});

test("Vergessen zeigt den Grund und lässt die Anzeige stehen, wenn die alte Verarbeitung noch läuft", async () => {
  const pending = { audio: AUDIO, meeting_id: "m-1", title: null, device: "npu", diarize: true };
  const ui = await boot({ pending_run: () => pending, discard_pending: () => Promise.reject("Die frühere Verarbeitung läuft vermutlich noch.") });
  await ui.click("pending-discard");
  assert.equal(ui.$("pending").hidden, false);
  assert.ok(ui.$("pending-status").textContent.includes("läuft vermutlich noch"));
});

test("Suchtreffer aus unvollständigen Besprechungen sind markiert", async () => {
  const hit = (id, complete) => ({ meeting_id: id, start_s: 5, display_name: "Anna", text: "Hallo", complete });
  const ui = await boot({ search_text: () => [hit("ganz", true), hit("teil", false)] });
  document.querySelector('[data-tab="search"]').click();
  ui.$("search-term").value = "hallo";
  ui.$("search-form").dispatchEvent(new dom.window.Event("submit", { cancelable: true }));
  await flush();
  const tags = [...document.querySelectorAll("#search-results li")].map((li) => li.querySelector(".tag")?.textContent ?? null);
  assert.deepEqual(tags, [null, "unvollständig"]);
});

test("der Hinweis auf die schnelle Lösung erscheint nur, wenn die Sprechertrennung wirklich fehlt", async () => {
  const on = await boot({ run_checks: () => ROWS });
  assert.equal(on.$("setup-hint").hidden, false);
  assert.ok(on.$("setup-hint").textContent.includes("Sprecher trennen"));
  dom.window.close();
  const off = await boot({ run_checks: () => ROWS }, { diarize: false });
  assert.equal(off.$("setup-hint").hidden, true);
});

test("lange Pfade laufen im Dialog nicht über den Rand", () => {
  const css = readFileSync(new URL("../src/style.css", import.meta.url), "utf8");
  for (const rule of [".dialog", ".muted", ".notice"]) assert.match(css, new RegExp(`${rule.replace(".", "\.")} \{[^}]*overflow-wrap: anywhere`));
});

const HARDWARE = ["auto", "npu", "gpu", "igpu", "cpu"];
const BACKENDS = ["auto", "openvino", "faster-whisper"];
const MODEL_IDS = ["auto", "small", "large-v3-turbo"];
const SIZES = { small: 257, "large-v3-turbo": 1628 };

function modelsInfo(installed = ["large-v3-turbo"]) {
  const choices = [];
  for (const b of BACKENDS) {
    for (const h of HARDWARE) {
      for (const m of MODEL_IDS) {
        const backend = b === "auto" ? "openvino" : b;
        const model = m === "auto" ? "large-v3-turbo" : m;
        let reason = null;
        if (backend === "faster-whisper") reason = ["auto", "cpu"].includes(h) ? "faster-whisper ist nicht installiert. Installieren mit: uv sync --extra faster-whisper" : "faster-whisper läuft nur auf der CPU und auf NVIDIA-Karten (CUDA).";
        const hardware = h === "auto" ? "npu" : h === "gpu" ? "igpu" : h;
        const variant = hardware === "npu" ? "fp16" : "int8";
        const ok = !reason;
        choices.push({
          backend: b, hardware: h, model: m, ok, reason, size_mb: ok ? SIZES[model] : null, installed: ok && installed.includes(model), status: hardware === "npu" && model !== "large-v3-turbo" ? "unverified" : "ok",
          notices: h === "gpu" ? ["Hinweis: --device gpu läuft hier auf der iGPU."] : [],
          resolved: ok ? { model, backend, hardware, device: { npu: "NPU", igpu: "iGPU", cpu: "CPU" }[hardware], variant, installed: installed.includes(model), status: "ok", reason: null, requested: h, notices: [] } : null,
        });
      }
    }
  }
  return {
    default_model: "large-v3-turbo",
    backends: [{ id: "openvino", label: "OpenVINO", installed: true, reason: null }, { id: "faster-whisper", label: "faster-whisper", installed: false, reason: "faster-whisper ist nicht installiert. Installieren mit: uv sync --extra faster-whisper" }],
    hardware: [
      { id: "npu", label: "NPU", present: true, name: "Intel(R) AI Boost", reason: null },
      { id: "gpu", label: "GPU", present: false, name: null, reason: "Keine dedizierte Grafikkarte gefunden." },
      { id: "igpu", label: "iGPU", present: true, name: "Intel(R) Arc(TM) 140V", reason: null },
      { id: "cpu", label: "CPU", present: true, name: "Test CPU", reason: null },
    ],
    models: MODEL_IDS.filter((m) => m !== "auto").map((model) => ({ backend: "openvino", model, variant: "int8", size_mb: SIZES[model], installed: installed.includes(model) })),
    choices,
  };
}

const optionStates = (ui, id) => Object.fromEntries([...ui.$(id).options].map((o) => [o.value, o.disabled]));
const option = (ui, id, value) => [...ui.$(id).options].find((o) => o.value === value);
const change = async (ui, id, value) => {
  ui.$(id).value = value;
  ui.$(id).dispatchEvent(new dom.window.Event("change"));
  await flush();
};
const openSettings = async () => {
  document.querySelector('[data-tab="settings"]').click();
  await flush();
};

test("die erste Seite zeigt eine ruhige Zeile mit der aufgelösten Wahl", async () => {
  const ui = await boot({ list_models: () => modelsInfo() });
  assert.equal(ui.$("summary").textContent, "NPU · large-v3-turbo · OpenVINO");
  assert.equal(ui.$("summary").classList.contains("warn"), false);
  assert.equal(ui.$("start").textContent, "Aufnahme starten");
  assert.equal(document.querySelectorAll("#view-idle select").length, 0);
});

test("die erste Seite nennt Mikrofon und Sprechertrennung, auch nach einer Änderung in den Einstellungen", async () => {
  const ui = await boot({ list_models: () => modelsInfo() }, { mic: "laptop", diarize: false });
  assert.equal(ui.$("summary-input").textContent, "Mikrofon: Laptop · Sprecher trennen: nein");
  assert.equal(ui.$("summary").textContent, "NPU · large-v3-turbo · OpenVINO");
  await openSettings();
  ui.$("set-diarize").checked = true;
  ui.$("set-diarize").dispatchEvent(new dom.window.Event("change"));
  await flush();
  assert.equal(ui.$("summary-input").textContent, "Mikrofon: Laptop · Sprecher trennen: ja");
  await change(ui, "set-mic", "dji");
  assert.equal(ui.$("summary-input").textContent, "Mikrofon: DJI · Sprecher trennen: ja");
});

test("ohne Antwort der Engine zeigt die Zeile die gespeicherte Wahl und die Einstellungen sagen es", async () => {
  const ui = await boot({}, { device: "cpu" });
  assert.equal(ui.$("summary").textContent, "CPU · Automatisch · OpenVINO");
  await openSettings();
  assert.ok(ui.$("choice-info").textContent.includes("Die Auswahl lässt sich nicht prüfen"));
});

test("die Einstellungen haben drei Auswahlfelder, Standard ist Automatisch und die Wahl wird gespeichert", async () => {
  const saved = [];
  const ui = await boot({ list_models: () => modelsInfo(), save_settings: (a) => saved.push(a.settings) });
  await openSettings();
  for (const id of ["set-backend", "set-hardware", "set-model"]) assert.equal(ui.$(id).value, "auto");
  assert.equal(ui.$("set-backend").options[0].textContent, "Automatisch");
  await change(ui, "set-hardware", "igpu");
  await change(ui, "set-model", "large-v3-turbo");
  assert.deepEqual({ d: saved.at(-1).device, b: saved.at(-1).backend, m: saved.at(-1).model }, { d: "igpu", b: "auto", m: "large-v3-turbo" });
  assert.equal(ui.$("summary").textContent, "iGPU · large-v3-turbo · OpenVINO");
  await ui.click("start");
  assert.deepEqual(ui.called("start_session")[0].args.options, { title: null, audio: null, mic: null, device: "igpu", backend: "auto", model: "large-v3-turbo", diarize: true });
});

test("eine gespeicherte Wahl bleibt nach dem Neustart stehen", async () => {
  const ui = await boot({ list_models: () => modelsInfo() }, { device: "cpu", backend: "openvino", model: "small" });
  assert.equal(ui.$("summary").textContent, "CPU · small · OpenVINO · nicht installiert");
  await openSettings();
  assert.deepEqual([ui.$("set-backend").value, ui.$("set-hardware").value, ui.$("set-model").value], ["openvino", "cpu", "small"]);
});

test("nicht vorhandene Hardware und unmögliche Kombinationen sind ausgegraut und nennen den Grund", async () => {
  const ui = await boot({ list_models: () => modelsInfo() });
  await openSettings();
  const hardware = optionStates(ui, "set-hardware");
  assert.equal(hardware.gpu, true);
  assert.deepEqual([hardware.auto, hardware.npu, hardware.igpu, hardware.cpu], [false, false, false, false]);
  const gpu = option(ui, "set-hardware", "gpu");
  assert.ok(gpu.textContent.includes("nicht vorhanden") && gpu.title.includes("Keine dedizierte Grafikkarte"));
  assert.equal(optionStates(ui, "set-backend")["faster-whisper"], true);
  assert.ok(option(ui, "set-backend", "faster-whisper").title.includes("uv sync --extra faster-whisper"));
  await change(ui, "set-hardware", "npu");
  const fw = option(ui, "set-backend", "faster-whisper");
  assert.ok(fw.disabled && fw.title.includes("nur auf der CPU"));
});

test("eine gespeicherte alte GPU-Wahl bleibt gültig und angezeigt, obwohl die Karte fehlt", async () => {
  const ui = await boot({ list_models: () => modelsInfo() }, { device: "gpu" });
  assert.equal(ui.$("summary").textContent, "iGPU · large-v3-turbo · OpenVINO");
  await openSettings();
  assert.equal(ui.$("set-hardware").value, "gpu");
  assert.ok(ui.$("choice-info").textContent.includes("läuft hier auf der iGPU"));
});

test("ein nicht installiertes Modell zeigt Größe und Download mit Fortschritt und Abschluss", async () => {
  let installed = ["large-v3-turbo"];
  const ui = await boot({ list_models: () => modelsInfo(installed), download_model: () => null }, { model: "small", device: "cpu" });
  await openSettings();
  assert.equal(ui.$("download-box").hidden, false);
  assert.ok(ui.$("download-text").textContent.includes("small") && ui.$("download-text").textContent.includes("257 MB"));
  const small = option(ui, "set-model", "small");
  assert.ok(small.textContent.includes("nicht installiert") && small.textContent.includes("257 MB"));
  await ui.click("download");
  assert.deepEqual(ui.called("download_model")[0].args, { backend: "openvino", model: "small:int8" });
  assert.equal(ui.$("download").disabled, true);
  await ui.emit("download-event", { event: "download", repo: "OpenVINO/whisper-small-int8-ov", bytes: 100_000_000, total: 257_000_000 });
  assert.equal(ui.$("download-bar").hidden, false);
  assert.equal(ui.$("download-bar").value, 100_000_000);
  assert.ok(ui.$("download-status").textContent.includes("100 MB") && ui.$("download-status").textContent.includes("257 MB"));
  installed = ["large-v3-turbo", "small"];
  await ui.emit("download-event", { event: "done", messages: [] });
  assert.equal(ui.$("download-box").hidden, true);
  assert.equal(ui.$("summary").textContent, "CPU · small · OpenVINO");
  assert.ok(ui.called("list_models").length >= 2);
});

test("ein fehlgeschlagener Download zeigt Meldung und Einzelheiten und lässt den Knopf wieder zu", async () => {
  const ui = await boot({ list_models: () => modelsInfo(), download_model: () => null }, { model: "small", device: "cpu" });
  await openSettings();
  await ui.click("download");
  await ui.emit("download-event", { event: "error", message: "Der Download ist fehlgeschlagen.", detail: "whisper OpenVINO/whisper-small-int8-ov: FEHLER OSError: offline" });
  assert.ok(ui.$("download-status").textContent.includes("Download fehlgeschlagen: Der Download ist fehlgeschlagen."));
  assert.equal(ui.$("download-status").classList.contains("warn"), true);
  assert.equal(ui.$("download-more").hidden, false);
  assert.ok(ui.$("download-detail").textContent.includes("offline"));
  assert.equal(ui.$("download").disabled, false);
  assert.equal(ui.$("download-bar").hidden, true);
  await ui.click("download");
  await ui.emit("download-exit", { code: -1, terminal: false, stderr: "abgestürzt" });
  assert.ok(ui.$("download-status").textContent.includes("unerwartet beendet"));
});

test("ein Fehler beim Starten des Downloads wird angezeigt", async () => {
  const ui = await boot({ list_models: () => modelsInfo(), download_model: () => Promise.reject("Es läuft bereits ein Download.") }, { model: "small", device: "cpu" });
  await openSettings();
  await ui.click("download");
  assert.ok(ui.$("download-status").textContent.includes("Es läuft bereits ein Download."));
  assert.equal(ui.$("download").disabled, false);
});

test("das Ergebnis nennt Hardware, Modell und Backend, die wirklich liefen, und einen Rückfall", async () => {
  const ui = await boot({ list_models: () => modelsInfo(), start_session: () => null });
  await startRun(ui);
  await ui.emit("engine-event", { event: "transcribing", audio: AUDIO, meeting_id: "m-1", backend: "openvino", device: "NPU", model: "large-v3-turbo", variant: "fp16" });
  assert.equal(ui.$("engine-name").textContent, "Läuft auf: NPU · large-v3-turbo · OpenVINO");
  await ui.emit("engine-event", { event: "warning", message: "NPU fehlgeschlagen: RuntimeError: DEVICE_LOST. Rückfall auf iGPU (OpenVINO)." });
  await ui.emit("engine-event", { event: "fallback", from: "NPU", to: "iGPU", backend: "openvino" });
  assert.equal(ui.$("engine-name").textContent, "Läuft auf: iGPU · large-v3-turbo · OpenVINO (Rückfall von NPU)");
  assert.ok(ui.$("warnings").textContent.includes("Rückfall auf iGPU") && ui.$("warnings").hidden === false);
  const ran = { backend: "openvino", devices: ["iGPU"], model: "large-v3-turbo", variant: "int8", requested: "Automatisch", fallback: true, text: "iGPU · large-v3-turbo · OpenVINO" };
  await ui.emit("engine-event", { event: "done", meeting_id: "m-1", report: "C:\\p.md", audio: AUDIO, segments: 2, speakers: 0, labels: [], ran });
  assert.ok(ui.$("result-ran").textContent.includes("Verarbeitet mit: iGPU · large-v3-turbo · OpenVINO"));
  assert.ok(ui.$("result-ran").textContent.includes("Angefordert war Automatisch, es gab einen Rückfall."));
});

test("ein Lauf ohne Rückfall nennt nur die tatsächliche Wahl", async () => {
  const ui = await boot({ start_session: () => null });
  await startRun(ui);
  const ran = { backend: "openvino", devices: ["NPU"], model: "large-v3-turbo", variant: "fp16", requested: "NPU", fallback: false, text: "NPU · large-v3-turbo · OpenVINO" };
  await ui.emit("engine-event", { event: "done", meeting_id: "m", report: "C:\\p.md", audio: AUDIO, segments: 1, speakers: 0, labels: [], ran });
  assert.equal(ui.$("result-ran").textContent, "Verarbeitet mit: NPU · large-v3-turbo · OpenVINO");
});
