import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { afterEach, test } from "node:test";
import { JSDOM } from "jsdom";

const HTML = readFileSync(new URL("../src/index.html", import.meta.url), "utf8").replace(/<script[\s\S]*?<\/script>/, "");
const BUNDLE = new URL("./.build/main.js", import.meta.url).href;
const SETTINGS = { engine: "", repo: "", device: "npu", mic: "", diarize: true };
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
  assert.equal(ui.$("warnings").children.length, 1);
  await ui.click("stop");
  assert.equal(ui.called("stop_session").length, 1);
});

test("nach einem Neuladen in der Verarbeitung gibt es keinen Stopp-Knopf, aber den Stand", async () => {
  const ui = await boot({ session_state: () => ({ active: true, stage: "processing", elapsed_s: 5, audio: AUDIO, warnings: [] }) });
  assert.equal(ui.$("view-running").hidden, false);
  assert.equal(ui.$("stop").hidden, true);
  assert.equal(ui.$("stage").textContent, "Transkription und Sprechertrennung");
  assert.ok(ui.$("input-name").textContent.includes(AUDIO));
});

test("eine unfertige Verarbeitung wird beim Start angeboten und lässt sich wiederholen oder vergessen", async () => {
  const pending = { audio: AUDIO, meeting_id: "m-1", title: "Runde", device: "gpu", diarize: false };
  let current = pending;
  const ui = await boot({ pending_run: () => current, start_session: () => null, discard_pending: () => (current = null) });
  assert.equal(ui.$("pending").hidden, false);
  assert.ok(ui.$("pending-audio").textContent.includes(AUDIO));
  await ui.click("pending-retry");
  assert.deepEqual(ui.called("start_session")[0].args.options, { title: "Runde", audio: AUDIO, mic: null, device: "gpu", diarize: false });
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
