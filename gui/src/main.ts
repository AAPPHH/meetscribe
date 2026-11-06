import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type Settings = { engine: string; repo: string; device: string; mic: string; diarize: boolean };
type Label = { label: string; sample: string };
type EngineEvent = {
  event: string;
  message?: string;
  input?: string;
  audio?: string | null;
  meeting_id?: string | null;
  retry?: string | null;
  report?: string;
  segments?: number;
  speakers?: number;
  labels?: Label[];
};
type ExitInfo = { code: number | null; terminal: boolean; stderr: string };
type Check = { id: string; label: string; ok: boolean; fix: string | null; detail?: string | null };
type Meeting = { meeting_id: string; title: string | null; meeting_date: string; segments: number; speakers: number; duration_s: number };
type Hit = { meeting_id: string; start_s: number; display_name: string; text: string };
type Devices = { inputs: { name: string }[]; inputs_error: string | null; openvino: { device: string; name: string }[] };

const TEXT = {
  stages: { preflight: "Prüfe Voraussetzungen", recording: "Aufnahme läuft", transcribing: "Transkription und Sprechertrennung", saving: "Beende Aufnahme" } as Record<string, string>,
  devices: { npu: "NPU", gpu: "GPU", cpu: "CPU" } as Record<string, string>,
  micFixed: [["", "Automatisch (DJI, sonst Laptop)"], ["dji", "DJI"], ["laptop", "Laptop"], ["both", "Beide Mikrofone"]],
  unknownExit: "Die Engine wurde unerwartet beendet.",
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const call = <T>(cmd: string, args: Record<string, unknown> = {}) => invoke<T>(cmd, args);
const text = (err: unknown) => (typeof err === "string" ? err : err instanceof Error ? err.message : JSON.stringify(err));
const clock = (seconds: number) => `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, content = "", cls = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.textContent = content;
  if (cls) node.className = cls;
  return node;
}

class Markdown {
  static inline(parent: HTMLElement, line: string) {
    line.split("**").forEach((part, i) => parent.append(i % 2 ? el("strong", part) : document.createTextNode(part)));
  }

  static render(source: string, target: HTMLElement) {
    target.replaceChildren();
    let list: HTMLUListElement | null = null;
    for (const raw of source.split("\n")) {
      const line = raw.trimEnd();
      const heading = /^(#{1,3})\s+(.*)$/.exec(line);
      if (line.startsWith("- ")) {
        list ??= target.appendChild(el("ul"));
        Markdown.inline(list.appendChild(el("li")), line.slice(2));
        continue;
      }
      list = null;
      if (heading) target.append(el(`h${heading[1].length}` as "h1", heading[2]));
      else if (line.trim()) Markdown.inline(target.appendChild(el("p")), line);
    }
  }
}

class Modal {
  private resolve: ((confirmed: boolean) => void) | null = null;

  constructor() {
    $("modal-keep").addEventListener("click", () => this.close(false));
    $("modal-confirm").addEventListener("click", () => this.close(true));
  }

  get open() {
    return !$("modal").hidden;
  }

  private close(confirmed: boolean) {
    $("modal").hidden = true;
    this.resolve?.(confirmed);
    this.resolve = null;
  }

  ask(title: string, body: string, keep: string, confirm: string) {
    if (this.open) return Promise.resolve(false);
    $("modal-title").textContent = title;
    $("modal-text").textContent = body;
    $("modal-keep").textContent = keep;
    $("modal-confirm").textContent = confirm;
    $("modal").hidden = false;
    $("modal-keep").focus();
    return new Promise<boolean>((resolve) => (this.resolve = resolve));
  }
}

class ReportView {
  constructor(private target: HTMLElement) {}

  async show(meetingId: string) {
    this.target.hidden = false;
    this.target.replaceChildren(el("p", "Lade Protokoll …", "muted"));
    try {
      const report = await call<{ markdown: string }>("get_report", { meetingId });
      Markdown.render(report.markdown, this.target);
      this.target.scrollIntoView({ block: "nearest" });
    } catch (err) {
      this.target.replaceChildren(el("p", text(err), "notice error"));
    }
  }
}

class Tabs {
  private listeners: Record<string, () => void> = {};

  constructor() {
    document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => b.addEventListener("click", () => this.select(b.dataset.tab!)));
  }

  on(tab: string, fn: () => void) {
    this.listeners[tab] = fn;
  }

  select(tab: string) {
    document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    document.querySelectorAll<HTMLElement>(".panel").forEach((p) => (p.hidden = p.id !== `tab-${tab}`));
    this.listeners[tab]?.();
  }
}

class SettingsStore {
  value: Settings = { engine: "", repo: "", device: "npu", mic: "", diarize: true };

  async load() {
    this.value = await call<Settings>("get_settings");
  }

  async save(patch: Partial<Settings>) {
    this.value = { ...this.value, ...patch };
    await call("save_settings", { settings: this.value });
  }
}

class StartView {
  private started = 0;
  private timer = 0;
  private recording = false;
  private lastOptions: Record<string, unknown> | null = null;
  private result: EngineEvent | null = null;
  private failure: EngineEvent | null = null;
  private ended = false;

  constructor(private settings: SettingsStore, private tabs: Tabs) {
    $("start").addEventListener("click", () => this.begin(this.options()));
    $("stop").addEventListener("click", () => this.stop());
    $("pick").addEventListener("click", async () => {
      const path = await call<string | null>("pick_audio");
      if (path) $<HTMLInputElement>("audio").value = path;
    });
    $("save-names").addEventListener("click", () => this.saveNames());
    $("open-report").addEventListener("click", () => this.openReport());
    $("again").addEventListener("click", () => this.reset());
    $("error-back").addEventListener("click", () => this.reset());
    $("retry").addEventListener("click", () => this.retry());
    $("error-meetings").addEventListener("click", () => this.tabs.select("meetings"));
    $("goto-checks").addEventListener("click", () => this.tabs.select("checks"));
    listen<EngineEvent>("engine-event", (e) => this.onEvent(e.payload));
    listen<ExitInfo>("engine-exit", (e) => this.onExit(e.payload));
  }

  summary() {
    const s = this.settings.value;
    const mic = TEXT.micFixed.find(([v]) => v === s.mic)?.[1] ?? s.mic;
    $("summary").textContent = `Mikrofon: ${mic} · Gerät: ${TEXT.devices[s.device]} · Sprecher trennen: ${s.diarize ? "ja" : "nein"}`;
  }

  options() {
    const s = this.settings.value;
    const audio = $<HTMLInputElement>("audio").value.trim();
    return { title: $<HTMLInputElement>("title").value.trim() || null, audio: audio || null, mic: s.mic || null, device: s.device, diarize: s.diarize };
  }

  show(view: "idle" | "running" | "names" | "error") {
    for (const name of ["idle", "running", "names", "error"]) $(`view-${name}`).hidden = name !== view;
    if (view !== "running") $("warnings").hidden = $("warnings").childElementCount === 0 || view === "idle";
  }

  tick = () => {
    $("elapsed").textContent = clock((Date.now() - this.started) / 1000);
  };

  async begin(options: Record<string, unknown>) {
    this.lastOptions = options;
    this.result = this.failure = null;
    this.ended = false;
    this.recording = false;
    $("warnings").replaceChildren();
    $("input-name").textContent = options.audio ? String(options.audio) : "";
    this.stage("preflight");
    $<HTMLButtonElement>("stop").hidden = true;
    this.show("running");
    this.started = Date.now();
    this.tick();
    window.clearInterval(this.timer);
    this.timer = window.setInterval(this.tick, 500);
    try {
      await call("start_session", { options });
    } catch (err) {
      this.fail({ event: "error", message: text(err), audio: null, meeting_id: null, retry: null });
    }
  }

  stage(key: string) {
    $("stage").textContent = TEXT.stages[key] ?? key;
  }

  async stop() {
    $<HTMLButtonElement>("stop").disabled = true;
    this.stage("saving");
    try {
      await call("stop_session");
    } catch (err) {
      this.warn(text(err));
      $<HTMLButtonElement>("stop").disabled = false;
    }
  }

  warn(message: string) {
    const list = $("warnings");
    list.appendChild(el("li", message));
    list.hidden = false;
  }

  onEvent(ev: EngineEvent) {
    if (ev.event === "preflight") this.stage("preflight");
    else if (ev.event === "recording") {
      this.recording = true;
      this.started = Date.now();
      this.stage("recording");
      $("input-name").textContent = `Eingang: ${ev.input ?? ""}`;
      const stop = $<HTMLButtonElement>("stop");
      stop.hidden = false;
      stop.disabled = false;
    } else if (ev.event === "warning") this.warn(ev.message ?? "");
    else if (ev.event === "transcribing") {
      this.recording = false;
      this.stage("transcribing");
      $("stop").hidden = true;
      if (ev.audio) $("input-name").textContent = `Audio: ${ev.audio}`;
    } else if (ev.event === "done") this.finish(ev);
    else if (ev.event === "error") this.fail(ev);
  }

  onExit(info: ExitInfo) {
    if (this.ended || info.terminal) return;
    const detail = info.stderr.trim();
    this.fail({ event: "error", message: `${TEXT.unknownExit} (Code ${info.code ?? "?"})${detail ? `\n${detail}` : ""}`, audio: null, meeting_id: null, retry: null });
  }

  endRun() {
    this.ended = true;
    this.recording = false;
    window.clearInterval(this.timer);
  }

  finish(ev: EngineEvent) {
    this.endRun();
    this.result = ev;
    const labels = ev.labels ?? [];
    $("result-title").textContent = "Fertig";
    $("result-info").textContent = `${ev.segments ?? 0} Abschnitte, ${ev.speakers ?? 0} Sprecher. Protokoll: ${ev.report ?? ""}`;
    const box = $("names");
    box.replaceChildren();
    if (!labels.length) box.append(el("p", "Keine Sprecher getrennt. Das Protokoll ist fertig.", "muted"));
    labels.forEach((l, i) => {
      const row = el("div", "", "speaker");
      const input = el("input");
      input.type = "text";
      input.dataset.label = l.label;
      input.id = `speaker-${i}`;
      input.placeholder = "Name (leer lassen = nicht ändern)";
      const caption = el("label", l.label, "label");
      caption.htmlFor = input.id;
      row.append(caption, el("div", `„${l.sample}“`, "sample"), input);
      box.append(row);
    });
    $<HTMLButtonElement>("save-names").hidden = !labels.length;
    $("names-status").textContent = "";
    this.show("names");
  }

  async saveNames() {
    if (!this.result) return;
    const names = Array.from($("names").querySelectorAll<HTMLInputElement>("input")).map((i) => ({ label: i.dataset.label!, name: i.value }));
    const status = $("names-status");
    const button = $<HTMLButtonElement>("save-names");
    button.disabled = true;
    status.textContent = "Speichere …";
    try {
      await call("save_speakers", { meetingId: this.result.meeting_id, report: this.result.report, names });
      status.textContent = "Namen gespeichert, Protokoll neu erzeugt.";
    } catch (err) {
      status.textContent = `Speichern fehlgeschlagen: ${text(err)}`;
    } finally {
      button.disabled = false;
    }
  }

  async openReport() {
    try {
      await call("open_report", { path: this.result?.report });
    } catch (err) {
      $("names-status").textContent = text(err);
    }
  }

  fail(ev: EngineEvent) {
    if (this.ended && this.failure) return;
    this.endRun();
    this.failure = ev;
    $("error-message").textContent = ev.message ?? "Unbekannter Fehler.";
    $("error-audio").textContent = ev.audio ? `Das Audio ist erhalten: ${ev.audio}` : "";
    $("error-retry").textContent = ev.retry ? `Kommandozeile: ${ev.retry}` : "";
    $("retry").hidden = !(ev.audio && !ev.meeting_id);
    $("error-meetings").hidden = !ev.meeting_id;
    this.show("error");
  }

  retry() {
    if (this.failure?.audio) this.begin({ ...this.lastOptions, audio: this.failure.audio });
  }

  reset() {
    $<HTMLInputElement>("audio").value = "";
    this.summary();
    this.show("idle");
  }
}

class CloseGuard {
  constructor(private modal: Modal, private start: StartView) {
    listen<{ stage: string }>("close-requested", (e) => this.ask(e.payload.stage));
  }

  async ask(stage: string) {
    if (stage === "recording") {
      const stopNow = await this.modal.ask(
        "Aufnahme läuft noch",
        "Das Fenster bleibt offen, bis die Aufnahme beendet ist. Beenden speichert das Audio und startet die Verarbeitung.",
        "Weiter aufnehmen",
        "Aufnahme beenden und speichern",
      );
      if (stopNow) await this.start.stop();
      return;
    }
    const leave = await this.modal.ask(
      "Verarbeitung läuft noch",
      "Das Audio ist gespeichert. Schließen bricht die Verarbeitung ab; sie lässt sich danach mit „Erneut versuchen“ wiederholen.",
      "Weiter warten",
      "Abbrechen und schließen",
    );
    if (!leave) return;
    try {
      await call("exit_app");
    } catch (err) {
      this.start.warn(text(err));
    }
  }
}

class MeetingsView {
  private report = new ReportView($("report-view"));

  constructor() {
    $("reload-meetings").addEventListener("click", () => this.load());
  }

  async load() {
    const status = $("meetings-status");
    const body = document.querySelector<HTMLTableSectionElement>("#meetings-table tbody")!;
    status.textContent = "Lade …";
    try {
      const meetings = await call<Meeting[]>("list_meetings");
      body.replaceChildren();
      status.textContent = meetings.length ? "" : "Noch keine Besprechungen.";
      for (const m of [...meetings].reverse()) {
        const row = el("tr");
        row.tabIndex = 0;
        row.dataset.id = m.meeting_id;
        row.append(el("td", String(m.meeting_date).slice(0, 10)), el("td", m.title || m.meeting_id), el("td", String(m.speakers)), el("td", clock(m.duration_s ?? 0)));
        const open = () => this.open(m.meeting_id, row);
        row.addEventListener("click", open);
        row.addEventListener("keydown", (e) => e.key === "Enter" && open());
        body.append(row);
      }
    } catch (err) {
      status.textContent = text(err);
    }
  }

  open(id: string, row?: HTMLElement) {
    document.querySelectorAll("#meetings-table tbody tr").forEach((r) => r.classList.toggle("selected", r === row || (r as HTMLElement).dataset.id === id));
    return this.report.show(id);
  }
}

class SearchView {
  constructor(private tabs: Tabs, private meetings: MeetingsView) {
    $("search-form").addEventListener("submit", (e) => {
      e.preventDefault();
      this.run();
    });
  }

  async run() {
    const term = $<HTMLInputElement>("search-term").value.trim();
    const status = $("search-status");
    const list = $("search-results");
    list.replaceChildren();
    if (!term) return (status.textContent = "Bitte einen Suchbegriff eingeben.");
    status.textContent = "Suche …";
    try {
      const hits = await call<Hit[]>("search_text", { term });
      status.textContent = hits.length ? `${hits.length} Treffer` : "Keine Treffer.";
      for (const h of hits) {
        const item = el("li");
        item.tabIndex = 0;
        item.append(el("div", `${h.meeting_id} · ${clock(h.start_s)} · ${h.display_name}`, "meta"), el("div", h.text));
        const open = async () => {
          this.tabs.select("meetings");
          await this.meetings.load();
          await this.meetings.open(h.meeting_id);
        };
        item.addEventListener("click", open);
        item.addEventListener("keydown", (e) => e.key === "Enter" && open());
        list.append(item);
      }
    } catch (err) {
      status.textContent = text(err);
    }
  }
}

class SettingsView {
  constructor(private store: SettingsStore, private start: StartView) {
    $("set-device").addEventListener("change", () => this.save({ device: $<HTMLSelectElement>("set-device").value }));
    $("set-mic").addEventListener("change", () => this.save({ mic: $<HTMLSelectElement>("set-mic").value }));
    $("set-diarize").addEventListener("change", () => this.save({ diarize: $<HTMLInputElement>("set-diarize").checked }));
    $("save-engine").addEventListener("click", () => this.saveEngine());
    $("reload-devices").addEventListener("click", () => this.loadDevices());
  }

  async save(patch: Partial<Settings>) {
    try {
      await this.store.save(patch);
      $("settings-status").textContent = "Gespeichert.";
    } catch (err) {
      $("settings-status").textContent = text(err);
    }
    this.start.summary();
  }

  async saveEngine() {
    await this.save({ engine: $<HTMLInputElement>("set-engine").value.trim(), repo: $<HTMLInputElement>("set-repo").value.trim() });
    await this.engineInfo();
  }

  async engineInfo() {
    try {
      const spec = await call<{ program: string; args: string[]; source: string }>("engine_info");
      $("engine-info").textContent = `Aktive Engine (${spec.source}): ${[spec.program, ...spec.args].join(" ")}`;
    } catch (err) {
      $("engine-info").textContent = text(err);
    }
  }

  fill() {
    const s = this.store.value;
    $<HTMLSelectElement>("set-device").value = s.device;
    $<HTMLInputElement>("set-diarize").checked = s.diarize;
    $<HTMLInputElement>("set-engine").value = s.engine;
    $<HTMLInputElement>("set-repo").value = s.repo;
    this.mics([]);
  }

  mics(names: string[]) {
    const select = $<HTMLSelectElement>("set-mic");
    const current = this.store.value.mic;
    const options = [...TEXT.micFixed, ...[...new Set(names)].map((n) => [n, n])];
    if (current && !options.some(([v]) => v === current)) options.push([current, current]);
    select.replaceChildren(...options.map(([v, label]) => Object.assign(el("option", label), { value: v })));
    select.value = current;
  }

  async loadDevices() {
    const info = $("devices-info");
    info.textContent = "Lese Geräte …";
    try {
      const d = await call<Devices>("list_devices");
      this.mics(d.inputs.map((i) => i.name));
      const ov = d.openvino.map((o) => `${o.device}: ${o.name}`).join(" · ");
      info.textContent = [d.inputs_error, `Beschleuniger: ${ov || "keine gefunden"}`].filter(Boolean).join(" ");
    } catch (err) {
      info.textContent = text(err);
    }
  }

  async activate() {
    this.fill();
    await Promise.all([this.engineInfo(), this.loadDevices()]);
  }
}

class ChecksView {
  constructor() {
    $("reload-checks").addEventListener("click", () => this.run());
  }

  async run() {
    const list = $("check-list");
    list.replaceChildren(el("li", "Prüfe …"));
    let rows: Check[];
    try {
      rows = await call<Check[]>("run_checks");
    } catch (err) {
      rows = [{ id: "check", label: "Prüfung", ok: false, fix: text(err) }];
    }
    list.replaceChildren(
      ...rows.map((r) => {
        const item = el("li", "", r.ok ? "ok" : "fail");
        const body = el("div");
        body.append(el("div", r.label, "label"));
        if (r.detail) body.append(el("div", r.detail, "muted"));
        if (r.fix) body.append(el("div", r.fix, "fix"));
        item.append(el("span", r.ok ? "In Ordnung" : "Fehlt", "mark"), body);
        return item;
      }),
    );
    const failing = rows.some((r) => !r.ok);
    $("check-badge").hidden = !failing;
    $("setup-warning").hidden = !failing;
  }
}

async function main() {
  const tabs = new Tabs();
  const modal = new Modal();
  const store = new SettingsStore();
  await store.load().catch(() => undefined);
  const start = new StartView(store, tabs);
  const meetings = new MeetingsView();
  new SearchView(tabs, meetings);
  const settings = new SettingsView(store, start);
  const checks = new ChecksView();
  new CloseGuard(modal, start);
  tabs.on("meetings", () => meetings.load());
  tabs.on("search", () => $("search-term").focus());
  tabs.on("settings", () => settings.activate());
  tabs.on("checks", () => checks.run());
  start.summary();
  checks.run();
}

main();
