import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type Settings = { engine: string; repo: string; device: string; backend: string; model: string; mic: string; diarize: boolean };
type Label = { label: string; sample: string };
type EngineEvent = {
  event: string;
  message?: string;
  detail?: string;
  input?: string;
  path?: string;
  audio?: string | null;
  meeting_id?: string | null;
  retry?: string | null;
  report?: string;
  segments?: number;
  speakers?: number;
  labels?: Label[];
  backend?: string;
  device?: string;
  model?: string;
  variant?: string;
  ran?: Ran;
  from?: string;
  to?: string;
  bytes?: number;
  total?: number;
};
type Ran = { backend: string; devices: string[]; model: string; variant: string; requested: string; fallback: boolean; text: string };
type ExitInfo = { code: number | null; terminal: boolean; stderr: string; audio?: string | null; log?: string | null };
type SessionState = { active: boolean; stage: string; elapsed_s?: number; audio?: string | null; input?: string | null; warnings?: string[] };
type Pending = { audio: string; meeting_id: string | null; title: string | null; device: string; backend?: string | null; model?: string | null; diarize: boolean };
type Platform = { mobile: boolean; notice: string | null };
type Check = { id: string; label: string; ok: boolean; fix: string | null; detail?: string | null; scope?: string | null };
type Meeting = { meeting_id: string; title: string | null; meeting_date: string; segments: number; speakers: number; duration_s: number; complete?: boolean };
type Hit = { meeting_id: string; start_s: number; display_name: string; text: string; complete?: boolean };
type Devices = { inputs: { name: string }[]; inputs_error: string | null; openvino: { device: string; name: string }[] };
type Resolved = { model: string; backend: string; hardware: string; device: string; variant: string; installed: boolean; status: string; reason: string | null; requested: string; notices: string[] };
type Choice = { backend: string; hardware: string; model: string; ok: boolean; reason: string | null; resolved: Resolved | null; size_mb: number | null; installed: boolean; status: string; notices: string[] };
type ModelsInfo = {
  default_model: string;
  backends: { id: string; label: string; installed: boolean; reason: string | null }[];
  hardware: { id: string; label: string; present: boolean; name: string | null; reason: string | null }[];
  models: { backend: string; model: string; variant: string; size_mb: number; installed: boolean }[];
  choices: Choice[];
};

const TEXT = {
  stages: {
    preflight: "Prüfe Voraussetzungen",
    recording: "Aufnahme läuft",
    transcribing: "Transkription und Sprechertrennung",
    transcribingPlain: "Transkription",
    saving: "Beende Aufnahme",
  } as Record<string, string>,
  auto: "Automatisch",
  hardware: { auto: "Automatisch (NPU, GPU, iGPU, CPU)", npu: "NPU", gpu: "GPU (dedizierte Karte)", igpu: "iGPU (integrierte Grafik)", cpu: "CPU" } as Record<string, string>,
  hardwareShort: { npu: "NPU", gpu: "GPU", igpu: "iGPU", cpu: "CPU" } as Record<string, string>,
  backends: { openvino: "OpenVINO", "faster-whisper": "faster-whisper" } as Record<string, string>,
  micFixed: [["", "Automatisch (DJI, sonst Laptop)"], ["dji", "DJI"], ["laptop", "Laptop"], ["both", "Beide Mikrofone"]],
  unknownExit: "Die Engine wurde unerwartet beendet.",
  close: {
    recording: ["Aufnahme läuft noch", "Das Fenster bleibt offen, bis die Aufnahme beendet ist. Beenden speichert das Audio{audio} und startet die Verarbeitung.", "Weiter aufnehmen", "Aufnahme beenden und speichern"],
    preflight: ["Vorbereitung läuft noch", "Es ist noch nichts aufgenommen und nichts gespeichert. Schließen bricht die Vorbereitung ab.", "Weiter warten", "Abbrechen und schließen"],
    processing: [
      "Verarbeitung läuft noch",
      "Das Audio bleibt erhalten{audio}. Schließen bricht die Verarbeitung ab und verwirft die unfertige Besprechung. Beim nächsten Start lässt sich die Verarbeitung über „Erneut versuchen“ wiederholen.",
      "Weiter warten",
      "Abbrechen und schließen",
    ],
  } as Record<string, string[]>,
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

  static cells(line: string) {
    return line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split(/(?<!\\)\|/)
      .map((cell) => cell.trim().replace(/\\\|/g, "|"));
  }

  static render(source: string, target: HTMLElement) {
    target.replaceChildren();
    let list: HTMLUListElement | null = null;
    let table: HTMLTableElement | null = null;
    for (const raw of source.split("\n")) {
      const line = raw.trimEnd();
      const heading = /^(#{1,3})\s+(.*)$/.exec(line);
      if (line.startsWith("|")) {
        list = null;
        if (/^\|[\s:|-]+$/.test(line)) continue;
        const head = !table;
        table ??= target.appendChild(el("table"));
        const row = table.appendChild(el("tr"));
        Markdown.cells(line).forEach((cell) => Markdown.inline(row.appendChild(el(head ? "th" : "td")), cell));
        continue;
      }
      table = null;
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

  dismiss() {
    this.close(false);
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

const size = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1).replace(".", ",")} GB` : `${mb} MB`);
const bytes = (n: number) => size(Math.round(n / 1_000_000));

class ModelCatalog {
  info: ModelsInfo | null = null;

  async load() {
    this.info = await call<ModelsInfo>("list_models").catch(() => null);
  }

  choice(settings: Settings, over: Partial<Settings> = {}) {
    const s = { ...settings, ...over };
    return this.info?.choices.find((c) => c.backend === s.backend && c.hardware === s.device && c.model === s.model);
  }

  backendLabel(id: string) {
    return this.info?.backends.find((b) => b.id === id)?.label ?? TEXT.backends[id] ?? id;
  }

  line(settings: Settings) {
    const choice = this.choice(settings);
    const r = choice?.resolved;
    if (!r) return { text: `${TEXT.hardware[settings.device]?.split(" (")[0] ?? settings.device} · ${settings.model === "auto" ? this.info?.default_model ?? TEXT.auto : settings.model} · ${settings.backend === "auto" ? this.backendLabel("openvino") : this.backendLabel(settings.backend)}`, problem: choice && !choice.ok ? choice.reason : null };
    const text = `${r.device} · ${r.model} · ${this.backendLabel(r.backend)}${choice.installed ? "" : " · nicht installiert"}`;
    return { text, problem: null };
  }
}

class SettingsStore {
  value: Settings = { engine: "", repo: "", device: "auto", backend: "auto", model: "auto", mic: "", diarize: true };

  async load() {
    this.value = await call<Settings>("get_settings");
  }

  async save(patch: Partial<Settings>) {
    this.value = { ...this.value, ...patch };
    await call("save_settings", { settings: this.value });
  }
}

class StartView {
  audio: string | null = null;
  private started = 0;
  private timer = 0;
  private result: EngineEvent | null = null;
  private failure: EngineEvent | null = null;
  private ended = false;
  private diarize = true;
  private pending: Pending | null = null;
  private planned = "";
  private model = "";

  constructor(private settings: SettingsStore, private tabs: Tabs, private modal: Modal, private catalog: ModelCatalog) {
    $("start").addEventListener("click", () => this.begin(this.options()));
    $("stop").addEventListener("click", () => this.stop());
    $("pick").addEventListener("click", async () => {
      const path = await call<string | null>("pick_audio");
      if (path) $<HTMLInputElement>("audio").value = path;
    });
    $("save-names").addEventListener("click", () => this.saveNames());
    $("open-report").addEventListener("click", () => this.openReport());
    $("again").addEventListener("click", () => this.reset());
    $("error-back").addEventListener("click", () => this.back());
    $("retry").addEventListener("click", () => this.retry());
    $("pending-retry").addEventListener("click", () => this.retryPending());
    $("pending-discard").addEventListener("click", () => this.discardPending());
    $("error-meetings").addEventListener("click", () => this.tabs.select("meetings"));
    $("goto-checks").addEventListener("click", () => this.tabs.select("checks"));
    listen<EngineEvent>("engine-event", (e) => this.onEvent(e.payload));
    listen<ExitInfo>("engine-exit", (e) => this.onExit(e.payload));
  }

  summary() {
    const { text, problem } = this.catalog.line(this.settings.value);
    const line = $("summary");
    line.textContent = problem ? `${text} · ${problem}` : text;
    line.classList.toggle("warn", Boolean(problem));
    const s = this.settings.value;
    const mic = TEXT.micFixed.find(([v]) => v === s.mic)?.[1] ?? s.mic;
    $("summary-input").textContent = `Mikrofon: ${mic} · Sprecher trennen: ${s.diarize ? "ja" : "nein"}`;
  }

  options() {
    const s = this.settings.value;
    const audio = $<HTMLInputElement>("audio").value.trim();
    return { title: $<HTMLInputElement>("title").value.trim() || null, audio: audio || null, mic: s.mic || null, device: s.device, backend: s.backend, model: s.model, diarize: s.diarize };
  }

  async restore() {
    const platform = await call<Platform>("platform_info").catch(() => null);
    if (platform?.mobile) {
      $("platform-notice").textContent = platform.notice ?? "";
      $("platform-notice").hidden = false;
      for (const id of ["start", "pick"]) $<HTMLButtonElement>(id).disabled = true;
      return;
    }
    const state = await call<SessionState>("session_state").catch(() => null);
    if (state?.active) this.resume(state);
    else await this.showPending();
  }

  async showPending() {
    const pending = await call<Pending | null>("pending_run").catch(() => null);
    $("pending").hidden = !pending;
    if (pending) $("pending-audio").textContent = `Audio: ${pending.audio}`;
    this.pending = pending;
  }

  retryPending() {
    const p = this.pending;
    if (p) this.begin({ title: p.title, audio: p.audio, mic: null, device: p.device, backend: p.backend ?? "auto", model: p.model ?? "auto", diarize: p.diarize });
  }

  async discardPending() {
    let status = "";
    try {
      await call("discard_pending");
    } catch (err) {
      status = text(err);
    }
    await this.showPending();
    $("pending-status").textContent = status;
  }

  show(view: "idle" | "running" | "names" | "error") {
    for (const name of ["idle", "running", "names", "error"]) $(`view-${name}`).hidden = name !== view;
    if (view !== "running") $("warnings").hidden = $("warnings").childElementCount === 0 || view === "idle";
  }

  tick = () => {
    $("elapsed").textContent = clock((Date.now() - this.started) / 1000);
  };

  startClock(elapsedS = 0) {
    this.started = Date.now() - elapsedS * 1000;
    this.tick();
    window.clearInterval(this.timer);
    this.timer = window.setInterval(this.tick, 500);
  }

  prepare(audio: string | null, diarize: boolean) {
    this.result = this.failure = null;
    this.ended = false;
    this.audio = audio;
    this.diarize = diarize;
    $("pending").hidden = true;
    $("input-name").textContent = "";
    $("engine-name").textContent = "";
    this.planned = "";
    this.showAudio();
    $<HTMLButtonElement>("stop").hidden = true;
    this.show("running");
  }

  async begin(options: Record<string, unknown>) {
    this.prepare(options.audio ? String(options.audio) : null, Boolean(options.diarize));
    $("warnings").replaceChildren();
    this.stage("preflight");
    this.startClock();
    try {
      await call("start_session", { options });
    } catch (err) {
      this.fail({ event: "error", message: text(err), audio: null, meeting_id: null, retry: null });
    }
  }

  resume(state: SessionState) {
    this.prepare(state.audio ?? null, this.settings.value.diarize);
    $("warnings").replaceChildren();
    (state.warnings ?? []).forEach((w) => this.warn(w));
    this.startClock(state.elapsed_s ?? 0);
    if (state.stage === "recording") this.recording(state.input ?? "");
    else if (state.stage === "processing") this.processing();
    else this.stage("preflight");
  }

  showAudio() {
    $("audio-path").textContent = this.audio ? `Audio: ${this.audio}` : "";
  }

  stage(key: string) {
    $("stage").textContent = TEXT.stages[key] ?? key;
  }

  recording(input: string) {
    this.stage("recording");
    $("input-name").textContent = `Eingang: ${input}`;
    this.showAudio();
    const stop = $<HTMLButtonElement>("stop");
    stop.hidden = false;
    stop.disabled = false;
  }

  processing() {
    this.stage(this.diarize ? "transcribing" : "transcribingPlain");
    $("stop").hidden = true;
    $("input-name").textContent = "";
    this.showAudio();
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

  engineName(device: string, suffix: string, backend?: string) {
    if (!device) return;
    this.planned = `${device} · ${this.model} · ${this.catalog.backendLabel(backend ?? "openvino")}`;
    $("engine-name").textContent = `Läuft auf: ${this.planned}${suffix}`;
  }

  warn(message: string) {
    const list = $("warnings");
    list.appendChild(el("li", message));
    list.hidden = false;
  }

  onEvent(ev: EngineEvent) {
    if (ev.event === "preflight") this.stage("preflight");
    else if (ev.event === "recording") {
      this.audio = ev.path ?? this.audio;
      this.startClock();
      this.recording(ev.input ?? "");
    } else if (ev.event === "warning") this.warn(ev.message ?? "");
    else if (ev.event === "fallback") this.engineName(ev.to ?? "", ` (Rückfall von ${ev.from ?? "?"})`, ev.backend);
    else if (ev.event === "transcribing") {
      this.audio = ev.audio ?? this.audio;
      this.model = ev.model ?? "";
      this.engineName(ev.device ?? "", "", ev.backend);
      this.processing();
    } else if (ev.event === "done") this.finish(ev);
    else if (ev.event === "error") this.fail(ev);
  }

  onExit(info: ExitInfo) {
    if (this.ended || info.terminal) return;
    const lines = [info.stderr.trim(), info.log ? `Protokolldatei der Engine: ${info.log}` : ""].filter(Boolean);
    this.fail({
      event: "error",
      message: `${TEXT.unknownExit} (Code ${info.code ?? "?"})`,
      detail: lines.join("\n"),
      audio: info.audio ?? this.audio,
      meeting_id: null,
      retry: null,
    });
  }

  endRun() {
    this.ended = true;
    window.clearInterval(this.timer);
    this.modal.dismiss();
  }

  finish(ev: EngineEvent) {
    this.endRun();
    this.result = ev;
    const labels = ev.labels ?? [];
    $("result-title").textContent = "Fertig";
    $("result-info").textContent = `${ev.segments ?? 0} Abschnitte${labels.length ? `, ${labels.length} Sprecher` : ""}. Protokoll: ${ev.report ?? ""}`;
    const ran = ev.ran;
    $("result-ran").textContent = ran ? `Verarbeitet mit: ${ran.text}${ran.fallback ? `. Angefordert war ${ran.requested}, es gab einen Rückfall.` : ""}` : "";
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
    $("error-detail").textContent = ev.detail ?? "";
    const detail = (ev.detail ?? "").trim();
    $("error-more").hidden = !detail || detail === (ev.message ?? "").trim();
    $("retry").hidden = !(ev.audio && !ev.meeting_id);
    $("error-meetings").hidden = !ev.meeting_id;
    this.show("error");
  }

  retry() {
    if (this.failure?.audio) this.begin({ ...this.options(), audio: this.failure.audio });
  }

  back() {
    this.summary();
    this.show("idle");
    this.showPending();
  }

  reset() {
    $<HTMLInputElement>("audio").value = "";
    $<HTMLInputElement>("title").value = "";
    this.back();
  }
}

class EngineNotice {
  constructor() {
    $("engine-notice-close").addEventListener("click", () => ($("engine-notice").hidden = true));
    listen<{ text: string }>("engine-notice", (e) => {
      $("engine-notice-text").textContent = e.payload.text;
      $("engine-notice").hidden = false;
    });
  }
}

class CloseGuard {
  constructor(private modal: Modal, private start: StartView) {
    listen<{ stage: string; audio?: string | null }>("close-requested", (e) => this.ask(e.payload.stage, e.payload.audio ?? this.start.audio));
  }

  async ask(stage: string, audio: string | null) {
    const [title, body, keep, confirm] = TEXT.close[stage] ?? TEXT.close.processing;
    const leave = await this.modal.ask(title, body.replace("{audio}", audio ? `: ${audio}` : ""), keep, confirm);
    if (!leave) return;
    if (stage === "recording") return this.start.stop();
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
      for (const m of meetings) {
        const row = el("tr");
        row.tabIndex = 0;
        row.dataset.id = m.meeting_id;
        const title = el("td", m.title || m.meeting_id);
        if (m.complete === false) title.append(el("span", "unvollständig", "tag"));
        row.append(el("td", String(m.meeting_date).slice(0, 10)), title, el("td", String(m.speakers)), el("td", clock(m.duration_s ?? 0)));
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
        const meta = el("div", `${h.meeting_id} · ${clock(h.start_s)} · ${h.display_name}`, "meta");
        if (h.complete === false) meta.append(el("span", "unvollständig", "tag"));
        item.append(meta, el("div", h.text));
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

class DownloadView {
  private running = false;

  constructor(private settings: SettingsStore, private catalog: ModelCatalog, private changed: () => void) {
    $("download").addEventListener("click", () => this.start());
    listen<EngineEvent>("download-event", (e) => this.onEvent(e.payload));
    listen<ExitInfo>("download-exit", (e) => this.onExit(e.payload));
  }

  target() {
    const choice = this.catalog.choice(this.settings.value);
    const r = choice?.resolved;
    return choice?.ok && r && !choice.installed ? { choice, r } : null;
  }

  render() {
    const target = this.target();
    $("download-box").hidden = !target && !this.running;
    if (!target) return;
    const { choice, r } = target;
    $("download-text").textContent = `Das Modell ${r.model} (${this.catalog.backendLabel(r.backend)}, ${r.variant}) ist noch nicht installiert. Download: ${size(choice.size_mb ?? 0)}.`;
    $<HTMLButtonElement>("download").disabled = this.running;
  }

  async start() {
    const target = this.target();
    if (!target) return;
    const { r } = target;
    this.running = true;
    $("download-status").textContent = "Download startet …";
    $("download-status").classList.remove("warn");
    $("download-more").hidden = true;
    $<HTMLProgressElement>("download-bar").value = 0;
    $("download-bar").hidden = false;
    this.render();
    try {
      await call("download_model", { backend: r.backend, model: `${r.model}:${r.variant}` });
    } catch (err) {
      this.failed(text(err), "");
    }
  }

  onEvent(ev: EngineEvent) {
    if (ev.event === "download") {
      const bar = $<HTMLProgressElement>("download-bar");
      bar.max = Math.max(ev.total ?? 0, ev.bytes ?? 0, 1);
      bar.value = ev.bytes ?? 0;
      $("download-status").textContent = `${bytes(ev.bytes ?? 0)} von etwa ${bytes(ev.total ?? 0)}`;
    } else if (ev.event === "done") this.finished();
    else if (ev.event === "error") this.failed(ev.message ?? "Der Download ist fehlgeschlagen.", ev.detail ?? "");
  }

  onExit(info: ExitInfo) {
    if (this.running && !info.terminal) this.failed(`Der Download wurde unerwartet beendet (Code ${info.code ?? "?"}).`, info.stderr.trim());
  }

  finished() {
    this.running = false;
    $("download-bar").hidden = true;
    $("download-status").textContent = "Fertig, das Modell ist installiert.";
    this.changed();
  }

  failed(message: string, detail: string) {
    this.running = false;
    $("download-bar").hidden = true;
    const status = $("download-status");
    status.textContent = `Download fehlgeschlagen: ${message}`;
    status.classList.add("warn");
    $("download-detail").textContent = detail;
    $("download-more").hidden = !detail.trim() || detail.trim() === message.trim();
    this.render();
  }
}

class SettingsView {
  constructor(private store: SettingsStore, private start: StartView, private checks: ChecksView, private catalog: ModelCatalog, private download: DownloadView) {
    $("set-backend").addEventListener("change", () => this.choose({ backend: $<HTMLSelectElement>("set-backend").value }));
    $("set-hardware").addEventListener("change", () => this.choose({ device: $<HTMLSelectElement>("set-hardware").value }));
    $("set-model").addEventListener("change", () => this.choose({ model: $<HTMLSelectElement>("set-model").value }));
    $("set-mic").addEventListener("change", () => this.save({ mic: $<HTMLSelectElement>("set-mic").value }));
    $("set-diarize").addEventListener("change", () => this.save({ diarize: $<HTMLInputElement>("set-diarize").checked }));
    $("save-engine").addEventListener("click", () => this.saveEngine());
    $("reload-devices").addEventListener("click", () => this.loadDevices());
  }

  async choose(patch: Partial<Settings>) {
    await this.save(patch);
    this.render();
  }

  async save(patch: Partial<Settings>) {
    try {
      await this.store.save(patch);
      $("settings-status").textContent = "Gespeichert.";
    } catch (err) {
      $("settings-status").textContent = text(err);
    }
    this.start.summary();
    if ("diarize" in patch) this.checks.run();
  }

  async saveEngine() {
    await this.save({ engine: $<HTMLInputElement>("set-engine").value.trim(), repo: $<HTMLInputElement>("set-repo").value.trim() });
    await this.engineInfo();
    await this.reloadCatalog();
    this.checks.run();
  }

  async reloadCatalog() {
    await this.catalog.load();
    this.render();
    this.start.summary();
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
    $<HTMLInputElement>("set-diarize").checked = s.diarize;
    $<HTMLInputElement>("set-engine").value = s.engine;
    $<HTMLInputElement>("set-repo").value = s.repo;
    this.mics([]);
    this.render();
  }

  fillSelect(id: string, options: { value: string; label: string; disabled: boolean; title: string }[], current: string) {
    const select = $<HTMLSelectElement>(id);
    select.replaceChildren(...options.map((o) => Object.assign(el("option", o.label), { value: o.value, disabled: o.disabled && o.value !== current, title: o.title })));
    if (!options.some((o) => o.value === current)) select.append(Object.assign(el("option", current), { value: current }));
    select.value = current;
  }

  render() {
    const s = this.store.value;
    const info = this.catalog.info;
    const reasonOf = (c?: Choice) => (c && !c.ok ? c.reason ?? "" : "");
    const mark = (label: string, reason: string) => (reason ? `${label} – nicht möglich` : label);
    const backends = ["auto", ...(info?.backends.map((b) => b.id) ?? ["openvino"])];
    this.fillSelect("set-backend", backends.map((id) => {
      const reason = reasonOf(this.catalog.choice(s, { backend: id })) || (info?.backends.find((b) => b.id === id)?.reason ?? "");
      return { value: id, label: mark(id === "auto" ? TEXT.auto : this.catalog.backendLabel(id), reason), disabled: Boolean(reason), title: reason };
    }), s.backend);
    const hardware = ["auto", "npu", "gpu", "igpu", "cpu"];
    this.fillSelect("set-hardware", hardware.map((id) => {
      const row = info?.hardware.find((h) => h.id === id);
      const reason = (row && !row.present ? row.reason ?? "Nicht vorhanden." : "") || reasonOf(this.catalog.choice(s, { device: id }));
      const name = row?.present && row.name ? ` – ${row.name}` : "";
      return { value: id, label: `${TEXT.hardware[id]}${reason ? " – " + (row && !row.present ? "nicht vorhanden" : "nicht möglich") : name}`, disabled: Boolean(reason), title: reason };
    }), s.device);
    const models = ["auto", ...(info ? [...new Set(info.models.map((m) => m.model))] : [s.model])];
    this.fillSelect("set-model", models.map((id) => {
      const choice = this.catalog.choice(s, { model: id });
      const reason = reasonOf(choice);
      const label = id === "auto" ? `${TEXT.auto} (${info?.default_model ?? ""})` : id;
      const notes = [choice?.ok && !choice.installed ? `nicht installiert, ${size(choice.size_mb ?? 0)}` : "", choice?.status === "unverified" ? "auf dieser Hardware ungeprüft" : ""].filter(Boolean);
      return { value: id, label: reason ? mark(label, reason) : notes.length ? `${label} – ${notes.join(", ")}` : label, disabled: Boolean(reason), title: reason || (choice?.resolved?.reason ?? "") };
    }), s.model);
    const current = this.catalog.choice(s);
    const note = $("choice-info");
    note.classList.toggle("warn", Boolean(current && !current.ok));
    if (!info) note.textContent = "Die Auswahl lässt sich nicht prüfen, die Engine antwortet nicht.";
    else if (current && !current.ok) note.textContent = current.reason ?? "";
    else {
      const r = current?.resolved;
      note.textContent = [...(current?.notices ?? []), r ? `Läuft auf ${this.catalog.line(s).text}.` : "", r?.status === "unverified" ? r.reason ?? "" : ""].filter(Boolean).join(" ");
    }
    this.download.render();
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
    await this.reloadCatalog();
  }

  async activate() {
    this.fill();
    await Promise.all([this.engineInfo(), this.loadDevices()]);
  }
}

class ChecksView {
  constructor(private settings: SettingsStore) {
    $("reload-checks").addEventListener("click", () => this.run());
  }

  skipped(row: Check) {
    return !row.ok && row.scope === "diarize" && !this.settings.value.diarize;
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
        const skipped = this.skipped(r);
        const item = el("li", "", r.ok || skipped ? "ok" : "fail");
        const body = el("div");
        body.append(el("div", r.label, "label"));
        if (r.detail) body.append(el("div", r.detail, "muted"));
        if (r.fix) body.append(el("div", skipped ? "Nicht nötig, solange „Sprecher trennen“ aus ist." : r.fix, skipped ? "muted" : "fix"));
        item.append(el("span", r.ok ? "In Ordnung" : skipped ? "Nicht nötig" : "Fehlt", "mark"), body);
        return item;
      }),
    );
    const failing = rows.some((r) => !r.ok && !this.skipped(r));
    $("check-badge").hidden = !failing;
    $("setup-warning").hidden = !failing;
    $("setup-hint").hidden = !rows.some((r) => !r.ok && r.scope === "diarize" && !this.skipped(r));
  }
}

async function main() {
  const tabs = new Tabs();
  const modal = new Modal();
  const store = new SettingsStore();
  await store.load().catch(() => undefined);
  const catalog = new ModelCatalog();
  await catalog.load();
  const start = new StartView(store, tabs, modal, catalog);
  const meetings = new MeetingsView();
  new SearchView(tabs, meetings);
  const checks = new ChecksView(store);
  let settings: SettingsView;
  const download = new DownloadView(store, catalog, () => settings.reloadCatalog());
  settings = new SettingsView(store, start, checks, catalog, download);
  new EngineNotice();
  new CloseGuard(modal, start);
  tabs.on("meetings", () => meetings.load());
  tabs.on("search", () => $("search-term").focus());
  tabs.on("settings", () => settings.activate());
  tabs.on("checks", () => checks.run());
  start.summary();
  checks.run();
  start.restore();
}

main();
