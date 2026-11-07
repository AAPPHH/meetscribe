import importlib.util
import platform
from pathlib import Path

import click

CATALOG = {
    "default_model": "large-v3-turbo",
    "hardware_order": ("npu", "gpu", "igpu", "cpu"),
    "backend_order": ("openvino", "faster-whisper"),
    "hardware": {
        "npu": {"label": "NPU", "name": "NPU", "absent": "Keine NPU gefunden."},
        "gpu": {"label": "GPU", "name": "Dedizierte Grafikkarte", "absent": "Keine dedizierte Grafikkarte gefunden."},
        "igpu": {"label": "iGPU", "name": "Integrierte Grafik", "absent": "Keine integrierte Grafik gefunden."},
        "cpu": {"label": "CPU", "name": "CPU", "absent": "Keine CPU gefunden."},
    },
    "backends": {
        "openvino": {
            "label": "OpenVINO",
            "module": "openvino_genai",
            "extra": None,
            "devices": {
                "npu": {"variants": ("fp16",), "default": "fp16", "refused": {"int8": "int8 läuft auf der NPU nicht: ZE_RESULT_ERROR_DEVICE_LOST gemessen. Auf der NPU nur fp16 verwenden."}},
                "gpu": {"variants": ("int8", "fp16"), "default": "int8", "absent": "Keine dedizierte Intel-Grafikkarte gefunden."},
                "igpu": {"variants": ("int8", "fp16"), "default": "int8"},
                "cpu": {"variants": ("int8", "fp16"), "default": "int8"},
            },
        },
        "faster-whisper": {
            "label": "faster-whisper",
            "module": "faster_whisper",
            "extra": "faster-whisper",
            "devices": {
                "gpu": {"variants": ("ct2",), "default": "ct2", "options": {"compute_type": "float16"}, "absent": "Keine NVIDIA-Karte (CUDA) gefunden."},
                "cpu": {"variants": ("ct2",), "default": "ct2", "options": {"compute_type": "int8"}},
            },
            "unsupported": "faster-whisper läuft nur auf der CPU und auf NVIDIA-Karten (CUDA).",
        },
    },
    "models": {
        "tiny": {"label": "tiny"},
        "base": {"label": "base"},
        "small": {"label": "small"},
        "medium": {"label": "medium"},
        "large-v3": {"label": "large-v3"},
        "large-v3-turbo": {"label": "large-v3-turbo"},
    },
    "variants": {
        "openvino": {
            "tiny": {"fp16": {"repo": "OpenVINO/whisper-tiny-fp16-ov", "mb": 84}, "int8": {"repo": "OpenVINO/whisper-tiny-int8-ov", "mb": 49}},
            "base": {"fp16": {"repo": "OpenVINO/whisper-base-fp16-ov", "mb": 154}, "int8": {"repo": "OpenVINO/whisper-base-int8-ov", "mb": 85}},
            "small": {"fp16": {"repo": "OpenVINO/whisper-small-fp16-ov", "mb": 493}, "int8": {"repo": "OpenVINO/whisper-small-int8-ov", "mb": 257}},
            "medium": {"fp16": {"repo": "OpenVINO/whisper-medium-fp16-ov", "mb": 1539}, "int8": {"repo": "OpenVINO/whisper-medium-int8-ov", "mb": 784}},
            "large-v3": {"fp16": {"repo": "OpenVINO/whisper-large-v3-fp16-ov", "mb": 3099}, "int8": {"repo": "OpenVINO/whisper-large-v3-int8-ov", "mb": 1568}},
            "large-v3-turbo": {"fp16": {"repo": "OpenVINO/whisper-large-v3-turbo-fp16-ov", "mb": 1628, "npu": "ok"}, "int8": {"repo": "OpenVINO/whisper-large-v3-turbo-int8-ov", "mb": 828}},
        },
        "faster-whisper": {
            "tiny": {"ct2": {"repo": "Systran/faster-whisper-tiny", "mb": 78}},
            "base": {"ct2": {"repo": "Systran/faster-whisper-base", "mb": 148}},
            "small": {"ct2": {"repo": "Systran/faster-whisper-small", "mb": 486}},
            "medium": {"ct2": {"repo": "Systran/faster-whisper-medium", "mb": 1531}},
            "large-v3": {"ct2": {"repo": "Systran/faster-whisper-large-v3", "mb": 3091}},
            "large-v3-turbo": {"ct2": {"repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "mb": 1622}},
        },
    },
    "unverified": "Auf der NPU für dieses Modell nicht geprüft; schlägt das Laden fehl, greift der Rückfall.",
    "npu_failed": "Gemessen: {model} ({variant}) lässt sich auf der NPU nicht übersetzen.",
    "notice_gpu": "Hinweis: Hier gibt es keine dedizierte Grafikkarte, --device gpu läuft deshalb auf der integrierten Grafik (iGPU). Künftig igpu angeben.",
}
HARDWARE_CHOICES = ("auto", *CATALOG["hardware_order"])
BACKEND_CHOICES = ("auto", *CATALOG["backend_order"])
CT2_FILES = ("model.bin", "config.json", "tokenizer.json")
VENDOR_INTEL = ("intel", "0x8086")
CUDA_NAME = "NVIDIA-Karte {index} (CUDA)"


class SelectionError(click.ClickException):
    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail

    def format_message(self):
        return f"{self.message}\n{self.detail}" if self.detail else self.message


class Catalog:
    @staticmethod
    def label(hardware):
        return CATALOG["hardware"][hardware]["label"]

    @staticmethod
    def entry(backend, model, variant):
        return CATALOG["variants"][backend][model][variant]

    @staticmethod
    def dir_name(entry):
        return entry["repo"].split("/")[-1]

    @staticmethod
    def spec(backend, hardware):
        return CATALOG["backends"][backend]["devices"].get(hardware)

    @staticmethod
    def variants(backend, model):
        return CATALOG["variants"][backend][model]

    @classmethod
    def backend_label(cls, backend):
        return CATALOG["backends"][backend]["label"]

    @staticmethod
    def installed(backend):
        return importlib.util.find_spec(CATALOG["backends"][backend]["module"]) is not None

    @classmethod
    def missing_reason(cls, backend):
        spec = CATALOG["backends"][backend]
        hint = f" Installieren mit: uv sync --extra {spec['extra']}" if spec["extra"] else ""
        return f"{spec['label']} ist nicht installiert.{hint}"


class Hardware:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else self.detect()

    @staticmethod
    def property(core, name, key):
        try:
            return str(core.get_property(name, key))
        except Exception:
            return ""

    @classmethod
    def openvino_rows(cls):
        try:
            import openvino as ov
            core = ov.Core()
            names = list(core.available_devices)
        except Exception:
            return []
        rows = []
        for name in names:
            base = name.split(".")[0]
            full = cls.property(core, name, "FULL_DEVICE_NAME")
            if base == "NPU":
                kind, kind_type = "npu", "integrated"
            elif base == "CPU":
                kind, kind_type = "cpu", "integrated"
            elif base == "GPU":
                discrete = "DISCRETE" in cls.property(core, name, "DEVICE_TYPE").upper()
                kind, kind_type = ("gpu", "discrete") if discrete else ("igpu", "integrated")
            else:
                continue
            vendor = f"{full} {cls.property(core, name, 'DEVICE_ARCHITECTURE')}".lower()
            intel = base != "GPU" or any(v in vendor for v in VENDOR_INTEL)
            rows.append({"id": kind, "device": name, "name": full, "type": kind_type, "cuda": None, "backends": ["openvino"] if intel else []})
        return rows

    @staticmethod
    def cuda_count():
        try:
            import ctranslate2
            return int(ctranslate2.get_cuda_device_count())
        except Exception:
            return 0

    @classmethod
    def detect(cls):
        rows = cls.openvino_rows()
        count, used = cls.cuda_count(), 0
        for row in rows:
            if row["id"] == "cpu":
                row["backends"].append("faster-whisper")
            elif row["id"] == "gpu" and "nvidia" in row["name"].lower() and used < count:
                row["cuda"] = used
                row["backends"].append("faster-whisper")
                used += 1
        rows += [
            {"id": "gpu", "device": f"cuda:{i}", "name": CUDA_NAME.format(index=i), "type": "discrete", "cuda": i, "backends": ["faster-whisper"]}
            for i in range(used, count)
        ]
        if not any(row["id"] == "cpu" for row in rows):
            rows.append({"id": "cpu", "device": "CPU", "name": platform.processor() or "CPU", "type": "integrated", "cuda": None, "backends": ["faster-whisper"]})
        return rows

    def of(self, kind, backend):
        return next((r for r in self.rows if r["id"] == kind and backend in r["backends"]), None)

    def present(self, kind):
        return any(r["id"] == kind for r in self.rows)


class Step:
    def __init__(self, backend, hardware, row, variant, entry, path, installed, status, reason=None):
        self.backend = backend
        self.hardware = hardware
        self.row = row
        self.variant = variant
        self.entry = entry
        self.path = path
        self.installed = installed
        self.status = status
        self.reason = reason

    @property
    def label(self):
        return Catalog.label(self.hardware)

    @property
    def options(self):
        return dict((Catalog.spec(self.backend, self.hardware) or {}).get("options", {}))

    def brief(self):
        return {"backend": self.backend, "hardware": self.hardware, "device": self.label, "variant": self.variant, "installed": self.installed, "status": self.status, "reason": self.reason}


class Plan:
    def __init__(self, model, steps, notices, requested):
        self.model = model
        self.steps = steps
        self.notices = notices
        self.requested = requested

    @property
    def primary(self):
        return self.steps[0]

    @property
    def chain(self):
        return [s for s in self.steps if s.installed]

    @property
    def installed(self):
        return self.primary.installed

    def brief(self):
        return {"model": self.model, **self.primary.brief(), "requested": self.requested, "notices": list(self.notices)}


class Selection:
    def __init__(self, models, hardware=None):
        self.models = models
        self.hardware = hardware or Hardware()

    @staticmethod
    def split(text):
        name, _, variant = (text or "auto").strip().partition(":")
        return name.strip() or "auto", variant.strip() or None

    def override(self, name):
        for path in (Path(name), self.models.root / name):
            if path.exists():
                return path
        return None

    def model_of(self, text):
        name, variant = self.split(text)
        if name.lower() == "auto":
            return CATALOG["default_model"], variant, None
        if name in CATALOG["models"]:
            return name, variant, None
        path = self.override((text or "").strip())
        if path is None:
            raise SelectionError(f"Unbekanntes Modell '{text}'. Verfügbar: {', '.join(CATALOG['models'])} (optional mit :Variante, z. B. small:fp16) oder ein Modellverzeichnis.")
        return path.name, None, path

    def check(self, backend, hardware, model, variant=None, override=None):
        spec = Catalog.spec(backend, hardware)
        label = Catalog.backend_label(backend)
        if not Catalog.installed(backend):
            return {"status": "refused", "reason": Catalog.missing_reason(backend), "variant": variant}
        if spec is None:
            return {"status": "refused", "reason": CATALOG["backends"][backend].get("unsupported") or f"{label} unterstützt {Catalog.label(hardware)} nicht.", "variant": variant}
        row = self.hardware.of(hardware, backend)
        if row is None:
            return {"status": "refused", "reason": spec.get("absent") or CATALOG["hardware"][hardware]["absent"], "variant": variant}
        if override is not None:
            return {"status": "ok", "reason": None, "variant": "custom", "row": row}
        variants = Catalog.variants(backend, model)
        if variant is not None and variant not in variants:
            return {"status": "refused", "reason": f"Unbekannte Variante '{variant}' für {model}. Verfügbar: {', '.join(variants)}.", "variant": variant, "row": row}
        if variant is not None and variant not in spec["variants"]:
            return {"status": "refused", "reason": spec.get("refused", {}).get(variant) or f"Variante {variant} läuft auf {Catalog.label(hardware)} mit {label} nicht.", "variant": variant, "row": row}
        return {"status": "ok", "reason": None, "variant": variant, "row": row}

    def step(self, backend, hardware, model, variant, override):
        verdict = self.check(backend, hardware, model, variant, override)
        if verdict["status"] == "refused":
            return None, verdict
        row = verdict["row"]
        if override is not None:
            return Step(backend, hardware, row, "custom", {"repo": None, "mb": None}, override, self.models.ready(backend, override), "ok"), verdict
        spec = Catalog.spec(backend, hardware)
        wanted = [variant] if variant else [spec["default"], *[v for v in spec["variants"] if v != spec["default"]]]
        steps, failed = [], None
        for name in wanted:
            entry = Catalog.entry(backend, model, name)
            path = self.models.root / Catalog.dir_name(entry)
            if hardware == "npu" and entry.get("npu") == "fail":
                failed = {"status": "refused", "reason": CATALOG["npu_failed"].format(model=model, variant=name)}
                continue
            status = "unverified" if hardware == "npu" and entry.get("npu") != "ok" else "ok"
            reason = CATALOG["unverified"] if status == "unverified" else None
            steps.append(Step(backend, hardware, row, name, entry, path, self.models.ready(backend, path), status, reason))
        if not steps:
            return None, failed
        return next((s for s in steps if s.installed), steps[0]), verdict

    def candidates(self, backends, kinds, model, variant, override):
        found, reasons = [], []
        for kind in kinds:
            for backend in backends:
                step, verdict = self.step(backend, kind, model, variant, override)
                if step is None:
                    reasons.append((kind, backend, verdict["reason"]))
                else:
                    found.append(step)
        return found, reasons

    def resolve(self, backend="auto", hardware="auto", model="auto", fallback=True):
        backend, hardware = (backend or "auto").lower(), (hardware or "auto").lower()
        if backend not in BACKEND_CHOICES:
            raise SelectionError(f"Unbekanntes Backend '{backend}'. Verfügbar: {', '.join(CATALOG['backend_order'])}.")
        if hardware not in HARDWARE_CHOICES:
            raise SelectionError(f"Unbekannte Hardware '{hardware}'. Verfügbar: {', '.join(HARDWARE_CHOICES)}.")
        name, variant, override = self.model_of(model)
        backends = list(CATALOG["backend_order"]) if backend == "auto" else [backend]
        order = list(CATALOG["hardware_order"])
        notices, requested = [], hardware
        kinds = order if hardware == "auto" else [hardware]
        found, reasons = self.candidates(backends, kinds, name, variant, override)
        if not found and hardware == "gpu" and not self.hardware.present("gpu"):
            found, more = self.candidates(backends, ["igpu"], name, variant, override)
            reasons += more
            if found:
                notices.append(CATALOG["notice_gpu"])
        if not found:
            raise SelectionError(self.refusal(backend, hardware, name, reasons))
        ready = [s for s in found if s.installed]
        primary = (ready or found)[0]
        if ready and primary is not found[0]:
            notices.append(f"{found[0].label} mit {Catalog.backend_label(found[0].backend)} übersprungen: Modell {name} ({found[0].variant}) ist nicht installiert.")
        steps = [primary]
        if fallback:
            start = order.index(primary.hardware)
            later = [k for k in order[start + 1:]]
            more, _ = self.candidates([primary.backend], later, name, variant, override)
            steps += [s for s in more if s.installed]
        return Plan(name, steps, notices, Catalog.label(requested) if requested != "auto" else "Automatisch")

    @staticmethod
    def refusal(backend, hardware, model, reasons):
        lines = list(dict.fromkeys(reason for _, _, reason in reasons if reason))
        head = f"Diese Kombination geht nicht (Backend {backend}, Hardware {hardware}, Modell {model})."
        return head + (" " + " ".join(lines) if lines else "")

    def matrix(self):
        rows = []
        for backend in BACKEND_CHOICES:
            for hardware in HARDWARE_CHOICES:
                for model in ("auto", *CATALOG["models"]):
                    row = {"backend": backend, "hardware": hardware, "model": model}
                    try:
                        plan = self.resolve(backend, hardware, model)
                        row.update(ok=True, reason=None, resolved=plan.brief(), size_mb=plan.primary.entry["mb"], installed=plan.installed, status=plan.primary.status, notices=plan.notices)
                    except SelectionError as exc:
                        row.update(ok=False, reason=exc.message, resolved=None, size_mb=None, installed=False, status="refused", notices=[])
                    rows.append(row)
        return rows

    def listing(self):
        rows = []
        for backend in CATALOG["backend_order"]:
            for model in CATALOG["models"]:
                for variant, entry in Catalog.variants(backend, model).items():
                    path = self.models.root / Catalog.dir_name(entry)
                    verdicts = {}
                    for hardware in CATALOG["hardware_order"]:
                        step, verdict = self.step(backend, hardware, model, variant, None)
                        verdicts[hardware] = {"status": step.status if step else "refused", "reason": step.reason if step else verdict["reason"]}
                    rows.append({
                        "backend": backend, "model": model, "variant": variant, "repo": entry["repo"], "dir": Catalog.dir_name(entry),
                        "size_mb": entry["mb"], "installed": self.models.ready(backend, path), "hardware": verdicts,
                    })
        return rows


class OpenVinoEngine:
    backend = "openvino"

    @staticmethod
    def ready(path):
        path = Path(path)
        graphs = list(path.glob("*.xml")) if path.is_dir() else []
        return bool(graphs) and all(g.stat().st_size and g.with_suffix(".bin").is_file() and g.with_suffix(".bin").stat().st_size for g in graphs)

    def __init__(self, step, models, language):
        import openvino_genai
        self.device, self.language, self.model = step.label, language, step.path.name
        self.pipe = openvino_genai.WhisperPipeline(
            str(step.path), step.row["device"], word_timestamps=True, CACHE_DIR=str(models.cache_dir)
        )

    def transcribe(self, audio):
        result = self.pipe.generate(
            audio.tolist(), language=f"<|{self.language}|>", task="transcribe", return_timestamps=True, word_timestamps=True
        )
        return {
            "words": [{"text": w.word, "start_s": w.start_ts, "end_s": w.end_ts} for w in (result.words or [])],
            "chunks": [{"text": c.text, "start_s": c.start_ts, "end_s": c.end_ts} for c in (result.chunks or [])],
        }


class FasterWhisperEngine:
    backend = "faster-whisper"

    @staticmethod
    def ready(path):
        path = Path(path)
        vocabulary = any((path / v).is_file() and (path / v).stat().st_size for v in ("vocabulary.txt", "vocabulary.json"))
        return path.is_dir() and vocabulary and all((path / n).is_file() and (path / n).stat().st_size for n in CT2_FILES)

    def __init__(self, step, models, language):
        from faster_whisper import WhisperModel
        options = step.options
        cuda = step.row["cuda"] is not None
        self.device, self.language = step.label, language
        self.model = f"{step.path.name}-{options['compute_type']}"
        self.pipe = WhisperModel(
            str(step.path), device="cuda" if cuda else "cpu", device_index=step.row["cuda"] or 0,
            compute_type=options["compute_type"], local_files_only=True,
        )

    def transcribe(self, audio):
        segments, _ = self.pipe.transcribe(audio, language=self.language, task="transcribe", word_timestamps=True, condition_on_previous_text=False)
        words, chunks = [], []
        for segment in segments:
            chunks.append({"text": segment.text, "start_s": segment.start, "end_s": segment.end})
            words += [{"text": w.word, "start_s": w.start, "end_s": w.end} for w in (segment.words or [])]
        return {"words": words, "chunks": chunks}


ENGINES = {"openvino": OpenVinoEngine, "faster-whisper": FasterWhisperEngine}
