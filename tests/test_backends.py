import sys
import types
from pathlib import Path

import numpy as np
import pytest

from meetscribe import backends, pipeline
from meetscribe.backends import CATALOG, Catalog, Hardware, Selection, SelectionError
from meetscribe.pipeline import Models

NPU = {"id": "npu", "device": "NPU", "name": "Intel AI Boost", "type": "integrated", "cuda": None, "backends": ["openvino"]}
CPU = {"id": "cpu", "device": "CPU", "name": "Test CPU", "type": "integrated", "cuda": None, "backends": ["openvino", "faster-whisper"]}
IGPU = {"id": "igpu", "device": "GPU.0", "name": "Intel Arc iGPU", "type": "integrated", "cuda": None, "backends": ["openvino"]}
ARC = {"id": "gpu", "device": "GPU.1", "name": "Intel Arc B580", "type": "discrete", "cuda": None, "backends": ["openvino"]}
NVIDIA = {"id": "gpu", "device": "GPU.1", "name": "NVIDIA RTX 4070", "type": "discrete", "cuda": 0, "backends": ["faster-whisper"]}
INTEL_ONLY = [NPU, IGPU, CPU]


def fill(root, *dirs, files=("openvino_model.xml", "openvino_model.bin")):
    for name in dirs:
        (root / name).mkdir(parents=True, exist_ok=True)
        for file in files:
            (root / name / file).write_bytes(b"x")


def fill_ct2(root, name):
    fill(root, name, files=("model.bin", "config.json", "tokenizer.json", "vocabulary.json"))


@pytest.fixture
def models(tmp_path):
    fill(tmp_path, "whisper-large-v3-turbo-fp16-ov", "whisper-large-v3-turbo-int8-ov")
    return Models(tmp_path)


@pytest.fixture(autouse=True)
def both_installed(monkeypatch):
    monkeypatch.setattr(Catalog, "installed", staticmethod(lambda backend: True))


def select(models, rows, **kwargs):
    return Selection(models, Hardware(rows)).resolve(**kwargs)


def test_catalog_is_complete_and_keeps_todays_defaults():
    assert CATALOG["default_model"] == "large-v3-turbo"
    assert set(CATALOG["models"]) >= {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}
    for backend, spec in CATALOG["backends"].items():
        for model in CATALOG["models"]:
            variants = CATALOG["variants"][backend][model]
            assert variants and all(v["repo"].count("/") == 1 and v["mb"] > 0 for v in variants.values())
        for hardware, device in spec["devices"].items():
            assert hardware in CATALOG["hardware"] and device["default"] in device["variants"]
            assert all(v in CATALOG["variants"][backend]["small"] for v in device["variants"])
    devices = CATALOG["backends"]["openvino"]["devices"]
    assert devices["npu"]["default"] == "fp16" and devices["igpu"]["default"] == "int8" and devices["cpu"]["default"] == "int8"
    assert Catalog.entry("openvino", "large-v3-turbo", "fp16")["repo"] == "OpenVINO/whisper-large-v3-turbo-fp16-ov"
    assert CATALOG["hardware_order"] == ("npu", "gpu", "igpu", "cpu")


def fake_core(devices):
    class Core:
        available_devices = list(devices)

        def get_property(self, name, key):
            if key not in devices[name]:
                raise RuntimeError("no property")
            return devices[name][key]

    return types.SimpleNamespace(Core=Core)


def test_hardware_uses_openvino_device_type_for_discrete_and_integrated(monkeypatch):
    core = fake_core({
        "CPU": {"FULL_DEVICE_NAME": "Test CPU", "DEVICE_TYPE": "Type.INTEGRATED"},
        "GPU.0": {"FULL_DEVICE_NAME": "Intel(R) Arc(TM) 140V GPU (iGPU)", "DEVICE_TYPE": "Type.INTEGRATED", "DEVICE_ARCHITECTURE": "GPU: vendor=0x8086"},
        "GPU.1": {"FULL_DEVICE_NAME": "Intel(R) Arc(TM) B580 (dGPU)", "DEVICE_TYPE": "Type.DISCRETE", "DEVICE_ARCHITECTURE": "GPU: vendor=0x8086"},
        "GPU.2": {"FULL_DEVICE_NAME": "Intel(R) Arc(TM) A770 (dGPU)", "DEVICE_TYPE": "Type.DISCRETE", "DEVICE_ARCHITECTURE": "GPU: vendor=0x8086"},
        "NPU": {"FULL_DEVICE_NAME": "Intel(R) AI Boost", "DEVICE_TYPE": "Type.INTEGRATED"},
    })
    monkeypatch.setitem(sys.modules, "openvino", core)
    monkeypatch.setattr(Hardware, "cuda_count", staticmethod(lambda: 0))
    rows = Hardware().rows
    assert [(r["id"], r["device"]) for r in rows] == [("cpu", "CPU"), ("igpu", "GPU.0"), ("gpu", "GPU.1"), ("gpu", "GPU.2"), ("npu", "NPU")]
    assert [r["type"] for r in rows] == ["integrated", "integrated", "discrete", "discrete", "integrated"]
    assert Hardware(rows).of("gpu", "openvino")["device"] == "GPU.1" and Hardware(rows).of("gpu", "faster-whisper") is None
    assert Hardware(rows).of("cpu", "faster-whisper")["device"] == "CPU"


def test_hardware_adds_cuda_cards_and_keeps_foreign_gpus_away_from_openvino(monkeypatch):
    core = fake_core({
        "CPU": {"FULL_DEVICE_NAME": "Test CPU"},
        "GPU.0": {"FULL_DEVICE_NAME": "NVIDIA GeForce RTX 4070", "DEVICE_TYPE": "Type.DISCRETE", "DEVICE_ARCHITECTURE": "GPU: vendor=0x10de"},
    })
    monkeypatch.setitem(sys.modules, "openvino", core)
    monkeypatch.setattr(Hardware, "cuda_count", staticmethod(lambda: 2))
    rows = Hardware.detect()
    cards = [r for r in rows if r["id"] == "gpu"]
    assert [r["cuda"] for r in cards] == [0, 1] and all(r["backends"] == ["faster-whisper"] for r in cards)
    assert cards[1]["name"] == "NVIDIA-Karte 1 (CUDA)"


def test_hardware_without_openvino_still_offers_the_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino", None)
    monkeypatch.setattr(Hardware, "cuda_count", staticmethod(lambda: 0))
    rows = Hardware.detect()
    assert [r["id"] for r in rows] == ["cpu"] and rows[0]["backends"] == ["faster-whisper"]


def test_auto_prefers_npu_then_discrete_then_igpu_then_cpu(models):
    plan = select(models, INTEL_ONLY)
    assert [s.hardware for s in plan.steps] == ["npu", "igpu", "cpu"] and plan.primary.variant == "fp16" and plan.steps[1].variant == "int8"
    assert plan.model == "large-v3-turbo" and plan.requested == "Automatisch" and plan.notices == []
    assert [s.hardware for s in select(models, [IGPU, ARC, CPU]).steps] == ["gpu", "igpu", "cpu"]
    assert [s.hardware for s in select(models, [CPU]).steps] == ["cpu"]


def test_explicit_hardware_starts_the_chain_there_and_no_fallback_stops_it(models):
    assert [s.hardware for s in select(models, INTEL_ONLY, hardware="igpu").steps] == ["igpu", "cpu"]
    assert [s.hardware for s in select(models, INTEL_ONLY, hardware="cpu").steps] == ["cpu"]
    assert [s.hardware for s in select(models, INTEL_ONLY, hardware="npu", fallback=False).steps] == ["npu"]


def test_gpu_means_the_discrete_card_and_maps_to_igpu_only_without_one(models):
    plan = select(models, INTEL_ONLY, hardware="gpu")
    assert plan.primary.hardware == "igpu" and [s.hardware for s in plan.steps] == ["igpu", "cpu"]
    assert plan.notices == [CATALOG["notice_gpu"]] and plan.requested == "GPU"
    assert select(models, [IGPU, ARC, CPU], hardware="gpu").primary.row["device"] == "GPU.1"
    assert select(models, [IGPU, ARC, CPU], hardware="gpu").notices == []
    assert select(models, [IGPU, ARC, CPU], hardware="igpu").primary.row["device"] == "GPU.0"


def test_nvidia_without_a_card_is_refused_with_a_german_reason(models):
    with pytest.raises(SelectionError, match="Keine NVIDIA-Karte \\(CUDA\\) gefunden"):
        select(models, INTEL_ONLY, backend="faster-whisper", hardware="gpu")
    with pytest.raises(SelectionError, match="Keine NVIDIA-Karte"):
        select(models, [IGPU, ARC, CPU], backend="faster-whisper", hardware="gpu")


def test_faster_whisper_serves_cpu_and_cuda_only(models):
    fill_ct2(models.root, "faster-whisper-large-v3-turbo")
    plan = select(models, INTEL_ONLY, backend="faster-whisper", hardware="cpu")
    assert plan.primary.backend == "faster-whisper" and plan.primary.options == {"compute_type": "int8"} and plan.primary.installed
    for kind in ("npu", "igpu"):
        with pytest.raises(SelectionError, match="nur auf der CPU und auf NVIDIA-Karten"):
            select(models, INTEL_ONLY, backend="faster-whisper", hardware=kind)
    cuda = select(models, [CPU, NVIDIA], backend="faster-whisper", hardware="gpu")
    assert cuda.primary.options == {"compute_type": "float16"} and cuda.primary.row["cuda"] == 0
    assert [s.hardware for s in cuda.steps] == ["gpu", "cpu"]
    assert select(models, [CPU, NVIDIA], backend="auto", hardware="gpu").primary.backend == "faster-whisper"


def test_backend_not_installed_is_refused_with_the_install_hint(models, monkeypatch):
    monkeypatch.setattr(Catalog, "installed", staticmethod(lambda backend: backend == "openvino"))
    with pytest.raises(SelectionError, match="faster-whisper ist nicht installiert. Installieren mit: uv sync --extra faster-whisper"):
        select(models, INTEL_ONLY, backend="faster-whisper", hardware="cpu")
    monkeypatch.setattr(Catalog, "installed", staticmethod(lambda backend: False))
    with pytest.raises(SelectionError, match="OpenVINO ist nicht installiert"):
        select(models, INTEL_ONLY)


def test_int8_on_the_npu_is_refused_but_fp16_on_the_igpu_is_not(models):
    with pytest.raises(SelectionError, match="int8 läuft auf der NPU nicht: ZE_RESULT_ERROR_DEVICE_LOST gemessen"):
        select(models, INTEL_ONLY, hardware="npu", model="large-v3-turbo:int8")
    assert select(models, INTEL_ONLY, hardware="igpu", model="large-v3-turbo:fp16").primary.variant == "fp16"
    assert select(models, INTEL_ONLY, hardware="igpu", model="large-v3-turbo:int8").primary.variant == "int8"
    with pytest.raises(SelectionError, match="Unbekannte Variante 'int4'"):
        select(models, INTEL_ONLY, model="large-v3-turbo:int4")


def test_npu_support_of_other_sizes_is_unverified_unless_measured(models, monkeypatch):
    assert select(models, INTEL_ONLY, hardware="npu").primary.status == "ok"
    plan = select(models, INTEL_ONLY, hardware="npu", model="medium")
    assert plan.primary.status == "unverified" and "nicht geprüft" in plan.primary.reason
    assert select(models, INTEL_ONLY, hardware="igpu", model="medium").primary.status == "ok"
    monkeypatch.setitem(CATALOG["variants"]["openvino"]["medium"]["fp16"], "npu", "fail")
    with pytest.raises(SelectionError, match="medium \\(fp16\\) lässt sich auf der NPU nicht übersetzen"):
        select(models, INTEL_ONLY, hardware="npu", model="medium")
    assert [s.hardware for s in select(models, INTEL_ONLY, model="medium").steps][0] == "igpu"


def test_auto_skips_hardware_whose_model_is_missing_and_says_so(models):
    for file in (models.root / "whisper-large-v3-turbo-fp16-ov").iterdir():
        file.unlink()
    plan = select(models, INTEL_ONLY)
    assert plan.primary.hardware == "igpu" and plan.installed
    assert plan.notices == ["NPU mit OpenVINO übersprungen: Modell large-v3-turbo (fp16) ist nicht installiert."]
    explicit = select(models, INTEL_ONLY, hardware="npu")
    assert not explicit.installed and explicit.primary.path.name == "whisper-large-v3-turbo-fp16-ov"


def test_missing_model_everywhere_gives_a_plan_that_preflight_can_refuse(models):
    plan = select(models, INTEL_ONLY, model="small")
    assert not plan.installed and plan.chain == [] and plan.primary.hardware == "npu"


def test_model_names_variants_and_overrides(models, tmp_path):
    assert select(models, INTEL_ONLY, model="auto").model == "large-v3-turbo"
    with pytest.raises(SelectionError, match="Unbekanntes Modell 'riesig'"):
        select(models, INTEL_ONLY, model="riesig")
    custom = tmp_path / "eigenes-modell"
    fill(tmp_path, "eigenes-modell")
    plan = select(models, INTEL_ONLY, model=str(custom))
    assert plan.model == "eigenes-modell" and plan.primary.path == custom and plan.primary.installed and plan.primary.variant == "custom"
    assert select(models, INTEL_ONLY, model="eigenes-modell").primary.path == custom


def test_unknown_backend_or_hardware_is_refused(models):
    with pytest.raises(SelectionError, match="Unbekanntes Backend 'x'"):
        select(models, INTEL_ONLY, backend="x")
    with pytest.raises(SelectionError, match="Unbekannte Hardware 'tpu'"):
        select(models, INTEL_ONLY, hardware="tpu")


def test_matrix_covers_every_combination_with_reasons(models):
    rows = Selection(models, Hardware(INTEL_ONLY)).matrix()
    assert len(rows) == 3 * 5 * 7
    by = {(r["backend"], r["hardware"], r["model"]): r for r in rows}
    auto = by[("auto", "auto", "auto")]
    assert auto["ok"] and auto["resolved"]["hardware"] == "npu" and auto["installed"] and auto["resolved"]["backend"] == "openvino"
    assert by[("openvino", "gpu", "small")]["resolved"]["hardware"] == "igpu" and by[("openvino", "gpu", "small")]["notices"]
    refused = by[("faster-whisper", "npu", "small")]
    assert not refused["ok"] and "nur auf der CPU" in refused["reason"] and refused["resolved"] is None
    assert not by[("faster-whisper", "gpu", "small")]["ok"] and by[("openvino", "npu", "medium")]["status"] == "unverified" and by[("openvino", "npu", "small")]["status"] == "ok"
    assert by[("openvino", "cpu", "small")]["installed"] is False and by[("openvino", "cpu", "small")]["size_mb"] == 257
    assert by[("openvino", "npu", "large-v3-turbo")]["size_mb"] == 1628


def test_listing_marks_install_state_and_per_hardware_status(models):
    rows = Selection(models, Hardware(INTEL_ONLY)).listing()
    turbo = next(r for r in rows if (r["backend"], r["model"], r["variant"]) == ("openvino", "large-v3-turbo", "int8"))
    assert turbo["installed"] and turbo["size_mb"] == 828 and turbo["hardware"]["npu"]["status"] == "refused" and "int8" in turbo["hardware"]["npu"]["reason"]
    assert turbo["hardware"]["gpu"]["status"] == "refused" and "dedizierte" in turbo["hardware"]["gpu"]["reason"]
    small = next(r for r in rows if (r["backend"], r["model"], r["variant"]) == ("openvino", "small", "fp16"))
    assert not small["installed"] and small["hardware"]["npu"]["status"] == "ok" and small["hardware"]["cpu"]["status"] == "ok"
    medium = next(r for r in rows if (r["backend"], r["model"], r["variant"]) == ("openvino", "medium", "fp16"))
    assert medium["hardware"]["npu"]["status"] == "unverified"
    assert {r["backend"] for r in rows} == {"openvino", "faster-whisper"} and len(rows) == 12 + 6


def install_fake_module(monkeypatch, name, **attrs):
    module = types.SimpleNamespace(**attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_openvino_engine_passes_the_real_device_name_and_cache(models, monkeypatch):
    seen = {}

    class Pipe:
        def __init__(self, path, device, **kwargs):
            seen.update(path=path, device=device, kwargs=kwargs)

        def generate(self, audio, **kwargs):
            seen["generate"] = kwargs
            word = types.SimpleNamespace(word=" Hallo", start_ts=0.0, end_ts=0.5)
            chunk = types.SimpleNamespace(text=" Hallo", start_ts=0.0, end_ts=1.0)
            return types.SimpleNamespace(words=[word], chunks=[chunk])

    install_fake_module(monkeypatch, "openvino_genai", WhisperPipeline=Pipe)
    plan = select(models, [IGPU, ARC, CPU], hardware="gpu")
    engine = backends.ENGINES["openvino"](plan.primary, models, "de")
    assert seen["device"] == "GPU.1" and seen["kwargs"]["CACHE_DIR"] == str(models.cache_dir) and seen["kwargs"]["word_timestamps"] is True
    assert engine.device == "GPU" and engine.backend == "openvino" and engine.model == "whisper-large-v3-turbo-int8-ov"
    result = engine.transcribe(np.zeros(16000, dtype="float32"))
    assert result == {"words": [{"text": " Hallo", "start_s": 0.0, "end_s": 0.5}], "chunks": [{"text": " Hallo", "start_s": 0.0, "end_s": 1.0}]}
    assert seen["generate"]["language"] == "<|de|>"


def test_faster_whisper_engine_yields_the_same_words_and_chunks_from_local_folders_only(models, monkeypatch):
    fill_ct2(models.root, "faster-whisper-large-v3-turbo")
    seen = {}

    class Model:
        def __init__(self, path, **kwargs):
            seen.update(path=path, kwargs=kwargs)

        def transcribe(self, audio, **kwargs):
            seen["transcribe"] = kwargs
            words = [types.SimpleNamespace(word=" Guten", start=0.1, end=0.4), types.SimpleNamespace(word=" Tag.", start=0.4, end=0.9)]
            return iter([types.SimpleNamespace(text=" Guten Tag.", start=0.0, end=1.0, words=words), types.SimpleNamespace(text=" Stille", start=2.0, end=3.0, words=None)]), None

    install_fake_module(monkeypatch, "faster_whisper", WhisperModel=Model)
    plan = select(models, [CPU, NVIDIA], backend="faster-whisper", hardware="gpu")
    engine = backends.ENGINES["faster-whisper"](plan.primary, models, "de")
    assert seen["kwargs"] == {"device": "cuda", "device_index": 0, "compute_type": "float16", "local_files_only": True}
    assert Path(seen["path"]).name == "faster-whisper-large-v3-turbo" and engine.model == "faster-whisper-large-v3-turbo-float16"
    result = engine.transcribe(np.zeros(16000, dtype="float32"))
    assert result["words"] == [{"text": " Guten", "start_s": 0.1, "end_s": 0.4}, {"text": " Tag.", "start_s": 0.4, "end_s": 0.9}]
    assert result["chunks"][1] == {"text": " Stille", "start_s": 2.0, "end_s": 3.0}
    assert seen["transcribe"]["word_timestamps"] is True and seen["transcribe"]["language"] == "de"
    cpu = backends.ENGINES["faster-whisper"](select(models, [CPU], backend="faster-whisper", hardware="cpu").primary, models, "de")
    assert seen["kwargs"]["device"] == "cpu" and seen["kwargs"]["compute_type"] == "int8" and cpu.device == "CPU"


def test_faster_whisper_words_flow_through_the_shared_transcriber(models, monkeypatch, tmp_path):
    from meetscribe.store import Query, Store, make_config
    fill_ct2(models.root, "faster-whisper-large-v3-turbo")

    class Model:
        def __init__(self, path, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            words = [types.SimpleNamespace(word=" Guten", start=0.1, end=0.4), types.SimpleNamespace(word=" Tag.", start=0.4, end=0.9)]
            return iter([types.SimpleNamespace(text=" Guten Tag.", start=0.0, end=1.0, words=words)]), None

    install_fake_module(monkeypatch, "faster_whisper", WhisperModel=Model)
    plan = select(models, [CPU], backend="faster-whisper", hardware="cpu")
    runner = pipeline.ASRRunner(plan.chain, lambda s: backends.ENGINES[s.backend](s, models, "de"))
    store = Store(make_config(tmp_path / "data"))
    audio = np.full(16000 * 2, 0.1, dtype="float32")
    with store.writer("m", title="t", meeting_date="2026-01-01", language="de") as writer:
        summary = pipeline.Transcriber(runner).run(audio, [], writer)
    assert summary["segments"] == 1 and summary["used"] == ["CPU"] and summary["engines"] == [("faster-whisper", "faster-whisper-large-v3-turbo-int8")]
    row = Query(store).execute("SELECT text, device, model, backend, words FROM segments")[1][0]
    assert row[0] == "Guten Tag." and row[1:4] == ("CPU", "faster-whisper-large-v3-turbo-int8", "faster-whisper")
    assert [w["text"] for w in row[4]] == ["Guten", "Tag."] and row[4][0]["start_s"] == pytest.approx(0.1)


def test_ct2_folder_readiness(tmp_path):
    engine = backends.FasterWhisperEngine
    assert not engine.ready(tmp_path / "none")
    fill_ct2(tmp_path, "c")
    assert engine.ready(tmp_path / "c")
    (tmp_path / "c" / "model.bin").write_bytes(b"")
    assert not engine.ready(tmp_path / "c")
    fill(tmp_path, "d", files=("model.bin", "config.json", "tokenizer.json"))
    assert not engine.ready(tmp_path / "d")


def test_targets_pick_the_variants_a_model_needs(models):
    job = pipeline.ModelSetup(models)
    assert job.targets(None, None, None) == ("openvino", "large-v3-turbo", ["fp16", "int8"])
    assert job.targets("small", None, "cpu") == ("openvino", "small", ["int8"])
    assert job.targets("small", "openvino", "npu") == ("openvino", "small", ["fp16"])
    assert job.targets("small:fp16", None, "cpu") == ("openvino", "small", ["fp16"])
    assert job.targets("small", "faster-whisper", None) == ("faster-whisper", "small", ["ct2"])
    for bad, text in ((("riesig", None, None), "Unbekanntes Modell"), (("small", "x", None), "Unbekanntes Backend"), (("small:int4", None, None), "Unbekannte Variante")):
        with pytest.raises(SelectionError, match=text):
            job.targets(*bad)


class Recorder:
    machine = False
    broken = False

    def __init__(self):
        self.events = []

    def event(self, name, **data):
        self.events.append((name, data))

    def warn(self, text, prefix=None):
        pass


def test_download_reports_progress_and_checks_the_result(models, monkeypatch):
    reporter = Recorder()
    job = pipeline.ModelSetup(models, reporter=reporter)
    monkeypatch.setitem(pipeline.SETTINGS, "download_poll_s", 0.01)

    def download(repo, target, token=None):
        import time
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "part.bin").write_bytes(b"x" * 1000)
        time.sleep(0.1)
        fill(Path(target).parent, Path(target).name, files=("model.bin", "config.json", "tokenizer.json", "vocabulary.json"))

    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(download))
    messages, failed = job.run("small", "faster-whisper", "cpu")
    assert not failed and messages == [f"whisper Systran/faster-whisper-small: ok ({models.root / 'faster-whisper-small'})"]
    sizes = [d["bytes"] for n, d in reporter.events if n == "download"]
    assert sizes and sizes == sorted(sizes) and sizes[-1] > 1000 and all(d["total"] == 486_000_000 for n, d in reporter.events)
    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(lambda repo, target, token=None: None))
    messages, failed = job.run("tiny", "faster-whisper", "cpu")
    assert failed and "unvollständig" in messages[0]
    monkeypatch.setattr(pipeline.ModelSetup, "download", staticmethod(lambda repo, target, token=None: (_ for _ in ()).throw(OSError("offline"))))
    assert job.run("tiny", "faster-whisper", "cpu")[1]
