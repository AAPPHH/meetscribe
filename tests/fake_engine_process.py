import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from meetscribe import backends, cli, pipeline  # noqa: E402
from test_cli import DJI, HARDWARE_ROWS, LAPTOP, WORDS  # noqa: E402


class SleepingEngine(backends.OpenVinoEngine):
    def __init__(self, step, models, language):
        self.device, self.model = step.label, "fake-model"

    def transcribe(self, audio):
        print("Rauschen einer Bibliothek")
        os.write(1, b"Rauschen aus nativem Code\n")
        time.sleep(float(os.environ.get("FAKE_SLEEP_S", "0")))
        return {"words": [{"text": t, "start_s": s, "end_s": e} for t, s, e in WORDS], "chunks": []}


class QuietRecorder:
    def __init__(self, device, path, hostapis=()):
        self.device, self.path, self.rate, self.hostapi = device, path, 16000, hostapis[device["hostapi"]]

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.path, np.full(16000, 0.1, dtype="float32"), 16000)

    def problem(self):
        return None

    def stop(self):
        return {"path": str(self.path), "seconds": 1.0, "overruns": 0, "error": None}


class NoDiarizer:
    def __init__(self, models):
        pass

    def diarize(self, audio, **options):
        return []


def install():
    backends.ENGINES["openvino"] = SleepingEngine
    backends.Hardware.detect = classmethod(lambda cls: [dict(r, backends=list(r["backends"])) for r in HARDWARE_ROWS])
    backends.Catalog.installed = staticmethod(lambda backend: backend == "openvino")
    pipeline.Diarizer = NoDiarizer
    pipeline.Recorder = QuietRecorder
    pipeline.AudioDevices.sounddevice_picker = staticmethod(lambda refresh=True: pipeline.InputPicker([LAPTOP, DJI], ["MME"]))
    pipeline.shutil.which = lambda name: "ffmpeg"
    pipeline.AudioLoader.load = staticmethod(lambda path, rate=None: np.full(48000, 0.1, dtype="float32"))
    pipeline.START["heartbeat_s"] = 0.1
    pipeline.START["abort_grace_s"] = float(os.environ.get("FAKE_GRACE_S", "5"))


if __name__ == "__main__":
    install()
    sys.argv = ["meetscribe", "serve"]
    cli.main()
