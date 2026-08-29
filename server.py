import os
import io
import time
import threading

import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from kokoro import KModel, KPipeline


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "kokoro-v1_0.pth"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "config.json"
)

VOICE_DIR = os.path.join(
    BASE_DIR,
    "voices"
)

DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

# Render CPU usually doesn't need a huge thread count.
# Can be overridden with an environment variable.
THREADS = int(
    os.environ.get(
        "TORCH_NUM_THREADS",
        "2"
    )
)

torch.set_num_threads(THREADS)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Jonah Kokoro TTS",
    version="1.0.0"
)


# ============================================================
# GLOBAL MODEL
# ============================================================

MODEL = None
PIPELINE = None
VOICE_CACHE = {}

MODEL_LOCK = threading.Lock()


# ============================================================
# REQUEST MODEL
# ============================================================

class TTSRequest(BaseModel):

    text: str

    voice: str = DEFAULT_VOICE

    speed: float = 1.0


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    global MODEL
    global PIPELINE

    if MODEL is not None:
        return

    print("=" * 60)
    print("[KOKORO] Starting model initialization")
    print("=" * 60)

    print(
        "[KOKORO] Model:",
        MODEL_PATH
    )

    print(
        "[KOKORO] Config:",
        CONFIG_PATH
    )

    print(
        "[KOKORO] Voices:",
        VOICE_DIR
    )

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(
            f"Missing Kokoro model: {MODEL_PATH}"
        )

    if not os.path.isfile(CONFIG_PATH):
        raise RuntimeError(
            f"Missing Kokoro config: {CONFIG_PATH}"
        )

    voice_path = os.path.join(
        VOICE_DIR,
        f"{DEFAULT_VOICE}.pt"
    )

    if not os.path.isfile(voice_path):
        raise RuntimeError(
            f"Missing default voice: {voice_path}"
        )

    start = time.time()

    print(
        "[KOKORO] Loading PyTorch model..."
    )

    MODEL = (
        KModel(
            config=CONFIG_PATH,
            model=MODEL_PATH
        )
        .to("cpu")
        .eval()
    )

    print(
        "[KOKORO] Model loaded in",
        round(time.time() - start, 2),
        "seconds"
    )

    print(
        "[KOKORO] Creating English pipeline..."
    )

    PIPELINE = KPipeline(
        lang_code="a",
        model=MODEL,
        device="cpu"
    )

    print(
        "[KOKORO] Loading voice:",
        DEFAULT_VOICE
    )

    PIPELINE.load_single_voice(
        voice_path
    )

    VOICE_CACHE[
        DEFAULT_VOICE
    ] = PIPELINE.load_voice(
        voice_path
    )

    print(
        "[KOKORO] Voice loaded"
    )

    print("=" * 60)
    print("[KOKORO] READY")
    print("=" * 60)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    try:

        load_model()

    except Exception as error:

        print(
            "[KOKORO] STARTUP FAILED:"
        )

        print(error)

        raise


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Jonah Kokoro TTS",
        "status": "ok",
        "model_loaded": MODEL is not None,
        "voice": DEFAULT_VOICE
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "voice_loaded": DEFAULT_VOICE in VOICE_CACHE
    }


# ============================================================
# VOICE LOADER
# ============================================================

def get_voice(voice_name):

    if voice_name in VOICE_CACHE:
        return VOICE_CACHE[voice_name]

    voice_path = os.path.join(
        VOICE_DIR,
        f"{voice_name}.pt"
    )

    if not os.path.isfile(voice_path):

        raise ValueError(
            f"Voice not found: {voice_name}"
        )

    print(
        "[KOKORO] Loading voice:",
        voice_name
    )

    voice = PIPELINE.load_single_voice(
        voice_path
    )

    VOICE_CACHE[
        voice_name
    ] = voice

    return voice


# ============================================================
# TTS
# ============================================================

@app.post("/tts")
def generate_tts(request: TTSRequest):

    if not request.text or not request.text.strip():

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    text = request.text.strip()

    voice_name = (
        request.voice
        or DEFAULT_VOICE
    )

    speed = float(
        request.speed
        or 1.0
    )

    if speed <= 0:
        speed = 1.0

    if speed < 0.5:
        speed = 0.5

    if speed > 2.0:
        speed = 2.0

    print(
        "[KOKORO] Request:",
        text[:100]
    )

    print(
        "[KOKORO] Voice:",
        voice_name
    )

    print(
        "[KOKORO] Speed:",
        speed
    )

    start = time.time()

    try:

        with MODEL_LOCK:

            voice = get_voice(
                voice_name
            )

            results = []

            generator = PIPELINE(
                text,
                voice=voice,
                speed=speed
            )

            for result in generator:

                audio = result.audio

                if audio is None:
                    continue

                results.append(
                    audio.detach()
                    .cpu()
                    .numpy()
                )

            if not results:

                raise RuntimeError(
                    "Kokoro returned no audio"
                )

            import numpy as np

            final_audio = np.concatenate(
                results
            )

            wav_buffer = io.BytesIO()

            sf.write(
                wav_buffer,
                final_audio,
                SAMPLE_RATE,
                format="WAV",
                subtype="PCM_16"
            )

            wav_bytes = (
                wav_buffer
                .getvalue()
            )

        elapsed = (
            time.time() - start
        )

        print(
            "[KOKORO] Generated:",
            len(final_audio),
            "samples"
        )

        print(
            "[KOKORO] Generation time:",
            round(elapsed, 2),
            "seconds"
        )

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Kokoro-Sample-Rate":
                    str(SAMPLE_RATE),

                "X-Kokoro-Generation-Time":
                    str(round(elapsed, 3))
            }
        )

    except Exception as error:

        print(
            "[KOKORO] GENERATION FAILED:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# PRELOAD
# ============================================================

@app.post("/warmup")
def warmup():

    load_model()

    return {
        "success": True,
        "message": "Kokoro is warm"
    }
