import os, time, wave, tempfile, threading
import numpy as np
import sounddevice as sd
import requests
import torch
import faiss
from sentence_transformers import SentenceTransformer
import whisper
from glob import glob
from pathlib import Path

from fractions import Fraction
from scipy.signal import resample_poly
from openwakeword.model import Model as WakeModel

import math
from collections import deque

# -------------------- Config --------------------
INPUT_DEV = 0                # your USB mic index from sd.query_devices()
WAKE_THRESHOLD = 0.6         # raise/lower to change sensitivity
WAKE_COOLDOWN_S = 2.0        # ignore retriggers for this long
RECORD_SECONDS = 5
GPU_LAYERS = 30              # (unused here; for llama-server startup only)

OWW_DIR = Path.home() / ".cache" / "openwakeword" / "models"
wake_models = glob(str(OWW_DIR / "*.tflite"))

# LLaMA server (legacy /completion endpoint)
LLAMA_URL = "http://127.0.0.1:8080/completion"

initial_prompt = (
    "You're an AI assistant specialized in AI development, embedded systems like the Jetson Nano, and Google technologies. "
    "Answer questions clearly and concisely in a friendly, professional tone. Do not use asterisks, do not ask new questions "
    "or act as the user. Keep replies short to speed up inference. If unsure, admit it and suggest looking into it further."
)

current_dir = os.path.dirname(os.path.abspath(__file__))
bip_sound = os.path.join(current_dir, "assets/bip.wav")
bip2_sound = os.path.join(current_dir, "assets/bip2.wav")

# -------------------- Models --------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("base").to(device)

# Wake word model: default bundles include phrases like “hey jarvis”, “alexa”, etc.
# You can pass a custom list of ONNX models via wakeword_models=[...]
#oww = WakeModel(wakeword_models=wake_models)
WAKE_THRESHOLD = 0.15
oww = WakeModel()
print("Loaded wake models:", list(getattr(oww, "models", {}).keys()))

# -------------------- Mini-DB (FAISS) --------------------
docs = [
    "The Jetson Nano is a compact, powerful computer designed by NVIDIA for AI applications at the edge.",
    "Developers can create AI assistants in under 100 lines of Python code using open-source libraries.",
    "Retrieval Augmented Generation enhances AI responses by combining language models with external knowledge bases.",
]

class VectorDatabase:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
    def add_documents(self, ds):
        embeddings = embedding_model.encode(ds)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(ds)
    def search(self, query, top_k=3):
        q = embedding_model.encode([query])[0].astype(np.float32)
        _, idx = self.index.search(np.array([q]), top_k)
        return [self.documents[i] for i in idx[0]]

db = VectorDatabase()
db.add_documents(docs)

# -------------------- Utils --------------------
def play_sound(path):
    os.system(f"aplay {path}")

def record_audio_to_wav(filename, seconds=RECORD_SECONDS):
    dinfo = sd.query_devices(INPUT_DEV)
    fs = int(dinfo["default_samplerate"])   # e.g., 44100 or 48000
    play_sound(bip_sound)
    print(f"Recording {seconds}s from '{dinfo['name']}' (idx {INPUT_DEV}) at {fs} Hz ...")

    audio = sd.rec(int(seconds * fs),
                   samplerate=fs,
                   channels=1,
                   dtype='float32',
                   device=INPUT_DEV)
    sd.wait()

    # quick level check
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    print(f"Recorded RMS level: {rms:.1f}")
    if rms < 20:
        print("Warning: very low level — mic muted or wrong device?")

    # Save as 16-bit PCM; Whisper/ffmpeg will resample as needed
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes())

    play_sound(bip2_sound)
    print("Recording completed")

def transcribe_audio(path):
    return whisper_model.transcribe(
        path,
        language="en",
        fp16=(device.type == "cuda")
    )['text']

def ask_llama(query, context):
    data = {
        "prompt": f"{initial_prompt}\nContext: {context}\nQuestion: {query}\nAnswer:",
        "n_predict": 80,        # llama.cpp expects n_predict in /completion mode
        "temperature": 0.7
    }
    r = requests.post(LLAMA_URL, json=data, headers={'Content-Type': 'application/json'})
    r.raise_for_status()
    return r.json().get('content', '').strip()

def rag_ask(q):
    ctx = " ".join(db.search(q))
    return ask_llama(q, ctx)

def speak(text):
    # escape double quotes in the response so `echo` doesn’t break
    safe_text = text.replace('"', r'\"')
    cmd = (
        f'echo "{safe_text}" | '
        f'/home/jetson/Documents/Github/piper/build/piper '
        f'--model /usr/local/share/piper/models/en_US-lessac-medium.onnx '
        f'--output_file response.wav && aplay response.wav'
    )
    os.system(cmd)


# -------------------- Wake Word Listener --------------------
wake_event = threading.Event()
last_trigger = 0.0
stream = None
STREAM_SR = None             # actual device sample rate (set at start)
OWW_SR = 16000               # openwakeword expects 16 kHz
DBG_EVERY_N_FRAMES = 15      # ~0.45s at 30ms frames
_dbg_counter = 0
_dbg_hist = deque(maxlen=30) # rolling debug of top probs

def _to_16k(x, sr_in):
    if sr_in == OWW_SR:
        return x
    # Compute rational resample ratio (e.g., 48k -> 16k : up=1, down=3; 44.1k -> 16k : up=320, down=882)
    r = Fraction(OWW_SR, int(sr_in)).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator).astype(np.float32)

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64)**2)))

def wake_callback(indata, frames, time_info, status):
    global last_trigger, _dbg_counter
    if status:
        # print(status)  # uncomment for XRUN debug
        pass

    mono = indata[:, 0].astype(np.float32)
    mono16 = _to_16k(mono, STREAM_SR)

    # Optional gain safety (helps very quiet mics without blowing up noise):
    # target_rms = 0.05
    # r = _rms(mono16)
    # if r > 1e-6 and r < target_rms:
    #     mono16 = mono16 * (target_rms / r)

    scores = oww.predict(mono16)  # dict: {wake_name: probability}

    # Debug print every ~0.45s
    _dbg_counter += 1
    if _dbg_counter % DBG_EVERY_N_FRAMES == 0:
        top = max(scores.items(), key=lambda kv: kv[1])
        rms16 = _rms(mono16)
        _dbg_hist.append(top[1])
        print(f"[lvl] rms16={rms16:.4f} | top={top[0]} {top[1]:.2f} | recent max={max(_dbg_hist, default=0):.2f}")

    now = time.time()
    for name, p in scores.items():
        if p >= WAKE_THRESHOLD and (now - last_trigger) > WAKE_COOLDOWN_S:
            last_trigger = now
            print(f"[wake] {name} ({p:.2f})")
            wake_event.set()
            break

def start_listener():
    global STREAM_SR
    dinfo = sd.query_devices(INPUT_DEV)
    preferred = [16000, int(dinfo.get("default_samplerate", 16000))]
    err = None
    for sr in preferred:
        try:
            print(f"Trying input stream on '{dinfo['name']}' @ {sr} Hz (device {INPUT_DEV})")
            s = sd.InputStream(
                device=INPUT_DEV,
                channels=1,
                samplerate=sr,
                dtype='float32',
                blocksize=int(sr * 0.03),  # ~30 ms frames
                callback=wake_callback,
                latency='low'
            )
            s.start()
            STREAM_SR = sr
            print(f"Wake listener active @ {sr} Hz; resampling -> {OWW_SR} Hz for OWW")
            return s
        except Exception as e:
            err = e
            print(f"Stream @ {sr} Hz failed ({e}); trying next…")
    raise RuntimeError(f"Could not open input stream at supported rates. Last error: {err}")


# -------------------- Main --------------------
def main():
    global stream
    # Start passive wake-word listener
    stream = start_listener()
    print("Say your wake word (e.g., 'hey jarvis'). Ctrl+C to exit.")

    try:
        while True:
            # Wait until the wake word triggers
            wake_event.wait()
            # Stop listener during active capture to avoid device contention
            try:
                stream.stop()
            except Exception:
                pass

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                record_audio_to_wav(tmp.name)
                text = transcribe_audio(tmp.name)
                print(f"Agent heard: {text}")
                if text.strip():
                    reply = rag_ask(text)
                    print(f"Agent response: {reply}")
                    if reply:
                        speak(reply)
                else:
                    print("No speech detected.")

            # Reset and resume wake listening
            wake_event.clear()
            try:
                stream.start()
            except Exception:
                # If the old stream can't be restarted, rebuild it
                try:
                    stream.close()
                except Exception:
                    pass
                time.sleep(0.1)
                stream = start_listener()

    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        try:
            stream.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
