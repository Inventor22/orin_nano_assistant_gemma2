import whisper, requests, os, sounddevice as sd, numpy as np, tempfile, wave
import faiss
from sentence_transformers import SentenceTransformer
import torch

# Optimization: Use a more efficient embedding model for Jetson Orin Nano
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Optimization: Explicitly use CUDA if available, with fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("base").to(device)

# Configuration for local LLM server
llama_url = "http://127.0.0.1:8080/completion"

# Initial prompt to guide the LLaMA model's behavior
initial_prompt = ("You're an AI assistant specialized in AI development, embedded systems like the Jetson Nano, and Google technologies. "
                  "Answer questions clearly and concisely in a friendly, professional tone. Do not use asterisks, do not ask new questions "
                  "or act as the user. Keep replies short to speed up inference. If unsure, admit it and suggest looking into it further.")

# Current directory and path for beep sound files (used to indicate recording start and end)
current_dir = os.path.dirname(os.path.abspath(__file__))
bip_sound = os.path.join(current_dir, "assets/bip.wav")
bip2_sound = os.path.join(current_dir, "assets/bip2.wav")

# Documents to be used in Retrieval-Augmented Generation (RAG)
docs = [
    "The Jetson Nano is a compact, powerful computer designed by NVIDIA for AI applications at the edge.",
    "Developers can create AI assistants in under 100 lines of Python code using open-source libraries.",
    "Retrieval Augmented Generation enhances AI responses by combining language models with external knowledge bases.",
]

# Vector Database class to handle document embedding and search using FAISS
class VectorDatabase:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
    def add_documents(self, docs):
        embeddings = embedding_model.encode(docs)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(docs)
    def search(self, query, top_k=3):
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.documents[i] for i in indices[0]]

# Create a VectorDatabase and add documents to it
db = VectorDatabase(dim=384)
db.add_documents(docs)

# Play sound (beep) to signal recording start/stop
def play_sound(sound_file):
    os.system(f"aplay {sound_file}")

# --- UPDATED: Record at device default rate; let Whisper resample from file ---
INPUT_DEV = 0  # your USB mic index from sd.query_devices()

def record_audio(filename, duration=5):
    dinfo = sd.query_devices(INPUT_DEV)
    fs = int(dinfo["default_samplerate"])  # e.g., 44100 or 48000
    play_sound(bip_sound)
    print(f"Recording {duration}s from '{dinfo['name']}' (idx {INPUT_DEV}) at {fs} Hz ...")
    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='float32',   # float32 plays nicest with Whisper/FFmpeg
        device=INPUT_DEV
    )
    sd.wait()

    # Basic level check to catch silence
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    print(f"Recorded RMS level: {rms:.1f}")

    # Save exactly as recorded (Whisper/ffmpeg will resample internally)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # we'll convert to int16 for the file
        wf.setframerate(fs)
        wf.writeframes(np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes())

    play_sound(bip2_sound)
    print("Recording completed")

# Transcribe recorded audio to text using Whisper (it will resample internally)
def transcribe_audio(filename):
    return whisper_model.transcribe(
        filename,
        language="en",
        fp16=(device.type == "cuda")
    )['text']

# Send a query and context to LLaMA server for completion
def ask_llama(query, context):
    data = {
        "prompt": f"{initial_prompt}\nContext: {context}\nQuestion: {query}\nAnswer:",
        "max_tokens": 80,
        "temperature": 0.7
    }
    response = requests.post(llama_url, json=data, headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        return response.json().get('content', '').strip()
    else:
        return f"Error: {response.status_code}"

# Generate a response using Retrieval-Augmented Generation (RAG)
def rag_ask(query):
    context = " ".join(db.search(query))
    return ask_llama(query, context)

# Convert text to speech using Piper TTS model
def text_to_speech(text):
    os.system(f'echo "{text}" | /home/jetson/Documents/Github/piper/build/piper --model /usr/local/share/piper/models/en_US-lessac-medium.onnx --output_file response.wav && aplay response.wav')

# Main loop for the assistant
def main():
    while True:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            record_audio(tmpfile.name)
            transcribed_text = transcribe_audio(tmpfile.name)
            print(f"Agent heard: {transcribed_text}")
            response = rag_ask(transcribed_text)
            print(f"Agent response: {response}")
            if response:
                text_to_speech(response)

# Entry point of the script
if __name__ == "__main__":
    main()
