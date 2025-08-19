from vosk import Model, KaldiRecognizer
import pyaudio
import json

model = Model("C:/Users/Usuario/OneDrive/Escritorio/Phyton/Robocup/vosk-model-small-es-0.42/vosk-model-small-es-0.42")  # Modelo del idioma a trabajar
rec = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=8192)
stream.start_stream()

print("Di algo")

while True:
    data = stream.read(4096, exception_on_overflow=False)
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        print("Texto:", result.get("text", ""))
