import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Optionally, set properties like voice, rate, and volume
voices = engine.getProperty('voices')

# Select an English voice (usually index 0 or 1, you can experiment)
for voice in voices:
    if 'english' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

while True:
    texto = input("Enter text to speak (type 'exit' to quit): ")
    if texto.lower().strip() == "exit":
        print("Program ended.")
        break

    engine.say(texto)
    engine.runAndWait()
