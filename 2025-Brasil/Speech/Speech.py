import speech_recognition as sr
import ollama
import pyttsx3

def Stt():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        text = r.recognize_sphinx(audio)
        if "bye" in text.lower():
            return False, text
        
        return True, text

    except sr.UnknownValueError:
        print("I don't understand what you said.")
        return True, None
    except sr.RequestError as e:
        print(f"Connection error: {e}")
        return True, None
    except Exception as e:
        print(f"Error: {e}")
        return True, None

def start_chat(user_input):
    history = [{
        'role': 'system',
        'content': 'You are a social robot, providing short answers and always trying to keep the conversation going by asking questions to interact.'}] 
    # Chatbot role
    history.append({'role': 'user', 'content': user_input})

    # Limit history
    history = history[:1] + history[-20:]

    # Call the model
    response = ollama.chat(model='llama3:8b', messages=history)

    last_response = response['message']['content'].strip()
    
    history.append({'role': 'assistant', 'content': last_response})
    return last_response

def Tts(CB_answer):
    # Initialize 
    engine = pyttsx3.init()

    # Set properties like voice, rate, and volume
    voices = engine.getProperty('voices')

    english_voice_found = False
    for voice in voices:
        if 'english' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            english_voice_found = True
            break

    if not english_voice_found:
        print("No English voice found. Using default voice.")

    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    try:
        engine.say(CB_answer)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during TTS: {e}")

if __name__ == "__main__":
    
    while True:
        continuar, text = Stt()
        
        # Check if the recognized text is valid
        if text is not None:
            user = text

            if user.lower() in ["bye", "goodbye", "exit"]:
                cb_answer = "See you later!"
                Tts(cb_answer)
                print(cb_answer)
                break

            # Start the chat with the recognized text
            Cb_answer = start_chat(user)
            print(f"Bot: {Cb_answer}")
            Tts(Cb_answer)
            
        else:
            print("No valid input received. Please try again.")
