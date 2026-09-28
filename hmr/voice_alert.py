import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)


def speak(message):
    print(f">>> VOICE: {message}")
    engine.say(message)
    engine.runAndWait()


if __name__ == "__main__":
    speak("DONUTS voice alert system is working")
