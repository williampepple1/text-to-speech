import pyttsx3
import platform

def text_to_speech(text: str, output_path: str):
    engine = pyttsx3.init()

    # Check the platform
    if platform.system() == "Windows":
        engine.setProperty('voice', 'sapi5')
    elif platform.system() == "Linux":
        engine.setProperty('voice', 'espeak')

    engine.save_to_file(text, output_path)
    engine.runAndWait()
