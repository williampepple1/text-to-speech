import pyttsx3
import platform

def text_to_speech(text: str, output_path: str):
    engine = pyttsx3.init()

    # Check the platform
    try:
        if platform.system() == "Windows":
            engine.setProperty('voice', 'sapi5')
        elif platform.system() == "Linux":
            engine.setProperty('voice', 'espeak')
        else:
            raise Exception("Unsupported platform")
        
        engine.save_to_file(text, output_path)
        engine.runAndWait()

    except Exception as e:
        print(f"Error in TTS: {e}")
        raise
