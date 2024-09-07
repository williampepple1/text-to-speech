import pyttsx3

# Initialize pyttsx3 engine
engine = pyttsx3.init()

def text_to_speech(text: str, output_path: str):
    # Convert text to speech and save it to an audio file
    engine.save_to_file(text, output_path)
    engine.runAndWait()
