from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid
import os
import time
from fastapi.responses import FileResponse
import database
from fastapi.middleware.cors import CORSMiddleware
import pyttsx3

import models, tts

app = FastAPI()

origins = [
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize the database
database.init_db()

class TextInput(BaseModel):
    text: str

# Dependency to get DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def delete_file_and_entry(file_path: str, file_id: int):
    # Wait for one minute
    time.sleep(60)

    # Check if the file exists, then delete it
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} deleted.")

    # Open a new database session
    db = database.SessionLocal()

    try:
        # Delete the corresponding database entry
        db_entry = db.query(models.TextToSpeech).filter(models.TextToSpeech.id == file_id).first()
        
        if db_entry:
            db.delete(db_entry)
            db.commit()
            print(f"Database entry with ID {file_id} deleted.")
    finally:
        # Always close the session
        db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to Text to Speech API!"}

@app.post("/generate-audio/")
def generate_audio(input: TextInput, db: Session = Depends(get_db), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        if input.text is None or input.text.strip() == "":
            raise HTTPException(status_code=400, detail="Text input cannot be empty.")

        output_filename = f"output_{uuid.uuid4()}.mp3"
        output_path = f"./audio/{output_filename}"

        print("Initializing TTS engine...")
        engine = pyttsx3.init()
        if engine is None:
            raise Exception("Failed to initialize TTS engine.")

        tts.text_to_speech(input.text, output_path)

        db_record = models.TextToSpeech(text=input.text, audio_file=output_path)
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        background_tasks.add_task(delete_file_and_entry, output_path, db_record.id)

        return {"audio_file": output_filename}
    except Exception as e:
        print(f"Error during audio generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/audio/{file_name}")
def get_audio(file_name: str):
    file_path = f"./audio/{file_name}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path)
