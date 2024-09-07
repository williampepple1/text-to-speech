from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid
import os
import time
from fastapi.responses import FileResponse
from app import models, database, tts
from fastapi.middleware.cors import CORSMiddleware

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

def delete_file(file_path: str):
    time.sleep(60)  # Wait for one minute
    if os.path.exists(file_path):
        os.remove(file_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to Text to Speech API!"}

@app.post("/generate-audio/")
def generate_audio(input: TextInput, db: Session = Depends(get_db), background_tasks: BackgroundTasks = BackgroundTasks()):
    output_filename = f"output_{uuid.uuid4()}.mp3"
    output_path = f"./app/audio/{output_filename}"

    # Convert text to speech using pyttsx3
    tts.text_to_speech(input.text, output_path)

    # Save the generated audio details in the database
    db_record = models.TextToSpeech(text=input.text, audio_file=output_path)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    # Add a background task to delete the audio file after one minute
    background_tasks.add_task(delete_file, output_path)

    return {"audio_file": output_filename}

@app.get("/audio/{file_name}")
def get_audio(file_name: str):
    file_path = f"./app/audio/{file_name}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path)
