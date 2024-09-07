from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid
import os
from fastapi.responses import FileResponse
from app import models, database, tts

app = FastAPI()

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

@app.post("/generate-audio/")
def generate_audio(input: TextInput, db: Session = Depends(get_db)):
    output_filename = f"output_{uuid.uuid4()}.mp3"
    output_path = f"./app/audio/{output_filename}"

    # Convert text to speech using pyttsx3
    tts.text_to_speech(input.text, output_path)

    # Save the generated audio details in the database
    db_record = models.TextToSpeech(text=input.text, audio_file=output_path)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return {"audio_file": output_filename}

@app.get("/audio/{file_name}")
def get_audio(file_name: str):
    file_path = f"./app/audio/{file_name}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path)
