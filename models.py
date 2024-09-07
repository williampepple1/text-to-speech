from sqlalchemy import Column, Integer, String
from database import Base

class TextToSpeech(Base):
    __tablename__ = 'tts'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    audio_file = Column(String)
