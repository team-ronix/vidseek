from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base

class OCRWord(Base):
    __tablename__ = "ocr_words"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    word = Column(String, nullable=False, index=True)
    start_time = Column(Integer)
    end_time = Column(Integer)
    frame_number = Column(Integer)

    video = relationship("Video", back_populates="ocr_words")