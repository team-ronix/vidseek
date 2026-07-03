from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id       = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    text     = Column(Text,    nullable=False)
    start    = Column(Float,   nullable=False)
    end      = Column(Float,   nullable=False)
    title    = Column(String,  nullable=False)

    video = relationship("Video", back_populates="transcript_segments")
