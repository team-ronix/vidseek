from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base

class ObjectVideo(Base):
    __tablename__ = "object_videos"

    id = Column(Integer, primary_key=True, index=True)
    object_id = Column(Integer, ForeignKey("objects.id"), nullable=False, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    start_time = Column(Float)
    end_time = Column(Float)
    frame_time = Column(Float)
    model_name = Column(String, nullable=True)

    object = relationship("Object", back_populates="object_videos", overlaps="objects,videos")
    video = relationship("Video", back_populates="object_videos", overlaps="objects,videos")