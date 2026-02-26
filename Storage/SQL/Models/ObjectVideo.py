from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base

class ObjectVideo(Base):
    __tablename__ = "object_videos"

    id = Column(Integer, primary_key=True, index=True)
    object_id = Column(Integer, ForeignKey("objects.id"), nullable=False, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)

    object = relationship("Object", back_populates="object_videos")
    video = relationship("Video", back_populates="object_videos")