from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base

class Object(Base):
    __tablename__ = "objects"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, index=True)

    object_videos = relationship("ObjectVideo", back_populates="object", overlaps="videos")
    videos = relationship("Video", secondary="object_videos", back_populates="objects", overlaps="object_videos")