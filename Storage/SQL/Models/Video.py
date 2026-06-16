from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    file_path = Column(String, unique=True)

    object_videos = relationship("ObjectVideo", back_populates="video", overlaps="objects")
    objects = relationship("Object", secondary="object_videos", back_populates="videos", overlaps="object_videos")
    vrd_videos = relationship("VRDVideo", back_populates="video")