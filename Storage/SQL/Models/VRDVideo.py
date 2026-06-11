from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base

class VRDVideo(Base):
    __tablename__ = "vrd_videos"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("vrd_subjects.id"), nullable=False, index=True)
    predicate_id = Column(Integer, ForeignKey("vrd_predicates.id"), nullable=False, index=True)
    object_id = Column(Integer, ForeignKey("vrd_objects.id"), nullable=False, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    start_time = Column(Integer)
    end_time = Column(Integer)

    subject = relationship("VRDSubject", back_populates="vrd_videos")
    predicate = relationship("VRDPredicate", back_populates="vrd_videos")
    object = relationship("VRDObject", back_populates="vrd_videos")
    video = relationship("Video", back_populates="vrd_videos")