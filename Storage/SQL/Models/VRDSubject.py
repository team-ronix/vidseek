from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base

class VRDSubject(Base):
    __tablename__ = "vrd_subjects"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, index=True)

    vrd_videos = relationship("VRDVideo", back_populates="subject")