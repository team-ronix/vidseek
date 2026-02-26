from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from Storage.SQL.Models.Base import Base

class VRDPredicate(Base):
    __tablename__ = "vrd_predicates"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, index=True)

    vrd_videos = relationship("VRDVideo", back_populates="predicate")