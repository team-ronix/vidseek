from sqlalchemy import Column, Float, Integer, String, Text
from Storage.SQL.Models.Base import Base
class VectorMetadata(Base):
    __tablename__ = "vector_metadata"

    id = Column(Integer, primary_key=True, index=True)

    vector_id =Column(Integer, nullable=False, unique=True, index=True)
    external_id= Column(String, nullable=True)
    entry_type =Column(String, nullable=True,index=True)
    text = Column(Text, nullable=True)

    video_path = Column(String, nullable=True,index=True)
    start_time = Column(Float,nullable=True)
    end_time = Column(Float,nullable=True)
    payload_json = Column(Text,nullable=True)
