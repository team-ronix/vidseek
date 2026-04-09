from SQL.DatabaseClient import SessionLocal
from SQL.Models.Video import Video

class VideoRepository:
    def __init__(self):
        self.db = SessionLocal()

    def create_video(self, file_name: str, file_path: str) -> Video:
        video = Video(file_name=file_name, file_path=file_path)
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        return video

    def get_video_by_id(self, video_id: int) -> Video:
        return self.db.query(Video).filter(Video.id == video_id).first()