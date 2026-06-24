from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.Object import Object
from Storage.SQL.Models.ObjectVideo import ObjectVideo
from Storage.SQL.Models.Video import Video


class ObjectRepository:
    def __init__(self):
        self.db = SessionLocal()

    def _get_or_create_object(self, key: str) -> Object:
        instance = self.db.query(Object).filter(Object.key == key).first()
        if not instance:
            instance = Object(key=key)
            self.db.add(instance)
            self.db.flush()
        return instance

    def save_from_inverted_index(self, inverted_index: dict, video_id: int):
        """
        inverted_index: { "dog": [{ scene, frame, video_path, start_time, end_time, confidence }, ...] }
        """
        try:
            for object_key, occurrences in inverted_index.items():
                obj = self._get_or_create_object(object_key)
                for occ in occurrences:
                    print(f"frame time is : {occ.get("frame_time")}")
                    exists = (
                        self.db.query(ObjectVideo)
                        .filter(
                            ObjectVideo.object_id == obj.id,
                            ObjectVideo.video_id  == video_id,
                            ObjectVideo.frame_time == occ.get("frame_time")
                        ).first()
                    )
                    if exists:
                        continue
                    self.db.add(ObjectVideo(
                        object_id=obj.id,
                        video_id=video_id,
                        start_time=occ.get("start_time"),
                        end_time=occ.get("end_time"),
                        frame_time = occ.get("frame_time")
                    ))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        finally:
            self.db.close()

    def search(self, query: str) -> list[dict]:
        """Keyword search across object keys."""
        q = f"%{query.lower()}%"
        rows = (
            self.db.query(ObjectVideo, Object, Video)
            .join(Object, ObjectVideo.object_id == Object.id)
            .join(Video,  ObjectVideo.video_id  == Video.id)
            .filter(Object.key.ilike(q))
            .all()
        )
        return [
            {
                "type":       "object",
                "text":       obj.key,
                "video_id":   ov.video_id,
                "video_path": video.file_path,
                "start_time": ov.start_time,
                "frame_time" : ov.frame_time,
                "end_time":   ov.end_time,
            }
            for ov, obj, video in rows
        ]
