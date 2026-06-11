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
        inverted_index shape:
          { "dog": [ { scene, frame, video_path, start_time, end_time, confidence }, ... ] }
        """
        try:
            for object_key, occurrences in inverted_index.items():
                obj = self._get_or_create_object(object_key)

                for occ in occurrences:
                    # avoid duplicate rows for same object + video + time range
                    exists = (
                        self.db.query(ObjectVideo)
                        .filter(
                            ObjectVideo.object_id == obj.id,
                            ObjectVideo.video_id == video_id,
                            ObjectVideo.start_time == occ.get("start_time"),
                            ObjectVideo.end_time == occ.get("end_time"),
                        ).first()
                    )
                    if exists:
                        continue

                    object_video = ObjectVideo(
                        object_id=obj.id,
                        video_id=video_id,
                        start_time=occ.get("start_time"),
                        end_time=occ.get("end_time"),
                    )
                    self.db.add(object_video)

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
            .join(Video, ObjectVideo.video_id == Video.id)
            .filter(Object.key.ilike(q))
            .all()
        )
        results = []
        for object_video, obj, video in rows:
            results.append(
                {
                    "type": "object",
                    "text": obj.key,
                    "video_id": object_video.video_id,
                    "video_path": video.file_path,
                    "start_time": object_video.start_time,
                    "end_time": object_video.end_time,
                }
            )
        return results
