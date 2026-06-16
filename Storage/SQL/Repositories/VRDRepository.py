from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.VRDSubject import VRDSubject
from Storage.SQL.Models.VRDPredicate import VRDPredicate
from Storage.SQL.Models.VRDObject import VRDObject
from Storage.SQL.Models.VRDVideo import VRDVideo


class VRDRepository:
    def __init__(self):
        self.db = SessionLocal()

    def _get_or_create(self, model, key: str):
        instance = self.db.query(model).filter(model.key == key).first()
        if not instance:
            instance = model(key=key)
            self.db.add(instance)
            self.db.flush()
        return instance

    def save_from_inverted_index(self, inverted_index: dict, video_id: int):
        """
        inverted_index: { "Dog, sitting on, Couch": [{ scene, frame, video_path, start_time, end_time }, ...] }
        Key is a comma-separated "subject, predicate, object" triple.
        """
        try:
            for triple_key, occurrences in inverted_index.items():
                parts = [p.strip() for p in triple_key.split(",")]
                if len(parts) != 3:
                    continue
                subject_key, predicate_key, object_key = parts

                subject   = self._get_or_create(VRDSubject,   subject_key)
                predicate = self._get_or_create(VRDPredicate, predicate_key)
                obj       = self._get_or_create(VRDObject,    object_key)

                for occ in occurrences:
                    exists = (
                        self.db.query(VRDVideo)
                        .filter(
                            VRDVideo.subject_id   == subject.id,
                            VRDVideo.predicate_id == predicate.id,
                            VRDVideo.object_id    == obj.id,
                            VRDVideo.video_id     == video_id,
                            VRDVideo.start_time   == occ.get("start_time"),
                            VRDVideo.end_time     == occ.get("end_time"),
                        ).first()
                    )
                    if exists:
                        continue
                    self.db.add(VRDVideo(
                        subject_id=subject.id,
                        predicate_id=predicate.id,
                        object_id=obj.id,
                        video_id=video_id,
                        start_time=occ.get("start_time"),
                        end_time=occ.get("end_time"),
                    ))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        finally:
            self.db.close()

    def search(self, query: str) -> list[dict]:
        """Keyword search across subject, predicate, and object keys."""
        q = f"%{query.lower()}%"
        rows = (
            self.db.query(VRDVideo, VRDSubject, VRDPredicate, VRDObject)
            .join(VRDSubject,   VRDVideo.subject_id   == VRDSubject.id)
            .join(VRDPredicate, VRDVideo.predicate_id == VRDPredicate.id)
            .join(VRDObject,    VRDVideo.object_id    == VRDObject.id)
            .filter(
                VRDSubject.key.ilike(q)
                | VRDPredicate.key.ilike(q)
                | VRDObject.key.ilike(q)
            ).all()
        )
        return [
            {
                "type":       "vrd",
                "text":       f"{subj.key}, {pred.key}, {obj.key}",
                "video_id":   vrd.video_id,
                "start_time": vrd.start_time,
                "end_time":   vrd.end_time,
            }
            for vrd, subj, pred, obj in rows
        ]
