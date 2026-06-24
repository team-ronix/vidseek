import json

from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.VectorMetadata import VectorMetadata


class VectorMetadataRepository:
    def __init__(self):
        self.db = SessionLocal()

    def upsert(self, vector_id: int, external_id: str | None, metadata: dict) -> None:
        row = self.db.query(VectorMetadata).filter(VectorMetadata.vector_id == vector_id).first()
        payload = json.dumps(metadata or {}, ensure_ascii=True)

        if row is None:
            row = VectorMetadata(
                vector_id=vector_id, 
                external_id=external_id, 
                entry_type=(metadata or  {}).get("type"),
                text=(metadata or {} ).get("text"),
                video_path=(metadata or  {} ).get("video_path"),
                start_time=(metadata or {}).get("start_time"),
                end_time=(metadata or {}).get("end_time"),
                payload_json=payload,
            )
            self.db.add(row)
        else:
            row.external_id = external_id
            row.entry_type = (metadata or {}).get("type")
            row.text = (metadata or {}).get("text")
            row.video_path = (metadata or {}).get("video_path")
            row.start_time = (metadata or {}).get("start_time")
            row.end_time = (metadata or {}).get("end_time")
            row.payload_json = payload

        self.db.commit()

    def get_by_vector_ids(self, vector_ids: list[int]) -> dict[int, dict]:
        if not vector_ids:
            return {}

        rows = (
            self.db.query(VectorMetadata)
            .filter(VectorMetadata.vector_id.in_(vector_ids))
            .all()
        )

        out = {}
        for row in rows:
            base_meta = {}
            if row.payload_json:
                try:
                    base_meta = json.loads(row.payload_json)
                except Exception:
                    base_meta = {}
            base_meta.setdefault("type", row.entry_type)
            base_meta.setdefault("text", row.text)
            base_meta.setdefault("video_path", row.video_path)
            base_meta.setdefault("start_time", row.start_time)
            base_meta.setdefault("end_time", row.end_time)
            out[int(row.vector_id)] = base_meta
        return out

    def close(self):
        self.db.close()
