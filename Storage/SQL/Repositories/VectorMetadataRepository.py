import json
from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.VectorMetadata import VectorMetadata


class VectorMetadataRepository:
    def __init__(self):
        self.db = SessionLocal()

    def upsert(self, vector_id: int, external_id: str | None,
               metadata: dict, embedder_model: str = "transformer") -> None:
        row = (
            self.db.query(VectorMetadata)
            .filter(
                VectorMetadata.vector_id      == vector_id,
                VectorMetadata.embedder_model == embedder_model,
            )
            .first()
        )
        payload = json.dumps(metadata or {}, ensure_ascii=True)
        meta    = metadata or {}

        if row is None:
            row = VectorMetadata(
                vector_id      = vector_id,
                embedder_model = embedder_model,
                external_id    = external_id,
                entry_type     = meta.get("type"),
                text           = meta.get("text"),
                video_path     = meta.get("video_path"),
                start_time     = meta.get("start_time"),
                end_time       = meta.get("end_time"),
                payload_json   = payload,
            )
            self.db.add(row)
        else:
            row.external_id  = external_id
            row.entry_type   = meta.get("type")
            row.text         = meta.get("text")
            row.video_path   = meta.get("video_path")
            row.start_time   = meta.get("start_time")
            row.end_time     = meta.get("end_time")
            row.payload_json = payload

        self.db.commit()

    def get_by_vector_ids(self, vector_ids: list[int],
                          embedder_model: str = "transformer") -> dict[int, dict]:
        if not vector_ids:
            return {}
        rows = (
            self.db.query(VectorMetadata)
            .filter(
                VectorMetadata.vector_id.in_(vector_ids),
                VectorMetadata.embedder_model == embedder_model,
            )
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
            base_meta.setdefault("type",       row.entry_type)
            base_meta.setdefault("text",       row.text)
            base_meta.setdefault("video_path", row.video_path)
            base_meta.setdefault("start_time", row.start_time)
            base_meta.setdefault("end_time",   row.end_time)
            out[int(row.vector_id)] = base_meta
        return out

    def delete_by_model(self, embedder_model: str) -> int:
        """Delete all vector_metadata rows for the given embedder. Returns deleted count."""
        deleted = (
            self.db.query(VectorMetadata)
            .filter(VectorMetadata.embedder_model == embedder_model)
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return deleted

    def close(self):
        self.db.close()
