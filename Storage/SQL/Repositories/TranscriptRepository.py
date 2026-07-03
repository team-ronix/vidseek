from typing import List, Dict, Any
from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.TranscriptSegment import TranscriptSegment


class TranscriptRepository:
    def __init__(self):
        self.db = SessionLocal()

    def save_segments(self, segments: List[Dict[str, Any]], video_id: int) -> None:
        for seg in segments:
            self.db.add(TranscriptSegment(
                video_id = video_id,
                text     = seg["text"],
                start    = seg["start"],
                end      = seg["end"],
                title    = seg["title"],
            ))
        self.db.commit()

    def get_by_video(self, video_id: int) -> List[TranscriptSegment]:
        return (
            self.db.query(TranscriptSegment)
            .filter(TranscriptSegment.video_id == video_id)
            .order_by(TranscriptSegment.start)
            .all()
        )

    def get_all_segments_with_video(self) -> List[Dict[str, Any]]:
        """Return all transcript segments as dicts for HybridEmbedder corpus fitting."""
        from Storage.SQL.Models.Video import Video as VideoModel
        rows = (
            self.db.query(TranscriptSegment, VideoModel)
            .join(VideoModel, TranscriptSegment.video_id == VideoModel.id)
            .order_by(TranscriptSegment.id)
            .all()
        )
        return [
            {
                "text":       seg.text,
                "video_path": video.file_path,
                "start":      seg.start,
                "end":        seg.end,
            }
            for seg, video in rows
            if seg.text and seg.text.strip()
        ]

    def close(self):
        self.db.close()
