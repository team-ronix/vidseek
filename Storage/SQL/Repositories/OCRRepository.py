from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.OCRWords import OCRWord


class OCRRepository:
    def __init__(self):
        self.db = SessionLocal()

    def save_word(self, word: str, video_id: int, start_time: float, end_time: float, frame_number: int):
        exists = (
            self.db.query(OCRWord)
            .filter(
                OCRWord.word == word,
                OCRWord.video_id == video_id,
                OCRWord.frame_number == frame_number,
            )
            .first()
        )
        if not exists:
            self.db.add(OCRWord(
                word=word,
                video_id=video_id,
                start_time=int(start_time),
                end_time=int(end_time),
                frame_number=frame_number,
            ))
            self.db.commit()

    def close(self):
        self.db.close()
