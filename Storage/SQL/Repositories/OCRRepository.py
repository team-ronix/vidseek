from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.OCRWords import OCRWord
from Storage.SQL.Models.Video import Video
from OCR.utils.word_distance import find_closest_word, get_edit_distance


class OCRRepository:
    def __init__(self):
        self.db = SessionLocal()

    def save_word(self, word: str, video_id: int, start_time: float, end_time: float, frame_number: int, word_detection_model: str, word_recognition_model: str):
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
                word_detection_model=word_detection_model,
                word_recognition_model=word_recognition_model
            ))
            self.db.commit()

    def find_closest_word_video(self, target: str, video_id = None):
        print("test find_closest_word_video")
        words = None
        if video_id is None:
            words = self.db.query(OCRWord).all()
        else:
            words = self.db.query(OCRWord).filter(OCRWord.video_id == video_id).all()
        sorted_words = find_closest_word(words, target)
        if sorted_words == -1:
            return None
        return [(words[i], confidence) for i, confidence in sorted_words]
    
    def find_closest_word_global(self, word: str):
        all_words = self.db.query(OCRWord).all()
        closest_word_index, confidence = find_closest_word(all_words, word)
        if closest_word_index == -1:
            return None, 0.0
        return all_words[closest_word_index], confidence
        

    def search(self, query: str, threshold: float = 0.6) -> list[dict]:
        rows = (
            self.db.query(OCRWord, Video)
            .join(Video, OCRWord.video_id == Video.id)
            .all()
        )
        results = []
        for word, video in rows:
            _, confidence = get_edit_distance(query.lower(), word.word.lower())
            if confidence >= threshold:
                results.append({
                    "type": "ocr",
                    "text": word.word,
                    "video_id": word.video_id,
                    "video_path": video.file_path,
                    "start_time": word.start_time,
                    "end_time": word.end_time,
                    "score": confidence,
                })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def close(self):
        self.db.close()
