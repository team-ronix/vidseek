from dataclasses import dataclass


@dataclass
class Scene:
    index: int
    start_time: float      # seconds
    end_time: float        # seconds
    start_frame: int
    end_frame: int
    fps: float

    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass
class PipelineContext:
    video_path: str
    video_id: str
    scenes: list[Scene]
    frames: list[dict]
    ocr_inverted_index: dict
    object_inverted_index: dict
    vrd_inverted_index: dict
    audio_transcription: str