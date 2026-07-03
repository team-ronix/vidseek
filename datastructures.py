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