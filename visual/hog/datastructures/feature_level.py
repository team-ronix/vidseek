from dataclasses import dataclass
import numpy as np

@dataclass
class FeatureLevel:
    feature_map: np.ndarray
    scale: float
    level: int
