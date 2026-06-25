import cv2
import math
import numpy as np
from datastructures import Scene
from OCR.utils.Hog import HoG, calc_gradients
_area_sums_cache = {}

def _area_sums(parent_size, num_parts):
    key = (parent_size, num_parts)
    if key in _area_sums_cache:
        return _area_sums_cache[key]
    if num_parts == 0:
        result = frozenset({0})
    elif num_parts == 1:
        result = frozenset({parent_size * parent_size})
    elif parent_size < num_parts:
        result = frozenset()
    else:
        res = set()
        for i in range(int(round(parent_size / 2))):
            sq = (i+1) * (i+1)
            for s in _area_sums(parent_size-i-1, num_parts-1):
                res.add(sq+s)
        result = frozenset(res)
    _area_sums_cache[key] = result
    return result


def _bgr_to_hsv(img):
    bgr = img.astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    rgb = np.stack([r, g, b], axis=-1)
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    delta = cmax-cmin
    h = np.zeros_like(cmax)
    mask_r = (cmax==r) & (delta!=0)
    mask_g = (cmax==g) & (delta!=0)
    mask_b = (cmax==b) & (delta!=0)
    h[mask_r] = 60.0* (((g[mask_r]-b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 *(((b[mask_g] -r[mask_g])/delta[mask_g])+ 2)
    h[mask_b] = 60.0 * (((r[mask_b] -g[mask_b])/delta[mask_b])+ 4)
    s = delta/np.maximum(cmax, 1e-12)
    h /= 2
    s *= 255
    cmax *= 255
    return np.stack([h.astype(np.uint8), s.astype(np.uint8), cmax.astype(np.uint8)], axis=-1)



import os
import cv2
import numpy as np


def detectShots(video_path,
                min_len=10,
                history_size=50,
                k=2.5):

    os.makedirs("debug/shots", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_hsv = None
    prev_edges = None

    history = []

    cuts = [0]
    fi = 0
    current_length = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (160, 90))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        current_length += 1

        if prev_hsv is not None:

            hsv_diff = np.abs(
                hsv.astype(np.int32) -
                prev_hsv.astype(np.int32)
            ).mean()

            edge_diff = np.mean(edges != prev_edges)

            score = hsv_diff + 15 * edge_diff

            if len(history) >= 10:

                mu = np.mean(history)
                sigma = np.std(history)

                adaptive_threshold = mu + k * sigma

                if score > adaptive_threshold:

                    cuts.append(fi)

                    cv2.imwrite(
                        f"debug/shots/shot_{len(cuts):04d}_frame_{fi}.jpg",
                        frame
                    )

                    current_length = 0
                    history = history[-10:]

            history.append(score)

            if len(history) > history_size:
                history.pop(0)

        prev_hsv = hsv
        prev_edges = edges

        fi += 1

    cap.release()

    cuts.append(fi)

    cleaned_cuts = [0]
    for i in range(1, len(cuts)):
        start = cuts[i-1]
        end = cuts[i]
        if (end - start) > min_len:
            cleaned_cuts.append(end)

    if cleaned_cuts[-1] != fi:
        cleaned_cuts.append(fi)

    return cleaned_cuts, fi, fps
class SceneSegmenter:
    def __init__(self, video_path, threshold=5, min_shot_len=10):
        self.video_path =video_path
        self.threshold= threshold
        self.min_shot_len= min_shot_len
        self.max_len_between_shots= 300
        self.scenes_list = []
        self.scene_frames = []
        self._video_cuts = []

    def segment_video(self, threshold=None):
        if threshold is None:
            threshold = self.threshold
        self.scenes_list = []
        self.scene_frames = []
        self._video_cuts = []
        cuts, total_frames, fps = detectShots(self.video_path)
        self._video_cuts = cuts

        for i in range(1, len(cuts)):
            start_f = cuts[i-1]
            end_f = cuts[i]
            self.scenes_list.append(Scene(
                index=i,
                start_time=start_f / fps,
                end_time=end_f / fps,
                start_frame=start_f,
                end_frame=end_f,
                fps=fps,
            ))

        return self.scenes_list

    def _frames_in_scene(self, scene, last_scene = False) -> list[int]:
        result = [f for f in self._video_cuts if scene.start_frame <= f < scene.end_frame]
        if last_scene:
            result.append(scene.end_frame)
        return result

    def extract_frames(self):
        print(f"Extracting frames from {len(self.scenes_list)} scenes...")
        if not self.scenes_list:
            return []

        self.scene_frames = []
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not cap.isOpened():
            print(f"Error: could not open {self.video_path}")
            return []

        
        for scene in self.scenes_list:
            frame_numbers = [scene.start_frame]
            for frame_num in frame_numbers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    self.scene_frames.append({
                        'scene_index': scene.index,
                        'frame_number': frame_num,
                        'frame_time': frame_num / fps,
                        'frame': frame,
                        'scene': scene,
                        'frame_count_in_scene': len(frame_numbers),
                    })
                else:
                    print(f"Warning: could not read frame {frame_num} (scene {scene.index})")

        cap.release()
        print(f"Extracted {len(self.scene_frames)} frames from {len(self.scenes_list)} scenes")
        return self.scene_frames

    def get_frames_for_ocr(self):
        if not self.scene_frames:
            self.extract_frames()
        return[d['frame'] for d in self.scene_frames]

    def get_frames_with_metadata(self):
        if not self.scene_frames:
            self.extract_frames()
        return self.scene_frames

    def get_scenes(self):
        return self.scenes_list
