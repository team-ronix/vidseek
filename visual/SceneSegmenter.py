import cv2
import numpy as np
from datastructures import Scene
import os
import cv2

def detectShots(video_path,min_len=10,
                history_size=50,
                k=2.5):

    base_name = os.path.basename(video_path)
    os.makedirs(f"debug/shots/{base_name}/", exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_hsv =None
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
            hsv_diff = np.abs(hsv.astype(np.int32) -prev_hsv.astype(np.int32)).mean()

            edge_diff = np.mean(edges != prev_edges)

            score = hsv_diff + 15 * edge_diff

            if len(history) >= 10:
                mu =np.mean(history)
                sigma = np.std(history)
                adaptive_threshold = mu + k * sigma

                if score > adaptive_threshold:

                    cuts.append(fi)
                    cv2.imwrite(
                        f"debug/shots/{base_name}/shot_{len(cuts):04d}_frame_{fi}.jpg",
                        frame
                    )
                    current_length= 0
                    history =history[-10:]

            history.append(score)

            if len(history) > history_size:
                history.pop(0)

        prev_hsv = hsv
        prev_edges = edges

        fi += 1

    cap.release()
    cuts.append(fi)

    cleaned_cuts= [0]
    for i in range(1, len(cuts)):
        start = cuts[i-1]
        end = cuts[i]
        if (end - start) > min_len:
            cleaned_cuts.append(end)

    if cleaned_cuts[-1] != fi:
        cleaned_cuts.append(fi)

    return cleaned_cuts,fi, fps
class SceneSegmenter:
    def __init__(self, video_path, threshold=5, min_shot_len=10):
        self.video_path =video_path
        self.threshold= threshold
        self.min_shot_len= min_shot_len
        self.max_len_between_shots= 300
        self.scenes_list =[]
        self.scene_frames= []
        self._video_cuts = []

    def segment_video(self, threshold=None):
        if threshold is None:
            threshold = self.threshold
        self.scenes_list = []
        cuts, _, fps = detectShots(self.video_path)
        self._video_cuts = cuts

        for i in range(1, len(cuts)):
            start_f = cuts[i-1]
            end_f = cuts[i]
            cuts[i] +=1
            self.scenes_list.append(Scene(
                index=i,
                start_time=start_f / fps,
                end_time=end_f / fps,
                start_frame=start_f,
                end_frame=end_f,
                fps=fps,
            ))

        return self.scenes_list

    def extract_frames(self):
        print(f"Extracting frames from {len(self.scenes_list)} scenes...")
        if not self.scenes_list:
            return []

        self.scene_frames = []
        cap = cv2.VideoCapture(self.video_path)
        fps =cap.get(cv2.CAP_PROP_FPS)

        if not cap.isOpened():
            print(f"Error: could not open {self.video_path}")
            return []

        
        for scene in self.scenes_list:
            frame_numbers = [scene.start_frame]
            for frame_num in frame_numbers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame =cap.read()
                
                if ret:
                    self.scene_frames.append({
                        'scene_index':scene.index,
                        'frame_number': frame_num,
                        'frame_time': frame_num/fps,
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
