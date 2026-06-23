import google.generativeai as genai
import json
import time
from PIL import Image
import cv2

class VRD:
    def __init__(self, video_path, frames, api_key: str, model_id="gemini-3.1-flash-lite"):
        self.video_path = video_path
        self.frames = frames
        self.api_key = api_key
        self.model_id = model_id
        self.model = None
        self.inverted_index = {}
        self.setup_model()

    def setup_model(self):
        if self.model is not None:
            return
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_id)

    def detect_relationships(self):
        prompt = (
            "Analyze this image and extract the visual relationships between the objects. "
            "Return the results strictly as a list of triplets in the format: [Subject, Predicate, Object]. "
            "The available subjects and objects are [Dog, Cat, Couch, Table, Chair, Person, Car, Tree, Building, Street]. "
            "The available predicates are [sitting on, standing on, next to, under, above, holding, looking at, running]. "
            "For example: [Dog, sitting on, Couch]. "
            "List only the triplets, one per line, nothing else."
            "Don't return anything other than the triplets if none exists return empty string"
        )
        for i, frame_data in enumerate(self.frames, 1):
            frame_number = frame_data['frame_number']
            frame_time = frame_data['frame_time']
            scene = frame_data['scene']
            frame_count = frame_data['frame_count_in_scene']
            frame = frame_data['frame']

            print(f"[{i}/{len(self.frames)}] Scene {scene.index}: Start at {scene.start_time:.2f}s, End at {scene.end_time:.2f}s, Duration {scene.duration:.2f}s ({frame_count} frames)")
            print(f"       Processing frame {frame_number}")

            if frame_number is None:
                continue

            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                output = None
                for attempt in range(4):
                    try:
                        response = self.model.generate_content([prompt, image])
                        
                        if response.candidates:
                            parts = response.candidates[0].content.parts

                            if parts and hasattr(parts[0], "text"):
                                output = parts[0].text
                            else:
                                output = None
                        else:
                            output = None

                        break
                    except Exception as e:
                        msg = str(e).lower()
                        if "429" in msg or "quota" in msg or "resource exhausted" in msg:
                            wait = 60
                            print(f"       - Rate limited, retrying in {wait}s ")
                            time.sleep(wait)
                        else:
                            print(f"       - Gemini error for frame {frame_number}: {e}")
                            break
                else:
                    print(f"       - Skipping frame {frame_number} after 4 rate-limit retries")

                if output is None:
                    continue

                if not output or not output.strip():
                    print(f"       - No relationships detected for frame {frame_number}")
                    continue

                relations: list[str] = output.strip().split(']')
                for relation in relations:
                    print(f"       - Detected relationship: {relation}")
                    while len(relation) > 0 and not relation[0].isalpha():
                        relation = relation[1:]
                    triple = relation.strip().split(',')
                    if len(triple) != 3:
                        continue
                    rel = ', '.join([part.strip() for part in triple])
                    if rel not in self.inverted_index:
                        self.inverted_index[rel] = []

                    if any(occ['scene'] == scene.index for occ in self.inverted_index[rel]):
                        continue

                    self.inverted_index[rel].append({
                        'scene': scene.index,
                        'frame': frame_number,
                        'video_path': self.video_path,
                        'frame_time': frame_time,
                        'start_time': scene.start_time,
                        'end_time': scene.end_time
                    })

    def save_inverted_index(self, output_path='vrd_inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)

    def load_inverted_index(self, input_path='vrd_inverted_index.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)

    def get_inverted_index(self):
        return self.inverted_index
