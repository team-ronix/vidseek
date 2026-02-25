from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import json
from PIL import Image
import cv2

class VRD:
    def __init__(self, video_path, frames, model_id="llava-hf/llava-1.5-7b-hf"):
        self.video_path = video_path
        self.frames = frames
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.inverted_index = {}
        self.setup_model()
        
    def setup_model(self):
        if self.model is not None:
            return
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            self.model.to("cuda")
        
    def detect_relationships(self):
        prompt = (
            "USER: <image>\n"
            "Analyze this image and extract the visual relationships between the objects. "
            "Return the results strictly as a list of triplets in the format: [Subject, Predicate, Object]. "
            "For example: [Dog, sitting on, Couch].\n"
            "ASSISTANT:"
        )
        for i, frame_data in enumerate(self.frames, 1):
            frame_number = frame_data['frame_number']
            scene = frame_data['scene']
            frame_count = frame_data['frame_count_in_scene']
            frame = frame_data['frame']
            
            print(f"[{i}/{len(self.frames)}] Scene {scene.index}: Start at {scene.start_time:.2f}s, End at {scene.end_time:.2f}s, Duration {scene.duration:.2f}s ({frame_count} frames)")
            print(f"       Processing frame {frame_number}")
            
            if frame_number is None:
                continue
            
            if frame is not None:
                # Convert BGR to RGB for LLaVA model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                generate_ids = self.model.generate(**inputs, max_new_tokens=150)
                output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                assistant_response = output.split("ASSISTANT:")[-1]
                relations: list[str] = assistant_response.strip().split(']')
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
                    
                    # Skip if already exists in this scene
                    if any(occ['scene'] == scene.index for occ in self.inverted_index[rel]):
                        continue

                    self.inverted_index[rel].append({
                        'scene': scene.index,
                        'frame': frame_number,
                        'video_path': self.video_path,
                        'start_time': scene.start_time,
                        'end_time': scene.end_time
                    })
    
    def save_inverted_index(self, output_path='vrd_inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)
            
    def get_inverted_index(self):
        return self.inverted_index
