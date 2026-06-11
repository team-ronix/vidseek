import easyocr
import cv2
import json

class OCR:
    def __init__(self, frames, video_path):
        self.frames = frames
        self.video_path = video_path
        self.inverted_index = {}
        self.reader = easyocr.Reader(['en'])
    
    def process_frames(self):
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
                # Preprocess frame for OCR
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Run OCR
                
                # TODO: Maybe we need to do some transfromation to the frame to enhance text detection
                
                print("       Running OCR...")
                result = self.reader.readtext(blur)
                print(f"       Detected {len(result)} text regions")
                
                # Merge close boxes to form full sentence
                merged_result = self.merge_nearby_text(result, horizontal_threshold=50, vertical_threshold=15)
                merged_result.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence

                for detection in merged_result:
                    text, confidence = detection
                    if confidence < 0.5 or len(text.strip()) < 5:  
                        continue
                    print(f"       - '{text}' (confidence: {confidence:.2f})")
                    # Add to inverted index
                    if text not in self.inverted_index:
                        self.inverted_index[text] = []
                    
                    # Skip if already exists in this scene
                    if any(occ['scene'] == scene.index for occ in self.inverted_index[text]):
                        continue

                    self.inverted_index[text].append({
                        'scene': scene.index,
                        'frame': frame_number,
                        'video_path': self.video_path,
                        'start_time': scene.start_time,
                        'end_time': scene.end_time,
                        'confidence': confidence
                    })
            else:
                print("       Warning: Frame data is None")


    def save_inverted_index(self, output_path='ocr_inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)
            
    def load_inverted_index(self, input_path='ocr_inverted_index.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)
            
    def get_inverted_index(self):
        return self.inverted_index
    
    def sorting_function(self, result):
        box = result[0]
        center_y = (box[0][1] + box[2][1]) / 2
        center_x = (box[0][0] + box[2][0]) / 2
        return (center_y, center_x)
    
    def merge_nearby_text(self, ocr_results, horizontal_threshold=50, vertical_threshold=15):
        if not ocr_results:
            return []
        sorted_results = sorted(ocr_results, key=self.sorting_function)
        merged = []
        current_group = [sorted_results[0]]
        
        for i in range(1, len(sorted_results)):
            prev_box = current_group[-1][0]
            curr_box = sorted_results[i][0]
            
            prev_center_y = (prev_box[0][1] + prev_box[2][1]) / 2
            curr_center_y = (curr_box[0][1] + curr_box[2][1]) / 2
            prev_height = abs(prev_box[2][1] - prev_box[0][1])
            curr_height = abs(curr_box[2][1] - curr_box[0][1])
            
            prev_right_x = prev_box[1][0]
            curr_left_x = curr_box[0][0]
            
            # Check if boxes are on the same line
            # Use the maximum height as reference for vertical alignment
            max_height = max(prev_height, curr_height)
            vertical_distance = abs(curr_center_y - prev_center_y)
            horizontal_distance = curr_left_x - prev_right_x
            
            # Boxes are on same line if their centers are within a fraction of the max height
            # and they're close horizontally
            same_line = vertical_distance <= max(vertical_threshold, max_height * 0.5)
            close_horizontal = horizontal_distance <= horizontal_threshold
            
            if same_line and close_horizontal:
                current_group.append(sorted_results[i])
            else:
                if current_group:
                    merged_text = ' '.join([item[1] for item in current_group])
                    avg_confidence = sum([item[2] for item in current_group]) / len(current_group)
                    merged.append((merged_text, avg_confidence))
                current_group = [sorted_results[i]]
        
        if current_group:
            merged_text = ' '.join([item[1] for item in current_group])
            avg_confidence = sum([item[2] for item in current_group]) / len(current_group)
            merged.append((merged_text, avg_confidence))
        
        return merged
        