from SceneSegmenter import SceneSegmenter
import os
import cv2
import easyocr
import json

videos_folder = './'
video_path = os.path.join(videos_folder, 'video2.mp4')

# Initialize inverted index: text -> list of occurrences
inverted_index = {}

scene_segmenter = SceneSegmenter(video_path)
scenes = scene_segmenter.segment_video()
print(f"Detected {len(scenes)} scenes:")

frames = scene_segmenter.get_frames_with_metadata()
print(f"\nExtracted {len(frames)} frames total")
print("Initializing OCR reader...")
reader = easyocr.Reader(['en'])
print("OCR reader ready")

# Create output directory for debug frames
debug_dir = os.path.join('.', 'debug_frames')
os.makedirs(debug_dir, exist_ok=True)
print(f"Saving annotated frames to: {debug_dir}")
print("\nProcessing frames...\n")

for i, frame_data in enumerate(frames, 1):
    frame_number = frame_data['frame_number']
    scene = frame_data['scene']
    frame_count = frame_data['frame_count_in_scene']
    frame = frame_data['frame']
    
    print(f"[{i}/{len(frames)}] Scene {scene.index}: Start at {scene.start_time:.2f}s, End at {scene.end_time:.2f}s, Duration {scene.duration:.2f}s ({frame_count} frames)")
    print(f"       Processing frame {frame_number}")
    
    if frame_number is None:
        continue
    
    if frame is not None:
        # Preprocess frame for OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Run OCR
        print("       Running OCR...")
        result = reader.readtext(blur)
        print(f"       Detected {len(result)} text regions")
        
        for detection in result:
            bbox, text, confidence = detection
            print(f"       - '{text}' (confidence: {confidence:.2f})")
            # Add to inverted index
            if text not in inverted_index:
                inverted_index[text] = []
            
            if text in inverted_index and any(occ['scene'] == scene.index for occ in inverted_index[text]):
                continue  # Skip if already exists in this scene

            inverted_index[text].append({
                'scene': scene.index,
                'frame': frame_number,
                'video_path': video_path,
                'start_time': scene.start_time,
                'end_time': scene.end_time,
                'confidence': confidence
            })
    else:
        print("       Warning: Frame data is None")

print(f"\nProcessing complete! Check {debug_dir}/ for annotated frames")

# Save inverted index to JSON
index_path = os.path.join('.', 'inverted_index.json')
with open(index_path, 'w', encoding='utf-8') as f:
    json.dump(inverted_index, f, indent=2, ensure_ascii=False)

print(f"\nInverted Index saved to: {index_path}")
print(f"Total unique texts detected: {len(inverted_index)}")
print("\nSample entries:")
for text, occurrences in list(inverted_index.items())[:5]:
    print(f"  '{text}' -> {len(occurrences)} occurrence(s)")
    for occ in occurrences[:2]:
        print(f"    - Scene {occ['scene']}, Frame {occ['frame']}, Time {occ['start_time']:.2f}s")