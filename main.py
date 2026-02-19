from SceneSegmenter import SceneSegmenter
from OCR import OCR
from ASR import ASR
import os


videos_folder = './'
video_path = os.path.join(videos_folder, 'video.mp4')

### Visual Processing

scene_segmenter = SceneSegmenter(video_path)
scenes = scene_segmenter.segment_video()
print(f"Detected {len(scenes)} scenes:")

frames = scene_segmenter.get_frames_with_metadata()
print(f"\nExtracted {len(frames)} frames total")

ocr_processor = OCR(frames, video_path)
ocr_processor.process_frames()
index_path = os.path.join('.', 'inverted_index.json')
ocr_processor.save_inverted_index(index_path)
inverted_index = ocr_processor.get_inverted_index()

print(f"\nInverted Index saved to: {index_path}")
print(f"Total unique texts detected: {len(inverted_index)}")
print("\nSample entries:")
for text, occurrences in list(inverted_index.items())[:5]:
    print(f"  '{text}' -> {len(occurrences)} occurrence(s)")
    for occ in occurrences[:2]:
        print(f"    - Scene {occ['scene']}, Frame {occ['frame']}, Time {occ['start_time']:.2f}s")
      
      
### Audio Processing  

asr_processor = ASR(video_path=video_path, model_name='openai/whisper-large-v3')
asr_processor.transcribe()
transcription_result = asr_processor.get_text()
asr_processor.save_transcription('transcription.json')
