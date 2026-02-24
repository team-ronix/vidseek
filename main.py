from SceneSegmenter import SceneSegmenter
from OCR import OCR
from ASR import ASR
from ObjectDetector import ObjectDetector
from SentenceSegmenter import SentenceSegmentation
import os
import sys


videos_folder = './videos'
video_path = os.path.join(videos_folder, '1.mp4')

if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    sys.exit(1)

## Visual Processing

print(f"Video segmenation processing...")
scene_segmenter = SceneSegmenter(video_path)
scenes = scene_segmenter.segment_video()
print(f"Detected {len(scenes)} scenes:")

frames = scene_segmenter.get_frames_with_metadata()
print(f"\nExtracted {len(frames)} frames total")

print("Starting OCR processing...")
ocr_processor = OCR(frames, video_path)
ocr_processor.process_frames()
ocr_index_path = os.path.join('.', 'ocr_inverted_index.json')
ocr_processor.save_inverted_index(ocr_index_path)
ocr_inverted_index = ocr_processor.get_inverted_index()
      
      
print(f"\nObject detection processing...")
object_detector = ObjectDetector(video_path, frames, model_name='yolo26l.pt')
object_detector.detect_objects()
object_index_path = os.path.join('.', 'object_inverted_index.json')
object_detector.save_inverted_index(object_index_path)
object_inverted_index = object_detector.get_inverted_index()
      
### Audio Processing  

print(f"\nAudio transcription processing...")
asr_processor = ASR(video_path=video_path, model_name='openai/whisper-large-v3')
asr_processor.transcribe(task="translate")
transcription_result = asr_processor.get_text()
asr_processor.save_transcription('transcription.json')

## Sentence Segmentation

print(f"\nSentence segmentation processing...")
SentenceSegmentation_processor = SentenceSegmentation(transcript_json="transcription.json", similarity_threshold=0.75)
SentenceSegmentation_processor.segment()
SentenceSegmentation_processor.save_segments("segmented_transcript.json")
