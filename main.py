from visual.SceneSegmenter import SceneSegmenter
from OCR.OCR import OCR
from audio.ASR import ASR
from visual.ObjectDetector import ObjectDetector
from visual.VRD import VRD
from audio.SentenceSegmenter import SentenceSegmentation
from Transformer import Transformer
from Storage.ChromaDBVectorStore import ChromaDBVectorStore
from Storage.SQL.Repositories.VideoRepository import VideoRepository
from Storage.SQL.Repositories.VRDRepository import VRDRepository
from Storage.SQL.Repositories.ObjectRepository import ObjectRepository
import os
import sys
import gc
import torch
import argparse

def run(args):
    videos_folder = args.videos_folder
    video_path = os.path.join(videos_folder, args.video_path)

    json_folder = args.json_folder
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    # ── 0. Register video in Postgres ────────────────────────────────────────────
    print("Registering video in database...")
    video_repo = VideoRepository()
    video = video_repo.get_video_by_path(video_path)
    if video is None:
        video = video_repo.create_video(
            file_name=os.path.basename(video_path),
            file_path=video_path,
        )
    video_id = video.id
    print(f"Video ID: {video_id}")
    video_repo.close()

    # ── 1. Visual Processing ──────────────────────────────────────────────────────
    print("Video segmentation processing...")
    scene_segmenter = SceneSegmenter(video_path)
    scenes = scene_segmenter.segment_video()
    print(f"Detected {len(scenes)} scenes")

    frames = scene_segmenter.get_frames_with_metadata()
    print(f"Extracted {len(frames)} frames total")

    del scene_segmenter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # OCR
    print("\nStarting OCR processing...")
    ocr_processor = OCR(frames, video_path)
    ocr_processor.process_frames()
    ocr_index_path = os.path.join(json_folder, args.ocr_output_path)
    ocr_processor.save_inverted_index(ocr_index_path)
    ocr_inverted_index = ocr_processor.get_inverted_index()

    del ocr_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Object detection → Postgres
    print("\nObject detection processing...")
    object_detector = ObjectDetector(video_path, frames, model_name='yolo26l.pt')
    object_detector.detect_objects()
    object_index_path = os.path.join(json_folder, args.object_output_path)
    object_detector.save_inverted_index(object_index_path)
    object_inverted_index = object_detector.get_inverted_index()

    print("Saving object detection results to Postgres...")
    object_repo = ObjectRepository()
    object_repo.save_from_inverted_index(object_inverted_index, video_id)

    del object_detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # VRD → Postgres
    print("\nVisual relationship detection processing...")
    vrd_processor = VRD(frames=frames, video_path=video_path, model_id='llava-hf/llava-1.5-7b-hf')
    vrd_processor.detect_relationships()
    vrd_index_path = os.path.join(json_folder, args.vrd_output_path)
    vrd_processor.save_inverted_index(vrd_index_path)
    vrd_inverted_index = vrd_processor.get_inverted_index()

    print("Saving VRD results to Postgres...")
    vrd_repo = VRDRepository()
    vrd_repo.save_from_inverted_index(vrd_inverted_index, video_id)

    del vrd_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── 2. Audio Processing ───────────────────────────────────────────────────────
    print("\nAudio transcription processing...")
    asr_processor = ASR(video_path=video_path, model_name='openai/whisper-large-v3')
    asr_processor.transcribe(task="translate")
    asr_processor.save_transcription(os.path.join(json_folder, args.transcription_output_path))

    del asr_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── 3. Sentence Segmentation ──────────────────────────────────────────────────
    print("\nSentence segmentation processing...")
    seg_processor = SentenceSegmentation(
        video_path=video_path,
        transcript_json=os.path.join(json_folder, args.transcription_output_path),
        similarity_threshold=0.75,
    )
    seg_processor.segment()
    seg_processor.save_segments(os.path.join(json_folder, args.segmented_transcript_path))
    transcript_segments = seg_processor.segments

    del seg_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── 4. Embed OCR + Transcript → ChromaDB ─────────────────────────────────────
    print("\nEmbedding OCR and transcript results...")
    transformer = Transformer(ocr_inverted_index, transcript_segments, model_id='all-MiniLM-L6-v2')
    transformer.transform()
    transformer.save_embeddings(ChromaDBVectorStore())

    print(f"Generated {len(transformer.get_embeddings())} total embeddings")
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the video processing pipeline")
    parser.add_argument("--videos-folder", default="videos",
                        help="Path to videos folder")
    parser.add_argument("--video-path", default="1.mp4",
                        help="Path to the video file")
    parser.add_argument("--json-folder", default="json_outputs",
                        help="Directory to save intermediate JSON outputs")
    parser.add_argument("--ocr-output-path", default="ocr_inverted_index.json",
                        help="Path to save OCR inverted index JSON")
    parser.add_argument("--object-output-path", default="object_inverted_index.json",
                        help="Path to save object detection inverted index JSON")
    parser.add_argument("--vrd-output-path", default="vrd_inverted_index.json",
                        help="Path to save VRD inverted index JSON")
    parser.add_argument("--transcription-output-path", default="transcription.json",
                        help="Path to save ASR transcription JSON")
    parser.add_argument("--segmented-transcript-path", default="segmented_transcript.json",
                        help="Path to save segmented transcript JSON")
    args = parser.parse_args()
    run(args)