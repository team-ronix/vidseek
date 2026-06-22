import os
import sys
import uuid
import threading
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import OCR.utils.ovo_svm as ovo_svm

sys.modules["ovo_svm"] = ovo_svm

from Transformer import Transformer
from Storage.CustomVectorStore import CustomVectorStore
from Storage.SQL.Repositories.VideoRepository import VideoRepository
from Storage.SQL.Repositories.VRDRepository import VRDRepository
from Storage.SQL.Repositories.ObjectRepository import ObjectRepository
from Storage.SQL.Repositories.OCRRepository import OCRRepository
from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.Object import Object as ObjectModel
from Storage.SQL.Models.ObjectVideo import ObjectVideo
from Storage.SQL.Models.VRDSubject import VRDSubject
from Storage.SQL.Models.VRDPredicate import VRDPredicate
from Storage.SQL.Models.VRDObject import VRDObject
from Storage.SQL.Models.VRDVideo import VRDVideo
from Storage.SQL.Models.Video import Video as VideoModel
import gc
import torch
from visual.SceneSegmenter import SceneSegmenter
from OCR.src.OCR import OCR
from visual.faster_rcnn.ObjectDetector import ObjectDetector
from visual.VRD import VRD
from audio.ASR import ASR
from audio.SentenceSegmenter import SentenceSegmentation

UPLOAD_DIR = Path("./videos")
UPLOAD_DIR.mkdir(exist_ok=True)
load_dotenv()

app = FastAPI(title="VidSeek API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# { job_id: { "status": "pending"|"running"|"done"|"error", "message": str } }
_jobs: dict[str, dict] = {}


# Pydantic schemas

class SearchResult(BaseModel):
    type: str           # "ocr" | "transcript" | "object" | "vrd"
    text: str
    video_path: str
    start_time: float
    frame_time: Optional[float] = None
    end_time: float
    sim: float         # similarity score; higher is better

class VideoGroup(BaseModel):
    video_path: str
    video_name: str
    match_count: int
    results: list[SearchResult]

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    videos: list[VideoGroup]

class GroupedResponse(BaseModel):
    videos: list[VideoGroup]
    total_results: int

def _group_by_video(results: list[SearchResult]) -> list[VideoGroup]:
    groups: dict[str, dict] = {}
    for r in results:
        vp = r.video_path
        if vp not in groups:
            groups[vp] = {
                "video_path": vp,
                "video_name": Path(vp).name,
                "match_count": 0,
                "results": [],
            }
        groups[vp]["match_count"] += 1
        groups[vp]["results"].append(r)
    return [
        VideoGroup(**g)
        for g in sorted(groups.values(), key=lambda g: g["match_count"], reverse=True)
    ]

def _run_pipeline(job_id: str, video_path: str, video_id: int):
    def update(msg: str, status: str = "running"):
        _jobs[job_id] = {"status": status, "message": msg}

    try:
        json_folder = "./json_outputs"
        os.makedirs(json_folder, exist_ok=True)

        update("Scene segmentation...")
        segmenter = SceneSegmenter(video_path)
        segmenter.segment_video()
        frames = segmenter.get_frames_with_metadata()
        del segmenter; gc.collect()

        update("OCR processing...")
        ocr = OCR(frames, video_path, video_id)
        ocr.process_frames()
        ocr_index = ocr.get_inverted_index()
        del ocr; gc.collect()

        update("Object detection...")
        obj_det = ObjectDetector(video_path, frames)
        obj_det.detect_objects()
        ObjectRepository().save_from_inverted_index(obj_det.get_inverted_index(), video_id)
        del obj_det; gc.collect()

        update("Visual relationship detection...")
        vrd = VRD(frames=frames, video_path=video_path, api_key=os.getenv("GEMINI_TOKEN"))
        vrd.detect_relationships()
        VRDRepository().save_from_inverted_index(vrd.get_inverted_index(), video_id)
        del vrd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        update("Audio transcription...")
        asr = ASR(video_path=video_path, model_name="openai/whisper-large-v3")
        asr.transcribe(task="translate")
        transcription_path = os.path.join(json_folder, f"{video_id}_transcription.json")
        asr.save_transcription(transcription_path)
        del asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        update("Sentence segmentation...")
        seg = SentenceSegmentation(
            video_path=video_path,
            transcript_json=transcription_path,
            similarity_threshold=0.75,
        )
        seg.segment()
        transcript_segments = seg.segments
        del seg; gc.collect()

        update("Embedding and storing...")
        transformer = Transformer(ocr_index, transcript_segments)
        transformer.transform()
        transformer.save_embeddings(CustomVectorStore())

        update("Done", status="done")

    except Exception as e:
        _jobs[job_id] = {"status": "error", "message": str(e)}
        raise

@app.get("/search", response_model=SearchResponse)
def search(q: str, top_k: int = 10):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    transformer = Transformer({}, [])
    embedding = transformer.transform_single_text(q)
    embedding = embedding.tolist()
    try:
        ids, metadatas, flat_similarities = CustomVectorStore().query(embedding, top_k=top_k)
        vector_results = [
            SearchResult(
                type=meta.get("type", "unknown"),
                text=meta.get("text", ""),
                video_path=meta.get("video_path", ""),
                start_time=float(meta.get("start_time", 0)),
                end_time=float(meta.get("end_time", 0)),
                sim=float(sim),
            )
            for _, meta, sim in zip(ids[0], metadatas[0], flat_similarities[0])
            if sim > 0.1
        ]
    except Exception as e:
        print(f"Error occurred while querying ChromaDB: {e}")
        vector_results = []
        
    return SearchResponse(query=q, results=vector_results, videos=_group_by_video(vector_results))

@app.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    video_repo = VideoRepository()
    video = video_repo.get_video_by_path(str(dest)) or \
            video_repo.create_video(file_name=file.filename, file_path=str(dest))
    video_id = video.id
    video_repo.close()

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "message": "Queued"}
    threading.Thread(
        target=_run_pipeline, args=(job_id, str(dest), video_id), daemon=True
    ).start()

    return JSONResponse({"job_id": job_id, "video_id": video_id})


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# Object & VRD option endpoints

@app.get("/objects")
def list_objects():
    db = SessionLocal()
    try:
        rows = db.query(ObjectModel).order_by(ObjectModel.key).all()
        return [{"id": r.id, "key": r.key} for r in rows]
    finally:
        db.close()


@app.get("/vrd/options")
def list_vrd_options():
    
    db = SessionLocal()
    try:
        return {
            "subjects": [r.key for r in db.query(VRDSubject).order_by(VRDSubject.key).all()],
            "relations": [r.key for r in db.query(VRDPredicate).order_by(VRDPredicate.key).all()],
            "objects": [r.key for r in db.query(VRDObject).order_by(VRDObject.key).all()],
        }
    finally:
        db.close()


@app.get("/search/ocr")
def search_by_ocr(q: str, video_id: Optional[int] = None):
    repo = OCRRepository()
    try:
        word, confidence = repo.find_closest_word_video(q, video_id)
        if word is None:
            return []
        return [{
            "type": "ocr",
            "text": word.word,
            "video_path": word.video.file_path,
            "video_name": word.video.file_name,
            "start_time": word.start_time,
            "end_time": word.end_time,
            "score": confidence,
        }]
    finally:
        repo.close()



@app.get("/search/object", response_model=GroupedResponse)
def search_by_object(key: str):
    db = SessionLocal()
    try:
        rows = (
            db.query(ObjectVideo, ObjectModel, VideoModel)
            .join(ObjectModel, ObjectVideo.object_id == ObjectModel.id)
            .join(VideoModel, ObjectVideo.video_id == VideoModel.id)
            .filter(ObjectModel.key == key)
            .all()
        )
        results = [
            SearchResult(
                type="object",
                text=obj.key,
                video_path=video.file_path,
                frame_time=float(ov.frame_time or 0),
                start_time=float(ov.start_time or 0),
                end_time=float(ov.end_time or 0),
                sim=0.0,
            )
            for ov, obj, video in rows
        ]
        return GroupedResponse(videos=_group_by_video(results), total_results=len(results))
    finally:
        db.close()


@app.get("/search/vrd", response_model=GroupedResponse)
def search_by_vrd(
    subject: Optional[str] = None,
    object: Optional[str] = None,
    relation: Optional[str] = None,
):
    db = SessionLocal()
    try:
        q = (
            db.query(VRDVideo, VRDSubject, VRDPredicate, VRDObject, VideoModel)
            .join(VRDSubject, VRDVideo.subject_id == VRDSubject.id)
            .join(VRDPredicate, VRDVideo.predicate_id == VRDPredicate.id)
            .join(VRDObject, VRDVideo.object_id == VRDObject.id)
            .join(VideoModel, VRDVideo.video_id == VideoModel.id)
        )
        if subject: q = q.filter(VRDSubject.key == subject)
        if relation: q = q.filter(VRDPredicate.key == relation)
        if object: q = q.filter(VRDObject.key == object)
        results = [
            SearchResult(
                type="vrd",
                text=f"{subj.key} - {pred.key} - {obj.key}",
                video_path=video.file_path,
                start_time=float(vrd.start_time or 0),
                end_time=float(vrd.end_time or 0),
                sim=0.0,
            )
            for vrd, subj, pred, obj, video in q.all()
        ]
        return GroupedResponse(videos=_group_by_video(results), total_results=len(results))
    finally:
        db.close()


@app.get("/video/stream")
def stream_video(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = Path(".") / p
    p = p.resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    ext = p.suffix.lower()
    media_types = {
        ".mp4": "video/mp4", ".webm": "video/webm",
        ".avi": "video/x-msvideo", ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
    }
    return FileResponse(str(p), media_type=media_types.get(ext, "video/mp4"))
