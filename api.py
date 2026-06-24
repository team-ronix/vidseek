import gc
import os
import sys
import time
import uuid
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import OCR.utils.ovo_svm as ovo_svm
sys.modules["ovo_svm"] = ovo_svm

import torch

from Transformer import Transformer
from HybridRetriever import HybridRetriever
from Storage.CustomVectorStore import CustomVectorStore
from Storage.HNSW import HNSWVectorStore
from Storage.SQL.Repositories.VideoRepository import VideoRepository
from Storage.SQL.Repositories.VRDRepository import VRDRepository
from Storage.SQL.Repositories.ObjectRepository import ObjectRepository
from Storage.SQL.Repositories.OCRRepository import OCRRepository
from Storage.SQL.Repositories.TranscriptRepository import TranscriptRepository
from Storage.SQL.DatabaseClient import SessionLocal
from Storage.SQL.Models.Object import Object as ObjectModel
from Storage.SQL.Models.ObjectVideo import ObjectVideo
from Storage.SQL.Models.VRDSubject import VRDSubject
from Storage.SQL.Models.VRDPredicate import VRDPredicate
from Storage.SQL.Models.VRDObject import VRDObject
from Storage.SQL.Models.VRDVideo import VRDVideo
from Storage.SQL.Models.Video import Video as VideoModel
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

_jobs: dict[str, dict] = {}


# -- Pydantic schemas --------------------------------------------------------

class SearchResult(BaseModel):
    type: str
    text: str
    video_path: str
    start_time: float
    frame_time: Optional[float] = None
    end_time: float
    sim: float
    source_model: str = "transformer"

class VideoGroup(BaseModel):
    video_path: str
    video_name: str
    match_count: int
    results: list[SearchResult]

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    videos: list[VideoGroup]
    latency_ms: dict[str, float] = {}

class GroupedResponse(BaseModel):
    videos: list[VideoGroup]
    total_results: int


def _group_by_video(results: list[SearchResult]) -> list[VideoGroup]:
    groups: dict[str, dict] = {}
    for r in results:
        vp = r.video_path
        if vp not in groups:
            groups[vp] = {"video_path": vp, "video_name": Path(vp).name,
                          "match_count": 0, "results": []}
        groups[vp]["match_count"] += 1
        groups[vp]["results"].append(r)
    return [VideoGroup(**g)
            for g in sorted(groups.values(), key=lambda g: g["match_count"], reverse=True)]


# -- Processing pipeline -----------------------------------------------------

def _run_pipeline(job_id: str, video_path: str, video_id: int,
                  detector: str = "craft", recognizer: str = "easyocr"):
    def update(msg: str, status: str = "running"):
        _jobs[job_id] = {"status": status, "message": msg}

    try:
        json_folder = "./json_outputs"
        os.makedirs(json_folder, exist_ok=True)

        update("Scene segmentation...")
        segmenter = SceneSegmenter(video_path)
        segmenter.segment_video()
        frames = segmenter.get_frames_with_metadata()
        print(f"Extracted {len(frames)} frames from {video_path}")
        del segmenter; gc.collect()

        update("OCR processing...")
        ocr = OCR(frames, video_path, video_id, detector=detector, recognizer=recognizer)
        ocr.process_frames()
        ocr_index = ocr.get_inverted_index()
        del ocr; gc.collect()

        update("Object detection...")
        obj_det = ObjectDetector(video_path, frames)
        obj_det.detect_objects()
        ObjectRepository().save_from_inverted_index(obj_det.get_inverted_index(), video_id)
        del obj_det; gc.collect()

        update("Visual relationship detection...")
        vrd = VRD(frames=frames, video_path=video_path, api_key=os.getenv("GEMINI_TOKEN") or "")
        vrd.detect_relationships()
        VRDRepository().save_from_inverted_index(vrd.get_inverted_index(), video_id)
        del vrd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        update("Audio transcription...")
        asr = ASR(video_path=video_path, model_name="openai/whisper-small")
        asr.transcribe(task="translate")
        transcription_path = os.path.join(json_folder, f"{video_id}_transcription.json")
        asr.save_transcription(transcription_path)
        del asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        update("Sentence segmentation...")
        seg = SentenceSegmentation(video_path=video_path, transcript_json=transcription_path)
        transcript_segments = seg.segment()
        transcript_repo = TranscriptRepository()
        transcript_repo.save_segments(transcript_segments, video_id)
        transcript_repo.close()
        del seg; gc.collect()

        update("Embedding (Transformer)...")
        transformer = Transformer(ocr_index, transcript_segments)
        transformer.transform()
        hnsw_store = HNSWVectorStore(source="transformer")
        transformer.save_embeddings(hnsw_store)
        hnsw_store.commit()
        del transformer, hnsw_store; gc.collect()

        update("Embedding (HybridEmbedder)...")
        hybrid = HybridRetriever(ocr_index, transcript_segments)
        hybrid.transform()
        hybrid_store = HNSWVectorStore(
            index_root="./data/hybrid_hnsw_index",
            dimension=128,
            source="hybrid",
        )
        hybrid.save_embeddings(hybrid_store)
        hybrid_store.commit()
        del hybrid, hybrid_store; gc.collect()

        update("Done", status="done")

    except Exception as e:
        _jobs[job_id] = {"status": "error", "message": str(e)}
        raise


# -- Search ------------------------------------------------------------------

@app.get("/search", response_model=SearchResponse)
def search(q: str, top_k: int = 10, model: str = "transformer"):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if model not in ("transformer", "hybrid", "both"):
        raise HTTPException(status_code=400, detail="model must be transformer | hybrid | both")

    results: list[SearchResult] = []
    latencies: dict[str, float] = {}

    if model in ("transformer", "both"):
        t0 = time.time()
        try:
            emb = Transformer({}, []).transform_single_text(q).tolist()
            _, metas, sims = HNSWVectorStore(source="transformer").query(emb, top_k=top_k)
            for meta, sim in zip(metas[0], sims[0]):
                if sim > 0.3:
                    results.append(SearchResult(
                        type=meta.get("type", "unknown"),
                        text=meta.get("text", ""),
                        video_path=meta.get("video_path", ""),
                        start_time=float(meta.get("start_time", 0)),
                        end_time=float(meta.get("end_time", 0)),
                        sim=float(sim),
                        source_model="transformer",
                    ))
        except Exception as e:
            print(f"Transformer search error: {e}")
        latencies["transformer"] = round((time.time() - t0) * 1000, 1)

    if model in ("hybrid", "both"):
        t0 = time.time()
        try:
            hits = HybridRetriever({}, []).query_hybrid(q, top_k=top_k)
            for hybrid_score, meta in hits:
                if hybrid_score > 0.05:
                    results.append(SearchResult(
                        type=meta.get("type", "unknown"),
                        text=meta.get("text", ""),
                        video_path=meta.get("video_path", ""),
                        start_time=float(meta.get("start_time", 0)),
                        end_time=float(meta.get("end_time", 0)),
                        sim=round(hybrid_score, 4),
                        source_model="hybrid",
                    ))
        except Exception as e:
            print(f"Hybrid search error: {e}")
        latencies["hybrid"] = round((time.time() - t0) * 1000, 1)

    results.sort(key=lambda r: r.sim, reverse=True)
    return SearchResponse(query=q, results=results,
                          videos=_group_by_video(results), latency_ms=latencies)


# -- Upload ------------------------------------------------------------------

@app.post("/videos/upload")
async def upload_video(
    file: UploadFile = File(...),
    detector: str = Form("craft"),
    recognizer: str = Form("easyocr"),
):
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")
    if detector not in ("east", "craft"):
        raise HTTPException(status_code=400, detail="detector must be 'east' or 'craft'")
    if recognizer not in ("mser", "easyocr"):
        raise HTTPException(status_code=400, detail="recognizer must be 'mser' or 'easyocr'")

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
        target=_run_pipeline,
        args=(job_id, str(dest), video_id, detector, recognizer),
        daemon=True,
    ).start()
    return JSONResponse({"job_id": job_id, "video_id": video_id})


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# -- Structured search -------------------------------------------------------

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
            "subjects":  [r.key for r in db.query(VRDSubject).order_by(VRDSubject.key).all()],
            "relations": [r.key for r in db.query(VRDPredicate).order_by(VRDPredicate.key).all()],
            "objects":   [r.key for r in db.query(VRDObject).order_by(VRDObject.key).all()],
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
        return [{"type": "ocr", "text": word.word, "video_path": word.video.file_path,
                 "video_name": word.video.file_name, "start_time": word.start_time,
                 "end_time": word.end_time, "score": confidence}]
    finally:
        repo.close()


@app.get("/search/object", response_model=GroupedResponse)
def search_by_object(key: str):
    db = SessionLocal()
    try:
        rows = (
            db.query(ObjectVideo, ObjectModel, VideoModel)
            .join(ObjectModel, ObjectVideo.object_id == ObjectModel.id)
            .join(VideoModel,  ObjectVideo.video_id  == VideoModel.id)
            .filter(ObjectModel.key == key).all()
        )
        results = [
            SearchResult(type="object", text=obj.key, video_path=video.file_path,
                         frame_time=float(ov.frame_time or 0),
                         start_time=float(ov.start_time or 0),
                         end_time=float(ov.end_time or 0), sim=0.0)
            for ov, obj, video in rows
        ]
        return GroupedResponse(videos=_group_by_video(results), total_results=len(results))
    finally:
        db.close()


@app.get("/search/vrd", response_model=GroupedResponse)
def search_by_vrd(
    subject:  Optional[str] = None,
    object:   Optional[str] = None,
    relation: Optional[str] = None,
):
    db = SessionLocal()
    try:
        q = (
            db.query(VRDVideo, VRDSubject, VRDPredicate, VRDObject, VideoModel)
            .join(VRDSubject,   VRDVideo.subject_id   == VRDSubject.id)
            .join(VRDPredicate, VRDVideo.predicate_id == VRDPredicate.id)
            .join(VRDObject,    VRDVideo.object_id    == VRDObject.id)
            .join(VideoModel,   VRDVideo.video_id     == VideoModel.id)
        )
        if subject:  q = q.filter(VRDSubject.key   == subject)
        if relation: q = q.filter(VRDPredicate.key == relation)
        if object:   q = q.filter(VRDObject.key    == object)
        results = [
            SearchResult(type="vrd", text=f"{subj.key} - {pred.key} - {obj.key}",
                         video_path=video.file_path,
                         start_time=float(vrd.start_time or 0),
                         end_time=float(vrd.end_time or 0),
                         frame_time=float(vrd.frame_time or 0), sim=0.0)
            for vrd, subj, pred, obj, video in q.all()
        ]
        return GroupedResponse(videos=_group_by_video(results), total_results=len(results))
    finally:
        db.close()


# -- Chapters ----------------------------------------------------------------

@app.get("/videos/chapters")
def get_chapters(path: str):
    db = SessionLocal()
    try:
        from Storage.SQL.Models.TranscriptSegment import TranscriptSegment as TSModel
        video = db.query(VideoModel).filter(VideoModel.file_path == path).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        rows = (db.query(TSModel)
                .filter(TSModel.video_id == video.id)
                .order_by(TSModel.start).all())
        return [{"id": r.id, "title": r.title, "start": r.start,
                 "end": r.end, "text": r.text} for r in rows]
    finally:
        db.close()


# -- Video stream ------------------------------------------------------------

@app.get("/video/stream")
def stream_video(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = Path(".") / p
    p = p.resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    media_types = {".mp4": "video/mp4", ".webm": "video/webm",
                   ".avi": "video/x-msvideo", ".mov": "video/quicktime",
                   ".mkv": "video/x-matroska"}
    return FileResponse(str(p), media_type=media_types.get(p.suffix.lower(), "video/mp4"))
