import os
import uuid
import threading
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from Transformer import Transformer
from Storage.ChromaDBVectorStore import ChromaDBVectorStore
from Storage.SQL.Repositories.VideoRepository import VideoRepository
from Storage.SQL.Repositories.VRDRepository import VRDRepository
from Storage.SQL.Repositories.ObjectRepository import ObjectRepository
from Storage.SQL.DatabaseClient import SessionLocal

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


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    type: str           # "ocr" | "transcript" | "object" | "vrd"
    text: str
    video_path: str
    start_time: float
    end_time: float
    score: float        # distance for ChromaDB results; 1.0 for SQL results

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

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
        ocr = OCR(frames, video_path)
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
        transformer = Transformer(ocr_index, transcript_segments, model_id="all-MiniLM-L6-v2")
        transformer.transform()
        transformer.save_embeddings(ChromaDBVectorStore())

        update("Done", status="done")

    except Exception as e:
        _jobs[job_id] = {"status": "error", "message": str(e)}
        raise

@app.get("/search", response_model=SearchResponse)
def search(q: str, top_k: int = 10):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    transformer = Transformer({}, [], model_id="all-MiniLM-L6-v2")
    embedding = transformer.transform_single_text(q)

    try:
        ids, metadatas, distances = ChromaDBVectorStore().query(embedding, top_k=top_k)
        chroma_results = [
            SearchResult(
                type=meta.get("type", "unknown"),
                text=meta.get("text", ""),
                video_path=meta.get("video_path", ""),
                start_time=float(meta.get("start_time", 0)),
                end_time=float(meta.get("end_time", 0)),
                score=float(dist),
            )
            for _, meta, dist in zip(ids[0], metadatas[0], distances[0])
        ]
    except Exception:
        chroma_results = []

    # Postgres: object + VRD keyword hits
    sql_rows: list[dict] = []
    try:
        sql_rows += ObjectRepository().search(q)
        sql_rows += VRDRepository().search(q)
    except Exception:
        pass

    video_repo = VideoRepository()
    sql_results = []
    for row in sql_rows:
        video_path = row.get("video_path", "")
        if not video_path and row.get("video_id"):
            video = video_repo.get_video_by_id(row["video_id"])
            video_path = video.file_path if video else ""
        sql_results.append(SearchResult(
            type=row["type"],
            text=row["text"],
            video_path=video_path,
            start_time=float(row.get("start_time") or 0),
            end_time=float(row.get("end_time") or 0),
            score=1.0,
        ))
    video_repo.close()

    results = sorted(chroma_results + sql_results, key=lambda r: r.score)
    return SearchResponse(query=q, results=results)

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


# ── Object & VRD option endpoints ─────────────────────────────────────────────

@app.get("/objects")
def list_objects():
    from Storage.SQL.Models.Object import Object as ObjectModel
    db = SessionLocal()
    try:
        rows = db.query(ObjectModel).order_by(ObjectModel.key).all()
        return [{"id": r.id, "key": r.key} for r in rows]
    finally:
        db.close()


@app.get("/vrd/options")
def list_vrd_options():
    from Storage.SQL.Models.VRDSubject   import VRDSubject
    from Storage.SQL.Models.VRDPredicate import VRDPredicate
    from Storage.SQL.Models.VRDObject    import VRDObject
    db = SessionLocal()
    try:
        return {
            "subjects":  [r.key for r in db.query(VRDSubject).order_by(VRDSubject.key).all()],
            "relations": [r.key for r in db.query(VRDPredicate).order_by(VRDPredicate.key).all()],
            "objects":   [r.key for r in db.query(VRDObject).order_by(VRDObject.key).all()],
        }
    finally:
        db.close()


@app.get("/search/object")
def search_by_object(key: str):
    from Storage.SQL.Models.Object      import Object as ObjectModel
    from Storage.SQL.Models.ObjectVideo import ObjectVideo
    from Storage.SQL.Models.Video       import Video as VideoModel
    db = SessionLocal()
    try:
        rows = (
            db.query(ObjectVideo, ObjectModel, VideoModel)
            .join(ObjectModel, ObjectVideo.object_id == ObjectModel.id)
            .join(VideoModel,  ObjectVideo.video_id  == VideoModel.id)
            .filter(ObjectModel.key == key)
            .all()
        )
        return [
            {
                "type":       "object",
                "text":       obj.key,
                "video_path": video.file_path,
                "video_name": video.file_name,
                "start_time": ov.start_time,
                "end_time":   ov.end_time,
            }
            for ov, obj, video in rows
        ]
    finally:
        db.close()


@app.get("/search/vrd")
def search_by_vrd(
    subject:  Optional[str] = None,
    object:   Optional[str] = None,
    relation: Optional[str] = None,
):
    from Storage.SQL.Models.VRDSubject   import VRDSubject
    from Storage.SQL.Models.VRDPredicate import VRDPredicate
    from Storage.SQL.Models.VRDObject    import VRDObject
    from Storage.SQL.Models.VRDVideo     import VRDVideo
    from Storage.SQL.Models.Video        import Video as VideoModel
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
        return [
            {
                "type":       "vrd",
                "text":       f"{subj.key} — {pred.key} — {obj.key}",
                "video_path": video.file_path,
                "video_name": video.file_name,
                "start_time": vrd.start_time,
                "end_time":   vrd.end_time,
            }
            for vrd, subj, pred, obj, video in q.all()
        ]
    finally:
        db.close()
