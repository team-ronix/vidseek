import os
import uuid
import asyncio
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
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
from OCR.OCR import OCR
from visual.ObjectDetector import ObjectDetector
from visual.VRD import VRD
from audio.ASR import ASR
from audio.SentenceSegmenter import SentenceSegmentation

UPLOAD_DIR = Path("./videos")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VidSeek API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────────────────
# { job_id: { "status": "pending"|"running"|"done"|"error", "message": str } }
_jobs: dict[str, dict] = {}


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class SearchResult(BaseModel):
    type: str           # "ocr" | "transcript" | "object" | "vrd"
    text: str
    video_path: str
    start_time: float
    end_time: float
    score: float        # lower = more similar for ChromaDB; 1.0 for SQL hits


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _merge_results(
    chroma_ids: list,
    chroma_metadatas: list,
    chroma_distances: list,
    sql_results: list[dict],
    video_repo: VideoRepository,
) -> list[SearchResult]:
    merged: list[SearchResult] = []

    # ChromaDB results (OCR + transcript)
    for id_, meta, dist in zip(chroma_ids[0], chroma_metadatas[0], chroma_distances[0]):
        merged.append(
            SearchResult(
                type=meta.get("type", "unknown"),
                text=meta.get("text", ""),
                video_path=meta.get("video_path", ""),
                start_time=float(meta.get("start_time", 0)),
                end_time=float(meta.get("end_time", 0)),
                score=float(dist),
            )
        )

    # SQL results (object + VRD) — fetch video path from DB
    for row in sql_results:
        video_path = row.get("video_path", "")
        if not video_path and row.get("video_id"):
            video = video_repo.get_video_by_id(row["video_id"])
            video_path = video.file_path if video else ""
        merged.append(
            SearchResult(
                type=row["type"],
                text=row["text"],
                video_path=video_path,
                start_time=row.get("start_time", 0.0),
                end_time=row.get("end_time", 0.0),
                score=1.0,
            )
        )

    # Sort: lower score (ChromaDB distance) first, SQL hits appended after
    merged.sort(key=lambda r: r.score)
    return merged


def _run_pipeline(job_id: str, video_path: str, video_id: int):
    def update(msg: str, status: str = "running"):
        _jobs[job_id] = {"status": status, "message": msg}
        print(f"[job {job_id}] {msg}")

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
        obj_det = ObjectDetector(video_path, frames, model_name="yolo26l.pt")
        obj_det.detect_objects()
        obj_index = obj_det.get_inverted_index()
        ObjectRepository().save_from_inverted_index(obj_index, video_id)
        del obj_det; gc.collect()

        update("Visual relationship detection...")
        vrd = VRD(frames=frames, video_path=video_path, model_id="llava-hf/llava-1.5-7b-hf")
        vrd.detect_relationships()
        vrd_index = vrd.get_inverted_index()
        VRDRepository().save_from_inverted_index(vrd_index, video_id)
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/search", response_model=SearchResponse)
def search(q: str, top_k: int = 10):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    transformer = Transformer({}, [], model_id="all-MiniLM-L6-v2")
    embedding = transformer.transform_single_text(q)

    # ChromaDB: OCR + transcript
    try:
        chroma = ChromaDBVectorStore()
        ids, metadatas, distances = chroma.query(embedding, top_k=top_k)
    except Exception:
        ids, metadatas, distances = [[]], [[]], [[]]

    # Postgres: objects + VRD
    sql_results: list[dict] = []
    try:
        sql_results += ObjectRepository().search(q)
        sql_results += VRDRepository().search(q)
    except Exception:
        pass

    video_repo = VideoRepository()
    results = _merge_results(ids, metadatas, distances, sql_results, video_repo)
    video_repo.close()

    return SearchResponse(query=q, results=results)


@app.get("/video/{video_id}")
def stream_video(video_id: int, request_range: Optional[str] = None):
    """Stream a video file with HTTP range request support for seeking."""
    video_repo = VideoRepository()
    video = video_repo.get_video_by_id(video_id)
    video_repo.close()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(video.file_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    file_size = video_path.stat().st_size
    content_type = "video/mp4"

    # Full file (no range)
    def iter_file():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type=content_type,
        headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"},
    )


@app.get("/video/{video_id}/stream")
async def stream_video_range(video_id: int, request: "Request"):  # type: ignore[name-defined]
    """Range-aware video streaming endpoint for HTML5 <video> seeking."""
    video_repo = VideoRepository()
    video = video_repo.get_video_by_id(video_id)
    video_repo.close()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(video.file_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        range_val = range_header.replace("bytes=", "")
        start_str, end_str = range_val.split("-")
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iter_range():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(1024 * 1024, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )

    def iter_full():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        iter_full(),
        media_type="video/mp4",
        headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"},
    )


@app.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save uploaded video and kick off the ingestion pipeline asynchronously."""
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    # Register in DB
    video_repo = VideoRepository()
    video = video_repo.get_video_by_path(str(dest))
    if video is None:
        video = video_repo.create_video(file_name=file.filename, file_path=str(dest))
    video_id = video.id
    video_repo.close()

    # Create job and run pipeline in background thread
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "message": "Queued"}
    thread = threading.Thread(
        target=_run_pipeline, args=(job_id, str(dest), video_id), daemon=True
    )
    thread.start()

    return JSONResponse({"job_id": job_id, "video_id": video_id})


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/videos")
def list_videos():
    video_repo = VideoRepository()
    # Simple full scan — fine for moderate collections
    from Storage.SQL.Models.Video import Video as VideoModel
    from Storage.SQL.DatabaseClient import SessionLocal
    db = SessionLocal()
    videos = db.query(VideoModel).all()
    db.close()
    video_repo.close()
    return [{"id": v.id, "file_name": v.file_name, "file_path": v.file_path} for v in videos]


# ── Object & VRD option endpoints ─────────────────────────────────────────────

@app.get("/objects")
def list_objects():
    """Return all distinct detected objects in the database."""
    from Storage.SQL.Models.Object import Object as ObjectModel
    db = SessionLocal()
    try:
        rows = db.query(ObjectModel).order_by(ObjectModel.key).all()
        return [{"id": r.id, "key": r.key} for r in rows]
    finally:
        db.close()


@app.get("/vrd/options")
def list_vrd_options():
    """Return all distinct subjects, predicates (relations), and objects from VRD."""
    from Storage.SQL.Models.VRDSubject   import VRDSubject
    from Storage.SQL.Models.VRDPredicate import VRDPredicate
    from Storage.SQL.Models.VRDObject    import VRDObject
    db = SessionLocal()
    try:
        subjects  = [r.key for r in db.query(VRDSubject).order_by(VRDSubject.key).all()]
        relations = [r.key for r in db.query(VRDPredicate).order_by(VRDPredicate.key).all()]
        objects   = [r.key for r in db.query(VRDObject).order_by(VRDObject.key).all()]
        return {"subjects": subjects, "relations": relations, "objects": objects}
    finally:
        db.close()


@app.get("/search/object")
def search_by_object(key: str):
    """Return all video scenes where a given object appears."""
    from Storage.SQL.Models.Object      import Object as ObjectModel
    from Storage.SQL.Models.ObjectVideo import ObjectVideo
    from Storage.SQL.Models.Video       import Video as VideoModel
    db = SessionLocal()
    try:
        rows = (
            db.query(ObjectVideo, ObjectModel, VideoModel)
            .join(ObjectModel,  ObjectVideo.object_id == ObjectModel.id)
            .join(VideoModel,   ObjectVideo.video_id  == VideoModel.id)
            .filter(ObjectModel.key == key)
            .all()
        )
        return [
            {
                "type": "object",
                "text": obj.key,
                "video_path": video.file_path,
                "video_name": video.file_name,
                "start_time": obj_vid.start_time,
                "end_time":   obj_vid.end_time,
            }
            for obj_vid, obj, video in rows
        ]
    finally:
        db.close()


@app.get("/search/vrd")
def search_by_vrd(
    subject:  Optional[str] = None,
    object:   Optional[str] = None,
    relation: Optional[str] = None,
):
    """Return all video scenes matching a VRD triple (any combination of subject/object/relation)."""
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
        rows = q.all()
        return [
            {
                "type":         "vrd",
                "subject":      subj.key,
                "predicate":    pred.key,
                "object":       obj.key,
                "text":         f"{subj.key} — {pred.key} — {obj.key}",
                "video_path":   video.file_path,
                "video_name":   video.file_name,
                "start_time":   vrd_vid.start_time,
                "end_time":     vrd_vid.end_time,
            }
            for vrd_vid, subj, pred, obj, video in rows
        ]
    finally:
        db.close()
