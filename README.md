# VidSeek

VidSeek is a multimodal video search engine that lets you query video content using natural language. It ingests videos through a multi-stage pipeline - extracting text via OCR, speech via automatic speech recognition, objects via computer vision, and visual relationships between objects - then indexes everything into a searchable vector store backed by PostgreSQL. A React frontend and FastAPI backend expose the full search and upload experience.

---

## Table of Contents

- [Demo](#demo-video)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Model & Data Files](#model--data-files)
- [Pipeline Phases](#pipeline-phases)
- [Run Vision Models Separately](#run-vision-models-separately)
- [Running the API Server](#running-the-api-server)
- [Running the Frontend](#running-the-frontend)
- [Quick-start Summary](#quick-start-summary)

---

## Demo Video

[![Demo](demo.png)](demo.mp4)

## Architecture

```
Video file
    │
    ▼
Scene Segmentation  ──►  Keyframes
    │
    ├──► OCR (CRAFT detector + EasyOCR)         ──► PostgreSQL
    │
    ├──► Object Detection (Faster R-CNN / HOG)  ──► PostgreSQL
    │
    ├──► Visual Relationship Detection (VRD)    ──► PostgreSQL
    │
    ├──► ASR (Whisper-small)
    │        │
    │        └──► Sentence Segmentation
    │                   │
    │                   ├──► Custom Transformer (384-dim)  ──► HNSW index
    │                   └──► Hybrid Embedder (BM25+dense)  ──► HNSW index
    │
    └──► FastAPI ◄──── React UI
```

---

## Prerequisites

Install these system tools before anything else.

| Tool                                                   | Version     | Purpose                     |
| ------------------------------------------------------ | ----------- | --------------------------- |
| Python                                                 | 3.12 – 3.14 | Backend runtime             |
| [Poetry](https://python-poetry.org/docs/#installation) | ≥ 2.0       | Python dependency manager   |
| Node.js + npm                                          | ≥ 18        | Frontend                    |
| [FFmpeg](https://ffmpeg.org/download.html)             | any recent  | Audio extraction for ASR    |
| PostgreSQL                                             | ≥ 14        | Structured data storage     |
| CUDA (optional)                                        | ≥ 11.8      | GPU acceleration for models |

> **Windows users:** Install FFmpeg and make sure `ffmpeg` is on your `PATH`. The easiest way is `winget install ffmpeg` or via [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/).

---

## Installation

### 1. Install Python dependencies

```bash
poetry install
```

This creates a `.venv/` inside the project and installs all 34 production dependencies (PyTorch, Transformers, FastAPI, SQLAlchemy, EasyOCR, Ultralytics, Whisper, etc.).

### 2. Install frontend dependencies

```bash
cd vidseek-ui
npm install
cd ..
```

---

## Configuration

Copy the environment template and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# PostgreSQL connection string
DATABASE_URL=postgresql+psycopg2://<user>:<password>@<host>:<port>/<dbname>?sslmode=require

# HuggingFace token - needed to download Whisper and other models
HF_TOKEN=hf_...

```

> Get a free HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Database Setup

### Option A - Hosted PostgreSQL (recommended for quick start)

Use any hosted PostgreSQL service (Aiven, Supabase, Neon, Railway). Paste the connection string into `DATABASE_URL`.

### Option B - Local PostgreSQL

```bash
# Create the database
createdb -h localhost -p 5432 -U postgres vidseek_db

# Update .env
DATABASE_URL=postgresql+psycopg2://postgres:yourpassword@localhost:5432/vidseek_db
```

### Apply migrations

```bash
make apply_migrate
```

This runs all Alembic migrations (13 versions) to create the full schema: Videos, Objects, OCR words, Transcript segments, Visual Relationship Detection tables, and Vector metadata.

---

## Model & Data Files

The trained model weights are **not** included in the repo (excluded via `.gitignore`). Download them before running the pipeline - or retrain using the datasets listed at the end of this section.

All of our own trained models live in a single Google Drive folder, organized to mirror the project's directory layout (`Transformer`, `hybrid_embedder`, `visual`, `OCR`):

**Google Drive:** https://drive.google.com/drive/folders/1GAu6hPLEjHTu5US-5kLkhyDcQSWjvKD6?usp=sharing

### Download required

Download each item from the source below and place it at the matching project path:

| Model | Files / folder | Source | Place at project path |
| ----- | -------------- | ------ | --------------------- |
| Custom Transformer Embedder | `best_model.pt`, `vocab.pkl` | Drive: `VidSeek Models/Transformer/results/allnli_specter` | `Transformer/results/allnli_specter` |
| Hybrid Embedder | `model_lsa.npz`, `model_bm25.npz` | Drive: `VidSeek Models/hybrid_embedder/models` | `hybrid_embedder/models` |
| HOG detector | `model/` folder | Drive: `VidSeek Models/visual/hog/model` | `visual/hog` |
| Visual Relationship Detection (VRD) | `checkpoint/` folder | Drive: `VidSeek Models/visual/vrd_ml/checkpoint` | `visual/vrd_ml` |
| Faster R-CNN | `faster_rcnn_final.pth` | [HuggingFace](https://huggingface.co/OmarMoh11/faster-RCNN/resolve/main/experiments/model_10/faster_rcnn_final.pth) | `visual/faster_rcnn` |
| All OCR models |`OCR/models` | Drive: `VidSeek Models/OCR` | `OCR/` |

**VRD word embeddings (extra step):** download `glove.2024.wikigiga.50d.zip` from [NLP Stanford](https://nlp.stanford.edu/projects/glove/) and extract it at `visual/vrd_ml`.

### Auto-downloaded (no action needed)

- **Whisper (ASR):** `openai/whisper-small` is downloaded automatically on first run (~500 MB) via HuggingFace Transformers and cached in your HuggingFace cache directory.
- **EasyOCR:** detection and recognition models download on first use (~200 MB).

### Training data (reference, for retraining)

#### Custom Transformer Embedder

Trained on a merge of publicly available sentence-embedding datasets:

- **AllNLI** - https://huggingface.co/datasets/sentence-transformers/embedding-training-data/blob/main/AllNLI.jsonl.gz
- **SPECTER Training Triples** - https://huggingface.co/datasets/sentence-transformers/embedding-training-data/blob/main/specter_train_triples.jsonl.gz

Both were originally in **JSONL** format, converted to **CSV**, and merged into a single dataset with the columns `anchor`, `positive`, `negative`.

Evaluation uses the **STS Benchmark** (https://huggingface.co/datasets/sentence-transformers/stsb). Additional pairs from the STS training split were added to the validation and test sets, resulting in approximately **100,000 samples** per split.

> **Note:** The additional samples were used only to enlarge the evaluation and testing datasets - they were never used to train the Transformer model.

#### Hybrid Embedder

Trained on the **MS MARCO Passage Ranking** dataset (https://microsoft.github.io/msmarco/). It combines:

- **BM25** - sparse lexical matching
- **LSA (TF-IDF + Randomized SVD)** - dense semantic representations

The fitted model is saved under `hybrid_embedder/models/` and is loaded automatically by the retrieval pipeline for indexing and search.

#### OCR models (training provenance)

- **CRAFT** text detector - trained on the SynthText dataset (Kaggle).
- **HoG** character recognizer - trained locally on the Char74K dataset.

> Training datasets for the vision detectors (Faster R-CNN, HOG, VRD) are listed under [Run Vision Models Separately](#run-vision-models-separately).

---

## Pipeline Phases

The pipeline stages run in order:

1. Scene segmentation and keyframe extraction
2. OCR on each keyframe
3. Object detection (Faster R-CNN by default)
4. Visual relationship detection
5. ASR transcription (Whisper)
6. Sentence segmentation
7. Embedding via custom Transformer -> HNSW index
8. Embedding via Hybrid Embedder -> HNSW index

---

## Run Vision Models Separately

These sections let you train or run each visual model independently from the full ingestion pipeline.

### Faster R-CNN (folder: `visual/faster_rcnn`)

That model is trained on the PASCAL VOC 2007 dataset available at 
[PASCAL VOC 2007](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html).  

Files:

- `visual/faster_rcnn/inference.py` for single-image inference
- `visual/faster_rcnn/train/pipeline.py` for training
- `visual/faster_rcnn/faster_rcnn_final.pth` pretrained/final weights

Train:

```bash
cd visual/faster_rcnn
make train DATA_PATH=./data CHECKPOINT_PATH=./checkpoints
```

Run inference on one image:

```bash
cd visual/faster_rcnn
make infer IMAGE_PATH=./sample.jpg MODEL_PATH=./faster_rcnn_final.pth SCORE_THRESH=0.5
```

Note:
- We can edit arguments in `visual/faster_rcnn/makefile`

### HOG Detector (folder: `visual/hog`)

That model is trained on the PASCAL VOC 2007 dataset available at 
[PASCAL VOC 2007](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html). 

Files:

- `visual/hog/pipeline.py` for training/evaluation pipeline
- `visual/hog/infer.py` for single-image inference
- `visual/hog/model/` saved detector model directory

Train pipeline:

```bash
cd visual/hog
make train TRAIN_CONFIG=./config/train.json
```

Evaluate only:

```bash
cd visual/hog
make eval TRAIN_CONFIG=./config/train.json
```

Run inference:

```bash
cd visual/hog
make infer DETECT_CONFIG=./config/detect.json
```

JSON inference output:

```bash
cd visual/hog
make infer-json DETECT_CONFIG=./config/detect.json
```
Note:
- We can edit arguments using configs `visual/hog/config/train.json` and `visual/hog/config/detect.json`

### VRD Model (folder: `visual/vrd_ml`)

That model is trained on Stanford VRD dataset available at 
[Stanford Dataset](https://cs.stanford.edu/people/ranjaykrishna/vrd/). 

Files:

- `visual/vrd_ml/vrd_pipeline.ipynb` end-to-end training/evaluation notebook
- `visual/vrd_ml/models/vrd_model.py` core model
- `visual/vrd_ml/VRD.py` runtime wrapper used by the API/pipeline
- `visual/vrd_ml/checkpoints/vrd_rf_real.pkl` and `visual/vrd_ml/checkpoints/vrd_rf.pkl` expected trained checkpoint
- `wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt`: the used word embedding and located in `visual/vrd_ml/`


The notebook trains and saves:

- `visual/vrd_ml/checkpoints/vrd_rf_real.pkl`
- `visual/vrd_ml/checkpoints/vrd_rf_real_clf.pkl`

Note:
- We need to edit the data path in notebook before running

---

## Running the API Server

```bash
make server
```

The server starts at `http://127.0.0.1:8000`. Override host/port if needed:

```bash
make server API_HOST=0.0.0.0 API_PORT=8080
```

### API endpoints

| Method | Path                                              | Description                                     |
| ------ | ------------------------------------------------- | ----------------------------------------------- |
| `POST` | `/videos/upload`                                  | Upload a video and start the ingestion pipeline |
| `GET`  | `/jobs/{job_id}`                                  | Poll pipeline job status                        |
| `GET`  | `/search/transcript?q=...&model=transformer`      | Semantic transcript search                      |
| `GET`  | `/search/ocr?q=...`                               | Text found on screen (OCR) search               |
| `GET`  | `/search/object?key=...`                          | Search by detected object label                 |
| `GET`  | `/search/vrd?subject=...&relation=...&object=...` | Visual relationship search                      |
| `GET`  | `/videos/chapters?path=...`                       | Get chapter/segment list for a video            |
| `GET`  | `/video/stream?path=...`                          | Stream a video file                             |
| `GET`  | `/objects`                                        | List all indexed object labels                  |
| `GET`  | `/vrd/options`                                    | List all VRD subjects, relations, objects       |

The upload endpoint accepts additional form fields:

- `detector`: `craft` (default) or `east`
- `recognizer`: `easyocr` (default) or `mser`
- `object_detector`: `faster_rcnn` (default) or `hog`

---

## Running the Frontend

```bash
make ui-start
```

The React dev server starts at `http://localhost:3000` and proxies API calls to `http://127.0.0.1:8000`.

To build for production:

```bash
make ui-build
```

---

## Quick-start Summary

```bash
# 1. Install dependencies
poetry install
cd vidseek-ui && npm install && cd ..

# 2. Configure environment
cp .env.example .env   # then edit .env with your DB URL and API tokens

# 3. Set up database
make apply_migrate

# 4. Start the API
make server

# 5. Start the frontend (new terminal)
make ui-start
```

---
