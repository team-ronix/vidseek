.DEFAULT_GOAL := help

PYTHON := poetry run python
UVICORN := poetry run uvicorn
ALEMBIC := poetry run alembic
CREATEDB := createdb

VIDEOS_DIR ?= videos
VIDEO_FILE ?= 1.mp4
JSON_DIR ?= json_outputs
OCR_OUTPUT_PATH ?= ocr_inverted_index.json
OBJECT_OUTPUT_PATH ?= object_inverted_index.json
VRD_OUTPUT_PATH ?= vrd_inverted_index.json
TRANSCRIPTION_OUTPUT_PATH ?= transcription.json
SEGMENTED_TRANSCRIPT_PATH ?= segmented_transcript.json
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
API_RELOAD ?= true
UI_DIR ?= vidseek-ui
DB_HOST ?= localhost
DB_PORT ?= 9051
DB_USER ?= postgres
DB_NAME ?= vidseek_db
MIGRATION_MESSAGE ?= Add time range to VRDVideo and ObjectVideo

.PHONY: help install run serve migrate ui-install ui-start ui-build all

help:
	@echo Available targets:
	@echo   install      Install project dependencies with Poetry
	@echo   run          Run the video ingestion pipeline in main.py
	@echo   serve        Start the FastAPI app from api.py with Uvicorn
	@echo   migrate      Apply Alembic migrations to the database
	@echo   ui-install   Install the React UI dependencies
	@echo   ui-start     Start the React development server
	@echo   ui-build     Build the React UI for production
	@echo   all          Install dependencies and run the pipeline
	@echo 
	@echo Environment variables to override defaults:
	@echo   VIDEOS_DIR            			Videos folder passed to main.py (default: videos)
	@echo   VIDEO_FILE            			Video filename passed to main.py (default: 1.mp4)
	@echo   JSON_DIR            			Output folder for intermediate JSON files
	@echo   OCR_OUTPUT_PATH            		Path to save OCR inverted index JSON (default: ocr_inverted_index.json)
	@echo   OBJECT_OUTPUT_PATH            	Path to save object detection inverted index JSON (default: object_inverted_index.json)
	@echo   VRD_OUTPUT_PATH            		Path to save VRD inverted index JSON (default: vrd_inverted_index.json)
	@echo   TRANSCRIPTION_OUTPUT_PATH 		Path to save ASR transcription JSON (default: transcription.json)
	@echo   SEGMENTED_TRANSCRIPT_PATH 		Path to save segmented transcript JSON (default: segmented_transcript.json)
	@echo   API_HOST              			FastAPI host (default: 127.0.0.1)
	@echo   API_PORT              			FastAPI port (default: 8000)
	@echo   API_RELOAD            			Enable Uvicorn reload mode (default: true)
	@echo   UI_DIR                			React UI directory (default: vidseek-ui)
	@echo 
	@echo Example:
	@echo   make run VIDEO_FILE=demo.mp4
	@echo   make serve API_PORT=8080
	@echo   make ui-start UI_DIR=vidseek-ui

install:
	poetry install

lock:
	poetry lock

run:
	$(PYTHON) main.py \
		--videos-folder "$(VIDEOS_DIR)" \
		--video-path "$(VIDEO_FILE)" \
		--json-folder "$(JSON_DIR)" \
		--ocr-output-path "$(OCR_OUTPUT_PATH)" \
		--object-output-path "$(OBJECT_OUTPUT_PATH)" \
		--vrd-output-path "$(VRD_OUTPUT_PATH)" \
		--transcription-output-path "$(TRANSCRIPTION_OUTPUT_PATH)" \
		--segmented-transcript-path "$(SEGMENTED_TRANSCRIPT_PATH)"

serve:
	$(UVICORN) api:app --host $(API_HOST) --port $(API_PORT) $(if $(filter true yes 1 on,$(API_RELOAD)),--reload,)

create_db:
	$(CREATEDB) -h $(DB_HOST) -p $(DB_PORT) -U $(DB_USER) $(DB_NAME)

create_migration:
	$(ALEMBIC) revision --autogenerate -m "$(MIGRATION_MESSAGE)"

apply_migrate:
	$(ALEMBIC) upgrade head

ui-install:
	cd /d "$(UI_DIR)" && npm install

ui-start:
	cd /d "$(UI_DIR)" && npm start

ui-build:
	cd /d "$(UI_DIR)" && npm run build

all: install run