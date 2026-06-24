PYTHON := poetry run python
TRAIN_CONFIG ?= ./config/train.json
DETECT_CONFIG ?= ./config/detect.json

.PHONY: all train eval pipeline resume infer infer-json clean help

all: pipeline

pipeline:
	$(PYTHON) pipeline.py --config $(TRAIN_CONFIG)

train:
	$(PYTHON) pipeline.py --config $(TRAIN_CONFIG)

eval:
	$(PYTHON) pipeline.py --config $(TRAIN_CONFIG) --eval-only

resume:
	$(PYTHON) pipeline.py --config $(TRAIN_CONFIG) --skip-step1

infer:
	$(PYTHON) infer.py --config $(DETECT_CONFIG)

infer-json:
	$(PYTHON) infer.py --config $(DETECT_CONFIG) --json

clean:
	rm -rf ./model ./checkpoint ./eval_ckpt.pkl ./result.jpg

help:
	@echo ""
	@echo "HOG Detector Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make pipeline   train + eval"
	@echo "  make train"
	@echo "  make eval"
	@echo "  make resume     skip step1"
	@echo "  make infer"
	@echo "  make infer-json"
	@echo "  make clean"
	@echo ""
	@echo "Config files:"
	@echo "  TRAIN_CONFIG=$(TRAIN_CONFIG)"
	@echo "  DETECT_CONFIG=$(DETECT_CONFIG)"
	@echo ""
